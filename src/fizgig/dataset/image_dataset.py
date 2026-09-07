"""Image dataset pipeline for Klein 9B LoRA training.

Handles image loading, bucketing, latent/text-encoder cache I/O, and batch
assembly.
"""

from concurrent.futures import ThreadPoolExecutor
import glob
from importlib.util import find_spec
import math
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from PIL import Image
import cv2

from fizgig.utils.safetensors import mem_eff_save_file, MemoryEfficientSafeOpen
from fizgig.training.metadata import (ARCHITECTURE_KLEIN_9B, ARCHITECTURE_KLEIN_9B_FULL,
                                      ARCHITECTURE_MINIMAX)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP",
    ".avif", ".AVIF",
]

if find_spec("jxlpy") is not None:
    from jxlpy import JXLImagePlugin  # noqa: F401
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if find_spec("pillow_jxl") is not None:
    import pillow_jxl  # noqa: F401
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

# Video clips, MiniMax H3 only — passed in per-dataset rather than added to IMAGE_EXTENSIONS, so
# a stray .mp4 in a Klein or Krea 2 folder is still ignored rather than handed to PIL. Declared
# here rather than imported from fizgig.minimax so the dataset layer keeps knowing nothing about
# any model.
VIDEO_EXTENSIONS = (".mp4", ".MP4")

# Audio-only voice items, MiniMax H3 only — same opt-in mechanism as clips. An audio item trains
# the audio stream against a tiny placeholder video latent; the dataset layer only needs to know
# these files exist and that they never enter the resolution bucketing (see AUDIO_SENTINEL_RESO).
AUDIO_EXTENSIONS = (".wav", ".WAV", ".mp3", ".MP3", ".flac", ".FLAC", ".m4a", ".M4A")

# The fixed pseudo-resolution every audio item carries. It is a real 128x128 (the placeholder
# latent is (24, T, 8, 8) at the 16x spatial factor), but audio items are PINNED here rather than
# routed through the bucket selector: with upscaling on, 128px would bucket to ~target-MP and the
# reso re-check would then drop every audio item as stale. Pinning also keeps 128-px stills from
# sharing a batch with audio items.
AUDIO_SENTINEL_RESO = (128, 128)


def is_audio_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {e.lower() for e in AUDIO_EXTENSIONS}

RESOLUTION_STEPS = 16  # Klein 9B resolution step

# Per-architecture bucket grid. A bucket edge must be divisible by (VAE spatial factor x DiT
# spatial patch) or the latent can't be patchified exactly. MiniMax H3 is 16x VAE with a 2x2
# patch = 32; on a 16 grid, half the buckets produce an odd latent that the trainer then has to
# crop (losing up to 16 px of edge). The reference trainers bucket at 32 for exactly this reason.
BUCKET_RESO_STEPS = {ARCHITECTURE_MINIMAX: 32}

# Pixel -> stored-latent spatial factor per architecture. Klein's FLUX.2 AE packs 2x2
# space-to-channel after its /8 encoder, so cached latents are pixel/16; Krea 2's
# Qwen-Image VAE stores plain /8 latents.
# MiniMax H3's video VAE is 16x spatial, same as Klein's packed latents (verified against a real
# cache: a 448x544 bucket stores `latent_28x34`). Missing here, latent_cache_matches_reso could
# only ever return None for H3 — so --skip_existing re-encoded every image anyway, which is
# exactly the "it re-caches everything" symptom.
LATENT_SPATIAL_FACTOR = {"klein9b": 16, "krea2": 8, ARCHITECTURE_MINIMAX: 16}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dtype_to_str(dtype: torch.dtype) -> str:
    """Return the short name of a torch dtype, e.g. 'bfloat16'."""
    return str(dtype).split(".")[-1]


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> Optional[torch.dtype]:
    """Convert a string like 'bf16' or 'bfloat16' to a torch.dtype."""
    if s is None:
        return default_dtype
    mapping = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
        "fp8_e4m3fn": torch.float8_e4m3fn, "e4m3fn": torch.float8_e4m3fn, "float8_e4m3fn": torch.float8_e4m3fn,
    }
    if hasattr(torch, "float8_e5m2"):
        mapping.update({"fp8_e5m2": torch.float8_e5m2, "e5m2": torch.float8_e5m2, "float8_e5m2": torch.float8_e5m2})
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unsupported dtype string: {s}")


def glob_images(directory: str, base: str = "*", caption_extension: Optional[str] = None,
                extra_extensions: tuple = ()) -> list[str]:
    """Glob image files in *directory*, optionally filtering to those with captions.

    extra_extensions adds formats this dataset accepts beyond images — VIDEO_EXTENSIONS for
    MiniMax H3, empty for everyone else. Opt-in per dataset so a stray .mp4 in a Klein folder is
    still skipped rather than opened as an image.
    """
    img_paths: list[str] = []
    for ext in list(IMAGE_EXTENSIONS) + list(extra_extensions):
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))

    if caption_extension is not None:
        caption_paths = glob.glob(os.path.join(glob.escape(directory), "*" + caption_extension))
        caption_bases = {os.path.splitext(os.path.basename(p))[0] for p in caption_paths}
        img_paths = [p for p in img_paths if os.path.splitext(os.path.basename(p))[0] in caption_bases]

    img_paths.sort()
    return img_paths


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: Tuple[int, int]) -> np.ndarray:
    """Resize *image* to *bucket_reso* ``(width, height)``, cropping the centre."""
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    bucket_width, bucket_height = bucket_reso
    if (bucket_width, bucket_height) == (image_width, image_height):
        return np.array(image) if is_pil else image

    scale = max(bucket_width / image_width, bucket_height / image_height)
    new_w = int(image_width * scale + 0.5)
    new_h = int(image_height * scale + 0.5)

    if scale > 1:
        img = Image.fromarray(image) if not is_pil else image
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
    else:
        arr = np.array(image) if is_pil else image
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    crop_left = (new_w - bucket_width) // 2
    crop_top = (new_h - bucket_height) // 2
    return arr[crop_top: crop_top + bucket_height, crop_left: crop_left + bucket_width]


# ---------------------------------------------------------------------------
# ItemInfo
# ---------------------------------------------------------------------------

class ItemInfo:
    """Metadata for a single training image (no video fields)."""

    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: Tuple[int, int],
        bucket_size: Optional[Tuple[int, int]] = None,
        content: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        latent_cache_path: Optional[str] = None,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.content = content  # np.ndarray or list[np.ndarray] for multi-target
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None
        self.control_content: Optional[list[np.ndarray]] = None

    def __str__(self) -> str:
        content_shape = None
        if isinstance(self.content, list):
            content_shape = [c.shape for c in self.content]
        elif self.content is not None:
            content_shape = self.content.shape
        ctrl_shape = None
        if isinstance(self.control_content, list):
            ctrl_shape = [c.shape for c in self.control_content]
        elif self.control_content is not None:
            ctrl_shape = self.control_content.shape
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            f"latent_cache_path={self.latent_cache_path}, "
            f"content={content_shape}, control_content={ctrl_shape})"
        )


# ---------------------------------------------------------------------------
# Cache I/O — Fizgig-native format
# ---------------------------------------------------------------------------

def save_latent_cache(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]] = None,
) -> None:
    """Save latent cache for Klein 9B (3-D latent: C, H, W)."""
    assert latent.dim() == 3, f"latent must be 3-D (C,H,W), got {latent.shape}"

    _, H, W = latent.shape
    sd: dict[str, torch.Tensor] = {f"latent_{H}x{W}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            assert cl.dim() == 3, f"control_latent[{i}] must be 3-D, got {cl.shape}"
            _, cH, cW = cl.shape
            sd[f"latent_control_{i}_{cH}x{cW}"] = cl.detach().cpu().contiguous()

    metadata = {
        "architecture": ARCHITECTURE_KLEIN_9B_FULL,
        "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]),
        "dtype": dtype_to_str(latent.dtype),
        "format_version": "2.0.0",
    }

    # NaN guard
    for key, value in sd.items():
        if torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} — replaced with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)
    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, ctx_vec: torch.Tensor) -> None:
    """Save text-encoder output cache for Klein 9B."""
    sd: dict[str, torch.Tensor] = {"text_embed": ctx_vec.detach().cpu()}

    # NaN guard
    for key, value in sd.items():
        if torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} — replaced with 0")
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": ARCHITECTURE_KLEIN_9B_FULL,
        "caption1": item_info.caption,
        "dtype": dtype_to_str(ctx_vec.dtype),
        "format_version": "2.0.0",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):
        # Merge with existing cache (may contain embeddings from another encoder)
        with MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_meta = f.metadata()
            for key in f.keys():
                if key not in sd:
                    sd[key] = f.get_tensor(key)
            assert existing_meta.get("architecture") == metadata["architecture"], "architecture mismatch in existing cache"
            if existing_meta.get("caption1") != metadata["caption1"]:
                logger.warning(
                    f"caption mismatch: existing={existing_meta.get('caption1')}, "
                    f"new={metadata['caption1']}; overwriting"
                )
            existing_meta.pop("caption1", None)
            existing_meta.pop("format_version", None)
            metadata.update(existing_meta)
    else:
        te_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(te_dir, exist_ok=True)

    mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


# ---------------------------------------------------------------------------
# BucketSelector
# ---------------------------------------------------------------------------

class BucketSelector:
    """Aspect-ratio bucketing with configurable resolution step."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        enable_bucket: bool = True,
        no_upscale: bool = False,
        reso_steps: int = RESOLUTION_STEPS,
    ):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.reso_steps = reso_steps

        if not enable_bucket:
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            resolutions: set[Tuple[int, int]] = set()
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                resolutions.add((w, h))
                resolutions.add((h, w))
            self.bucket_resolutions = sorted(resolutions)

        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Return the closest bucket resolution for *image_size* ``(width, height)``."""
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w = divisible_by(image_size[0], self.reso_steps)
            h = divisible_by(image_size[1], self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        bucket_id = int(np.abs(self.aspect_ratios - aspect_ratio).argmin())
        return self.bucket_resolutions[bucket_id]

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: Tuple[int, int],
        resolution: Tuple[int, int],
        reso_steps: int = RESOLUTION_STEPS,
    ) -> Tuple[int, int]:
        """Compute best bucket resolution for a given image size and target area."""
        max_area = resolution[0] * resolution[1]
        width, height = image_size
        aspect_ratio = width / height

        bucket_width = int(math.sqrt(max_area * aspect_ratio))
        bucket_height = int(math.sqrt(max_area / aspect_ratio))
        bucket_width = divisible_by(bucket_width, reso_steps)
        bucket_height = divisible_by(bucket_height, reso_steps)

        best_resolution: Optional[Tuple[int, int]] = None
        best_diff = float("inf")
        for i in range(-2, 3):
            w = bucket_width + i * reso_steps
            if w <= 0:
                continue
            h = divisible_by(max_area // w, reso_steps)
            diff = abs((w / h) - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                best_resolution = (w, h)

        return best_resolution if best_resolution is not None else (bucket_width, bucket_height)


# ---------------------------------------------------------------------------
# BucketBatchManager
# ---------------------------------------------------------------------------

class BucketBatchManager:
    """Manages batched iteration over bucketed items, loading cached tensors."""

    def __init__(
        self,
        bucketed_item_info: dict[Tuple[int, int], list[ItemInfo]],
        batch_size: int,
        num_timestep_buckets: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        # Keys are (w, h) tuples — except the audio sentinel ("audio", w, h), which a plain
        # sort would crash on (str vs int). Stringify per element; the order is only cosmetic.
        self.bucket_resos = sorted(self.buckets.keys(), key=lambda k: tuple(str(x) for x in k))
        self.num_timestep_buckets = num_timestep_buckets
        self.timestep_pool: Optional[list[list[float]]] = None

        self.bucket_batch_indices: list[Tuple[Tuple[int, int], int]] = []
        for reso in self.bucket_resos:
            num_batches = math.ceil(len(self.buckets[reso]) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((reso, i))

    def show_bucket_info(self):
        for reso in self.bucket_resos:
            logger.info(f"bucket: {reso}, count: {len(self.buckets[reso])}")
        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        for bucket in self.buckets.values():
            random.shuffle(bucket)
        random.shuffle(self.bucket_batch_indices)

        if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
            num_batches = len(self.bucket_batch_indices)
            total_needed = num_batches * self.batch_size
            all_timesteps: list[float] = []
            per_bucket = math.ceil(total_needed / self.num_timestep_buckets)
            for i in range(self.num_timestep_buckets):
                lo = i / self.num_timestep_buckets
                hi = (i + 1) / self.num_timestep_buckets
                all_timesteps.extend(random.uniform(lo, hi) for _ in range(per_bucket))
            random.shuffle(all_timesteps)
            all_timesteps = all_timesteps[:total_needed]
            self.timestep_pool = [
                all_timesteps[i * self.batch_size: (i + 1) * self.batch_size]
                for i in range(num_batches)
            ]

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))

        batch_tensor_data: dict[str, list[torch.Tensor]] = {}

        for item_info in bucket[start:end]:
            sd_latent = load_file(item_info.latent_cache_path)
            sd_te = load_file(item_info.text_encoder_output_cache_path)
            if getattr(item_info, "first_frame_only", False):
                # the clip's first frame as a still: [C, T, H, W] -> [C, H, W] (frame 0 of a
                # causal VAE encodes pixel frame 0 alone), and no sound — a still has none
                sd_latent = {k: (v[:, 0].contiguous() if (k.startswith("latent_") and v.dim() == 4) else v)
                             for k, v in sd_latent.items()
                             if k not in ("audio_latent", "audio_only")}
            sd = {**sd_latent, **sd_te}
            # MiniMax H3 reference distillation only: sibling `..._teref{N}.safetensors` files
            # hold the TEACHER's conditioning (caption + a `<Picture 1>` vision block), its
            # per-row modality tags, and that reference's own latent. Each item has one slot per
            # paired reference; a slot is picked at RANDOM per step so the LoRA sees every
            # pairing across an epoch while the teacher's sequence stays short.
            #
            # Gated on the files EXISTING, which they only do when a MiniMax cache pass ran with
            # --reference_count — so Klein, Krea 2 and non-distillation H3 batches are unchanged.
            _ref_te = item_info.text_encoder_output_cache_path
            if _ref_te:
                _stem, _ext = os.path.splitext(_ref_te)
                _slots = [i for i in range(8) if os.path.exists(f"{_stem}ref{i}{_ext}")]
                if _slots:
                    _sd_ref = load_file(f"{_stem}ref{random.choice(_slots)}{_ext}")
                    sd["ref_hidden_states"] = _sd_ref["hidden_states"]
                    sd["ref_token_tags"] = _sd_ref["token_tags"]
                    sd["ref_latent"] = _sd_ref["ref_latent"]

            for key, tensor in sd.items():
                # Map Fizgig-native keys to training batch keys
                if key.startswith("latent_control_"):
                    # latent_control_{i}_{H}x{W} -> latents_control_{i}
                    parts = key.split("_")  # ['latent', 'control', '{i}', '{H}x{W}']
                    content_key = f"latents_control_{parts[2]}"
                elif key.startswith("latent_"):
                    # latent_{H}x{W} -> latents
                    content_key = "latents"
                elif key == "text_embed":
                    content_key = "ctx_vec"
                elif key in ("ref_hidden_states", "ref_token_tags", "ref_latent",
                             "audio_latent", "audio_only"):
                    # H3: distillation refs, the clip's packed audio rows, and the audio-only
                    # flag, verbatim. Named explicitly because the fallback below would rsplit
                    # `audio_latent` AND `audio_only` into `audio` — and an unfound audio_only
                    # flag doesn't crash, it silently trains the video head against the zeros
                    # placeholder. The worst kind of miss.
                    content_key = key
                elif key in ("hidden_states", "attention_mask"):
                    # Krea 2 text cache: multi-layer Qwen3-VL stack + validity mask
                    content_key = key
                else:
                    # Fallback: strip dtype suffix for compatibility with any
                    # legacy keys that might exist in merged caches
                    content_key = key.rsplit("_", 1)[0] if not key.endswith("_mask") else key
                    if content_key.startswith("latents_"):
                        content_key = content_key.rsplit("_", 1)[0]

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(tensor)

        # Stack tensors per key
        stacked: dict[str, Any] = {}
        for key, tensors in batch_tensor_data.items():
            stacked[key] = torch.stack(tensors)

        if self.timestep_pool is not None:
            stacked["timesteps"] = self.timestep_pool[idx][:end - start]
        else:
            stacked["timesteps"] = None

        # Per-image identity for the (passive) per-image loss logger. Harmless list of strings that
        # trainers ignore unless logging is enabled.
        stacked["item_keys"] = [item_info.item_key for item_info in bucket[start:end]]

        return stacked


# ---------------------------------------------------------------------------
# Datasource — image directory (the primary data loader)
# ---------------------------------------------------------------------------

class ImageDirectoryDatasource:
    """Loads images (and optional control images) from a directory on disk."""

    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        extra_extensions: tuple = (),
    ):
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.extra_extensions = tuple(extra_extensions)
        self.caption_only = False
        self.has_control = False
        self._current_idx = 0

        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory,
                                       caption_extension=self.caption_extension,
                                       extra_extensions=self.extra_extensions)
        _clips = sum(1 for p in self.image_paths
                     if os.path.splitext(p)[1].lower() in {e.lower() for e in VIDEO_EXTENSIONS})
        _audio = sum(1 for p in self.image_paths if is_audio_path(p))
        if _clips or _audio:
            logger.info(f"found {len(self.image_paths) - _clips - _audio} images, "
                        f"{_clips} clip(s) and {_audio} audio file(s)")
        else:
            logger.info(f"found {len(self.image_paths)} images")

        # Two items sharing a stem (`voice.wav` beside `voice.mp4`) share a TEXT cache file —
        # its name carries no size token — so both would train with whichever caption was
        # encoded last, silently. Refuse up front, naming the pair.
        _by_stem: dict[str, str] = {}
        for p in self.image_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in _by_stem:
                raise ValueError(
                    f"'{os.path.basename(_by_stem[stem])}' and '{os.path.basename(p)}' share the "
                    f"name '{stem}' — their caches would collide and one caption would silently "
                    f"win. Rename one of them.")
            _by_stem[stem] = p

        # An audio file without a caption .txt is silently dropped by the caption filter above —
        # reasonable for a stray photo, baffling for the voice recording someone deliberately
        # added. Say so once, with the fix.
        if self.caption_extension and os.path.isdir(self.image_directory or ""):
            _globbed = {os.path.basename(p) for p in self.image_paths}
            _accepted = {e.lower() for e in self.extra_extensions}
            _dropped = [f for f in os.listdir(self.image_directory)
                        if is_audio_path(f) and f not in _globbed
                        and os.path.splitext(f)[1].lower() in _accepted]
            if _dropped:
                logger.warning(
                    f"[dataset] {len(_dropped)} audio file(s) have no caption "
                    f"{self.caption_extension} and will NOT train: {', '.join(_dropped[:5])}"
                    f"{'…' if len(_dropped) > 5 else ''}. Gizmo's audio tab writes captions "
                    f"as it exports segments.")

        # Optional control images
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths: dict[str, list[str]] = {}

            sorted_by_len = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)
            all_ctrl = set(glob_images(self.control_directory))

            for img_path in sorted_by_len:
                base_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                matches = [
                    p for p in all_ctrl
                    if os.path.basename(p).startswith(base_no_ext + ".")
                    or os.path.basename(p).startswith(base_no_ext + "_")
                ]
                all_ctrl.difference_update(matches)
                if matches:
                    def _sort_key(path, _base=base_no_ext):
                        bn = os.path.splitext(os.path.basename(path))[0]
                        if bn == _base:
                            return 0
                        suffix = bn.rsplit("_", 1)[-1]
                        return int(suffix) + 1 if suffix.isdigit() else 999999
                    matches.sort(key=_sort_key)
                    self.control_paths[img_path] = matches

            missing = len(self.image_paths) - len(self.control_paths)
            if missing > 0:
                missing_paths = set(self.image_paths) - set(self.control_paths.keys())
                raise ValueError(f"Missing control images for {missing} images: {missing_paths}")

            logger.info(f"found {len(self.control_paths)} matching control image sets")

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self) -> bool:
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> Tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        image_path = self.image_paths[idx]
        if is_audio_path(image_path):
            # An audio item's "image" is a fixed placeholder — the caching layer writes a zeros
            # video latent and the trainer zeroes the video loss, so these pixels are never
            # encoded or trained on. The placeholder exists so the item flows through the same
            # size/bucket bookkeeping as everything else (pinned to AUDIO_SENTINEL_RESO, never
            # the bucket selector). The audio itself is decoded at cache time, not here.
            _, caption = self.get_caption(idx)
            return image_path, [Image.new("RGB", AUDIO_SENTINEL_RESO)], caption, None
        if os.path.splitext(image_path)[1].lower() in {e.lower() for e in VIDEO_EXTENSIONS}:
            # A clip arrives as its frames, which the resize step downstream already handles as a
            # list. Imported lazily and only inside this branch: the dataset layer stays free of
            # model imports at load time, and only MiniMax datasets ever glob a video in the
            # first place.
            from fizgig.minimax.clip import read_frames, validate
            frames = read_frames(image_path)
            validate(image_path, frames=len(frames))
            imgs = [Image.fromarray(f) for f in frames]
            _, caption = self.get_caption(idx)
            return image_path, imgs, caption, None

        img = Image.open(image_path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for cp in self.control_paths[image_path]:
                c = Image.open(cp)
                if c.mode not in ("RGB", "RGBA"):
                    c = c.convert("RGB")
                controls.append(c)

        return image_path, [img], caption, controls

    def get_caption(self, idx: int) -> Tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:
            def _fetch(i=self._current_idx):
                return self.get_caption(i)
        else:
            def _fetch(i=self._current_idx):
                return self.get_image_data(i)

        self._current_idx += 1
        return _fetch


# ---------------------------------------------------------------------------
# ImageDataset
# ---------------------------------------------------------------------------

class ImageDataset(torch.utils.data.Dataset):
    """Klein 9B image dataset with bucketing, caching, and batch management."""

    # MiniMax H3 only (Peter, 7 Sep 2026): every cached CLIP also contributes its FIRST FRAME
    # as a photo item — its own step, the clip's own text encoding. H3's video VAE is causal
    # and its token grid opens with a lone frame, so a clip latent's first temporal slice IS
    # the still's latent: no extra encode, no extra cache file, just a derived item sliced at
    # load. Set on the CLASS by the trainer before the dataset group is generated (the
    # blueprint path builds and prepares the datasets in one go).
    clip_first_frame_as_photo: bool = False

    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        image_directory: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = ARCHITECTURE_KLEIN_9B,
        is_reg: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale if enable_bucket else False
        self.image_directory = image_directory
        self.control_directory = control_directory
        self.cache_directory = cache_directory
        self.debug_dataset = debug_dataset
        self.architecture = architecture
        # Regularisation set: a prior anchor, not a subject. The trainer trains it at a fixed
        # reduced LR and keeps the per-image loss watch off it.
        self.is_reg = is_reg
        self.reso_steps = BUCKET_RESO_STEPS.get(architecture, RESOLUTION_STEPS)

        self.seed: Optional[int] = None
        self.current_epoch = 0
        self.shared_epoch: SharedEpoch = None
        self.max_train_steps = 0

        if image_directory is None:
            raise ValueError("image_directory must be specified")

        # Only MiniMax H3 can train on clips and voice recordings, so only its datasets glob for
        # them. Kept on the dataset as well as handed to the datasource, because
        # prepare_for_training re-derives what counts as a training item when it cross-checks
        # the cache — and if the two answers disagree, every clip's cache is discarded as an
        # orphan and the run trains on nothing.
        _extra = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS if architecture == ARCHITECTURE_MINIMAX else ()
        self.extra_extensions = _extra
        self.datasource = ImageDirectoryDatasource(image_directory, caption_extension,
                                                   control_directory, extra_extensions=_extra)

        if self.cache_directory is None:
            self.cache_directory = self.image_directory

        self.batch_manager: Optional[BucketBatchManager] = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control

    # -- metadata ----------------------------------------------------------

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
        }
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["has_control"] = self.has_control
        return metadata

    def get_total_image_count(self) -> Optional[int]:
        return len(self.datasource) if self.datasource.is_indexable() else None

    # -- cache paths -------------------------------------------------------

    def get_all_latent_cache_files(self) -> list[str]:
        return glob.glob(os.path.join(glob.escape(self.cache_directory), f"*_{self.architecture}.safetensors"))

    def get_all_text_encoder_output_cache_files(self) -> list[str]:
        return glob.glob(os.path.join(glob.escape(self.cache_directory), f"*_{self.architecture}_te.safetensors"))

    def _plan_clip_bucket(self, bucket_reso, image_key):
        """Cap a clip's bucket to what the VAE can actually encode in the free VRAM.

        Free VRAM is read ONCE and remembered. Reading it per clip would shrink every clip after
        the first, since the encoder's own allocations are still resident by then — the plan has
        to be made against the memory the pass starts with, not against what it has left.
        """
        if not hasattr(self, "_clip_free_gb"):
            try:
                import torch
                from fizgig.utils.device import plannable_free_vram
                self._clip_free_gb = (plannable_free_vram()   # honours FIZGIG_SIM_VRAM_GB
                                      if torch.cuda.is_available() else 0.0)
            except Exception:
                self._clip_free_gb = 0.0
            self._clip_capped = set()
        if not self._clip_free_gb:
            return bucket_reso                       # CPU, or no way to ask — leave it alone

        from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder
        planned = MiniMaxH3VideoVAEEncoder.plan_clip_bucket(self._clip_free_gb, *bucket_reso)
        if tuple(planned) != tuple(bucket_reso) and image_key not in self._clip_capped:
            self._clip_capped.add(image_key)
            logger.warning(
                "[clip] %s: %dx%d needs more VRAM than the %.1f GB free can encode — caching it "
                "at %dx%d instead. Stills in this dataset are unaffected. Lower Target "
                "Megapixels, or close whatever else is on the GPU, to keep the full size.",
                os.path.basename(str(image_key)), bucket_reso[0], bucket_reso[1],
                self._clip_free_gb, planned[0], planned[1])
        return planned

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None
        return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None
        return os.path.join(self.cache_directory, f"{basename}_{self.architecture}_te.safetensors")

    # -- latent cache batch retrieval (for caching phase) ------------------

    def retrieve_latent_cache_batches(self, num_workers: int):
        """Yield ``(bucket_key, [ItemInfo])`` batches for latent caching."""
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale,
                                        self.reso_steps)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[Tuple[int, ...], list[ItemInfo]] = {}
        futures: list = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed = [f for f in futures if f.done()]
                if not completed:
                    if len(futures) >= num_workers or consume_all:
                        time.sleep(0.1)
                        continue
                    else:
                        break

                for future in completed:
                    original_size, item_key, images, caption, controls = future.result()
                    image = images[0]
                    bucket_h, bucket_w = image.shape[:2]
                    bucket_reso = (bucket_w, bucket_h)

                    item_info = ItemInfo(
                        item_key, caption, original_size, bucket_reso,
                        content=image if len(images) == 1 else images,
                    )
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)

                    bkey: Tuple = bucket_reso
                    if controls is not None:
                        item_info.control_content = controls
                        # Different control sizes go into different batches
                        extra = tuple(s for ctrl in controls for s in ctrl.shape[:2])
                        bkey = bucket_reso + extra
                    if is_audio_path(str(item_key)):
                        # A 128px STILL legitimately buckets to (128,128) too — the marker keeps
                        # audio items out of its batch (the encoder takes different paths).
                        bkey = ("audio",) + bucket_reso

                    batches.setdefault(bkey, []).append(item_info)
                    futures.remove(future)

        def submit_batch(flush: bool = False):
            for key in list(batches.keys()):
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][:self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size:]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:
            def fetch_and_resize(op=fetch_op):
                image_key, images, caption, controls = op()
                image: Image.Image = images[0]
                image_size = image.size
                if is_audio_path(image_key):
                    # Never through the bucket selector: with upscaling on, the 128px
                    # placeholder would bucket to ~target-MP, and prepare_for_training's reso
                    # re-check would then drop every audio item as stale (and --skip_existing
                    # would re-encode them every launch). The placeholder is already at
                    # AUDIO_SENTINEL_RESO — no resize, just the ndarray the aggregator expects.
                    return image_size, image_key, [np.asarray(img) for img in images], caption, None
                bucket_reso = bucket_selector.get_bucket_resolution(image_size)
                # A clip costs far more to ENCODE than a still of the same size — a whole
                # 17-frame group is live in the VAE at once — so a mixed dataset lets someone
                # pick a megapixel target that is fine for their photographs and impossible for
                # their clips. Cap the clip rather than fail the run: the stills are unaffected,
                # and a clip at a size that fits is worth more than a crash at one that does not.
                if len(images) > 1:
                    bucket_reso = self._plan_clip_bucket(bucket_reso, image_key)
                resized = [resize_image_to_bucket(img, bucket_reso) for img in images]

                resized_controls = None
                if controls is not None:
                    resized_controls = [resize_image_to_bucket(c, bucket_reso) for c in controls]
                return image_size, image_key, resized, caption, resized_controls

            future = executor.submit(fetch_and_resize)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    # -- text encoder cache batch retrieval --------------------------------

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        """Yield ``[ItemInfo]`` batches for text-encoder output caching."""
        self.datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures: list = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed = [f for f in futures if f.done()]
                if not completed:
                    if len(futures) >= num_workers or consume_all:
                        time.sleep(0.1)
                        continue
                    else:
                        break
                for future in completed:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    data.append(item_info)
                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= self.batch_size or (data and flush):
                batch = data[:self.batch_size]
                data = data[self.batch_size:]
                return batch
            return None

        for fetch_op in self.datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()

    # -- training preparation (after caching) ------------------------------

    @staticmethod
    def latent_cache_matches_reso(cache_file: str, bucket_reso: Tuple[int, int],
                                  architecture: str) -> Optional[bool]:
        """Does the cached latent inside `cache_file` match `bucket_reso` (pixel w, h)?

        The cache FILENAME encodes the original image size, which never changes — so a cache
        written at one Target Megapixels setting is indistinguishable by name from one written
        at another, and __getitem__ maps whatever `latent_*` key the file holds onto `latents`.
        Without this check a resolution change trains silently at the OLD resolution wherever
        re-encoding is skipped. The pixel->latent factor is per-architecture
        (LATENT_SPATIAL_FACTOR): Klein stores packed pixel/16 latents, Krea 2 plain pixel/8 —
        comparing both against /8 rejected every valid Klein cache (issue #27).
        Header-only read (safe_open.keys()); returns None if the file can't be read, holds no
        parseable latent key, or the architecture is unknown — callers choose their own
        failure mode.
        """
        factor = LATENT_SPATIAL_FACTOR.get(architecture)
        if factor is None:
            return None
        try:
            from safetensors import safe_open
            with safe_open(cache_file, framework="pt") as f:
                keys = list(f.keys())
        except Exception:
            return None
        expected = {int(bucket_reso[0]) // factor, int(bucket_reso[1]) // factor}
        for k in keys:
            if k.startswith("latent_") and not k.startswith("latent_control_"):
                try:
                    # `latent_{H}x{W}` for a still, `latent_{T}x{H}x{W}` for a video clip. Only
                    # the spatial pair is compared, so take the LAST two — a clip cache is
                    # resolution-checked exactly like a still. Reading the first two instead
                    # would compare T against a height and quietly re-encode every clip, every
                    # launch.
                    dims = k[len("latent_"):].split("x")
                    a, b = dims[-2], dims[-1]
                    if {int(a), int(b)} == expected:
                        return True
                    # A CLIP cached BELOW its bucket is a legitimate state, not staleness:
                    # the VRAM cap (_plan_clip_bucket) shrinks a clip to what the VAE can
                    # encode in the free memory at cache time, so demanding exact equality
                    # dropped every capped clip at train time — a clips-only dataset then
                    # crashed with "No training items" straight after a clean cache run
                    # (field report, 16 GB card). "smaller" lets the trainer accept it
                    # (loudly) while --skip_existing stays strict (`is not True`), so a
                    # fresh cache run still re-attempts full size. Stills keep the exact
                    # check — the v3.0.0 silent-stale incident was a still, and stills
                    # are never VRAM-capped.
                    if len(dims) == 3:
                        _c = sorted((int(a), int(b)))
                        _e = sorted((int(bucket_reso[0]) // factor,
                                     int(bucket_reso[1]) // factor))
                        if _c[0] <= _e[0] and _c[1] <= _e[1]:
                            return "smaller"
                    return False
                except Exception:
                    return None
        return None

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        """Build the BucketBatchManager from cached latent files on disk."""
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale,
                                        self.reso_steps)

        latent_cache_files = glob.glob(os.path.join(glob.escape(self.cache_directory), f"*_{self.architecture}.safetensors"))

        # The training set is built from CACHE files, not from the image folder — so stale cache
        # entries get silently TRAINED ON: an image deleted from the dataset lingers as its cache,
        # and a cache directory shared across datasets mixes the previous dataset straight into
        # this run. Cross-check every cache item against the images actually present and skip the
        # orphans. (Cache filenames are the image basename without extension.)
        valid_keys = None
        if self.image_directory and os.path.isdir(self.image_directory):
            # The SAME extensions the datasource globbed with, video included. Images only would
            # mean every clip's cache looks like an orphan of a deleted image: a folder of clips
            # caches perfectly, then reports "ignored 17 stale cache file(s)" and trains on
            # nothing. A caption .txt still cannot keep a deleted item alive, which is the point
            # of listing extensions here rather than taking every stem in the folder.
            _exts = {e.lower() for e in IMAGE_EXTENSIONS}
            _exts |= {e.lower() for e in getattr(self, "extra_extensions", ())}
            valid_keys = {os.path.splitext(f)[0] for f in os.listdir(self.image_directory)
                          if os.path.splitext(f)[1].lower() in _exts}
        # The stems that are audio files. Cache filenames strip the extension, so the folder
        # listing is the only place an item's audio-ness can still be seen at this point.
        audio_keys = set()
        if self.image_directory and os.path.isdir(self.image_directory):
            audio_keys = {os.path.splitext(f)[0] for f in os.listdir(self.image_directory)
                          if is_audio_path(f)}
        skipped_stale = 0
        skipped_wrong_reso = 0
        accepted_smaller_clips = 0
        first_frame_items = 0

        bucketed: dict[Tuple, list[ItemInfo]] = {}   # (w, h), or ("audio", w, h) for voice items
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")
            # Filename: {basename}_{W:04d}x{H:04d}_{arch}.safetensors
            size_token = tokens[-2]  # e.g. "0768x0512"
            image_width, image_height = map(int, size_token.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            if valid_keys is not None and item_key not in valid_keys:
                skipped_stale += 1
                continue
            te_cache = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(te_cache):
                logger.warning(f"Text encoder cache not found: {te_cache}")
                continue

            if item_key in audio_keys:
                # Audio items are pinned, never bucketed: the selector would upscale their
                # 128px placeholder to ~target-MP and the reso check below would drop them all
                # as stale. The Target Megapixels setting genuinely does not apply to a voice —
                # there is nothing to render — so the check is skipped, not fudged. The marker
                # in the bucket key keeps them out of any real 128px bucket: their latents are
                # (24, T, 8, 8) against a still's (24, 8, 8) and cannot stack together.
                bucket_reso = AUDIO_SENTINEL_RESO
                bucket_key: Tuple = ("audio",) + AUDIO_SENTINEL_RESO
            else:
                bucket_reso = bucket_selector.get_bucket_resolution(image_size)
                # A cache written at a different Target Megapixels has the SAME filename (it
                # encodes the original size, not the bucket) — training on it would silently run
                # at the old resolution. Skip it and tell the user to re-run cache preparation.
                _reso_ok = self.latent_cache_matches_reso(cache_file, bucket_reso, self.architecture)
                if _reso_ok is False:
                    skipped_wrong_reso += 1
                    continue
                if _reso_ok == "smaller":
                    # A VRAM-capped clip (or one cached at an older, lower Target Megapixels)
                    # — train it at its cached size rather than dropping it: dropping made a
                    # clips-only dataset crash with "No training items" right after a clean
                    # cache run. Batch size 1 only: mixed sizes cannot stack into one batch
                    # (H3 always runs batch 1, so this guard is belt-and-braces).
                    if self.batch_size != 1:
                        skipped_wrong_reso += 1
                        continue
                    accepted_smaller_clips += 1
                bucket_key = bucket_reso
            item_info = ItemInfo(item_key, "", image_size, bucket_reso, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = te_cache

            bucket = bucketed.get(bucket_key, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            # A clip's first frame as a photo of its own (see the class attribute): a
            # derived item sharing both cache files, sliced in __getitem__. Voice items
            # (audio sentinel bucket) never qualify; a still (3-D latent) is left alone.
            if (self.clip_first_frame_as_photo and bucket_key != ("audio",) + AUDIO_SENTINEL_RESO
                    and self.latent_cache_frames(cache_file) > 1):
                ff = ItemInfo(item_key + "#frame0", "", image_size, bucket_reso,
                              latent_cache_path=cache_file)
                ff.text_encoder_output_cache_path = te_cache
                ff.first_frame_only = True
                for _ in range(self.num_repeats):
                    bucket.append(ff)
                first_frame_items += self.num_repeats
            bucketed[bucket_key] = bucket

        if skipped_stale:
            logger.warning(
                f"[dataset] ignored {skipped_stale} stale cache file(s) in {self.cache_directory} "
                f"with no matching image in {self.image_directory} — deleted images or another "
                f"dataset's leftovers. They are NOT trained on; delete them to silence this.")
        if skipped_wrong_reso:
            logger.warning(
                f"[dataset] ignored {skipped_wrong_reso} cache file(s) written at a DIFFERENT "
                f"resolution than this run's Target Megapixels — re-run cache preparation "
                f"(latent + text) to train at the current setting.")
        if accepted_smaller_clips:
            logger.info(
                f"[dataset] {accepted_smaller_clips} clip(s) are cached below their bucket size "
                f"(VRAM-capped at encode time, or cached at an older Target Megapixels) — "
                f"training them at the cached size. Re-running cache preparation with more "
                f"free VRAM re-encodes them at full size.")

        if first_frame_items:
            logger.info(
                f"[dataset] {first_frame_items} clip first frame(s) added as photo items — each "
                f"clip's frame 0 trains as a still on its own step, with the clip's caption "
                f"(sliced from the clip's own latent; nothing re-encoded).")
        self.batch_manager = BucketBatchManager(bucketed, self.batch_size, num_timestep_buckets=num_timestep_buckets)
        self.batch_manager.show_bucket_info()
        self.num_train_items = sum(len(b) for b in bucketed.values())

    @staticmethod
    def latent_cache_frames(cache_file: str) -> int:
        """Latent frames in a cached item, from the header alone (1 for a still or on any
        trouble reading)."""
        try:
            from safetensors import safe_open
            with safe_open(cache_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("latent_") and not k.startswith("latent_control_"):
                        shape = f.get_slice(k).get_shape()
                        return int(shape[1]) if len(shape) == 4 else 1
        except Exception:
            pass
        return 1

    # -- epoch / seed management -------------------------------------------

    def set_seed(self, seed: int, shared_epoch: SharedEpoch):
        self.seed = seed
        self.shared_epoch = shared_epoch

    def set_current_epoch(self, epoch: int):
        assert self.shared_epoch is not None
        assert self.shared_epoch.value == epoch

    def set_max_train_steps(self, max_train_steps: int):
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    # -- Dataset protocol --------------------------------------------------

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy before prepare_for_training
        return len(self.batch_manager)

    def __getitem__(self, idx):
        # Handle epoch transitions
        assert self.shared_epoch is not None
        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            logger.info(f"epoch incremented: {self.current_epoch} -> {epoch}")
            for _ in range(epoch - self.current_epoch):
                self.current_epoch += 1
                self.shuffle_buckets()
        elif epoch < self.current_epoch:
            logger.warning(f"epoch not incremented: current={self.current_epoch}, got={epoch}")
            self.current_epoch = epoch

        return self.batch_manager[idx]


# ---------------------------------------------------------------------------
# DatasetGroup
# ---------------------------------------------------------------------------

class DatasetGroup(torch.utils.data.ConcatDataset):
    """Thin wrapper around multiple ImageDataset instances."""

    def __init__(self, datasets: Sequence[ImageDataset]):
        super().__init__(datasets)
        self.datasets: list[ImageDataset] = list(datasets)
        self.num_train_items = sum(ds.num_train_items for ds in self.datasets)

    def set_current_epoch(self, epoch: int):
        for ds in self.datasets:
            ds.set_current_epoch(epoch)

    def set_max_train_steps(self, max_train_steps: int):
        for ds in self.datasets:
            ds.set_max_train_steps(max_train_steps)
