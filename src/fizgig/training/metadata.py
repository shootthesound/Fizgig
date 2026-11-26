"""SAI Model Spec metadata builder for LoRA safetensors files.

Based on https://github.com/Stability-AI/ModelSpec
Klein 9B specific — other architectures stripped.
"""

import base64
import datetime
import hashlib
from io import BytesIO
import os
import re
from typing import List, Optional, Tuple, Union

import safetensors
import logging

logger = logging.getLogger(__name__)

# Fizgig-native architecture identifiers
ARCHITECTURE_KLEIN_9B = "klein9b"
ARCHITECTURE_KLEIN_9B_FULL = "klein_9b"
ARCHITECTURE_KREA2 = "krea2"  # Krea 2 single-stream MMDiT (Qwen-Image VAE + Qwen3-VL-4B)
# NOTE: no underscore — the dataset cache filename is {basename}_{WxH}_{arch}.safetensors, parsed
# by split("_") with the size at tokens[-2]. An arch string with an underscore (e.g. "minimax_h3")
# shifts that index and breaks size parsing, so the cache-facing id is a single token (same reason
# Klein uses "klein9b", not "klein_9b").
ARCHITECTURE_MINIMAX = "minimaxh3"  # MiniMax H3 omni DiT (video VAE + Qwen3-VL-32B), image-only training

# SAI model spec architecture strings
ARCH_KLEIN_9B = "Flux.2-klein-9b"
IMPL_KLEIN = "https://github.com/black-forest-labs/flux2"
ARCH_KREA2 = "Krea-2"
IMPL_KREA2 = "https://github.com/krea-ai/krea-2"
ARCH_MINIMAX = "MiniMax-H3"
IMPL_MINIMAX = "https://github.com/MiniMax-AI/MiniMax-H3"

ADAPTER_LORA = "lora"
MODELSPEC_TITLE = "modelspec.title"

BASE_METADATA = {
    "modelspec.sai_model_spec": "1.0.0",
    "modelspec.architecture": None,
    "modelspec.implementation": None,
    "modelspec.title": None,
    "modelspec.resolution": None,
    "modelspec.description": None,
    "modelspec.author": None,
    "modelspec.date": None,
    "modelspec.license": None,
    "modelspec.tags": None,
    "modelspec.merged_from": None,
    "modelspec.prediction_type": None,
    "modelspec.timestep_range": None,
    "modelspec.encoder_layer": None,
    "modelspec.trigger_phrase": None,
    "modelspec.thumbnail": None,
    "modelspec.usage_hint": None,
}

# The GUI's default, unedited value for the Output Name field (lora_trainer_gui.py's
# LORA_NAME setting). _apply_lora_name_suffix there only ever swaps the trailing "_k9b"/
# "_krea2" tag to match the selected family, so an untouched field is always exactly one of
# these two — never anything else. Used to keep obvious placeholder text like
# "LoraName_TokenName_krea2" out of modelspec.title.
PLACEHOLDER_OUTPUT_NAMES = {"loraname_tokenname_k9b", "loraname_tokenname_krea2"}


def load_bytes_in_safetensors(tensors):
    bytes_data = safetensors.torch.save(tensors)
    b = BytesIO(bytes_data)
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
    offset = n + 8
    b.seek(offset)
    return b.read()


def precalculate_safetensors_hashes(state_dict):
    hash_sha256 = hashlib.sha256()
    for tensor in state_dict.values():
        single_tensor_sd = {"tensor": tensor}
        bytes_for_tensor = load_bytes_in_safetensors(single_tensor_sd)
        hash_sha256.update(bytes_for_tensor)
    return f"0x{hash_sha256.hexdigest()}"


def _described_family(architecture):
    try:
        from fizgig.families.registry import by_arch_id
        return by_arch_id(architecture)
    except Exception:
        return None


def build_metadata(
    state_dict: Optional[dict],
    architecture: str,
    timestamp: float,
    title: Optional[str] = None,
    reso: Optional[Union[str, int, Tuple[int, int]]] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[str] = None,
    merged_from: Optional[str] = None,
    timesteps: Optional[Tuple[int, int]] = None,
    is_lora: bool = True,
    trigger_phrase: Optional[str] = None,
    thumbnail: Optional[str] = None,
    usage_hint: Optional[str] = None,
) -> dict:
    """Build SAI model spec metadata for Klein 9B LoRA files."""
    metadata = {}
    metadata.update(BASE_METADATA)

    # A reasonable default beats a blank field nobody's ever been able to set before now —
    # same reasoning as the auto-thumbnail. Only kicks in when there's nothing to lose: an
    # explicit usage_hint always wins, and this needs a trigger_phrase to say anything useful.
    if usage_hint is None and trigger_phrase:
        usage_hint = f"Include '{trigger_phrase}' in your prompt."

    # Map Fizgig architecture to SAI model spec
    if architecture in (ARCHITECTURE_KLEIN_9B, ARCHITECTURE_KLEIN_9B_FULL):
        arch = ARCH_KLEIN_9B
        impl = IMPL_KLEIN
    elif architecture == ARCHITECTURE_KREA2:
        arch = ARCH_KREA2
        impl = IMPL_KREA2
    elif architecture == ARCHITECTURE_MINIMAX:
        arch = ARCH_MINIMAX
        impl = IMPL_MINIMAX
    elif _described_family(architecture) is not None:
        # families added through the standard layer (fizgig.families) carry their own modelspec names
        _fam = _described_family(architecture)
        arch = _fam.modelspec_arch
        impl = _fam.implementation
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if is_lora:
        arch += f"/{ADAPTER_LORA}"
    metadata["modelspec.architecture"] = arch
    metadata["modelspec.implementation"] = impl

    if title is None:
        title = f"LoRA@{timestamp}" if is_lora else f"Klein-9B@{timestamp}"
    metadata[MODELSPEC_TITLE] = title

    # Optional fields — set or remove
    for key, value in [
        ("modelspec.author", author),
        ("modelspec.description", description),
        ("modelspec.license", license),
        ("modelspec.tags", tags),
        ("modelspec.merged_from", merged_from),
        ("modelspec.trigger_phrase", trigger_phrase),
        ("modelspec.thumbnail", thumbnail),
        ("modelspec.usage_hint", usage_hint),
    ]:
        if value is not None:
            metadata[key] = value
        else:
            del metadata[key]

    # Date
    int_ts = int(timestamp)
    metadata["modelspec.date"] = datetime.datetime.fromtimestamp(int_ts).isoformat()

    # Resolution
    if reso is not None:
        if isinstance(reso, str):
            reso = tuple(map(int, reso.split(",")))
        if isinstance(reso, int):
            reso = (reso, reso)
        if len(reso) == 1:
            reso = (reso[0], reso[0])
    else:
        reso = (1024, 1024)  # Klein default

    metadata["modelspec.resolution"] = f"{reso[0]}x{reso[1]}"

    # Remove unused fields
    del metadata["modelspec.prediction_type"]
    del metadata["modelspec.encoder_layer"]

    # Timestep range
    if timesteps is not None:
        if isinstance(timesteps, (str, int)):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        metadata["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
    else:
        del metadata["modelspec.timestep_range"]

    if not all(v is not None for v in metadata.values()):
        logger.error(f"Internal error: some metadata values are None: {metadata}")

    return metadata


def latest_sample_image(output_dir: Optional[str]) -> Optional[str]:
    """Most recently written preview under <output_dir>/sample/, if any.

    Used as the default source for `modelspec.thumbnail` — training already produces these,
    so a LoRA can carry a representative preview with no extra input from the user.
    """
    if not output_dir:
        return None
    sample_dir = os.path.join(output_dir, "sample")
    if not os.path.isdir(sample_dir):
        return None
    exts = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                  if f.lower().endswith(exts)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def sample_for_epoch(output_dir: Optional[str], output_name: Optional[str], epoch: int) -> Optional[str]:
    """The newest preview written for THIS epoch (`<name>_e{epoch:06d}_*`), or None.

    `latest_sample_image` picks whatever is newest on disk, which at checkpoint-save time is
    the PREVIOUS epoch's preview — the epoch checkpoint is saved before its own preview renders
    (#122: every checkpoint's thumbnail was one epoch behind). This picks by epoch number."""
    if not output_dir or not output_name:
        return None
    sample_dir = os.path.join(output_dir, "sample")
    if not os.path.isdir(sample_dir):
        return None
    prefix = f"{output_name}_e{int(epoch):06d}_"
    exts = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                  if f.startswith(prefix) and f.lower().endswith(exts)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def refresh_checkpoint_thumbnail(lora_path: str, image_path: str) -> bool:
    """Re-embed `modelspec.thumbnail` in an already-saved checkpoint from `image_path`.

    Used after an epoch's preview renders, so the checkpoint saved a moment earlier carries
    its OWN epoch's picture rather than the previous one's. Tensors are untouched; the file is
    rewritten atomically (tmp + replace). Best-effort — a failure logs and leaves the file."""
    uri = thumbnail_data_uri(image_path)
    if not uri or not lora_path or not os.path.exists(lora_path):
        return False
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        tensors, meta = {}, {}
        with safe_open(lora_path, framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        if meta.get("modelspec.thumbnail") == uri:
            return True
        meta["modelspec.thumbnail"] = uri
        tmp = lora_path + ".thumb.tmp"
        save_file(tensors, tmp, metadata=meta)
        os.replace(tmp, lora_path)
        logger.info("[thumbnail] %s now carries its own epoch's preview (%s)",
                    os.path.basename(lora_path), os.path.basename(image_path))
        return True
    except Exception:
        logger.warning("could not refresh the thumbnail of %s", lora_path, exc_info=True)
        return False


def thumbnail_data_uri(image_path: Optional[str], max_size: int = 512, quality: int = 85) -> Optional[str]:
    """Downscale an image and embed it as a `modelspec.thumbnail` data URI (what ComfyUI's
    model browser renders as the card art). Best-effort: a missing or broken thumbnail must
    never fail a checkpoint save."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        from fizgig.utils.upright import upright
        with Image.open(image_path) as im:
            from fizgig.utils.image import flatten
            im = flatten(upright(im))
            im.thumbnail((max_size, max_size))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        logger.warning(f"could not build metadata thumbnail from {image_path}", exc_info=True)
        return None


def resolve_title(output_name: Optional[str], trigger_phrase: Optional[str]) -> Optional[str]:
    """The output name, unless it's still the GUI's untouched placeholder — in which case
    fall back to the trigger phrase (if there is one) rather than writing obvious filler like
    "LoraName_TokenName_krea2" into modelspec.title. Deliberately narrow: this only ever
    affects the metadata title, never the actual filename/output_name used everywhere else in
    the run (cache dirs, state dirs, the checkpoint's own name), which stays exactly what the
    user set regardless."""
    if output_name and output_name.strip().lower() in PLACEHOLDER_OUTPUT_NAMES and trigger_phrase:
        return trigger_phrase
    return output_name


def get_title(metadata: dict) -> Optional[str]:
    return metadata.get(MODELSPEC_TITLE, None)


def load_metadata_from_safetensors(model: str) -> dict:
    if not model.endswith(".safetensors"):
        return {}
    with safetensors.safe_open(model, framework="pt") as f:
        metadata = f.metadata()
    return metadata if metadata is not None else {}


def build_merged_from(models: List[str]) -> str:
    def _get_title(model: str):
        metadata = load_metadata_from_safetensors(model)
        title = metadata.get(MODELSPEC_TITLE, None)
        if title is None:
            title = os.path.splitext(os.path.basename(model))[0]
        return title
    return ", ".join(_get_title(m) for m in models)


# ---- training facts (kohya ss_* keys + one fizgig_training_config JSON) -----------------------------------------
# Technical recipe only: no captions, tag lists, prompts, author names, dataset paths, or folder names.
TRAINING_CONFIG_KEY = "fizgig_training_config"
_SECRETISH = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                        r"xox[abprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_\-]{20,}|[A-Fa-f0-9]{40,}|[A-Za-z0-9+/]{48,})")
_PRIVATE_SPEC_KEY = re.compile(
    r"^(?:author|authors|user|user_id|username|tag|tags|caption|captions|prompt|prompts|trigger_phrase|"
    r"description|image_directory|cache_directory|directory|directories|folder_name|folder_names|"
    r"dataset_name|dataset_names)$"
    r"|(?:^|_)(?:path|paths|file_path|file_paths|directory|directories|folder_name|folder_names|"
    r"username|user_id)(?:_|$)"
    r"|^(?:caption|prompt|tag)_(?:text|list|content|bodies|strings)$",
    re.IGNORECASE,
)
_OMIT = object()
_VERSION = []


def _safe_name(value) -> str:
    """A file name only: the last part of a path (either slash), with secret-like strings removed."""
    s = str(value or "").replace("\\", "/").rstrip("/")
    s = s.rsplit("/", 1)[-1]
    return _SECRETISH.sub("[removed]", s)


def _safe_text(value) -> str:
    """A short setting value: path-like values are reduced to a basename and secrets are removed."""
    s = str(value if value is not None else "")
    if "/" in s or "\\" in s:
        s = _safe_name(s)
    return _SECRETISH.sub("[removed]", s)[:200]


def _safe_training_spec(value, key=None):
    """Recursively keep scalar training settings while filtering content, identity, and path-bearing fields."""
    if key is not None and _PRIVATE_SPEC_KEY.search(str(key)):
        return _OMIT
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            safe = _safe_training_spec(v, k)
            if safe is not _OMIT and safe is not None:
                out[str(k)] = safe
        return out
    if isinstance(value, (list, tuple)):
        out = [_safe_training_spec(v) for v in value]
        return [v for v in out if v is not _OMIT and v is not None]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _safe_text(value)
    return _safe_text(value)


def fizgig_version() -> dict:
    """The running Fizgig version and commit, read once, offline."""
    if not _VERSION:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        def git(*a):
            try:
                r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=5,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return r.stdout.strip() if r.returncode == 0 else ""
            except (OSError, ValueError, Exception):
                return ""
        try:
            from fizgig import __version__ as pkg
        except Exception:
            pkg = ""
        commit = git("rev-parse", "HEAD")
        info = {"version": git("describe", "--tags", "--always") or pkg or "unknown",
                "commit": commit if re.fullmatch(r"[0-9a-f]{40}", commit or "") else "",
                "changed": bool(git("status", "--porcelain", "--untracked-files=no"))}
        _VERSION.append(info)
    return dict(_VERSION[0])


def dataset_dirs(datasets) -> Tuple[dict, dict, dict]:
    """Kohya-shaped counts plus anonymous, technical per-dataset recipe summaries; no names or paths."""
    dirs, reg, table = {}, {}, {}
    train_i = reg_i = 0
    for ds in datasets or []:
        if not getattr(ds, "image_directory", None):
            continue
        is_reg = bool(getattr(ds, "is_reg", False))
        if is_reg:
            reg_i += 1
            key, target = f"reg_dataset_{reg_i:02d}", reg
        else:
            train_i += 1
            key, target = f"dataset_{train_i:02d}", dirs
        try:
            count = ds.get_total_image_count()
        except Exception:
            count = None
        reps = int(getattr(ds, "num_repeats", 1) or 1)
        img_count = int(count) if count is not None else 0
        target[key] = {"n_repeats": reps, "img_count": img_count}

        recipe = {
            "weight": float(getattr(ds, "loss_weight", 1.0) or 1.0),
            "repeats": reps,
            "img_count": img_count,
            "batch_size": int(getattr(ds, "batch_size", 1) or 1),
            "enable_bucket": bool(getattr(ds, "enable_bucket", False)),
            "bucket_no_upscale": bool(getattr(ds, "bucket_no_upscale", False)),
            "cache_enabled": bool(getattr(ds, "cache_directory", None)),
            "tag_shuffle": bool(getattr(ds, "tag_shuffle", False)),
            "keep_first_tags": int(getattr(ds, "keep_first_tags", 1) or 0),
            "tag_dropout": float(getattr(ds, "tag_dropout", 0.0) or 0.0),
            "caption_dropout": float(getattr(ds, "caption_dropout", 0.0) or 0.0),
            "caption_variants": int(getattr(ds, "caption_variants", 0) or 0),
            "caption_variant_cap": int(getattr(ds, "caption_variant_cap", 0) or 0),
            "paired_data": bool(getattr(ds, "control_directory", None)),
            "extra_reference_data": bool(getattr(ds, "extra_reference_directory", None)),
            "mask_directory_enabled": bool(getattr(ds, "mask_directory", None)),
            "photos_only": bool(getattr(ds, "photos_only", False)),
        }
        resolution = getattr(ds, "resolution", None)
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            recipe["resolution"] = [int(resolution[0]), int(resolution[1])]
        extension = str(getattr(ds, "caption_extension", "") or "")
        if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", extension):
            recipe["caption_extension"] = extension
        clip_mp = getattr(ds, "clip_megapixels", None)
        if clip_mp is not None:
            try:
                recipe["clip_megapixels"] = float(clip_mp)
            except (TypeError, ValueError):
                pass
        if not is_reg:
            table[key] = recipe
        else:
            # The regularization dataset gets the same technical recipe, without a folder identity.
            table[key] = {**recipe, "regularization": True}
    return dirs, reg, table


def bucket_resolutions(datasets) -> List[str]:
    """The resolutions the pictures actually trained at (each dataset's buckets), "WxH", largest area first."""
    seen = set()
    for ds in datasets or []:
        bm = getattr(ds, "batch_manager", None)
        for r in getattr(bm, "bucket_resos", None) or []:
            try:
                seen.add((int(r[0]), int(r[1])))
            except (TypeError, ValueError, IndexError):
                pass
    return [f"{w}x{h}" for w, h in sorted(seen, key=lambda r: (-r[0] * r[1], r))]


def training_metadata(*, seed=None, batch_size=None, grad_accum=1, epochs=None, epoch=None, steps=None,
                      steps_per_epoch=None, max_steps=None, num_images=None, scheduler=None, warmup_steps=None,
                      resolution=None, buckets=None, mixed_precision="bf16", base_precision=None, base_model=None,
                      datasets=None, training_method=None, selective_block_check=False, folder_balance=None,
                      validation_groups=None, best_epoch=None, best_val_loss=None, current_lr=None,
                      training_spec=None, started_at=None, now=None) -> dict:
    """Kohya ss_* keys plus a sanitized Fizgig training recipe; no paths, folder names, captions or tags."""
    import json
    import time as _time
    now = float(now if now is not None else _time.time())
    dirs, reg, table = dataset_dirs(datasets)
    buckets = list(buckets or [])
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        reso = f"({int(resolution[0])}, {int(resolution[1])})"
    elif resolution:
        reso = f"({int(resolution)}, {int(resolution)})"
    else:
        reso = ""
    md = {}

    def put(k, v):
        if v is not None and v != "":
            md[k] = _safe_text(v)
    put("ss_seed", seed)
    put("ss_batch_size_per_device", batch_size)
    put("ss_gradient_accumulation_steps", grad_accum)
    put("ss_num_epochs", epochs)
    put("ss_epoch", epoch)
    put("ss_steps", steps)
    put("ss_num_batches_per_epoch", steps_per_epoch)
    put("ss_max_train_steps", max_steps)
    put("ss_num_train_images", num_images)
    put("ss_num_reg_images", sum(int(x.get("img_count", 0)) for x in reg.values()))
    put("ss_num_datasets", len(dirs))
    put("ss_num_reg_datasets", len(reg))
    put("ss_lr_scheduler", scheduler)
    put("ss_lr_warmup_steps", warmup_steps)
    put("ss_resolution", reso)
    put("ss_mixed_precision", mixed_precision)
    put("ss_base_precision", base_precision)
    put("ss_sd_model_name", _safe_name(base_model) if base_model else None)
    if started_at:
        put("ss_training_started_at", f"{float(started_at):.0f}")
    put("ss_training_finished_at", f"{now:.0f}")
    # The Kohya field names are retained, but directory keys are anonymous indices.
    md["ss_dataset_dirs"] = json.dumps(dirs, sort_keys=True)
    if reg:
        md["ss_reg_dataset_dirs"] = json.dumps(reg, sort_keys=True)
    if buckets:
        md["ss_bucket_info"] = json.dumps({"buckets": buckets}, sort_keys=True)
    ver = fizgig_version()
    cfg = {
        "training_method": _safe_text(training_method or "standard"),
        "selective_block_check": bool(selective_block_check),
        "dataset_layout": table,
        "folder_balance": folder_balance if folder_balance is None else bool(folder_balance),
        "validation_groups": int(validation_groups) if validation_groups else None,
        "epoch": int(epoch) if epoch is not None else None,
        "epochs": int(epochs) if epochs is not None else None,
        "steps": int(steps) if steps is not None else None,
        "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch is not None else None,
        "max_steps": int(max_steps) if max_steps is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "current_lr": float(current_lr) if current_lr is not None else None,
        "training_seconds": round(now - float(started_at), 1) if started_at else None,
        "resolution": reso.strip("()").replace(", ", "x") or None,
        "trained_resolutions": buckets,
        "base_precision": _safe_text(base_precision) if base_precision else None,
        "base_model": _safe_name(base_model) if base_model else None,
        "dataset_counts": {
            "training_datasets": len(dirs),
            "regularization_datasets": len(reg),
            "training_images": int(num_images or 0),
            "regularization_images": sum(int(x.get("img_count", 0)) for x in reg.values()),
        },
        "training_spec": _safe_training_spec(training_spec or {}),
        "fizgig_version": _safe_text(ver.get("version")),
        "fizgig_commit": ver.get("commit") or None,
        "fizgig_files_changed": ver.get("changed"),
    }
    md[TRAINING_CONFIG_KEY] = json.dumps(cfg, sort_keys=True)
    return md
