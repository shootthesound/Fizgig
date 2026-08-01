"""SAI Model Spec metadata builder for LoRA safetensors files.

Based on https://github.com/Stability-AI/ModelSpec
Klein 9B specific — other architectures stripped.
"""

import base64
import datetime
import hashlib
from io import BytesIO
import os
from typing import List, Optional, Tuple, Union

import safetensors
import logging

logger = logging.getLogger(__name__)

# Fizgig-native architecture identifiers
ARCHITECTURE_KLEIN_9B = "klein9b"
ARCHITECTURE_KLEIN_9B_FULL = "klein_9b"
ARCHITECTURE_KREA2 = "krea2"  # Krea 2 single-stream MMDiT (Qwen-Image VAE + Qwen3-VL-4B)

# SAI model spec architecture strings
ARCH_KLEIN_9B = "Flux.2-klein-9b"
IMPL_KLEIN = "https://github.com/black-forest-labs/flux2"
ARCH_KREA2 = "Krea-2"
IMPL_KREA2 = "https://github.com/krea-ai/krea-2"

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
}


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
) -> dict:
    """Build SAI model spec metadata for Klein 9B LoRA files."""
    metadata = {}
    metadata.update(BASE_METADATA)

    # Map Fizgig architecture to SAI model spec
    if architecture in (ARCHITECTURE_KLEIN_9B, ARCHITECTURE_KLEIN_9B_FULL):
        arch = ARCH_KLEIN_9B
        impl = IMPL_KLEIN
    elif architecture == ARCHITECTURE_KREA2:
        arch = ARCH_KREA2
        impl = IMPL_KREA2
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


def thumbnail_data_uri(image_path: Optional[str], max_size: int = 512, quality: int = 85) -> Optional[str]:
    """Downscale an image and embed it as a `modelspec.thumbnail` data URI (what ComfyUI's
    model browser renders as the card art). Best-effort: a missing or broken thumbnail must
    never fail a checkpoint save."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        logger.warning(f"could not build metadata thumbnail from {image_path}", exc_info=True)
        return None


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
