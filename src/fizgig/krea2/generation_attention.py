"""Krea 2's inference-only attention preference, read once per generated image.

Use ComfyUI-style fixed-length calls, batching equal-length samples and splitting
ragged batches to exclude padding. ROCm builds need not export the variable-length API.
"""
import json
import logging
import importlib.util
import os
from functools import wraps

from fizgig import user_data


def preference_available():
    """Offered only on an AMD launch with SageAttention installed in Fizgig's Python (the Linux AMD installer builds
    it for RX 9070 / 9070 XT). Never on NVIDIA. Discovery must not load native kernels, so find_spec, not import."""
    if os.environ.get("FIZGIG_GPU_BACKEND", "").lower() != "rocm":
        return False
    try:
        return importlib.util.find_spec("sageattention") is not None
    except (ImportError, ValueError):
        return False


def wanted():
    """The preference is on AND offered here: a setting saved on another machine, or before SageAttention was
    removed, does nothing where the tick isn't shown."""
    try:
        with open(user_data.PREFS_FILE, encoding="utf-8") as f:
            on = str(json.load(f).get("inference_sageattention", "0")).lower() in ("1", "true")
    except (OSError, ValueError):
        return False
    return on and preference_available()


def configured_generation(fn):
    @wraps(fn)
    def generate(self, dit, *args, **kwargs):
        if not wanted():
            return fn(self, dit, *args, **kwargs)
        from fizgig.krea2 import attention
        if not callable(attention.sageattn):
            raise RuntimeError("Force SageAttention is enabled, but sageattention.sageattn is unavailable. "
                               "Install a SageAttention build compatible with your GPU and PyTorch/ROCm in "
                               "Fizgig's Python environment, or disable it in Preferences → Inference performance.")
        previous = dit.attn_mode, dit.split_attn
        try:
            dit.attn_mode, dit.split_attn = "sageattn", True
            logging.getLogger(__name__).info("[attention] forcing SageAttention for Krea 2 generation (ComfyUI-style calls)")
            return fn(self, dit, *args, **kwargs)
        finally:
            dit.attn_mode, dit.split_attn = previous
    return generate
