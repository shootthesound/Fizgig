"""Krea 2's inference-only attention preference, read once per generated image.

Use fixed-length SageAttention calls per sample: ROCm builds need not export the
variable-length CUDA API. The model's existing split path removes padding first.
"""
import json
import logging
import importlib.util
import os
from functools import wraps

from fizgig import user_data


def preference_available():
    """Offer on AMD launches or when installed; discovery must not load native kernels."""
    if os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm":
        return True
    try:
        return importlib.util.find_spec("sageattention") is not None
    except (ImportError, ValueError):
        return False


def wanted():
    try:
        with open(user_data.PREFS_FILE, encoding="utf-8") as f:
            return str(json.load(f).get("inference_sageattention", "0")).lower() in ("1", "true")
    except (OSError, ValueError):
        return False


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
            logging.getLogger(__name__).info("[attention] forcing SageAttention for Krea 2 generation")
            return fn(self, dit, *args, **kwargs)
        finally:
            dit.attn_mode, dit.split_attn = previous
    return generate
