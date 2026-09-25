"""Detect whether PyTorch is running on NVIDIA CUDA or AMD ROCm (HIP).

ROCm builds expose ``torch.cuda`` (HIP backend) just like CUDA builds do,
so most device code works unchanged — but feature probes (fp8, cuDNN SDPA,
bitsandbytes wheels) need to know which stack is underneath.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def is_rocm() -> bool:
    try:
        import torch
    except Exception:
        return False
    # Identify the installed backend from build metadata. Checking device
    # availability here needlessly enters the driver during imports/startup.
    # torch 2.10+: torch.version.rocm; older builds only had .hip
    rocm = getattr(torch.version, "rocm", None)
    if rocm:
        return True
    hip = getattr(torch.version, "hip", None)
    if hip:
        return True
    ver = getattr(torch, "__version__", "") or ""
    return "+rocm" in ver.lower()


def backend_label() -> str:
    return "ROCm" if is_rocm() else "CUDA"
