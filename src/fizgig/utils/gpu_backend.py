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
    if not torch.cuda.is_available():
        return False
    # torch 2.10+: torch.version.rocm; older builds only had .hip
    rocm = getattr(torch.version, "rocm", None)
    if rocm:
        return True
    hip = getattr(torch.version, "hip", None)
    if hip:
        return True
    ver = getattr(torch, "__version__", "") or ""
    return "+rocm" in ver.lower()


def pin_rocm_device() -> None:
    """AMD PCs with built-in graphics listed first (#186, #342): Launch Fizgig AMD.bat puts the graphics card's index
    in HIP_VISIBLE_DEVICES so only that card is visible. Some ROCm builds on Windows ignore it - torch still lists
    both, and "cuda" then means the built-in graphics (gfx1036 on a Ryzen 7000), whose kernels the card's PyTorch
    doesn't carry ("invalid kernel file"). When that happens, make the chosen card the current device, so every
    "cuda" tensor lands on it. When the variable worked (one device left), nothing changes."""
    import os
    want = (os.environ.get("HIP_VISIBLE_DEVICES") or "").strip()
    if not want.isdigit():
        return
    try:
        import torch
        if not is_rocm():
            return
        n = torch.cuda.device_count()
        idx = int(want)
        if n > 1 and idx < n:
            torch.cuda.set_device(idx)
            import logging
            logging.getLogger(__name__).info(
                f"[AMD-ROCm] HIP_VISIBLE_DEVICES={idx} was not applied by this ROCm build ({n} devices listed): "
                f"using device {idx}, {torch.cuda.get_device_name(idx)}")
    except Exception:
        pass


def backend_label() -> str:
    return "ROCm" if is_rocm() else "CUDA"
