"""Hardware-selected FP32 GEMMs for frozen NF4 weights on RDNA2.

Keep BF16 model/activation storage. Disable autocast locally so it cannot undo
the conversion. No global dtype change or persistent FP32 weight cache.
"""
import functools
import os

import torch


@functools.lru_cache(maxsize=None)
def _is_rdna2(device_index):
    from fizgig.utils.gpu_backend import is_rocm
    if not is_rocm():
        return False
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":")[0].startswith("gfx103")


def enabled(x):
    return (os.environ.get("FIZGIG_RDNA2_LINEAR", "auto") in ("auto", "1", "fp32")
            and x.device.type == "cuda" and x.dtype == torch.bfloat16
            and _is_rdna2(x.device.index))


def stream_nf4_enabled(device):
    """Stream on RDNA2 only; allow disabling it for comparison."""
    device = torch.device(device)
    return (os.environ.get("FIZGIG_STREAM_NF4", "1") != "0"
            and device.type == "cuda" and _is_rdna2(device.index))


def matmul(a, b):
    with torch.autocast(device_type=a.device.type, enabled=False):
        return (a.float() @ b.float()).to(a.dtype)
