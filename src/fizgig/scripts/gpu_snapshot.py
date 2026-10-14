"""One-shot RDNA2 GUI metadata snapshot; no matrix kernels or package imports."""
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def snapshot():
    """Collect the launch decision in a child process without calling detect()."""
    if not (os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm"
            and os.environ.get("GPU_ARCH", "").lower().startswith("gfx103")):
        raise RuntimeError("RDNA2 GPU snapshot requires the gfx103 ROCm launcher")

    from fizgig.utils.capabilities import Capabilities
    import torch

    caps = Capabilities()
    if not torch.cuda.is_available():
        caps.notes.append("GPU unavailable")
        return caps

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "").split(":")[0].lower()
    if not arch.startswith("gfx103"):
        raise RuntimeError(f"GPU architecture changed since launch: {arch or 'unknown'}")
    caps.has_cuda = True
    caps.is_rocm = True
    caps.device_name = props.name
    caps.vram_gb = props.total_memory / (1024 ** 3)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(0)
        caps.vram_free_gb = free_bytes / (1024 ** 3)
    except Exception:
        caps.vram_free_gb = caps.vram_gb
        caps.notes.append("could not read free VRAM — using card total")

    # The RDNA2 Auto plan uses NF4 or fp8; INT8 GEMM kernel probes in the
    # GUI caused stalls on gfx103. Other ROCm cards still use detect().
    caps.flash_attn = importlib.util.find_spec("flash_attn") is not None
    caps.bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
    if not caps.bitsandbytes:
        caps.notes.append("bitsandbytes missing — NF4 unavailable")
    caps.notes.append("RDNA2 GUI kernel probes skipped")
    return caps


if __name__ == "__main__":
    print(json.dumps(asdict(snapshot())), flush=True)