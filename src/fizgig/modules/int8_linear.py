"""comfy-kitchen's INT8 linear for inference renders on an int8 ConvRot base (MiniMax H3), as an opt-in switch.

The checkpoint's int8 weights against int8 activations: the kernel rotates the activation (the same block Hadamard),
quantizes it to int8 with one scale per row, and runs an int8 GEMM - what ComfyUI runs on these files. Fizgig's own
preview path casts the int8 weight to bf16 on every call and runs a bf16 GEMM with the activation at full precision.

Measured on a 5090 (12 Nov 2026) at a 5-second 512x512 clip's 7,200 tokens: one MLP layer 8.5 ms -> 2.1 ms, the
output ~1% relative from Fizgig's path (the activation's int8 rounding; the weights are the same codes either way).

How it plugs in: ConvRotInt8Linear.forward calls `linear(...)` first and uses the result when it is not None. None
means "run your own path": the switch is off, a gradient is being taken, the input isn't on a CUDA card, or the kernel
is missing / failed (announced once). The workbench engines turn the switch on around their renders (`renders()`);
training never turns it on.
"""
import contextlib

import torch

_STATE = {"wanted": False, "checked": False, "fn": None}


def set_wanted(on: bool) -> None:
    _STATE["wanted"] = bool(on)


def wanted() -> bool:
    return bool(_STATE["wanted"])


@contextlib.contextmanager
def renders(on: bool = True):
    """The switch on (or as given) for the renders inside, back as it was after."""
    had = _STATE["wanted"]
    _STATE["wanted"] = bool(on)
    try:
        yield
    finally:
        _STATE["wanted"] = had


def _fn():
    st = _STATE
    if not st["checked"]:
        st["checked"] = True
        try:
            import comfy_kitchen as _ck
            st["fn"] = _ck.int8_linear
            print("[int8-linear] comfy-kitchen kernel active", flush=True)
        except Exception as _e:
            print(f"[int8-linear] comfy-kitchen not available ({type(_e).__name__}) — the bf16 path instead",
                  flush=True)
    return st["fn"]


def kernel_available() -> bool:
    return _fn() is not None


def linear(x, qdata, wscale, bias, rot, out_dtype):
    """x [..., in] through the int8 kernel against the int8 ConvRot weight (codes [out, in], scale [out, 1], rotation
    group `rot`), or None when the caller's own path should run."""
    if not _STATE["wanted"] or torch.is_grad_enabled() or not x.is_cuda or not qdata.is_cuda:
        return None
    fn = _fn()
    if fn is None:
        return None
    try:
        return fn(x, qdata, wscale, bias, out_dtype=out_dtype, convrot=rot > 1, convrot_groupsize=int(rot))
    except Exception as _e:
        _STATE["fn"] = None
        print(f"[int8-linear] the kernel failed ({type(_e).__name__}: {_e}) — the bf16 path for the rest of the run",
              flush=True)
        return None
