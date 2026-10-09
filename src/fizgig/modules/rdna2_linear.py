"""Hardware-selected FP32 GEMMs for frozen NF4 weights on RDNA2.

Keep BF16 model/activation storage. Disable autocast locally so it cannot undo
the conversion. No global dtype change or persistent FP32 weight cache.
"""
import torch
import torch.nn as nn


def is_rdna2_device(device):
    """One architecture check at model load; no NVIDIA device probe."""
    if not (getattr(torch.version, "rocm", None) or getattr(torch.version, "hip", None)
            or "+rocm" in (getattr(torch, "__version__", "") or "").lower()):
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device.index)
    return getattr(props, "gcnArchName", "").split(":")[0].startswith("gfx103")


def matmul(a, b):
    with torch.autocast(device_type=a.device.type, enabled=False):
        return (a.float() @ b.float()).to(a.dtype)


class _RDNA2NF4Linear(torch.autograd.Function):
    """Rebuild a frozen NF4 weight in backward instead of retaining its BF16 copy."""
    @staticmethod
    def forward(ctx, x, packed, state):
        from bitsandbytes.functional import dequantize_nf4
        # Saved tensors are released when backward completes. A plain ctx.packed
        # reference survives in the last loss graph and retains the entire NF4
        # base on GPU when a preview parks it on CPU and restores a new copy.
        ctx.save_for_backward(packed)
        ctx.state, ctx.shape = state, x.shape
        weight = dequantize_nf4(packed, state).to(x.dtype)
        return matmul(x.reshape(-1, x.shape[-1]), weight.t()).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        from bitsandbytes.functional import dequantize_nf4
        (packed,) = ctx.saved_tensors
        weight = dequantize_nf4(packed, ctx.state).to(grad.dtype)
        dx = matmul(grad.reshape(-1, grad.shape[-1]), weight)
        return dx.reshape(ctx.shape), None, None


def nf4_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    out = _RDNA2NF4Linear.apply(x, self._nf4_packed, self._nf4_state)
    return out if self.bias is None else out + self.bias


def install_nf4_forward(model):
    """Replace forwards only on the RDNA2 model's already-packed NF4 Linears."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, "_is_nf4", False):
            module.forward = nf4_forward.__get__(module, type(module))
            count += 1
    return count
