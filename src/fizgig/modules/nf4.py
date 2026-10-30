"""4-bit (NF4) base quantization for the frozen Klein 9B DiT — opt-in low-VRAM mode.

Quantizes the frozen base-block Linear weights to NF4 (bitsandbytes), roughly
halving DiT residency (~9.6 GB fp8 → ~5.6 GB) so a full 9B LoRA fits on an
10–12 GB card with no block swap (the gap fp8's ~14 GB training can't reach). The
trainable LoRA still runs in bf16 on top —
this is QLoRA's recipe applied to a diffusion DiT.

Design mirrors modules/fp8.py: we keep each `nn.Linear` (so LoRA targeting by
module path is unchanged) and monkey-patch its `forward` to dequantize NF4 → bf16
per matmul, through the custom autograd Function `_NF4DequantLinear`: the packed
4-bit weight and quant state ride on the context and the bf16 weight is rebuilt
inside the backward, so the unpacked copy never enters autograd's saved-tensor machinery.

That replaces the first shape of this file, which did plain `dequantize_nf4` + `F.linear` with no Function: standard
autograd SAVED the bf16 weight of every quantised Linear for grad_x, so with checkpointing OFF they were all resident
at once and 4-bit storage billed like 16-bit (measured: nf4 uncheckpointed 33.8 GB against int8's 22.8 GB, which pays
no such gap because its matmul is 8-bit and no bf16 copy is ever made). The Function is bit-identical to the plain
path and reclaims the residency IN EAGER MODE only - a compiled run keeps the dequantised weights alive across the
step however the Python forward is written, so 33.8 GB is not fixable at this layer (that needs bnb's C++
`matmul_4bit`, opaque to dynamo).

Source-agnostic: quantizes from whatever the loaded base is. An fp8 base is
dequantized fp8 → bf16 first, then NF4 (fp8→NF4 is slightly lossier — double
quantization); a bf16 base goes straight to NF4 (cleaner, closer to QLoRA's
native error). Either way the result is a 4-bit resident base.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_NF4_TARGET_DEFAULT = ("double_blocks", "single_blocks")


class _NF4DequantLinear(torch.autograd.Function):
    """Memory-frugal dequant matmul for a FROZEN NF4 base Linear — the fp8 `_FP8DequantLinear` pattern.

    The packed weight and quant state ride on the context as plain attributes rather than save_for_backward, so
    the bf16 dequant exists only inside each call and never across the step; backward rebuilds it from the packed
    weight (weight and state are frozen → None).

    The bias is handled HERE rather than added by the caller, because F.linear fuses the bias into the GEMM
    epilogue: adding it afterwards reorders the rounding by one bf16 ulp (measured 0.0078 at O(1) values), which
    this path is meant not to have. With the bias inside, the result is bit-identical to the plain dequant path
    by construction.
    """

    @staticmethod
    def forward(ctx, x, packed, state, bias):
        from bitsandbytes.functional import dequantize_nf4
        w = dequantize_nf4(packed, state).to(x.dtype)
        ctx.packed = packed
        ctx.state = state
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_out):
        from bitsandbytes.functional import dequantize_nf4
        w = dequantize_nf4(ctx.packed, ctx.state).to(grad_out.dtype)
        grad_bias = None
        if ctx.needs_input_grad[3]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0)
        return grad_out @ w, None, None, grad_bias


def nf4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Patched forward for an NF4-quantized frozen Linear: the matmul runs through `_NF4DequantLinear`, so the
    bf16 weight is rebuilt in the backward instead of saved by autograd. The dequantized weight carries no
    gradient (the base is frozen), so grad flows only to `x` (and any LoRA upstream) and to the bias - the fp8
    dequant path's shape, from 4-bit storage.
    """
    return _NF4DequantLinear.apply(x, self._nf4_packed, self._nf4_state, self.bias)


def _dequantize_source_weight(module: nn.Linear) -> torch.Tensor:
    """Return the module's weight as bf16, dequantizing fp8 if needed."""
    w = module.weight.data
    if w.dtype == torch.float8_e4m3fn and getattr(module, "scale_weight", None) is not None:
        # fp8 base → bf16 (same math as fp8_linear_forward_patch's dequant)
        return (w.to(torch.float32) * module.scale_weight.to(torch.float32)).to(torch.bfloat16)
    return w.to(torch.bfloat16)


def apply_nf4_quantization(
    model: nn.Module,
    target_keys=_NF4_TARGET_DEFAULT,
    exclude_keys=(),
    compute_device: torch.device = torch.device("cuda"),
    compress_statistics: bool = False,
) -> int:
    """Quantize the frozen base-block Linears to NF4 in place.

    For each target Linear: dequantize its weight to bf16 (fp8→bf16 if needed),
    NF4-quantize it on `compute_device`, stash the packed weight + quant state on
    the module, FREE the original weight (so we actually save the memory), and
    patch `forward` to the dequant path. Done one Linear at a time so peak VRAM is
    ~(NF4 total so far) + one bf16 weight, never the whole bf16 model — which is
    what lets it run on a small card.

    Returns the number of Linears quantized.
    """
    from bitsandbytes.functional import quantize_nf4

    compute_device = torch.device(compute_device)
    count = 0
    freed_bytes = 0
    packed_bytes = 0

    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None):
            continue
        if not any(t in name for t in target_keys):
            continue
        if any(e in name for e in exclude_keys):
            continue

        w_bf16 = _dequantize_source_weight(module).to(compute_device).contiguous()
        packed, state = quantize_nf4(w_bf16, compress_statistics=compress_statistics)

        module._nf4_packed = packed
        module._nf4_state = state
        module._is_nf4 = True

        # Free the original full-precision weight — keep the Parameter object (so
        # in_features/out_features and the module identity stay intact for LoRA),
        # but drop its storage to a 0-element tensor.
        freed_bytes += module.weight.numel() * module.weight.element_size()
        module.weight.data = torch.empty(0, device=packed.device, dtype=torch.bfloat16)
        module.weight.requires_grad_(False)
        packed_bytes += packed.numel() * packed.element_size()

        # Bind the dequant forward (same mechanism as apply_fp8_monkey_patch).
        module.forward = nf4_linear_forward_patch.__get__(module, type(module))

        del w_bf16
        count += 1

    # Flag the model so callers can detect the 4-bit base (e.g. the sample path,
    # which must NOT block-swap an NF4 model — the weights live in _nf4_packed,
    # not .weight).
    if count > 0:
        model._nf4_quantized = True

    logger.info(
        f"NF4 quantization: {count} base Linears → 4-bit "
        f"(freed ~{freed_bytes/1e9:.2f} GB of fp8/bf16 weights, packed ~{packed_bytes/1e9:.2f} GB)."
    )
    return count


def _move_quant_state(state, device: torch.device):
    """Move a bitsandbytes QuantState's tensors to `device`. QuantState.to() is broken for the
    non-nested (compress_statistics=False) case in bnb 0.48.x — it dereferences a None state2 —
    so we move the constituent tensors ourselves. Recurses into the nested state2 when present."""
    for attr in ("absmax", "code", "offset"):
        t = getattr(state, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(state, attr, t.to(device))
    s2 = getattr(state, "state2", None)
    if s2 is not None:
        _move_quant_state(s2, device)


def move_nf4_to_device(model: nn.Module, device) -> int:
    """Move an NF4-quantized model's packed weights + quant state to `device`.

    `nn.Module.to()` leaves these behind — `_nf4_packed` and `_nf4_state` are plain attributes,
    not registered buffers/params — so `model.to("cpu")` frees only the (empty) `.weight` shells
    and the ~6 GB of 4-bit data stays put. To actually park an NF4 model on CPU for a preview
    (freeing that VRAM for the preview model) and restore it afterwards, call this explicitly.
    Returns the number of NF4 modules moved."""
    device = torch.device(device)
    n = 0
    for module in model.modules():
        if not getattr(module, "_is_nf4", False):
            continue
        packed = getattr(module, "_nf4_packed", None)
        if packed is not None:
            module._nf4_packed = packed.to(device)
        state = getattr(module, "_nf4_state", None)
        if state is not None:
            _move_quant_state(state, device)
        n += 1
    return n
