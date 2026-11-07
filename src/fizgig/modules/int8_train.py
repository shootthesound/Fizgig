"""INT8 (W8A8) matmul for a FROZEN training base — differentiable, unlike modules/int8.py.

modules/int8.py runs int8 under no_grad for previews. Training needs a gradient path: the LoRA
sits downstream of these Linears, so grad must flow *through* the frozen base to reach it.

Two things make this cheaper and safer here than a general W8A8 implementation:

1. **The base is frozen.** We never need grad w.r.t. the weight — only w.r.t. the input. Weight
   gradients are where int8 quantisation does the most damage, and we simply don't compute them.

2. **Gradient checkpointing is on by default**, so the forward runs TWICE per step (once
   forward, once recomputed inside backward). An int8 forward therefore covers two thirds of
   the matmul work even if the backward stays in bf16.

The backward is bf16 (exact gradients, the weight dequantised in the compute dtype). An INT8 backward was
measured and removed (#221): rounding each gradient row to INT8 lets its outliers set the scale and rounds most
of the row to zero, so a LoRA's gradients came out at cosine 0.2-0.6 to the exact ones and likeness lagged.

Measured at Krea 2's main shape (M=1024, K=N=6144), RTX 5090:
    dequant + bf16 matmul (current)   0.767 ms
    _int_mm + activation quant        0.224 ms
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_WARNED_FALLBACK = False


def _quant_per_token(t: torch.Tensor):
    """Symmetric per-row int8 quantisation. Returns (int8 tensor, fp32 scale of shape (M,1))."""
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(torch.float32) / 127.0
    q = (t.to(torch.float32) / scale).round_().clamp_(-127, 127).to(torch.int8)
    return q, scale


class _Int8FrozenLinear(torch.autograd.Function):
    """y = (x @ W) with W frozen int8. Only grad_input is produced — the base never trains."""

    @staticmethod
    def forward(ctx, x, w_i8_nk, w_scale_1n, bias):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        a_i8, a_scale = _quant_per_token(x2d)
        try:
            # torch._int_mm refuses M <= 16. Rather than let those fall through to the fp32 path,
            # pad the rows with zeros and slice the result back: zero rows produce zero output and
            # are discarded, so this is exact. Without it, whether a matmul runs quantised depends
            # on how many tokens happen to be in the tensor — which made the same weights give
            # different answers for short captions vs long ones, silently.
            m = x2d.shape[0]
            pad_rows = 17 - m if m <= 16 else 0
            if pad_rows:
                a_i8 = torch.nn.functional.pad(a_i8, (0, 0, 0, pad_rows))
            acc = torch._int_mm(a_i8, w_i8_nk.t())                 # (M,N) int32, free view
            if pad_rows:
                acc = acc[:m]
            # Scale in the compute dtype: an fp32 (M,N) intermediate is 4x the bytes and this
            # step is bandwidth-bound, not arithmetic-bound.
            out = acc.to(x.dtype) * a_scale.to(x.dtype) * w_scale_1n.to(x.dtype)
            ctx._fell_back = False
        except Exception:
            # _int_mm requires M > 16 (and dislikes some alignments), so short sequences land
            # here and run in fp32 instead — MORE accurate than int8, not less, but it means
            # numerics depend on token count: the same weights give different answers either
            # side of that threshold. Worth knowing about, because it silently made a text
            # padding experiment look like a masking bug (see docs/PERF_ROADMAP.md), so say it
            # once rather than never.
            w = w_i8_nk.to(torch.float32) * w_scale_1n.reshape(-1, 1)
            out = (x2d.to(torch.float32) @ w.t()).to(x.dtype)
            ctx._fell_back = True
            # Global write + logging are ONLY legal in eager: dynamo rejects a module-global
            # mutation inside an autograd.Function outright, and this branch is rare enough
            # that a compiled run would train for hours before an odd shape finally took it
            # and died. is_compiling() is a trace-time constant, so the compiled graph simply
            # omits this block (the warning still fires on the first EAGER fallback).
            if not torch.compiler.is_compiling():
                global _WARNED_FALLBACK
                if not _WARNED_FALLBACK:
                    _WARNED_FALLBACK = True
                    logger.info("[int8] some matmuls fall back to fp32 (torch._int_mm needs M > 16, "
                                "got %d) — those run exact rather than quantised, so int8 numerics "
                                "vary with sequence length", x2d.shape[0])
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(w_i8_nk, w_scale_1n)
        ctx.orig_shape = orig_shape
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out):
        w_i8_nk, w_scale_1n = ctx.saved_tensors
        g2d = grad_out.reshape(-1, grad_out.shape[-1])
        # bf16 backward: exact gradients. Dequantise in the COMPUTE dtype - an fp32 (K,N) copy of a 6144x6144 weight
        # is 150 MB and made this path slower than dense bf16.
        w = w_i8_nk.to(grad_out.dtype) * w_scale_1n.reshape(-1, 1).to(grad_out.dtype)  # (N,K)
        grad_x = g2d @ w
        grad_bias = g2d.sum(dim=0).to(grad_out.dtype) if ctx.needs_bias_grad else None
        return grad_x.reshape(*ctx.orig_shape), None, None, grad_bias


def int8_weight_only_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Weight-only INT8: the int8 weight (and its per-channel scale) back to the activations' dtype for a plain matmul -
    the activations are never quantised. The memory of INT8 without its per-token activation rounding, for a model whose
    activations carry outliers that rounding crushes (Vimoo: mottled, flickering backgrounds). Slower than _int_mm."""
    w = self.weight.data.to(x.dtype) * self._int8_wscale.reshape(-1, 1).to(x.dtype)
    return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def int8_train_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return _Int8FrozenLinear.apply(x, self.weight.data, self._int8_wscale, self.bias)
