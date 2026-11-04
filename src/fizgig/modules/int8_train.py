"""INT8 (W8A8) matmul for a FROZEN training base — differentiable, unlike modules/int8.py.

modules/int8.py runs int8 under no_grad for previews. Training needs a gradient path: the LoRA
sits downstream of these Linears, so grad must flow *through* the frozen base to reach it.

Two things make this cheaper and safer here than a general W8A8 implementation:

1. **The base is frozen.** We never need grad w.r.t. the weight — only w.r.t. the input. Weight
   gradients are where int8 quantisation does the most damage, and we simply don't compute them.

2. **Gradient checkpointing is on by default**, so the forward runs TWICE per step (once
   forward, once recomputed inside backward). An int8 forward therefore covers two thirds of
   the matmul work even if the backward stays in bf16.

Hence `grad_mode`:
    "bf16"  – int8 forward, bf16 backward. Gradients are exact; ~2/3 of the win. Default.
    "int8"  – both quantised (what OneTrainer does). Faster, and the accuracy question is real:
              gradients have a far wider dynamic range than weights, and small updates are
              exactly what per-image LR and fine-tuning depend on.

The backward's weight scale is per-output-channel, and the backward reduces OVER that axis, so
it cannot be factored out after the matmul. It is folded into grad_output *before* quantising
instead — cheap, and exact.

Measured at Krea 2's main shape (M=1024, K=N=6144), RTX 5090:
    dequant + bf16 matmul (current)   0.767 ms
    _int_mm + activation quant        0.224 ms
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

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
    def forward(ctx, x, w_i8_nk, w_scale_1n, bias, grad_mode):
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
        ctx.grad_mode = grad_mode
        ctx.orig_shape = orig_shape
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out):
        w_i8_nk, w_scale_1n = ctx.saved_tensors
        g2d = grad_out.reshape(-1, grad_out.shape[-1])

        # grad_x[m,k] = sum_n grad_out[m,n] * w[k,n],  w[k,n] = w_i8[k,n] * w_scale[n].
        # The reduction runs over n, which is exactly the axis w_scale varies along, so the
        # scale cannot be applied after the matmul — fold it in first.
        if ctx.grad_mode == "int8" and not ctx._fell_back:
            gs = g2d * w_scale_1n.to(g2d.dtype)
            g_i8, g_scale = _quant_per_token(gs)
            try:
                # (M,N)@(N,K): w_i8_nk is already (N,K) contiguous — no transpose needed.
                # Same M > 16 row padding as the forward, for the same reason.
                m = g2d.shape[0]
                pad_rows = 17 - m if m <= 16 else 0
                if pad_rows:
                    g_i8 = torch.nn.functional.pad(g_i8, (0, 0, 0, pad_rows))
                acc = torch._int_mm(g_i8, w_i8_nk)                     # (M,K) int32
                if pad_rows:
                    acc = acc[:m]
                grad_x = acc.to(grad_out.dtype) * g_scale.to(grad_out.dtype)
            except Exception:
                w = w_i8_nk.to(grad_out.dtype) * w_scale_1n.reshape(-1, 1).to(grad_out.dtype)
                grad_x = g2d @ w
        else:
            # bf16 backward: exact gradients. Dequantise in the COMPUTE dtype — an fp32 (K,N)
            # copy of a 6144x6144 weight is 150 MB and made this path slower than dense bf16.
            w = w_i8_nk.to(grad_out.dtype) * w_scale_1n.reshape(-1, 1).to(grad_out.dtype)  # (N,K)
            grad_x = g2d @ w

        grad_bias = g2d.sum(dim=0).to(grad_out.dtype) if ctx.needs_bias_grad else None
        return grad_x.reshape(*ctx.orig_shape), None, None, grad_bias, None


def int8_train_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return _Int8FrozenLinear.apply(x, self.weight.data, self._int8_wscale, self.bias,
                                   getattr(self, "_int8_grad_mode", "bf16"))


def apply_int8_training(model: nn.Module, target_keys=("blocks.",), exclude_keys=(),
                        compute_device: torch.device = torch.device("cuda"),
                        grad_mode: str = "bf16") -> int:
    """Quantise target Linears to int8 and patch them onto the differentiable path.

    The int8 weight lives in `.weight.data` in the natural (N,K) orientation, so block swap
    streams it exactly as before and `.weight.shape` stays meaningful.
    """
    from fizgig.modules.nf4 import _dequantize_source_weight

    if grad_mode not in ("bf16", "int8"):
        raise ValueError(f"grad_mode must be 'bf16' or 'int8', got {grad_mode!r}")

    compute_device = torch.device(compute_device)
    count = 0
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None):
            continue
        if not any(t in name for t in target_keys) or any(e in name for e in exclude_keys):
            continue

        w = _dequantize_source_weight(module).to(compute_device).contiguous()      # (N,K)
        w_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0        # (N,1)
        w_i8 = (w / w_scale).round_().clamp_(-127, 127).to(torch.int8)              # (N,K)

        module.weight.requires_grad_(False)     # must clear before .data can hold int8
        # Store (N,K) CONTIGUOUS — the natural nn.Linear orientation — and pass .t() to
        # _int_mm. That transpose is a free view, and the resulting column-major layout is
        # what the int8 tensor cores want: 0.131 ms vs 0.452 ms for a (K,N)-contiguous
        # weight at 6144x6144. Storing "pre-transposed to avoid a transpose" is 3.4x SLOWER.
        module.weight.data = w_i8.contiguous()                                      # (N,K)
        module.register_buffer("_int8_wscale", w_scale.reshape(1, -1).to(torch.float32),
                               persistent=False)
        module._is_int8 = True
        module._int8_grad_mode = grad_mode
        module.forward = int8_train_forward.__get__(module, type(module))
        del w, w_i8
        count += 1

    if count:
        model._int8_quantized = True
    logger.info("INT8 training path: %d Linears -> W8A8 (grad_mode=%s)", count, grad_mode)
    return count
