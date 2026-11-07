"""Muon — Momentum Orthogonalized by Newton-Schulz, with RMS-matched AdamW scaling.

Adapted from PyTorch's `torch.optim.Muon` (torch/optim/_muon.py: its Newton-Schulz constants, checks and
`adjust_lr` rules), BSD-3-Clause, Copyright (c) 2016- Facebook, Inc. and the PyTorch contributors; the
licence is in docs/THIRD_PARTY_NOTICES.md.


Used as a fallback when `torch.optim.Muon` is unavailable or when a parameter list includes
non-2D tensors (e.g. >2D convolutional weights flattened to 2D `(out, -1)` or 1D vectors).
When all parameters are 2D and `torch.optim.Muon` is present (PyTorch 2.10+), `create_optimizer`
uses `torch.optim.Muon` directly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _zeropower_batched(x: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float) -> Tensor:
    """_zeropower_via_newtonschulz for a stack of same-shaped matrices (B, m, n) at once: each matrix gets the same
    maths (its own norm, the same iterations), in a few batched multiplies instead of 3 * ns_steps per matrix. A LoRA
    is hundreds of small matrices, so the per-matrix loop was a third of a MOSS step (#221)."""
    a, b, c = ns_coefficients
    x = x.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(ns_steps):
        gram = x @ x.transpose(-2, -1)
        upd = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, upd, x, beta=a)
    return x.transpose(-2, -1) if tall else x


class BatchedMuon(torch.optim.Optimizer):
    """torch.optim.Muon's update (same momentum, Newton-Schulz and adjust_lr rules) for all-2D parameter lists, with
    the per-parameter work batched: momentum and weight decay as foreach ops, Newton-Schulz once per group of
    same-shaped matrices."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.1, momentum: float = 0.95,
                 nesterov: bool = True, ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
                 eps: float = EPS, ns_steps: int = DEFAULT_NS_STEPS, adjust_lr_fn: str | None = None) -> None:
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw", "spectral_unclamped"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps,
                                      adjust_lr_fn=adjust_lr_fn))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"BatchedMuon takes 2D parameters only, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mom = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            bufs = []
            for p, g in zip(ps, grads):
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                bufs.append(st["momentum_buffer"])
            torch._foreach_lerp_(bufs, grads, 1.0 - mom)
            ups = [g.lerp(b_, mom) for g, b_ in zip(grads, bufs)] if group["nesterov"] else bufs
            if wd != 0.0:
                torch._foreach_mul_(ps, 1.0 - lr * wd)
            by_shape = {}
            for i, p in enumerate(ps):
                by_shape.setdefault((tuple(p.shape), p.device, p.dtype), []).append(i)
            for (shape, _dev, dtype), idx in by_shape.items():
                ortho = _zeropower_batched(torch.stack([ups[i] for i in idx]), group["ns_coefficients"],
                                           int(group["ns_steps"]), float(group["eps"]))
                alr = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                torch._foreach_add_([ps[i] for i in idx], list(ortho.to(dtype).unbind(0)), alpha=-alr)
        return loss


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: tuple[int, int]) -> float:
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    elif adjust_lr_fn == "spectral_unclamped":
        adjusted_ratio = math.sqrt(A / B)
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(torch.optim.Optimizer):
    """Muon optimizer supporting 2D matrices (and >2D tensors reshaped to 2D, plus 1D fallback)."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = "match_rms_adamw",
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
            "spectral_unclamped",
        ):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = group["ns_coefficients"]
            eps = float(group["eps"])
            ns_steps = int(group["ns_steps"])
            adjust_lr_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                if p.ndim >= 2:
                    shape_2d = (p.size(0), p.numel() // p.size(0))
                    update_2d = update.reshape(shape_2d)
                    ortho = _zeropower_via_newtonschulz(update_2d, ns_coefficients, ns_steps, eps)
                    update = ortho.reshape(p.shape).to(p.dtype)
                    adjusted_lr = _adjust_lr(lr, adjust_lr_fn, shape_2d)
                else:
                    rms = update.float().pow(2).mean().sqrt().clamp(min=eps)
                    update = (update.float() / rms).to(p.dtype) * 0.2
                    adjusted_lr = lr

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-adjusted_lr)

        return loss
