# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import ast
import hashlib
import math
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of multi-head attention.
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # NOTE: rank is NOT capped at min(in, out), even though B@A can never exceed that rank.
        # A cap was added and then removed (4 Aug) after measuring what it costs: on MiniMax H3's
        # pruned AdaLN ([96768, 8]) capping rank 32 -> 8 cut that layer's learning to 27% of the
        # reference trainer's after one matched epoch, and AdaLN carries ~45% of all weight
        # movement there. The expressible SET is identical either way; the optimisation dynamics
        # are not — an over-complete factorisation trains markedly faster under Adam (implicit
        # acceleration of overparameterised linear networks). The extra columns are not dead
        # weight, they are the gradient path. Keep whatever rank the caller asked for.

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported for split_dims
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # for save/load

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
                for i in range(len(lxs)):
                    if len(lxs[i].size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lxs[i].size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # for reference
        self.enabled = True
        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)  # for calculation

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            up_weight = sd["lora_up.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)

            # merge weight
            if len(weight.size()) == 2:
                # linear
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + self.multiplier * conved * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype=dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight

                # merge weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)

    # return weight for merge
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def default_forward(self, x):
        # Inference only (the training class above is untouched): the epilogue is ONE
        # add with a scalar alpha instead of two bf16 multiplies and an add — three
        # elementwise kernels over [tokens, out] per module became one. On the H3 Repair
        # Studio that is ~1,000 module forwards per Dial render (measured 4 Sep: ~0.4 s of
        # a 3.8 s move). Same maths; multiplier x scale is formed once in fp32.
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            lx = torch.cat(lxs, dim=-1)
        m = self.multiplier
        if isinstance(m, torch.Tensor):                    # a tensor multiplier keeps the old path
            return self.org_forward(x) + lx * m * self.scale
        return torch.add(self.org_forward(x), lx, alpha=float(m) * float(self.scale))

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return just the LoRA contribution (base output NOT added) for a given
        input x. Used by the extractor's activation-capture hook so it works
        uniformly across standard LoRA and LyCORIS variants."""
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return lx * self.multiplier * self.scale
        lxs = [lora_down(x) for lora_down in self.lora_down]
        lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
        return torch.cat(lxs, dim=-1) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


# ---------------------------------------------------------------------------
# LyCORIS modules: LoKR (Kronecker-product) and LoHa (Hadamard-product)
#
# Quacks like LoRAInfModule — same `.lora_name` / `.enabled` / `.multiplier`
# / `.apply_to()` interface, so LoRANetwork's set_module_*_by_pattern, Repair
# Studio sliders, and the profiler all work unchanged. Parameters are named to
# match the file's key suffixes (lokr_w1, lokr_w2, hada_w1_a, ...) so
# load_state_dict populates them directly.
# ---------------------------------------------------------------------------

def lycoris_scale_from_keys(mod_keys: Dict[str, torch.Tensor]) -> float:
    """Effective scale for a LoKR/LoHa module's dense delta, mirroring
    LoKRInfModule / LoHaInfModule exactly. Bake and extraction materialize
    through this so the saved file matches what live inference showed —
    a divergent convention here silently rescales the whole LoRA.
    """
    alpha_t = mod_keys.get("alpha")
    alpha = float(alpha_t.item()) if alpha_t is not None else 1.0
    if alpha >= LoKRInfModule._ALPHA_SENTINEL_THRESHOLD:
        return 1.0  # refined-export sentinel → scale baked into the weights
    if mod_keys.get("hada_w1_a") is not None:  # LoHa
        r1 = int(mod_keys["hada_w1_a"].shape[1])
        r2 = int(mod_keys["hada_w2_a"].shape[1])
        return alpha / max(1, min(r1, r2))
    # LoKR: dim = min of the decomposition ranks that are actually in use;
    # a full-matrix LoKR (no w1_a/w2_a factors) has dim 1 → scale = alpha.
    dims = []
    if mod_keys.get("lokr_w1_a") is not None:
        dims.append(int(mod_keys["lokr_w1_a"].shape[1]))
    if mod_keys.get("lokr_w2_a") is not None:
        dims.append(int(mod_keys["lokr_w2_a"].shape[1]))
    dim = max(1, min(dims)) if dims else 1
    return alpha / dim


def _lokr_forward_update(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                        a: int, b: int, c: int, d: int) -> torch.Tensor:
    """Compute kron(w1, w2) @ x efficiently via the identity
       kron(A, B) @ vec(X) = vec(B @ X @ A.T)   (column-major vec)

    Shapes:  x (..., N=b·d)  →  out (..., M=a·c)
    """
    orig = x.shape[:-1]
    # x viewed as column-major vec(X) with X of shape (d, b).
    # Row-major x.reshape(b, d) gives X.T; transpose gives X of shape (d, b).
    X = x.reshape(*orig, b, d).transpose(-1, -2)              # (..., d, b)
    Y = w2 @ X @ w1.transpose(-1, -2)                          # (..., c, a)
    return Y.transpose(-1, -2).reshape(*orig, a * c)           # (..., M=a·c)


def factorization(n: int, factor: int) -> tuple:
    """Split n into (small, large) with small * large == n and small the LARGEST divisor
    of n that is <= factor — the LyCORIS convention, so a trained file's shapes look like
    every other LoKR in the wild. A prime n degenerates to (1, n), which is still a valid
    Kronecker factor (w1 becomes a row/column), not an error.
    """
    if factor < 1:
        factor = 1
    small = 1
    for cand in range(min(factor, n), 0, -1):
        if n % cand == 0:
            small = cand
            break
    return small, n // small


class LoKRModule(torch.nn.Module):
    """TRAINABLE LoKR (Kronecker-product) module — the training-side counterpart of
    LoKRInfModule below, full-matrix flavour: w1 (a,b) small, w2 (c,d) large, delta =
    multiplier · scale · kron(w1, w2), never materialized (see _lokr_forward_update).

    Where LoRA covers the weight with a rank-r slice, kron(w1, w2) tiles it with structure —
    the reason for LoKR's likeness-per-megabyte reputation. Full-matrix w2 is the quality
    configuration; `factor` is the single dial (w1 is ~factor x factor).

    Contract notes (create_modules calls all module classes identically):
    - signature matches LoRAModule positionally; `lora_dim`/`alpha` args are accepted and
      IGNORED — factor rules. `.lora_dim` is set to factor so logs stay meaningful.
    - alpha buffer is 1.0 and scale is 1.0: the one value that means the same thing under
      LoKRInfModule.__init__ (full-matrix path: dim 1 → scale = alpha) and
      lycoris_scale_from_keys — so the saved file renders identically everywhere.
    - init mirrors LoRAModule's zeroed lora_up: w1 kaiming, w2 zeros → delta exactly 0 at
      step 0, and grads flow to BOTH factors from step 1 (d/dw2 kron(w1,w2) ≠ 0 when w1 ≠ 0).
    - `apply_to` must `del self.org_module`, or the frozen base Linear's weight would be
      registered as a submodule — saved into the LoRA file and cast/moved by network.to().
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        factor=8,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError(f"LoKR training supports Linear modules only, got Conv2d at {lora_name}")
        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.factor = int(factor)
        a, c = factorization(out_dim, self.factor)   # out = a·c, a small
        b, d = factorization(in_dim, self.factor)    # in  = b·d, b small
        self.a, self.b, self.c, self.d = a, b, c, d

        self.lokr_w1 = torch.nn.Parameter(torch.empty(a, b))
        self.lokr_w2 = torch.nn.Parameter(torch.empty(c, d))
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lokr_w2)

        self.register_buffer("alpha", torch.tensor(1.0))
        self.scale = 1.0
        self.multiplier = multiplier
        self.enabled = True
        self.lora_dim = self.factor  # for create_modules' verbose log; not a rank
        self.split_dims = None
        self.org_module = org_module  # remove in applying

        # rank/neuron dropout act on a rank axis kron doesn't have; module_dropout still applies.
        if dropout is not None or rank_dropout is not None:
            logger.warning(f"LoKR: dropout/rank_dropout are not applicable and ignored ({lora_name})")
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        if not self.enabled or self.multiplier == 0.0:
            return org_forwarded
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded
        update = _lokr_forward_update(x, self.lokr_w1, self.lokr_w2,
                                      self.a, self.b, self.c, self.d)
        return org_forwarded + update * self.multiplier * self.scale


class LoKRInfModule(torch.nn.Module):
    """Inference-time LoKR module. Stores w1 (a,b) and w2 (c,d) — or their
    low-rank factors w1_a (a,k1) w1_b (k1,b) / w2_a (c,k2) w2_b (k2,d) —
    and applies update = multiplier · scale · kron(w1, w2) @ x additively.

    Scale handling: if alpha looks like a sentinel (very large, e.g. 1e10
    used by Comfy-Realtime-Lora refined exports to mean "scale is already
    baked into the weights"), scale = 1.0. Otherwise scale = alpha / dim
    following the standard LyCORIS convention.
    """

    _ALPHA_SENTINEL_THRESHOLD = 1e6  # alpha >= this → treat as "scale = 1.0"

    def __init__(self, lora_name: str, org_module: torch.nn.Module,
                 multiplier: float, alpha: float,
                 a: int, b: int, c: int, d: int,
                 k1: Optional[int] = None, k2: Optional[int] = None):
        super().__init__()
        self.lora_name = lora_name
        self.a, self.b, self.c, self.d = a, b, c, d
        self.k1, self.k2 = k1, k2
        self.w1_decomposed = k1 is not None
        self.w2_decomposed = k2 is not None

        # Placeholder parameters — real values come from load_state_dict().
        if self.w1_decomposed:
            self.lokr_w1_a = torch.nn.Parameter(torch.zeros(a, k1))
            self.lokr_w1_b = torch.nn.Parameter(torch.zeros(k1, b))
        else:
            self.lokr_w1 = torch.nn.Parameter(torch.zeros(a, b))
        if self.w2_decomposed:
            self.lokr_w2_a = torch.nn.Parameter(torch.zeros(c, k2))
            self.lokr_w2_b = torch.nn.Parameter(torch.zeros(k2, d))
        else:
            self.lokr_w2 = torch.nn.Parameter(torch.zeros(c, d))

        # Alpha as a buffer (matches standard LoRA convention; saves in state_dict).
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        if alpha >= self._ALPHA_SENTINEL_THRESHOLD:
            self.scale = 1.0  # refined-export sentinel → scale baked in
        else:
            # LyCORIS convention: scale = alpha / dim, dim = min of active rank dims
            dim_candidates = []
            if self.w1_decomposed and k1 is not None:
                dim_candidates.append(k1)
            if self.w2_decomposed and k2 is not None:
                dim_candidates.append(k2)
            if not dim_candidates:
                # No decomposition — use a safe dim of 1 so scale = alpha (LyCORIS default).
                dim_candidates.append(1)
            dim = max(1, min(dim_candidates))
            self.scale = float(alpha) / dim

        self.multiplier = float(multiplier)
        self.enabled = True

        # LoRAInfModule compat fields so profiler / extractor code that queries
        # them doesn't crash when given a LoKR module.
        self.lora_dim = k1 if k1 is not None else a
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.split_dims = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _w1(self) -> torch.Tensor:
        return self.lokr_w1_a @ self.lokr_w1_b if self.w1_decomposed else self.lokr_w1

    def _w2(self) -> torch.Tensor:
        return self.lokr_w2_a @ self.lokr_w2_b if self.w2_decomposed else self.lokr_w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org = self.org_forward(x)
        if not self.enabled or self.multiplier == 0.0:
            return org
        update = _lokr_forward_update(x, self._w1(), self._w2(),
                                      self.a, self.b, self.c, self.d)
        return org + self.multiplier * self.scale * update

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """LoKR contribution for input x (base NOT included)."""
        update = _lokr_forward_update(x, self._w1(), self._w2(),
                                      self.a, self.b, self.c, self.d)
        return self.multiplier * self.scale * update

    # Inference merge — materializes kron(w1, w2) once and adds to base weight.
    # Not fast but only done once per LoRA load via pipeline.load_lora.
    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype, org_device = weight.dtype, weight.device
        w = weight.to(device, dtype=torch.float, non_blocking=non_blocking)
        w1 = self._w1().to(device, dtype=torch.float)
        w2 = self._w2().to(device, dtype=torch.float)
        kron = torch.kron(w1, w2)  # (a·c, b·d) = (out, in)
        if kron.shape != w.shape:
            logger.warning(f"{self.lora_name}: kron shape {list(kron.shape)} != weight shape {list(w.shape)}, skipping merge")
            return
        w = w + self.multiplier * self.scale * kron
        org_sd["weight"] = w.to(org_device, dtype=org_dtype if dtype is None else dtype)
        self.org_module.load_state_dict(org_sd)


class LoHaInfModule(torch.nn.Module):
    """Inference-time LoHa module. Stores w1 = w1_a @ w1_b and w2 = w2_a @ w2_b,
    applies update = multiplier · scale · (W1 ⊙ W2) @ x additively.

    Unlike LoKR, LoHa can't avoid materializing W1 and W2 each forward (the
    Hadamard product doesn't factor). Per-call cost is modest (~2× standard
    LoRA forward) and peak VRAM stays the same as standard LoRA because the
    materialized tensors are released after the update is added to org output.
    """

    _ALPHA_SENTINEL_THRESHOLD = 1e6

    def __init__(self, lora_name: str, org_module: torch.nn.Module,
                 multiplier: float, alpha: float,
                 out_dim: int, in_dim: int, r1: int, r2: int):
        super().__init__()
        self.lora_name = lora_name
        self.out_dim, self.in_dim = out_dim, in_dim
        self.r1, self.r2 = r1, r2

        self.hada_w1_a = torch.nn.Parameter(torch.zeros(out_dim, r1))
        self.hada_w1_b = torch.nn.Parameter(torch.zeros(r1, in_dim))
        self.hada_w2_a = torch.nn.Parameter(torch.zeros(out_dim, r2))
        self.hada_w2_b = torch.nn.Parameter(torch.zeros(r2, in_dim))

        self.register_buffer("alpha", torch.tensor(float(alpha)))
        if alpha >= self._ALPHA_SENTINEL_THRESHOLD:
            self.scale = 1.0
        else:
            self.scale = float(alpha) / max(1, min(r1, r2))

        self.multiplier = float(multiplier)
        self.enabled = True
        self.lora_dim = min(r1, r2)
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.split_dims = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org = self.org_forward(x)
        if not self.enabled or self.multiplier == 0.0:
            return org
        W1 = self.hada_w1_a @ self.hada_w1_b    # (out_dim, in_dim)
        W2 = self.hada_w2_a @ self.hada_w2_b    # (out_dim, in_dim)
        W = W1 * W2                              # Hadamard product — same shape
        # x @ W.T: for x of shape (..., in_dim), produces (..., out_dim).
        return org + self.multiplier * self.scale * (x @ W.transpose(-1, -2))

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """LoHa contribution for input x (base NOT included)."""
        W1 = self.hada_w1_a @ self.hada_w1_b
        W2 = self.hada_w2_a @ self.hada_w2_b
        W = W1 * W2
        return self.multiplier * self.scale * (x @ W.transpose(-1, -2))

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype, org_device = weight.dtype, weight.device
        w = weight.to(device, dtype=torch.float, non_blocking=non_blocking)
        W1 = (self.hada_w1_a.to(device, torch.float) @ self.hada_w1_b.to(device, torch.float))
        W2 = (self.hada_w2_a.to(device, torch.float) @ self.hada_w2_b.to(device, torch.float))
        W = W1 * W2
        if W.shape != w.shape:
            logger.warning(f"{self.lora_name}: loha W shape {list(W.shape)} != weight shape {list(w.shape)}, skipping merge")
            return
        w = w + self.multiplier * self.scale * W
        org_sd["weight"] = w.to(org_device, dtype=org_dtype if dtype is None else dtype)
        self.org_module.load_state_dict(org_sd)


# ---------------------------------------------------------------------------
# Hash utilities for safetensors metadata
# ---------------------------------------------------------------------------

def _addnet_hash_legacy(b: BytesIO) -> str:
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()
    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def _addnet_hash_safetensors(b: BytesIO) -> str:
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def _precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""
    import safetensors.torch

    # Only retain training metadata for hash calculation (meant to be immutable)
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    raw_bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(raw_bytes)

    model_hash = _addnet_hash_safetensors(b)
    legacy_hash = _addnet_hash_legacy(b)
    return model_hash, legacy_hash


# ---------------------------------------------------------------------------
# Network creation functions
# ---------------------------------------------------------------------------

def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    module_class: Type[object] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Architecture-independent network creation."""
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # regular expression for module selection: exclude and include
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    if module_class is None:
        module_class = LoRAModule

    network = LoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


class LoRANetwork(torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Optional[Union[List[nn.Module], nn.Module]],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        module_kwargs: Optional[Dict[str, Any]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        self.module_kwargs = module_kwargs or {}

        self.loraplus_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        # compile regular expression if specified
        exclude_re_patterns = []
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")
                    continue
                exclude_re_patterns.append(re_pattern)

        include_re_patterns = []
        if include_patterns is not None:
            for pattern in include_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")
                    continue
                include_re_patterns.append(re_pattern)

        # create module instances
        def create_modules(
            is_unet: bool,
            pfx: str,
            root_module: torch.nn.Module,
            target_replace_mods: Optional[List[str]] = None,
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                    if target_replace_mods is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        # bitsandbytes 4-bit/8-bit linears are Linears for LoRA purposes: they
                        # expose in_features/out_features and LoRAModule only wraps their
                        # forward. Needed to LoRA-train on an NF4-quantized frozen base
                        # (MiniMax H3's 33B DiT), which Fizgig loads as Linear4bit.
                        # Match by CLASS NAME, so every quantized Linear stand-in must be listed
                        # here or its modules are silently skipped — a LoRA that trains a
                        # fraction of what was asked for, with no error. ConvRotInt8Linear /
                        # Nvfp4Linear (MiniMax H3's int8-ConvRot and nvfp4 bases) were exactly
                        # that: 58 of 258 modules wrapped. An isinstance(nn.Linear) test would
                        # be more robust; the name list is kept for Conv2d symmetry, so ADD TO
                        # IT when introducing a Linear subclass.
                        is_linear = child_module.__class__.__name__ in (
                            "Linear", "Linear4bit", "Linear8bitLt",
                            "ConvRotInt8Linear", "Nvfp4Linear", "HQQ4bitLinear")
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{pfx}.{original_name}".replace(".", "_")

                            # exclude/include filter.
                            # - exclude_patterns drop matching modules.
                            # - include_patterns, when present, RESTRICT training to matching
                            #   modules (this is what the GUI's "Model Area to Train" relies on).
                            # - a module matching both is kept (include overrides exclude).
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.fullmatch(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.fullmatch(original_name):
                                    included = True
                                    break
                            if not included and (excluded or include_re_patterns):
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # filter by name (not used in the current implementation)
                            if filter is not None and filter not in lora_name:
                                continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha.get(lora_name, dim)
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                **self.module_kwargs,
                            )
                            loras.append(lora)

                    if target_replace_mods is None:
                        break  # all modules are searched
            return loras, skipped

        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []

        # create LoRA for U-Net / DiT
        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, prefix, unet, target_replace_modules)

        # A block-targeted network that matched nothing must fail loudly: training would
        # otherwise run to completion and save a LoRA containing no weights.
        if include_re_patterns and unet is not None and len(self.unet_loras) == 0:
            raise ValueError(
                f"include_patterns matched no modules — nothing would train. "
                f"Patterns: {[p.pattern for p in include_re_patterns]}"
            )

        logger.info(f"create LoRA for U-Net/DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped / dim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def prepare_network(self, args):
        """
        called after the network is created
        """
        pass

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def set_module_enabled_by_pattern(self, pattern: str, enabled: bool, target: str = "unet"):
        """Enable/disable LoRA modules whose lora_name matches the regex pattern.

        Args:
            pattern: Regex pattern to match against lora_name.
            enabled: True to enable matching modules, False to disable.
            target: "unet", "text_encoder", or "all".
        """
        import re
        regex = re.compile(pattern)
        targets = []
        if target in ("unet", "all"):
            targets.extend(self.unet_loras)
        if target in ("text_encoder", "all"):
            targets.extend(self.text_encoder_loras)

        count = 0
        for lora in targets:
            if regex.search(lora.lora_name):
                lora.enabled = enabled
                count += 1
        return count

    def set_module_multiplier_by_pattern(self, pattern: str, multiplier: float, target: str = "unet"):
        """Set multiplier on LoRA modules whose lora_name matches the regex pattern.

        Args:
            pattern: Regex pattern to match against lora_name.
            multiplier: Multiplier value to set.
            target: "unet", "text_encoder", or "all".
        """
        import re
        regex = re.compile(pattern)
        targets = []
        if target in ("unet", "all"):
            targets.extend(self.unet_loras)
        if target in ("text_encoder", "all"):
            targets.extend(self.text_encoder_loras)

        count = 0
        for lora in targets:
            if regex.search(lora.lora_name):
                lora.multiplier = multiplier
                count += 1
        return count

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(
        self,
        text_encoders: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        if len(self.text_encoder_loras) == 0 and len(self.unet_loras) == 0:
            # Diagnostic dump to help identify the mismatch source
            unet_classes = set()
            if unet is not None:
                for _n, _m in unet.named_modules():
                    unet_classes.add(_m.__class__.__name__)
            target_list = getattr(self, "target_replace_modules", None) or []
            target_matches = [c for c in target_list if c in unet_classes]
            logger.error(
                "No LoRA modules found.\n"
                "  target_replace_modules: %s\n"
                "  matches found in DiT:   %s\n"
                "Likely cause: the LoRA's weight keys don't match Fizgig's lora_unet_<block>_<linear> naming, "
                "OR the DiT's block classes were renamed, OR the LoRA was trained against a different model.",
                target_list or "<unknown>",
                target_matches or "NONE — no DoubleStreamBlock/SingleStreamBlock found in DiT!",
            )
            raise RuntimeError("No LoRA modules found")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for lora in self.text_encoder_loras + self.unet_loras:
                sd_for_lora = {}
                for key in weights_sd.keys():
                    if key.startswith(lora.lora_name):
                        sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
                if len(sd_for_lora) == 0:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue

                futures.append(executor.submit(lora.merge_to, sd_for_lora, dtype, device, non_blocking))

        for future in futures:
            future.result()

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            if loraplus_ratio is not None and len(param_groups["plus"]) == 0:
                logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        if self.unet_loras:
            params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            if metadata is None:
                metadata = {}
            # The sshs_* hashes serialize the ENTIRE state dict into memory a second time —
            # at the end of an epoch that is the moment RAM is tightest, and on constrained
            # boxes it raised MemoryError (or a safetensors-rust PanicException) BEFORE the
            # checkpoint was written, killing the run over optional indexing metadata
            # (#92, diagnosed by David Maybank on a 12 GB / 64 GB box). Optional metadata
            # must never cost a checkpoint: on those two failures, skip the hashes and save;
            # anything else is a real bug and still raises.
            try:
                model_hash, legacy_hash = _precalculate_safetensors_hashes(state_dict, metadata)
                metadata["sshs_model_hash"] = model_hash
                metadata["sshs_legacy_hash"] = legacy_hash
            except BaseException as e:
                if not (isinstance(e, MemoryError)
                        or type(e).__name__ == "PanicException"):
                    raise
                logger.warning("skipping optional sshs_* hash metadata — not enough memory "
                               "to compute it (%s). The checkpoint itself saves normally.",
                               type(e).__name__)

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()

        # guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not any("lora_down" in k and "weight" in k for k in state_dict.keys()):
            logger.warning("max_norm_regularization is only supported for LoRA")
            return 0, 0.0, 0.0

        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


# =====================================================================================
# LoRA format compatibility — PEFT / diffusers → Fizgig/kohya conversion
# =====================================================================================

def detect_lora_format(weights_sd: Dict[str, torch.Tensor]) -> str:
    """Detect the naming convention of a LoRA state dict.

    Returns one of:
      - "kohya"   : kohya-style (lora_unet_*, .lora_down/up, .alpha) — what Fizgig emits
      - "peft"    : PEFT/diffusers style (diffusion_model.*.lora_A/B.weight)
      - "lokr"    : LoKR (Kronecker-product) — uses .lokr_w1 / .lokr_w2
      - "loha"    : LoHa (Hadamard-product) — uses .hada_w1_a / .hada_w2_a etc.
      - "glora"   : GLoRA — uses .glora_a / .glora_b / .glora_c / .glora_d
      - "unknown" : no supported signature matched

    LoKR/LoHa/GLoRA are low-rank parametrizations different from standard LoRA;
    Fizgig's LoRAInfModule only implements the standard `up @ down` form, so
    these variants are detected here for the loader to reject cleanly.
    """
    has_kohya = False
    has_peft = False
    for k in weights_sd.keys():
        kl = k.lower()
        # Check exotic formats first — they often have a 'diffusion_model.' prefix
        # too (e.g. lycoris-exports from ComfyUI-Realtime-Lora), so detecting these
        # before peft avoids mis-classification.
        if ".lokr_w1" in kl or ".lokr_w2" in kl:
            return "lokr"
        if ".hada_w1_a" in kl or ".hada_w2_a" in kl or ".hada_w1_b" in kl or ".hada_w2_b" in kl:
            return "loha"
        if ".glora_a" in kl or ".glora_b" in kl or ".glora_c" in kl or ".glora_d" in kl:
            return "glora"
        if k.startswith("lora_unet_") and (".lora_down" in k or ".lora_up" in k or ".alpha" in k or ".lora_A" in k or ".lora_B" in k):
            has_kohya = True
        # OneTrainer legacy format: lora_transformer_ prefix with lora_down/lora_up
        if k.startswith("lora_transformer_") and (".lora_down" in k or ".lora_up" in k or ".alpha" in k):
            has_kohya = True
        if k.startswith("diffusion_model.") and (".lora_A." in k or ".lora_B." in k or ".lora_down" in k or ".lora_up" in k):
            has_peft = True
        if k.startswith("transformer.") and (".lora_A." in k or ".lora_B." in k or ".lora_down" in k or ".lora_up" in k):
            has_peft = True  # AI-Toolkit / OneTrainer OMI format
        if "." in k and not k.startswith(("lora_unet_", "lora_transformer_")) \
                and (".lora_A." in k or ".lora_B." in k):
            has_peft = True  # bare dotted paths, no prefix — MiniMax H3 ecosystem exports
    if has_kohya and not has_peft:
        return "kohya"
    if has_peft and not has_kohya:
        return "peft"
    if has_kohya and has_peft:
        return "kohya"  # mixed — loader can consume kohya keys
    return "unknown"


# ---------------------------------------------------------------------------
# Diffusers → Klein key conversion for Flux LoRAs (OneTrainer, etc.)
# Maps diffusers naming (transformer_blocks, attn.to_q, etc.) to Klein's
# internal naming (double_blocks, img_attn.qkv, etc.) including QKV fusion.
# Reference: ComfyUI comfy/utils.py::flux_to_diffusers()
# ---------------------------------------------------------------------------

# 1:1 mappings (no QKV fusion needed)
_DIFFUSERS_DOUBLE_BLOCK_MAP = {
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.net.0.proj": "img_mlp.0",
    "ff.net.2": "img_mlp.2",
    "ff_context.net.0.proj": "txt_mlp.0",
    "ff_context.net.2": "txt_mlp.2",
    "norm1.linear": "img_mod.lin",
    "norm1_context.linear": "txt_mod.lin",
    # LyCORIS / LoKR variants
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}

_DIFFUSERS_SINGLE_BLOCK_MAP = {
    "proj_out": "linear2",
    "norm.linear": "modulation.lin",
    "attn.to_out": "linear2",          # Flux 2 variant
    "attn.to_qkv_mlp_proj": "linear1", # Flux 2 fused variant
}

# QKV fusion groups: diffusers separate keys → Klein fused key
# (diffusers_suffix, klein_target, slot_index)
_DIFFUSERS_DOUBLE_QKV_IMG = [
    ("attn.to_q", "img_attn.qkv", 0),
    ("attn.to_k", "img_attn.qkv", 1),
    ("attn.to_v", "img_attn.qkv", 2),
]
_DIFFUSERS_DOUBLE_QKV_TXT = [
    ("attn.add_q_proj", "txt_attn.qkv", 0),
    ("attn.add_k_proj", "txt_attn.qkv", 1),
    ("attn.add_v_proj", "txt_attn.qkv", 2),
]
_DIFFUSERS_SINGLE_QKV = [
    ("attn.to_q", "linear1", 0),
    ("attn.to_k", "linear1", 1),
    ("attn.to_v", "linear1", 2),
    ("proj_mlp", "linear1", 3),  # MLP projection fused into linear1 slot 3
]


# Diffusers-name -> Fizgig-native renames for Krea 2, applied to FLATTENED kohya module names.
# Straight from OneTrainer's own diffusers_to_original table (modules/model/Krea2Model.py) —
# q/k/v are split in BOTH namespaces, so unlike Flux there is no QKV fusion stage, only renames.
# Order matters: longest/most-specific first, so _attn_to_out_0 wins before _attn_to_o could.
_KREA2_DIFFUSERS_RENAMES = (
    ("_attn_to_out_0", "_attn_wo"),
    ("_attn_to_gate", "_attn_gate"),
    ("_attn_to_q", "_attn_wq"),
    ("_attn_to_k", "_attn_wk"),
    ("_attn_to_v", "_attn_wv"),
    ("_ff_up", "_mlp_up"),
    ("_ff_gate", "_mlp_gate"),
    ("_ff_down", "_mlp_down"),
)


def _is_diffusers_krea2_lora(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Krea 2 in diffusers naming (OneTrainer, AI-Toolkit, diffusers exports).

    Must be checked BEFORE the Flux converter: both architectures use `transformer_blocks.` in
    diffusers form, and the Flux converter would silently map a Krea 2 file onto
    double_blocks/single_blocks names that do not exist there. The discriminators are modules
    only Krea 2 has: the gated-MLP `ff.up/gate/down` (Flux's MLP is `ff.net.*`), the attention
    output gate `attn.to_gate`, and the `text_fusion.` tower.
    """
    for k in weights_sd:
        kk = k.replace("_", ".")        # catch the pre-flattened legacy form too
        if (".ff.up." in kk or ".ff.gate." in kk or ".ff.down." in kk
                or ".attn.to.gate." in kk or "text.fusion." in kk):
            return True
    return False


def _krea2_diffusers_to_native(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename flattened diffusers-Krea 2 module names to the native ones our KreaDiT exposes.

    Runs AFTER prefix stripping + dot flattening, so every key looks like
    `lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight`. Without this pass none of
    those names exist in the model and create_network_from_weights builds zero modules —
    "No LoRA modules found" (issue #50, OneTrainer Krea 2 LoRAs).
    """
    out: Dict[str, torch.Tensor] = {}
    n = 0
    for k, v in weights_sd.items():
        mod, dot, rest = k.partition(".")
        new_mod = mod
        new_mod = re.sub(r"^lora_unet_transformer_blocks_(\d+)_", lambda m_: "lora_unet_blocks_" + m_.group(1) + "_", new_mod)
        new_mod = new_mod.replace("lora_unet_text_fusion_", "lora_unet_txtfusion_")
        for a, b in _KREA2_DIFFUSERS_RENAMES:
            if a in new_mod:
                new_mod = new_mod.replace(a, b)
        # IO layers, for completeness (rarely trained, cheap to map): img_in -> first,
        # final_layer.linear -> last.linear, txt_in linears -> the txtmlp Sequential slots.
        new_mod = (new_mod
                   .replace("lora_unet_img_in", "lora_unet_first")
                   .replace("lora_unet_final_layer_linear", "lora_unet_last_linear")
                   .replace("lora_unet_txt_in_linear_1", "lora_unet_txtmlp_1")
                   .replace("lora_unet_txt_in_linear_2", "lora_unet_txtmlp_3")
                   .replace("lora_unet_time_embed_linear_1", "lora_unet_tmlp_0")
                   .replace("lora_unet_time_embed_linear_2", "lora_unet_tmlp_2"))
        if new_mod != mod:
            n += 1
        out[new_mod + dot + rest] = v
    logger.info("Krea 2 diffusers naming detected — renamed %d key(s) to native module names.", n)
    return out


def _is_diffusers_flux_lora(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Check if the state dict uses diffusers-style Flux key names."""
    for k in weights_sd:
        if "transformer_blocks." in k or "single_transformer_blocks." in k:
            return True
    return False


def _convert_diffusers_flux_lora(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a diffusers-format Flux LoRA to kohya convention.

    Handles QKV fusion: separate Q/K/V LoRA weights are concatenated into
    a single fused LoRA module targeting Klein's fused QKV linear.
    """
    converted: Dict[str, torch.Tensor] = {}
    # Collect QKV groups for fusion: {klein_module_name: {slot: {suffix: tensor}}}
    qkv_pending: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}

    # Strip transformer. prefix if present
    stripped_sd = {}
    for k, v in weights_sd.items():
        if k.startswith("transformer."):
            stripped_sd[k[len("transformer."):]] = v
        else:
            stripped_sd[k] = v

    for key, tensor in stripped_sd.items():
        # Determine suffix type
        if key.endswith(".lora_down.weight"):
            suffix = "lora_down.weight"
            module_path = key[:-len(".lora_down.weight")]
        elif key.endswith(".lora_up.weight"):
            suffix = "lora_up.weight"
            module_path = key[:-len(".lora_up.weight")]
        elif key.endswith(".lora_A.weight"):
            suffix = "lora_down.weight"
            module_path = key[:-len(".lora_A.weight")]
        elif key.endswith(".lora_B.weight"):
            suffix = "lora_up.weight"
            module_path = key[:-len(".lora_B.weight")]
        elif key.endswith(".alpha"):
            suffix = "alpha"
            module_path = key[:-len(".alpha")]
        elif key.endswith(".dora_scale"):
            continue
        else:
            converted[key] = tensor
            continue

        # Parse block type and index
        klein_module = None
        is_qkv = False

        # Double blocks
        if module_path.startswith("transformer_blocks."):
            rest = module_path[len("transformer_blocks."):]
            dot = rest.find(".")
            if dot == -1:
                continue
            idx = rest[:dot]
            layer_path = rest[dot + 1:]
            block_prefix = f"double_blocks.{idx}"

            # Check QKV fusion groups
            for diff_name, klein_target, slot in _DIFFUSERS_DOUBLE_QKV_IMG + _DIFFUSERS_DOUBLE_QKV_TXT:
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_target}"
                    qkv_pending.setdefault(klein_module, {}).setdefault(slot, {})[suffix] = tensor
                    is_qkv = True
                    break
            if is_qkv:
                continue

            # Check 1:1 mappings
            for diff_name, klein_name in _DIFFUSERS_DOUBLE_BLOCK_MAP.items():
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_name}"
                    break

        # Single blocks
        elif module_path.startswith("single_transformer_blocks."):
            rest = module_path[len("single_transformer_blocks."):]
            dot = rest.find(".")
            if dot == -1:
                # Could be proj_mlp (no dot after index)
                # e.g. single_transformer_blocks.0  — shouldn't happen with a layer suffix
                continue
            idx = rest[:dot]
            layer_path = rest[dot + 1:]
            block_prefix = f"single_blocks.{idx}"

            # Check QKV fusion groups
            for diff_name, klein_target, slot in _DIFFUSERS_SINGLE_QKV:
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_target}"
                    qkv_pending.setdefault(klein_module, {}).setdefault(slot, {})[suffix] = tensor
                    is_qkv = True
                    break
            if is_qkv:
                continue

            # proj_mlp has no dot separator in the layer path for single blocks
            if layer_path == "proj_mlp" or module_path.endswith(".proj_mlp"):
                klein_module = f"{block_prefix}.linear1"
                qkv_pending.setdefault(klein_module, {}).setdefault(3, {})[suffix] = tensor
                continue

            # Check 1:1 mappings
            for diff_name, klein_name in _DIFFUSERS_SINGLE_BLOCK_MAP.items():
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_name}"
                    break

        if klein_module is not None and not is_qkv:
            flat = klein_module.replace(".", "_")
            lora_name = f"lora_unet_{flat}"
            converted[f"{lora_name}.{suffix}"] = tensor

    # Fuse QKV groups
    for klein_module, slots in qkv_pending.items():
        flat = klein_module.replace(".", "_")
        lora_name = f"lora_unet_{flat}"

        # Collect lora_down and lora_up from each slot, sorted by slot index
        sorted_slots = sorted(slots.keys())
        downs = [slots[s]["lora_down.weight"] for s in sorted_slots if "lora_down.weight" in slots[s]]
        ups = [slots[s]["lora_up.weight"] for s in sorted_slots if "lora_up.weight" in slots[s]]

        if not downs or not ups:
            # Incomplete QKV group — emit whatever we have as-is
            for s in sorted_slots:
                for suf, t in slots[s].items():
                    converted[f"{lora_name}_slot{s}.{suf}"] = t
            continue

        # All Q/K/V lora_down: (rank, in_dim) each → cat along dim 0 → (N*rank, in_dim)
        fused_down = torch.cat(downs, dim=0)

        # All Q/K/V lora_up: (out_dim, rank) each → build block-diagonal
        # so each slot's output lands in the right slice of the fused output.
        # Each slot's own alpha/rank scale is baked into its columns here, and the
        # fused module ships alpha = total_rank (scale 1.0) — so a source exporting
        # e.g. alpha = rank/2 (OneTrainer, ai-toolkit) keeps its true strength
        # instead of being inflated to scale 1.0.
        slot_ups = [(s, slots[s]["lora_up.weight"]) for s in sorted_slots
                    if "lora_up.weight" in slots[s]]
        ranks = [u.shape[1] for _, u in slot_ups]
        out_dims = [u.shape[0] for _, u in slot_ups]
        total_rank = sum(ranks)
        total_out = sum(out_dims)

        fused_up = torch.zeros(total_out, total_rank, dtype=slot_ups[0][1].dtype)
        r_offset = 0
        o_offset = 0
        for i, (s, up) in enumerate(slot_ups):
            alpha_t = slots[s].get("alpha")
            # PEFT convention when alpha is absent: alpha = rank → scale 1.0
            slot_scale = (float(alpha_t) / ranks[i]) if alpha_t is not None else 1.0
            if slot_scale != 1.0:
                up = (up.to(torch.float32) * slot_scale).to(up.dtype)
            fused_up[o_offset:o_offset + out_dims[i], r_offset:r_offset + ranks[i]] = up
            r_offset += ranks[i]
            o_offset += out_dims[i]

        converted[f"{lora_name}.lora_down.weight"] = fused_down
        converted[f"{lora_name}.lora_up.weight"] = fused_up
        converted[f"{lora_name}.alpha"] = torch.tensor(float(total_rank))

    n_modules = len([k for k in converted if k.endswith(".lora_down.weight")])
    logger.info(f"[diffusers→kohya] converted {len(weights_sd)} keys → {len(converted)} "
                f"({n_modules} LoRA modules, QKV groups fused)")
    return converted


def peft_to_kohya(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a PEFT/diffusers-format LoRA (or LyCORIS-prefixed LoKR/LoHa) state
    dict to Fizgig/kohya module-name convention.

    Key mapping (standard LoRA):
      diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
        → lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
      diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight
        → lora_unet_double_blocks_0_img_attn_proj.lora_up.weight

    LyCORIS suffixes (LoKR / LoHa) are kept as-is — only the prefix is stripped
    and dots flattened:
      diffusion_model.double_blocks.0.img_attn.proj.lokr_w1
        → lora_unet_double_blocks_0_img_attn_proj.lokr_w1

    For standard LoRA: synthesizes `.alpha` tensor = rank where absent (PEFT
    convention, producing scale = 1.0). For LyCORIS: preserves the file's
    `.alpha` exactly (LyCORIS tools set it to an explicit value or a sentinel
    like 1e10 meaning "scale = 1.0 already baked in").

    Handles common prefix variants seen in the wild:
      - `diffusion_model.` (ComfyUI LoRA export, Colorful-style)
      - `transformer.`     (AI-Toolkit, OneTrainer OMI format)
      - `lora_transformer_` (OneTrainer legacy-diffusers format — dots already flattened)
    """
    # Check for diffusers-style Flux keys (transformer_blocks / single_transformer_blocks)
    # These need full key remapping + QKV fusion, not just prefix stripping.
    # Checked FIRST: Krea 2 and Flux both say `transformer_blocks.` in diffusers form, and the
    # Flux converter would map a Krea 2 file onto block names that do not exist there. Krea 2
    # needs no QKV fusion (q/k/v are split in both namespaces), so the ordinary strip+flatten
    # below does the heavy lifting and the rename pass runs on the result at the end.
    _krea2 = _is_diffusers_krea2_lora(weights_sd)
    if not _krea2 and _is_diffusers_flux_lora(weights_sd):
        logger.info("Diffusers-format Flux LoRA detected (transformer_blocks) — converting with QKV fusion.")
        return _convert_diffusers_flux_lora(weights_sd)

    converted: Dict[str, torch.Tensor] = {}
    ranks: Dict[str, int] = {}  # lora_name → rank for standard-LoRA alpha synthesis

    # Candidate prefixes we'll strip
    prefixes = ("diffusion_model.", "transformer.")

    # LyCORIS suffixes kept verbatim (just strip prefix + flatten dots in module path)
    LYCORIS_SUFFIXES = (
        ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
        ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    )

    for key, tensor in weights_sd.items():
        # Strip one recognized prefix; skip keys that don't have one
        stripped = None
        for pfx in prefixes:
            if key.startswith(pfx):
                stripped = key[len(pfx):]
                break
        if stripped is None:
            # No prefix at all, but a dotted module path with a LoRA suffix — the MiniMax H3
            # ecosystem exports like this (blocks.0.attn.qkv_proj.lora_A.weight, e.g. the
            # Turbo LoRA). Treat the whole key as the stripped path.
            _bare_suffixes = (".lora_A.weight", ".lora_B.weight", ".lora_down.weight",
                              ".lora_up.weight", ".alpha") + LYCORIS_SUFFIXES
            if "." in key and key.endswith(_bare_suffixes):
                stripped = key
            else:
                # Not a LoRA weight in a format we convert — preserve as-is (rare)
                converted[key] = tensor
                continue

        # LyCORIS suffix? Preserve it verbatim.
        lycoris_match = None
        for suf in LYCORIS_SUFFIXES:
            if stripped.endswith(suf):
                lycoris_match = suf
                break
        if lycoris_match is not None:
            module_path = stripped[: -len(lycoris_match)]
            flat = module_path.replace(".", "_")
            lora_name = f"lora_unet_{flat}"
            converted[f"{lora_name}{lycoris_match}"] = tensor
            continue

        # Map standard-LoRA suffixes: lora_A/B (PEFT) or lora_down/up (OneTrainer/kohya)
        if stripped.endswith(".lora_A.weight"):
            module_path = stripped[: -len(".lora_A.weight")]
            matrix_name = "lora_down"
        elif stripped.endswith(".lora_B.weight"):
            module_path = stripped[: -len(".lora_B.weight")]
            matrix_name = "lora_up"
        elif stripped.endswith(".lora_down.weight"):
            module_path = stripped[: -len(".lora_down.weight")]
            matrix_name = "lora_down"
        elif stripped.endswith(".lora_up.weight"):
            module_path = stripped[: -len(".lora_up.weight")]
            matrix_name = "lora_up"
        elif stripped.endswith(".alpha"):
            module_path = stripped[: -len(".alpha")]
            matrix_name = "alpha"
        elif stripped.endswith(".dora_scale"):
            # DoRA scale — skip (not supported but don't error)
            continue
        else:
            # Unrecognized suffix — skip (could be biases or other non-LoRA aux data)
            continue

        # Flatten dots to underscores inside the module path, prepend lora_unet_
        flat = module_path.replace(".", "_")
        lora_name = f"lora_unet_{flat}"

        if matrix_name == "alpha":
            converted[f"{lora_name}.alpha"] = tensor
        else:
            new_key = f"{lora_name}.{matrix_name}.weight"
            converted[new_key] = tensor
            if matrix_name == "lora_down":
                # lora_down shape = (rank, in_dim) — first dim IS the rank
                ranks[lora_name] = int(tensor.shape[0])

    # Synthesize alpha tensors where not already present (standard LoRA only).
    # LyCORIS modules always ship their own alpha; don't fabricate one for them.
    for lora_name, rank in ranks.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in converted:
            converted[alpha_key] = torch.tensor(float(rank))

    logger.info(
        f"[peft→kohya] converted {len(weights_sd)} keys → {len(converted)} "
        f"({len(ranks)} LoRA modules, alphas synthesized where absent)"
    )
    if _krea2:
        converted = _krea2_diffusers_to_native(converted)
    return converted


class UnsupportedLoRAFormat(RuntimeError):
    """Raised when a LoRA file uses a low-rank parametrization Fizgig can't load
    (LoKR / LoHa / GLoRA). Fizgig's LoRAInfModule only implements standard LoRA
    (`up @ down`); these variants use Kronecker / Hadamard / four-matrix
    decompositions that would need separate module classes."""


class LoRAFamilyMismatch(RuntimeError):
    """Raised when a LoRA's trained-for family doesn't match the family a caller
    (Royale / Explorer / Repair Studio) is about to load it against. Callers should
    catch this the same way they catch UnsupportedLoRAFormat — a clean single-line
    message, no traceback — since it's an expected user-facing outcome, not a bug."""


# Family display names, keyed by the same "klein" / "krea2" / "minimax" strings the
# GUI's *_family_var StringVars already use — lets callers compare lora_family_from_file's
# result against a family_var.get() directly with no translation step.
FAMILY_DISPLAY_NAMES = {
    "klein": "Klein 9B",
    "krea2": "Krea 2",
    "minimax": "MiniMax H3",
}

# Families that actually have an inference-side selector (Royale, Explorer, Repair Studio).
# A caller auto-following lora_family_from_file's result must only .set() values a radio
# button carries (issue #62 follow-up: a value with no radio leaves both blank and the family
# that loads is whichever "else" branch the caller's ternary falls back to).
INFERENCE_FAMILIES = ("klein", "krea2", "minimax")

# Klein 9B's block depth (Klein9BParams in klein/model.py: depth=8, depth_single_blocks=24).
# double_blocks/single_blocks (native/kohya) and transformer_blocks/single_transformer_blocks
# (diffusers) are BFL naming shared with other Flux variants at different depths — Flux.1
# dev/schnell is 19+38, full Flux.2 is 8+48 — so the family can't be read off the block-name
# substring alone; the deepest block index actually present has to fall inside Klein's own
# shape. A negative lookbehind keeps "single_transformer_blocks.N" from also matching the
# double/transformer_blocks pattern (it contains "transformer_blocks.N" as a substring).
_KLEIN_MAX_DOUBLE_BLOCKS = 8
_KLEIN_MAX_SINGLE_BLOCKS = 24
_KLEIN_DOUBLE_BLOCK_RE = re.compile(r"(?<!single_)(?:double_blocks|transformer_blocks)[._](\d+)")
_KLEIN_SINGLE_BLOCK_RE = re.compile(r"single_(?:transformer_)?blocks[._](\d+)")


def lora_family_from_file(path: str) -> Optional[str]:
    """Identify which model family a LoRA/LyCORIS file was trained for, from its
    safetensors header alone — no tensor data read, no pipeline load, microseconds.

    Returns "klein", "krea2", "minimax", or None if the file doesn't match any
    known signature (callers should not block on None — fail open and let the
    real loader be the judge, the same as an unreadable/corrupt file). None also
    covers a file that uses Klein's own block-name segments at a depth Klein 9B
    doesn't have (a Flux.1 or full-Flux.2 LoRA) — see _KLEIN_MAX_DOUBLE_BLOCKS.

    Krea 2 and MiniMax H3 are each identified by a substring nothing else uses, so a
    plain membership check is enough for them regardless of whether the file uses
    Fizgig's own native prefixes (`diffusion_model.`, `transformer.`), the kohya/
    sd-scripts flattened form (`lora_unet_*`), or another tool's dotted diffusers
    export. Klein 9B shares its block-name segments with other Flux depths, so it's
    identified by block count instead (checked last, after Krea 2 and MiniMax H3 are
    ruled out — diffusers-form Krea 2 also says `transformer_blocks.`).
    """
    from fizgig.utils.safetensors import MemoryEfficientSafeOpen
    try:
        with MemoryEfficientSafeOpen(path) as f:
            keys = f.keys()
    except Exception:
        return None
    for k in keys:
        if "text_fusion" in k or "txtfusion" in k:
            return "krea2"
    if _is_diffusers_krea2_lora(keys):
        return "krea2"
    for k in keys:
        if "token_refiner" in k or "adaln_proj" in k:
            return "minimax"
    max_double = -1
    max_single = -1
    for k in keys:
        m = _KLEIN_DOUBLE_BLOCK_RE.search(k)
        if m:
            max_double = max(max_double, int(m.group(1)))
            continue
        m = _KLEIN_SINGLE_BLOCK_RE.search(k)
        if m:
            max_single = max(max_single, int(m.group(1)))
    if max_double < 0 and max_single < 0:
        return None
    if max_double < _KLEIN_MAX_DOUBLE_BLOCKS and max_single < _KLEIN_MAX_SINGLE_BLOCKS:
        return "klein"
    return None


def assert_lora_family_matches(path: str, selected_family: str, tab_label: str) -> None:
    """Guard for the family/LoRA mismatch in issue #62: check the file's family against
    what the tab's selector is set to BEFORE a caller commits to a full pipeline load.
    No-op when the file's family can't be determined (fail open) or already matches.

    Raises LoRAFamilyMismatch with a plain-English message on a real mismatch — callers
    should catch it alongside UnsupportedLoRAFormat and show it without a traceback.
    """
    detected = lora_family_from_file(path)
    if detected is None or detected == selected_family:
        return
    detected_name = FAMILY_DISPLAY_NAMES.get(detected, detected)
    selected_name = FAMILY_DISPLAY_NAMES.get(selected_family, selected_family)
    raise LoRAFamilyMismatch(
        f"This LoRA was trained for {detected_name}, but the {tab_label} family selector "
        f"is on {selected_name}. Switch it to {detected_name} to use this file."
    )


def ensure_kohya_lora_state_dict(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert the state dict to kohya module-name convention if needed.
    Pass-through for already-kohya files. Raises UnsupportedLoRAFormat for
    GLoRA (4-matrix form, not yet implemented)."""
    fmt = detect_lora_format(weights_sd)

    # OneTrainer legacy-diffusers format: lora_transformer_ prefix → rename to lora_unet_
    if fmt == "kohya" and any(k.startswith("lora_transformer_") for k in weights_sd.keys()):
        renamed = {}
        for k, v in weights_sd.items():
            if k.startswith("lora_transformer_"):
                renamed["lora_unet_" + k[len("lora_transformer_"):]] = v
            else:
                renamed[k] = v
        n_renamed = sum(1 for k in weights_sd if k.startswith("lora_transformer_"))
        logger.info("OneTrainer legacy format detected — renamed %d lora_transformer_ keys to lora_unet_", n_renamed)
        weights_sd = renamed
        # The module paths are still diffusers-shaped after the prefix swap. A Krea 2 file
        # (ff_up / attn_to_gate / text_fusion markers) needs the same rename pass the dotted
        # form gets in peft_to_kohya, or none of its names exist in the model (issue #50).
        if _is_diffusers_krea2_lora(weights_sd):
            weights_sd = _krea2_diffusers_to_native(weights_sd)

    if fmt == "peft":
        logger.info("PEFT/diffusers-format LoRA detected — converting key names to kohya convention.")
        return peft_to_kohya(weights_sd)
    if fmt in ("lokr", "loha"):
        # LyCORIS files almost always use a diffusion_model. prefix in the wild;
        # run them through peft_to_kohya to strip the prefix and flatten dots.
        # The LyCORIS suffixes (.lokr_w1, .hada_w1_a, etc.) are preserved verbatim.
        has_prefix = any(k.startswith("diffusion_model.") or k.startswith("transformer.")
                         for k in weights_sd.keys())
        if has_prefix:
            logger.info(f"{fmt.upper()} LoRA with diffusion_model prefix detected — flattening to kohya module names.")
            return peft_to_kohya(weights_sd)
        return weights_sd
    if fmt == "glora":
        raise UnsupportedLoRAFormat(
            "This LoRA uses the 'GLoRA' parametrization (four matrices A/B/C/D), "
            "which Fizgig doesn't implement yet. LoKR and LoHa are supported; "
            "GLoRA is a follow-up. Re-extract as a standard LoRA, LoKR, or LoHa."
        )
    return weights_sd


def _build_dit_linear_map(
    unet: nn.Module,
    target_replace_modules: Optional[List[str]],
    prefix: str = "lora_unet",
) -> Dict[str, nn.Module]:
    """Map kohya-style lora_name → target Linear/Conv2d module in the DiT.

    Replicates the naming logic from `LoRANetwork.create_modules` so we can
    look up a specific Linear from a LyCORIS module name without rescanning.
    """
    result: Dict[str, nn.Module] = {}
    if unet is None:
        return result
    for name, module in unet.named_modules():
        if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
            for child_name, child_module in module.named_modules():
                # Must match create_modules' whitelist (lora.py ~:860) — quantized bases hold
                # Linear SUBCLASSES, and a plain-Linear test silently drops every LyCORIS
                # module on them (H3's LoKR default loaded 0 modules before this).
                if child_module.__class__.__name__ in (
                        "Linear", "Linear4bit", "Linear8bitLt",
                        "ConvRotInt8Linear", "Nvfp4Linear", "HQQ4bitLinear", "Conv2d"):
                    original_name = (name + "." if name else "") + child_name
                    lora_name = f"{prefix}.{original_name}".replace(".", "_")
                    result[lora_name] = child_module
    return result


def _append_lycoris_modules(
    network: "LoRANetwork",
    weights_sd: Dict[str, torch.Tensor],
    target_replace_modules: List[str],
    unet: Optional[nn.Module],
    for_inference: bool,
    multiplier: float,
) -> None:
    """Scan the state dict for LoKR/LoHa modules and append them to
    network.unet_loras. Must be called BEFORE network.apply_to() so they get
    patched in with the rest (apply_to iterates unet_loras)."""
    if not for_inference:
        # LyCORIS training modules are out of scope for Fizgig currently.
        return
    if unet is None:
        return

    # Group state-dict keys by module name (the part before the first dot).
    by_module: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in weights_sd.items():
        if "." not in key:
            continue
        lora_name, _, suffix = key.partition(".")
        if not lora_name.startswith("lora_unet_"):
            continue
        by_module.setdefault(lora_name, {})[suffix] = tensor

    linear_map = _build_dit_linear_map(unet, target_replace_modules)
    existing_names = {m.lora_name for m in network.unet_loras}
    added_lokr = 0
    added_loha = 0

    for lora_name, keys in by_module.items():
        # Skip if standard LoRA already created this module.
        if lora_name in existing_names:
            continue

        # Detect variant.
        has_lokr = any(k.startswith("lokr_w") for k in keys)
        has_loha = any(k.startswith("hada_") for k in keys)
        if not (has_lokr or has_loha):
            continue  # standard LoRA without lora_down — skip; or non-LoRA key

        org_module = linear_map.get(lora_name)
        if org_module is None:
            logger.warning(f"LyCORIS module {lora_name} has no matching Linear in DiT — skipping")
            continue

        # Extract alpha (LyCORIS convention: scalar tensor).
        alpha_tensor = keys.get("alpha")
        alpha_val = float(alpha_tensor.item()) if alpha_tensor is not None else 1.0

        org_out = getattr(org_module, "out_features", None) or getattr(org_module, "out_channels", None)
        org_in = getattr(org_module, "in_features", None) or getattr(org_module, "in_channels", None)

        if has_lokr:
            # Resolve w1 shape — either direct (a,b) or decomposed ((a,k1),(k1,b)).
            if "lokr_w1" in keys:
                a, b = keys["lokr_w1"].shape
                k1 = None
            elif "lokr_w1_a" in keys and "lokr_w1_b" in keys:
                a = keys["lokr_w1_a"].shape[0]
                k1 = keys["lokr_w1_a"].shape[1]
                b = keys["lokr_w1_b"].shape[1]
            else:
                logger.warning(f"LoKR {lora_name}: missing w1 (or w1_a/w1_b pair) — skipping")
                continue
            # Same for w2.
            if "lokr_w2" in keys:
                c, d = keys["lokr_w2"].shape
                k2 = None
            elif "lokr_w2_a" in keys and "lokr_w2_b" in keys:
                c = keys["lokr_w2_a"].shape[0]
                k2 = keys["lokr_w2_a"].shape[1]
                d = keys["lokr_w2_b"].shape[1]
            else:
                logger.warning(f"LoKR {lora_name}: missing w2 (or w2_a/w2_b pair) — skipping")
                continue

            if a * c != org_out or b * d != org_in:
                logger.warning(
                    f"LoKR {lora_name}: kron({a}x{b}, {c}x{d}) = {a*c}x{b*d} "
                    f"doesn't match Linear {org_out}x{org_in} — skipping"
                )
                continue

            module = LoKRInfModule(lora_name, org_module, multiplier, alpha_val,
                                   a=a, b=b, c=c, d=d, k1=k1, k2=k2)
            network.unet_loras.append(module)
            existing_names.add(lora_name)
            added_lokr += 1

        elif has_loha:
            required = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
            missing = [k for k in required if k not in keys]
            if missing:
                logger.warning(f"LoHa {lora_name}: missing {missing} — skipping")
                continue
            out_dim = keys["hada_w1_a"].shape[0]
            r1 = keys["hada_w1_a"].shape[1]
            in_dim = keys["hada_w1_b"].shape[1]
            r2 = keys["hada_w2_a"].shape[1]
            if out_dim != org_out or in_dim != org_in:
                logger.warning(
                    f"LoHa {lora_name}: shape ({out_dim}, {in_dim}) doesn't match "
                    f"Linear ({org_out}, {org_in}) — skipping"
                )
                continue

            module = LoHaInfModule(lora_name, org_module, multiplier, alpha_val,
                                   out_dim=out_dim, in_dim=in_dim, r1=r1, r2=r2)
            network.unet_loras.append(module)
            existing_names.add(lora_name)
            added_loha += 1

    if added_lokr:
        logger.info(f"Appended {added_lokr} LoKR modules to network.")
    if added_loha:
        logger.info(f"Appended {added_loha} LoHa modules to network.")


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    module_class: Optional[Type[object]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> LoRANetwork:
    # Scan standard-LoRA dims/alphas (LyCORIS modules are detected separately below).
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim

    if module_class is None:
        module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
    )

    # Append LoKR / LoHa modules found in the state dict. A single file can
    # mix variants (uncommon but permitted) — each module is dispatched by
    # its own key suffix.
    _append_lycoris_modules(
        network, weights_sd, target_replace_modules, unet, for_inference, multiplier
    )

    return network
