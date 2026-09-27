"""Conditional Immiscible Diffusion noise targets.

The sample's conditioning (its caption) is never moved. Only the Gaussian
noise that sample is flowed toward is chosen, which is the conditional
fine-tuning setup in https://github.com/yhli123/Immiscible-Diffusion
(``conditional_ft_train_sd.py``), as opposed to an optimal-transport plan that
permutes data and labels together.

  knn         — draw ``k`` noises for this sample and keep the nearest.
                Improved Immiscible Diffusion (arXiv:2505.18521). Works at
                batch size 1, which is how MiniMax H3 trains.
  assignment  — pair a group of samples with a group of noises by minimising
                total L2 distance. Immiscible Diffusion (arXiv:2406.12303).

Distances are in fp32. The reference scales both sides by 0.10 before an fp16
norm; that factor cancels in the argmin and is omitted. An fp16 norm also
overflows on a long H3 video latent.

The pairing rule is reimplemented from the MIT-licensed reference
(Copyright (c) 2024 Yiheng Li). See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch


def knn_noise(x: torch.Tensor, k: int, generator: torch.Generator | None = None,
              chunk: int = 8) -> torch.Tensor:
    """Nearest of ``k`` standard normals to each row of ``x`` ([B, ...]).

    Candidates are drawn in chunks so a long video latent is not materialised
    ``k`` times at once. ``k == 1`` is an ordinary draw.
    """
    if k < 1:
        raise ValueError(f"immiscible k must be >= 1 (got {k})")
    if x.ndim < 1:
        raise ValueError("knn_noise expects a batched tensor")
    b = x.shape[0]
    if b == 0:
        return torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    x_flat = x.detach().reshape(b, -1).float()
    best = None
    best_dist = None
    left = int(k)
    while left:
        c = min(int(chunk), left)
        left -= c
        cand = torch.randn((b, c) + tuple(x.shape[1:]), device=x.device,
                           dtype=torch.float32, generator=generator)
        dist = _squared_rows(x_flat, cand.reshape(b, c, -1))
        dmin, idx = dist.min(dim=1)
        chosen = cand[torch.arange(b, device=x.device), idx]
        if best is None:
            best, best_dist = chosen, dmin
        else:
            take = dmin < best_dist
            view = (b,) + (1,) * (best.ndim - 1)
            best = torch.where(take.view(view), chosen, best)
            best_dist = torch.where(take, dmin, best_dist)
    return best


def assignment_noise(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Noise of ``x``'s shape whose pairing minimises total L2 distance.

    Row ``i`` of the result is the noise assigned to row ``i`` of ``x``. The
    rows of ``x`` are not reordered. A single row has only one pairing, so it
    is an ordinary draw.
    """
    if x.ndim < 1:
        raise ValueError("assignment_noise expects a batched tensor")
    noise = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    b = x.shape[0]
    if b < 2:
        return noise
    # cost[i, j] = ||x[i] - noise[j]||^2. x's row order stays; noise is permuted.
    cost = _pairwise_sq(x.detach().reshape(b, -1).float(), noise.reshape(b, -1).float())
    index = torch.tensor(min_cost_assignment(cost), device=noise.device, dtype=torch.long)
    return noise.index_select(0, index)


def min_cost_assignment(cost: torch.Tensor) -> list[int]:
    """Column index per row of a square cost matrix (minimum total)."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"assignment cost must be square, got {tuple(cost.shape)}")
    n = cost.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    return _jonker_volgenant(cost.detach().float().cpu().tolist())


def _squared_rows(x_flat: torch.Tensor, n_flat: torch.Tensor) -> torch.Tensor:
    """||x[b] - n[b, c]||^2. ``x_flat`` is [B, D], ``n_flat`` is [B, C, D]."""
    x2 = (x_flat * x_flat).sum(-1, keepdim=True)
    n2 = (n_flat * n_flat).sum(-1)
    dot = torch.matmul(n_flat, x_flat.unsqueeze(-1)).squeeze(-1)
    return (x2 + n2 - 2 * dot).clamp_min(0)


def _pairwise_sq(x_flat: torch.Tensor, n_flat: torch.Tensor) -> torch.Tensor:
    """||x[i] - n[j]||^2 as [B, B], without materialising [B, B, D]."""
    x2 = (x_flat * x_flat).sum(-1, keepdim=True)
    n2 = (n_flat * n_flat).sum(-1).unsqueeze(0)
    return (x2 + n2 - 2 * (x_flat @ n_flat.T)).clamp_min(0)


def _jonker_volgenant(cost: list[list[float]]) -> list[int]:
    """Min-cost assignment. ``cols[row]`` is the chosen column.

    The dense Jonker–Volgenant shortest-augmenting-path algorithm, 1-indexed
    the way the classical writeup is. ``n`` is the group size H3 actually
    pairs (a handful to a few dozen), so the Python loops are not the step.
    """
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = [0] * n
    for j in range(1, n + 1):
        cols[p[j] - 1] = j - 1
    return cols
