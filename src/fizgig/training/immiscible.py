"""Conditional Immiscible Diffusion noise targets.

The sample's conditioning (its caption) is never moved. Only the Gaussian
noise that sample is flowed toward is chosen, which is the conditional
fine-tuning setup in https://github.com/yhli123/Immiscible-Diffusion
(``conditional_ft_train_sd.py``), as opposed to an optimal-transport plan that
permutes data and labels together.

  knn         — draw ``k`` noises for this sample and keep the nearest.
                Improved Immiscible Diffusion (arXiv:2505.18521). Works at
                batch size 1, which is how MiniMax H3 trains.
  knn_coarse  — the same draw, scored on an average-pooled copy of the latent
                (default 8×8). The tensor that enters the flow is still the
                full-resolution candidate, so high-frequency detail stays a
                normal Gaussian. An H3 still is ~55,000 values; full-latent
                nearest-of-k either barely moves (k=4) or hugs the image
                (k=64) and the training target collapses.
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
              chunk: int = 8, coarse: int = 0) -> torch.Tensor:
    """Nearest of ``k`` standard normals to each row of ``x`` ([B, ...]).

    Candidates are drawn in chunks so a long video latent is not materialised
    ``k`` times at once. ``k == 1`` is an ordinary draw.

    ``coarse`` > 1 scores the match on an average-pool of the last two axes
    (both must be larger than ``coarse``). The returned tensor is still the
    full-resolution candidate. A tensor with no spatial pair of axes, such as
    audio rows ``[1, A, 32]``, keeps the full-vector score.
    """
    if k < 1:
        raise ValueError(f"immiscible k must be >= 1 (got {k})")
    if x.ndim < 1:
        raise ValueError("knn_noise expects a batched tensor")
    b = x.shape[0]
    if b == 0:
        return torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    x_score = _score_flat(x, coarse)
    best = None
    best_dist = None
    left = int(k)
    while left:
        c = min(int(chunk), left)
        left -= c
        cand = torch.randn((b, c) + tuple(x.shape[1:]), device=x.device,
                           dtype=torch.float32, generator=generator)
        idx, dist = _knn_index(x_score, cand, coarse)
        chosen = cand[torch.arange(b, device=x.device), idx]
        if best is None:
            best, best_dist = chosen, dist
        else:
            take = dist < best_dist
            view = (b,) + (1,) * (best.ndim - 1)
            best = torch.where(take.view(view), chosen, best)
            best_dist = torch.where(take, dist, best_dist)
    return best


def _score_flat(x: torch.Tensor, coarse: int) -> torch.Tensor:
    """``[N, ...]`` -> ``[N, D]`` used only to rank candidates.

    The flow target is never this tensor. Pooling applies when ``coarse`` > 1
    and the last two axes are both strictly larger than ``coarse``.
    """
    if coarse > 1 and x.ndim >= 4:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h > coarse and w > coarse:
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                x.detach().reshape(x.shape[0], -1, h, w).float(),
                (int(coarse), int(coarse)),
            )
            return pooled.reshape(x.shape[0], -1)
    return x.detach().reshape(x.shape[0], -1).float()


def _knn_index(x_score: torch.Tensor, cand: torch.Tensor, coarse: int):
    """Index and distance of the nearest candidate. ``cand`` is ``[B, K, ...]``."""
    b, k = cand.shape[:2]
    scored = _score_flat(cand.reshape(b * k, *cand.shape[2:]), coarse).reshape(b, k, -1)
    if scored.shape[-1] != x_score.shape[-1]:
        raise ValueError(
            f"coarse score width {scored.shape[-1]} does not match the latent "
            f"score {x_score.shape[-1]}")
    dist = _squared_rows(x_score, scored)
    dmin, idx = dist.min(dim=1)
    return idx, dmin


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
