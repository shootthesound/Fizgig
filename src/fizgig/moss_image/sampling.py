"""MOSS-Image schedule, timestep sampler and Euler sampler (the reference pipeline's, without diffusers).

* sigmas(n, shift): MOSSImageScheduler.set_timesteps - a shifted grid snapped to the nearest of the 1000 training
  sigmas (training shift 1.0), duplicates dropped, then a terminal 0. The DiT's timestep is 1000 sigma.
* sample(): fp32 noise, fp32 CFG and Euler updates; the negative branch skips block 22 (uncond_skip_layers) and an
  optional positive SLG branch adds slg_scale (cond - cond_skip), both inside [slg_start, slg_stop) of the steps.
* sample_t(): the paper's final-stage training distribution (App. A.2 eq. 4) on the discrete 1000-sigma grid;
  options can replace it with plain logit-normal or uniform.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

NUM_TRAIN_TIMESTEPS = 1000
SHIFT = 5.0                 # scheduler_config.json shift
TRAINING_SHIFT = 1.0        # scheduler_config.json training_shift
CFG = 3.0
UNCOND_SKIP_LAYERS = (22,)
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128
# paper App. A.2 eq. 4: S = 1 - sigmoid(N(mean, std^2)) truncated at S <= cutoff, mixed with uniform
T_MEAN, T_STD, T_CUTOFF, T_UNIFORM = -1.09, 0.95, 0.95, 0.01


def _grid(count: int, shift: float, total: int = NUM_TRAIN_TIMESTEPS) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def sigmas(steps: int, shift: float = SHIFT, training_shift: float = TRAINING_SHIFT) -> torch.Tensor:
    """fp32 [n + 1], from ~1 down to the terminal 0 (n <= steps: snapped duplicates are dropped)."""
    if steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = NUM_TRAIN_TIMESTEPS
    s = _grid(steps, shift)
    training = _grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, s).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((s - training[lower]).abs() <= (training[upper] - s).abs(), lower, upper)
    s = torch.unique_consecutive(training[nearest])
    return torch.cat([s, s.new_zeros(1)]).float()


def snap_sigma(sigma: float) -> float:
    """The training grid's nearest sigma (training shift 1.0: k / 1000)."""
    training = _grid(NUM_TRAIN_TIMESTEPS, TRAINING_SHIFT).flip(0)
    i = int(torch.argmin((training - float(sigma)).abs()))
    return float(training[i])


def latent_hw(height: int, width: int) -> tuple:
    if height <= 0 or width <= 0 or height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError("height and width must be positive multiples of 16.")
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


def initial_noise(seed: int, height: int, width: int, channels: int = LATENT_CHANNELS) -> torch.Tensor:
    """(1, 128, h, w) fp32 on CPU - the reference draws its noise in fp32 (never via bf16)."""
    h, w = latent_hw(height, width)
    return torch.randn((1, channels, h, w), generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float32)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sample_t(generator, kind: str = "paper") -> float:
    """sigma on the training grid {0.001, ..., 1.000} for one step.

    "paper" is App. A.2 eq. 4 (the final Stage 3 interval): Z ~ N(-1.09, 0.95^2), S = 1 - sigmoid(Z) kept only where
    S <= 0.95, rescaled by 1 / 0.95 onto the grid (sigma_j = j / 1000 for S in (0.95 (j-1)/1000, 0.95 j/1000]), mixed
    with 1% uniform over the grid. Mean sigma ~0.78: the base was finished on noisy steps.
    "logit_normal": sigmoid(N(0, 1)); "uniform": uniform over the grid."""
    T = NUM_TRAIN_TIMESTEPS
    u = torch.rand(1, generator=generator, dtype=torch.float64).item()
    if kind == "logit_normal":
        z = torch.randn(1, generator=generator, dtype=torch.float64).item()
        return min(T, max(1, math.ceil(T / (1 + math.exp(-z))))) / T
    if kind == "uniform":
        return min(T, int(u * T) + 1) / T
    if kind != "paper":
        raise ValueError(f"unknown timestep sampler {kind!r}")
    v = torch.rand(1, generator=generator, dtype=torch.float64).item()
    if u < T_UNIFORM:
        return min(T, int(v * T) + 1) / T
    # Z truncated to S <= a, i.e. Z >= logit(1 - a), drawn by inverse CDF
    lo = _phi((math.log((1 - T_CUTOFF) / T_CUTOFF) - T_MEAN) / T_STD)
    p = min(max(lo + v * (1 - lo), 1e-15), 1 - 1e-15)
    z = T_MEAN + T_STD * math.sqrt(2) * torch.erfinv(torch.tensor(2 * p - 1, dtype=torch.float64)).item()
    S = 1 - 1 / (1 + math.exp(-z))
    return min(T, max(1, math.ceil(S / T_CUTOFF * T - 1e-9))) / T


def paper_pmf() -> torch.Tensor:
    """Eq. 4's p_j for j = 1..1000 (float64) - the test's reference."""
    T, a, rho = NUM_TRAIN_TIMESTEPS, T_CUTOFF, T_UNIFORM

    def F(x):     # Pr(S <= x), S = 1 - sigmoid(Z)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return 1 - _phi((math.log((1 - x) / x) - T_MEAN) / T_STD)
    Fa = F(a)
    return torch.tensor([(1 - rho) * (F(a * j / T) - F(a * (j - 1) / T)) / Fa + rho / T for j in range(1, T + 1)],
                        dtype=torch.float64)


def scale_t(sigma: float, min_t: float = 0.0, max_t: float = 1.0) -> float:
    """Fizgig's [min_t, max_t] range (timestep presets) mapped onto the grid, never below 0.001."""
    if min_t <= 0.0 and max_t >= 1.0:
        return sigma
    return max(1, round((min_t + (max_t - min_t) * sigma) * NUM_TRAIN_TIMESTEPS)) / NUM_TRAIN_TIMESTEPS


def predict(dit, x, sigma, cond, mask=None, skip_layers=()):
    """The DiT's velocity at sigma (fp32 out). x fp32 [B, 128, h, w]; cond [B, L, 4096]; mask [B, L] or None."""
    dtype = next(p for p in dit.parameters() if p.is_floating_point()).dtype
    t = torch.as_tensor(sigma * NUM_TRAIN_TIMESTEPS, dtype=torch.float32, device=x.device)
    return dit(x.to(dtype), t, cond.to(dtype), mask, skip_layers=skip_layers).float()


@torch.no_grad()
def sample(dit, cond, height, width, *, steps=50, seed=0, cfg=CFG, neg=None, mask=None, neg_mask=None,
           shift=SHIFT, uncond_skip_layers=UNCOND_SKIP_LAYERS, slg_layers=(), slg_scale=0.0, slg_start=0.0,
           slg_stop=1.0, sigma_schedule=None, noise=None, device=None, on_step: Optional[Callable] = None):
    """The reference pipeline's loop -> latents fp32 [1, 128, h, w] (the normalised space, as the cache holds them).
    cond [1, L, 4096] (+ mask [1, L]); neg the encoded negative (the empty prompt by default) when cfg != 1."""
    device = device or next(dit.parameters()).device
    if noise is None:
        noise = initial_noise(seed, height, width, dit.latent_channels)
    x = noise.to(device=device, dtype=torch.float32)
    if x.dim() == 3:
        x = x[None]
    s = sigmas(steps, shift) if sigma_schedule is None else torch.as_tensor(sigma_schedule, dtype=torch.float32)
    if float(s[-1]) != 0.0:
        s = torch.cat([s, s.new_zeros(1)])
    cond = cond.to(device)
    mask = mask.to(device) if mask is not None else None
    do_cfg = cfg != 1.0
    if do_cfg and neg is None:
        raise ValueError("MOSS-Image samples with CFG against an encoded negative (the empty prompt); none was given")
    if do_cfg:
        neg = neg.to(device)
        neg_mask = neg_mask.to(device) if neg_mask is not None else None
    uncond_skip = tuple(uncond_skip_layers or ())
    slg = tuple(slg_layers or ())
    n = len(s) - 1
    for i in range(n):
        if on_step is not None:
            on_step(i, n)
        skip_active = slg_start <= i / n < slg_stop
        sigma = float(s[i])
        pred = predict(dit, x, sigma, cond, mask)
        conditional = pred
        if do_cfg:
            uncond = predict(dit, x, sigma, neg, neg_mask, uncond_skip if skip_active else ())
            pred = uncond + cfg * (pred - uncond)
        if slg and slg_scale > 0 and skip_active:
            skipped = predict(dit, x, sigma, cond, mask, slg)
            if not do_cfg:
                pred = skipped
            pred = pred + slg_scale * (conditional - skipped)
        x = x + (float(s[i + 1]) - sigma) * pred
    if on_step is not None:
        on_step(n, n)
    return x


def unpack_for_vae(latents: torch.Tensor, running_mean, running_var, eps: float = 1e-4) -> torch.Tensor:
    """The reference's decode prelude: de-normalise with the VAE BatchNorm, then 2x2-unpatchify to (B, 32, 2h, 2w)."""
    z = latents * (running_var + eps).sqrt().to(latents).view(1, -1, 1, 1) + running_mean.to(latents).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
