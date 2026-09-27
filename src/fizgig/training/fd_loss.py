"""EMA Fréchet loss on pooled MiniMax H3 latents.

Yang et al., Representation Fréchet Loss for Visual Generation, arXiv:2604.28190.
The paper matches a large population of *image* features (Inception, DINOv2,
SigLIP) while backpropagating only through the current batch. Two of their
conditions do not fit an H3 LoRA step: the gradient batch they use is 1024,
and the feature net runs on decoded pixels. The int8 H3 base already fills a
24 GB card, so this module uses the same estimator on a pooled VAE latent.

  reference cloud  — mean and second moment of the cached training latents,
                     computed once before the DiT is asked to generate
  generated cloud  — EMA of those moments (beta 0.999 in the paper)
  loss             — Fréchet distance, divided by stopgrad(FD) + eps

The EMA buffers are detached. Gradients enter only through the current
predicted clean latent. A voice item's zero video placeholder is not a member
of the cloud; the caller skips it.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def latent_descriptor(latent: torch.Tensor, pool: int = 4) -> torch.Tensor:
    """Pooled H3 latent as one row ``[1, 24 * pool * pool]``.

    Accepts ``[C,H,W]``, ``[C,T,H,W]``, ``[1,C,H,W]`` or ``[1,C,T,H,W]``.
    Time is averaged. H and W are cropped to an even size, matching the
    patch crop in the training step, then average-pooled.
    """
    if pool < 1:
        raise ValueError(f"fd pool must be >= 1 (got {pool})")
    z = latent.detach().float() if not latent.requires_grad else latent.float()
    if z.ndim == 3:
        z = z.unsqueeze(0).unsqueeze(2)
    elif z.ndim == 4:
        # [C,T,H,W] when C is the VAE width, else a batched still [1,C,H,W].
        if z.shape[0] == 1:
            z = z.unsqueeze(2)
        else:
            z = z.unsqueeze(0)
    elif z.ndim != 5:
        raise ValueError(f"H3 latent must be 3D, 4D or 5D, got {tuple(z.shape)}")
    h, w = int(z.shape[-2]), int(z.shape[-1])
    hc, wc = h - (h % 2), w - (w % 2)
    if hc < 2 or wc < 2:
        raise ValueError(f"latent spatial size {h}x{w} is too small to describe")
    if (hc, wc) != (h, w):
        z = z[..., :hc, :wc]
    z = z.mean(dim=2)
    z = F.adaptive_avg_pool2d(z, (int(pool), int(pool)))
    return z.flatten(1)


def _sqrtm(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD square root. Jitter keeps a rank-deficient cloud invertible."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
    mat = mat + (1e-5 * eye)
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(0).sqrt()
    return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)


def frechet_distance(mu_r: torch.Tensor, sig_r: torch.Tensor,
                     mu_g: torch.Tensor, sig_g: torch.Tensor) -> torch.Tensor:
    """Fréchet distance between two Gaussians. Gradients flow into ``mu_g`` and ``sig_g``."""
    diff = mu_r - mu_g
    sig_r_sqrt = _sqrtm(sig_r)
    mid = sig_r_sqrt @ sig_g @ sig_r_sqrt
    mid = 0.5 * (mid + mid.transpose(-1, -2))
    fd = diff.dot(diff) + torch.trace(sig_r) + torch.trace(sig_g) - 2.0 * torch.trace(_sqrtm(mid))
    return fd.clamp_min(0)


class EMAFrechet:
    """Paper EMA estimator. Reference moments are fixed. Generated moments are an EMA.

    ``set_reference`` warm-starts the EMA at the real cloud. The paper warm-starts
    from 50k generated samples, which this card cannot draw up front. Starting at
    the data means the first step is not a match against the zero vector.
    """

    def __init__(self, beta: float = 0.999, pool: int = 4):
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError(f"fd beta must be in [0, 1) (got {beta})")
        self.beta = float(beta)
        self.pool = int(pool)
        self.dim = 0
        self.mu_r = None
        self.sig_r = None
        self.mu_ema = None
        self.M_ema = None

    def set_reference(self, feats: torch.Tensor) -> None:
        feats = feats.detach().float()
        if feats.ndim != 2 or feats.shape[0] < 2:
            raise ValueError(
                f"fd reference needs at least 2 descriptors, got {tuple(feats.shape)}")
        n, d = feats.shape
        mu = feats.mean(dim=0)
        second = (feats.transpose(0, 1) @ feats) / n
        self.dim = int(d)
        self.mu_r = mu.detach()
        self.sig_r = (second - mu.outer(mu)).detach()
        self.mu_ema = mu.detach().clone()
        self.M_ema = second.detach().clone()

    def step(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalized FD. ``feat`` is ``[B, D]`` and is the only tensor with gradient."""
        if self.mu_r is None:
            raise RuntimeError("fd reference cloud was not set")
        feat = feat.float().reshape(-1, self.dim)
        if self.mu_ema.device != feat.device:
            self.mu_r = self.mu_r.to(feat.device)
            self.sig_r = self.sig_r.to(feat.device)
            self.mu_ema = self.mu_ema.to(feat.device)
            self.M_ema = self.M_ema.to(feat.device)
        b = feat.shape[0]
        mu_b = feat.mean(dim=0)
        second_b = (feat.transpose(0, 1) @ feat) / b
        mu_g = self.beta * self.mu_ema.detach() + (1.0 - self.beta) * mu_b
        second_g = self.beta * self.M_ema.detach() + (1.0 - self.beta) * second_b
        sig_g = second_g - mu_g.outer(mu_g)
        fd = frechet_distance(self.mu_r, self.sig_r, mu_g, sig_g)
        self.mu_ema = mu_g.detach()
        self.M_ema = second_g.detach()
        return fd / (fd.detach() + 1e-6)


def reference_from_dataset(group, pool: int = 4) -> torch.Tensor:
    """One descriptor per cached training latent. Skips caches with no video latent."""
    from safetensors.torch import load_file

    rows = []
    seen = set()
    datasets = getattr(group, "datasets", None) or [group]
    for ds in datasets:
        manager = getattr(ds, "batch_manager", None)
        buckets = getattr(manager, "buckets", None) if manager is not None else None
        if not buckets:
            buckets = getattr(ds, "buckets", None) or {}
        for items in buckets.values():
            for item in items:
                path = getattr(item, "latent_cache_path", None)
                if not path or path in seen or not os.path.isfile(path):
                    continue
                seen.add(path)
                sd = load_file(path)
                flag = sd.get("audio_only")
                if torch.is_tensor(flag) and bool(flag.detach().reshape(-1)[0].item()):
                    continue
                latent = sd.get("still_latent")
                if latent is None:
                    for key, value in sd.items():
                        if key.startswith("latent_"):
                            latent = value
                            break
                if not torch.is_tensor(latent):
                    continue
                rows.append(latent_descriptor(latent, pool).cpu())
    if len(rows) < 2:
        raise ValueError(
            f"fd-loss found {len(rows)} cached video latent(s); it needs at least 2")
    return torch.cat(rows, dim=0)
