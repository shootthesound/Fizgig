"""Qwen Image 2.1 sampling: FlowMatch Euler with the reference's dynamic exponential shift.

Mirrors diffusers `QwenImage21Pipeline.__call__` + `FlowMatchEulerDiscreteScheduler` with the checkpoint's
scheduler_config: use_dynamic_shifting, time_shift_type exponential, base_shift 0.5 @ 256 image tokens, max_shift
0.9 @ 8192, shift_terminal 0.02 (None for the Viggle turbo LoRA, which also supplies its own raw sigmas).
The model predicts the flow velocity (noise - x0); Euler: x <- x + (sigma_next - sigma) * v.
"""
import math

import numpy as np
import torch

BASE_SEQ, MAX_SEQ, BASE_SHIFT, MAX_SHIFT = 256, 8192, 0.5, 0.9
SHIFT_TERMINAL = 0.02


def calculate_mu(image_seq_len, base_seq=BASE_SEQ, max_seq=MAX_SEQ, base_shift=BASE_SHIFT, max_shift=MAX_SHIFT):
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return image_seq_len * m + (base_shift - m * base_seq)


def sigma_schedule(steps: int, image_seq_len: int, sigmas=None, shift_terminal=SHIFT_TERMINAL,
                   dynamic_shift=True) -> torch.Tensor:
    """Shifted sigmas for `steps` steps, with the terminal 0 appended (length steps + 1). dynamic_shift False: the
    sigmas as given (Qwen-Image-2.1-Turbo's scheduler_config: use_dynamic_shifting false, shift 1.0)."""
    s = np.linspace(1.0, 1.0 / steps, steps) if sigmas is None else np.asarray(sigmas, dtype=np.float64)
    if dynamic_shift:
        mu = calculate_mu(image_seq_len)
        s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))          # exponential time shift, sigma exponent 1
    if shift_terminal:
        one_minus = 1.0 - s
        s = 1.0 - one_minus / (one_minus[-1] / (1.0 - shift_terminal))
    return torch.tensor(np.append(s, 0.0), dtype=torch.float32)


def latent_hw(height: int, width: int):
    """Latent tokens per side (16x VAE, kept even like the reference's prepare_latents)."""
    return 2 * (height // 32), 2 * (width // 32)


def pack(latents: torch.Tensor) -> torch.Tensor:
    """[B, 64, h, w] or [B, 64, 1, h, w] -> [B, h*w, 64]."""
    if latents.ndim == 5:
        latents = latents[:, :, 0]
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2)


def unpack(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B, h*w, 64] -> [B, 64, 1, h, w] (the VAE's 5D layout)."""
    b, _, c = latents.shape
    return latents.transpose(1, 2).reshape(b, c, 1, h, w)


def model_inputs(text_emb: torch.Tensor, n_tokens: int, device, text_mask=None, ref_mask=None):
    """Batch-1 conditioning tensors for QwenImage21DiT.forward from one [L, 4096] embedding.
    text_mask: optional [L] bool marking real tokens (padded prompts, e.g. prompt travel); default all real.
    ref_mask: optional [L] bool marking reference images' vision tokens (edit), whose rows the DiT replaces with the
    references' latents (IMG_TOKENS_PER_SLOT latent tokens per vision token)."""
    L = text_emb.shape[0]
    txt_img = torch.zeros(L, dtype=torch.bool) if ref_mask is None else ref_mask.to(torch.bool).cpu()
    img_mask = torch.cat([txt_img, torch.ones(n_tokens // 4, dtype=torch.bool)])[None]
    enc_mask = torch.ones(1, L, dtype=torch.bool) if text_mask is None else text_mask.reshape(1, L)
    return text_emb[None].to(device), img_mask.to(device), enc_mask.to(device)


@torch.no_grad()
def sample(dit, text_emb, height, width, steps=25, seed=0, sigmas=None, shift_terminal=SHIFT_TERMINAL,
           dynamic_shift=True, cfg=1.0, neg_emb=None, device="cuda", dtype=torch.bfloat16, generator_device="cpu", noise=None,
           on_step=None, text_mask=None, neg_mask=None, ref_latents=None, ref_mask=None, neg_ref_mask=None):
    """Denoise one image; returns packed latents [1, N, 64] in float32. noise: a [1, 64, h, w] start (seed travel)
    instead of the seed's; on_step(done, total) is called before each step and may raise to abort."""
    h, w = latent_hw(height, width)
    n = h * w
    if noise is None:
        noise = initial_noise(seed, height, width, generator_device)
    x = noise.float().to(device).reshape(1, 64, n).transpose(1, 2).contiguous()   # reference pack
    sched = sigma_schedule(steps, n, sigmas, shift_terminal, dynamic_shift).to(device)
    enc, img_mask, enc_mask = model_inputs(text_emb, n, device, text_mask, ref_mask)
    if cfg > 1.0 and neg_emb is not None:
        nenc, nmask, nenc_mask = model_inputs(neg_emb, n, device, neg_mask, neg_ref_mask)
    refs = [r.float().to(device) for r in (ref_latents or [])]        # each [1, 64, rh, rw], clean
    ref_tokens = torch.cat([pack(r) for r in refs], dim=1) if refs else None
    shapes = [[(1, r.shape[-2], r.shape[-1]) for r in refs] + [(1, h, w)]]
    for i in range(len(sched) - 1):
        if on_step is not None:
            on_step(i, len(sched) - 1)
        s, s_next = sched[i], sched[i + 1]
        t = s.view(1).to(dtype)
        xin = x if ref_tokens is None else torch.cat([ref_tokens, x], dim=1)
        v = dit(xin.to(dtype), enc.to(dtype), t, shapes, img_mask, enc_mask)[:, -n:].float()
        if cfg > 1.0 and neg_emb is not None:
            vn = dit(xin.to(dtype), nenc.to(dtype), t, shapes, nmask, nenc_mask)[:, -n:].float()
            v = vn + cfg * (v - vn)
        x = x + (s_next - s) * v
    return x


def initial_noise(seed, height, width, generator_device="cpu"):
    """The seed's starting noise, [1, 64, h, w] float32 (what sample() draws when no noise is given)."""
    h, w = latent_hw(height, width)
    gen = torch.Generator(generator_device).manual_seed(int(seed))
    return torch.randn(1, 1, 64, h, w, generator=gen, dtype=torch.float32)[:, 0]


@torch.no_grad()
def decode(vae, packed: torch.Tensor, height: int, width: int):
    """Packed latents -> RGBA float in [-1, 1], [1, 4, H, W]."""
    h, w = latent_hw(height, width)
    z = unpack(packed, h, w)
    cfg = vae.config
    mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1, 1).to(z)
    std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1, 1).to(z)
    z = (z * std + mean).to(next(vae.parameters()).dtype)
    return vae.decode(z).sample[:, :, 0]


def encode_image(vae, rgb: torch.Tensor):
    """[B, 3 or 4, H, W] in [-1, 1] -> normalised latents [B, 64, h, w] (RGB gets an opaque alpha channel)."""
    if rgb.shape[1] == 3:
        rgb = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)
    x = rgb[:, :, None].to(next(vae.parameters()).dtype)
    z = vae.encode(x).latent_dist.mode()
    cfg = vae.config
    mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1, 1).to(z)
    std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1, 1).to(z)
    return ((z - mean) / std)[:, :, 0]
