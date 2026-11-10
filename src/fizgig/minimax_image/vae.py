# The tiling, the decoder's 2D variant and the encode / decode steps in this file are derived from ComfyUI
# (https://github.com/comfyanonymous/ComfyUI, comfy/ldm/minimax/vae.py: MiniMaxH3VideoVAE.split_tiles / blend /
# tiled_encode / tiled_decode, H3ImageVAE, MiniMaxH3ImageVAE), Copyright (C) comfyanonymous and the ComfyUI
# contributors, licensed under the GNU General Public License v3.0. This file therefore stays under GPL-3.0 and is not
# covered by Fizgig's Apache-2.0 licence. See docs/THIRD_PARTY_NOTICES.md (ComfyUI).
"""The MiniMax-H3-Image VAE: MiniMax H3's video VAE encoder (bit-identical weights: the 118 encoder + quant_conv
tensors of minimax_h3_image_vae_fp16.safetensors equal minimax_h3_video_vae_fp16's) and a decoder of its own, a 24-layer
ViT over one frame (16 heads x 64, rope over the (h, w) axes on half of each head) behind a 2D post-quant conv.

As ComfyUI runs it (H3ImageVAE / MiniMaxH3ImageVAE): both directions tile the picture in 256 px tiles overlapping by at
least 64 px, blended linearly; encoding returns a sample of the posterior drawn from a fixed seed (42, the H3 keyframe
recipe's) rather than its mean; decoding returns pixels in [0, 1].
"""
import math

import torch
import torch.nn as nn

from fizgig.minimax.vae import (IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAEEncoder,
                                RotaryEmbeddingND, ViT3DDecoder, create_token_ids)

IMAGE_ENCODE_SEED = 42
TILE_PX = 256
TILE_OVERLAP_MIN = 64
VAE_RATIO = 16


class ImageViTDecoder(ViT3DDecoder):
    """H3's ViT decoder with ComfyUI's image settings: one frame per latent (patch_size_t 1), 24 layers, 16 heads,
    rope over the two trailing axes (h, w) on half of each head (rope_dims=2, rope_dim_ratio=0.5)."""

    def __init__(self, num_layers=24, heads=16, dim_head=64, rope_theta=100.0, rope_dim_ratio=0.5):
        super().__init__(patch_size=VAE_RATIO, patch_size_t=1, num_layers=num_layers, heads=heads,
                         dim_head=dim_head, rope_theta=rope_theta, rope_dim_ratio=rope_dim_ratio)
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=2)

    def forward(self, x):
        B, _C, lt, lh, lw = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = h.shape[1]
        h = torch.cat([h, self.register_tokens.expand(B, -1, -1).to(h.dtype), torch.zeros_like(h[:, 0:1, :])], dim=1)
        ids = create_token_ids((lt, lh, lw)[-2:], x.device, torch.float32).expand(B, -1, -1)
        suffix = torch.zeros((B, 1 + self.num_register_tokens, 2), device=x.device, dtype=ids.dtype)
        rope = self.pos_embed(torch.cat([ids, suffix], dim=1))
        for block in self.transformer_blocks:
            h = block(h, rope)
        out = self.proj_out(self.norm_out(h))[:, :num_patches, :]
        out = out.view(B, lt, lh, lw, self.out_channels, 1, self.patch_size, self.patch_size)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return out.reshape(B, self.out_channels, lt, lh * self.patch_size, lw * self.patch_size)


def split_tiles(input_len, tile_size=TILE_PX, overlap_min=TILE_OVERLAP_MIN, ratio=VAE_RATIO):
    if tile_size >= input_len:
        return [0], [input_len], []
    n = math.ceil(input_len / tile_size)
    while True:
        overlaps = [overlap_min] * (n - 1)
        remaining = tile_size * n - sum(overlaps) - input_len
        if remaining < 0:
            n += 1
        else:
            break
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    idx = [0]
    for i in range(n - 1):
        idx.append(idx[-1] + tile_size - overlaps[i])
    return idx, [tile_size] * n, overlaps


def blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    pos = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    wa, wb = (1 - pos / extent).view(shape), (pos / extent).view(shape)
    out = b.clone()
    sl_a = [slice(None)] * a.ndim
    sl_a[dim] = slice(a.shape[dim] - extent, a.shape[dim])
    sl_b = [slice(None)] * b.ndim
    sl_b[dim] = slice(0, extent)
    out[tuple(sl_b)] = a[tuple(sl_a)] * wa + b[tuple(sl_b)] * wb
    return out


class MiniMaxH3ImageVAE(nn.Module):
    """Encode (pixels in [-1, 1] -> normalised latents) and decode (latents -> pixels in [0, 1]), both tiled. Either
    half can be left out: training caches with the encoder only, previews decode with the decoder only."""

    def __init__(self, encoder=True, decoder=True):
        super().__init__()
        self.enc = MiniMaxH3VideoVAEEncoder() if encoder else None
        if decoder:
            self.post_quant_conv = nn.Conv2d(24, 24, 1)
            self.decoder = ImageViTDecoder()
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def load_file(self, path):
        from safetensors import safe_open
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(("encoder.", "quant_conv.")):
                    if self.enc is not None:
                        sd["enc." + k] = f.get_tensor(k)
                elif k.startswith(("decoder.", "post_quant_conv.")):
                    if hasattr(self, "decoder"):
                        sd[k] = f.get_tensor(k)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith(("latents_mean", "latents_std")) and not m.startswith("enc.l")]
        if missing or unexpected:
            raise ValueError(f"{path} is not a MiniMax-H3-Image VAE: missing {missing[:4]}, unexpected {unexpected[:4]}")
        return self

    # ---- encode -----------------------------------------------------------------------------------
    def _moments(self, x):
        return self.enc.quant_conv(self.enc.encoder(x))

    def _tiled_moments(self, x):
        H, W = x.shape[-2], x.shape[-1]
        yi, yl, yo = split_tiles(H)
        xi, xl, xo = split_tiles(W)
        rows = [[self._moments(x[..., a:a + al, b:b + bl]) for b, bl in zip(xi, xl)] for a, al in zip(yi, yl)]
        ly, lx = [o // VAE_RATIO for o in yo], [o // VAE_RATIO for o in xo]
        out_rows = []
        for i, row in enumerate(rows):
            parts = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = blend(rows[i - 1][j], tile, ly[i - 1], dim=-2)
                if j > 0:
                    tile = blend(row[j - 1], tile, lx[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-ly[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-lx[j]]
                parts.append(tile)
            out_rows.append(torch.cat(parts, dim=-1))
        return torch.cat(out_rows, dim=-2)

    @torch.no_grad()
    def encode(self, x):
        """x [B, 3, H, W] (or [B, 3, 1, H, W]) in [-1, 1], H and W multiples of 16 -> [B, 24, 1, H/16, W/16]."""
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = (x + 1.0) * 0.5
        x = (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)
        moments = self._tiled_moments(x)[:, :, -1:]
        mean, logvar = moments.float().chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(IMAGE_ENCODE_SEED)).to(mean.device)
        z = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)
        lm, ls = self.latents_mean.view(1, -1, 1, 1, 1).to(z), self.latents_std.view(1, -1, 1, 1, 1).to(z)
        return (z - lm) / ls

    # ---- decode -----------------------------------------------------------------------------------
    def _pixels(self, z):
        return self.decoder(self.post_quant_conv(z.squeeze(2)).unsqueeze(2))

    @torch.no_grad()
    def decode(self, z):
        """z [B, 24, 1, h, w] normalised -> float32 pixels [B, 3, 1, 16h, 16w] in [0, 1]."""
        if z.shape[2] != 1:
            raise ValueError(f"the MiniMax-H3-Image VAE decodes single-frame latents, got {z.shape[2]} frames")
        z = z * self.latents_std.view(1, -1, 1, 1, 1).to(z) + self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        H, W = z.shape[-2] * VAE_RATIO, z.shape[-1] * VAE_RATIO
        yi, yl, yo = split_tiles(H)
        xi, xl, xo = split_tiles(W)
        canvas = None
        strip = None
        out_y = 0
        for i, (a, al) in enumerate(zip(yi, yl)):
            zrow = z[..., a // VAE_RATIO:(a + al) // VAE_RATIO, :]
            new_strip, left_tail, out_x = None, None, 0
            for j, (b, bl) in enumerate(zip(xi, xl)):
                tile = self._pixels(zrow[..., b // VAE_RATIO:(b + bl) // VAE_RATIO]).float()
                if i > 0:
                    tile = blend(strip[..., :, b:b + bl], tile, yo[i - 1], dim=-2)
                if j > 0:
                    tile = blend(left_tail, tile, xo[j - 1], dim=-1)
                left_tail = tile[..., :, -xo[j]:].clone() if j < len(xi) - 1 else None
                if j < len(xi) - 1:
                    tile = tile[..., :, :-xo[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], H, W, dtype=tile.dtype, device=tile.device)
                if i < len(yi) - 1:
                    if new_strip is None:
                        new_strip = torch.empty(*tile.shape[:-2], yo[i], W, dtype=tile.dtype, device=tile.device)
                    new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -yo[i]:, :]
                    tile = tile[..., :-yo[i], :]
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                tile_h = tile.shape[-2]
                out_x += tile.shape[-1]
            strip = new_strip
            out_y += tile_h
        px = canvas * self.pixel_std.to(canvas) + self.pixel_mean.to(canvas)
        return px.clamp_(0.0, 1.0)


def load_image_vae(path, device, *, encoder=True, decoder=True, dtype=torch.float32):
    return MiniMaxH3ImageVAE(encoder=encoder, decoder=decoder).load_file(path).to(device=device, dtype=dtype).eval()
