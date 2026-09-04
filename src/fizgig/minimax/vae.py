"""MiniMax H3 video VAE — ENCODE PATH ONLY (image or clip -> 24-channel latent).

Pure-PyTorch port of the encoder half of ComfyUI's comfy/ldm/minimax/vae.py. Image-only
training needs to turn a still image into the DiT's 24-channel latent exactly once (caching),
so only the 3D-causal-CNN encoder + quant_conv + latent normalization are ported. The ViT3D
decoder and spatial/temporal tiling are omitted — no sampling/decode in scope.

Weight names match the checkpoint's `encoder.*` / `quant_conv.*` / `latents_mean/std`, so the
official minimax_h3_video_vae_fp16.safetensors loads with strict=False (decoder/post_quant keys
ignored). 16x spatial downscale; 4x causal temporal downscale, so a still (T=1) stays T=1 and a
clip of T frames yields ceil(T/4) latent frames.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608886, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.4498890042304993, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235595, 3.0496184825897216, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811524,
]


class CausalConv3d(nn.Conv3d):
    """Reflect spatial padding, causal (front-only, zero) temporal padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        cp = self.causal_padding
        if sum(cp) == 0:
            return super().forward(x)
        # spatial reflect (H, W), then temporal causal front-zeros — unifies the reference's
        # single-frame and multi-frame paths (front-pad by 2*cp[0] zeros is numerically the
        # single-frame "causal_zero" optimization).
        x = F.pad(x, (cp[2], cp[2], cp[1], cp[1], 0, 0), mode="reflect")
        x = F.pad(x, (0, 0, 0, 0, cp[0] * 2, 0), mode="constant")
        return super().forward(x)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    """GroupNorm with per-frame statistics (time folded into batch)."""
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
                                 stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(ResnetBlock3D(
                    in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                    out_channels=block_mid[i_level]))
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(block_mid[i_level], block_out[i_level],
                                               time_stride=time_down[i_level], space_stride=space_down[i_level])
            self.down.append(down)
        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(block_out[-1], 2 * z_channels if double_z else z_channels,
                                     kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class MiniMaxH3VideoVAEEncoder(nn.Module):
    """Encode-only. Load the full checkpoint with strict=False (decoder keys ignored)."""

    def __init__(self, in_channels=3, ch=128, embed_dim=24, z_channels=24,
                 ch_mult=(1, 2, 2, 4, 4, 8), num_res_blocks=2,
                 space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1)):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))        # 16
        self.vae_ratio_t = int(math.prod(time_down))       # 4
        self.encoder = EncoderFCN3D(ch=ch, ch_mult=list(ch_mult), space_down=list(space_down),
                                    time_down=list(time_down), num_res_blocks=num_res_blocks,
                                    in_channels=in_channels, z_channels=z_channels, double_z=True)
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN[:embed_dim]))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD[:embed_dim]))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @torch.no_grad()
    @staticmethod
    def plan_clip_bucket(free_gb, width, height, reserve_gb=1.5):
        """Largest /32 bucket of this shape whose clip encode fits in `free_gb`.

        Encoding a clip is far more expensive than encoding a still of the same size — the whole
        17-frame group is live at once — and a mixed dataset lets someone pick 1 MP for their
        photographs without knowing that no consumer card can cache a clip at it. Refusing the
        run would be the wrong answer: the stills are fine, and the clips are still worth having
        at a size that fits.

        MEASURED on a 5090, fp32, encoding in 17-frame groups:

            0.09 MP  5.6 GiB     0.26 MP  14.2 GiB     0.52 MP  28.0 GiB     0.66 MP  OOM

        which is 0.9 + 52 x megapixels, within 0.3 GiB across that range, and flat in clip length
        (124 frames costs 0.3 GiB more than 22). So the cap is a straight solve, with the same
        1.5 GiB left for the allocator and the display that the training planner reserves.
        """
        budget = max(0.0, float(free_gb) - float(reserve_gb) - 0.9)
        max_pixels = budget / 52.0 * 1e6
        if width * height <= max_pixels or max_pixels < 32 * 32:
            return int(width), int(height)
        scale = (max_pixels / (width * height)) ** 0.5
        w = max(32, int(width * scale) // 32 * 32)
        h = max(32, int(height * scale) // 32 * 32)
        return w, h

    def encode_clip(self, x):
        """A whole clip -> [B,24,T',H/16,W/16], encoded in the groups the model expects.

        Encoding a clip in ONE call is wrong twice over, and the first way is silent.

        H3's latent clock is (1,4,4,4,4) repeating — five latent frames covering seventeen pixel
        frames, restarting with a keyframe. Feeding all 22 frames of a clip to `encode` gives six
        latent frames; feeding 17 then 5 gives 5+2 = 7, which is what the DiT's position ids and
        audio clock are built for. Both numbers are plausible tensor shapes, so the wrong one does
        not raise — it just trains against a misaligned target. (Measured on the real weights.)

        The second way is loud: `encode` has no temporal chunking, so peak activation follows the
        whole clip at once. A 39-frame clip peaks at 30 GiB and a 56-frame one will not fit on a
        32 GB card at all. In 17-frame groups the peak is fixed by the group, so length costs
        nothing extra here and the ceiling moves to the training step where it belongs.
        """
        t = x.shape[2] if x.ndim == 5 else 1
        if t <= 1:
            return self.encode(x)
        if t < 5 or (t - 5) % 17:
            raise ValueError(f"{t} frames is not on H3's 17n+5 grid — the VAE could not have "
                             f"produced a latent for it")
        # n groups of 17, then the 5 that every grid length ends on: 22 = 17+5, 39 = 17+17+5.
        sizes = [17] * ((t - 5) // 17) + [5]
        out, i = [], 0
        for size in sizes:
            out.append(self.encode(x[:, :, i:i + size]))
            i += size
        return torch.cat(out, dim=2)

    def encode(self, x):
        """x: [B,3,H,W] or [B,3,T,H,W] in [-1,1] -> normalized latent [B,24,ceil(T/4),H/16,W/16].

        A CLIP should go through `encode_clip` instead — see there for why calling this with all
        of a clip's frames gives the wrong number of latent frames without raising.

        The stack is temporally causal with a 4x stride, so latent frame k depends only on pixel
        frames up to 4k (verified by perturbation) and a still gives T'=1 — the image path is
        unchanged. Feed 1+4k frames for a clean 1+k latent frames.

        No temporal chunking: peak activation is dominated by the first level at full T x H x W,
        so long clips want tiling that this port does not implement. Encode in clips.
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = (x + 1.0) * 0.5
        x = (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)
        moments = self.quant_conv(self.encoder(x))
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - lm) / ls


# ---------------------------------------------------------------------------------------
# DECODE PATH — the ViT3D decoder (latent -> pixels), ported from the ComfyUI reference.
#
# Every dimension is hardcoded there too (comfy/sd.py builds the VAE with zero arguments) and
# matches the checkpoint: 36 blocks, dim 2048 (32 heads x 64), gated-SiLU FFN, RMSNorm inside
# blocks with per-branch LayerScale, LayerNorm at the output, and proj_out emitting
# 3*4*16*16 — so ONE latent voxel becomes a 4-frame 16x16 pixel block.
#
# Three details are easy to get silently wrong, so each is called out where it happens:
#   * QKV is HEAD-MAJOR INTERLEAVED ([q|k|v] per head), not [3, heads, dim] like the DiT's;
#   * rope is split-half over the FIRST 48 of 64 head dims, the remaining 16 passing through;
#   * token ids are normalised to [-1, 1] over the actual T/H/W, so the decode shape is
#     semantically load-bearing, not merely a memory choice.
# ---------------------------------------------------------------------------------------

def create_token_ids(patch_dims, device=None, dtype=torch.float32):
    """[1, prod(dims), len(dims)] coordinates, each axis normalised to [-1, 1]."""
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device) / dim_size
        coords_list.append(2.0 * coords - 1.0)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    """3-axis rope producing a [B, S, 1, pairs, 2, 2] rotation table."""

    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        angles = (self.angle_scale * img_ids[:, :, :, None].float()
                  * self.inv_freq.to(img_ids.device)[None, None, None, :])
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        table = torch.stack([c, -s, s, c], dim=-1).reshape(
            *angles.shape[:2], 1, angles.shape[-1], 2, 2)
        return table.to(img_ids.dtype)


def _apply_rope_split_half(x, freqs_cis):
    """Split-half (GPT-NeoX style) rotation: the rotated span splits [first half | second half],
    NOT adjacent pairs. Pure-torch equivalent of comfy_kitchen's kernel."""
    t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
    out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return out.movedim(-1, -2).reshape(*x.shape).type_as(x)


class _VaeFeedForward(nn.Module):
    """Gated SiLU: w1 emits 2x inner width and the FIRST chunk is the gate."""

    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, inner * 2, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)

    def forward(self, x):
        gate, y = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * y)


class _VaeAttention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.heads, self.dim_head, self.eps = heads, dim_head, eps
        inner = heads * dim_head
        self.to_qkv = nn.Linear(inner, inner * 3, bias=bias)
        self.to_out = nn.Linear(inner, inner, bias=bias)

    def forward(self, x, rotary_pos_emb=None):
        b, s, _ = x.shape
        # HEAD-MAJOR INTERLEAVED: (b, s, heads, 3*dim_head), then split -> per head [q|k|v].
        # The DiT's attention splits [3, heads, dim]; copying either into the other silently
        # scrambles the projection with no error.
        qkv = self.to_qkv(x).view(b, s, -1, 3 * self.dim_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = F.rms_norm(q, (self.dim_head,), eps=self.eps)   # affine=False -> no learned weight
        k = F.rms_norm(k, (self.dim_head,), eps=self.eps)
        if rotary_pos_emb is not None:
            rot = rotary_pos_emb.shape[-3] * 2              # 48 of 64; the rest pass through
            q = torch.cat([_apply_rope_split_half(q[..., :rot], rotary_pos_emb), q[..., rot:]], -1)
            k = torch.cat([_apply_rope_split_half(k[..., :rot], rotary_pos_emb), k[..., rot:]], -1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        out = out.transpose(1, 2).reshape(b, s, -1).nan_to_num_(0.0)
        return self.to_out(out)


class _VaeTransformerBlock(nn.Module):
    """Pre-norm RMSNorm + LayerScale (scale1/scale2 are learned per-channel residual gains)."""

    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = _VaeAttention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = _VaeFeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rotary_pos_emb=None):
        x = x + self.scale1 * self.attn(self.norm1(x), rotary_pos_emb)
        return x + self.scale2 * self.ff(self.norm2(x))


class ViT3DDecoder(nn.Module):
    """[B, 24, T, H, W] latent -> [B, 3, 4T, 16H, 16W] pixels."""

    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3,
                 num_layers=36, heads=32, dim_head=64, rope_theta=100.0, rope_dim_ratio=0.75,
                 bias=True, eps=1e-5, num_register_tokens=4):
        super().__init__()
        dim = heads * dim_head
        self.patch_size, self.patch_size_t = patch_size, patch_size_t
        self.out_channels, self.num_register_tokens = out_channels, num_register_tokens
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # Present only so the checkpoint loads without an unexpected key — never read. The
        # reference appends a literal zero row where you might expect this to be used.
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [_VaeTransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
             for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)   # LayerNorm, not RMS
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x):
        B, _C, lt, lh, lw = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))       # one latent voxel = one token
        num_patches = h.shape[1]
        h = torch.cat([h, self.register_tokens.expand(B, -1, -1).to(h.dtype),
                       torch.zeros_like(h[:, 0:1, :])], dim=1)
        img_ids = create_token_ids((lt, lh, lw), x.device, torch.float32).expand(B, -1, -1)
        suffix = torch.zeros((B, 1 + self.num_register_tokens, 3),
                             device=x.device, dtype=img_ids.dtype)
        rope = self.pos_embed(torch.cat([img_ids, suffix], dim=1))
        for block in self.transformer_blocks:
            h = block(h, rope)
        out = self.proj_out(self.norm_out(h))[:, :num_patches, :]
        out = out.view(B, lt, lh, lw, self.out_channels,
                       self.patch_size_t, self.patch_size, self.patch_size)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return out.reshape(B, self.out_channels, lt * self.patch_size_t,
                           lh * self.patch_size, lw * self.patch_size)


class MiniMaxH3VideoVAEDecoder(nn.Module):
    """Decode-only companion to the encoder above: normalized latent -> pixels in [0, 1].

    Single-frame path only (latent_t == 1), which is the model's native image convention and
    exactly what the encoder produces for a still — the reference has the same special case
    (`if z.shape[2] == 1`). Longer clips need its temporal chunking, which is not ported.

    Spatial tiling is likewise not ported: the reference tiles above 256 px per axis, and since
    token ids are resolution-normalised a single-pass decode of a larger image is not identical
    to its tiled one. For diagnostic previews that is an acceptable difference — but it is a
    real one, so it is measured (see the round-trip test) rather than assumed."""

    def __init__(self, embed_dim=24, z_channels=24):
        super().__init__()
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)   # a real learned 1x1x1 mix
        self.decoder = ViT3DDecoder(in_channels=z_channels)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN[:embed_dim]))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD[:embed_dim]))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1),
                             persistent=False)

    # Spatial tiling. NOT optional above 256 px: because create_token_ids normalises coordinates
    # over the actual H/W, a single-pass decode of a larger image drifts off the trained regime
    # and leaves visible 16-px patch seams. Measured seam energy (1.0 = none): 256 px single-pass
    # 1.09, but 512 px single-pass 2.31 — a mesh you can see. Tiled, 512 px comes back to ~1.1.
    _TILE_PX = 256
    _TILE_OVERLAP_MIN = 64
    vae_ratio = 16          # spatial compression, same constant the encoder uses

    # A lone temporal token is out of distribution for this chunk-trained decoder, so the single
    # latent is replicated before decoding. Two schemes, selected by `single_frame_mode`:
    #
    #   "group" (DEFAULT) — replicate to a full 5-latent temporal group, keep frame 3 (past the
    #       causal lead-in). Wins on every measure taken: round-trip fidelity 29.99 dB mean vs
    #       16.96 dB (tests/diag_frame_choice.py, real photos), and confirmed better on real
    #       rendered previews too. Costs 2.5x the decode tokens, which is small at preview size.
    #   "reference" — cat([z, z]) -> a 2-latent clip, keep pixel frame 0. What ai-toolkit does.
    #       Tried as the default on 4 Aug and reverted: a 2-latent pad is still too short for a
    #       decoder trained on 5-latent chunks, and it scores barely above the raw lone token
    #       (16.96 vs 16.64 dB). Only a COMPLETE (1, 4, 4, 4, 4) group puts the ViT back in its
    #       training regime, and then only an interior frame is clean — frames 0 and 4 sit at
    #       the group boundary and lose ~10 dB.
    #
    # This is the one place Fizgig deliberately diverges from the reference. Set
    # `decoder.single_frame_mode = "reference"` to A/B.
    _T_GROUP = 5
    _LEAD_IN = 3            # "group": the reference drops a 3-frame causal lead-in
    _REF_T = 2              # "reference": cat([z, z])
    _REF_FRAME = 0
    single_frame_mode = "group"

    def _pad_and_index(self):
        """(replication count, pixel frame to keep) for the active single-frame scheme."""
        if self.single_frame_mode == "group":
            return self._T_GROUP, self._LEAD_IN
        if self.single_frame_mode == "reference":
            return self._REF_T, self._REF_FRAME
        raise ValueError(f"single_frame_mode must be 'reference' or 'group', "
                         f"got {self.single_frame_mode!r}")

    @torch.no_grad()
    def decode(self, z):
        """z: normalized latent [B, 24, 1, H, W] -> pixels [B, 3, H*16, W*16] in [0, 1].

        The single latent frame is replicated before decoding — `create_token_ids` normalises
        the t coordinate over `latent_T`, so a lone token sits at t=0, outside the range the
        decoder ever saw (measured: 16 dB and visibly dark). See `single_frame_mode` above for
        which replication scheme runs."""
        if z.shape[2] != 1:
            return self.decode_clip(z)
        n_rep, keep = self._pad_and_index()
        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        # Follow the module's own dtype rather than forcing fp32: this decoder is 2.4 B params,
        # so fp32 residency is 9.7 GB against 4.8 GB for a 16-bit dtype — a difference that
        # matters when it is loaded on top of the resident base for a preview. Callers should
        # load it FP16 (the weights' native format, and the only 16-bit dtype ComfyUI permits
        # for this VAE), not bf16.
        w_dtype = self.post_quant_conv.weight.dtype
        # post_quant_conv is a real learned 1x1x1 channel mix, NOT an identity — skipping it
        # leaves the image structurally recognisable but badly wrong (measured: 7 dB PSNR).
        zz = self.post_quant_conv((z * ls + lm).to(w_dtype)).repeat(1, 1, n_rep, 1, 1)
        dec = self._tiled_decode(zz)[:, :, keep]        # -> [B, 3, H*16, W*16]
        dec = dec.float() * self.pixel_std.to(dec)[:, :, 0] + self.pixel_mean.to(dec)[:, :, 0]
        return dec.clamp_(0.0, 1.0)

    # --- multi-frame decode: the reference's temporal chunking, ported ------------------------
    # comfy/ldm/minimax/vae.py::decode_temporal with its constants resolved for this VAE:
    # vae_ratio_t=4, clip_length=17 -> tokens_chunk_size=5, token_overlap=(-3)%5=2,
    # frame_pre_padding=(-17)%4=3, frame_overlap=max(2*4-3,0)=5. Each chunk decodes 5+2
    # latent tokens spatially tiled, drops the 3-frame causal lead-in, keeps the first 17
    # frames, and cross-fades a 5-frame overlap into the next chunk — so 5n+2 latents come
    # back as exactly 17n+5 pixel frames.
    _RATIO_T = 4
    _CHUNK_TOK = 5
    _TOK_OVERLAP = 2
    _PRE_PAD = 3
    _FRAME_OVERLAP = 5

    @torch.no_grad()
    def decode_clip(self, z):
        """z: normalized latent [B, 24, T_lat, H, W] (T_lat on the 5n+2 grid) ->
        pixels [B, 3, 17n+5, H*16, W*16] in [0, 1]."""
        from fizgig.minimax.model import pixel_frames_for_latent
        out_frames = pixel_frames_for_latent(int(z.shape[2]))    # validates the grid too
        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        w_dtype = self.post_quant_conv.weight.dtype
        zz = self.post_quant_conv((z * ls + lm).to(w_dtype))

        # token padding: pseudo length includes the 3 dropped-by-encode tokens
        pseudo = zz.shape[2] + self._PRE_PAD
        pad_tokens = (-pseudo) % self._CHUNK_TOK
        num_chunks = (pseudo + pad_tokens) // self._CHUNK_TOK - 1
        if num_chunks < 1:
            pad_tokens += self._CHUNK_TOK
            num_chunks += 1
        if pad_tokens:
            zz = torch.cat([zz, zz[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        chunk_dec = self._CHUNK_TOK * self._RATIO_T
        dec, dec_overlap, write_pos = None, None, 0

        def write_part(part):
            nonlocal dec, write_pos
            if part.shape[2] <= 0:
                return
            if dec is None:
                # fp32 from the start: copy_ casts each part exactly as the old final
                # .float() did, and the scale below runs in place — no second full-clip
                # copy (316 MB at 768×640×56 — the allocation a 24 GB card ran out on).
                shape = list(part.shape)
                shape[2] = out_frames
                dec = torch.empty(shape, dtype=torch.float32, device=part.device)
            n = min(part.shape[2], max(0, dec.shape[2] - write_pos))
            if n > 0:
                dec[:, :, write_pos:write_pos + n].copy_(part[:, :, :n])
                write_pos += n

        for i in range(num_chunks):
            t0 = i * self._CHUNK_TOK
            clip = zz[:, :, t0:t0 + self._CHUNK_TOK + self._TOK_OVERLAP]
            clip_dec = self._tiled_decode(clip)                  # [B,3,4*tok,h,w], spatially tiled
            for j in range(2):                                   # split_count with token_drop > 0
                f0 = j * chunk_dec
                f1 = min(f0 + chunk_dec, clip_dec.shape[2])
                # the reference drops the 3-frame causal lead-in from EVERY split chunk:
                # j=0 -> frames [3:20] (17 kept), j=1 -> [23:28] (the 5-frame overlap)
                if j == 0:
                    part = clip_dec[:, :, self._PRE_PAD:f1]
                    if dec_overlap is not None:
                        part = self._blend(dec_overlap, part, self._FRAME_OVERLAP, dim=-3)
                        dec_overlap = None
                    write_part(part)
                else:
                    dec_overlap = clip_dec[:, :, f0 + self._PRE_PAD:f1].contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
            del clip_dec, clip

        dec.mul_(self.pixel_std.to(dec)).add_(self.pixel_mean.to(dec))
        return dec.clamp_(0.0, 1.0)

    @torch.no_grad()
    def decode_middle_frame(self, z, frame_idx=None):
        """Decode ONE frame of a clip latent by decoding only the temporal chunk(s) that
        contribute to it. Returns pixels [B, 3, H*16, W*16] in [0, 1] — numerically identical
        to `decode_clip(z)[:, :, frame_idx]`, including the cross-fade when the frame sits in
        a chunk-boundary blend zone (then the previous chunk decodes too).

        At 22 frames the whole clip is one chunk, so this saves nothing; at 56 frames it
        decodes 1 of 3 chunks, at 124 one of 7. `frame_idx=None` means the middle frame."""
        from fizgig.minimax.model import pixel_frames_for_latent
        out_frames = pixel_frames_for_latent(int(z.shape[2]))
        if z.shape[2] == 1:
            return self.decode(z)
        if frame_idx is None:
            frame_idx = out_frames // 2
        frame_idx = max(0, min(out_frames - 1, int(frame_idx)))

        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        w_dtype = self.post_quant_conv.weight.dtype
        zz = self.post_quant_conv((z * ls + lm).to(w_dtype))

        # Same chunk bookkeeping as decode_clip.
        pseudo = zz.shape[2] + self._PRE_PAD
        pad_tokens = (-pseudo) % self._CHUNK_TOK
        num_chunks = (pseudo + pad_tokens) // self._CHUNK_TOK - 1
        if num_chunks < 1:
            pad_tokens += self._CHUNK_TOK
            num_chunks += 1
        if pad_tokens:
            zz = torch.cat([zz, zz[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        chunk_dec = self._CHUNK_TOK * self._RATIO_T
        chunk_len = chunk_dec - self._PRE_PAD              # 17 kept pixel frames per chunk

        def _chunk_parts(i):
            t0 = i * self._CHUNK_TOK
            cd = self._tiled_decode(zz[:, :, t0:t0 + self._CHUNK_TOK + self._TOK_OVERLAP])
            main = cd[:, :, self._PRE_PAD:chunk_dec]       # the 17 frames this chunk owns
            overlap = cd[:, :, chunk_dec + self._PRE_PAD:]  # up to 5 frames fed forward
            return main, overlap

        tail_start = num_chunks * chunk_len
        if frame_idx >= tail_start:
            # The final 17n..17n+4 frames are the last chunk's overlap, written raw.
            _, overlap = _chunk_parts(num_chunks - 1)
            frame = overlap[:, :, frame_idx - tail_start]
        else:
            ci = frame_idx // chunk_len
            local = frame_idx - ci * chunk_len
            main, _ = _chunk_parts(ci)
            frame = main[:, :, local]
            if ci > 0 and local < self._FRAME_OVERLAP:
                # Cross-fade zone: blend with the previous chunk's overlap using the same
                # weights _blend applies at this position (wa = 1 - pos/extent).
                _, prev_overlap = _chunk_parts(ci - 1)
                extent = min(prev_overlap.shape[2], chunk_len, self._FRAME_OVERLAP)
                if local < extent:
                    wb = torch.tensor(local / extent, dtype=frame.dtype, device=frame.device)
                    frame = prev_overlap[:, :, local] * (1 - wb) + frame * wb
        frame = frame.float() * self.pixel_std.to(frame)[:, :, 0] + self.pixel_mean.to(frame)[:, :, 0]
        return frame.clamp_(0.0, 1.0)

    def _decode_tile(self, zz):
        """One spatial tile: the full decoded clip [B, 3, 4*T, h*16, w*16] — temporal selection
        is the caller's job (single-frame keeps one frame; decode_clip slices per chunk)."""
        return self.decoder(zz)

    def _split_tiles(self, input_len):
        """Tile starts/lengths/overlaps in PIXELS, matching the reference's layout."""
        tile = self._TILE_PX
        if tile >= input_len:
            return [0], [input_len], []
        n = math.ceil(input_len / tile)
        while True:
            overlaps = [self._TILE_OVERLAP_MIN] * (n - 1)
            remaining = tile * n - sum(overlaps) - input_len
            if remaining < 0:
                n += 1
            else:
                break
        for i in range(remaining // self.vae_ratio):
            overlaps[i % (n - 1)] += self.vae_ratio
        starts = [0]
        for i in range(n - 1):
            starts.append(starts[-1] + tile - overlaps[i])
        return starts, [tile] * n, overlaps

    @staticmethod
    def _blend(a, b, extent, dim):
        extent = min(a.shape[dim], b.shape[dim], extent)
        pos = torch.arange(extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = extent
        wa, wb = (1 - pos / extent).view(shape), (pos / extent).view(shape)
        sa = [slice(None)] * a.ndim; sa[dim] = slice(-extent, None)
        sb = [slice(None)] * b.ndim; sb[dim] = slice(0, extent)
        blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
        if extent < b.shape[dim]:
            rest = [slice(None)] * b.ndim; rest[dim] = slice(extent, None)
            return torch.cat([blended, b[tuple(rest)]], dim=dim)
        return blended

    def _tiled_decode(self, zz):
        """Decode in overlapping 256-px tiles, cross-faded — the reference's scheme. Collapses to
        a single pass (no seams to blend) whenever the image already fits one tile."""
        r = self.vae_ratio
        height, width = zz.shape[-2] * r, zz.shape[-1] * r
        y_idx, y_len, y_ov = self._split_tiles(height)
        x_idx, x_len, x_ov = self._split_tiles(width)
        if len(y_idx) == 1 and len(x_idx) == 1:
            return self._decode_tile(zz)
        canvas, row_tails, out_y = None, [], 0
        for i, (ip, il) in enumerate(zip(y_idx, y_len)):
            zi, zl = ip // r, il // r
            new_tails, left_tail, out_x = [], None, 0
            for j, (jp, jl) in enumerate(zip(x_idx, x_len)):
                zj, zw = jp // r, jl // r
                tile = self._decode_tile(zz[..., zi:zi + zl, zj:zj + zw])
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_ov[i]:, :].clone())
                next_left = tile[..., :, -x_ov[j]:].clone() if j < len(x_idx) - 1 else None
                if i > 0:
                    tile = self._blend(row_tails[j], tile, y_ov[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(left_tail, tile, x_ov[j - 1], dim=-1)
                left_tail = next_left
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_ov[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_ov[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width,
                                         dtype=tile.dtype, device=tile.device)
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas
