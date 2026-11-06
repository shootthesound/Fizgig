"""Vimoo 1.0 Lite's video transformer (Vimoo-AI, Apache-2.0): 40 dual-stream blocks over a 2 x 2 hidden patch merge.

Ported from Vimoo's diffusers fork (models/transformers/transformer_vimoo.py, numerical_profile "lite") with the
module names of their ComfyUI single file (vimoo_v1_lite_bf16.safetensors), so the file loads as it is and a LoRA
Fizgig writes is the kohya form ComfyUI's LoRA loader maps onto the same names. The arithmetic follows the "lite"
profile, which the checkpoint was made with: FP32 timestep features (FP64 frequencies), FP32 low-rank modulation and
residual additions, exact GELU, output-projection biases added in FP32, the packed QK RMS-norm summed in its fixed
order, and an FP32 prediction.

Tokens: the 97-channel input (48 noisy latent + 48 reference latent + 1 mask) is projected per latent cell
(img_in.proj, 1 x 1 x 1) and each 2 x 2 window is merged in hidden space (img_in.patch_merge); the final layer splits
each token back into its 4 cells (final_layer.patch_split). Text: Qwen2.5-VL states through a 2-block token
refiner (txt_in), ByT5 glyph states through byt5_in, SigLIP states (image-to-video) through vision_in; the three are
concatenated with their valid tokens first (the reference's reorder) and join the video stream in every block.

Block swap streams double_blocks through Fizgig's shared offloader, as Fibo 2.0 and MOSS-Image use it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LITE_CONFIG = dict(hidden_size=3072, heads=24, depth=40, mlp_ratio=4.0, in_channels=48, mod_rank=16,
                   text_dim=3584, byt5_dim=1472, vision_dim=1152, vision_tokens=729, rope_dims=(16, 56, 56),
                   rope_theta=256.0, yarn_ref_area=1600.0, ref_fps=16.0, patch=(1, 2, 2))


# ---- linear layers with the reference's precision ---------------------------------------------------------------
class FP32Linear(nn.Linear):
    """Weights kept in FP32 and the product taken in FP32 (img_in, time_in, mod_projection, the block modulation,
    the final split / projection): the reference's fp32_io_time policy."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = F.linear(x.float(), self.weight.float())
            return out + self.bias.float() if self.bias is not None else out


class BiasLinear(nn.Linear):
    """The product in the weight's dtype, the bias added after it (the reference's VimooSeparateBiasLinear)."""

    def forward(self, x):
        out = F.linear(x.to(self.weight.dtype), self.weight)
        return out + self.bias.to(out.dtype) if self.bias is not None else out


class BiasUpLinear(nn.Linear):
    """The product in the weight's dtype and the bias added in FP32 (the attention and MLP output projections of
    the video stream, and both MLP outputs: the reference's VimooDeferredBiasProjection(upcast_bias=True))."""

    def forward(self, x):
        out = F.linear(x.to(self.weight.dtype), self.weight)
        return out.float() + self.bias.float() if self.bias is not None else out


def _modulate(x, shift, scale, observed=None):
    """FP32 adaptive LayerNorm modulation; observed: [1, N, 1] True for the first latent frame's tokens, which take
    the second of two modulations (shift / scale shaped [2, B, D])."""
    xf = x.float()
    if observed is not None:
        return torch.where(observed, xf * (1 + scale[1].float().unsqueeze(1)) + shift[1].float().unsqueeze(1),
                           xf * (1 + scale[0].float().unsqueeze(1)) + shift[0].float().unsqueeze(1)).to(x.dtype)
    return (xf * (1 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)).to(x.dtype)


def _gated_add(residual, update, gate, observed=None):
    u = update.float()
    if observed is not None:
        u = torch.where(observed, u * gate[1].float().unsqueeze(1), u * gate[0].float().unsqueeze(1))
    else:
        u = u * gate.float().unsqueeze(1)
    return (residual.float() + u).to(residual.dtype)


class QKNorm(nn.Module):
    """RMS norm of query and key over the head dim, its sum of squares taken in the reference's fixed order
    (groups of 4, then pairwise), FP32 scales. Names match the file: norm.query_norm / norm.key_norm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.query_norm = _Scale(dim)
        self.key_norm = _Scale(dim)

    def forward(self, q, k):
        qk = torch.stack((q, k), 0)
        h = qk.float()
        d = h.shape[-1]
        if d % 4 == 0 and (d // 4) & (d // 4 - 1) == 0:
            sq = h.unflatten(-1, (d // 4, 4)).square()
            var = sq[..., 0] + sq[..., 1]
            var = (var + sq[..., 2]) + sq[..., 3]
            while var.shape[-1] > 1:
                var = var[..., ::2] + var[..., 1::2]
            var = var / d
        else:
            var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        q2, k2 = h.unbind(0)
        return ((q2 * self.query_norm.weight.float()).to(qk.dtype), (k2 * self.key_norm.weight.float()).to(qk.dtype))


class _Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


# ---- rotary position embedding ------------------------------------------------------------------------------------
def _rotate_half(x):
    re, im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-im, re], dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    """Interleaved RoPE on [B, N, H, D], in FP32."""
    shape = [1, x.shape[1], 1, x.shape[-1]]
    c, s = cos.view(*shape), sin.view(*shape)
    xf = x.float()
    return (xf * c + _rotate_half(xf) * s).to(x.dtype)


def rope_table(grid, latent_hw, fps, cfg, device):
    """cos / sin [N, 128] for a (T, H/2, W/2) token grid: positions on the latent lattice (spatial x2), YaRN on the
    spatial axes from the latent area (reference area 1600 cells), the temporal axis divided by fps / 16."""
    lh, lw = latent_hw
    spatial = max(1.0, math.sqrt((lh * lw) / cfg["yarn_ref_area"]))
    temporal = float(fps) / cfg["ref_fps"]
    if temporal <= 0:
        raise ValueError("fps must be positive")
    axes = []
    for axis, size in enumerate(grid):
        pos = torch.arange(int(size), dtype=torch.float32, device=device) * float(cfg["patch"][axis])
        if axis == 0:
            pos = pos / torch.tensor(temporal, device=device, dtype=torch.float32)
        axes.append(pos)
    mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    cos_parts, sin_parts = [], []
    for i, dim in enumerate(cfg["rope_dims"]):
        freqs = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2]
                                             / dim))
        if i != 0:                                      # YaRN ramp on height and width
            pair = torch.linspace(0.0, 1.0, dim // 2, dtype=freqs.dtype, device=device)
            mix = torch.clamp((pair - 0.25) / 0.5, 0.0, 1.0)
            freqs = freqs / (1.0 + (spatial - 1.0) * mix)
        ang = torch.outer(mesh[i].reshape(-1).float(), freqs)
        cos_parts.append(ang.cos().repeat_interleave(2, dim=1))
        sin_parts.append(ang.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos_parts, 1), torch.cat(sin_parts, 1)


def _attention(q, k, v, key_mask=None):
    """[B, N, H, D] -> [B, N, H * D]; key_mask [B, L] bool (False = padding) or None."""
    from fizgig.modules.sdpa import sdpa_backend_ctx
    from fizgig.modules import int8_attention as _i8a
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = _i8a.attend(q, k, v, key_mask)         # workbench renders: the INT8 kernel when it's switched on, no padding
    if out is not None:
        return out.transpose(1, 2).flatten(2)
    mask = key_mask[:, None, None, :] if key_mask is not None else None
    with sdpa_backend_ctx():
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


# ---- conditioning ---------------------------------------------------------------------------------------------------
def _timestep_features(t, dim=256, max_period=10000):
    """The lite profile's sinusoid: FP64 frequencies where the device has them, cos then sin."""
    half = dim // 2
    dt = torch.float64 if t.device.type not in ("mps",) else torch.float32
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=dt, device=t.device) / half)
    ang = t.to(dt)[:, None] * freqs[None]
    return torch.cat((ang.cos(), ang.sin()), dim=-1).float()


class TimeMLP(nn.Module):
    """time_in / txt_in.t_embedder: sinusoid -> Linear -> SiLU -> Linear, all FP32."""

    def __init__(self, hidden):
        super().__init__()
        self.in_layer = FP32Linear(256, hidden)
        self.out_layer = FP32Linear(hidden, hidden)

    def forward(self, t):
        with torch.autocast(device_type=t.device.type, enabled=False):
            return self.out_layer(F.silu(self.in_layer(_timestep_features(t))))


class TextProjection(nn.Module):
    """txt_in.c_embedder: Linear -> SiLU -> Linear (PixArt's text projection)."""

    def __init__(self, din, hidden):
        super().__init__()
        self.in_layer = BiasLinear(din, hidden)
        self.out_layer = BiasLinear(hidden, hidden)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class RefinerAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.qkv = BiasLinear(hidden, hidden * 3)
        self.proj = BiasLinear(hidden, hidden)

    def forward(self, x, key_mask):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, -1).unbind(2)
        return self.proj(_attention(q, k, v, key_mask))


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.self_attn = RefinerAttention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.mlp = nn.Sequential(BiasLinear(hidden, inner), nn.SiLU(), BiasLinear(inner, hidden))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), BiasLinear(hidden, 2 * hidden))

    def forward(self, x, c, key_mask):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _gated_add(x, self.self_attn(self.norm1(x), key_mask), gate_msa)
        return _gated_add(x, self.mlp(self.norm2(x)), gate_mlp)


class IndividualRefiner(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, mlp_ratio) for _ in range(depth)])


class TextRefiner(nn.Module):
    """txt_in: Qwen states -> hidden, refined by 2 self-attention blocks modulated by the timestep and the
    prompt's mean state."""

    def __init__(self, din, hidden, heads, mlp_ratio):
        super().__init__()
        self.input_embedder = BiasLinear(din, hidden)
        self.t_embedder = TimeMLP(hidden)
        self.c_embedder = TextProjection(din, hidden)
        self.individual_token_refiner = IndividualRefiner(hidden, heads, mlp_ratio)

    def forward(self, x, t, mask):
        t_rep = self.t_embedder(t)
        # the prompt's mean state, summed and divided in the stream's dtype as the reference does (always through its
        # mask) - x.mean() accumulates in higher precision and rounds differently, which every refiner gate then carries
        mf = (mask if mask is not None else torch.ones(x.shape[:2], device=x.device)).to(x.dtype).unsqueeze(-1)
        ctx = (x * mf).sum(dim=1) / mf.sum(dim=1)
        c = t_rep + self.c_embedder(ctx.to(x.dtype))
        x = self.input_embedder(x)
        key_mask = None
        if mask is not None and not bool(mask.all()):
            key_mask = mask.clone().bool()
            key_mask[:, 0] = True
        for block in self.individual_token_refiner.blocks:
            x = block(x, c, key_mask)
        return x


class ByT5Mapper(nn.Module):
    """byt5_in: LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear."""

    def __init__(self, din, hidden):
        super().__init__()
        self.layernorm = nn.LayerNorm(din, eps=1e-6)
        self.fc1 = BiasLinear(din, hidden)
        self.fc2 = BiasLinear(hidden, hidden)
        self.fc3 = BiasLinear(hidden, hidden)

    def forward(self, x, dtype):
        x = self.layernorm(x.to(dtype))
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


class VisionProjection(nn.Module):
    """vision_in.proj: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm (indexes 0, 1, 3, 4 as in the file)."""

    def __init__(self, din, hidden):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(din, eps=1e-6), BiasLinear(din, din), nn.GELU(),
                                  BiasLinear(din, hidden), nn.LayerNorm(hidden, eps=1e-6))

    def forward(self, x):
        return self.proj(x)


def _valid_first(a, b, ma, mb, zero=False):
    """The reference's reorder: [a, b] with each row's valid tokens first (stable), the rest after."""
    x = torch.cat([a, b], dim=1)
    m = torch.cat([ma, mb], dim=1).bool()
    order = torch.argsort(m.to(torch.int32), dim=1, descending=True, stable=True)
    m = torch.gather(m, 1, order)
    x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if zero:
        x = x * m.unsqueeze(-1).to(x.dtype)
    return x, m


# ---- the patch merge and the final split ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """img_in: a 1 x 1 x 1 projection of the 97 channels per latent cell, then each 2 x 2 window merged (ordered row
    offset, column offset, channel), both in FP32."""

    def __init__(self, cin, hidden, patch):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(cin, hidden, kernel_size=(1, 1, 1))
        self.patch_merge = FP32Linear(patch[1] * patch[2] * hidden, hidden)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            e = F.conv3d(x.float(), self.proj.weight.float(), self.proj.bias.float())
            ph, pw = self.patch[1], self.patch[2]
            w = e.unfold(3, ph, ph).unfold(4, pw, pw)            # B, C, T, H/2, W/2, 2, 2
            tokens = w.movedim(1, -1).flatten(4).flatten(1, 3)    # B, N, 4C
            return self.patch_merge(tokens).to(x.dtype)


class FinalLayer(nn.Module):
    """final_layer: each token split back into its 4 cells (patch_split, a residual per offset), adaptive LayerNorm,
    then the 48-channel projection; FP32."""

    def __init__(self, hidden, cout, patch):
        super().__init__()
        self.vol = patch[0] * patch[1] * patch[2]
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.patch_split = FP32Linear(hidden, self.vol * hidden)
        self.linear = FP32Linear(hidden, cout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), BiasLinear(hidden, 2 * hidden))

    def forward(self, x, c):
        mod = self.adaLN_modulation(c.float())
        shift, scale = mod.chunk(2, dim=1)
        _, n, d = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            offsets = self.patch_split(x.float()).unflatten(-1, (self.vol, d))
        h = self.norm_final((x[:, :, None] + offsets).flatten(1, 2))
        h = (h.float() * (1.0 + scale.float()[:, None]) + shift.float()[:, None]).to(h.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            vals = self.linear(h.float())
        return vals.unflatten(1, (n, self.vol)).transpose(2, 3).flatten(2)     # B, N, C * vol (channel-major)


# ---- the dual-stream block ----------------------------------------------------------------------------------------
class StreamAttention(nn.Module):
    """One stream's projections in a joint attention (file names: qkv, norm.query_norm / key_norm, proj)."""

    def __init__(self, hidden, heads, out_cls):
        super().__init__()
        self.qkv = BiasLinear(hidden, hidden * 3)
        self.norm = QKNorm(hidden // heads)
        self.proj = out_cls(hidden, hidden)


class LowRankModulation(nn.Module):
    """img_mod / txt_mod: the shared 16-wide timestep projection expanded to 6 x hidden, FP32."""

    def __init__(self, rank, hidden):
        super().__init__()
        self.lin = FP32Linear(rank, 6 * hidden)

    def forward(self, v, dtype):
        return self.lin(v).to(dtype)


class DoubleBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio, rank):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.heads = heads
        self.img_mod = LowRankModulation(rank, hidden)
        self.img_norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.img_attn = StreamAttention(hidden, heads, BiasUpLinear)
        self.img_norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(BiasLinear(hidden, inner), nn.GELU(), BiasUpLinear(inner, hidden))
        self.txt_mod = LowRankModulation(rank, hidden)
        self.txt_norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.txt_attn = StreamAttention(hidden, heads, BiasLinear)
        self.txt_norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(BiasLinear(hidden, inner), nn.GELU(), BiasUpLinear(inner, hidden))

    def forward(self, img, txt, vec, cos, sin, key_mask, txt_mask, compute_dtype, observed=None):
        i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = self.img_mod(vec, compute_dtype).chunk(6, dim=-1)
        t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = self.txt_mod(vec if observed is None else vec[0],
                                                              compute_dtype).chunk(6, dim=-1)
        img_m = _modulate(self.img_norm1(img), i_sh1, i_sc1, observed).to(compute_dtype)
        txt_m = _modulate(self.txt_norm1(txt), t_sh1, t_sc1).to(compute_dtype)
        b, n_img, _ = img_m.shape
        iq, ik, iv = self.img_attn.qkv(img_m).unflatten(-1, (3, self.heads, -1)).unbind(2)
        tq, tk, tv = self.txt_attn.qkv(txt_m).unflatten(-1, (3, self.heads, -1)).unbind(2)
        iq, ik = self.img_attn.norm(iq, ik)
        iq, ik = apply_rope(iq.to(iv.dtype), cos, sin), apply_rope(ik.to(iv.dtype), cos, sin)
        tq, tk = self.txt_attn.norm(tq, tk)
        tq, tk = tq.to(tv.dtype), tk.to(tv.dtype)
        out = _attention(torch.cat((iq, tq), 1), torch.cat((ik, tk), 1), torch.cat((iv, tv), 1), key_mask)
        img_a = self.img_attn.proj(out[:, :n_img].contiguous())
        txt_a = self.txt_attn.proj(out[:, n_img:].contiguous())
        if txt_mask is not None:
            txt_a = txt_a * txt_mask.to(txt_a.dtype).unsqueeze(-1)
        img = _gated_add(img, img_a, i_g1, observed)
        txt = _gated_add(txt, txt_a, t_g1)
        img = _gated_add(img, self.img_mlp(_modulate(self.img_norm2(img), i_sh2, i_sc2, observed).to(compute_dtype)),
                         i_g2, observed)
        txt = _gated_add(txt, self.txt_mlp(_modulate(self.txt_norm2(txt), t_sh2, t_sc2).to(compute_dtype)), t_g2)
        return img, txt


# ---- the model ----------------------------------------------------------------------------------------------------
class VimooLite(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = self.cfg = {**LITE_CONFIG, **cfg}
        h = c["hidden_size"]
        self.img_in = PatchEmbed(c["in_channels"] * 2 + 1, h, c["patch"])
        self.time_in = TimeMLP(h)
        self.mod_projection = FP32Linear(h, c["mod_rank"])
        self.txt_in = TextRefiner(c["text_dim"], h, c["heads"], c["mlp_ratio"])
        self.byt5_in = ByT5Mapper(c["byt5_dim"], h)
        self.vision_in = VisionProjection(c["vision_dim"], h)
        self.cond_type_embedding = nn.Embedding(3, h)
        self.double_blocks = nn.ModuleList([DoubleBlock(h, c["heads"], c["mlp_ratio"], c["mod_rank"])
                                            for _ in range(c["depth"])])
        self.final_layer = FinalLayer(h, c["in_channels"], c["patch"])
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    # ---- block swap (Fizgig's shared offloader) -------------------------------------------------------------------
    def enable_block_swap(self, num_blocks, device, supports_backward=True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()
        n = len(self.double_blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"Vimoo block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("vimoo", list(self.double_blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        if self.blocks_to_swap:
            blocks = self.double_blocks
            self.double_blocks = nn.ModuleList()
            try:
                self.to(device)
            finally:
                self.double_blocks = blocks
        else:
            self.to(device)
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.double_blocks))

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def _run(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # ---- conditioning ------------------------------------------------------------------------------------------------
    def condition(self, t, text, text_mask, glyph, glyph_mask, vision, vision_mask, dtype):
        """(vec, text stream, its mask): the text stream is [vision, glyph, qwen] with valid tokens first."""
        b = text.shape[0]
        dev = t.device
        vec = self.time_in(t)
        txt = self.txt_in(text.to(device=dev, dtype=dtype), t, text_mask)
        if text_mask is None:
            text_mask = torch.ones(b, txt.shape[1], dtype=torch.int64, device=dev)
        emb = self.cond_type_embedding.weight
        txt = txt + emb[0].to(txt.dtype)
        if glyph is None:
            glyph = torch.zeros(b, 1, self.cfg["byt5_dim"], device=dev, dtype=dtype)
            glyph_mask = torch.zeros(b, 1, dtype=torch.int64, device=dev)
        g = self.byt5_in(glyph.to(dev), dtype) + emb[1].to(dtype)
        txt, m = _valid_first(g, txt, glyph_mask.to(dev), text_mask.to(dev), zero=True)
        if vision is None:
            vis = self.vision_in(torch.zeros(b, 1, self.cfg["vision_dim"], device=dev, dtype=dtype)) * 0.0
            vmask = torch.zeros(b, 1, dtype=torch.int64, device=dev)
        else:
            vis = self.vision_in(vision.to(device=dev, dtype=dtype))
            vmask = (vision_mask.to(device=dev, dtype=torch.int64) if vision_mask is not None else
                     torch.ones(b, vis.shape[1], dtype=torch.int64, device=dev))
            if vision_mask is not None:
                vis = vis * vmask.unsqueeze(-1).to(vis.dtype)
        vis = vis + emb[2].to(vis.dtype)
        txt, m = _valid_first(vis, txt, vmask, m.to(torch.int64))
        keep = int(m.sum(dim=1).max())                   # padding sits at the end: drop what no row uses
        return vec, txt[:, :keep], m[:, :keep]

    def forward(self, x, t, text, text_mask=None, glyph=None, glyph_mask=None, vision=None, vision_mask=None,
                fps=24.0, observed_t=None):
        """x [B, 97, T, H, W] (noisy latent, reference latent, mask); t [B] in 0..1000; text [B, L, 3584] Qwen
        states (+ mask); glyph [B, Lg, 1472] ByT5 states (+ mask) or None; vision [B, 729, 1152] SigLIP states or
        None (text-to-video); observed_t: the first latent frame's own timestep (image-to-video) or None.
        Returns the predicted velocity (noise - clean) [B, 48, T, H, W], FP32."""
        b, _, nt, nh, nw = x.shape
        compute_dtype = x.dtype
        ph, pw = self.cfg["patch"][1], self.cfg["patch"][2]
        x = F.pad(x, (0, -nw % pw, 0, -nh % ph))        # a 45-row latent (720p) pads to the patch lattice
        gt, gh, gw = nt, x.shape[3] // ph, x.shape[4] // pw
        cos, sin = rope_table((gt, gh, gw), (nh, nw), fps, self.cfg, x.device)
        img = self.img_in(x)
        t = t.to(x.device).flatten().float()
        if t.numel() == 1 and b > 1:
            t = t.expand(b)
        vec, txt, tmask = self.condition(t, text, text_mask, glyph, glyph_mask, vision, vision_mask, img.dtype)
        observed = None
        if observed_t is not None:
            ot = observed_t.to(x.device).flatten().float()
            if ot.numel() == 1 and b > 1:
                ot = ot.expand(b)
            block_vec = self.mod_projection(F.silu(torch.stack((vec, self.time_in(ot)))))
            observed = (torch.arange(img.shape[1], device=img.device) < gh * gw)[None, :, None]
        else:
            block_vec = self.mod_projection(F.silu(vec))
        rdt = torch.promote_types(img.dtype, txt.dtype)
        img, txt = img.to(rdt), txt.to(rdt)
        key_mask = None
        if not bool(tmask.all()):
            key_mask = F.pad(tmask, (img.shape[1], 0), value=True)
        txt_mask = None if key_mask is None else tmask
        for i, block in enumerate(self.double_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            img, txt = self._run(block, img, txt, block_vec, cos, sin, key_mask, txt_mask, compute_dtype, observed)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.double_blocks, i)
        out = self.final_layer(img, vec)
        c = self.cfg["in_channels"]
        out = out.reshape(b, gt, gh, gw, c, 1, ph, pw)
        out = torch.einsum("nthwcopq->nctohpwq", out).reshape(b, c, gt, gh * ph, gw * pw)
        return out[:, :, :nt, :nh, :nw].float()
