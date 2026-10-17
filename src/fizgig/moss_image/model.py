"""MOSS-Image DiT (OpenMOSS, MOSS-Image-Base / -SFT) for Fizgig.

Ported from the reference `pipeline_moss_image.py` (Apache-2.0, OpenMOSS-Team/MOSS-Image-Base on Hugging Face, the
`MOSSImageTransformer` class; https://github.com/OpenMOSS/MOSS-Image). Module names and maths are the reference's, so
Fizgig's module names are the ones ComfyUI's LoRA loader addresses (`diffusion_model.layers.N.proj.q`). Fizgig adds:

* a dtype rule that survives INT8 / NF4 bases (the reference reads the compute dtype off proj.q's weight, which a
  quantised base turns into int8 - here it is the block norms' dtype, which stay bf16)
* a masked-batch path: prompts of different lengths in one batch, the padded text removed by an attention key mask and
  every sample's image RoPE placed after its OWN valid text length (the reference loops per sample for this; the loop
  stays available as per_sample=True and the tests hold the two equal)
* gradient checkpointing per block, Fizgig's block swap (modules/offloading.ModelOffloader) on `layers`, attention
  through fizgig.modules.int8_attention.attend first (workbench renders that opt in)
* a loader for ComfyUI's single file (q/k/v fused into proj.qkv, split back here) and the official diffusers shards

Shapes: 42 blocks of width 3072 (24 heads x 128), SiTU-GLU FFN 8192, one modulation shared by every block, a 2-block
text refiner (4096 -> 3072) in front, 4-axis RoPE (r, f, h, w) with the image FIRST in the sequence and its f
coordinate equal to the prompt's unpadded length. Latents are FLUX.2 VAE latents 2x2-patchified: 128 channels at 16x.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# transformer/config.json of OpenMOSS-Team/MOSS-Image-Base (rev 1ceede80), spelled out
CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128,
              rope_base=10000.0)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    """SiTU-GLU: soft-capped gate (beta 4) times soft-capped value (beta 25) - not SwiGLU."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    """x [B, S, H, D]; rope [1 | B, S, D] = [sin(D/2) ; cos(D/2)], rotate-half pairing (i, i + D/2)."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _sdpa(q, k, v, mask=None):
    """q/k/v [B, H, S, D]; mask None or bool [B, 1, 1, S] (True = attend). An opted-in workbench render takes
    comfy-kitchen's INT8 kernel when nothing is masked (fizgig/modules/int8_attention)."""
    if mask is None:
        try:
            from fizgig.modules import int8_attention as _i8a
            out = _i8a.attend(q, k, v, None)
        except Exception:           # the module is optional on the box; attend() itself never raises on CUDA
            out = None
        if out is not None:
            return out
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def attention(q, k, v, rope, heads, mask=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return _sdpa(q.to(v.dtype), k.to(v.dtype), v, mask).transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    """The text refiner's block: no modulation, biased Linears, text-only attention. Never trained in Fizgig."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads, mask))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, mask=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, mask)
        return x


class Block(nn.Module):
    """One of the 42 joint blocks. LoRA targets: proj.q / k / v / o / ffn_up_gate / ffn_down."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)     # [hidden ; gate]
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.norm1.weight.dtype          # the reference's proj['q'].weight.dtype, quantisation-proof
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads, mask)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


def rope_bands(head_dim: int, base: float) -> torch.Tensor:
    """The reference's fp32 CPU construction (num_bands = head_dim / 8 per axis)."""
    n = head_dim // 8
    return 1 / (base ** (torch.arange(n, device="cpu", dtype=torch.float32) / n))


def rope_table(bands: torch.Tensor, height: int, width: int, text_length: int, device) -> torch.Tensor:
    """[1, h*w + L, head_dim] in fp32: image tokens (0, L, row - (h-1)/2, col - (w-1)/2), then text (0, t, 0, 0)."""
    bands = bands.to(device)
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def time_embedding_input(times: torch.Tensor, freq_dim: int) -> torch.Tensor:
    """The reference's sinusoid: float64 phases, [cos ; sin], cast to float32. times on the 0-1000 scale (1000 sigma)."""
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


class MossImageDiT(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                 latent_channels=128, rope_base=10000.0):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.config = dict(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers, text_dim=text_dim,
                           freq_dim=freq_dim, latent_channels=latent_channels, rope_base=rope_base)
        self.heads = num_heads
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        # derived constants, never a parameter: recomputed in fp32 (the reference rebuilds them on every _apply)
        self._bands = rope_bands(dim // num_heads, rope_base)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    # ---- training plumbing ----------------------------------------------------------------------------------------
    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()
        n = len(self.layers)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("mossimage", list(self.layers), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        layers = self.layers
        self.layers = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.layers = layers
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.layers))

    def disable_block_swap(self):
        if self.offloader is not None:
            self.offloader.remove_hooks()
        self.offloader = None
        self.blocks_to_swap = 0

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    # ---- forward ----------------------------------------------------------------------------------------------------
    def _conditioning(self, times, batch):
        emb = time_embedding_input(times, self.freq_dim)
        time = self.time_embedding(emb.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    def _run_blocks(self, x, rope, mod, mask, skip):
        blocks = list(self.layers) if self.blocks_to_swap else None
        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if index not in skip:       # a skipped block (SLG) still keeps the swap's schedule
                if getattr(layer, "_handles_checkpointing", False):
                    x = layer(x, rope, mod, mask)
                elif torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = checkpoint(layer, x, rope, mod, mask, use_reentrant=False)
                else:
                    x = layer(x, rope, mod, mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)
        return x

    def _head(self, x, head, n):
        x = self.final_norm_video(x[:, :n].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        return self.final_linear_video(x.to(self.final_linear_video.weight.dtype))

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, skip_layers=(),
                per_sample: bool = False):
        """hidden_states [B, 128, H, W] (any float dtype); timestep [B] or scalar on the 0-1000 scale (1000 sigma);
        encoder_hidden_states [B, L, 4096]; encoder_attention_mask [B, L] bool (None = every token valid);
        skip_layers: block indices skipped (skip-layer guidance). Returns [B, 128, H, W] in the weights' dtype."""
        skip = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H, W]")
        batch, _, height, width = hidden_states.shape
        n = height * width
        device = hidden_states.device
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim]")
        mask = encoder_attention_mask
        if mask is not None:
            mask = mask.to(device).bool()
            if mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError("encoder_attention_mask must be a [B, S] mask")
            if not bool(mask.any(dim=1).all()):
                raise ValueError("Each sample needs at least one valid text token.")
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        mod, head = self._conditioning(times, batch)

        padded = mask is not None and not bool(mask.all())
        if per_sample or (padded and batch == 1):
            # the reference's loop: each sample at its true text length (also the cheap path for one padded prompt)
            out = []
            for i in range(batch):
                ctx = encoder_hidden_states[i:i + 1]
                if mask is not None:
                    ctx = ctx[:, mask[i]]
                out.append(self._forward_same_length(hidden_states[i:i + 1], ctx, mod[i:i + 1], head[i:i + 1], skip))
            return torch.cat(out)
        if not padded:
            return self._forward_same_length(hidden_states, encoder_hidden_states, mod, head, skip)
        return self._forward_masked(hidden_states, encoder_hidden_states, mask, mod, head, skip)

    def _forward_same_length(self, hidden_states, context, mod, head, skip):
        batch, _, height, width = hidden_states.shape
        n = height * width
        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        rope = rope_table(self._bands, height, width, context.shape[1], hidden_states.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, n:])
        x = torch.cat((video, text), 1)
        x = self._run_blocks(x, rope, mod, None, skip)
        x = self._head(x, head, n)
        return x.transpose(1, 2).reshape(batch, -1, height, width)

    def _forward_masked(self, hidden_states, context, mask, mod, head, skip):
        """Prompts of different lengths in one batch. Valid tokens must lead each row (right padding, as the cache and
        pad_conditioning write them). Padded text never acts as a key, text RoPE runs 0..l-1 per row and the image's
        f coordinate is each row's own l - so every row equals its unpadded single forward."""
        batch, _, height, width = hidden_states.shape
        n = height * width
        device = hidden_states.device
        lengths = mask.sum(1)
        expect = torch.arange(mask.shape[1], device=device)[None] < lengths[:, None]
        if not torch.equal(expect, mask):
            raise ValueError("encoder_attention_mask must mark a prefix of each row (right padding)")
        L = mask.shape[1]
        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        # per-row rope: image f = that row's length; text positions 0..L-1 (rows past l are padding, never keys)
        rope = torch.cat([rope_table(self._bands, height, width, int(l), device)[:, :n] for l in lengths.tolist()])
        text_rope = rope_table(self._bands, 1, 1, L, device)[:, 1:]          # text part only, positions 0..L-1
        rope = torch.cat((rope, text_rope.expand(batch, -1, -1)), 1).to(video.dtype)
        text_keys = mask[:, None, None, :]
        text = self.text_embedder(context, rope[:, n:], text_keys)
        x = torch.cat((video, text), 1)
        keys = torch.cat((torch.ones(batch, n, dtype=torch.bool, device=device), mask), 1)[:, None, None, :]
        x = self._run_blocks(x, rope, mod, keys, skip)
        x = self._head(x, head, n)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


# ---- weights ----------------------------------------------------------------------------------------------------
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "net.", "model.")


def strip_prefix(sd: dict) -> dict:
    for p in _PREFIXES:
        if sd and all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd


def split_fused_qkv(sd: dict) -> dict:
    """ComfyUI's single file fuses each block's q/k/v into layers.N.proj.qkv.weight [3 dim, dim] (rows q, k, v);
    Fizgig keeps them apart so a LoRA addresses each (ComfyUI maps lora names proj.q/k/v onto the fused rows)."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".proj.qkv.weight"):
            stem = k[: -len("qkv.weight")]
            q, kk, vv = v.chunk(3, 0)
            out[stem + "q.weight"], out[stem + "k.weight"], out[stem + "v.weight"] = q, kk, vv
        else:
            out[k] = v
    return out


def config_from_state_dict(sd: dict) -> dict:
    dim = sd["video_embedder.weight"].shape[0]
    head_dim = sd["layers.0.q_norm.weight"].shape[0]
    return dict(dim=dim, ffn_dim=sd["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2, num_heads=dim // head_dim,
                num_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers.")),
                text_dim=sd["text_embedder.proj.weight"].shape[1], freq_dim=sd["time_embedding.0.weight"].shape[1],
                latent_channels=sd["video_embedder.weight"].shape[1], rope_base=CONFIG["rope_base"])


def is_convrot(path) -> bool:
    """A ComfyUI int8 ConvRot file (quantised Linears carry `<module>.comfy_quant`)."""
    if not (isinstance(path, str) and os.path.isfile(path) and path.endswith(".safetensors")):
        return False
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return any(k.endswith(".comfy_quant") or k.endswith(".weight_scale") for k in f.keys())


def weight_files(path: str) -> list:
    """The safetensors files behind `path`: a single file, a diffusers shard index (*.index.json), or a folder
    holding either (MOSS-Image-Base/ or its transformer/ subfolder)."""
    if os.path.isdir(path):
        for sub in ("", "transformer"):
            d = os.path.join(path, sub)
            idx = [f for f in os.listdir(d) if f.endswith(".safetensors.index.json")] if os.path.isdir(d) else []
            if idx:
                return weight_files(os.path.join(d, idx[0]))
            sts = sorted(f for f in os.listdir(d) if f.endswith(".safetensors")) if os.path.isdir(d) else []
            if sts:
                return [os.path.join(d, f) for f in sts]
        raise FileNotFoundError(f"{path}: no MOSS-Image DiT weights found")
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(os.path.dirname(path), s) for s in shards]
    return [path]


def load_state_dict(path: str) -> dict:
    from safetensors.torch import load_file
    sd = {}
    for f in weight_files(path):
        sd.update(load_file(f, device="cpu"))
    return sd


def take_convrot(sd: dict) -> dict:
    """Pop the int8 ConvRot Linears out of a ComfyUI state dict (prefix already stripped) -> {module name: (codes,
    per-row scale, rotation)}. The fused proj.qkv splits by rows into q / k / v - the per-row scales split with them,
    and the rotation runs along the input dim, so each third stays valid. ComfyUI's attention-kernel hint
    (`<block>.comfy_attention.config`) carries no weights and is dropped."""
    from fizgig.minimax.convrot import parse_comfy_quant
    for k in [k for k in sd if k.endswith(".comfy_attention.config")]:
        sd.pop(k)
    out = {}
    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        base = k[: -len(".comfy_quant")]
        conf = parse_comfy_quant(sd.pop(k))
        if conf.get("format") != "int8_tensorwise":
            raise ValueError(f"{base}: unsupported ComfyUI quant format {conf.get('format')!r} (int8_tensorwise "
                             "ConvRot files load)")
        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
        q, sc = sd.pop(base + ".weight"), sd.pop(base + ".weight_scale").float().reshape(-1, 1)
        if base.endswith(".proj.qkv"):
            stem = base[: -len("qkv")]
            for i, (qq, ss) in enumerate(zip(q.chunk(3, 0), sc.chunk(3, 0))):
                out[stem + "qkv"[i]] = (qq.contiguous(), ss.contiguous(), rot)
        else:
            out[base] = (q, sc, rot)
    return out


def prepare_state_dict(sd: dict, convrot: str = "refuse"):
    """Prefix stripped, q/k/v split, rope.bands dropped. A ComfyUI int8 ConvRot file (the SFT / Base int8 uploads):
    convrot="refuse" (the training base) raises, "decode" returns its weights decoded to dense, "keep" pops them out
    for ConvRotInt8Linear - returned as (sd, {module: (codes, scale, rotation)}) for "keep", else sd."""
    if any(k.endswith(".comfy_quant") for k in sd):
        if convrot == "refuse":
            raise ValueError("this is ComfyUI's int8 ConvRot MOSS-Image file - Fizgig trains from the bf16 file "
                             "(moss_image_base_bf16.safetensors) and applies its own INT8 / NF4")
        sd = strip_prefix(sd)
        rot = take_convrot(sd)
        if convrot == "decode":
            from fizgig.minimax.convrot import dequantize_int8_convrot
            for name, (q, sc, r) in rot.items():
                sd[name + ".weight"] = dequantize_int8_convrot(
                    q, sc, {"format": "int8_tensorwise", "convrot": r > 1, "convrot_groupsize": r},
                    out_dtype=torch.float32)
            rot = {}
    else:
        sd, rot = strip_prefix(sd), {}
    sd = split_fused_qkv(sd)
    sd.pop("rope.bands", None)
    if "shared_block_modulation.weight" not in sd or ("layers.0.proj.ffn_up_gate.weight" not in sd
                                                      and "layers.0.proj.ffn_up_gate" not in rot):
        raise KeyError("not a MOSS-Image DiT (no shared_block_modulation / layers.0.proj.ffn_up_gate)")
    return (sd, rot) if convrot == "keep" else sd


def load_moss_dit(path, device="cuda", dtype=torch.bfloat16, sd: Optional[dict] = None,
                  convrot: str = "refuse") -> MossImageDiT:
    """bf16 DiT from ComfyUI's single file (fused qkv), the official shards (index / folder) or a state dict.
    convrot: what a ComfyUI int8 ConvRot file does - "refuse" (the default: the training base is the bf16 file),
    "keep" (its int8 weights stay int8 as frozen ConvRotInt8Linear - a preview / workbench model, never trained) or
    "decode" (decoded to `dtype`)."""
    raw = load_state_dict(path) if sd is None else dict(sd)
    rot = {}
    if convrot == "keep":
        sd, rot = prepare_state_dict(raw, convrot="keep")
    else:
        sd = prepare_state_dict(raw, convrot=convrot)
    del raw
    shapes = dict(sd)
    for name, (q, _sc, _r) in rot.items():           # config_from_state_dict reads Linear shapes
        shapes[name + ".weight"] = q
    cfg = config_from_state_dict(shapes)
    if cfg["num_layers"] != CONFIG["num_layers"] or cfg["dim"] != CONFIG["dim"]:
        logger.warning(f"[moss_image] unusual DiT shape {cfg} (MOSS-Image-Base is {CONFIG}) - loading it anyway")
    with torch.device("meta"):
        model = MossImageDiT(**cfg)
    model._bands = rope_bands(cfg["dim"] // cfg["num_heads"], cfg["rope_base"])
    expected = set(model.state_dict())
    have = set(sd) | {n + ".weight" for n in rot}
    missing, unexpected = sorted(expected - have), sorted(have - expected)
    if missing or unexpected:
        raise KeyError(f"MOSS-Image DiT: missing {missing[:6]}, unexpected {unexpected[:6]}")
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False, assign=True)
    if rot:
        from fizgig.minimax.convrot import ConvRotInt8Linear
        for name, (q, sc, r) in rot.items():
            parent, attr = name.rsplit(".", 1)
            old = model.get_submodule(name)
            with torch.device("meta"):
                lin = ConvRotInt8Linear(old.in_features, old.out_features, bias=old.bias is not None, rot=r,
                                        compute_dtype=dtype)
            lin.qdata, lin.wscale = q.contiguous(), sc.contiguous()
            if old.bias is not None:
                lin.bias = torch.nn.Parameter(old.bias.detach(), requires_grad=False)
            container = model.get_submodule(parent)
            if isinstance(container, nn.ModuleDict):
                container[attr] = lin
            else:
                setattr(container, attr, lin)
        model._convrot = True
        logger.info(f"[moss_image] {len(rot)} int8 ConvRot Linears kept as stored")
    return model.to(device)
