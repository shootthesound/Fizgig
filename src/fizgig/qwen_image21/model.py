# Ported from diffusers `models/transformers/transformer_qwenimage21.py` (main, commit 6256aa7666, 2026-09-18).
# Copyright 2026 Qwen-Image Team, The HuggingFace Team. Licensed under the Apache License, Version 2.0.
# Changes for Fizgig: plain torch modules (no diffusers mixins), attention through torch SDPA using the exact
# per-segment block-causal decomposition (the reference's QwenImage21AttnProcessor; the flex path is not ported),
# non-reentrant gradient checkpointing, a loader for the ComfyUI single file (fused MLP) and the diffusers shards.
"""Qwen Image 2.1 single-stream DiT (7.1B, 32 identical blocks, patch size 1, block-causal attention)."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IMG_TOKENS_PER_SLOT = 4          # each vision-language image slot stands for a 2x2 group of latent tokens

QWEN21_CONFIG = dict(patch_size=1, in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128,
                     num_attention_heads=32, context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56),
                     eps=1e-6, causal_condition=True)


def apply_rotary_emb_complex(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x [B, S, H, D]; freqs_cis complex [S, D/2]."""
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_c * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)


class RMSNorm(nn.Module):
    """diffusers.models.normalization.RMSNorm (elementwise weight, fp32 variance)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        x = x * self.weight
        return x.to(input_dtype)


class TemporalTimesteps(nn.Module):
    """Sinusoidal embedding, cos first then sin, t scaled by 1000."""

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_factor * timestep.float()
        args = t[:, None] * self.freqs[None].to(t.device)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb.to(timestep.dtype)


class TimestepEmbedding(nn.Module):
    """diffusers TimestepEmbedding(256, dim, sample_proj_bias=False): linear_1 -> SiLU -> linear_2, no biases."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = TemporalTimesteps(256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timestep).to(dtype=hidden_states.dtype))


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is `scale - 1` (effective scale weight + 1), computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms * (self.weight.float() + 1)).to(dt)


class TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(self.act(self.in_layer(self.text_norm(x))))


class SwiGLUFeedForward(nn.Module):
    """out(silu(gate_layer(x)) * proj(x)). Kept unfused so LoRA targets gate_layer / proj separately."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        return self.out(self.activation_fn(self.gate_layer(x)) * self.proj(x))


def select_modulation_rows(params: torch.Tensor, target_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """With causal_condition, params has batch+1 rows: real-t rows then the trailing t=0 row. Text and condition-image
    tokens take the t=0 row, target-image tokens their own sample's row."""
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class AdaLayerNormContinuous(nn.Module):
    """Final adaptive norm, scale only (no shift)."""

    def __init__(self, embedding_dim: int, conditioning_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(self, x, cond, target_token_mask=None):
        scale = self.linear(self.silu(cond).to(x.dtype))
        scale = select_modulation_rows(scale, target_token_mask)
        return self.norm(x) * (1 + scale)


def prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list:
    """(start, end, is_text) runs of equal image_ids over the prefix (text + condition images)."""
    ids = image_ids[:prefix_len].tolist()
    segs, start = [], 0
    for i in range(1, prefix_len + 1):
        if i == prefix_len or ids[i] != ids[start]:
            segs.append((start, i, ids[start] < 0))
            start = i
    return segs


def _sdpa(q, k, v, mask=None):
    """q/k/v in [B, S, H, D] (diffusers layout); SDPA wants [B, H, S, D]; bool mask True = attend. An opted-in
    workbench render takes comfy-kitchen's INT8 kernel where no mask excludes a token (fizgig/modules/int8_attention)."""
    from fizgig.modules import int8_attention as _i8a
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = _i8a.attend(qt, kt, vt, mask)
    if out is None:
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, x, rotary_emb, segments, key_valid):
        q = self.to_q(x).unflatten(-1, (self.heads, -1))
        k = self.to_k(x).unflatten(-1, (self.heads, -1))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        q = self.norm_q(q).to(v.dtype)
        k = self.norm_k(k).to(v.dtype)
        if rotary_emb is not None:
            q = apply_rotary_emb_complex(q, rotary_emb)
            k = apply_rotary_emb_complex(k, rotary_emb)
        # Block-causal prefill (QwenImage21AttnProcessor): each prefix segment attends to keys [0, end); text segments
        # add a causal triangle over their own keys; the target image attends to everything. Padded keys dropped.
        prefix_len = segments[-1][1] if segments else 0
        outs = []
        for start, end, is_text in segments:
            m = None
            if is_text:
                n = end - start
                m = torch.cat([torch.ones(n, start, dtype=torch.bool, device=q.device),
                               torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))], dim=1)[None, None]
            if key_valid is not None:
                kv = key_valid[:, None, None, :end]
                if kv.dtype == torch.bool:
                    m = kv if m is None else (m & kv)
                else:                                   # float bias (soft prompt-travel weights)
                    m = kv if m is None else kv.masked_fill(~m, float("-inf"))
            outs.append(_sdpa(q[:, start:end], k[:, :end], v[:, :end], m))
        outs.append(_sdpa(q[:, prefix_len:], k, v, None if key_valid is None else key_valid[:, None, None, :]))
        h = torch.cat(outs, dim=1).flatten(2, 3).type_as(q)
        return self.to_out[1](self.to_out[0](h))


class TransformerBlock(nn.Module):
    """Single-stream block; modulation is shared model-wide and passed in."""

    def __init__(self, dim, heads, head_dim, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    @staticmethod
    def _modulate(x, mod, target_token_mask):
        scale, gate = mod.chunk(2, dim=-1)
        return x * (1 + select_modulation_rows(scale, target_token_mask)), select_modulation_rows(gate, target_token_mask)

    def forward(self, x, modulation, rotary_emb, target_token_mask, segments, key_valid):
        mod1, mod2 = modulation.chunk(2, dim=-1)
        h, g1 = self._modulate(self.img_norm1(x), mod1, target_token_mask)
        x = x + g1.tanh() * self.attn(h, rotary_emb, segments, key_valid)
        h, g2 = self._modulate(self.img_norm2(x), mod2, target_token_mask)
        x = x + g2.tanh() * self.img_mlp(h)
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class Rope3D(nn.Module):
    """3-axis (frame, height, width) RoPE. Text advances a shared position on all axes; each image block freezes the
    frame axis and lays tokens on an h/w grid centred on zero."""

    def __init__(self, theta: int, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        pos = torch.arange(8192)
        neg = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [torch.cat([self._params(pos, d, theta), self._params(neg, d, theta)], 0) for d in self.axes_dim]

    @staticmethod
    def _params(index, dim, theta):
        f = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(f), f)

    def forward(self, img_shapes, image_pad_mask, device, text_valid=None):
        """text_valid: optional [joint_len] weights (1 real, 0 padded, in between while prompt travel fades a token
        in). The image's frame position then starts after the real text only - fractionally while tokens fade - so a
        padded prompt places the image exactly as the unpadded prompt does, and travel moves it smoothly."""
        self.freqs = [f.to(device) for f in self.freqs]
        frame, img_h, img_w = [], [], []
        cursor, position = 0, 0
        total = image_pad_mask.shape[-1]
        is_img = image_pad_mask.tolist()
        valid = text_valid.tolist() if text_valid is not None else None
        for _, h, w in img_shapes:
            block_start = is_img.index(True, cursor)
            text_len = block_start - cursor
            frame.extend(range(position, position + text_len))
            position += text_len if valid is None else float(sum(valid[cursor:block_start]))
            cursor = block_start + h * w
            frame.extend([position] * (h * w))
            position += max(h, w)
            img_h.extend([y for y in range(-(h - h // 2), h // 2) for _ in range(w)])
            img_w.extend([x for _ in range(h) for x in range(-(w - w // 2), w // 2)])
        if cursor < total:
            frame.extend(range(position, position + total - cursor))
        if valid is not None and any(float(f) != int(f) for f in frame):
            frame_f = torch.tensor(frame, dtype=torch.float32)
            f0 = self._params(frame_f, self.axes_dim[0], self.theta).to(device)    # fractional start (travel)
            frame = frame_f.floor().long().to(device)                               # text slots are whole numbers
        else:
            frame = torch.tensor(frame, dtype=torch.long, device=device)
            f0 = self.freqs[0][frame]
        hh, ww = frame.clone(), frame.clone()
        hh[image_pad_mask] = torch.tensor(img_h, dtype=torch.long, device=device)
        ww[image_pad_mask] = torch.tensor(img_w, dtype=torch.long, device=device)
        return torch.cat([f0, self.freqs[1][hh], self.freqs[2][ww]], dim=-1)


class QwenImage21DiT(nn.Module):
    def __init__(self, patch_size=1, in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128,
                 num_attention_heads=32, context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56), eps=1e-6,
                 causal_condition=True):
        super().__init__()
        self.causal_condition = causal_condition
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Rope3D(10000, axes_dims_rope)
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)
        # one shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate]
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
            for _ in range(num_layers)])
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    # ---- block swap (Fizgig's shared offloader, as Klein and Krea 2 use it) --------------------------------------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.transformer_blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("qwen21", list(self.transformer_blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped blocks' weights (the model is assumed to be on CPU)."""
        blocks = self.transformer_blocks
        self.transformer_blocks = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.transformer_blocks = blocks
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.transformer_blocks))

    def disable_block_swap(self):
        """Tear block swap down (the offloader's backward hooks would otherwise fire on the next training backward):
        a run without swap gets its model back after the Turbo previews parked it."""
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

    @staticmethod
    def build_token_metadata(image_pad_mask, img_shapes):
        """image_ids (-1 at text, unique id per image block) and target_token_mask (the last image block)."""
        pos = image_pad_mask.nonzero(as_tuple=True)[0]
        lengths = [math.prod(s) for s in img_shapes]
        if sum(lengths) != pos.numel():
            raise ValueError(f"img_shapes accounts for {sum(lengths)} image tokens but the mask marks {pos.numel()}")
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        image_ids[pos] = torch.repeat_interleave(torch.arange(len(lengths), device=image_pad_mask.device),
                                                 torch.tensor(lengths, device=image_pad_mask.device))
        target = torch.zeros_like(image_pad_mask)
        target[pos[-lengths[-1]:]] = True
        return image_ids, target

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                encoder_hidden_states_mask=None):
        """hidden_states [B, N, 64] packed target latents (condition images first if any); encoder_hidden_states
        [B, L, 4096]; timestep [B] in [0, 1] (= sigma); img_shapes [[(1, h, w), ...]]; img_mask [B, L + target/4]
        True at image slots. Returns [B, joint_len, 64]; the caller keeps the last N rows."""
        batch = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        enc = self.txt_in(encoder_hidden_states)

        repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = math.prod(img_shapes[0][-1])
        joint = torch.cat([enc, enc.new_zeros(batch, target_tokens // 4, enc.shape[2])], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = hidden_states.to(joint.dtype)

        key_valid, text_w = None, None
        if encoder_hidden_states_mask is not None:
            text_pos = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            m = encoder_hidden_states_mask[:, vlm_text]
            if m.dtype == torch.bool:
                key_valid = torch.ones(batch, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device)
                key_valid[:, text_pos] = m
                text_w = key_valid[0].float()
            else:
                # soft key weights (prompt travel): attention bias log(w), -inf at 0 = masked, 0 at 1 = plain
                w = torch.ones(batch, image_pad_mask.shape[0], dtype=torch.float32, device=hidden_states.device)
                w[:, text_pos] = m.float()
                text_w = w[0]
                key_valid = torch.log(w).to(hidden_states.dtype)
        padded = text_w is not None and not bool((text_w == 1).all())
        rotary = self.pos_embed(img_shapes[0], image_pad_mask, hidden_states.device,
                                text_valid=text_w if padded else None)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            mod_mask = target_token_mask
        else:
            mod_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        prefix_len = int((~target_token_mask).sum())
        segments = prefix_segments(image_ids, prefix_len)

        blocks = list(self.transformer_blocks) if self.blocks_to_swap else None
        for index, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if getattr(block, "_handles_checkpointing", False):    # compiled: the wrapper checkpoints itself
                joint = block(joint, modulation, rotary, mod_mask, segments, key_valid)
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                joint = checkpoint(block, joint, modulation, rotary, mod_mask, segments, key_valid,
                                   use_reentrant=False)
            else:
                joint = block(joint, modulation, rotary, mod_mask, segments, key_valid)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)

        joint = self.norm_out(joint, temb, mod_mask)
        return self.proj_out(joint)


# ---- weights ---------------------------------------------------------------------------------------------------
def convert_comfy_state_dict(sd: dict) -> dict:
    """ComfyUI single file -> this module's keys. The Comfy repack fuses img_mlp.gate_layer + img_mlp.proj into
    img_mlp.gate_up = [gate_layer; proj] (first half gate, second half proj - comfy/lora.py maps the halves so)."""
    out = {}
    for k, v in sd.items():
        k2 = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
        if k2.endswith("img_mlp.gate_up.weight"):
            half = v.shape[0] // 2
            base = k2[: -len("gate_up.weight")]
            out[base + "gate_layer.weight"] = v[:half]
            out[base + "proj.weight"] = v[half:]
        else:
            out[k2] = v
    return out


def load_qwen21_dit(path, device="cuda", dtype=torch.bfloat16, config=None) -> QwenImage21DiT:
    """Load from the ComfyUI single file, a diffusers shard directory, or its index.json. Strict key match."""
    import glob
    import os

    from safetensors.torch import load_file
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    elif path.endswith(".json"):
        files = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.safetensors")))
    else:
        files = [path]
    with torch.device("meta"):
        model = QwenImage21DiT(**(config or QWEN21_CONFIG))
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    sd = convert_comfy_state_dict(sd)
    sd = {k: v.to(dtype) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Qwen Image 2.1 DiT keys mismatch: missing {missing[:8]} ({len(missing)}), "
                         f"unexpected {unexpected[:8]} ({len(unexpected)})")
    model.pos_embed = Rope3D(10000, (config or QWEN21_CONFIG)["axes_dims_rope"])       # rebuilt off meta
    model.time_text_embed.time_proj = TemporalTimesteps(256)
    return model.to(device)
