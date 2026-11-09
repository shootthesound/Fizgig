# Ported from diffusers `models/transformers/transformer_bria_fibo2.py` (v0.42.1, 2026-10-27).
# Copyright 2026 Bria AI and The HuggingFace Team. Licensed under the Apache License, Version 2.0.
# Changes for Fizgig: plain torch modules (no diffusers mixins or config), attention through torch SDPA (diffusers'
# default "native" backend) with comfy-kitchen's INT8 kernel for long renders, the timestep embedding cast to the
# model's compute dtype (so an INT8 or NF4 base reads no weight dtype), blocks called positionally, non-reentrant
# gradient checkpointing, and block swap over the 44 main layers.
"""Fibo 2.0 DiT (8.2B: 44 main blocks, 5 of them reading the text, 2 noise refiners, a 2-layer Perceiver).

Block swap streams the main layers through Fizgig's shared offloader. The stream calls sit inside the block loop,
not in forward hooks: a hook would re-fire during a gradient-checkpointing recompute and park weights the backward
still needs."""
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from fizgig.modules import int8_attention as _i8a
from fizgig.training import attn_probe


if TYPE_CHECKING:
    from fizgig.modules.offloading import ModelOffloader

ADALN_EMBED_DIM = 256
INT8_MIN_TOKENS = 2048      # the INT8 kernel loses below ~1300 keys and wins clearly from here (5090, 26 Oct 2026)

FIBO2_CONFIG = dict(in_channels=32, patch_size=2, dim=3584, n_layers=44, n_refiner_layers=2, n_heads=28,
                    norm_eps=1e-5, cap_feat_dim=7680, injection_layer_ids=(6, 14, 22, 30, 38),
                    perceiver_num_layers=2, min_num_gist_tokens=256, max_num_gist_tokens=512, gist_step=64,
                    gist_min_text_len=256, gist_max_text_len=2048, rope_theta=256.0, t_scale=1000.0,
                    axes_dims=(32, 48, 48), axes_lens=(2048, 512, 512))

# the pipeline's text recipe (diffusers pipelines/bria_fibo2/pipeline_bria_fibo2.py, v0.42.1): the Qwen3-VL hidden
# states it reads, as six bundles of three layers - the Perceiver's first, then one per injection block
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def _parse_prompt(prompt):
    """A prompt's caption (as Fibo 2.0 was trained with it) and its edit instruction - diffusers' _parse_prompt: a
    prompt that isn't a JSON object is returned as it is with no instruction; empty values and the score keys go."""
    import json
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    def empty(v):
        return math.isnan(v) if isinstance(v, float) else v is None or (isinstance(v, (str, dict, list)) and not v)

    def clean(v):
        if isinstance(v, dict):
            v = {k: clean(x) for k, x in v.items()}
            return {k: x for k, x in v.items() if not empty(x)}
        if isinstance(v, list):
            return [x for x in map(clean, v) if not empty(x)]
        return v

    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    caption = {k: {a: b for a, b in v.items() if a not in scores} if isinstance(v, dict) else v
               for k, v in clean(caption).items() if k not in scores}
    caption = {k: v for k, v in caption.items() if v != {}}
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "",
                   "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def edit_prompt(prompt, num_images):
    """An edit prompt as the pipeline builds it (_get_edit_prompt, without text images): one JSON object - a vision
    marker per image, the result's caption when the prompt has one, then the instruction."""
    import json
    caption, instruction = _parse_prompt(prompt)
    if instruction is None:
        caption, instruction = None, prompt
    edit = {"image": VISION_MARKER} if num_images == 1 else \
        {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)


# A masked attention (a training batch's padded captions): PyTorch's default choice falls back to its memory-efficient
# kernel with a mask, and cuDNN takes one at almost no cost (#221, measured on MOSS-Image). A preference, falling back.
_MASKED_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def attention(query, key, value, attn_mask=None):
    """[B, S, H, D] in and out. A render past INT8_MIN_TOKENS keys takes the INT8 kernel when the workbench has it
    on (attend() refuses a live grad, a mask that drops tokens, or no kernel); everything else is torch SDPA."""
    q, k, v = (x.transpose(1, 2) for x in (query, key, value))
    if _i8a.wanted() and k.shape[2] >= INT8_MIN_TOKENS:
        out = _i8a.attend(q, k, v, attn_mask)
        if out is not None:
            return out.transpose(1, 2)
    if attn_mask is not None:
        with sdpa_kernel(_MASKED_BACKENDS, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)


class RMSNorm(nn.Module):
    """diffusers.models.normalization.RMSNorm (elementwise weight, fp32 variance)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size, bias=True), nn.SiLU(),
                                 nn.Linear(mid_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device)
                              / half)
            args = t[:, None].float() * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return emb

    def forward(self, t, dtype):
        # diffusers casts to mlp[0].weight.dtype; the compute dtype is the same on a bf16 base and still right on
        # an INT8 one, whose weight dtype would leave the embedding in fp32
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(dtype))


def select_per_token(value_noisy, value_clean, noise_mask, seq_len):
    return torch.where(noise_mask.unsqueeze(-1) == 1, value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                       value_clean.unsqueeze(1).expand(-1, seq_len, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size,
                                                                   bias=True))

    def forward(self, x, c):
        return self.linear(self.norm_final(x) * (1.0 + self.adaLN_modulation(c)).unsqueeze(1))


class RopeEmbedder:
    def __init__(self, theta, axes_dims, axes_lens):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dims, ends, theta):
        out = []
        for d, e in zip(dims, ends):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            freqs = torch.outer(torch.arange(e, dtype=torch.float64), freqs).float()
            out.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return out

    def __call__(self, ids):
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def compute_gist_count(text_lengths, min_gist, max_gist, gist_step, min_text_len, max_text_len):
    """Gist tokens per prompt: text lengths between min_text_len and max_text_len map linearly onto
    [min_gist, max_gist], rounded up to a multiple of gist_step."""
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        q = int(math.ceil((min_gist + ratio * (max_gist - min_gist)) / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, q)))
    return counts


def _rope(x, freqs_cis):
    """RoPE in float32, cast back to the input's dtype."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)


class Attention(nn.Module):
    """Self-attention (RoPE when freqs_cis is given) or cross-attention to encoder_hidden_states."""

    def __init__(self, query_dim, heads, dim_head, kv_dim=None, eps=1e-5):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(ctx).unflatten(-1, (self.heads, -1)))
        v = self.to_v(ctx).unflatten(-1, (self.heads, -1))
        if freqs_cis is not None:
            q, k = _rope(q, freqs_cis), _rope(k, freqs_cis)
        if attn_probe._ACTIVE is not None and encoder_hidden_states is not None:     # training maps: cross only
            attn_probe.see(q.transpose(1, 2), k.transpose(1, 2), attention_mask)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]     # [B, S] -> broadcast over heads and queries
        return self.to_out[0](attention(q, k, v, attention_mask).flatten(2, 3))


class TransformerBlock(nn.Module):
    """Z-Image's block. With cross_attention_dim it is an injection block that also reads the text encoder."""

    def __init__(self, dim, n_heads, norm_eps, cross_attention_dim=None):
        super().__init__()
        self.attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim, (dim * 8 // 3) // 64 * 64)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None, noise_mask=None, adaln_noisy=None,
                adaln_clean=None, encoder_hidden_states=None, encoder_attention_mask=None):
        seq_len = x.shape[1]
        if noise_mask is not None:            # per token: one modulation for the noisy tokens, one for the clean
            sn, gn, smn, gmn = self.adaLN_modulation(adaln_noisy).chunk(4, dim=1)
            sc, gc, smc, gmc = self.adaLN_modulation(adaln_clean).chunk(4, dim=1)
            scale_msa = select_per_token(1.0 + sn, 1.0 + sc, noise_mask, seq_len)
            scale_mlp = select_per_token(1.0 + smn, 1.0 + smc, noise_mask, seq_len)
            gate_msa = select_per_token(gn.tanh(), gc.tanh(), noise_mask, seq_len)
            gate_mlp = select_per_token(gmn.tanh(), gmc.tanh(), noise_mask, seq_len)
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        x = x + gate_msa * self.attention_norm2(
            self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis))
        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), encoder_hidden_states=encoder_hidden_states,
                                         attention_mask=encoder_attention_mask)
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(cross)
        return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))


class PerceiverLayer(nn.Module):
    """Gist tokens read the text encoder (cross-attention), then each other (self-attention), then an FFN."""

    def __init__(self, dim, n_heads, cross_attention_dim, norm_eps):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def forward(self, latents, encoder_hidden_states, encoder_attention_mask=None, attention_mask=None):
        latents = latents + self.cross_attention(self.cross_attn_norm(latents),
                                                 encoder_hidden_states=encoder_hidden_states,
                                                 attention_mask=encoder_attention_mask)
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        return latents + self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    """Squeezes the text encoder's hidden states into num_gist_tokens gist tokens."""

    def __init__(self, dim, n_heads, cross_attention_dim, num_queries, num_layers, norm_eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps)
                                     for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(self, encoder_hidden_states, num_gist_tokens, encoder_attention_mask=None, attention_mask=None):
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
        return self.norm_out(latents)


class Fibo2DiT(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = self.config = SimpleNamespace(**{**FIBO2_CONFIG, **cfg})
        dim = c.dim
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader: "ModelOffloader | None" = None
        self.x_embedder = nn.Linear(c.patch_size * c.patch_size * c.in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(c.rope_theta, c.axes_dims, c.axes_lens)
        self.noise_refiner = nn.ModuleList([TransformerBlock(dim, c.n_heads, c.norm_eps)
                                            for _ in range(c.n_refiner_layers)])
        self.perceiver = PerceiverResampler(dim, c.n_heads, c.cap_feat_dim, c.max_num_gist_tokens,
                                            c.perceiver_num_layers, c.norm_eps)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, c.n_heads, c.norm_eps,
                             cross_attention_dim=c.cap_feat_dim if i in c.injection_layer_ids else None)
            for i in range(c.n_layers)])
        self.final_layer = FinalLayer(dim, c.patch_size * c.patch_size * c.in_channels)

    # ---- block swap (Fizgig's shared offloader, as zimage and moss_image use it) ----------------------------------
    def enable_block_swap(self, num_blocks, device, supports_backward: bool = True):
        """Park the last num_blocks of the 44 main layers on CPU and stream them just in time. The 2 noise refiners
        and the 2 Perceiver layers stay resident."""
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks
        n = len(self.layers)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"Fibo 2.0 block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("fibo2", list(self.layers), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        if self.blocks_to_swap:
            layers = self.layers
            self.layers = nn.ModuleList()
            try:                                # an OOM in .to() must not leave the model with no blocks
                self.to(device)
            finally:
                self.layers = layers
        else:
            self.to(device)
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.layers))

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    # ---- forward ---------------------------------------------------------------------------------------------------
    @staticmethod
    def create_coordinate_grid(size, start, device):
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _patchify(self, latents):
        # 2x2 latent patches become tokens, features ordered (channel, dy, dx) as in training
        b, ch, h, w = latents.shape
        p = self.config.patch_size
        return latents.view(b, ch, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), -1)

    def _block(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None,
                context_latents=None):
        """hidden_states (B, 32, H, W) latents; timestep (B,) noise level in [0, 1]; encoder_hidden_states
        (B, 6, L, 7680) text bundles (the Perceiver's first); encoder_attention_mask (B, L) bool for padded prompts;
        context_latents a list of (B, 32, h, w) clean images to edit. Returns the velocity, shaped as hidden_states."""
        c = self.config
        b, ch, h, w = hidden_states.shape
        p = c.patch_size
        ht, wt = h // p, w // p
        n_img = ht * wt
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. timestep embeddings: image tokens get the real timestep, clean tokens (gist, context) t = 0
        adaln_noisy = self.t_embedder(timestep * c.t_scale, dtype)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep), dtype)

        # 2. patchify the image and the context images; the image sits on RoPE plane max_num_gist_tokens + 1, context
        # image k on the k-th plane after it
        plane = c.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, ht, wt), (plane, 0, 0), device)]
        for k, lat in enumerate(context_latents or []):
            tokens.append(self._patchify(lat))
            ids.append(self.create_coordinate_grid((1, lat.shape[2] // p, lat.shape[3] // p), (plane + k + 1, 0, 0),
                                                   device))
        x = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat([g.flatten(0, 2) for g in ids])).unsqueeze(0)

        # 3. noise refiner; with context images the noise mask marks the image tokens as the only noisy ones
        refiner_mask = None
        if context_latents:
            refiner_mask = torch.zeros(b, x.shape[1], dtype=torch.long, device=device)
            refiner_mask[:, :n_img] = 1
        for block in self.noise_refiner:
            x = self._block(block, x, None, image_freqs, adaln_noisy, refiner_mask, adaln_noisy, adaln_clean)

        # 4. gist count per prompt from its text length; the Perceiver turns the first bundle into gist tokens
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * b
        gist_counts = compute_gist_count(text_lengths, c.min_num_gist_tokens, c.max_num_gist_tokens, c.gist_step,
                                         c.gist_min_text_len, c.gist_max_text_len)
        n_gist = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < n_gist:
            gist_mask = torch.arange(n_gist, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(encoder_hidden_states[:, 0], n_gist, encoder_attention_mask, gist_mask)
        gist_freqs = self.rope_embedder(
            self.create_coordinate_grid((n_gist, 1, 1), (1, 0, 0), device).flatten(0, 2)).unsqueeze(0)

        # 5. one stream: gist, image, context - the order Fibo 2.0 was trained with (gist last when a batch pads it)
        n_ctx = x.shape[1] - n_img
        attention_mask = None
        if gist_mask is None:
            start = n_gist
            x = torch.cat([gist, x], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            start = 0
            x = torch.cat([x, gist], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            attention_mask = torch.cat([torch.ones(b, n_img + n_ctx, dtype=torch.bool, device=device), gist_mask],
                                       dim=1)
        noise_mask = torch.zeros(b, x.shape[1], dtype=torch.long, device=device)
        noise_mask[:, start:start + n_img] = 1

        attn_probe.layout((start, start + n_img), 0, x.shape[1], kv_total=encoder_hidden_states.shape[2])
        # 6. the main blocks; injection blocks also read their own bundle
        bundle = {layer_id: i + 1 for i, layer_id in enumerate(c.injection_layer_ids)}
        for i, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle[i]] if i in bundle else None
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            x = self._block(block, x, attention_mask, freqs_cis, None, noise_mask, adaln_noisy, adaln_clean, text,
                            encoder_attention_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, i)

        # 7. final layer on the image tokens, then unpatchify
        x = self.final_layer(x[:, start:start + n_img], adaln_noisy)
        return x.view(b, ht, wt, ch, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, ch, h, w)
