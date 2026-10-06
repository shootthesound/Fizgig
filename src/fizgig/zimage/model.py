# Adapted from the Z-Image reference implementation (https://github.com/Tongyi-MAI/Z-Image, src/zimage/transformer.py),
# Copyright 2025 Alibaba Z-Image Team, licensed under the Apache License, Version 2.0.
# Changes for Fizgig: constants inlined, PyTorch SDPA instead of the backend dispatcher, gradient checkpointing on the
# blocks, a loader for ComfyUI's single file (fused qkv split back into to_q / to_k / to_v).
"""Z-Image's S3-DiT: a 6B single-stream transformer (30 layers, width 3840, 30 heads of 128) with two image-only
"noise refiner" blocks and two text-only "context refiner" blocks before the joint stack; image tokens then caption
tokens; 3-axis RoPE (32 / 48 / 48, theta 256); AdaLN from the timestep only (scale and tanh gate, no shift).

Flow convention (the reference pipeline): x_sigma = (1 - sigma) x0 + sigma noise; the model is called at t = 1 - sigma
and returns x0 - noise (the negated velocity).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from fizgig.modules.int8_attention import attend

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
ROPE_THETA = 256.0
ROPE_AXES_DIMS = (32, 48, 48)
ROPE_AXES_LENS = (1536, 512, 512)
FREQUENCY_EMBEDDING_SIZE = 256
MAX_PERIOD = 10000


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size, bias=True), nn.SiLU(),
                                 nn.Linear(mid_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in, freqs_cis):
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        return torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3).type_as(x_in)


def _mask_or_none(mask):
    """None when every token is real (one item, or equal lengths): the attention then needs no mask."""
    return None if bool(mask.all()) else mask


class ZImageAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm=True, eps=1e-5):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])
        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        q = self.to_q(hidden_states).unflatten(-1, (self.n_heads, -1))
        k = self.to_k(hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        v = self.to_v(hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        dtype = q.dtype
        # (B, S) True = attend -> SDPA's boolean mask; None when nothing is padded (decided in the model's forward,
        # outside the compiled layers)
        mask = attention_mask[:, None, None, :] if attention_mask is not None else None
        q, k, v = q.transpose(1, 2), k.to(dtype).transpose(1, 2), v.transpose(1, 2)
        out = attend(q, k, v) if mask is None else None       # workbench renders: INT8 attention when switched on
        if out is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.to_out[0](out.transpose(1, 2).flatten(2, 3).to(dtype))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True):
        super().__init__()
        self.dim, self.layer_id, self.modulation = dim, layer_id, modulation
        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


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
    def __init__(self, theta=ROPE_THETA, axes_dims=ROPE_AXES_DIMS, axes_lens=ROPE_AXES_LENS):
        self.theta, self.axes_dims, self.axes_lens = theta, list(axes_dims), list(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=ROPE_THETA):
        out = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            freqs = torch.outer(torch.arange(e, dtype=torch.float64), freqs).float()
            out.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return out

    def __call__(self, ids):
        if self.freqs_cis is None or self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [f.to(ids.device) for f in self.precompute_freqs_cis(self.axes_dims, self.axes_lens,
                                                                                  self.theta)]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


class ZImageTransformer2DModel(nn.Module):
    def __init__(self, all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=3840, n_layers=30,
                 n_refiner_layers=2, n_heads=30, n_kv_heads=30, norm_eps=1e-5, qk_norm=True, cap_feat_dim=2560,
                 rope_theta=ROPE_THETA, t_scale=1000.0, axes_dims=ROPE_AXES_DIMS, axes_lens=ROPE_AXES_LENS):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.all_patch_size, self.all_f_patch_size = all_patch_size, all_f_patch_size
        self.dim, self.n_heads, self.t_scale = dim, n_heads, t_scale
        self.all_x_embedder = nn.ModuleDict({
            f"{p}-{f}": nn.Linear(f * p * p * in_channels, dim, bias=True) for p, f in zip(all_patch_size, all_f_patch_size)})
        self.all_final_layer = nn.ModuleDict({
            f"{p}-{f}": FinalLayer(dim, p * p * f * in_channels) for p, f in zip(all_patch_size, all_f_patch_size)})
        self.noise_refiner = nn.ModuleList([ZImageTransformerBlock(1000 + i, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                                                                   modulation=True) for i in range(n_refiner_layers)])
        self.context_refiner = nn.ModuleList([ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                                                                     modulation=False) for i in range(n_refiner_layers)])
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))
        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.layers = nn.ModuleList([ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                                     for i in range(n_layers)])
        assert dim // n_heads == sum(axes_dims)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def _run(self, layer, *args):
        if getattr(layer, "_handles_checkpointing", False):     # compiled: the wrapper checkpoints itself
            return layer(*args)
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    # ---- block swap over the 30 main layers (Fizgig's shared offloader, as Qwen 2.1 uses it) ---------------------
    def enable_block_swap(self, num_blocks, device, supports_backward=True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.layers)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("zimage", list(self.layers), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped layers' weights (the model is assumed to be on CPU)."""
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

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        start = start or tuple(0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def unpatchify(self, x, size, patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        for i in range(len(x)):
            Fr, H, W = size[i]
            n = (Fr // pF) * (H // pH) * (W // pW)
            x[i] = (x[i][:n].view(Fr // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, Fr, H, W))
        return x

    def patchify_and_embed(self, all_image, all_cap_feats, patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device
        img_out, img_size, img_pos, img_pad, cap_pos, cap_pad, cap_out = [], [], [], [], [], [], []
        for image, cap_feat in zip(all_image, all_cap_feats):
            cl = len(cap_feat)
            cp = (-cl) % SEQ_MULTI_OF
            cap_pos.append(self.create_coordinate_grid((cl + cp, 1, 1), (1, 0, 0), device).flatten(0, 2))
            cap_pad.append(torch.cat([torch.zeros(cl, dtype=torch.bool, device=device),
                                      torch.ones(cp, dtype=torch.bool, device=device)]))
            cap_out.append(torch.cat([cap_feat, cap_feat[-1:].repeat(cp, 1)]) if cp else cap_feat)
            C, Fr, H, W = image.size()
            img_size.append((Fr, H, W))
            Ft, Ht, Wt = Fr // pF, H // pH, W // pW
            image = image.view(C, Ft, pF, Ht, pH, Wt, pW).permute(1, 3, 5, 2, 4, 6, 0).reshape(Ft * Ht * Wt, pF * pH * pW * C)
            il = len(image)
            ip = (-il) % SEQ_MULTI_OF
            pos = self.create_coordinate_grid((Ft, Ht, Wt), (cl + cp + 1, 0, 0), device).flatten(0, 2)
            if ip:
                pos = torch.cat([pos, self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(ip, 1)])
            img_pos.append(pos)
            img_pad.append(torch.cat([torch.zeros(il, dtype=torch.bool, device=device),
                                      torch.ones(ip, dtype=torch.bool, device=device)]))
            img_out.append(torch.cat([image, image[-1:].repeat(ip, 1)]) if ip else image)
        return img_out, cap_out, img_size, img_pos, cap_pos, img_pad, cap_pad

    def forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1):
        """x: list of latents (C, F, H, W); t: (B,) at 1 - sigma; cap_feats: list of (L, 2560). -> list of
        (C, F, H, W), the model's x0 - noise."""
        bsz = len(x)
        device = x[0].device
        adaln = self.t_embedder(t * self.t_scale)
        x, cap_feats, x_size, x_pos, cap_pos, x_pad, cap_pad = self.patchify_and_embed(x, cap_feats, patch_size,
                                                                                         f_patch_size)
        x_lens = [len(a) for a in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x))
        adaln = adaln.type_as(x)
        x = torch.where(torch.cat(x_pad)[:, None], self.x_pad_token.to(x), x)
        x = list(x.split(x_lens))
        x_freqs = list(self.rope_embedder(torch.cat(x_pos)).split([len(a) for a in x_pos]))
        x = pad_sequence(x, batch_first=True)
        x_freqs = pad_sequence(x_freqs, batch_first=True)[:, :x.shape[1]]
        x_mask = torch.zeros((bsz, max(x_lens)), dtype=torch.bool, device=device)
        for i, n in enumerate(x_lens):
            x_mask[i, :n] = True
        x_mask = _mask_or_none(x_mask)
        for layer in self.noise_refiner:
            x = self._run(layer, x, x_mask, x_freqs, adaln)

        cap_lens = [len(a) for a in cap_feats]
        c = self.cap_embedder(torch.cat(cap_feats))
        c = torch.where(torch.cat(cap_pad)[:, None], self.cap_pad_token.to(c), c)
        c = list(c.split(cap_lens))
        c_freqs = list(self.rope_embedder(torch.cat(cap_pos)).split([len(a) for a in cap_pos]))
        c = pad_sequence(c, batch_first=True)
        c_freqs = pad_sequence(c_freqs, batch_first=True)[:, :c.shape[1]]
        c_mask = torch.zeros((bsz, max(cap_lens)), dtype=torch.bool, device=device)
        for i, n in enumerate(cap_lens):
            c_mask[i, :n] = True
        c_mask = _mask_or_none(c_mask)
        for layer in self.context_refiner:
            c = self._run(layer, c, c_mask, c_freqs)

        u = [torch.cat([x[i][:x_lens[i]], c[i][:cap_lens[i]]]) for i in range(bsz)]
        u_freqs = [torch.cat([x_freqs[i][:x_lens[i]], c_freqs[i][:cap_lens[i]]]) for i in range(bsz)]
        u_lens = [len(a) for a in u]
        u = pad_sequence(u, batch_first=True)
        u_freqs = pad_sequence(u_freqs, batch_first=True)
        u_mask = torch.zeros((bsz, max(u_lens)), dtype=torch.bool, device=device)
        for i, n in enumerate(u_lens):
            u_mask[i, :n] = True
        u_mask = _mask_or_none(u_mask)
        layers = list(self.layers) if self.blocks_to_swap else None
        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            u = self._run(layer, u, u_mask, u_freqs, adaln)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(layers, index)
        u = self.all_final_layer[f"{patch_size}-{f_patch_size}"](u, adaln)
        return self.unpatchify(list(u.unbind(0)), x_size, patch_size, f_patch_size)


# ---- ComfyUI's single file -> this module's names -----------------------------------------------------------------
_RENAMES = (("attention.out.", "attention.to_out.0."), ("attention.q_norm.", "attention.norm_q."),
            ("attention.k_norm.", "attention.norm_k."))


def convert_comfy_state_dict(sd):
    """ComfyUI's Z-Image file (fused qkv, `out`, q_norm / k_norm, x_embedder, final_layer; also with a
    `model.diffusion_model.` prefix) -> the reference module names. Already-reference keys pass through."""
    out = {}
    for k, v in sd.items():
        for p in ("model.diffusion_model.", "diffusion_model."):
            if k.startswith(p):
                k = k[len(p):]
        if k.endswith("attention.qkv.weight"):
            q, kk, vv = v.chunk(3, dim=0)
            base = k[:-len("qkv.weight")]
            out[base + "to_q.weight"], out[base + "to_k.weight"], out[base + "to_v.weight"] = q, kk, vv
            continue
        for a, b in _RENAMES:
            k = k.replace(a, b)
        if k.startswith("x_embedder."):
            k = "all_x_embedder.2-1." + k[len("x_embedder."):]
        elif k.startswith("final_layer."):
            k = "all_final_layer.2-1." + k[len("final_layer."):]
        out[k] = v
    return out


def load_zimage_dit(path, device="cpu", dtype=torch.bfloat16):
    """The DiT from a ComfyUI single file (or the reference names), frozen, in `dtype`."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    with init_empty_weights():
        dit = ZImageTransformer2DModel()
    sd = convert_comfy_state_dict({k: v.to(dtype) for k, v in load_file(path).items()})
    missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"not a Z-Image DiT file: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return dit.to(device).eval().requires_grad_(False)
