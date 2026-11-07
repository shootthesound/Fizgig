# Ported from OpenMOSS-Team/MOSS-Image-Base `pipeline_moss_image.py` (the diffusers custom pipeline shipped with the
# weights, md5 ba68f9308b1d9ad9bb90ae9755b2e3b7, 16 Oct 2026). Copyright 2026 OpenMOSS. Licensed under the Apache
# License, Version 2.0 (LICENSE beside this file).
# Changes for Fizgig: plain torch modules (no diffusers mixins), the forward split so the shared block offloader,
# gradient checkpointing, torch.compile and Reuse unchanged blocks wrap the 42 blocks, attention through Fizgig's INT8 hook
# (fizgig/modules/int8_attention) before torch SDPA, and a loader for the ComfyUI single file (fused q/k/v), the
# diffusers shards and ComfyUI's int8 ConvRot repack. The maths of every module is the reference's, line for line.
"""MOSS-Image DiT: 5B single-stream transformer, 42 blocks at width 3072 (24 heads x 128), shared modulation, a
two-block text adapter over Qwen3.5-9B's final hidden states, FLUX.2 latents 2x2-packed to 128 channels per token."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

# transformer/config.json (OpenMOSS-Team/MOSS-Image-Base) - the only configuration released
MOSS_CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                   latent_channels=128, rope_base=10000.0)
BLOCK_LINEARS = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
TEXT_BLOCK_LINEARS = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


# A training batch's padded captions need a key mask, and with one PyTorch's default choice falls back to its
# memory-efficient kernel; cuDNN takes the mask at almost no cost (#221: batch 3 at 0.25 MP 1.60 -> 1.31 s/step on an
# A6000, with stack_conds padding the captions to a multiple of 16 so cuDNN plans only a few shapes). A preference,
# not a demand: a GPU or shape cuDNN can't take falls back down the list.
_MASKED_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _attention(q, k, v, rope, heads, key_mask=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if key_mask is not None:            # a training batch (forward_batch): each picture's padded caption tokens dropped
        with sdpa_kernel(_MASKED_BACKENDS, set_priority=True):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        return out.transpose(1, 2).flatten(2)
    from fizgig.modules import int8_attention as _i8a      # workbench renders only, when opted in; else None
    out = _i8a.attend(q, k, v)
    if out is None:
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, key_mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, key_mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, key_mask=None):
        x = self.proj(x.to(getattr(self.proj, "base", self.proj).weight.dtype))   # through a LoRA wrapper (#298)
        for block in self.blocks:
            x = block(x, rope) if key_mask is None else block(x, rope, key_mask)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, key_mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        # the reference reads proj["q"].weight.dtype; Fizgig wraps (LoRA) or quantises (INT8 / NF4) that Linear, so the
        # block's own norm weight carries the same value - the dtype the checkpoint loaded in (bf16)
        dtype = self.norm1.weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads, key_mask)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    """(r, t, h, w) axes, head_dim / 8 bands each; r = 0 without references, so its quarter stays unrotated."""

    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base

    def bands(self, device):
        # the reference's CPU fp32 construction ("Match training's CPU fp32 construction")
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def forward(self, height, width, text_length, device):
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageDiT(nn.Module):
    """Module names are the checkpoint's (layers.N.proj.q ...), so LoRA keys are ComfyUI's."""

    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                 latent_channels=128, rope_base=10000.0):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.freq_dim, self.latent_channels = freq_dim, latent_channels
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    # ---- block swap (Fizgig's shared offloader, as Qwen Image 2.1's model uses it) ------------------------------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.layers)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("moss_image", list(self.layers), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped blocks' weights (the model is assumed to be on CPU)."""
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

    # ---- forward ------------------------------------------------------------------------------------------------
    def _io_dtype(self, lin):
        lin = getattr(lin, "base", lin)                 # a LoRA-wrapped Linear (a full-model difference wraps this one)
        dt = lin.weight.dtype
        return dt if dt.is_floating_point else torch.bfloat16

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, skip_layers=()):
        """hidden_states [B, 128, h, w] (normalised packed latents), timestep [B] or scalar on the reference's
        0-1000 scale (1000 * sigma), encoder_hidden_states [B, L, 4096], mask [B, L] (True = real token; padded tokens
        are dropped, as the reference does). skip_layers: block indices left out (the negative branch's layer 22).
        Returns the velocity [B, 128, h, w]."""
        skip = frozenset(skip_layers or ())
        batch, _, height, width = hidden_states.shape
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # the reference computes the sinusoid in float64 and feeds float32 into its MLP
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self._io_dtype(self.time_embedding[0]))).float()
        mod = self.shared_block_modulation(time.to(self._io_dtype(self.shared_block_modulation))).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self._io_dtype(self.video_head_modulation))).view(batch, 2, -1)
        layers = list(self.layers) if self.blocks_to_swap else None
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self._io_dtype(self.video_embedder)).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            m = mod[index:index + 1]
            for i, layer in enumerate(self.layers):
                if self.blocks_to_swap:
                    self.offloader.wait_for_block(i)
                if i not in skip:
                    if getattr(layer, "_handles_checkpointing", False):     # compiled: the wrapper checkpoints
                        x = layer(x, rope, m)
                    elif torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = checkpoint(layer, x, rope, m, use_reentrant=False)
                    else:
                        x = layer(x, rope, m)
                if self.blocks_to_swap:
                    self.offloader.submit_move_blocks_forward(layers, i)
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self._io_dtype(self.final_linear_video)))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)

    def forward_batch(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        """Batch Size above 1 (training): forward's pictures in one pass instead of one at a time. Each picture's
        caption is zero-padded at the end of the sequence with its padded keys masked out, and gets the positions its
        own caption length gives (the image's text axis is that length), so each picture's velocity is what forward
        gives it alone. hidden_states [B, 128, h, w], timestep [B] (1000 * sigma), encoder_hidden_states [B, L, 4096],
        encoder_attention_mask [B, L] True on real tokens."""
        batch, _, height, width = hidden_states.shape
        n = height * width
        dev = hidden_states.device
        mask = encoder_attention_mask.to(dev).bool()
        L = encoder_hidden_states.shape[1]
        times = torch.as_tensor(timestep, device=dev, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=dev, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self._io_dtype(self.time_embedding[0]))).float()
        mod = self.shared_block_modulation(time.to(self._io_dtype(self.shared_block_modulation))).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self._io_dtype(self.video_head_modulation))).view(batch, 2, -1)
        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self._io_dtype(self.video_embedder)).contiguous())
        ropes = []
        for index in range(batch):          # each picture's own: its image's text axis is its caption's length
            real = int(mask[index].sum())
            if real == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            r = self.rope(height, width, real, dev)
            if real < L:
                r = torch.cat([r, r[:, -1:].expand(1, L - real, r.shape[-1])], 1)   # padded slots: masked keys
            ropes.append(r)
        rope = torch.cat(ropes).to(video.dtype)
        key_mask = torch.cat([torch.ones(batch, n, dtype=torch.bool, device=dev), mask], 1)
        text = self.text_embedder(encoder_hidden_states.to(dev), rope[:, n:], mask)
        x = torch.cat((video, text), 1)
        layers = list(self.layers) if self.blocks_to_swap else None
        for i, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            if getattr(layer, "_handles_checkpointing", False):
                x = layer(x, rope, mod, key_mask)
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(layer, x, rope, mod, key_mask, use_reentrant=False)
            else:
                x = layer(x, rope, mod, key_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(layers, i)
        x = self.final_norm_video(x[:, :n].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self._io_dtype(self.final_linear_video)))
        return x.transpose(1, 2).reshape(batch, -1, height, width)


# ---- weights ---------------------------------------------------------------------------------------------------
def _files(path):
    import glob
    import os
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if path.endswith(".json"):                       # a diffusers index.json: its shards beside it
        return sorted(glob.glob(os.path.join(os.path.dirname(path), "*.safetensors")))
    if "-of-" in os.path.basename(path):            # one shard of a set: the whole set
        stem = os.path.basename(path).rsplit("-", 3)[0]
        return sorted(glob.glob(os.path.join(os.path.dirname(path), stem + "-*-of-*.safetensors")))
    return [path]


def read_state_dict(path, dtype=torch.bfloat16, plain=False) -> dict:
    """The DiT's tensors under this module's names, from any released layout:
    * ComfyUI's single file (Comfy-Org/MOSS-Image diffusion_models/moss_image_*_bf16): bare keys, q/k/v fused into
      proj.qkv = [q; k; v] (comfy/lora.py's MOSSImage branch maps LoRA thirds the same way) - split back here
    * the diffusers shards (OpenMOSS-Team/MOSS-Image-Base transformer/): this module's own names
    * ComfyUI's int8 ConvRot repack (moss_image_*_int8_convrot): decoded to `dtype` (Fizgig quantises itself)
    A "diffusion_model." / "model.diffusion_model." prefix is stripped; rope.bands (a derived constant) and ComfyUI's
    attention-kernel hints are dropped."""
    from safetensors.torch import load_file
    from fizgig.moss_image.embedder import from_file_wanted
    sd = {}
    for f in _files(path):
        if plain or from_file_wanted():
            # a computer under 30 GB of RAM, or a file loaded beside a live model (plain): plain reads - load_file maps
            # the whole file, which Windows counts against the commit limit on top of the copies
            from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
            with MemoryEfficientSafeOpen(f) as fh:
                sd.update({k: fh.get_tensor(k) for k in fh.keys()})
        else:
            sd.update(load_file(f, device="cpu"))
    for p in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(p) for k in sd):
            sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.endswith(".comfy_attention.config") and k != "rope.bands"}
    quant = [k[: -len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")]
    if quant:
        from fizgig.minimax.convrot import dequantize_int8_convrot, parse_comfy_quant
        for base in quant:
            conf = parse_comfy_quant(sd.pop(base + ".comfy_quant"))
            sd.pop(base + ".input_scale", None)       # an activation scale: these formats leave the input unquantised
            if conf.get("format") != "int8_tensorwise":
                raise ValueError(f"{base}: unsupported comfy quant format {conf.get('format')!r}")
            q, sc = sd.pop(base + ".weight"), sd.pop(base + ".weight_scale").float().reshape(-1, 1)
            sd[base + ".weight"] = dequantize_int8_convrot(q, sc, conf, out_dtype=dtype)
    out = {}
    for k, v in sd.items():
        if k.endswith(".proj.qkv.weight"):
            q, kk, vv = v.chunk(3, dim=0)
            base = k[: -len("qkv.weight")]
            out[base + "q.weight"], out[base + "k.weight"], out[base + "v.weight"] = q, kk, vv
        else:
            out[k] = v
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in out.items()}


def load_moss_dit(path, device="cuda", dtype=torch.bfloat16, config=None, plain=False) -> MOSSImageDiT:
    """Load with a strict key match (no silent drops, as the reference's load_training_state_dict)."""
    with torch.device("meta"):
        model = MOSSImageDiT(**(config or MOSS_CONFIG))
    sd = read_state_dict(path, dtype, plain)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"MOSS-Image DiT keys mismatch: missing {missing[:8]} ({len(missing)}), "
                         f"unexpected {unexpected[:8]} ({len(unexpected)})")
    return model.to(device)


def is_fused_file(path) -> bool:
    """The file stores q/k/v fused (ComfyUI's repack) - the fine-tune writes them back that way."""
    from safetensors import safe_open
    with safe_open(_files(path)[0], "pt") as f:
        return any(k.endswith(".proj.qkv.weight") for k in f.keys())


def per_block_gb(dtype_bytes=2.0, cfg=MOSS_CONFIG) -> float:
    """One main block's Linear weights in GB (what one swapped block frees)."""
    d, f = cfg["dim"], cfg["ffn_dim"]
    return (4 * d * d + 2 * f * d + f * d) * dtype_bytes / 1024 ** 3
