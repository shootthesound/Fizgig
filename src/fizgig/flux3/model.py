"""FLUX 3 Image Klein DiT: per-mode blocks (image / reference image / text), then shared single-stream blocks.

Vendored from black-forest-labs/flux-3-image-klein src/flux3/dit.py (Apache-2.0, see LICENSE beside this file) with
the module names unchanged: LoRA keys come from named_modules(), and ComfyUI (comfy/lora.py, the Flux3 branch) maps
lora_unet_<path> onto these names, including q_proj / k_proj / v_proj on a file that fuses them into qkv.

Fizgig additions: gradient checkpointing around each block, and a loader that accepts the reference package
(model.safetensors) or Comfy-Org's single file (diffusion_models/flux3-image-klein*-bf16.safetensors).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from fizgig.modules import int8_attention as _i8a
from fizgig.training import attn_probe


MODES = ("image", "image_cond", "text")  # sorted; the order tokens take inside a sample
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
HIDDEN = 3072
HEADS = 24
DEPTH = 5
DEPTH_SINGLE = 28
AXES_DIM = (32, 32, 32, 32)
MLP_RATIO = 3.0
THETA = 10000
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"   # only the distilled (flow-map) checkpoint has it


@dataclass
class Packed:
    """Mode-major token pack. mode_indices[m][s] = (start, end) of sample s in data[m]."""

    data: dict
    ids: dict
    timesteps: dict
    mode_indices: dict

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


def timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids: Tensor) -> Tensor:
    """ids (B, L, 4) -> rotation matrices (B, 1, L, head_dim // 2, 2, 2)."""
    with torch.autocast(device_type=ids.device.type, enabled=False):
        emb = torch.cat([rope(ids[..., i], AXES_DIM[i], THETA) for i in range(len(AXES_DIM))], dim=-3)
    return emb.unsqueeze(1)


def rmsnorm_rope(x: Tensor, scale: Tensor, freqs: Tensor, eps: float = 1e-6) -> Tensor:
    with torch.autocast(device_type="cuda", enabled=False):
        b, h, l, d = x.shape
        x_f = x.float()
        x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        x_r, freqs = x_norm.reshape(b, h, l, d // 2, 1, 2), freqs.float()
        rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
        return rotated.reshape(b, h, l, d).to(x.dtype)


def layernorm_modulation(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return (1.0 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift


def packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds: list) -> Tensor:
    """Per-sample attention over (1, H, L, D); samples never attend to each other."""
    _, heads, length, dim = q.shape
    out = torch.zeros(length, heads * dim, device=q.device, dtype=q.dtype)
    backend = SDPBackend.CUDNN_ATTENTION if q.is_cuda else SDPBackend.MATH
    for start, end in bounds:
        if start == end:
            continue
        qs, ks, vs = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
        chunk = _i8a.attend(qs, ks, vs)          # workbench renders: the INT8 kernel when it's switched on
        if chunk is None:
            with sdpa_kernel(backend):
                chunk = F.scaled_dot_product_attention(qs, ks, vs)
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, freqs: Tensor):
        return rmsnorm_rope(q, self.query_norm.scale, freqs), rmsnorm_rope(k, self.key_norm.scale, freqs)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: Tensor):
        return tuple(self.lin(F.silu(vec)).chunk(3, dim=-1))  # shift, scale, gate


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(layernorm_modulation(x, scale, shift))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def forward(self, x: Tensor, freqs: Tensor, bounds: list, mod: tuple) -> Tensor:
        shift, scale, gate = mod
        x_mod = layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(1, 2)  # L (H D) -> 1 H L D
        q, k = self.norm(split(self.q_proj(x_mod)), split(self.k_proj(x_mod)), freqs)
        if attn_probe._ACTIVE is not None:      # training maps: the trigger word's attention (training/attn_probe.py)
            attn_probe.see(q, k)
        v = split(self.v_proj(x_mod))
        attn = packed_attention(q, k, v.to(q.dtype), bounds)
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))


class Flux3DiT(nn.Module):
    def __init__(self, flow_map: bool = False) -> None:
        super().__init__()
        self.flow_map = flow_map
        self.gradient_checkpointing = False
        self.emb_in = nn.ModuleDict({m: nn.Linear(IN_CHANNELS[m], HIDDEN, bias=False) for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(IN_CHANNELS["text"], HIDDEN, bias=False)
        self.time_in = MLPEmbedder(256, HIDDEN)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, HIDDEN)
        self.early_stream_modulations = nn.ModuleDict({k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.single_stream_modulations = nn.ModuleDict({k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.content_mode_blocks = nn.ModuleDict({
            m: nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]) for m in ("image", "image_cond")
        })
        self.txt_mode_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.single_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH_SINGLE)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(HIDDEN, IN_CHANNELS["image"])})
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, on: bool = True) -> None:
        self.gradient_checkpointing = on

    # ---- block swap: the 28 single blocks stream between CPU and GPU (the shared offloader, as Qwen 2.1's) ----------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.single_blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("flux3", list(self.single_blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped blocks' weights (the model is assumed to be on CPU)."""
        blocks = self.single_blocks
        self.single_blocks = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.single_blocks = blocks
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.single_blocks))

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

    def _run(self, block, *args):
        if getattr(block, "_handles_checkpointing", False):    # compiled: the wrapper checkpoints itself
            return block(*args)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, packed: Packed, timesteps_ref: Tensor = None) -> Tensor:
        """Velocity prediction for the image tokens, in packed.data["image"] order.
        timesteps_ref: flow map only, the time each sample jumps to."""
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a flow-map DiT")
        data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for m in MODES:  # an empty mode gets one phantom token so the projections run; it is never attended
            if data[m].shape[0] == 0:
                data[m], ids[m], timesteps[m] = data[m].new_zeros(1, data[m].shape[1]), ids[m].new_zeros(1, 4), timesteps[m].new_zeros(1)
        mod_key = lambda m: "txt" if m == "text" else m
        vec = {m: self.time_in(timestep_embedding(timesteps[m])) for m in MODES}
        if timesteps_ref is not None:
            ref = self.ref_time_in(timestep_embedding(timesteps_ref.to(timesteps["image"].dtype)))
            for m in MODES:
                parts = [ref[s].expand(b - a, -1) for s, (a, b) in enumerate(packed.mode_indices[m]) if b > a]
                if parts:
                    vec[m] = vec[m] + torch.cat(parts)
        early = {m: self.early_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        single = {m: self.single_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(data[m]) for m in MODES}
        freqs = {m: rope_embed(ids[m][None]) for m in MODES}
        for d in range(DEPTH):
            for m in ("image", "image_cond"):
                x[m] = self._run(self.content_mode_blocks[m][d], x[m], freqs[m], packed.mode_indices[m], early[m])
            x["text"] = self._run(self.txt_mode_blocks[d], x["text"], freqs["text"], packed.mode_indices["text"], early["text"])

        chunks, id_chunks, mods, bounds, placement, offset = [], [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        if attn_probe._ACTIVE is not None and packed.num_samples == 1:     # the picture's rows, where the text starts
            at = {m: (o, o + b - a) for m, a, b, o in placement}
            if "image" in at and "text" in at:
                attn_probe.layout(at["image"], at["text"][0], h.shape[0])
        freqs_sm = rope_embed(torch.cat(id_chunks)[None])
        mods = tuple(torch.cat(c) for c in mods)
        blocks = list(self.single_blocks) if self.blocks_to_swap else None
        for index, block in enumerate(self.single_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            h = self._run(block, h, freqs_sm, bounds, mods)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)

        image = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                image[a:b] = h[start : start + b - a]
        return self.final_layer["image"](image, vec["image"])


# ---- loading ------------------------------------------------------------------------------------------------------
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")


def _strip_prefix(keys) -> str:
    for p in _PREFIXES:
        if any(k.startswith(p) for k in keys):
            return p
    return ""


def is_flow_map(path: str) -> bool:
    """True for the distilled checkpoint (sampled without CFG through the flow map)."""
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
    p = _strip_prefix(keys)
    return p + FLOW_MAP_KEY in keys


def is_quantized(path: str) -> bool:
    """A Comfy int8 ConvRot / NVFP4 / FP8 file: not loadable here yet (bf16 only)."""
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return any(k.endswith((".weight_scale", ".input_scale", ".comfy_quant")) for k in f.keys())


def convert_state_dict(sd: dict) -> dict:
    """Comfy-Org's single file -> this module's names: drop the diffusion_model. prefix and split any fused
    <block>.qkv.weight into q_proj / k_proj / v_proj (thirds along the output dim, the order comfy/lora.py slices)."""
    p = _strip_prefix(sd.keys())
    out = {}
    for k, v in sd.items():
        k = k[len(p):] if p and k.startswith(p) else k
        if k.endswith(".qkv.weight"):
            stem = k[: -len(".qkv.weight")]
            q, kk, vv = v.chunk(3, dim=0)
            out[f"{stem}.q_proj.weight"], out[f"{stem}.k_proj.weight"], out[f"{stem}.v_proj.weight"] = q, kk, vv
        else:
            out[k] = v
    return out


def load_flux3_dit(path: str, device="cpu", dtype=torch.bfloat16) -> Flux3DiT:
    """bf16 file -> Flux3DiT on `device`. Distilled vs base is read from the file (the flow-map key)."""
    from safetensors.torch import load_file
    if is_quantized(path):
        raise NotImplementedError(f"{path}: quantised Flux 3 files are not supported yet - use a bf16 file "
                                  f"(Fizgig quantises to INT8 / NF4 itself)")
    sd = convert_state_dict(load_file(path, device="cpu"))
    with torch.device("meta"):
        dit = Flux3DiT(flow_map=FLOW_MAP_KEY in sd)
    expected = set(dit.state_dict())
    missing, unexpected = sorted(expected - set(sd)), sorted(set(sd) - expected)
    if missing or unexpected:
        raise KeyError(f"{path}: not a Flux 3 Image Klein DiT - missing {missing[:5]}, unexpected {unexpected[:5]}")
    dit.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    return dit.to(device)
