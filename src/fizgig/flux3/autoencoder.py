"""Vendored verbatim from black-forest-labs/flux-3-image-klein src/flux3/autoencoder.py (Apache-2.0, LICENSE beside
this file); Fizgig adds load_flux3_vae at the end. Same 0.34 GB FLUX.2 autoencoder (Comfy-Org ships it as
vae/flux2-vae.safetensors), with BatchNorm latent statistics."""


import math

import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EMBED_DIM = 32
CH = 128
CH_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        b, c, height, width = h.shape
        flat = lambda t: t.reshape(b, c, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        from fizgig.modules.wide_attention import wide_head_attention
        h = wide_head_attention(flat(self.q(h)), flat(self.k(h)), flat(self.v(h)))   # one wide head: direct on ROCm (#383)
        return x + self.proj_out(h.squeeze(1).permute(0, 2, 1).reshape(b, c, height, width))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * EMBED_DIM, 2 * EMBED_DIM, 1)
        self.conv_in = nn.Conv2d(3, CH, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = CH
        for level, mult in enumerate(CH_MULT):
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS):
                down.block.append(ResnetBlock(block_in, CH * mult))
                block_in = CH * mult
            if level != len(CH_MULT) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * EMBED_DIM, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != len(CH_MULT) - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(self, ch=CH):
        """ch: 128 for the FLUX.2 decoder, 96 for BFL's FLUX.2 small decoder (the one MOSS-Image ships)."""
        super().__init__()
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, EMBED_DIM, 1)
        block_in = ch * CH_MULT[-1]
        self.conv_in = nn.Conv2d(EMBED_DIM, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for level in reversed(range(len(CH_MULT))):
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS + 1):
                up.block.append(ResnetBlock(block_in, ch * CH_MULT[level]))
                block_in = ch * CH_MULT[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(CH_MULT))):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class AutoEncoder(nn.Module):
    def __init__(self, decoder_ch=CH):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decoder_ch)
        self.bn_eps = 1e-4
        self.bn = nn.BatchNorm2d(LATENT_CHANNELS, eps=self.bn_eps, affine=False, track_running_stats=True)

    @torch.inference_mode()
    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """image (3, H, W) in [-1, 1] -> tokens (h*w, 128) and ids (h*w, 4) = (0, row, col, 0)."""
        mean, _ = self.encoder(image.unsqueeze(0)).chunk(2, dim=1)
        z = mean.reshape(1, EMBED_DIM, mean.shape[2] // 2, 2, mean.shape[3] // 2, 2).permute(0, 1, 3, 5, 2, 4)
        z = self.bn(z.reshape(1, LATENT_CHANNELS, mean.shape[2] // 2, mean.shape[3] // 2))
        h, w = z.shape[2:]
        return z[0].permute(1, 2, 0).reshape(h * w, LATENT_CHANNELS), latent_ids(h, w, z.device)

    @torch.inference_mode()
    def decode(self, tokens: Tensor, ids: Tensor) -> Tensor:
        """tokens (h*w, 128) in id order -> image (3, H, W) in [-1, 1]."""
        h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
        z = tokens.float().reshape(h, w, LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
        z = z * torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps) + self.bn.running_mean.view(1, -1, 1, 1)
        z = z.reshape(1, EMBED_DIM, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, EMBED_DIM, 2 * h, 2 * w)
        return self.decoder(z)[0]


def latent_ids(h: int, w: int, device: torch.device | str = "cpu") -> Tensor:
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def latent_shape(height: int, width: int) -> tuple[int, int]:
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


def _from_diffusers(sd):
    """diffusers AutoencoderKLFlux2 names -> the reference's: down_blocks.i / up_blocks.i (up_blocks counted from the
    lowest resolution) -> down.i / up.(levels-1-i), resnets.j -> block.j, mid_block -> mid.block_1 / attn_1 / block_2,
    conv_shortcut -> nin_shortcut, conv_norm_out -> norm_out; quant convs move into the encoder / decoder; the
    attention's Linear weights become 1x1 convs."""
    n = len(CH_MULT)
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith(("quant_conv.", "post_quant_conv.")):
            k2 = ("encoder." if k2.startswith("quant") else "decoder.") + k2
        k2 = re.sub(r"\.down_blocks\.(\d+)\.resnets\.", r".down.\1.block.", k2)
        k2 = re.sub(r"\.down_blocks\.(\d+)\.downsamplers\.0\.", r".down.\1.downsample.", k2)
        k2 = re.sub(r"\.up_blocks\.(\d+)\.resnets\.", lambda m: f".up.{n - 1 - int(m.group(1))}.block.", k2)
        k2 = re.sub(r"\.up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f".up.{n - 1 - int(m.group(1))}.upsample.", k2)
        k2 = k2.replace(".mid_block.resnets.0.", ".mid.block_1.").replace(".mid_block.resnets.1.", ".mid.block_2.")
        k2 = k2.replace(".mid_block.attentions.0.", ".mid.attn_1.")
        k2 = (k2.replace("attn_1.group_norm.", "attn_1.norm.").replace("attn_1.to_q.", "attn_1.q.")
              .replace("attn_1.to_k.", "attn_1.k.").replace("attn_1.to_v.", "attn_1.v.")
              .replace("attn_1.to_out.0.", "attn_1.proj_out."))
        k2 = k2.replace(".conv_shortcut.", ".nin_shortcut.").replace(".conv_norm_out.", ".norm_out.")
        if ".attn_1." in k2 and k2.endswith(".weight") and v.dim() == 2:
            v = v[:, :, None, None]
        out[k2] = v
    return out


def load_flux3_vae(path: str, device="cpu", dtype=torch.float32) -> AutoEncoder:
    """ae.safetensors / flux2-vae.safetensors (or flux2-vae_small: the full encoder with BFL's small decoder) ->
    AutoEncoder (fp32, as the reference pipeline runs it)."""
    from safetensors.torch import load_file
    sd = load_file(path, device="cpu")
    for p in ("vae.", "first_stage_model."):
        if all(k.startswith(p) for k in sd):
            sd = {k[len(p):]: v for k, v in sd.items()}
    if "decoder.conv_norm_out.weight" in sd:      # diffusers names (Comfy-Org's flux2-vae_small, BFL's small decoder)
        sd = _from_diffusers(sd)
    ae = AutoEncoder(decoder_ch=sd["decoder.conv_out.weight"].shape[1] // CH_MULT[0])
    expected = set(ae.state_dict())
    missing = sorted(k for k in expected - set(sd) if not k.endswith("num_batches_tracked"))
    unexpected = sorted(set(sd) - expected)
    if missing or unexpected:
        raise KeyError(f"{path}: not the FLUX.2 / Flux 3 autoencoder - missing {missing[:5]}, unexpected {unexpected[:5]}")
    ae.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
    return ae.to(device).eval().requires_grad_(False)
