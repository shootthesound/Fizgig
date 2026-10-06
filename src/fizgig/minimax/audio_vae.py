# This file is derived from ComfyUI (https://github.com/comfyanonymous/ComfyUI), Copyright (C) comfyanonymous and
# the ComfyUI contributors, which is licensed under the GNU General Public License v3.0. This file therefore stays
# under GPL-3.0 and is not covered by Fizgig's Apache-2.0 licence. See THIRD_PARTY_NOTICES.md (ComfyUI).
"""MiniMax H3 audio VAE — encoder (waveform -> 32-ch latents at 40 Hz) AND decoder (back).

H3 denoises audio and video jointly, and Fizgig's DiT side has always packed the audio rows —
but as SILENCE, because nothing could turn a waveform into latents. The encoder is that missing
piece: with it, a training clip's real sound becomes a real target. The DECODER is the other
direction, for previews with sound: the sampler already denoises the audio rows for real, and
this turns them back into a waveform you can play (BigVGAN vocoder — the other half of the
same checkpoint file).

Ported from the ComfyUI reference (`comfy/ldm/minimax/audio_vae.py`: DAC-lineage encoder,
BigVGAN decoder, both MIT-lineage). Same treatment as the video VAE port next door: comfy's
`ops` become plain `nn` modules; each half loads only its own keys from the official
`minimax_h3_audio_vae_fp32.safetensors`.

Shapes and rates, all of which the DiT already assumes:

  * waveform  stereo [B, 2, L], 32 kHz, in [-1, 1]
  * hop       800 samples -> 32000/800 = **40 latents per second**, which is exactly the
              `AUDIO_LATENTS_PER_SECOND` model.py has always been written against
  * latents   [B, 32, 2, T] normalized; 32 is `audio_latents_dim`

Each stereo channel rides the mono encoder/decoder independently (`b*2` reshape, as in the
reference). Encode right-pads to a hop multiple; decode returns exactly T*800 samples.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 32000
HOP_LENGTH = 800                 # prod of the encoder strides (2*4*4*5*5)
LATENTS_PER_SECOND = SAMPLE_RATE // HOP_LENGTH      # 40 — matches model.AUDIO_LATENTS_PER_SECOND


def snake(x, alpha, beta):
    """x + 1/beta * sin^2(alpha * x). Not in-place: the reference can mutate its own temporaries,
    this runs under autograd where that would break the backward."""
    t = torch.sin(alpha * x)
    return x + (t * t) * (beta + 1e-9).reciprocal()


class Snake1d(nn.Module):
    """Snake with one alpha per channel, alpha serving as beta too (the encoder's variant)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        a = self.alpha.to(x)
        return snake(x, a, a)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y + x


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                      padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """DAC encoder: [B, 1, L] waveform -> [B, d_latent, L/800]."""

    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    """Causal attention that also POOLS 2048 -> 32.

    Two easy things to get wrong, both mirrored from the reference:
      * qkv has no bias parameter of its own; q and v biases are separate tensors and k's bias is
        a registered ZERO buffer, concatenated at call time. Dropping the zeros shifts k.
      * the head axis is averaged (not concatenated) and then adaptive-avg-pooled down to the
        latent width, which is how 2048 becomes 32.
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias.to(self.q_bias), self.v_bias)).to(x)
        qkv = F.linear(x, self.qkv.weight.to(x), bias)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    """The posterior head: [B, T, 2048] -> [B, T, 32]."""

    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio))

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# --- decoder (BigVGAN vocoder) -----------------------------------------------------------------
# Latents -> waveform, for previews with sound. Mirrored from the same ComfyUI reference file
# as the encoder above; runs under no_grad only, so plain ops throughout.

class SnakeBeta(nn.Module):
    """x + 1/beta * sin^2(alpha * x), alpha/beta stored in LOG scale (the decoder's variant)."""

    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        alpha = torch.exp(self.alpha.to(x)).view(1, -1, 1)
        beta = torch.exp(self.beta.to(x)).view(1, -1, 1)
        return snake(x, alpha, beta)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Kaiser-windowed sinc low-pass, [1, 1, kernel_size], sum-normalized."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filt /= filt.sum()               # no leakage of the constant component
    return filt.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, self.filter.to(x).expand(C, -1, -1),
                               stride=self.stride, groups=C) * self.ratio
        return x[..., self.pad_left:-self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.to(x).expand(C, -1, -1),
                        stride=self.stride, groups=C)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio,
                                       stride=ratio, kernel_size=kernel_size)

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Anti-aliased pointwise activation: upsample x2 -> act -> downsample x2."""

    def __init__(self, activation):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d,
                      padding=_get_padding(kernel_size, d)) for d in dilation])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1,
                      padding=_get_padding(kernel_size, 1)) for _ in range(len(dilation))])
        self.activations = nn.ModuleList(
            [Activation1d(SnakeBeta(channels))
             for _ in range(len(self.convs1) + len(self.convs2))])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class BigVGAN(nn.Module):
    """BigVGAN vocoder, MiniMax H3 32 kHz configuration (no bias/tanh at the final conv;
    output clamped to [-1, 1]). Upsample rates 5*5*2^5 = 800 — one latent back to one hop."""

    def __init__(self, num_mels=2048, upsample_initial_channel=1024,
                 upsample_rates=(5, 5, 2, 2, 2, 2, 2),
                 upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                   upsample_initial_channel // (2 ** (i + 1)),
                                   k, u, padding=(k - u) // 2)]))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d))
        self.activation_post = Activation1d(SnakeBeta(ch))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        x = self.activation_post(x)
        return self.conv_post(x).clamp(-1.0, 1.0)


def unpack_audio(rows: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """[2*T, 32] channel-major rows -> [1, 32, 2, T] — pack_audio's exact inverse."""
    total, c = rows.shape
    t = total // channels
    return rows.reshape(channels, t, c).permute(2, 0, 1).unsqueeze(0)


class MiniMaxH3AudioVAEDecoder(nn.Module):
    """Decode-only: normalized latents [B, 32, 2, T] -> stereo waveform [B, 2, T*800]."""

    def __init__(self, latent_dim=2048, decoder_dim=1024, vae_latent_channels=32):
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(num_mels=latent_dim, upsample_initial_channel=decoder_dim)
        self.register_buffer("latents_mean", torch.zeros(vae_latent_channels))
        self.register_buffer("latents_std", torch.ones(vae_latent_channels))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, c, s, t = z.shape
        z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
        mean = self.latents_mean.view(1, -1, 1).to(z)
        std = self.latents_std.view(1, -1, 1).to(z)
        z = z * std + mean                       # denormalize — encode's exact inverse
        x = self.decoder(self.dec_in_proj(z))    # [b*s, 1, L], clamped
        return x.reshape(b, s, -1)


def load_minimax_h3_audio_vae_decoder(path: str, device="cuda", dtype=torch.float32
                                      ) -> MiniMaxH3AudioVAEDecoder:
    """Build the decoder and load its half of the official checkpoint.

    The kaiser sinc filters are computed at construction and may legitimately be absent from
    the file; every WEIGHT must be present — a silently random vocoder would hiss instead of
    raising, which is the failure mode this refuses."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

    model = MiniMaxH3AudioVAEDecoder()
    wanted = set(model.state_dict().keys())
    sd = {}
    with MemoryEfficientSafeOpen(path) as f:
        for k in f.keys():
            if k in wanted:
                sd[k] = f.get_tensor(k)
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if not m.endswith(".filter")]
    if real_missing:
        raise ValueError(
            f"{path} is missing {len(real_missing)} audio-decoder weight(s), e.g. "
            f"{real_missing[:3]} — expected the MiniMax H3 audio VAE "
            f"(minimax_h3_audio_vae_fp32.safetensors).")
    return model.to(device=device, dtype=dtype).eval()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """[B, 32, 2, T] -> [2*T, 32] rows, CHANNEL-MAJOR (ch0 t0..T-1, then ch1 t0..T-1).

    The ordering is a checkpoint contract, not a choice — it is what the frozen base was trained
    on, and it is `pack_audio` in the reference DiT verbatim. Time-major would run silently and
    train against scrambled targets.
    """
    _, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


class MiniMaxH3AudioVAEEncoder(nn.Module):
    """Encode-only. The official `minimax_h3_audio_vae_fp32.safetensors` loads with
    `strict=False`; the decoder / logs_proj / dec_in_proj keys are simply unused."""

    def __init__(self, encoder_dim=64, encoder_rates=(2, 4, 4, 5, 5),
                 latent_dim=2048, vae_latent_channels=32):
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self.hop_length = int(math.prod(encoder_rates))          # 800
        self.latents_per_second = self.sample_rate // self.hop_length
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)
        self.mean_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.register_buffer("latents_mean", torch.zeros(vae_latent_channels))
        self.register_buffer("latents_std", torch.ones(vae_latent_channels))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Stereo [B, 2, L] at 32 kHz in [-1, 1] -> normalized latents [B, 32, 2, T].

        The posterior MEAN is used directly, with no sampling — same choice as the video VAE
        path, where a frozen draw is strictly worse than the mean for a cached target.
        """
        b, s, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad:
            waveform = F.pad(waveform, (0, right_pad))
        x = self.encoder(waveform.reshape(b * s, 1, -1))
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        mean = self.latents_mean.view(1, -1, 1).to(z)
        std = self.latents_std.view(1, -1, 1).to(z)
        z = (z - mean) / std
        return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)


def load_minimax_h3_audio_vae(path: str, device="cuda", dtype=torch.float32
                              ) -> MiniMaxH3AudioVAEEncoder:
    """Build the encoder and load the official checkpoint, ignoring the decoder half.

    Read through MemoryEfficientSafeOpen for the same reason every other loader here does: the
    official safe_open mmap path hard-crashes on Windows for large files.
    """
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

    model = MiniMaxH3AudioVAEEncoder()
    wanted = set(model.state_dict().keys())
    sd = {}
    with MemoryEfficientSafeOpen(path) as f:
        for k in f.keys():
            if k in wanted:
                sd[k] = f.get_tensor(k)
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    # zero_k_bias is a constant this port registers itself; anything else missing means the file
    # is not the encoder we think it is, and a silently random encoder would poison every cached
    # audio target without raising.
    real_missing = [m for m in missing if not m.endswith("zero_k_bias")]
    if real_missing:
        raise ValueError(
            f"{path} is missing {len(real_missing)} audio-encoder weight(s), e.g. "
            f"{real_missing[:3]} — expected the MiniMax H3 audio VAE "
            f"(minimax_h3_audio_vae_fp32.safetensors).")
    return model.to(device=device, dtype=dtype).eval()
