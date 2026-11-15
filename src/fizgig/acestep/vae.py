"""ACE-Step 1.5's audio VAE: an Oobleck autoencoder (48 kHz stereo <-> 64 channels at 25 Hz, hop 1920), run through
diffusers' AutoencoderOobleck. The Comfy-Org file stores it in stable-audio-tools' layout (encoder.layers.N...,
weight-normed convs, flat Snake alpha / beta); load_vae renames every key and checks every shape."""
import re

import torch

CONFIG = {"audio_channels": 2, "channel_multiples": [1, 2, 4, 8, 16], "decoder_channels": 128,
          "decoder_input_channels": 64, "downsampling_ratios": [2, 4, 4, 6, 10], "encoder_hidden_size": 128,
          "sampling_rate": 48000}      # ACE-Step/Ace-Step1.5 vae/config.json
SAMPLE_RATE = 48000
HOP = 1920


def _unit(n):
    """A residual unit's layer inside stable-audio-tools' Sequential: 0 Snake, 1 conv, 2 Snake, 3 conv."""
    return {"0": "snake1", "1": "conv1", "2": "snake2", "3": "conv2"}[n]


def diffusers_name(key):
    """stable-audio-tools key -> diffusers AutoencoderOobleck key (None for a key outside the VAE)."""
    m = re.match(r"(encoder|decoder)\.layers\.(\d+)\.(.+)$", key)
    if not m:
        return None
    side, i, rest = m.group(1), int(m.group(2)), m.group(3)
    if i == 0:
        return f"{side}.conv1.{rest}"
    if i == 6:
        return f"{side}.snake1.{rest}"
    if i == 7:
        return f"{side}.conv2.{rest}"
    b = i - 1
    m2 = re.match(r"layers\.(\d+)\.(.+)$", rest)
    j, tail = int(m2.group(1)), m2.group(2)
    if side == "encoder":                  # 0-2 residual units, 3 Snake, 4 downsampling conv
        if j <= 2:
            u = re.match(r"layers\.(\d)\.(.+)$", tail)
            return f"encoder.block.{b}.res_unit{j + 1}.{_unit(u.group(1))}.{u.group(2)}"
        return f"encoder.block.{b}.{'snake1' if j == 3 else 'conv1'}.{tail}"
    if j <= 1:                             # decoder: 0 Snake, 1 upsampling conv, 2-4 residual units
        return f"decoder.block.{b}.{'snake1' if j == 0 else 'conv_t1'}.{tail}"
    u = re.match(r"layers\.(\d)\.(.+)$", tail)
    return f"decoder.block.{b}.res_unit{j - 1}.{_unit(u.group(1))}.{u.group(2)}"


def load_vae(path, device, dtype=torch.float32):
    from diffusers import AutoencoderOobleck
    from safetensors.torch import load_file
    vae = AutoencoderOobleck(**CONFIG)
    target = vae.state_dict()
    sd = {}
    for k, v in load_file(path).items():
        name = diffusers_name(k[len("vae."):] if k.startswith("vae.") else k)
        if name is None:
            continue
        if name not in target:
            raise RuntimeError(f"ACE-Step VAE: {k} maps to {name}, which the Oobleck VAE doesn't have")
        sd[name] = v.reshape(target[name].shape)
    missing = set(target) - set(sd)
    if missing:
        raise RuntimeError(f"ACE-Step VAE: {len(missing)} weights missing from {path}, e.g. {sorted(missing)[:3]}")
    vae.load_state_dict(sd)
    return vae.to(device, dtype).eval().requires_grad_(False)


# A song goes through the encoder in pieces (#347). Whole, a 4-minute song's first layers hold 128 channels x 11.5 M
# samples - about 6 GB per activation in float32 - and caching ran out of memory on 16 and 24 GB cards. Each piece
# carries OVERLAP of real audio either side, so the convolutions see the same neighbours they would in one pass, and
# only its middle is kept; both lengths are whole hops, so every piece's latents land exactly on the song's frames.
CHUNK = 15 * SAMPLE_RATE                 # 375 latent frames
OVERLAP = 2 * SAMPLE_RATE                # 50 frames either side


@torch.no_grad()
def posterior(vae, audio):
    """audio (2, n) float at 48 kHz -> the posterior's (mean, std), each (1, 64, T), encoded piece by piece."""
    x = audio[None]
    n = x.shape[-1]
    if n <= CHUNK + 2 * OVERLAP:
        d = vae.encode(x.to(vae.device, vae.dtype)).latent_dist
        return d.mean, d.std
    means, stds = [], []
    s = 0
    while True:
        lo, hi = max(0, s - OVERLAP), min(n, s + CHUNK + OVERLAP)
        d = vae.encode(x[..., lo:hi].to(vae.device, vae.dtype)).latent_dist
        a = (s - lo) // HOP
        keep = slice(a, None) if hi == n else slice(a, a + CHUNK // HOP)   # the last piece keeps the song's end
        means.append(d.mean[..., keep])
        stds.append(d.std[..., keep])
        if hi == n:
            break
        s += CHUNK
    return torch.cat(means, -1), torch.cat(stds, -1)


@torch.no_grad()
def encode(vae, audio, generator=None):
    """audio (2, n) float at 48 kHz -> latents (1, T, 64), sampled from the posterior as ACE-Step's trainer does."""
    mean, std = posterior(vae, audio)
    lat = mean + std * torch.randn(mean.shape, generator=generator).to(mean)
    return lat.transpose(1, 2).float()


@torch.no_grad()
def encode_mean(vae, audio):
    """audio (2, n) float at 48 kHz -> the posterior's mean as latents (1, T, 64)."""
    return posterior(vae, audio)[0].transpose(1, 2).float()


@torch.no_grad()
def decode(vae, latents):
    """latents (1, T, 64) -> audio (2, n) float at 48 kHz, clamped to [-1, 1]."""
    out = vae.decode(latents.transpose(1, 2).to(vae.device, vae.dtype)).sample
    return out[0].float().clamp(-1, 1).cpu()
