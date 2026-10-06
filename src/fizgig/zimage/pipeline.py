# Adapted from the Z-Image reference pipeline (https://github.com/Tongyi-MAI/Z-Image, src/zimage/pipeline.py),
# Copyright 2025 Alibaba Z-Image Team, licensed under the Apache License, Version 2.0.
# Changes for Fizgig: the text encoder from ComfyUI's single Qwen3-4B file, the FLUX.1 VAE through diffusers, a plain
# Euler loop over the reference shift-3 schedule, latents in and out as tensors.
"""Z-Image text encoding, sampling and the VAE.

* text: Qwen3-4B on the chat-templated prompt (thinking on, generation prompt added), hidden_states[-2] with no final
  norm, real tokens only, at most 512
* sampling: Euler on sigma = shift * s / (1 + (shift - 1) s), s from linspace(1, 0, steps + 1) (shift 3: Turbo's 8
  steps are the reference 9-step schedule without its skipped final sigma 0); the model runs at t = 1 - sigma and its
  output is negated into the velocity; CFG as the reference: pos + scale * (pos - neg)
* VAE: the FLUX.1 autoencoder, latents = (z - 0.1159) * 0.3611
"""
import torch

HELPER = "Tongyi-MAI/Z-Image-Turbo"               # tokenizer (helper_files)
MAX_TOKENS = 512
SCALING, SHIFT_FACTOR = 0.3611, 0.1159
DEFAULT_SHIFT = 3.0                              # scheduler_config.json shift 3.0, use_dynamic_shifting false
QWEN3_4B = dict(vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128, hidden_act="silu",
                max_position_embeddings=40960, rms_norm_eps=1e-6, rope_theta=1000000.0, attention_bias=False,
                tie_word_embeddings=True, use_sliding_window=False)


class TextEncoder:
    def __init__(self, path, device, dtype=torch.bfloat16, tokenizer=None):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        self.tok = AutoTokenizer.from_pretrained(tokenizer or HELPER, subfolder=None if tokenizer else "tokenizer")
        model = Qwen3Model(Qwen3Config(**QWEN3_4B))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or [k for k in unexpected if not k.startswith("lm_head")]:
            raise RuntimeError(f"not Z-Image's Qwen3-4B file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model = model.to(device, dtype).eval().requires_grad_(False)
        self.device = device

    def template(self, prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)

    @torch.no_grad()
    def encode(self, prompts):
        """prompts -> list of (L, 2560) on CPU, real tokens only."""
        out = []
        for p in prompts:
            t = self.tok([self.template(p)], padding="max_length", max_length=MAX_TOKENS, truncation=True,
                         return_tensors="pt")
            ids, mask = t.input_ids.to(self.device), t.attention_mask.to(self.device).bool()
            h = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
            out.append(h[0][mask[0]].cpu())
        return out

    def unload(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_vae(path, device, dtype=torch.float32, config=HELPER):
    """The FLUX.1 autoencoder from its single file (LDM key layout), with the reference VAE config."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_single_file(path, config=config, subfolder="vae", torch_dtype=dtype)
    return vae.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_image(vae, pixels):
    """pixels (B, 3, H, W) in [-1, 1] -> latents (B, 16, H/8, W/8), the posterior mean."""
    z = vae.encode(pixels.to(vae.device, vae.dtype)).latent_dist.mean
    return ((z - SHIFT_FACTOR) * SCALING).float()


@torch.no_grad()
def decode_latents(vae, latents):
    """latents (B, 16, h, w) -> pixels (B, 3, H, W) in [-1, 1]."""
    return vae.decode((latents.to(vae.device, vae.dtype) / SCALING) + SHIFT_FACTOR).sample.float().clamp(-1, 1)


def sigmas(steps, shift=DEFAULT_SHIFT):
    s = torch.linspace(1.0, 0.0, int(steps) + 1)
    return shift * s / (1 + (shift - 1) * s)


def velocity(dit, x, sigma, cap):
    """x (B, 16, h, w) at noise level sigma (B,), cap: list of (L, 2560) -> the flow velocity noise - x0."""
    t = (1.0 - sigma).to(x.device, torch.float32)
    out = dit([xi.unsqueeze(1) for xi in x.to(next(dit.parameters()).dtype)], t, [c.to(x.device) for c in cap])
    return -torch.stack([o.float() for o in out]).squeeze(2)


@torch.no_grad()
def sample(dit, cap, height, width, *, steps=8, seed=0, shift=3.0, cfg=0.0, neg=None, noise=None, on_step=None):
    """One image's latents: Euler over the reference schedule. cfg > 1 with `neg` (list of one (L, 2560)) applies the
    reference guidance; Turbo samples at cfg 0."""
    device = next(dit.parameters()).device
    if noise is None:
        g = torch.Generator("cpu").manual_seed(int(seed))
        noise = torch.randn((1, 16, height // 8, width // 8), generator=g, dtype=torch.float32)
    x = noise.to(device)
    sig = sigmas(steps, shift)
    for i in range(len(sig) - 1):
        if on_step is not None:
            on_step(i, len(sig) - 1)
        s = torch.full((1,), float(sig[i]))
        v = velocity(dit, x, s, cap)
        if cfg > 1.0 and neg is not None:
            u = velocity(dit, x, s, neg)
            v = v + cfg * (v - u)
        x = x + (float(sig[i + 1]) - float(sig[i])) * v
    return x
