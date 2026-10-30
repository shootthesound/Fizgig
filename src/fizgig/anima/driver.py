"""Anima driver for Fizgig's standard layer (families/driver.py) - CircleStone Labs' Anima (anime / illustration),
a Cosmos-Predict2 2B DiT with an LLM adapter, a Qwen3 0.6B text encoder and the Qwen-Image VAE.

* model: anima/model.py (sd-scripts' Apache-2.0 port of NVIDIA's MiniTrainDIT), loaded from ComfyUI's single file
  (keys "net." or "model.diffusion_model."), the LLM adapter inside it
* conditioning: Qwen3 0.6B's last hidden state (final norm applied) for the raw caption, no template or BOS/EOS
  -> {"qwen": (Lq, 1024)}, plus the same caption's T5 token ids (with EOS) -> {"t5_ids": (Lt,)}. The adapter turns
  them into the DiT's cross-attention context inside every forward (so a LoRA on the adapter, e.g. the Turbo LoRA,
  applies), zero-padded to 512 tokens - ComfyUI's path exactly (comfy/ldm/anima/model.py preprocess_text_embeds)
* latents: Qwen-Image VAE (16 channels, 8x), per-channel mean / std (the Krea 2 VAE code), (16, h, w)
* training: rectified flow, x_t = (1 - t) x0 + t noise, target noise - x0, t logit-normal (diffusion-pipe's default),
  unweighted MSE
* sampling: Euler on the flow schedule with ComfyUI's shift 3 (sigma = 3t / (1 + 2t)), CFG as two passes
* LoRA: each block's self- and cross-attention and MLP Linears; kohya keys (lora_unet_blocks_0_self_attn_q_proj),
  which ComfyUI maps to diffusion_model.blocks.0.self_attn.q_proj
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import FamilyDriver, prior_base, weighted_loss, with_prior

DTYPE = torch.bfloat16
HELPER = "circlestone-labs/Anima-Base-v1.0-Diffusers"   # tokenizers + the Qwen3 config (helper_files)
QWEN_PAD = 151643
T5_EOS = 1
CONTEXT_LEN = 512
SHIFT = 3.0

# Qwen3 0.6B as Anima uses it (Anima-Base-v1.0-Diffusers text_encoder/config.json), spelled out: that file stores
# rope_theta under rope_parameters (transformers 5), which older transformers ignore and fall back to 10000 - a
# different text encoding from position 1 on
QWEN3_06B = dict(vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_act="silu",
                 max_position_embeddings=32768, rms_norm_eps=1e-6, rope_theta=1000000.0, attention_bias=False,
                 tie_word_embeddings=False, use_sliding_window=False)

DIT_CONFIG = dict(
    max_img_h=512, max_img_w=512, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    model_channels=2048, concat_padding_mask=True, crossattn_emb_channels=1024, pos_emb_cls="rope3d",
    pos_emb_learnable=True, pos_emb_interpolation="crop", min_fps=1, max_fps=30, use_adaln_lora=True,
    adaln_lora_dim=256, num_blocks=28, num_heads=16, extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False, use_llm_adapter=True)


def _strip(key):
    for p in ("net.", "model.diffusion_model.", "diffusion_model."):
        if key.startswith(p):
            return key[len(p):]
    return key


def _batched(key, v):
    """qwen (Lq, 1024) -> (1, Lq, 1024); t5_ids (Lt,) -> (1, Lt)."""
    return v if v.dim() == (3 if key == "qwen" else 2) else v[None]


class _TextEncoder:
    """Qwen3 0.6B (base) from its single file, with the Qwen and T5 tokenizers."""

    def __init__(self, path, device):
        from safetensors.torch import load_file
        from fizgig.utils import hf_public
        from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
        self.tok = AutoTokenizer.from_pretrained(HELPER, subfolder="tokenizer")
        # the T5 vocabulary straight from its tokenizer.json (its post-processor appends </s>): the repo's
        # tokenizer_config.json is written by a newer transformers than Fizgig pins and does not load in it
        self.t5 = PreTrainedTokenizerFast(tokenizer_file=hf_public.download(
            repo_id=HELPER, filename="t5_tokenizer/tokenizer.json"),
                                          eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
        model = Qwen3Model(Qwen3Config(**QWEN3_06B))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or [k for k in unexpected if not k.startswith("lm_head")]:
            raise RuntimeError(f"Qwen3 0.6B file does not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        # fp32, as ComfyUI runs it: in bf16 the first token (Qwen's very large activation) is 16% off and the rest ~1%
        self.model = model.to(device, torch.float32).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, captions):
        out = []
        for cap in captions:
            q = self.tok(cap, add_special_tokens=False, truncation=True, max_length=CONTEXT_LEN).input_ids or [QWEN_PAD]
            h = self.model(input_ids=torch.tensor([q], device=self.device)).last_hidden_state[0]
            t5 = self.t5(cap, truncation=True, max_length=CONTEXT_LEN).input_ids or [T5_EOS]
            out.append({"qwen": h.to(DTYPE).cpu(), "t5_ids": torch.tensor(t5, dtype=torch.int64)})
        return out

    def unload(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AnimaDriver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from fizgig.anima.model import Anima
        with init_empty_weights():
            dit = Anima(**DIT_CONFIG)
        sd = {_strip(k): v.to(DTYPE) for k, v in load_file(path).items()}
        missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
        missing = [k for k in missing if not any(b in k for b in ("seq", "dim_spatial_range", "dim_temporal_range",
                                                                  "inv_freq"))]
        if missing or unexpected:
            raise RuntimeError(f"not an Anima DiT file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return dit.to(device).eval().requires_grad_(False)

    def load_vae(self, path, device):
        from fizgig.krea2.vae_loader import load_vae
        return load_vae(path, input_channels=3, device=device, disable_mmap=True).to(device)

    def load_text_encoder(self, path, device):
        return _TextEncoder(path, device)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).unsqueeze(2) / 127.5 - 1.0                      # (B, C, 1, H, W)
        z = vae.encode_pixels_to_latents(x.to(vae.device, dtype=vae.dtype))
        return [(zi.squeeze(1) if zi.dim() == 4 else zi).cpu() for zi in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return te.encode(captions)

    # ---- the model call -------------------------------------------------------------------------
    @staticmethod
    def _context(dit, cond, device):
        qwen = _batched("qwen", cond["qwen"]).to(device, DTYPE)
        ids = _batched("t5_ids", cond["t5_ids"]).to(device).long()
        ctx = dit.llm_adapter(qwen, ids)
        if ctx.shape[1] < CONTEXT_LEN:
            ctx = F.pad(ctx, (0, 0, 0, CONTEXT_LEN - ctx.shape[1]))
        return ctx

    def _velocity(self, dit, x, t, cond):
        """x (B, 16, 1, h, w), t (B,) in [0, 1] -> the predicted velocity (noise - x0), (B, 16, 1, h, w)."""
        ctx = self._context(dit, cond, x.device)
        if ctx.shape[0] != x.shape[0]:
            ctx = ctx.expand(x.shape[0], -1, -1)
        mask = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=DTYPE)
        return dit.forward_mini_train_dit(x.to(DTYPE), t.to(x.device, DTYPE)[:, None], ctx, padding_mask=mask)

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        x0 = latents.float()
        if x0.dim() == 4:
            x0 = x0.unsqueeze(2)                                                    # (B, 16, 1, h, w)
        n = x0.shape[0]
        t = torch.sigmoid(torch.randn(n, generator=generator))
        t = (min_t + (max_t - min_t) * t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        tb = t.view(-1, 1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * noise
        base = prior_base(prior, loss_mask, lambda: self._velocity(dit, xt, t, cond))
        pred = self._velocity(dit, xt, t, cond).float()
        w = None
        if diff_ref is not None and diff_weight > 0.0:
            # slider pairs: positions where the two poles differ count more (Krea 2's formula, as Qwen and SDXL)
            ref = diff_ref.to(x0.device).float()
            ref = ref.unsqueeze(2) if ref.dim() == 4 else ref
            d = (x0 - ref).abs().mean(dim=1).flatten(1)                                 # (B, h*w)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
        mw = None if loss_mask is None else loss_mask.float().to(x0.device).flatten(1)        # (B, h*w), one frame
        if w is None and mw is None:
            return F.mse_loss(pred, noise - x0), {"t": float(t.mean())}
        loss = weighted_loss((pred - (noise - x0)).pow(2).mean(dim=1).flatten(1), w, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred - base).pow(2).mean(dim=1).flatten(1), mw, w)
        return loss, {"t": float(t.mean())}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = latents.float()
        if x0.dim() == 4:
            x0 = x0.unsqueeze(2)
        t = torch.sigmoid(torch.randn(1, generator=generator))
        t = (min_t + (max_t - min_t) * t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        tb = t.view(-1, 1, 1, 1, 1)
        return {"xt": (1 - tb) * x0 + tb * noise, "t": t}

    def predict(self, dit, state, cond):
        return self._velocity(dit, state["xt"], state["t"], cond)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        g = torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn((1, 16, 1, height // 8, width // 8), generator=g, dtype=torch.float32)

    @staticmethod
    def _sigmas(steps, shift):
        t = torch.linspace(1.0, 0.0, int(steps) + 1)
        return shift * t / (1 + (shift - 1) * t)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, **_ignored):
        device = next(dit.parameters()).device
        opts = dict(options or ())
        if sigmas is not None and len(sigmas) == int(steps) + 1:
            sig = torch.tensor([float(s) for s in sigmas])
        elif sigmas is not None and len(sigmas) == int(steps):
            sig = torch.tensor([float(s) for s in sigmas] + [0.0])
        else:
            sig = self._sigmas(steps, float(opts.get("shift", SHIFT)))
        x = (self.initial_noise(seed, width, height) if noise is None else noise).to(device).float()
        if cfg > 1.0 and neg_cond is None:
            # no negative encoded: the dropped-caption conditioning (no Qwen signal, T5 EOS only)
            neg_cond = {"qwen": torch.zeros(1, 1024), "t5_ids": torch.tensor([T5_EOS])}
        for i in range(len(sig) - 1):
            if on_step is not None:
                on_step(i, len(sig) - 1)
            s, s_next = float(sig[i]), float(sig[i + 1])
            ts = torch.full((1,), s, device=device)
            v = self._velocity(dit, x, ts, cond).float()
            if cfg > 1.0:
                u = self._velocity(dit, x, ts, neg_cond).float()
                v = u + cfg * (v - u)
            x = x + (s_next - s) * v
        return x

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        dev = next(vae.parameters()).device
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        pixels = vae.decode_to_pixels(latents.to(dev, torch.bfloat16))                # [0, 1], (B, C, H, W)
        arr = (pixels[0].float().permute(1, 2, 0).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
        return Image.fromarray(arr)

    def ft_spec(self, dit):
        """Fine-tuning: the 28 blocks' attention and MLP Linears (the NF4 trunk's), in four windows of similar size per
        block (self-attention ~17M parameters, cross-attention ~13M, each MLP matrix ~17M). The LLM adapter is never
        trained (the model card: it degrades easily), nor are the modulation layers. The file names every DiT weight
        under "net."; checkpoints keep that layout, so ComfyUI loads them as it loads the base."""
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="blocks", components=("self_attn", "cross_attn", "mlp.layer1", "mlp.layer2"),
                      file_prefix="net.",
                      # measured 5 Oct on a 5090: window peaks 2.7-3.7 GB at 0.25 MP, 4.0-4.3 GB at 1 MP (1.92 GB
                      # allocated at planning, 0.93 GB NF4 trunk) -> resident base 2.4 GB at 1 MP (+0.1; the planner
                      # keeps 1.5 GB besides); the attention windows grew 1.7 GB/MP from 0.25 to 1 MP. Calibrated at
                      # 1 MP, so smaller runs plan with ~0.6 GB to spare. A simulated 8 GB card plans the 4 full windows
                      overhead_gb=2.5, calib_mp=1.0, act_gb_per_mp=2.0)

    def compile_blocks(self, dit, boundary="inside", blocks_to_swap=0):
        """torch.compile the 28 blocks in place, after the LoRA has wrapped its Linears. Each block does its own
        gradient checkpoint, so the checkpoint compiles inside the graph. A real 1 MP run on a 5090 went 0.71 ->
        0.31 s/step with fused AdamW (peak 5.7 -> 6.1 GB)."""
        import logging
        from fizgig.families.compile import ready_to_compile
        if not ready_to_compile(blocks_to_swap):
            return
        for i, blk in enumerate(dit.blocks):
            dit.blocks[i] = torch.compile(blk, fullgraph=False)
        logging.getLogger(__name__).info("[compile] %d Anima blocks compiled - the first step of each new shape "
                                         "pauses to compile", len(dit.blocks))

    # ---- LoRA -----------------------------------------------------------------------------------
    def alias_flat(self, flat):
        """diffusers naming (AI-Toolkit's Anima files before its ComfyUI rename): transformer_blocks_N_attn1_to_q ->
        blocks_N_self_attn_q_proj."""
        import re
        m = re.match(r"^transformer_blocks_(\d+)_(attn1|attn2|ff)_(.+)$", flat)
        if not m:
            return None
        blk, part, leaf = m.groups()
        if part == "ff":
            leaf = {"net_0_proj": "layer1", "net_2": "layer2"}.get(leaf)
            return f"blocks_{blk}_mlp_{leaf}" if leaf else None
        leaf = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out_0": "output_proj"}.get(leaf)
        return f"blocks_{blk}_{'self_attn' if part == 'attn1' else 'cross_attn'}_{leaf}" if leaf else None
