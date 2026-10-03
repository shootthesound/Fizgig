"""Krea 2 driver for Fizgig's standard layer (families/driver.py).

The original Krea 2 trainer's rules behind the FamilyDriver interface, so the generic cache / train / preview code
trains Krea 2 the way src/fizgig/krea2/trainer.py does:
* latents: Qwen-Image VAE (16 channels, 8x), normalised with its latents_mean / std, (16, h, w)
* conditioning: Qwen3-VL-4B multi-layer hidden stack + validity mask, padded to max_length
  -> {"hidden_states": (seq, layers, dim), "attention_mask": (seq,)}; valid tokens are gathered per forward
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0; t logit-normal with the resolution-
  dependent shift exp(mu), mu linear in the image-token count (0.5 at 256 to 1.15 at 6400); unweighted MSE
* sampling: Euler on the same schedule (sampling.timesteps), mu pinned at 1.15 for the turbo previews
* LoRA: every Linear in the DiT (264: the 28 blocks, the text-fusion stack, the input / timestep / output layers);
  quantisation (INT8 / NF4) covers the blocks' Linears only, as the original's fp8 / int8 / NF4 paths
"""
import re

import numpy as np

import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

DTYPE = torch.bfloat16
_BLOCK_MODULES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
_IO = ("first", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3", "tproj.1", "last.linear")


def _mu(num_img_tokens):
    """musubi's get_lin_function through (256, 0.5) and (6400, 1.15): the flow shift's log for this token count."""
    m = (1.15 - 0.5) / (6400 - 256)
    return m * num_img_tokens + (0.5 - m * 256)


class Krea2Driver(FamilyDriver):

    # INT8 scales computed in bf16, exactly as the original trainer's apply_int8_training quantises Krea 2 - so a
    # driver run starts from the same INT8 weights as a Krea 2 run.
    int8_fp32_scales = False

    def ft_spec(self, dit):
        # the original's component windows (balanced: attention ~30% of a block, the MLP split in its three
        # matrices), text fusion trained throughout, and its measured NF4-trunk planner calibration
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="blocks", components=("attn", "mlp.gate", "mlp.up", "mlp.down"),
                      always_on=("txtfusion",), overhead_gb=9.5, trunk_gb_per_block=0.217,
                      stream_base_gb=3.2,      # measured 30 Sep: 2.7 GB worst window (12 GB card) + 0.5 slack
                      calib_mp=0.25, act_gb_per_mp=3.2)   # 0.25 -> 0.98 MP measured +2.2 GB (3.0 GB/MP)

    def compile_targets(self, dit):
        return dit.blocks

    def compile_plan(self, mode, total_steps, precision, blocks_to_swap, mp=0.25):
        """Krea 2's own measured rule (utils/capabilities: should_compile / compile_boundary) - its compiled-memory
        figures place the checkpoint inside the graph where it fits and outside where it does not."""
        from fizgig.utils.capabilities import compile_boundary, should_compile
        q4, q8 = precision == "nf4", ("int8" if precision == "int8" else "")
        if mode == "auto":
            return should_compile(total_steps, q4, q8, blocks_to_swap, mp=mp)
        if mode == "outside":
            return "outside", ""
        b = compile_boundary(q4, q8, mp=mp)
        return b, ("on: inside-the-graph won't fit at this token load - compiling with the checkpoint OUTSIDE the "
                   "region instead." if b == "outside" else "")

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.krea2.utils import load_krea2_dit
        dit = load_krea2_dit(path, device=device, dtype=DTYPE, fp8_scaled=False, loading_device=device)
        return dit.eval().requires_grad_(False)

    def on_base_loaded(self, dit, precision, device):
        """RDNA2 (gfx103*) only: bind the grouped attention and FP32 NF4 GEMMs to this model instance. Every other
        card returns at the ROCm build check, before any device query; the shared attention, NF4 and SDPA code is
        untouched."""
        from fizgig.modules.rdna2_linear import is_rdna2_device
        if not is_rdna2_device(device):
            return
        import contextlib
        import logging
        import os
        log = logging.getLogger(__name__)
        if precision == "nf4" and os.environ.get("FIZGIG_RDNA2_LINEAR", "1") != "0":
            from fizgig.modules.rdna2_linear import install_nf4_forward
            log.info("[rdna2] installed FP32 frozen NF4 GEMMs on %d Linears", install_nf4_forward(dit))
        if os.environ.get("FIZGIG_RDNA2_ATTENTION", "1") != "0":
            from fizgig.modules.rdna2_attention import install_attention
            log.info("[rdna2] installed grouped attention on %d Krea 2 modules", install_attention(dit))
        # This RDNA2 process uses the math SDPA backend. Priming it here avoids the NVIDIA-only cuDNN probe later.
        from fizgig.modules import sdpa as _sdpa
        _sdpa._SDPA_CTX = contextlib.nullcontext

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.blocks) if dit is not None else self.description.n_blocks) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(num_blocks, BlockSwapConfig(torch.device(device), supports_backward=supports_backward))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from fizgig.krea2.vae_loader import load_vae
        vae = load_vae(path, input_channels=3, device=device, disable_mmap=True)
        return vae.to(device)

    def load_text_encoder(self, path, device):
        from fizgig.krea2.utils import load_krea2_text_encoder
        return load_krea2_text_encoder(path, dtype=DTYPE, device=device)

    def unload_text_encoder(self, te):
        try:
            te.to("cpu")
        except Exception:
            pass
        del te
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding (the original's caching.py, per item) -----------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).unsqueeze(2) / 127.5 - 1.0                      # (B, C, 1, H, W)
        z = vae.encode_pixels_to_latents(x.to(vae.device, dtype=vae.dtype))
        return [(zi.squeeze(1) if zi.dim() == 4 else zi).cpu() for zi in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        """One caption per forward, as the original cache script: a batched forward rounds a bf16 step or two
        differently, so a caption's conditioning would depend on which captions shared its batch."""
        out = []
        for cap in captions:
            hiddens, mask = te([cap])                                            # (1, seq, L, D), (1, seq)
            out.append({"hidden_states": hiddens[0].cpu(), "attention_mask": mask[0].cpu().to(torch.bool)})
        return out

    def encode_text_with_image(self, te, captions, image, megapixels=1.0):
        """The original's preview reference: the image through Qwen3-VL's vision path (1 MP for training previews;
        the workbench passes its MP setting), one caption per forward."""
        out = []
        for cap in captions:
            hiddens, mask = te([cap], images=[[image]], vision_megapixels=float(megapixels))
            out.append({"hidden_states": hiddens[0].cpu(), "attention_mask": mask[0].cpu().to(torch.bool)})
        return out

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(num_img_tokens, generator, min_t=0.0, max_t=1.0):
        """sample_krea2_timesteps: logit-normal base, shift exp(mu), rescaled INTO the window (never clamped)."""
        shift = float(np.exp(_mu(num_img_tokens)))
        t = torch.randn(1, generator=generator).sigmoid()
        t = (t * shift) / (1.0 + (shift - 1.0) * t)
        if min_t > 0.0 or max_t < 1.0:
            lo, hi = max(0.0, float(min_t)), min(1.0, float(max_t))
            t = lo + t * max(hi - lo, 1e-6)
        return t

    @staticmethod
    def _text(cond, device):
        from fizgig.krea2.sampling import gather_valid_text
        h, m = cond["hidden_states"], cond["attention_mask"]
        if h.dim() == 3:                      # one caption as encoded (previews); training batches carry a batch dim
            h, m = h[None], m[None]
        return gather_valid_text(h.to(device=device, dtype=DTYPE), m.to(device).bool())

    def _forward(self, dit, noised, t, cond):
        """The DiT's velocity for a noised latent (B, 16, h, w) at t (B,), image tokens only."""
        from fizgig.krea2.sampling import prepare
        device = noised.device
        txt, txtmask = self._text(cond, device)
        img, pos, mask = prepare(noised, txt.shape[1], dit.config.patch, txtmask)
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            return dit(img=img, context=txt, t=t.to(DTYPE), pos=pos, mask=mask)

    def loss_at(self, dit, latents, noise, t, cond, *, diff_ref=None, diff_weight=0.0):
        """The training loss for a given latent, noise and t - the original compute_loss's arithmetic, in the same
        dtype order. training_loss draws noise and t; tests call this directly to compare with the original."""
        from fizgig.krea2.sampling import patchify_block
        patch = dit.config.patch
        latent = latents.to(dtype=DTYPE)
        noise = noise.to(device=latent.device, dtype=DTYPE)
        t = t.to(latent.device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        noised = (1.0 - t_) * latent + t_ * noise
        target, _, _ = patchify_block(noise - latent, patch)
        pred = self._forward(dit, noised, t, cond)
        if diff_ref is not None and diff_weight > 0.0:
            d, _, _ = patchify_block((latent - diff_ref.to(device=latent.device, dtype=DTYPE)).abs(), patch)
            d = d.float().mean(dim=-1)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
            se = (pred.float() - target.float()).pow(2).mean(dim=-1)
            return (se * w).mean()
        return F.mse_loss(pred.float(), target.float())

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        if refs:
            raise RuntimeError("Krea 2 has no edit training (its DiT takes no reference latents)")
        patch = dit.config.patch
        n_tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
        noise = torch.randn(latents.shape, generator=generator)
        t = self._sample_t(n_tokens, generator, min_t, max_t)
        loss = self.loss_at(dit, latents, noise, t, cond, diff_ref=diff_ref, diff_weight=diff_weight)
        return loss, {"t": float(t.mean())}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        patch = 2
        n_tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
        noise = torch.randn(latents.shape, generator=generator).to(latents.device, DTYPE)
        t = self._sample_t(n_tokens, generator, min_t, max_t).to(latents.device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        return {"xt": (1.0 - t_) * latents.to(DTYPE) + t_ * noise, "t": t}

    def predict(self, dit, state, cond):
        return self._forward(dit, state["xt"], state["t"], cond)

    # ---- sampling (sampling.sample, split at the decode) ----------------------------------------
    @staticmethod
    def _grid(width, height):
        from fizgig.krea2.sampling import roundup
        align = 16                                                      # VAE 8 x patch 2
        return roundup(width, align, "width"), roundup(height, align, "height")

    def pad_conditioning(self, conds):
        """Prompt travel: every Krea 2 prompt is already the encoder's fixed 512 tokens with a validity mask, so the
        shapes match as they are; blending the masks (as weights) fades in the tokens only one prompt has."""
        return [dict(c) for c in conds]

    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        width, height = self._grid(width, height)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=dev).manual_seed(int(seed))
        return torch.randn(1, 16, height // 8, width // 8, device=dev, dtype=DTYPE, generator=g).float().cpu()

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        from fizgig.krea2.sampling import prepare, timesteps
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        width, height = self._grid(width, height)
        opts = dict(options)
        patch = dit.config.patch
        txt, txtmask = self._text(cond, device)
        guided = cfg > 1.0 and neg_cond is not None
        if guided:
            untxt, untxtmask = self._text(neg_cond, device)
        x0 = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device, DTYPE)
        img, pos, mask = prepare(x0, txt.shape[1], patch, txtmask)
        if guided:
            _, unpos, unmask = prepare(x0, untxt.shape[1], patch, untxtmask)
        x1, x2 = (256 // 16) ** 2, (1280 // 16) ** 2
        ts = list(sigmas) + [0.0] if sigmas is not None and len(sigmas) == steps else \
            timesteps(img.shape[1], steps, x1, x2, mu=opts.get("mu"))
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
                if on_step is not None:
                    on_step(i, len(ts) - 1)
                t = torch.full((1,), tc, dtype=img.dtype, device=device)
                v = dit(img=img, context=txt, t=t, pos=pos, mask=mask)
                if guided:
                    u = dit(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                    v = u + cfg * (v - u)
                img = img + (tp - tc) * v
        from einops import rearrange
        return rearrange(img, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)", ph=patch, pw=patch,
                         h=height // 16, w=width // 16)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        dev = next(vae.parameters()).device
        pixels = vae.decode_to_pixels(latents.to(dev, torch.bfloat16))                # [0, 1], (B, C, H, W)
        arr = (pixels[0].float().permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
        return Image.fromarray(arr)

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        blocks = [Block(f"block_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                  for i in range(self.description.n_blocks)]
        # the text-fusion stack as the original's four sliders (its layerwise and refiner blocks); its projector
        # rides with the input / output layers
        fusion = [Block(f"txt_{short}_{i}", f"Text fusion {name} {i}",
                        keep([f"txtfusion.{stack}_blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                  for short, name, stack in (("lw", "layerwise", "layerwise"), ("rf", "refiner", "refiner"))
                  for i in range(2)]
        return [BlockGroup("Blocks", blocks),
                BlockGroup("Text fusion", fusion),
                BlockGroup("Input and output", [Block("io", "Input, timestep and output layers",
                                                      keep(list(_IO) + ["txtfusion.projector"]))])]

    # diffusers naming (OneTrainer, AI-Toolkit, diffusers exports) -> Krea 2's own, on flattened names; the same
    # table the original Krea 2 loader used (networks.lora._KREA2_DIFFUSERS_RENAMES, issue #50)
    _ALIASES = ((r"^transformer_blocks_(\d+)_", r"blocks_\1_"), (r"^text_fusion_", "txtfusion_"),
                (r"_attn_to_out_0$", "_attn_wo"), (r"_attn_to_gate$", "_attn_gate"), (r"_attn_to_q$", "_attn_wq"),
                (r"_attn_to_k$", "_attn_wk"), (r"_attn_to_v$", "_attn_wv"), (r"_ff_up$", "_mlp_up"),
                (r"_ff_gate$", "_mlp_gate"), (r"_ff_down$", "_mlp_down"), (r"^img_in$", "first"),
                (r"^final_layer_linear$", "last_linear"), (r"^txt_in_linear_1$", "txtmlp_1"),
                (r"^txt_in_linear_2$", "txtmlp_3"), (r"^time_embed_linear_1$", "tmlp_0"),
                (r"^time_embed_linear_2$", "tmlp_2"), (r"^time_mod_proj$", "tproj_1"))

    def alias_flat(self, flat):
        out = flat
        for pat, new in self._ALIASES:
            out = re.sub(pat, new, out)
        return out if out != flat else None

    def quant_target_names(self, dit):
        """The blocks' Linears only: the text-fusion stack and the I/O layers stay bf16, as in the original."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]
