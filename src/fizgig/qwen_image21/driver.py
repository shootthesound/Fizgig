"""Qwen Image 2.1 driver for Fizgig's standard layer (families/driver.py).

Everything Qwen-specific the generic cache/train/preview code needs, behind the FamilyDriver interface:
* conditioning: Qwen3-VL-8B last layer before the final RMSNorm, system tokens dropped -> {"hidden_states": (L, 4096)}
* latents: RGBA VAE (RGB gets an opaque alpha), normalised with the checkpoint's latents_mean/std, (64, h, w) at /16
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0, logit-normal t with the checkpoint's
  resolution-dependent exponential shift (the sampler's mu); block-causal joint forward, keep the image tokens
* sampling: FlowMatch Euler with the reference schedule (sampling.py)
* edit: reference images go into the text encoder's template as vision tokens (cond["ref_mask"] marks them) and
  into the DiT sequence as clean latents ahead of the target; the checkpoint is one model for both (up to 10 refs)
"""
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import FamilyDriver
from fizgig.qwen_image21 import sampling as S

logger = logging.getLogger(__name__)


class QwenImage21Driver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.qwen_image21.model import load_qwen21_dit
        return load_qwen21_dit(path, device=device).eval().requires_grad_(False)

    def max_blocks_to_swap(self, dit=None):
        return len(dit.transformer_blocks) - 2 if dit is not None else self.description.n_blocks - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    # ---- training previews on the Turbo checkpoint (Klein's Distilled handoff) ----------------------------------
    def park_for_preview(self, dit, device):
        """Stream all but two of the training model's blocks so the Turbo fits beside it; an NF4 base cannot stream and
        is small - left as it is (token None)."""
        if getattr(dit, "_nf4_quantized", False):
            return None
        orig = int(dit.blocks_to_swap or 0)
        dit.enable_block_swap(len(dit.transformer_blocks) - 2, torch.device(device), True)
        dit.prepare_block_swap_before_forward()
        return orig

    def load_preview_checkpoint(self, path, device, int8=False):
        """The Turbo DiT for this preview round, sized to the VRAM left beside the parked training model: bf16 when it
        fits with room for the render, else INT8, else INT8 with blocks streamed (forward-only)."""
        from fizgig.families import quant
        torch.cuda.empty_cache()
        free = quant.free_vram_gb()       # FIZGIG_SIM_VRAM_GB: the simulated card's
        size = os.path.getsize(path) / 1024 ** 3
        room = 4.0                          # 1024 px render + VAE decode
        precision = "int8" if int8 or free < size + room else "bf16"
        need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2) / max(1, self.description.n_blocks)       # INT8 GB per block
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        if swapped:
            self.block_swap_mode(m, inference=True)
        logger.info(f"[sample] Turbo checkpoint: {precision}, {swapped} blocks streamed ({free:.1f} GB free)")
        return m.eval().requires_grad_(False), swapped

    def unpark_after_preview(self, dit, device, token):
        """The run's own swap back, or with none the sample-time offloader torn down and the model back on the GPU."""
        if token is None:
            return
        device = torch.device(device)
        if token > 0:
            dit.enable_block_swap(token, device, True)
            dit.move_to_device_except_swap_blocks(device)
        else:
            from fizgig.families import quant
            dit.disable_block_swap()
            quant.move(dit, device)
        dit.switch_block_swap_for_training()

    def ft_spec(self, dit):
        # four balanced windows per block (attention ~0.13 GB, each MLP matrix ~0.10); the file fuses the MLP's gate
        # and projection into one gate_up tensor, [gate_layer; proj]
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="transformer_blocks", components=("attn", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"),
                      file_layout=(("img_mlp.gate_layer.weight", "img_mlp.gate_up.weight", 0, 2),
                                   ("img_mlp.proj.weight", "img_mlp.gate_up.weight", 1, 2)),
                      # measured 30 Sep on a 5090 at 0.5 MP: resident base 5.06 GB (MLP windows), streaming base 2.2;
                      # 0.5 -> 0.98 MP grew the peaks by up to +2.1 GB (~4 GB/MP)
                      overhead_gb=5.5, stream_base_gb=2.7, calib_mp=0.5, act_gb_per_mp=4.2)

    def load_vae(self, path, device):
        from fizgig.qwen_image21.vae import load_qwen21_vae
        return load_qwen21_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Text-only (no LM head or vision tower): bf16 when it fits the free VRAM with room to run, else INT8 -
        about 8 GB, which fits a 10 GB card."""
        from fizgig.qwen_image21.embedder import Qwen21TextEncoder
        from fizgig.families.quant import free_vram_gb
        return Qwen21TextEncoder(path, device=device, int8=free_vram_gb() < 19.5)

    supports_references = True

    def load_reference_text_encoder(self, path, device):
        """With its vision tower (about 1 GB more than load_text_encoder's)."""
        from fizgig.qwen_image21.embedder import Qwen21TextEncoder
        from fizgig.families.quant import free_vram_gb
        return Qwen21TextEncoder(path, device=device, int8=free_vram_gb() < 21.0, vision=True)

    def unload_text_encoder(self, te):
        te.unload()

    def compile_targets(self, dit):
        return dit.transformer_blocks

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        p = next(vae.parameters())
        return [z.to(torch.bfloat16).cpu() for z in S.encode_image(vae, x.to(p.device, p.dtype))]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in te.encode(captions)]

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        from PIL import Image
        out = []
        for cap, imgs in zip(captions, references):
            h, m = te.encode_with_references(cap, [Image.fromarray(np.ascontiguousarray(a[..., :3])) for a in imgs])
            out.append({"hidden_states": h.to(torch.bfloat16).cpu(), "ref_mask": m.cpu()})
        return out

    @staticmethod
    def _ref_sequence(refs, ref_mask, device):
        """Packed reference tokens + their img_shapes entries. Each vision token stands for 2x2 latent tokens, so the
        references' latents must match the size the text encoder saw them at."""
        refs = [r.float().to(device) for r in refs]
        n_ref = sum(r.shape[-2] * r.shape[-1] for r in refs)
        slots = int(ref_mask.sum()) if ref_mask is not None else 0
        if slots * 4 != n_ref:
            raise RuntimeError(f"reference size mismatch: the conditioning holds {slots} image tokens for "
                               f"{n_ref // 4} expected - re-cache the text (Target Megapixels or the before-images "
                               f"changed since it was cached)")
        return torch.cat([S.pack(r) for r in refs], dim=1), [(1, r.shape[-2], r.shape[-1]) for r in refs]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(n_tokens, generator, min_t, max_t):
        t = torch.sigmoid(torch.randn(1, generator=generator)).item()
        mu = S.calculate_mu(n_tokens)
        t = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
        return min_t + (max_t - min_t) * t

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        device = latents.device
        h, w = latents.shape[-2:]
        n = h * w
        x0 = S.pack(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(device)
        t = self._sample_t(n, generator, min_t, max_t)
        xt = (1 - t) * x0 + t * noise
        ref_mask = cond["ref_mask"][0] if "ref_mask" in cond else None
        enc, img_mask, enc_mask = S.model_inputs(cond["hidden_states"][0], n, device, ref_mask=ref_mask)
        shapes = [(1, h, w)]
        if refs:
            ref_tokens, ref_shapes = self._ref_sequence(refs, ref_mask, device)
            xt, shapes = torch.cat([ref_tokens, xt], dim=1), ref_shapes + shapes
        elif ref_mask is not None and ref_mask.any():
            raise RuntimeError("edit conditioning without its reference latents - re-cache the latents")
        tt = torch.tensor([t], device=device, dtype=torch.bfloat16)
        pred = dit(xt.to(torch.bfloat16), enc.to(torch.bfloat16), tt, [shapes], img_mask, enc_mask)[:, -n:]
        if diff_ref is not None and diff_weight > 0.0:
            # slider disentanglement: tokens where the two poles differ count more (Krea 2's formula)
            d = S.pack((latents.float() - diff_ref.to(device).float()).abs()).mean(dim=-1)     # (1, N)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
            se = (pred.float() - (noise - x0)).pow(2).mean(dim=-1)
            return (se * w).mean(), {"t": t}
        return F.mse_loss(pred.float(), noise - x0), {"t": t}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        h, w = latents.shape[-2:]
        x0 = S.pack(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        t = self._sample_t(h * w, generator, min_t, max_t)
        return {"xt": (1 - t) * x0 + t * noise, "t": t, "hw": (h, w)}

    def predict(self, dit, state, cond):
        h, w = state["hw"]
        n = h * w
        device = state["xt"].device
        enc, img_mask, enc_mask = S.model_inputs(cond["hidden_states"][0], n, device)
        tt = torch.tensor([state["t"]], device=device, dtype=torch.bfloat16)
        return dit(state["xt"].to(torch.bfloat16), enc.to(torch.bfloat16), tt, [[(1, h, w)]], img_mask,
                   enc_mask)[:, -n:]

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        neg = neg_cond["hidden_states"] if (neg_cond is not None and cfg > 1.0) else None
        neg_mask = neg_cond.get("mask") if neg is not None else None
        opts = dict(options)
        shift_terminal = opts.get("shift_terminal", S.SHIFT_TERMINAL)
        if sigmas is not None and len(sigmas) != steps:
            sigmas = None                   # an explicit schedule only applies at its own step count
        return S.sample(dit, cond["hidden_states"], height, width, steps=steps, seed=seed, cfg=cfg, neg_emb=neg,
                        device=device, sigmas=sigmas, shift_terminal=shift_terminal,
                        dynamic_shift=bool(opts.get("dynamic_shift", True)), noise=noise, on_step=on_step,
                        text_mask=cond.get("mask"), neg_mask=neg_mask, ref_latents=refs or None,
                        ref_mask=cond.get("ref_mask"), neg_ref_mask=neg_cond.get("ref_mask") if neg is not None else None)

    def pad_conditioning(self, conds):
        """Pad prompts to one length (zeros) with a mask so they can be blended (prompt travel). The DiT masks the
        padded keys and starts the image's positions after the real tokens, so a padded prompt renders as the
        unpadded one does."""
        L = max(c["hidden_states"].shape[0] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"]
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        img = S.decode(vae, latents, height, width)[0, :3].float().clamp(-1, 1)
        return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))
