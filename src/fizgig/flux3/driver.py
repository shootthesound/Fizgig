"""Flux 3 Image Klein driver for Fizgig's standard layer (families/driver.py).

* conditioning: Qwen3-VL-4B, layers 4..32 stacked -> {"tokens": (L, 20480)}, L padded to a multiple of 80 (embedder.py)
* latents: the FLUX.2 autoencoder, BatchNorm-normalised, 2x2-packed -> (128, h, w) at /16 (autoencoder.py)
* training: flow matching on the Base checkpoint, x_t = (1 - t) x0 + t noise, target = noise - x0 (the reference
  sampler integrates dx/dt = velocity from t = 1 to 0, so this is the velocity it was trained to predict)
* sampling: the reference's shifted schedule (alpha 6.93); Base with CFG against the empty prompt, the distilled
  checkpoint as a flow map without CFG (sampling.py)

Not built yet: sliders' pair weighting, fine-tune, compile.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver, picture_losses, prior_base, weighted_loss, with_prior
from fizgig.flux3 import sampling as S

logger = logging.getLogger(__name__)

N_MODE = 5            # model.py DEPTH: per-mode blocks for image, reference image and text
N_SINGLE = 28         # model.py DEPTH_SINGLE
BLOCK_MODULES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")


class Flux3Driver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.flux3.model import load_flux3_dit
        return load_flux3_dit(path, device=device).eval().requires_grad_(False)

    def load_vae(self, path, device):
        from fizgig.flux3.autoencoder import load_flux3_vae
        return load_flux3_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """bf16 (about 8 GB without the LM head and vision tower) when it fits the free VRAM with room to run, else
        INT8 weights (about 4.5 GB)."""
        from fizgig.families.quant import free_vram_gb
        from fizgig.flux3.embedder import Flux3TextEncoder
        return Flux3TextEncoder(path, device=device, int8=free_vram_gb() < 11.0)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    def compile_targets(self, dit):
        """The 28 single blocks (most of the compute; the 15 per-mode blocks see only their own tokens)."""
        return dit.single_blocks

    # ---- full fine-tune (families/ft.py) ---------------------------------------------------------
    def ft_spec(self, dit):
        # the per-mode blocks (image, reference, text) then the single blocks as one cycle; each block's six Linears
        # are the parts (the planner packs as many per window as the card holds). The embedders, modulations and final
        # layer stay frozen. The per-mode blocks run interleaved by depth, not as one ordered list, so frozen blocks
        # can't stream: plans keep the whole NF4 trunk resident (5.6B: ~3 GB)
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks=("content_mode_blocks.image", "content_mode_blocks.image_cond", "txt_mode_blocks",
                              "single_blocks"),
                      components=BLOCK_MODULES, streams=False,
                      # whole model in one window: 12.6 GB at 0.25 MP, 14.0 at 1.0 (RTX 5090, 15 Oct 2026); the
                      # default 4 GB/MP planned 15.4 at 1.0. 8.4 puts the 1.0 MP estimate 0.5 GB over
                      overhead_gb=8.4, calib_mp=1.0, act_gb_per_mp=1.9)

    # ---- block swap: the single blocks (model.py) -----------------------------------------------------------------
    def max_blocks_to_swap(self, dit=None):
        return (len(dit.single_blocks) if dit is not None else N_SINGLE) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    # ---- training previews on the distilled checkpoint (Klein's Distilled handoff; adapted from Qwen 2.1's) ----------
    def park_for_preview(self, dit, device):
        """Stream all but two of the training model's single blocks so the distilled fits beside it. An NF4 base
        cannot stream: left as it is (token None), or on a card that can't hold the INT8 distilled's resident part
        and a render beside it, parked on the CPU for the round (token 0; unpark moves it back). Measured on a
        simulated 10 GB card: left resident, the second preview ran out of memory."""
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.families import quant
            torch.cuda.empty_cache()
            if quant.free_vram_gb() >= 8.0:   # INT8 distilled ~5.5 GB + a 1024 px render and decode
                return None
            quant.move(dit, "cpu")
            torch.cuda.empty_cache()
            return 0
        orig = int(dit.blocks_to_swap or 0)
        dit.enable_block_swap(len(dit.single_blocks) - 2, torch.device(device), True)
        dit.prepare_block_swap_before_forward()
        return orig

    def load_preview_checkpoint(self, path, device, int8=False):
        """The distilled DiT for this preview round, sized to the VRAM left beside the parked training model: bf16 when
        it fits with room for the render, else INT8, else INT8 with blocks streamed (forward-only)."""
        import os
        from fizgig.families import quant
        torch.cuda.empty_cache()
        free = quant.free_vram_gb()       # FIZGIG_SIM_VRAM_GB: the simulated card's
        size = os.path.getsize(path) / 1024 ** 3
        room = 4.0                          # 1024 px render + VAE decode
        precision = "int8" if int8 or free < size + room else "bf16"
        need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2) / (N_SINGLE + 3 * N_MODE)                  # INT8 GB per block
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        if swapped:
            self.block_swap_mode(m, inference=True)
        logger.info(f"[sample] distilled checkpoint: {precision}, {swapped} blocks streamed ({free:.1f} GB free)")
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

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        p = next(vae.parameters())
        out = []
        for a in images:
            x = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float().div(127.5).sub(1.0)
            tokens, ids = vae.encode(x.to(p.device, p.dtype))
            h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
            out.append(S.to_latents(tokens, h, w).float().cpu())
        return out

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"tokens": t} for t in te.encode(captions)]

    # ---- edit training: references reach the DiT as latents (the image_cond stream), never the text encoder -------
    supports_references = True

    def load_reference_text_encoder(self, path, device):
        return self.load_text_encoder(path, device)

    def encode_text_with_references(self, te, captions, references):
        """The text encoder never sees the references (pipeline.py: they're VAE-encoded into the image_cond stream)."""
        return self.encode_text(te, captions)

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(generator, min_t, max_t):
        """Logit-normal, shifted by the sampler's alpha so training covers the times sampling spends its steps on.
        BFL hasn't published the training distribution: this is an assumption to verify (see README)."""
        t = torch.sigmoid(torch.randn(1, generator=generator)).item()
        return min_t + (max_t - min_t) * S.shift(t)

    def _forward(self, dit, xt, t, cond, h, w, refs=None):
        device = xt.device
        ref = None
        if refs:              # as sampling.sample: each reference's tokens, its l axis -(1 + i), at time 0
            toks = [S.to_tokens(r.float()).to(device, torch.bfloat16) for r in refs]
            rids = [S.image_ids(r.shape[-2], r.shape[-1], device, ref_index=i) for i, r in enumerate(refs)]
            ref = (torch.cat(toks), torch.cat(rids))
        packed = S.pack([[S.text_item(cond["tokens"], device), ref, (xt, S.image_ids(h, w, device))]])
        packed.timesteps["image"].fill_(t)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xt.is_cuda):
            if dit.flow_map:   # a distilled DiT predicts the jump to a target time; training uses the Base
                raise RuntimeError("Flux 3 trains on the Base checkpoint, not the distilled one")
            return dit(packed)

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        h, w = latents.shape[-2:]
        x0 = S.to_tokens(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        t = self._sample_t(generator, min_t, max_t)
        xt = (1 - t) * x0 + t * noise
        base = prior_base(prior, loss_mask, lambda: self._forward(dit, xt, t, cond, h, w, refs=refs))
        pred = self._forward(dit, xt, t, cond, h, w, refs=refs)
        target = noise - x0
        wgt = None
        if diff_ref is not None and diff_weight > 0.0:
            # image-pair sliders (Krea 2's weighting, on Flux 3's tokens): each token's error weighted by how much the
            # two poles differ there, so the slider learns what changes between them, not what they share
            d = (x0 - S.to_tokens(diff_ref.float().to(x0.device))).abs().mean(dim=-1)        # (h * w,)
            dm = d.mean().clamp_min(1e-8)
            r = (d / dm).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean().clamp_min(1e-8)
            if float(dm) <= 1e-6:
                wgt = torch.ones_like(wgt)                      # identical pair: uniform, never all-zero
        mw = None if loss_mask is None else S.to_tokens(loss_mask.float().to(x0.device)).mean(dim=-1)   # (h * w,)
        if wgt is None and mw is None:
            return F.mse_loss(pred.float(), target), {"t": t}
        loss = weighted_loss((pred.float() - target).pow(2).mean(dim=-1), wgt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred.float() - base).pow(2).mean(dim=-1), mw, wgt)
        return loss, {"t": t}

    def stack_conds(self, conds):
        """Captions of any length batch as they are: the DiT packs each picture as its own sample (its own text,
        image and attention bounds), so nothing is padded."""
        return {"items": conds}

    def training_loss_batch(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, loss_mask=None):
        """Batch Size above 1: each picture's noise and t drawn as training_loss draws them, picture by picture, then
        one forward over the pictures packed as samples (each its own text, image and timestep). The loss is the
        pictures' mean: a mask weighs each picture's tokens on their own."""
        h, w = latents.shape[-2:]
        device = latents.device
        n = latents.shape[0]
        x0s, xts, ts = [], [], []
        for i in range(n):
            x0 = S.to_tokens(latents[i].float())
            noise = torch.randn(x0.shape, generator=generator).to(device)
            t = self._sample_t(generator, min_t, max_t)
            x0s.append(x0)
            xts.append((1 - t) * x0 + t * noise)
            ts.append((t, noise))
        packed = S.pack([[S.text_item(c["tokens"], device), None, (xt, S.image_ids(h, w, device))]
                         for c, xt in zip(cond["items"], xts)])
        for (a, b), (t, _) in zip(packed.mode_indices["image"], ts):
            packed.timesteps["image"][a:b] = t
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latents.is_cuda):
            if dit.flow_map:
                raise RuntimeError("Flux 3 trains on the Base checkpoint, not the distilled one")
            pred = dit(packed)
        target = torch.cat([noise - x0 for x0, (_, noise) in zip(x0s, ts)])
        mw = None if loss_mask is None else \
            torch.stack([S.to_tokens(loss_mask[i].float().to(device)).mean(dim=-1) for i in range(n)])
        per = picture_losses((pred.float() - target).pow(2).mean(dim=-1).view(n, -1), mw)
        return per.mean(), {"t": sum(t for t, _ in ts) / n, "per": per, "ts": [float(t) for t, _ in ts]}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        h, w = latents.shape[-2:]
        x0 = S.to_tokens(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        t = self._sample_t(generator, min_t, max_t)
        return {"xt": (1 - t) * x0 + t * noise, "t": t, "hw": (h, w)}

    def predict(self, dit, state, cond):
        h, w = state["hw"]
        return self._forward(dit, state["xt"], state["t"], cond, h, w)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        if sigmas is not None and len(sigmas) not in (steps, steps + 1):
            sigmas = None                   # an explicit schedule only applies at its own step count
        if "mask" in cond:      # a prompt-travel blend (pad_conditioning): only the tokens either side has
            cond = {"tokens": (cond["tokens"][0] if cond["tokens"].dim() == 3 else cond["tokens"])[cond["mask"] > 0]}
        neg = neg_cond["tokens"] if neg_cond is not None else None
        if cfg != 1.0 and neg is None and not dit.flow_map:
            raise ValueError("Flux 3 Base samples with CFG against an encoded empty prompt; none was given")
        tokens = S.sample(dit, cond["tokens"], height, width, steps=steps, seed=seed, cfg=cfg, neg_txt=neg,
                        noise=noise, on_step=on_step, sigmas=sigmas, alpha=dict(options).get("alpha", S.ALPHA),
                        refs=[r[0] if r.dim() == 4 else r for r in refs] if refs else None)   # the workbench's
        #                                                       references carry a batch dim; the sampler's are (C, h, w)
        # (1, C, h, w) as every driver's: the workbench keeps lat[0] as a travel frame's reference
        return S.to_latents(tokens, height // S.DOWNSAMPLING_RATIO, width // S.DOWNSAMPLING_RATIO)[None]

    def pad_conditioning(self, conds):
        """Prompt travel (LoRA Royale): prompts zero-padded to one length with a mask of each one's own tokens (its
        real tokens and its pad to the multiple of 80). The DiT attends every token it is given, so generate keeps
        only the masked-in ones: each waypoint renders exactly as its own prompt, and between two lengths the extra
        tokens fade toward zero."""
        L = max(c["tokens"].shape[-2] for c in conds)
        out = []
        for c in conds:
            h = c["tokens"][0] if c["tokens"].dim() == 3 else c["tokens"]
            pad = L - h.shape[0]
            out.append({"tokens": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        h, w = height // S.DOWNSAMPLING_RATIO, width // S.DOWNSAMPLING_RATIO
        tokens = latents if latents.dim() == 2 else S.to_tokens(latents[0] if latents.dim() == 4 else latents)
        p = next(vae.parameters())
        img = vae.decode(tokens.to(p.device), S.latent_ids(h, w, p.device)).float().clamp(-1, 1)
        return Image.fromarray(((img + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

    # ---- LoRA structure -------------------------------------------------------------------------
    def block_map(self, dit=None):
        """Four areas in the model's own terms: the per-mode blocks for the image being made, the reference images
        and the text (5 each, before the streams meet), then the 28 shared single-stream blocks."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def blocks(prefix, bid, label, n):
            out = []
            for i in range(n):
                mods = [f"{prefix}.{i}.{m}" for m in BLOCK_MODULES]
                out.append(Block(f"{bid}_{i}", f"{label} {i}", [m for m in mods if names is None or m in names]))
            return out
        return [BlockGroup("Image blocks", blocks("content_mode_blocks.image", "image", "Image", N_MODE)),
                BlockGroup("Reference blocks", blocks("content_mode_blocks.image_cond", "ref", "Reference", N_MODE)),
                BlockGroup("Text blocks", blocks("txt_mode_blocks", "text", "Text", N_MODE)),
                BlockGroup("Single blocks", blocks("single_blocks", "single", "Single", N_SINGLE))]
