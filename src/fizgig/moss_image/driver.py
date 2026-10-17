"""MOSS-Image driver for Fizgig's standard layer (families/driver.py).

* conditioning: Qwen3.5-9B last_hidden_state of the chat template (thinking on), per caption, unpadded, from the
  side-venv worker (te_proxy.py) -> {"hidden_states": (L, 4096) bf16}. The text encoder and the DiT's text refiner
  are never trained.
* latents: the FLUX.2 autoencoder (flux3/autoencoder.py, Comfy-Org's flux2-vae file), BatchNorm-normalised and
  2x2-packed -> (128, h, w) at /16, exactly the layout the reference pipeline decodes
* training: flow matching, x_t = (1 - sigma) x0 + sigma noise, target noise - x0, DiT timestep 1000 sigma, plain
  token-mean MSE (paper eq. 3); sigma from the paper's eq. 4 grid distribution (or logit-normal / uniform). Caption
  dropout (family option, default 0) swaps in the encoded empty prompt, which the text cache stage writes beside the
  captions' cache.
* sampling: the reference Euler loop - shift 5, CFG against an encoded negative, the negative branch skipping block 22
  (uncond_skip_layers), optional SLG (sampling.py)
"""
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver
from fizgig.moss_image import sampling as S

logger = logging.getLogger(__name__)

BLOCK_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
N_BLOCKS = 42
UNCOND_FILE = "mossimage_empty_prompt.safetensors"   # not *_mossimage[_te].safetensors: the
#                                                       text stage would prune it as a stale item cache


def _dtype(dit):
    """The activations' dtype: the norms' (they stay bf16 under an INT8 / NF4 base). Read off the output norm, not a
    block's: compiled blocks (compile_blocks) are wrappers without the block's attributes."""
    return dit.final_norm_video.weight.dtype


TEXT_CACHE_BATCH = 1 << 20    # a dataset's captions in one request: the worker's layer-major pass (te_worker.py)


class MossImageDriver(FamilyDriver):
    _uncond = None
    _text_datasets = None
    _empty_h = None

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.moss_image.model import load_moss_dit
        return load_moss_dit(path, device=device).eval().requires_grad_(False)

    def load_planned(self, path, device, precision, blocks_to_swap):
        """A ComfyUI int8 ConvRot file (Comfy-Org's moss_image_sft_int8_convrot, and repackages of it): its own int8
        weights, as stored - the precision is the file's. Streaming needs weights the shared offloader can move, so a
        swapped load decodes it and takes Fizgig's INT8. Such a model renders previews and workbench pictures only:
        prepare_training refuses it as a training base. Any other file: the shared load."""
        from fizgig.moss_image.model import is_convrot, load_moss_dit
        if not is_convrot(path):
            return None
        swap = min(int(blocks_to_swap or 0), self.max_blocks_to_swap())
        if swap <= 0:
            logger.info(f"[precision] {os.path.basename(path)}: int8 ConvRot, as stored")
            return load_moss_dit(path, device=device, convrot="keep").eval().requires_grad_(False), 0
        from fizgig.families import quant
        dit = load_moss_dit(path, device="cpu", convrot="decode").eval().requires_grad_(False)
        quant.quantize(dit, self, "int8", device, store_device="cpu")
        self.enable_block_swap(dit, swap, device, False)
        logger.info(f"[precision] {os.path.basename(path)}: decoded to INT8 with {swap} blocks streamed")
        return dit, swap

    # ---- training previews on the SFT checkpoint (train_preview_checkpoint) ---------------------------------------
    def park_for_preview(self, dit, device):
        """Stream all but two of the training model's blocks so the SFT model fits beside it; an NF4 base cannot
        stream and is small - left as it is (token None)."""
        if getattr(dit, "_nf4_quantized", False):
            return None
        orig = int(dit.blocks_to_swap or 0)
        dit.enable_block_swap(len(dit.layers) - 2, torch.device(device), True)
        dit.prepare_block_swap_before_forward()
        return orig

    def load_preview_checkpoint(self, path, device, int8=False):
        """The SFT DiT for this preview round, sized to the VRAM left beside the parked training model: an int8
        ConvRot file as stored; a bf16 file in bf16 when it fits with room for the render, else INT8; blocks streamed
        (forward-only) when even that does not fit."""
        from fizgig.families import quant
        from fizgig.moss_image.model import is_convrot
        cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
        free = quant.free_vram_gb() if cuda else float("inf")     # FIZGIG_SIM_VRAM_GB: the simulated card's
        size = os.path.getsize(path) / 1024 ** 3
        room = 4.0                          # a 1024 px render + the VAE decode
        if is_convrot(path):
            precision, need = "int8", size + room
        else:
            precision = "int8" if int8 or free < size + room else "bf16"
            need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2 if precision == "int8" or is_convrot(path) else size) / N_BLOCKS
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        if swapped:
            self.block_swap_mode(m, inference=True)
        logger.info(f"[sample] SFT checkpoint: {precision}, {swapped} blocks streamed"
                    + (f" ({free:.1f} GB free)" if cuda else ""))
        return m.eval().requires_grad_(False), swapped

    def unpark_after_preview(self, dit, device, token):
        """The run's own swap back, or with none the preview-time offloader torn down and the model back on the GPU."""
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

    # ---- full fine-tune (families/ft.py) ---------------------------------------------------------------------------
    def ft_spec(self, dit):
        """The 42 blocks' six Linears on the NF4 trunk, one component each (per block: q / k / v / o 9.4M parameters
        each, ffn_up_gate 50M, ffn_down 25M - the planner trains as many together as the card holds). The text
        refiner, embedders, shared modulation, norms and final layer stay frozen, as for the LoRA. ComfyUI's file
        fuses q / k / v into proj.qkv ([q; k; v] rows); the checkpoint is written back in that layout, so ComfyUI
        loads it as it loads the base. Memory: the shared estimates (NF4 trunk + 3.5 GB) until measured on a GPU."""
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="layers",
                      components=("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"),
                      file_layout=(("proj.q.weight", "proj.qkv.weight", 0, 3),
                                   ("proj.k.weight", "proj.qkv.weight", 1, 3),
                                   ("proj.v.weight", "proj.qkv.weight", 2, 3)))

    def ft_source_unfit(self, path):
        """A fine-tune reads its master from, and writes its checkpoint as, one bf16 file - ComfyUI's
        moss_image_base_bf16.safetensors (not the diffusers shards, nor an int8 ConvRot file)."""
        if not (isinstance(path, str) and os.path.isfile(path) and path.endswith(".safetensors")):
            return ("is not a single .safetensors file - the fine-tune trains ComfyUI's "
                    "moss_image_base_bf16.safetensors and writes its checkpoint in that layout")
        return super().ft_source_unfit(path)

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.layers) if dit is not None else N_BLOCKS) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    def compile_targets(self, dit):
        return dit.layers

    def load_vae(self, path, device):
        from fizgig.flux3.autoencoder import load_flux3_vae
        return load_flux3_vae(path, device=device)

    def load_text_encoder(self, path, device):
        from fizgig.moss_image.te_proxy import MossTextEncoder
        return MossTextEncoder(path, device=device)

    def unload_text_encoder(self, te):
        """Before the worker stops: after a text cache stage, the empty prompt's encoding goes beside each dataset's
        cache (caption dropout reads it), unless it is already there."""
        try:
            dirs = [getattr(ds, "cache_directory", None) for ds in (self._text_datasets or ())]
            todo = [d for d in dirs if d and not os.path.isfile(os.path.join(d, UNCOND_FILE))]
            if todo:
                from safetensors.torch import save_file
                h = self._empty_h if self._empty_h is not None else te.encode([""])[0]
                h = h.to(torch.bfloat16).contiguous()
                for d in todo:
                    os.makedirs(d, exist_ok=True)
                    save_file({"hidden_states": h}, os.path.join(d, UNCOND_FILE),
                              metadata={"architecture": "mossimage", "caption1": ""})
                logger.info(f"[caption_dropout] empty-prompt embed cached ({h.shape[0]} tokens)")
        finally:
            self._text_datasets = None
            self._empty_h = None
            te.unload()

    def _needs_empty(self):
        return self._empty_h is None and any(
            d and not os.path.isfile(os.path.join(d, UNCOND_FILE))
            for d in (getattr(ds, "cache_directory", None) for ds in (self._text_datasets or ())))

    def cache_stage(self, stage, datasets, args, device, aux):
        if stage == "text":
            self._text_datasets = list(datasets)
            self._empty_h = None
            if getattr(args, "batch_size", None) is None:
                args.batch_size = TEXT_CACHE_BATCH     # not the shared 8: every layer would cross 85 times for 676
        return False                       # the shared encode path runs

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        from fizgig.flux3 import sampling as F3
        p = next(vae.parameters())
        out = []
        for a in images:
            x = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float().div(127.5).sub(1.0)
            tokens, ids = vae.encode(x.to(p.device, p.dtype))
            h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
            out.append(F3.to_latents(tokens, h, w).float().cpu())
        return out

    @torch.no_grad()
    def encode_text(self, te, captions):
        captions = list(captions)
        extra = self._text_datasets is not None and self._needs_empty()
        hs = te.encode(captions + [""] if extra else captions)   # the empty prompt rides along in the same pass
        if extra:
            self._empty_h = hs.pop()
        return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in hs]

    # ---- options --------------------------------------------------------------------------------
    #   caption_dropout   e.g. 0.05: that share of steps trains on the empty prompt (default 0 = off)
    #   timesteps         paper (default) | logit_normal | uniform
    def set_options(self, options):
        super().set_options(options)
        self._uncond = None
        p = float(self.options.get("caption_dropout") or 0)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"caption_dropout must be between 0 and 1, got {p}")
        kind = self.options.get("timesteps") or "paper"
        if kind not in ("paper", "logit_normal", "uniform"):
            raise ValueError(f"timesteps: paper, logit_normal or uniform - got {kind!r}")

    def prepare_training(self, dit, group, net=None):
        if getattr(dit, "_convrot", False):
            raise RuntimeError("the DiT is ComfyUI's int8 ConvRot MOSS-Image file - Fizgig trains from the bf16 file "
                               "(moss_image_base_bf16.safetensors) and applies its own INT8 / NF4. The int8 SFT "
                               "file belongs in the SFT preview row of Preferences")
        p = float(self.options.get("caption_dropout") or 0)
        if p <= 0:
            return
        from safetensors.torch import load_file
        for ds in getattr(group, "datasets", ()):
            f = os.path.join(getattr(ds, "cache_directory", "") or "", UNCOND_FILE)
            if os.path.isfile(f):
                self._uncond = load_file(f)["hidden_states"][None]
                break
        if self._uncond is None:
            logger.warning("[caption_dropout] no empty-prompt embed in the cache folders (re-run the text cache) - "
                           "caption dropout is off for this run")
        else:
            logger.info(f"[caption_dropout] {p:.2f} of steps train on the empty prompt")

    def run_metadata(self):
        return {"ss_caption_dropout": str(self.options.get("caption_dropout") or 0),
                "ss_timestep_sampling": self.options.get("timesteps") or "paper"}

    # ---- training -------------------------------------------------------------------------------
    def _sample_sigma(self, generator, min_t, max_t):
        return S.scale_t(S.sample_t(generator, self.options.get("timesteps") or "paper"), min_t, max_t)

    def _text(self, cond, device, generator=None):
        h = cond["hidden_states"]
        h = h if h.dim() == 3 else h[None]
        mask = cond.get("mask")
        p = float(self.options.get("caption_dropout") or 0)
        if self._uncond is not None and p > 0 and generator is not None:
            if torch.rand(1, generator=generator).item() < p:
                return self._uncond.to(device), None
        if mask is not None:
            mask = (mask if mask.dim() == 2 else mask[None]).to(device).bool()
        return h.to(device), mask

    def _forward(self, dit, xt, sigma, text, mask):
        dt = _dtype(dit)
        if torch.is_tensor(sigma):                  # per sample (a batch): sigma (B,)
            t = sigma.reshape(-1).to(xt.device, torch.float32) * S.NUM_TRAIN_TIMESTEPS
        else:
            t = torch.full((xt.shape[0],), float(sigma) * S.NUM_TRAIN_TIMESTEPS, device=xt.device,
                           dtype=torch.float32)
        return dit(xt.to(dt), t, text.to(dt), mask)

    # ---- batch size > 1: the masked path (model._forward_masked, per-row RoPE) ------------------------------------
    supports_batch = True

    def batch_cond(self, batch, device):
        """The cached captions, batched: a ragged batch arrives right-padded with `cond__hidden_states_mask`."""
        c = super().batch_cond(batch, device)
        if "hidden_states_mask" in c:
            c["mask"] = c.pop("hidden_states_mask")
        return c

    @staticmethod
    def _batch_rows(cond, n):
        """Each sample's own caption tokens (the valid rows of a right-padded batch)."""
        h = cond["hidden_states"]
        h = h if h.dim() == 3 else h[None]
        if h.shape[0] != n:
            raise RuntimeError(f"MOSS-Image: {h.shape[0]} captions for a batch of {n} latents")
        mask = cond.get("mask")
        mask = None if mask is None else (mask if mask.dim() == 2 else mask[None]).bool()
        return [h[b][mask[b].to(h.device)] if mask is not None else h[b] for b in range(n)]

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        """Flow matching: xt = (1 - s) x0 + s noise, target noise - x0, per-image token-mean MSE. A batch draws
        noise, sigma and caption dropout per sample in the order batch 1 draws them (sample by sample), so a batch
        is the mean of its samples' batch-1 losses; captions of different lengths take the masked path."""
        if refs:
            raise RuntimeError("MOSS-Image has no edit training")
        device = latents.device
        x0 = latents.float()
        x0 = x0 if x0.dim() == 4 else x0[None]
        n = x0.shape[0]
        if n == 1:
            noise = torch.randn(x0.shape, generator=generator).to(device)
            sigma = self._sample_sigma(generator, min_t, max_t)
            text, mask = self._text(cond, device, generator)
            xt = (1 - sigma) * x0 + sigma * noise
            info_t = sigma
        else:
            rows = self._batch_rows(cond, n)        # dropout swaps a row for the empty prompt; then padded + mask
            p = float(self.options.get("caption_dropout") or 0)
            noise, sig = [], []
            for b in range(n):
                noise.append(torch.randn((1,) + tuple(x0.shape[1:]), generator=generator))
                sig.append(self._sample_sigma(generator, min_t, max_t))
                if self._uncond is not None and p > 0 and torch.rand(1, generator=generator).item() < p:
                    rows[b] = self._uncond[0].to(rows[b].device, rows[b].dtype)
            noise = torch.cat(noise).to(device)
            sigma = torch.tensor(sig, dtype=torch.float32, device=device)
            xt = (1 - sigma.view(-1, 1, 1, 1)) * x0 + sigma.view(-1, 1, 1, 1) * noise
            L = max(r.shape[0] for r in rows)
            if all(r.shape[0] == L for r in rows):
                text, mask = torch.stack(rows).to(device), None
            else:
                text = torch.stack([torch.cat([r, r.new_zeros(L - r.shape[0], r.shape[1])]) for r in rows]).to(device)
                mask = torch.stack([torch.arange(L) < r.shape[0] for r in rows]).to(device)
            info_t = float(sigma.mean())
        pred = self._forward(dit, xt, sigma, text, mask)
        target = noise - x0
        if diff_ref is not None and diff_weight > 0.0:
            # image-pair sliders (Krea 2's weighting): each token's error weighted by how much the two poles differ
            d = (x0 - diff_ref.to(device).float().reshape(x0.shape)).abs().mean(dim=1)       # (B, h, w)
            dm = d.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            r = (d / dm).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            wgt = torch.where(dm <= 1e-6, torch.ones_like(wgt), wgt)
            return ((pred.float() - target).pow(2).mean(dim=1) * wgt).mean(), {"t": info_t}
        return F.mse_loss(pred.float(), target), {"t": info_t}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = latents.float()
        x0 = x0 if x0.dim() == 4 else x0[None]
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        sigma = self._sample_sigma(generator, min_t, max_t)
        return {"xt": (1 - sigma) * x0 + sigma * noise, "t": sigma}

    def predict(self, dit, state, cond):
        text, mask = self._text(cond, state["xt"].device)
        return self._forward(dit, state["xt"], state["t"], text, mask)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        opts = dict(options)
        device = next(p for p in dit.parameters()).device
        if sigmas is not None and len(sigmas) not in (steps, steps + 1):
            sigmas = None                   # an explicit schedule only applies at its own step count
        text, mask = self._text(cond, device)
        neg = neg_mask = None
        if cfg != 1.0:
            if neg_cond is None:
                raise ValueError("MOSS-Image samples with CFG against an encoded negative (the empty prompt counts); "
                                 "none was given")
            neg, neg_mask = self._text(neg_cond, device)
        uncond_skip = opts.get("uncond_skip_layers", S.UNCOND_SKIP_LAYERS)
        if isinstance(uncond_skip, (int, str)):
            uncond_skip = [int(x) for x in str(uncond_skip).replace(",", " ").split()] if str(uncond_skip) else []
        return S.sample(dit, text, height, width, steps=steps, seed=seed, cfg=cfg, neg=neg, mask=mask,
                        neg_mask=neg_mask, shift=float(opts.get("shift", S.SHIFT)), uncond_skip_layers=uncond_skip,
                        slg_layers=tuple(opts.get("slg_layers", ())), slg_scale=float(opts.get("slg_scale", 0.0)),
                        slg_start=float(opts.get("slg_start", 0.0)), slg_stop=float(opts.get("slg_stop", 1.0)),
                        sigma_schedule=sigmas, noise=noise, device=device, on_step=on_step)

    def pad_conditioning(self, conds):
        """Prompt travel: zero-padded to one length with a mask of each prompt's own tokens. The DiT never lets a
        padded token act as a key and starts the image's positions after the real tokens, so a padded prompt renders
        exactly as the unpadded one."""
        rows = [c["hidden_states"][0] if c["hidden_states"].dim() == 3 else c["hidden_states"] for c in conds]
        L = max(r.shape[0] for r in rows)
        out = []
        for h in rows:
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image

        from fizgig.flux3 import sampling as F3
        z = latents[0] if latents.dim() == 4 else latents
        h, w = z.shape[-2:]
        p = next(vae.parameters())
        img = vae.decode(F3.to_tokens(z.float()).to(p.device, p.dtype), F3.latent_ids(h, w, p.device))
        img = img.float().clamp(-1, 1)
        # the reference pipeline's postprocess: (x + 1) / 2, then round to uint8
        arr = ((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    # ---- LoRA structure -------------------------------------------------------------------------
    def block_map(self, dit=None):
        """The 42 single-stream blocks, six Linears each. The text refiner (text_embedder.*) is not in the map: it
        and the text encoder stay frozen."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        n = len(dit.layers) if dit is not None else N_BLOCKS
        blocks = []
        for i in range(n):
            mods = [f"layers.{i}.{m}" for m in BLOCK_MODULES]
            blocks.append(Block(f"block_{i}", f"Block {i}", [m for m in mods if names is None or m in names]))
        return [BlockGroup("Blocks", blocks)]

    def alias_flat(self, flat):
        """diffusers / PEFT names from other trainers: transformer.layers.N... is already ours once the prefix goes.
        A LoRA on the fused proj.qkv of ComfyUI's file (one extracted from a fine-tuned checkpoint) is split by
        lorafile.fused_parts through ft_spec's file_layout."""
        for p in ("transformer_", "diffusion_model_"):
            if flat.startswith(p):
                return flat[len(p):]
        return None
