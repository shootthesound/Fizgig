"""MOSS-Image driver for Fizgig's standard layer (families/driver.py).

* conditioning: Qwen3.5-9B's final hidden state (after the final norm), the chat template with an open think block,
  one unpadded prompt at a time -> {"hidden_states": (L, 4096)} (embedder.py)
* latents: the FLUX.2 autoencoder Fizgig already ships for Flux 3 / Klein (fizgig/flux3/autoencoder.py), BatchNorm-
  normalised and 2x2-packed -> (128, h, w) at /16. MOSS-Image's own VAE file is this encoder (bf16-rounded) with the
  FLUX.2 small decoder; the latent space is the same
* training: flow matching on the Base checkpoint, x_t = (1 - sigma) x0 + sigma noise, target noise - x0, the DiT told
  1000 * sigma; sigma drawn from the final pretraining stage's distribution (technical report Appendix A.2, Table 6:
  truncated logit-normal, mu -1.09, s 0.95, cutoff 0.95 rescaled onto the 1000-step grid, 1% uniform), not AI
  Toolkit's shift-5 uniform. The LoRA trains all 42 blocks, block 22 included (leaving it out, as AI Toolkit does,
  cost likeness in A/B runs: the positive branch always runs it)
* sampling: the reference's snapped shift-5 schedule, Euler, CFG 3 with block 22 skipped in the negative branch

No edit training: the released checkpoints are text-to-image only (technical report section 8, "does not directly
condition on reference images or support image editing"; the pipeline's defaults "match the finalized dense
MOSSImage model, not ... reference-image variants").
"""
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver, prior_base, weighted_loss, with_prior
from fizgig.moss_image import sampling as S

logger = logging.getLogger(__name__)

N_BLOCKS = 42                 # transformer/config.json num_layers
N_TEXT_BLOCKS = 2             # the text adapter's blocks (pipeline _TextProjection)
BLOCK_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
ATTN_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o")
TEXT_MODULES = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")

# technical report Appendix A.2, Table 6, the last Stage 3 interval (steps 274,001-310,000, the released weights)
T_MU, T_S, T_CUTOFF, T_UNIFORM = -1.09, 0.95, 0.95, 0.01


def _t_cdf(x, mu):
    """P(S <= x) for S = 1 - sigmoid(Z), Z ~ N(mu, T_S^2)."""
    return 1.0 - 0.5 * math.erfc(-((math.log((1 - x) / x) - mu) / T_S) / math.sqrt(2))


def mu_for_clean_share(pct):
    """The Z mean that puts `pct`% of training steps below sigma 0.5 (the clean end), the 1% uniform included; the
    curve keeps the reference's shape and width. The reference's own mu -1.09 gives about 11%."""
    want = (float(pct) / 100.0 - T_UNIFORM * 0.5) / (1.0 - T_UNIFORM)
    want = min(max(want, 1e-4), 1.0 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(80):                       # share rises with mu: bisect
        mid = 0.5 * (lo + hi)
        if _t_cdf(0.5 * T_CUTOFF, mid) / _t_cdf(T_CUTOFF, mid) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class MOSSImageDriver(FamilyDriver):
    _plain_reads = False         # load_dit's file read: plain reads (the preview checkpoint, beside the training model)

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.moss_image.model import load_moss_dit
        return load_moss_dit(path, device=device, plain=self._plain_reads).eval().requires_grad_(False)

    # ---- training previews on the SFT checkpoint (train_preview_checkpoint; after @Murf-Code, PR #195) -------------
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
        """The SFT DiT for this preview round, sized to the VRAM left beside the parked training model: bf16 when it
        fits with room for the render, else INT8, with blocks streamed (forward-only) when even that does not fit.
        ComfyUI's int8 ConvRot file is decoded at load (model.load_moss_dit), so it sizes as the bf16 file."""
        import os
        from fizgig.families import quant
        cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
        free = quant.free_vram_gb() if cuda else float("inf")     # FIZGIG_SIM_VRAM_GB: the simulated card's
        size = os.path.getsize(path) / 1024 ** 3
        if "convrot" in os.path.basename(path).lower():
            size *= 2                       # int8 on disk, bf16 once decoded
        room = 4.0                          # a 1024 px render + the VAE decode
        precision = "int8" if int8 or free < size + room else "bf16"
        need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2 if precision == "int8" else size) / N_BLOCKS
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        # plain reads: mapping the 10 GB file beside the loaded training model crashed Windows with 32 GB of RAM (#229)
        self._plain_reads = True
        try:
            m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        finally:
            self._plain_reads = False
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

    def load_vae(self, path, device):
        """The FLUX.2 autoencoder (flux2-vae.safetensors / Klein's ae.safetensors), fp32 as the reference runs it."""
        from fizgig.flux3.autoencoder import load_flux3_vae
        return load_flux3_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Text-only Qwen3.5-9B (no vision tower, LM head or MTP head) in bf16: resident (about 15 GB) when it fits
        the free VRAM with room to run, else streamed layer-major from the CPU (embedder.MOSSTextEncoder) - still
        exact bf16, which OpenMOSS ask for (their text encoder is "much more unstable than DiT"), not 8-bit."""
        from fizgig.families.quant import free_vram_gb
        from fizgig.moss_image.embedder import MOSSTextEncoder, TOKENIZER_REPO, from_file_wanted
        from transformers import AutoTokenizer
        tok = self.from_pretrained(AutoTokenizer, TOKENIZER_REPO, subfolder="tokenizer")
        stream = free_vram_gb() < 18.0
        # streamed on a computer without the RAM to hold it (16 / 24 GB): read from the file, a layer at a time
        return MOSSTextEncoder(path, device=device, stream=stream, tokenizer=tok,
                               keep_in_ram=self.keep_text_encoder, from_file=stream and from_file_wanted())

    def text_encoder_gb(self, path):
        """The language model alone once it has loaded (the file also holds the vision tower and heads)."""
        from fizgig.moss_image.embedder import kept_gb
        return kept_gb(path) or super().text_encoder_gb(path)

    def drop_text_encoder(self):
        from fizgig.moss_image.embedder import drop_kept_state_dict
        drop_kept_state_dict()

    def cache_stage(self, stage, datasets, args, device, aux):
        """The shared text cache, with the dataset's captions in large groups so a streamed encoder moves each layer
        to the GPU once per group rather than once per 8 captions."""
        if stage == "text" and args.batch_size is None:
            args.batch_size = 4096
        return False

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    def compile_targets(self, dit):
        return dit.layers

    # ---- full fine-tune (families/ft.py) ---------------------------------------------------------
    def ft_spec(self, dit):
        # each block's six Linears are the parts (attention q / k / v / o ~0.018 GB each, the fused MLP up+gate 0.094,
        # the down projection 0.047 in bf16). ComfyUI's file fuses q / k / v into one proj.qkv = [q; k; v] tensor; the
        # diffusers shards keep them apart. The text adapter, embedders, modulations and the final layer stay frozen,
        # as for a LoRA; on SFT or a fine-tune of it (lora_sft) only attention trains, as for a LoRA.
        # Memory: measured 19 Oct on an H100 at 1 MP, the whole model in one window: all six parts peaked 12.6 GB,
        # attention only 8.3 GB. overhead_gb is fitted to each in the planner's frame (4.5 = 0.6 GB above the full
        # run's fit; 6.0 attention only, whose window is credited the whole 4-bit trunk though the FFN stays in it).
        # act_gb_per_mp is not measured yet (2.0, as Anima and Flux 3).
        from fizgig.families.ft import FTSpec
        sft = str(self.options.get("lora_sft", "")) == "1"
        return FTSpec(blocks="layers", components=ATTN_MODULES if sft else BLOCK_MODULES,
                      overhead_gb=6.0 if sft else 4.5, calib_mp=1.0, act_gb_per_mp=2.0,
                      file_layout=(("proj.q.weight", "proj.qkv.weight", 0, 3),
                                   ("proj.k.weight", "proj.qkv.weight", 1, 3),
                                   ("proj.v.weight", "proj.qkv.weight", 2, 3)))

    def ft_cycle(self, cycle, offset, total):
        # a fine-tune's Clean-end share counts rotations, not epochs: with several windows each epoch trains one, so
        # an epoch-counted Split / Slide gave each window a different share, and its epochs moved with the card
        self._ft_cycle = max(1, int(cycle))

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        p = next(vae.parameters())
        out = []
        for a in images:
            x = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float().div(127.5).sub(1.0)
            tokens, ids = vae.encode(x.to(p.device, p.dtype))
            h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
            out.append(tokens.reshape(h, w, -1).permute(2, 0, 1).float().cpu())        # (128, h, w)
        return out

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h} for h in te.encode(captions)]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(generator, min_t, max_t, mu=T_MU):
        """The final pretraining interval's sigma (report eq. 4): with probability 0.01 uniform over the grid j / 1000,
        else S = 1 - sigmoid(Z), Z ~ N(mu, 0.95^2) (mu -1.09 unless a clean-end share moves it), truncated to
        S <= 0.95 (inverse CDF), rescaled by 1 / 0.95 and taken to the grid point whose cell holds it. Then scaled into [min_t, max_t] (the Timestep Range)."""
        T = S.TRAIN_STEPS
        u = torch.rand(2, generator=generator)
        if float(u[0]) < T_UNIFORM:
            sigma = (math.floor(float(u[1]) * T) + 1) / T
        else:
            # draw F(S) uniformly in (0, F(cutoff)], F = _t_cdf
            p = max(float(u[1]), 1e-12) * _t_cdf(T_CUTOFF, mu)
            z = mu + T_S * float(torch.special.ndtri(torch.tensor(1.0 - p, dtype=torch.float64)))
            s = 1.0 / (1.0 + math.exp(z))                 # S = 1 - sigmoid(Z) = sigmoid(-Z)
            sigma = min(max(math.ceil(T * s / T_CUTOFF), 1), T) / T
        return min_t + (max_t - min_t) * sigma

    def _share(self, epoch):
        """(the Clean-end share for 1-based `epoch` as typed / computed, "" = the reference schedule; whether the epoch
        trains at half the LR). lownoise_pct=P: one share. Split, "A:N:B": A% for the first N epochs, then B%. Slide,
        "A:H:B:E": A% until epoch H, then sliding in a straight line to B% by epoch E (and B% after); "A:H:B:E:R": and
        after E, the higher of A and B, with every Rth epoch at the lower one, at half the LR (A or B first): those
        epochs reset structure, but at the full rate they cost likeness each time (Peter, 28 Oct 2026)."""
        pct = str(self.options.get("lownoise_pct", "") or "").replace("%", "").strip()
        half = False
        if ":" in pct:
            try:
                parts = [float(x) for x in pct.split(":")]
                if len(parts) not in (3, 4, 5):
                    raise ValueError
            except ValueError:
                raise RuntimeError(f"Clean-end share '{pct}': give share:epochs:share (75:5:40 = 75% for the first 5 "
                                   "epochs, then 40%) or share:epoch:share:epoch (75:6:40:10 = 75% until epoch 6, "
                                   "then slides to 40% by epoch 10; a fifth number R = after that, the higher share, "
                                   "with every Rth epoch at the lower one)") from None
            first, until, rest = parts[:3]
            end = parts[3] if len(parts) >= 4 else until
            every = int(parts[4]) if len(parts) == 5 else 0
            cyc = getattr(self, "_ft_cycle", 0)
            e = -(-int(epoch) // cyc) if cyc else int(epoch)      # a fine-tune: the rotation this epoch is in
            share = first if e <= until else rest if e >= end else first + (rest - first) * (e - until) / (end - until)
            if every > 0 and e > end:
                refresh = (e - end) % every == 0      # the refresh epoch is at the lower share, the rest at the higher
                share = min(first, rest) if refresh else max(first, rest)
                half = first != rest and refresh
            pct = f"{round(share, 2):g}"
        return pct, half

    def step_policy(self, batch, epoch):
        return False, (0.5 if self._share(epoch)[1] else 1.0)

    def _t_mu(self):
        """The timestep curve's centre for the epoch: lownoise_pct (_share) moves it so that share of steps lands
        below sigma 0.5; unset = the reference's."""
        pct, half = self._share(self.epoch)
        if not pct:
            return T_MU
        cached = getattr(self, "_mu_cached", None)
        if cached is None or cached[0] != (pct, half):
            cached = self._mu_cached = ((pct, half), mu_for_clean_share(pct))
            cyc = getattr(self, "_ft_cycle", 0)
            when = f"rotation {-(-self.epoch // cyc)} (epoch {self.epoch})" if cyc else f"epoch {self.epoch}"
            logger.info(f"MOSS-Image: {pct}% of training steps at the clean end (sigma < 0.5) from {when}, "
                        f"Z mean {cached[1]:.2f} (reference -1.09, about 11%)"
                        + (" - at half the learning rate (the lower share, after the slide)" if half else ""))
        return cached[1]

    def _forward(self, dit, xt, t, cond):
        hidden = cond["hidden_states"]
        hidden = hidden if hidden.dim() == 3 else hidden[None]
        tt = torch.tensor([t * S.TRAIN_STEPS], device=xt.device)
        return dit(xt, tt, hidden.to(xt.device))

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        if refs:
            raise RuntimeError("MOSS-Image has no edit (reference-image) conditioning")
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        t = self._sample_t(generator, min_t, max_t, self._t_mu())
        xt = (1 - t) * x0 + t * noise
        base = prior_base(prior, loss_mask, lambda: self._forward(dit, xt, t, cond))
        pred = self._forward(dit, xt, t, cond).float()
        target = noise - x0
        wgt = None
        if diff_ref is not None and diff_weight > 0.0:
            # image-pair sliders (Krea 2's weighting): each token's error weighted by how much the two poles differ
            # there, so the slider learns what changes between them, not what they share
            d = (x0 - diff_ref.to(x0.device).float()).abs().mean(dim=1)                       # (1, h, w)
            dm = d.mean().clamp_min(1e-8)
            r = (d / dm).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean().clamp_min(1e-8)
            if float(dm) <= 1e-6:
                wgt = torch.ones_like(wgt)                      # identical pair: uniform, never all-zero
        mw = None if loss_mask is None else loss_mask.float().to(x0.device)[:, 0]             # (1, h, w)
        if wgt is None and mw is None:
            # report eq. 3: per-token mean over channels, no sigma weighting
            return F.mse_loss(pred, target), {"t": t}
        loss = weighted_loss((pred - target).pow(2).mean(dim=1), wgt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred - base).pow(2).mean(dim=1), mw, wgt)
        return loss, {"t": t}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        t = self._sample_t(generator, min_t, max_t, self._t_mu())
        return {"xt": (1 - t) * x0 + t * noise, "t": t}

    def predict(self, dit, state, cond):
        return self._forward(dit, state["xt"], state["t"], cond)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @staticmethod
    def _real_tokens(cond):
        """A prompt-travel blend (pad_conditioning) keeps only the tokens either waypoint has; the DiT drops padding
        the same way (it evaluates each prompt at its true length)."""
        h = cond["hidden_states"]
        h = h[0] if h.dim() == 3 else h
        if "mask" in cond:
            m = cond["mask"]
            m = (m[0] if m.dim() == 2 else m).to(h.device)
            h = h[m > 0]
        return h

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        if refs:
            raise RuntimeError("MOSS-Image has no edit (reference-image) conditioning")
        if sigmas is not None and len(sigmas) not in (steps, steps + 1):
            sigmas = None                   # an explicit schedule only applies at its own step count
        # the description's default sampler under the caller's options (training previews pass none)
        opts = {**dict(self.description.default_sampling().options), **dict(options)}
        neg = self._real_tokens(neg_cond) if (neg_cond is not None and cfg != 1.0) else None
        if cfg != 1.0 and neg is None:
            raise ValueError("MOSS-Image samples with CFG against an encoded negative (the empty prompt by default); "
                             "none was given")
        skip = tuple(opts.get("uncond_skip", S.UNCOND_SKIP) or ())
        return S.sample(dit, self._real_tokens(cond), height, width, steps=steps, seed=seed, cfg=cfg, neg_text=neg,
                        noise=noise, on_step=on_step, sigmas=sigmas, shift=float(opts.get("shift", S.SHIFT)),
                        uncond_skip=skip, sampler=str(opts.get("sampler", "euler")),
                        scheduler=str(opts.get("schedule", "reference")), slg_end=float(opts.get("slg_end", 1.0)))

    def pad_conditioning(self, conds):
        """Prompt travel (LoRA Royale): prompts zero-padded to one length with a mask of each one's real tokens.
        generate keeps only the masked-in tokens, so each waypoint renders exactly as its own prompt."""
        L = max(c["hidden_states"].shape[-2] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"][0] if c["hidden_states"].dim() == 3 else c["hidden_states"]
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        from fizgig.flux3.autoencoder import latent_ids
        z = latents[0] if latents.dim() == 4 else latents                         # (128, h, w)
        h, w = z.shape[-2:]
        tokens = z.permute(1, 2, 0).reshape(h * w, -1)
        p = next(vae.parameters())
        img = vae.decode(tokens.to(p.device), latent_ids(h, w, p.device)).float().clamp(-1, 1)
        return Image.fromarray(((img + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

    # ---- LoRA structure -------------------------------------------------------------------------
    def block_map(self, dit=None):
        """The 42 joint blocks, then the two-block text adapter that refines Qwen3.5's states before they join the
        image tokens (the report's "Text adapter: linear projection 4096 -> 3072; two text-only Transformer
        blocks"). Works with no model loaded (the Profiler and Extract)."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"block_{i}", f"Block {i}", keep([f"layers.{i}.{m}" for m in BLOCK_MODULES]))
                for i in range(N_BLOCKS if dit is None else len(dit.layers))]
        text = [Block(f"text_{i}", f"Text adapter {i}",
                      keep([f"text_embedder.blocks.{i}.{m}" for m in TEXT_MODULES])) for i in range(N_TEXT_BLOCKS)]
        return [BlockGroup("Blocks", main), BlockGroup("Text adapter", text)]

    def lora_target_names(self, dit):
        """The 42 blocks' Linears, block 22 included: the negative branch skips it, but the positive one always runs
        it, and training it gave clearly better likeness than leaving it out as AI Toolkit does. The text adapter
        stays frozen with the text encoder (OpenMOSS's guide: keep the text encoder frozen; @Murf-Code, PR #195;
        training it exaggerated expressions epoch by epoch). Repair Studio still shows it, for LoRAs from trainers
        that do adapt it. On SFT or a fine-tune of it (the lora_sft option) only attention trains: its feed-forward
        layers trained gave body horror within a few epochs."""
        names = [m for m in super().lora_target_names(dit) if m.startswith("layers.")]
        if str(self.options.get("lora_sft", "")) == "1":
            names = [m for m in names if m.endswith(ATTN_MODULES)]
        return names

    def quant_target_names(self, dit):
        """Every block Linear, whichever of them the LoRA trains (an SFT base's feed-forward layers too): block swap
        pairs blocks layer by layer, so every block must hold the same dtype - block 22 left bf16 among INT8 blocks
        got INT8 weights swapped into a plain Linear (#201)."""
        return [m for m in super().lora_target_names(dit) if m.startswith("layers.")]
