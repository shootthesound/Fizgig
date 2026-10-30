"""MiniMax H3 driver for Fizgig's standard layer (families/driver.py) - first stage of the H3 port.

The old H3 trainer's own code behind the FamilyDriver interface, so the shared cache / train code runs H3 exactly as
it does today:
* caching: the old scripts' cache functions (scripts/minimax_cache_latents.cache_latents / minimax_cache_text
  .cache_text) over the datasets the shared cache loaded - the same minimaxh3 files, byte for byte (cache_stage)
* conditioning: the cached Qwen3-VL-32B layer-50 states, plus a clip's audio rows and a voice item's audio_only flag
  (batch_cond)
* training: minimax/trainer.compute_loss - flow matching at the shift-12 sigma density, audio on its own remapped
  schedule, a voice item's video term left out
* LoRA: the blocks' attention + MLP Linears (AdaLN left out, as --no_train_adaln), block ids h3blk_N / h3_rf_N

Still to come (the migration doc's §5.7b order): quant tiers + H2D rings, the Ostris clip adapter, two bases, the
workbench wrap, FT.
"""
import gc
import logging

import torch

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
_BLOCK_MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")



_ADALN_ROWS = {}          # (dit id, file, mtime) -> a frozen file's unscaled AdaLN rows (MiniMaxDriver._adaln_pairs)

class _TrainableOff:
    """compute_distill_loss's teacher switch (its lora_disabled zeroes `unet_loras` multipliers): here the family
    LoRA's trainable scale goes to 0 for the teacher pass - the frozen adapter and context stay on, as before."""

    def __init__(self, net):
        self.unet_loras = [_Mult(net)]


class _Mult:
    def __init__(self, net):
        self.net, self._m = net, 1.0

    @property
    def multiplier(self):
        return self._m

    @multiplier.setter
    def multiplier(self, v):
        self._m = float(v)
        self.net.set_trainable_multiplier(self._m)


class MiniMaxDriver(FamilyDriver):

    _allowed = {}
    _uncond = None

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        return self.load_planned(path, device, "int8", 0)[0]

    # ---- the base: H3's tiers and rings (the old trainer's plan_base_quant / enable_block_swap) -------------
    def max_blocks_to_swap(self, dit=None):
        return 40                     # the old planner's cap (50 blocks, >= 10 resident)

    def nogc_load(self, group):
        """A clip dataset's heaviest clip as effective megapixels (spatial size x latent frames), read against the
        clip figures; a stills dataset its largest bucket."""
        from fizgig.minimax import common as T
        eff, clip_t = T._max_effective_mp(group)
        if clip_t > 1:
            return "clips", eff
        return super().nogc_load(group)

    def plan_run(self, precision, blocks_to_swap, *, group, run):
        """The old trainer's Auto plan, unchanged: the precision and the streamed-block count chosen together from
        free VRAM, the run's real adapter size (header shapes, optimizer, frozen LoRAs, EMA shadow) and the
        dataset's heaviest item (spatial size x clip frames). int8 first (streamed H2D when it does not fit), NF4
        under the tested streaming floor or when the staging would starve system RAM. Previews are trimmed to the
        plan up front: a streamed plan leaves ~4 GB, so clips start at 22 frames; a 16 GB-class card caps them at
        768x640 / 22 frames."""
        from fizgig.minimax import common as T
        from fizgig.utils.device import plannable_free_vram
        path = run["dit_path"]
        free = plannable_free_vram()
        pruned = T.is_pruned_checkpoint(path)
        mp = 0.25
        try:
            mp = max(k[-2] * k[-1] / 1e6 for ds in group.datasets for k in ds.batch_manager.bucket_resos)
        except Exception:
            pass
        eff, clip_t = T._max_effective_mp(group)
        if eff > 0:
            mp = eff
        if clip_t > 1:
            logger.info(f"[vram] the heaviest item is a {clip_t}-latent-frame clip - planning against its effective "
                        f"{mp:.2f} MP (spatial size x frames).")
        pats = [x for x in T.DEFAULT_INCLUDE_PATTERNS
                if "token_refiner" not in x or self.options.get("train_token_refiner") == "1"]
        params = T.adapter_param_count(path, pats,
                                       network_type=run.get("network_type") or "lora",
                                       network_dim=int(run.get("network_dim") or 16),
                                       lokr_factor=int(run.get("lokr_factor") or 8))
        adapter, frozen, ema = T.plan_adapter_gb(params, run.get("optimizer_type") or "adamw8bit",
                                                 training_adapter_path=run.get("training_adapter"),
                                                 context_lora_path=run.get("context_lora_path"),
                                                 ema_decay=run.get("ema_decay") or 0.0)
        if precision == "auto" and blocks_to_swap >= 0:
            # a hand-set swap skips the planner: the precision is the file's own, as in the old trainer
            mode, n = ("int8" if pruned else "nf4"), blocks_to_swap
            why = (f"chosen from the checkpoint, not from free VRAM, because Blocks Swap is set to {n} rather than "
                   f"Auto. Set Blocks Swap to Auto to have the precision and the swap count planned together.")
        elif precision == "auto":
            mode, n, _ckpt, why = T.plan_base_quant(free, pruned, mp=mp, adapter_gb=adapter)
        else:
            mode = precision
            resident = {"int8": T._RESIDENT_INT8_GB,
                        "hqq": T._RESIDENT_HQQ_PRUNED_GB if pruned else T._RESIDENT_HQQ_GB}.get(
                mode, T._RESIDENT_PRUNED_GB if pruned else T._RESIDENT_GB)
            n, _ckpt = T.plan_vram(free, mp=mp, resident_gb=resident,
                                   transient_gb=T._INT8_TRANSIENT_GB if mode == "int8" else 0.0, adapter_gb=adapter)
            why = f"base precision pinned to {mode} by the user"
        extra = (f" +{frozen:.2f} GB frozen LoRAs" if frozen else "") + (f" +{ema:.2f} GB EMA shadow" if ema else "")
        logger.info(f"[vram] auto plan: free {free:.1f} GB, largest item {mp:.2f} MP, adapter ~{adapter:.1f} GB "
                    f"({params / 1e6:.0f} M params){extra} -> {mode}, blocks_to_swap={n}")
        if mode == "nf4" and pruned and precision == "auto":
            logger.warning("[vram] this run trains on a 4-bit base (~9% error) instead of the checkpoint's own int8 "
                           "(~0.17%). It is much faster here, but the LoRA spends some capacity correcting "
                           "quantization error that will NOT exist at inference. To force the accurate base, set Base "
                           "Precision to int8 - expect block swap and a slower run - or close other GPU apps and "
                           "re-launch.")
        self._plan_previews(n)
        return mode, n, why

    def _plan_previews(self, n_swap):
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            total = 99.0
        frames = int(self.options.get("preview_frames", 1) or 1)
        if n_swap > 0 and total >= 20.0 and frames > 22:
            logger.info(f"[preview] the plan streams {n_swap} blocks, which leaves clip previews ~4 GB - {frames} "
                        f"frames -> 22 up front (a 56-frame 768 clip OOMs at that headroom). Sound kept.")
            self.options["preview_frames"] = 22
        if total < 20.0:
            if frames > 22:
                self.options["preview_frames"] = 22
            self._small_card = True
            logger.info("[preview] 16 GB card: previews cap at 768x640 / 22 frames on this class of GPU (sound kept)")

    def load_planned(self, path, device, precision, blocks_to_swap):
        """The base at int8 (the file's ConvRot codes), NF4 or HQQ, the last n blocks streamed host->device through
        the ring that matches their type (rintic-13's int8 / HQQ rings, @mabseyuk's NF4 ring); classic parking if
        a ring cannot build. AdaLN stays fp32 (not a LoRA target), as in the old --no_train_adaln runs."""
        import os
        from fizgig.minimax.loader import load_minimax_h3_dit
        mode = precision if precision in ("int8", "nf4", "hqq") else "int8"
        n = max(0, int(blocks_to_swap or 0))
        dit = load_minimax_h3_dit(path, device=device, compute_dtype=DTYPE, quantize=True, blocks_to_swap=n,
                                  base_quant=mode, adaln_fp32=True).requires_grad_(False)
        if n > 0:
            ring = mode == "int8" or os.environ.get("FIZGIG_NO_NF4_H2D") != "1"
            n = dit.enable_block_swap(n, h2d_only=ring, ring_size=2)
            off = getattr(dit, "_h2d_offloader", None)
            if off is not None:
                gb = getattr(off, "staged_gb", None)
                gb = n * 0.39 if gb is None else gb
                staging = ("pinned in RAM" if not getattr(off, "_pin_failed", False) else
                           "staged in ordinary RAM (pinning unavailable or RAM too tight) - copies synchronous")
                logger.info(f"[vram] block swap active: last {n} blocks streamed H2D-only "
                            f"({getattr(off, 'kind', '?')}, ring 2, ~{gb:.1f} GB {staging}) - no writeback, "
                            f"prefetch overlaps compute")
            else:
                logger.info(f"[vram] block swap active: last {n} blocks parked on CPU (~{n * 0.34:.1f} GB VRAM "
                            f"freed, packed {mode} in RAM)")
        self._n_swap, self._base_mode = n, mode
        return dit, n

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(bool(on))

    # ---- caching: today's H3 cache code, over the datasets the shared cache loaded ------------------
    def cache_stage(self, stage, datasets, args, device, aux):
        import argparse
        common = dict(skip_existing=args.skip_existing, keep_cache=args.keep_cache, num_workers=args.num_workers)
        if stage == "latents":
            from fizgig.scripts.minimax_cache_latents import cache_latents
            cache_latents(argparse.Namespace(vae=args.model, audio_vae=aux.get("audio_vae") or None,
                                             clip_still=aux.get("clip_still") == "1", batch_size=args.batch_size,
                                             **common), datasets, device)
        else:
            from fizgig.scripts.minimax_cache_text import cache_text
            # --aux reference_count=K directly, or the GUI's pair: distill_refs=K counted only with distill=1
            refs = int(aux.get("reference_count") or (aux.get("distill_refs", 0) if aux.get("distill") == "1" else 0))
            cache_text(argparse.Namespace(text_encoder=args.model, reference_count=refs,
                                          no_quantize=aux.get("no_quantize") == "1",
                                          batch_size=args.batch_size or 16, **common), datasets, device)
        return True

    def clip_bucket_cap(self, free_gb, width, height):
        from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder
        return MiniMaxH3VideoVAEEncoder.plan_clip_bucket(free_gb, width, height)

    # ---- the old trainer's training options (--family_option KEY=VALUE) ------------------------------
    #   photo_blocks / clip_blocks / audio_blocks   e.g. 20-49: that modality's steps train only these blocks (the
    #                                               old --photo_blocks / --clip_blocks / --audio_blocks, backward cut)
    #   tread             ratio@start-end, e.g. 0.5@2-47: clip steps route that share of video tokens past the blocks
    #   caption_dropout   e.g. 0.05: swap in the cached empty-prompt embed on that share of steps
    #   shift             the sigma density (the old --shift); unset = H3's own shift 12
    #   clip_still_as_photo  1: each clip's cached still also trains as a photo
    def set_options(self, options):
        super().set_options(options)
        from fizgig.minimax.common import parse_block_spec
        n = self.description.n_blocks
        self._allowed = {m: set(parse_block_spec(options[f"{m}_blocks"], n)) for m in ("photo", "clip", "audio")
                         if options.get(f"{m}_blocks")}
        self._uncond = None
        from fizgig.dataset.image_dataset import ImageDataset
        ImageDataset.clip_still_as_photo = options.get("clip_still_as_photo") == "1"

    def prepare_training(self, dit, group, net=None):
        import logging
        import os
        log = logging.getLogger(__name__)
        self._net = net
        self._distill_setup(group, log)
        self._ramp_setup(log)
        hn = self.options.get("highnoise_lr_pct") or (float(self.options.get("highnoise_lr") or 1.0) * 100)
        if abs(float(str(hn).rstrip("%")) - 100.0) > 1e-9:
            log.info(f"[lr] steps above sigma 0.5 train at {float(str(hn).rstrip('%')):.0f}% of the learning rate.")
        tread = self.options.get("tread")
        if tread:
            ratio, span = tread.split("@")
            start, end = (int(x) for x in span.split("-"))
            dit._tread = (float(ratio), start, end)
            log.info(f"[tread] token routing ON - {float(ratio) * 100:.0f}% of the video tokens skip blocks {start}-"
                     f"{end - 1} on every clip step")
        p = float(self.options.get("caption_dropout") or 0)
        if p > 0:
            from safetensors.torch import load_file
            for ds in group.datasets:
                f = os.path.join(getattr(ds, "cache_directory", "") or "", "uncond_minimaxh3_te.safetensors")
                if os.path.isfile(f):
                    self._uncond = load_file(f)["hidden_states"].unsqueeze(0)
                    break
            if self._uncond is None:
                log.warning("[caption_dropout] no uncond embed in the cache dirs - dropout disabled for this run")
            else:
                log.info(f"[caption_dropout] {p:.2f} - empty-prompt embed loaded")
        for m, allowed in self._allowed.items():
            log.info(f"[likeness] {m} steps train blocks {min(allowed)}-{max(allowed)} only (backward cut at the "
                     f"window)")

    @staticmethod
    def _modality(batch):
        if batch.get("audio_only") is not None and bool(batch["audio_only"].any()):
            return "audio"
        return "photo" if batch["latents"].dim() == 4 else "clip"

    def step_frozen_blocks(self, batch):
        """A routed step's out-of-window blocks, and the token refiner with them: its text rows enter at block 0, so
        a trainable refiner would drag the backward through every block (the old backward cut froze it too). Under
        a fine-tune, the modality's route over the cycle (the old _ft_freeze): the always-on refiner trains on
        every modality."""
        ftb = getattr(self, "_ftb", None)
        if ftb is not None:
            cat = {"audio": "voice"}.get(self._modality(batch), self._modality(batch))
            route = ftb.routes.get(cat)
            return () if route is None else tuple(f"h3blk_{i}" for i in range(self.description.n_blocks)
                                                  if i not in route)
        allowed = self._allowed.get(self._modality(batch))
        if not allowed:
            return ()
        return tuple(f"h3blk_{i}" for i in range(self.description.n_blocks) if i not in allowed) + ("h3_rf_0", "h3_rf_1")

    def _ramp_setup(self, log):
        """The old adapter-relative LR ramp (option adapter_ramp=R, off by default): each step held at fraction R of
        the adapter's current size, starting at 10% of the LR and climbing as the adapter grows. The old AdapterRamp
        controller, reading the family LoRA's ||dW||."""
        try:
            r = float(self.options.get("adapter_ramp") or 0)
        except ValueError:
            r = 0.0
        self._ramp = None
        if r > 0:
            from fizgig.minimax.common import AdapterRamp
            ramp = AdapterRamp.__new__(AdapterRamp)
            ramp.target, ramp.mult, ramp._smooth, ramp._prev = r, 0.1, None, None
            ramp._size = self._adapter_size
            self._ramp = ramp
            log.info(f"[ramp] adapter-relative LR ON - each step held at {100 * r:.3f}% of the adapter's current size")

    @torch.no_grad()
    def _adapter_size(self):
        from fizgig.families.lora import TRAINABLE
        tot = 0.0
        for w in self._net.wrapped.values():
            m = w.adapters[TRAINABLE] if TRAINABLE in w.adapters else None
            if m is None:
                continue
            if hasattr(m, "lokr_w1"):
                n = (m.lokr_w1.float().norm() * m.lokr_w2.float().norm()) ** 2
            else:
                a, b = m[0].weight.float(), m[1].weight.float()
                n = torch.trace((a @ a.T) @ (b.T @ b)).clamp(min=0)
            tot += float(n) * w.scales[TRAINABLE] ** 2
        return tot ** 0.5

    def after_optimizer_step(self):
        if getattr(self, "_ramp", None) is not None:
            self._ramp.step()

    def run_metadata(self):
        def stop(k):
            spec = self.options.get(k)
            if not spec and self.options.get("stop_epoch") and self.options.get("stop_category") == k.split("_")[0]:
                spec = f"{self.options['stop_epoch']}:{self.options.get('stop_mode') or 'anchor'}"
            n, _, mode = str(spec or "").partition(":")
            return f"{int(n)}:{mode or 'anchor'}" if n.strip().isdigit() and int(n) else "off"
        o = self.options
        md = {"ss_visual_stop": stop("visual_stop"), "ss_audio_stop": stop("audio_stop"),
              # the old trainer's run record, from the options this run had
              "ss_photo_blocks": o.get("photo_blocks") or "all", "ss_clip_blocks": o.get("clip_blocks") or "all",
              "ss_audio_blocks": o.get("audio_blocks") or "all", "ss_tread": o.get("tread") or "off",
              "ss_caption_dropout": str(o.get("caption_dropout") or 0),
              "ss_clip_still_as_photo": "1" if o.get("clip_still_as_photo") == "1" else "0",
              "ss_distill": "dataset" if o.get("distill") == "1" else "off",
              "ss_adapter_ramp": str(o.get("adapter_ramp") or "off"),
              "ss_train_token_refiner": "1" if o.get("train_token_refiner") == "1" else "0"}
        if self._shift() is not None:
            md["ss_timestep_density"] = f"{float(self._shift()):g}"
        if getattr(self, "_base_mode", None):
            md["ss_base_quant"] = self._base_mode
        return md

    def _distill_setup(self, group, log):
        """Reference distillation (options distill=1, distill_weight 0.8, distill_phase1 -1 = auto): the cached r2v
        conditioning (--aux reference_count=K) is the teacher. Identity-first (the old default): epochs 1..P train
        against the teacher ONLY at a third of the LR, then the photos alone; P auto = ~650 steps."""
        import math
        self._distill = self.options.get("distill") == "1"
        self._p1 = 0
        if not self._distill:
            return
        self._dw = float(self.options.get("distill_weight") or 0.8)
        p1 = int(self.options.get("distill_phase1") or -1)
        steps = max(1, int(getattr(group, "num_train_items", 1)))
        self._p1 = max(1, math.ceil(650 / steps)) if p1 < 0 else p1
        if self._p1:
            log.info(f"[distill] IDENTITY-FIRST: the first {self._p1} epoch(s) (~{self._p1 * steps} steps, or the whole "
                     f"run if shorter) train against the teacher ONLY at a third of the LR, then the photographs alone "
                     f"at the full LR.")
        else:
            log.info(f"[distill] reference distillation ON - teacher weight {self._dw:.2f}, photo {1 - self._dw:.2f}")

    def _teacher_phase(self):
        return bool(getattr(self, "_distill", False) and self._p1 and getattr(self, "_epoch", 1) <= self._p1)

    def step_policy(self, batch, epoch):
        """Per-category retirement, as the old trainer: past its stop epoch (options visual_stop / audio_stop,
        "N:anchor" or "N:stop") photos & clips or voice items either train on at 10% LR (anchor - a drift guard on
        the shared adapters) or are skipped outright (stop - faster epochs)."""
        from fizgig.minimax.common import _P1_LR_SCALE, ANCHOR_LR_SCALE
        self._epoch = epoch
        if getattr(self, "_slider_pairs", False) and "latents_control_0" not in batch:
            return True, 1.0              # a pair slider: a clip's derived still has no pair, so it sits out
        if getattr(self, "_ftb", None) is not None and self.options.get("ft_scope") == "photo" and (
                self._modality(batch) != "photo"):
            return True, 1.0              # 'Train on: Photos only': clip and voice batches are skipped outright
        phase = _P1_LR_SCALE if self._teacher_phase() else 1.0      # identity-first: the whole phase-1 epoch
        if getattr(self, "_ramp", None) is not None:
            phase *= self._ramp.mult
        voice = self._modality(batch) == "audio"
        spec = self.options.get("audio_stop" if voice else "visual_stop")
        if not spec and not getattr(self, "_ftb", None) and self.options.get("stop_epoch") and self.options.get(
                "stop_category") == (
                "audio" if voice else "visual"):
            spec = f"{self.options['stop_epoch']}:{self.options.get('stop_mode') or 'anchor'}"     # the GUI's three rows
        if not spec:
            return False, phase
        n, _, mode = str(spec).partition(":")
        if not int(n or 0) or epoch <= int(n):
            return False, phase
        if (mode or "anchor") == "anchor":
            return False, phase * ANCHOR_LR_SCALE
        return True, 1.0

    def batch_cond(self, batch, device):
        import random
        text = batch["hidden_states"]
        if self._uncond is not None and random.random() < float(self.options.get("caption_dropout") or 0):
            text = self._uncond                                               # caption dropout step
        cond = {"hidden_states": text.to(device)}
        if "ref_hidden_states" in batch:          # reference distillation's teacher conditioning
            cond.update(ref_hidden_states=batch["ref_hidden_states"].to(device), ref_latent=batch["ref_latent"],
                        ref_token_tags=batch["ref_token_tags"][0])
        if batch.get("audio_latent") is not None:
            cond["audio_latent"] = batch["audio_latent"].to(device)
        if batch.get("audio_only") is not None and bool(batch["audio_only"].any()):
            cond["audio_only"] = True
        return cond

    def _shift(self):
        """shift=X, or lownoise_pct=P (the share of steps below sigma 0.5: shift = (1 - p) / p, the old GUI's rule)."""
        pct = self.options.get("lownoise_pct")
        if pct not in (None, ""):
            p = float(str(pct).rstrip("%")) / 100.0
            if 0.0 < p < 1.0:
                return (1.0 - p) / p
        shift = self.options.get("shift")
        return float(shift) if shift not in (None, "", "sigmoid", "resolution") else (shift or None)

    # ---- training: the old compute_loss ------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        from fizgig.minimax.common import compute_distill_loss, compute_loss, sample_sigmas
        if diff_ref is not None and diff_weight > 0.0:
            return self._slider_loss(dit, latents, cond, generator, min_t, max_t, diff_ref, diff_weight)
        teacher = self._teacher_phase()
        if (getattr(self, "_distill", False) and (teacher or not self._p1) and "ref_hidden_states" in cond
                and not cond.get("audio_only")):
            # the old distillation step: teacher = base + frozen adapters WITH the reference, trainable LoRA off;
            # student = the LoRA from text alone (a voice has no face to distill - it takes the plain path)
            rz = cond["ref_latent"].to(latents.device, DTYPE)
            rz = rz.unsqueeze(2) if rz.dim() == 4 else rz
            # a picture's latent is (1, 24, H, W): without its frame axis the 5-D sigma broadcast read the 24
            # channels as frames (mat1 and mat2 shapes cannot be multiplied (22200x4 and 96x5376), #264)
            lat = latents if latents.dim() == 5 else latents.unsqueeze(2)
            loss, s = compute_distill_loss(dit, _TrainableOff(self._net), lat, cond["hidden_states"].to(DTYPE),
                                           text_ref=cond["ref_hidden_states"].to(DTYPE), ref_latents=[rz],
                                           text_token_tags=cond["ref_token_tags"],
                                           distill_weight=1.0 if teacher else self._dw, shift=self._shift())
            return loss, {"t": float(s)}
        lat = latents if latents.dim() == 5 else latents.unsqueeze(2)          # (1, 24, T, H, W)
        _pt, ph, pw = getattr(dit, "patch_size", (1, 2, 2))
        tokens = (lat.shape[-2] // ph) * (lat.shape[-1] // pw)
        shift = self._shift()
        sigma = sample_sigmas(1, "cpu", shift=shift, generator=generator, image_tokens=tokens)
        if min_t > 0.0 or max_t < 1.0:
            sigma = min_t + (max_t - min_t) * sigma
        noise = torch.randn(lat.shape, generator=generator, dtype=torch.float32)
        audio = cond.get("audio_latent")
        if audio is not None and audio.dim() == 3:
            audio = audio[0]                                                   # cached (2T, 32), batched
        # audio_weight (CLI only, default 1.0 = parity): the weight on the sound term for clips that carry it - audio is
        # ~4% of the packed sequence, so parity may teach a voice too quietly. Stills and muted clips have no sound term.
        loss, s = compute_loss(dit, lat.to(DTYPE), cond["hidden_states"].to(DTYPE), sigma=sigma.to(lat.device),
                               noise=noise, audio_latent=audio,
                               audio_weight=float(self.options.get("audio_weight") or 1.0),
                               video_weight=0.0 if cond.get("audio_only") else 1.0)
        info = {"t": float(s)}
        hn = float(self.options.get("highnoise_lr") or 1.0)
        if self.options.get("highnoise_lr_pct") not in (None, ""):
            hn = max(0.0, min(1.0, float(str(self.options["highnoise_lr_pct"]).rstrip("%")) / 100.0))
        if hn != 1.0 and not cond.get("audio_only"):
            # the old noise-band LR: steps drawn above sigma 0.5 train at this share (a voice step's gradient lives on
            # the audio schedule, which this classification does not describe - it sits out)
            from fizgig.minimax.common import MINIMAX_LOWNOISE_SIGMA
            info["lr_mult"] = hn if float(s) >= MINIMAX_LOWNOISE_SIGMA else 1.0
        return loss, info

    # ---- sliders (families/train.py): image / clip pairs through training_loss's diff_ref, prompt pairs through
    #      noise_latents / predict. compute_loss's still / clip path, video only (a pole's sound is not the dial) ----
    def _noised(self, latents, generator, min_t, max_t):
        """compute_loss's training input in training_loss's draw order: the latent cropped to the patch grid, the
        run's sigma, the noise. -> (x0, noise, sigma (1,))."""
        from fizgig.minimax.common import sample_sigmas
        lat = latents["latent"] if isinstance(latents, dict) else latents
        lat = lat if lat.dim() == 5 else lat.unsqueeze(2)                     # (1, 24, T, H, W)
        _pt, ph, pw = getattr(self, "_patch", (1, 2, 2))
        x0 = lat[..., :(lat.shape[-2] // ph) * ph, :(lat.shape[-1] // pw) * pw].float()
        sigma = sample_sigmas(1, "cpu", shift=self._shift(), generator=generator,
                              image_tokens=(x0.shape[-2] // ph) * (x0.shape[-1] // pw))
        if min_t > 0.0 or max_t < 1.0:
            sigma = min_t + (max_t - min_t) * sigma
        noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device)
        return x0, noise, sigma

    def _slider_loss(self, dit, latents, cond, generator, min_t, max_t, diff_ref, diff_weight):
        """An image- or clip-pair slider step: the plain flow loss with each patch token weighted by how much the two
        poles differ there (the Krea 2 / Qwen formula), over every frame of a clip pair; poles of different lengths
        weigh every token the same."""
        import torch.nn.functional as F
        self._patch = getattr(dit, "patch_size", (1, 2, 2))
        _pt, ph, pw = self._patch
        x0, noise, sigma = self._noised(latents, generator, min_t, max_t)
        s = float(sigma.reshape(-1)[0])
        noised = (1.0 - s) * x0 + s * noise
        txt = cond["hidden_states"]
        txt = (txt[None] if txt.dim() == 2 else txt).to(x0.device, DTYPE)
        pred = dit(noised.to(DTYPE), (1.0 - sigma).to(x0.device), txt).float()
        ref = diff_ref if diff_ref.dim() == 5 else diff_ref.unsqueeze(2)
        ref = ref[..., :x0.shape[-2], :x0.shape[-1]].to(x0.device).float()
        if ref.shape != x0.shape:            # poles of different lengths (an edit that drops frames): even weights
            se = (pred - (x0 - noise)).pow(2)
            return se.mean(), {"t": s}
        d = (x0 - ref).abs().mean(dim=1)                                       # (1, T, H, W)
        b, t_, h, w = d.shape
        d = F.interpolate(F.avg_pool2d(d.reshape(b * t_, 1, h, w), (ph, pw)), scale_factor=(ph, pw),
                          mode="nearest").reshape(b, t_, h, w)                 # one value per patch token
        dm = d.mean()
        if float(dm) > 1e-6:
            wt = (1.0 - float(diff_weight)) + float(diff_weight) * (d / dm).clamp(max=8.0)
            wt = wt / wt.mean().clamp_min(1e-8)
        else:
            wt = torch.ones_like(d)                                            # identical pair: uniform
        se = (pred - (x0 - noise)).pow(2).mean(dim=1)
        return (se * wt).mean(), {"t": s}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0, noise, sigma = self._noised(latents, generator, min_t, max_t)
        s = float(sigma.reshape(-1)[0])
        return {"x": ((1.0 - s) * x0 + s * noise).to(DTYPE), "tt": (1.0 - sigma).to(x0.device), "t": s}

    def predict(self, dit, state, cond):
        txt = cond["hidden_states"]
        txt = (txt[None] if txt.dim() == 2 else txt).to(state["x"].device, DTYPE)
        return dit(state["x"], state["tt"], txt)

    def encode_images(self, vae, images):
        """uint8 (H, W, 3) arrays -> (24, h, w) latents through the H3 VAE encoder, loaded on first use (the
        previews' VAE is the decoder only) - a prompt slider's practice renders are re-encoded here."""
        import numpy as np
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder
        enc = vae.get("encoder")
        if enc is None:
            enc = MiniMaxH3VideoVAEEncoder()
            with safe_open(vae["path"], framework="pt", device="cpu") as f:
                enc.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
            vae["encoder"] = enc = enc.to(torch.float32).eval()
        device = vae["device"]
        enc.to(device)
        try:
            x = torch.from_numpy(np.stack([np.asarray(im)[..., :3] for im in images])).permute(0, 3, 1, 2)
            with torch.no_grad():
                z = enc.encode(x.float().div(127.5).sub(1.0).to(device, torch.float32))   # (B, 24, 1, h, w)
            return [(zi.squeeze(1) if zi.dim() == 4 else zi).cpu() for zi in z]
        finally:
            enc.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def slider_setup(self, group):
        """A pair slider's derived clip stills sit out (step_policy); the dial's previews render at the Samples tab's
        Sample length - clips side by side, or a still strip."""
        self._slider_pairs = any(getattr(ds, "has_control", False) for ds in getattr(group, "datasets", []))
        frames = int(self.options.get("preview_frames", 1) or 1)
        logger.info("[slider] previews: %s", f"{frames}-frame clips side by side" if frames > 1 else "stills side by side")

    def still_renders(self):
        """generate() renders stills inside (a prompt slider's practice pictures train as stills)."""
        import contextlib

        @contextlib.contextmanager
        def _stills():
            had = self.options.get("preview_frames")
            self.options["preview_frames"] = 1
            try:
                yield
            finally:
                if had is None:
                    self.options.pop("preview_frames", None)
                else:
                    self.options["preview_frames"] = had
        return _stills()

    def slider_preview(self, frames, multipliers):
        """Clip previews at -1 / 0 / +1 laid side by side as one clip (the PNG is the labelled middle-frame strip);
        stills take the shared strip."""
        if not frames or not isinstance(frames[0], dict):
            return None
        from fizgig.families.train import _slider_strip
        n = min(f["frames"].shape[1] for f in frames)
        h = frames[0]["frames"].shape[2]
        gap = torch.zeros(3, n, h, 8)
        parts = []
        for k, f in enumerate(frames):
            if k:
                parts.append(gap)
            parts.append(f["frames"][:, :n, :h])
        return {"frames": torch.cat(parts, dim=3), "image": _slider_strip([f["image"] for f in frames], multipliers),
                "wave": None, "every_frame": True}    # every frame + an mp4: a frame-skip dial shows in the motion

    # ---- previews (first stage: the reference sampler, the old clip contract) -----------------------
    #   options: preview_frames (1 = a still; 22 / 39 / 56 ... clips), preview_audio=1 (a clip's sound),
    #            audio_vae=PATH (the decoder for that sound)
    def load_text_encoder(self, path, device):
        from fizgig.minimax.embedder import load_minimax_h3_te_planned
        return load_minimax_h3_te_planned(path, device=device, compute_dtype=DTYPE, quantize=True)

    def unload_text_encoder(self, te):
        import gc
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": te.encode(c)[0].detach().cpu()} for c in captions]       # (L, 5120)

    def load_vae(self, path, device):
        """The video decoder in fp16 on the CPU (it rides to the GPU for the decode only, as the old previews) and,
        with the audio_vae option, the audio decoder."""
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
        dec = MiniMaxH3VideoVAEDecoder()
        with safe_open(path, framework="pt", device="cpu") as f:
            dec.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        audio = None
        apath = self.options.get("audio_vae")
        if apath:
            from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae_decoder
            audio = load_minimax_h3_audio_vae_decoder(apath, device="cpu")
        return {"video": dec.to(torch.float16).eval(), "audio": audio, "device": device, "path": path}

    def initial_noise(self, seed, width, height):
        raise NotImplementedError("H3 travel previews come with the workbench wrap")

    @staticmethod
    def _adaln_pairs(dit, path, strength):
        """A LoRA file's full-model AdaLN rows as (AdalnProj, A, B * strength) - the old _prefilter_frozen_lora's
        AdaLN half (the Linears are the family LoRA's)."""
        import os
        key = (id(dit), path, os.path.getmtime(path))
        raw = _ADALN_ROWS.get(key)
        if raw is None:
            # read once per file: a strength change (the workbench's Dial / Confirm) only rescales the rows
            from safetensors.torch import load_file
            from fizgig.networks.lora import ensure_kohya_lora_state_dict
            sd = ensure_kohya_lora_state_dict(load_file(path))
            parents = {f"lora_unet_{n.replace('.', '_')}_linear": m for n, m in dit.named_modules()
                       if type(m).__name__ == "AdalnProj"}
            raw = []
            for name, ap in parents.items():
                down, up = sd.get(f"{name}.lora_down.weight"), sd.get(f"{name}.lora_up.weight")
                lin = ap.linear.base if hasattr(ap.linear, "base") else ap.linear
                # rows that fit the Linear itself (a LoRA trained on this pruned base) ride the family LoRA, as the
                # old _prefilter_frozen_lora kept them; only full-model rows (another input width) are injected
                if (down is not None and up is not None and up.shape[0] == lin.out_features
                        and down.shape[1] != lin.in_features):
                    raw.append((ap, down.clone(), up.clone()))
            while len(_ADALN_ROWS) >= 4:    # a run holds a file per role (adapter, context, speed) at most
                _ADALN_ROWS.pop(next(iter(_ADALN_ROWS)))
            _ADALN_ROWS[key] = raw
        return [(ap, down.clone(), up * float(strength)) for ap, down, up in raw]

    def frozen_file_added(self, dit, path, strength, role):
        """A frozen file's full-model AdaLN rows (an older H3 LoRA trained with AdaLN on, as a Context LoRA): the
        pruned base has no AdaLN Linears to wrap, so they are injected at run time (turbo_adaln_patch), as the old
        load_context_lora did - on for every training step; in previews the adapter's come off, the context's stay."""
        pairs = self._adaln_pairs(dit, path, strength)
        if not pairs:
            return
        if not hasattr(self, "_frozen_adaln"):
            self._frozen_adaln = {}
        self._frozen_adaln[role] = pairs
        logger.info(f"[{role}] {len(pairs)} AdaLN rows injected at run time")
        self._patch_adaln(dit, ("adapter", "context"))

    def _patch_adaln(self, dit, roles, extra=()):
        """One patch for the union of the active frozen files' AdaLN rows (a patch replaces the module forward
        wholesale, so every set on a module goes in together)."""
        from fizgig.minimax.common import turbo_adaln_patch, turbo_adaln_unpatch
        held = getattr(self, "_frozen_adaln", {})
        turbo_adaln_unpatch([p for v in held.values() for p in v])
        pairs = [p for r in roles for p in held.get(r, [])] + list(extra)
        if pairs:
            device = next(p for p in dit.parameters() if p.device.type == "cuda").device
            turbo_adaln_patch(dit, pairs, device, DTYPE)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, frames=None, audio=None):
        """A preview render. The shared trainer has the Turbo (the family's speed LoRA) switched on around this call
        and passes its steps and CFG; its full-model AdaLN rows, and a context LoRA's, are injected for the render,
        the training adapter's taken off - and the training set put back after."""
        if not getattr(self, "_frozen_adaln", None):
            return self._generate(dit, cond, width, height, steps, seed, cfg, neg_cond, frames, audio, on_step)
        try:
            self._patch_adaln(dit, ("context", "speed"))
            return self._generate(dit, cond, width, height, steps, seed, cfg, neg_cond, frames, audio, on_step)
        finally:
            self._patch_adaln(dit, ("adapter", "context"))     # the training set back (nothing if none)

    def _generate(self, dit, cond, width, height, steps, seed, cfg, neg_cond, frames, audio, on_step=None):
        from fizgig.minimax import sampling
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        frames = int(self.options.get("preview_frames", 1) if frames is None else frames)
        want_audio = (self.options.get("preview_audio") == "1" if audio is None else bool(audio)) and frames > 1
        txt = cond["hidden_states"]
        txt = (txt[None] if txt.dim() == 2 else txt).to(device, DTYPE)
        unc = None
        if cfg > 1.0 and neg_cond is not None:
            unc = neg_cond["hidden_states"]
            unc = (unc[None] if unc.dim() == 2 else unc).to(device, DTYPE)
        lat, arows, width, height, frames = self._ladder(sampling, dit, txt, unc, width, height, steps, cfg, seed,
                                                         device, frames, on_step=on_step)
        want_audio = want_audio and frames > 1
        return {"latent": lat.cpu(), "audio": arows.cpu() if (want_audio and arows is not None) else None,
                "size": (width, height)}

    def _ladder(self, sampling, dit, txt, unc, width, height, steps, cfg, seed, device, frames, on_step=None):
        """The old previews' OOM ladder: an out-of-memory render - or one paging into system RAM, which Windows never
        raises - retries one rung down, a shorter clip first (141 -> 56 -> 22 -> still never: 22 is the floor), then
        the resolution down the standard sizes to 512. Both caps stick for later epochs; a resolution step prints the
        marker the GUI writes back into the Samples tab. At the floor of both it re-raises.
        on_step(done, total) (the workbench): called as each step ends - the next one's start - and may raise to
        cancel, which leaves as RenderCancelled, never as a ladder retry."""
        from fizgig.minimax.common import clip_fallback_frames, next_preview_res
        if getattr(self, "_small_card", False):
            from fizgig.minimax.common import cap_preview_res_small_card
            width, height = cap_preview_res_small_card(width, height)
        cap = getattr(self, "_res_cap", None)
        if cap and (cap[0] < width or cap[1] < height):
            width, height = min(width, cap[0]), min(height, cap[1])
        state = {"wh": (width, height), "frames": frames, "slow_told": False, "cancel": None}

        def slow(seconds, step, total):
            if on_step is not None:
                try:
                    on_step(step, total)
                except Exception as e:      # the workbench's cancel: out of the sampler, past the ladder
                    state["cancel"] = e
                    return True
                if seconds <= 120.0:        # polled every step; the paging notice keeps its own threshold
                    return False
            w, h = state["wh"]
            f = state["frames"]
            if (f > 1 and clip_fallback_frames(f) > 1) or next_preview_res(w, h) != (w, h):
                logger.warning(f"[preview] step {step}/{total} took {seconds:.0f}s - the render is paging into system "
                               f"RAM (Windows never raises an OOM for this). Abandoning this sample and retrying one "
                               f"rung down (a shorter clip first, then resolution).")
                return True
            if not state["slow_told"]:
                state["slow_told"] = True
                logger.warning(f"[preview] step {step}/{total} took {seconds:.0f}s - the render is spilling into "
                               f"system RAM even at the ladder floor (shortest clip, 512x512). It will finish, just "
                               f"slowly. The Turbo LoRA (Preferences) or a still Sample length make it lighter.")
            return False

        oom = (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", torch.cuda.OutOfMemoryError),
               sampling.PreviewAborted)
        while True:
            state["wh"], state["frames"] = (width, height), frames
            try:
                lat, arows = sampling.sample_image(dit, txt, width=width, height=height, steps=steps, cfg_scale=cfg,
                                                   uncond_embeds=unc, seed=seed, device=device, dtype=DTYPE,
                                                   num_frames=frames, return_audio=True, on_slow_step=slow,
                                                   **({"slow_step_s": 0.0} if on_step is not None else {}))
                return lat, arows, width, height, frames
            except oom:
                if state["cancel"] is not None:
                    from fizgig.families.workbench import RenderCancelled
                    raise RenderCancelled() from state["cancel"]
                gc.collect()
                torch.cuda.empty_cache()
                nf = clip_fallback_frames(frames) if frames > 1 else 1
                if nf > 1:
                    logger.warning(f"[preview] OOM at {frames} frames - retrying this sample at {nf} frames "
                                   f"({width}x{height} kept)")
                    frames = nf
                    self.options["preview_frames"] = nf
                    continue
                nw, nh = next_preview_res(width, height)
                if (nw, nh) == (width, height):
                    raise
                msg = f"[preview] OOM at {width}x{height} - retrying at {nw}x{nh}"
                if min(nw, nh) < 768 and not getattr(self, "_res_warned", False):
                    self._res_warned = True
                    msg += (". NOTE: below H3's 768 training canvas the model is outside its expected regime - treat "
                            "these previews as a rough guide, and judge the LoRA at full size in ComfyUI.")
                logger.warning(msg)
                width, height = nw, nh
                self._res_cap = (nw, nh)
                print(f"[preview] resolution settled: {nw}x{nh}", flush=True)

    def park_for(self, dit, device, need_gb, purpose):
        """The old trainer's park: when the card cannot offer `need_gb` beside the resident base (the decode wants
        ~7.5 GB, an override encode the text encoder + 2), only the missing gigabytes (+1) of tail blocks go to CPU
        (park_dit_partial, which unbinds a streaming ring first) - every extra gigabyte moved is paging churn and a
        slower restore on a WDDM card."""
        from fizgig.minimax.common import park_dit_partial
        from fizgig.utils.device import plannable_free_vram
        need_gb = 7.5 if need_gb is None else need_gb
        gc.collect()
        torch.cuda.empty_cache()
        free = plannable_free_vram()
        if free >= need_gb:
            return False
        need = (need_gb - free) + 1.0
        logger.info(f"[preview] {free:.1f} GB free is too tight for {purpose} - parking ~{need:.1f} GB of tail blocks "
                    f"for this pass.")
        park_dit_partial(dit, need_gb=need)
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def unpark(self, dit, device, token):
        if token:
            from fizgig.minimax.common import restore_parked_dit
            restore_parked_dit(dit, device, getattr(self, "_n_swap", 0))     # swap-aware: never the whole base
            torch.cuda.empty_cache()

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        """A still -> PIL. A clip -> a dict (frames [3, F, H, W] in [0, 1], the middle frame, the waveform) that
        save_preview writes as the old contract."""
        from PIL import Image
        device = vae["device"]
        dec = vae["video"].to(device)
        lat = latents["latent"].to(device).float()
        try:
            if lat.shape[2] > 1:
                px = dec.decode_clip(lat)[0].float().cpu()                    # [3, F, H, W]
                mid = (px[:, px.shape[1] // 2].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
                wave = None
                if latents.get("audio") is not None and vae.get("audio") is not None:
                    from fizgig.minimax.audio_vae import unpack_audio
                    adec = vae["audio"].to(device)
                    wave = adec.decode(unpack_audio(latents["audio"]).to(device, torch.float32))[0].cpu()
                    vae["audio"].to("cpu")
                return {"frames": px, "image": Image.fromarray(mid), "wave": wave}
            px = dec.decode(lat)[0]
            return Image.fromarray((px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
        finally:
            if not self.keep_vae_resident:
                vae["video"].to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def decode_audio(self, vae, audio):
        if audio is None or vae.get("audio") is None:
            return None
        from fizgig.minimax.audio_vae import unpack_audio
        device = vae["device"]
        adec = vae["audio"].to(device)
        try:
            return adec.decode(unpack_audio(audio).to(device, torch.float32))[0].float().clamp(-1, 1).cpu()
        finally:
            if not self.keep_vae_resident:
                vae["audio"].to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def save_preview(self, result, path):
        """The old previews' contract: a clip writes every 2nd frame as JPEG in <stem>.clip/, the wav and a playable
        mp4 beside it, then the middle frame as the PNG - LAST, the gallery's 'finished' signal."""
        import os
        from PIL import Image
        if not isinstance(result, dict):
            result.save(path)
            return [path]
        stem = path[:-4]
        px, out = result["frames"], []
        clip_dir = stem + ".clip"
        os.makedirs(clip_dir, exist_ok=True)
        keep = list(range(0, px.shape[1], 1 if result.get("every_frame") else 2))
        if keep[-1] != px.shape[1] - 1:
            keep.append(px.shape[1] - 1)
        for k in keep:
            fr = (px[:, k].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            Image.fromarray(fr).save(os.path.join(clip_dir, f"f{k:03d}.jpg"), quality=87)
        if result.get("wave") is not None:
            from fizgig.minimax.common import write_preview_mp4, write_wav
            write_wav(stem + ".wav", result["wave"])
            out.append(stem + ".wav")
            try:
                write_preview_mp4(stem + ".mp4", px, stem + ".wav")
                out.append(stem + ".mp4")
            except Exception:
                pass                                     # the wav and the scrub frames still work
        elif result.get("every_frame"):
            # a clip slider's strip: the motion is the dial, so a playable mp4 at the true frame rate, silent
            from fizgig.minimax.common import write_preview_mp4
            try:
                write_preview_mp4(stem + ".mp4", px, None)
                out.append(stem + ".mp4")
            except Exception:
                pass
        result["image"].save(path)
        return out + [path]

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"h3blk_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                for i in range(self.description.n_blocks)]
        refiner = [Block(f"h3_rf_{i}", f"Refiner {i}",
                         keep([f"token_refiner.blocks.{i}.{m}" for m in _BLOCK_MODULES])) for i in range(2)]
        return [BlockGroup("Blocks", main), BlockGroup("Token Refiner", refiner)]   # the old H3 panel's wording

    def expand_train_blocks(self, items):
        """The old Blocks to Train field: ranges and singles ("3-12, 22, 31-33"), "all", or h3blk_N ids."""
        from fizgig.minimax.common import parse_block_spec
        ids, spec = [], []
        for it in items:
            it = str(it).split("·")[0].strip()
            if it.startswith("h3"):
                ids.append(it)
            elif it and it.lower() != "all":
                spec.append(it)
            elif it.lower() == "all":
                return None
        if spec:
            ids += [f"h3blk_{i}" for i in parse_block_spec(",".join(spec), self.description.n_blocks)]
        return ids or None

    # ---- the fine-tune (the old rotation FT, through the shared runner) ------------------------------------------
    def ft_spec(self, dit):
        """H3's component windows - the shared runner's checks and the Training tab's plan line read this; the work
        itself is H3FTBackend's."""
        from fizgig.families.ft import FTSpec
        from fizgig.minimax.rotation_ft import H3_COMPONENT_PREFIXES
        return FTSpec(blocks="blocks", components=H3_COMPONENT_PREFIXES, overhead_gb=14.5, trunk_gb_per_block=0.21,
                      slots_gb=2.0)

    def ft_card_plan(self, path, free_gb, mp=None, options=None, max_parts=0):
        """H3's planner on the idle card: the trainer budgets after its NF4 trunk (~10.5 GB) and the non-block layers
        and VAE (~1.1 GB) are in, then adds the trunk back - so usable ~ free - 2.6; the fine-tune Blocks range
        narrows the cycle. Clips' activation reserve is left to the trainer (it sees the dataset)."""
        from fizgig.minimax.rotation_ft import plan_h3_ft_windows
        from fizgig.minimax.common import parse_block_spec
        spec = str((options or {}).get("ft_blocks") or "").strip()
        subset = sorted(parse_block_spec(spec, self.description.n_blocks)) if spec and spec.lower() != "all" else None
        windows, stream, _why = plan_h3_ft_windows(free_gb - 2.6, subset=subset, n_blocks=self.description.n_blocks,
                                                   max_parts=max_parts)
        return (windows, stream) if windows else None

    def ft_source_unfit(self, path):
        from fizgig.minimax.ft_backend import source_unfit_reason
        return source_unfit_reason(path)

    def ft_backend(self, dit, device, src, group):
        from fizgig.minimax.ft_backend import H3FTBackend
        if getattr(dit, "_tread", None):
            logging.getLogger(__name__).info("[h3-ft] TREAD token routing is LoRA-only - off for this fine-tune")
            dit._tread = None
        self._ftb = H3FTBackend(self, dit, device, src, group)
        return self._ftb

    def ft_cycle(self, cycle, offset, total):
        """Retirement under a fine-tune: stop mode only (the anchor rides optimizer machinery the per-tensor steps
        don't have) and the epoch snapped to rotation-cycle boundaries, cumulative across continuations - the old
        _snap_stop."""
        from fizgig.minimax.common import snap_ft_stop
        log = logging.getLogger(__name__)
        for k, label in (("visual_stop", "photos & clips"), ("audio_stop", "voice")):
            spec = self.options.get(k)
            if not spec and self.options.get("stop_epoch") and self.options.get("stop_category") == k.split("_")[0]:
                spec = str(self.options["stop_epoch"])
            n = int(str(spec or "0").partition(":")[0] or 0)
            if not n:
                continue
            v, kind = snap_ft_stop(n, cycle, offset, total)
            if kind == "past":
                log.info("[h3-ft] %s retirement at epoch %d is already behind this run (continuing from epoch %d) - "
                         "retired from the start.", label, v, offset)
            elif kind == "snapped":
                log.info("[h3-ft] %s retirement lands at rotation-cycle boundaries - epoch %d snaps to %d (%d-epoch "
                         "cycle).", label, n, v, cycle)
            elif kind == "never":
                log.warning("[h3-ft] %s retirement at epoch %d is at or past the run's end (epoch %d) - it will never "
                            "fire.", label, v, total + offset)
            self.options[k] = f"{v}:stop"

    def legacy_state_order(self, dit):
        """The old H3 trainer's parameter order: its network walked named_modules - the token refiner (registered
        first) then blocks 0-49, each qkv / out / fc1 / fc2."""
        groups = self.block_map(dit)
        return [m for b in groups[1].blocks for m in b.modules] + [m for b in groups[0].blocks for m in b.modules]

    def lora_target_names(self, dit):
        """The 50 main blocks (the old default); the token refiner too with option train_token_refiner=1."""
        groups = self.block_map(dit)
        if self.options.get("train_token_refiner") == "1":
            return [m for g in groups for b in g.blocks for m in b.modules]
        return [m for b in groups[0].blocks for m in b.modules]
