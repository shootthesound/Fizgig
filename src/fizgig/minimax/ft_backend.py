"""MiniMax H3's side of the shared fine-tune (families/train.py _FineTune): the old H3 rotation fine-tune's model
pieces, unchanged, behind the backend interface the shared runner drives (families/ft.py SharedBackend).

What differs from the shared (bf16-file, families/quant NF4) backend, and why it is H3's own:
- the source is the int8 ConvRot checkpoint - the master is DEQUANTISED from its codes (build_bf16_master_h3, or the
  lazy disk MasterStore), and saves re-encode trained stems to int8 with stochastic rounding (save_full_checkpoint_h3);
- the NF4 trunk is bnb Linear4bit (the H3 loader's), swapped by H3NF4Rotator's class swap;
- the window plan carries H3's calibration and a clip dataset's activation reserve (plan_h3_ft_windows);
- streaming is H3's NF4 H2D ring, rescoped to the frozen out-of-window blocks at every rotation;
- the cycle tightens to the blocks the dataset's modalities train, and each modality is confined to its blocks per
  step (plan_ft_modality_routing) - photos to the likeness blocks, voice to the audio blocks, clips to the clip blocks.
Everything here is the old trainer's code path (minimax/trainer.py train_minimax, the ft_rotation branches), cited.
"""
import gc
import hashlib
import logging
import os
import re
import shutil

import torch
import torch.nn as nn

from fizgig.minimax.rotation_ft import H3NF4Rotator

logger = logging.getLogger(__name__)

_BASE = re.compile(r"\.base(?=\.|$)")


def _model_name(name):
    """A Linear's name as the checkpoint knows it: FamilyLoRA wraps each target and keeps the real Linear as .base."""
    return _BASE.sub("", name)


class H3FTRotator(H3NF4Rotator):
    """H3NF4Rotator on a DiT whose Linears the family LoRA has wrapped (the training adapter and the preview Turbo
    ride on the wrappers): discovery, always-on activation and the flushed master use the checkpoint's names."""

    def __init__(self, blocks, master, key_prefix="blocks", device="cuda", block_subset=None):
        super().__init__(blocks, master, key_prefix=key_prefix, device=device, block_subset=block_subset)
        self._by_key = {}
        for bi, block in enumerate(blocks):
            if self.block_subset is not None and bi not in self.block_subset:
                continue
            for lname, lin in block.named_modules():
                if type(lin).__name__ == "Linear4bit" and ".adapters." not in f".{lname}.":
                    self._by_key[self._key(bi, _model_name(lname))] = (bi, _model_name(lname), lin)

    def activate_always(self, prefix, module):
        n_swapped = n_direct = 0
        for lname, lin in [(_model_name(n), m) for n, m in module.named_modules()
                           if isinstance(m, nn.Linear) and ".adapters." not in f".{n}."]:
            key = f"{prefix}.{lname}.weight"
            if type(lin).__name__ == "Linear4bit":
                w = self.master.get(key)
                if w is None:
                    logger.warning("[h3-nf4] no master weight for always-on %s - leaving frozen", key)
                    continue
                self._orig_class[id(lin)] = type(lin)
                lin.__class__ = nn.Linear
                lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16), requires_grad=True)
                n_swapped += 1
            else:
                lin.weight.requires_grad_(True)
                n_direct += 1
            self.touched.add(key)
            if lin.bias is not None:
                lin.bias.requires_grad_(True)
        self.always.append((prefix, module))
        logger.info("[h3-nf4] always-on: %s (%d Linears trainable for the whole run%s)", prefix,
                    n_swapped + n_direct, f"; {n_swapped} de-quantized, {n_direct} already dense" if n_swapped else "")
        return n_swapped + n_direct

    def _always_linears(self):
        for prefix, module in self.always:
            for n, m in module.named_modules():
                if isinstance(m, nn.Linear) and ".adapters." not in f".{n}.":
                    yield f"{prefix}.{_model_name(n)}.weight", m

    def master_state_dict(self):
        """The old _h3_flushed_state_dict / BlockRotator.master_state_dict, with the checkpoint's names."""
        from fizgig.minimax.rotation_ft import MasterStore, _FlushedView
        if isinstance(self.master, MasterStore):
            def _puller(lin):
                return lambda: lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
            live = {k: _puller(lin) for k, lin in self._targets(list(self.active))}
            live.update({k: _puller(lin) for k, lin in self._always_linears()})
            return _FlushedView(self.master, live)
        out = dict(self.master)
        for key, lin in self._targets(list(self.active)):
            if key in out:
                out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        for key, lin in self._always_linears():
            out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        return out


class H3FTBackend:
    """The shared runner's backend for H3 (see the module docstring)."""

    def __init__(self, driver, dit, device, src, group):
        from fizgig.minimax.common import parse_block_spec, plan_ft_modality_routing
        self.driver, self.dit, self.device, self.src = driver, dit, torch.device(device), src
        self.n_blocks = len(dit.blocks)
        o = driver.options
        subset = None
        if o.get("ft_blocks") and str(o["ft_blocks"]).strip().lower() != "all":
            subset = sorted(parse_block_spec(o["ft_blocks"], self.n_blocks))
            if not subset:
                raise RuntimeError(f"[h3-ft] ft_blocks {o['ft_blocks']!r} selects no blocks")
        n_photo, n_clip, n_voice = self._composition(group)
        if o.get("ft_scope") == "photo":
            n_clip = n_voice = 0          # 'Train on: Photos only' filters the dataset: no demand on the cycle
        self.n_clip_items = n_clip
        pb, cb, ab = o.get("photo_blocks"), o.get("clip_blocks"), o.get("audio_blocks")
        if subset is not None and ((pb and n_photo) or (ab and n_voice)):
            logger.info("[h3-ft] Blocks is set explicitly (%s) - the explicit range wins over photo/voice routing.",
                        o["ft_blocks"])
        plan_subset, self.routes = plan_ft_modality_routing(self.n_blocks, pb, ab, n_photo, n_voice, n_clip,
                                                            explicit_subset=subset, clip_blocks=cb)
        if subset is None and plan_subset is not None:
            from fizgig.minimax.common import format_block_spec
            subset = plan_subset
            logger.info("[h3-ft] the cycle tightens to blocks %s - the union of what this dataset actually trains.",
                        format_block_spec(subset))
        for cat, label, blocks in (("photo", "photo", pb), ("voice", "audio", ab), ("clip", "clip", cb)):
            if self.routes[cat] is not None:
                logger.info("[h3-ft] %s batches freeze every block outside %s.", label, blocks)
        self.subset = subset
        self.group = group
        self.rot = None
        self.stream = False
        self._ring = None
        self.scratch = None

    @staticmethod
    def _composition(group):
        from fizgig.dataset.image_dataset import VIDEO_EXTENSIONS, is_audio_path
        vext = {e.lower() for e in VIDEO_EXTENSIONS}
        try:
            paths = [p for ds in group.datasets for p in getattr(getattr(ds, "datasource", None), "image_paths", [])
                     or []]
            n_voice = sum(1 for p in paths if is_audio_path(p))
            n_clip = sum(1 for p in paths if os.path.splitext(p)[1].lower() in vext)
            return len(paths) - n_voice - n_clip, n_clip, n_voice
        except Exception:
            return 1, 1, 1                # unreadable: the widest mix, so the cycle never under-spans

    # ---- the master ------------------------------------------------------------------------------------------------
    def build_master(self, where, scratch_dir):
        """RAM (dequantised now) or disk (MasterStore: untouched tensors re-dequantised on read, trained ones spilled
        beside the dataset caches - the old trainer's choice, by the same 40%-of-free-RAM rule)."""
        from fizgig.minimax.rotation_ft import MasterStore, build_bf16_master_h3
        include = ("token_refiner",)
        est = (len(self.subset) if self.subset else self.n_blocks) * 0.771 + 0.4
        mode = str(self.driver.options.get("ft_master") or where or "auto").lower()
        if mode == "auto":
            try:
                from fizgig.utils.capabilities import _available_ram_gb
                avail, _ = _available_ram_gb()
            except Exception:
                avail = None
            mode = "disk" if (avail is not None and est > 0.40 * avail) else "ram"
            logger.info("[ft-master] auto -> %s (master ~%.1f GB vs %.0f GB RAM available)", mode, est,
                        avail if avail is not None else -1)
        if mode == "disk":
            cd = next((getattr(d, "cache_directory", None) for d in self.group.datasets
                       if getattr(d, "cache_directory", None)), None)
            base = os.path.dirname(str(cd).rstrip("/\\")) if cd else (scratch_dir or ".")
            self.scratch = os.path.join(base, "ft-scratch-" + hashlib.sha1(
                os.path.abspath(scratch_dir or ".").encode()).hexdigest()[:8])
            if os.path.isdir(self.scratch):
                shutil.rmtree(self.scratch, ignore_errors=True)       # a fresh run = fresh training state
            master = MasterStore(self.src, self.scratch, block_subset=self.subset, include_prefixes=include)
        else:
            master = build_bf16_master_h3(self.src, block_subset=self.subset, include_prefixes=include)
        self.rot = H3FTRotator(self.dit.blocks, master, key_prefix="blocks", device=str(self.device),
                               block_subset=self.subset)
        return est, mode

    def start_always(self):
        ref = getattr(self.dit, "token_refiner", None)
        if ref is not None and self.driver.options.get("train_token_refiner") == "1":
            return self.rot.activate_always("token_refiner", ref)
        if ref is not None:
            logger.info("[h3-ft] text token refiner frozen (tick 'Train the text token refiner' to include it - off "
                        "by default, as for LoRA runs)")
        return 0

    always_label = "token_refiner (when ticked)"

    # ---- the window plan ---------------------------------------------------------------------------------------------
    def plan(self, free_gb, allow_stream=True, mp=None, max_parts=0, held_grads=False, state_factor=0.0):
        """plan_h3_ft_windows over usable = free + the NF4 trunk already resident - 1.5, less a clip dataset's
        activation reserve (the stills calibration has no idea a 56-frame clip adds ~2.3 GB a step). held_grads (a DOP
        fine-tune, the shared trainer's argument): H3's window sizes are its own calibration and aren't re-planned for
        it; taken so the call matches families/ft.py's. state_factor likewise: H3 fine-tunes plan for Adafactor
        (no plans_optimizer_state, so the trainer keeps it)."""
        from fizgig.minimax.rotation_ft import ft_clip_activation_gb, plan_h3_ft_windows
        act = margin = 0.0
        if self.n_clip_items:
            from fizgig.minimax.common import _max_clip_act_item
            lt, smp = _max_clip_act_item(self.group)
            act, margin = ft_clip_activation_gb(lt, smp)
            if act > 0:
                logger.info("[h3-ft] clip activations ~%.1f GB + %.1f GB fragmentation margin (%d-latent-frame clips "
                            "at %.2f MP) reserved before window sizing", act, margin, lt, smp)
        usable = free_gb + 0.21 * self.n_blocks - 1.5 - act - margin
        windows, stream, why = plan_h3_ft_windows(usable, subset=self.subset, n_blocks=self.n_blocks,
                                                  allow_stream=allow_stream, max_parts=max_parts)
        self.stream = bool(stream)
        return windows, stream, why, usable

    def cycle_len(self):
        return len(self.subset) if self.subset else self.n_blocks

    # ---- rotation ----------------------------------------------------------------------------------------------------
    def _window_gb(self, spec):
        from fizgig.minimax.rotation_ft import H3_COMPONENT_GB_PER_BLOCK
        span = self.subset if self.subset else range(self.n_blocks)
        total = 0.0
        for e in spec:
            if isinstance(e, str):
                total += H3_COMPONENT_GB_PER_BLOCK.get(e, 0.31) * len(list(span))
            else:
                p, lo, hi = e
                total += H3_COMPONENT_GB_PER_BLOCK.get(p, 0.31) * sum(1 for b in span if lo <= b <= hi)
        return total

    def resident_blocks(self, spec):
        span = set(self.subset) if self.subset else set(range(self.n_blocks))
        res = set()
        for e in spec:
            res |= span if isinstance(e, str) else {b for b in span if e[1] <= b <= e[2]}
        return res

    def rotate(self, want):
        """The old boundary, in its order: deactivate the outgoing window, defrag if the incoming one would not fit
        (non-streaming), rescope the ring (evicting the streamed set first), then activate."""
        from fizgig.minimax.common import park_dit_to_cpu, restore_parked_dit
        from fizgig.utils.device import plannable_free_vram
        want = list(want)
        if want == list(self.rot.active):
            return 0
        if self.rot.active:
            self.rot.deactivate(list(self.rot.active))
            gc.collect()
            torch.cuda.empty_cache()
        if not want:
            return 0
        free, need = plannable_free_vram(), self._window_gb(want) + 2.0
        if free < need and not self.stream:
            logger.info("[h3-ft] %.1f GB free before activating the next window (needs ~%.1f) - defragmenting via a "
                        "full park/restore round-trip.", free, need)
            park_dit_to_cpu(self.dit)
            gc.collect()
            torch.cuda.empty_cache()
            restore_parked_dit(self.dit, self.device, 0)
            logger.info("[h3-ft] post-defrag: %.1f GB free", plannable_free_vram())
        self._rebuild_ring(want)
        return self.rot.activate(want)

    def _rebuild_ring(self, spec):
        """The old _ft_rebuild_ring: the NF4 H2D ring scoped to the frozen out-of-window blocks, rebuilt (with a
        ring-aware park/restore defrag) at every rotation - see minimax/trainer.py for the field history."""
        if not self.stream:
            return
        from fizgig.minimax.h3_nf4_h2d_offload import H3NF4H2DOffloader, bind_block_packed_to
        from fizgig.minimax.common import format_block_spec, park_dit_to_cpu
        dit, device = self.dit, self.device
        old = self._ring
        if old is not None:
            park_dit_to_cpu(dit)
            old.release()
            self._ring = None
            dit._h2d_offloader = None
            for blk in dit.blocks:
                blk._h2d_offloader = None
            old = None
            gc.collect()
            torch.cuda.empty_cache()
            for cname, child in dit.named_children():
                if cname != "blocks":
                    child.to(device)
        resident = self.resident_blocks(spec)
        streamed = [i for i in range(len(dit.blocks)) if i not in resident]
        ring = H3NF4H2DOffloader(dit.blocks, streamed, device)
        ring.move_static_weights_to_gpu()
        ring.prepare()
        gc.collect()
        torch.cuda.empty_cache()
        for i in sorted(resident):
            bind_block_packed_to(dit.blocks[i], device)
        gc.collect()
        torch.cuda.empty_cache()
        dit._h2d_offloader = ring
        for blk in dit.blocks:
            blk._h2d_offloader = ring
        dit._swap_from = 0
        self._ring = ring
        logger.info("[h3-ft] streaming ring rescoped: %d resident block(s) [%s], %d streamed (~%.1f GB staged in RAM)",
                    len(resident), format_block_spec(sorted(resident)), len(streamed), ring.staged_gb)

    def trainable_params(self):
        return self.rot.trainable_params()

    def park(self):
        """Every window back to NF4 (the master complete) - before a save, a preview, or the end."""
        if self.rot.active:
            self.rot.deactivate(list(self.rot.active))
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def active(self):
        return list(self.rot.active)

    @property
    def master(self):
        return self.rot.master

    def params_in_blocks(self, block_ids):
        """The active window's weights in these blocks (h3blk_N) - a routed step freezes them."""
        want = {int(b.split("_")[1]) for b in block_ids if str(b).startswith("h3blk_")}
        return [lin.weight for key, lin in self.rot._targets(list(self.rot.active)) if int(key.split(".")[1]) in want]

    # ---- the checkpoint ----------------------------------------------------------------------------------------------
    def save(self, path, meta):
        from fizgig.minimax.rotation_ft import save_full_checkpoint_h3
        save_full_checkpoint_h3(self.rot, self.src, path, extra_metadata=meta)
        return len(self.rot.touched), None

    def cleanup(self):
        if self.scratch and os.path.isdir(self.scratch):
            shutil.rmtree(self.scratch, ignore_errors=True)


def source_unfit_reason(path):
    """Why a file cannot be H3-fine-tuned, or None: the rotation FT needs the pre-quantised int8 ConvRot checkpoint
    (the master dequantises from its codes)."""
    from fizgig.minimax.common import is_pruned_checkpoint
    if not is_pruned_checkpoint(path):
        return ("is not the pre-quantized int8 checkpoint (minimax_h3_*_pruned_int8_convrot.safetensors) - the "
                "fine-tune builds its master from its ConvRot codes")
    return None
