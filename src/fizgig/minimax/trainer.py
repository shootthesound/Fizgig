"""MiniMax H3 — image-only training core: flow-matching loss + timestep sampling.

The heart of the trainer, isolated so it's headless-testable with the tiny model (no GPU,
no 66 GB base, no 32 B text encoder). The full LoRA/rotating-FT wiring, caching and GUI come
later; this pins the maths of one training step.

Flow / sign convention (matched to ComfyUI's comfy/ldm/minimax/model.py):
  x0 = clean latent, noise ~ N(0,1), sigma in (0,1) the noise level.
  noised = (1 - sigma)*x0 + sigma*noise            (sigma 0 = clean, 1 = pure noise)
  t = 1 - sigma                                     the "cleanness" fed to the time embedder
  the DiT's raw video_out predicts (x0 - noise)     (the reference NEGATES it to get the
                                                     sampler's velocity noise - x0)
So the training target for the model's output is `x0 - noise`.
"""

import argparse
import contextlib
import gc
import hashlib
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F

from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)

VIDEO_SIGMA_SHIFT_TRAIN = 12.0     # H3's video shift — also the reference TRAINING density

# Where "low noise" stops and the noisy half begins. The GUI defines its clean-end percentage
# against this same 0.5, so the box and --highnoise_lr_scale always mean the same boundary; the
# two must move together if either ever moves.
MINIMAX_LOWNOISE_SIGMA = 0.5

# What a RETIRED category trains at in anchor mode — the Krea per-image ladder's tested floor.
# One constant, not a knob: "anchor" is legible, "0.085 vs 0.12" is not.
ANCHOR_LR_SCALE = 0.1

# Identity-first phase 1 trains at this fraction of the Learning Rate box (Peter, 11 Aug). Phase
# 1 places the identity on a near-zero adapter, where a full-size Adam stride does the most
# damage and the least good; phase 2 then gets the full rate from a sensible starting point.
_P1_LR_SCALE = 1.0 / 3.0

# LoRA targets the transformer blocks' ATTENTION + MLP Linears (+ the 2-block text refiner).
# The fp32 patch/head IO layers are left alone (wrapping them clashes fp32-base vs bf16-adapter).
#
# `adaln_proj` is per-checkpoint (matching the reference trainer on the pruned build):
#   * FULL bf16 model ([96768, 2688]): EXCLUDED — the up-matrices are 96768-out (6x qkv),
#     soaked up the largest share of LoRA capacity, and ComfyUI's pruned inference builds
#     drop every adaln key anyway (~50% likeness until excluded, real run).
#   * PRUNED model ([96768, 8]): INCLUDED — deploy-consistent, and what ai-toolkit trains.
#     It carries ~45% of all weight movement in a matched reference epoch, and it is the
#     timestep-conditioned modulation, so starving it reads from outside as "the mid/low-noise
#     range never gets trained". Train it at the REQUESTED rank: capping to min(in,out)=8 cost
#     73% of its learning (see the no-cap note in networks/lora.py). An epoch-1 melt was once
#     pinned on these adapters (tests/diag_epoch1_ab.py) but the distortion predated adaln and
#     persisted without it — the real culprit was the training density (see sample_sigmas).
DEFAULT_INCLUDE_PATTERNS = [r"blocks\.\d+\.attn\..*", r"blocks\.\d+\.mlp\..*",
                            r"token_refiner\.blocks\..*"]
# NOTE: the per-block AdaLNs only — NOT `final_layer.adaln_proj`. The reference trains 258
# modules and we were training 259; the extra one was added here by symmetry, not by matching
# them. It also happened to carry our single highest per-element drift after a matched epoch
# (0.0133 vs their 0.0068 max), so it was contributing noise rather than capability.
PRUNED_INCLUDE_PATTERNS = DEFAULT_INCLUDE_PATTERNS + [r"blocks\.\d+\.adaln_proj\..*"]


def clip_fallback_frames(frames: int) -> int:
    """Next shorter clip length to retry with after a clip preview fails (in practice, OOM).

    Halves the request and snaps down onto the model's 17n+5 grid, so a 141-frame OOM retries
    at 56, then 22, and only then gives up on clips: 141 -> 56 -> 22 -> 1.

    Stepping down rather than collapsing straight to a still matters because a still is the
    MOST out-of-distribution render H3 has — ComfyUI cannot even construct one (its video
    latent floor is 2 frames) and the trained band is ~124-362. Dropping a clip run to stills
    on one OOM quietly replaces the previews being judged with the least trustworthy kind,
    for the rest of the run. A shorter clip is still a clip.
    """
    half = int(frames) // 2
    if half < 22:                      # below the first real grid point above a keyframe pair
        return 1
    return half - (half - 5) % 17      # largest 17n+5 value <= half


_VRAM_LOG = None            # decided on first call: small cards log, big cards stay quiet


def vram_line(tag: str):
    """One honest line of VRAM accounting. `allocated` is live tensors; `reserved` is what
    torch's allocator holds from the driver (the gap is fragmentation — inactive-split is the
    pinned part empty_cache cannot return); driver free is what everyone else sees. The
    16 GB hunt kept stalling because each theory only ever saw ONE of these numbers.

    Logs on cards under 20 GB (where the numbers are the diagnosis when a report comes in);
    quiet on bigger cards. FIZGIG_VRAM_LOG=1/0 forces it on/off anywhere."""
    global _VRAM_LOG
    if not torch.cuda.is_available():
        return
    if _VRAM_LOG is None:
        _env = os.environ.get("FIZGIG_VRAM_LOG", "").strip()
        if _env in ("1", "0"):
            _VRAM_LOG = _env == "1"
        else:
            try:
                _VRAM_LOG = torch.cuda.mem_get_info()[1] / 2**30 < 20.0
            except Exception:
                _VRAM_LOG = False
    if not _VRAM_LOG:
        return
    try:
        s = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        logger.info(f"[vram:{tag}] allocated {torch.cuda.memory_allocated()/2**30:.2f} / "
                    f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GB "
                    f"(inactive-split {s.get('inactive_split_bytes.all.current', 0)/2**30:.2f}), "
                    f"driver free {free/2**30:.2f} of {total/2**30:.2f} GB")
    except Exception:
        pass


def park_dit_to_cpu(dit):
    """Whole-DiT park that REUSES its CPU arena across cycles.

    ``dit.to("cpu")`` allocates ~9 GB of fresh CPU tensors every preview and frees them on
    restore — and the Windows heap keeps the freed pages, compounding: measured ~5 GB of RSS
    retained after ONE park/restore cycle with every tensor reference clean (16 GB 4090 field
    case: RAM locked at 31/32 GB after the first preview, 12.4 GB before it). The arena is
    allocated once, written into on every park, and deliberately KEPT across restores — the
    same storage is reused forever, so RSS plateaus at baseline + one packed base instead of
    climbing. Assigning ``.data`` sidesteps bnb Params4bit's ``.to()`` override, so NF4 stays
    packed and quant_state never moves (it's small and the parked weights are never computed
    with). Restore stays ``restore_parked_dit`` — Module.to(device) repoints ``.data`` at a
    fresh CUDA copy while the arena keeps its CPU tensor for the next park."""
    park_dit_partial(dit, need_gb=None)


def park_dit_partial(dit, need_gb=None):
    """Arena park, tail-blocks-first, stopping once ``need_gb`` has been freed.

    Parking the WHOLE base to fit a ~7 GB decode frees ~9.6 GB when ~3 were missing — and on
    a WDDM card every unnecessary gigabyte moved is more driver paging churn and a slower
    restore (field: post-preview steps fell from 1.0 to 24.7 s/it on the 16 GB 4090). Blocks
    park from the tail (matching the swap order — under swap they are already on CPU and are
    skipped as not-cuda); the non-block modules go last and only if the blocks were not
    enough. need_gb=None parks everything (the fit-the-text-encoder case)."""
    arena = getattr(dit, "_park_arena", None)
    if arena is None:
        arena = {}
        dit._park_arena = arena
    freed = 0.0
    target = float("inf") if need_gb is None else max(0.0, float(need_gb)) * 1e9
    _alloc0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Under H2D swap, the swapped blocks' qdata/wscale LOOK cuda-resident to the walk below
    # (they point at ring staging views), and evicting one view resize_(0)s the whole shared
    # slot storage — the next copy from that storage is a sticky CUDA 'invalid argument'
    # that killed the preview decode and the first training step (24 GB card, swap 6, 56-
    # frame preview at 4.9 GB free; step 0/6900 died in the rebuilt offloader). Their real
    # masters are already on CPU in the offloader's pinned flats: re-bind, and the walk
    # skips them exactly as its own "already on CPU" rule intends.
    _off = getattr(dit, "_h2d_offloader", None)
    if _off is not None:
        _off.unbind_to_cpu()

    def _park_module(mod, prefix):
        nonlocal freed
        for name, t in (list(mod.named_parameters(prefix=prefix))
                        + list(mod.named_buffers(prefix=prefix))):
            if not t.data.is_cuda:
                continue
            slot = arena.get(name)
            if slot is None or slot.shape != t.data.shape or slot.dtype != t.data.dtype:
                slot = torch.empty_like(t.data, device="cpu")
                arena[name] = slot
            slot.copy_(t.data)
            # Free the STORAGE, not just our reference: after a training epoch something
            # (unnamed — not autograd-saved, not a python-visible holder; found only as
            # census orphans) still references the old weight tensors, so `t.data = slot`
            # alone freed nothing and the restore then uploaded duplicates. resize_(0)
            # releases the bytes under every tensor sharing the storage — phantom holders
            # keep a zero-byte husk. The model itself never touches the old storage again:
            # its params now point at the arena, and restore allocates fresh.
            _storage = t.data.untyped_storage()
            t.data = slot
            try:
                _storage.resize_(0)
            except Exception:
                pass
            freed += slot.numel() * slot.element_size()

    def _walk():
        for i in range(len(dit.blocks) - 1, -1, -1):
            if freed >= target:
                return
            _park_module(dit.blocks[i], f"blocks.{i}")
        if freed < target:
            for cname, child in dit.named_children():
                if cname == "blocks":
                    continue
                if freed >= target:
                    return
                _park_module(child, cname)
            _park_module(dit, "")      # any direct params/buffers on the root

    _walk()
    # The verdict: evicting a weight only frees its VRAM if nothing else references the old
    # tensor. Field case (16 GB 4090): the epoch-1 park evicted 6.3 GB and allocated fell
    # 0.11 — the restore then uploaded DUPLICATES and the run drowned. When that happens,
    # census the stale tensors (they are module-orphans now) and name their holders.
    if torch.cuda.is_available() and freed > 1e9:
        gc.collect()
        torch.cuda.empty_cache()
        _dropped = _alloc0 - torch.cuda.memory_allocated()
        if _dropped < freed * 0.5:
            logger.warning(f"[park] evicted {freed/1e9:.2f} GB of weights but allocated only "
                           f"fell {max(_dropped, 0)/1e9:.2f} GB — something still references "
                           f"the old GPU tensors. Census:")
            try:
                from fizgig.utils.device import report_cuda_leak
                report_cuda_leak("park-failed", threshold_gb=0.0, orphan_min_mb=24)
            except Exception:
                pass
            _reg = globals().get("_SAVED_TENSOR_REG")
            if _reg:
                from collections import Counter
                _tot = sum(b for _, b, _ in _reg.values())
                logger.warning(f"[audit] {len(_reg)} saved tensors still live "
                               f"({_tot/2**30:.2f} GB cuda) — top save sites:")
                _c = Counter()
                for shape, b, stk in _reg.values():
                    if b:
                        _c[stk] += b
                for stk, b in _c.most_common(6):
                    logger.warning(f"[audit]   {b/2**30:.2f} GB saved at {stk or '(small)'}")


def restore_parked_dit(dit, device, n_swap: int):
    """Bring a whole-DiT park (``dit.to("cpu")``) back WITHOUT materializing the full base on
    the GPU.

    ``dit.to(device)`` moves all 50 blocks up before ``enable_block_swap`` can re-park its
    tail — invisible on a card that briefly fits the whole ~21 GB int8 base (32 GB), a
    guaranteed OOM on the tier that never could (16 GB — found by the VRAM sim, where the
    restore's failure was then mis-blamed on the clip render that had already succeeded).
    Move only the resident head and the non-block modules; the swapped tail stays on CPU,
    which is exactly where ``enable_block_swap`` expects to find it (the H2D offloader
    re-pins from wherever the sources live, and classic parking re-parks CPU→CPU for free).
    """
    n = max(0, min(int(n_swap or 0), len(dit.blocks) - 2))
    if n <= 0:
        dit.to(device)
        return
    _old = getattr(dit, "_h2d_offloader", None)
    if _old is not None:
        # Free the stale GPU ring BEFORE refilling the card — enable_block_swap would release
        # it too, but only after the head blocks are already up, and on a tight card that
        # ordering is the difference between fitting and not.
        _old.release()
        dit._h2d_offloader = None
        for _blk in dit.blocks:
            if hasattr(_blk, "_h2d_offloader"):
                _blk._h2d_offloader = None
        # Drop the reference before enable_block_swap builds the fresh ring below —
        # holding it doubled the CPU staging transient at every preview restore
        # (audit, 25 Aug; twin of the fix in enable_block_swap's own teardown).
        _old = None
    keep = len(dit.blocks) - n
    for i, blk in enumerate(dit.blocks):
        blk.to(device if i < keep else "cpu")
    for name, child in dit.named_children():
        if name != "blocks":
            child.to(device)
    for _pn, _p in list(dit._parameters.items()):
        if _p is not None:
            _p.data = _p.data.to(device)
    for _bn, _b in list(dit._buffers.items()):
        if _b is not None:
            dit._buffers[_bn] = _b.to(device)
    dit.enable_block_swap(n)              # rebuilds the H2D ring / re-parks, mode preserved


def parse_block_spec(spec, num_blocks: int = None):
    """"3-12, 14-15, 22,27,31-33" -> [3,4,...,12,14,15,22,27,31,32,33].

    Ranges and singles, comma-separated, whitespace anywhere. Returns sorted unique indices.
    Raises ValueError on anything it cannot read — a typo here must stop the run, not silently
    train a different set of blocks than the one being tested.

    num_blocks, when given, bounds-checks: an out-of-range index would otherwise just match
    nothing and quietly shrink the experiment.
    """
    text = str(spec if spec is not None else "").strip()
    if not text:
        raise ValueError("no blocks given")
    out = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue                       # tolerate a trailing or doubled comma
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"range runs backwards: {chunk!r}")
            out.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", chunk):
            out.add(int(chunk))
        else:
            raise ValueError(f"cannot read {chunk!r} — use numbers and ranges, "
                             f"e.g. '3-12, 14-15, 22, 31-33'")
    if not out:
        raise ValueError("no blocks given")
    if num_blocks is not None:
        bad = sorted(i for i in out if i >= num_blocks)
        if bad:
            raise ValueError(f"block(s) {bad} do not exist — this model has {num_blocks} "
                             f"(0-{num_blocks - 1})")
    return sorted(out)


def format_block_spec(indices):
    """[3,4,5,7] -> "3-5,7" — the canonical form recorded in metadata and logged."""
    if not indices:
        return ""
    runs, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def snap_ft_stop(n, cycle, offset, local_max):
    """The effective (cumulative) FT retirement epoch: (value, kind).

    Stop epochs are CUMULATIVE across pause/resume, exactly like checkpoint numbering —
    on a continuation the flag means the same calendar epoch it meant in the original
    run, so "pause at 12, set the stop to 12, resume" gives an audio-only continuation
    instead of 12 more epochs of photos. Cycle-snapping happens in LOCAL space (each
    process's rotation windows are what must see the identical data mix), then converts
    back to cumulative. kind: "off" (0/disabled), "past" (already retired when this run
    starts — value returned verbatim), "snapped"/"exact" (fires mid-run), "never"
    (at or past this run's end). Pure so the table is pinnable without a training run."""
    n = max(0, int(n or 0))
    if not n:
        return 0, "off"
    local = n - int(offset or 0)
    if local <= 0:
        return n, "past"
    snapped = ((local + cycle - 1) // cycle) * cycle
    if snapped >= local_max:
        return snapped + int(offset or 0), "never"
    return snapped + int(offset or 0), ("snapped" if snapped != local else "exact")


def plan_ft_modality_routing(n_blocks, photo_blocks, audio_blocks,
                             n_photo, n_voice, n_clip, explicit_subset=None,
                             clip_blocks=None):
    """The fine-tune's modality-routing plan, as pure data: (cycle_subset, routes).

    cycle_subset: sorted block list the rotation cycle should span, or None for the full
    model — the UNION of what each modality present in the dataset needs (photos -> the
    likeness set when given, voice -> the audio zone, clips -> clip_blocks when given,
    full model otherwise). clip_blocks landed 29 Aug from a field result: an overnight
    video run confined to the likeness blocks worked, so "clips -> full model" is now the
    fallback, not the law — the GUI's "Restrict video to likeness blocks" tickbox passes
    the likeness set here, and unticking it (or CLI runs without --clip_blocks) keeps the
    whole-model behaviour.
    routes: {"photo": set|None, "voice": set|None, "clip": set|None} — the per-batch
    confinement set for each modality, or None when that modality may train the whole
    span (absent from the dataset, no set configured, or the span already sits inside
    its set — the caller's freeze list then comes out empty anyway; None here keeps the
    intent legible in logs).

    An explicit --finetune_blocks subset wins: it is returned verbatim and all routes are
    None (the validated manual workflow — the A/B that produced the 34-49 rule was run
    exactly this way). Kept as a module-level pure function so the truth table is pinnable
    without a training run."""
    routes = {"photo": None, "voice": None, "clip": None}
    if explicit_subset is not None:
        return sorted(explicit_subset), routes
    pb = set(parse_block_spec(photo_blocks, n_blocks)) if photo_blocks else None
    aud = set(parse_block_spec(audio_blocks, n_blocks)) if audio_blocks else None
    cb = set(parse_block_spec(clip_blocks, n_blocks)) if clip_blocks else None
    full = set(range(n_blocks))
    union = set()
    if n_photo:
        union |= (pb if pb else full)
    if n_voice:
        union |= (aud if aud else full)
    if n_clip:
        union |= (cb if cb else full)
    if not union:
        union = full
    span = union
    if pb is not None and n_photo and not span.issubset(pb):
        routes["photo"] = pb
    if aud is not None and n_voice and not span.issubset(aud):
        routes["voice"] = aud
    if cb is not None and n_clip and not span.issubset(cb):
        routes["clip"] = cb
    return (sorted(union) if union != full else None), routes


def restrict_patterns_to_blocks(patterns, block_spec, num_blocks: int = None):
    """Narrow `blocks.N.*` patterns to a block selection. Non-block patterns pass through.

    H3 is 50 IDENTICAL blocks with no published map of what each one does, so training a subset is
    an experiment, not a recipe — this exists to make that experiment cheap to run. The token
    refiner is deliberately never narrowed: it is text-side (where a trigger token gets shaped),
    it is 8 of 258 modules, and holding it constant keeps two selections comparable to each other
    rather than confounding the block question with a conditioning change.

    Applied ON TOP of the per-checkpoint pattern list rather than replacing it, so the pruned vs
    bf16 AdaLN decision stays in exactly one place.
    """
    idx = parse_block_spec(block_spec, num_blocks)
    alt = "|".join(str(i) for i in idx)
    out = []
    for p in patterns:
        if p.startswith(r"blocks\.\d+"):
            out.append(p.replace(r"blocks\.\d+", rf"blocks\.(?:{alt})", 1))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# VRAM planner — resolves "auto" block swap + gradient checkpointing from the card's actual
# free VRAM and the run's real token load (bucket megapixels x batch). Simpler than Krea 2's:
# one quant mode (NF4), batch is 1, no preview co-residency.
# ---------------------------------------------------------------------------
# Measured anchors (5090, real 33B, rank 16, ~0.2 MP batch 1 — GPU validation pass, 4 Aug):
#   no swap, no ckpt : resident 17.6, step peak 22.7  (overhead ~5.1)
#   no swap, ckpt    : resident 17.5, step peak 18.3  (overhead 0.9 — and only ~+0.1 s/step)
#   swap 16 + ckpt   : resident 11.9 (0.34 GB/block), steady 12.8, step peak 19.3 — the swap
#                      path carries a ~7.4 GB backward transient (checkpoint recompute segments
#                      held by the engine), which the planner must budget on top of residency.
#
# Re-measured 6 Aug on the SHIPPED default (int8 base, LoKR factor 8 + adamw, AdaLN off), because
# those anchors were taken with a rank-16 LoRA on adamw8bit — an adapter of ~0.4 GB against the
# ~3.1 GB the defaults now carry, so the planner was budgeting for a run nobody does:
#   resident         : base 21.07 + LoKR weights 0.63 + fp32 Adam state 2.50 = 24.20 GB
#   0.23 MP  no ckpt : 29.18      |  ckpt: 24.39
#   0.50 MP  no ckpt : OOM (>31)  |  ckpt: 24.47
#   0.98 MP  no ckpt : OOM        |  ckpt: 24.56
# Two things fall out. Un-checkpointed really does scale hard (0.5 MP OOMs a 32 GB card, so
# forcing ckpt on there is correct), and CHECKPOINTED IS ALMOST FLAT — 1 MP costs 0.17 GB more
# than 0.23 MP, not four times as much. Hence _ACT_GB_CKPT below.
_RESIDENT_GB = 17.5          # full bf16 model, NF4 resident (measured 17.3-17.6)
# The PRUNED checkpoint drops the full-width AdaLN (~40% of the model's weight mass) for a curve
# table, so the same NF4 pass lands far smaller: ~20.1 B params quantized -> ~10.1 GB, plus the
# unquantized remainder. Estimated from the file's own tensor census, not yet GPU-measured, so
# it carries margin.
# MEASURED 6 Aug (was 11.0, estimated from the file's tensor census): the pruned checkpoint
# decoded and re-quantized to NF4 sits at 10.46 GB resident, and a checkpointed step peaks at
# 13.46 / 13.56 / 13.63 GB at 0.23 / 0.50 / 0.98 MP — flat in megapixels, exactly like int8.
# Un-checkpointed it is 18.27 / 23.52 / OOM. Now that Auto can CHOOSE this mode, the number it
# chooses against had to stop being a guess.
_RESIDENT_PRUNED_GB = 10.5
# int8 base (base_quant=int8, the reference's own storage): the 200 block linears stay 1 byte
# per param instead of NF4's 0.5, and the refiner/AdaLN load dense — ~19.3 + ~1.5 GB.
_RESIDENT_INT8_GB = 21.0
# HQQ 4-bit g16 (rintic-13, #102): 0.5 B/param of codes + two bf16 group vectors at 1/16 =
# 0.75 B/param against NF4's ~0.52 (block-64 absmax, double-quantized) — ~45% more resident
# for the same quantized mass. PRUNED figure MEASURED (2 Sep, 5090, pruned int8 file
# decoded to HQQ): ~15.5 GB process right after load incl. the 0.6 GB adapter, ~22 GB
# steady in training at 0.25 MP with no checkpointing, 1.45 it/s vs NF4's 2.6 it/s on the
# same run (PyTorch-path dequant in forward AND backward; hqq's CUDA kernel is not
# installed). The bf16-file figure is still the B/param ratio over _RESIDENT_GB — no bf16
# checkpoint on the bench box. HQQ stays explicit-pick only (Auto never chooses it).
_RESIDENT_HQQ_GB = 22.0
_RESIDENT_HQQ_PRUNED_GB = 15.0
# int8 dequantizes a bf16 weight per matmul (fc1 is 28672x5376 = 308 MB). A few are live at
# once, but they are NOT retained for backward — _Int8RotLinearFn recomputes the weight in its
# own backward, so the cost is a handful of transients rather than one per layer. (Before that
# custom backward, autograd saved every one and a 0.25 MP run OOM'd the moment the planner
# turned checkpointing off: measured 0.45 GB of retained weight over 12 test linears against
# 0.12 GB now, and the real DiT has 200.)
_INT8_TRANSIENT_GB = 1.0
_PER_BLOCK_GB = 0.34         # one parked block's GPU share (measured: (17.5-11.9)/16)
_ACT_GB_NOCKPT = 5.5         # step overhead at 0.25 MP batch 1, no checkpointing (measured 4.98)
# Checkpointed memory is very nearly FLAT in megapixels — that is the whole point of recompute,
# and the old 2.0 (which then got multiplied by the MP scale) modelled it as growing four times
# faster than it does. Measured on the shipped default (int8 base, LoKR 8 + adamw, 6 Aug 2026),
# peak above the resident 24.20 GB:
#     0.23 MP  0.19 GB        0.50 MP  0.27 GB        0.98 MP  0.36 GB
# i.e. ~0.15 + 0.2 x scale. 0.5 keeps a wide margin at every size and still leaves the planner
# free to say "no swap" where the card genuinely fits — the old value invented 25 blocks of swap
# for a 1 MP run that actually peaks at 24.6 GB, costing ~4x the step time for nothing.
_ACT_GB_CKPT = 0.5           # step overhead at 0.25 MP batch 1, checkpointed (measured 0.19)
_SWAP_TRANSIENT_GB = 7.5     # extra backward-time peak whenever swap is active (measured 7.4 @ n=16)
# H2D-only streaming (#73) keeps ring_size blocks resident at once (~0.8 GB at ring 2) —
# inside this transient budget. Re-measure only if diag_h2d_speedup shows the peak moving.
_H2D_PER_BLOCK_GB = 0.39     # one streamed int8 block's VRAM share (checkpoint header: 0.385)
_H2D_TRANSIENT_GB = 2.0      # ring (2 x 0.39) + margin — validated on a simulated 16 GB card
                             # at BOTH 0.25 MP and 1 MP buckets, swap 40 (~1.85 s/it at 1 MP)
_MIN_INT8_H2D_FREE_GB = 13.5  # INT8 H2D was validated at ~14.2 GB free (16 GB-class cards).
                              # A 12 GB card tops out below this and must stay on NF4: letting
                              # the streaming arithmetic alone approve 38-40 streamed INT8
                              # blocks made Auto pick a larger base that crashed before step
                              # one (@mabseyuk's 5070 field report). Explicit int8 remains
                              # available for anyone benchmarking new floors.
_RESERVE_GB = 1.5            # display / allocator / fragmentation headroom
# Skipping checkpointing has to EARN it. Measured on H3, recompute costs ~0.1 s/step and saves
# ~5 GB — so choosing "no checkpointing" on a thin margin trades five gigabytes of headroom for
# a tenth of a second. Peter's 6 Aug run picked it with 0.37 GB of predicted margin (needed
# 32.13 of 32.5 GB free) and then ran at 4-6 s/step instead of ~1: on Windows the driver spills
# to system RAM rather than OOMing, so an over-tight plan does not fail, it just crawls, with
# nothing in the log to say why. The un-checkpointed peak is also the one that scales with
# megapixels, so a plan that barely fits at one bucket size will not fit at the next.
_NOCKPT_MARGIN_GB = 3.0      # extra headroom demanded before skipping recompute


def adapter_param_count(dit_path: str, include_patterns, network_type: str = "lora",
                        network_dim: int = 16, lokr_factor: int = 8,
                        train_blocks: str = None) -> int:
    """Trainable parameter count, read from the checkpoint HEADER — no model, no GPU.

    The VRAM plan runs before the DiT is built, so the shapes come from the safetensors header
    (which is just JSON at the front of the file). That keeps this exact rather than an
    architecture guess: it sees the real targeted Linears for whichever checkpoint is loaded,
    respects include_patterns and the Blocks to Train restriction, and works the same on the
    pruned and full builds.
    """
    import json
    import re as _re
    import struct
    try:
        with open(dit_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
    except Exception:
        return 0

    pats = list(include_patterns or [])
    if train_blocks:
        n_blocks = len({int(m.group(1)) for k in hdr
                        for m in [_re.match(r"blocks\.(\d+)\.", k)] if m} or {0})
        pats = restrict_patterns_to_blocks(pats, train_blocks, n_blocks)
    if not pats:
        return 0
    rx = [_re.compile(p) for p in pats]

    total = 0
    for key, ent in hdr.items():
        if key == "__metadata__" or not key.endswith(".weight"):
            continue
        shape = ent.get("shape") or []
        if len(shape) != 2:                     # Linears only, as create_modules wraps
            continue
        name = key[:-len(".weight")]
        if not any(r.search(name) for r in rx):
            continue
        out_dim, in_dim = int(shape[0]), int(shape[1])
        if str(network_type).lower() == "lokr":
            from fizgig.networks.lora import factorization   # local: avoids a circular import
            a, _c = factorization(out_dim, int(lokr_factor))
            b, _d = factorization(in_dim, int(lokr_factor))
            total += a * b + _c * _d            # w1 (a,b) + w2 (c,d)
        else:
            total += int(network_dim) * (in_dim + out_dim)
    return total


def adapter_vram_gb(params: int, optimizer_type: str = "adamw8bit") -> float:
    """GB the adapter holds for the WHOLE run: bf16 weights + optimizer state.

    Not a rounding error at these sizes. LoKR factor 8 on H3 trains ~313 M parameters against a
    rank-16 LoRA's ~77 M, and the state dtype widens the gap again: fp32 Adam keeps two 4-byte
    moments per parameter where the 8-bit optimizers keep two 1-byte ones. LoKR + adamw is
    ~3.1 GB against ~0.4 GB for the rank-16 + adamw8bit configuration the original anchors were
    measured on — which is why planning without this term was planning for a run nobody does.

    Gradients are deliberately NOT counted here. They are transient, and fused AdamW frees them
    per parameter as it steps, so they never all coexist: measured, a checkpointed step peaks
    only 0.19 GB above this figure even though the gradients would be 0.63 GB if they were all
    live at once. They belong in the activation term's margin, not in the resident one.

    Verified against a real step (6 Aug 2026): base 21.07 + weights 0.63 + fp32 state 2.50 =
    24.20 GB resident, exactly what this returns for 313.1 M parameters on adamw.
    """
    key = (optimizer_type or "adamw8bit").lower()
    n_states = 1 if "lion" in key else 2        # Lion keeps momentum only
    state_bytes = (1 if "8bit" in key else 4) * n_states
    return params * (2 + state_bytes) / 1e9     # bf16 weight + optimizer state


def plan_base_quant(free_gb: float, pruned: bool, mp: float = 0.25, adapter_gb: float = 0.0):
    """Pick the base quantisation AND the swap plan together -> (mode, blocks_to_swap, ckpt, why).

    Choosing a swap count from VRAM alone, with the quantisation already fixed, produces the
    worst available outcome on mid-range cards: the int8 base is ~21 GB, so a 24 GB card cannot
    hold it and the old CLASSIC swap parked 38 of 50 blocks on CPU — every one of them
    round-tripping PCIe every step, ~4x the step time. That trade DIED with the H2D-only
    streamer (#73, @rintic-13): int8 blocks are frozen, so they stream host->device only, on a
    copy stream that overlaps compute — measured on the real base at swap 40 (a simulated
    16 GB card, six epochs): ~1.35 s/it steady, where classic parking ran several times that.
    So int8 no longer has to fit to be picked.

    Order of preference:
      1. int8, no swap   — the most accurate base (~0.17% error against the reference's own
                           storage) with no PCIe cost at all.
      2. int8 + H2D swap — same accurate base; parked blocks stream one-way with prefetch.
                           This replaced "4-bit, no swap": a LoRA fitted on NF4's ~9.5%-
                           perturbed base spends capacity correcting error that will not exist
                           at inference, and the speed argument for accepting that is gone.
      3. 4-bit (+classic swap if even 11 GB doesn't fit) — the floor for cards the int8
                           residual footprint (~5.6 GB of non-streamed weights + activations)
                           genuinely cannot fit, and for the bf16 checkpoint (no int8 weights
                           to stream — H2D is ConvRot-specific).

    Only applies to a pruned int8 checkpoint — the bf16 file has no int8 weights to keep, so
    there is nothing to choose between.
    """
    if not pruned:
        n, c = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_GB, adapter_gb=adapter_gb)
        return "nf4", n, c, "bf16 checkpoint — NF4 is the only option"

    i_swap, i_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_INT8_GB,
                               transient_gb=_INT8_TRANSIENT_GB, adapter_gb=adapter_gb)
    if i_swap == 0:
        return "int8", i_swap, i_ckpt, "int8 fits with no block swap — the most accurate base"
    # The int8 streaming path was originally validated with 16 GB-class headroom (~14.2 GB
    # free) — @mabseyuk's 5070 crashed before step one on a 12 GB int8-streaming plan,
    # which is where this floor came from. That crash predates the v4.4.0 pin fallback and
    # ring hardening, and the SAME card now runs an EXPLICIT int8 pick at ~1.1 s/it with
    # 40 streamed blocks (#101) — but at ~15 GB of pinned system RAM, which a 16 GB-RAM
    # box cannot survive. So Auto keeps the conservative floor (nf4 stays on-card and
    # stages nothing) and the reason string hands big-RAM users the explicit escape hatch;
    # relaxing Auto itself is #101 and wants a RAM-aware gate plus measurement first.
    if free_gb < _MIN_INT8_H2D_FREE_GB:
        n_swap, n_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_PRUNED_GB,
                                   adapter_gb=adapter_gb)
        return ("nf4", n_swap, n_ckpt,
                f"{free_gb:.1f} GB free is below the tested int8-streaming floor "
                f"({_MIN_INT8_H2D_FREE_GB:.1f} GB) — using the smaller 4-bit base. "
                f"(A machine with 48 GB+ of system RAM can pick Base Precision: int8 "
                f"explicitly — the accurate base streams through the ring at this tier, "
                f"staging ~15 GB in pinned RAM)")
    # H2D-specific arithmetic — the classic anchors are WRONG for streaming and would refuse
    # cards that measurably work. Classic swap's 7.5 GB backward transient is engine-held
    # recompute segments of physically-moving blocks; H2D blocks never move — the transient is
    # the ring (2 x 0.39 GB) plus margin. And an int8 block frees _H2D_PER_BLOCK_GB = 0.39
    # (measured from the checkpoint header), not NF4's 0.34. Validated: a simulated 16 GB
    # card (14.2 GB free) ran swap 40 for six epochs at ~1.35 s/it, peak within budget.
    _need = (_ckpt_need_gb(mp, 1, _RESIDENT_INT8_GB, _INT8_TRANSIENT_GB, adapter_gb)
             + _H2D_TRANSIENT_GB)
    _h2d_swap = int((_need - free_gb) / _H2D_PER_BLOCK_GB + 0.999)
    # H2D staging lives in SYSTEM RAM — and on Windows so does the GPU itself: WDDM backs
    # GPU allocations with commit charge, so exhausting RAM makes the driver refuse even
    # tiny VRAM allocations ("CUDA error: out of memory" with headroom on the card). Field
    # case (16 GB 4090, 32 GB RAM): 31 staged blocks (~12 GB) + a preview decode parking
    # the whole base to CPU pegged RAM at 32 GB — pinning failed, then the first training
    # step died at latent.float(). The 5090's simulated-16GB validation never saw this
    # because that machine has RAM to spare. So the staging plus a working margin (parked-
    # base transient ~8 GB + WDDM commit headroom) must genuinely fit in AVAILABLE RAM, or
    # the accurate-base argument loses to the machine falling over: NF4 keeps everything on
    # the card and stages nothing.
    _stage_gb = _h2d_swap * _H2D_PER_BLOCK_GB
    _avail_ram = None
    _ram_short = False
    if 0 < _h2d_swap <= 40:
        try:
            import psutil
            _avail_ram = psutil.virtual_memory().available / 1e9
            _ram_short = _avail_ram < _stage_gb + 14.0
        except Exception:
            _ram_short = False
        if not _ram_short:
            return ("int8", _h2d_swap, True,
                    f"int8 with {_h2d_swap} blocks streamed H2D-only — the accurate base, and "
                    f"streaming (not parking) keeps the swap cheap")

    n_swap, n_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_PRUNED_GB,
                               adapter_gb=adapter_gb)
    if _ram_short:
        return ("nf4", n_swap, n_ckpt,
                f"int8 would stage {_stage_gb:.0f} GB of blocks in system RAM with only "
                f"{_avail_ram:.0f} GB available — Windows backs GPU memory with RAM commit, "
                f"so that starves the whole machine. 4-bit (~10.5 GB) stays on the card")
    return ("nf4", n_swap, n_ckpt,
            f"too tight even for streamed int8 — 4-bit parks {n_swap} blocks against "
            f"int8's {i_swap}")


def _max_effective_mp(group):
    """The heaviest single ITEM in the dataset, as effective megapixels: T x H x W per file.

    Header-only: safetensors key names carry the shape, so this is a directory scan and no
    tensor is read. A still's `latent_HxW` contributes its own area; a clip's `latent_TxHxW`
    contributes area x T. The per-file PRODUCT is the point — the old form took the largest
    bucket and the longest T as two separate maxima, which planned a 1 MP stills + tiny-latent
    voice dataset as ~37 MP and forced a several-times-slower max-swap run for nothing. (A
    voice item's placeholder is (24, 37, 8, 8): 0.6 effective MP, smaller than one 1 MP still.)

    Returns (max_mp, latent_t_of_that_item); (0.0, 1) when no cache exists yet.
    """
    from safetensors import safe_open
    best_mp, best_t = 0.0, 1
    seen = set()
    for ds in getattr(group, "datasets", []):
        cache_dir = getattr(ds, "cache_directory", None)
        if not cache_dir or cache_dir in seen or not os.path.isdir(cache_dir):
            continue
        seen.add(cache_dir)
        # Only caches whose stems are CURRENT images — the stale-cache guard's rule, applied
        # to planning. Without it, a leftover cache from a previous Target Megapixels in the
        # same dir inflates the plan (seen live: 992x992-era headers made a 0.25 MP run plan
        # for 0.98 MP — conservative direction, but the plan should describe THIS run).
        _stems = None
        _img_dir = getattr(ds, "image_directory", None)
        if _img_dir and os.path.isdir(_img_dir):
            _stems = {os.path.splitext(f)[0] for f in os.listdir(_img_dir)}
        for name in os.listdir(cache_dir):
            if not name.endswith(".safetensors"):
                continue
            if _stems is not None:
                _stem = "_".join(name.split("_")[:-2])       # {basename}_{WxH}_{arch}
                if _stem and _stem not in _stems:
                    continue
            try:
                with safe_open(os.path.join(cache_dir, name), framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("latent_") and not k.startswith("latent_control_"):
                            dims = [int(d) for d in k[len("latent_"):].split("x")]
                            t = dims[0] if len(dims) == 3 else 1
                            h, w = dims[-2], dims[-1]
                            mp = t * (h * 16) * (w * 16) / 1e6
                            if mp > best_mp:
                                best_mp, best_t = mp, t
            except Exception:
                continue                      # unreadable cache: the caching pass will say so
    return best_mp, best_t


def _max_clip_act_item(group):
    """The dataset's heaviest CLIP, as (latent_t, spatial_mp) — (1, 0.0) when it has none.

    Feeds the FT planner's clip activation term (ft_clip_activation_gb) and nothing else —
    _max_effective_mp stays untouched because it feeds the LoRA-side plan_vram. Same
    header-only scan and stale-stem filter as _max_effective_mp, with two differences:

    * A CLIP is identified by its cache HEADER, never by filename: a 3-dim latent key
      (`latent_{T}x{H}x{W}`) in a file with no `audio_only` key. Cache filenames strip the
      source extension, so an extension test would need the source directory listing — and
      that listing is optional here (missing dir just disables the staleness filter), which
      would silently zero the activation term and reintroduce the exact OOM it prevents.
    * VOICE items are excluded even though their placeholder latents are 3-dim and their
      video frames genuinely forward: their spatial grid is 8x8 latent, so their own
      activation cost tops out ~0.35 GB — inside the stills overhead's round-up, and
      field-proven at every tier. What the exclusion actually protects is the flat
      fragmentation MARGIN, which gates on T>1 and would tax every voice-only dataset
      ~2.4 GB for nothing. The `audio_only` header key is the guarantee.

    Heaviest = argmax of the per-item (T-1) x spatial_mp product (one step's peak belongs
    to one item; two separate maxima would re-create the trap documented above)."""
    from safetensors import safe_open
    best_score, best_t, best_mp = 0.0, 1, 0.0
    seen = set()
    for ds in getattr(group, "datasets", []):
        cache_dir = getattr(ds, "cache_directory", None)
        if not cache_dir or cache_dir in seen or not os.path.isdir(cache_dir):
            continue
        seen.add(cache_dir)
        _stems = None
        _img_dir = getattr(ds, "image_directory", None)
        if _img_dir and os.path.isdir(_img_dir):
            _stems = {os.path.splitext(f)[0] for f in os.listdir(_img_dir)}
        for name in os.listdir(cache_dir):
            if not name.endswith(".safetensors"):
                continue
            if _stems is not None:
                _stem = "_".join(name.split("_")[:-2])       # {basename}_{WxH}_{arch}
                if _stem and _stem not in _stems:
                    continue
            try:
                with safe_open(os.path.join(cache_dir, name), framework="pt") as f:
                    keys = list(f.keys())
                    if "audio_only" in keys:
                        continue                              # voice item — see docstring
                    for k in keys:
                        if k.startswith("latent_") and not k.startswith("latent_control_"):
                            dims = [int(d) for d in k[len("latent_"):].split("x")]
                            if len(dims) != 3:
                                continue                      # a still — no activation term
                            t, h, w = dims
                            spatial_mp = (h * 16) * (w * 16) / 1e6
                            score = (t - 1) * spatial_mp
                            if score > best_score:
                                best_score, best_t, best_mp = score, t, spatial_mp
            except Exception:
                continue                      # unreadable cache: the caching pass will say so
    return best_t, best_mp


# The 0.5 GB / 0.25 MP checkpointed-activation anchor was MEASURED at 0.25 MP; everything
# above it is linear extrapolation, and attention workspaces do not owe us linearity (4090
# field OOM, 25 Aug — video items plan at effective MP 10-50x the anchor). The extrapolated
# PORTION of the activation term gets this safety fraction — exactly zero at the anchor, so
# every validated stills-tier plan is bit-identical.
_ACT_EXTRAP_FRAC = 0.15


def _ckpt_need_gb(mp, batch, resident, transient_gb, adapter_gb):
    """Checkpointed-VRAM need (GB) before any swap — shared by plan_vram, plan_base_quant's
    H2D branch, and plan_swap_shortfall_gb so the three can never drift apart."""
    base = float(resident) + float(transient_gb) + float(adapter_gb)
    scale = max(0.25, float(mp)) / 0.25 * max(1, int(batch))
    act = _ACT_GB_CKPT * scale
    act += _ACT_EXTRAP_FRAC * max(0.0, act - _ACT_GB_CKPT)
    return base + act + _RESERVE_GB


def plan_swap_shortfall_gb(free_gb, mp=0.25, batch=1, resident_gb=None, transient_gb=0.0,
                           adapter_gb=0.0, swap_transient_gb=None, per_block_gb=None):
    """GB the swap plan is still short AFTER the 40-block cap — 0.0 when the cap covers it.

    plan_vram's `min(40, ...)` is a CAP, not a guarantee (4090 field OOM, 25 Aug): at
    video-tier effective MP the deficit can exceed what 40 parked blocks free, and the old
    planner proceeded anyway — a confident [vram] line, then an OOM at the first training
    step. Pure like plan_vram so the truth table pins on CPU. Callers pass the transient
    and per-block numbers matching how the swap actually runs (ring vs classic)."""
    resident = _RESIDENT_GB if resident_gb is None else float(resident_gb)
    swap_t = _SWAP_TRANSIENT_GB if swap_transient_gb is None else float(swap_transient_gb)
    per_block = _PER_BLOCK_GB if per_block_gb is None else float(per_block_gb)
    need = _ckpt_need_gb(mp, batch, resident, transient_gb, adapter_gb)
    deficit = need + swap_t - float(free_gb)
    return max(0.0, deficit - 40 * per_block)


def plan_vram(free_gb: float, mp: float = 0.25, batch: int = 1, resident_gb: float = None,
              transient_gb: float = 0.0, adapter_gb: float = 0.0):
    """Pure planner: (blocks_to_swap, gradient_checkpointing) from free VRAM + token load.

    Token load scales the activation term linearly (tokens ∝ mp x batch). Checkpointing is
    preferred OFF (faster) when everything fits without it; forced ON whenever swap is needed
    (without recompute, autograd would pin every swapped block's weights through backward).
    Swap additionally budgets _SWAP_TRANSIENT_GB: the backward pass transiently holds
    recompute segments beyond the parked residency (measured, see anchors above)."""
    resident = _RESIDENT_GB if resident_gb is None else float(resident_gb)
    # adapter_gb is resident for the whole run (weights + grads + optimizer state), so it belongs
    # in the base, not the activation term — gradient checkpointing does not reduce it.
    base = resident + float(transient_gb) + float(adapter_gb)
    scale = max(0.25, float(mp)) / 0.25 * max(1, int(batch))
    # _NOCKPT_MARGIN_GB, not just _RESERVE_GB: see the note on the constant. Recompute is ~0.1 s
    # a step and worth ~5 GB, so skipping it on a thin margin is a bad trade in both directions.
    need_nockpt = base + _ACT_GB_NOCKPT * scale + _RESERVE_GB + _NOCKPT_MARGIN_GB
    if free_gb >= need_nockpt:
        return 0, False
    need_ckpt = _ckpt_need_gb(mp, batch, resident, transient_gb, adapter_gb)
    if free_gb >= need_ckpt:
        return 0, True
    deficit = need_ckpt + _SWAP_TRANSIENT_GB - free_gb
    blocks = min(40, int(deficit / _PER_BLOCK_GB + 0.999))
    return blocks, True


def is_pruned_checkpoint(path: str) -> bool:
    """Does this file carry the curve-table AdaLN? Reads only the safetensors header.

    Needed before the base loads, because the pruned build's NF4 residency is ~6 GB smaller and
    the swap planner would otherwise park blocks nobody needs parked."""
    import json
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return "adaln_t_table" in json.loads(f.read(n))
    except Exception:
        return False


def cap_preview_res_small_card(w, h):
    """The 16 GB-class preview resolution cap, orientation-preserving: long side <= 768,
    short side <= 640 (a full 768 square ran a real 16 GB 4090 at 15.9/16 — one bad frame
    from the OOM ladder). Applied to the Samples-tab values at startup AND to the live
    sample override every time it's read — the override box must not be a way around the
    cap. Returns (w, h[, changed])-style: the clamped pair. H3-only by construction."""
    try:
        if torch.cuda.get_device_properties(0).total_memory / 1e9 >= 20.0:
            return w, h
    except Exception:
        return w, h
    _long, _short = max(w, h), min(w, h)
    if _long <= 768 and _short <= 640:
        return w, h
    _nl, _ns = min(_long, 768), min(_short, 640)
    return (_nl, _ns) if w >= h else (_ns, _nl)


def read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height} while active, else None. Unlike Krea 2 there is no
    ref_image: H3 is not an edit model, so a reference is meaningless here and a prompt is
    required for the override to count."""
    import json
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if not prompt:
            return None
        return {"prompt": prompt,
                "seed": int(d.get("seed", 1234)),
                "width": int(d.get("width", 768)),
                "height": int(d.get("height", 768))}
    except Exception:
        return None


def sample_sigmas(batch: int, device, shift=None, generator=None,
                  image_tokens: int = None) -> torch.Tensor:
    """Noise levels in (0,1) for training.

    shift=None (the default): sigma = 12u/(1+11u), u ~ uniform — H3's OWN training density.
    ai-toolkit's per-model defaults override the global 'sigmoid' with timestep_type='shift'
    through a scheduler configured shift=12 (ui options.tsx + their scheduler_config), so this
    is what MiniMax LoRAs are actually trained with there: median sigma ~0.92, ~57% of steps
    above 0.9, ~3% below 0.3. Training lives at the high-noise end, where each step nudges
    broad structure gently — which is why 1e-4 is a sane LR there and scorching at low shifts.
    (An earlier run here blamed shift-12 for poor likeness; that verdict was confounded —
    bf16 adaln was eating half the LoRA and being dropped at inference, and the pack had no
    audio rows yet. Withdrawn.)

    shift="sigmoid": UNSHIFTED logit-normal, sigma = sigmoid(N(0,1)), median 0.5 — the
    SD3/Flux-style density (ai-toolkit's GLOBAL default, but NOT its MiniMax one). Trains the
    mid/low-noise zone hard: at 1e-4 a 46-image epoch visibly overdrove the adapters
    (real-run finding, twice). A/B use only.

    shift="resolution": logit-normal with a resolution-dependent shift (~1.7 @768^2, median
    0.62 — Krea 2's mapping). Fizgig's original replacement density; same overdrive failure.

    shift=<float>: the uniform-u + shift map at any other value.
    """
    if shift is None:
        shift = VIDEO_SIGMA_SHIFT_TRAIN
    if shift == "sigmoid":
        return torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    if shift == "resolution":
        tokens = float(image_tokens or 225)                       # ~0.25 MP default
        mu = 0.5 + (tokens - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0)
        s = math.exp(mu)
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    elif isinstance(shift, str) and shift.startswith("lognorm:"):
        # SHAPE, not amount. Same shift map, but a logit-normal base instead of a uniform one:
        # the mass piles up in the middle and thins at BOTH ends, where a uniform base has fat
        # tails. Krea 2 and Klein both draw logit-normal, so this is the one axis the numeric
        # ladder cannot reach — it only ever varies how much low-noise training there is, never
        # where the rest of the mass sits.
        s = float(shift.split(":", 1)[1])
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    else:
        s = float(shift)
        base = torch.rand(batch, device=device, generator=generator)
    return (s * base) / (1.0 + (s - 1.0) * base)


def compute_loss(model, latent: torch.Tensor, text_embeds: torch.Tensor, *,
                 sigma: torch.Tensor = None, shift: float = None, generator=None,
                 noise: torch.Tensor = None, audio_latent: torch.Tensor = None,
                 audio_weight: float = 1.0, video_weight: float = 1.0,
                 parts_out: dict = None):
    """One training step's loss.

    latent      : [1, 24, T, H, W] clean VAE latent (x0). T=1 is a still.
    text_embeds : [1, L, text_dim] Qwen3-VL states.
    noise       : optional fixed noise (reproducible steps / tests); else sampled.
    audio_latent: optional [A*2, 32] clean audio rows (channel-major, as cached). Given, the
                  audio stream gets a REAL target instead of silence and its error joins the
                  loss. Absent — a still, or a clip the user muted — nothing changes: the rows
                  are still packed as noised silence so the frozen base runs in the layout it
                  was trained in, they simply contribute no gradient.
    audio_weight: multiplier on the audio term. Audio is only ~4% of the packed sequence at any
                  clip length, so an unweighted term barely moves; this is the dial for that,
                  and it starts at parity until a measurement says otherwise.
    video_weight: multiplier on the video term — 0 for an audio-only voice item, whose video
                  latent is a zeros placeholder. At 0 the video loss never enters the graph:
                  MSE against the dataset-mean latent is a real, wrong gradient ("every frame
                  looks like the average"), not a harmless no-op.

    Returns (loss, sigma_used). parts_out, if given, receives the video and audio terms
    separately — they are on different noise schedules and averaging them into one number hides
    which stream is actually learning.
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    # The DiT patchifies with patch_size (1, ph, pw), so the latent's H and W must be divisible by
    # the spatial patch. The dataset buckets on a 16-px step and the VAE is 16x, so a latent can be
    # odd (e.g. a 496-px bucket -> 31-px latent, not divisible by 2). Crop to the patch multiple
    # (drops at most one latent row/col = <=16 px of image edge) so patchify is exact and the target
    # (x0 - noise) stays the same shape as the model's prediction.
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]
    if sigma is None:
        # Resolution-aware auto schedule: token count from the (cropped) latent's patch grid.
        _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)

    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)

    if audio_latent is None:
        pred = model(noised.to(latent.dtype), t, text_embeds)
        loss = F.mse_loss(pred.float(), (x0 - noise).to(pred.dtype).float())
        if parts_out is not None:
            parts_out.update(video=float(loss.detach()), audio=None)
        if video_weight != 1.0:              # degenerate (an audio item missing its rows) but honest
            loss = video_weight * loss
        return loss, float(sigma.reshape(-1)[0])

    # The audio stream denoises on its OWN schedule — shift 3 against video's 12 — and
    # remap_sigma is the closed form that keeps the two at the same underlying point. Noising the
    # audio rows at the VIDEO sigma would put the stream somewhere the base has never seen it,
    # and the frozen model would spend the step disagreeing with the layout rather than learning.
    from fizgig.minimax.model import remap_sigma
    sigma_v = float(sigma.reshape(-1)[0])
    sigma_a = float(remap_sigma(torch.tensor(sigma_v)))

    a0 = audio_latent.to(device=device, dtype=torch.float32)
    a_noise = torch.randn(a0.shape, device=device, generator=generator, dtype=torch.float32)
    a_noised = (1.0 - sigma_a) * a0 + sigma_a * a_noise

    pred, pred_a = model(noised.to(latent.dtype), t, text_embeds,
                         audio_rows=a_noised, return_audio=True)
    v_loss = F.mse_loss(pred.float(), (x0 - noise).to(pred.dtype).float())
    if pred_a is None:                      # pack_audio_rows off — nothing to train against
        if parts_out is not None:
            parts_out.update(video=float(v_loss.detach()), audio=None)
        return video_weight * v_loss, sigma_v
    a_loss = F.mse_loss(pred_a.float(), (a0 - a_noise).float())
    if parts_out is not None:
        parts_out.update(video=float(v_loss.detach()), audio=float(a_loss.detach()),
                         sigma_audio=sigma_a)
    if video_weight == 0.0:
        # Not `0 * v_loss`: the multiplied form still builds the video branch's backward graph
        # and autograd walks it for nothing. The audio term alone IS this item's loss.
        return audio_weight * a_loss, sigma_v
    if video_weight != 1.0:
        return video_weight * v_loss + audio_weight * a_loss, sigma_v
    return v_loss + audio_weight * a_loss, sigma_v


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def write_preview_mp4(path, frames, wav_path, fps=24):
    """Mux decoded preview frames [3, F, H, W] in [0,1] with their wav into a playable mp4.

    The gallery plays THIS for samples with sound — a real clip at the true frame rate with
    its soundtrack, instead of a scrub slider plus a separate audio player. Raises on any
    failure; the caller treats the mp4 as a nicety."""
    import subprocess
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("no ffmpeg available")
    f, h, w = frames.shape[1], frames.shape[2], frames.shape[3]
    raw = (frames.permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().cpu().numpy().tobytes()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-i", wav_path,
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", path]
    p = subprocess.run(cmd, input=raw, capture_output=True,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    if p.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:]
                           or "ffmpeg failed")


_PREVIEW_RUNGS = (1536, 1280, 1024, 768, 640, 512)


def _rung_below(v):
    for r in _PREVIEW_RUNGS:
        if r < v:
            return r
    return v


def next_preview_res(w, h):
    """One rung down the preview OOM ladder, walking the STANDARD resolutions: the taller
    axis drops to the next standard size first, then the other — 1024x1024 -> 1024x768 ->
    768x768 -> 768x640 -> 640x640 -> 640x512 -> 512x512 (Peter). Floors at 512; returns
    the same pair when nothing is below (the caller re-raises then)."""
    if h >= w and h > 512:
        nh = _rung_below(h)
        if nh < h:
            return w, nh
    if w > 512:
        nw = _rung_below(w)
        if nw < w:
            return nw, h
    if h > 512:
        nh = _rung_below(h)
        if nh < h:
            return w, nh
    return w, h


def write_wav(path, wav, sample_rate=32000):
    """[2, L] float waveform in [-1, 1] -> 16-bit interleaved stereo wav. Stdlib only."""
    import wave as _wave
    data = (wav.detach().float().clamp(-1, 1) * 32767.0).to(torch.int16)
    with _wave.open(path, "wb") as f:
        f.setnchannels(int(data.shape[0]))
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(data.t().contiguous().numpy().tobytes())


_EGRID_CACHE = [None]


def _load_h3_egrid():
    """The full model's silu(t_emb) rows on a 1025-point t grid, [1025, 2688].

    Bundled from larryvrh's ComfyUI-MiniMax-H3-Turbo node (Apache-2.0) — it exists because
    the PRUNED base collapsed the time embedder into an 8-wide curve table, so silu(t_emb)
    cannot be computed from the loaded weights; the grid is the full model's answer,
    precomputed."""
    if _EGRID_CACHE[0] is None:
        import fizgig
        from safetensors.torch import load_file
        p = os.path.join(os.path.dirname(fizgig.__file__), "assets",
                         "h3_silu_temb_grid.safetensors")
        _EGRID_CACHE[0] = load_file(p)["silu_t_emb_grid"]
    return _EGRID_CACHE[0]


def _turbo_adaln_forward(base, updates, table, egrid):
    """A replacement AdalnProj.forward that adds one or more full-model AdaLN LoRA updates.

    Each update lives in the full model's silu(t_emb) space; the pruned base only has curve
    rows, so each incoming t_emb row is matched to its nearest table row (the model built it
    by lerping adjacent rows — half a grid step of error at worst, larryvrh's own approach)
    and the corresponding full-width grid row stands in: x += B @ A @ silu(t_emb). Strength
    is folded into B at collection time. `updates` is a list of (A, B): a context LoRA's
    rows and the preview Turbo's rows both land on the same module, and they must ADD, not
    replace each other."""
    def forward(t_emb):
        import torch.nn.functional as _F
        x = base.linear(_F.silu(t_emb) if base.apply_silu else t_emb)
        idx = torch.cdist(t_emb.detach().float(),
                          table.to(t_emb.device, torch.float32)).argmin(dim=1)
        st = egrid.to(t_emb.device)[idx].to(x.dtype)
        for A, B in updates:
            x = x + (B.to(x) @ (A.to(x) @ st.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


def turbo_adaln_patch(dit, pairs, device, dtype, egrid=None):
    """Install the AdaLN injection for the preview render. Returns modules patched.

    Instance-attribute forwards, like the reference node: assignment shadows the class
    method, deletion restores it — the module tree is never rebuilt, and a training AdaLN
    adapter wrapped around .linear keeps firing because the replacement still calls
    base.linear."""
    if not pairs or not getattr(dit, "pruned_adaln", False):
        return 0
    egrid = _load_h3_egrid() if egrid is None else egrid
    table = dit.adaln_t_table
    if table.shape[0] != egrid.shape[0]:
        logger.warning(f"[turbo] adaln grid rows {egrid.shape[0]} != table rows "
                       f"{table.shape[0]} — adaln injection skipped")
        return 0
    eg = egrid.to(device)
    n = 0
    by_mod = {}
    for mod, A, B in pairs:
        if A.shape[1] != eg.shape[1]:
            continue
        by_mod.setdefault(id(mod), (mod, []))[1].append((A.to(device, dtype),
                                                          B.to(device, dtype)))
    for mod, updates in by_mod.values():
        mod.forward = _turbo_adaln_forward(mod, updates, table, eg)
        n += 1
    return n


def turbo_adaln_unpatch(pairs):
    """Remove the injection (idempotent) — the class forward comes back, and the GPU copies
    of A/B/grid die with the closures."""
    for mod, _a, _b in pairs:
        try:
            del mod.forward
        except AttributeError:
            pass


def load_preview_turbo(dit, path, strength, tag="turbo"):
    """The Turbo LoRA, wired for previews: applied ONCE to the live DiT with every module
    DISABLED, weights parked on CPU. The preview phase flips `enabled` on and moves the
    weights to the GPU; afterwards both revert. A disabled LoRAInfModule's forward is a pure
    passthrough that never touches its weights, so the training step pays one Python branch
    per wrapped Linear and nothing else — no weight surgery on the training model, ever.

    Returns (network, adaln_pairs). The file's backbone modules are prefiltered to Linears
    that exist on THIS base with matching shapes. Its AdaLN modules (2688-wide, full-model
    space) cannot be hosted by the pruned curve-table base as weight modules — but they are
    NOT discarded: they carry the per-timestep modulation for the video AND audio streams,
    and dropping them is what made few-step audio fall apart (Peter; same finding as
    larryvrh's dedicated loader node). They come back as (adaln_module, A, B*strength)
    pairs for turbo_adaln_patch's run-time injection during previews."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import (create_network_from_weights,
                                      ensure_kohya_lora_state_dict)
    sd = ensure_kohya_lora_state_dict(load_file(path))
    linears = {f"lora_unet_{n.replace('.', '_')}": m
               for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
    adaln_parents = {f"lora_unet_{n.replace('.', '_')}_linear": m
                     for n, m in dit.named_modules()
                     if type(m).__name__ == "AdalnProj"}
    keep, adaln_pairs, dropped = {}, [], []
    for name in sorted({k.split(".")[0] for k in sd}):
        m = linears.get(name)
        down = sd.get(f"{name}.lora_down.weight")
        up = sd.get(f"{name}.lora_up.weight")
        if down is None or up is None:
            dropped.append(name)
            continue
        if (m is not None and down.shape[1] == m.in_features
                and up.shape[0] == m.out_features):
            for suf in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                if f"{name}{suf}" in sd:
                    keep[f"{name}{suf}"] = sd[f"{name}{suf}"]
            continue
        ap = adaln_parents.get(name)
        if ap is not None and up.shape[0] == ap.linear.out_features:
            # full-model AdaLN rows: hosted at preview time by the e-grid injection
            adaln_pairs.append((ap, down.clone(), up.clone() * float(strength)))
            continue
        dropped.append(name)
    if not keep:
        raise RuntimeError("no module in this LoRA matches the loaded base — wrong file?")
    net = create_network_from_weights(None, float(strength), keep, None, dit,
                                      for_inference=True)
    net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    # AFTER apply_to, or the modules keep their zero init and contribute nothing — the same
    # trap the Krea 2 context-LoRA path documents.
    net.load_state_dict(keep, strict=False)
    net.requires_grad_(False)
    for m in net.unet_loras:
        m.enabled = False
    logger.info(f"[{tag}] {len(net.unet_loras)} modules wired at strength {strength:g}"
                + (f" + {len(adaln_pairs)} adaln via run-time injection"
                   if adaln_pairs else "")
                + (f" ({len(dropped)} skipped)" if dropped else ""))
    return net, adaln_pairs


def load_context_lora(dit, path, strength, device, dtype, tag="context", label="Context LoRA"):
    """A Context LoRA: an existing LoRA loaded FROZEN and ACTIVE on the base before the
    trainable network wraps it, so the new LoRA learns to coexist with it (Klein and Krea 2
    have the same feature). Same loader as the preview Turbo — H3's shape prefilter and the
    pruned-base AdaLN injection are exactly what a foreign H3 LoRA needs — but the modules
    stay ENABLED and resident for every training step, and its AdaLN rows are injected for
    the same span. Previews render with it OFF (context_lora_set_active) — the deployment
    view, which for a training adapter is the whole point.

    Returns (network, adaln_pairs)."""
    from fizgig.networks.lora import assert_lora_family_matches
    assert_lora_family_matches(path, "minimax", label)
    net, pairs = load_preview_turbo(dit, path, strength, tag=tag)
    net.to(device=device, dtype=dtype).eval()
    for m in net.unet_loras:
        m.enabled = True
    n_ad = turbo_adaln_patch(dit, pairs, device, dtype)
    logger.info(f"[{tag}] {os.path.basename(path)} active at {float(strength):g} — "
                f"{len(net.unet_loras)} modules"
                + (f" + {n_ad} adaln injected" if n_ad else "")
                + "; the trainable LoRA learns on top of it")
    return net, pairs


@contextlib.contextmanager
def lora_disabled(network):
    """Run the frozen BASE inside this block — every adapter's multiplier is temporarily 0.

    Every module type (LoRA, LoKR, LoHa) reads self.multiplier live in its forward and
    short-circuits on 0.0, so this needs no re-apply and no weight surgery. Restores whatever
    each module had, not a blanket 1.0 — a context LoRA rides at its own strength."""
    mods = list(getattr(network, "unet_loras", []))
    saved = [m.multiplier for m in mods]
    try:
        for m in mods:
            m.multiplier = 0.0
        yield
    finally:
        for m, v in zip(mods, saved):
            m.multiplier = v


def compute_distill_loss(model, network, latent, text_plain, *, text_ref, ref_latents,
                         text_token_tags=None, distill_weight=0.8, shift=None, generator=None,
                         noise=None, seed=0, parts_out=None):
    """Reference distillation: teach the LoRA to behave, from text alone, as if it had been
    shown the reference photo.

    Two predictions of the SAME noised latent at the SAME timestep:
      teacher — frozen base, LoRA off, conditioning WITH the reference (vision blocks + ref rows)
      student — LoRA on, conditioning WITHOUT it
    loss = w * MSE(student, teacher) + (1 - w) * MSE(student, x0 - noise)

    The photo term is what keeps real photographic detail available: pure distillation caps the
    LoRA at exactly the teacher's habits and can never exceed them. The teacher term is what
    stops the run spending capacity on backgrounds and framing, because the target is no longer
    a particular photograph.

    Everything the two passes share is drawn ONCE — noise, timestep, and the audio silence rows.
    The audio rows especially: model.forward redraws them per call when not given, so letting
    each pass draw its own would put a different soundtrack under teacher and student and add
    pure noise to the very signal being distilled.
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]

    _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
    sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)
    noised = ((1.0 - s) * x0 + s * noise).to(latent.dtype)
    t = (1.0 - sigma).to(device)

    # one soundtrack for both passes (see the docstring)
    audio_noise = None
    if getattr(model, "pack_audio_rows", False):
        from fizgig.minimax.model import AUDIO_CHANNELS, audio_latents_for_frames
        n_a = audio_latents_for_frames(1) * AUDIO_CHANNELS
        audio_noise = torch.randn(n_a, model.config.audio_latents_dim, device=device,
                                  generator=generator, dtype=torch.float32)

    with torch.no_grad(), lora_disabled(network):
        teacher = model(noised, t, text_ref, audio_noise, ref_latents=ref_latents,
                        text_token_tags=text_token_tags, seed=seed).float()
    student = model(noised, t, text_plain, audio_noise).float()

    w = float(distill_weight)
    teacher_mse = F.mse_loss(student, teacher.detach())
    loss = w * teacher_mse
    photo_mse = None
    if w < 1.0:
        photo_mse = F.mse_loss(student, (x0 - noise).float())
        loss = loss + (1.0 - w) * photo_mse
    if parts_out is not None:
        # The RAW errors, before the 0.8/0.2 weights. The weights are already known; what is not
        # is how BIG each error is — and "how much of the learning comes from real pixels" is a
        # question about the errors, not the weights. Matching a real photograph is harder than
        # matching the model's own output, so the photo term can punch well above its weight.
        parts_out["teacher"] = float(teacher_mse.detach())
        parts_out["photo"] = float(photo_mse.detach()) if photo_mse is not None else 0.0
    return loss, float(sigma.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Adaptive LR — bi-directional plateau tracker (architecture-agnostic; a faithful port of the
# Klein/Krea 2 watcher). Stability signal is weight-norm growth (>30%), same as Krea 2 (the H3
# loop clips gradients but the watcher reads weight-norm growth, not the clip ratio).
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """Each epoch boundary: probe UP x1.25 on steady loss descent (patience 2); reduce DOWN x0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the LoRA
    weights 70/30 toward the previous epoch's snapshot and restores the optimizer state (kills bad
    Adam momentum). The CPU rollback snapshot is in-memory only; the streak/best_loss scalars are
    JSON round-trippable (kept for parity — this barebones trainer has no resume yet)."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            # Respect a depth-split LR: each group carries its own lr_scale, so the watcher moves
            # the whole schedule up or down while KEEPING the ratio between groups. Writing new_lr
            # flat would silently undo the split on the first adaptive move.
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr * pg.get("lr_scale", 1.0)
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


# ---------------------------------------------------------------------------
# Per-block movement limiter — a compressor on the block bus.
#
# Empirical finding (8 Aug, three runs + a block-range A/B): whichever block sits LAST in the
# trained range absorbs wildly disproportionate movement — 2-4x the median block from epoch 1,
# and still diverging 40 epochs later. Cut blocks 46-49 and blocks 43-45 inherit the exact
# same signature: the pathology is POSITIONAL, not a property of particular layers. The
# deepest trained block gets the most coherent, least-attenuated gradient (everything after
# it is frozen and decorrelates nothing), and Adam turns coherence into relentless movement.
# The visible symptom is output-adjacent over-editing: distorted eyes and other
# high-frequency damage.
#
# LR penalties and block cuts are positional patches for a positional problem — they just
# relocate the hot spot. This limiter is self-targeting: after each optimizer step, any
# block whose TOTAL RELATIVE movement (sum over its adapters of ||dW||/||W_base||, the same
# metric the offline analysis used) exceeds `cap_factor x median block` is projected back to
# the cap by scaling its up-factors. Blocks move freely until one hogs; then only that one
# is pulled back, wherever the trained range ends.
# ---------------------------------------------------------------------------
class StepClipper:
    """Cap how far any block may move in a SINGLE optimizer step.

    Replaces the cumulative BlockLimiter that shipped in 3.5.0. That one clamped a block's
    TOTAL accumulated movement back to cap x median, which necessarily scaled down everything
    the block had legitimately learned in earlier epochs along with the overshoot — measured on
    real runs as a genuine likeness ceiling: limiter ON was visibly worse than OFF, while OFF
    corrupted. Clipping the STEP prevents the overshoot instead of undoing the history, so
    there is no quality to trade for the safety.

    Being per-step also removes the calibration problem that sank the movement governor: a
    per-epoch budget has to be scaled by dataset size, and got it wrong by 7x on a 272-step
    epoch, starving a run for 84 epochs. A step is a step on any dataset.

    Measured in MODEL space — the change this step in each block's effective delta, summed in
    quadrature across the block's modules — and cheap, because every term is a rank-sized
    product via <kron(a,b), kron(c,d)> = <a,c><b,d> and <UV, XY> = tr(U^T X Y V^T). A full
    weight matrix is never materialised. Over-cap blocks are lerped back toward their pre-step
    weights, which is exact to first order in the step size (the delta is bilinear in the
    factors, so the second-order term is negligible at real step sizes).

    Self-calibrating: the cap is a multiple of the MEDIAN block's step, so it needs no absolute
    threshold and targets whichever block is actually running hot — the caboose, wherever the
    trained range happens to end.
    """

    def __init__(self, network, cap_factor: float = 1.25):
        import re as _re
        self.cap = float(cap_factor)
        self.clamped_total = 0
        self.clamp_counts = {}
        self.groups = {}              # block id -> [module]
        for m in getattr(network, "unet_loras", []):
            blk = _re.search(r"blocks_(\d+)_", m.lora_name)
            if blk is None or "token_refiner" in m.lora_name:
                continue              # text-side refiner is not part of the depth argument
            self.groups.setdefault(int(blk.group(1)), []).append(m)
        # Pre-step parameter snapshot, allocated ONCE and copied into each step.
        self._params = {blk: [p for m in mods for p in m.parameters() if p.requires_grad]
                        for blk, mods in self.groups.items()}
        self._prev = {blk: [p.detach().clone() for p in ps] for blk, ps in self._params.items()}
        self._prev_f = {}             # per-module factor snapshot for the delta measurement
        # Clip on each block's SMOOTHED step rate, not its instantaneous step. The caboose is a
        # PERSISTENTLY hot block; a single step landing above the median is just noise, and
        # per-step movement is far noisier than the cumulative quantity the retired limiter
        # measured. Reusing 1.25x on the raw per-step value therefore braked whichever blocks
        # were learning fastest on any given step — measured as a real quality loss that no LR
        # change touched (halving the LR moved the dose by 8%).
        self._rate = {}               # block id -> EMA of its per-step movement
        self._clipped_steps = 0
        self._total_steps = 0
        self._tail = 0.0              # last measured peak/median ACCUMULATED movement

    @staticmethod
    def _factors(m):
        """(a, b, scale) such that the module's delta is scale * a (x) b, for whichever form."""
        if hasattr(m, "lokr_w1"):
            return m.lokr_w1, m.lokr_w2, 1.0        # Fizgig LoKR: alpha 1.0, scale 1.0
        return m.lora_up.weight, m.lora_down.weight, float(m.scale)

    @classmethod
    def _cum_sq(cls, m) -> float:
        """||D||_F^2 for one module — its ACCUMULATED delta, not this step's."""
        a, b, sc = cls._factors(m)
        a, b = a.float(), b.float()
        if hasattr(m, "lokr_w1"):
            n = (a.norm() * b.norm()) ** 2
        else:
            n = torch.trace((a.T @ a) @ (b @ b.T)).clamp(min=0)
        return float(n) * sc * sc

    @classmethod
    def _step_delta_sq(cls, m, prev) -> float:
        """||D_post - D_pre||_F^2 for one module, without materialising D."""
        a1, b1, sc = cls._factors(m)
        a0, b0 = prev
        a1, b1, a0, b0 = a1.float(), b1.float(), a0.float(), b0.float()
        if hasattr(m, "lokr_w1"):
            # <kron(a,b), kron(c,d)> = <a,c><b,d>
            n1 = (a1.norm() * b1.norm()) ** 2
            n0 = (a0.norm() * b0.norm()) ** 2
            cross = (a1 * a0).sum() * (b1 * b0).sum()
        else:
            # <U1 V1, U0 V0> = tr(U1^T U0 V0 V1^T); ||UV||^2 = tr((U^T U)(V V^T))
            n1 = torch.trace((a1.T @ a1) @ (b1 @ b1.T))
            n0 = torch.trace((a0.T @ a0) @ (b0 @ b0.T))
            cross = torch.trace((a1.T @ a0) @ (b0 @ b1.T))
        return float((n1 + n0 - 2 * cross).clamp(min=0)) * sc * sc

    @torch.no_grad()
    def pre_step(self):
        """Snapshot the weights the optimizer is about to move. Call BEFORE optimizer.step()."""
        for blk, ps in self._params.items():
            for dst, p in zip(self._prev[blk], ps):
                dst.copy_(p.detach())
        self._prev_f = {id(m): tuple(t.detach().clone() for t in self._factors(m)[:2])
                        for mods in self.groups.values() for m in mods}

    @torch.no_grad()
    def step(self):
        """Clip blocks whose SMOOTHED movement rate is running above cap x the pack's."""
        import statistics as _st
        if len(self.groups) < 3 or not self._prev_f:
            return
        moved = {blk: sum(self._step_delta_sq(m, self._prev_f[id(m)]) for m in mods) ** 0.5
                 for blk, mods in self.groups.items()}
        for blk, d in moved.items():
            r = self._rate.get(blk)
            self._rate[blk] = d if r is None else 0.9 * r + 0.1 * d
        med = _st.median(self._rate.values())
        self._total_steps += 1
        if med <= 0:
            return                                  # nothing has moved yet
        cap = self.cap * med
        # ACCUMULATION AWARENESS. Capping strides bounds how fast a block moves but not how far
        # it has GOT — and a coherent run (which is what gradient accumulation produces) lets
        # the caboose accumulate imbalance even while every stride is legal: measured at 2.02x
        # the median block by epoch 2 with strides capped at 1.25x. The old limiter fixed that
        # by scaling the block's accumulated delta down, which also destroyed what it had
        # legitimately learned. Instead, a block that is ALREADY ahead simply gets a tighter
        # step allowance until the pack catches up: no history is ever touched, the block just
        # stops pulling further away. Squeeze is proportional and floored so it never freezes.
        cums = {blk: sum(self._cum_sq(m) for m in mods) ** 0.5
                for blk, mods in self.groups.items()}
        med_cum = _st.median(cums.values())
        self._tail = (max(cums.values()) / med_cum) if med_cum > 0 else 0.0
        _fired = False
        for blk, d in moved.items():
            blk_cap = cap
            if med_cum > 0 and cums[blk] > self.cap * med_cum:
                # SQRT, not the raw ratio, and floored at 0.5. An already-ahead block is
                # otherwise penalised twice over — once by the per-step cap for being above the
                # median, again by this squeeze for being ahead — and those are the same late
                # blocks every time, so the stacked penalty reads as a treble cut. At a 2.41x
                # tail under a 2.0 cap the raw ratio pulled the effective cap down to 1.66;
                # softened it is 1.82, so raising the cap actually raises it. Genuine runaways
                # still get squeezed, just proportionally less hard.
                blk_cap = cap * max(0.5, ((self.cap * med_cum) / cums[blk]) ** 0.5)
            # TREND decides whether to act — that is what makes a persistently hot block (the
            # caboose) the target and lets a one-off noisy step from a healthy block through.
            # The TRIM is then applied to this actual step, not to the lagging average: scaling
            # by cap/rate under-corrects badly (a 10x hog only came back to ~5.9x).
            if self._rate[blk] <= blk_cap or d <= blk_cap:
                continue
            cap_ = blk_cap
            s = cap_ / d
            for p, prev in zip(self._params[blk], self._prev[blk]):
                p.data.lerp_(prev, 1.0 - s)
            # The trend must reflect what actually happened, not the pre-trim step, or it stays
            # inflated and keeps re-triggering on a block that is now behaving.
            self._rate[blk] -= 0.1 * (d - cap_)
            self.clamped_total += 1
            self.clamp_counts[blk] = self.clamp_counts.get(blk, 0) + 1
            _fired = True
        if _fired:
            self._clipped_steps += 1

    def epoch_report(self):
        # The clip-rate is the number that matters as much as WHICH blocks: a cap that fires on
        # most steps is braking the whole pack, not trimming a caboose, and that reads from the
        # outside as "quality is worse" with no distortion to point at. A healthy run trims a
        # few persistent blocks; if this says most steps, the cap is too tight for the dataset.
        pct = (100.0 * self._clipped_steps / self._total_steps) if self._total_steps else 0.0
        self._clipped_steps = self._total_steps = 0
        tail = f" · tail {self._tail:.2f}x median" if self._tail else ""
        if not self.clamp_counts:
            return f"[clip] no block ran above the cap this epoch{tail}"
        top = sorted(self.clamp_counts.items(), key=lambda kv: -kv[1])[:6]
        n_blocks = len(self.clamp_counts)
        self.clamp_counts = {}
        return (f"[clip] fired on {pct:.0f}% of steps across {n_blocks} block(s){tail} — "
                + ", ".join(f"block {b} x{n}" for b, n in top))


class BlockLimiter:
    """RETIRED (10 Aug) — kept only because the offline analysis scripts import _movement.

    Superseded by StepClipper: clamping CUMULATIVE movement also scaled down legitimately
    learned history, which measurably capped likeness. Do not wire this into the loop."""

    def __init__(self, network, dit, cap_factor: float = 1.5):
        import re as _re
        self.cap = float(cap_factor)
        self.clamped_total = 0
        self.clamp_counts = {}
        # Own the module map, by ISINSTANCE — the shared _build_dit_linear_map filters on the
        # exact class NAME "Linear", which made the int8 base's ConvRotInt8Linear (and bnb's
        # Linear4bit) invisible: on a real base the limiter watched ZERO blocks and reported
        # "no block exceeded the cap" while the caboose ran hot. Both are nn.Linear subclasses.
        linear_map = {"lora_unet_" + n.replace(".", "_"): m
                      for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
        self.groups = {}          # block id -> [(module, base_norm)]
        for m in getattr(network, "unet_loras", []):
            blk = _re.search(r"blocks_(\d+)_", m.lora_name)
            if blk is None or "token_refiner" in m.lora_name:
                continue          # text-side refiner is not part of the depth argument
            target = linear_map.get(m.lora_name)
            bn = self._base_norm(target) if target is not None else None
            if bn:
                self.groups.setdefault(int(blk.group(1)), []).append((m, bn))

    @staticmethod
    def _base_norm(mod) -> float:
        """||W_base||_F for whichever storage the base uses. The ConvRot rotation is
        orthogonal, so the norm of the int8 codes x scales IS the true weight norm."""
        import torch as _t
        with _t.no_grad():
            if hasattr(mod, "qdata"):                        # ConvRotInt8Linear
                return float((mod.qdata.float() * mod.wscale.float()).norm())
            if hasattr(mod, "W_q"):                          # HQQ4bitLinear: dequant on demand
                try:
                    return float(mod.weight.float().norm())
                except Exception:
                    return 0.0
            w = getattr(mod, "weight", None)
            if w is None:
                return 0.0
            if w.__class__.__name__ == "Params4bit":         # bnb NF4 shell
                try:
                    import bitsandbytes.functional as _bf
                    return float(_bf.dequantize_4bit(w.data, w.quant_state).float().norm())
                except Exception:
                    return 0.0
            return float(w.float().norm())

    @staticmethod
    def _movement(m) -> float:
        """||dW||_F for one adapter, exactly as the offline analysis computes it."""
        import torch as _t
        with _t.no_grad():
            if hasattr(m, "lokr_w1"):
                return float(m.lokr_w1.float().norm() * m.lokr_w2.float().norm()) * float(m.scale)
            up, dn = m.lora_up.weight.float(), m.lora_down.weight.float()
            g = _t.trace((up.T @ up) @ (dn @ dn.T)).clamp(min=0)
            return float(g.sqrt()) * float(m.scale)

    @torch.no_grad()
    def step(self):
        """Project any over-cap block back to cap_factor x median. Cheap: rank-sized matmuls.

        The metric is RAW ||dW|| per block — NOT movement relative to the block's base norm.
        It used to be relative, and a real run showed why that leaks: damage correlates with
        raw delta (the whole dose-response table is in raw units), late blocks have LARGER
        base weights, so the relative metric granted the tail extra raw allowance — while
        the limiter held every block to 1.25x in ITS units, the tail crept to 2.34x median
        in raw units, over the damage threshold the governor was holding the pack under.
        All H3 blocks are identical shapes, so raw norms are directly comparable."""
        import statistics as _st
        rel = {}
        for blk, mods in self.groups.items():
            rel[blk] = sum(self._movement(m) for m, _bn in mods)
        if len(rel) < 3:
            return
        med = _st.median(rel.values())
        if med <= 0:
            return                                            # nothing has moved yet
        cap = self.cap * med
        for blk, r in rel.items():
            if r <= cap:
                continue
            s = cap / r
            for m, _bn in self.groups[blk]:
                if hasattr(m, "lokr_w2"):
                    m.lokr_w2.mul_(s)                         # delta scales linearly in w2
                else:
                    m.lora_up.weight.mul_(s)
            self.clamped_total += 1
            self.clamp_counts[blk] = self.clamp_counts.get(blk, 0) + 1

    def epoch_report(self):
        if not self.clamp_counts:
            return "[limiter] no block exceeded the cap this epoch"
        top = sorted(self.clamp_counts.items(), key=lambda kv: -kv[1])[:6]
        msg = "[limiter] clamped " + ", ".join(f"block {b} x{n}" for b, n in top)
        self.clamp_counts = {}
        return msg


class AdapterRamp:
    """Hold each step at a constant FRACTION of the adapter's current size, ramping the LR up
    toward the configured ceiling as the adapter grows.

    The observation this comes from: an adapter at ||dW|| ~53, trained slowly for 92 epochs,
    took a full 2e-4 for ten epochs with no distortion at all and produced the best likeness of
    the project. A fresh adapter at ||dW|| ~3 is visibly damaged by half that. The rate was
    never the problem — the SAME step is a 9% perturbation of a mature adapter and a 150%
    perturbation of a new one. A LoRA starts at exactly zero, so the ratio of step size to
    adapter size is at its worst on step one and improves monotonically from there.

    Which means the conventional schedule is backwards for adapters. Warmup-then-decay is built
    for models that start from a sensible initialisation; here it is too hot when the adapter is
    tiny and too cold once the adapter could take it. This ramps the other way.

    Why it needs no calibration, unlike the retired movement governor: the governor servoed on
    an ABSOLUTE movement rate, which depends on dataset size, network type and model width — it
    was wrong by 7x on a 272-step epoch. `step / ||dW||` is dimensionless, so one target
    transfers across datasets, LoRA vs LoKR, and any model size.

    At equilibrium the adapter grows exponentially (d||dW||/dt = rho*||dW||) until the LR hits
    the ceiling, after which growth returns to linear. rho is therefore best read as a growth
    rate: 0.005/step doubles the adapter roughly every 140 steps."""

    def __init__(self, network, target_rel: float = 0.005, start_mult: float = 0.1):
        self.target = float(target_rel)
        self.mult = float(start_mult)
        self._smooth = None
        self._prev = None
        self.params = [p for p in network.parameters() if p.requires_grad]
        self._mods = [m for m in getattr(network, "unet_loras", [])]

    @torch.no_grad()
    def _size(self) -> float:
        """||dW|| across the whole adapter — model-space, not parameter-space."""
        return sum(StepClipper._cum_sq(m) for m in self._mods) ** 0.5

    @torch.no_grad()
    def step(self) -> float:
        cur = self._size()
        if self._prev is None or cur <= 1e-9:
            self._prev = cur
            return self.mult
        rel = max(0.0, cur - self._prev) / cur      # this step as a fraction of what exists
        self._prev = cur
        self._smooth = rel if self._smooth is None else 0.9 * self._smooth + 0.1 * rel
        if self._smooth > 1e-12:
            err = self._smooth / self.target
            # Per-step gain caps, both damped after a real run hunted and then DAMAGED the
            # model on the way back up: 22 -> 77 -> 73 -> 68 -> 63 -> 29 -> 100 across
            # consecutive epochs, and the jump to 100% hit an adapter that was not ready for
            # it. The RELEASE rate is therefore a safety parameter in its own right, not a
            # tuning nicety — the old 1.03 compounds to 3.9x over a 46-step epoch, enough to
            # go from a third of the ceiling to all of it in one epoch. 1.01 caps that at
            # ~1.6x per epoch, so the ceiling is approached over several epochs and the
            # adapter has time to grow into it.
            #
            # The old 0.70 down cap compounds to 4e-8 over the same epoch — a 12:1 asymmetry
            # against the up-gain that caused the slam-to-floor half of the oscillation, whose
            # rebound was what overshot. 0.95 keeps a safety bias (still ~5x faster down than
            # up) without flooring the LR from a single noisy reading.
            self.mult = min(1.0, max(0.02, self.mult * min(1.01, max(0.95, err ** -0.3))))
        return self.mult

    def epoch_report(self) -> str:
        rel = (self._smooth or 0.0)
        return (f"[ramp] adapter ||dW||={self._prev or 0:.2f}, growing {100 * rel:.3f}%/step "
                f"(target {100 * self.target:.3f}%) — LR at {100 * self.mult:.0f}% of the "
                f"configured ceiling")


def should_reassert_lr(*, resuming, adaptive, ramp, warmup_steps, global_step) -> bool:
    """Does anything write param_group['lr'] from here on? If not, a resume must reassert the
    CONFIGURED rate.

    torch's optimizer.load_state_dict restores the saved param_groups INCLUDING lr, and the
    step loop only writes lr while warmup is still ramping. A state written while something
    WAS modulating the LR (the retired movement governor throttled it) therefore handed its
    last throttled rate to a run that no longer modulates anything, and kept it for the whole
    run — measured on a real run as 3.28e-5 against a configured 2e-4.

    The subtle case, and the one the first version of this fix got wrong: warmup CONFIGURED but
    already FINISHED. warmup_steps > 0 is not the question — `global_step < warmup_steps` is."""
    if not resuming:
        return False
    if adaptive is not None:
        return False        # adaptive owns the LR; its restored mid-flight value is correct
    if ramp is not None:
        return False        # the adapter ramp rewrites lr every step
    if warmup_steps and global_step < warmup_steps:
        return False        # the warmup ramp rewrites lr every step until it ends
    return True


class EMAWeights:
    """Exponential moving average of the trainable adapter — the smooth center of a rough
    trajectory.

    High static LRs take big Adam strides that zigzag around the good solution; the raw
    weights at any single step are one corner of the zigzag, and that roughness reads as
    distortion in samples. The EMA is the running average of the path, so what gets SAVED
    (and previewed) is the center the strides orbit — the standard diffusion-training cure
    for exactly this. Training itself always runs on the raw weights: swap_in/swap_out
    bracket saves and previews only.

    Decay ramps in as min(decay, (1+n)/(10+n)) so the first steps track the weights closely
    instead of anchoring to the zero init. Shadow is fp32 (the adapter is small)."""

    def __init__(self, network, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.params = [p for p in network.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self._backup = None

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self):
        """Put the averaged weights into the live network (for a save or a preview).

        The raw-weight backup lives on CPU: swap_in brackets previews, and a clip preview
        is exactly when GPU headroom is scarcest — a GPU-resident backup (~0.6 GB at LoKR
        factor 8) was part of what tipped 32 GB cards back into the Windows VRAM spill."""
        self._backup = [p.detach().to("cpu", copy=True) for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.to(p.device, p.dtype))

    @torch.no_grad()
    def swap_out(self):
        """Restore the raw training weights. Must always pair with swap_in."""
        for b, p in zip(self._backup, self.params):
            p.data.copy_(b.to(p.device, p.dtype))
        self._backup = None

    def state_dict(self):
        return {"n": self.n, "decay": self.decay,
                "shadow": [s.detach().cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.n = int(sd["n"])
        if len(sd["shadow"]) != len(self.shadow):
            raise ValueError(f"EMA state has {len(sd['shadow'])} tensors, network has "
                             f"{len(self.shadow)} — different run configuration?")
        self.shadow = [t.to(s.device, torch.float32) for t, s in zip(sd["shadow"], self.shadow)]


# ---------------------------------------------------------------------------
# Full image-only training loop (NF4 base + LoRA) over the H3 caches.
# ---------------------------------------------------------------------------
class _Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_training_state(output_dir, output_name, network, optimizer, *, epoch, global_step,
                         dtype, extra=None, ema=None):
    """Save a resumable training-state dir matching Klein/Krea 2 naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). Holds the
    network weights in NATIVE state_dict naming (never the LyCORIS comfy-format rewrite — resume
    load_state_dict needs the module keys), the optimizer state, RNG, and a small JSON. The
    GUI's _detect_latest_state_dir finds the highest-numbered dir and passes it to --resume."""
    import json
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    try:
        return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                  global_step=global_step, dtype=dtype, extra=extra, ema=ema)
    except Exception as _first:
        # Clean the partial dir (no training_state.json = no commit marker, but it would shadow
        # the previous good state in the GUI's latest-state scan), then retry ONCE after a short
        # pause. Network filesystems (RunPod volumes) throw transient stream errors that clear
        # in seconds — a real run lost its epoch-8 state to exactly one of those. If the retry
        # also fails it is not transient; re-raise and let the caller decide fatality.
        import shutil
        import time
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.warning("[state] save failed (%s: %s) — retrying once in 5s",
                       type(_first).__name__, _first)
        time.sleep(5)
        try:
            os.makedirs(state_dir, exist_ok=True)
            return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                      global_step=global_step, dtype=dtype, extra=extra, ema=ema)
        except Exception:
            shutil.rmtree(state_dir, ignore_errors=True)
            raise


def _write_state_files(state_dir, network, optimizer, *, epoch, global_step,
                       dtype, extra=None, ema=None):
    import json
    network.save_weights(os.path.join(state_dir, "lora.safetensors"), dtype,
                         {"ss_architecture": ARCHITECTURE_MINIMAX,
                          "ss_network_module": "fizgig.minimax (state dir, native keys)"})
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    if ema is not None:
        # The RAW weights are what lora.safetensors holds (training resumes from them); the
        # EMA shadow rides alongside so the average survives pause/resume too.
        torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    if extra:
        meta.update(extra)
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    # training_state.json is written LAST on purpose: it is the commit marker. A save that
    # dies partway leaves no json, and both the resume validator and the GUI's latest-state
    # detection treat a json-less dir as not-a-state rather than resuming garbage.
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _validate_state_dir(state_dir):
    """Refuse anything that is not a saved training state, and say what to pick instead.

    Issue #48: choosing the OUTPUT directory rather than a state folder failed with a bare
    "lora.safetensors not found", and the obvious workaround — putting a LoRA there under that
    name — then appeared to work. It cannot: without training_state.json there is no epoch or
    step, and without optimizer.pt there is no Adam state, so the run silently starts over from
    epoch 0 while looking like a resume, and overwrites the finished LoRA on the way. Refusing
    is the only safe answer, and the message has to name the folder they actually wanted.
    """
    if os.path.isfile(state_dir):
        sib = ""
        base = os.path.dirname(state_dir)
        try:
            states = sorted(d for d in os.listdir(base) if d.endswith("-state")
                            and os.path.isfile(os.path.join(base, d, "training_state.json")))
            if states:
                sib = " Next to it: " + ", ".join(states[-3:])
        except OSError:
            pass
        raise RuntimeError(
            f"[resume] {os.path.basename(state_dir)} is a file — resume takes the saved-state "
            f"FOLDER (named like '<lora name>-000012-state'), not a .safetensors.{sib}")
    if not os.path.isdir(state_dir):
        raise RuntimeError(f"[resume] {state_dir} does not exist — was the state folder moved "
                           f"or renamed?")
    missing = [f for f in ("lora.safetensors", "training_state.json")
               if not os.path.isfile(os.path.join(state_dir, f))]
    if not missing:
        return
    lines = [
        f"[resume] {state_dir} is not a saved training state — missing {', '.join(missing)}.",
        "[resume] Pick the folder named like '<lora name>-000012-state'. Renaming a LoRA to "
        "lora.safetensors does not make one: there would be no optimizer state and no epoch "
        "to resume from, so the run would quietly start again from scratch.",
    ]
    try:
        # The usual mistake is picking the parent output directory, one level above the state
        # folders — so if they are sitting right there, name them.
        here = sorted(d for d in os.listdir(state_dir)
                      if d.endswith("-state")
                      and os.path.isfile(os.path.join(state_dir, d, "training_state.json")))
        if here:
            lines.append("[resume] That looks like your output directory. The saved states in "
                         "it are: " + ", ".join(here[-5:]))
    except OSError:
        pass
    raise RuntimeError(os.linesep.join(lines))


def resume_network_shape(state_dir, network_type, network_dim, network_alpha, lokr_factor):
    """The network shape a RESUME must build: the checkpoint's, not the GUI's.

    The state dir carries the adapter exactly as it trained; rebuilding from whatever the
    boxes say NOW crashed the relaunch the moment settings had moved on (a rank-8 pause
    resumed under a rank-16 preset died on a size-mismatch wall, Peter 17 Aug). This reads
    the shape out of the saved lora.safetensors — parametrization by key suffix, LoKR factor
    from w1, rank/alpha from a non-AdaLN module (AdaLN's rank caps at 8, so it lies about
    the network's dim) — and returns (type, dim, alpha, factor, notes). The caller logs the
    notes and builds THAT network; a resume continues the run it resumes."""
    path = os.path.join(state_dir, "lora.safetensors")
    notes = []
    if not os.path.isfile(path):
        return network_type, network_dim, network_alpha, lokr_factor, notes
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if any(k.endswith(".lokr_w1") for k in keys):
            if network_type != "lokr":
                notes.append(f"the checkpoint is LoKR — overriding the configured "
                             f"{network_type}")
                network_type = "lokr"
            w1 = f.get_slice(next(k for k in keys if k.endswith(".lokr_w1")))
            if int(w1.get_shape()[0]) != int(lokr_factor):
                notes.append(f"LoKR factor {int(w1.get_shape()[0])} from the checkpoint "
                             f"(configured: {lokr_factor})")
                lokr_factor = int(w1.get_shape()[0])
        else:
            if network_type != "lora":
                notes.append(f"the checkpoint is a standard LoRA — overriding the "
                             f"configured {network_type}")
                network_type = "lora"
            dk = next((k for k in keys if k.endswith(".lora_down.weight")
                       and "adaln" not in k), None)
            if dk is not None:
                dim = int(f.get_slice(dk).get_shape()[0])
                ak = dk.replace(".lora_down.weight", ".alpha")
                alpha = float(f.get_tensor(ak)) if ak in keys else float(dim)
                if dim != int(network_dim) or abs(alpha - float(network_alpha)) > 1e-6:
                    notes.append(f"rank {dim}/alpha {alpha:g} from the checkpoint "
                                 f"(configured: {network_dim}/{float(network_alpha):g})")
                    network_dim, network_alpha = dim, alpha
    return network_type, network_dim, network_alpha, lokr_factor, notes


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
    _validate_state_dir(state_dir)
    import json
    from safetensors.torch import load_file
    # strict=False tolerates benign key drift, but if NOTHING matched the network silently stays
    # at its zero init and the run "succeeds" while training from scratch — then overwrites the
    # finished LoRA with a no-op. Refuse that outright.
    _incompat = network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
    _missing = getattr(_incompat, "missing_keys", [])
    if _missing and len(_missing) >= len(network.state_dict()):
        raise RuntimeError(
            f"[state] {state_dir} matched none of this network's {len(network.state_dict())} keys — "
            f"refusing to resume into a zero-initialised network. The state was almost certainly "
            f"saved with a different config (rank/alpha/factor, network type, or target modules).")
    opt_path = os.path.join(state_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    meta_path = os.path.join(state_dir, "training_state.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    is_lokr = getattr(network, "_network_type", "lora") == "lokr"
    if is_lokr:
        metadata = {
            "ss_network_module": "fizgig.minimax (lokr, transformer blocks)",
            "ss_lokr_factor": str(getattr(network, "_lokr_factor", "")),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    else:
        metadata = {
            "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": str(network_alpha),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    if extra_metadata:
        metadata.update(extra_metadata)
    if is_lokr:
        # LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*) — the format every ComfyUI
        # LoKR in the wild uses. Unlike Krea 2 (whose internal saves stay native for resume and
        # previews), MiniMax has neither, and every checkpoint's only consumer is ComfyUI — so
        # every LoKR save is comfy-format. Fizgig's own loader ingests both namings via
        # ensure_kohya_lora_state_dict.
        from fizgig.networks.lora import _precalculate_safetensors_hashes
        from safetensors.torch import save_file
        dotted = getattr(network, "_dotted_names", {})
        sd = {}
        for k, v in network.state_dict().items():
            mod, _, suffix = k.partition(".")
            path_dotted = dotted.get(mod)
            nk = f"diffusion_model.{path_dotted}.{suffix}" if path_dotted else k
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            sd[nk] = v
        model_hash, legacy_hash = _precalculate_safetensors_hashes(sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(sd, path, metadata)
        return
    network.save_weights(path, dtype, metadata)


def _reg_subject_keys(datasets):
    """(reg item_keys, subject item_keys) across the dataset group — bare stems, the same
    strings the collate puts in batch['item_keys']. Split out of train_minimax so the
    collision handling is pinnable without a full trainer boot."""
    reg, subj = set(), set()
    for _ds in datasets:
        _bm = getattr(_ds, "batch_manager", None)
        if _bm is None:
            continue
        _tgt = reg if getattr(_ds, "is_reg", False) else subj
        for _bucket in _bm.buckets.values():
            for _it in _bucket:
                _tgt.add(str(_it.item_key))
    return reg, subj


def train_minimax(
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit_path: str,
    *,
    network_dim: int = 16,
    network_alpha: float = 16,
    network_type: str = "lora",      # "lora" | "lokr" (Kronecker, full-matrix w2)
    lokr_factor: int = 8,            # LoKR only: w1 is ~factor x factor; dim/alpha unused
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    # Resumable state dirs (network + optimizer + RNG + adaptive scalars). Pause saves state
    # regardless of these — they govern only the automatic per-checkpoint / end-of-run saves.
    save_state: bool = False,
    save_state_on_train_end: bool = False,
    keep_last_n_states: int = 2,
    resume_state_dir: str = None,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    caption_dropout: float = 0.05,
    # Clips with sound only. Audio is ~4% of the packed sequence at any clip length, so parity
    # may well be too quiet to teach anything — but it starts there, and moves on a measurement
    # rather than a guess. The per-epoch [audio] line reports the share it is actually winning.
    audio_weight: float = 1.0,
    base_quant: str = "auto",
    include_patterns: list = None,
    train_blocks: str = None,        # "14-37" = train only that block range (experiment)
    photo_blocks: str = None,        # Optimised Likeness Learning: photo steps update only these
                                     # blocks (+refiners); video/audio clips update everything.
                                     # The 20-49 recipe: photo gradients into the front trunk are
                                     # pure prior damage (deformed previews, eroded prompt
                                     # following) while identity lives in the back 30 blocks.
    clip_blocks: str = None,         # Confine CLIP steps to these blocks too (the GUI's "Restrict
                                     # video to likeness blocks" tickbox passes the likeness set).
                                     # FT: the rotation cycle honours it; LoRA (2 Sep): the same
                                     # per-step gradient mask as photo_blocks, keyed on clip-only
                                     # windows. Field result 29 Aug: an overnight video run
                                     # confined this way trained perfectly well. Unset = clips
                                     # train the whole model, the original behaviour.
    audio_blocks: str = None,        # Voice routing: audio-only steps update only these blocks
                                     # (+refiners). The 34-49 recipe (voice core 38-48 + shoulder,
                                     # RESEARCH_h3_block_map.md): audio gradients outside it
                                     # measurably corrupt the visual blocks (A/B, 24 Aug —
                                     # audio-only @34-49 clean, @20-49 damaged visuals).
    train_adaln: bool = True,        # False = drop adaln_proj from the targets (pruned only)
    distill: bool = False,           # reference distillation (references come from the dataset)
    distill_weight: float = 0.8,     # teacher share of the loss; the rest is the real photo
    distill_phase1_epochs: int = -1,  # identity-first: teacher-ONLY epochs, then photos-only
                                      # (-1 = auto from dataset size, 0 = off/blended)
    slow_blocks: str = None,         # block spec trained at a reduced LR ("21-49")
    block_limit: float = 0.0,   # >0 = per-block movement cap at N x the median block (the limiter)
    adapter_ramp: float = 0.0,  # >0 = hold each step at this FRACTION of the adapter's size
    gradient_accumulation_steps: int = 1,  # batches summed per optimizer step (effective batch)
    lr_warmup_epochs: float = 0.0,  # >0 = linear LR ramp over the first N epochs (static LR only)
    ema_decay: float = 0.0,     # >0 = save/preview the EMA of the adapter instead of raw weights
    slow_block_lr_scale: float = 1.0,  # the multiplier applied to those blocks' LR
    quantize: bool = True,           # NF4 the base (QLoRA); False = bf16 base (needs ~66 GB VRAM)
    shift: float = None,             # None = auto resolution schedule (logit-normal); float = legacy
    highnoise_lr_scale: float = 1.0,  # LR multiplier for steps above sigma 0.5; 1.0 = unchanged
    # Mixed visual+voice datasets: each category can retire at its own epoch, because the two
    # need not converge together (a much smaller category can finish, or start to overbake,
    # well before the larger one). "anchor" keeps the
    # retired category training at ANCHOR_LR_SCALE — rehearsal against drift on the shared
    # adapters, and its epoch ledger stays live as the drift alarm; "stop" skips its steps
    # entirely (faster epochs, blind). 0 = never retire.
    visual_stop_epoch: int = 0,
    visual_stop_mode: str = "anchor",
    audio_stop_epoch: int = 0,
    audio_stop_mode: str = "anchor",
    blocks_to_swap="auto",           # "auto" | int — park the last N blocks on CPU between uses
    gradient_checkpointing="auto",   # "auto" | "on" | "off" — forced on when swap > 0
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # In-training previews. Prompts come from the Samples tab; the text encoder is loaded ONCE
    # before the DiT (it must never be resident alongside it) and freed.
    sample_prompts: list = None,
    te_path: str = None,
    vae_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_at_first: bool = False,
    # H3's native canvas: 768 short edge, 768*1344 pixel cap.
    sample_width: int = 768,
    sample_height: int = 768,
    # 28, matching the reference pipeline's default. 8 leaves the latent well off the
    # encoder's manifold, which is exactly where the decoder produces patchy output
    # (measured seam energy 4.0 on an off-manifold latent vs 1.05 on a real one).
    sample_steps: int = 28,
    sample_cfg_scale: float = 1.0,
    sample_frames: int = 1,      # pixel frames on the 17n+5 grid; 1 = classic still
    sample_negative: str = None,
    sample_seed: int = 42,
    # Turbo LoRA for previews ONLY: applied once with every module disabled, flipped on for
    # the sampling phase, off again after. The training math never sees it.
    turbo_lora_path: str = None,
    turbo_lora_strength: float = 0.75,
    # Context LoRA: an existing LoRA held frozen + active under the trainable one for every
    # training step; OFF for previews, so they show the LoRA as it deploys. Recorded in the
    # output metadata. Not available under rotation fine-tune.
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    # Training adapter (Ostris's assistant LoRA, ostris/minimax_h3_training_adapter): the
    # same frozen-layer mechanism at a fixed strength of 1.0, stacked UNDER the context
    # LoRA. De-distills the base while the LoRA learns; off for previews like the context.
    training_adapter_path: str = None,
    # Previews with sound: decode the jointly-denoised audio rows to a .wav beside each clip
    # sample. Needs the audio VAE (its decoder half); silently off without it.
    sample_audio: bool = False,
    audio_vae_path: str = None,
    # Output metadata (recorded in the saved LoRA).
    metadata_title: str = None,
    metadata_author: str = None,
    metadata_description: str = None,
    metadata_license: str = None,
    metadata_tags: str = None,
    metadata_trigger_phrase: str = None,
    # Rotation full fine-tune (trains the BASE, not a LoRA). finetune_rotation > 0 is the
    # master switch. COMPONENT windows only (24 Aug: block/numeric modes removed — component
    # on the NF4 base is the mode that actually works): each window is one matmul
    # (qkv/out/fc1/fc2) across every block, 4 windows per cycle. Scope "photo" skips
    # clip/audio batches; finetune_blocks restricts the rotation cycle to a block subset
    # (the likeness recipe is "20-49"). Continuation = point --dit at the last saved
    # checkpoint and set --finetune_start_window (printed at every save); state-dir resume
    # is refused.
    finetune_rotation: int = 0,
    finetune_rotate_every: int = 1,
    finetune_rotation_mode: str = "component",
    finetune_start_window: int = 0,
    finetune_fused_backward: bool = True,
    finetune_scope: str = "all",            # "all" | "photo"
    finetune_blocks: str = None,
    finetune_master: str = "auto",          # "auto" | "ram" | "disk" — see the mode select
    finetune_scratch_dir: str = None,       # disk mode's spill dir; default: beside the caches
    reg_lr_multiplier: float = 0.2,         # FT only: LR nudge for `is_reg` dataset blocks
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native MiniMax H3 image-only LoRA training: bucketed dataloader over the H3 caches ->
    flow-matching loss -> optimizer -> save a ComfyUI-compatible LoRA. No samples, no preview."""
    from torch.utils.data import DataLoader

    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.networks.lora import create_network
    from fizgig.training.optimizers import create_optimizer
    from fizgig.training.train_utils import LossRecorder, validate_output_name
    from fizgig.training.metadata import build_metadata, resolve_title, ARCHITECTURE_MINIMAX
    from fizgig.minimax.loader import load_minimax_h3_dit
    from tqdm import tqdm
    import math

    torch.manual_seed(seed)
    user_include_patterns = include_patterns   # None -> resolved per checkpoint below
    # Parse the block selection NOW, before the 21 GB base streams in: a typo surfacing after
    # the load costs minutes and reads like a crash rather than a correction. Bounds-checking
    # waits until the model is up (that is when the real block count is known).
    if train_blocks:
        parse_block_spec(train_blocks)
    validate_output_name(output_name)          # same reason, one epoch later otherwise (#70)

    # ---- rotation full fine-tune: resolve the config and disarm what can't coexist ----
    # Mirrors krea2/trainer.py's FT coercion block. Everything forced here is forced for a
    # structural reason, not taste — each line names it.
    ft_rotation = max(0, int(finetune_rotation or 0))
    rotator = None
    ft_subset = None
    ft_epoch_offset = 0
    # FT streaming state — defined unconditionally so closures shared with LoRA mode
    # (e.g. _encode_override) can test it without tripping a NameError.
    ft_stream = False
    _ft_ring = {"ring": None}
    if ft_rotation:
        from fizgig.utils.capabilities import wait_for_gpu_handoff, wait_for_ram_recovery
        # Before OUR first CUDA call (the recommender's free-VRAM read inits the context):
        # a back-to-back fine-tune can start while the previous trainer process is still
        # tearing down, and the resulting WDDM demotion is sticky — see the guards' docstrings.
        # Two legs: VRAM (driver view) and RAM (a finished FT hands back ~120 GB of commit).
        wait_for_gpu_handoff()
        wait_for_ram_recovery()
        if resume_state_dir:
            raise RuntimeError(
                "[h3-ft] --resume with a state dir is not supported under rotation fine-tune: "
                "state dirs hold the (inert) LoRA, not base weights. To continue a fine-tune, "
                "point --dit at the last saved checkpoint and set --finetune_start_window to "
                "the value printed at that save.")
        if distill:
            raise RuntimeError(
                "[h3-ft] reference distillation is LoRA-only: the teacher pass assumes an "
                "adapter it can switch off (lora_disabled), which a fine-tune doesn't have. "
                "Turn one of the two off.")
        # Component is THE mode (24 Aug): block/numeric windows are gone — they never
        # matched component's likeness speed and their int8-residency path is dead weight.
        if not str(finetune_rotation_mode).startswith("comp"):
            raise RuntimeError(
                f"[h3-ft] rotation mode {finetune_rotation_mode!r} was removed — component "
                "is the only H3 fine-tune mode (one matmul across every block per window).")
        if finetune_scope not in ("all", "photo"):
            raise RuntimeError(f"[h3-ft] finetune_scope must be 'all' or 'photo', "
                               f"got {finetune_scope!r}")
        if finetune_blocks:
            parse_block_spec(finetune_blocks)   # early typo check; bounds after the load
        # Continuation numbering (mirrors krea2): a fine-tune continued from a saved
        # checkpoint restarts local epochs at 1, so `<name>-000001.safetensors` would
        # silently OVERWRITE the original run's first checkpoint. The --dit file IS the
        # previous checkpoint — offset every checkpoint filename by its trailing epoch
        # number (the FINAL file has no number; its count rides in fizgig_ft_epochs_done).
        _m = re.search(r"-(\d{6})\.safetensors$", os.path.basename(dit_path or ""))
        if _m:
            ft_epoch_offset = int(_m.group(1))
        else:
            try:
                from safetensors import safe_open
                with safe_open(dit_path, framework="pt") as _f:
                    _md = _f.metadata() or {}
                ft_epoch_offset = int(_md.get("fizgig_ft_epochs_done", 0))
            except Exception:
                ft_epoch_offset = 0
        if ft_epoch_offset:
            logger.info("[h3-ft] continuing from %s — checkpoint numbering starts at "
                        "epoch %d", os.path.basename(dit_path), ft_epoch_offset + 1)
        # Structural disarms (each mirrors a Krea FT coercion):
        blocks_to_swap = 0          # the H2D offloader would fight the rotator for qdata
        # Component windows only coexist with an NF4 trunk: one matmul across every block
        # is up to 15.4 GB of bf16, which fits beside a ~10.5 GB NF4 base, not the ~21 GB
        # int8 one. The trunk's ~9.5% NF4 error is training-time only; the SAVED checkpoint
        # still writes bf16-master -> int8 ConvRot. (The master always dequantizes from the
        # int8 FILE, so residency never affects its fidelity.)
        base_quant = "nf4"
        adaptive_lr = False         # rotation boundaries read as instability to the watcher
        # photo_blocks is KEPT under FT: it is the likeness intent, honoured with the same
        # semantics as LoRA mode — photos feed the identity blocks, clips/voice the full
        # model. Resolution (cycle-tighten vs per-window gating) happens after the dataset
        # is known, in the rotator construction block.
        network_type = "lora"       # the network is built inert; LoKR keys would just churn
        block_limit = 0.0           # movement governors measure LoRA movement
        adapter_ramp = 0.0
        ema_decay = 0.0
        lr_warmup_epochs = 0.0
        slow_blocks = None
        # The band multiplier and retirement ANCHORS both work by rewriting the optimizer's
        # param-group LR at boundary steps — machinery FT doesn't have (fused: no optimizer
        # object at all; non-fused: rotation rebuilds discard the stashed base LR). Left
        # armed they crash, not degrade. So: the band multiplier is coerced off, and
        # retirement is kept but in STOP form only — the skip path owns no optimizer state,
        # and the stop epochs are snapped to rotation-cycle boundaries below (once the
        # schedule exists) so every window sees the identical data mix for equal passes
        # before the mix changes.
        if abs(float(highnoise_lr_scale or 1.0) - 1.0) > 1e-9:
            logger.info("[h3-ft] Medium to High LR is a LoRA-mode knob — the fine-tune "
                        "trains every noise level at the configured rate.")
        highnoise_lr_scale = 1.0
        if ((int(visual_stop_epoch or 0) and visual_stop_mode == "anchor")
                or (int(audio_stop_epoch or 0) and audio_stop_mode == "anchor")):
            logger.info("[h3-ft] the 'anchor at 10% LR' retirement mode is LoRA-mode "
                        "machinery — under the fine-tune a retired category stops entirely.")
        visual_stop_mode = "stop"
        audio_stop_mode = "stop"
        # Previews stay ON under FT, but at CYCLE cadence only: they render when every block
        # has had the same number of passes, via a deactivate-all -> standard preview ->
        # reactivate bracket (the Sample-every-N box is overridden with a log line below).
        if finetune_fused_backward:
            max_grad_norm = 0.0             # grads are consumed per-tensor as they land
            gradient_accumulation_steps = 1  # nothing left to accumulate
        logger.info("[h3-ft] ROTATION FINE-TUNE: component windows on an NF4 base — each "
                    "window is one matmul (qkv/out/fc1/fc2) across every block, so a "
                    "concept trains at full model depth each epoch. The frozen trunk "
                    "carries NF4's ~9.5%% error during training (the saved checkpoint "
                    "is still exact int8, written from the bf16 master). Scope=%s%s%s.",
                    finetune_scope,
                    f", blocks {finetune_blocks}" if finetune_blocks else "",
                    ", fused backward" if finetune_fused_backward else "")
        # FT has never been run on AMD/ROCm — every measured tier is NVIDIA, and the NF4
        # rotator leans on bitsandbytes 4-bit, the least-travelled part of the ROCm stack.
        # Log-only: a power user gets the facts, not a gate. (Twin of the Krea 2 warning.)
        if getattr(torch.version, "hip", None):
            logger.warning("[h3-ft] heads-up: fine-tuning is UNTESTED on AMD/ROCm — every "
                           "measured tier is NVIDIA. The NF4 rotator depends on bitsandbytes "
                           "4-bit, the least-tested part of the ROCm stack. It may work; if "
                           "it does (or doesn't), a report on GitHub genuinely helps.")

    # ---- dataset (built from the caches the two cache scripts wrote) ----
    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run minimax_cache_latents then minimax_cache_text first.")
    logger.info(f"MiniMax H3 training: {group.num_train_items} items, {max_train_epochs} epochs")

    # FIZGIG_SAVED_TENSOR_AUDIT=1: account every tensor autograd saves for backward, with the
    # stack that saved it. Holders unregister on free, so whatever remains at a failed park is
    # the live graph — named by file:line. Diagnostic for the 16 GB weight-retention hunt.
    if os.environ.get("FIZGIG_SAVED_TENSOR_AUDIT"):
        import traceback as _tb
        import torch.autograd.graph as _ag
        _SAVED_REG = {}
        globals()["_SAVED_TENSOR_REG"] = _SAVED_REG

        class _SavedHold:
            __slots__ = ("t", "k")

            def __init__(self, t):
                self.t = t
                self.k = id(self)
                try:
                    if t.is_cuda and t.numel() * t.element_size() > 8 * 2**20:
                        stk = "|".join(f"{f.filename.rsplit(chr(92), 1)[-1]}:{f.lineno}"
                                       for f in _tb.extract_stack(limit=8)[:-2][-4:])
                    else:
                        stk = ""
                    _SAVED_REG[self.k] = (tuple(t.shape),
                                          t.numel() * t.element_size() if t.is_cuda else 0,
                                          stk)
                except Exception:
                    pass

            def __del__(self):
                _SAVED_REG.pop(self.k, None)

        _audit_ctx = _ag.saved_tensors_hooks(lambda t: _SavedHold(t), lambda h: h.t)
        _audit_ctx.__enter__()
        logger.warning("[audit] saved-tensor accounting ON — expect slower steps")
    if shift is None:
        logger.info("[timesteps] shift-12 uniform map (median sigma ~0.92) — H3's own training "
                    "density, matching the reference trainer")
    elif shift == "sigmoid":
        logger.warning("[timesteps] UNSHIFTED logit-normal (median 0.5) — A/B mode. At 1e-4 this "
                       "overdrives adapters within an epoch on small datasets; the default "
                       "(omit --shift) is the reference recipe.")
    elif shift == "resolution":
        logger.warning("[timesteps] logit-normal + resolution shift (median ~0.62) — A/B mode; "
                       "same overdrive caveat as sigmoid.")
    elif isinstance(shift, str) and str(shift).startswith("lognorm:"):
        logger.info(f"[timesteps] logit-normal base at shift {str(shift).split(':', 1)[1]} — "
                    f"mid-concentrated spread at the requested low-noise share.")
    else:
        logger.info(f"[timesteps] explicit shift={shift} — uniform-u map.")

    # ---- VRAM plan: block swap + gradient checkpointing (before the base loads) ----
    # A CLIP costs its spatial size TIMES its latent frames, and the planner's activation term is
    # linear in exactly that — so the plan is built from the heaviest single ITEM's product, read
    # from the cache headers. The bucket alone makes a 124-frame run look like the still it is
    # one frame of; two separate maxima (largest bucket x longest T) would charge a 1 MP stills +
    # voice dataset for a 37 MP step that no item performs.
    #
    # Measured, 0.25 MP buckets, gradient checkpointing on: activations run 1.7 GiB at 22 frames
    # to 9.2 at 124, dead linear. Left as the bucket size, a 32 GB card is told it can afford
    # int8 (19.8 GiB resident), the 124-frame step then peaks at 28.9 of 31.8 and the allocator
    # thrashes: 300 s a step against 17.7 s for the same step on NF4. Not an error — just a run
    # that never finishes, for want of telling the planner what it was planning for.
    _mp = 0.25
    try:
        # Bucket fallback for when no cache exists yet. key[-2:] rather than unpacking the
        # key: the audio sentinel is ("audio", w, h) and a bare (w, h) unpack would throw,
        # silently leaving the whole estimate at 0.25.
        _mp = max(key[-2] * key[-1] / 1e6
                  for ds in group.datasets for key in ds.batch_manager.bucket_resos)
    except Exception:
        pass
    _eff_mp, _clip_t = _max_effective_mp(group)
    if _eff_mp > 0:
        _mp = _eff_mp
    if _clip_t > 1:
        logger.info(f"[vram] the heaviest item is a {_clip_t}-latent-frame clip — planning "
                    f"against its effective {_mp:.2f} MP (spatial size x frames).")
    _ckpt_req = str(gradient_checkpointing).lower()
    _base_mode = (base_quant if base_quant != "auto"
                  else ("int8" if is_pruned_checkpoint(dit_path) else "nf4"))
    if not quantize:
        _base_mode = "none"
    # Any swap on a quantized base rides an H2D ring now — int8 through rintic-13's
    # ConvRot ring (#73), NF4 through @mabseyuk's Linear4bit ring, HQQ through rintic-13's
    # HQQ ring (#102). Planner-owned, no opt-in; FIZGIG_NO_NF4_H2D=1 is the debug
    # kill-switch back to classic parking for both 4-bit rings.
    # Evaluated at USE time, not here: _base_mode is reassigned to the planner's
    # RESOLVED mode below (Auto's pre-plan guess of int8 can resolve to nf4 under the
    # streaming floor), and a snapshot taken now made the kill-switch dead on exactly
    # the default path where the NF4 ring is reached (audit, 25 Aug).
    def _ring_planned():
        return (_base_mode == "int8"
                or (_base_mode in ("nf4", "hqq")
                    and os.environ.get("FIZGIG_NO_NF4_H2D") != "1"))

    def _resident_for(mode, pruned):
        if mode == "int8":
            return _RESIDENT_INT8_GB
        if mode == "hqq":
            return _RESIDENT_HQQ_PRUNED_GB if pruned else _RESIDENT_HQQ_GB
        return _RESIDENT_PRUNED_GB if pruned else _RESIDENT_GB
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available() and quantize:
            from fizgig.utils.device import plannable_free_vram
            _free_gb = plannable_free_vram()   # honours FIZGIG_SIM_VRAM_GB
            _pruned = is_pruned_checkpoint(dit_path)
            # The adapter is NOT a rounding error and it is not fixed: LoKR 8 trains ~313 M
            # parameters against a rank-16 LoRA's ~75 M, and fp32 Adam state is 4x the 8-bit
            # one. Planning without it was planning for a configuration nobody runs — the
            # anchors were measured on rank-16 + adamw8bit (~0.45 GB) while the shipped default
            # is LoKR 8 + adamw (~3.8 GB). Shapes come from the checkpoint header, so this is
            # the real targeted module set for whichever file is loaded.
            _pat = PRUNED_INCLUDE_PATTERNS if _pruned else DEFAULT_INCLUDE_PATTERNS
            if not train_adaln:
                _pat = [p for p in _pat if "adaln" not in p]
            _ad_params = adapter_param_count(dit_path, _pat, network_type=network_type,
                                             network_dim=network_dim, lokr_factor=lokr_factor,
                                             train_blocks=train_blocks)
            _adapter = adapter_vram_gb(_ad_params, optimizer_type)

            if base_quant == "auto":
                _mode, n_swap, _ckpt_auto, _why = plan_base_quant(
                    _free_gb, _pruned, mp=_mp, adapter_gb=_adapter)
            else:
                # An explicit choice is never overridden — the plan is built AROUND it, or the
                # swap count would be sized for a quantisation that will not run.
                _mode = base_quant
                n_swap, _ckpt_auto = plan_vram(
                    _free_gb, mp=_mp, resident_gb=_resident_for(_mode, _pruned),
                    transient_gb=_INT8_TRANSIENT_GB if _mode == "int8" else 0.0,
                    adapter_gb=_adapter)
                _why = f"base precision pinned to {_mode} by the user"
                if _mode == "hqq":
                    _why += (" (HQQ: ~45% more resident than nf4 and roughly half the "
                             "step speed — PyTorch-path dequant)")
            _base_mode = _mode
            _resident = _resident_for(_mode, _pruned)

            logger.info(f"[vram] auto plan: free {_free_gb:.1f} GB, largest bucket {_mp:.2f} MP, "
                        f"base ~{_resident:.0f} GB ({_mode}, {'pruned' if _pruned else 'bf16'}), "
                        f"adapter ~{_adapter:.1f} GB ({_ad_params/1e6:.0f} M params, "
                        f"{optimizer_type}) -> blocks_to_swap={n_swap}, "
                        f"checkpointing={'on' if _ckpt_auto else 'off'}")
            logger.info(f"[vram] base precision: {_mode} — {_why}")
            if _mode == "nf4" and _pruned and base_quant == "auto":
                # Say it plainly rather than quietly downgrading the base: this costs likeness,
                # and the user has a real alternative (train slower on int8, or free some VRAM).
                logger.warning(
                    "[vram] this run trains on a 4-bit base (~9% error) instead of the "
                    "checkpoint's own int8 (~0.17%). It is much faster here, but the LoRA spends "
                    "some capacity correcting quantization error that will NOT exist at "
                    "inference. To force the accurate base, set Base Precision to int8 — expect "
                    "block swap and a several-times-slower run — or close other GPU apps and "
                    "re-launch.")
            if n_swap > 0 and not _ring_planned():
                # Only the CLASSIC parking swap earns the scary line — ring-streamed
                # blocks (int8 and NF4 alike) cross PCIe one-way with prefetch and cost
                # a fraction of that. The ring path logs its own line at activation.
                logger.warning(
                    f"[vram] {n_swap} of 50 blocks will live on CPU and cross PCIe every step, "
                    f"which is several times slower. Lower Target Megapixels, or free VRAM, to "
                    f"avoid it.")
            if n_swap >= 40:
                # A plan AT the cap may be a plan that doesn't fit at all (4090 field
                # OOM): say so loudly with the number, instead of a confident plan line
                # followed by a step-1 OOM. The run still proceeds — Windows can spill
                # to shared memory and limp — but nobody should discover an ~8 GB hole
                # from a traceback. Transient/per-block match how the swap actually
                # runs: ring-streamed blocks never round-trip, so their backward
                # transient is the ring, not classic parking's recompute segments.
                _swap_t = (_H2D_TRANSIENT_GB if _ring_planned()
                           else _SWAP_TRANSIENT_GB)
                _short = plan_swap_shortfall_gb(
                    _free_gb, mp=_mp, resident_gb=_resident,
                    transient_gb=_INT8_TRANSIENT_GB if _mode == "int8" else 0.0,
                    adapter_gb=_adapter,
                    swap_transient_gb=_swap_t,
                    per_block_gb=(_H2D_PER_BLOCK_GB if _mode == "int8"
                                  else _PER_BLOCK_GB))
                # Margins are not shortfall (#101, MEASURED): the arithmetic's "need"
                # includes the reserve and the swap transient, which exist for spikes —
                # plans it called 0.9 GB short (field 5070) and 2.5 GB short (sim-12
                # gate, HARD allocator cap) both trained at ~1.3 s/it with headroom.
                # Crying wolf there costs the warning its credibility, so it fires only
                # once the deficit eats THROUGH the margins — the 8-12 GB monster-clip
                # holes it was built for (the 4090 report) sail past this bar.
                _margins = _RESERVE_GB + _swap_t
                if _short > _margins + 0.5:
                    logger.warning(
                        f"[vram] this plan does NOT fit: even at the 40-block swap cap, "
                        f"and with every safety margin spent, it is ~{_short - _margins:.1f} "
                        f"GB short for the heaviest item in this dataset ({_mp:.2f} "
                        f"effective MP — spatial size x clip frames). Expect an "
                        f"out-of-memory error at the first training step, or a severe "
                        f"slowdown if Windows spills to shared memory. Lower Target "
                        f"Megapixels, shorten or downscale the heaviest clips, or free "
                        f"VRAM and re-launch.")
                elif _short > 0.5:
                    logger.info(
                        f"[vram] tight fit: the plan leans ~{_short:.1f} GB into its "
                        f"safety margins at the 40-block cap — training measured fine "
                        f"at this tier, but previews may downgrade or switch off "
                        f"(training and checkpoints are never at risk). Close other "
                        f"GPU apps if the first step OOMs.")
        else:
            n_swap, _ckpt_auto = 0, False
    else:
        n_swap = max(0, int(blocks_to_swap))
        _ckpt_auto = n_swap > 0
        if base_quant == "auto" and quantize:
            # A hand-set swap count skips the planner, and the planner is what weighs int8's
            # residency against free VRAM. Auto precision then falls back to "whatever the file
            # is", which on a pruned checkpoint is always int8 — ~21 GB and a slow run on a card
            # the planner would have put on 4-bit with no swap at all. It does not fail, so
            # nothing would otherwise say why the run is crawling.
            logger.info(f"[vram] base precision: {_base_mode} — chosen from the checkpoint, not "
                        f"from free VRAM, because Blocks Swap is set to {n_swap} rather than "
                        f"Auto. Set Blocks Swap to Auto to have the precision and the swap count "
                        f"planned together.")
    use_ckpt = {"on": True, "off": False}.get(_ckpt_req, _ckpt_auto)
    if n_swap > 0 and not use_ckpt:
        logger.info("[vram] block swap needs gradient checkpointing (autograd would pin swapped "
                    "weights through backward) — forcing it on.")
        use_ckpt = True
    if ft_rotation:
        # FT plans its own VRAM: no swap, NF4 base, and
        # checkpointing ON — the window budget was sized against checkpointed activations.
        # The int8 file is a hard requirement: the master dequantizes from its codes.
        if not is_pruned_checkpoint(dit_path):
            raise RuntimeError(
                "[h3-ft] rotation fine-tune needs the pre-quantized int8 checkpoint "
                "(minimax_h3_*_pruned_int8_convrot.safetensors) — this file has no ConvRot "
                "codes to build the master from.")
        use_ckpt = True
        logger.info("[h3-ft] budget: ~10.5 GB NF4 resident + one component of bf16 "
                    "(worst window fc1, ~%.1f GB); bf16 CPU master ~%.1f GB RAM%s",
                    (len(parse_block_spec(finetune_blocks, 50)) if finetune_blocks
                     else 50) * 0.308,
                    (len(parse_block_spec(finetune_blocks, 50)) if finetune_blocks
                     else 50) * 0.771,
                    f" (blocks {finetune_blocks} only)" if finetune_blocks else "")

    # ---- previews: encode the prompts BEFORE the DiT loads ----
    # Order matters more here than anywhere else in Fizgig: the Qwen3-VL-32B text encoder is
    # ~14 GB even at NF4, and the DiT is ~17 GB. They must never be resident together, so the
    # prompts are encoded once, up front, and the encoder is freed before the base streams in.
    do_previews = bool((sample_every_n_epochs or sample_at_first) and sample_prompts and te_path)
    encoded_prompts = encoded_negative = sample_dir = None
    if do_previews:
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info(f"[preview] pre-encoding {len(sample_prompts)} sample prompt(s) "
                    f"(the text encoder is freed before the DiT loads)...")
        # 16 GB-class cards: previews hard-cap at 768x768 and 22 frames (sound untouched —
        # it's a separate flag and ~4% of the sequence). The Samples menu still offers the
        # longer clips; here they clamp rather than crash-and-ladder: a 56-frame clip's
        # sampling tokens plus its chunked decode is exactly what pushed a real 16 GB 4090
        # into the paging/OOM spiral (Peter, 19 Aug). Clamp, say so once, move on.
        try:
            _total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            _total_gb = 99.0
        if _total_gb < 20.0:
            _clamped = []
            if int(sample_frames or 1) > 22:
                _clamped.append(f"{sample_frames} frames -> 22")
                sample_frames = 22
            # Resolution rule lives in cap_preview_res_small_card — shared with the live
            # sample override so neither path can exceed the other.
            _new = cap_preview_res_small_card(sample_width, sample_height)
            if _new != (sample_width, sample_height):
                _clamped.append(f"{sample_width}x{sample_height} -> {_new[0]}x{_new[1]}")
                sample_width, sample_height = _new
            if _clamped:
                logger.info(f"[preview] 16 GB card: {'; '.join(_clamped)} — previews cap at "
                            f"768x640 / 22 frames on this class of GPU (sound kept)")
        try:
            encoded_prompts = encode_sample_prompts(te_path, sample_prompts, device=device,
                                                    quantize=quantize)
            if sample_negative and sample_cfg_scale and sample_cfg_scale > 1.0:
                encoded_negative = encode_sample_prompts(te_path, [sample_negative],
                                                         device=device, quantize=quantize)[0]
            sample_dir = os.path.join(output_dir, "sample")
            os.makedirs(sample_dir, exist_ok=True)
            # State the whole preview recipe once, up front — steps in particular, since too
            # few leaves the latent off-manifold and the decode patchy.
            logger.info(
                f"[preview] {sample_steps} steps @ {sample_width}x{sample_height}, "
                f"cfg {sample_cfg_scale:g}"
                f"{'' if sample_cfg_scale > 1.0 else ' (off — H3 is guidance-distilled)'}, "
                f"seed {sample_seed if sample_seed else 'random'}, "
                f"every {sample_every_n_epochs} epoch(s)"
                f"{', plus epoch 0' if sample_at_first else ''} — "
                f"{'full VAE decode' if vae_path else 'RGB approximation (no VAE path set)'}")
        except Exception as _e:
            logger.warning(f"[preview] prompt encoding failed ({type(_e).__name__}: {_e}) — "
                           f"previews disabled; training continues normally.")
            do_previews = False

    # ---- base (NF4-frozen) + trainable LoRA over the transformer blocks ----
    # adaln_fp32 matches ComfyUI's curve-checkpoint dtype, but only when AdaLN is NOT a LoRA
    # target — a bf16 adapter cannot take an fp32 activation from the Linear it wraps.
    # base_quant is the RESOLVED mode, never the raw "auto" (issue #55). The loader has its own
    # auto rule — int8 whenever the file is pre-quantized — which ignores how much VRAM is free,
    # so handing it "auto" threw away the plan: swap sized for an ~11 GB NF4 base while a ~21 GB
    # int8 one loaded, and the log printed both decisions a few lines apart. It also made
    # ss_base_quant a lie in the output metadata.
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=quantize,
                              blocks_to_swap=n_swap, base_quant=_base_mode,
                              adaln_fp32=not train_adaln)
    dit.requires_grad_(False)                                   # frozen base (QLoRA-style)
    if n_swap > 0:
        # Quantized bases stream H2D-only: the base is frozen, so the classic swap's
        # writeback half was always waste — a ring buffer + copy stream prefetches each
        # block while the previous computes. int8 rides rintic-13's ConvRot ring (#73);
        # NF4 rides @mabseyuk's Linear4bit ring (the tier 12 GB cards land on — his 5070
        # went from 12-14 s/step parked to ~1 s/step streamed); HQQ rides rintic-13's
        # HQQ ring (#102, same per-slot design). enable_block_swap dispatches by module
        # type and falls back to classic parking if a ring can't build; the later
        # preview-restore calls re-enter it bare and inherit this mode.
        _use_h2d = _ring_planned()
        n_swap = dit.enable_block_swap(n_swap, h2d_only=_use_h2d, ring_size=2)
        _off = getattr(dit, "_h2d_offloader", None)
        if _off is not None:
            _staging = ("pinned in RAM"
                        if not getattr(_off, "_pin_failed", False)
                        else "staged in ordinary RAM (pinning unavailable or RAM too "
                             "tight) — copies synchronous")
            # All three ring classes declare kind/staged_gb; the explicit None test matters
            # because `or` would swallow a legitimate 0.0 into the int8 estimate.
            _kind = getattr(_off, "kind", "?")
            _gb = getattr(_off, "staged_gb", None)
            if _gb is None:
                _gb = n_swap * 0.39
            logger.info(f"[vram] block swap active: last {n_swap} blocks streamed H2D-only "
                        f"({_kind}, ring 2, ~{_gb:.1f} GB {_staging}) — no "
                        f"writeback, prefetch overlaps compute")
        else:
            # Classic parking: the planned NF4 path with the kill-switch on, OR any
            # base whose ring failed to build — so name the mode rather than assuming.
            logger.info(f"[vram] block swap active: last {n_swap} blocks parked on CPU "
                        f"(~{n_swap * 0.34:.1f} GB VRAM freed, packed {_base_mode} "
                        f"in RAM)")
    if use_ckpt:
        dit.enable_gradient_checkpointing()
        logger.info("[vram] gradient checkpointing ON")
    if ft_rotation:
        # ---- rotation FT: master + rotator + schedule ----
        # No LoRA network is built or applied under FT. This is deliberate and H3-specific:
        # LoRA's apply_to captures each module's BOUND forward at apply time, and the
        # rotator's class-swap would leave that captured ConvRot forward pointing at freed
        # int8 codes. Krea builds its network inert; H3 cannot even do that.
        from fizgig.krea2.rotation import RotationSchedule
        from fizgig.minimax.rotation_ft import build_bf16_master_h3
        # The LoRA-path ring (enable_block_swap) must not be armed here — it spans a tail
        # that includes trainable blocks and would fight the rotator for qdata. FT's OWN
        # ring (built below when the plan streams) is different: scoped to fully-frozen
        # out-of-window blocks only, rebuilt at every rotation.
        assert getattr(dit, "_h2d_offloader", None) is None, \
            "[h3-ft] H2D offloader present under FT — it would fight the rotator for qdata"
        _n_blocks = len(dit.blocks)
        if finetune_blocks:
            ft_subset = sorted(parse_block_spec(finetune_blocks, _n_blocks))
            if not ft_subset:
                raise RuntimeError(f"[h3-ft] finetune_blocks {finetune_blocks!r} selects "
                                   "no blocks")
        # Modality routing. Each modality trains only where it belongs: photos -> the
        # likeness set when ticked (photo gradients into the front trunk are pure prior
        # damage), voice -> the audio zone (audio gradients OUTSIDE 34-49 measurably
        # corrupt the visual blocks — A/B, 24 Aug), clips -> full model for now. Two
        # mechanisms compose:
        #   cycle tighten — the rotation cycle spans the UNION of what the modalities
        #       present in the dataset need, so a photos+voice dataset never spends an
        #       epoch on the front trunk and an audio-only dataset tightens to 34-49
        #       automatically (the validated A/B config, no Blocks typing required);
        #   per-batch freeze — a modality whose set is narrower than the span keeps its
        #       hands off the rest via requires_grad, rebuilt per window in
        #       _ft_rebind_optimizer (component windows span every block, so a window
        #       skip cannot express partial confinement — a parameter freeze can, and
        #       the fused per-tensor hooks simply never fire for a no-grad param).
        # An explicit --finetune_blocks range wins and disables all routing.
        try:
            from fizgig.dataset.image_dataset import VIDEO_EXTENSIONS, is_audio_path
            _vexts = {e.lower() for e in VIDEO_EXTENSIONS}
            _mr_paths = [p for ds in group.datasets
                         for p in getattr(getattr(ds, "datasource", None),
                                          "image_paths", []) or []]
            _n_voice_items = sum(1 for p in _mr_paths if is_audio_path(p))
            _n_clip_items = sum(1 for p in _mr_paths
                                if os.path.splitext(p)[1].lower() in _vexts)
            _n_photo_items = len(_mr_paths) - _n_voice_items - _n_clip_items
        except Exception:
            # Composition unreadable: assume the widest mix so the cycle never under-spans.
            _n_voice_items, _n_clip_items, _n_photo_items = 1, (1 if _clip_t > 1 else 0), 1
        if finetune_scope == "photo":
            # 'Train on: Photos only' is a dataset FILTER — clips and voice never train,
            # so they place no demand on the cycle span.
            _n_voice_items = _n_clip_items = 0
        if ft_subset is not None and ((photo_blocks and _n_photo_items)
                                      or (audio_blocks and _n_voice_items)):
            logger.info("[h3-ft] Blocks is set explicitly (%s) — the explicit range "
                        "wins over photo/voice routing.", finetune_blocks)
        _plan_subset, _ft_route = plan_ft_modality_routing(
            _n_blocks, photo_blocks, audio_blocks,
            _n_photo_items, _n_voice_items, _n_clip_items, explicit_subset=ft_subset,
            clip_blocks=clip_blocks)
        if ft_subset is None and _plan_subset is not None:
            ft_subset = _plan_subset
            logger.info("[h3-ft] the cycle tightens to blocks %s — the union of what "
                        "this dataset actually trains (%s).",
                        format_block_spec(ft_subset),
                        ", ".join(filter(None, [
                            (f"photos -> {photo_blocks}" if photo_blocks
                             else "photos -> full model") if _n_photo_items else None,
                            (f"voice -> {audio_blocks}" if audio_blocks
                             else "voice -> full model") if _n_voice_items else None,
                            ((f"clips -> {clip_blocks} (restricted)" if clip_blocks
                              else "clips -> full model")
                             if _n_clip_items else None)])))
        if _ft_route["photo"] is not None:
            logger.info("[h3-ft] Optimised Likeness Learning on a mixed dataset: photo "
                        "batches freeze every block outside %s — the same photos-"
                        "protect-the-trunk behaviour as LoRA mode, per parameter.",
                        photo_blocks)
        if _ft_route["voice"] is not None:
            logger.info("[h3-ft] voice routing: audio batches freeze every block "
                        "outside %s (the voice zone — audio gradients beyond it "
                        "corrupt the visual blocks).", audio_blocks)
        if _ft_route["clip"] is not None:
            logger.info("[h3-ft] video routing: clip batches freeze every block "
                        "outside %s (Restrict video to likeness blocks).", clip_blocks)
        # RAM vs disk-backed master. The in-RAM dict is only irreplaceable for TRAINED
        # tensors; MasterStore reads untouched ones lazily from the int8 file and spills
        # trained bf16 to scratch — master RAM drops from whole-model to ~one tensor, which
        # is what fits full-model FT on 64 GB boxes. Auto: disk when the estimated master
        # would eat >40% of available RAM (so a 128 GB box keeps the field-proven RAM path
        # for the likeness-sized masters and flips only where it matters).
        _inc = ("token_refiner",)
        _n_master_blocks = len(ft_subset) if ft_subset else _n_blocks
        _est_master_gb = _n_master_blocks * 0.771 + 0.4
        _mmode = str(finetune_master or "auto").lower()
        if _mmode == "auto":
            try:
                from fizgig.utils.capabilities import _available_ram_gb
                _avail_ram, _ = _available_ram_gb()
            except Exception:
                _avail_ram = None
            _mmode = ("disk" if (_avail_ram is not None
                                 and _est_master_gb > 0.40 * _avail_ram) else "ram")
            logger.info("[ft-master] auto -> %s (master ~%.1f GB vs %.0f GB RAM available)",
                        _mmode, _est_master_gb,
                        _avail_ram if _avail_ram is not None else -1)
        if _mmode == "disk":
            from fizgig.minimax.rotation_ft import MasterStore
            _scratch = finetune_scratch_dir
            if not _scratch:
                # Default beside the dataset caches — the cache pref lives on a fast local
                # drive; the OUTPUT dir must not be assumed fast (checkpoints often land on
                # a big slow volume).
                _cd = next((getattr(d, "cache_directory", None) for d in group.datasets
                            if getattr(d, "cache_directory", None)), None)
                _base = (os.path.dirname(str(_cd).rstrip("/\\")) if _cd else output_dir)
                _scratch = os.path.join(
                    _base, f"ft-scratch-{hashlib.sha1(output_name.encode()).hexdigest()[:8]}")
            # Fresh run = fresh training state: this run's own stale scratch is wiped
            # (continuation runs carry state via --dit <checkpoint>, never via scratch),
            # and crashed siblings older than a week are swept.
            if os.path.isdir(_scratch):
                shutil.rmtree(_scratch, ignore_errors=True)
            try:
                import glob as _glob
                import time as _time
                for _old in _glob.glob(os.path.join(os.path.dirname(_scratch),
                                                    "ft-scratch-*")):
                    if (os.path.isdir(_old) and _old != _scratch
                            and _time.time() - os.path.getmtime(_old) > 7 * 86400):
                        shutil.rmtree(_old, ignore_errors=True)
            except Exception:
                pass
            master = MasterStore(dit_path, _scratch, block_subset=ft_subset,
                                 include_prefixes=_inc)
        else:
            master = build_bf16_master_h3(dit_path, block_subset=ft_subset,
                                          include_prefixes=_inc)
        from fizgig.minimax.rotation_ft import (H3NF4Rotator, H3_COMPONENT_PREFIXES,
                                                H3_COMPONENT_GB_PER_BLOCK,
                                                plan_h3_ft_windows)
        # Window plan: how many depth-splits (and whether the frozen out-of-window blocks
        # must stream from CPU) this card's budget forces. Measured AFTER the NF4 base
        # loaded, so the trunk it already holds is added back — the plan's model counts
        # the whole process footprint. FIZGIG_NO_FT_STREAM=1 is the debug kill-switch to
        # the resident-only plan (the streaming tier then refuses rather than OOMs).
        #
        # Clip datasets reserve their activation term BEFORE window sizing: the stills-
        # calibrated overhead has no idea a 56-frame clip adds ~2.3 GB per step, and the
        # measured consequence was a 32 GB card picking the resident 4-window plan and
        # dying at fc1 while the 16/24 GB tiers (forced onto split/streamed plans) passed
        # the same dataset. The term is per-STEP and plan-independent, so subtracting it
        # from usable is exact for both the resident cap and the streaming budget.
        # Gated on _n_clip_items — already zeroed above for finetune_scope == "photo",
        # whose clip batches `continue` before any forward and so never spike.
        from fizgig.minimax.rotation_ft import ft_clip_activation_gb
        _act_gb = _act_margin_gb = 0.0
        if _n_clip_items:
            _clip_lt, _clip_smp = _max_clip_act_item(group)
            _act_gb, _act_margin_gb = ft_clip_activation_gb(_clip_lt, _clip_smp)
            if _act_gb > 0:
                logger.info("[h3-ft] clip activations ~%.1f GB + %.1f GB fragmentation "
                            "margin (%d-latent-frame clips at %.2f MP) reserved before "
                            "window sizing", _act_gb, _act_margin_gb, _clip_lt, _clip_smp)
        try:
            from fizgig.utils.device import plannable_free_vram as _pfv0
            _usable = _pfv0() + 0.21 * _n_blocks - 1.5      # + resident trunk − reserve
        except Exception:
            _usable = 99.0
        _usable -= _act_gb + _act_margin_gb
        _windows, ft_stream, _plan_why = plan_h3_ft_windows(
            _usable, subset=ft_subset, n_blocks=_n_blocks,
            allow_stream=os.environ.get("FIZGIG_NO_FT_STREAM") != "1")
        for _line in _plan_why:
            logger.info("[h3-ft] %s", _line)
        if _windows is None:
            _clip_advice = (
                f" This dataset's clips are the driver (~{_act_gb + _act_margin_gb:.1f} GB "
                "of the budget is reserved for their activations, which is why the figure "
                "above reads lower than your card's free VRAM): cutting the clips to the "
                "56-frame / 2.3 s Gizmo slot, or lowering Target Megapixels, shrinks what "
                "the plan needs." if _act_gb > 0 else "")
            raise RuntimeError(
                f"[h3-ft] ~{_usable:.1f} GB of usable VRAM is below what the rotation "
                "fine-tune needs even with depth-split windows and streamed frozen blocks. "
                f"Close other GPU apps, or train a LoRA instead.{_clip_advice}")
        rotator = H3NF4Rotator(dit.blocks, master, key_prefix="blocks", device=device,
                               block_subset=ft_subset)
        _refiner = getattr(dit, "token_refiner", None)
        if _refiner is not None:
            # The always-on analogue of Krea's txtfusion: small, text-side, unquantized in
            # the int8 checkpoint (the NF4 rotator swaps its Params4bit Linears in from the
            # master; small dense Linears just unfreeze).
            rotator.activate_always("token_refiner", _refiner)
        _cycle_n = len(ft_subset) if ft_subset else _n_blocks
        rot_schedule = RotationSchedule(_cycle_n, active=ft_rotation,
                                        rotate_every=max(1, int(finetune_rotate_every or 1)),
                                        mode="component",
                                        components=_windows,
                                        start_window=int(finetune_start_window or 0))
        # Continuation across a different card/plan: the stamped window index only lines
        # up when the window COUNT matches. A mismatch is survivable (the cycle re-walks
        # from an approximate position) but must not be silent.
        if ft_epoch_offset:
            try:
                from safetensors import safe_open as _so_w
                with _so_w(dit_path, framework="pt") as _fw:
                    _prev_nw = int((_fw.metadata() or {}).get("fizgig_ft_n_windows", 0))
            except Exception:
                _prev_nw = 0
            if _prev_nw and _prev_nw != rot_schedule.n_windows:
                logger.warning("[h3-ft] this card's plan has %d windows; the checkpoint "
                               "was trained with %d — the rotation cycle cannot line up "
                               "exactly across the change, so expect one cycle of mild "
                               "imbalance while it settles.",
                               rot_schedule.n_windows, _prev_nw)

        def _ft_want(_epoch):
            """The rotation spec for an epoch: component window entries, verbatim."""
            return list(rot_schedule.active_at(_epoch))

        def _ft_window_gb(spec):
            """Projected bf16 size of a window spec (bare prefixes and depth-splits)."""
            _span_l = ft_subset if ft_subset else range(_n_blocks)
            total = 0.0
            for _e in spec:
                if isinstance(_e, str):
                    total += H3_COMPONENT_GB_PER_BLOCK.get(_e, 0.31) * len(list(_span_l))
                else:
                    _p, _lo, _hi = _e
                    _nb = sum(1 for b in _span_l if _lo <= b <= _hi)
                    total += H3_COMPONENT_GB_PER_BLOCK.get(_p, 0.31) * _nb
            return total

        _ft_span_set = set(ft_subset) if ft_subset else set(range(_n_blocks))

        def _ft_resident_blocks(spec):
            """Blocks holding trainable Linears under `spec` — the set that must stay
            GPU-resident when the frozen remainder streams."""
            res = set()
            for _e in spec:
                if isinstance(_e, str):
                    res |= _ft_span_set
                else:
                    _p, _lo, _hi = _e
                    res |= {b for b in _ft_span_set if _lo <= b <= _hi}
            return res

        def _ft_rebuild_ring(spec):
            """(Re)scope the NF4 H2D ring to the frozen out-of-window blocks.

            Called BEFORE each window activates (setup and every rotation): evicting the
            streamed set first is what makes room for the incoming bf16 window — the
            other order OOMs at setup on the 16 GB tier. Every resident block is pushed
            to the GPU through bind_block_packed_to (idempotent for blocks already
            there), so this is also the restore path after an override-encode park.

            On a rotation it also DEFRAGS (see below) — on a 16 GB budget the rebuild's
            allocations otherwise strand ~3.6 GB of unusable segments per boundary.

            ORDER IS LOAD-BEARING, both ways (field, 27 Aug, a 16 GB-sim OOM):
              * the outgoing residents are unbound to CPU by the new ring's prepare()
                BEFORE the incoming ones are bound to the GPU. The other order holds
                BOTH packed sets (~3 GB each at 15 blocks) on the card at once, and that
                transient is what tips a Windows allocator with no expandable_segments
                over the edge — it died with only 8.65 GiB live but 5.88 GiB stranded in
                fragments.
              * `old` is dropped BEFORE the new ring is built. Releasing the old ring
                frees its GPU slots but NOT its CPU staging dict, so a lingering local
                reference doubles the pinned staging (~7 GB -> ~14 GB here) straight
                through construction. enable_block_swap solved exactly this in
                model.py; this is the same hazard on the rotation path."""
            if not ft_stream:
                return
            from fizgig.minimax.h3_nf4_h2d_offload import (H3NF4H2DOffloader,
                                                           bind_block_packed_to)
            old = _ft_ring["ring"]
            if old is not None:
                # RING-AWARE DEFRAG. Ordering fixes alone did not save the 16 GB tier:
                # it still died at the epoch-2 boundary with only ~8.9 GiB LIVE (exactly
                # what the window plan predicts) but ~5.5 GiB stranded in segments the
                # allocator could not reuse. Windows has no expandable_segments, so the
                # rebuild's fresh allocations never fit the holes the old ones left and
                # `reserved` climbed ~3.6 GB per rotation until it hit the cap.
                #
                # The cure is the one the non-streaming path already uses, expressed
                # THROUGH the ring instead of around it: park everything first, so for
                # one moment no block tensor is live and empty_cache can hand whole
                # segments back, then let the rebuild land contiguously. park_dit_partial
                # is safe against a live ring by design — it calls unbind_to_cpu() before
                # its walk, so the streamed blocks' ring VIEWS are never resize_(0)'d
                # (the sticky 'invalid argument' that once killed step 0). It must
                # therefore run BEFORE the offloader refs are cleared, or that guard
                # cannot fire.
                park_dit_to_cpu(dit)
                old.release()
                _ft_ring["ring"] = None
                dit._h2d_offloader = None
                for _blk in dit.blocks:
                    _blk._h2d_offloader = None
                # Each block's weight still views its old CPU flat until it rebinds
                # below, so the staging peak stays ~one window's worth instead of two.
                old = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # The park took the non-block modules (token refiner, embeddings, final
                # layer) down with it; the blocks come back below, but these have no
                # other restore path. The refiner is always-on TRAINABLE, and .to() only
                # repoints .data — Parameter identity and requires_grad survive, and the
                # optimizer is rebuilt after activate() regardless.
                for _cname, _child in dit.named_children():
                    if _cname != "blocks":
                        _child.to(device)
            resident = _ft_resident_blocks(spec)
            streamed = [i for i in range(len(dit.blocks)) if i not in resident]
            ring = H3NF4H2DOffloader(dit.blocks, streamed, torch.device(device))
            ring.move_static_weights_to_gpu()
            # prepare() stages every streamed block into a CPU flat and rebinds it —
            # which is what releases the OUTGOING residents' GPU packed weights.
            ring.prepare()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # ...only now pull the incoming window's blocks onto the card.
            for i in sorted(resident):
                bind_block_packed_to(dit.blocks[i], device)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dit._h2d_offloader = ring
            for _blk in dit.blocks:
                _blk._h2d_offloader = ring
            dit._swap_from = 0      # every block consults the ring; residents no-op inside it
            _ft_ring["ring"] = ring
            logger.info("[h3-ft] streaming ring rescoped: %d resident block(s) [%s], "
                        "%d streamed (~%.1f GB staged in RAM)", len(resident),
                        format_block_spec(sorted(resident)), len(streamed), ring.staged_gb)

        if rot_schedule.cycle_epochs > max_train_epochs:
            logger.warning("[h3-ft] a full rotation cycle is %d epochs but the run is only "
                           "%d — some blocks will never train. Raise Max Epochs to at least "
                           "the cycle length.", rot_schedule.cycle_epochs, max_train_epochs)

        _first = _ft_want(0)
        _ft_rebuild_ring(_first)      # evict the streamed set BEFORE the window allocates
        rotator.activate(_first)
        logger.info("[h3-ft] cycle: %d component windows %s across %d block(s), "
                    "%d epoch(s) per cycle; first window -> %s",
                    rot_schedule.n_windows, list(rot_schedule.components),
                    _cycle_n, rot_schedule.cycle_epochs, _first)
        # Save cadence snaps to the cycle too — each save is a full ~21 GB checkpoint, and
        # per-cycle saves compare like-for-like (every block equally trained). A multiple
        # of the cycle is respected; anything else snaps UP to the next multiple — the
        # typed number expressed how SPARSE the user wants 20 GB saves (and, since
        # previews follow saves, previews), so rounding down to one-per-cycle would give
        # 2.5x the saves they asked for (field: 10 on a 4-cycle silently became 4).
        # 0 = final only.
        _cyc = rot_schedule.cycle_epochs
        # Max epochs snaps UP to end on a cycle boundary too — an off-cycle total leaves
        # the FINAL checkpoint (the one people keep) with some components trained one
        # more pass than others. Start-aware, so a resumed leg still lands on the
        # original total when that total was cycle-aligned.
        from fizgig.krea2.rotation import snap_ft_epochs as _snap_ep
        _snapped_ep = _snap_ep(max_train_epochs, _cyc,
                               start_window=int(finetune_start_window or 0),
                               rotate_every=max(1, int(finetune_rotate_every or 1)))
        if _snapped_ep != max_train_epochs:
            logger.info("[h3-ft] Max epochs %d would end mid-cycle (%d-epoch cycle) — "
                        "snapping to %d so the final checkpoint ends with every "
                        "component evenly trained.",
                        max_train_epochs, _cyc, _snapped_ep)
            max_train_epochs = _snapped_ep
        if save_every_n_epochs and save_every_n_epochs % _cyc != 0:
            _snapped_save = ((save_every_n_epochs + _cyc - 1) // _cyc) * _cyc
            logger.info("[h3-ft] Save every %d epoch(s) doesn't line up with the %d-epoch "
                        "cycle — snapping to %d, the next multiple (each save is a full "
                        "~21 GB checkpoint, and previews ride along with saves).",
                        save_every_n_epochs, _cyc, _snapped_save)
            save_every_n_epochs = _snapped_save
        # Gate on do_previews, not on prompts+TE. Previews ALSO need a cadence
        # (sample_every_n_epochs or sample_at_first), so a run with prompts and a TE but
        # neither flag renders nothing — and this line used to announce previews anyway,
        # for a whole run. Same class as the vision-rung streaming lie and the compile-"On"
        # lie: a log that describes intent rather than what was decided. Say which it is.
        if do_previews:
            logger.info("[h3-ft] previews follow CHECKPOINT SAVES (every %d epoch(s), plus "
                        "the final one) — each sample is the rehearsal of a checkpoint you "
                        "can deploy, overriding Sample-every-N. Rendered via a "
                        "deactivate/reactivate bracket with the Turbo applied fresh each "
                        "time.",
                        save_every_n_epochs if save_every_n_epochs else max_train_epochs)
        elif sample_prompts and te_path:
            logger.warning("[h3-ft] previews are OFF for this run: prompts and a text "
                           "encoder are set, but no sample cadence — set Sample every N "
                           "epochs (--sample_every_n_epochs), or sample-at-first, and "
                           "previews will ride the checkpoint saves.")

        # Category retirement lands at cycle boundaries only: every window must see the
        # identical data mix for equal passes before the mix changes, or late-cycle
        # windows train on different data than early ones. Snapped UP, never down —
        # the user asked for at least that much training. Cumulative across pause/resume
        # (snap_ft_stop): a stop at-or-below the continuation's start epoch means that
        # category is retired for this whole run — "pause, set the stop to the current
        # epoch, resume" IS the supported way to finish a mixed run audio- or visual-only.
        def _snap_stop(_n, _label):
            _v, _kind = snap_ft_stop(_n, _cyc, ft_epoch_offset, max_train_epochs)
            if _kind == "past":
                logger.info("[h3-ft] %s retirement at epoch %d is already behind this run "
                            "(continuing from epoch %d) — retired from the start.",
                            _label, _v, ft_epoch_offset)
            elif _kind == "snapped":
                logger.info("[h3-ft] %s retirement lands at rotation-cycle boundaries — "
                            "epoch %d snaps to %d (%d-epoch cycle).",
                            _label, int(_n or 0), _v, _cyc)
            elif _kind == "never":
                logger.warning("[h3-ft] %s retirement at epoch %d is at or past the run's "
                               "end (epoch %d) — it will never fire.",
                               _label, _v, max_train_epochs + ft_epoch_offset)
            return _v
        visual_stop_epoch = _snap_stop(visual_stop_epoch, "photos & clips")
        audio_stop_epoch = _snap_stop(audio_stop_epoch, "voice")
        if train_blocks:
            logger.info("[h3-ft] Blocks to Train (%s) is LoRA machinery and does nothing "
                        "under fine-tune — use the FT card's Blocks field "
                        "(--finetune_blocks) to restrict the rotation cycle instead.",
                        train_blocks)
            train_blocks = None
    # AdaLN targeting is per-checkpoint — see the pattern note at the top of this file.
    include_patterns = user_include_patterns or (
        PRUNED_INCLUDE_PATTERNS if dit.pruned_adaln else DEFAULT_INCLUDE_PATTERNS)
    # AdaLN is a pure function of the TIMESTEP — DiTBlock.forward calls adaln_proj(t_emb) and
    # nothing else, so its adapters cannot tell one subject from another. They can only reshape
    # how strongly each block fires at each noise level. On the pruned checkpoint they carry
    # ~45% of all weight movement in a matched epoch, which is a lot of a LoRA's capacity spent
    # somewhere structurally incapable of holding a face — hence the toggle. See
    # docs/MINIMAX_BLOCKS.md. No-op on the bf16 checkpoint, which never targets AdaLN.
    _adaln_on = bool(train_adaln) and dit.pruned_adaln
    if not train_adaln:
        _before = len(include_patterns)
        include_patterns = [p for p in include_patterns if "adaln" not in p]
        if len(include_patterns) < _before:
            logger.info("[base] EXPERIMENT: AdaLN adapters OFF. AdaLN sees only the timestep, so "
                        "it cannot encode identity — this frees the capacity it was taking. "
                        "Compare against the same run with it on.")
        else:
            logger.info("[base] AdaLN was not a target on this checkpoint; the toggle changes "
                        "nothing here.")
    _blocks_used = "all"
    if train_blocks:
        _n_blocks = len(dit.blocks)
        include_patterns = restrict_patterns_to_blocks(include_patterns, train_blocks, _n_blocks)
        _sel = parse_block_spec(train_blocks, _n_blocks)
        _blocks_used = format_block_spec(_sel)
        logger.info("[base] EXPERIMENT: training blocks %s only (%d of %d), text refiner "
                    "included. Nobody has mapped what H3's blocks do — judge this against a "
                    "full-model run on the same dataset, not on its own.",
                    _blocks_used, len(_sel), _n_blocks)
    # Report what is ACTUALLY targeted: this used to key off the checkpoint alone, so a run with
    # --no_train_adaln announced "+ AdaLN" one line after saying AdaLN adapters were off.
    _adaln_on = bool(dit.pruned_adaln and train_adaln)
    logger.info("[base] %s checkpoint; LoRA targets: attention + MLP + token refiner%s",
                "pruned (curve-table AdaLN)" if dit.pruned_adaln else "full bf16",
                " + AdaLN (deploy-consistent on this build; rank caps at 8)" if _adaln_on
                else (" (AdaLN excluded - turned off for this run)" if dit.pruned_adaln
                      else " (AdaLN excluded - dropped by pruned inference builds)"))
    if resume_state_dir:
        # A resume builds the CHECKPOINT'S network, whatever the boxes say now — settings
        # that moved on since the pause used to crash the relaunch on a size-mismatch wall.
        network_type, network_dim, network_alpha, lokr_factor, _rs_notes = \
            resume_network_shape(resume_state_dir, network_type, network_dim,
                                 network_alpha, lokr_factor)
        for _n in _rs_notes:
            logger.warning(f"[resume] {_n} — a resume continues the run it resumes")
    # Context LoRA goes on FIRST, so the trainable network wraps a forward that already
    # includes it (additive stack: base + context + trainable). Refused under rotation FT for
    # the same reason the Turbo is deferred there — apply_to captures bound forwards that
    # the rotator's class-swap would orphan — and unlike the Turbo it can't be re-applied
    # per window without changing what the model trains against mid-run.
    # Frozen layers, innermost first: training adapter, then the user's context LoRA. Both
    # ride at their strength for every training step. Previews differ PER LAYER (Peter):
    # the adapter switches OFF (it is a training aid — Ostris's assistant LoRA is inactive
    # for every sample pass, and the LoRA deploys without it), the context LoRA stays ON
    # (the LoRA deploys paired with it — Klein/Krea 2 semantics).
    adapter_net, adapter_adaln = None, []
    context_net, context_adaln = None, []
    if training_adapter_path or context_lora_path:
        if rotator is not None:
            _what = "The training adapter" if training_adapter_path else "Context LoRA"
            raise RuntimeError(f"{_what} is not available with fine-tuning on MiniMax H3 "
                               "— untick Fine-tune (train a LoRA) or turn it off.")
    if training_adapter_path:
        if not os.path.isfile(training_adapter_path):
            raise FileNotFoundError(f"Training adapter not found: {training_adapter_path} — "
                                    "run the updater or the Preferences model download.")
        adapter_net, adapter_adaln = load_context_lora(dit, training_adapter_path, 1.0,
                                                       device, dtype, tag="adapter",
                                                       label="Training adapter")
    if context_lora_path:
        if not os.path.isfile(context_lora_path):
            raise FileNotFoundError(f"Context LoRA not found: {context_lora_path}")
        context_net, context_adaln = load_context_lora(dit, context_lora_path,
                                                       context_lora_strength, device, dtype)

    def _frozen_for_preview():
        """Adapter off, context on; AdaLN rows = context (+ the Turbo's, added by the
        caller). Unpatch the training set FIRST — the rows share modules, and a patch
        replaces the module forward wholesale."""
        turbo_adaln_unpatch(adapter_adaln + context_adaln)
        if adapter_net is not None:
            for _m in adapter_net.unet_loras:
                _m.enabled = False
            logger.info("[preview] training adapter off for the render (deployment view)")

    def _frozen_for_training():
        """Back to the training stack: Turbo/preview rows out, adapter on, adapter +
        context rows in. Idempotent — safe on the exception path."""
        turbo_adaln_unpatch(context_adaln + turbo_adaln)
        if adapter_net is not None:
            for _m in adapter_net.unet_loras:
                _m.enabled = True
        if adapter_adaln or context_adaln:
            turbo_adaln_patch(dit, adapter_adaln + context_adaln, device, dtype)
    if rotator is not None:
        # Rotation FT: no adapter at all. LoRA's apply_to captures each wrapped module's
        # BOUND forward, which the rotator's class-swap would orphan — a wrapped window
        # would call the old ConvRot forward against freed codes. network stays None and
        # every downstream network consumer branches on the rotator.
        network = None
        _n_targeted = 0
        logger.info("[h3-ft] no LoRA network — the base model's own weights train")
    elif network_type == "lokr":
        # LoKR (Kronecker) — same mechanism as Krea 2's: module_class swaps the parametrization
        # inside the identical scan/wrap machinery, so include_patterns (adaln exclusion) and the
        # NF4/Linear4bit base compose unchanged. dim/alpha are ignored; factor is the dial.
        from fizgig.networks.lora import LoKRModule
        logger.info(f"network: LoKR (Kronecker), factor {lokr_factor}, full-matrix w2 — "
                    "dim/alpha do not apply")
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns,
                                 module_class=LoKRModule, module_kwargs={"factor": int(lokr_factor)})
    else:
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns)
    if network is not None:
        network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
        network.requires_grad_(True)
        network.to(device=device, dtype=dtype)
        network._network_type = network_type
        network._lokr_factor = int(lokr_factor)
    if network is not None:
        # Dotted module paths for the LyCORIS-standard save (diffusion_model.<path>.lokr_*) — built
        # from the DiT itself with the same flattening create_modules used, so the reverse mapping
        # is exact even where module names contain underscores. isinstance covers bnb Linear4bit
        # (an nn.Linear subclass).
        network._dotted_names = {
            f"lora_unet_{name.replace('.', '_')}": name
            for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
        }
        _n_targeted = len(network.unet_loras)
        if network_type == "lokr":
            logger.info(f"LoKR: {len(network.unet_loras)} modules wrapped (factor {lokr_factor})")
        else:
            logger.info(f"LoRA: {len(network.unet_loras)} modules wrapped (dim {network_dim}, alpha {network_alpha})")

        # How many Linears did the include_patterns actually TARGET? create_modules matches by
        # class NAME, so a quantized Linear stand-in that is not on that list is skipped in
        # silence — which once shipped a run training 58 of 258 modules with no error anywhere.
        import re as _re
        _targeted = [n for n, m in dit.named_modules()
                     if isinstance(m, torch.nn.Linear)
                     and any(_re.search(p, n) for p in include_patterns)]
        if len(network.unet_loras) < len(_targeted):
            _kinds = sorted({type(dit.get_submodule(n)).__name__ for n in _targeted})
            raise RuntimeError(
                f"only {len(network.unet_loras)} of {len(_targeted)} targeted Linears were wrapped — "
                f"the network builder matches by class name and one of {_kinds} is not on its list "
                f"(networks/lora.py, create_modules). Training now would silently learn a fraction "
                f"of the model.")
        _n_targeted = len(_targeted)
        logger.info(f"[network] {len(network.unet_loras)}/{_n_targeted} targeted Linears wrapped")

    # The preview Turbo LoRA — a nicety, never a run-killer: any failure logs and trains on.
    turbo_net = None
    turbo_adaln = []
    _ft_turbo_path = None
    if turbo_lora_path and rotator is not None:
        # The Turbo must NOT be applied at setup under FT: apply_to captures bound forwards
        # that the rotator's class-swap would orphan. It is instead applied FRESH at each
        # cycle-boundary preview (against the fully-deactivated, all-ConvRot model) and
        # cleanly un-applied before the next window activates.
        _ft_turbo_path = turbo_lora_path
        turbo_lora_path = None
        logger.info("[h3-ft] preview Turbo LoRA deferred — applied per preview, inside the "
                    "deactivate/reactivate bracket")
    if turbo_lora_path:
        if not os.path.isfile(turbo_lora_path):
            logger.warning(f"[turbo] file not found — previews render without it: "
                           f"{turbo_lora_path}")
        else:
            try:
                turbo_net, turbo_adaln = load_preview_turbo(dit, turbo_lora_path,
                                                            turbo_lora_strength)
                logger.info(f"[turbo] previews render with "
                            f"{os.path.basename(turbo_lora_path)} at "
                            f"{turbo_lora_strength:g} x {sample_steps} steps")
            except Exception as _te:
                logger.warning(f"[turbo] could not load ({type(_te).__name__}: {_te}) — "
                               f"previews render without it")
        if turbo_net is None and sample_steps < 20:
            # The launched command carried the TURBO pace — a few steps only make sense with
            # the Turbo applied, and rendering the fallback at 6 would produce mush and read
            # as a broken LoRA (Peter). Standard pass instead, said out loud.
            logger.warning(f"[turbo] previews fall back to the standard pass: {sample_steps} "
                           f"steps was the Turbo pace — using 20 instead")
            sample_steps = 20

    # Previews with sound: the audio VAE's DECODER half, loaded once on first use and parked
    # on CPU (~0.45 GB RAM) between previews. Any failure means silent samples, never a dead
    # run.
    _audio_dec_state = {"dec": None, "tried": False}
    # The video decoder gets the same load-once lifecycle: reloading its ~4.85 GB from disk
    # into fresh CPU tensors every preview and del-ing it after left the freed pages in the
    # Windows heap — ~6 GB of RSS growth PER PREVIEW on a 16 GB 4090 (RAM 25 -> 31.6 across
    # one epoch). Loaded once, parked on CPU between previews, moved to GPU only for decode.
    _video_dec_state = {"dec": None, "tried": False}

    def _get_audio_decoder():
        if _audio_dec_state["tried"]:
            return _audio_dec_state["dec"]
        _audio_dec_state["tried"] = True
        if not (audio_vae_path and os.path.isfile(audio_vae_path)):
            logger.warning("[preview] sound requested but the audio VAE path is not set — "
                           "samples render silent (Preferences → Audio VAE)")
            return None
        try:
            from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae_decoder
            _audio_dec_state["dec"] = load_minimax_h3_audio_vae_decoder(
                audio_vae_path, device="cpu")
            logger.info("[preview] audio decoder loaded — clip samples carry a .wav")
        except Exception as _ae:
            logger.warning(f"[preview] audio decoder failed to load "
                           f"({type(_ae).__name__}: {_ae}) — samples render silent")
        return _audio_dec_state["dec"]

    params = rotator.trainable_params() if rotator is not None \
        else list(network.get_trainable_params())

    # Per-modality confinement under FT (the routing decided in the rotator block): the
    # active window's Parameters that each modality must NOT touch, rebuilt per window.
    # Applied as a per-batch requires_grad freeze around the forward/backward — the fused
    # per-tensor hooks never fire for a no-grad param, and the non-fused path simply
    # accumulates nothing. The always-on refiner is never in these lists (its Linears are
    # not block-indexed): it trains on every modality, matching the validated 34-49 run.
    _ft_freeze = {"photo": [], "voice": [], "clip": []}

    def _ft_rebuild_freeze():
        _ft_freeze["photo"] = []
        _ft_freeze["voice"] = []
        _ft_freeze["clip"] = []
        if rotator is None or not (_ft_route["photo"] or _ft_route["voice"]
                                   or _ft_route["clip"]):
            return
        for _k, _lin in rotator._targets(list(rotator.active)):
            _bi = int(_k.split(".")[1])
            for _cat, _allowed in _ft_route.items():
                if _allowed is not None and _bi not in _allowed:
                    _ft_freeze[_cat].append(_lin.weight)
        for _cat, _label in (("photo", "photo"), ("voice", "audio"), ("clip", "video")):
            if _ft_freeze[_cat]:
                logger.info("[h3-ft] this window: %d of %d tensors frozen on %s batches",
                            len(_ft_freeze[_cat]), len(params), _label)

    _ft_rebuild_freeze()

    # Adaptive LR ignores the Learning Rate box: it starts at the GEOMETRIC MIDPOINT of Min/Max
    # and the watcher owns the LR from there (matches Klein/Krea 2). Two knobs, not three.
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} (geometric midpoint) "
                    f"min={adaptive_lr_min:.3e} max={adaptive_lr_max:.3e}; the Learning Rate box is ignored")

    # Weight-decay parity with the reference trainer: ai-toolkit's job template passes
    # optimizer_params weight_decay=1e-4; bitsandbytes' default is 0.01 (100x). Only applied
    # when the user hasn't set their own via Optimizer Args.
    if "weight_decay" not in (optimizer_args or "") and "adam" in optimizer_type.lower():
        optimizer_args = (optimizer_args + " weight_decay=1e-4").strip()

    # Depth-dependent LR. A perturbation injected at block 5 passes through 45 more blocks that
    # absorb and renormalize it; one injected at block 45 lands almost directly on the output. So
    # the same |dW| is far more disruptive the later it sits, and ONE learning rate is wrong by
    # construction — it is either too low for the early blocks or too high for the late ones.
    # Observed here: at 1e-4, blocks 0-20 train cleanly but slowly while anything past 20 wrecks
    # the samples (block swap ruled out — those runs recorded blocks_swapped=0).
    # Built AFTER the adaptive block above, so `learning_rate` is already the resolved start LR.
    _slow_used, _slow_n = "", 0
    opt_params = params          # the optimizer may get groups; `params` stays flat for clipping
    if slow_blocks and abs(float(slow_block_lr_scale) - 1.0) > 1e-9:
        _slow_idx = set(parse_block_spec(slow_blocks, len(dit.blocks)))
        _slow_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:      # text-side, never part of the depth argument
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) in _slow_idx:
                _slow_ids.update(id(p) for p in _lora.parameters())
        if _slow_ids:
            _slow = [p for p in params if id(p) in _slow_ids]
            _fast = [p for p in params if id(p) not in _slow_ids]
            _scaled = learning_rate * float(slow_block_lr_scale)
            # lr_scale rides along on the group so the adaptive watcher can move both groups
            # together without flattening them back to one rate.
            # NOTE: assign to opt_params, NOT params. `params` stays the flat tensor list because
            # clip_grad_norm_ iterates it every step and cannot take param-group dicts.
            opt_params = [{"params": _fast, "lr": learning_rate, "lr_scale": 1.0},
                          {"params": _slow, "lr": _scaled, "lr_scale": float(slow_block_lr_scale)}]
            _slow_used = format_block_spec(sorted(_slow_idx))
            _slow_n = len(_slow)
            logger.info("[lr] depth-split: blocks %s train at %.3e (x%g), the rest at %.3e "
                        "(%d of %d tensors slowed)", _slow_used, _scaled, slow_block_lr_scale,
                        learning_rate, _slow_n, len(_slow) + len(_fast))
        else:
            logger.warning("[lr] slow_blocks %r matched no trained modules — is it outside "
                           "Blocks to Train? Depth-split LR is not active.", slow_blocks)

    # Optimised Likeness Learning: on photo-only optimizer windows, the params of blocks OUTSIDE
    # photo_blocks get grad=None before the clip — AdamW skips None-grad params entirely (no step,
    # no momentum, no weight decay), which is exactly "this step never touched them". Their LoRA
    # deltas stay ACTIVE in the forward; they just don't learn from photos. Refiners and non-block
    # modules always train (same rule as restrict_patterns_to_blocks — text-side, held constant).
    _photo_mask_params, _photo_used = [], ""
    # LoRA-only: under FT there is no network — the likeness intent was already resolved in
    # the rotator block (cycle-tighten or per-window gating).
    if photo_blocks and rotator is None:
        _pb_allowed = set(parse_block_spec(photo_blocks, len(dit.blocks)))
        _mask_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) not in _pb_allowed:
                _mask_ids.update(id(p) for p in _lora.parameters())
        _photo_mask_params = [p for p in params if id(p) in _mask_ids]
        _photo_used = format_block_spec(sorted(_pb_allowed))
        if _photo_mask_params:
            logger.info("[likeness] Optimised Likeness Learning ON — photo steps train blocks "
                        "%s (+refiners, %d of %d tensors frozen on photos); video clips train %s",
                        _photo_used, len(_photo_mask_params), len(params),
                        (f"blocks {clip_blocks} (restricted)" if clip_blocks else "the full model"))
        else:
            logger.info("[likeness] photo_blocks %s covers every trained block — nothing to "
                        "mask (Blocks to Train already inside it?)", _photo_used)
    # Clip routing, LoRA mode (Peter, 2 Sep — same behaviour as the FT tickbox): clip-only
    # windows update only clip_blocks. Same mechanism as the photo mask.
    _clip_mask_params = []
    if clip_blocks and rotator is None:
        _cb_allowed = set(parse_block_spec(clip_blocks, len(dit.blocks)))
        _cmask_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) not in _cb_allowed:
                _cmask_ids.update(id(p) for p in _lora.parameters())
        _clip_mask_params = [p for p in params if id(p) in _cmask_ids]
        if _clip_mask_params:
            logger.info("[likeness] video restriction ON — clip steps train blocks %s "
                        "(+refiners, %d of %d tensors frozen on clips)",
                        format_block_spec(sorted(_cb_allowed)),
                        len(_clip_mask_params), len(params))
    # Voice routing, LoRA mode: audio-only steps update only audio_blocks (the measured
    # voice zone — audio gradients outside it corrupt the visual blocks). Same mechanism
    # as the photo mask, keyed on voice-only optimizer windows.
    _audio_mask_params = []
    if audio_blocks and rotator is None:
        _ab_allowed = set(parse_block_spec(audio_blocks, len(dit.blocks)))
        _amask_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) not in _ab_allowed:
                _amask_ids.update(id(p) for p in _lora.parameters())
        _audio_mask_params = [p for p in params if id(p) in _amask_ids]
        if _audio_mask_params:
            logger.info("[likeness] voice routing ON — audio steps train blocks %s "
                        "(+refiners, %d of %d tensors frozen on voice)",
                        format_block_spec(sorted(_ab_allowed)),
                        len(_audio_mask_params), len(params))

    # FT ignores the Optimizer Type box (Krea parity): Adafactor's factored state is ~10x
    # smaller than Adam's, which is part of what keeps a window inside 32 GB. Optionally
    # fused into backward — one single-parameter optimizer per tensor, stepped from a
    # post-accumulate hook so a gradient is consumed the moment it lands and the whole
    # window's gradients never coexist.
    _fused = {"on": False, "opts": {}, "handles": []}

    def _make_ft_optimizer(_params):
        try:
            from transformers.optimization import Adafactor
            return (Adafactor(_params, lr=learning_rate, scale_parameter=False,
                              relative_step=False, warmup_init=False),
                    "adafactor (rotation)")
        except Exception:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(_params, lr=learning_rate), "adamw8bit (rotation)"
            except Exception:
                return torch.optim.AdamW(_params, lr=learning_rate), "adamw (rotation)"

    def _detach_fused():
        for h in _fused["handles"]:
            h.remove()
        _fused["handles"].clear()
        _fused["opts"].clear()
        _fused["on"] = False

    def _attach_fused(_params):
        for h in _fused["handles"]:
            h.remove()
        _fused["handles"].clear()
        _fused["opts"].clear()
        for p in _params:
            opt1, _ = _make_ft_optimizer([p])
            _fused["opts"][p] = opt1

            def _hook(param, _o=None):
                o = _fused["opts"].get(param)
                if o is not None:
                    o.step()
                    o.zero_grad(set_to_none=True)
            _fused["handles"].append(p.register_post_accumulate_grad_hook(_hook))
        _fused["on"] = True

    if rotator is not None:
        # opt_params is the non-FT path's alias of the FIRST window's param list — under FT
        # it would pin generation-1 params (3+ GB of bf16) for the whole run after the first
        # rotation deactivates them (field: peak 23.9 -> 26.8 GB at rotation one, and the
        # next ~0.6 GB of creep crossed the WDDM spill line at ~9 s/it).
        opt_params = None
        if finetune_fused_backward:
            _attach_fused(params)
            optimizer, optimizer_label = None, "adafactor (rotation, fused backward)"
            logger.info("[h3-ft] optimizer-in-backward: each gradient is consumed and freed "
                        "as it lands (grad clipping and accumulation are off)")
        else:
            optimizer, optimizer_label = _make_ft_optimizer(params)
        logger.info(f"optimizer: {optimizer_label} @ lr={learning_rate:.3e}")
    else:
        # eps_floor_8bit: H3-only. The 8-bit second moment underflows on this model's most
        # structured tensors and the update degrades to lr*m/eps — measured at ~100x the
        # configured LR, which presented as melted anatomy at epoch 1. The floor caps that. It
        # is passed here and nowhere else: Krea 2 has never shown the failure.
        optimizer, optimizer_label = create_optimizer(optimizer_type, opt_params, learning_rate,
                                                      optimizer_args, eps_floor_8bit=True)
        logger.info(f"optimizer: {optimizer_label} @ lr={learning_rate:.3e}")

    limiter = None
    if block_limit and float(block_limit) > 0:
        limiter = StepClipper(network, float(block_limit))
        logger.info(f"[clip] per-step movement cap ON at {float(block_limit):g}x the median "
                    f"block ({len(limiter.groups)} blocks watched) — whichever block overshoots "
                    f"in a single step gets pulled back to the pack, wherever the trained range "
                    f"ends. Only the offending STEP is shortened; nothing already learned is "
                    f"scaled down.")

    ramp = None
    if adapter_ramp and float(adapter_ramp) > 0:
        if adaptive is not None:
            logger.info("[ramp] ignored — Adaptive LR owns the schedule.")
        else:
            ramp = AdapterRamp(network, float(adapter_ramp))
            logger.info(f"[ramp] adapter-relative LR ON — each step held at "
                        f"{100 * float(adapter_ramp):.3f}% of the adapter's current size, so the "
                        f"LR starts low and climbs toward your configured ceiling as the adapter "
                        f"grows. A step is a huge perturbation of a new adapter and a small one "
                        f"of a mature adapter; this keeps the RATIO steady instead of the rate.")

    ema = None
    if ema_decay and float(ema_decay) > 0:
        ema = EMAWeights(network, float(ema_decay))
        logger.info(f"[ema] ON at decay {float(ema_decay):g} — checkpoints and previews use the "
                    f"smoothed average of the training path; training itself runs on the raw "
                    f"weights. Big-LR strides zigzag; the EMA is the center of the zigzag.")

    # LR warmup: the epoch-1 damage from a high static LR comes from oversized strides landing
    # on zero-init adapters at the steepest part of the loss surface. A linear ramp over the
    # first N epochs eases in, then runs at full speed — the cost is a fraction of one epoch's
    # worth of movement. Adaptive LR owns its own schedule, so warmup only applies without it.
    if adaptive is not None and lr_warmup_epochs and lr_warmup_epochs > 0:
        logger.info("[warmup] ignored — Adaptive LR owns the schedule (it already starts at "
                    "the midpoint and probes from there).")
        lr_warmup_epochs = 0.0

    # Caption dropout (reference default 0.05): swap in the cached empty-prompt embed for a
    # random ~5% of steps. The uncond file is written by minimax_cache_text next to the caches.
    uncond_text = None
    if caption_dropout and caption_dropout > 0:
        for _ds in group.datasets:
            _f = os.path.join(getattr(_ds, "cache_directory", "") or "",
                              f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors")
            if os.path.isfile(_f):
                from safetensors.torch import load_file as _lf
                uncond_text = _lf(_f)["hidden_states"].unsqueeze(0)      # (1, L, 5120)
                break
        if uncond_text is None:
            logger.warning("[caption_dropout] no uncond embed in the cache dirs (re-run text "
                           "caching to enable it) — dropout disabled for this run")
        else:
            logger.info(f"[caption_dropout] {caption_dropout:.2f} — empty-prompt embed loaded")

    # Reference distillation needs nothing at run start: each item's reference conditioning AND
    # that reference's latent both ride in from the cache, one slot picked at random per step.
    if distill:
        logger.info("[distill] reference distillation ON — teacher weight %.2f, photo %.2f. "
                    "References come from the dataset itself (each image paired with others by "
                    "the caching pass); no image is ever its own reference.",
                    distill_weight, 1.0 - distill_weight)

    # Regularisation images (`is_reg = true` dataset blocks) — a PRIOR ANCHOR, not a subject
    # (same doctrine as Krea 2 FT): a full fine-tune moves the base weights with no low-rank
    # bound, so a handful of subjects drifts the model's whole notion of people. Reg stills
    # pull back, and only work as a fixed reduced-LR nudge. Their lifecycle rides the visual
    # category for free: they are stills, so likeness routing confines them to the photo
    # window and 'Finish photos & clips early' stops them with the photos (by design —
    # once subject-visual pressure stops, the counter-pressure must stop too). LoRA runs
    # ignore them with a warning: the update is rank-bounded, the problem doesn't exist.
    reg_keys = set()
    reg_mult = float(reg_lr_multiplier)
    if rotator is not None:
        reg_keys, _subj_keys = _reg_subject_keys(group.datasets)
        # item_keys are bare image stems — a subject and a reg image sharing a filename
        # would silently throttle the SUBJECT to reg LR for the whole run. Never punish
        # the subject: colliding stems leave the reg set (that reg image then trains
        # unthrottled, which the warning says out loud).
        _clash = reg_keys & _subj_keys
        if _clash:
            reg_keys -= _clash
            logger.warning(
                f"[reg] {len(_clash)} filename(s) appear in BOTH the subject and the "
                f"regularisation folders ({', '.join(sorted(_clash)[:5])}"
                f"{'...' if len(_clash) > 5 else ''}) — those items train as ordinary "
                "subjects at full LR. Rename the regularisation copies to re-anchor them.")
        if reg_keys:
            logger.info(f"[reg] {len(reg_keys)} regularisation image(s) at x{reg_mult:g} LR "
                        f"({group.num_train_items - len(reg_keys)} subject items). They follow "
                        "the photo routing and the visual category's stop epoch.")
            if len(reg_keys) >= group.num_train_items - len(reg_keys):
                logger.warning("[reg] regularisation images are at least half the training "
                               "set — keep them the minority or the multiplier stops reading "
                               "as a nudge.")
            if not finetune_fused_backward and max_grad_norm and max_grad_norm > 0:
                # The boundary clip renormalises the whole gradient to max_grad_norm —
                # a x0.2 reg gradient and a full subject gradient both land at the SAME
                # norm, so the multiplier is erased. Fused FT is immune (grad clipping
                # is structurally off there).
                logger.warning(
                    "[reg] gradient clipping is ON (non-fused fine-tune, max_grad_norm="
                    f"{max_grad_norm:g}) — it renormalises every step to the same norm, "
                    "which CANCELS the regularisation multiplier. Re-tick 'Free each "
                    "gradient as it lands' or pass --max_grad_norm 0 for the anchor to "
                    "work.")
    elif any(getattr(_ds, "is_reg", False) for _ds in group.datasets):
        logger.warning("[reg] the dataset config has a regularisation block, but this is a "
                       "LoRA run — regularisation images are a fine-tune feature and are "
                       "IGNORED here. They will train as ordinary images at full LR; remove "
                       "the `is_reg` block from the TOML if that is not what you want.")

    collator = _Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    # --- identity-first (two-phase distillation) --------------------------------------------
    # Phase 1 trains ONLY against the teacher, so the adapter learns who the trigger means
    # before it is asked to reproduce any particular photograph. Phase 2 then drops the teacher
    # entirely and trains on the photos alone, starting from an adapter that already has the
    # identity in the right places. A hard switch, not a decay: this is an INITIALISATION
    # strategy, so what phase 2 forgets about the teacher does not matter.
    #
    # AUTO length comes from a real run (11 Aug, 82 images): the teacher error fell 0.069 ->
    # 0.051 -> 0.050 over epochs 7-9 — converged by epoch 8, i.e. ~650 gradient STEPS. Steps,
    # not epochs, is the invariant: a 24-image set needs the same number of steps, which is
    # many more epochs. Held at one epoch minimum.
    _p1_epochs = 0
    if distill:
        _p1_epochs = (max(1, math.ceil(650 / max(1, steps_per_epoch)))
                      if distill_phase1_epochs is None or distill_phase1_epochs < 0
                      else int(distill_phase1_epochs))
        _p1_epochs = min(_p1_epochs, max_train_epochs)
        if _p1_epochs > 0:
            logger.info(
                f"[distill] IDENTITY-FIRST: epochs 1-{_p1_epochs} train against the teacher "
                f"ONLY (~{_p1_epochs * steps_per_epoch} steps) at "
                f"{learning_rate * _P1_LR_SCALE:.2e} — a third of the box — then the teacher is "
                f"dropped and epochs {_p1_epochs + 1}-{max_train_epochs} train on the "
                f"photographs alone at the full {learning_rate:.2e}. "
                f"The teacher weight box does not apply in this mode."
                + ("" if distill_phase1_epochs is not None and distill_phase1_epochs >= 0 else
                   "  (length chosen from the dataset size — the teacher objective converges in "
                   "roughly 650 steps whatever the image count.)"))

    # Gradient accumulation. Batch size 1 means every step is aimed by ONE image, so a large
    # stride follows an equally large random walk — the roughness that reads as "quality loss
    # without distortion" when a run covers ground fast. Averaging the gradient over N images
    # before stepping makes a big step PRECISE instead of rough, at the same wall-clock per
    # epoch (same forwards, N times fewer optimizer steps).
    _accum_n = max(1, int(gradient_accumulation_steps or 1))
    if _accum_n > 1:
        logger.info(f"[accum] gradient accumulation {_accum_n} — effective batch {_accum_n}, "
                    f"{steps_per_epoch // _accum_n} optimizer steps per epoch instead of "
                    f"{steps_per_epoch}. Each step is aimed by {_accum_n} images, so the same "
                    f"stride carries far less sampling noise.")

    warmup_steps = int(round(float(lr_warmup_epochs or 0.0) * steps_per_epoch))
    if warmup_steps > 0:
        logger.info(f"[warmup] LR ramps linearly over the first {lr_warmup_epochs:g} epoch(s) "
                    f"= {warmup_steps} steps, then holds at the configured LR.")

    os.makedirs(output_dir, exist_ok=True)
    pause_flag = os.path.join(output_dir, ".pause_requested")

    # ---- resume: restore network + optimizer + RNG + (epoch, step) + adaptive scalars ----
    from fizgig.training.train_utils import prune_state_dirs
    global_step = 0
    start_epoch = 0
    # `if resume_state_dir` — NOT `and os.path.isdir(...)`: a requested resume whose path is bad
    # used to skip this block silently and train from scratch. Resume, or refuse — never ignore.
    if resume_state_dir:
        start_epoch, global_step, _resume_meta = _load_training_state(
            resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
        if ema is not None:
            _ema_path = os.path.join(resume_state_dir, "ema.pt")
            if os.path.isfile(_ema_path):
                ema.load_state_dict(torch.load(_ema_path, map_location="cpu"))
                logger.info(f"[ema] restored the running average ({ema.n} updates)")
            else:
                # The state predates EMA (or it was off then). Restart the average from the
                # RESTORED weights — the shadow currently holds the zero init from construction.
                ema.shadow = [p.detach().clone().float() for p in ema.params]
                logger.info("[ema] no EMA state in the resume dir — restarting the average "
                            "from the restored weights.")
        _rs = _resume_meta.get("adapter_ramp")
        if ramp is not None and _rs:
            ramp.mult = float(_rs.get("mult", ramp.mult))
            ramp._smooth = (float(_rs["smooth"]) if _rs.get("smooth") is not None else None)
            ramp._prev = (float(_rs["prev"]) if _rs.get("prev") is not None else None)
            logger.info(f"[ramp] restored — LR resumes at {100 * ramp.mult:.0f}% of the "
                        f"configured ceiling rather than re-climbing from the floor")
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch "
                    f"{start_epoch + 1}/{max_train_epochs} (global_step {global_step})")
        if start_epoch >= max_train_epochs:
            # Pausing ON the last epoch exits before the final LoRA is written — Resume is what
            # completes it, so this fall-through writes the final file from the restored state.
            logger.warning(f"[resume] state is at epoch {start_epoch} of {max_train_epochs} — "
                           f"nothing left to train. Writing the final LoRA from the restored "
                           f"state. To train further, raise Max Train Epochs and resume again.")

    # Steps drawn above sigma 0.5 — the noisy half, where pose and composition are decided — can
    # be damped relative to the clean-end steps that carry identity. It has to scale the
    # OPTIMIZER'S LR and not the loss: Adam's update is m / (sqrt(v) + eps), which is invariant to
    # a constant factor on the gradient, so a loss multiplier here would do essentially nothing
    # while reading as though it worked.
    _hn_scale = float(highnoise_lr_scale)
    _hn_active = abs(_hn_scale - 1.0) > 1e-9
    _band_acc = []                       # this window's multipliers; averaged at the step
    if _hn_active:
        logger.info(f"[lr] steps above sigma {MINIMAX_LOWNOISE_SIGMA:g} train at "
                    f"{_hn_scale * 100:.0f}% of the learning rate.")

    # Per-category retirement (mixed visual+voice datasets). The anchor multiplies
    # param_group["lr"] through the same composed product as everything else — NEVER the loss:
    # Adam's update is m/(sqrt(v)+eps), invariant to a constant on the gradient, so a loss
    # multiplier reads as a working throttle and changes almost nothing (measured in
    # tests/test_minimax_highnoise_lr.py; realized travel is the only honest check).
    _vis_stop = max(0, int(visual_stop_epoch or 0))
    _aud_stop = max(0, int(audio_stop_epoch or 0))
    _cat_acc = []                        # this window's per-category multipliers
    # Under FT retirement is pure batch-skipping (mode is coerced to "stop" and the epochs
    # snap to cycle boundaries) — it must NOT arm the LR-composition machinery below, which
    # rewrites optimizer param groups FT doesn't have (fused: no optimizer object at all).
    _retire_active = bool(_vis_stop or _aud_stop) and rotator is None
    if _vis_stop:
        logger.info(f"[retire] photos & clips after epoch {_vis_stop}: "
                    + (f"anchored at {ANCHOR_LR_SCALE * 100:.0f}% LR (drift guard, ledger "
                       f"stays live)" if visual_stop_mode == "anchor" else
                       "stopped entirely (faster epochs)"))
    if _aud_stop:
        logger.info(f"[retire] voice recordings after epoch {_aud_stop}: "
                    + (f"anchored at {ANCHOR_LR_SCALE * 100:.0f}% LR (drift guard, ledger "
                       f"stays live)" if audio_stop_mode == "anchor" else
                       "stopped entirely (faster epochs)"))

    if warmup_steps > 0 or ramp is not None or _p1_epochs or _hn_active or _retire_active:
        # Stashed AFTER the resume block: optimizer.load_state_dict replaces the param-group
        # dicts, so a stash made earlier would not survive a resume. Derived from the CONFIGURED
        # rate (x the group's depth-split scale), not the group's current lr, which a resumed
        # mid-ramp state would have left partway up.
        for _g in optimizer.param_groups:
            _g["_warmup_base_lr"] = learning_rate * float(_g.get("lr_scale", 1.0))
    # WHOEVER OWNS THE LR SETS IT — and when nobody does, the configured value must win.
    # NOT an elif on the block above: warmup CONFIGURED but already FINISHED lands here too,
    # and that was exactly the case the first version of this fix missed.
    if should_reassert_lr(resuming=bool(resume_state_dir), adaptive=adaptive, ramp=ramp,
                          warmup_steps=warmup_steps, global_step=global_step):
        _stale = float(optimizer.param_groups[0].get("lr", learning_rate))
        for _g in optimizer.param_groups:
            _g["lr"] = learning_rate * float(_g.get("lr_scale", 1.0))
        if abs(_stale - learning_rate) > 1e-12:
            logger.info("[resume] the saved state carried lr=%.3e (a throttled value from when "
                        "it was written); nothing is modulating the LR this run, so the "
                        "configured %.3e is reasserted.", _stale, learning_rate)

    def _run_provenance():
        """What actually produced this LoRA — the facts you need to compare two of them.

        Added after an A/B where the file could not answer "was this the int8 base or NF4?",
        "how many modules were really wrapped?" or "how many steps?" — all of which changed the
        interpretation completely, and one of which (58 of 258 modules) had been a silent bug.
        A LoRA that cannot describe its own run is a measurement you have to take on trust."""
        try:
            # key[-2:], not an unpack: the audio sentinel key is ("audio", w, h), and a bare
            # (w, h) unpack would throw and silently blank EVERY resolution out of the metadata.
            _res = sorted({("voice" if isinstance(key[0], str) else f"{key[-2]}x{key[-1]}")
                           for ds in group.datasets for key in ds.batch_manager.bucket_resos})
        except Exception:
            _res = []
        try:
            from fizgig.minimax.audio import is_audio as _is_af
            _n_voice = sum(1 for ds in group.datasets
                           for p in getattr(getattr(ds, "datasource", None), "image_paths", []) or []
                           if _is_af(p))
        except Exception:
            _n_voice = 0
        _dens = ("shift12" if shift is None else
                 shift if isinstance(shift, str) else f"shift{shift:g}")
        return {
            "ss_base_checkpoint": os.path.basename(dit_path),
            "ss_base_quant": _base_mode,
            "ss_lora_modules": (str(len(network.unet_loras)) if network is not None
                                else "0 (rotation fine-tune)"),
            "ss_targeted_modules": str(_n_targeted),
            "ss_steps": str(global_step),
            "ss_epochs": str(max_train_epochs),
            "ss_learning_rate": f"{learning_rate:g}",
            "ss_optimizer": optimizer_label,
            "ss_timestep_density": _dens,
            "ss_highnoise_lr_scale": f"{float(highnoise_lr_scale):g}",
            "ss_train_blocks": _blocks_used,
            "ss_train_adaln": "1" if _adaln_on else "0",
            "ss_distill": "dataset" if distill else "off",
            "ss_distill_weight": (f"{distill_weight:g}" if distill else "0"),
            # Context LoRA: the file this LoRA was trained ON TOP OF and the strength it rode
            # at — pair them the same way at inference (Klein/Krea 2 record the same keys).
            "ss_context_lora": (os.path.basename(context_lora_path)
                                if context_lora_path else "none"),
            "ss_context_lora_strength": (f"{float(context_lora_strength):g}"
                                         if context_lora_path else "0"),
            "ss_training_adapter": (os.path.basename(training_adapter_path)
                                    if training_adapter_path else "none"),
            "ss_slow_blocks": _slow_used or "none",
            "ss_photo_blocks": (_photo_used if _photo_mask_params else "off"),
            "ss_clip_blocks": (str(clip_blocks) if (clip_blocks and (_clip_mask_params or rotator is not None))
                               else "off"),
            "ss_block_limit": str(block_limit or 0),
            "ss_gradient_accumulation": str(_accum_n),
            "ss_adapter_ramp": f"{adapter_ramp:g}" if ramp is not None else "0",
            "ss_lr_warmup_epochs": f"{lr_warmup_epochs:g}",
            "ss_ema_decay": f"{ema_decay:g}" if ema is not None else "0",
            "ss_slow_block_lr_scale": (f"{slow_block_lr_scale:g}" if _slow_used else "1"),
            "ss_caption_dropout": f"{caption_dropout:g}" if uncond_text is not None else "0",
            # One [[datasets]] block per subject is how Multi Concept keeps two people apart, so
            # a deployed LoRA should say how many it carries and where they came from — six
            # months later the trigger words are the only other clue.
            "ss_multi_concept": str(len(group.datasets)),
            "ss_concept_dirs": ",".join(
                os.path.basename(str(getattr(d, "image_directory", "") or "").rstrip("/\\"))
                for d in group.datasets),
            "ss_max_grad_norm": f"{max_grad_norm:g}",
            "ss_audio_only_items": str(_n_voice),
            "ss_visual_stop": (f"{_vis_stop}:{visual_stop_mode}" if _vis_stop else "off"),
            "ss_audio_stop": (f"{_aud_stop}:{audio_stop_mode}" if _aud_stop else "off"),
            "ss_bucket_resolutions": ",".join(_res),
            "ss_gradient_checkpointing": "1" if use_ckpt else "0",
            "ss_blocks_swapped": str(n_swap),
        }

    def _meta():
        md = build_metadata(
            None, ARCHITECTURE_MINIMAX, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author, description=metadata_description,
            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase)
        md.update(_run_provenance())
        return md

    def _state_extra():
        extra = {}
        if adaptive:
            extra["adaptive_lr_state"] = adaptive.state_dict()
        if ramp is not None:
            # Three JSON-safe scalars. Without them a resume restarts the climb at the floor
            # and spends ~78 steps re-earning a multiplier it had already established — the
            # same defect the retired governor shipped with, so it does not ship again.
            extra["adapter_ramp"] = {"mult": ramp.mult, "smooth": ramp._smooth,
                                     "prev": ramp._prev}
        return extra or None

    # Encoded override prompt, kept between epochs: re-encoding costs a TE load, so only redo it
    # when the prompt text actually changes.
    _ov_state = {"prompt": None, "enc": None}

    def _encode_override(prompt):
        """Encode one override prompt mid-run.

        The TE is ~14.5 GB and the int8 base ~21 GB, so unlike Krea 2 they cannot both be
        resident on a 32 GB card — the normal flow deliberately encodes every prompt BEFORE the
        DiT loads. To honour a live override we park the DiT on CPU for the duration, then
        restore it (and its block-swap split). That is a ~21 GB round trip, which is why the
        result is cached against the prompt text and only paid when you actually change it."""
        from fizgig.minimax.sampling import encode_sample_prompts
        from fizgig.utils.device import plannable_free_vram
        _free = plannable_free_vram() if torch.cuda.is_available() else 0.0
        _park = torch.cuda.is_available() and _free < 17.0     # TE + headroom
        _ring_was_live = _ft_ring["ring"] is not None
        if _park:
            logger.info(f"[sample override] parking the base on CPU to fit the text encoder "
                        f"({_free:.1f} GB free) — one-off for this prompt")
            if _ring_was_live:
                # The ring's GPU slots go too, and its CPU-bound flats must never ride a
                # dit.to(device) restore (that would materialize the whole base) — so the
                # ring is torn down here and rebuilt below through its one normal path.
                _ft_ring["ring"].release()
                _ft_ring["ring"] = None
                dit._h2d_offloader = None
                for _blk in dit.blocks:
                    _blk._h2d_offloader = None
            park_dit_to_cpu(dit)
            gc.collect()
            torch.cuda.empty_cache()
        try:
            return encode_sample_prompts(te_path, [prompt], device=device, quantize=quantize)
        finally:
            if _park and _ring_was_live:
                for _cname, _child in dit.named_children():
                    if _cname != "blocks":
                        _child.to(device)
                _ft_rebuild_ring(list(rotator.active))   # residents back up, ring re-armed
                gc.collect()
                torch.cuda.empty_cache()
            elif _park:
                restore_parked_dit(dit, device, n_swap)   # swap-aware: never the whole base
                gc.collect()
                torch.cuda.empty_cache()

    # Clip previews carry real failure risk a still never had (a 124-frame clip is ~30x the
    # sampling tokens plus a chunked multi-frame decode), and the epoch loop LATCHES previews
    # off on any preview exception. A clip-specific failure must degrade to a SHORTER clip that
    # fits, not take every future preview down with it — so the frame count lives in mutable
    # state the failure handlers can lower.
    _clip_state = {"frames": max(1, int(sample_frames or 1)), "notice_done": False,
                   "slow_done": False}
    # An OOM'd preview downgrades its resolution one ladder rung and retries; the cap is
    # STICKY so later epochs don't re-OOM their way down the same ladder every time.
    _res_cap = {"wh": None, "warned": False}

    def _slow_step_notice(seconds, step, total):
        """Told once per render when a preview step runs absurdly long.

        A preview that does not fit in VRAM does NOT raise on Windows — the driver pages to
        system RAM and the render succeeds at roughly a hundred times the cost. Wall time is
        the only symptom that survives, so it is what we watch — and when the ladder still
        has a rung below (a shorter clip first, then a resolution step), we don't just
        advise: returning True aborts the render and the preview loop retries one rung down,
        same as a hard OOM (Peter). Only at the floor of both axes does this fall back to
        advice.
        """
        _cur = _clip_state.get("cur_wh")
        _curf = int(_clip_state.get("cur_frames", 1))
        _has_rung = ((_curf > 1 and clip_fallback_frames(_curf) > 1)
                     or (_cur and next_preview_res(*_cur) != tuple(_cur)))
        if _has_rung:
            logger.warning(
                f"[preview] step {step}/{total} took {seconds:.0f}s — the render is paging "
                f"into system RAM (Windows never raises an OOM for this). Abandoning this "
                f"sample and retrying one rung down (a shorter clip first, then resolution).")
            return True
        if _clip_state["slow_done"]:
            return False
        _clip_state["slow_done"] = True
        logger.warning(
            f"[preview] step {step}/{total} took {seconds:.0f}s — the render is spilling "
            f"into system RAM even at the ladder floor (shortest clip, 512x512). It will "
            f"finish, just slowly. In order of impact: set the Turbo LoRA in Preferences "
            f"(6-step previews — over 3x fewer forwards than the standard 20), or switch "
            f"Sample length to a still. Previews are a heartbeat between checkpoints, not "
            f"the verdict: every epoch saves a .safetensors, and you can Pause the run, "
            f"judge an epoch in ComfyUI, then Resume.")

    def _render_previews(epoch):
        # Cumulative numbering, matching the checkpoints: a continuation run passes its
        # LOCAL epoch here, but the sample tag and the console must carry the run-total
        # epoch — the gallery's per-epoch tools (visualiser scrub, likeness trend) sort
        # by this tag, so a resumed leg re-tagging from e000000 collides with the first
        # leg's samples in the same folder (field, 29 Aug). Rebased once, here, because
        # everything below uses `epoch` for display and the filename only.
        epoch = epoch + ft_epoch_offset
        """Render one still per prompt on the RESIDENT training DiT and write them where the
        samples gallery looks. The filename format is the gallery/likeness/Visualiser contract
        (parse_sample_filename in the GUI) — do not change it casually.

        The DiT never moves: only eval mode is toggled, and block swap's JIT .to() is already
        forward-safe, so there is no swap-mode dance like Krea 2 needs."""
        import time as _time
        import numpy as _np
        from PIL import Image
        from fizgig.minimax import sampling
        was_training = dit.training
        decoder = None
        _base_parked = False        # set in phase 2; the finally MUST restore a parked base
        _opt_parked = []            # set in phase 1; ditto for the parked optimizer state
        _ema_parked = False         # set in phase 1; ditto for the parked fp32 EMA shadow
        try:
            dit.eval()
            if vae_path and _video_dec_state["dec"] is None and not _video_dec_state["tried"]:
                # Loaded ONCE per run (see _video_dec_state above) — it lives on CPU between
                # previews and only rides to the GPU for the decode phase.
                _video_dec_state["tried"] = True
                from safetensors import safe_open as _safe_open
                from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
                decoder = MiniMaxH3VideoVAEDecoder()
                with _safe_open(vae_path, framework="pt", device="cpu") as _f:
                    decoder.load_state_dict({k: _f.get_tensor(k) for k in _f.keys()}, strict=False)
                # FP16, not the training dtype and not fp32. ComfyUI allows this VAE exactly
                # [float16, float32] (sd.py:951) where its class default and every neighbouring
                # video VAE also list bfloat16 — bf16 was singled out and removed for this
                # decoder. The weights ship fp16 (minimax_h3_video_vae_fp16.safetensors), so
                # casting to bf16 threw away 3 mantissa bits at load, and 36 pre-norm residual
                # blocks feed a proj_out that emits 3072 pixel values per token: the error lands
                # straight on pixels as softness and gradient banding, with nothing downstream to
                # smooth it. fp16 costs the same 4.8 GB as bf16 (fp32 would be 9.7), so this is
                # free. Overflow is covered by the same nan_to_num guard ComfyUI relies on
                # (vae.py, attention output) — fp16 is the regime that guard was written for.
                # It also stays on CPU until the DECODE phase: previews used to put it on the GPU
                # before sampling even started, which cost the sampling forward 4.85 GB of
                # headroom it never used — harmless for a 256-token still, an OOM for a 124-frame
                # clip whose forward is ~30x the tokens (real 32 GB-card failure, 8 Aug).
                decoder = decoder.to(torch.float16).eval()
                _video_dec_state["dec"] = decoder
            elif vae_path:
                decoder = _video_dec_state["dec"]
            # Live override from the GUI, re-read every epoch so it can be turned on, changed or
            # switched off mid-run without touching the paused/resume path.
            _prompts, _w, _h = encoded_prompts, sample_width, sample_height
            _seed = sample_seed
            _ov = read_sample_override(output_dir)
            if _ov and not te_path:
                logger.warning("[sample override] a prompt is set but no --text_encoder is "
                               "configured, so it cannot be encoded — using the Samples tab.")
                _ov = None
            if _ov:
                if _ov["prompt"] != _ov_state["prompt"]:
                    # A failed encode must not take previews down with it. This loads the
                    # 14.5 GB TE mid-run (parking the 21 GB base to fit), and on a tight card
                    # that can OOM — and an exception from here used to propagate into the
                    # epoch loop's preview catch, which LATCHES previews off for the rest of
                    # the run. One bad encode silently ended every preview and read from the
                    # outside as "the override just stopped working". Fall back to the Samples
                    # tab prompts for this epoch instead; the state is left untouched, so the
                    # next boundary retries.
                    try:
                        _ov_state["enc"] = _encode_override(_ov["prompt"])
                        _ov_state["prompt"] = _ov["prompt"]
                    except Exception as _oe:
                        logger.warning(
                            f"[sample override] could not encode the new prompt "
                            f"({type(_oe).__name__}) — using the Samples tab prompts this "
                            f"epoch; will retry at the next preview.")
                        _ov = None
            if _ov:
                _prompts, _w, _h, _seed = _ov_state["enc"], _ov["width"], _ov["height"], _ov["seed"]
                # The override obeys the same 16 GB resolution cap as the Samples tab —
                # typing 1024x1024 into the box must not become a way around it.
                _cw, _ch = cap_preview_res_small_card(_w, _h)
                if (_cw, _ch) != (_w, _h):
                    logger.info(f"[sample override] {_w}x{_h} exceeds this card's preview cap "
                                f"— rendering {_cw}x{_ch}")
                    _w, _h = _cw, _ch
                logger.info(f"[sample override] active — '{_ov['prompt'][:60]}' "
                            f"seed={_seed} {_w}x{_h}")

            if _res_cap["wh"] is not None:
                _cw, _ch = _res_cap["wh"]
                if (_cw, _ch) != (_w, _h) and (_cw <= _w and _ch <= _h):
                    _w, _h = _cw, _ch
            _seed = _seed if _seed != 0 else random.randint(1, 2 ** 31 - 1)
            ts = _time.strftime("%Y%m%d%H%M%S")
            _frames = max(1, int(_clip_state["frames"]))
            if _frames > 1 and decoder is None:
                logger.warning("[preview] clip samples need the video VAE for decode — no VAE "
                               "path is configured, so this epoch renders stills instead.")
                _frames = 1
            if _frames > 1 and not _clip_state["notice_done"]:
                _clip_state["notice_done"] = True
                logger.info(f"[preview] clip mode: {_frames} frames per sample at {_w}x{_h} — "
                            f"clips take longer than stills, and longer clips take longer "
                            f"still. Cadence is 'Sample every N epochs' and size is "
                            f"Width/Height, both on the Samples tab.")
            # PHASE 1 — sample every prompt with the decoder still on CPU. The latents are a
            # few MB each, so parking them on CPU between phases costs nothing; the clip
            # forward gets the whole non-base headroom instead of sharing it with a decoder
            # it is not using yet.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # drop the training step's allocator slack
            # Clip forwards are big enough that VRAM pressure does not OOM on Windows - the
            # driver spills to system RAM and every step silently runs 3-6x slower (the
            # same failure signature as the checkpointing-margin bug). The fp32 Adam state
            # is ~2.5 GB of dead weight during a no-grad preview: park it on CPU for the
            # sampling phase. Costs about a second each way, once per preview epoch.
            if _frames > 1 and torch.cuda.is_available():
                # optimizer is None under FT's fused backward (per-tensor Adafactor whose
                # factored state is a rounding error) — nothing to park there.
                if optimizer is not None:
                    for _st in optimizer.state.values():
                        for _k, _v in list(_st.items()):
                            if torch.is_tensor(_v) and _v.is_cuda:
                                _st[_k] = _v.to('cpu')
                                _opt_parked.append((_st, _k))
                if ema is not None:
                    # The fp32 shadow (~1.25 GB at LoKR factor 8) is dead weight during a
                    # no-grad preview — the EMA weights are already swapped INTO the live
                    # network. Park it with the optimizer state; next ema.update() is a
                    # training step away, after the finally restores it.
                    ema.shadow = [s.to('cpu') for s in ema.shadow]
                    _ema_parked = True
                if _opt_parked or _ema_parked:
                    gc.collect()
                    torch.cuda.empty_cache()
                from fizgig.utils.device import plannable_free_vram
                _free0 = plannable_free_vram()
                logger.info(f'[preview] clip sampling with {_free0:.1f} GB free '
                            f'({len(_opt_parked)} optimizer tensors parked'
                            f'{", EMA shadow parked" if _ema_parked else ""})')
                # Leak tripwire, baseline-relative to the FIRST preview. Driver-free is the
                # wrong signal here: it also falls with allocator fragmentation (inactive-split
                # segments the decode-park absorbs — benign, self-limiting). A real leak is
                # LIVE allocation growth, so the census keys on memory_allocated().
                _alloc_now = torch.cuda.memory_allocated() / 1e9
                _base_free = _clip_state.get("free0")
                if _base_free is None:
                    _clip_state["free0"] = _free0
                    _clip_state["alloc0"] = _alloc_now
                elif _alloc_now > _clip_state.get("alloc0", _alloc_now) + 1.5:
                    try:
                        from fizgig.utils.device import report_cuda_leak, flush_reserved_vram
                        logger.info(f"[preview] live allocation grew "
                                    f"{_clip_state['alloc0']:.1f} -> {_alloc_now:.1f} GB "
                                    f"since the first preview — census:")
                        report_cuda_leak("preview-start", threshold_gb=0.0)
                        # The reserved-side census: names the small survivors pinning
                        # fragmented segments that empty_cache cannot return.
                        flush_reserved_vram("preview-start", threshold_gb=0.5)
                    except Exception:
                        pass
                vram_line("preview-start")
            # Training adapter OFF for the render, context LoRA ON (see the load block).
            _frozen_for_preview()
            if turbo_net is not None:
                # On for the sampling phase only: weights to the GPU (~0.8 GB), modules
                # enabled at their strength, AdaLN injected. Off + back to CPU before decode.
                turbo_net.to(device=device, dtype=dtype)
                for _tm in turbo_net.unet_loras:
                    _tm.enabled = True
                _n_ad = turbo_adaln_patch(dit, context_adaln + turbo_adaln, device, dtype)
                logger.info(f"[preview] Turbo LoRA on — {sample_steps} steps at "
                            f"{turbo_lora_strength:g}"
                            + (f", {_n_ad} adaln injected" if _n_ad else ""))
            elif context_adaln:
                turbo_adaln_patch(dit, context_adaln, device, dtype)
            _want_audio = bool(sample_audio and _frames > 1)
            _rendered = []
            for i, txt in enumerate(_prompts):
                print(f"[preview] epoch {epoch}: prompt {i + 1}/{len(_prompts)} "
                      f"({_w}x{_h}, {_frames} frame(s), seed {_seed + i})", flush=True)
                while True:
                    _clip_state["cur_wh"] = (_w, _h)       # the slow-step callback reads these
                    _clip_state["cur_frames"] = _frames
                    try:
                        lat, _arows = sampling.sample_image(
                            dit, txt.to(device, dtype),
                            width=_w, height=_h, steps=sample_steps,
                            cfg_scale=sample_cfg_scale,
                            uncond_embeds=(encoded_negative.to(device, dtype)
                                           if encoded_negative is not None else None),
                            seed=_seed + i, device=device, dtype=dtype, log_steps=True,
                            num_frames=_frames, on_slow_step=_slow_step_notice,
                            return_audio=True)
                        break
                    except (torch.cuda.OutOfMemoryError,
                            getattr(torch, "AcceleratorError", torch.cuda.OutOfMemoryError),
                            sampling.PreviewAborted):
                        # AcceleratorError: driver-level "CUDA error: out of memory" (seen on
                        # a 16 GB 4090 at the epoch-0 preview) arrives as this type, NOT as
                        # the allocator's OutOfMemoryError — without it here the ladder never
                        # ran and the preview was simply skipped.
                        # Downgrade one ladder rung and retry rather than losing previews for
                        # the run. Two triggers, one ladder: a hard CUDA OOM (Linux, or a
                        # too-big single allocation), and the slow-step abort (Windows paging
                        # never raises — the callback bails after the first crawling step
                        # instead). FRAMES give back first (Peter): a 56-frame clip retries at
                        # 22 frames before touching 768x768, because a shorter clip is still a
                        # clip while below 768 the model leaves its training regime. Only with
                        # frames at their clip floor does resolution start walking down
                        # (768x640 -> 640x640 -> ... -> 512x512). Both caps stick for later
                        # epochs. At the floor of BOTH there is nothing left to give back —
                        # re-raise into the trainer's usual preview handling (the abort never
                        # fires there by construction).
                        _nf = clip_fallback_frames(_frames) if _frames > 1 else 1
                        if _nf > 1:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.warning(f"[preview] OOM at {_frames} frames — retrying "
                                           f"this sample at {_nf} frames ({_w}x{_h} kept)")
                            _frames = _nf
                            _clip_state["frames"] = _nf
                            continue
                        _nw, _nh = next_preview_res(_w, _h)
                        if (_nw, _nh) == (_w, _h):
                            raise
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        _msg = f"[preview] OOM at {_w}x{_h} — retrying at {_nw}x{_nh}"
                        if min(_nw, _nh) < 768 and not _res_cap["warned"]:
                            _res_cap["warned"] = True
                            _msg += (". NOTE: below H3's 768 training canvas the model is "
                                     "outside its expected regime — treat these previews as "
                                     "a rough guide, and judge the LoRA at full size in "
                                     "ComfyUI.")
                        logger.warning(_msg)
                        _w, _h = _nw, _nh
                        _res_cap["wh"] = (_w, _h)
                        # A stable marker the GUI watches for: it writes this resolution back
                        # into the Samples tab, so the NEXT run starts where this one settled
                        # instead of re-walking the ladder. Printed per downgrade; the last
                        # one wins.
                        print(f"[preview] resolution settled: {_w}x{_h}", flush=True)
                _rendered.append((f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{_seed + i}",
                                  lat.to("cpu"),
                                  _arows.to("cpu") if (_want_audio and _arows is not None)
                                  else None))
                del lat, _arows

            if turbo_net is not None:
                for _tm in turbo_net.unet_loras:
                    _tm.enabled = False
                turbo_adaln_unpatch(turbo_adaln)
                turbo_net.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()      # its ~0.8 GB back before the decode phase
            _frozen_for_training()

            # optimizer state back before anything else - the next training step needs it
            for _st, _k in _opt_parked:
                _st[_k] = _st[_k].to(device)
            _opt_parked = []

            # PHASE 2 — decode. Clip decode wants ~6 GB (decoder weights + chunk transients);
            # if the card cannot offer that next to the resident base, park the base on CPU
            # for the duration, exactly as the override-encode path does. A ~21 GB round trip
            # costs seconds once per preview epoch; an OOM used to cost the previews entirely.
            if decoder is not None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    from fizgig.utils.device import plannable_free_vram as _pfv
                    _free = _pfv()
                    vram_line("pre-decode")
                    if _frames > 1 and _free < 7.5:
                        _need = (7.5 - _free) + 1.0
                        logger.info(f"[preview] {_free:.1f} GB free is too tight for clip "
                                    f"decode — parking ~{_need:.1f} GB of tail blocks for "
                                    f"this decode pass.")
                        park_dit_partial(dit, need_gb=_need)
                        gc.collect()
                        torch.cuda.empty_cache()
                        _base_parked = True
                        vram_line("post-park")
                decoder = decoder.to(device)
                vram_line("decoder-up")
            for stem, lat, _arows in _rendered:
                _px_mp4 = None                # full frames held only for a with-sound mp4
                lat = lat.to(device)
                if lat.shape[2] > 1 and decoder is not None:
                    # Clip: decode every frame, store EVERY 2ND frame as JPEG in a sibling
                    # .clip dir, and save the MIDDLE frame as the contract PNG — written LAST,
                    # so the gallery/likeness settle guard sees one finished unit. The PNG name
                    # is the gallery/likeness/Visualiser contract; the .clip dir is additive.
                    px = decoder.decode_clip(lat.float())[0]     # [3, F, H, W] in [0, 1]
                    n_f = px.shape[1]
                    clip_dir = os.path.join(sample_dir, stem + ".clip")
                    os.makedirs(clip_dir, exist_ok=True)
                    _keep = list(range(0, n_f, 2))
                    if _keep[-1] != n_f - 1:
                        _keep.append(n_f - 1)          # always include the final frame
                    for k in _keep:
                        fr = (px[:, k].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                        Image.fromarray(fr).save(os.path.join(clip_dir, f"f{k:03d}.jpg"),
                                                 quality=87)
                    mid = (px[:, n_f // 2].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(mid)
                    print(f"[preview] decoded {n_f}-frame clip at {_w}x{_h} "
                          f"({(n_f + 1) // 2} scrub frames)", flush=True)
                    if _arows is not None:
                        _px_mp4 = px.cpu()     # every frame, for the playable mp4 below
                    del px
                elif decoder is not None:
                    px = decoder.decode(lat.float())[0]          # [3, H, W] in [0, 1]
                    arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(arr)
                    print(f"[preview] decoded {_w}x{_h}", flush=True)
                else:
                    # No VAE path configured — fall back to the 24ch->RGB linear approximation
                    # (a 1/16-scale rough look) rather than dropping previews entirely.
                    arr = sampling.latent_to_rgb(lat)
                    img = Image.fromarray(arr).resize((_w, _h), Image.NEAREST)
                    print(f"[preview] decoded {_w}x{_h}", flush=True)
                # The wav is written BEFORE the contract PNG on purpose: the gallery's
                # settle guard treats the PNG as "this sample is finished", so everything
                # belonging to the sample must already be on disk when it lands.
                if _arows is not None:
                    _adec = _get_audio_decoder()
                    if _adec is not None:
                        try:
                            from fizgig.minimax.audio_vae import unpack_audio
                            _adec.to(device)
                            _wave = _adec.decode(
                                unpack_audio(_arows).to(device, torch.float32))
                            _wav_path = os.path.join(sample_dir, stem + ".wav")
                            write_wav(_wav_path, _wave[0].cpu())
                            print(f"[preview] wrote sound: {stem}.wav", flush=True)
                            if _px_mp4 is not None:
                                # a real playable clip — frames at true rate, sound muxed in;
                                # the gallery plays this instead of scrub + separate audio
                                try:
                                    write_preview_mp4(
                                        os.path.join(sample_dir, stem + ".mp4"),
                                        _px_mp4, _wav_path)
                                    print(f"[preview] wrote video: {stem}.mp4", flush=True)
                                except Exception as _me:
                                    logger.warning(f"[preview] mp4 mux skipped "
                                                   f"({type(_me).__name__}: {_me}) — the "
                                                   f"wav and scrub frames still work")
                        except Exception as _we:
                            logger.warning(f"[preview] audio decode failed "
                                           f"({type(_we).__name__}: {_we}) — this sample "
                                           f"renders silent")
                    _px_mp4 = None
                img.save(os.path.join(sample_dir, stem + ".png"))
                del lat
            if _audio_dec_state["dec"] is not None:
                _audio_dec_state["dec"].to("cpu")     # ~0.45 GB back off the card
            vram_line("post-decode")
            if _base_parked:
                restore_parked_dit(dit, device, n_swap)   # swap-aware: never the whole base
                gc.collect()
                torch.cuda.empty_cache()
                _base_parked = False
                vram_line("post-restore")
            logger.info(f"[preview] epoch {epoch}: wrote {len(_prompts)} sample(s) "
                        f"({sample_steps} steps, seed {_seed}) to {sample_dir}")
        finally:
            if decoder is not None:
                decoder.to("cpu")                        # park, don't free — reloading 4.85 GB
                if torch.cuda.is_available():            # per preview is what leaked the heap
                    torch.cuda.empty_cache()
            if _audio_dec_state["dec"] is not None:
                _audio_dec_state["dec"].to("cpu")        # idempotent; covers a mid-decode raise
            # The training stack must come BACK the same way (an exception mid-sample would
            # otherwise leave every subsequent training step without the adapter).
            _frozen_for_training()
            if turbo_net is not None:
                # Idempotent, and NON-NEGOTIABLE on an exception mid-sample: a Turbo left
                # enabled (or an injected AdaLN forward left installed) would ride every
                # subsequent TRAINING step.
                for _tm in turbo_net.unet_loras:
                    _tm.enabled = False
                turbo_adaln_unpatch(turbo_adaln)
                turbo_net.to("cpu")
            for _st, _k in _opt_parked:      # exception during phase 1: state must return
                _st[_k] = _st[_k].to(device)
            if _ema_parked:                  # next ema.update() needs the shadow on-device
                ema.shadow = [s.to(device) for s in ema.shadow]
            if _base_parked:
                # An exception mid-decode left the 21 GB base on CPU — the next training step
                # would die with "mat2 is on cpu". Restore residency (and the swap split)
                # before anything else runs.
                restore_parked_dit(dit, device, n_swap)
            if was_training:
                dit.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Fragmentation guard for tight cards (#92, David Maybank's 12 GB report): the
            # preview's block up/down churn + the 4.85 GB decoder round-trip fragment the
            # allocator reserve (his post-preview census: ~4.8 GB inactive split), and
            # Windows torch has no expandable_segments — the NEXT training step's first
            # contiguous allocation then OOMs even though everything was restored. Same
            # disease and same cure as the FT bracket previews: a full park/restore round
            # trip re-lands the resident set contiguously. Seconds, and only when free VRAM
            # is actually tight — a 32 GB card never triggers this.
            if torch.cuda.is_available():
                try:
                    from fizgig.utils.device import plannable_free_vram as _pfv2
                    _free_after = _pfv2()
                except Exception:
                    _free_after = 99.0
                if _free_after < 4.0 and _ft_ring["ring"] is None:
                    logger.info(f"[preview] {_free_after:.1f} GB free after the preview — "
                                "defragmenting via a full park/restore round-trip so the "
                                "next training step doesn't fight the fragments.")
                    park_dit_to_cpu(dit)
                    gc.collect()
                    torch.cuda.empty_cache()
                    restore_parked_dit(dit, device, n_swap)
                    gc.collect()
                    torch.cuda.empty_cache()
                    vram_line("post-defrag")
                elif _free_after < 4.0:
                    logger.info(f"[preview] {_free_after:.1f} GB free after the preview — "
                                "the park/restore defrag is unavailable while the FT "
                                "streaming ring is live; the next rotation's ring rebuild "
                                "re-lands the resident set instead.")
            vram_line("finally-done")

    def _ft_rebind_optimizer():
        """Rebuild the optimizer wiring for the CURRENT window params. Must run after ANY
        re-activation — rotation or preview bracket — because activation creates fresh
        Parameter objects: per-tensor fused optimizers keyed on the old objects would keep
        3+ GB of zombie weights alive while the new params accumulate un-stepped, un-freed
        gradients (field bug: 29.5 GB peak on a 24 GB window, and the window silently not
        training after an epoch-0 preview)."""
        nonlocal params, optimizer
        params = rotator.trainable_params()
        if finetune_fused_backward:
            _attach_fused(params)
        else:
            optimizer, _ = _make_ft_optimizer(params)
        # Fresh Parameter objects -> the modality freeze lists must be rebuilt too, or
        # they'd freeze the deactivated window's orphans and leave the live one open.
        _ft_rebuild_freeze()

    def _ft_render_previews(n):
        """Cycle-boundary preview under rotation FT: deactivate the whole window so the model
        is a consistent all-ConvRot checkpoint (the master holds every trained weight —
        reactivation-exactness is test-pinned), apply the Turbo FRESH against it, run the
        bog-standard preview, then un-apply the Turbo completely and reactivate. The un-apply
        pops each wrapped module's instance `forward` — apply_to deleted the module ref but
        the bound org_forward's __self__ IS the module, and the pre-Turbo forward here is
        always the class-level one, so popping restores it exactly."""
        nonlocal turbo_net, turbo_adaln, params, optimizer
        import gc as _gc
        _act = list(rotator.active)
        if _act:
            rotator.deactivate(_act)
            # Drop every optimizer reference to the window's now-orphaned bf16 Parameters —
            # the fused opts dict is KEYED on them, so without this they stay VRAM-resident
            # (~6 GB at 8 blocks) through the whole preview. Measured: the bracket saw
            # 3.7-4.2 GB free on an otherwise-clean card with them pinned. The next epoch's
            # rotation activates + rebinds through the one normal path, exactly as it does
            # for the deferred re-activation below.
            params = None
            if _fused["on"]:
                _detach_fused()
            else:
                optimizer = None
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            # Windows has no expandable_segments, so the activate/deactivate churn (worst at
            # 8-block windows) fragments the allocator — a bracket preview measured 3.9 GB
            # free and spilled to 50 s/step. A PARTIAL park can't help: sampling runs the
            # DiT, so parked blocks crash the render (field, RuntimeError at 56 and 22
            # frames). The real cure is a full defrag round-trip: park the ENTIRE base to
            # the CPU arena, hand every emptied segment back to the driver, then restore —
            # the re-upload lands contiguously. Seconds per preview, both halves are the
            # long-proven park machinery.
            try:
                from fizgig.utils.device import plannable_free_vram
                _free_now = plannable_free_vram()
            except Exception:
                _free_now = 99.0
            if _free_now < 8.0 and _ft_ring["ring"] is not None:
                # Never park/restore over a live ring: the restore's dit.to(device) would
                # push the streamed CPU flats up and materialize the whole base. The ring
                # itself keeps residency low; the render proceeds with what's free.
                logger.info("[h3-ft] %.1f GB free before the bracket preview — the "
                            "park/restore defrag is unavailable while the streaming ring "
                            "is live; rendering with what's free.", _free_now)
            elif _free_now < 8.0:
                logger.info("[h3-ft] %.1f GB free before the bracket preview — defragmenting "
                            "via a full park/restore round-trip.", _free_now)
                import gc as _gc3
                park_dit_to_cpu(dit)
                _gc3.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                restore_parked_dit(dit, device, 0)
                _gc3.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    _free_now = plannable_free_vram()
                    logger.info("[h3-ft] post-defrag: %.1f GB free", _free_now)
                except Exception:
                    pass
            # Still tight after the defrag = the shortage is TOTAL VRAM (usually other apps
            # holding the card), not fragmentation. Spilling turns a 5 s preview step into
            # 60 s — degrade THIS render to a shorter clip instead, and say so. Temporary:
            # unlike the crash ladder, the next preview tries the full length again.
            # Threshold from measured runs: 5.7 GB free rendered full-length fast; 4.1 GB
            # spilled to 61 s/step. 5.0 splits them.
            _frames_held = None
            if _free_now < 5.0 and int(_clip_state.get("frames", 1) or 1) > 22:
                _frames_held = _clip_state["frames"]
                _clip_state["frames"] = 22
                logger.info("[h3-ft] %.1f GB free is not enough for a %d-frame preview "
                            "without spilling — rendering 22 frames this time. Full length "
                            "returns automatically when more VRAM is free (closing other "
                            "GPU apps can help).", _free_now, _frames_held)
            if _ft_turbo_path:
                turbo_net, turbo_adaln = load_preview_turbo(dit, _ft_turbo_path,
                                                            turbo_lora_strength)
            try:
                _render_previews(n)
            finally:
                if _frames_held is not None:
                    _clip_state["frames"] = _frames_held
                if turbo_net is not None:
                    for _l in turbo_net.unet_loras:
                        _m = getattr(getattr(_l, "org_forward", None), "__self__", None)
                        if _m is not None:
                            _m.__dict__.pop("forward", None)
                    turbo_net.to("cpu")
                turbo_net, turbo_adaln = None, []
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            # Deliberately NO re-activation here. The model stays deactivated (fully
            # consistent — the master holds everything) and the NEXT epoch's rotation
            # check sees active=[] != wanted and activates + rebinds through the normal
            # path. Two wins: one activation code path, and on a FAILED preview the
            # activation happens only after the exception (whose traceback pins the
            # render's tensors) has been fully released — re-activating inside the
            # exception's lifetime OOM'd in the field (30.1 GB at the activate).
            import gc as _gc2
            _gc2.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- epoch loop ----
    loss_recorder = LossRecorder()
    if do_previews and sample_at_first and start_epoch == 0:
        try:
            if rotator is not None:
                _ft_render_previews(0)
            else:
                _render_previews(0)
        except Exception as _e0:
            if _clip_state["frames"] > 1:
                _was = _clip_state["frames"]
                _clip_state["frames"] = clip_fallback_frames(_was)
                logger.warning(
                    f"[preview] Sample at Start failed in CLIP mode ({type(_e0).__name__}) at "
                    f"{_was} frames — later previews retry at "
                    f"{_clip_state['frames']} frame(s). Training continues.")
            else:
                logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                               f"continues; per-epoch previews will still be attempted.")
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="minimax-h3")
    # Distillation only: the two loss terms, summed over the epoch. The 0.8/0.2 weights are known
    # up front; what is not is how BIG each error is, and that is what actually decides how much
    # of the learning comes from real pixels versus from the teacher's rendering of them.
    _distill_parts = {}
    _distill_acc = [0.0, 0.0, 0]        # teacher sum, photo sum, count
    # Clips with sound only. Reported separately per epoch because the two streams sit on
    # different noise schedules — one averaged number would hide which of them is learning.
    _audio_parts = {}
    _audio_acc = [0.0, 0.0, 0]          # video sum, audio sum, count — clips with sound
    _voice_acc = [0.0, 0]               # audio sum, count — audio-only voice items

    _pending = [0]                       # backwards accumulated since the last optimizer step
    _window_photo_only = [True]          # likeness mask: does this window hold ONLY photo steps?
    _window_voice_only = [True]          # voice-routing mask: ONLY audio steps in this window?
    _window_clip_only = [True]           # clip-routing mask: ONLY video-clip steps in this window?

    def _boundary_step():
        """The optimizer step at a window boundary — shared by live iterations and by the
        stopped-category skip path (whose iterations can land ON a boundary with earlier
        grads still pending). Guarded: a window of nothing steps nothing.

        Warmup, the ramp, the identity-first phase scale, the noise-band multiplier and the
        retirement anchor all COMPOSE into param_group["lr"] — never the loss: Adam's update
        is m/(sqrt(v)+eps), invariant to a constant on the gradient, so a loss multiplier
        reads as a working throttle and changes almost nothing. Both window multipliers are
        the MEAN over the window, not the last draw — taking the last would make the setting
        depend on the order batches happened to arrive in.
        """
        if not _pending[0]:
            return
        # Likeness mask, decided per WINDOW: photo-only windows drop the masked params' grads
        # before the clip (so the global norm reflects only what actually trains) and before the
        # step (grad=None params are skipped by AdamW outright — no moment update, no decay).
        # A window containing any clip/voice step trains the full model — conservative; at the
        # default accumulation of 1 this is exact per-step masking. StepClipper note: a photo
        # window shows zero movement for the masked blocks, which would pull its median down if
        # the limiter were on (it ships retired).
        if _photo_mask_params and _window_photo_only[0]:
            for _p in _photo_mask_params:
                _p.grad = None
        _window_photo_only[0] = True
        # Voice routing, same rule: a voice-only window drops the out-of-zone params' grads
        # (mixed windows mask nothing — conservative, exact at the default accumulation of 1).
        if _audio_mask_params and _window_voice_only[0]:
            for _p in _audio_mask_params:
                _p.grad = None
        _window_voice_only[0] = True
        # Clip routing, same rule again: a clip-only window drops the out-of-range grads.
        if _clip_mask_params and _window_clip_only[0]:
            for _p in _clip_mask_params:
                _p.grad = None
        _window_clip_only[0] = True
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        _bm = (sum(_band_acc) / len(_band_acc)) if _band_acc else 1.0
        _band_acc.clear()
        _cm = (sum(_cat_acc) / len(_cat_acc)) if _cat_acc else 1.0
        _cat_acc.clear()
        # `or _hn_active or _retire_active` is not decoration: warmup is retired for this
        # family and the ramp is OFF in the Fast preset, so without them this block never
        # runs for the preset most people use and the settings would silently do nothing.
        if warmup_steps or ramp is not None or _p1_epochs or _hn_active or _retire_active:
            _wf = (min(1.0, (global_step + 1) / warmup_steps) if warmup_steps else 1.0)
            _rm = ramp.mult if ramp is not None else 1.0
            for _g in optimizer.param_groups:
                _g["lr"] = _g["_warmup_base_lr"] * _wf * _rm * _phase_lr * _bm * _cm
        if limiter is not None:
            limiter.pre_step()           # snapshot BEFORE the optimizer moves anything
        if optimizer is not None:        # None under FT's fused backward (per-tensor hooks)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if limiter is not None:
            limiter.step()
        if ramp is not None:
            ramp.step()                  # post-clip weights; sets the multiplier for NEXT step
        if ema is not None:
            ema.update()                 # after the clip, so the shadow tracks clipped weights
        _pending[0] = 0

    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        if network is not None:
            network.train()
        if rotator is not None:
            # Rotate BEFORE the epoch's first step. The optimizer (or the fused per-tensor
            # set) is rebuilt from scratch each window — the outgoing window's Adafactor
            # state is meaningless to the incoming one — and `params` is REASSIGNED so
            # clip_grad_norm_ and the boundary step see the live window, not the old one.
            _want = _ft_want(epoch)
            if _want != list(rotator.active):
                # Decomposed rotate_to (deactivate -> activate is exactly what it does) so a
                # defrag can run at the one moment it helps: after the old window frees,
                # before the new one allocates. Windows torch has no expandable_segments, and
                # the ~6 GB in/out churn of every rotation fragments the reserve a little
                # each epoch — empty_cache only returns fully-EMPTY segments — until driver
                # memory hits the WDDM ceiling and steps silently crawl (field: a clean
                # first run pinned at 32.0/32.6 GB by epochs 6-7). The full park/restore
                # round-trip re-lands everything contiguously; seconds, and only when the
                # post-deactivate free is too tight for the incoming window.
                _act_now = list(rotator.active)
                if _act_now:
                    # Release the optimizer's grip BEFORE deactivating — the fused opts are
                    # KEYED on the outgoing window's Parameters, and _ft_rebind_optimizer
                    # only runs after the next window activates, so without this every
                    # rotation held BOTH windows at once. Block mode squeaked under the
                    # ceiling; full-model component mode did not (field: fc1's 15.4 GB
                    # stayed pinned — orphaned params are invisible to the park, so the
                    # defrag read 3.4 GB free and fc2's activation OOMed).
                    params = None
                    if _fused["on"]:
                        _detach_fused()
                    else:
                        optimizer = None
                    rotator.deactivate(_act_now)
                    import gc as _gcr
                    _gcr.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                try:
                    from fizgig.utils.device import plannable_free_vram as _pfv
                    _free_r = _pfv()
                except Exception:
                    _free_r = 99.0
                # Component window: projected bf16 size of the incoming spec — depth-split
                # entries count only their slice, so a 17-block fc1 chunk no longer reads
                # as the full-depth 15.4 GB.
                _need_r = _ft_window_gb(_want) + 2.0
                if _free_r < _need_r and not ft_stream:
                    logger.info("[h3-ft] %.1f GB free before activating the next window "
                                "(needs ~%.1f) — defragmenting via a full park/restore "
                                "round-trip.", _free_r, _need_r)
                    park_dit_to_cpu(dit)
                    import gc as _gcr2
                    _gcr2.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    restore_parked_dit(dit, device, 0)
                    try:
                        logger.info("[h3-ft] post-defrag: %.1f GB free", _pfv())
                    except Exception:
                        pass
                elif _free_r < _need_r:
                    # Streaming: the ring rebuild below evicts the streamed set before the
                    # window allocates — that IS the defrag here. A park/restore round-trip
                    # would push the ring's CPU-bound flats through dit.to(device) and
                    # materialize the whole base, so it never runs while the ring is live.
                    logger.info("[h3-ft] %.1f GB free before the next window (needs ~%.1f) "
                                "— the streaming ring rebuild reclaims the streamed set "
                                "first.", _free_r, _need_r)
                _ft_rebuild_ring(_want)   # evict the outgoing/streamed set, then activate
                rotator.activate(_want)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()   # per-window peak (logged per epoch)
                _ft_rebind_optimizer()
                logger.info("[h3-ft] epoch %d: window -> %s (%d trainable tensors)",
                            epoch + 1, _want, len(params))
        _epoch_trained = 0
        _distill_acc[:] = [0.0, 0.0, 0]
        # Identity-first: teacher-only while inside phase 1, photos-only after. Phase 2 takes the
        # ORDINARY loss path, so it never runs the teacher forward at all — half the compute of a
        # blended step, and no reference cache is touched.
        _teacher_phase = bool(distill and _p1_epochs and epoch < _p1_epochs)
        # Phase 1 runs at a THIRD of the Learning Rate box. It is placing the identity, not
        # reproducing detail, and it does that on a near-zero adapter where a full-size stride is
        # at its most destructive. Phase 2 gets the rate you actually asked for, starting from an
        # adapter that is already in the right place.
        _phase_lr = _P1_LR_SCALE if _teacher_phase else 1.0
        if _p1_epochs and epoch == _p1_epochs and epoch > start_epoch:
            logger.info(f"[distill] identity-first phase 1 complete after {_p1_epochs} epoch(s) "
                        f"— dropping the teacher; from here it trains on the photographs alone, "
                        f"at the full {learning_rate:.2e} (phase 1 ran at "
                        f"{learning_rate * _P1_LR_SCALE:.2e}).")
        for i, batch in enumerate(loader):
            # A still arrives 4-D (1,24,H,W); clips and voice placeholders are 5-D — tested
            # BEFORE the unsqueeze below erases the difference. Feeds the likeness mask.
            _is_photo = batch["latents"].dim() == 4
            latents = batch["latents"].to(device, dtype)           # (1, 24, H, W)
            if latents.dim() == 4:
                latents = latents.unsqueeze(2)                     # -> (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)        # (1, L, 5120)
            if uncond_text is not None and random.random() < caption_dropout:
                text = uncond_text.to(device, dtype)               # caption dropout step
            _is_voice = bool(batch.get("audio_only") is not None and batch["audio_only"].any())
            if not _is_photo or _is_voice:
                _window_photo_only[0] = False
            if not _is_voice:
                _window_voice_only[0] = False
            if _is_photo or _is_voice:
                _window_clip_only[0] = False
            # Per-category retirement: past its stop epoch a category is either ANCHORED
            # (trains on at ANCHOR_LR_SCALE — rehearsal against drift on the shared adapters,
            # ledger stays live) or STOPPED (skipped outright — faster epochs, blind).
            # The comparison is CUMULATIVE: ft_epoch_offset re-bases a continuation's local
            # epochs onto the original run's calendar (0 everywhere else, incl. LoRA resume,
            # whose restored numbering is already cumulative).
            _retired = ((not _is_voice and _vis_stop and (epoch + 1 + ft_epoch_offset) > _vis_stop)
                        or (_is_voice and _aud_stop and (epoch + 1 + ft_epoch_offset) > _aud_stop))
            _ret_mode = audio_stop_mode if _is_voice else visual_stop_mode
            if _retired and _ret_mode != "anchor":
                # No forward, no loss, no record — but a boundary landing here still owes any
                # pending grads from the window's live iterations their step.
                if (i + 1) % _accum_n == 0 or (i + 1) >= steps_per_epoch:
                    _boundary_step()
                global_step += 1
                progress_bar.update(1)
                continue
            if rotator is not None and finetune_scope == "photo" and (not _is_photo or _is_voice):
                # Photo-scope FT: clip/voice batches are skipped outright (same shape as the
                # retirement skip — a full clip forward at 4.5k tokens would be pure waste).
                # Fairness holds across the cycle: windows advance per epoch and every epoch
                # sees the identical photo subset.
                if (i + 1) % _accum_n == 0 or (i + 1) >= steps_per_epoch:
                    _boundary_step()
                global_step += 1
                progress_bar.update(1)
                continue
            # Modality routing (FT): freeze the blocks this batch's modality must not touch
            # for the span of its forward+backward — a component window spans every block,
            # so this per-parameter freeze is the only way to express "photos stay inside
            # 20-49, voice stays inside 34-49, clips inside clip_blocks when restricted".
            # The fused per-tensor hooks never fire for a no-grad param; restored right
            # after the backward (any exception here is fatal to the run anyway).
            _frz = []
            if rotator is not None:
                _frz = (_ft_freeze["voice"] if _is_voice
                        else (_ft_freeze["photo"] if _is_photo
                              else _ft_freeze["clip"]))
                for _p in _frz:
                    _p.requires_grad_(False)
            if (distill and (_teacher_phase or not _p1_epochs) and "ref_hidden_states" in batch
                    and not _is_voice):
                # `not _is_voice`: compute_distill_loss packs still-sized audio noise (4 rows)
                # against a voice item's hundreds — the sequence and RoPE table would disagree
                # in length and crash mid-run. A voice has no face to distill anyway; it takes
                # the plain path below at video_weight 0.
                _rz = batch["ref_latent"].to(device, dtype)      # (1, 24, h, w) from the cache
                if _rz.dim() == 4:
                    _rz = _rz.unsqueeze(2)                       # -> (1, 24, 1, h, w)
                loss, _ = compute_distill_loss(
                    dit, network, latents, text,
                    text_ref=batch["ref_hidden_states"].to(device, dtype),
                    ref_latents=[_rz],
                    text_token_tags=batch["ref_token_tags"][0],
                    # Phase 1 is teacher-ONLY (weight 1.0); the blended mode keeps the box value.
                    distill_weight=(1.0 if _teacher_phase else distill_weight),
                    shift=shift, seed=seed, parts_out=_distill_parts)
                _distill_acc[0] += _distill_parts["teacher"]
                _distill_acc[1] += _distill_parts["photo"]
                _distill_acc[2] += 1
            else:
                # A clip that carried usable sound cached an audio_latent; a still, a muted clip
                # and a silent one did not, and for those this is the original call unchanged.
                # A VOICE item is its audio: the video latent is a zeros placeholder, and
                # video_weight=0 keeps "every frame looks like the dataset mean" out of the run.
                _a = batch.get("audio_latent")
                if _a is not None:
                    _a = _a[0].to(device)            # (1, A*2, 32) -> the DiT's row block
                loss, _step_sigma = compute_loss(dit, latents, text, shift=shift,
                                                 audio_latent=_a, audio_weight=audio_weight,
                                                 video_weight=0.0 if _is_voice else 1.0,
                                                 parts_out=_audio_parts)
                if _is_voice:
                    # Its own ledger. The clip ledger's "video err" is a real number about real
                    # footage; a voice item's video term is its error against the placeholder —
                    # folding that in would poison both averages in the epoch report.
                    if _audio_parts.get("audio") is not None:
                        _voice_acc[0] += _audio_parts["audio"]
                        _voice_acc[1] += 1
                elif _a is not None and _audio_parts.get("audio") is not None:
                    _audio_acc[0] += _audio_parts["video"]
                    _audio_acc[1] += _audio_parts["audio"]
                    _audio_acc[2] += 1
                # The band multiplier needs to know where this step landed. Only the plain path
                # reports it; a distillation step keeps the multiplier at 1.0, since its loss is
                # a teacher comparison rather than a draw from the noise schedule. With sound in
                # the batch the reported sigma is the VIDEO draw's — the box is defined against
                # the video schedule, and audio's own remapped curve must not vote on the LR.
                # A voice step also sits out: its entire gradient lives on the audio schedule
                # (shift 3), which the video-sigma classification does not describe.
                _band_acc.append(1.0 if _is_voice else
                                 (_hn_scale if _step_sigma >= MINIMAX_LOWNOISE_SIGMA else 1.0))
            # An anchored retired step trains at the anchor scale; everything else at 1.0.
            # Appended for EVERY executed backward (distill path included — a retired visual
            # category's distillation steps anchor exactly like its plain ones).
            _cat_acc.append(ANCHOR_LR_SCALE if _retired else 1.0)
            # Regularisation images (FT): scale THIS step's gradient down to the nudge —
            # loss scaling, because the fused per-tensor update has no optimizer object
            # whose LR could be rewritten. The RAW loss is what the ledgers record below,
            # so avr_loss and the drift lines see unscaled numbers (Krea 2 FT parity).
            #
            # Honesty note (review, 26 Aug — and the exception to _boundary_step's
            # "never the loss" doctrine): Adafactor's second-moment normalisation
            # partially cancels a constant gradient scale, so x0.2 realises as roughly
            # x0.2-0.25 ONLY while reg steps are the minority feeding each tensor's EMA
            # (the [reg] majority warning above is that condition). The very first step
            # after each rotation rebind is fully UNthrottled (fresh state: beta2t=0
            # makes the update scale-invariant) — ~one stray full-LR reg step per
            # rotation at typical reg ratios. Krea 2 FT has identical behaviour; the
            # doctrine comment stays right for everything that CAN ride param-group LR.
            _bk = loss
            if reg_keys and any(str(k) in reg_keys for k in (batch.get("item_keys") or ())):
                _bk = loss * reg_mult
            # Divide so the accumulated gradient is the MEAN over the window, not the sum —
            # otherwise the effective LR scales with the accumulation count.
            (_bk / _accum_n if _accum_n > 1 else _bk).backward()
            for _p in _frz:
                _p.requires_grad_(True)
            _pending[0] += 1
            # Step on the window boundary, and always on the last batch of the epoch so a
            # partial tail window is never silently discarded.
            if (i + 1) % _accum_n == 0 or (i + 1) >= steps_per_epoch:
                _boundary_step()
            global_step += 1
            _epoch_trained += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)

        # The last step's `loss` keeps its whole autograd graph alive across the epoch
        # boundary, and each leaf AccumulateGrad node holds a STRONG ref to its Parameter —
        # under rotation FT that pins the entire deactivated window (~6 GB of orphaned bf16
        # at 8 blocks, census-confirmed) straight through the bracket preview.
        loss = batch = None
        # Cumulative across pause/resume, matching checkpoint numbering — a resumed run
        # logging "epoch 1/95" beside checkpoints named -000006 read as a numbering bug
        # (field, 29 Aug). Offset is 0 on a fresh run, so nothing changes there.
        logger.info(f"epoch {epoch + 1 + ft_epoch_offset}/{max_train_epochs + ft_epoch_offset} "
                    f"done — avr_loss {loss_recorder.moving_average:.4f}")
        if rotator is not None and torch.cuda.is_available():
            # Per-window peak, reset at each rotation — the measured record the window
            # planner's constants are calibrated from. GB (1e9), NOT GiB: this line used
            # to divide by 2**30 while every planner constant was in GB, and the 7% gap
            # silently put _FT_OVERHEAD_GB a full GB too low (found 27 Aug, field gate).
            logger.info("[h3-ft] window peak VRAM: %.1f GB",
                        torch.cuda.max_memory_allocated() / 1e9)
        if rotator is not None and _epoch_trained == 0:
            if finetune_scope == "photo":
                raise RuntimeError(
                    "[h3-ft] photo-scope fine-tune trained ZERO steps this epoch — the "
                    "dataset has no photos (clips/voice only). Use --finetune_scope all, "
                    "or add photos.")
            if _vis_stop or _aud_stop:
                raise RuntimeError(
                    "[h3-ft] category retirement left this epoch with nothing to train — "
                    "every batch belongs to a retired category. Lower the stop epoch, or "
                    "end the run at it with Max Epochs.")
        # Optimizer sanity: lora_up starts at zero and an Adam-family step is bounded by ~lr, so
        # after N steps no element can honestly exceed ~3*N*lr. When the 8-bit second moment
        # misbehaves (v quantized to zero -> update degrades to lr*m/eps) the drift blows through
        # that bound by orders of magnitude — caught here per epoch instead of per melted preview.
        # LoRA-specific by construction — FT's movement signal is the per-window write-back log.
        if rotator is None:
            try:
                _lr_now = optimizer.param_groups[0]["lr"]
                _drift = max((float(l.lora_up.weight.detach().abs().max())
                              for l in network.unet_loras if hasattr(l, "lora_up")), default=0.0)
                _bound = 3.0 * global_step * _lr_now
                if _drift > _bound:
                    logger.warning(f"[drift] max|lora_up|={_drift:.4f} EXCEEDS the Adam bound "
                                   f"~{_bound:.4f} ({global_step} steps @ lr={_lr_now:.1e}) — the "
                                   f"optimizer is stepping far beyond the configured LR (8-bit "
                                   f"state underflow?). Expect degraded samples.")
                else:
                    logger.info(f"[drift] max|lora_up|={_drift:.4f} (bound ~{_bound:.4f} — healthy)")
            except Exception:
                pass
        if _p1_epochs and not _teacher_phase and distill:
            logger.info(f"[distill] photos only (identity-first phase 2) — the teacher was "
                        f"dropped after epoch {_p1_epochs}.")
        if _audio_acc[2]:
            _av, _aa = _audio_acc[0] / _audio_acc[2], _audio_acc[1] / _audio_acc[2]
            _wa = float(audio_weight) * _aa
            _share = 100 * _wa / (_av + _wa) if (_av + _wa) else 0.0
            # The share is the number to watch. Audio is only ~4% of the packed sequence, so if
            # its weighted contribution is a rounding error the LoRA is not learning sound no
            # matter how healthy the total loss looks — that is what audio_weight is for.
            logger.info(
                f"[audio] {_audio_acc[2]} clip(s) with sound | video err {_av:.4f} | "
                f"audio err {_aa:.4f} x{float(audio_weight):.2f} = {_wa:.4f} | "
                f"sound is {_share:.1f}% of their loss")
            _audio_acc[:] = [0.0, 0.0, 0]
        if _voice_acc[1]:
            logger.info(f"[voice] {_voice_acc[1]} voice step(s) | "
                        f"audio err {_voice_acc[0] / _voice_acc[1]:.4f} "
                        f"x{float(audio_weight):.2f} (video loss zeroed — placeholder frames)")
            _voice_acc[:] = [0.0, 0]
        if _distill_acc[2]:
            _t = _distill_acc[0] / _distill_acc[2]
            _p = _distill_acc[1] / _distill_acc[2]
            _w = 1.0 if _teacher_phase else float(distill_weight)
            # Weighted contributions are what the optimizer actually sees. The raw errors are
            # printed too, because the interesting question is whether the photo term is HARDER
            # (bigger error) than the teacher term, which is what lets 20% punch above its weight.
            _wt, _wp = _w * _t, (1.0 - _w) * _p
            _tot = _wt + _wp
            logger.info(
                f"[distill] teacher err {_t:.4f} x{_w:.2f} = {_wt:.4f} | "
                f"photo err {_p:.4f} x{1 - _w:.2f} = {_wp:.4f} | "
                f"real pixels are {100 * _wp / _tot if _tot else 0:.0f}% of this epoch's loss "
                f"(the weight alone says {100 * (1 - _w):.0f}%)")
        if limiter is not None:
            logger.info(limiter.epoch_report())
        if ramp is not None:
            logger.info(ramp.epoch_report())
        if adaptive is not None:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)
        ft_ckpt_saved_this_epoch = False
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            ckpt = os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors")
            if rotator is not None:
                # The full checkpoint IS the resumable state under FT (the continuation is
                # --dit <this file> + the printed start_window). ~21 GB per save.
                from fizgig.minimax.rotation_ft import save_full_checkpoint_h3
                ckpt = os.path.join(output_dir,
                                    f"{output_name}-{epoch + 1 + ft_epoch_offset:06d}.safetensors")
                _next_w = rot_schedule.window_at(epoch + 1)
                save_full_checkpoint_h3(rotator, dit_path, ckpt, extra_metadata={
                    **_meta(), "fizgig_next_start_window": str(_next_w),
                    "fizgig_ft_n_windows": str(rot_schedule.n_windows),
                    "fizgig_ft_epochs_done": str(epoch + 1 + ft_epoch_offset)})
                ft_ckpt_saved_this_epoch = True
                logger.info("[h3-ft] to continue from this checkpoint: --dit %s "
                            "--finetune_start_window %d", os.path.basename(ckpt), _next_w)
            else:
                if ema is not None:
                    ema.swap_in()
                try:
                    _save_lora(network, ckpt, network_dim, network_alpha, dtype, _meta())
                finally:
                    if ema is not None:
                        ema.swap_out()
            logger.info(f"saved {ckpt}")
            if save_state and rotator is not None:
                logger.info("[h3-ft] state dirs are skipped — the full checkpoint is the "
                            "state; continue with --dit + --finetune_start_window.")
            if save_state and rotator is None:
                # Non-fatal (see the krea2 twin): a failed convenience save must never kill a
                # run whose checkpoint already wrote. Truly-full disks fail the next epoch
                # CHECKPOINT, and that one is rightly fatal.
                try:
                    _save_training_state(output_dir, output_name, network, optimizer,
                                         epoch=epoch + 1, global_step=global_step,
                                         dtype=dtype, extra=_state_extra(), ema=ema)
                    prune_state_dirs(output_dir, output_name, keep_last_n_states)
                except Exception as _se:
                    logger.error("[state] saving the resume state FAILED (%s: %s) — likely the "
                                 "disk (on RunPod the volume quota is only visible in the "
                                 "dashboard). Training continues; this epoch has no resume "
                                 "point. The epoch checkpoint itself already saved.",
                                 type(_se).__name__, _se)
        # Under FT previews FOLLOW CHECKPOINT SAVES (Peter, 24 Aug): one preview per saved
        # checkpoint, so the sample gallery maps 1:1 onto files you can actually deploy.
        # Saves only land on cycle boundaries, so the old per-cycle equal-training honesty
        # is preserved; a sparser save cadence (a multiple of the cycle) now previews
        # sparser too, and Save-every 0 (final only) previews only at the end. The final
        # epoch always previews — its checkpoint is the final save after the loop.
        # Sample-every-N still doesn't apply; the Samples tab still gates previews on/off.
        _ft_saved_this_epoch = bool(save_every_n_epochs
                                    and (epoch + 1) % save_every_n_epochs == 0
                                    and (epoch + 1) < max_train_epochs)
        _prev_due = ((_ft_saved_this_epoch or (epoch + 1) >= max_train_epochs)
                     if rotator is not None
                     else bool(sample_every_n_epochs
                               and (epoch + 1) % sample_every_n_epochs == 0))
        if do_previews and _prev_due:
            try:
                # Previews render on the EMA weights when EMA is on — a preview must show what
                # the saved checkpoint will look like, not the raw zigzag the EMA exists to hide.
                if ema is not None:
                    ema.swap_in()
                try:
                    if rotator is not None:
                        _ft_render_previews(epoch + 1)
                    else:
                        _render_previews(epoch + 1)
                    vram_line("post-preview")
                finally:
                    if ema is not None:
                        ema.swap_out()
            except Exception as _pe:
                # Latch previews OFF for the rest of the run rather than re-failing (and
                # re-OOMing) every epoch. Training and checkpoints are never at risk.
                _oom = "out of memory" in str(_pe).lower()
                if _clip_state["frames"] > 1:
                    # The failure arrived in CLIP mode — the mode a still preview never
                    # exercised. Step DOWN the frame grid and keep clips rather than ending
                    # every preview for the run; only a failure at stills latches off.
                    _was = _clip_state["frames"]
                    _clip_state["frames"] = clip_fallback_frames(_was)
                    logger.warning(
                        f"[preview] epoch {epoch + 1} CLIP preview failed at {_was} frames "
                        f"({'CUDA OOM' if _oom else type(_pe).__name__}) — retrying at "
                        f"{_clip_state['frames']} frame(s) from the next preview on. Training "
                        f"continues and LoRAs still save normally.")
                else:
                    logger.warning(
                        f"[preview] epoch {epoch + 1} preview failed "
                        f"({'CUDA OOM' if _oom else type(_pe).__name__}); disabling previews for "
                        f"the rest of the run. Training continues and LoRAs still save normally.")
                    do_previews = False
            if network is not None:
                network.train()
        if os.path.exists(pause_flag):
            # Pause = graceful epoch-end exit with FULL state (regardless of the save-state
            # toggles), so Resume continues exactly here — matching Klein/Krea 2. The final
            # LoRA is deliberately NOT written; Resume (or the natural run end) writes it.
            # Under FT the "state" is a full checkpoint + the printed continuation command
            # (state-dir resume is structurally impossible — the LoRA is absent and the
            # optimizer may be per-tensor hooks).
            if rotator is not None:
                from fizgig.minimax.rotation_ft import save_full_checkpoint_h3
                _pp = os.path.join(output_dir,
                                   f"{output_name}-{epoch + 1 + ft_epoch_offset:06d}.safetensors")
                _next_w = rot_schedule.window_at(epoch + 1)
                try:
                    # ~21 GB and minutes per save: if the cadence just wrote this epoch's
                    # checkpoint above, don't rewrite the identical file — same dedupe as the
                    # LoRA path's state_saved_this_epoch.
                    if not ft_ckpt_saved_this_epoch:
                        save_full_checkpoint_h3(rotator, dit_path, _pp, extra_metadata={
                            **_meta(), "fizgig_next_start_window": str(_next_w),
                            "fizgig_ft_n_windows": str(rot_schedule.n_windows),
                            "fizgig_ft_epochs_done": str(epoch + 1 + ft_epoch_offset)})
                    logger.info("[h3-ft] paused. Continue with: --dit %s "
                                "--finetune_start_window %d", os.path.basename(_pp), _next_w)
                    # The checkpoint now carries everything — the scratch is superseded.
                    from fizgig.minimax.rotation_ft import MasterStore as _MS
                    if isinstance(rotator.master, _MS):
                        rotator.master.cleanup()
                except Exception as _se:
                    logger.error("[pause] checkpoint save FAILED (%s: %s) — there is NO new "
                                 "continuation point for this pause.", type(_se).__name__, _se)
            else:
                try:
                    _save_training_state(output_dir, output_name, network, optimizer,
                                         epoch=epoch + 1, global_step=global_step,
                                         dtype=dtype, extra=_state_extra(), ema=ema)
                except Exception as _se:
                    logger.error("[pause] state save FAILED (%s: %s) — there is NO new resume "
                                 "point for this pause. Free disk space (RunPod: dashboard "
                                 "quota) and resume from the previous saved state.",
                                 type(_se).__name__, _se)
            try:
                os.remove(pause_flag)
            except OSError:
                pass
            progress_bar.close()
            logger.info(f"[pause] requested — state saved at epoch {epoch + 1}. Exiting cleanly.")
            sys.exit(0)

    progress_bar.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    if rotator is not None:
        from fizgig.minimax.rotation_ft import save_full_checkpoint_h3
        _next_w = rot_schedule.window_at(max_train_epochs)
        save_full_checkpoint_h3(rotator, dit_path, final, extra_metadata={
            **_meta(), "fizgig_next_start_window": str(_next_w),
            "fizgig_ft_n_windows": str(rot_schedule.n_windows),
            "fizgig_ft_epochs_done": str(max_train_epochs + ft_epoch_offset)})
        logger.info("[h3-ft] saved final fine-tuned checkpoint: %s — test it in ComfyUI as a "
                    "normal H3 model, or distil it to a LoRA with Checkpoint to LoRA. To train "
                    "it further: --dit %s --finetune_start_window %d",
                    final, os.path.basename(final), _next_w)
        # The final checkpoint supersedes the scratch — reclaim the disk.
        from fizgig.minimax.rotation_ft import MasterStore as _MS
        if isinstance(rotator.master, _MS):
            rotator.master.cleanup()
        try:
            os.remove(pause_flag)
        except OSError:
            pass
        return final
    if ema is not None:
        ema.swap_in()
    try:
        _save_lora(network, final, network_dim, network_alpha, dtype, _meta())
    finally:
        if ema is not None:
            ema.swap_out()
    logger.info(f"saved final LoRA: {final}")
    if save_state_on_train_end and max_train_epochs > start_epoch:
        # Non-fatal: the final LoRA is already on disk; dying here would turn a finished run red.
        try:
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=max_train_epochs, global_step=global_step,
                                 dtype=dtype, extra=_state_extra(), ema=ema)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
        except Exception as _se:
            logger.error("[state] end-of-run state save FAILED (%s: %s) — the finished LoRA is "
                         "saved and fine; only train-further-by-resume is affected.",
                         type(_se).__name__, _se)
    try:
        os.remove(pause_flag)
    except OSError:
        pass
    return final
