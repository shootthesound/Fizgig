"""MiniMax H3's training helpers that outlive the old trainer: the loss, the timestep sampler, the VRAM and base
planners, the preview helpers (OOM ladder, decode park, clip / sound writers), the AdaLN injection for frozen LoRAs on
the pruned base, the block-spec parser and the fine-tune's routing / stop snapping. Moved verbatim from
minimax/trainer.py (3 Oct 2026) so the driver (minimax/driver.py, ft_backend.py), RefMod and the workbench engine
keep them when the old trainer goes; trainer.py imports them from here until then.
"""
import contextlib
import gc
import logging
import math
import os

import torch
import torch.nn.functional as F

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


from fizgig.utils.block_spec import format_block_spec, parse_block_spec  # noqa: E402,F401


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
    likeness set when given, voice -> audio_blocks when given, clips -> clip_blocks when given,
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


# HQQ 4-bit g8 w/ 8-bit group vectors (rintic-13, #102): 0.5 B/param of codes + two uint8 vectors at 1/8 =
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
        if not any(r.fullmatch(name) for r in rx):   # fullmatch, as create_modules matches
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


def frozen_lora_vram_gb(path: str, bytes_per_elem: int = 2) -> float:
    """GB a FROZEN side LoRA holds on the card for the whole run — the training adapter, or a
    Context LoRA. Read from the safetensors HEADER, so this works before the DiT is built, like
    adapter_param_count.

    load_context_lora does `net.to(device=device, dtype=dtype)`, so the file lands in the training
    dtype (bf16 = 2 bytes), not the dtype it was stored in — count elements, not file bytes. It is
    applied to the DiT and never freed, so it belongs in the resident term exactly like the
    trainable adapter's weights. Any AdaLN rows the file carries are injected from the same
    tensors and are counted here too.

    Not a rounding error: the training adapter is on by default in every H3 preset. Fizgig's
    default is Circlestone's (rank 64, 0.62 GB resident); Ostris's v1 is rank 16, 0.155 GB, and
    his v2 at rank 32 is 0.310 GB if a user points the path at it. This reads whichever file
    is configured rather than assuming either.
    """
    if not path:
        return 0.0
    try:
        if not os.path.isfile(path):
            return 0.0
        import json
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        elems = 0
        for key, ent in hdr.items():
            if key == "__metadata__" or not isinstance(ent, dict):
                continue
            shape = ent.get("shape") or []
            c = 1
            for d in shape:
                c *= int(d)
            elems += c if shape else 0
        return elems * int(bytes_per_elem) / 1e9
    except Exception:
        return 0.0     # unreadable header: plan as before rather than refuse to plan


def ema_shadow_gb(params: int) -> float:
    """GB the EMA shadow holds. EMAWeights keeps `p.detach().clone().float()` per trainable
    parameter — a full FP32 copy on the same device, live for the whole run (swap_in/swap_out
    only bracket saves and previews). Four bytes, not two: 1.25 GB on LoKR factor 8, which is
    most of the planner's whole reserve. Zero when EMA is off, and FT rotation forces it off
    before the plan runs."""
    return max(0, int(params)) * 4 / 1e9


def plan_adapter_gb(params: int, optimizer_type: str = "adamw8bit", *,
                    training_adapter_path: str = None, context_lora_path: str = None,
                    ema_decay: float = 0.0, ft_rotation: int = 0):
    """Everything the adapter side of a run keeps resident, as (total, frozen_gb, ema_gb) GB.

    Pure so the gating is testable: the planner runs deep inside train_minimax behind a GPU and a
    21 GB checkpoint, and until this existed the only coverage of the gates was grepping the
    source for the lines that implement them, which passes whether or not they run.

    The gates mirror what the run actually does. The training adapter is resident under both
    modes (under fine-tune rotation it rides as forward hooks). A Context LoRA is refused
    under rotation (it raises at load time) and EMA is forced off by the FT coercion block,
    so neither is resident there. Otherwise each term counts only when it is really present.
    """
    total = adapter_vram_gb(params, optimizer_type)
    frozen = frozen_lora_vram_gb(training_adapter_path)
    ema = 0.0
    if not ft_rotation:
        frozen += frozen_lora_vram_gb(context_lora_path)
        if ema_decay and float(ema_decay) > 0:
            ema = ema_shadow_gb(params)
    return total + frozen + ema, frozen, ema


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


_H3_CACHE_SUFFIX = "_minimaxh3.safetensors"     # {basename}_{WxH}_minimaxh3 - this family's own latent caches


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
            if not name.endswith(_H3_CACHE_SUFFIX):     # the folder is shared: other families' latents are 8x, not 16x
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
            if not name.endswith(_H3_CACHE_SUFFIX):     # the folder is shared: other families' latents are 8x, not 16x
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
                 parts_out: dict = None, ref_latents=None, loss_mask=None, prior=None):
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

    loss_mask   : optional (1, 1, h, w) per-cell weights on the latent's grid, as the dataset
                  builds them (masked training, dataset/masks.py): the video term's error is
                  weighted per position and rescaled to the same total, so a masked picture
                  counts for as much as an unmasked one. Never the audio term — a mask is a
                  picture's, and the dataset gives one only to a still.
    prior       : optional masked prior preservation (families/train.py _MaskPrior): with a
                  loss_mask, the target outside the mask is the model's own prediction with the
                  LoRA at 0, taken before the training forward.

    Returns (loss, sigma_used). parts_out, if given, receives the video and audio terms
    separately — they are on different noise schedules and averaging them into one number hides
    which stream is actually learning. The prior term is added after parts_out is filled, so the video term it
    reports is the video term alone.
    """
    from fizgig.families.driver import prior_base, weighted_loss, with_prior
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
    # The mask arrives on the UNCROPPED latent grid (the dataset sizes it from the cached latent), so
    # it takes exactly the crop x0 just had - otherwise se and mw differ by a row or a column on any
    # latent whose edges aren't a patch multiple.
    mw = None if loss_mask is None else loss_mask.float().to(device)[..., :_Hc, :_Wc]
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

    # ref_latents: RefMod mode — the mod rides as the reference block on every step (the LoRA
    # learns what the reference can't carry). None = the ordinary step.
    _ref_kw = {"ref_latents": ref_latents} if ref_latents else {}
    if audio_latent is None:
        # masked prior preservation's forward runs BEFORE the training forward (families/driver.py)
        base = prior_base(prior, loss_mask, lambda: model(noised.to(latent.dtype), t, text_embeds, **_ref_kw))
        pred = model(noised.to(latent.dtype), t, text_embeds, **_ref_kw)
        target = (x0 - noise).to(pred.dtype).float()
        if mw is None:
            loss = F.mse_loss(pred.float(), target)
        else:
            loss = weighted_loss((pred.float() - target).pow(2).mean(dim=1), mw)      # one weight per cell
        if parts_out is not None:
            parts_out.update(video=float(loss.detach()), audio=None)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred.float() - base).pow(2).mean(dim=1), mw)
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
                         audio_rows=a_noised, return_audio=True, **_ref_kw)
    target = (x0 - noise).to(pred.dtype).float()
    if mw is None:
        v_loss = F.mse_loss(pred.float(), target)
    else:
        v_loss = weighted_loss((pred.float() - target).pow(2).mean(dim=1), mw)
    # No prior term on this path: it is a clip with sound (a still never reaches it), and the
    # dataset gives a mask to a still alone - a clip step's mask is all-white, so it preserves nothing.
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
    """Mux decoded preview frames [3, F, H, W] in [0,1] with their wav into a playable mp4 (wav_path None: silent).

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
           *(["-i", wav_path] if wav_path else []),
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           *(["-c:a", "aac", "-b:a", "192k", "-shortest"] if wav_path else []), "-movflags", "+faststart", path]
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
    from fizgig.networks.lora import create_network_from_weights
    keep, adaln_pairs, dropped = _prefilter_frozen_lora(dit, path, strength)
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


def _prefilter_frozen_lora(dit, path, strength):
    """A frozen LoRA file against THIS base: -> (keep, adaln_pairs, dropped). `keep` is the
    kohya state dict restricted to Linears that exist here with matching shapes; the
    full-model AdaLN rows come back as (AdalnProj, A, B*strength) pairs for the run-time
    injection; `dropped` names what matched nothing (see load_preview_turbo)."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import ensure_kohya_lora_state_dict
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
    return keep, adaln_pairs, dropped


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
