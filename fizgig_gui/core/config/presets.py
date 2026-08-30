import os
import re

from fizgig_gui.core.domain.minimax_math import MINIMAX_BASE_QUANT_OPTIONS


def ft_checkpoint_continuation(path):
    """(next_window, epochs_done) for a rotation fine-tune checkpoint.

    A fine-tune's continuation point is the full checkpoint itself, not a state dir. The
    trainers stamp `fizgig_next_start_window` (where the rotation cycle picks back up) and
    `fizgig_ft_epochs_done` (cumulative epochs trained) into every FT save; the epoch count
    also lives in the filename's -NNNNNN, which wins when present (the un-numbered FINAL
    artifact has only the metadata). Anything missing reads as 0 — a continuation that
    restarts the cycle at window 0 is degraded, not wrong."""
    next_window, epochs_done = 0, 0
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
        next_window = int(md.get("fizgig_next_start_window", 0))
        epochs_done = int(md.get("fizgig_ft_epochs_done", 0))
    except Exception:
        pass
    m = re.search(r"-(\d{6})\.safetensors$", os.path.basename(path or ""))
    if m:
        epochs_done = int(m.group(1))
    return next_window, epochs_done


# Built-in presets — always available in the Load Preset dropdown, prefixed with ✨ to distinguish
# from user-saved presets. Defined in code so they ship with the app and can't be deleted accidentally.
# Tune these as empirical findings accumulate.
BUILT_IN_PRESETS = {
    "✨ Old Reliable (rank 16, full model, single subject)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 55, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)": {
        "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 55, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Identity (rank 4, single subject)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Identity (rank 8, harder dataset)": {
        "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 20, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Multi-Character (rank 16, multi character or concept)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 2e-4,
        "MAX_TRAIN_EPOCHS": 50, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Style (late timesteps)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Style", "MIN_TIMESTEP": "0", "MAX_TIMESTEP": "400",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Style+Composition (all timesteps)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Style+Composition", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
}

# Built-in presets for Krea 2. The Klein block-targeting / adaptive-LR / timestep presets
# above don't apply (Krea 2 trains all linears, no block map yet, no adaptive LR), so Krea 2
# ships a single sensible-defaults entry. Users can still save their own via Save Preset —
# those land in the per-architecture preset folder and appear alongside this one.
KREA2_BUILT_IN_PRESETS = {
    "✨ Krea 2 Defaults (rank 32, full model)": {
        "NETWORK_DIM": 32, "NETWORK_ALPHA": 32, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 30, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "0.25",
        # Memory settings all auto — each resolves from the actual GPU at launch.
        # BLOCKS_SWAP must be the combobox's exact label: _apply_preset_values matches a
        # preset value against the offered options on its first token, case-sensitively,
        # so a bare "auto" would not select "Auto (detect from GPU)".
        "BLOCKS_SWAP": "Auto (detect from GPU)",
        "QUANT_4BIT_MODE": "auto", "COMPILE_BLOCKS": "Auto",
        # Per-image loss watch: detection + the LR throttle on, the two interventions that
        # rewrite captions or pre-judge images left off — those want a deliberate choice.
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
    },
    # Rank 8 + Adaptive LR at an aggressive floor: fewer epochs to a usable LoRA. Everything
    # else identical to Krea 2 Defaults (which stays the preset applied on family switch).
    "✨ Krea 2 Ultra Fast (rank 8, adaptive LR)": {
        "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 20, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "0.25",
        "BLOCKS_SWAP": "Auto (detect from GPU)",
        "QUANT_4BIT_MODE": "auto", "COMPILE_BLOCKS": "Auto",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
    },
    # Style. Two deliberate departures from the identity-shaped presets above:
    #
    #   Rank 16 — style is a broad global direction (palette, texture, rendering), not the
    #   fine detail identity memorises. The only style-specific parameter advice published
    #   for Krea 2 is about rank, not LR ("rank 16 for simple styles, 64 for complex
    #   subjects" — Krea2Trainer), so 16 follows the one datapoint that exists.
    #
    #   LR ceiling 2e-4, not 4e-4 — this matters MORE than the floor. The adaptive watcher
    #   probes UP x1.25 on steady descent, and style loss descends steadily early, so it
    #   climbs toward whatever ceiling it's given; 4e-4 is where style overbake lives. The
    #   geometric-midpoint start rule puts min 5e-5 / max 2e-4 at exactly 1e-4 — the LR the
    #   whole Krea 2 ecosystem defaults to (Krea's own docs, Krea2Trainer, RunComfy) — while
    #   the watcher keeps its protection in both directions.
    #
    # Fewer epochs (15) because style overbakes fast, and on Krea 2 that shows up as
    # generations dragging toward the training set's COMPOSITIONS, not just its look —
    # so save every epoch and scrub for the sweet spot in LoRA Royale.
    "✨ Krea 2 Style (rank 16, gentle LR)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "5e-5", "ADAPTIVE_LR_MAX": "2e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "0.25",
        "BLOCKS_SWAP": "Auto (detect from GPU)",
        "QUANT_4BIT_MODE": "auto", "COMPILE_BLOCKS": "Auto",
        # Detection + throttle on as usual. Look-outlier warm-up stays off for an extra
        # reason here: it scores images by FACE embedding, which is meaningless on a style
        # set that may have no faces at all.
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
    },
}

# LoRA Royale seed-travel presets — recipes of the *mechanics* knobs (reference /
# anchor / journey / identity-lock), NOT a prompt system (seed travel uses the Setup
# prompt). Reference is kept light (0.1–0.4) so the seeds can actually travel.
SEED_TRAVEL_PRESETS = {
    # Each varies a clearly different knob:
    "Hold subject":   dict(ref_strength="0.4",  ref_mp="1.0", sequential=False, waypoints="2"),
    "Journey":        dict(ref_strength="0.2",  ref_mp="1.0", sequential=False, waypoints="6"),
    "Feedback dream": dict(ref_strength="0.25", ref_mp="1.0", sequential=True,  waypoints="4"),
    # No reference at all — pure noise wandering through many seeds.
    "Free ride":      dict(ref_strength="0.0",  ref_mp="1.0", sequential=False, waypoints="8"),
}

# MiniMax H3 built-in presets — barebones image-only LoRA. Only the knobs the H3 trainer reads
# apply (rank/alpha/lr/epochs/save/seed/adaptive/optimizer/grad-accum/max-grad-norm/megapixels);
# the H3 base is always NF4 (no swap / fp8 / quant knobs). The first entry is applied on switch.
MINIMAX_BUILT_IN_PRESETS = {
    # LoKR factor 8 rather than standard LoRA: the same call Krea 2 landed on after measurement
    # (highest likeness recorded here, no skin sheen). Dim/alpha still ride along for when the
    # Network Type is flipped back to LoRA — the H3 trainer ignores them under LoKR.
    #
    # Static 1e-4, adaptive OFF. The adaptive watcher probes the LR UP on steady descent, and a
    # density sweep needs the LR held still or the runs aren't comparable.
    #
    # 0.5 MP rather than the 0.25 the other families default to: detail has to be present in the
    # training image before any schedule can teach it, and 0.5 still fits without block swap.
    #
    # 60% low-noise WITH mid-concentrated. The two controls answer different questions: the box
    # is how much training sits below sigma 0.5, the checkbox is where that mass actually lands.
    # At 60% the split is 58.8% of steps in sigma 0.3-0.7, 12.9% below 0.2, 5.0% above 0.8
    # (measured, 2M draws through sample_sigmas). The same 60% on a UNIFORM base would dump far
    # more into the near-clean end, where the latent is almost the finished image and the step
    # teaches little. Peter ran 70% + mid-concentrated well; 60 keeps slightly more mid-band and
    # gives back some sigma>0.8 mass, which is the only part of the schedule that lets the LoRA
    # touch composition at all. Both beat the 22% this shipped with.
    #
    # Note the mid-band fraction peaks around 50% and is nearly flat from 45-60, so it does NOT
    # discriminate on its own — it is a description of where the steps go, not a prediction of
    # likeness. 60 is Peter's call from real runs, with the arithmetic agreeing rather than
    # leading. 50% is worth trying: its shift is exactly 1.00, i.e. the plain logit-normal that
    # earlier runs recorded as overdriving adapters — on adamw8bit, before the optimizer was
    # understood.
    # LoRA, not LoKR (Peter, 10 Aug). The runs that produced the best likeness on this family
    # were standard LoRA at dim/alpha 16, and LoKR moves ~7-10x further per unit LR — which made
    # the same Learning Rate box mean two very different things depending on the Network Type
    # sitting above it. LoKR stays one dropdown away for anyone who wants it.
    "✨ MiniMax H3 (Lower LR - slower)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16,
        "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": 8,
        # Flat 1e-4 (Peter, 17 Aug). With the ramp off this IS the rate — and rank 16 wants
        # half of what the rank-8 Fast preset runs at (which keeps its flat 2e-4).
        "LEARNING_RATE": 1e-4,
        # Ships OFF (Peter, 17 Aug — reversing 11 Aug): the slow build spent the early epochs
        # crawling and the flat 2e-4 runs have been the ones delivering. The ramp stays a
        # dropdown away for anyone who wants the held-ratio start.
        "MINIMAX_ADAPTER_RAMP": "Off",
        # Carried so "load Defaults" genuinely resets it. Multi Concept overrides this to Off
        # when it is on, and the command builder locks it there regardless.
        "MINIMAX_CAPTION_DROPOUT": "0.05 (default)",
        "MAX_TRAIN_EPOCHS": 60, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
        # adamw, NOT adamw8bit — the single biggest likeness change measured on H3 (2026-08-06).
        # Every other knob had been swept with likeness stuck around 40-50%; full-precision
        # optimizer state moved it night-and-day on the same dataset. The 8-bit optimizer stores
        # the second moment blockwise-quantized, and on this model that is evidently costing the
        # fine detail. Costs ~1.2 GB of fp32 state against a 21 GB resident base.
        "OPTIMIZER_TYPE": "adamw",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        # Back to 0.25 MP (Peter, 11 Aug). The 1.0 default lasted a day: it came from the theory
        # that 496px trains below H3's 768 short-edge canvas and must therefore starve detail —
        # the same out-of-distribution argument that was being made about previews at the time.
        # A day of real runs did not bear it out, and 0.25 is four times cheaper per step.
        # The canvas number is about what the model RENDERS; it turned out to say much less than
        # expected about what it can be TRAINED on.
        "DATASET_MEGAPIXELS": "0.25",
        "MINIMAX_LOWNOISE_PCT": "60", "MINIMAX_HIGHNOISE_LR_PCT": "100",
        # The experiment knobs all ship OFF, so the preset is the plain baseline every A/B is
        # measured against. Each of these was built to be TRIED, not to be on by default:
        #   blocks "all"      — no block-range restriction
        #   AdaLN False       — Peter's call from real runs; the reference trains it, we do not
        #   slow blocks ""    — one LR everywhere, no depth split
        #   distill False     — ordinary photo training, no r2v teacher (which also needs the
        #                       _teref cache built, so defaulting it on would break a fresh run)
        "MINIMAX_BLOCKS": "all", "MINIMAX_BASE_QUANT": MINIMAX_BASE_QUANT_OPTIONS[0],
        "MINIMAX_TRAIN_ADALN": False,
        # Optimised Likeness Learning ships ON: photos train the identity blocks (20-49) only,
        # clips train the full model. The one measured exception is style — the Style preset
        # turns it off (style needs the early blocks).
        "MINIMAX_LIKENESS_OPT": True,
        "MINIMAX_SLOW_BLOCKS": "", "MINIMAX_SLOW_LR_SCALE": "0.2",
        # The one experiment that graduated: the limiter ships ON. Validated on a real A/B
        # (8 Aug) — the last trained block always hogs 2-4x the median block's movement and
        # over-edits fine detail (distorted eyes); capping it fixed epoch-1 quality outright
        # Per-step clip and LR warmup are RETIRED (Peter, 10 Aug) and hidden — both were
        # after-the-fact answers to epoch-1 overshoot, which the Adapter-relative LR ramp
        # removes at its root by holding the step/size ratio steady. Kept as keys so old
        # presets and saved configs still load; the command builder never emits them.
        "MINIMAX_BLOCK_LIMIT": "Off",
        "MINIMAX_LR_WARMUP": "Off",
        # EMA stays available but OFF by default: it saves the smoothed centre of the stride
        # zigzag instead of a raw corner of it, which is worth having when a run is pushed hard
        # and unnecessary when it is not.
        "MINIMAX_EMA": "Off",
        "MINIMAX_DISTILL": False,
    },
    # (An earlier "Fast (adaptive LR)" preset was retired 9 Aug when its recipe drifted from the
    # main one. The Fast preset below avoids that by DERIVING from Defaults — see the comment.)
}

# --- MiniMax H3 Fast ------------------------------------------------------------------------
# Peter's rank-8 recipe (11 Aug): the run that reached full likeness in ~400 steps and came out
# noticeably more flexible than the rank-16 ones. Low rank cannot memorise backgrounds and
# framing in the time available, so it is forced to encode the subject instead.
#
# Built by SPREADING Defaults rather than copying it, so a change to the shipped recipe cannot
# silently leave this one behind — which is exactly how the last Fast preset earned its
# retirement. Keyed off the first entry rather than the title, because the title has been
# renamed twice already.
_MM_DEFAULTS_KEY = next(iter(MINIMAX_BUILT_IN_PRESETS))
MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)"] = {
    **MINIMAX_BUILT_IN_PRESETS[_MM_DEFAULTS_KEY],
    "NETWORK_DIM": 8, "NETWORK_ALPHA": 8,
    "MAX_TRAIN_EPOCHS": 50,
    "LEARNING_RATE": 2e-4,
    # Flat, not ramped. The ramp exists to stop a full-size stride landing on a near-zero
    # adapter; at rank 8 there are half as many directions to move, and the measured run that
    # this preset reproduces had no ramp at all.
    "MINIMAX_ADAPTER_RAMP": "Off",
    "ADAPTIVE_LR": False,
}

# --- MiniMax H3 Style -------------------------------------------------------------------------
# The 19 Aug style ablation (Repair Studio, same instrument that found the likeness set): a
# style LoRA's deltas matter across nearly the WHOLE model — droppable only at 4-5 (the dead
# band / audio-embedder pipe) and 48-49 (subject-specific last-mile work: load-bearing for
# likeness and voice, silent for style). Hence 0-3, 6-47. LR matches Fast's 2e-4 — Peter's
# real style runs (20 Aug) found the halved 1e-4 unnecessary; drop it manually for an extra-
# gentle run if a style ever fries.
MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Style (LoRA 8)"] = {
    **MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)"],
    "LEARNING_RATE": 2e-4,
    "MINIMAX_BLOCKS": "0-3, 6-47",
    # MUST be off here: style measurably needs the early blocks the likeness mask freezes, and
    # with it on the blocks spec above would be ignored outright.
    "MINIMAX_LIKENESS_OPT": False,
}

# Fast is the shipped default (Peter, 22 Aug): the FIRST entry is what a family switch and a
# fresh start apply, and the rank-8 Fast recipe is where most datasets should begin. The
# rank-16 recipe stays one dropdown away, flagged in the GUI as the larger-dataset choice.
# (Re-inserting an existing key keeps its first position, so Fast leads and nothing else moves.)
_MM_FAST_KEY = "✨ MiniMax H3 Fast (LoRA 8, 50 epochs)"
MINIMAX_BUILT_IN_PRESETS = {
    _MM_FAST_KEY: MINIMAX_BUILT_IN_PRESETS[_MM_FAST_KEY],
    **MINIMAX_BUILT_IN_PRESETS,
}
