// =============================================================================
// Fizgig MiniMax H3 Presets — EXACT 1:1 mapping from jj.py lines 848-1001
// =============================================================================

import { MINIMAX_BASE_QUANT_OPTIONS } from './constants';

export const MINIMAX_BUILT_IN_PRESETS: Record<string, Record<string, any>> = {};

// LoRA Royale seed-travel presets — recipes of the *mechanics* knobs (reference /
// anchor / journey / identity-lock), NOT a prompt system (seed travel uses the Setup
// prompt). Reference is kept light (0.1–0.4) so the seeds can actually travel.
// Lines 839-846 of jj.py
export const SEED_TRAVEL_PRESETS: Record<string, Record<string, any>> = {
  // Each varies a clearly different knob:
  "Hold subject":   { ref_strength: "0.4",  ref_mp: "1.0", sequential: false, waypoints: "2" },
  "Journey":        { ref_strength: "0.2",  ref_mp: "1.0", sequential: false, waypoints: "6" },
  "Feedback dream": { ref_strength: "0.25", ref_mp: "1.0", sequential: true,  waypoints: "4" },
  // No reference at all — pure noise wandering through many seeds.
  "Free ride":      { ref_strength: "0.0",  ref_mp: "1.0", sequential: false, waypoints: "8" },
};

// MiniMax H3 built-in presets — barebones image-only LoRA. Only the knobs the H3 trainer reads
// apply (rank/alpha/lr/epochs/save/seed/adaptive/optimizer/grad-accum/max-grad-norm/megapixels);
// the H3 base is always NF4 (no swap / fp8 / quant knobs). The first entry is applied on switch.
// Lines 851-951 of jj.py

// ✨ MiniMax H3 (Lower LR - slower)
const _MM_DEFAULTS: Record<string, any> = {
  NETWORK_DIM: 16, NETWORK_ALPHA: 16,
  NETWORK_TYPE: "LoRA (standard)", LOKR_FACTOR: 8,
  // Flat 1e-4 (Peter, 17 Aug). With the ramp off this IS the rate — and rank 16 wants
  // half of what the rank-8 Fast preset runs at (which keeps its flat 2e-4).
  LEARNING_RATE: 1e-4,
  // Ships OFF (Peter, 17 Aug — reversing 11 Aug): the slow build spent the early epochs
  // crawling and the flat 2e-4 runs have been the ones delivering. The ramp stays a
  // dropdown away for anyone who wants the held-ratio start.
  MINIMAX_ADAPTER_RAMP: "Off",
  // Carried so "load Defaults" genuinely resets it. Multi Concept overrides this to Off
  // when it is on, and the command builder locks it there regardless.
  MINIMAX_CAPTION_DROPOUT: "0.05 (default)",
  MAX_TRAIN_EPOCHS: 60, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
  ADAPTIVE_LR: false, ADAPTIVE_LR_MIN: "1e-5", ADAPTIVE_LR_MAX: "4e-4",
  // adamw, NOT adamw8bit — the single biggest likeness change measured on H3 (2026-08-06).
  // Every other knob had been swept with likeness stuck around 40-50%; full-precision
  // optimizer state moved it night-and-day on the same dataset. The 8-bit optimizer stores
  // the second moment blockwise-quantized, and on this model that is evidently costing the
  // fine detail. Costs ~1.2 GB of fp32 state against a 21 GB resident base.
  OPTIMIZER_TYPE: "adamw",
  GRADIENT_ACCUMULATION: 1, MAX_GRAD_NORM: 1.0,
  // Back to 0.25 MP (Peter, 11 Aug). The 1.0 default lasted a day: it came from the theory
  // that 496px trains below H3's 768 short-edge canvas and must therefore starve detail —
  // the same out-of-distribution argument that was being made about previews at the time.
  // A day of real runs did not bear it out, and 0.25 is four times cheaper per step.
  // The canvas number is about what the model RENDERS; it turned out to say much less than
  // expected about what it can be TRAINED on.
  DATASET_MEGAPIXELS: "0.25",
  MINIMAX_LOWNOISE_PCT: "60", MINIMAX_HIGHNOISE_LR_PCT: "100",
  // The experiment knobs all ship OFF, so the preset is the plain baseline every A/B is
  // measured against. Each of these was built to be TRIED, not to be on by default:
  //   blocks "all"      — no block-range restriction
  //   AdaLN False       — Peter's call from real runs; the reference trains it, we do not
  //   slow blocks ""    — one LR everywhere, no depth split
  //   distill False     — ordinary photo training, no r2v teacher (which also needs the
  //                       _teref cache built, so defaulting it on would break a fresh run)
  MINIMAX_BLOCKS: "all", MINIMAX_BASE_QUANT: MINIMAX_BASE_QUANT_OPTIONS[0],
  MINIMAX_TRAIN_ADALN: false,
  // Optimised Likeness Learning ships ON: photos train the identity blocks (20-49) only,
  // clips train the full model. The one measured exception is style — the Style preset
  // turns it off (style needs the early blocks).
  MINIMAX_LIKENESS_OPT: true,
  // Training adapter ships ON (Peter, 2 Sep): measured on the same dataset/seed it hit
  // 50% likeness seven epochs sooner and peaked higher (61 vs 57). Every H3 preset
  // inherits this — Style included, the adapter is about the base, not the blocks.
  MINIMAX_TRAINING_ADAPTER: true,
  // TREAD token routing ships ON (Peter, 7 Sep, after his A/B): clip steps route half
  // their video tokens around blocks 2-46; photos and clip stills always run in full.
  MINIMAX_TREAD: true,
  // Each clip's sharpest face frame trains as a photo too (picked at cache time). ON.
  MINIMAX_CLIP_STILL: true,
  MINIMAX_SLOW_BLOCKS: "", MINIMAX_SLOW_LR_SCALE: "0.2",
  // The one experiment that graduated: the limiter ships ON. Validated on a real A/B
  // (8 Aug) — the last trained block always hogs 2-4x the median block's movement and
  // over-edits fine detail (distorted eyes); capping it fixed epoch-1 quality outright
  MINIMAX_BLOCK_LIMIT: "Off",
  MINIMAX_LR_WARMUP: "Off",
  // EMA stays available but OFF by default: it saves the smoothed centre of the stride
  // zigzag instead of a raw corner of it, which is worth having when a run is pushed hard
  // and unnecessary when it is not.
  MINIMAX_EMA: "Off",
  MINIMAX_DISTILL: false,
};

MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 (Lower LR - slower)"] = { ..._MM_DEFAULTS };

// --- MiniMax H3 Fast ---
// Peter's rank-8 recipe (11 Aug): the run that reached full likeness in ~400 steps and came out
// noticeably more flexible than the rank-16 ones. Low rank cannot memorise backgrounds and
// framing in the time available, so it is forced to encode the subject instead.
// Lines 963-973 of jj.py
MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)"] = {
  ..._MM_DEFAULTS,
  NETWORK_DIM: 8, NETWORK_ALPHA: 8,
  MAX_TRAIN_EPOCHS: 50,
  LEARNING_RATE: 2e-4,
  // Flat, not ramped. The ramp exists to stop a full-size stride landing on a near-zero
  // adapter; at rank 8 there are half as many directions to move, and the measured run that
  // this preset reproduces had no ramp at all.
  MINIMAX_ADAPTER_RAMP: "Off",
  ADAPTIVE_LR: false,
};

// --- MiniMax H3 Style ---
// The 19 Aug style ablation (Repair Studio, same instrument that found the likeness set): a
// style LoRA's deltas matter across nearly the WHOLE model — droppable only at 4-5 (the dead
// band / audio-embedder pipe) and 48-49 (subject-specific last-mile work: load-bearing for
// likeness and voice, silent for style). Hence 0-3, 6-47.
// Lines 982-991 of jj.py
MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Style (LoRA 8)"] = {
  ...MINIMAX_BUILT_IN_PRESETS["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)"],
  LEARNING_RATE: 2e-4,
  MINIMAX_BLOCKS: "0-3, 6-47",
  // MUST be off here: style measurably needs the early blocks the likeness mask freezes, and
  // with it on the blocks spec above would be ignored outright.
  MINIMAX_LIKENESS_OPT: false,
  // Style is about the look, not the face: no extra sharp-face stills from the clips.
  MINIMAX_CLIP_STILL: false,
};

// Fast is the shipped default (Peter, 22 Aug): the FIRST entry is what a family switch and a
// fresh start apply, and the rank-8 Fast recipe is where most datasets should begin.
// Lines 993-1001 of jj.py — re-order so Fast comes first
const _reordered: Record<string, Record<string, any>> = {};
const _MM_FAST_KEY = "✨ MiniMax H3 Fast (LoRA 8, 50 epochs)";
_reordered[_MM_FAST_KEY] = MINIMAX_BUILT_IN_PRESETS[_MM_FAST_KEY];
for (const [k, v] of Object.entries(MINIMAX_BUILT_IN_PRESETS)) {
  _reordered[k] = v;
}
// Replace the export
Object.keys(MINIMAX_BUILT_IN_PRESETS).forEach(k => delete MINIMAX_BUILT_IN_PRESETS[k]);
Object.assign(MINIMAX_BUILT_IN_PRESETS, _reordered);
