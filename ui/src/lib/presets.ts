// =============================================================================
// Fizgig Presets — EXACT 1:1 mapping from jj.py lines 707-850
// Every preset name, key, and value is character-identical to the Python.
// =============================================================================

// Built-in presets — always available in the Load Preset dropdown, prefixed with ✨ to distinguish
// from user-saved presets. Defined in code so they ship with the app and can't be deleted accidentally.
// Tune these as empirical findings accumulate.
// Lines 707-757 of jj.py
export const BUILT_IN_PRESETS: Record<string, Record<string, any>> = {
  "✨ Old Reliable (rank 16, full model, single subject)": {
    NETWORK_DIM: 16, NETWORK_ALPHA: 16, LEARNING_RATE: 1e-4,
    MAX_TRAIN_EPOCHS: 55, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "1e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Full Model", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)": {
    NETWORK_DIM: 8, NETWORK_ALPHA: 8, LEARNING_RATE: 1e-4,
    MAX_TRAIN_EPOCHS: 55, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "1e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Full Model", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Identity (rank 4, single subject)": {
    NETWORK_DIM: 4, NETWORK_ALPHA: 4, LEARNING_RATE: 4e-4,
    MAX_TRAIN_EPOCHS: 15, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "2e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Identity", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Identity (rank 8, harder dataset)": {
    NETWORK_DIM: 8, NETWORK_ALPHA: 8, LEARNING_RATE: 4e-4,
    MAX_TRAIN_EPOCHS: 20, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "2e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Identity", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Multi-Character (rank 16, multi character or concept)": {
    NETWORK_DIM: 16, NETWORK_ALPHA: 16, LEARNING_RATE: 2e-4,
    MAX_TRAIN_EPOCHS: 50, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "1e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Identity", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Style (late timesteps)": {
    NETWORK_DIM: 4, NETWORK_ALPHA: 4, LEARNING_RATE: 4e-4,
    MAX_TRAIN_EPOCHS: 15, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "1e-5", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Style", MIN_TIMESTEP: "0", MAX_TIMESTEP: "400",
    OPTIMIZER_TYPE: "adamw8bit",
  },
  "✨ Style+Composition (all timesteps)": {
    NETWORK_DIM: 4, NETWORK_ALPHA: 4, LEARNING_RATE: 4e-4,
    MAX_TRAIN_EPOCHS: 15, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "1e-5", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Style+Composition", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
  },
};

// Built-in presets for Krea 2. The Klein block-targeting / adaptive-LR / timestep presets
// above don't apply (Krea 2 trains all linears, no block map yet, no adaptive LR), so Krea 2
// ships a single sensible-defaults entry. Users can still save their own via Save Preset —
// those land in the per-architecture preset folder and appear alongside this one.
// Lines 763-850 of jj.py
export const KREA2_BUILT_IN_PRESETS: Record<string, Record<string, any>> = {
  "✨ Krea 2 Defaults (rank 32, full model)": {
    NETWORK_DIM: 32, NETWORK_ALPHA: 32, NETWORK_TYPE: "LoRA (standard)",
    LEARNING_RATE: 1e-4,
    MAX_TRAIN_EPOCHS: 30, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: false, ADAPTIVE_LR_MIN: "1e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Full Model", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
    GRADIENT_ACCUMULATION: 1, MAX_GRAD_NORM: 1.0,
    DATASET_MEGAPIXELS: "0.25",
    // Memory settings all auto — each resolves from the actual GPU at launch.
    // BLOCKS_SWAP must be the combobox's exact label: _apply_preset_values matches a
    // preset value against the offered options on its first token, case-sensitively,
    // so a bare "auto" would not select "Auto (detect from GPU)".
    BLOCKS_SWAP: "Auto (detect from GPU)",
    QUANT_4BIT_MODE: "auto", COMPILE_BLOCKS: "Auto",
    // Per-image loss watch: detection + the LR throttle on, the two interventions that
    // rewrite captions or pre-judge images left off — those want a deliberate choice.
    KREA2_LOSS_WATCH: true, KREA2_PER_IMAGE_LR: true,
    KREA2_AUTO_RECAPTION: false, KREA2_WARMUP_LOOK: false,
  },
  // Rank 8 + Adaptive LR at an aggressive floor: fewer epochs to a usable LoRA. Everything
  // else identical to Krea 2 Defaults (which stays the preset applied on family switch).
  "✨ Krea 2 Ultra Fast (rank 8, adaptive LR)": {
    NETWORK_DIM: 8, NETWORK_ALPHA: 8, NETWORK_TYPE: "LoRA (standard)",
    LEARNING_RATE: 1e-4,
    MAX_TRAIN_EPOCHS: 20, SAVE_EVERY_N_EPOCHS: 1, SEED: 42,
    ADAPTIVE_LR: true, ADAPTIVE_LR_MIN: "2e-4", ADAPTIVE_LR_MAX: "4e-4",
    TARGET_LAYERS: "Full Model", MIN_TIMESTEP: "", MAX_TIMESTEP: "",
    OPTIMIZER_TYPE: "adamw8bit",
    GRADIENT_ACCUMULATION: 1, MAX_GRAD_NORM: 1.0,
    DATASET_MEGAPIXELS: "0.25",
    BLOCKS_SWAP: "Auto (detect from GPU)",
    QUANT_4BIT_MODE: "auto", COMPILE_BLOCKS: "Auto",
    KREA2_LOSS_WATCH: true, KREA2_PER_IMAGE_LR: true,
    KREA2_AUTO_RECAPTION: false, KREA2_WARMUP_LOOK: false,
  },
};
