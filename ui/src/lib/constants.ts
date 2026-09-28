// =============================================================================
// Fizgig Constants — EXACT 1:1 mapping from jj.py
// Every string, number, key name, and label is character-identical to the Python.
// These values are the backend's contract — do NOT modify.
// =============================================================================

// Refined Color Palette (Fizgig Visual Style Guide)
// Lines 55-94 of jj.py
export const COLORS = {
  bg_deep: "#1E2530",        // Main window background
  bg_surface: "#252D38",     // Cards, panels, inputs
  bg_hover: "#2A3542",       // Hover states
  bg_header: "#1A2028",      // Collapsible section headers

  text_primary: "#F0F4F8",   // Main text
  text_secondary: "#8A9BAE", // Labels
  text_explain: "#C3CDD9",
  text_muted: "#5A6B7E",

  accent: "#3B82F6",         // Primary actions, links
  accent_hover: "#60A5FA",   // Accent hover
  accent_subtle: "#1E3A5F",  // Accent backgrounds

  queue_blue: "#93C5FD",
  queue_blue_hover: "#BFDBFE",

  border: "#3A4555",         // Borders, dividers
  border_focus: "#3B82F6",   // Focus rings

  scrollbar_thumb: "#3B82F6",
  scrollbar_thumb_hover: "#60A5FA",

  success: "#10B981",        // Success states
  warning: "#F59E0B",        // Warnings
  error: "#EF4444",          // Errors
} as const;

// Typography — Lines 97-98
export const FONT_FAMILY = "Segoe UI";
export const FONT_MONO = "Consolas";

// Legacy color constants — Lines 100-108
export const BG_COLOR = COLORS.bg_deep;
export const FG_COLOR = COLORS.text_primary;
export const ACCENT_COLOR = COLORS.accent;
export const ENTRY_BG = COLORS.bg_surface;
export const BUTTON_ACTIVE = COLORS.bg_hover;
export const BORDER_COLOR = COLORS.border;
export const ACTIVE_ENTRY_BG = "white";
export const ACTIVE_ENTRY_FG = "black";

// Architecture configurations — Lines 297-414
export interface ArchitectureConfig {
  train_script: string;
  cache_latents_script: string;
  cache_text_script: string;
  network_module: string;
  use_fizgig_venv: boolean;
  timestep_sampling: string;
  discrete_flow_shift: number | null;
  weighting_scheme: string;
  blocks_swap_max: number;
  fp8_text_encoder_flag: string | null;
  uses_clip: boolean;
  uses_t5: boolean;
  uses_text_encoder: boolean;
  uses_model_type: boolean;
  uses_model_version: boolean;
  model_version: string;
  vae_label: string;
  text_encoder_label: string;
  is_distilled: boolean;
  supports_weighting_scheme: boolean;
  supports_discrete_flow_shift: boolean;
  supports_samples: boolean;
  sample_cfg_default: number;
  sample_flow_shift_default: number | null;
  sample_steps_default: number;
  sample_width_default: number;
  sample_height_default: number;
  lora_name_suffix: string;
  is_krea2?: boolean;
  is_minimax?: boolean;
  sample_is_distilled?: boolean;
  sample_cfg_fixed?: boolean;
}

export const ARCHITECTURES: Record<string, ArchitectureConfig> = {
  "Flux 2 Klein Base 9B": {
    train_script: "FizgigIndependent/src/fizgig/scripts/train.py",
    cache_latents_script: "FizgigIndependent/src/fizgig/scripts/cache_latents.py",
    cache_text_script: "FizgigIndependent/src/fizgig/scripts/cache_text.py",
    network_module: "fizgig.networks.lora_klein",
    use_fizgig_venv: true,
    timestep_sampling: "flux2_shift",
    discrete_flow_shift: null,
    weighting_scheme: "none",
    blocks_swap_max: 16,
    fp8_text_encoder_flag: "--fp8_text_encoder",
    uses_clip: false,
    uses_t5: false,
    uses_text_encoder: true,
    uses_model_type: false,
    uses_model_version: true,
    model_version: "klein-base-9b",
    vae_label: "AE Model (ae.safetensors from FLUX.2-dev — NOT the Diffusers subfolder VAE)",
    text_encoder_label: "Text Encoder (Qwen3-8B)",
    is_distilled: false,
    supports_weighting_scheme: false,
    supports_discrete_flow_shift: false,
    supports_samples: true,
    sample_cfg_default: 4.5,
    sample_flow_shift_default: null,
    sample_steps_default: 40,
    sample_width_default: 768,
    sample_height_default: 768,
    lora_name_suffix: "k9b",
  },
  "Krea 2": {
    train_script: "src/fizgig/scripts/krea2_train.py",
    cache_latents_script: "src/fizgig/scripts/krea2_cache_latents.py",
    cache_text_script: "src/fizgig/scripts/krea2_cache_text.py",
    is_krea2: true,
    network_module: "fizgig.networks.lora_klein",
    use_fizgig_venv: true,
    timestep_sampling: "shift",
    discrete_flow_shift: 2.5,
    weighting_scheme: "none",
    blocks_swap_max: 26,
    fp8_text_encoder_flag: null,
    uses_clip: false,
    uses_t5: false,
    uses_text_encoder: true,
    uses_model_type: false,
    uses_model_version: false,
    model_version: "krea-2",
    vae_label: "Qwen-Image VAE",
    text_encoder_label: "Qwen3-VL-4B",
    is_distilled: false,
    supports_weighting_scheme: false,
    supports_discrete_flow_shift: false,
    supports_samples: true,
    sample_cfg_default: 1.0,
    sample_flow_shift_default: null,
    sample_steps_default: 8,
    sample_width_default: 1024,
    sample_height_default: 1024,
    lora_name_suffix: "krea2",
  },
  "MiniMax H3": {
    train_script: "src/fizgig/scripts/minimax_train.py",
    cache_latents_script: "src/fizgig/scripts/minimax_cache_latents.py",
    cache_text_script: "src/fizgig/scripts/minimax_cache_text.py",
    is_minimax: true,
    network_module: "fizgig.networks.lora_klein",
    use_fizgig_venv: true,
    timestep_sampling: "shift",
    discrete_flow_shift: 12.0,
    weighting_scheme: "none",
    blocks_swap_max: 40,
    fp8_text_encoder_flag: null,
    uses_clip: false,
    uses_t5: false,
    uses_text_encoder: true,
    uses_model_type: false,
    uses_model_version: false,
    model_version: "minimax-h3",
    vae_label: "MiniMax H3 Video VAE",
    text_encoder_label: "Qwen3-VL-32B",
    is_distilled: false,
    supports_weighting_scheme: false,
    supports_discrete_flow_shift: false,
    supports_samples: true,
    sample_cfg_default: 1.0,
    sample_flow_shift_default: null,
    sample_is_distilled: true,
    sample_cfg_fixed: true,
    sample_steps_default: 20,
    sample_width_default: 768,
    sample_height_default: 768,
    lora_name_suffix: "mmh3",
  },
};

// Arch aliases — Lines 420-423
export const _ARCH_ALIASES: Record<string, string> = {
  "MiniMax H3 (experimental)": "MiniMax H3",
  "Krea 2 (experimental)": "Krea 2",
};
for (const [old, newName] of Object.entries(_ARCH_ALIASES)) {
  ARCHITECTURES[old] = ARCHITECTURES[newName];
}

export const ARCHITECTURE_LIST = Object.keys(ARCHITECTURES).filter(
  k => !(k in _ARCH_ALIASES)
);

export function _canon_arch(name: string): string {
  return _ARCH_ALIASES[name] ?? name;
}

export const LORA_NAME_SUFFIXES = new Set(
  Object.values(ARCHITECTURES).map(c => c.lora_name_suffix).filter(Boolean)
);

// Line 444
export const SAMPLE_RESOLUTIONS = ["512", "640", "768", "1024", "1280", "1536"];

// Lines 476-527
export const MINIMAX_LOWNOISE_SIGMA = 0.5;

export function minimax_lownoise_to_shift(pct: string | number): number | null {
  try {
    const p = parseFloat(String(pct).trim().replace(/%$/, '')) / 100.0;
    if (!(p > 0.0 && p < 1.0)) return null;
    return (1.0 - p) / p;
  } catch { return null; }
}

export function minimax_highnoise_lr(pct: string | number): number | null {
  try {
    const p = parseFloat(String(pct).trim().replace(/%$/, '')) / 100.0;
    if (!(p >= 0.0 && p <= 1.0)) return null;
    return p;
  } catch { return null; }
}

export function minimax_shift_to_lownoise(shift: number): number | null {
  if (shift <= 0) return null;
  return 100.0 / (1.0 + shift);
}

// Lines 592-607
export const MINIMAX_STRUCTURE_OPTIONS: Record<string, [number, number] | null> = {
  "Likeness and Style — 60% clean-end": [60, 100],
  "Model default, movement — 8% clean-end": [8, 100],
  "Custom": null,
};
export const MINIMAX_STRUCTURE_DESC: Record<string, string> = {
  "Likeness and Style — 60% clean-end":
    "Most of the run on nearly-clean images — the tuned default for stills. See the MiniMax section of the README.",
  "Model default, movement — 8% clean-end":
    "The reference trainer's schedule, weighted to movement and composition. See the MiniMax section of the README.",
  "Custom":
    "Type your own clean-end share. See the MiniMax section of the README.",
};
export const MINIMAX_STRUCTURE_DEFAULT = "Likeness and Style — 60% clean-end";

// Lines 609-615
export const MINIMAX_BLOCK_OPTIONS = [
  "all · every block (50 of 50)",
  "10-49 · skip the first 10",
  "14-37 · middle band",
  "25-49 · back half",
  "0-24 · front half",
];

export const MINIMAX_NUM_BLOCKS = 50;
export const MINIMAX_LIKENESS_BLOCKS = "20-49";
export const MINIMAX_AUDIO_BLOCKS = "34-49";

// Lines 635-640
export const MINIMAX_BASE_QUANT_OPTIONS = [
  "Auto (recommended)",
  "int8 · most accurate, needs ~30 GB free",
  "4-bit · fits smaller cards",
  "4-bit HQQ · lower error than 4-bit, slower",
];

export function minimax_base_quant(raw: string): string {
  const s = (raw || "").split("·")[0].trim().toLowerCase();
  if (s.startsWith("int8")) return "int8";
  if (s.includes("hqq")) return "hqq";
  if (s.startsWith("4-bit") || s.startsWith("nf4")) return "nf4";
  return "auto";
}

export function minimax_block_spec(raw: string): string {
  return (raw || "").split("·")[0].trim() || "all";
}

// Lines 665-677
export const REPAIR_H3_BASE_OPTIONS = [
  "Auto (by free VRAM)",
  "Stream blocks (exact int8, room for big clips)",
  "NF4 (smallest, 9.5% base error)",
] as const;

export const MINIMAX_TRAIN_BASE_OPTIONS = [
  "First/last frame (fl2va) — standard",
  "Reference (ref2va)",
];

export function minimax_train_base(raw: string): string {
  return (raw || "").toLowerCase().includes("ref2va") ? "ref2va" : "fl2va";
}

// Optimizer types — Line 1884
export const OPTIMIZER_TYPES = ["adamw", "adamw8bit", "bitsandbytes.optim.AdEMAMix8bit", "bitsandbytes.optim.PagedAdEMAMix8bit"];
export const KREA2_OPTIMIZER_TYPES_DEFAULT = ["adamw8bit", "adamw"];
