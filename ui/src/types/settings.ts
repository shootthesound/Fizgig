// =============================================================================
// Fizgig Settings & Preferences Types — EXACT 1:1 mapping from jj.py
// Every string, number, key name, and label is character-identical to the Python.
// These values are the backend's contract — do NOT modify.
// =============================================================================

export interface FizgigSettings {
  // Initialize settings with default values, including conversion settings
  // Klein 9B Base is the only supported architecture. A few legacy keys
  // (CLIP_MODEL / T5_MODEL / MODEL_TYPE) remain as empty defaults so dead
  // code paths gated behind `config["uses_*"]` flags don't KeyError.
  ARCHITECTURE: string;
  DATASET_CONFIG: string;
  // Model paths resolved at runtime from prefs_vars — blank fallback.
  VAE_MODEL: string;
  CLIP_MODEL: string;
  T5_MODEL: string;
  TEXT_ENCODER: string;
  DIT_MODEL: string;
  LORA_OUTPUT_DIR: string;
  LORA_NAME: string;
  MODEL_TYPE: string;
  LEARNING_RATE: number;
  LORA_LR_RATIO: number;
  NETWORK_DIM: number;
  NETWORK_ALPHA: number;
  // Krea 2 only (Klein hides it and trains standard). LoRA default; LoKR is the
  // quality pick (validated 31 Jul: highest likeness measured here, no skin sheen).
  NETWORK_TYPE: string;
  LOKR_FACTOR: number;
  MAX_TRAIN_EPOCHS: number;
  SAVE_EVERY_N_EPOCHS: number;
  SEED: number;
  BLOCKS_SWAP: string; // Klein valid range 0-16; "auto" detects from GPU
  // MiniMax H3 only. Percent of steps trained below sigma 0.5 (H3's own default works out at ~7.7%).
  // 60 + mid-concentrated matches the MiniMax preset, which is what a switch to that
  // family applies anyway; these are the values the widgets are BUILT with, so they are
  // what shows before any preset lands. Both keys are MiniMax-only — no other family
  // reads them. (OPTIMIZER_TYPE below is deliberately NOT changed to match the preset:
  // it is shared with Klein and Krea 2, and the MiniMax preset supplies adamw on switch.)
  MINIMAX_LOWNOISE_PCT: string;
  // What the steps ABOVE sigma 0.5 do to the LR, as a percentage. 100 = unchanged, which
  // is every run before this existed.
  MINIMAX_HIGHNOISE_LR_PCT: string;
  MINIMAX_BLOCKS: string;
  MINIMAX_BASE_QUANT: string;
  MINIMAX_BLOCK_LIMIT: string;
  MINIMAX_LR_WARMUP: string;
  MINIMAX_EMA: string;
  MINIMAX_ADAPTER_RAMP: string;
  MINIMAX_CAPTION_DROPOUT: string;
  // OFF by default (Peter's call from real runs). The reference trains AdaLN on the
  // pruned checkpoint, but AdaLN is a pure function of the timestep — adaln_proj(t_emb)
  // and nothing else — so its adapters cannot tell one subject from another, and on the
  // pruned build they were taking ~45% of all weight movement to do it.
  MINIMAX_TRAIN_ADALN: boolean;
  // Optimised Likeness Learning — photo steps train blocks 20-49 only, clips train
  // everything. On by default: it is the measured best recipe for the character/voice
  // work H3 is for. The Style preset turns it OFF (style needs the early blocks).
  MINIMAX_LIKENESS_OPT: boolean;
  MINIMAX_TRAINING_ADAPTER: boolean;
  MINIMAX_TREAD: boolean;         // clip steps route half their video tokens (7 Sep)
  MINIMAX_CLIP_STILL: boolean;    // each clip's sharpest face frame trains as a photo
  MINIMAX_DISTILL: boolean;      // off = ordinary training
  // Which H3 base ordinary training runs on ("fl2va"/"ref2va"). NOT in any preset —
  // the Training Base dropdown's var lives outside self.entries by design.
  MINIMAX_TRAIN_BASE: string;
  MINIMAX_DISTILL_WEIGHT: string;
  MINIMAX_DISTILL_REFS: string;
  MINIMAX_DISTILL_PHASE1: string;
  MINIMAX_SLOW_BLOCKS: string;     // blank = one LR everywhere
  MINIMAX_SLOW_LR_SCALE: string;
  // Per-category retirement — MIXED datasets only
  MIXED_STOP_CATEGORY: string;
  MIXED_STOP_EPOCH: string;
  MIXED_STOP_MODE: string;
  RESUME_TRAINING: string;
  OPTIMIZER_TYPE: string;
  OPTIMIZER_ARGS: string;
  GRADIENT_ACCUMULATION: number;  // Effective batch size = batch × this
  MAX_GRAD_NORM: number;  // Gradient clipping (0 to disable)
  NETWORK_DROPOUT: number;  // LoRA dropout for regularization
  ATTENTION_MECHANISM: string;  // Default attention (was "none", causing duplicate flags)
  LOGGING_DIR: string;
  LOG_WITH: string;
  LOG_PREFIX: string;
  IMG_IN_TXT_IN_OFFLOADING: boolean;
  LR_SCHEDULER: string;
  LR_WARMUP_STEPS: string;
  LR_DECAY_STEPS: string;
  ADAPTIVE_LR: boolean;
  ADAPTIVE_LR_MIN: string;
  ADAPTIVE_LR_MAX: string;
  CONTEXT_LORA_PATH: string;
  CONTEXT_LORA_STRENGTH: string;
  TIMESTEP_SAMPLING: string;
  DISCRETE_FLOW_SHIFT: string;
  SIGMOID_SCALE: string;
  MIN_TIMESTEP: string;
  MAX_TIMESTEP: string;
  PRESERVE_DISTRIBUTION: boolean;
  WEIGHTING_SCHEME: string;
  LOGIT_MEAN: string;
  LOGIT_STD: string;
  MODE_SCALE: string;
  METADATA_TITLE: string;
  METADATA_AUTHOR: string;
  METADATA_DESCRIPTION: string;
  METADATA_LICENSE: string;
  METADATA_TAGS: string;
  METADATA_TRIGGER_PHRASE: string;
  METADATA_THUMBNAIL: string;
  FP8: boolean;  // Default FP8 setting (--fp8_base)
  SCALED: boolean;  // Default Scaled setting (--fp8_scaled, recommended with fp8_base)
  QUANT_4BIT: boolean;  // 4-bit NF4 base (low-VRAM); supersedes fp8 when on
  COMPILE_BLOCKS: string;  // torch.compile the DiT blocks (krea2): auto | on | off
  GRADIENT_CHECKPOINTING: boolean;  // ON by default — recompute activations to fit 9B on most cards
  FP8_TEXT_ENCODER: boolean;  // FP8 for text encoder (T5/LLM)
  // Resumable state dirs. Pause/Resume writes state regardless — these only govern the
  // automatic saves. Keep-N matters: a state is LoRA + optimizer (~470 MB at rank 32).
  SAVE_STATE: boolean;
  SAVE_STATE_ON_TRAIN_END: boolean;
  KEEP_LAST_N_STATES: number;
  KREA2_LOSS_WATCH: boolean;   // per-image loss tracking + stuck-image detection (krea2)
  KREA2_PER_IMAGE_LR: boolean;  // per-image adaptive LR (throttle stuck images) — experimental
  KREA2_AUTO_RECAPTION: boolean;  // Qwen3-VL rewrites stuck images' captions mid-run — experimental
  // Sample generation settings
  SAMPLE_ENABLED: boolean;
  SAMPLE_PROMPT: string;
  SAMPLE_WIDTH: number;
  SAMPLE_HEIGHT: number;
  SAMPLE_STEPS: number;
  SAMPLE_SEED: number;
  SAMPLE_EVERY_N_EPOCHS: number;
  SAMPLE_EVERY_N_STEPS: number;
  SAMPLE_AT_FIRST: boolean;
  CACHE_SAMPLE_MODEL: string;  // keep Distilled sample model in RAM between epochs
  SAMPLE_FLOW_SHIFT: string;
  SAMPLE_NEGATIVE: string;
  SAMPLE_CFG_SCALE: number;
  SAMPLE_FRAMES: string;
  // MiniMax Turbo previews (used only when the Turbo LoRA is set in Preferences)
  MINIMAX_TURBO_STEPS: number;
  MINIMAX_TURBO_STRENGTH: number;
  // Florence captioning settings
  CAPTION_TRIGGER_WORD: string;
  CAPTION_MODEL: string;
  CAPTION_TASK: string;
  CAPTION_MAX_TOKENS: number;
}

export interface Preferences {
  // Model paths (absolute — point to external model downloads).
  base_dit: string;
  distilled_dit: string;
  vae: string;
  text_encoder: string;
  krea2_raw_dit: string;
  krea2_turbo_dit: string;
  krea2_vae: string;
  krea2_text_encoder: string;
  krea2_turbo_lora: string;
  minimax_dit: string;
  minimax_ref_dit: string;
  minimax_text_encoder: string;
  minimax_vae: string;
  minimax_audio_vae: string;
  minimax_turbo_lora: string;
  minimax_training_adapter: string;
  minimax_ref_training_adapter: string;
  lora_output_dir: string;
  profiles_dir: string;
  cache_dir: string;
  input_lora_dir: string;
  input_ref_dir: string;
  input_dataset_dir: string;
  runpod_stop_when_done: string;
  runpod_api_key: string;
  inference_blocks_to_swap: string;
  inference_int8: string;
  cuda_device: string;
  cloud_provider?: string;
  modal_gpu?: string;
}

export interface LastUsed {
  image_prep_source: string;
  image_folder: string;  // Start tab: training image folder (shared with Captions)
  image_folder2: string;  // Multi Concept (MiniMax): second subject, TRAINING ONLY
  caption_trigger: string;
  dataset_cache_dir: string;
  sample_prompt: string;
  caption_model?: string;
  caption_task?: string;
  caption_max_tokens?: string;
  caption_tasks?: Record<string, string>;
  prep_mode?: string;
  prep_replace_originals?: boolean;
  prep_megapixels?: string;
  architecture?: string;
  lora_output_dir?: string;
  status_bar_visible?: boolean;
}

export interface DatasetConfig {
  dataset_name: string;
  dataset_type: string;
  dataset_video_dir: string;
  dataset_cache_dir: string;
  dataset_caption_ext: string;
  dataset_jsonl_file: string;
  dataset_megapixels: string;
  dataset_batch_size: string;
  dataset_num_repeats: string;
  dataset_enable_bucket: boolean;
  dataset_no_upscale: boolean;
  dataset_target_frames: string;
  dataset_frame_extraction: string;
  dataset_source_fps: string;
}
