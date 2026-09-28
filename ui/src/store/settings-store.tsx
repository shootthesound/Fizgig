'use client';

import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { FizgigSettings, Preferences, LastUsed, DatasetConfig } from '../types/settings';

export interface AppState {
  settings: FizgigSettings;
  prefs: Preferences;
  lastUsed: LastUsed;

  // Standalone variables from __init__ in jj.py
  image_folder: string;
  image_folder2: string;
  caption_text: string;
  overwrite_captions: boolean;
  skip_bilingual: boolean;
  convert_output: string;
  delete_originals: boolean;
  prep_mode: string;
  face_selection: string;
  face_padding: string;
  prep_megapixels: string;

  // Dataset variables
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

  // System & VRAM/RAM monitor
  vram_used: number;
  vram_total: number;
  ram_used: number;
  ram_total: number;
  status_hint: string;

  // Training state
  current_process: number | null;
  training_state: 'idle' | 'running' | 'paused' | 'stopped';
  training_logs: string[];
  entries: Record<string, any>;
}

export type Action =
  | { type: 'UPDATE_SETTINGS'; payload: Partial<FizgigSettings> }
  | { type: 'UPDATE_PREFS'; payload: Partial<Preferences> }
  | { type: 'UPDATE_LAST_USED'; payload: Partial<LastUsed> }
  | { type: 'UPDATE_ENTRIES'; payload: Record<string, any> }
  | { type: 'SET_IMAGE_FOLDER'; payload: string }
  | { type: 'SET_IMAGE_FOLDER2'; payload: string }
  | { type: 'SET_CAPTION_TRIGGER'; payload: string }
  | { type: 'APPEND_LOG'; payload: string }
  | { type: 'CLEAR_LOGS' }
  | { type: 'SET_SYSTEM_STATS'; payload: { vram_used: number; vram_total: number; ram_used: number; ram_total: number } }
  | { type: 'UPDATE_STATE'; payload: Partial<AppState> };

const initialState: AppState = {
  settings: {
    ARCHITECTURE: "Flux 2 Klein Base 9B",
    DATASET_CONFIG: "dataset/Fizgig_train.toml",
    VAE_MODEL: "",
    CLIP_MODEL: "",
    T5_MODEL: "",
    TEXT_ENCODER: "",
    DIT_MODEL: "",
    LORA_OUTPUT_DIR: "output_loras",
    LORA_NAME: "LoraName_TokenName_k9b",
    MODEL_TYPE: "",
    LEARNING_RATE: 4e-4,
    LORA_LR_RATIO: 1,
    NETWORK_DIM: 4,
    NETWORK_ALPHA: 4,
    NETWORK_TYPE: "LoRA (standard)",
    LOKR_FACTOR: 8,
    MAX_TRAIN_EPOCHS: 12,
    SAVE_EVERY_N_EPOCHS: 1,
    SEED: 42,
    BLOCKS_SWAP: "auto",
    MINIMAX_LOWNOISE_PCT: "60",
    MINIMAX_HIGHNOISE_LR_PCT: "100",
    MINIMAX_BLOCKS: "all",
    MINIMAX_BASE_QUANT: "fp8",
    MINIMAX_BLOCK_LIMIT: "all · every block (50 of 50)",
    MINIMAX_LR_WARMUP: "0",
    MINIMAX_EMA: "Off",
    MINIMAX_ADAPTER_RAMP: "Off",
    MINIMAX_CAPTION_DROPOUT: "0.05 (default)",
    MINIMAX_TRAIN_ADALN: false,
    MINIMAX_LIKENESS_OPT: true,
    MINIMAX_TRAINING_ADAPTER: true,
    MINIMAX_TREAD: true,
    MINIMAX_CLIP_STILL: true,
    MINIMAX_DISTILL: false,
    MINIMAX_TRAIN_BASE: "fl2va",
    MINIMAX_DISTILL_WEIGHT: "0.8",
    MINIMAX_DISTILL_REFS: "2",
    MINIMAX_DISTILL_PHASE1: "Auto (from dataset size)",
    MINIMAX_SLOW_BLOCKS: "",
    MINIMAX_SLOW_LR_SCALE: "0.2",
    MIXED_STOP_CATEGORY: "voice",
    MIXED_STOP_EPOCH: "",
    MIXED_STOP_MODE: "anchor at 10% LR (recommended)",
    RESUME_TRAINING: "",
    OPTIMIZER_TYPE: "adamw8bit",
    OPTIMIZER_ARGS: "",
    GRADIENT_ACCUMULATION: 1,
    MAX_GRAD_NORM: 1.0,
    NETWORK_DROPOUT: 0,
    ATTENTION_MECHANISM: "sdpa",
    LOGGING_DIR: "",
    LOG_WITH: "none",
    LOG_PREFIX: "",
    IMG_IN_TXT_IN_OFFLOADING: false,
    LR_SCHEDULER: "constant",
    LR_WARMUP_STEPS: "",
    LR_DECAY_STEPS: "",
    ADAPTIVE_LR: false,
    ADAPTIVE_LR_MIN: "1e-5",
    ADAPTIVE_LR_MAX: "4e-4",
    CONTEXT_LORA_PATH: "",
    CONTEXT_LORA_STRENGTH: "1.0",
    TIMESTEP_SAMPLING: "shift",
    DISCRETE_FLOW_SHIFT: "3.0",
    SIGMOID_SCALE: "1.0",
    MIN_TIMESTEP: "",
    MAX_TIMESTEP: "",
    PRESERVE_DISTRIBUTION: false,
    WEIGHTING_SCHEME: "none",
    LOGIT_MEAN: "0.0",
    LOGIT_STD: "1.0",
    MODE_SCALE: "1.29",
    METADATA_TITLE: "",
    METADATA_AUTHOR: "",
    METADATA_DESCRIPTION: "",
    METADATA_LICENSE: "",
    METADATA_TAGS: "",
    METADATA_TRIGGER_PHRASE: "",
    METADATA_THUMBNAIL: "",
    FP8: true,
    SCALED: true,
    QUANT_4BIT: false,
    COMPILE_BLOCKS: "auto",
    GRADIENT_CHECKPOINTING: true,
    FP8_TEXT_ENCODER: true,
    SAVE_STATE: true,
    SAVE_STATE_ON_TRAIN_END: true,
    KEEP_LAST_N_STATES: 2,
    KREA2_LOSS_WATCH: false,
    KREA2_PER_IMAGE_LR: false,
    KREA2_AUTO_RECAPTION: false,
    SAMPLE_ENABLED: true,
    SAMPLE_PROMPT: "A high quality photo",
    SAMPLE_WIDTH: 768,
    SAMPLE_HEIGHT: 768,
    SAMPLE_STEPS: 40,
    SAMPLE_SEED: 1234,
    SAMPLE_EVERY_N_EPOCHS: 1,
    SAMPLE_EVERY_N_STEPS: 0,
    SAMPLE_AT_FIRST: true,
    CACHE_SAMPLE_MODEL: "auto",
    SAMPLE_FLOW_SHIFT: "",
    SAMPLE_NEGATIVE: "blurry, low detail, noisy, washed out, oversaturated, distorted anatomy, extra limbs, duplicate objects, text, watermark, logo, frame, cropped subject, flat lighting, muddy colors",
    SAMPLE_CFG_SCALE: 1.0,
    SAMPLE_FRAMES: "56 frames with sound (~2.3s)",
    MINIMAX_TURBO_STEPS: 6,
    MINIMAX_TURBO_STRENGTH: 75,
    CAPTION_TRIGGER_WORD: "",
    CAPTION_MODEL: "MiaoshouAI/Florence-2-base-PromptGen",
    CAPTION_TASK: "<DETAILED_CAPTION>",
    CAPTION_MAX_TOKENS: 256,
  },
  prefs: {
    base_dit: "",
    distilled_dit: "",
    vae: "",
    text_encoder: "",
    krea2_raw_dit: "",
    krea2_turbo_dit: "",
    krea2_vae: "",
    krea2_text_encoder: "",
    krea2_turbo_lora: "",
    minimax_dit: "",
    minimax_ref_dit: "",
    minimax_text_encoder: "",
    minimax_vae: "",
    minimax_audio_vae: "",
    minimax_turbo_lora: "",
    minimax_training_adapter: "",
    minimax_ref_training_adapter: "",
    lora_output_dir: "output_loras",
    profiles_dir: "profiles",
    cache_dir: "cache",
    input_lora_dir: "",
    input_ref_dir: "",
    input_dataset_dir: "",
    runpod_stop_when_done: "0",
    runpod_api_key: "",
    inference_blocks_to_swap: "Auto (detect from GPU)",
    inference_int8: "1",
    cuda_device: "",
    cloud_provider: "modal",
    modal_gpu: "A100-40GB",
  },
  lastUsed: {
    image_prep_source: "",
    image_folder: "",
    image_folder2: "",
    caption_trigger: "trigger_word",
    dataset_cache_dir: "cache",
    sample_prompt: "A high quality photo",
    prep_mode: "Auto Prep (Face Crops)",
    prep_replace_originals: false,
    prep_megapixels: "1.0",
  },
  image_folder: "",
  image_folder2: "",
  caption_text: "trigger_word",
  overwrite_captions: true,
  skip_bilingual: true,
  convert_output: "",
  delete_originals: false,
  prep_mode: "Auto Prep (Face Crops)",
  face_selection: "Largest Face",
  face_padding: "20",
  prep_megapixels: "1.0",
  dataset_name: "Fizgig_train",
  dataset_type: "Image with Captions",
  dataset_video_dir: "",
  dataset_cache_dir: "",
  dataset_caption_ext: ".txt",
  dataset_jsonl_file: "",
  dataset_megapixels: "0.25",
  dataset_batch_size: "1",
  dataset_num_repeats: "1",
  dataset_enable_bucket: true,
  dataset_no_upscale: true,
  dataset_target_frames: "1, 25, 45",
  dataset_frame_extraction: "head",
  dataset_source_fps: "30.0",
  vram_used: 0,
  vram_total: 0,
  ram_used: 0,
  ram_total: 0,
  status_hint: "Ready",
  current_process: null,
  training_state: 'idle',
  training_logs: [],
  entries: {},
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'UPDATE_SETTINGS':
      return { ...state, settings: { ...state.settings, ...action.payload } };
    case 'UPDATE_PREFS':
      return { ...state, prefs: { ...state.prefs, ...action.payload } };
    case 'UPDATE_LAST_USED':
      return { ...state, lastUsed: { ...state.lastUsed, ...action.payload } };
    case 'UPDATE_ENTRIES':
      return { ...state, entries: { ...state.entries, ...action.payload } };
    case 'SET_IMAGE_FOLDER':
      return {
        ...state,
        image_folder: action.payload,
        lastUsed: { ...state.lastUsed, image_folder: action.payload },
      };
    case 'SET_IMAGE_FOLDER2':
      return {
        ...state,
        image_folder2: action.payload,
        lastUsed: { ...state.lastUsed, image_folder2: action.payload },
      };
    case 'SET_CAPTION_TRIGGER':
      return {
        ...state,
        caption_text: action.payload,
        settings: { ...state.settings, CAPTION_TRIGGER_WORD: action.payload },
        lastUsed: { ...state.lastUsed, caption_trigger: action.payload },
      };
    case 'APPEND_LOG':
      return {
        ...state,
        training_logs: [...state.training_logs.slice(-500), action.payload],
      };
    case 'CLEAR_LOGS':
      return {
        ...state,
        training_logs: [],
      };
    case 'SET_SYSTEM_STATS':
      return {
        ...state,
        vram_used: action.payload.vram_used,
        vram_total: action.payload.vram_total,
        ram_used: action.payload.ram_used,
        ram_total: action.payload.ram_total,
      };
    case 'UPDATE_STATE':
      return { ...state, ...action.payload };
    default:
      return state;
  }
}

const SettingsContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
}>({
  state: initialState,
  dispatch: () => null,
});

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Auto-sync initial config from disk
  useEffect(() => {
    fetch('/api/prefs')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.prefs) {
          dispatch({ type: 'UPDATE_PREFS', payload: data.prefs });
        }
        if (data && data.lastUsed) {
          dispatch({ type: 'UPDATE_LAST_USED', payload: data.lastUsed });
          if (data.lastUsed.image_folder) {
            dispatch({ type: 'SET_IMAGE_FOLDER', payload: data.lastUsed.image_folder });
          }
          if (data.lastUsed.caption_trigger) {
            dispatch({ type: 'SET_CAPTION_TRIGGER', payload: data.lastUsed.caption_trigger });
          }
        }
      })
      .catch((e) => console.error('Could not load prefs from backend:', e));
  }, []);

  return (
    <SettingsContext.Provider value={{ state, dispatch }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettingsStore() {
  return useContext(SettingsContext);
}
