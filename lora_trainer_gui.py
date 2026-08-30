import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu, scrolledtext, simpledialog
import subprocess
import sys
import threading
import json
import os
import signal
import math
import re
import webbrowser
import glob
import time
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image, ImageTk

# CUDA allocator policy, set before ANYTHING imports torch — the backend is fixed at CUDA init,
# so this has to happen at startup rather than when a tool loads its model.
#
# The workbench tools (Repair Studio, LoRA the Explorer, LoRA Royale) run IN this process and
# repeatedly allocate and free large, differently-sized blocks: a model load, then a preview per
# slider change or per variant, then a swap to another LoRA. The default allocator carves those
# from fixed-size segments, which fragments under exactly that pattern and bites hardest on cards
# with little headroom. expandable_segments lets a segment grow and shrink instead.
#
# Note this is inherited by the training subprocess too, which is intended — the same churn
# happens there. Respects an existing value, and FIZGIG_NO_EXPANDABLE=1 opts out for A/B testing.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# OpenMP wait policy, also before anything loads torch (which loads libiomp on Windows).
# Intel OpenMP's default is to keep its whole thread pool ACTIVELY SPINNING for 200 ms
# (KMP_BLOCKTIME) after every parallel region, "in case more work arrives". Training work is
# on the GPU, but each step touches small CPU tensors (collate, noise, timestep bookkeeping),
# so the pool — sized to every core — re-arms its spin constantly and burns 100% of every
# core doing nothing (issue #18: 4080 Super pegged on all cores at CUDA 85%). Measured here:
# simulated per-step CPU ops with GPU-sized gaps burn 14.8 cores spinning by default, 0.0
# with BLOCKTIME=0, no step-time cost. Inherited by the training subprocess, which is the
# point. Both vars set: KMP_* is Intel-runtime-specific, OMP_WAIT_POLICY is the portable one.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# Face detection imports (optional - graceful fallback if not installed)
try:
    from face_utils import (FaceDetector, FaceEmbedder, crop_to_face, draw_face_boxes,
                            is_face_detection_available)
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    FaceDetector = None
    FaceEmbedder = None

from fizgig_gui.core.config.constants import (
    COLORS, FONT_FAMILY, FONT_MONO,
    BG_COLOR, FG_COLOR, ACCENT_COLOR, ENTRY_BG, BUTTON_ACTIVE, BORDER_COLOR,
    ACTIVE_ENTRY_BG, ACTIVE_ENTRY_FG, SAMPLE_RESOLUTIONS,
)


from fizgig_gui.core.ui_base.widgets import _GUIWriter, ToolTip, CollapsibleFrame

from fizgig_gui.core.domain.architectures import (
    ARCHITECTURES, _ARCH_ALIASES, ARCHITECTURE_LIST, _canon_arch, LORA_NAME_SUFFIXES,
)

# Preview resolutions offered by BOTH the Samples tab and the live "Override next sample" panel.
# They used to be two hardcoded lists that had drifted — the Samples tab reached 1536 while the
# override stopped at 1024, so a run previewing at 1280+ could not be reproduced by the override,
# which silently downgraded it. Nothing downstream caps the value (Krea 2 rounds up to alignment,
# Klein floors to a multiple of 16), so the ceiling was purely this list.
SAMPLE_RESOLUTIONS = ["512", "640", "768", "1024", "1280", "1536"]

# Fizgig installation directory (where this GUI lives)
FIZGIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for custom presets (per architecture)
PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")

# Snapshot of settings from the most recent training launch — restorable via "Load Last Train" button
LAST_TRAIN_FILE = os.path.join(PRESETS_DIR, ".last_train_settings.json")
# Training queue: full settings snapshots waiting to run back-to-back. Survives restart;
# never auto-starts on launch (a queue found at startup waits for the user's first Start).
QUEUE_FILE = os.path.join(PRESETS_DIR, "training_queue.json")

from fizgig_gui.core.domain.minimax_math import (
    MINIMAX_LOWNOISE_SIGMA, minimax_lownoise_to_shift, minimax_highnoise_lr,
    minimax_shift_to_lownoise, MINIMAX_STRUCTURE_OPTIONS, MINIMAX_STRUCTURE_DESC,
    MINIMAX_STRUCTURE_DEFAULT, MINIMAX_BLOCK_OPTIONS, MINIMAX_NUM_BLOCKS,
    MINIMAX_LIKENESS_BLOCKS, MINIMAX_AUDIO_BLOCKS, MINIMAX_BASE_QUANT_OPTIONS,
    minimax_base_quant, minimax_block_spec, MINIMAX_TRAIN_BASE_OPTIONS, minimax_train_base,
)

from fizgig_gui.core.config.presets import (
    ft_checkpoint_continuation, BUILT_IN_PRESETS, KREA2_BUILT_IN_PRESETS,
    SEED_TRAVEL_PRESETS, MINIMAX_BUILT_IN_PRESETS, _MM_DEFAULTS_KEY,
)
from fizgig_gui.core.ui_base.styling import StylingMixin
from fizgig_gui.core.ui_base.tab_scaffold import TabScaffoldMixin

from fizgig_gui.core.tabs.start_tab import StartTabMixin
from fizgig_gui.core.tabs.metadata_tab import MetadataTabMixin
from fizgig_gui.core.tabs.prefs_tab import PrefsTabMixin
from fizgig_gui.core.tabs.profiler_tab import ProfilerTabMixin
from fizgig_gui.core.tabs.explorer_tab import ExplorerTabMixin
from fizgig_gui.core.tabs.extract_tab import ExtractTabMixin
from fizgig_gui.core.tabs.repair_studio_tab import RepairStudioTabMixin
from fizgig_gui.core.tabs.caption_tab import CaptionTabMixin

from fizgig_gui.core.models.caption_model import CaptionModelMixin

from fizgig_gui.core.ui_base.console_validation import ConsoleValidationMixin

from fizgig_gui.core.config.last_used import (
    DATASET_DIR, CACHE_DIR, OUTPUT_LORAS_DIR, LAST_USED_FILE,
    load_last_used, save_last_used,
)


from fizgig_gui.core.config.prefs import (
    PREFS_FILE, HELP_FILE, _FIZGIG_DIR,
    FLORENCE_DEFAULT_MODEL, FLORENCE_MODELS, FLORENCE_REVISIONS,
    FLORENCE_CODE_REVISIONS, FLORENCE_TASKS, QWEN_CAPTION_MODEL, QWEN_CUSTOM_TASK,
    _PORTABLE_DIR_KEYS, _resolve_pref_path, _serialize_pref_path, DEFAULT_PREFS,
    _enumerate_gpus, _apply_cuda_device_pref, _auto_detect_blocks_to_swap,
    load_prefs, save_prefs,
    _persist_disabled, _running_on_pod, _pod_id, _app_commit,
    _git, _git_describe_version, _latest_release_tag, _update_status_from,
    _git_ok, _check_for_update, _pod_stop_key_env,
)


from fizgig_gui.core.config.settings_map import SETTING_TO_PREF, PRESETS

class LoRATrainerGUI(
    StartTabMixin, MetadataTabMixin, PrefsTabMixin,
    ProfilerTabMixin, ExplorerTabMixin, ExtractTabMixin,
    RepairStudioTabMixin, StylingMixin,
    TabScaffoldMixin, CaptionModelMixin,
    CaptionTabMixin, ConsoleValidationMixin
):
    def __init__(self, master):
        self.master = master
        master.title("Fizgig — Klein 9B & Krea 2 LoRA Studio")
        master.geometry("1450x1124")  # wide enough that the IDLE/BUSY light clears the last tab ("Preferences") with the Metadata tab in the strip; +100 height for the bottom status bar
        master.minsize(1180, 900)  # keeps the tab row clear of the status light + tab content not cut off
        master.configure(bg=BG_COLOR)
        # Closing the window must not orphan a training subprocess: Tk's default destroy
        # exits the interpreter but the trainer runs in its own process (no job object on
        # Windows) — it SURVIVED, holding 14-20 GB of VRAM and still writing checkpoints,
        # stoppable only via Task Manager. Confirm, then taskkill the tree via stop_training.
        try:
            master.protocol("WM_DELETE_WINDOW", self._on_app_close)
        except Exception:
            pass

        # Window/taskbar icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            try:
                icon_img = Image.open(icon_path).resize((32, 32), Image.LANCZOS)
                self._window_icon = ImageTk.PhotoImage(icon_img)
                master.iconphoto(True, self._window_icon)
            except Exception:
                pass

        self.current_process = None
        self.training_thread = None
        self.process_group_id = None
        self.user_scrolled = False  # Flag for manual console scrolling
        self.samples_watcher_running = False  # For live gallery updates
        self.samples_watcher_thread = None

        # Global log buffer + console popup state
        self._log_buffer = []
        self._console_popup = None
        self._console_popup_text = None
        self._captioning_running = False
        self._translating = False

        # Redirect stdout/stderr when running under pythonw.exe (no console)
        if sys.stdout is None or sys.stderr is None:
            sys.stdout = _GUIWriter(self, "stdout")
            sys.stderr = _GUIWriter(self, "stderr")

        # HTTP server for samples gallery (avoids CORS issues)
        self.gallery_server = None
        self.gallery_server_port = None
        self.gallery_server_thread = None

        # Load last-used folder paths
        self.last_used = load_last_used()

        # Load user preferences (model paths, output directories)
        self.prefs = load_prefs()
        # Before ANY CUDA work: _auto_detect_blocks_to_swap and the workbench tools both build a
        # CUDA context, and the visible set is frozen the moment one exists.
        self._cuda_device_env_locked = bool(
            os.environ.get("CUDA_VISIBLE_DEVICES")) and not str(
            self.prefs.get("cuda_device", "")).strip()
        self._cuda_device_applied = _apply_cuda_device_pref(self.prefs)
        self.prefs_vars = {}
        for key, default in DEFAULT_PREFS.items():
            var = tk.StringVar(value=self.prefs.get(key, default))
            var.trace_add("write", lambda *a, k=key: self._save_pref(k))
            self.prefs_vars[key] = var

        # Caption Generator variables
        # Unified training-folder var — shared between Start tab (authoritative),
        # Captions tab, and the Fizgig_train.toml auto-saver. Replaces the old
        # caption_folder_var / dataset_image_dir_var pair + their propagation.
        self.image_folder_var = tk.StringVar(value=self.last_used.get("image_folder", ""))
        self.caption_text_var = tk.StringVar(value=self.last_used.get("caption_trigger", "trigger_word"))
        self.overwrite_captions_var = tk.BooleanVar(value=True)
        self.skip_bilingual_var = tk.BooleanVar(value=True)

        # Image Converter variables — source is unified with self.image_folder_var
        # (the Start-tab picker); only the output folder is Image-Prep-specific.
        # Always empty since the Output Folder UI was removed — prep always writes into the
        # training folder. Kept because the convert pipeline reads it ("or source_folder").
        self.convert_output_var = tk.StringVar()
        # Image Prep's size target is a MEGAPIXEL AREA, not a longest edge — declared with the
        # dataset vars below (it seeds from the Training tab's Target Megapixels).
        # Default flipped to KEEP (False) 2026-07-28 — safe-by-default like the Look Filter and
        # the Captions Remove button; remembered across restarts now that it's a real choice.
        self.delete_originals_var = tk.BooleanVar(
            value=bool(self.last_used.get("prep_replace_originals", False)))

        # Prep Mode and Face Cropping variables
        self.prep_mode_var = tk.StringVar(value=self.last_used.get("prep_mode", "Auto Prep (Face Crops)"))
        self.face_selection_var = tk.StringVar(value="Largest Face")
        self.face_padding_var = tk.StringVar(value="20")

        # Face detector instance (lazy loaded)
        self._face_detector = None

        # Dataset Manager variables
        # Hardcoded: dataset name and type are fixed; num_repeats always 1; cache dir comes from Preferences.
        self.dataset_name_var = tk.StringVar(value="Fizgig_train")
        self.dataset_type_var = tk.StringVar(value="Image with Captions")
        # dataset_image_dir_var removed — unified into self.image_folder_var above.
        self.dataset_video_dir_var = tk.StringVar()
        self.dataset_cache_dir_var = tk.StringVar()  # legacy/back-compat — UI removed; cache dir now lives in prefs_vars["cache_dir"]
        self.dataset_caption_ext_var = tk.StringVar(value=".txt")
        self.dataset_jsonl_file_var = tk.StringVar()
        self.dataset_megapixels_var = tk.StringVar(value="0.25")
        # Image Prep's target area. Training buckets by AREA and never upscales, so prepping to a
        # longest-edge cap silently pushed every non-square image below the training target and
        # threw away detail training could not get back (issue #44). Defaults to 1.0 MP — NOT the
        # Training tab's 0.25 default: prepping above the training target is free (training just
        # downscales at cache time), so the default keeps resolution in hand for training at any
        # target up to 1.0 MP. The inline warning covers the one harmful direction (prep < train).
        # Remembered across restarts.
        self.prep_megapixels_var = tk.StringVar(
            value=str(self.last_used.get("prep_megapixels", "1.0")))
        self.dataset_batch_size_var = tk.StringVar(value="1")
        self.dataset_num_repeats_var = tk.StringVar(value="1")
        self.dataset_enable_bucket_var = tk.BooleanVar(value=True)
        self.dataset_no_upscale_var = tk.BooleanVar(value=True)
        self.dataset_target_frames_var = tk.StringVar(value="1, 25, 45")
        self.dataset_frame_extraction_var = tk.StringVar(value="head")
        self.dataset_source_fps_var = tk.StringVar(value="30.0")

        # Ensure all three portable output directories exist (honours user's
        # prefs overrides when present; otherwise creates the defaults:
        # output_loras/, profiles/, cache/ inside FIZGIG_DIR).
        for pref_key in ("lora_output_dir", "profiles_dir", "cache_dir"):
            try:
                os.makedirs(self.prefs_vars[pref_key].get(), exist_ok=True)
            except Exception:
                pass
        # Dataset dir is hardcoded to FIZGIG_DIR/dataset/ (never a pref).
        # The Dataset tab auto-writes Fizgig_train.toml here via
        # auto_save_dataset_config_silent(); no example template needed.
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Add traces to auto-save last-used folder paths and settings.
        # image_folder_var is the single source of truth shared between the
        # Start tab, Captions tab, and the Fizgig_train.toml auto-saver —
        # no propagation helpers needed.
        self.image_folder_var.trace_add("write", self._save_last_used_paths)
        self.caption_text_var.trace_add("write", self._save_last_used_paths)
        self.prep_mode_var.trace_add("write", self._save_last_used_paths)
        self.delete_originals_var.trace_add("write", self._save_last_used_paths)
        self.prep_megapixels_var.trace_add("write", self._save_last_used_paths)
        # Prep's target area and the Training tab's target are independent, but prepping BELOW
        # the training target is the one harmful direction — refresh the summary (which carries
        # that warning) whenever either side moves.
        self.prep_megapixels_var.trace_add("write", self._update_prep_note)
        self.dataset_megapixels_var.trace_add("write", self._update_prep_note)
        # The Image Prep summary shows a live image count + resolution check for the folder,
        # so a folder change on the Start tab must refresh it. Guarded: fires before the
        # Image Prep tab exists during startup, and _update_prep_note no-ops then.
        self.image_folder_var.trace_add("write", self._update_prep_note)
        self.image_folder_var.trace_add("write", self._refresh_audio_only_ui)
        # Auto-save the dataset TOML on every relevant change (no Save button needed)
        def _auto_save_ds(*_a):
            if hasattr(self, "auto_save_dataset_config_silent"):
                self.auto_save_dataset_config_silent()
        for _v in (self.image_folder_var, self.dataset_caption_ext_var,
                   self.dataset_megapixels_var, self.dataset_batch_size_var,
                   self.dataset_enable_bucket_var, self.dataset_no_upscale_var):
            _v.trace_add("write", _auto_save_ds)
        # Multi Concept adds [[datasets]] blocks, so its toggle and folders have to rewrite the
        # TOML too — they are created later (Training tab), hence the deferred hook-up.
        self._auto_save_ds = _auto_save_ds

        # Initialize settings with default values, including conversion settings
        # Klein 9B Base is the only supported architecture. A few legacy keys
        # (CLIP_MODEL / T5_MODEL / MODEL_TYPE) remain as empty defaults so dead
        # code paths gated behind `config["uses_*"]` flags don't KeyError.
        default_arch = "Flux 2 Klein Base 9B"
        self.settings = {
            "ARCHITECTURE": default_arch,
            "DATASET_CONFIG": os.path.join(DATASET_DIR, "Fizgig_train.toml"),
            # Model paths resolved at runtime from prefs_vars — blank fallback.
            "VAE_MODEL": "",
            "CLIP_MODEL": "",
            "T5_MODEL": "",
            "TEXT_ENCODER": "",
            "DIT_MODEL": "",
            "LORA_OUTPUT_DIR": OUTPUT_LORAS_DIR,
            "LORA_NAME": "LoraName_TokenName_k9b",
            "MODEL_TYPE": "",
            "LEARNING_RATE": 4e-4,
            "LORA_LR_RATIO": 1,
            "NETWORK_DIM": 4,
            "NETWORK_ALPHA": 4,
            # Krea 2 only (Klein hides it and trains standard). LoRA default; LoKR is the
            # quality pick (validated 31 Jul: highest likeness measured here, no skin sheen).
            "NETWORK_TYPE": "LoRA (standard)",
            "LOKR_FACTOR": 8,
            "MAX_TRAIN_EPOCHS": 12,
            "SAVE_EVERY_N_EPOCHS": 1,
            "SEED": 42,
            "BLOCKS_SWAP": "auto",  # Klein valid range 0-16; "auto" detects from GPU
            # MiniMax H3 only. Percent of steps trained below sigma 0.5 (H3's own default works out at ~7.7%).
            # 60 + mid-concentrated matches the MiniMax preset, which is what a switch to that
            # family applies anyway; these are the values the widgets are BUILT with, so they are
            # what shows before any preset lands. Both keys are MiniMax-only — no other family
            # reads them. (OPTIMIZER_TYPE below is deliberately NOT changed to match the preset:
            # it is shared with Klein and Krea 2, and the MiniMax preset supplies adamw on switch.)
            "MINIMAX_LOWNOISE_PCT": "60",
            # What the steps ABOVE sigma 0.5 do to the LR, as a percentage. 100 = unchanged, which
            # is every run before this existed.
            "MINIMAX_HIGHNOISE_LR_PCT": "100",
            "MINIMAX_BLOCKS": "all",
            "MINIMAX_BASE_QUANT": MINIMAX_BASE_QUANT_OPTIONS[0],
            # OFF by default (Peter's call from real runs). The reference trains AdaLN on the
            # pruned checkpoint, but AdaLN is a pure function of the timestep — adaln_proj(t_emb)
            # and nothing else — so its adapters cannot tell one subject from another, and on the
            # pruned build they were taking ~45% of all weight movement to do it.
            "MINIMAX_TRAIN_ADALN": False,
            # Optimised Likeness Learning — photo steps train blocks 20-49 only, clips train
            # everything. On by default: it is the measured best recipe for the character/voice
            # work H3 is for. The Style preset turns it OFF (style needs the early blocks).
            "MINIMAX_LIKENESS_OPT": True,
            "MINIMAX_DISTILL": False,      # off = ordinary training
            # Which H3 base ordinary training runs on ("fl2va"/"ref2va"). NOT in any preset —
            # the Training Base dropdown's var lives outside self.entries by design.
            "MINIMAX_TRAIN_BASE": "fl2va",
            "MINIMAX_DISTILL_WEIGHT": "0.8",
            "MINIMAX_DISTILL_REFS": "2",
            "MINIMAX_SLOW_BLOCKS": "",     # blank = one LR everywhere
            "MINIMAX_SLOW_LR_SCALE": "0.2",
            "RESUME_TRAINING": "",
            "OPTIMIZER_TYPE": "adamw8bit",
            "OPTIMIZER_ARGS": "",
            "GRADIENT_ACCUMULATION": 1,  # Effective batch size = batch × this
            "MAX_GRAD_NORM": 1.0,  # Gradient clipping (0 to disable)
            "NETWORK_DROPOUT": 0,  # LoRA dropout for regularization
            "ATTENTION_MECHANISM": "sdpa",  # Default attention (was "none", causing duplicate flags)
            "LOGGING_DIR": "",
            "LOG_WITH": "none",
            "LOG_PREFIX": "",
            "IMG_IN_TXT_IN_OFFLOADING": False,
            "LR_SCHEDULER": "constant",
            "LR_WARMUP_STEPS": "",
            "LR_DECAY_STEPS": "",
            "ADAPTIVE_LR": False,
            "ADAPTIVE_LR_MIN": "1e-5",
            "ADAPTIVE_LR_MAX": "4e-4",
            "CONTEXT_LORA_PATH": "",
            "CONTEXT_LORA_STRENGTH": "1.0",
            "TIMESTEP_SAMPLING": "shift",
            "DISCRETE_FLOW_SHIFT": "3.0",
            "SIGMOID_SCALE": "1.0",
            "MIN_TIMESTEP": "",
            "MAX_TIMESTEP": "",
            "PRESERVE_DISTRIBUTION": False,
            "WEIGHTING_SCHEME": "none",
            "LOGIT_MEAN": "0.0",
            "LOGIT_STD": "1.0",
            "MODE_SCALE": "1.29",
            "METADATA_TITLE": "",
            "METADATA_AUTHOR": "",
            "METADATA_DESCRIPTION": "",
            "METADATA_LICENSE": "",
            "METADATA_TAGS": "",
            "METADATA_TRIGGER_PHRASE": "",
            "METADATA_THUMBNAIL": "",
            "FP8": True,  # Default FP8 setting (--fp8_base)
            "SCALED": True,  # Default Scaled setting (--fp8_scaled, recommended with fp8_base)
            "QUANT_4BIT": False,  # 4-bit NF4 base (low-VRAM); supersedes fp8 when on
            "COMPILE_BLOCKS": "auto",  # torch.compile the DiT blocks (krea2): auto | on | off
                "GRADIENT_CHECKPOINTING": True,  # ON by default — recompute activations to fit 9B on most cards
            "FP8_TEXT_ENCODER": True,  # FP8 for text encoder (T5/LLM)
            # Resumable state dirs. Pause/Resume writes state regardless — these only govern the
            # automatic saves. Keep-N matters: a state is LoRA + optimizer (~470 MB at rank 32).
            "SAVE_STATE": True,
            "SAVE_STATE_ON_TRAIN_END": True,
            "KEEP_LAST_N_STATES": 2,
            "KREA2_LOSS_WATCH": False,   # per-image loss tracking + stuck-image detection (krea2)
            "KREA2_PER_IMAGE_LR": False,  # per-image adaptive LR (throttle stuck images) — experimental
            "KREA2_AUTO_RECAPTION": False,  # Qwen3-VL rewrites stuck images' captions mid-run — experimental
            # Sample generation settings
            "SAMPLE_ENABLED": True,
            "SAMPLE_PROMPT": "A high quality photo",
            "SAMPLE_WIDTH": 768,
            "SAMPLE_HEIGHT": 768,
            "SAMPLE_STEPS": 40,
            "SAMPLE_SEED": 1234,
            "SAMPLE_EVERY_N_EPOCHS": 1,
            "SAMPLE_EVERY_N_STEPS": 0,
            "SAMPLE_AT_FIRST": True,
            "CACHE_SAMPLE_MODEL": "auto",  # keep Distilled sample model in RAM between epochs
            "SAMPLE_FLOW_SHIFT": "",
            "SAMPLE_NEGATIVE": "blurry, low detail, noisy, washed out, oversaturated, distorted anatomy, extra limbs, duplicate objects, text, watermark, logo, frame, cropped subject, flat lighting, muddy colors",
            "SAMPLE_CFG_SCALE": 1.0,
            # MiniMax Turbo previews (used only when the Turbo LoRA is set in Preferences)
            "MINIMAX_TURBO_STEPS": 6,
            "MINIMAX_TURBO_STRENGTH": 75,
            # Florence captioning settings
            "CAPTION_TRIGGER_WORD": "",
            "CAPTION_MODEL": "MiaoshouAI/Florence-2-base-PromptGen",
            "CAPTION_TASK": "<DETAILED_CAPTION>",
            "CAPTION_MAX_TOKENS": 256,
        }

        # Backing store for the active dataset config path. The Model Paths section was removed
        # from the Training tab; Dataset-tab callbacks write here, training command builders read here.
        self._dataset_config_var = tk.StringVar(value=self.settings["DATASET_CONFIG"])

        # Override with last-used LoRA output directory if available
        if self.last_used.get("lora_output_dir"):
            self.settings["LORA_OUTPUT_DIR"] = self.last_used["lora_output_dir"]

        # Training queue — loaded before any UI so the status-bar button can show its count.
        # A queue restored from a previous session waits for the user; it never auto-starts.
        self.training_queue = self._load_training_queue()

        # Startup update check: which newer tag (if any) is available. Populated on a daemon
        # thread a couple of seconds after launch (see the after() call at the end of __init__).
        self._update_info = None

        # Klein's trainer resolves these itself (name-or-module-path). Krea 2 goes through
        # fizgig.training.optimizers, which offers a different set — filtered to what's actually
        # installed — so the dropdown is re-populated when the Base Model selector changes.
        self.optimizer_types = ["adamw", "adamw8bit", "bitsandbytes.optim.AdEMAMix8bit", "bitsandbytes.optim.PagedAdEMAMix8bit"]
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            from fizgig.training.optimizers import available_optimizers
            self.krea2_optimizer_types = available_optimizers()
        except Exception:
            self.krea2_optimizer_types = ["adamw8bit", "adamw"]

        self.setup_styles()

        # Live VRAM/RAM status bar pinned to the bottom (packed first so it
        # reserves the strip; the notebook then fills the space above it).
        self._build_status_bar(master)

        # Create notebook and tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status indicator — overlaid on top-right of notebook, zero vertical space
        self._status_canvas = tk.Canvas(master, width=84, height=32,
                                        bg=COLORS["bg_deep"], highlightthickness=0,
                                        cursor="hand2")
        self._status_canvas.place(relx=1.0, x=-16, y=11, anchor="ne")
        self._status_canvas.bind("<Button-1>", lambda e: self._open_console_popup())
        ToolTip(self._status_canvas, "Click to view console log")

        # Tabs ordered by natural workflow: Start -> Prep -> Caption -> Train -> everything else.
        # The old Dataset tab was folded into Training (Other Options → Dataset subsection);
        # its Image Directory field is now the Start tab's folder picker (shared with Captions).
        self.start_tab = ttk.Frame(self.notebook)
        self.start_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.start_tab, text="1. Start")

        self.image_converter_tab = ttk.Frame(self.notebook)
        self.image_converter_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.image_converter_tab, text="2. Image Prep")

        self.caption_gen_tab = ttk.Frame(self.notebook)
        self.caption_gen_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.caption_gen_tab, text="3. Captions")

        self.samples_tab = ttk.Frame(self.notebook)
        self.samples_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.samples_tab, text="4. Samples")

        self.training_tab = ttk.Frame(self.notebook)
        self.training_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.training_tab, text="5. Training")

        self.profiler_tab = ttk.Frame(self.notebook)
        self.profiler_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.profiler_tab, text="Profiler")

        self.repair_studio_tab = ttk.Frame(self.notebook)
        self.repair_studio_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.repair_studio_tab, text="Repair Studio")

        self.explorer_tab = ttk.Frame(self.notebook)
        self.explorer_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.explorer_tab, text="LoRA the Explorer")

        self.lora_royale_tab = ttk.Frame(self.notebook)
        self.lora_royale_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.lora_royale_tab, text="LoRA Royale")

        self.extract_tab = ttk.Frame(self.notebook)
        self.extract_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.extract_tab, text="Extract")

        self.metadata_tab = ttk.Frame(self.notebook)
        self.metadata_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.metadata_tab, text="Metadata")

        self.prefs_tab = ttk.Frame(self.notebook)
        self.prefs_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.prefs_tab, text="Preferences")

        # Initialize tab contents
        self.entries = {}
        self.labels = {}  # Store label widgets for dynamic updates
        self.rows = {}    # Store row widgets for show/hide
        self.create_start_tab()
        self.create_training_settings()
        self.create_caption_generator()
        self.create_image_converter()
        self.create_samples_settings()
        self.create_profiler_tab()
        self.create_repair_studio_tab()
        self.create_explorer_tab()
        self.create_lora_royale_tab()
        self.create_extract_tab()
        self.create_metadata_tab()
        self.create_prefs_tab()
        # Restore remembered Repair Studio / Explorer Setup fields + attach save traces.
        # After ALL tabs exist: restoring fires their traces, which touch other tabs' widgets.
        self._restore_workbench_setup_fields()
        # Retag the LoRA name for the restored family. The settings dict is built long before
        # architecture_var exists, so its default is hardcoded Klein-shaped — without this, a
        # user whose last session was Krea 2 reopens to a name ending _k9b.
        try:
            self._apply_lora_name_suffix(self.architecture_var.get())
        except Exception:
            pass

        # AI captioning runs in batch_caption.py worker — no in-process models.
        self.caption_process = None
        self._caption_worker_stdin = None
        self._caption_worker_key = None
        self._caption_worker_ready = threading.Event()
        self._caption_worker_warm = False
        self._training_start_pending = False
        self._caption_worker_released_for_training = False
        self._caption_stop_file = ""
        self.captioning_stop_flag = False
        self.caption_thumbnails = {}
        self.current_caption_page = 0
        self.images_per_page = 12

        # Load architecture defaults first (populates optimizer / fp8 / timestep
        # fields that the built-in presets don't explicitly set), then overlay
        # the first built-in preset ("Old Reliable") on top. Load Settings From
        # Last Train still works — it just overrides whenever the user clicks it.
        self.load_default_preset(show_message=False)
        try:
            _builtins = self._builtins_for_arch(self.architecture_var.get())
            first_preset = next(iter(_builtins))
            self._apply_preset_values(_builtins[first_preset])
            self.custom_preset_var.set(first_preset)
        except Exception:
            pass

        # Create context menu for copying console text
        self.context_menu = Menu(self.master, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selected_text)

        # Bind tab change event to load caption images when visiting Captions tab
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.caption_images_loaded = False  # Track if we've loaded images for current folder

        # Reset flag when folder changes so images reload on next tab visit
        self.image_folder_var.trace_add("write", self._on_caption_folder_changed)

        # Prevent mousewheel from accidentally changing Combobox/Spinbox values
        # ONE global wheel router instead of per-panel bind_all tug-of-wars (which broke in
        # both directions: an open tool window stole the main window's wheel, and a stray
        # <Leave> killed the tool window's). The router sends the wheel wherever the POINTER
        # is: a Text (console) or Listbox scrolls itself natively; otherwise the nearest
        # scrollable Canvas up the ancestry scrolls — an inner panel when hovered, the tab's
        # main scrollbar everywhere else. Button-4/5 are the X11 wheel (pods).
        for _seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.master.bind_all(_seq, self._route_mousewheel)
            # Dropdowns/spinboxes must NEVER change value on wheel (an accidental flick over
            # the LR box mid-scroll is how a run gets silently mis-configured) — but they
            # must not be dead zones either: route the scroll to the page, then break so
            # the widget's own value-spin binding never runs. The old bare-"break" bindings
            # covered <MouseWheel> only, leaving X11 pods spinning values via Button-4/5.
            self.master.bind_class("TCombobox", _seq, self._wheel_over_dropdown)
            self.master.bind_class("TSpinbox", _seq, self._wheel_over_dropdown)
            self.master.bind_class("Spinbox", _seq, self._wheel_over_dropdown)

        # Start status indicator polling
        self._update_status_indicator()

        # Grey out sample fields overridden by Distilled (default on)
        if hasattr(self, '_on_distilled_samples_toggled'):
            self._on_distilled_samples_toggled()

        # Auto-save dataset config on startup if all fields are valid
        # This ensures training works immediately without manual "Save and Activate"
        self.auto_save_dataset_config_silent()

    # ── Global log + status indicator ───────────────────────────────────

    def _append_global_log(self, text):
        """Append text to the global log buffer and push to popup if open.

        Thread-safe by marshalling: several workers (profiler, extract, engine loads) log
        from their own threads, and writing the console popup's Text widget off the main
        thread is a Tcl panic — a hard process crash no try/except can catch. That is how
        clicking the IDLE/BUSY light during a model load killed the app: the click queued,
        the popup opened as the load returned, and the next worker log write hit it."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._append_global_log, text)
            return
        self._log_buffer.append(text)
        if len(self._log_buffer) > 50000:
            self._log_buffer = self._log_buffer[-40000:]
        if self._console_popup_text is not None:
            try:
                at_bottom = self._console_popup_text.yview()[1] >= 0.999
                self._console_popup_text.configure(state="normal")
                self._console_popup_text.insert(tk.END, text)
                if at_bottom:
                    self._console_popup_text.see(tk.END)
                self._console_popup_text.configure(state="disabled")
            except Exception:
                pass

    def _royale_is_busy(self):
        """True while any LoRA Royale work is in flight (epoch render / seed / prompt /
        strength travel / morph export / likeness scoring)."""
        return any(getattr(self, f, False) for f in (
            '_royale_rendering', '_royale_traveling', '_royale_pt_running',
            '_royale_lora_running', '_royale_exporting', '_royale_scoring',
            '_royale_cmp_running'))

    @staticmethod
    def _wheel_units(event):
        """Scroll units from either wheel encoding: Windows <MouseWheel> delta (±120 per
        notch, high-res mice send smaller values) or X11 Button-4/5 (pods)."""
        num = getattr(event, "num", 0)
        if num == 4:
            return -3
        if num == 5:
            return 3
        d = getattr(event, "delta", 0)
        if not d:
            return 0
        return int(-1 * (d / 120)) if abs(d) >= 120 else (-1 if d > 0 else 1)

    def _route_mousewheel(self, event):
        """Global wheel dispatch — see the install site for the routing rules."""
        try:
            w = self.master.winfo_containing(event.x_root, event.y_root)
        except Exception:
            w = None
        if w is None:
            return
        units = self._wheel_units(event)
        if not units:
            return
        node, hops = w, 0
        while node is not None and hops < 40:
            # Native scrollers own the wheel while hovered — their class bindings already
            # scroll them, and routing the page underneath as well would double-scroll.
            if isinstance(node, (tk.Text, tk.Listbox, ttk.Treeview)):
                return
            if isinstance(node, tk.Canvas):
                try:
                    if node.cget("yscrollcommand"):
                        node.yview_scroll(units, "units")
                        return "break"
                except tk.TclError:
                    pass
            node = getattr(node, "master", None)
            hops += 1
        return

    def _wheel_over_dropdown(self, event):
        """Wheel over a Combobox/Spinbox: scroll the page, never the value."""
        self._route_mousewheel(event)
        return "break"

    def _is_render_busy(self):
        """In-process GPU render on a tab that unloads its engine on switch (Repair
        Studio / Explorer / Royale). Switching tabs here would reset a busy engine and
        hang the app — used to lock tab switching. Excludes the training subprocess
        (separate process; switching during a run is fine)."""
        return (getattr(self, '_repair_preview_in_flight', False)
                or getattr(self, '_explorer_generating', False)
                or self._royale_is_busy())

    def _is_any_busy(self):
        """Return True if any background work is in progress."""
        if self.current_process is not None:
            try:
                if self.current_process.poll() is None:
                    return True
            except Exception:
                pass
        if getattr(self, '_captioning_running', False):
            return True
        if getattr(self, '_translating', False):
            return True
        if getattr(self, '_fetch_running', False):
            return True   # model download in flight — a queued run must not read partial files
        if getattr(self, '_repair_preview_in_flight', False):
            return True
        if getattr(self, '_explorer_generating', False):
            return True
        if self._royale_is_busy():
            return True
        # Profiler running (button disabled while active)
        if hasattr(self, 'profiler_run_btn'):
            try:
                if str(self.profiler_run_btn.cget("state")) == "disabled":
                    return True
            except Exception:
                pass
        # Extractor running
        if hasattr(self, 'extract_run_btn'):
            try:
                if str(self.extract_run_btn.cget("state")) == "disabled":
                    return True
            except Exception:
                pass
        return False

    def _update_status_indicator(self):
        """Poll busy state and redraw the IDLE/BUSY 'studio light': a lit circle
        with a soft glow + all-caps word + a matching-colour frame, all in the
        status colour (green idle / red busy)."""
        # Lock tab switching during an in-process render — switching can unload a busy
        # engine and hang the app. (Training is a subprocess, so it doesn't lock.)
        try:
            render_busy = self._is_render_busy()
            if render_busy != getattr(self, "_tabs_locked", None):
                self._tabs_locked = render_busy
                cur = self.notebook.select()
                for tid in self.notebook.tabs():
                    self.notebook.tab(tid, state=("disabled" if (render_busy and tid != cur) else "normal"))
        except Exception:
            pass
        try:
            busy = self._is_any_busy()
            color = COLORS["error"] if busy else COLORS["success"]
            label = "BUSY" if busy else "IDLE"
            bg = COLORS["bg_deep"]
            c = self._status_canvas
            c.delete("all")
            w = int(c["width"]); h = int(c["height"])
            cy = h // 2
            dx = 13
            d = 9
            # soft glow: concentric rings fading from the background up to the
            # status colour (drawn outer→inner so the brightest sits nearest).
            for gd, t in ((d + 12, 0.16), (d + 8, 0.32), (d + 4, 0.55)):
                c.create_oval(dx - gd / 2, cy - gd / 2, dx + gd / 2, cy + gd / 2,
                              fill=self._lerp_color(bg, color, t), outline="")
            # outer frame (the warning-light surround)
            c.create_rectangle(1, 1, w - 1, h - 1, outline=color, width=2)
            # the lit dot
            c.create_oval(dx - d / 2, cy - d / 2, dx + d / 2, cy + d / 2,
                          fill=color, outline=color)
            # the word — centred in the gap between the dot's right edge and the
            # right edge of the frame
            text_cx = ((dx + d / 2) + (w - 2)) / 2
            c.create_text(text_cx, cy + 1, text=label, anchor="center",
                          fill=color, font=(FONT_FAMILY, 9, "bold"))
        except Exception:
            pass
        self.master.after(500, self._update_status_indicator)

    def _open_console_popup(self):
        """Open (or raise) the console log popup window."""
        if self._console_popup is not None:
            try:
                if self._console_popup.winfo_exists():
                    self._console_popup.lift()
                    return
            except Exception:
                pass
            self._console_popup = None

        win = tk.Toplevel(self.master)
        win.title("Fizgig — Console Log")
        win.geometry("900x500")
        win.minsize(400, 200)
        win.configure(bg=COLORS["bg_deep"])

        text = tk.Text(win, bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                       font=("Consolas", 9), wrap="word", state="disabled",
                       selectbackground=COLORS["accent"], borderwidth=0,
                       padx=12, pady=8)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate with existing history
        text.configure(state="normal")
        text.insert(tk.END, "".join(self._log_buffer))
        text.see(tk.END)
        text.configure(state="disabled")

        self._console_popup = win
        self._console_popup_text = text

        def _on_close():
            self._console_popup = None
            self._console_popup_text = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _on_caption_folder_changed(self, *args):
        """Reset caption images loaded flag when folder changes"""
        self.caption_images_loaded = False

    def on_tab_changed(self, event):
        """Handle notebook tab changes"""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        # When Captions tab is selected, load images if folder is set
        if tab_text == "3. Captions":
            folder = self.image_folder_var.get()
            if folder and os.path.isdir(folder) and not self.caption_images_loaded:
                self.refresh_caption_images()
                self.caption_images_loaded = True

        # When leaving Repair Studio / Explorer / Royale, unload pipeline to free VRAM —
        # but NEVER unload an engine that's mid-render on a worker thread (resetting it
        # under the worker hard-hangs the app). Leave it loaded; it frees on a later
        # idle switch. Tab switching is also locked while rendering (see status poll),
        # so this is a belt-and-suspenders guard.
        if tab_text != "Repair Studio" and not getattr(self, '_repair_preview_in_flight', False):
            self._unload_repair_studio_models()
        if tab_text != "LoRA the Explorer" and not getattr(self, '_explorer_generating', False):
            self._unload_explorer_models()
        if tab_text != "LoRA Royale" and not self._royale_is_busy():
            self._royale_unload()

        # Entering a heavy-engine tab releases the warm caption worker (~8 GB Qwen): those
        # engines are 10-20 GB each and plan against free VRAM at load. Guarded on a job in
        # flight, like every other unload here. Elsewhere the worker deliberately stays warm
        # (fast Regenerate) until Unload / Start Training / app close.
        if (tab_text in ("Repair Studio", "LoRA the Explorer", "LoRA Royale")
                and self._caption_worker_alive()
                and not getattr(self, "_captioning_running", False)):
            self.update_caption_log("Caption model released (freeing VRAM for "
                                    f"{tab_text}).\n")
            self._stop_caption_worker_async(lambda: None, graceful=False)

    def remove_focus(self, event):
        """Remove focus from active widget when clicking background"""
        self.master.focus_set()

    def _open_in_file_manager(self, path: str):
        """Open a file or folder in the OS's native file manager.

        Uses os.startfile on Windows, `open` on macOS, and `xdg-open` on Linux.
        `os.name == 'posix'` matches both macOS and Linux, so we fall back to
        sys.platform for the Mac/Linux split.
        """
        try:
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Open Failed", f"Could not open:\n{path}\n\n{e}")

    # Setup-area fields remembered across restarts for the two hands-on workbench tabs.
    # (attr, last_used key) — one table drives restore, the save stanza, and the traces.
    _WORKBENCH_REMEMBER = [
        ("repair_primary_var", "repair_primary"),
        ("repair_donor_var", "repair_donor"),
        ("repair_prompt_var", "repair_prompt"),
        ("repair_seed_var", "repair_seed"),
        ("repair_res_var", "repair_res"),
        ("repair_turbo_var", "repair_turbo"),
        ("repair_ref_path_var", "repair_ref_path"),
        ("repair_ref_mp_var", "repair_ref_mp"),
        ("repair_ref_strength_var", "repair_ref_strength"),
        ("repair_metrics_ref_var", "repair_metrics_ref"),
        ("explorer_lora_var", "explorer_lora"),
        ("explorer_prompt_var", "explorer_prompt"),
        ("explorer_ref_path_var", "explorer_ref_path"),
        ("explorer_ref_mp_var", "explorer_ref_mp"),
        ("explorer_ref_strength_var", "explorer_ref_strength"),
        ("explorer_seed_var", "explorer_seed"),
        ("explorer_res_var", "explorer_res"),
        ("explorer_intensity_var", "explorer_intensity"),
        ("explorer_mutations_var", "explorer_mutations"),
        ("explorer_structure_var", "explorer_structure"),
    ]

    def _restore_workbench_setup_fields(self):
        """Restore remembered Repair Studio / Explorer Setup values, then attach debounced
        save traces so edits persist without needing a clean app close. Restore is safe at
        startup: every var trace on these fields no-ops while its engine is unloaded."""
        for attr, key in self._WORKBENCH_REMEMBER:
            var = getattr(self, attr, None)
            if var is None or key not in self.last_used:
                continue
            try:
                var.set(self.last_used[key])
            except Exception:
                pass
        # Traces AFTER restore, so restoring doesn't immediately rewrite the file N times.
        self._workbench_save_after = None

        def _debounced_save(*_):
            if self._workbench_save_after is not None:
                try:
                    self.master.after_cancel(self._workbench_save_after)
                except Exception:
                    pass
            self._workbench_save_after = self.master.after(600, self._save_last_used_paths)

        for attr, _key in self._WORKBENCH_REMEMBER:
            var = getattr(self, attr, None)
            if var is not None:
                try:
                    var.trace_add("write", _debounced_save)
                except Exception:
                    pass

    def _save_last_used_paths(self, *args):
        """Save last-used folder paths and settings to config file"""
        if _persist_disabled():
            return
        # Seed from what's already remembered, THEN overwrite with live widget values. Keys that
        # are set directly on self.last_used (the per-tab Klein/Krea 2 family selectors, for
        # instance) have no widget stanza below, so building `data` from scratch silently dropped
        # them on the very next save — the setting looked persisted until you restarted.
        data = dict(self.last_used) if isinstance(getattr(self, "last_used", None), dict) else {}
        # Caption model / task / token budget — these were read at widget-creation time and never
        # written back, so every restart reset them. Now that the model choice matters (Florence
        # vs Qwen3-VL), losing it silently would be a real annoyance.
        for _attr, _key in (("caption_model_var", "caption_model"),
                            ("caption_task_var", "caption_task"),
                            ("caption_max_tokens_var", "caption_max_tokens")):
            _v = getattr(self, _attr, None)
            if _v is not None:
                try:
                    data[_key] = _v.get()
                except Exception:
                    pass
        # Per-model task memory. The flat "caption_task" above is still written so an older
        # build (or a downgrade) finds something sensible rather than nothing.
        if getattr(self, "_caption_task_memory", None):
            data["caption_tasks"] = dict(self._caption_task_memory)
        data.update({
            "prep_mode": self.prep_mode_var.get(),
            "prep_replace_originals": bool(self.delete_originals_var.get()),
            "prep_megapixels": self.prep_megapixels_var.get(),
            "image_folder": self.image_folder_var.get(),
            "image_folder2": (self._concept_folder_vars[0].get()
                              if getattr(self, "_concept_folder_vars", None) else ""),
            "caption_trigger": self.caption_text_var.get(),
            "dataset_cache_dir": self.dataset_cache_dir_var.get(),
        })
        # Save architecture if variable exists
        if hasattr(self, 'architecture_var'):
            data["architecture"] = self.architecture_var.get()
        # Save sample prompt if widget exists
        if hasattr(self, 'sample_prompt_text'):
            data["sample_prompt"] = self.sample_prompt_text.get("1.0", tk.END).strip()
        # Save the sample reference image path
        if hasattr(self, 'sample_ref_image_var'):
            data["sample_ref_image"] = self.sample_ref_image_var.get()
        if hasattr(self, 'sample_frames_var'):
            data["sample_frames"] = self.sample_frames_var.get()
        # Krea 2 preview engine (Samples tab) — stored canonical, not the display label
        if hasattr(self, 'krea2_preview_engine_var'):
            data["krea2_preview_engine"] = self._krea2_preview_engine()
        # Repair Studio / Explorer Setup fields (one shared table drives restore + save)
        for _attr, _key in self._WORKBENCH_REMEMBER:
            _var = getattr(self, _attr, None)
            if _var is not None:
                try:
                    data[_key] = _var.get()
                except Exception:
                    pass
        # Save LoRA output directory if entry exists
        if "LORA_OUTPUT_DIR" in self.entries:
            data["lora_output_dir"] = self.entries["LORA_OUTPUT_DIR"].get()
        # Remember the last LoRA Royale checkpoint folder + render inputs
        if hasattr(self, 'royale_folder_var'):
            data["royale_folder"] = self.royale_folder_var.get()
        if hasattr(self, 'royale_mode_var'):
            data["royale_mode"] = self.royale_mode_var.get()
            data["royale_single"] = self.royale_single_var.get()
        if hasattr(self, 'royale_prompt_var'):
            data["royale_prompt"] = self.royale_prompt_var.get()
            data["royale_seed"] = self.royale_seed_var.get()
            data["royale_max"] = self.royale_max_var.get()
            data["royale_ref"] = self.royale_ref_var.get()
            data["royale_ref_strength"] = self.royale_ref_strength_var.get()
            data["royale_w"] = self.royale_w_var.get()
            data["royale_h"] = self.royale_h_var.get()
        if hasattr(self, 'royale_like_ref_var'):
            data["royale_like_ref"] = self.royale_like_ref_var.get()
        if hasattr(self, 'royale_travel_seed_a_var'):
            data["royale_travel_seed_a"] = self.royale_travel_seed_a_var.get()
            data["royale_travel_seed_b"] = self.royale_travel_seed_b_var.get()
            data["royale_travel_w"] = self.royale_travel_w_var.get()
            data["royale_travel_h"] = self.royale_travel_h_var.get()
            data["royale_travel_ref"] = self.royale_travel_ref_var.get()
            data["royale_travel_use_epoch_ref"] = bool(self.royale_travel_use_epoch_ref_var.get())
            data["royale_travel_ref_strength"] = self.royale_travel_ref_strength_var.get()
            data["royale_travel_ref_mp"] = self.royale_travel_ref_mp_var.get()
            data["royale_travel_seq_ref"] = bool(self.royale_travel_seq_ref_var.get())
            data["royale_travel_waypoints"] = self.royale_travel_waypoints_var.get()
        if hasattr(self, 'royale_lora_start_var'):
            data["royale_lora_start"] = self.royale_lora_start_var.get()
            data["royale_lora_end"] = self.royale_lora_end_var.get()
            data["royale_lora_frames"] = self.royale_lora_frames_var.get()
            data["royale_lora_w"] = self.royale_lora_w_var.get()
            data["royale_lora_h"] = self.royale_lora_h_var.get()
        if hasattr(self, 'royale_cmp_mode_var'):
            data["royale_cmp_mode"] = self.royale_cmp_mode_var.get()
            data["royale_cmp_trigger"] = self.royale_cmp_trigger_var.get()
            data["royale_cmp_seed"] = self.royale_cmp_seed_var.get()
            data["royale_cmp_w"] = self.royale_cmp_w_var.get()
            data["royale_cmp_h"] = self.royale_cmp_h_var.get()
            data["royale_cmp_epochs"] = self.royale_cmp_epochs_var.get()
            try:
                data["royale_cmp_prompts"] = self.royale_cmp_prompts.get("1.0", tk.END).strip()
            except Exception:
                pass
        if hasattr(self, 'royale_pt_prompt_var'):
            data["royale_pt_prompt"] = self.royale_pt_prompt_var.get()
            data["royale_pt_dim"] = self.royale_pt_dim_var.get()
            data["royale_pt_custom"] = self.royale_pt_custom_var.get()
            data["royale_pt_frames"] = self.royale_pt_frames_var.get()
            data["royale_pt_ref"] = self.royale_pt_ref_var.get()
            data["royale_pt_w"] = self.royale_pt_w_var.get()
            data["royale_pt_h"] = self.royale_pt_h_var.get()
            data["royale_pt_use_epoch_ref"] = bool(self.royale_pt_use_epoch_ref_var.get())
            data["royale_pt_ref_strength"] = self.royale_pt_ref_strength_var.get()
            data["royale_pt_vary_seed"] = bool(self.royale_pt_vary_seed_var.get())
            data["royale_pt_ref_mp"] = self.royale_pt_ref_mp_var.get()
            data["royale_pt_seq_ref"] = bool(self.royale_pt_seq_ref_var.get())
            data["royale_pt_anchor"] = bool(self.royale_pt_anchor_var.get())
            data["royale_pt_anchor_str"] = self.royale_pt_anchor_str_var.get()
            data["royale_pt_start"] = self.royale_pt_start_var.get()
            data["royale_pt_end"] = self.royale_pt_end_var.get()
            data["royale_pt_interp"] = self.royale_pt_interp_var.get()
            data["royale_pt_drift"] = self.royale_pt_drift_var.get()
            data["royale_pt_subject"] = self.royale_pt_subject_var.get()
        # Remember whether the bottom status bar is shown
        data["status_bar_visible"] = bool(getattr(self, "_status_bar_visible", True))
        save_last_used(data)

    def _save_pref(self, key):
        """Save a single pref value back to prefs.json."""
        if key in self.prefs_vars:
            self.prefs[key] = self.prefs_vars[key].get()
            save_prefs(self.prefs)

    # ------------------------------------------------------------------
    # Live VRAM / RAM status bar (bottom of window)
    # ------------------------------------------------------------------
    def _build_status_bar(self, master):
        """Bottom status panel: stacked VRAM + system-RAM gradient fill bars (with
        per-run peak ticks) on the left, the live sample override on the right,
        and a remembered hide/show toggle. A daemon thread does the reads so the
        Tk redraw never stalls on an nvidia-smi call."""
        container = tk.Frame(master, bg=COLORS["bg_deep"])
        container.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_container = container

        # Thin always-visible handle carrying the show/hide toggle.
        handle = tk.Frame(container, bg=COLORS["bg_deep"])
        handle.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_handle = handle
        self._status_toggle_btn = tk.Button(
            handle, text="▾ Hide stats", font=(FONT_FAMILY, 8),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"],
            activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
            relief="flat", bd=0, padx=10, pady=1, cursor="hand2",
            command=self._toggle_status_bar)
        self._status_toggle_btn.pack(side=tk.RIGHT, padx=(0, 12))

        # The expandable bar (sits above the handle).
        bar = tk.Frame(container, bg=COLORS["bg_deep"], height=82)
        bar.pack(side=tk.BOTTOM, fill=tk.X, before=handle)
        bar.pack_propagate(False)
        self._status_bar_frame = bar

        # --- left: stacked VRAM (top) + RAM (bottom) gradient bars ---
        bars_col = tk.Frame(bar, bg=COLORS["bg_deep"])
        bars_col.pack(side=tk.LEFT, padx=(14, 18), pady=10)
        self._vram_canvas = tk.Canvas(bars_col, width=360, height=27, bg=COLORS["bg_surface"],
                                      highlightthickness=0)
        self._vram_canvas.pack(side=tk.TOP, pady=(0, 6))
        self._ram_canvas = tk.Canvas(bars_col, width=360, height=27, bg=COLORS["bg_surface"],
                                     highlightthickness=0)
        self._ram_canvas.pack(side=tk.TOP)

        # --- far right: training-queue button (lower-right corner of the app) ---
        # Packed BEFORE the override panel so it owns the corner; the override panel's
        # expand soaks up whatever is left in the middle.
        # fill=Y on both this column and the override panel below is what makes the two
        # blocks exactly the same height (the bar is a fixed 82 px with pack_propagate off,
        # so each ends up 82 - 2*pady). Without it each block sizes to its own content and
        # the button sat visibly shorter than the panel beside it.
        qcol = tk.Frame(bar, bg=COLORS["bg_deep"])
        qcol.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 14), pady=10)
        self._queue_btn = tk.Button(
            qcol, text="📋 Queue", font=(FONT_FAMILY, 9, "bold"),
            bg=COLORS["queue_blue"], fg=COLORS["bg_deep"],
            activebackground=COLORS["queue_blue_hover"], activeforeground=COLORS["bg_deep"],
            relief="flat", bd=0, padx=12, cursor="hand2",
            command=self._open_queue_window)
        self._queue_btn.pack(fill=tk.BOTH, expand=True)
        self._refresh_queue_button()

        # --- right: live sample override (surface-coloured mini panel) ---
        # Widths trimmed vs the original layout to make room for the queue button.
        ov = tk.Frame(bar, bg=COLORS["bg_surface"])
        ov.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10, ipadx=8, ipady=4)
        _sbg = COLORS["bg_surface"]
        r1 = tk.Frame(ov, bg=_sbg); r1.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.sample_override_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="Override next sample", variable=self.sample_override_var,
                        command=self._on_sample_override_changed,
                        style="Surface.TCheckbutton").pack(side=tk.LEFT)
        tk.Label(r1, text="seed", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(10, 3))
        self.sample_override_seed_var = tk.StringVar(value="1234")
        ttk.Entry(r1, textvariable=self.sample_override_seed_var, width=6).pack(side=tk.LEFT)
        # Same list as the Samples tab (SAMPLE_RESOLUTIONS) — these two had drifted, and
        # the override's lower ceiling silently downgraded a 1280/1536 preview.
        _res_vals = SAMPLE_RESOLUTIONS
        tk.Label(r1, text="W", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_w_var = tk.StringVar(value="768")
        ttk.Combobox(r1, textvariable=self.sample_override_w_var, values=_res_vals,
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(r1, text="H", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_h_var = tk.StringVar(value="768")
        ttk.Combobox(r1, textvariable=self.sample_override_h_var, values=_res_vals,
                     state="readonly", width=5).pack(side=tk.LEFT)
        # Reference image — auto-capped to ~0.20 MP by the trainer so a big image can't OOM the
        # sample. Shown for BOTH families: Klein uses it as edit conditioning, Krea 2 routes it
        # through the Qwen3-VL vision path. (This comment used to claim it was hidden under
        # Krea 2 — it never was; there is no hide call for these widgets anywhere.)
        self._override_ref_caption = tk.Label(r1, text="Ref", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8))
        self._override_ref_caption.pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_ref_var = tk.StringVar(value="")
        # Compact button so it matches the seed/resolution input height (the
        # default ttk.Button padding is taller and pushes the prompt row down).
        ttk.Style().configure("OverrideRef.TButton", padding=(8, 0), font=(FONT_FAMILY, 8))
        self._override_ref_browse_btn = ttk.Button(r1, text="Browse…", width=9, style="OverrideRef.TButton",
                   command=self._browse_override_ref)
        self._override_ref_browse_btn.pack(side=tk.LEFT)
        self._override_ref_label = tk.Label(r1, text="(none)", bg=_sbg,
                                            fg=COLORS["text_muted"], font=(FONT_FAMILY, 8))
        self._override_ref_label.pack(side=tk.LEFT, padx=(6, 2))
        self._override_ref_clear_btn = tk.Button(r1, text="✕", font=(FONT_FAMILY, 8), bg=_sbg, fg=COLORS["text_muted"],
                  activebackground=COLORS["border"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, cursor="hand2",
                  command=self._clear_override_ref)
        self._override_ref_clear_btn.pack(side=tk.LEFT)
        r2 = tk.Frame(ov, bg=_sbg); r2.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(r2, text="Prompt", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(0, 6))
        self.sample_override_prompt_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.sample_override_prompt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        for _v in (self.sample_override_prompt_var, self.sample_override_seed_var,
                   self.sample_override_w_var, self.sample_override_h_var,
                   self.sample_override_ref_var):
            _v.trace_add("write", lambda *a: self._on_sample_override_changed())

        self._vram_peak = 0
        self._ram_peak = 0
        self._status_latest = (None, None)
        self._status_stop = False
        # Restore remembered visibility (default shown).
        self._status_bar_visible = bool(self.last_used.get("status_bar_visible", True))
        if not self._status_bar_visible:
            bar.pack_forget()
            self._status_toggle_btn.configure(text="▴ Show stats")
        import threading
        self._status_thread = threading.Thread(target=self._status_reader_loop, daemon=True)
        self._status_thread.start()
        self.master.after(800, self._poll_status_bar)

    def _toggle_status_bar(self):
        """Show/hide the stats bar; remember the choice across launches."""
        self._status_bar_visible = not getattr(self, "_status_bar_visible", True)
        if self._status_bar_visible:
            self._status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X, before=self._status_handle)
            self._status_toggle_btn.configure(text="▾ Hide stats")
        else:
            self._status_bar_frame.pack_forget()
            self._status_toggle_btn.configure(text="▴ Show stats")
        try:
            self._save_last_used_paths()
        except Exception:
            pass

    def _visible_gpu_index(self):
        """The physical GPU index training will actually use.

        NVML and nvidia-smi index every card in the machine; CUDA_VISIBLE_DEVICES does not
        change that, it only changes what torch can see. So on a two-GPU box with
        CUDA_VISIBLE_DEVICES=1, training runs on physical card 1 while an unqualified NVML read
        reports card 0 — the status bar then shows a card that is doing nothing (issue #60).
        Torch's cuda:0 is the FIRST entry in the list, hence [0]. When CUDA_VISIBLE_DEVICES
        holds a UUID (new format), map it back to NVML index via _gpu_info."""
        raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        if raw.isdigit():
            return int(raw)
        # UUID — look up NVML index from _gpu_info
        for _k, _info in getattr(self, "_gpu_info", {}).items():
            if _info[3] == raw:
                return _info[0]
        return 0

    def _read_vram(self):
        """Return (used_bytes, total_bytes) for the GPU training uses, or None. Prefers pynvml
        (fast); falls back to a one-shot nvidia-smi query. AMD ROCm paths are
        tried only when NVIDIA readers return nothing."""
        try:
            import pynvml
            if not getattr(self, "_nvml_init", False):
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._visible_gpu_index())
                self._nvml_init = True
            m = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return int(m.used), int(m.total)
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self._visible_gpu_index()),
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            used, total = out.stdout.strip().splitlines()[0].split(",")
            return int(used) * 1024 * 1024, int(total) * 1024 * 1024
        except Exception:
            pass
        try:
            from fizgig.utils.vram_monitor import read_amd_gpu_vram
            return read_amd_gpu_vram()
        except Exception:
            return None

    def _status_reader_loop(self):
        import time
        while not getattr(self, "_status_stop", False):
            vram = self._read_vram()
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram = (vm.total - vm.available, vm.total)
            except Exception:
                ram = None
            self._status_latest = (vram, ram)
            time.sleep(1.0)

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, t))
        r = round(int(c1[1:3], 16) + (int(c2[1:3], 16) - int(c1[1:3], 16)) * t)
        g = round(int(c1[3:5], 16) + (int(c2[3:5], 16) - int(c1[3:5], 16)) * t)
        b = round(int(c1[5:7], 16) + (int(c2[5:7], 16) - int(c1[5:7], 16)) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_status_segment(self, canvas, used, total, peak, label, c_start, c_end):
        canvas.delete("all")
        try:
            w = int(canvas["width"]); h = int(canvas["height"])
        except Exception:
            return
        frac = max(0.0, min(1.0, used / total)) if total else 0.0
        # track
        canvas.create_rectangle(0, 0, w, h, fill=COLORS["bg_deep"], outline="")
        # gradient fill: colour interpolates c_start -> c_end across the FULL
        # width, drawn up to the current fill (fuller = closer to c_end).
        fill_w = int(w * frac)
        step = 3
        for x in range(0, fill_w, step):
            col = self._lerp_color(c_start, c_end, x / max(1, w - 1))
            canvas.create_rectangle(x, 1, min(x + step, fill_w), h - 1, fill=col, outline="")
        # per-run peak tick
        if peak and total:
            px = int(w * max(0.0, min(1.0, peak / total)))
            canvas.create_line(px, 0, px, h, fill="#FFFFFF", width=2)
        canvas.create_text(10, h // 2,
                           text=(f"{label}  {used/1073741824:.1f} / {total/1073741824:.1f} GB"
                                 f" · peak {peak/1073741824:.1f}"),  # GiB (binary) — matches '32 GB' labels
                           anchor="w", fill="#FFFFFF", font=(FONT_FAMILY, 9, "bold"))

    def _poll_status_bar(self):
        vram, ram = getattr(self, "_status_latest", (None, None))
        visible = getattr(self, "_status_bar_visible", True)
        if vram:
            u, t = vram
            self._vram_peak = max(self._vram_peak, u)
            if visible:
                self._draw_status_segment(self._vram_canvas, u, t, self._vram_peak,
                                          "VRAM", "#3FB950", "#E5534B")  # green → red
        if ram:
            u, t = ram
            self._ram_peak = max(self._ram_peak, u)
            if visible:
                self._draw_status_segment(self._ram_canvas, u, t, self._ram_peak,
                                          "RAM", "#3B82F6", "#EAC54F")   # blue → yellow
        self.master.after(1000, self._poll_status_bar)

    def reset_status_peaks(self):
        """Zero the VRAM/RAM peak markers — call at the start of a training run."""
        self._vram_peak = 0
        self._ram_peak = 0

    def _sample_override_path(self):
        # Prefer the live entry (always current) so this matches the --output_dir
        # the trainer is launched with, even before settings is synced.
        out_dir = ""
        try:
            out_dir = self.entries["LORA_OUTPUT_DIR"].get().strip()
        except Exception:
            pass
        if not out_dir:
            out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".sample_override.json")

    def _on_sample_override_changed(self):
        """Write or remove the live sample-override sentinel the trainer reads.

        Active when the toggle is on AND there's a prompt OR a reference image. A reference with
        no prompt is a valid Krea 2 'generate from this picture' override (routed through the
        Qwen3-VL vision path); the Klein trainer still ignores prompt-less overrides (it requires
        a prompt). Off removes the sentinel → samples fall back to the Samples tab."""
        path = self._sample_override_path()
        try:
            active = self.sample_override_var.get() and (
                self.sample_override_prompt_var.get().strip()
                or self.sample_override_ref_var.get().strip())
            if active:
                try:
                    seed = int(self.sample_override_seed_var.get() or "1234")
                except ValueError:
                    seed = 1234
                try:
                    width = int(self.sample_override_w_var.get() or "768")
                except ValueError:
                    width = 768
                try:
                    height = int(self.sample_override_h_var.get() or "768")
                except ValueError:
                    height = 768
                data = {"prompt": self.sample_override_prompt_var.get().strip(),
                        "seed": seed, "width": width, "height": height,
                        "ref_image": self.sample_override_ref_var.get().strip()}
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                # Atomic: the trainer polls this file at epoch boundaries, and a plain write
                # can be caught half-finished mid-keystroke — unparseable JSON reads as "no
                # override" for that epoch.
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _browse_override_ref(self):
        """Pick a reference image for the live sample override (Klein edit
        conditioning). The trainer auto-caps it to ~0.20 MP, so any size is safe."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image for samples",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.sample_override_ref_var.set(path)
            self._override_ref_label.configure(text=os.path.basename(path)[:24])
            self._on_sample_override_changed()

    def _clear_override_ref(self):
        self.sample_override_ref_var.set("")
        self._override_ref_label.configure(text="(none)")
        self._on_sample_override_changed()

    def _browse_sample_ref(self):
        """Pick the persistent reference image for training samples (Samples tab)."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image for samples",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.sample_ref_image_var.set(path)

    def create_training_settings(self):
        """Create the Training tab (Start-tab styled)."""
        scrollable_frame, self.training_canvas = self.create_scrollable_frame(self.training_tab)

        # Outer bg_deep container — all sections pack into this so the banner
        # and collapsibles share the same horizontal alignment.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Store collapsible sections for later access
        self.collapsible_sections = {}

        self._add_tab_banner(
            outer,
            "Training",
            "Pick a preset or dial in a custom run. Sections below collapse to reduce clutter — "
            "click any header to toggle. Dataset-config knobs live in Other Options.",
        )

        # Model family selector. Restore the last choice; fall back to Klein if the
        # saved value isn't a known architecture (e.g. a removed/renamed entry).
        _saved_arch = "Flux 2 Klein Base 9B"
        try:
            _candidate = _canon_arch(self.last_used.get("architecture", _saved_arch))
            if _candidate in ARCHITECTURE_LIST:
                _saved_arch = _candidate
        except Exception:
            pass
        self.architecture_var = tk.StringVar(value=_saved_arch)
        # Seeded so _on_architecture_selected can tell a real family change from the user
        # re-picking the entry that's already selected (both fire <<ComboboxSelected>>).
        self._arch_last_selected = _saved_arch
        # Per-family settings memory, session-scoped: architecture -> the Training-tab
        # snapshot it had when you last switched away from it, and the preset label it was
        # showing. First visit to a family falls back to that family's default preset.
        self._arch_settings_memory = {}
        self._arch_preset_name_memory = {}

        # === Base Model card (only shown when more than one architecture is available) ===
        if len(ARCHITECTURE_LIST) > 1:
            model_card = self._start_section_card(
                outer, "Base Model",
                "Pick the model family to train. Krea 2 needs its model paths set on the "
                "Preferences tab before training.",
            )
            model_row = tk.Frame(model_card, bg=COLORS["bg_surface"])
            model_row.pack(anchor=tk.W)
            tk.Label(
                model_row, text="Model:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            arch_combo = ttk.Combobox(
                model_row, textvariable=self.architecture_var, state="readonly",
                width=28, values=ARCHITECTURE_LIST,
            )
            arch_combo.pack(side=tk.LEFT)
            arch_combo.bind("<<ComboboxSelected>>", self._on_architecture_selected)
            ToolTip(arch_combo, "Model family to train (Klein 9B, Krea 2 or MiniMax H3)")

            # Training Base (MiniMax only) — which H3 fine-tune the run trains against, right
            # where the family was just chosen. A dedicated var kept OUT of self.entries and
            # never collected, so presets can't flip it; last-train and the queue carry it
            # explicitly. Shown/hidden by _apply_training_arch_visibility alongside the note.
            self.minimax_train_base_var = tk.StringVar(
                value=MINIMAX_TRAIN_BASE_OPTIONS[
                    1 if minimax_train_base(self.settings.get("MINIMAX_TRAIN_BASE")) == "ref2va"
                    else 0])
            self._minimax_base_frame = tk.Frame(model_card, bg=COLORS["bg_surface"])
            tk.Label(
                self._minimax_base_frame, text="Training Base:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            self._minimax_base_combo = ttk.Combobox(
                self._minimax_base_frame, textvariable=self.minimax_train_base_var,
                values=list(MINIMAX_TRAIN_BASE_OPTIONS), state="readonly", width=36)
            self._minimax_base_combo.pack(side=tk.LEFT)
            self._minimax_base_hint = tk.Label(
                model_card,
                text="Pick the H3 model you deploy on. First/last frame (fl2va) is the standard "
                     "model most workflows run. Reference (ref2va) is the Reference-to-Video "
                     "fine-tune — choose it if your LoRA's home is the r2v workflow (needs 'DiT "
                     "(reference)' set in Preferences). Presets never change this; reference "
                     "distillation always trains on ref2va regardless.",
                font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                bg=COLORS["bg_surface"], wraplength=760, justify=tk.LEFT)
            self._minimax_base_frame.pack(anchor=tk.W, pady=(10, 0))
            self._minimax_base_hint.pack(anchor=tk.W, pady=(2, 0))
            if not self._is_minimax_arch():
                self._minimax_base_frame.pack_forget()
                self._minimax_base_hint.pack_forget()

            # Previews track likeness honestly but are not the place to compare quality — say
            # so where the family is chosen, along with the Pause/Resume route that makes
            # judging in ComfyUI practical on one GPU.
            self._minimax_sample_note = tk.Label(
                model_card,
                text=("⏱ Previews track LIKENESS, not quality. Judge quality in ComfyUI — Pause "
                      "frees the GPU, so you can check an epoch there and Resume.\n"
                      "Defaults are 768×768 56-frame clips with sound; Sample length has "
                      "stills and other lengths. 📖 Full write-ups in the README."),
                font=(FONT_FAMILY, 9), fg=COLORS["warning"], bg=COLORS["bg_surface"],
                wraplength=760, justify=tk.LEFT,
            )
            self._minimax_sample_note.pack(anchor=tk.W, pady=(10, 0))
            if not self._is_minimax_arch():
                self._minimax_sample_note.pack_forget()

        # === Presets card ===
        preset_card = self._start_section_card(
            outer, "Presets",
            "Save the current settings under a name, load a saved preset, or restore the exact "
            "configuration from your last training run.",
        )
        # Row 1: Save + Load Preset
        preset_row1 = tk.Frame(preset_card, bg=COLORS["bg_surface"])
        preset_row1.pack(anchor=tk.W, pady=(0, 8))
        save_preset_btn = ttk.Button(preset_row1, text="Save Preset", command=self.save_custom_preset)
        save_preset_btn.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(save_preset_btn, "Save current training parameters as a named preset")
        tk.Label(
            preset_row1, text="Load Preset:",
            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.custom_preset_var = tk.StringVar()
        self.custom_preset_combo = ttk.Combobox(preset_row1, textvariable=self.custom_preset_var, state="readonly", width=28)
        self.custom_preset_combo.pack(side=tk.LEFT)
        self.custom_preset_combo.bind("<<ComboboxSelected>>", self.load_custom_preset)
        ToolTip(self.custom_preset_combo, "Your saved training presets")
        # Bracketed nudge for the rank-16 recipe, shown only while the MiniMax Defaults preset
        # is selected: Fast is the default now, and this says when the bigger one earns its keep.
        self._preset_hint_label = tk.Label(
            preset_row1, text="(more suitable for larger datasets with longer trains)",
            font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
        )
        self.custom_preset_var.trace_add("write", lambda *_: self._update_preset_hint())

        # Row 2: Load Settings From Last Train
        load_last_btn = ttk.Button(preset_card, text="Load Settings From Last Train",
                                   command=self._load_last_train_settings, width=32)
        load_last_btn.pack(anchor=tk.W)
        ToolTip(load_last_btn, "Restore the exact settings used in your most recent training launch")

        # === Output Section (Expanded by default) ===
        output_section = CollapsibleFrame(outer,"Output", default_expanded=True)
        output_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["output"] = output_section

        output_content = output_section.get_content_frame()
        output_content.columnconfigure(1, weight=1)

        self._add_field_to_section(output_content, "LORA_OUTPUT_DIR", "Output Directory", "directory", 0)
        self._add_field_to_section(output_content, "LORA_NAME", "LoRA Name", "text", 1)

        # Save LoRA output directory when it changes
        self.entries["LORA_OUTPUT_DIR"].bind("<FocusOut>", lambda e: self._save_last_used_paths())
        self.entries["LORA_OUTPUT_DIR"].bind("<Return>", lambda e: self._save_last_used_paths())

        # === Training Parameters Section (Expanded by default) ===
        training_section = CollapsibleFrame(outer,"Training Parameters", default_expanded=True)
        training_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["training"] = training_section

        training_content = training_section.get_content_frame()
        training_content.columnconfigure(1, weight=1)

        self._add_field_to_section(training_content, "MODEL_TYPE", "Model Type", "dropdown", 0)
        self._add_field_to_section(training_content, "LEARNING_RATE", "Learning Rate", "float", 1)

        # --- Adaptive LR (bi-directional plateau tracker) — placed under Learning Rate so both bracket the starting LR
        self.adaptive_lr_var = tk.BooleanVar(value=False)
        adaptive_cb = ttk.Checkbutton(
            training_content, text="Adaptive LR (auto-adjust based on loss, gradient clipping & weight-norm growth)",
            variable=self.adaptive_lr_var, command=self._on_adaptive_lr_toggle,
        )
        adaptive_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(4, 0))
        self._adaptive_cb = adaptive_cb

        adaptive_frame = ttk.Frame(training_content)
        adaptive_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 2))
        self._adaptive_frame = adaptive_frame
        ttk.Label(adaptive_frame, text="Min LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MIN"] = ttk.Combobox(adaptive_frame, width=28, values=["1e-5", "5e-5", "1e-4", "2e-4 - rank 4/8 only", "3e-4 - low-rank only"], state="readonly")
        self.entries["ADAPTIVE_LR_MIN"].set("1e-5")
        self.entries["ADAPTIVE_LR_MIN"].pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(adaptive_frame, text="Max LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MAX"] = ttk.Combobox(adaptive_frame, width=8, values=["1e-4", "2e-4", "3e-4", "4e-4"], state="readonly")
        self.entries["ADAPTIVE_LR_MAX"].set("4e-4")
        self.entries["ADAPTIVE_LR_MAX"].pack(side=tk.LEFT, padx=(0, 12))
        self._adaptive_reset_btn = ttk.Button(adaptive_frame, text="Reset Defaults", command=self._reset_adaptive_lr_defaults)
        self._adaptive_reset_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._adaptive_desc_label = ttk.Label(training_content,
                  text="When on, the Learning Rate box is IGNORED (greyed out) — the run starts at the geometric "
                       "midpoint of Min/Max (e.g. 1e-4 & 4e-4 → 2e-4) and the watcher owns the LR from there: "
                       "probes UP on steady loss descent; reduces DOWN on loss plateau, heavy gradient clipping, "
                       "or runaway weight-norm growth (with a rollback to the previous epoch's weights on "
                       "stability events).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._adaptive_desc_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 6))
        self._on_adaptive_lr_toggle()  # sync initial enabled/disabled state

        # LoRA LR Ratio — hidden, always 1 (LoRA+ default). Widget exists for preset/save compat.
        self.entries["LORA_LR_RATIO"] = ttk.Entry(training_content, width=12)
        self.entries["LORA_LR_RATIO"].insert(0, str(self.settings["LORA_LR_RATIO"]))
        self._add_field_to_section(training_content, "NETWORK_DIM", "Network Dim (Rank)", "int", 5)
        self._add_field_to_section(training_content, "NETWORK_ALPHA", "Network Alpha", "float", 6)
        self._add_field_to_section(training_content, "MAX_TRAIN_EPOCHS", "Max Epochs", "int", 7)
        self._add_field_to_section(training_content, "SAVE_EVERY_N_EPOCHS", "Save Every N Epochs", "int", 8)
        self._add_field_to_section(training_content, "SEED", "Seed", "int", 9)

        # Network Type (Krea 2 only): standard LoRA or LoKR (Kronecker). Rows 18/19 sit between
        # the Target Megapixels hint (17) and the Krea 2 loss-watch block (20). LoKR replaces
        # rank/alpha with a single Factor dial, so the rows swap with the selection.
        _nt_label = tk.Label(training_content, text="Network Type:", font=(FONT_FAMILY, 10),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        _nt_label.grid(row=18, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels["NETWORK_TYPE"] = _nt_label
        # Widget + hint share a row frame so the hint hugs the control instead of being
        # pushed to the far column edge by the full-width rows above.
        self._network_type_rowf = tk.Frame(training_content, bg=COLORS["bg_surface"])
        self._network_type_rowf.grid(row=18, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        self.entries["NETWORK_TYPE"] = ttk.Combobox(
            self._network_type_rowf, values=["LoRA (standard)", "LoKR (Kronecker)"],
            state="readonly", width=18)
        self.entries["NETWORK_TYPE"].set(self.settings.get("NETWORK_TYPE", "LoRA (standard)"))
        self.entries["NETWORK_TYPE"].pack(side=tk.LEFT)
        self.entries["NETWORK_TYPE"].bind("<<ComboboxSelected>>",
                                          lambda e: self._on_network_type_changed())
        self._network_type_hint = tk.Label(
            self._network_type_rowf,
            text="LoKR: higher quality · LoRA: ~20% faster training",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT)
        self._network_type_hint.pack(side=tk.LEFT, padx=(10, 0))
        # rows entry is the FRAME (the gridded thing show_row/hide_row must toggle).
        self.rows["NETWORK_TYPE"] = {"row": 18, "label": _nt_label,
                                     "entry": self._network_type_rowf,
                                     "browse": None, "parent": training_content}

        _lf_label = tk.Label(training_content, text="LoKR Factor:", font=(FONT_FAMILY, 10),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        _lf_label.grid(row=19, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels["LOKR_FACTOR"] = _lf_label
        self._lokr_factor_rowf = tk.Frame(training_content, bg=COLORS["bg_surface"])
        self._lokr_factor_rowf.grid(row=19, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        self.entries["LOKR_FACTOR"] = ttk.Entry(self._lokr_factor_rowf, width=8)
        self.entries["LOKR_FACTOR"].insert(0, str(self.settings.get("LOKR_FACTOR", 8)))
        self.entries["LOKR_FACTOR"].pack(side=tk.LEFT)
        self._lokr_factor_hint = tk.Label(
            self._lokr_factor_rowf,
            text="8 is the sweet spot · 4 = stronger, bigger files · above 8: just use LoRA",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT)
        self._lokr_factor_hint.pack(side=tk.LEFT, padx=(10, 0))
        self.rows["LOKR_FACTOR"] = {"row": 19, "label": _lf_label,
                                    "entry": self._lokr_factor_rowf,
                                    "browse": None, "parent": training_content}

        self._build_minimax_structure_row(training_content)

        # Model Area to Train dropdown (blocks + timestep auto-fill)
        self._modelarea_label = ttk.Label(training_content, text="Model Area to Train:")
        self._modelarea_label.grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
        self.training_preset_var = tk.StringVar(value="Full Model")
        training_preset_combo = ttk.Combobox(
            training_content, textvariable=self.training_preset_var,
            values=["Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom"],
            state="readonly", width=20
        )
        training_preset_combo.grid(row=10, column=1, sticky=tk.W, padx=5, pady=2)
        training_preset_combo.bind("<<ComboboxSelected>>", self._on_training_preset_changed)
        self._modelarea_combo = training_preset_combo
        self._modelarea_desc_label = ttk.Label(training_content,
                  text="Identity = single 1-16  |  Style = style+comp blocks @ late ts (0-400)  |  Style+Composition = double 0-7 + single 0-1  |  Details = single 12-23",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"))
        self._modelarea_desc_label.grid(row=11, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Custom block picker panel (hidden unless preset == Custom)
        self._training_custom_frame = ttk.Frame(training_content)
        self._training_custom_frame.grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=15, pady=(4, 4))
        self.training_block_vars = {}  # block_name -> BooleanVar

        tc_header = ttk.Frame(self._training_custom_frame)
        tc_header.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        ttk.Label(tc_header, text="Select blocks to train:",
                  font=(FONT_FAMILY, 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tc_header, text="All", width=5,
                   command=lambda: self._set_all_training_blocks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="None", width=5,
                   command=lambda: self._set_all_training_blocks(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Identity", width=8,
                   command=lambda: self._set_category_training_blocks("identity")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Style+Comp", width=10,
                   command=lambda: self._set_category_training_blocks("style_composition")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Details", width=8,
                   command=lambda: self._set_category_training_blocks("details")).pack(side=tk.LEFT, padx=2)

        tc_double = ttk.Frame(self._training_custom_frame)
        tc_double.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_double, text="Double:", width=8, foreground="#5B9BD5").pack(side=tk.LEFT)
        for i in range(8):
            key = f"double_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_double, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single1 = ttk.Frame(self._training_custom_frame)
        tc_single1.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single1, text="Single:", width=8, foreground="#70AD47").pack(side=tk.LEFT)
        for i in range(12):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single1, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single2 = ttk.Frame(self._training_custom_frame)
        tc_single2.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single2, text="Single:", width=8, foreground="#ED7D31").pack(side=tk.LEFT)
        for i in range(12, 24):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single2, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        ttk.Label(self._training_custom_frame,
                  text="double + single 0-1 = style+composition  |  single 1-16 = identity (overlaps at 1 and 12-16)  |  single 12-23 = details  |  edit MIN/MAX_TIMESTEP on Advanced tab",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).pack(anchor=tk.W, pady=(4, 0))

        self._training_custom_frame.grid_remove()  # hidden until preset == Custom

        # Context LoRA (optional) — train new LoRA with an existing one frozen + active on the base
        self._contextlora_label = ttk.Label(training_content, text="Context LoRA:")
        self._contextlora_label.grid(row=13, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        ctx_frame = ttk.Frame(training_content)
        ctx_frame.grid(row=13, column=1, sticky=tk.W, padx=5, pady=(8, 2))
        self._contextlora_frame = ctx_frame
        self.entries["CONTEXT_LORA_PATH"] = ttk.Entry(ctx_frame, width=42)
        self.entries["CONTEXT_LORA_PATH"].pack(side=tk.LEFT)
        ttk.Button(ctx_frame, text="Browse",
                   command=lambda: self._browse_context_lora()).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(ctx_frame, text="Strength:").pack(side=tk.LEFT, padx=(4, 4))
        self.entries["CONTEXT_LORA_STRENGTH"] = ttk.Entry(ctx_frame, width=6)
        self.entries["CONTEXT_LORA_STRENGTH"].insert(0, "1.0")
        self.entries["CONTEXT_LORA_STRENGTH"].pack(side=tk.LEFT)
        self._contextlora_desc_label = ttk.Label(training_content,
                  text="Train this LoRA with an existing LoRA already active on the base model. "
                       "Pair with same context+strength at inference.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"))
        self._contextlora_desc_label.grid(row=14, column=0, columnspan=2, sticky=tk.W, padx=5)
        self._contextlora_warn_label = ttk.Label(training_content,
                  text="⚠ Context LoRAs usually look better in ComfyUI than in training samples — "
                       "don't worry if previews look rough, test the output LoRA in ComfyUI.",
                  foreground="#E67E22", font=(FONT_FAMILY, 9, "italic"))
        self._contextlora_warn_label.grid(row=15, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Target Megapixels (training resolution) — moved here from Other Options
        ttk.Label(training_content, text="Target Megapixels:").grid(row=16, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        mp_frame = ttk.Frame(training_content)
        mp_frame.grid(row=16, column=1, sticky=tk.W, padx=5, pady=(8, 2))
        self._mp_combo = ttk.Combobox(
            mp_frame, textvariable=self.dataset_megapixels_var,
            values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4", "3.0", "4.2"], width=8)
        self._mp_combo.pack(side=tk.LEFT, padx=(0, 10))
        # Shown (and the combo greyed) when the training folder is voice recordings only —
        # there are no pixels for this number to size. Managed by _refresh_audio_only_ui.
        self._mp_audio_note = ttk.Label(mp_frame, text="— audio-only dataset: nothing to size",
                                        foreground="#F59E0B", font=(FONT_FAMILY, 9))
        ttk.Label(mp_frame,
                  text="MP  (0.25 ≈ 512², 1.0 ≈ 1024², 2.4 ≈ 1536², 4.2 ≈ 2048²)   "
                       "example: 512² = 512×512 pixels, or any other width × height with a similar pixel area",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9), wraplength=620,
                  justify=tk.LEFT).pack(side=tk.LEFT)
        ttk.Label(training_content,
                  text="Images are automatically resized to fit this target area — no need to resize your dataset "
                       "beforehand. 0.25 MP ≈ 512×512 of pixel area, and your images do NOT have to be square: any "
                       "aspect ratio works (bucketing handles mixed shapes). Higher = more detail, but more VRAM per "
                       "step: 4.2 MP is 4x the pixels of 1.0 and realistically wants 24-32 GB (or heavy block swap) — "
                       "a 16 GB card will OOM well before it.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720).grid(
            row=17, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Per-image loss watch (Krea 2 only for now — hidden under Klein via
        # _apply_training_arch_visibility). Two tiers sharing one watcher in the trainer:
        # detection reports stuck images; per-image LR additionally throttles them.
        self.krea2_loss_watch_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_LOSS_WATCH", False)))
        self._krea2_losswatch_frame = ttk.Frame(training_content)
        self._krea2_losswatch_frame.grid(row=20, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._krea2_detect_cb = ttk.Checkbutton(
            self._krea2_losswatch_frame,
            text="Detect problem images (per-image loss tracking)",
            variable=self.krea2_loss_watch_var,
        )
        self._krea2_detect_cb.pack(side=tk.LEFT)
        ttk.Button(self._krea2_losswatch_frame, text="👁 View Problem Images",
                   command=self._open_problem_images_window).pack(side=tk.LEFT, padx=(12, 0))
        self.krea2_per_image_lr_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_PER_IMAGE_LR", False)))
        self._krea2_perimglr_cb = ttk.Checkbutton(
            training_content,
            text="Per-image adaptive LR (throttle stuck images, boost healthy learned ones) — experimental",
            variable=self.krea2_per_image_lr_var,
        )
        self._krea2_perimglr_cb.grid(row=21, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        self.krea2_auto_recaption_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_AUTO_RECAPTION", False)))
        self._krea2_autorecap_cb = ttk.Checkbutton(
            training_content,
            text="Auto-recaption stuck images (Qwen3-VL rewrites the caption between epochs) — experimental",
            variable=self.krea2_auto_recaption_var,
        )
        self._krea2_autorecap_cb.grid(row=22, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        self.krea2_warmup_look_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_WARMUP_LOOK", False)))
        self._krea2_warmuplook_cb = ttk.Checkbutton(
            training_content,
            text="Warm up look outliers (unusual angles ease in at low LR — needs a Look Filter scan) — experimental",
            variable=self.krea2_warmup_look_var,
        )
        self._krea2_warmuplook_cb.grid(row=23, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        # All four features attribute a step's loss to ONE image — a batch-mean isn't a
        # per-image signal, so they grey out whenever Batch Size > 1 (note shown instead).
        self._krea2_perimage_batch_note = tk.Label(
            training_content,
            text="Per-image features need Batch Size 1 (Dataset section) — a batch-mean loss "
                 "isn't a per-image signal, so these are disabled at the current batch size.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=680, justify=tk.LEFT)
        self._krea2_perimage_batch_note.grid(row=24, column=0, columnspan=2, sticky=tk.W,
                                             padx=5, pady=(2, 0))
        self._krea2_perimage_batch_note.grid_remove()
        try:
            self.dataset_batch_size_var.trace_add(
                "write", lambda *_a: self._refresh_perimage_toggle_state())
        except Exception:
            pass
        self._refresh_perimage_toggle_state()
        self._krea2_losswatch_hint = ttk.Label(training_content,
                  text="Tracks each image's loss (normalized for the random noise level) across epochs. Detection "
                       "flags images that stay hard without improving — usually mislabeled/off-concept data — in the "
                       "console, the Problem Images window, and loss_log/problem_images.json. Per-image LR also "
                       "throttles them (suspects ×0.7 from ~epoch 3, confirmed stuck ×0.5 from ~epoch 5 escalating "
                       "to ×0.1), eases off mined-out images (×0.6) and gives healthy learned ones a gentle boost (×1.1). Auto-recaption goes "
                       "further: when an image is confirmed stuck, the Qwen3-VL text encoder looks at it and rewrites "
                       "its caption from what's actually visible (appending your Captions-tab trigger word, if set), "
                       "re-encodes it, and gives the image a fresh start (a 2nd attempt goes extra-detailed; still "
                       "stuck after that = excluded from training entirely — edit its caption to re-admit it). "
                       "Warm-up: images the Image Prep Look Filter scored as look-outliers (tight angles, "
                       "profiles — real but unusual) start at ×0.4 LR and ramp to ×1.0 over the first ~4 epochs, "
                       "so they refine the identity instead of fighting it while it forms; released early the "
                       "moment they start improving. Run the Look Filter (scan with 3 baselines) first — it saves "
                       "the scores with your dataset. Batch size 1.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._krea2_losswatch_hint.grid(row=24, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Full fine-tune (rotating windows) — Krea 2 only, experimental branch.
        # Trains the BASE MODEL directly instead of a LoRA, a window of weights at a time so
        # a 12.9B full fine-tune fits a consumer card. Output is a full checkpoint.
        self.krea2_finetune_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FINETUNE", False)))
        self._krea2_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚗ Fine-tune the BASE MODEL instead of training a LoRA (experimental)",
            variable=self.krea2_finetune_var,
            command=lambda: self._on_krea2_ft_toggle(),
        )
        self._krea2_ft_cb.grid(row=60, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))

        self._krea2_ft_frame = ttk.Frame(training_content)
        self._krea2_ft_frame.grid(row=61, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        ttk.Label(self._krea2_ft_frame, text="Window:").pack(side=tk.LEFT, padx=(16, 4))
        self.krea2_ft_mode_var = tk.StringVar(
            value=str(self.settings.get("KREA2_FT_MODE", "Auto (by VRAM)")))
        _ftm = ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_mode_var,
                            values=["Auto (by VRAM)", "component", "block"],
                            state="readonly", width=15)
        _ftm.pack(side=tk.LEFT)
        _ftm.bind("<<ComboboxSelected>>", lambda e: self._apply_krea2_ft_visibility())
        ToolTip(_ftm, "Auto (recommended): sizes the window to the VRAM free at launch and prints "
                      "what it picked and why. Component needs ~29.5 GB free; below that it drops to "
                      "block mode with frozen-block streaming, which fits a 24 GB card.  |  "
                      "component: attention, then each MLP matrix, across ALL 28 blocks — "
                      "every window trains the model's full depth, so a concept is learned by every "
                      "layer at once. 4 windows per cycle.  |  "
                      "block: contiguous slices of blocks. Fewer windows, but each trains only part "
                      "of the depth at a time.")
        self._krea2_ft_blocks_lbl = ttk.Label(self._krea2_ft_frame, text="Blocks/window:")
        self._krea2_ft_blocks_lbl.pack(side=tk.LEFT, padx=(14, 4))
        self.krea2_ft_blocks_var = tk.StringVar(value=str(self.settings.get("KREA2_FT_BLOCKS", "14")))
        self._krea2_ft_blocks_cb = ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_blocks_var,
                                                values=["4", "8", "12", "14", "18"], state="readonly", width=4)
        self._krea2_ft_blocks_cb.pack(side=tk.LEFT)
        ToolTip(self._krea2_ft_blocks_cb, "How many of the 28 blocks train at once (block mode). "
                                          "Measured on a 32 GB card: 4 -> 24.8 GB, 8 -> 24.2 GB, "
                                          "14 -> 27.5 GB, 18 -> 29.5 GB. More blocks = fewer windows "
                                          "= each block gets a bigger share of the run.")
        ttk.Label(self._krea2_ft_frame, text="Rotate every:").pack(side=tk.LEFT, padx=(14, 4))
        self.krea2_ft_every_var = tk.StringVar(value=str(self.settings.get("KREA2_FT_EVERY", "1")))
        ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_every_var,
                     values=["1", "2", "3", "5"], state="readonly", width=4).pack(side=tk.LEFT)
        ttk.Label(self._krea2_ft_frame, text="epoch(s)").pack(side=tk.LEFT, padx=(4, 0))
        # One-click launcher for the extractor — the H3 card's twin (pod users only have
        # the browser GUI; hunting for run_diff_to_lora.bat/.sh there is real friction).
        self._krea2_c2l_btn = ttk.Button(self._krea2_ft_frame, text="Checkpoint to LoRA…",
                                         command=self._launch_diff_to_lora)
        self._krea2_c2l_btn.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(self._krea2_c2l_btn,
                "Open the Checkpoint to LoRA tool (its own window): diff a fine-tuned "
                "checkpoint against its base and extract a shareable LoRA at several ranks "
                "at once. Same tool as run_diff_to_lora.bat / .sh in the Fizgig folder.")

        self.krea2_ft_fused_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FT_FUSED", True)))
        self._krea2_ft_fused_cb = ttk.Checkbutton(
            training_content,
            text="Free each gradient as it lands (saves ~5 GB; disables gradient clipping)",
            variable=self.krea2_ft_fused_var,
        )
        self._krea2_ft_fused_cb.grid(row=62, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(2, 0))

        self.krea2_fast_ft_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FAST_FT", False)))
        self._krea2_fast_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚡ Fast FT — per-tensor fp8 + _scaled_mm on the frozen base (experimental)",
            variable=self.krea2_fast_ft_var,
        )
        self._krea2_fast_ft_cb.grid(row=63, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(2, 0))
        ToolTip(self._krea2_fast_ft_cb,
                "Runs the frozen base through torch._scaled_mm instead of dequantising every "
                "weight on every forward. Needs an RTX 40-series or newer (SM 8.9+); silently "
                "falls back to the normal path per-Linear if anything doesn't fit, and the "
                "console says so.\n\n"
                "Costs accuracy: ~1.5x the per-Linear forward error of the default path "
                "(3.7e-02 vs 2.5e-02, measured on real Krea 2 weights). Most of that is NOT the "
                "scale change (only 1.10x) — _scaled_mm needs the activations in fp8 too, and the "
                "default path keeps them in bf16. That is the price of the fp8 GEMM.\n\n"
                "Off by default so the default path stays exactly as it was. The saved checkpoint "
                "comes from the bf16 master either way, so this never changes what lands on disk — "
                "only the frozen forward the trainable window sees.\n\n"
                "Measured 1.14x on a 5090 (0.849 -> 0.742 s/it, component mode, epoch 4) — about "
                "12% off the wall clock. Loss runs ~1.4% higher at the same step, as a lossier "
                "frozen base predicts. Whether that costs output quality is NOT established — "
                "compare checkpoints before trusting it on a real run.")

        # Optional regularisation set. Real photos, not model output: anchoring to the model's
        # own samples distils its artifacts back in, and a full fine-tune moves every weight so
        # there is nothing bounding the drift. Trained at a fixed low LR so it tethers the prior
        # rather than teaching a new one.
        self._krea2_reg_frame = ttk.Frame(training_content)
        self._krea2_reg_frame.grid(row=64, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(6, 0))
        ttk.Label(self._krea2_reg_frame, text="Regularisation images (optional):").pack(side=tk.LEFT)
        self.krea2_reg_dir_var = tk.StringVar(value=str(self.settings.get("KREA2_REG_DIR", "")))
        _regent = ttk.Entry(self._krea2_reg_frame, textvariable=self.krea2_reg_dir_var, width=40)
        _regent.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(self._krea2_reg_frame, text="Browse", width=8,
                   command=self._browse_krea2_reg_dir).pack(side=tk.LEFT)
        # Typed paths need the same TOML rewrite the Browse button triggers.
        self.krea2_reg_dir_var.trace_add(
            "write", lambda *_a: self.auto_save_dataset_config_silent())
        ttk.Label(self._krea2_reg_frame, text="LR ×").pack(side=tk.LEFT, padx=(14, 2))
        self.krea2_reg_mult_var = tk.StringVar(value=str(self.settings.get("KREA2_REG_MULT", "0.2")))
        ttk.Combobox(self._krea2_reg_frame, textvariable=self.krea2_reg_mult_var,
                     values=["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"],
                     state="normal", width=5).pack(side=tk.LEFT)
        ToolTip(_regent,
                "A folder of ordinary photos of the broader class — men, women, people — with "
                "normal detailed captions. Leave empty to train without one.\n\n"
                "Why real photos and not model output: generated regularisation images anchor "
                "the model to its own artifacts, and a full fine-tune moves every weight, so "
                "there is nothing bounding that drift. Real photos are an external reference.\n\n"
                "They train at the LR multiplier beside this box. 0.1-0.3 keeps them a nudge — "
                "tethering the model's prior rather than replacing it. Higher values train them "
                "more like real data: at 1.0 they are simply a second subject set, which is a "
                "different (valid) thing — class-balanced training rather than a light anchor.\n\n"
                "Captions matter: anything you leave unsaid gets attributed to the class word "
                "itself. Caption them as you would any training image.")

        self._krea2_ft_hint = ttk.Label(training_content,
                  text="Trains the base model's own weights, not an adapter — no rank bottleneck, so concepts "
                       "don't compete for the same directions. Only part of the model is trainable at a time and "
                       "the window rotates, which is what makes a 12.9B fine-tune fit; a full cycle is 4 epochs in "
                       "component mode, so run at least that many or some weights never train (the console warns "
                       "you). Use a LOW learning rate — 1e-5 or below; LoRA rates will wreck a base model. "
                       "Network Rank/Alpha are ignored. Adaptive LR and in-training previews are turned off "
                       "automatically, so there are no in-training previews — judge the saved checkpoints in "
                       "ComfyUI. EACH SAVE IS A FULL ~26 GB CHECKPOINT; saving every 4 epochs lands one per "
                       "full cycle, when every component has had the same number of passes and checkpoints "
                       "compare like-for-like (~260 GB over a 40-epoch run). Checkpoints are written to the Output Directory above "
                       "(the usual LoRA folder) — point it somewhere with room, e.g. your ComfyUI models/unet. "
                       "Test the result in ComfyUI as a normal Krea 2 model.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720)
        self._krea2_ft_hint.grid(row=65, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))

        # --- Full fine-tune (rotating windows) — MiniMax H3. Same idea as the Krea 2 card:
        # trains the BASE (the int8 checkpoint's own weights), a window of blocks at a time.
        # Output is a full ~21 GB checkpoint per save. Rows 66-69 (Krea's card is 60-65; each
        # family's card hides under the other).
        self.minimax_finetune_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FINETUNE", False)))
        self._minimax_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚗ Fine-tune the BASE MODEL instead of training a LoRA (experimental)",
            variable=self.minimax_finetune_var,
            command=lambda: self._on_minimax_ft_toggle(),
        )
        self._minimax_ft_cb.grid(row=66, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))

        self._minimax_ft_frame = ttk.Frame(training_content)
        self._minimax_ft_frame.grid(row=67, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        # Component windows are THE mode (24 Aug: the old Blocks/window picker with its
        # 4/6/8-block windows is gone — block mode never matched component's likeness speed).
        # Each window is one matmul (attention qkv/out, MLP fc1/fc2) across EVERY block:
        # full model depth per window, 4 windows per cycle, on an NF4-resident base.
        ttk.Label(self._minimax_ft_frame, text="Rotate every:").pack(side=tk.LEFT, padx=(16, 4))
        self.minimax_ft_every_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_EVERY", "1")))
        ttk.Combobox(self._minimax_ft_frame, textvariable=self.minimax_ft_every_var,
                     values=["1", "2", "3"], state="readonly", width=4).pack(side=tk.LEFT)
        ttk.Label(self._minimax_ft_frame, text="epoch(s)").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(self._minimax_ft_frame, text="Train on:").pack(side=tk.LEFT, padx=(14, 4))
        self.minimax_ft_scope_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_SCOPE", "All media")))
        _mfts = ttk.Combobox(self._minimax_ft_frame, textvariable=self.minimax_ft_scope_var,
                             values=["All media", "Photos only"], state="readonly", width=11)
        _mfts.pack(side=tk.LEFT)
        ToolTip(_mfts, "A dataset FILTER, not a mode. All media (default) fine-tunes on "
                       "everything in the folder — photos, clips, voice. Photos only is the "
                       "override for a mixed folder: clips and voice are skipped and the run "
                       "behaves as if the dataset were photos-only (with Optimised Likeness "
                       "on, the cycle then tightens to the identity blocks). On a dataset "
                       "that's already just photos this choice changes nothing.")
        ttk.Label(self._minimax_ft_frame, text="Blocks:").pack(side=tk.LEFT, padx=(14, 4))
        self.minimax_ft_blockspec_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_BLOCKSPEC", "")))
        _mftbs = ttk.Entry(self._minimax_ft_frame, textvariable=self.minimax_ft_blockspec_var,
                           width=10)
        _mftbs.pack(side=tk.LEFT)
        ToolTip(_mftbs, "Optional: restrict the rotation cycle to a block range — the whole "
                        "fine-tune touches only these blocks. '20-49' is the measured likeness "
                        "recipe (protects the fragile 0-19 trunk) and roughly halves the "
                        "system-RAM master copy. Empty = the full model.")
        # One-click launcher for the extractor — pod users only have the browser GUI, so
        # "find run_diff_to_lora.bat in the folder" is real friction there (field, 29 Aug).
        self._minimax_c2l_btn = ttk.Button(self._minimax_ft_frame, text="Checkpoint to LoRA…",
                                           command=self._launch_diff_to_lora)
        self._minimax_c2l_btn.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(self._minimax_c2l_btn,
                "Open the Checkpoint to LoRA tool (its own window): diff a fine-tuned "
                "checkpoint against its base and extract a shareable LoRA at several ranks "
                "at once. Same tool as run_diff_to_lora.bat / .sh in the Fizgig folder.")

        # Save-every follows the CYCLE, and the cycle follows these controls — so the box is
        # kept live rather than seeded with a stale constant (it used to sit at 13, the old
        # full-model N=4 cycle, whatever mode was picked; the trainer's launch-time snap then
        # silently corrected it and the box looked ignored).
        for _v in (self.minimax_ft_every_var, self.minimax_ft_blockspec_var):
            _v.trace_add("write", lambda *_a: self._refresh_minimax_ft_save_box())

        self.minimax_ft_fused_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FT_FUSED", True)))
        self._minimax_ft_fused_cb = ttk.Checkbutton(
            training_content,
            text="Free each gradient as it lands (fits the window; disables gradient clipping)",
            variable=self.minimax_ft_fused_var,
        )
        self._minimax_ft_fused_cb.grid(row=68, column=0, columnspan=2, sticky=tk.W,
                                       padx=(21, 5), pady=(2, 0))

        # Optional regularisation set — same doctrine as the Krea 2 FT card (real photos of
        # the broader class, trained at a fixed low LR as a prior anchor), with H3-specific
        # lifecycle: reg stills follow the photo routing (likeness window) and stop with the
        # visual category under 'Finish one category early'. Entirely optional — empty = off.
        self._minimax_reg_frame = ttk.Frame(training_content)
        self._minimax_reg_frame.grid(row=69, column=0, columnspan=2, sticky=tk.W,
                                     padx=(21, 5), pady=(6, 0))
        ttk.Label(self._minimax_reg_frame, text="Regularisation images (optional):").pack(side=tk.LEFT)
        self.minimax_reg_dir_var = tk.StringVar(value=str(self.settings.get("MINIMAX_REG_DIR", "")))
        _mregent = ttk.Entry(self._minimax_reg_frame, textvariable=self.minimax_reg_dir_var, width=40)
        _mregent.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(self._minimax_reg_frame, text="Browse", width=8,
                   command=self._browse_minimax_reg_dir).pack(side=tk.LEFT)
        self.minimax_reg_dir_var.trace_add(
            "write", lambda *_a: self.auto_save_dataset_config_silent())
        ttk.Label(self._minimax_reg_frame, text="LR ×").pack(side=tk.LEFT, padx=(14, 2))
        self.minimax_reg_mult_var = tk.StringVar(value=str(self.settings.get("MINIMAX_REG_MULT", "0.2")))
        ttk.Combobox(self._minimax_reg_frame, textvariable=self.minimax_reg_mult_var,
                     values=["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"],
                     state="normal", width=5).pack(side=tk.LEFT)
        ToolTip(_mregent,
                "A folder of ordinary REAL photos of the broader class — people, faces — with "
                "normal detailed captions. Leave empty to train without one.\n\n"
                "A full fine-tune moves the base weights with nothing bounding the drift; these "
                "anchor the model's prior while your subject data pulls it. Real photos, not "
                "model output — generated images anchor the model to its own artifacts.\n\n"
                "They train at the LR multiplier beside this box (0.1-0.3 keeps them a nudge; "
                "keep them the MINORITY of the dataset or the nudge stops reading as one), "
                "follow the same photo routing as your subject stills, and stop when photos & "
                "clips stop under 'Finish one category early' — once subject pressure ends, the "
                "counter-pressure ends with it. Stills only: they tether the visual prior; the "
                "audio prior is protected by voice routing instead.\n\n"
                "With Optimised Likeness on, the anchor pulls only on the likeness blocks — "
                "the same territory your subject photos train, which is the point. On an "
                "audio-only dataset, adding reg stills widens the rotation cycle to include "
                "the photo blocks (the console prints the new span).")

        self._minimax_ft_hint = ttk.Label(training_content,
                  text="Trains the base model's own weights, not an adapter — Network Type "
                       "and Blocks to Train hide while this is on (they're LoRA machinery); "
                       "Optimised Likeness Learning keeps working with its usual meaning. "
                       "Each window is one matmul (attention qkv/out, MLP fc1/fc2) across "
                       "every block — full model depth per window, 4 windows per cycle, on "
                       "an NF4-resident base (the saved checkpoint is still exact int8). "
                       "The Blocks field above is an optional manual restriction of the "
                       "whole fine-tune. Needs a 32 GB card and ~64 GB of system RAM. "
                       "A full ~21 GB checkpoint saves once per COMPLETED CYCLE (every "
                       "block equally trained — the save box snaps to the cycle "
                       "automatically), and previews ride along with each save (plus the "
                       "final one), so every sample matches a checkpoint you can deploy. "
                       "Photos-only dataset? Leave 'Train on' "
                       "alone — there's nothing to skip. Use a LOW learning rate (1e-5 to "
                       "start; H3 is uncalibrated — compare checkpoints). Point the Output "
                       "Directory somewhere with room, judge results in ComfyUI, and distil "
                       "to a shareable LoRA with Checkpoint to LoRA.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720)
        self._minimax_ft_hint.grid(row=70, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))
        # --- Per-step movement clip (MiniMax only) -----------------------------------------
        # Whichever block sits LAST in the trained range absorbs 2-4x the median block's
        # movement from epoch 1 (measured across four runs; cutting blocks just moves the hot
        # spot to the new last block). This caps any block's movement WITHIN A SINGLE STEP at
        # N x the median block's step. The 3.5.0 version capped CUMULATIVE movement instead,
        # which also scaled down everything the block had legitimately learned — measured as a
        # real likeness ceiling (on was visibly worse than off, off corrupted). Clipping the
        # step prevents the overshoot instead of undoing history, so there is nothing to trade.
        self._minimax_limiter_label = ttk.Label(training_content, text="Per-step movement clip:")
        self._minimax_limiter_label.grid(row=37, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_limiter_frame = ttk.Frame(training_content)
        self._minimax_limiter_frame.grid(row=37, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_BLOCK_LIMIT"] = ttk.Combobox(
            self._minimax_limiter_frame, values=["Off", "1.1 x median (tightest)",
                                                 "1.25 x median (default)",
                                                 "1.5 x median",
                                                 "2.0 x median (loose)",
                                                 "2.5 x median",
                                                 "3.0 x median (safety net only)"],
            width=26, state="readonly")
        self.entries["MINIMAX_BLOCK_LIMIT"].set(
            str(self.settings.get("MINIMAX_BLOCK_LIMIT", "Off")))
        self.entries["MINIMAX_BLOCK_LIMIT"].pack(side=tk.LEFT)
        self._minimax_limiter_hint = ttk.Label(
            training_content,
            text="STRONGLY RECOMMENDED ON — stops any single block overshooting in a step, the "
                 "classic source of distortion. Only the offending step is shortened, so it "
                 "costs nothing that was already learned. Full write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_limiter_hint.grid(row=38, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Reference distillation (MiniMax only, experimental) ---------------------------
        # No picker: the dataset IS the reference pool, so there is nothing to choose.
        self.minimax_distill_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_DISTILL", False)))
        self._minimax_distill_frame = ttk.Frame(training_content)
        self._minimax_distill_frame.grid(row=35, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_distill_cb = ttk.Checkbutton(
            self._minimax_distill_frame, text="Learn identity from my dataset (reference distillation)",
            variable=self.minimax_distill_var, command=self._on_minimax_distill_clicked)
        self._minimax_distill_cb.pack(side=tk.LEFT)
        # Multi Concept shows a warning while identity-learn is OFF (no reference steering), so
        # that hint has to refresh when this checkbox moves, not only when the mode is toggled.
        self.minimax_distill_var.trace_add(
            "write", lambda *_a: self._on_minimax_multiconcept_toggle())
        ttk.Label(self._minimax_distill_frame, text="   teacher ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_WEIGHT"] = ttk.Combobox(
            # 0.4/0.5 added 11 Aug — an even split is a reasonable thing to want and the list
            # stopped at 0.6, so it could not be asked for. 1.0 removes the photo term entirely,
            # which caps the LoRA at what reference mode can already do.
            self._minimax_distill_frame,
            values=["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], width=5)
        self.entries["MINIMAX_DISTILL_WEIGHT"].set(
            str(self.settings.get("MINIMAX_DISTILL_WEIGHT", "0.8")))
        self.entries["MINIMAX_DISTILL_WEIGHT"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_distill_frame, text="   references each ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_REFS"] = ttk.Combobox(
            self._minimax_distill_frame, values=["1", "2", "3", "4"], width=4)
        self.entries["MINIMAX_DISTILL_REFS"].set(
            str(self.settings.get("MINIMAX_DISTILL_REFS", "2")))
        self.entries["MINIMAX_DISTILL_REFS"].pack(side=tk.LEFT)
        # Identity-first: teacher-ONLY for the first stretch, then photos-only. A hard switch,
        # not a blend — the point is where the adapter STARTS, so what phase 2 forgets about the
        # teacher does not matter. Auto sizes phase 1 from the dataset (~650 steps, which is
        # where the teacher error was measured to converge on a real run).
        ttk.Label(self._minimax_distill_frame, text="   identity-first ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_PHASE1"] = ttk.Combobox(
            self._minimax_distill_frame, state="readonly", width=22,
            values=["Auto (from dataset size)", "Off — blend throughout",
                    "2 epochs", "4 epochs", "8 epochs", "16 epochs", "30 epochs"])
        self.entries["MINIMAX_DISTILL_PHASE1"].set(
            str(self.settings.get("MINIMAX_DISTILL_PHASE1", "Auto (from dataset size)")))
        self.entries["MINIMAX_DISTILL_PHASE1"].pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_PHASE1"].bind(
            "<<ComboboxSelected>>", lambda _e: self._sync_distill_weight_state())
        self._minimax_distill_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — teaches your LoRA to reproduce identity from the trigger word "
                 "the way H3 does when shown a photo, using your own dataset as the "
                 "references. Needs the ref2va model in Preferences. Full write-up in the "
                 "README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_distill_hint.grid(row=36, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))





        # --- Multi Concept (MiniMax only) ---------------------------------------------------
        # Two subjects in ONE folder get cross-referenced by reference distillation: the pairing
        # rotation runs per [[datasets]] block, so a single block marks subject A's answers
        # against photos of subject B — which blends them rather than separating them. Giving
        # each subject its own folder makes the rotation per-subject for free.
        self.minimax_multiconcept_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_MULTICONCEPT", False)))
        self._minimax_mc_frame = ttk.Frame(training_content)
        self._minimax_mc_frame.grid(row=49, column=0, columnspan=2, sticky=tk.W, padx=5,
                                    pady=(8, 0))
        self._minimax_mc_cb = ttk.Checkbutton(
            self._minimax_mc_frame, text="Multi Concept — a second subject in its own folder",
            variable=self.minimax_multiconcept_var,
            command=self._on_minimax_multiconcept_clicked)
        self._minimax_mc_cb.pack(side=tk.LEFT)

        # A LIST even though the UI shows one — a third concept is then a widget, not a rewrite.
        self._concept_folder_vars = [tk.StringVar(
            value=str(self.last_used.get("image_folder2", "") or ""))]
        self._minimax_mc_dir_frame = ttk.Frame(training_content)
        self._minimax_mc_dir_frame.grid(row=50, column=0, columnspan=2, sticky=tk.EW, padx=5,
                                        pady=(2, 0))
        ttk.Label(self._minimax_mc_dir_frame, text="Subject 2 folder:").pack(side=tk.LEFT,
                                                                             padx=(20, 6))
        self._minimax_mc_entry = ttk.Entry(self._minimax_mc_dir_frame,
                                           textvariable=self._concept_folder_vars[0],
                                           state="readonly", width=52)
        self._minimax_mc_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self._minimax_mc_dir_frame, text="Browse…",
                   command=self._browse_concept_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(self._minimax_mc_dir_frame, text="Clear",
                   command=lambda: self._concept_folder_vars[0].set("")).pack(side=tk.LEFT,
                                                                             padx=(4, 0))
        self._minimax_mc_hint = ttk.Label(
            training_content,
            text="Each folder needs its OWN trigger word, in every caption — that is the only "
                 "thing telling the two apart. Caption and prep both folders yourself first; "
                 "this box is training-only. Ticking the mode also sets the settings that suit "
                 "it (identity-learn on, 4 references, identity-first 2 epochs, dropout 0.10, "
                 "adapter-relative LR off) — all still yours to change.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT,
            wraplength=720)
        self._minimax_mc_hint.grid(row=51, column=0, columnspan=2, sticky=tk.W, padx=5,
                                   pady=(0, 4))
        # Only shown when Multi Concept is on AND identity-learn is off — see the toggle handler.
        self._minimax_mc_nodistill_hint = ttk.Label(
            training_content,
            text="Identity-learn is off, so the reference steering that keeps two subjects "
                 "apart is not running — separation rests on your trigger words alone.",
            foreground=COLORS["warning"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT,
            wraplength=720)
        self._minimax_mc_nodistill_hint.grid(row=52, column=0, columnspan=2, sticky=tk.W,
                                             padx=5, pady=(0, 4))
        # These change the [[datasets]] blocks, so the TOML has to be rewritten when they move.
        # Without this the mode looks enabled and silently trains the old single-folder config.
        if hasattr(self, "_auto_save_ds"):
            self.minimax_multiconcept_var.trace_add("write", self._auto_save_ds)
            for _cv in self._concept_folder_vars:
                _cv.trace_add("write", self._auto_save_ds)
                _cv.trace_add("write", lambda *_a: self._save_last_used_paths())

        # --- Slow blocks (MiniMax only, experimental): depth-dependent LR -------------------
        self._minimax_slow_label = ttk.Label(training_content, text="Slower LR for blocks:")
        self._minimax_slow_label.grid(row=33, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self._minimax_slow_frame = ttk.Frame(training_content)
        self._minimax_slow_frame.grid(row=33, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(8, 2))
        self.entries["MINIMAX_SLOW_BLOCKS"] = ttk.Entry(self._minimax_slow_frame, width=22)
        self.entries["MINIMAX_SLOW_BLOCKS"].insert(
            0, str(self.settings.get("MINIMAX_SLOW_BLOCKS", "") or ""))
        self.entries["MINIMAX_SLOW_BLOCKS"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_slow_frame, text="  at ×").pack(side=tk.LEFT)
        self.entries["MINIMAX_SLOW_LR_SCALE"] = ttk.Combobox(
            self._minimax_slow_frame, values=["0.1", "0.2", "0.3", "0.5", "0.7"],
            state="normal", width=6)
        self.entries["MINIMAX_SLOW_LR_SCALE"].set(
            str(self.settings.get("MINIMAX_SLOW_LR_SCALE", "0.2")))
        self.entries["MINIMAX_SLOW_LR_SCALE"].pack(side=tk.LEFT, padx=(2, 0))
        self._minimax_slow_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — leave blank for one learning rate everywhere (normal). A change in "
                 "a late block goes almost straight to the output, while a change early on gets "
                 "smoothed out by the 40-odd blocks after it — so the same learning rate is "
                 "gentle at the front of the model and violent at the back. If the later blocks "
                 "wreck your samples at a rate the early ones handle fine, put those blocks here "
                 "with a multiplier instead of dropping them: 21-49 at ×0.2 trains them at a "
                 "fifth the rate. Same syntax as Blocks to Train, and only blocks you're actually "
                 "training count. Adaptive LR still works — it moves both rates together and "
                 "keeps the ratio.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_slow_hint.grid(row=34, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Train AdaLN (MiniMax only, experimental) --------------------------------------
        # A BooleanVar kept in self.entries so the preset/queue machinery picks it up for free.
        self.entries["MINIMAX_TRAIN_ADALN"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_TRAIN_ADALN", False)))
        self._minimax_adaln_cb = ttk.Checkbutton(
            training_content, text="Train AdaLN (timestep modulation)",
            variable=self.entries["MINIMAX_TRAIN_ADALN"])
        self._minimax_adaln_cb.grid(row=31, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_adaln_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — off by default (the reference trainer leaves it on). AdaLN is "
                 "the part of the model that decides how strongly each block fires at each noise "
                 "level. It only ever sees the noise level: not your image, not your prompt. So it "
                 "CAN'T learn who someone is — but on this base it soaks up roughly 45% of "
                 "everything your LoRA learns. Turning it off hands that capacity to the parts "
                 "that do see the image. It may sharpen likeness, or it may cost you the timing "
                 "control that makes the rest work — run it both ways on the same dataset. Only "
                 "applies to the pruned int8 base; the bf16 one never trains AdaLN anyway.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_adaln_hint.grid(row=32, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # Optimised Likeness Learning — photo steps train the identity blocks only; clips train
        # the full model. BooleanVar in self.entries so presets/queue/last-train carry it free.
        self.entries["MINIMAX_LIKENESS_OPT"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_LIKENESS_OPT", True)))
        # Under FT the tickbox changes the rotation-cycle length (50 blocks -> the 20-49
        # tighten), so the Save-every suggestion follows it live.
        self.entries["MINIMAX_LIKENESS_OPT"].trace_add(
            "write", lambda *_a: self._refresh_minimax_ft_save_box())
        self._minimax_likeness_cb = ttk.Checkbutton(
            training_content, text="Optimised Likeness Learning",
            variable=self.entries["MINIMAX_LIKENESS_OPT"])
        self._minimax_likeness_cb.grid(row=39, column=0, columnspan=2, sticky=tk.W,
                                       padx=5, pady=(8, 0))
        self._minimax_likeness_hint = ttk.Label(
            training_content,
            text=f"Photos train the identity blocks ({MINIMAX_LIKENESS_BLOCKS}) only and "
                 f"voice recordings train the audio zone ({MINIMAX_AUDIO_BLOCKS}) only — "
                 "protecting the base model's rendering, anatomy and prompt following — while "
                 "video clips train the full model. Measured result: sharper, more "
                 "prompt-responsive, better sound, faster to converge. Untick for style or "
                 "scene training (the Style preset does).",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_likeness_hint.grid(row=40, column=0, columnspan=2, sticky=tk.W,
                                         padx=5, pady=(0, 4))
        self._MINIMAX_LIKENESS_HINT_LORA = self._minimax_likeness_hint.cget("text")
        self._MINIMAX_LIKENESS_HINT_FT = (
            f"Under fine-tune this keeps its exact LoRA meaning: photos feed only the "
            f"identity blocks ({MINIMAX_LIKENESS_BLOCKS}), voice feeds only the audio zone "
            f"({MINIMAX_AUDIO_BLOCKS}) — and video follows the restriction tickbox below "
            f"(on: clips train {MINIMAX_LIKENESS_BLOCKS} too; off: clips train the full "
            f"model). The rotation cycle tightens automatically to the union of what your "
            f"dataset actually trains. Untick for style/scene fine-tunes — voice still "
            f"routes to its zone either way. An explicit Blocks range above always wins.")
        # Restrict video to likeness blocks — FT-only sub-tick of likeness mode (Peter,
        # 29 Aug: a confined overnight video run trained perfectly well; on by default,
        # untick for whole-model video). Emitted as --clip_blocks by the FT builder only;
        # shown only when the family is MiniMax AND Fine-tune AND likeness are all on
        # (managed by _sync_minimax_likeness_state, which fires on all three).
        self.entries["MINIMAX_FT_CLIP_LIKENESS"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FT_CLIP_LIKENESS", True)))
        self._minimax_ft_clip_cb = ttk.Checkbutton(
            training_content,
            text=f"Restrict video to likeness blocks ({MINIMAX_LIKENESS_BLOCKS}) — in our "
                 "tests this trains video just as well, and it makes clips far lighter on "
                 "VRAM. Untick to train video on the whole model.",
            variable=self.entries["MINIMAX_FT_CLIP_LIKENESS"])
        self._minimax_ft_clip_cb.grid(row=41, column=0, columnspan=2, sticky=tk.W,
                                      padx=(21, 5), pady=(0, 4))
        # trace, not command=: preset loads set the var programmatically and must re-grey too.
        self.entries["MINIMAX_LIKENESS_OPT"].trace_add(
            "write", lambda *_a: self._sync_minimax_likeness_state())

        # Answers "when do changes take effect?" (issue #40) right where people wonder it.
        ttk.Label(training_content,
                  text="When do changes apply? Settings are read when a run launches — changing "
                       "them mid-run does nothing. Pause → Resume relaunches with your current "
                       "settings, so these can be changed at a pause. Dataset/caption changes "
                       "need a fresh run (Resume skips re-caching).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720).grid(
            row=30, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))

        # === Optimizer Section (Collapsed by default) ===
        optimizer_section = CollapsibleFrame(outer,"Optimizer", default_expanded=False)
        optimizer_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["optimizer"] = optimizer_section

        optimizer_content = optimizer_section.get_content_frame()
        optimizer_content.columnconfigure(1, weight=1)

        self._add_field_to_section(optimizer_content, "OPTIMIZER_TYPE", "Optimizer Type", "dropdown", 0)
        self._add_field_to_section(optimizer_content, "OPTIMIZER_ARGS", "Optimizer Args", "text", 1)
        self._add_field_to_section(optimizer_content, "GRADIENT_ACCUMULATION", "Gradient Accumulation", "int", 2)
        self._add_field_to_section(optimizer_content, "MAX_GRAD_NORM", "Max Grad Norm", "float", 3)
        self._add_field_to_section(optimizer_content, "NETWORK_DROPOUT", "Network Dropout", "float", 4)

        # === Scheduler Section (Collapsed by default) ===
        scheduler_section = CollapsibleFrame(outer,"Other Options", default_expanded=False)
        scheduler_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["scheduler"] = scheduler_section

        scheduler_content = scheduler_section.get_content_frame()
        scheduler_content.columnconfigure(1, weight=1)

        # === Dataset subsection (migrated from the removed Dataset tab) ===
        tk.Label(
            scheduler_content,
            text="Dataset",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(8, 4))

        ttk.Label(scheduler_content, text="Caption Extension:").grid(row=1, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        ttk.Entry(scheduler_content, textvariable=self.dataset_caption_ext_var, width=16).grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)
        tk.Label(scheduler_content, text="(default .txt)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=1, column=2, sticky=tk.W, padx=5)

        # (Target Megapixels moved to the Training Parameters section — it's a core
        # training setting, so it lives with the rest of them rather than buried here.)

        ttk.Label(scheduler_content, text="Batch Size:").grid(row=4, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bs_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bs_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(bs_frame, textvariable=self.dataset_batch_size_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(bs_frame, text="(recommended: 1 — higher values need more VRAM)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        ttk.Label(scheduler_content, text="Bucket Options:").grid(row=5, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bucket_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bucket_frame.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Checkbutton(bucket_frame, text="Enable Bucket", variable=self.dataset_enable_bucket_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(bucket_frame, text="No Upscale (keep small images at native size)",
                        variable=self.dataset_no_upscale_var).pack(side=tk.LEFT)

        ttk.Separator(scheduler_content, orient="horizontal").grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 6))

        # === LR Scheduler ===
        tk.Label(
            scheduler_content,
            text="LR Scheduler",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(4, 4))

        self._add_field_to_section(scheduler_content, "LR_SCHEDULER", "LR Scheduler", "dropdown", 8)

        # Warmup/Decay steps in a sub-frame
        lr_steps_label = tk.Label(
            scheduler_content,
            text="Warmup / Decay:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        lr_steps_label.grid(row=9, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        lr_steps_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        lr_steps_frame.grid(row=9, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(lr_steps_frame, text="Warmup:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_WARMUP_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_WARMUP_STEPS"].pack(side=tk.LEFT, padx=(0, 16))

        # Decay is Klein-only (its warmup_stable_decay path); krea2's scheduler set has no
        # decay-steps knob, so this pair is hidden under Krea 2 rather than silently ignored.
        self._lr_decay_label = tk.Label(lr_steps_frame, text="Decay:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self._lr_decay_label.pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_DECAY_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_DECAY_STEPS"].pack(side=tk.LEFT)

        # Separator before the migrated Advanced fields
        ttk.Separator(scheduler_content, orient="horizontal").grid(row=10, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 4))
        # Inline Attention / Logging / Memory / Metadata fields (formerly the Advanced tab)
        self._populate_other_options(scheduler_content, start_row=11)

        # === Memory & precision section ===
        # Header names every path the section now offers, INT8 included — it is the default on
        # 20 GB+ cards and had no presence anywhere in the UI.
        memory_section = CollapsibleFrame(outer, "Memory & Precision (INT8 / FP8 / NF4)",
                                          default_expanded=True)
        memory_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["memory"] = memory_section

        memory_content = memory_section.get_content_frame()
        memory_content.columnconfigure(1, weight=1)

        # Blocks Swap dropdown — labeled VRAM presets first, then leftover numbers (Klein 9B max=16)
        ttk.Label(memory_content, text="Blocks Swap:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        # VRAM guidance is per Base DiT precision: fp8 Base (~9.6 GB resident,
        # recommended) needs far less than bf16 Base (~18 GB). fp8 fits 16 GB cards
        # with no swap at all.
        blocks_swap_options = [
            "Auto (detect from GPU)",
            "0  (No swap — 16GB fp8 / 24GB bf16)",
            "4  (Light — 14GB fp8 / 20GB bf16)",
            "8  (Moderate — 12GB fp8 / 16GB bf16)",
            "12 (Aggressive — 10GB fp8 / 12GB bf16)",
            "16 (Max — 8GB fp8 / 10GB bf16)",
            "1", "2", "3", "5", "6", "7", "9", "10", "11", "13", "14", "15",
        ]
        _bs_max_len = max(len(v) for v in blocks_swap_options)
        self.entries["BLOCKS_SWAP"] = ttk.Combobox(memory_content, values=blocks_swap_options, width=_bs_max_len + 2)
        self.entries["BLOCKS_SWAP"].grid(row=0, column=1, sticky=tk.W, padx=5, pady=4)
        try:
            _bs_val = self.settings.get("BLOCKS_SWAP", "auto")
            if str(_bs_val).lower() == "auto":
                self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])
            else:
                _bs_int = int(_bs_val)
                _label_map = {0: blocks_swap_options[1], 4: blocks_swap_options[2], 8: blocks_swap_options[3],
                              12: blocks_swap_options[4], 16: blocks_swap_options[5]}
                self.entries["BLOCKS_SWAP"].set(_label_map.get(_bs_int, str(_bs_int)))
        except (ValueError, TypeError):
            self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])

        self._add_field_to_section(memory_content, "RESUME_TRAINING", "Resume Training", "directory", 1)
        # The field had no explanation at all, which made the whole resume feature invisible
        # unless you already knew a state dir was a folder (not a .safetensors) and where it lived.
        tk.Label(memory_content,
                 text="Leave empty for a normal run. To carry on from a saved state, Browse to a folder named "
                      "like myLora-000012-state in your LoRA output folder — the number is the epoch it "
                      "finished. Training continues at the next epoch with the optimizer, learning rate and "
                      "seed exactly as they were, so it picks up mid-run rather than starting over. To train a "
                      "FINISHED LoRA further, pick its highest-numbered state and raise Max Train Epochs first "
                      "— otherwise there are no epochs left to run. Pausing writes one of these for you, and "
                      "the Resume button fills this in automatically; you only need Browse for an older "
                      "checkpoint or a run from a previous session.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=2, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # FP8 Checkboxes. Row label + hint are captured on self (not locals) so
        # _apply_training_arch_visibility can hide them alongside the checkboxes for Krea 2 —
        # otherwise hiding the controls leaves their label and explanation floating.
        self._fp8_row_label = tk.Label(
            memory_content,
            text="Weight Optimization:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        self._fp8_row_label.grid(row=3, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        fp8_frame = tk.Frame(memory_content, bg=COLORS["bg_surface"])
        fp8_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=4)

        self.fp8_var = tk.BooleanVar(value=self.settings["FP8"])
        self.scaled_var = tk.BooleanVar(value=self.settings["SCALED"])

        self.fp8_check = ttk.Checkbutton(fp8_frame, text="FP8 Base", variable=self.fp8_var, command=self.toggle_scaled, style="Surface.TCheckbutton")
        self.fp8_check.pack(side=tk.LEFT, padx=(0, 16))

        self.scaled_check = ttk.Checkbutton(fp8_frame, text="FP8 Scaled", variable=self.scaled_var, state=tk.DISABLED if not self.fp8_var.get() else tk.NORMAL, style="Surface.TCheckbutton")
        self.scaled_check.pack(side=tk.LEFT)
        self._fp8_hint = tk.Label(
            memory_content,
            text="Converts a bf16 model to fp8 at load time. If your Base DiT is already fp8 "
                 "(e.g. flux-2-klein-base-9b-fp8), leave this unchecked — Fizgig detects "
                 "pre-quantised fp8 files automatically.",
            font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=600, justify=tk.LEFT)
        self._fp8_hint.grid(row=4, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # FP8 Text Encoder
        self.fp8_text_encoder_label = tk.Label(
            memory_content,
            text="FP8 Text Encoder:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        self.fp8_text_encoder_label.grid(row=5, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.fp8_text_encoder_var = tk.BooleanVar(value=self.settings["FP8_TEXT_ENCODER"])
        self.fp8_text_encoder_check = ttk.Checkbutton(memory_content, text="Enable FP8 T5/LLM", variable=self.fp8_text_encoder_var, style="Surface.TCheckbutton")
        self.fp8_text_encoder_check.grid(row=5, column=1, sticky=tk.W, padx=5, pady=4)

        # Base precision (Krea 2 only — hidden for Klein by _apply_training_arch_visibility).
        #
        # Was "4-bit Base: Auto / On / Off", which was actively misleading: "Off" meant "not
        # NF4", and on a capable card you would then silently get INT8 — a quantisation the UI
        # never named anywhere. Reading Off as "no quantisation" is what produced the v2.8.6
        # regression; the UI invited that misreading. Every path is now named and selectable,
        # INT8 included, and each one is planned for properly by the memory strategy.
        #
        # quant_4bit_var stays the BooleanVar every downstream consumer reads (True only for
        # NF4); quant_4bit_mode_var holds the canonical key and the combobox shows its label.
        self._quant_4bit_label = tk.Label(memory_content, text="Base precision:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._quant_4bit_label.grid(row=6, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.quant_4bit_var = tk.BooleanVar(value=False)
        self.quant_4bit_mode_var = tk.StringVar(value=self._BASE_PRECISION_LABELS[
            self._normalize_base_precision(
                self.settings.get("QUANT_4BIT_MODE", "")
                or ("On" if self.settings.get("QUANT_4BIT", False) else "auto"))])
        _q4_row = ttk.Frame(memory_content)
        _q4_row.grid(row=6, column=1, sticky=tk.W, padx=5, pady=4)
        self.quant_4bit_check = ttk.Combobox(
            _q4_row, textvariable=self.quant_4bit_mode_var,
            values=list(self._BASE_PRECISION_LABELS.values()), state="readonly", width=34)
        self.quant_4bit_check.pack(side=tk.LEFT)
        self.quant_4bit_check.bind("<<ComboboxSelected>>",
                                   lambda e: self._on_quant_4bit_mode_changed())
        self._quant_4bit_hint = tk.Label(memory_content,
                 text="Auto (recommended) picks the fastest option that fits your FREE VRAM, and sizes block "
                      "swap to match. INT8 is 8-bit — fastest, and ~7x more accurate than 4-bit, but needs "
                      "~18 GB free. 4-bit NF4 is the smallest (~5.6 GB base) so it fits 10–12 GB cards with "
                      "no swap, at a slight quality cost. fp8 is the least compressed of the three and needs "
                      "the most VRAM, so it swaps blocks to fit. Anything you pick explicitly is planned "
                      "for — swap is sized for the option that will actually run.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._quant_4bit_hint.grid(row=7, column=1, sticky=tk.W, padx=5, pady=(0, 4))
        self._on_quant_4bit_mode_changed()  # derive the boolean + sync dependent locks

        # Gradient checkpointing — trades compute for VRAM.
        self._grad_checkpoint_label = tk.Label(memory_content, text="Grad Checkpoint:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._grad_checkpoint_label.grid(row=8, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.grad_checkpoint_var = tk.BooleanVar(value=self.settings.get("GRADIENT_CHECKPOINTING", True))
        self.grad_checkpoint_check = ttk.Checkbutton(
            memory_content, text="Gradient checkpointing (recommended — lower VRAM)",
            variable=self.grad_checkpoint_var, command=self._on_grad_checkpoint_toggle,
            style="Surface.TCheckbutton")
        self.grad_checkpoint_check.grid(row=8, column=1, sticky=tk.W, padx=5, pady=4)
        self._grad_checkpoint_hint = tk.Label(memory_content,
                 text="On (default) recomputes activations during the backward pass to save VRAM — it's what lets "
                      "a 9B LoRA fit on a 16 GB card. Turning it OFF makes training ~20–30% faster but uses far more "
                      "VRAM, so it's only for big cards (24 GB+, ideally 32 GB) with Blocks Swap at 0. On 16 GB, or "
                      "with block swap on, leave it ON.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._grad_checkpoint_hint.grid(row=9, column=1, sticky=tk.W, padx=5, pady=(0, 4))
        # torch.compile (Krea 2 only — hidden under Klein by _apply_training_arch_visibility).
        self._compile_blocks_label = tk.Label(memory_content, text="Compile Blocks:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._compile_blocks_label.grid(row=10, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        # Auto / On / Off rather than a checkbox: compile is a long-run win and a short-run loss,
        # so the right default is a judgement the trainer makes from the actual run length, the
        # same way Blocks Swap defaults to "Auto (detect from GPU)".
        self.compile_blocks_var = tk.StringVar(value=str(self.settings.get("COMPILE_BLOCKS", "auto")).capitalize())
        self.compile_blocks_check = ttk.Combobox(
            memory_content, textvariable=self.compile_blocks_var, state="readonly", width=36,
            values=["Auto", "On", "Off"])
        self.compile_blocks_check.grid(row=10, column=1, sticky=tk.W, padx=5, pady=4)
        self._compile_blocks_hint = tk.Label(memory_content,
                 text="Auto (recommended) turns torch.compile on only when this run is long enough to repay it. "
                      "It fuses the per-matmul quantise/dequantise work that bounds the INT8 and NF4 paths — "
                      "2.0× per step on INT8 (0.59 → 0.29 s/step, matching OneTrainer) and 1.28× on "
                      "NF4 (0.71 → 0.56) — but costs a ~90 s compile pause first, so a short run is SLOWER "
                      "overall. Break-even is around 600 steps on INT8, 1200 on NF4. NF4 + compile still fits a 16 GB "
                      "card (verified under a 13.5 GB cap). INT8 + compile fits from ~22 GB free: at high resolution "
                      "the checkpoint automatically moves outside the compiled region, which keeps memory at eager "
                      "levels (~18 GB at 1024px, measured ~27% faster than uncompiled). Requires Triton and, on "
                      "Windows, a C++ compiler (VS Build Tools) — both located automatically. Never used with "
                      "Blocks Swap, since swapping moves weights and compiled graphs assume they stay put.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._compile_blocks_hint.grid(row=11, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Save State ------------------------------------------------------------------
        # Deliberately next to Resume Training (row 1): what writes state and what reads it back
        # belong together. Both families, so NOT added to either list in
        # _apply_training_arch_visibility.
        tk.Label(memory_content, text="Save State:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]
                 ).grid(row=12, column=0, sticky=tk.NW, padx=(12, 8), pady=(10, 4))
        _state_frame = tk.Frame(memory_content, bg=COLORS["bg_surface"])
        _state_frame.grid(row=12, column=1, sticky=tk.W, padx=5, pady=(10, 4))
        self.save_state_var = tk.BooleanVar(value=self.settings.get("SAVE_STATE", True))
        ttk.Checkbutton(_state_frame, text="At each checkpoint", variable=self.save_state_var,
                        style="Surface.TCheckbutton").pack(anchor=tk.W)
        self.save_state_on_train_end_var = tk.BooleanVar(
            value=self.settings.get("SAVE_STATE_ON_TRAIN_END", True))
        ttk.Checkbutton(_state_frame, text="At end of training",
                        variable=self.save_state_on_train_end_var,
                        style="Surface.TCheckbutton").pack(anchor=tk.W)
        tk.Label(memory_content,
                 text="A state dir holds the LoRA plus the optimizer, so a run can pick up exactly where it "
                      "left off — after a crash, or to train a finished LoRA further by raising Max Train "
                      "Epochs and resuming. \"At each checkpoint\" follows Save Every N Epochs. Pause always "
                      "saves state whether these are ticked or not.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=13, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        tk.Label(memory_content, text="Keep Last:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]
                 ).grid(row=14, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.entries["KEEP_LAST_N_STATES"] = ttk.Entry(memory_content, width=40)
        self.entries["KEEP_LAST_N_STATES"].insert(0, str(self.settings.get("KEEP_LAST_N_STATES", 2)))
        self.entries["KEEP_LAST_N_STATES"].grid(row=14, column=1, sticky=tk.W, padx=5, pady=4)
        tk.Label(memory_content,
                 text="States are big — roughly 470 MB at rank 32, 240 MB at rank 16 — so older ones are "
                      "deleted as new ones are written. Only state dirs for THIS LoRA name are touched, and "
                      "the newest is always kept.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=15, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # Re-sync now that the GC checkbox exists: the earlier _on_quant_4bit_toggle
        # call ran before it was created, so this applies the NF4→force-GC-on lock
        # when a saved config has 4-bit already enabled.
        self._on_quant_4bit_toggle()

        # === MiniMax H3 rows that belong in OTHER sections ===============================
        # Created here rather than up in Training Parameters because Tkinter cannot re-parent
        # a widget, and these two content frames do not exist until this point in the method.
        # Section display order follows CollapsibleFrame construction order, so the sections
        # themselves must not be reordered to make the parents available earlier.
        # Base Precision -> Memory & Precision; the rest -> Other Options.

        # --- Base Precision (MiniMax only) -------------------------------------------------
        # Auto picks the quantisation and the block-swap count TOGETHER. Deciding swap alone,
        # with the precision already fixed by which file you loaded, gives mid-range cards the
        # worst of both: the int8 base is ~21 GB, so a 24 GB card parks 38 of 50 blocks on CPU
        # and crosses PCIe every step for ~4x the runtime, when the same file loaded 4-bit is
        # ~11 GB and needs no swap at all.
        self._minimax_quant_label = ttk.Label(memory_content, text="Base Precision:")
        self._minimax_quant_label.grid(row=16, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_quant_frame = ttk.Frame(memory_content)
        self._minimax_quant_frame.grid(row=16, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_BASE_QUANT"] = ttk.Combobox(
            self._minimax_quant_frame, values=list(MINIMAX_BASE_QUANT_OPTIONS), width=30,
            state="readonly")
        self.entries["MINIMAX_BASE_QUANT"].set(
            str(self.settings.get("MINIMAX_BASE_QUANT", MINIMAX_BASE_QUANT_OPTIONS[0])))
        self.entries["MINIMAX_BASE_QUANT"].pack(side=tk.LEFT)
        self._minimax_quant_hint = ttk.Label(
            memory_content,
            text="Auto reads your FREE VRAM at launch and picks the base precision and block "
                 "swap together — int8 is the most accurate, 4-bit fits smaller cards. Full "
                 "write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_quant_hint.grid(row=17, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Weight averaging (MiniMax only): EMA ------------------------------------------
        # Damage at a high static LR comes from oversized Adam strides: worst at epoch 1
        # (zero-init adapters, steepest surface) and rough thereafter (the weights zigzag around
        # the good solution). EMA addresses the second by saving a smoothed average.
        #
        # WARMUP is retired (Peter, 10 Aug): the Adapter-relative LR ramp is the better answer to
        # the epoch-1 problem — it holds the step/size RATIO steady instead of guessing an epoch
        # count, so it eases in by construction and keeps doing so. The widget is kept (the
        # launch dict and presets still carry the key) but is never packed and is forced Off.
        self._minimax_smooth_label = ttk.Label(scheduler_content, text="Weight averaging (EMA):")
        self._minimax_smooth_label.grid(row=25, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_smooth_frame = ttk.Frame(scheduler_content)
        self._minimax_smooth_frame.grid(row=25, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_LR_WARMUP"] = ttk.Combobox(     # retired — never packed
            self._minimax_smooth_frame, values=["Off", "1 epoch", "2 epochs", "3 epochs"],
            width=10, state="readonly")
        self.entries["MINIMAX_LR_WARMUP"].set("Off")
        self.entries["MINIMAX_EMA"] = ttk.Combobox(
            self._minimax_smooth_frame, values=["Off", "0.98 (light)", "0.99 (recommended)",
                                                "0.995 (strong)"],
            width=18, state="readonly")
        self.entries["MINIMAX_EMA"].set(str(self.settings.get("MINIMAX_EMA", "Off")))
        self.entries["MINIMAX_EMA"].pack(side=tk.LEFT)
        self._minimax_smooth_hint = ttk.Label(
            scheduler_content,
            text="Saves a smoothed average of the weights, so checkpoints come out crisper "
                 "when you push the LR hard. Costs no speed. Full write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_smooth_hint.grid(row=26, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Adapter-relative LR ramp (MiniMax only, EXPERIMENT, default Off) ---------------
        # From a real run: an adapter at ||dW||~53 took a full 2e-4 for ten epochs with no
        # distortion and gave the best likeness of the project, while a fresh adapter is
        # visibly damaged by half that. Same step, 9% perturbation vs 150% — a LoRA starts at
        # zero, so step/size is worst at step 1 and improves from there. This holds that RATIO
        # steady, which ramps the LR up toward the box value as the adapter grows.
        self._minimax_ramp_label = ttk.Label(scheduler_content, text="Adapter-relative LR:")
        self._minimax_ramp_label.grid(row=27, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_ramp_frame = ttk.Frame(scheduler_content)
        self._minimax_ramp_frame.grid(row=27, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_ADAPTER_RAMP"] = ttk.Combobox(
            self._minimax_ramp_frame, values=["Off", "0.003 (slow build)",
                                              "0.005 (recommended)", "0.01 (fast build)"],
            width=24, state="readonly")
        self.entries["MINIMAX_ADAPTER_RAMP"].set(
            str(self.settings.get("MINIMAX_ADAPTER_RAMP", "Off")))
        self.entries["MINIMAX_ADAPTER_RAMP"].pack(side=tk.LEFT)
        self._minimax_ramp_hint = ttk.Label(
            scheduler_content,
            text="Makes the Learning Rate box a CEILING the run climbs toward instead of a rate "
                 "it starts at, so set the LR to where you want to end up. Full write-up in the "
                 "README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_ramp_hint.grid(row=28, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Caption dropout (MiniMax only) -------------------------------------------------
        # A fraction of steps train the image against the EMPTY prompt instead of its caption.
        # On a single-concept set that is healthy regularisation — it stops the LoRA leaning on
        # the trigger token alone. On a MULTI-concept set it is the opposite: those steps teach
        # the model to produce the concept with no trigger at all, which is exactly how one
        # subject leaks into the other's prompts. Was hardcoded at 0.05 (the CLI default) with
        # no way to change it from the GUI.
        self._minimax_capdrop_label = ttk.Label(scheduler_content, text="Caption dropout:")
        self._minimax_capdrop_label.grid(row=29, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_capdrop_frame = ttk.Frame(scheduler_content)
        self._minimax_capdrop_frame.grid(row=29, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_CAPTION_DROPOUT"] = ttk.Combobox(
            self._minimax_capdrop_frame,
            values=["Off", "0.05 (default)", "0.10 (strong)"],
            width=24, state="readonly")
        self.entries["MINIMAX_CAPTION_DROPOUT"].set(
            str(self.settings.get("MINIMAX_CAPTION_DROPOUT", "0.05 (default)")))
        self.entries["MINIMAX_CAPTION_DROPOUT"].pack(side=tk.LEFT)
        self._minimax_capdrop_hint = ttk.Label(
            scheduler_content,
            text="Trains a few percent of steps with no caption, so the LoRA does not lean "
                 "entirely on the trigger word.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_capdrop_hint.grid(row=30, column=0, columnspan=2, sticky=tk.W, padx=5,
                                        pady=(0, 4))

        # --- Blocks to Train (MiniMax only, experimental) ---------------------------------
        self._minimax_blocks_label = ttk.Label(scheduler_content, text="Blocks to Train:")
        self._minimax_blocks_label.grid(row=31, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self._minimax_blocks_frame = ttk.Frame(scheduler_content)
        self._minimax_blocks_frame.grid(row=31, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(8, 2))
        # Editable, not readonly — the presets are starting points and the real control is typing
        # a spec. Anything the trainer's parser takes is legal here.
        self.entries["MINIMAX_BLOCKS"] = ttk.Combobox(
            self._minimax_blocks_frame, values=MINIMAX_BLOCK_OPTIONS, width=34)
        self.entries["MINIMAX_BLOCKS"].pack(side=tk.LEFT)
        self._select_combo_by_token(self.entries["MINIMAX_BLOCKS"],
                                    self.settings.get("MINIMAX_BLOCKS", "all"))
        # Live readout: a typed spec is easy to fat-finger, and "trained 3 blocks when you meant
        # 30" is invisible in the output. Says how many blocks the box currently means.
        self._minimax_blocks_count = tk.Label(self._minimax_blocks_frame, text="",
                                              font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        self._minimax_blocks_count.pack(side=tk.LEFT, padx=(10, 0))
        self.entries["MINIMAX_BLOCKS"].bind(
            "<KeyRelease>", lambda _e: self._refresh_minimax_blocks_count())
        self.entries["MINIMAX_BLOCKS"].bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_minimax_blocks_count())
        self._minimax_blocks_hint = ttk.Label(
            scheduler_content,
            text=self._MINIMAX_BLOCKS_HINT,
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_blocks_hint.grid(row=32, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))
        self._refresh_minimax_blocks_count()

        # Training Structure lives in Training Parameters now — see _build_minimax_structure_row,
        # called from that section. It used to sit here in Other Options, collapsed, which is
        # where the single most consequential MiniMax setting was least likely to be found.

        # === Timestep & Noise Schedule Section (Collapsed by default) ===
        timestep_section = CollapsibleFrame(outer,"Timestep & Noise Schedule", default_expanded=False)
        timestep_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["timestep"] = timestep_section

        ts_content = timestep_section.get_content_frame()
        ts_content.columnconfigure(1, weight=1)

        ts_row = 0

        # Quick Preset Buttons
        preset_btn_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        preset_btn_frame.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(8, 4))

        tk.Label(preset_btn_frame, text="Quick Presets:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))

        for preset_name, preset_fn in [
            ("Full Range", self._ts_preset_full_range),
            ("Structure Focus", self._ts_preset_structure),
            ("Detail Focus", self._ts_preset_detail),
            ("Balanced Sigmoid", self._ts_preset_sigmoid),
        ]:
            btn = ttk.Button(preset_btn_frame, text=preset_name, command=preset_fn)
            btn.pack(side=tk.LEFT, padx=2)

        ts_row += 1

        # Timestep Sampling (editable dropdown)
        ts_sampling_label = tk.Label(ts_content, text="Timestep Sampling:",
                                     font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_sampling_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.ts_sampling_var = tk.StringVar(value=self.settings["TIMESTEP_SAMPLING"])
        # Must mirror the trainer's argparse choices exactly — "qwen_shift" was offered here
        # but rejected by the trainer at launch; "qinglong_flux" existed but wasn't offered.
        ts_sampling_options = ["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "logsnr", "qinglong_flux"]
        self.ts_sampling_combo = ttk.Combobox(ts_content, textvariable=self.ts_sampling_var,
                                               values=ts_sampling_options, state="readonly", width=20)
        self.ts_sampling_combo.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        self.ts_sampling_combo.bind("<<ComboboxSelected>>", self._on_timestep_sampling_changed)
        self.entries["TIMESTEP_SAMPLING"] = self.ts_sampling_combo
        ts_row += 1

        # Discrete Flow Shift — not used by Klein 9B (uses flux2_shift automatic).
        # Widget created but not gridded so presets/save still work.
        self.ts_flow_shift_label = tk.Label(ts_content, text="Discrete Flow Shift:",
                                            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.entries["DISCRETE_FLOW_SHIFT"] = ttk.Entry(ts_content, width=12)
        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, self.settings["DISCRETE_FLOW_SHIFT"])

        # Sigmoid Scale
        self.ts_sigmoid_label = tk.Label(ts_content, text="Sigmoid Scale:",
                                         font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_sigmoid_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.entries["SIGMOID_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["SIGMOID_SCALE"].insert(0, self.settings["SIGMOID_SCALE"])
        self.entries["SIGMOID_SCALE"].grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        ts_row += 1

        # Min / Max Timestep on one row
        ts_range_label = tk.Label(ts_content, text="Timestep Range:",
                                  font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_range_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        ts_range_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        ts_range_frame.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(ts_range_frame, text="Min:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MIN_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MIN_TIMESTEP"].insert(0, self.settings["MIN_TIMESTEP"])
        self.entries["MIN_TIMESTEP"].pack(side=tk.LEFT, padx=(0, 16))
        self.entries["MIN_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        tk.Label(ts_range_frame, text="Max:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MAX_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MAX_TIMESTEP"].insert(0, self.settings["MAX_TIMESTEP"])
        self.entries["MAX_TIMESTEP"].pack(side=tk.LEFT)
        self.entries["MAX_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        ts_row += 1

        # Noise range description label
        self.noise_range_label = tk.Label(ts_content, text="", font=(FONT_FAMILY, 9),
                                          fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.noise_range_label.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=(0, 4))
        self._update_noise_range_label()
        ts_row += 1

        # Preserve Distribution Shape
        self.preserve_dist_var = tk.BooleanVar(value=self.settings["PRESERVE_DISTRIBUTION"])
        self.preserve_dist_check = ttk.Checkbutton(ts_content, text="Preserve Distribution Shape",
                                                    variable=self.preserve_dist_var, style="Surface.TCheckbutton")
        self.preserve_dist_check.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=4)
        ToolTip(self.preserve_dist_check, "Use rejection sampling to preserve the original\n"
                "distribution shape within the min/max range.\n"
                "Only effective when min/max timestep is set.")
        self.entries["PRESERVE_DISTRIBUTION"] = self.preserve_dist_var
        ts_row += 1

        # --- Hidden fields (not used by Klein 9B) ---
        # Widgets created but not gridded so presets/save/command-building still work.

        # Weighting Scheme
        self.ts_weighting_label = tk.Label(ts_content, text="Weighting Scheme:")
        self.weighting_scheme_var = tk.StringVar(value=self.settings["WEIGHTING_SCHEME"])
        weighting_options = ["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"]
        self.ts_weighting_combo = ttk.Combobox(ts_content, textvariable=self.weighting_scheme_var,
                                                values=weighting_options, state="readonly", width=20)
        self.ts_weighting_combo.bind("<<ComboboxSelected>>", self._on_weighting_scheme_changed)
        self.entries["WEIGHTING_SCHEME"] = self.ts_weighting_combo

        # Logit Mean / Std
        self.ts_logit_label = tk.Label(ts_content, text="Logit Normal:")
        logit_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        self.entries["LOGIT_MEAN"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_MEAN"].insert(0, self.settings["LOGIT_MEAN"])
        self.entries["LOGIT_STD"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_STD"].insert(0, self.settings["LOGIT_STD"])

        # Mode Scale
        self.ts_mode_label = tk.Label(ts_content, text="Mode Scale:")
        self.entries["MODE_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["MODE_SCALE"].insert(0, self.settings["MODE_SCALE"])

        # Initial state for conditional fields
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()

        # === Reorder collapsible sections: Training → Memory & FP8 → Timestep → Optimizer → Other Options ===
        # Sections were created in declaration order; re-pack in the desired display order.
        # Training Parameters and Memory & FP8 are open by default since most users need both.
        for _sec in (training_section, memory_section, timestep_section, optimizer_section, scheduler_section):
            try:
                _sec.pack_forget()
                _sec.pack(fill=tk.X, padx=36, pady=(0, 16))
            except Exception:
                pass

        # === Run card — Enable Cache + Start/Pause/Resume/Stop buttons ===
        run_card = self._start_section_card(outer, "Run", None)

        self.enable_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_card, text="Enable Cache Preparation", variable=self.enable_cache_var).pack(anchor=tk.W, pady=(0, 12))

        button_frame = tk.Frame(run_card, bg=COLORS["bg_surface"])
        button_frame.pack(anchor=tk.W)

        self._start_training_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training, style="Primary.TButton")
        self._start_training_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._pause_training_btn = ttk.Button(button_frame, text="Pause Training", command=self._pause_training)
        self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._pause_training_btn.pack_forget()  # hidden until training is running

        self._resume_training_btn = ttk.Button(button_frame, text="Resume Training", command=self._resume_training, style="Primary.TButton")
        self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._resume_training_btn.pack_forget()  # hidden until paused state exists

        stop_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, style="Danger.TButton")
        stop_btn.pack(side=tk.LEFT, padx=(0, 24))

        ttk.Button(button_frame, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT)

        # === Console Output card ===
        console_card = self._start_section_card(outer, "Console Output", None)
        self.console_frame = tk.Frame(console_card, bg=COLORS["bg_surface"])
        self.console_frame.pack(fill=tk.BOTH, expand=True)

        self.console_output = tk.Text(
            self.console_frame,
            height=12,
            width=80,
            bg=COLORS["bg_deep"],
            fg=COLORS["text_primary"],
            font=(FONT_MONO, 9),
            wrap="word",
            state="disabled",
            selectbackground=COLORS["accent"],
            selectforeground="white",
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border_focus"],
            padx=12,
            pady=8
        )
        self.console_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console_scrollbar = ttk.Scrollbar(
            self.console_frame,
            orient="vertical",
            command=self.console_output.yview,
            style="Vertical.TScrollbar"
        )
        self.console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_output.configure(yscrollcommand=self.console_scrollbar.set)

        self.console_output.bind("<MouseWheel>", self.on_mousewheel)
        self.console_output.bind("<Button-4>", self.on_mousewheel)
        self.console_output.bind("<Button-5>", self.on_mousewheel)
        self.console_output.bind("<Button-3>", self.show_context_menu)

        # Initial UI update based on architecture
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()

        self._add_youtube_help_button(outer, "training")

    def on_architecture_changed(self, event=None):
        """Handle architecture change"""
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()
        self.load_default_preset(show_message=False)  # Auto-load defaults for new architecture
        # Update samples tab UI for new architecture
        if hasattr(self, 'sample_settings_frame'):
            self.update_samples_ui_for_architecture()

    def get_preset_dir_for_architecture(self, arch):
        """Get the preset directory for an architecture, creating if needed"""
        preset_dir = os.path.join(PRESETS_DIR, arch)
        # One-shot folder rename from the pre-2026-07-28 arch name, or user-saved Krea 2
        # presets would silently vanish from the dropdown.
        if arch == "Krea 2" and not os.path.isdir(preset_dir):
            _legacy = os.path.join(PRESETS_DIR, "Krea 2 (experimental)")
            if os.path.isdir(_legacy):
                try:
                    os.rename(_legacy, preset_dir)
                except OSError:
                    pass
        os.makedirs(preset_dir, exist_ok=True)
        return preset_dir

    def get_saved_presets(self, arch):
        """Get list of saved preset names for an architecture"""
        preset_dir = self.get_preset_dir_for_architecture(arch)
        presets = []
        for f in os.listdir(preset_dir):
            if f.endswith('.json'):
                presets.append(f[:-5])  # Remove .json extension
        return sorted(presets)

    def _builtins_for_arch(self, arch):
        """Return the built-in preset dict for an architecture. Krea 2 gets its single
        defaults entry (Klein's block/timestep/adaptive presets don't apply); everything
        else gets the full Klein built-in set."""
        cfg = ARCHITECTURES.get(arch, {})
        if cfg.get("is_minimax"):
            return MINIMAX_BUILT_IN_PRESETS
        return KREA2_BUILT_IN_PRESETS if cfg.get("is_krea2") else BUILT_IN_PRESETS

    def _update_preset_hint(self):
        """The bracketed note beside Load Preset: visible only while the MiniMax rank-16
        Defaults preset is the selection — with Fast as the shipped default, this label is
        what tells the user when the bigger recipe is the right reach."""
        lbl = getattr(self, "_preset_hint_label", None)
        if lbl is None:
            return
        try:
            if self._is_minimax_arch() and self.custom_preset_var.get() == _MM_DEFAULTS_KEY:
                lbl.pack(side=tk.LEFT, padx=(8, 0))
            else:
                lbl.pack_forget()
        except Exception:
            pass

    def refresh_preset_combobox(self):
        """Refresh the preset combobox: built-in presets first, then user-saved presets."""
        arch = self.architecture_var.get()
        user_presets = self.get_saved_presets(arch)
        builtins_map = self._builtins_for_arch(arch)
        builtins = list(builtins_map.keys())
        # Built-ins first; if a user saves a preset with same name as a built-in, it appears once (under user)
        combined = builtins + [p for p in user_presets if p not in builtins_map]
        self.custom_preset_combo['values'] = combined
        # Dynamic width: fit longest entry so names like "✨ Multi-Character (rank 16, noisy dataset)" don't truncate
        max_len = max((len(v) for v in combined), default=20)
        self.custom_preset_combo.config(width=max(20, min(max_len + 2, 60)))
        self.custom_preset_var.set('')  # Clear selection

    def load_default_preset(self, show_message=True):
        """Load recommended preset values for the current architecture"""
        arch = self.architecture_var.get()
        if arch not in PRESETS:
            if show_message:
                messagebox.showinfo("Info", f"No preset available for {arch}")
            return

        preset = PRESETS[arch]
        self._apply_preset_values(preset)
        if show_message:
            messagebox.showinfo("Preset Loaded", f"Loaded recommended preset for {arch}")

    # Comboboxes whose values feed directly into the launch command: a saved value the
    # current family doesn't offer (cross-family last-train leak, withdrawn LR floors,
    # removed optimizers) must NOT be .set() onto them — readonly Comboboxes accept any
    # value without complaint, and the bad name then dies (or misbehaves) at launch.
    _STRICT_COMBO_KEYS = {"OPTIMIZER_TYPE", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX", "LR_SCHEDULER",
                          "NETWORK_TYPE"}

    def _apply_preset_values(self, preset):
        """Apply preset values to the UI (shared by load_default_preset and load_custom_preset)"""
        for key, value in preset.items():
            if key in self.entries:
                entry = self.entries[key]
                if isinstance(entry, ttk.Combobox):
                    value = str(value)
                    try:
                        # A saved plain value (e.g. "2e-4") should select its labeled combobox
                        # entry ("2e-4 - rank 4/8 only") so the warning suffix still shows.
                        opts = entry.cget("values") or ()
                        if value not in opts:
                            for opt in opts:
                                if str(opt).split(" ")[0] == value.split(" ")[0]:
                                    value = str(opt)
                                    break
                        if value not in opts and key in self._STRICT_COMBO_KEYS:
                            self.update_console(
                                f"[preset] {key}: saved value {value!r} isn't offered here — "
                                f"keeping {entry.get()!r}\n")
                            continue
                    except tk.TclError:
                        pass
                    entry.set(value)
                elif isinstance(entry, tk.BooleanVar):
                    # Some boolean settings (e.g. IMG_IN_TXT_IN_OFFLOADING, PRESERVE_DISTRIBUTION)
                    # are stored in self.entries as BooleanVars — they don't support .delete/.insert.
                    entry.set(bool(value))
                else:
                    try:
                        entry.delete(0, tk.END)
                        entry.insert(0, str(value))
                    except (AttributeError, tk.TclError):
                        # Unknown widget type — skip rather than crash
                        pass

        # Update timestep settings from preset. Validate against what the trainer accepts —
        # old presets can carry values a past version offered (e.g. "qwen_shift", which
        # argparse rejects at launch).
        if "TIMESTEP_SAMPLING" in preset:
            _ts_val = str(preset["TIMESTEP_SAMPLING"])
            _ts_ok = ("sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift",
                      "logsnr", "qinglong_flux")
            if _ts_val in _ts_ok:
                self.ts_sampling_var.set(_ts_val)
            else:
                self.update_console(f"[preset] TIMESTEP_SAMPLING {_ts_val!r} isn't supported — "
                                    f"keeping {self.ts_sampling_var.get()!r}\n")
        if "WEIGHTING_SCHEME" in preset:
            self.weighting_scheme_var.set(preset["WEIGHTING_SCHEME"])
        if "PRESERVE_DISTRIBUTION" in preset:
            self.preserve_dist_var.set(preset["PRESERVE_DISTRIBUTION"])
        # Refresh conditional field states after preset load
        if hasattr(self, 'ts_sampling_var'):
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()
        # Network Type drives a ROW SWAP (rank/alpha <-> LoKR factor), and setting a combobox
        # programmatically does NOT fire <<ComboboxSelected>> — so without this a preset that
        # changes the type left the old rows on screen: "LoRA (standard)" selected with the
        # LoKR Factor box still underneath it.
        if "NETWORK_TYPE" in preset and "LOKR_FACTOR" in getattr(self, "rows", {}):
            self._on_network_type_changed()

        # Update FP8/SCALED checkboxes from preset
        if "FP8" in preset:
            self.fp8_var.set(preset["FP8"])
        if "SCALED" in preset:
            self.scaled_var.set(preset["SCALED"])

        # Base precision — a dedicated var (not in self.entries), so the generic loop never
        # restores it. _normalize_base_precision takes the canonical key, the display label,
        # or a legacy Auto/On/Off alike, so old presets keep working.
        if hasattr(self, 'quant_4bit_mode_var'):
            _key = None
            if "QUANT_4BIT_MODE" in preset:
                _key = self._normalize_base_precision(preset["QUANT_4BIT_MODE"])
            elif "QUANT_4BIT" in preset:
                # Legacy boolean. False means "4-bit not requested", i.e. no opinion — NOT an
                # explicit demand for fp8, which would pin every old preset (including Klein's
                # defaults, which carry QUANT_4BIT: False) away from Auto.
                _key = "nf4" if bool(preset["QUANT_4BIT"]) else "auto"
            if _key:
                self.quant_4bit_mode_var.set(self._BASE_PRECISION_LABELS[_key])
                self._on_quant_4bit_mode_changed()

        # torch.compile mode — collected by _collect_preset_values but, until now, never
        # restored, so a preset's COMPILE_BLOCKS was silently dropped on load.
        if "COMPILE_BLOCKS" in preset and hasattr(self, 'compile_blocks_var'):
            _cb = str(preset["COMPILE_BLOCKS"]).capitalize()
            if _cb in ("Auto", "On", "Off"):
                self.compile_blocks_var.set(_cb)

        # Save-state toggles — dedicated vars, same not-in-self.entries situation. Presets saved
        # before these existed simply don't carry the keys, so they keep the current setting.
        if "SAVE_STATE" in preset and hasattr(self, 'save_state_var'):
            self.save_state_var.set(bool(preset["SAVE_STATE"]))
        if "SAVE_STATE_ON_TRAIN_END" in preset and hasattr(self, 'save_state_on_train_end_var'):
            self.save_state_on_train_end_var.set(bool(preset["SAVE_STATE_ON_TRAIN_END"]))

        # Per-image loss watch toggles (krea2) — dedicated vars, same not-in-self.entries situation.
        if "KREA2_LOSS_WATCH" in preset and hasattr(self, 'krea2_loss_watch_var'):
            self.krea2_loss_watch_var.set(bool(preset["KREA2_LOSS_WATCH"]))
        if "KREA2_PER_IMAGE_LR" in preset and hasattr(self, 'krea2_per_image_lr_var'):
            self.krea2_per_image_lr_var.set(bool(preset["KREA2_PER_IMAGE_LR"]))
        if "KREA2_AUTO_RECAPTION" in preset and hasattr(self, 'krea2_auto_recaption_var'):
            self.krea2_auto_recaption_var.set(bool(preset["KREA2_AUTO_RECAPTION"]))
        if "KREA2_WARMUP_LOOK" in preset and hasattr(self, 'krea2_warmup_look_var'):
            self.krea2_warmup_look_var.set(bool(preset["KREA2_WARMUP_LOOK"]))

        # Adaptive LR checkbox + sync enabled state of Min/Max LR dropdowns
        if "ADAPTIVE_LR" in preset and hasattr(self, 'adaptive_lr_var'):
            self.adaptive_lr_var.set(bool(preset["ADAPTIVE_LR"]))
            if hasattr(self, '_on_adaptive_lr_toggle'):
                self._on_adaptive_lr_toggle()
            # Re-apply the Learning Rate AFTER the toggle has settled. The entries loop above
            # runs while the LR box still reflects the PREVIOUS adaptive state — and a write to
            # a disabled ttk.Entry is silently dropped. Restoring a non-adaptive run while
            # adaptive happened to be on therefore kept the old rate, which on a fine-tune is
            # the difference between 1e-5 and a LoRA-grade 1e-4.
            _lr_entry = self.entries.get("LEARNING_RATE")
            if "LEARNING_RATE" in preset and _lr_entry is not None:
                try:
                    _was = _lr_entry.cget("state")
                    _lr_entry.config(state="normal")
                    _lr_entry.delete(0, tk.END)
                    _lr_entry.insert(0, str(preset["LEARNING_RATE"]))
                    _lr_entry.config(state=_was)
                except (AttributeError, tk.TclError):
                    pass

        # Krea 2 base-model fine-tune. Captured by _collect_preset_values but never applied
        # back, so "Load Settings From Last Train" silently dropped the entire fine-tune
        # config — mode, window size, fused backward, Fast FT and the regularisation set.
        # Order matters: the fine-tune flag goes first, because the regularisation block is
        # only written to the dataset TOML while fine-tune is on.
        _ft_map = [
            ("KREA2_FINETUNE", "krea2_finetune_var", bool),
            ("KREA2_FT_MODE", "krea2_ft_mode_var", str),
            ("KREA2_FT_BLOCKS", "krea2_ft_blocks_var", str),
            ("KREA2_FT_EVERY", "krea2_ft_every_var", str),
            ("KREA2_FT_FUSED", "krea2_ft_fused_var", bool),
            ("KREA2_FAST_FT", "krea2_fast_ft_var", bool),
            ("KREA2_REG_DIR", "krea2_reg_dir_var", str),
            ("KREA2_REG_MULT", "krea2_reg_mult_var", str),
            ("MINIMAX_FINETUNE", "minimax_finetune_var", bool),
            ("MINIMAX_FT_EVERY", "minimax_ft_every_var", str),
            ("MINIMAX_FT_SCOPE", "minimax_ft_scope_var", str),
            ("MINIMAX_FT_BLOCKSPEC", "minimax_ft_blockspec_var", str),
            ("MINIMAX_FT_FUSED", "minimax_ft_fused_var", bool),
            ("MINIMAX_REG_DIR", "minimax_reg_dir_var", str),
            ("MINIMAX_REG_MULT", "minimax_reg_mult_var", str),
        ]
        _ft_touched = False
        for _key, _attr, _cast in _ft_map:
            if _key in preset and hasattr(self, _attr):
                try:
                    getattr(self, _attr).set(_cast(preset[_key]))
                    _ft_touched = True
                except Exception:
                    pass
        if _ft_touched:
            # Show/hide the fine-tune panel to match, and rewrite the dataset TOML so the
            # regularisation block tracks the restored state rather than the previous run's.
            if hasattr(self, "_apply_krea2_ft_visibility"):
                self._apply_krea2_ft_visibility()
            if hasattr(self, "_apply_minimax_ft_visibility"):
                self._apply_minimax_ft_visibility()
            if hasattr(self, "auto_save_dataset_config_silent"):
                self.auto_save_dataset_config_silent()

        # LEARNING_RATE is state-gated: the adaptive checkbox greys the LR box, and a tk
        # Entry silently DROPS delete/insert while disabled — so the generic loop above
        # lost the preset's LR whenever adaptive was on at that moment (e.g. Old Reliable
        # active, then loading Identity kept 1e-4 instead of 4e-4). Re-apply after the
        # adaptive toggle has settled, forcing the widget writable for the write.
        if "LEARNING_RATE" in preset and "LEARNING_RATE" in self.entries:
            _lr_ent = self.entries["LEARNING_RATE"]
            try:
                _prev_state = str(_lr_ent.cget("state"))
                _lr_ent.config(state="normal")
                _lr_ent.delete(0, tk.END)
                _lr_ent.insert(0, str(preset["LEARNING_RATE"]))
                _lr_ent.config(state=_prev_state)
            except (AttributeError, tk.TclError):
                pass

        if "MINIMAX_DISTILL" in preset and hasattr(self, "minimax_distill_var"):
            self.minimax_distill_var.set(bool(preset["MINIMAX_DISTILL"]))

        # Multi Concept: a BooleanVar plus a LIST of folders, so neither is reachable by the
        # generic self.entries loop above. Restore the folders BEFORE the toggle so the handler
        # that rewrites the TOML and locks caption dropout sees the finished state.
        if "MINIMAX_CONCEPT_DIRS" in preset:
            _dirs = preset.get("MINIMAX_CONCEPT_DIRS") or []
            if isinstance(_dirs, str):                     # tolerate an older single-string save
                _dirs = [_dirs] if _dirs.strip() else []
            for _i, _v in enumerate(getattr(self, "_concept_folder_vars", [])):
                _v.set(str(_dirs[_i]).strip() if _i < len(_dirs) else "")
        if "MINIMAX_MULTICONCEPT" in preset and hasattr(self, "minimax_multiconcept_var"):
            self.minimax_multiconcept_var.set(bool(preset["MINIMAX_MULTICONCEPT"]))
        # Re-run unconditionally: a preset that carries MINIMAX_CAPTION_DROPOUT (the Defaults one
        # does) would otherwise leave the box showing 0.05 while Multi Concept is on. Training
        # was never at risk - the command builder locks it either way - but the UI would lie.
        if hasattr(self, "minimax_multiconcept_var"):
            try:
                self._on_minimax_multiconcept_toggle()
            except Exception:
                pass
        try:
            self._sync_distill_weight_state()
        except Exception:
            pass

        # Model Area to Train (training preset dropdown)
        if "TARGET_LAYERS" in preset and hasattr(self, 'training_preset_var'):
            legacy_map = {
                "All Layers": "Full Model",
                "Identity Blocks": "Identity",
                "Style+Composition Blocks": "Style+Composition",
                "Details Blocks": "Details",
            }
            raw = preset["TARGET_LAYERS"]
            mapped = legacy_map.get(raw, raw)
            valid = ("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom")
            self.training_preset_var.set(mapped if mapped in valid else "Full Model")
            if hasattr(self, '_on_training_preset_changed'):
                self._on_training_preset_changed()
                # _on_training_preset_changed auto-fills MIN/MAX_TIMESTEP from the Model
                # Area — which just overwrote the values the generic loop restored from
                # the preset. The preset's explicit values are the user's saved choice:
                # put them back.
                for _ts_key in ("MIN_TIMESTEP", "MAX_TIMESTEP"):
                    if _ts_key in preset and _ts_key in self.entries:
                        try:
                            self.entries[_ts_key].delete(0, tk.END)
                            self.entries[_ts_key].insert(0, str(preset[_ts_key]))
                        except Exception:
                            pass
        if "FP8_TEXT_ENCODER" in preset:
            self.fp8_text_encoder_var.set(preset["FP8_TEXT_ENCODER"])
        if "ENABLE_BUCKET" in preset:
            self.dataset_enable_bucket_var.set(preset["ENABLE_BUCKET"])
        if "BUCKET_NO_UPSCALE" in preset:
            self.dataset_no_upscale_var.set(preset["BUCKET_NO_UPSCALE"])
        # Dataset subsection (Training → Other Options → Dataset)
        if "DATASET_CAPTION_EXT" in preset and hasattr(self, "dataset_caption_ext_var"):
            self.dataset_caption_ext_var.set(preset["DATASET_CAPTION_EXT"])
        if "DATASET_MEGAPIXELS" in preset and hasattr(self, "dataset_megapixels_var"):
            self.dataset_megapixels_var.set(preset["DATASET_MEGAPIXELS"])
        if "DATASET_BATCH_SIZE" in preset and hasattr(self, "dataset_batch_size_var"):
            self.dataset_batch_size_var.set(preset["DATASET_BATCH_SIZE"])
        # Run card's Enable Cache checkbox
        if "ENABLE_CACHE" in preset and hasattr(self, "enable_cache_var"):
            self.enable_cache_var.set(bool(preset["ENABLE_CACHE"]))
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if "TRAINING_BLOCKS" in preset and hasattr(self, "training_block_vars"):
            for block_key, block_on in preset["TRAINING_BLOCKS"].items():
                if block_key in self.training_block_vars:
                    self.training_block_vars[block_key].set(bool(block_on))
        # Gradient mining
        if "GRADIENT_MINING" in preset and hasattr(self, "gradient_mining_var"):
            self.gradient_mining_var.set(bool(preset["GRADIENT_MINING"]))
        self.toggle_scaled()  # Update checkbox state

    def _save_last_train_settings(self):
        """Snapshot current settings just before launching training, so 'Load Last Train' can restore them."""
        if _persist_disabled():
            return   # headless tests must never overwrite the real .last_train_settings.json
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            snapshot = self._collect_preset_values()
            # Presets deliberately don't carry the family (a Krea 2 preset must not hijack your
            # model choice), but "restore my last launch" plainly includes WHICH model it was —
            # the same reasoning the training queue uses when it stores the architecture beside
            # its snapshot. Namespaced so _apply_preset_values ignores it as an unknown key.
            snapshot["__architecture__"] = self.architecture_var.get()
            # Training Base is preset-immune (outside self.entries, never collected) but
            # "restore my last launch" plainly includes which base it ran on — same reasoning
            # as the architecture above. Namespaced so _apply_preset_values ignores it.
            if hasattr(self, "minimax_train_base_var"):
                snapshot["__minimax_train_base__"] = minimax_train_base(
                    self.minimax_train_base_var.get())
            with open(LAST_TRAIN_FILE, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            print(f"[last_train] Failed to save snapshot: {e}")

    def _load_last_train_settings(self):
        """Restore settings from the most recent training launch."""
        if not os.path.exists(LAST_TRAIN_FILE):
            messagebox.showinfo(
                "No Last Train",
                "No previous training settings found.\n\n"
                "Launch a training run first; afterwards this button will restore those settings."
            )
            return
        try:
            with open(LAST_TRAIN_FILE, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            # Switch family FIRST if the launch was on a different one: on_architecture_changed
            # loads that family's default preset, so doing it after would clobber everything the
            # snapshot just restored. Older snapshots have no architecture — they simply skip this.
            _arch = snapshot.pop("__architecture__", None)
            _switched = ""
            _arch = _canon_arch(_arch) if _arch else _arch
            if _arch and _arch in ARCHITECTURES and _arch != self.architecture_var.get():
                self.architecture_var.set(_arch)
                self.on_architecture_changed()
                _switched = f"\n\nSwitched the Base Model back to {_arch}."
            # Training Base rides beside the preset, not in it (preset-immune by design) —
            # pop before applying so _apply_preset_values never sees it even by accident.
            _base = snapshot.pop("__minimax_train_base__", None)
            self._apply_preset_values(snapshot)
            if _base and hasattr(self, "minimax_train_base_var"):
                self.minimax_train_base_var.set(MINIMAX_TRAIN_BASE_OPTIONS[
                    1 if minimax_train_base(_base) == "ref2va" else 0])
            messagebox.showinfo("Loaded",
                                f"Restored settings from your last training launch.{_switched}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load last train settings:\n{e}")

    # ------------------------------------------------------------
    # Training queue — settings snapshots that run back-to-back
    # ------------------------------------------------------------
    # A queue item is everything a run needs that the GUI would otherwise read live:
    # the preset snapshot (_collect_preset_values), the architecture (presets are
    # per-arch and deliberately don't carry it), the Start-tab dataset folder, and the
    # Samples-tab entries (presets deliberately skip those too). Restoring an item is
    # "load these into the GUI, then press Start" — the queue never bypasses
    # start_training, so validation, TOML regeneration, snapshotting and the pause
    # machinery all behave exactly as for a hand-started run.

    def _load_training_queue(self):
        if _persist_disabled():
            return []
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                print(f"[queue] {QUEUE_FILE} does not hold a list — starting with an empty queue")
                return []
            good = [i for i in items if self._queue_item_valid(i)]
            if len(good) != len(items):
                print(f"[queue] dropped {len(items) - len(good)} unreadable entr(ies) from {QUEUE_FILE}")
            return good
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"[queue] failed to load {QUEUE_FILE}: {e}")
            return []

    def _save_training_queue(self):
        if _persist_disabled():
            return
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            tmp = QUEUE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.training_queue, f, indent=2, default=str)
            os.replace(tmp, QUEUE_FILE)
        except Exception as e:
            print(f"[queue] failed to save: {e}")

    @staticmethod
    def _queue_item_valid(item):
        """Deep-enough validation for anything about to flow into apply/summary/advance.
        Shallow checks (dict with a dict preset) let hand-edited variants through that then
        crashed AFTER the item was popped and saved away — losing it before the traceback."""
        return (isinstance(item, dict)
                and isinstance(item.get("preset"), dict)
                and isinstance(item.get("image_folder", ""), str)
                and isinstance(item.get("architecture", ""), str)
                and isinstance(item.get("samples", {}), dict))

    def _schedule_queue_advance(self, delay_ms):
        """The ONE way to arm a queue-advance timer. A generation counter makes every
        previously-armed timer a no-op: Stop, Pause, a failure-HOLD, or a manual start bumps
        the generation, so a stale after() callback from before the state change can never
        fire into a paused/held queue or double-launch across a pipeline phase gap."""
        gen = getattr(self, "_queue_advance_gen", 0)

        def _tick():
            if getattr(self, "_queue_advance_gen", 0) == gen:
                self._start_next_queued()
        self.master.after(delay_ms, _tick)

    def _cancel_pending_queue_advance(self):
        self._queue_advance_gen = getattr(self, "_queue_advance_gen", 0) + 1

    _QUEUE_SAMPLE_KEYS = ("SAMPLE_ENABLED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
                          "SAMPLE_SEED", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_EVERY_N_STEPS",
                          "SAMPLE_AT_FIRST", "SAMPLE_FLOW_SHIFT", "SAMPLE_NEGATIVE",
                          "SAMPLE_CFG_SCALE", "SAMPLE_FRAMES",
                          "MINIMAX_TURBO_STEPS", "MINIMAX_TURBO_STRENGTH")

    def _queue_snapshot(self):
        """Capture the currently configured run as a queue item."""
        import time as _time
        samples = {}
        for k in self._QUEUE_SAMPLE_KEYS:
            entry = self.entries.get(k)
            if entry is None:
                continue
            try:
                samples[k] = entry.get()
            except Exception:
                pass
        return {
            "id": f"q{int(_time.time() * 1000)}",
            "queued_at": _time.strftime("%Y-%m-%d %H:%M"),
            "architecture": self.architecture_var.get(),
            "image_folder": self.image_folder_var.get().strip(),
            # A queued Multi Concept run loses its second subject without this.
            "concept_folders": [v.get().strip() for v in
                                getattr(self, "_concept_folder_vars", [])],
            "preset": self._collect_preset_values(),
            "samples": samples,
            # Training Base rides beside the preset, not in it (preset-immune by design) — a
            # queued ref2va run would otherwise silently launch on fl2va.
            "minimax_train_base": minimax_train_base(
                getattr(self, "minimax_train_base_var", None)
                and self.minimax_train_base_var.get()),
        }

    def _apply_queue_item(self, item):
        """Load a queue item's settings back into the GUI (arch first — it swaps the UI)."""
        arch = _canon_arch(item.get("architecture", ""))
        if isinstance(arch, str) and arch and arch in ARCHITECTURES and self.architecture_var.get() != arch:
            self.architecture_var.set(arch)
            try:
                self.update_ui_for_architecture()
            except Exception as e:
                self.update_console(f"[queue] arch switch to {arch!r} failed: {e}\n")
        self._apply_preset_values(item.get("preset", {}))
        # Items queued before the Training Base dropdown existed carry no key — leave the
        # dropdown as it stands rather than forcing a default onto an old queue file.
        _base = item.get("minimax_train_base")
        if _base and hasattr(self, "minimax_train_base_var"):
            self.minimax_train_base_var.set(MINIMAX_TRAIN_BASE_OPTIONS[
                1 if minimax_train_base(_base) == "ref2va" else 0])
        folder = str(item.get("image_folder") or "").strip()
        if folder:
            self.image_folder_var.set(folder)   # traces regenerate Fizgig_train.toml
        # Multi Concept's extra folders ride separately: they are not the Start folder and must
        # not overwrite it. Restored before the toggle so the TOML rewrite sees them.
        _cf = item.get("concept_folders") or []
        for _i, _v in enumerate(getattr(self, "_concept_folder_vars", [])):
            _v.set(str(_cf[_i]).strip() if _i < len(_cf) else "")
        _samples = item.get("samples")
        for k, v in (_samples.items() if isinstance(_samples, dict) else ()):
            entry = self.entries.get(k)
            if entry is None:
                continue
            try:
                if isinstance(entry, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                    entry.set(v)
                elif isinstance(entry, ttk.Combobox):
                    entry.set(str(v))
                else:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(v))
            except Exception:
                pass

    @staticmethod
    def _queue_signature(item):
        """What makes two queue entries THE SAME RUN: everything except id/queued_at."""
        try:
            return json.dumps({k: item.get(k) for k in
                               ("architecture", "image_folder", "preset", "samples",
                                "minimax_train_base")},
                              sort_keys=True, default=str)
        except Exception:
            return repr(item)

    @staticmethod
    def _queue_output_key(item):
        """(output dir, LoRA name) — two runs writing here overwrite each other's files."""
        p = item.get("preset", {}) if isinstance(item.get("preset"), dict) else {}
        return (str(p.get("LORA_OUTPUT_DIR", "")).strip().lower().replace("\\", "/").rstrip("/"),
                str(p.get("LORA_NAME", "")).strip().lower())

    def _queue_current_run(self):
        """Snapshot the current config to the end of the queue (Start pressed mid-run)."""
        # Queueing skips validate_inputs entirely (Start returns above), so without this a bad
        # name is written into the queue file, compared dirty by the clash check below, and only
        # rejected an hour later when the queue tries to launch it — modal, unattended, held.
        _name, _name_error = self._tidy_lora_name()
        if _name_error:
            messagebox.showwarning("Check the LoRA name", _name_error)
            return
        item = self._queue_snapshot()
        if not item["image_folder"]:
            messagebox.showwarning(
                "Nothing to queue",
                "Pick a training image folder on the Start tab first — a queued run "
                "needs to know its dataset.")
            return
        # An exact duplicate (same everything) is never useful — it would just train the
        # identical run twice. Point at the existing entry instead of adding another.
        sig = self._queue_signature(item)
        for pos, q in enumerate(self.training_queue):
            if self._queue_signature(q) == sig:
                messagebox.showinfo(
                    "Already queued",
                    f"This exact run is already in the queue (position {pos + 1}).\n\n"
                    "Change something — the dataset, the output name, any setting — "
                    "to queue a different run.")
                return
        # Same output dir + name as another queued job (or the run in progress) with
        # DIFFERENT settings: the later run would overwrite the earlier one's checkpoints,
        # state dirs and samples. Flag it; queueing anyway is a legitimate choice.
        okey = self._queue_output_key(item)
        if okey != ("", ""):
            clash = next((f"queued job {pos + 1}" for pos, q in enumerate(self.training_queue)
                          if self._queue_output_key(q) == okey), None)
            if clash is None:
                _active = getattr(self, "_active_run_item", None)
                if (_active is not None
                        and getattr(self, "training_state", "idle") in ("running", "pausing")
                        and self._queue_output_key(_active) == okey):
                    clash = "the run in progress"
            if clash is not None and not messagebox.askyesno(
                    "Same output name",
                    f"This run writes to the same output folder and LoRA name as {clash} — "
                    f"its checkpoints, state dirs and samples would be overwritten.\n\n"
                    f"Queue it anyway? (Change the Output Name to keep both.)"):
                return
        self.training_queue.append(item)
        self._save_training_queue()
        self._refresh_queue_button()
        self._render_queue_window()
        name = item["preset"].get("LORA_NAME") or os.path.basename(item["image_folder"])
        self.update_console(f"[queue] added '{name}' — position {len(self.training_queue)} in the "
                            f"queue. It starts automatically when the current run finishes.\n")

    def _start_next_queued(self):
        """Pop the head of the queue into the GUI and start it. Never called while busy."""
        _proc = getattr(self, "current_process", None)
        if _proc is not None and _proc.poll() is None:
            return
        # Process-gone is NOT idle: between pipeline phases current_process is briefly None
        # while training_state is still "running", and paused/pausing runs own the GPU's
        # future. A stale timer or an eager click must not launch into any of those.
        if getattr(self, "training_state", "idle") in ("running", "pausing", "paused"):
            return
        if not self.training_queue:
            return
        # A training subprocess isn't the only thing that owns the GPU: a Royale export, a
        # caption batch, an Extract or a live preview are all in-process threads the process
        # check can't see. Launching a run on top of them OOMs it (and a failed run HOLDS
        # the queue — the worst outcome for an unattended batch). Wait and retry — capped,
        # so a stuck busy flag can't spin forever: after ~10 minutes the queue HOLDs loudly.
        try:
            if self._is_any_busy():
                self._queue_busy_retries = getattr(self, "_queue_busy_retries", 0) + 1
                if self._queue_busy_retries > 40:
                    self._queue_busy_retries = 0
                    self.update_console("[queue] HELD — the app has reported other GPU work "
                                        "for 10+ minutes. Finish or cancel it, then use "
                                        "'Start next now' in the queue window.\n")
                    self._render_queue_window()
                    return
                self.update_console("[queue] GPU work in progress elsewhere in the app — "
                                    "next run retries in 15 s.\n")
                self._schedule_queue_advance(15000)
                return
        except Exception:
            pass
        self._queue_busy_retries = 0
        head = self.training_queue[0]
        # Malformed item (hand-edited/corrupted queue file): remove it LOUDLY, then move on
        # to the next — one bad entry must not wedge the whole queue or crash the advance.
        if not self._queue_item_valid(head):
            self.training_queue.pop(0)
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()
            self.update_console("[queue] removed an unreadable queue entry (corrupt or "
                                "hand-edited queue file) — continuing with the next.\n")
            self._schedule_queue_advance(100)
            return
        # Dataset gone (deleted/renamed/moved since queueing): without this check the stale
        # TOML silently trains the PREVIOUS job's dataset under this job's name. HOLD with
        # the item still queued so nothing is lost.
        folder = (head.get("image_folder") or "").strip()
        if not os.path.isdir(folder):
            self.update_console(f"[queue] HELD — the next run's image folder no longer exists:\n"
                                f"        {folder}\n"
                                f"        Restore the folder (or edit/delete the queued job in "
                                f"the queue window), then 'Start next now'.\n")
            self._render_queue_window()
            return
        item = self.training_queue.pop(0)
        self._save_training_queue()
        self._refresh_queue_button()
        name = item.get("preset", {}).get("LORA_NAME") or os.path.basename(item.get("image_folder", "?"))
        self.update_console(f"\n[queue] starting next run: '{name}' "
                            f"({len(self.training_queue)} still queued)\n")
        self._apply_queue_item(item)
        self.start_training()
        # start_training can decline (validation, disk warning declined). The item's settings
        # are in the GUI either way; put it back at the head so nothing is silently lost.
        # _training_start_pending counts as LAUNCHED: with a warm caption worker the real
        # launch is marshalled through after(0) and training_state is still "idle" here —
        # re-inserting then would run the same item twice (review agent, 25 Aug).
        if (getattr(self, "training_state", "idle") != "running"
                and not getattr(self, "_training_start_pending", False)):
            self.training_queue.insert(0, item)
            self._save_training_queue()
            self._refresh_queue_button()
            # Invalidate any timer armed before this decline — otherwise a pending advance
            # re-pops the same head and repeats the same modal validation error in a loop.
            self._cancel_pending_queue_advance()
            self.update_console("[queue] run did not start — it stays at the head of the queue. "
                                "Fix the issue and use the queue window's 'Start next' button.\n")
        self._render_queue_window()

    def _refresh_queue_button(self):
        btn = getattr(self, "_queue_btn", None)
        if btn is None:
            return
        n = len(getattr(self, "training_queue", []))
        try:
            # Dark text on the light blue in BOTH states — the old accent-blue-when-queued
            # would now be mid-blue on baby blue (~2:1, unreadable). The count carries the
            # signal instead.
            btn.config(text=f"📋 Queue ({n})" if n else "📋 Queue",
                       bg=COLORS["queue_blue"], fg=COLORS["bg_deep"])
        except Exception:
            pass

    def _queue_thumbnail(self, folder, size=56):
        """PhotoImage of the first image in `folder`, or None. Caller keeps the reference."""
        try:
            from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
            exts = {e.lower() for e in IMAGE_EXTENSIONS}
            first = next((f for f in sorted(os.listdir(folder))
                          if os.path.splitext(f)[1].lower() in exts), None)
            if first is None:
                return None
            img = Image.open(os.path.join(folder, first))
            img.thumbnail((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _open_queue_window(self):
        """The queue manager: one row per queued run — thumbnail, key settings, and the
        operations (reorder / edit in tab / update from tab / delete / start next)."""
        win = getattr(self, "_queue_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            self._render_queue_window()
            return
        win = tk.Toplevel(self.master)
        win.title("Training Queue")
        win.geometry("860x560")
        win.configure(bg=COLORS["bg_deep"])
        self._queue_win = win

        tk.Label(win, text="Training Queue", font=(FONT_FAMILY, 16, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor=tk.W, padx=16, pady=(14, 0))
        self._queue_win_status = tk.Label(win, text="", font=(FONT_FAMILY, 9),
                                          bg=COLORS["bg_deep"], fg=COLORS["text_muted"],
                                          justify=tk.LEFT)
        self._queue_win_status.pack(anchor=tk.W, padx=16, pady=(2, 8))

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=16)
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vsb = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        cw = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(cw, width=e.width))
        # Wheel: global router — scrolls this canvas when the pointer is over the window,
        # the main app everywhere else, with no bind_all steal in either direction.
        self._queue_rows_frame = rows

        foot = tk.Frame(win, bg=COLORS["bg_deep"])
        foot.pack(fill=tk.X, padx=16, pady=12)
        self._queue_start_next_btn = ttk.Button(foot, text="▶ Start next now",
                                                command=self._start_next_queued, style="Primary.TButton")
        self._queue_start_next_btn.pack(side=tk.LEFT)
        ttk.Button(foot, text="Clear queue", command=self._queue_clear_all).pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(foot, text="Queued runs start automatically when the current run finishes cleanly. "
                            "After a failure, a Stop, or an app restart, the queue waits for you.",
                 font=(FONT_FAMILY, 8), bg=COLORS["bg_deep"], fg=COLORS["text_explain"],
                 wraplength=420, justify=tk.LEFT).pack(side=tk.RIGHT)
        self._render_queue_window()

    def _queue_clear_all(self):
        if self.training_queue and messagebox.askyesno(
                "Clear queue", f"Remove all {len(self.training_queue)} queued run(s)?"):
            self.training_queue.clear()
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()

    def _queue_row_summary(self, item):
        p = item.get("preset", {}) if isinstance(item.get("preset"), dict) else {}
        folder = str(item.get("image_folder") or "")
        try:
            from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
            _exts = {e.lower() for e in IMAGE_EXTENSIONS}
            # Clips and voice recordings are training items too — count them or a MiniMax
            # clip/audio folder reads "(0 images)" and looks like a queued mistake.
            if "MiniMax" in str(item.get("architecture", "")):
                _exts |= {".mp4"} | self.TRAINING_AUDIO_EXTENSIONS
            n_imgs = sum(1 for f in os.listdir(folder)
                         if os.path.splitext(f)[1].lower() in _exts) if os.path.isdir(folder) else 0
        except Exception:
            n_imgs = 0
        name = p.get("LORA_NAME") or os.path.basename(folder) or "(unnamed)"
        bits = [f"{item.get('architecture', '?')}",
                f"{os.path.basename(folder) or '?'} ({n_imgs} items)"]
        for label, key in (("LR", "LEARNING_RATE"), ("epochs", "MAX_TRAIN_EPOCHS"),
                           ("dim", "NETWORK_DIM"), ("type", "NETWORK_TYPE"),
                           ("area", "TARGET_LAYERS")):
            v = p.get(key)
            if v not in (None, ""):
                bits.append(f"{label} {v}")
        # Detail Focus only means anything for MiniMax, and it's the whole point of queueing a
        # shift sweep — without it two rows of an A/B look identical in the manager.
        if ARCHITECTURES.get(item.get("architecture", ""), {}).get("is_minimax"):
            _sh = str(p.get("MINIMAX_LOWNOISE_PCT") or "").strip()
            if _sh:
                bits.append(f"low-noise {_sh}%")
            _hl = str(p.get("MINIMAX_HIGHNOISE_LR_PCT") or "100").strip()
            if _hl and _hl != "100":
                bits.append(f"high-noise LR {_hl}%")
            if p.get("MINIMAX_LIKENESS_OPT"):
                bits.append("likeness-opt")
            else:
                _bl = minimax_block_spec(p.get("MINIMAX_BLOCKS"))
                if _bl.lower() != "all":
                    bits.append(f"blocks {_bl}")
            if p.get("MINIMAX_TRAIN_ADALN") is False:
                bits.append("no adaln")
            if p.get("MINIMAX_DISTILL"):
                bits.append(f"distill x{p.get('MINIMAX_DISTILL_WEIGHT', '0.8')}"
                            f" ({p.get('MINIMAX_DISTILL_REFS', '2')} refs)")
            _sl = str(p.get("MINIMAX_SLOW_BLOCKS") or "").strip()
            if _sl and str(p.get("MINIMAX_SLOW_LR_SCALE", "1")).strip() not in ("", "1", "1.0"):
                bits.append(f"slow {_sl} ×{p.get('MINIMAX_SLOW_LR_SCALE')}")
        return name, "  ·  ".join(str(b) for b in bits) + f"\nqueued {item.get('queued_at', '?')}"

    def _render_queue_window(self):
        rows = getattr(self, "_queue_rows_frame", None)
        if rows is None or not rows.winfo_exists():
            return
        for w in rows.winfo_children():
            w.destroy()
        self._queue_thumb_refs = []
        _busy = getattr(self, "current_process", None)
        _busy = _busy is not None and _busy.poll() is None
        _state = getattr(self, "training_state", "idle")
        _active = getattr(self, "_active_run_item", None)
        _show_active = _active is not None and (_busy or _state in ("running", "pausing", "paused"))
        try:
            n = len(self.training_queue)
            # Starting the next run while one is PAUSED would silently abandon the paused
            # run (its state dir resumes nothing once another run overwrites the GUI), so
            # paused disables the button just like busy does.
            _blocked = _busy or _state in ("pausing", "paused")
            self._queue_start_next_btn.config(
                state=(tk.DISABLED if (_blocked or not n) else tk.NORMAL))
            if _busy and _state == "pausing":
                txt = (f"{n} run(s) queued — the current run is pausing at the epoch end. "
                       f"A pause HOLDS the queue: Resume from the Training tab, or start the "
                       f"next run from here after it exits.") if n else \
                      "The current run is pausing at the epoch end."
            elif _busy:
                txt = (f"{n} run(s) queued — a run is active; the queue continues when it "
                       f"finishes cleanly." if n else
                       "A run is active and nothing is queued. The Start Training button reads "
                       "'Queue Train' — click it to add the currently configured run.")
            elif _state == "paused":
                txt = (f"{n} run(s) queued — a run is PAUSED. Resume it from the Training tab; "
                       f"'Start next now' is disabled because it would abandon the paused run."
                       if n else
                       "A run is paused — Resume it from the Training tab.")
            elif n:
                txt = (f"{n} run(s) queued — nothing is training. The queue HOLDS after a "
                       f"failure or Stop; use 'Start next now' to begin or continue.")
            else:
                txt = ("Queue is empty. While a run is active, the Start Training button "
                       "becomes 'Queue Train' — click it to add the currently configured run.")
            self._queue_win_status.config(text=txt)
        except Exception:
            pass

        # The run in progress, pinned on top — it isn't a queue item (never saved, can't be
        # reordered or deleted), but after editing a queued job in the Training tab, its ✎ is
        # the way BACK to the settings that are actually running.
        if _show_active:
            badge = ("⏸ paused" if _state == "paused" else
                     "⏸ pausing at epoch end" if _state == "pausing" else "▶ training now")
            card = tk.Frame(rows, bg=COLORS["bg_surface"],
                            highlightbackground=COLORS["accent"], highlightthickness=2)
            card.pack(fill=tk.X, pady=(0, 8))
            thumb = self._queue_thumbnail(_active.get("image_folder", ""))
            if thumb is not None:
                self._queue_thumb_refs.append(thumb)
                tk.Label(card, image=thumb, bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=10, pady=8)
            else:
                tk.Label(card, text="🖼", font=(FONT_FAMILY, 20), width=3,
                         bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=10, pady=8)
            act = tk.Frame(card, bg=COLORS["bg_surface"])
            act.pack(side=tk.RIGHT, padx=10, pady=8)
            name, summary = self._queue_row_summary(_active)
            txt = tk.Frame(card, bg=COLORS["bg_surface"])
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)
            tk.Label(txt, text=f"{badge}  —  {name}", font=(FONT_FAMILY, 11, "bold"),
                     bg=COLORS["bg_surface"], fg=COLORS["accent"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            tk.Label(txt, text=summary.split("\n")[0], font=(FONT_FAMILY, 8),
                     bg=COLORS["bg_surface"], fg=COLORS["text_muted"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            abtn = tk.Button(act, text="✎", font=(FONT_FAMILY, 10), width=3,
                             bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                             activebackground=COLORS["border"], relief="flat", bd=0,
                             cursor="hand2", command=self._queue_restore_active)
            abtn.pack(side=tk.LEFT, padx=2)
            ToolTip(abtn, "Load this run's settings back into the Training tab — the way back "
                          "after editing a queued job")
            if _busy:
                cbtn = tk.Button(act, text="■", font=(FONT_FAMILY, 10), width=3,
                                 bg=COLORS["bg_surface"], fg=COLORS["error"],
                                 activebackground=COLORS["border"], relief="flat", bd=0,
                                 cursor="hand2", command=self._queue_cancel_active)
                cbtn.pack(side=tk.LEFT, padx=2)
                ToolTip(cbtn, "Stop this run (no save). Queued runs HOLD — they won't "
                              "auto-start after a cancel")

        if not self.training_queue:
            return
        for i, item in enumerate(list(self.training_queue)):
            # One corrupt entry (hand-edited file, interrupted write) must not take the
            # whole window down — render it as removable wreckage instead.
            if not isinstance(item, dict) or not isinstance(item.get("preset"), dict):
                bad = tk.Frame(rows, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["error"], highlightthickness=1)
                bad.pack(fill=tk.X, pady=(0, 8))
                tk.Label(bad, text=f"{i + 1}.  ⚠ unreadable queue entry (corrupt or hand-edited "
                                   f"queue file)", font=(FONT_FAMILY, 10),
                         bg=COLORS["bg_surface"], fg=COLORS["error"]).pack(side=tk.LEFT, padx=10, pady=10)
                tk.Button(bad, text="✕", font=(FONT_FAMILY, 10), width=3,
                          bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                          activebackground=COLORS["border"], relief="flat", bd=0, cursor="hand2",
                          command=lambda i=i: self._queue_delete(i)).pack(side=tk.RIGHT, padx=10)
                continue
            card = tk.Frame(rows, bg=COLORS["bg_surface"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
            card.pack(fill=tk.X, pady=(0, 8))
            thumb = self._queue_thumbnail(item.get("image_folder", ""))
            if thumb is not None:
                self._queue_thumb_refs.append(thumb)
                tk.Label(card, image=thumb, bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=10, pady=8)
            else:
                tk.Label(card, text="🖼", font=(FONT_FAMILY, 20), width=3,
                         bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=10, pady=8)
            # Buttons pack FIRST (from the right): pack allocates space in order, so a long
            # unwrapped summary used to squeeze the ↑↓✎⤓✕ column clean off the card —
            # "my queued jobs have no delete button".
            btns = tk.Frame(card, bg=COLORS["bg_surface"])
            btns.pack(side=tk.RIGHT, padx=10, pady=8)
            name, summary = self._queue_row_summary(item)
            txt = tk.Frame(card, bg=COLORS["bg_surface"])
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)
            tk.Label(txt, text=f"{i + 1}.  {name}", font=(FONT_FAMILY, 11, "bold"),
                     bg=COLORS["bg_surface"], fg=COLORS["text_primary"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            tk.Label(txt, text=summary, font=(FONT_FAMILY, 8),
                     bg=COLORS["bg_surface"], fg=COLORS["text_muted"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)

            def _mk(parent, label, cmd, tip):
                b = tk.Button(parent, text=label, font=(FONT_FAMILY, 10), width=3,
                              bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                              activebackground=COLORS["border"], relief="flat", bd=0,
                              cursor="hand2", command=cmd)
                b.pack(side=tk.LEFT, padx=2)
                ToolTip(b, tip)
                return b
            _mk(btns, "↑", lambda i=i: self._queue_move(i, -1), "Move up")
            _mk(btns, "↓", lambda i=i: self._queue_move(i, +1), "Move down")
            _mk(btns, "✎", lambda i=i: self._queue_edit(i),
                "Load this run's settings into the Training tab to edit them")
            _mk(btns, "⤓", lambda i=i: self._queue_update_from_tab(i),
                "Overwrite this queued run with the Training tab's current settings")
            _mk(btns, "✕", lambda i=i: self._queue_delete(i), "Remove from queue")

    def _queue_move(self, i, delta):
        j = i + delta
        if 0 <= i < len(self.training_queue) and 0 <= j < len(self.training_queue):
            q = self.training_queue
            q[i], q[j] = q[j], q[i]
            self._save_training_queue()
            self._render_queue_window()

    def _queue_delete(self, i):
        if 0 <= i < len(self.training_queue):
            self.training_queue.pop(i)
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()

    def _queue_cancel_active(self):
        """Stop the run in progress from the queue window's pinned card. Confirmed first —
        the Training tab's own Stop button stays instant, but here a misclick between rows
        would kill hours of work. The existing hold policy applies: queued runs do NOT
        auto-start after a cancel."""
        _proc = getattr(self, "current_process", None)
        if _proc is None or _proc.poll() is not None:
            self._render_queue_window()
            return
        name = (getattr(self, "_active_run_item", None) or {}).get("preset", {}).get("LORA_NAME", "this run")
        if not messagebox.askyesno(
                "Stop training?",
                f"Stop '{name}' now? Progress since the last checkpoint is lost, and queued "
                f"runs will HOLD rather than auto-start.\n\n(To finish the epoch and save "
                f"first, use Pause Training on the Training tab instead.)"):
            return
        self.stop_training()
        self._render_queue_window()

    def _queue_restore_active(self):
        """Put the RUNNING job's settings back into the Training tab (the ✎ on the pinned
        'training now' card) — the undo for having edited a queued job in the tab."""
        item = getattr(self, "_active_run_item", None)
        if item is None:
            return
        self._apply_queue_item(item)
        self.update_console("[queue] Training tab restored to the run in progress.\n")

    def _queue_edit(self, i):
        """Load the item into the Training tab. The item stays queued — after editing,
        use ⤓ on the same row to write the changes back."""
        if not (0 <= i < len(self.training_queue)):
            return
        self._apply_queue_item(self.training_queue[i])
        self.update_console(f"[queue] loaded run {i + 1} into the Training tab — edit, then use "
                            f"the ⤓ button on its queue row to save the changes back.\n")

    def _queue_update_from_tab(self, i):
        if not (0 <= i < len(self.training_queue)):
            return
        old = self.training_queue[i]
        item = self._queue_snapshot()
        item["id"], item["queued_at"] = old.get("id", item["id"]), old.get("queued_at", item["queued_at"])
        self.training_queue[i] = item
        self._save_training_queue()
        self._render_queue_window()
        self.update_console(f"[queue] run {i + 1} updated from the Training tab's current settings.\n")

    # Keys in self.entries that belong to OTHER tabs — skipped when collecting
    # a training-tab preset. Everything else in self.entries is fair game.
    # RESUME_TRAINING is run-specific state, not a preset knob: capturing it baked an
    # absolute state-dir path into every saved preset (and Load Last Train), silently
    # turning future runs into resumes of an old checkpoint.
    _NON_TRAINING_ENTRY_KEYS = {
        "SAMPLE_ENABLED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
        "SAMPLE_SEED", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_EVERY_N_STEPS",
        "SAMPLE_AT_FIRST", "SAMPLE_FLOW_SHIFT",
        "SAMPLE_NEGATIVE", "SAMPLE_CFG_SCALE",
        "MINIMAX_TURBO_STEPS", "MINIMAX_TURBO_STRENGTH",
        "RESUME_TRAINING",
    }

    def _collect_preset_values(self):
        """Snapshot every user-editable value on the Training tab into a preset dict.

        Iterates all of self.entries (skipping keys that belong to other tabs or to
        system-level settings) plus every known Training-tab Boolean/StringVar — so
        saved presets capture anything the user touched in the Training UI, not just
        a hand-curated subset.
        """
        preset = {}

        # Everything in self.entries that's on the Training tab
        for key, entry in self.entries.items():
            if key in self._NON_TRAINING_ENTRY_KEYS:
                continue
            try:
                if isinstance(entry, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                    preset[key] = entry.get()
                else:
                    # ttk.Entry / ttk.Combobox / ttk.Spinbox all expose .get()
                    preset[key] = entry.get()
            except Exception:
                pass

        # Training-tab toggles that live on dedicated vars (not in self.entries)
        def _grab(attr, key):
            if hasattr(self, attr):
                try:
                    preset[key] = getattr(self, attr).get()
                except Exception:
                    pass

        # Multi Concept's folders are a LIST on a dedicated attribute, so neither the entries
        # loop nor _grab reaches them — and without this "Load Settings From Last Train" brings
        # the toggle back with no second subject behind it.
        if getattr(self, "_concept_folder_vars", None):
            preset["MINIMAX_CONCEPT_DIRS"] = [v.get().strip()
                                              for v in self._concept_folder_vars]

        _grab("preserve_dist_var", "PRESERVE_DISTRIBUTION")
        _grab("fp8_var", "FP8")
        _grab("scaled_var", "SCALED")
        _grab("quant_4bit_var", "QUANT_4BIT")
        # Store the canonical key, never the display label — the wording can change without
        # invalidating everyone's saved presets.
        if hasattr(self, "quant_4bit_mode_var"):
            preset["QUANT_4BIT_MODE"] = self._base_precision()
        _grab("compile_blocks_var", "COMPILE_BLOCKS")
        # BooleanVars aren't in self.entries, so they need grabbing explicitly — unlike
        # KEEP_LAST_N_STATES, which is an Entry and is captured by the generic sweep above.
        _grab("save_state_var", "SAVE_STATE")
        _grab("save_state_on_train_end_var", "SAVE_STATE_ON_TRAIN_END")
        _grab("krea2_loss_watch_var", "KREA2_LOSS_WATCH")
        _grab("krea2_per_image_lr_var", "KREA2_PER_IMAGE_LR")
        _grab("krea2_auto_recaption_var", "KREA2_AUTO_RECAPTION")
        _grab("krea2_warmup_look_var", "KREA2_WARMUP_LOOK")
        _grab("krea2_finetune_var", "KREA2_FINETUNE")
        _grab("krea2_fast_ft_var", "KREA2_FAST_FT")
        _grab("krea2_reg_dir_var", "KREA2_REG_DIR")
        _grab("krea2_reg_mult_var", "KREA2_REG_MULT")
        _grab("krea2_ft_mode_var", "KREA2_FT_MODE")
        _grab("krea2_ft_blocks_var", "KREA2_FT_BLOCKS")
        _grab("krea2_ft_every_var", "KREA2_FT_EVERY")
        _grab("krea2_ft_fused_var", "KREA2_FT_FUSED")
        _grab("minimax_finetune_var", "MINIMAX_FINETUNE")
        _grab("minimax_ft_every_var", "MINIMAX_FT_EVERY")
        _grab("minimax_ft_scope_var", "MINIMAX_FT_SCOPE")
        _grab("minimax_ft_blockspec_var", "MINIMAX_FT_BLOCKSPEC")
        _grab("minimax_ft_fused_var", "MINIMAX_FT_FUSED")
        _grab("minimax_reg_dir_var", "MINIMAX_REG_DIR")
        _grab("minimax_reg_mult_var", "MINIMAX_REG_MULT")
        # MiniMax reference distillation. A plain StringVar, so the generic self.entries sweep
        # above does NOT see it — without this a queued distillation run loses its reference
        # and silently becomes an ordinary run (tests/test_minimax_distill_gui.py).
        _grab("minimax_distill_var", "MINIMAX_DISTILL")
        _grab("minimax_multiconcept_var", "MINIMAX_MULTICONCEPT")
        _grab("grad_checkpoint_var", "GRADIENT_CHECKPOINTING")
        _grab("fp8_text_encoder_var", "FP8_TEXT_ENCODER")
        _grab("adaptive_lr_var", "ADAPTIVE_LR")
        _grab("training_preset_var", "TARGET_LAYERS")
        _grab("ts_sampling_var", "TIMESTEP_SAMPLING")
        _grab("weighting_scheme_var", "WEIGHTING_SCHEME")
        _grab("enable_cache_var", "ENABLE_CACHE")
        # Dataset subsection (now living in Training → Other Options)
        _grab("dataset_enable_bucket_var", "ENABLE_BUCKET")
        _grab("dataset_no_upscale_var", "BUCKET_NO_UPSCALE")
        _grab("dataset_caption_ext_var", "DATASET_CAPTION_EXT")
        _grab("dataset_megapixels_var", "DATASET_MEGAPIXELS")
        _grab("dataset_batch_size_var", "DATASET_BATCH_SIZE")
        # Gradient mining
        _grab("gradient_mining_var", "GRADIENT_MINING")
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if hasattr(self, "training_block_vars") and self.training_block_vars:
            preset["TRAINING_BLOCKS"] = {k: v.get() for k, v in self.training_block_vars.items()}

        return preset

    def save_custom_preset(self):
        """Save current settings as a custom preset for the current architecture"""
        arch = self.architecture_var.get()

        # Prompt for preset name
        preset_name = simpledialog.askstring(
            "Save Preset",
            f"Enter a name for your preset (for {arch}):",
            parent=self.master
        )

        if not preset_name:
            return  # User cancelled

        # Validate name (no special chars that could cause filesystem issues)
        invalid_chars = '<>:"/\\|?*'
        if any(c in preset_name for c in invalid_chars):
            messagebox.showerror("Invalid Name", f"Preset name cannot contain: {invalid_chars}")
            return

        preset_name = preset_name.strip()
        if not preset_name:
            messagebox.showerror("Invalid Name", "Preset name cannot be empty")
            return

        # Check if preset already exists
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if os.path.exists(preset_path):
            overwrite = messagebox.askyesno(
                "Preset Exists",
                f"A preset named '{preset_name}' already exists.\nDo you want to overwrite it?"
            )
            if not overwrite:
                return

        # Collect and save preset
        preset = self._collect_preset_values()
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=4)
            self.refresh_preset_combobox()
            messagebox.showinfo("Preset Saved", f"Preset '{preset_name}' saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")

    def load_custom_preset(self, event=None):
        """Load a preset from the combobox selection — built-in or user-saved."""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            return

        # Check built-in presets first (architecture-specific)
        builtins = self._builtins_for_arch(self.architecture_var.get())
        if preset_name in builtins:
            self._apply_preset_values(builtins[preset_name])
            messagebox.showinfo("Preset Loaded", f"Loaded built-in preset '{preset_name}'")
            return

        # Otherwise, look for a user-saved preset on disk
        arch = self.architecture_var.get()
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if not os.path.exists(preset_path):
            messagebox.showerror("Error", f"Preset file not found: {preset_name}")
            self.refresh_preset_combobox()
            return

        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
            self._apply_preset_values(preset)
            messagebox.showinfo("Preset Loaded", f"Loaded preset '{preset_name}'")
        except json.JSONDecodeError:
            messagebox.showerror("Error", f"Preset file is corrupted: {preset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {str(e)}")

    def delete_custom_preset(self):
        """Delete the currently selected custom preset"""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            messagebox.showinfo("Info", "Please select a preset to delete")
            return

        arch = self.architecture_var.get()

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the preset '{preset_name}'?\n\nThis action cannot be undone."
        )
        if not confirm:
            return

        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        try:
            if os.path.exists(preset_path):
                os.remove(preset_path)
                self.refresh_preset_combobox()
                messagebox.showinfo("Preset Deleted", f"Preset '{preset_name}' deleted successfully")
            else:
                messagebox.showerror("Error", f"Preset file not found: {preset_name}")
                self.refresh_preset_combobox()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete preset: {str(e)}")

    def update_ui_for_architecture(self):
        """Update UI elements based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Window title is set once in __init__ and stays put — architecture is
        # hardcoded to Klein Base 9B, no need to mirror it in the title bar.

        # Show/hide CLIP model field
        if config["uses_clip"]:
            self.show_row("CLIP_MODEL")
        else:
            self.hide_row("CLIP_MODEL")

        # Show/hide T5 model field
        if config["uses_t5"]:
            self.show_row("T5_MODEL")
        else:
            self.hide_row("T5_MODEL")

        # Show/hide Text Encoder field (for Z-Image and Flux 2)
        if config["uses_text_encoder"]:
            self.show_row("TEXT_ENCODER")
            if "TEXT_ENCODER" in self.labels:  # Model Paths section may have been removed
                self.labels["TEXT_ENCODER"].config(text=f"{config['text_encoder_label']}:")
        else:
            self.hide_row("TEXT_ENCODER")

        # Show/hide Model Type dropdown (Wan only) and update options
        if config["uses_model_type"]:
            self.show_row("MODEL_TYPE")
            # Update MODEL_TYPE dropdown values for this architecture
            model_types = config.get("model_types", ["t2v-14B", "i2v-14B"])
            self.entries["MODEL_TYPE"]["values"] = model_types
            current_val = self.entries["MODEL_TYPE"].get()
            if current_val not in model_types:
                self.entries["MODEL_TYPE"].current(0)
        else:
            self.hide_row("MODEL_TYPE")

        # Update VAE label (Model Paths section may have been removed)
        if "VAE_MODEL" in self.labels:
            self.labels["VAE_MODEL"].config(text=f"{config['vae_label']}:")

        # Update FP8 text encoder checkbox label
        if arch.startswith("Wan"):
            self.fp8_text_encoder_check.config(text="Enable FP8 T5")
        elif arch.startswith("Z-Image"):
            self.fp8_text_encoder_check.config(text="Enable FP8 LLM")
        else:
            self.fp8_text_encoder_check.config(text="Enable FP8 Text Encoder")

        # Update blocks swap max (enforce limit). "Auto" is left ALONE: resolving it here
        # ran the auto strategy (a GPU probe that can flip the 4-bit toggle) as a side
        # effect of a bounds check, and writing the resolved number into the combobox
        # silently turned an auto choice into a permanent manual one. Auto strategies
        # already return in-range values; only explicit numbers need clamping.
        try:
            _raw_swap = self.entries["BLOCKS_SWAP"].get().strip()
            if not _raw_swap.lower().startswith("auto"):
                current_blocks = self._parse_blocks_swap()
                if current_blocks > config["blocks_swap_max"]:
                    self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                    self.entries["BLOCKS_SWAP"].insert(0, str(config["blocks_swap_max"]))
        except ValueError:
            pass

        # Update timestep section for architecture.
        # Values are only touched when the ARCHITECTURE actually changed: several callers
        # use this method as a pure visibility refresh, and unconditionally resetting the
        # section wiped the user's timestep settings on every call. On a real switch, the
        # outgoing family's values are stashed and restored when the user switches back.
        if hasattr(self, 'ts_sampling_var'):
            _prev_arch = getattr(self, "_ts_defaults_arch", None)
            if _prev_arch != arch:
                if not hasattr(self, "_arch_ts_stash"):
                    self._arch_ts_stash = {}
                if _prev_arch is not None:
                    self._arch_ts_stash[_prev_arch] = {
                        "sampling": self.ts_sampling_var.get(),
                        "shift": self.entries["DISCRETE_FLOW_SHIFT"].get(),
                        "min_ts": self.entries["MIN_TIMESTEP"].get(),
                        "max_ts": self.entries["MAX_TIMESTEP"].get(),
                        "preserve": self.preserve_dist_var.get(),
                        "weighting": self.weighting_scheme_var.get(),
                    }
                stash = self._arch_ts_stash.get(arch)
                if stash is not None:
                    # Returning to a family the user already configured — restore, don't reset.
                    self.ts_sampling_var.set(stash["sampling"])
                    self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                    self.entries["DISCRETE_FLOW_SHIFT"].delete(0, tk.END)
                    self.entries["DISCRETE_FLOW_SHIFT"].insert(0, stash["shift"])
                    self.entries["MIN_TIMESTEP"].delete(0, tk.END)
                    self.entries["MIN_TIMESTEP"].insert(0, stash["min_ts"])
                    self.entries["MAX_TIMESTEP"].delete(0, tk.END)
                    self.entries["MAX_TIMESTEP"].insert(0, stash["max_ts"])
                    self.preserve_dist_var.set(stash["preserve"])
                    self.weighting_scheme_var.set(stash["weighting"])
                else:
                    # First visit to this family — apply its defaults.
                    self.ts_sampling_var.set(config.get("timestep_sampling", "shift"))
                    default_shift = config.get("discrete_flow_shift")
                    if default_shift is not None:
                        self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                        self.entries["DISCRETE_FLOW_SHIFT"].delete(0, tk.END)
                        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, str(default_shift))
                    min_ts = config.get("min_timestep")
                    max_ts = config.get("max_timestep")
                    self.entries["MIN_TIMESTEP"].delete(0, tk.END)
                    self.entries["MAX_TIMESTEP"].delete(0, tk.END)
                    if min_ts is not None:
                        self.entries["MIN_TIMESTEP"].insert(0, str(min_ts))
                    if max_ts is not None:
                        self.entries["MAX_TIMESTEP"].insert(0, str(max_ts))
                    self.preserve_dist_var.set(config.get("preserve_distribution_shape", False))
                self._ts_defaults_arch = arch

            # Enable/disable states are pure display — refresh them on every call.
            supports_shift = config.get("supports_discrete_flow_shift", True)
            if supports_shift:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                self.ts_flow_shift_label.config(fg=COLORS["text_secondary"])
            else:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="disabled")
                self.ts_flow_shift_label.config(fg=COLORS["text_muted"])

            supports_weighting = config.get("supports_weighting_scheme", True)
            if supports_weighting:
                self.ts_weighting_combo.config(state="readonly")
                self.ts_weighting_label.config(fg=COLORS["text_secondary"])
            else:
                self.weighting_scheme_var.set("none")
                self.ts_weighting_combo.config(state="disabled")
                self.ts_weighting_label.config(fg=COLORS["text_muted"])

            # Refresh conditional field states
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()

        # Krea 2 hides Training-tab features that aren't wired into its native trainer yet.
        # These are DEFERRED, not removed — each is re-enabled simply by dropping it from the
        # hide lists in _apply_training_arch_visibility once krea2_train supports it.
        self._apply_training_arch_visibility(config.get("is_krea2", False))

    @staticmethod
    def _select_combo_by_token(combo, value):
        """Select the option whose FIRST whitespace-separated token equals `value`.

        Labelled dropdowns here carry a trailing note ("12 - reference default, 5% detail band")
        while settings/presets store only the bare token. Matching on the token means the note
        can be reworded without invalidating anyone's saved runs or queued items."""
        want = str(value).split(" ")[0]
        try:
            for opt in (combo.cget("values") or ()):
                if str(opt).split(" ")[0] == want:
                    combo.set(str(opt))
                    return True
        except tk.TclError:
            pass
        return False

    def _refresh_minimax_shift_match(self):
        """Show the schedule the typed percentage produces, or why it can't be read.

        The relationship is very non-linear at the ends — 5% is shift 19, 2% is shift 49 — so the
        resulting shift and median noise level are worth seeing next to the number you typed."""
        lbl = getattr(self, "_minimax_shift_match", None)
        ent = self.entries.get("MINIMAX_LOWNOISE_PCT")
        if lbl is None or ent is None or not lbl.winfo_exists():
            return
        shift = minimax_lownoise_to_shift(ent.get())
        if shift is None:
            lbl.config(text="✗ enter a number above 0 and below 100", fg="#E74C3C")
            return
        # The median is the shift map at the uniform base's median draw, so shift/(shift+1).
        med = shift / (shift + 1.0)
        lbl.config(text=f"→ shift {shift:.3g}, median noise {med:.2f}", fg="#27AE60")

    def _build_minimax_structure_row(self, parent):
        """Training Structure — the MiniMax timestep density, named.

        Rows 22-26 of Training Parameters, under Network Type. The structure dropdown is a VIEW of
        MINIMAX_LOWNOISE_PCT rather than a setting of its own, so every existing preset and saved
        run keeps working with no migration: 60 shows Face likeness, anything unrecognised shows
        Custom and reveals the box it came from.
        """
        self._minimax_structure_label = ttk.Label(parent, text="Training Structure:")
        self._minimax_structure_label.grid(row=22, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self.minimax_structure_var = tk.StringVar(value=MINIMAX_STRUCTURE_DEFAULT)
        self._minimax_structure_combo = ttk.Combobox(
            parent, textvariable=self.minimax_structure_var,
            values=list(MINIMAX_STRUCTURE_OPTIONS), state="readonly", width=36)
        self._minimax_structure_combo.grid(row=22, column=1, columnspan=2, sticky=tk.W,
                                           padx=5, pady=(8, 2))
        self._minimax_structure_combo.bind("<<ComboboxSelected>>",
                                           lambda _e: self._on_minimax_structure_changed())

        self._minimax_structure_desc = tk.Label(
            parent, text="", font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
            bg=COLORS["bg_surface"], justify=tk.LEFT, wraplength=700)
        self._minimax_structure_desc.grid(row=23, column=0, columnspan=3, sticky=tk.W,
                                          padx=(12, 5), pady=(0, 4))

        # Shown when the dataset carries voice recordings and the structure ISN'T Likeness —
        # A/B tested (Aug 2026): voices train much faster and sound better at Likeness and
        # Style than at Model default, for the same reason faces do: identity lives at the
        # clean end, and the audio schedule is chained to the video one. Managed by
        # _refresh_audio_only_ui.
        self._minimax_structure_voice_note = tk.Label(
            parent, text="🎙 Voice recordings in this dataset — Likeness and Style trains "
                         "voices much faster than Model default (tested). Consider switching.",
            font=(FONT_FAMILY, 9), fg="#F59E0B", bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=700)

        # Per-category retirement — MIXED datasets only (managed by _refresh_audio_only_ui).
        # Visuals and voice need not converge together (a much smaller category can finish,
        # or start to overbake, well before the larger one),
        # so each can retire at its own epoch. "Anchor" keeps the finished category training at
        # a REAL 10% LR (it multiplies the optimizer's lr — a loss multiplier would be an Adam
        # no-op) as a drift guard, with its epoch ledger staying live as the drift alarm;
        # "stop" skips its steps outright for faster epochs.
        self._mixed_stop_label = ttk.Label(parent, text="Finish one category early:")
        self._mixed_stop_frame = ttk.Frame(parent)
        _msf = self._mixed_stop_frame
        _RETIRE_MODES = ["anchor at 10% LR (recommended)", "stop completely (faster)"]
        self.entries["MIXED_STOP_CATEGORY"] = ttk.Combobox(
            _msf, values=["voice", "photos & clips"], width=14, state="readonly")
        self.entries["MIXED_STOP_CATEGORY"].set(
            str(self.settings.get("MIXED_STOP_CATEGORY", "")) or "voice")
        self.entries["MIXED_STOP_CATEGORY"].pack(side=tk.LEFT)
        ttk.Label(_msf, text=" after epoch ").pack(side=tk.LEFT)
        self.entries["MIXED_STOP_EPOCH"] = ttk.Entry(_msf, width=5)
        self.entries["MIXED_STOP_EPOCH"].insert(
            0, str(self.settings.get("MIXED_STOP_EPOCH", "")))
        self.entries["MIXED_STOP_EPOCH"].pack(side=tk.LEFT, padx=(0, 8))
        # Under FT the hint shows live where a typed epoch will land (cycle-boundary snap).
        self.entries["MIXED_STOP_EPOCH"].bind(
            "<KeyRelease>", lambda _e: self._refresh_mixed_stop_hint())
        self.entries["MIXED_STOP_MODE"] = ttk.Combobox(_msf, values=_RETIRE_MODES, width=26,
                                                       state="readonly")
        self.entries["MIXED_STOP_MODE"].set(
            str(self.settings.get("MIXED_STOP_MODE", "")) or _RETIRE_MODES[0])
        self.entries["MIXED_STOP_MODE"].pack(side=tk.LEFT)
        self._mixed_stop_hint = tk.Label(
            parent, text="If one category is a substantially different size from the other, "
                         "it may be done (or start to overbake) well before the rest — finish "
                         "it early instead of overtraining it. Blank = both train to the end. "
                         "Anchor keeps the finished category at a true 10% learning rate — "
                         "holding its quality against drift from the still-training category, "
                         "with its epoch report staying live as the drift alarm. Stop skips "
                         "its steps entirely: faster epochs, but that category goes unwatched.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=720)
        self._MIXED_STOP_HINT_LORA = self._mixed_stop_hint.cget("text")
        # FT text is rebuilt live by _refresh_mixed_stop_hint (the cycle length rides on
        # Rotate every); this placeholder is only ever shown for a frame at build time.
        self._MIXED_STOP_HINT_FT = ""

        # The raw share, revealed only under Custom — the named options are the point.
        self._minimax_shift_label = ttk.Label(parent, text="Clean-end share:")
        self._minimax_shift_label.grid(row=24, column=0, sticky=tk.W, padx=5, pady=2)
        self._minimax_shift_frame = ttk.Frame(parent)
        self._minimax_shift_frame.grid(row=24, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.entries["MINIMAX_LOWNOISE_PCT"] = ttk.Entry(self._minimax_shift_frame, width=8)
        self.entries["MINIMAX_LOWNOISE_PCT"].insert(
            0, str(self.settings.get("MINIMAX_LOWNOISE_PCT", "60")))
        self.entries["MINIMAX_LOWNOISE_PCT"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_shift_frame, text="% of steps").pack(side=tk.LEFT, padx=(4, 0))
        # Live readout: the typed number is what you care about, but the schedule it produces is
        # worth seeing — a couple of percent swings the shift enormously at the ends.
        self._minimax_shift_match = tk.Label(self._minimax_shift_frame, text="",
                                             font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        self._minimax_shift_match.pack(side=tk.LEFT, padx=(10, 0))
        self.entries["MINIMAX_LOWNOISE_PCT"].bind(
            "<KeyRelease>", lambda _e: self._refresh_minimax_shift_match())

        # Always visible: a preset recommends a value, the user can override it without that
        # counting as a different structure.
        self._minimax_hnlr_label = ttk.Label(parent, text="Medium to High LR:")
        self._minimax_hnlr_label.grid(row=25, column=0, sticky=tk.W, padx=5, pady=(2, 8))
        self._minimax_hnlr_frame = ttk.Frame(parent)
        self._minimax_hnlr_frame.grid(row=25, column=1, columnspan=2, sticky=tk.W,
                                      padx=5, pady=(2, 8))
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"] = ttk.Entry(self._minimax_hnlr_frame, width=8)
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"].insert(
            0, str(self.settings.get("MINIMAX_HIGHNOISE_LR_PCT", "100")))
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"].pack(side=tk.LEFT)
        tk.Label(self._minimax_hnlr_frame,
                 text="%  — best left at 100 unless you are experimenting.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(4, 0))
        # Says what it does and what was measured, so lowering it is a decision rather than a
        # guess: across five datasets, at both densities, 0% and 100% render cleanly at 20 steps
        # without the Turbo LoRA and 100% holds face SHAPE better every time.
        self._minimax_hnlr_hint = tk.Label(
            parent,
            text="What the noisier steps — where pose, framing and face shape are decided — do to "
                 "the learning rate. Lowering it biases the run toward surface detail at the cost "
                 "of shape — useful for a skin-texture LoRA, not for a likeness one. Across five "
                 "datasets 100 held face shape better, and nothing distorted at any setting.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=700)
        self._minimax_hnlr_hint.grid(row=26, column=0, columnspan=3, sticky=tk.W,
                                     padx=(12, 5), pady=(0, 8))

        self._sync_minimax_structure_from_pct()
        self._refresh_minimax_shift_match()

    def _on_minimax_structure_changed(self):
        """A named option writes the numbers behind it; Custom just reveals them."""
        vals = MINIMAX_STRUCTURE_OPTIONS.get(self.minimax_structure_var.get())
        if vals is not None:
            pct, hnlr = vals
            for key, value in (("MINIMAX_LOWNOISE_PCT", pct), ("MINIMAX_HIGHNOISE_LR_PCT", hnlr)):
                ent = self.entries.get(key)
                if ent is not None:
                    ent.delete(0, tk.END)
                    ent.insert(0, str(value))
        self._refresh_minimax_structure_ui()
        self._refresh_minimax_shift_match()
        self._refresh_audio_only_ui()      # the voice hint clears the moment Likeness is picked

    def _sync_minimax_structure_from_pct(self):
        """Pick the dropdown entry the current percentage corresponds to, else Custom.

        Derived rather than stored, which is what lets every preset written before this control
        existed keep working untouched.
        """
        try:
            pct = float(str(self.entries["MINIMAX_LOWNOISE_PCT"].get()).strip().rstrip("%"))
        except (KeyError, TypeError, ValueError, tk.TclError):
            pct = None
        name = "Custom"
        if pct is not None:
            for label, vals in MINIMAX_STRUCTURE_OPTIONS.items():
                if vals is not None and abs(vals[0] - pct) < 1e-9:
                    name = label
                    break
        self.minimax_structure_var.set(name)
        self._refresh_minimax_structure_ui()

    def _refresh_minimax_structure_ui(self):
        """Description text, and the raw share shown only under Custom."""
        name = self.minimax_structure_var.get()
        desc = getattr(self, "_minimax_structure_desc", None)
        if desc is not None and desc.winfo_exists():
            desc.config(text=MINIMAX_STRUCTURE_DESC.get(name, ""))
        custom = MINIMAX_STRUCTURE_OPTIONS.get(name) is None
        for w in (getattr(self, "_minimax_shift_label", None),
                  getattr(self, "_minimax_shift_frame", None)):
            if w is None or not w.winfo_exists():
                continue
            if custom and self._is_minimax_arch():
                w.grid()
            else:
                w.grid_remove()

    def _refresh_minimax_blocks_count(self):
        """Say how many blocks the Blocks to Train box currently means, or why it can't be read.

        A typed spec fails silently in the worst way: "3-12, 4" trains 11 blocks and looks like a
        run, and nothing downstream ever says otherwise. This turns that into a number you can
        see before you launch."""
        lbl = getattr(self, "_minimax_blocks_count", None)
        combo = self.entries.get("MINIMAX_BLOCKS")
        if lbl is None or combo is None or not lbl.winfo_exists():
            return
        spec = minimax_block_spec(combo.get())
        if spec.lower() == "all":
            lbl.config(text="all 50 blocks", fg=COLORS["text_explain"])
            return
        try:
            from fizgig.minimax.trainer import parse_block_spec
            idx = parse_block_spec(spec, MINIMAX_NUM_BLOCKS)
        except ValueError as e:
            lbl.config(text=f"✗ {e}", fg="#E74C3C")
            return
        except ImportError:
            lbl.config(text="")
            return
        lbl.config(text=f"✓ {len(idx)} of {MINIMAX_NUM_BLOCKS} blocks", fg="#27AE60")

    # The Blocks to Train hint in both of its states — module-level truth so the greying
    # handler can swap them without duplicating the strings inline.
    _MINIMAX_BLOCKS_HINT = ("Train only a subset of the 50 blocks. Type ranges and single "
                            "blocks, comma-separated, like 3-12, 22, 31-33 (blocks 0-49). "
                            "Measured answers: 20-49 for likeness — sharper, more "
                            "prompt-responsive, better sound, faster and smoother to converge "
                            "(Optimised Likeness Learning applies it to photos automatically) — "
                            "and 0-3, 6-47 for style (the Style preset sets it). Full write-up "
                            "in the README.")
    _MINIMAX_BLOCKS_HINT_LOCKED = ("Disabled by Optimised Likeness Learning above — untick it "
                                   "to hand-pick blocks. While it's on, photos train "
                                   f"{MINIMAX_LIKENESS_BLOCKS}; video follows the restriction tickbox.")

    def _sync_minimax_likeness_state(self):
        """Grey Blocks to Train while Optimised Likeness Learning owns the block choice.

        The combobox VALUE is deliberately preserved — a hand-typed spec survives a toggle
        round-trip; only the widget state and the hint change. Driven by the checkbox trace
        (fires on preset loads too) and by arch switches."""
        combo = self.entries.get("MINIMAX_BLOCKS")
        hint = getattr(self, "_minimax_blocks_hint", None)
        if combo is None or hint is None or not combo.winfo_exists():
            return
        locked = self._is_minimax_arch() and bool(
            self.entries["MINIMAX_LIKENESS_OPT"].get())
        # The video-restriction sub-tick shows only where it means something: MiniMax
        # family, Fine-tune ON, likeness ON. (LoRA-mode clips keep whole-model behaviour;
        # the builder only emits --clip_blocks under FT regardless, so this is
        # presentation — the flag gate is the guard.)
        _clip_cb = getattr(self, "_minimax_ft_clip_cb", None)
        if _clip_cb is not None and _clip_cb.winfo_exists():
            _show = locked and bool(getattr(self, "minimax_finetune_var", None)
                                    and self.minimax_finetune_var.get())
            self._set_widget_visible(_clip_cb, _show)
        if locked:
            combo.config(state="disabled")
            hint.config(text=self._MINIMAX_BLOCKS_HINT_LOCKED)
            lbl = getattr(self, "_minimax_blocks_count", None)
            if lbl is not None and lbl.winfo_exists():
                lbl.config(text=f"photos: {MINIMAX_LIKENESS_BLOCKS} · clips: see video restriction",
                           fg=COLORS["text_explain"])
        else:
            combo.config(state="")               # editable, the widget's natural state
            hint.config(text=self._MINIMAX_BLOCKS_HINT)
            self._refresh_minimax_blocks_count()

    def _is_krea2_arch(self) -> bool:
        return ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_krea2", False)

    def _is_minimax_arch(self) -> bool:
        return ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_minimax", False)

    def _set_widget_visible(self, w, show: bool):
        """Show/hide a single widget, working for both grid- and pack-managed widgets.
        grid widgets use grid_remove()/grid() (position preserved); pack widgets stash their
        pack_info on hide and restore it on show (with the 'in'→'in_' kwarg fix)."""
        if w is None:
            return
        try:
            if show:
                saved = getattr(w, "_fizgig_pack_info", None)
                if saved is not None:
                    w.pack(**saved)
                    w._fizgig_pack_info = None
                elif w.winfo_manager() == "":   # grid_remove'd → restore remembered slot
                    w.grid()
            else:
                mgr = w.winfo_manager()
                if mgr == "pack":
                    info = {k: v for k, v in w.pack_info().items()}
                    if "in" in info:
                        info["in_"] = info.pop("in")
                    w._fizgig_pack_info = info
                    w.pack_forget()
                elif mgr == "grid":
                    w.grid_remove()
        except Exception:
            pass

    def _set_training_section_visible(self, key: str, before_key: str, visible: bool):
        """Show/hide a whole collapsible section, preserving the canonical pack order.
        When showing, pack it before `before_key` (which must be a currently-packed section)
        so it lands back in the right place rather than at the bottom of the tab."""
        sec = self.collapsible_sections.get(key)
        if sec is None:
            return
        try:
            if visible:
                before = self.collapsible_sections.get(before_key)
                if before is not None and before.winfo_manager() == "pack":
                    sec.pack(fill=tk.X, padx=36, pady=(0, 16), before=before)
                else:
                    sec.pack(fill=tk.X, padx=36, pady=(0, 16))
            else:
                sec.pack_forget()
        except Exception:
            pass

    # Recommended base-model fine-tune setup. Applied when the checkbox is ticked so the
    # whole recipe comes as one decision instead of six. Values come from the measured runs
    # on this branch; the LR especially — LoRA rates (1e-4+) destroy a base model.
    KREA2_FT_DEFAULTS = {
        "LEARNING_RATE": "1e-5",
        "MAX_TRAIN_EPOCHS": "40",         # 10 full 4-window cycles — an overnight run you can
                                          # scrub through; nobody has tuned this recipe on a
                                          # diffusion DiT, so compare checkpoints to find where
                                          # it peaks rather than trusting the number
        "SAVE_EVERY_N_EPOCHS": "4",       # one per full cycle: every component has had the same
                                          # number of passes, so checkpoints are comparable
                                          # like-for-like. ~26 GB each -> 10 files / ~260 GB
                                          # over a 40-epoch run
        "GRADIENT_ACCUMULATION": "1",     # fused backward consumes grads as they land
        "MAX_GRAD_NORM": "0",             # global clipping is impossible under fused backward
        "NETWORK_TYPE": "LoRA (standard)",  # FT trains the BASE — a LoKR adapter would sit
                                            # inert burning VRAM, so the recipe resets it
    }

    def _on_krea2_ft_toggle(self):
        """User ticked/unticked base-model fine-tuning. Only push the recipe on the way ON,
        so re-showing the tab never stomps values the user has since tuned."""
        self._apply_krea2_ft_visibility()
        # The regularisation block is fine-tune-only, so the TOML changes with this toggle.
        self.auto_save_dataset_config_silent()
        if bool(self.krea2_finetune_var.get()):
            self._apply_krea2_ft_defaults()

    def _apply_krea2_ft_defaults(self):
        """Set the whole fine-tune recipe in one go, and say what changed."""
        changed = []
        # Adaptive LR goes off FIRST: the trainer disables it anyway (rotation boundaries read
        # as instability), and while it's on the Learning Rate box is greyed out — writing the
        # recipe's 1e-5 into a disabled Entry silently does nothing, which left fine-tune runs
        # starting at a LoRA-grade LR.
        if getattr(self, "adaptive_lr_var", None) is not None and self.adaptive_lr_var.get():
            self.adaptive_lr_var.set(False)
            try:
                self._on_adaptive_lr_toggle()
            except Exception:
                pass
            changed.append("Adaptive LR: on -> off (incompatible with rotation)")
        for key, val in self.KREA2_FT_DEFAULTS.items():
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                before = entry.get()
                if str(before).strip() == val:
                    continue
                # Belt and braces: a disabled Entry rejects delete/insert, so re-enable it for
                # the write and put its state back.
                _was = str(entry.cget("state"))
                if _was == "disabled":
                    entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, val)
                if _was == "disabled":
                    entry.config(state=_was)
                changed.append(f"{key.replace('_', ' ').title()}: {before} -> {val}")
            except Exception:
                pass
        if getattr(self, "lr_scheduler_var", None) is not None and self.lr_scheduler_var.get() != "constant":
            was = self.lr_scheduler_var.get()
            self.lr_scheduler_var.set("constant")
            changed.append(f"LR Scheduler: {was} -> constant")
        if changed:
            self.update_console("[fine-tune] applied the recommended base-model setup:\n  "
                                + "\n  ".join(changed) + "\n")

    def _browse_krea2_reg_dir(self):
        """Pick the optional regularisation image folder (Krea 2 fine-tune)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Regularisation images (optional)",
                                    initialdir=self.krea2_reg_dir_var.get() or None)
        if d:
            self.krea2_reg_dir_var.set(d)
            self.auto_save_dataset_config_silent()   # the TOML carries the reg block

    def _browse_minimax_reg_dir(self):
        """Pick the optional regularisation image folder (MiniMax H3 fine-tune)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Regularisation images (optional)",
                                    initialdir=self.minimax_reg_dir_var.get() or None)
        if d:
            self.minimax_reg_dir_var.set(d)
            self.auto_save_dataset_config_silent()   # the TOML carries the reg block

    def _apply_krea2_ft_visibility(self):
        """Show the fine-tune knobs only when base-model fine-tuning is on, and the
        blocks-per-window picker only in block mode (component windows are fixed)."""
        if not hasattr(self, "_krea2_ft_frame"):
            return
        on = bool(self.krea2_finetune_var.get())
        for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_fast_ft_cb,
                  self._krea2_reg_frame, self._krea2_ft_hint):
            self._set_widget_visible(w, on)
        # Auto-recaption is hidden under a fine-tune: its between-epoch VLM load moves the
        # whole DiT off the card through a blocks_to_swap-aware restore that knows nothing
        # about the FT rotation streamer — on the 16 GB streamed tier the restore would
        # hoist every streamed block back onto the card behind the offloader's bookkeeping.
        # The other three watch toggles stay: their multipliers ride the same loss-scaling
        # the FT regularisation path uses, and detection compares each image against the
        # cohort at the same epoch, so rotation's boundary shifts cancel. The trainer
        # disarms a ticked-but-hidden box too, so this is presentation, not the guard.
        if hasattr(self, "_krea2_autorecap_cb"):
            self._set_widget_visible(self._krea2_autorecap_cb, not on)
        # Network Type (LoRA/LoKR) is meaningless under a base-model fine-tune — the adapter
        # is inert. Hide the row while FT is on; restore the normal swap when it goes off.
        if hasattr(self, "_network_type_rowf"):
            self._set_widget_visible(self.labels["NETWORK_TYPE"], not on)
            self._set_widget_visible(self._network_type_rowf, not on)
            if on:
                self.hide_row("LOKR_FACTOR")
                self.show_row("NETWORK_DIM")
                self.show_row("NETWORK_ALPHA")
            else:
                self._on_network_type_changed()
        block_mode = str(self.krea2_ft_mode_var.get()) == "block"   # Auto picks its own
        if on and block_mode:
            self._krea2_ft_blocks_lbl.pack(side=tk.LEFT, padx=(14, 4))
            self._krea2_ft_blocks_cb.pack(side=tk.LEFT)
        else:
            self._krea2_ft_blocks_lbl.pack_forget()
            self._krea2_ft_blocks_cb.pack_forget()

    # --- MiniMax H3 rotation fine-tune (mirrors the Krea 2 card) --------------------------
    MINIMAX_FT_DEFAULTS = {
        "LEARNING_RATE": "1e-5",          # a starting point, NOT a calibrated H3 recipe —
                                          # nobody has tuned FT rates on this model yet
        "MAX_TRAIN_EPOCHS": "100",        # a realistic fine-tune length (Peter, 29 Aug:
                                          # 26 was "far too small"; his field A/Bs ran 64
                                          # and kept improving). Clean at BOTH full-speed
                                          # plans — 25 cycles at 4 windows, 20 at 5 — and
                                          # the trainer now snaps any total UP to a cycle
                                          # boundary at launch anyway (snap_ft_epochs),
                                          # so odd window counts still end evenly trained.
        # SAVE_EVERY_N_EPOCHS is NOT a static recipe value: the cycle length depends on the
        # window mode/size, so _refresh_minimax_ft_save_box keeps the box in step live.
        "GRADIENT_ACCUMULATION": "1",     # fused backward consumes grads as they land
        "MAX_GRAD_NORM": "0",             # global clipping is impossible under fused backward
        "NETWORK_TYPE": "LoRA (standard)",  # FT trains the BASE — reset the adapter selector
    }

    def _on_minimax_ft_toggle(self):
        """Recipe pushed on the way ON only, so re-showing the tab never stomps tuned values.

        The likeness tickbox needs NO bridging here: --photo_blocks (and --audio_blocks)
        travel under FT and the TRAINER resolves them — the cycle tightens to the union of
        what the dataset trains, and each modality is confined to its own blocks per batch.
        The Blocks field stays purely manual."""
        self._apply_minimax_ft_visibility()
        # The video-restriction sub-tick lives with likeness but only under FT — re-sync
        # so toggling FT shows/hides it without touching the likeness box itself.
        self._sync_minimax_likeness_state()
        if bool(self.minimax_finetune_var.get()):
            self._apply_minimax_ft_defaults()
            self._refresh_minimax_ft_save_box()

    def _minimax_ft_cycle_estimate(self):
        """Epochs per full rotation cycle — the 32 GB BASELINE of 4 component windows
        (qkv / out / fc1 / fc2) x rotate-every. Since the small-card tiers landed this is
        an estimate, not exact: the trainer's window planner depth-splits fat windows on
        24 GB (5 windows) and streams on 16 GB (more), resolved from free VRAM at LAUNCH —
        unknowable here. The trainer's own cycle snap stays authoritative and logs when
        it corrects the Save-every box."""
        try:
            _every = max(1, int(str(self.minimax_ft_every_var.get()).strip() or 1))
        except ValueError:
            _every = 1
        return 4 * _every

    def _refresh_minimax_ft_save_box(self):
        """Keep Save-every in step with the cycle the FT controls imply.

        Ownership decides what may be rewritten: a value the GUI itself wrote (tracked in
        _minimax_ft_save_autoset) is only ever a suggestion and always follows the cycle —
        without this, rotate-every 1 -> 2 -> 1 stranded the box at 8, because 8 is a
        multiple of 4 and looked like a user choice (field). A USER-typed value is kept
        when it's 0 (final-only) or a non-zero multiple of the cycle (a deliberate sparser
        cadence); anything else is rewritten to the suggestion: EVERY SECOND CYCLE (8 on
        the 4-window baseline, 10 on a 5-window plan) — saves are ~21 GB each and previews
        ride them, so once-per-cycle doubled the disk and preview cost for no gain (Peter,
        29 Aug: ~10 epochs is the right feel). Trainer-side snap stays authoritative at
        launch."""
        # The stop-epoch hint quotes the same cycle length — keep the two in step (cheap,
        # and this refresh fires on every cycle-affecting control).
        try:
            self._refresh_mixed_stop_hint()
        except Exception:
            pass
        if not bool(getattr(self, "minimax_finetune_var", None)
                    and self.minimax_finetune_var.get()):
            return
        entry = self.entries.get("SAVE_EVERY_N_EPOCHS")
        if entry is None:
            return
        cyc = self._minimax_ft_cycle_estimate()
        try:
            cur = int(str(entry.get()).strip() or 0)
        except ValueError:
            cur = -1
        if cur == 2 * cyc:
            self._minimax_ft_save_autoset = 2 * cyc  # already right — claim it as ours
            return
        if cur != getattr(self, "_minimax_ft_save_autoset", None):
            if cur == 0 or (cur > 0 and cur % cyc == 0):
                return                              # the user's own deliberate cadence
        # A user-typed non-multiple snaps UP to the next cycle multiple (10 on a 4-cycle
        # -> 12): the typed number expressed how SPARSE they want 20 GB saves — and,
        # since previews follow saves, previews — so one-per-cycle would be 2.5x what
        # they asked for. A GUI-owned value just tracks the cycle itself.
        _target = (((cur + cyc - 1) // cyc) * cyc
                   if cur > 0 and cur != getattr(self, "_minimax_ft_save_autoset", None)
                   else 2 * cyc)
        try:
            _was = str(entry.cget("state"))
            if _was == "disabled":
                entry.config(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, str(_target))
            if _was == "disabled":
                entry.config(state=_was)
            self._minimax_ft_save_autoset = _target
        except Exception:
            pass

    def _apply_minimax_ft_defaults(self):
        """Same shape as _apply_krea2_ft_defaults — one recipe write, with a console report."""
        changed = []
        if getattr(self, "adaptive_lr_var", None) is not None and self.adaptive_lr_var.get():
            self.adaptive_lr_var.set(False)
            try:
                self._on_adaptive_lr_toggle()
            except Exception:
                pass
            changed.append("Adaptive LR: on -> off (incompatible with rotation)")
        for key, val in self.MINIMAX_FT_DEFAULTS.items():
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                before = entry.get()
                if str(before).strip() == val:
                    continue
                _was = str(entry.cget("state"))
                if _was == "disabled":
                    entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, val)
                if _was == "disabled":
                    entry.config(state=_was)
                changed.append(f"{key.replace('_', ' ').title()}: {before} -> {val}")
            except Exception:
                pass
        if changed:
            self.update_console("[fine-tune] applied the recommended base-model setup:\n  "
                                + "\n  ".join(changed) + "\n")

    def _launch_diff_to_lora(self):
        """Open the Checkpoint to LoRA tool (its own window) from inside the app.

        One click matters most on a RunPod pod, where the GUI in a browser tab is all the
        user has — hunting for run_diff_to_lora.bat/.sh in a noVNC desktop is exactly how
        a field user ended up invoking venv/bin/python by hand (29 Aug). Windows prefers
        pythonw so no console flashes; everywhere else the venv python running this GUI
        launches it directly."""
        import subprocess
        exe = sys.executable
        if os.name == "nt":
            _w = os.path.join(FIZGIG_DIR, "venv", "Scripts", "pythonw.exe")
            if os.path.exists(_w):
                exe = _w
        try:
            subprocess.Popen([exe, os.path.join(FIZGIG_DIR, "diff_to_lora_gui.py")],
                             cwd=FIZGIG_DIR)
        except Exception as e:
            messagebox.showerror("Checkpoint to LoRA", f"Could not launch the tool: {e}")

    def _apply_minimax_ft_visibility(self):
        """FT sub-controls only while the checkbox is on. Everything that is ADAPTER machinery
        hides under FT rather than sitting there silently ignored — Network Type, Optimised
        Likeness Learning, and Blocks to Train (the FT card's own Blocks field is the
        fine-tune's block restriction)."""
        if not hasattr(self, "_minimax_ft_frame"):
            return
        on = bool(self.minimax_finetune_var.get())
        for w in (self._minimax_ft_frame, self._minimax_ft_fused_cb,
                  self._minimax_reg_frame, self._minimax_ft_hint):
            self._set_widget_visible(w, on)
        # The likeness tickbox STAYS — same meaning, different mechanism: under FT it drives
        # the Blocks field (whole fine-tune on the identity blocks) instead of masking photo
        # steps. Its hint swaps to say so. Blocks to Train is adapter-only and hides.
        if hasattr(self, "_minimax_likeness_hint"):
            self._minimax_likeness_hint.config(
                text=self._MINIMAX_LIKENESS_HINT_FT if on else self._MINIMAX_LIKENESS_HINT_LORA)
        for w in (getattr(self, "_minimax_blocks_label", None),
                  getattr(self, "_minimax_blocks_frame", None),
                  getattr(self, "_minimax_blocks_hint", None),
                  # Medium to High LR is a LoRA-mode knob (it rewrites the optimizer's
                  # param-group LR at boundary steps — machinery FT doesn't have). Hidden
                  # under FT; the builder also suppresses the flag.
                  getattr(self, "_minimax_hnlr_label", None),
                  getattr(self, "_minimax_hnlr_frame", None),
                  getattr(self, "_minimax_hnlr_hint", None)):
            if w is not None:
                self._set_widget_visible(w, not on)
        if hasattr(self, "_network_type_rowf"):
            self._set_widget_visible(self.labels["NETWORK_TYPE"], not on)
            self._set_widget_visible(self._network_type_rowf, not on)
            if on:
                self.hide_row("LOKR_FACTOR")
                self.show_row("NETWORK_DIM")
                self.show_row("NETWORK_ALPHA")
            else:
                self._on_network_type_changed()
        # 'Finish one category early' STAYS under FT (retirement works there now, stop-only
        # at cycle boundaries) — its mode picker hides and its hint swaps.
        self._refresh_mixed_stop_hint()
        # A restored session can come up with FT already ON and a stale non-multiple in
        # the save box (field: 10 survived an app restart and the trainer silently snapped
        # it) — visibility runs on every restore/arch-switch, so re-snap here too. No-op
        # when FT is off (the refresh early-returns).
        self._refresh_minimax_ft_save_box()

    def _refresh_mixed_stop_hint(self):
        """Swap the 'Finish one category early' hint and hide the anchor/stop picker under
        FT — retirement there is stop-only and lands on rotation-cycle boundaries. The FT
        text is rebuilt live: the cycle length rides on Rotate every, and a typed epoch
        gets its snap target spelled out (the trainer's snap stays authoritative)."""
        if not hasattr(self, "_mixed_stop_hint"):
            return
        on = bool(getattr(self, "minimax_finetune_var", None)
                  and self.minimax_finetune_var.get())
        _mode = self.entries.get("MIXED_STOP_MODE")
        if _mode is not None:
            self._set_widget_visible(_mode, not on)
        if not on:
            self._mixed_stop_hint.config(text=self._MIXED_STOP_HINT_LORA)
            return
        cyc = self._minimax_ft_cycle_estimate()
        try:
            _n = int(str(self.entries["MIXED_STOP_EPOCH"].get()).strip() or 0)
        except (ValueError, KeyError, tk.TclError):
            _n = 0
        _snap = ((_n + cyc - 1) // cyc) * cyc if _n > 0 else 0
        _ex = (f" Your epoch {_n} lands at {_snap}."
               if _n > 0 and _snap != _n else "")
        self._MIXED_STOP_HINT_FT = (
            "Under fine-tune the finished category STOPS outright (no anchor mode), and "
            "the stop lands on a rotation-cycle boundary — epochs snap UP to the next "
            f"multiple of the {cyc}-epoch cycle, so every window sees the same data mix "
            f"for equal passes before it changes.{_ex} Great for a polish tail: stop "
            "photos & clips and let the voice keep refining its own blocks, or the "
            "reverse.")
        self._mixed_stop_hint.config(text=self._MIXED_STOP_HINT_FT)

    def _refresh_optimizer_choices(self, is_krea2: bool):
        """Point the Optimizer Type dropdown at the selected family's catalog."""
        combo = self.entries.get("OPTIMIZER_TYPE")
        if combo is None:
            return
        choices = self.krea2_optimizer_types if is_krea2 else self.optimizer_types
        combo["values"] = choices
        if combo.get() not in choices:
            combo.set("adamw8bit" if "adamw8bit" in choices else choices[0])

    def _apply_training_arch_visibility(self, is_krea2: bool):
        """Hide Training-tab controls not yet wired into the Krea 2 native trainer; re-show for Klein.

        Deferred-for-Krea-2 feature groups (re-enable by removing from these lists as they land):
          • Model Area to Train (dropdown + desc + Custom panel) — no Krea 2 block map yet
          • Network Dropout                                     — not implemented for krea2
            (the rest of the Optimizer section is wired: Type, Args, Gradient Accumulation,
             Max Grad Norm. The Type dropdown re-populates per family — see
             _refresh_optimizer_choices — because the two resolve names differently.)
          • LR Decay steps                                      — Klein-only warmup_stable_decay
          • Timestep & Noise section                            — krea2 uses a fixed shift schedule
          • FP8 Scaled (in Memory & FP8)                        — krea2's fp8 path is always scaled
          • FP8 Text Encoder (in Memory & FP8)                  — krea2 caches the TE in bf16
          • Gradient Checkpointing (in Memory & FP8)            — krea2_train hardcodes it ON
          • FP8 Base (in Memory & FP8)                          — see below; an OOM trap on krea2
        Kept (model-agnostic / wired): the full live "Override next sample" status-bar panel
        including its Reference image — krea2 reads the override sentinel for previews
        (prompt/seed/resolution) and routes the reference through the Qwen3-VL vision path.

        FP8 Base is hidden for Krea 2 (29 Jul) because unticking it was a guaranteed OOM, not a
        useful option. It sends --no_fp8, i.e. a bf16 base: 25.8 GB of weights alone, ~28 GB in
        total, which no consumer card can hold. The auto swap planner never saw it — the plan is
        identical ticked or unticked — so the run got a swap count sized for fp8/INT8/NF4 and
        then loaded something twice that size. The command builder's elif chain also meant
        unticking it silently dropped the INT8 flag the planner had just chosen. Krea 2's real
        base-precision choices all live on the 4-bit control (Auto / On / Off -> NF4 / INT8 /
        fp8), which the planner does see.
        """
        # Guard: this may run via update_ui_for_architecture before the Training tab is built.
        if not hasattr(self, "_adaptive_cb"):
            return
        # MiniMax H3 is a THIRD, even-more-minimal native family. It shares Krea 2's "hide the
        # Klein-only controls" set, and ALSO hides Krea 2's own extras (base-precision dropdown,
        # per-image loss watch, torch.compile, LoKR network type) — it's LoRA-over-NF4 only, no
        # samples. `native` = "not Klein" (hide Klein-only); `is_krea2` still gates the Krea-2-only
        # widgets, so MiniMax (is_krea2 False) hides them too.
        is_minimax = self._is_minimax_arch()
        native = is_krea2 or is_minimax

        # The single-frame preview caveat belongs to MiniMax only — show it under the Base Model
        # selector when that family is picked, hide it otherwise.
        _note = getattr(self, "_minimax_sample_note", None)
        if _note is not None:
            if is_minimax:
                if not _note.winfo_manager():
                    _note.pack(anchor=tk.W, pady=(10, 0))
            elif _note.winfo_manager():
                _note.pack_forget()
        # Training Base rides in the same card, above the note (before= keeps the order when
        # the note is already on screen).
        _brow = getattr(self, "_minimax_base_frame", None)
        if _brow is not None:
            _bhint = self._minimax_base_hint
            if is_minimax:
                if not _brow.winfo_manager():
                    _kw = {"before": _note} if (_note is not None
                                                and _note.winfo_manager()) else {}
                    _brow.pack(anchor=tk.W, pady=(10, 0), **_kw)
                    _bhint.pack(anchor=tk.W, pady=(2, 0), **_kw)
            elif _brow.winfo_manager():
                _brow.pack_forget()
                _bhint.pack_forget()

        # The live-override REFERENCE image is a Klein edit-model feature. Neither native family
        # is an edit model, and their trainers ignore the field — so hide the picker rather than
        # leave a control that silently does nothing.
        for _n in ("_override_ref_browse_btn", "_override_ref_label", "_override_ref_clear_btn"):
            _w = getattr(self, _n, None)
            if _w is None:
                continue
            if native and _w.winfo_manager():
                _w.pack_forget()
            elif not native and not _w.winfo_manager():
                _w.pack(side=tk.LEFT, **({"padx": (6, 2)} if _n.endswith("label") else {}))
        if native:
            try:
                self.sample_override_ref_var.set("")
            except Exception:
                pass
        # Per-widget groups across the Training Parameters + Memory & FP8 sections.
        widgets = [
            self._modelarea_label, self._modelarea_combo, self._modelarea_desc_label,
            # The whole Weight Optimization row: label, both checkboxes, and the hint under
            # them. Hiding only the controls left an orphaned label and a paragraph of text
            # explaining something no longer on screen.
            self._fp8_row_label, self.fp8_check, self.scaled_check, self._fp8_hint,
            self.fp8_text_encoder_label, self.fp8_text_encoder_check,
            self._grad_checkpoint_label, self.grad_checkpoint_check, self._grad_checkpoint_hint,
            # LR Decay steps: Klein-only (warmup_stable_decay). LR Scheduler + Warmup ARE wired
            # for krea2 (--lr_scheduler / --lr_warmup_steps) so they stay visible.
            self._lr_decay_label, self.entries.get("LR_DECAY_STEPS"),
            # Optimizer Type / Args are wired for BOTH now (krea2 -> --optimizer_type /
            # --optimizer_args); only network dropout has no krea2 equivalent.
            self.labels.get("NETWORK_DROPOUT"), self.entries.get("NETWORK_DROPOUT"),
        ]
        for w in widgets:
            self._set_widget_visible(w, not native)

        # Base precision is the inverse: Krea 2 ONLY. Its options (Auto / INT8 / NF4 / fp8) and
        # the memory strategy behind them are entirely krea2_train's; Klein's trainer has no
        # INT8 path and no auto strategy, so offering the dropdown there would list options
        # Klein cannot run. (Klein's --quant_4bit still exists on its CLI.)
        for w in (self._quant_4bit_label, self.quant_4bit_check, self._quant_4bit_hint):
            self._set_widget_visible(w, is_krea2)

        # The two families resolve optimizer names differently, so the dropdown's contents follow
        # the selector. A name valid for one may not exist in the other (Klein takes module paths;
        # krea2 takes catalog names), so fall back to the shared default rather than carrying a
        # value across that the trainer would then have to reject.
        # MiniMax resolves optimizer names the same catalog-based way Krea 2 does (its trainer
        # takes catalog names too), so it shares Krea 2's dropdown contents.
        self._refresh_optimizer_choices(native)

        # Krea 2-ONLY controls (inverse of the above): the per-image loss watch toggles are only
        # wired into krea2_train for now — hide them under Klein.
        for w in (self._krea2_losswatch_frame, self._krea2_perimglr_cb,
                  self._krea2_autorecap_cb, self._krea2_warmuplook_cb,
                  self._krea2_losswatch_hint, self._krea2_ft_cb,
                  # torch.compile is wired into krea2_train only.
                  self._compile_blocks_label, self.compile_blocks_check, self._compile_blocks_hint):
            self._set_widget_visible(w, is_krea2)
        # The FT sub-controls are gated by the checkbox as well as by the family. Gate on
        # the family, NOT native: each family's FT visibility logic also swaps the Network
        # Type rows, which the other family's logic must never touch. Away from a family,
        # its FT sub-widgets hide outright (the family loop above only covers the checkbox).
        if is_krea2:
            self._apply_krea2_ft_visibility()
        elif hasattr(self, "_krea2_ft_frame"):
            for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_fast_ft_cb,
                      self._krea2_reg_frame, self._krea2_ft_hint):
                self._set_widget_visible(w, False)
        if is_minimax:
            self._apply_minimax_ft_visibility()
        elif hasattr(self, "_minimax_ft_frame"):
            for w in (self._minimax_ft_frame, self._minimax_ft_fused_cb,
                      self._minimax_reg_frame, self._minimax_ft_hint):
                self._set_widget_visible(w, False)
        # Network Type (LoRA/LoKR) is wired for BOTH native families (krea2_train and
        # minimax_train take --network_type/--lokr_factor); Klein trains standard only.
        # The row frame carries the combo + hint together. The speed note is Krea 2-only:
        # on MiniMax the governor holds both types at the same movement rate, so the ~20%
        # LoRA speed edge measured on Krea 2 doesn't translate.
        for w in (self.labels["NETWORK_TYPE"], self._network_type_rowf):
            self._set_widget_visible(w, native)
        self._network_type_hint.config(
            text="LoRA recommended for MiniMax" if is_minimax
            else "LoKR: higher quality · LoRA: ~20% faster training")

        # Detail Focus is the inverse: MiniMax ONLY. Klein and Krea 2 already derive their shift
        # from the sample's token count, so there is nothing to dial there.
        for w in (self._minimax_structure_label, self._minimax_structure_combo,
                  self._minimax_structure_desc,
                  self._minimax_hnlr_label, self._minimax_hnlr_frame, self._minimax_hnlr_hint,
                  self._minimax_blocks_label, self._minimax_blocks_frame, self._minimax_blocks_hint,
                  self._minimax_likeness_cb, self._minimax_likeness_hint,
                  self._minimax_distill_frame, self._minimax_distill_hint,
                  self._minimax_quant_label, self._minimax_quant_frame,
                  self._minimax_quant_hint,
                  self._minimax_smooth_label, self._minimax_smooth_frame,
                  self._minimax_smooth_hint,
                  self._minimax_ramp_label, self._minimax_ramp_frame, self._minimax_ramp_hint,
                  self._minimax_capdrop_label, self._minimax_capdrop_frame,
                  self._minimax_capdrop_hint,
                  self._minimax_mc_frame,
                  self._minimax_ft_cb,
                  ):
            self._set_widget_visible(w, is_minimax)
        # The clean-end box answers to BOTH the family and the dropdown: visible only for MiniMax,
        # and only when the structure is Custom.
        self._refresh_minimax_structure_ui()
        # Blocks to Train greys while Optimised Likeness Learning owns it — arch-dependent, so
        # re-sync on every family switch (a Klein session must not leave it locked).
        self._sync_minimax_likeness_state()
        # The Multi Concept sub-rows are owned by its own toggle handler (they are hidden even
        # under MiniMax until the box is ticked), so route them through it rather than the loop.
        if is_minimax:
            self._on_minimax_multiconcept_toggle()
            self._sync_distill_weight_state()
        else:
            for w in (self._minimax_mc_dir_frame, self._minimax_mc_hint,
                      self._minimax_mc_nodistill_hint):
                self._set_widget_visible(w, False)
        # Retired MiniMax controls — never shown under any family. AdaLN can't deploy on the
        # pruned builds; depth-split LR was superseded by the limiter (9 Aug). The per-step clip
        # and LR warmup joined them 10 Aug: the Adapter-relative LR ramp addresses the same
        # epoch-1 overshoot at its root by holding the step/size ratio steady, so a movement cap
        # and a fixed warmup count are both guesses at a problem that no longer needs them. The
        # command builder locks every one of these regardless of saved settings.
        for w in (self._minimax_adaln_cb, self._minimax_adaln_hint,
                  self._minimax_slow_label, self._minimax_slow_frame, self._minimax_slow_hint,
                  self._minimax_limiter_label, self._minimax_limiter_frame,
                  self._minimax_limiter_hint):
            self._set_widget_visible(w, False)
        if is_minimax:
            for _k, _off in (("MINIMAX_BLOCK_LIMIT", "Off"), ("MINIMAX_LR_WARMUP", "Off")):
                if str(self.entries[_k].get()) != _off:
                    self.entries[_k].set(_off)     # a preset or saved config must not revive it
        # Adaptive LR is hidden under MiniMax: ticking it silently disabled the governor +
        # warmup (they defer to it). The var is forced off so the greyed-LR-box state and the
        # curated launch dict can't carry a stale True into a run.
        for w in (self._adaptive_cb, self._adaptive_frame, self._adaptive_desc_label):
            self._set_widget_visible(w, not is_minimax)
        if is_minimax:
            if self.adaptive_lr_var.get():
                self.adaptive_lr_var.set(False)
                try:
                    self._on_adaptive_lr_toggle()      # un-grey the Learning Rate box
                except Exception:
                    pass
            # Optimizer locked to adamw (the likeness finding) — hide the dropdown row.
            self.hide_row("OPTIMIZER_TYPE")
        else:
            self.show_row("OPTIMIZER_TYPE")

        # Context LoRA is wired for Klein and Krea 2 but NOT MiniMax — hide the whole row there
        # rather than show a picker the trainer silently ignores.
        for w in (self._contextlora_label, self._contextlora_frame,
                  self._contextlora_desc_label, self._contextlora_warn_label):
            self._set_widget_visible(w, not is_minimax)
        if native:
            # Restore the rank/alpha <-> factor row swap for the current selection.
            self._on_network_type_changed()
        else:
            for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_ft_hint):
                self._set_widget_visible(w, False)
            # Klein always shows rank/alpha and never the factor, whatever the combo holds.
            self.show_row("NETWORK_DIM")
            self.show_row("NETWORK_ALPHA")
            self.hide_row("LOKR_FACTOR")

        # Custom block picker: always hidden under the native families (no Krea 2 / MiniMax block
        # map); under Klein, let the Model-Area dropdown decide (only shown when preset = "Custom").
        try:
            if native:
                self._training_custom_frame.grid_remove()
            else:
                self._on_training_preset_changed()
        except Exception:
            pass

        # Whole collapsible sections. Re-show in canonical order (Timestep before Optimizer,
        # Optimizer before Other Options) — show Optimizer first so Timestep's anchor is packed.
        # Optimizer section now stays visible for Krea 2 (Gradient Accumulation + Max Grad Norm
        # are wired); its unwired fields are hidden individually above.
        self._set_training_section_visible("optimizer", "scheduler", True)
        self._set_training_section_visible("timestep", "optimizer", not native)

    # ── Problem Images window (per-image loss watch) ────────────────────

    def _loss_log_dir(self) -> str:
        """<output_dir>/loss_log from the LIVE Output Directory field (settings only refresh at
        start_training, so a user who edits the field pre-launch would otherwise see stale data)."""
        out = ""
        try:
            if hasattr(self, "entries") and "LORA_OUTPUT_DIR" in self.entries:
                out = self.entries["LORA_OUTPUT_DIR"].get().strip()
        except Exception:
            out = ""
        out = out or (self.settings.get("LORA_OUTPUT_DIR", "") or "")
        return os.path.join(out, "loss_log") if out else ""

    def _problem_images_json_path(self) -> str:
        d = self._loss_log_dir()
        return os.path.join(d, "problem_images.json") if d else ""

    def _find_dataset_image(self, key: str):
        """Resolve a loss-watch item key (image basename, no extension) to a file in the training
        image folder. Returns a path or None."""
        folder = self.image_folder_var.get().strip() if hasattr(self, "image_folder_var") else ""
        if not folder or not os.path.isdir(folder):
            return None
        base = os.path.basename(key)
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            p = os.path.join(folder, base + ext)
            if os.path.exists(p):
                return p
        return None

    def _load_thumbs_async(self, jobs, cache):
        """Decode row thumbnails OFF the Tk main thread. Decoding a whole dataset of
        full-resolution PNGs inline froze the Problem Images / Look Filter windows for
        seconds; rows now show a placeholder and fill in as each decode completes.
        jobs: [(image_path, placeholder_label)]; cache: path -> PhotoImage (holds refs)."""
        def work():
            done = []
            for p, lbl in jobs:
                try:
                    # with-block: PIL otherwise keeps the file handle open until GC, and an
                    # open handle makes Windows fail a later move of that image (Look Filter's
                    # "Move Marked" raced this and left a copy behind). thumbnail() forces a
                    # full decode, so the raster stays usable after close.
                    with Image.open(p) as im:
                        im.thumbnail((96, 96), Image.LANCZOS)
                        done.append((p, im, lbl))
                except Exception:
                    done.append((p, None, lbl))
            def apply():
                for p, im, lbl in done:
                    ph = cache.get(p)
                    if ph is None and im is not None:
                        try:
                            ph = ImageTk.PhotoImage(im)
                            cache[p] = ph
                        except Exception:
                            ph = None
                    try:
                        if not lbl.winfo_exists():
                            continue
                        if ph is not None:
                            lbl.config(image=ph, text="")
                        else:
                            lbl.config(text="no\npreview")
                    except Exception:
                        pass
            try:
                self.master.after(0, apply)
            except Exception:
                pass   # GUI torn down mid-decode
        threading.Thread(target=work, daemon=True).start()

    def _open_problem_images_window(self):
        """Live viewer for the per-image loss watch — thumbnails + verdicts, auto-refreshing
        during training from <output_dir>/loss_log/problem_images.json."""
        win = getattr(self, "_problem_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            self._refresh_problem_images(force=True)
            return
        win = tk.Toplevel(self.master)
        win.title("Problem Images — per-image loss watch")
        win.geometry("1010x640")
        win.configure(bg=COLORS["bg_deep"])
        self._problem_win = win
        self._problem_mtime = None
        self._problem_thumbs = {}  # path -> PhotoImage, cached across refreshes (and kept alive)
        self._problem_row_ui = {}  # key -> persistent row widgets (in-place refresh; new window = fresh)
        self._problem_last_order = []
        self._problem_img_paths = getattr(self, "_problem_img_paths", {})  # key -> resolved image path

        head = tk.Frame(win, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=14, pady=(12, 6))
        tk.Label(head, text="Problem Images", font=(FONT_FAMILY, 15, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh", command=lambda: self._refresh_problem_images(force=True)).pack(side=tk.RIGHT)
        self._problem_status = tk.Label(win, text="", font=(FONT_FAMILY, 9),
                                        fg=COLORS["text_muted"], bg=COLORS["bg_deep"],
                                        justify=tk.LEFT, anchor="w")
        self._problem_status.pack(fill=tk.X, padx=14)

        # Wrap the status text to the live window width — without a wraplength, long lines
        # (plateau banners especially) clip off the right edge. The label is packed, so extra
        # wrapped lines push the rows list down cleanly rather than overlapping it.
        def _status_wrap(e):
            wl = max(300, e.width - 32)
            if getattr(self._problem_status, "_wl", None) != wl:
                self._problem_status._wl = wl
                self._problem_status.config(wraplength=wl)
        win.bind("<Configure>", lambda e: _status_wrap(e) if e.widget is win else None, add="+")

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 12))
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vbar = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        rows_id = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(rows_id, width=e.width))
        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.
        self._problem_rows = rows
        self._problem_canvas = canvas

        self._refresh_problem_images(force=True)

        def _tick():
            # Close over THIS window: an orphaned timer from a closed popup must never latch
            # onto a reopened one (each reopen starts its own loop — they'd multiply).
            if not win.winfo_exists() or getattr(self, "_problem_win", None) is not win:
                return
            try:
                self._refresh_problem_images()
            except Exception:
                pass  # one bad refresh (e.g. odd JSON shape) must not kill the auto-refresh loop
            win.after(4000, _tick)
        win.after(4000, _tick)

    def _refresh_problem_images(self, force: bool = False):
        """Re-read problem_images.json (only rebuilds when the file changed, unless forced)."""
        win = getattr(self, "_problem_win", None)
        if win is None or not win.winfo_exists():
            return
        path = self._problem_images_json_path()
        data = None
        if path and os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if not force and mtime == self._problem_mtime:
                    return  # unchanged
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
                    data = None
                self._problem_mtime = mtime  # only after a good parse — a bad read must retry next tick
            except Exception:
                data = None
        elif self._problem_mtime is not None:
            self._problem_mtime = None  # file removed (fresh run wiped it) — fall through to clear the rows
        elif not force:
            return  # still no file; nothing to redraw

        if not data or not data.get("images"):
            for w in self._problem_rows.winfo_children():
                w.destroy()
            self._problem_row_ui = {}
            self._problem_last_order = []
            self._problem_status.config(text="No data yet. Enable “Detect problem images” on the Training tab, "
                                             "start a Krea 2 run, and give it 3+ epochs of warmup.")
            return

        images = data["images"]
        counts = {}
        for s in images.values():
            v = s.get("verdict", "mid")
            counts[v] = counts.get(v, 0) + 1
        _known = ("excluded", "stuck", "suspect", "watch", "warmup", "exhausted", "learning", "mid", "easy")
        tally = "  ·  ".join([f"{counts.get(v, 0)} {v}" for v in _known]
                             + [f"{n} {v}" for v, n in sorted(counts.items()) if v not in _known])
        mode = ("per-image LR active (stuck ×0.5→×0.1 escalating, suspect ×0.7, mined-out ×0.6, learned ×1.1 boost)"
                if data.get("apply_lr") else "detection only")
        imp = data.get("improving_count")
        pend = int(data.get("pending_count") or 0)
        if data.get("plateaued") and data.get("best_epoch_estimate"):
            be = int(data["best_epoch_estimate"])
            if pend:
                progress = (f"⏳ Plateau (provisional) — the settled images finished ≈ epoch {be}, "
                            f"but {pend} image(s) are still being adjudicated (throttled or freshly "
                            f"recaptioned). If they resolve, training may get a second wind and a "
                            f"LATER epoch may be the better checkpoint — wait for the confirmed "
                            f"plateau (0 pending) before stopping.")
            else:
                progress = (f"📍 TRAINING PLATEAUED — best checkpoint ≈ epoch {be}. "
                            f"Scrub epochs {max(1, be - 2)}–{be + 2} in LoRA Royale to pick by eye; "
                            f"later epochs mainly add overbake risk.")
        elif data.get("plateaued"):
            progress = ("⏳ Plateau (provisional) — nothing improving, but "
                        f"{pend} image(s) still being adjudicated." if pend else
                        "📍 TRAINING PLATEAUED — no image is still improving.")
        elif imp is not None:
            progress = f"{imp} image(s) still improving" + (
                f"  ·  best checkpoint so far ≈ epoch {int(data['best_epoch_estimate'])}"
                if data.get("best_epoch_estimate") else "")
        else:
            progress = ""
        self._problem_status.config(
            text=f"Epoch {data.get('epoch', '?')}  ·  {len(images)} images tracked  ·  {mode}\n"
                 f"{tally}" + (f"\n{progress}" if progress else "") + "\n"
                 f"Residual = loss vs. the average at the same noise level (higher = harder than typical). "
                 f"Stuck = hard AND not improving → check the image + caption.")

        # Caption-fix queue/ack state for the row badges: queued = edit waiting for the next epoch
        # boundary (or mid-re-encode), applied = trainer re-encoded it (with the epoch number).
        loss_log_dir = os.path.dirname(path) if path else ""
        queued_keys, applied_info = set(), {}
        for qname in ("caption_updates.json", "caption_updates.json.processing"):
            try:
                qp = os.path.join(loss_log_dir, qname)
                if os.path.exists(qp):
                    with open(qp, encoding="utf-8") as f:
                        queued_keys.update(json.load(f).keys())
            except Exception:
                pass
        try:
            ap = os.path.join(loss_log_dir, "caption_updates_applied.json")
            if os.path.exists(ap):
                with open(ap, encoding="utf-8") as f:
                    applied_info = json.load(f)
        except Exception:
            pass

        style = {
            "excluded": ("#7F8C8D", "EXCLUDED from training — two AI captions couldn't fix it. Edit the caption to re-admit it, or remove it from the dataset."),
            "stuck":    ("#E74C3C", "STUCK — persistently hard, not improving. Review this image/caption."),
            "suspect":  ("#D35400", "Suspect — extremely hard from the start; provisionally slowed while the trend confirms. Worth a caption check now."),
            "watch":    ("#E67E22", "Watching — looked stuck this epoch; needs more epochs to confirm."),
            "warmup":   ("#8E7CC3", "Look-filter outlier easing in — unusual view on an LR ramp toward ×1.0 while the identity core forms; releases early once it starts improving."),
            "exhausted": ("#16A085", "Fully mined — improved a lot, then plateaued. Caption is fine; LR eased to prevent overbake."),
            "learning": ("#5B9BD5", "Learning — hard but improving. Leave it alone."),
            "mid":      ("#95A5A6", "Normal."),
            "easy":     ("#70AD47", "Learned — consistently easy. Gets a gentle ×1.1 boost to keep the healthy signal strong."),
        }
        order = {"excluded": 0, "stuck": 1, "suspect": 2, "watch": 3, "warmup": 4, "exhausted": 5,
                 "learning": 6, "mid": 7, "easy": 8}
        items = sorted(images.items(),
                       key=lambda kv: (order.get(kv[1].get("verdict", "mid"), 2),
                                       -float(kv[1].get("mean_residual", 0.0))))

        # Persistent rows: refreshes UPDATE existing rows in place instead of destroying and
        # recreating hundreds of widgets on the main thread every epoch boundary — that rebuild
        # was the window's remaining lag source. Only appearing/disappearing images create or
        # destroy widgets, and the list only re-packs when the sort order actually changed.
        new_keys = [key for key, _ in items]
        key_set = set(new_keys)
        for k in list(self._problem_row_ui):
            if k not in key_set:
                ui = self._problem_row_ui.pop(k)
                try:
                    ui["frame"].destroy()
                except Exception:
                    pass
        thumb_jobs = []
        for key, s in items:
            ui = self._problem_row_ui.get(key)
            if ui is None:
                ui = self._problem_build_row(key, thumb_jobs)
                self._problem_row_ui[key] = ui
            self._problem_update_row(ui, key, s, data, queued_keys, applied_info, style)
        if new_keys != self._problem_last_order:
            try:
                scroll_pos = self._problem_canvas.yview()[0]
            except Exception:
                scroll_pos = 0.0
            for key in new_keys:
                self._problem_row_ui[key]["frame"].pack_forget()
            for key in new_keys:
                self._problem_row_ui[key]["frame"].pack(fill=tk.X, pady=4)
            self._problem_last_order = new_keys
            try:
                self._problem_rows.update_idletasks()
                self._problem_canvas.yview_moveto(scroll_pos)
            except Exception:
                pass
        if thumb_jobs:
            self._load_thumbs_async(thumb_jobs, self._problem_thumbs)

    def _problem_build_row(self, key, thumb_jobs):
        """Create one persistent Problem Images row (static widgets only — per-refresh state is
        painted by _problem_update_row)."""
        row = tk.Frame(self._problem_rows, bg=COLORS["bg_surface"],
                       highlightbackground=COLORS["border"], highlightthickness=2)
        row.pack(fill=tk.X, pady=4)

        thumb_holder = tk.Frame(row, width=100, height=100, bg=COLORS["bg_surface"])
        thumb_holder.pack_propagate(False)
        thumb_holder.pack(side=tk.LEFT, padx=8, pady=8)
        thumb_lbl = tk.Label(thumb_holder, text="…", font=(FONT_FAMILY, 8),
                             fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        thumb_lbl.pack(expand=True)
        # Path resolution cached per key — probing 5 extensions per image per refresh added
        # hundreds of stat() calls against (often network/spinning) dataset drives.
        img_path = self._problem_img_paths.get(key)
        if img_path is None:
            img_path = self._find_dataset_image(key)
            if img_path:
                self._problem_img_paths[key] = img_path
        if img_path:
            ph = self._problem_thumbs.get(img_path)
            if ph is not None:
                thumb_lbl.config(image=ph, text="")
            else:
                thumb_jobs.append((img_path, thumb_lbl))
        else:
            thumb_lbl.config(text="no\npreview")

        info = tk.Frame(row, bg=COLORS["bg_surface"])
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)
        name_row = tk.Frame(info, bg=COLORS["bg_surface"])
        name_row.pack(fill=tk.X, anchor="w")
        tk.Label(name_row, text=os.path.basename(key), font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        verdict_lbl = tk.Label(name_row, text="", font=(FONT_FAMILY, 9, "bold"),
                               bg=COLORS["bg_surface"])
        verdict_lbl.pack(side=tk.LEFT)
        badge_lbl = tk.Label(name_row, text="", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        badge_lbl.pack(side=tk.LEFT)
        ttk.Button(name_row, text="✏ Edit Caption", width=14,
                   command=lambda k=key: self._open_caption_editor(k)).pack(side=tk.RIGHT, padx=(8, 0))
        stats_lbl = tk.Label(info, text="", font=(FONT_FAMILY, 9),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        stats_lbl.pack(anchor="w", pady=(2, 0))
        blurb_lbl = tk.Label(info, text="", font=(FONT_FAMILY, 8, "italic"),
                             fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        blurb_lbl.pack(anchor="w", pady=(2, 0))
        return {"frame": row, "verdict": verdict_lbl, "badge": badge_lbl,
                "stats": stats_lbl, "blurb": blurb_lbl}

    def _problem_update_row(self, ui, key, s, data, queued_keys, applied_info, style):
        """Paint one row's per-refresh state (verdict colour, badges, stats) in place."""
        verdict = s.get("verdict", "mid")
        color, blurb = style.get(verdict, style["mid"])
        try:
            ui["frame"].config(highlightbackground=color)
            ui["verdict"].config(text=f"  {verdict.upper()}", fg=color)
            if key in queued_keys:
                ui["badge"].config(text="  ✏ fix queued", fg="#F1C40F")
            elif key in applied_info:
                # Ledger entries are per-fix history lists (older files carry a single dict);
                # the badge shows the LATEST fix.
                _entry = applied_info[key]
                if isinstance(_entry, list):
                    _entry = _entry[-1] if _entry else {}
                _ai = _entry.get("auto")
                _att = int(_entry.get("attempt", 1) or 1)
                _ep = _entry.get("epoch", "?")
                if _ai and _att >= 2:
                    _txt = f"  🤖 AI re-captioned ×2 (detailed) @ epoch {_ep} — last chance"
                elif _ai:
                    _txt = f"  🤖 AI re-captioned @ epoch {_ep}"
                else:
                    _txt = f"  ✓ caption re-encoded @ epoch {_ep}"
                ui["badge"].config(text=_txt, fg="#2ECC71")
            else:
                ui["badge"].config(text="")
            # Trend shows the DECISION metric (the half-window drop test the verdicts actually
            # use), not the raw slope — the old slope arrow could say "improving" while the
            # decision bar said otherwise, which read as a contradiction next to a stuck badge.
            slope = float(s.get("slope", 0.0))
            if "improving" in s:
                trend = "↓ improving" if s["improving"] else ("↑ worsening" if slope > 1e-4 else "→ plateau")
            else:
                trend = "↓ improving" if slope < -1e-4 else ("↑ worsening" if slope > 1e-4 else "→ flat")
            # Stuck + improving = release countdown; make the state legible instead of confusing.
            rv = int(s.get("release_votes", 0))
            if verdict == "stuck" and s.get("improving"):
                trend += f" — releasing ({rv}/3 clean epochs)"
            elif verdict == "stuck" and s.get("stuck_epochs"):
                trend += f" — stuck {int(s['stuck_epochs'])} epochs"
            mult = s.get("multiplier", 1.0)
            ui["stats"].config(text=(
                f"difficulty {float(s.get('mean_residual', 0.0)):+.4f}   ·   trend {trend}   ·   "
                f"mean loss {float(s.get('mean_loss', 0.0)):.4f}   ·   "
                f"{int(s.get('epochs', 0))} epochs tracked"
                + (f"   ·   LR ×{mult:g}" if data.get("apply_lr") and mult != 1.0 else "")))
            ui["blurb"].config(text=blurb, fg=COLORS["text_muted"])
        except Exception:
            pass   # a dying widget mid-refresh must not kill the loop

    def _find_dataset_caption(self, key: str):
        """Caption file for a loss-watch item key: <image_folder>/<basename><caption_ext>."""
        folder = self.image_folder_var.get().strip() if hasattr(self, "image_folder_var") else ""
        if not folder or not os.path.isdir(folder):
            return None
        ext = ".txt"
        try:
            ext = self.dataset_caption_ext_var.get().strip() or ".txt"
        except Exception:
            pass
        return os.path.join(folder, os.path.basename(key) + ext)

    def _queue_caption_update(self, key: str, caption: str) -> bool:
        """Merge one caption edit into <output_dir>/loss_log/caption_updates.json (atomic write).
        The trainer consumes it at the next epoch boundary and re-encodes the embedding."""
        d = self._loss_log_dir()
        if not d:
            return False
        try:
            os.makedirs(d, exist_ok=True)
            qp = os.path.join(d, "caption_updates.json")
            updates = {}
            if os.path.exists(qp):
                try:
                    with open(qp, encoding="utf-8") as f:
                        updates = json.load(f)
                    if not isinstance(updates, dict):
                        updates = {}
                except Exception:
                    # An unreadable EXISTING queue means other pending edits we can't see —
                    # writing just this key would silently discard them. Fail instead.
                    print("[caption-fix] queue exists but could not be read — not overwriting it")
                    return False
            updates[str(key)] = caption
            tmp = qp + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(updates, f, indent=2)
            os.replace(tmp, qp)  # atomic — the trainer never sees a half-written file
            return True
        except Exception as e:
            print(f"[caption-fix] queue failed: {e}")
            return False

    def _open_caption_editor(self, key: str):
        """Standalone caption editor for one problem image. Deliberately a SEPARATE Toplevel from
        the auto-refreshing list, so rows can rebuild/reorder underneath without eating your edit."""
        editors = getattr(self, "_caption_editors", None)
        if editors is None:
            editors = self._caption_editors = {}
        existing = editors.get(key)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return

        win = tk.Toplevel(self.master)
        win.title(f"Edit Caption — {os.path.basename(key)}")
        win.geometry("560x680")
        win.minsize(520, 560)
        win.configure(bg=COLORS["bg_deep"])
        editors[key] = win
        win.bind("<Destroy>", lambda e: editors.pop(key, None) if e.widget is win else None)

        # Bottom bar FIRST with side=BOTTOM so the buttons can never be clipped off the window,
        # whatever the thumbnail aspect/caption length pushes the middle content to.
        btns = tk.Frame(win, bg=COLORS["bg_deep"])
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(6, 12))
        status = tk.Label(win, text="Tip: caption what the image actually shows — viewpoint "
                                    "(“from behind”, “side profile”), pose, occlusions.",
                          font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_deep"],
                          wraplength=520, justify=tk.LEFT)
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=14)

        img_path = self._find_dataset_image(key)
        if img_path:
            try:
                im = Image.open(img_path)
                im.thumbnail((280, 280), Image.LANCZOS)
                ph = ImageTk.PhotoImage(im)
                win._thumb_ref = ph  # keep alive
                tk.Label(win, image=ph, bg=COLORS["bg_deep"]).pack(pady=(14, 6))
            except Exception:
                img_path = None
        if not img_path:
            tk.Label(win, text="(image preview unavailable)", font=(FONT_FAMILY, 9),
                     fg=COLORS["text_muted"], bg=COLORS["bg_deep"]).pack(pady=(14, 6))

        cap_path = self._find_dataset_caption(key)
        caption = ""
        cap_read_failed = False
        if cap_path and os.path.exists(cap_path):
            try:
                # utf-8-sig strips a BOM (which would otherwise ride into the embedding);
                # errors="replace" keeps a legacy-ANSI caption editable instead of blank.
                with open(cap_path, encoding="utf-8-sig", errors="replace") as f:
                    caption = f.read().strip()
            except Exception:
                cap_read_failed = True

        tk.Label(win, text=os.path.basename(key), font=(FONT_FAMILY, 11, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack()
        txt = tk.Text(win, height=6, wrap=tk.WORD, font=(FONT_FAMILY, 10),
                      bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                      insertbackground=COLORS["text_primary"], relief=tk.FLAT, padx=8, pady=8)
        txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        txt.insert("1.0", caption)
        if cap_read_failed:
            status.config(fg="#E74C3C", text="Couldn't read the existing caption file — the box "
                          "starts empty. Saving will OVERWRITE the .txt with what you type.")

        def _save():
            new_cap = txt.get("1.0", tk.END).strip()
            if not new_cap:
                status.config(text="Caption is empty — not saved.", fg="#E74C3C")
                return
            wrote_txt = False
            if cap_path:
                try:
                    with open(cap_path, "w", encoding="utf-8") as f:
                        f.write(new_cap)
                    wrote_txt = True
                except Exception as e:
                    print(f"[caption-fix] .txt write failed: {e}")
            queued = self._queue_caption_update(key, new_cap)
            if queued:
                status.config(fg="#2ECC71", text="Saved & queued ✓ — the trainer re-encodes it at the next "
                              "epoch boundary; this image's history resets and it should turn blue "
                              "(learning) if the fix worked."
                              + ("" if wrote_txt else "  (Note: couldn't write the .txt — the live run is "
                                 "fixed, but re-caching later will use the old caption.)"))
            else:
                status.config(fg="#E74C3C", text="Could not queue the update (no output directory?). "
                              + ("The .txt was updated for future runs." if wrote_txt else ""))

        ttk.Button(btns, text="Save & Queue for Re-encode", command=_save).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    # ── Timestep section helpers ────────────────────────────────────────

    def _on_adaptive_lr_toggle(self):
        """Enable/disable the Min/Max LR dropdowns based on the Adaptive LR checkbox,
        and grey out the Learning Rate box — it is IGNORED while adaptive is on (the run
        starts at the geometric midpoint of Min/Max; the watcher owns the LR from there)."""
        if not hasattr(self, 'entries') or "ADAPTIVE_LR_MIN" not in self.entries:
            return
        on = self.adaptive_lr_var.get()
        # Comboboxes: "readonly" when enabled (dropdown active, no free typing), "disabled" when not
        combo_state = "readonly" if on else "disabled"
        btn_state = "normal" if on else "disabled"
        self.entries["ADAPTIVE_LR_MIN"].config(state=combo_state)
        self.entries["ADAPTIVE_LR_MAX"].config(state=combo_state)
        if hasattr(self, '_adaptive_reset_btn'):
            self._adaptive_reset_btn.config(state=btn_state)
        # The LR box is the inverse: live when adaptive is OFF, greyed when ON.
        try:
            if "LEARNING_RATE" in self.entries:
                self.entries["LEARNING_RATE"].config(state="disabled" if on else "normal")
            if hasattr(self, "labels") and "LEARNING_RATE" in self.labels:
                self.labels["LEARNING_RATE"].config(
                    fg=COLORS["text_muted"] if on else COLORS["text_secondary"])
        except Exception:
            pass

    def _parse_blocks_swap(self) -> int:
        """Extract integer from the BLOCKS_SWAP combobox value.
        'Auto' resolves to a value based on GPU VRAM (training needs more headroom than inference)."""
        import re as _re
        raw = self.entries["BLOCKS_SWAP"].get().strip()
        if raw.lower().startswith("auto"):
            cfg = ARCHITECTURES.get(self.architecture_var.get(), {})
            if cfg.get("is_krea2"):
                return self._auto_krea2_strategy()
            return self._auto_training_blocks_swap()
        # Explicit swap value: any INT8 pick from a PREVIOUS auto pass must not leak into
        # this launch (stale --quant_int8 alongside --blocks_to_swap N OOM'd small cards).
        self._auto_quant_int8 = ""
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _auto_krea2_strategy(self) -> int:
        """Choose Krea 2 quantisation AND swap together, then return the swap count.

        Picking a swap count from VRAM alone produced the worst possible outcome on 16 GB
        cards: fp8 doesn't fit, so it swapped 20 of 28 blocks every step. Measured on a 5090
        (Krea 2, 36 imgs @ 0.25 MP, batch 1):

            fp8, no swap   0.85 s/it   20.1 GB   12.5% CPU
            fp8, swap 20   3.09 s/it   12.3 GB   49.9% CPU
            NF4, no swap   0.70 s/it   13.8 GB   14.0% CPU

        NF4 is both faster and smaller, so it leads. Only touches the 4-bit toggle when the
        user has left block swap on Auto — an explicit swap choice is left alone.
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "src"))
            from fizgig.utils.capabilities import detect, recommend_krea2_strategy
        except Exception:
            self._auto_quant_int8 = ""   # no strategy ran — a stale INT8 pick must not leak
            return self._auto_krea2_blocks_swap()

        try:
            caps = detect()
            # Budget for THIS run's shape — batch size is the largest term (+2.4 GB/image);
            # a single-constant budget let batch 2 sail through the check and OOM.
            try:
                _mp = float(self.dataset_megapixels_var.get().strip() or 0.25)
            except (ValueError, AttributeError):
                _mp = 0.25
            try:
                _bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
            except (ValueError, AttributeError):
                _bs = 1
            try:
                _rk = int(self.entries["NETWORK_DIM"].get().strip() or 32)
            except (ValueError, KeyError, AttributeError):
                _rk = 32
            # LoKR: the hidden rank box is meaningless (baseline it at 32) and the factor
            # carries the real state cost — params scale 1/factor², so factor 4 is ~+4 GB
            # the budget must know about on tight cards.
            _ntype = "lokr" if self._network_type_is_lokr() else "lora"
            if _ntype == "lokr":
                _rk = 32
            try:
                _lf = int(self.entries["LOKR_FACTOR"].get().strip() or 8)
            except (ValueError, KeyError, AttributeError):
                _lf = 8
            # If the user pinned the 4-bit control, the plan must be built AROUND that choice —
            # otherwise the swap count is sized for a quantisation that will not run. That
            # exact mismatch (fp8 given NF4's swap-0 plan) OOM'd 16 GB cards; reproduced and
            # fixed 28 Jul.
            #
            # "Off" maps to no_4bit, not fp8: the control is labelled *4-bit Base*, so turning
            # it off is a vote against NF4, not against every quantisation. INT8 is 8-bit,
            # faster than NF4 and far more accurate, so it still applies where it fits —
            # briefly making Off mean plain fp8 cost 20 GB+ cards the fastest path for nothing.
            _force = self._krea2_force_quant() if hasattr(self, "quant_4bit_mode_var") else None
            plan = recommend_krea2_strategy(caps=caps, mp=_mp, batch=_bs, rank=_rk,
                                            force_quant=_force,
                                            network_type=_ntype, lokr_factor=_lf)
        except Exception:
            self._auto_quant_int8 = ""   # no strategy ran — a stale INT8 pick must not leak
            return self._auto_krea2_blocks_swap()

        try:
            self.update_console(f"[auto] {caps.summary()}\n[auto] {plan.reason}\n")
        except Exception:
            pass
        # INT8 has no GUI toggle (it is newer than the 4-bit control) — carry it on the
        # instance so the krea2 command builder can pass --quant_int8.
        self._auto_quant_int8 = getattr(plan, "quant_int8", "") or ""
        # The plan only drives the NF4 flag when Base precision is on Auto — an explicit
        # choice is the user's call and the strategy must not override it. (Compared via the
        # canonical key, not the display label, which is why the label can change freely.)
        _q4_auto = (not hasattr(self, "quant_4bit_mode_var")
                    or self._base_precision() == "auto")
        if (_q4_auto and hasattr(self, "quant_4bit_var")
                and bool(self.quant_4bit_var.get()) != plan.quant_4bit):
            self.quant_4bit_var.set(plan.quant_4bit)
            try:
                self._on_quant_4bit_toggle()
            except Exception:
                pass
            try:
                self.update_console(
                    f"[auto] 4-bit NF4 base turned {'ON' if plan.quant_4bit else 'OFF'} "
                    "(block swap is on Auto — set it explicitly to control this yourself)\n")
            except Exception:
                pass
        return int(plan.blocks_to_swap)

    def _auto_krea2_blocks_swap(self) -> int:
        """Pick Krea 2 training block swap from GPU VRAM. Krea 2's RAW DiT is ~14 GB in fp8,
        so the training step fits a 32 GB card with no swap (fastest — no PCIe transfers); the
        in-training preview parks the training DiT on CPU separately, so swap only governs the
        training step. Smaller cards swap progressively. Max swap is 26 (28 main blocks − 2)."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 30:
                    return 0    # 32 GB — no swap; fp8 base (~14 GB) trains resident
                if vram_gb >= 22:
                    return 12   # 24 GB
                if vram_gb >= 15:
                    return 20   # 16 GB
                return 26       # <16 GB — maximum
        except Exception:
            pass
        return 12  # safe default for an unknown smaller card

    def _auto_krea2_inference_blocks_swap(self) -> int:
        """Pick Krea 2 INFERENCE/preview block swap from GPU VRAM, tuned for the fp8 Turbo.
        Measured: the Turbo peaks ~22.6 GB at swap 0 (DiT + the transient Qwen3-VL encode
        spike) and drops ~0.43 GB per swapped block — heavier than Klein's ~9 GB Distilled, so
        reusing the Klein inference preset would under-swap and OOM smaller cards. This adapts to
        the actual card so the workbench + previews 'just work'. Forward-only (lighter than the
        training step); max swap is 26 (28 main blocks − 2)."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 30:
                    return 0    # 32 GB — Turbo (~22.6 GB peak) fits resident, fastest
                if vram_gb >= 22:
                    return 4    # 24 GB — light swap for headroom over the encode spike
                if vram_gb >= 18:
                    return 12   # 20 GB
                if vram_gb >= 15:
                    return 20   # 16 GB
                return 26       # <16 GB — maximum
        except Exception:
            pass
        return 20  # safe default for an unknown smaller card

    def _auto_training_blocks_swap(self) -> int:
        """Pick training block swap based on GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # 16 GB and up → no swap. The fp8 Base is only ~9.6 GB resident, so
                # 16 GB cards train without swapping, and skipping swap is faster (no
                # PCIe block transfers). Threshold is 15, not 16: a 16 GB card reports
                # ~15.9 GiB total (drivers reserve a little), so a >=16 gate would
                # wrongly exclude it. Only genuinely smaller cards (<15 GB) swap.
                if vram_gb >= 15:
                    return 0   # 16 GB / 24 GB / 32 GB — no swap needed
                if vram_gb >= 10:
                    return 12  # 12 GB cards
                return 16      # <10 GB — maximum swap
        except Exception:
            pass
        return 0  # safe fallback — avoid the buggy swap path on detection failure

    def _on_gpu_choice(self, _event=None):
        """Save the picked GPU by UUID. Label -> UUID, so CUDA_VISIBLE_DEVICES is immune to
        NVML vs. CUDA index reordering (issue #104)."""
        _picked = self._gpu_choice_var.get()
        _uuid = ""
        for k, v in self._gpu_choice_labels.items():
            if v == _picked:
                _uuid = self._gpu_info.get(k, (None, None, None, ""))[3] or ""
                break
        self.prefs_vars["cuda_device"].set(_uuid)          # trace writes prefs.json
        self.update_console(
            f"[gpu] training will use {_picked}. Restart Fizgig to move the workbench tools too.\n"
            if _uuid else "[gpu] back to the system default GPU.\n")

    def _cuda_env_for_subprocess(self, env):
        """Stamp the chosen GPU onto a subprocess environment.

        The child would inherit our own CUDA_VISIBLE_DEVICES anyway, but only the value set at
        startup - so without this, changing the pref would not reach a run until the app was
        restarted, which is the one place it easily can take effect immediately."""
        _want = str(self.prefs_vars["cuda_device"].get()).strip() if hasattr(
            self, "prefs_vars") else ""
        if _want and not getattr(self, "_cuda_device_env_locked", False):
            env["CUDA_VISIBLE_DEVICES"] = _want
        return env

    def _get_inference_blocks_to_swap(self) -> int:
        """Resolve the Preferences inference_blocks_to_swap pref to an int.

        'Auto (detect from GPU)' resolves from VRAM at call time (same as the
        training Blocks Swap setting). Labeled options like '16 (Max — …)' store
        as the full string; we take the leading integer. Returns 0 on failure."""
        import re as _re
        raw = ""
        try:
            raw = str(self.prefs_vars["inference_blocks_to_swap"].get()).strip()
        except Exception:
            return 0
        if raw.lower().startswith("auto"):
            return _auto_detect_blocks_to_swap()
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _get_inference_int8(self) -> bool:
        """Resolve the Preferences 'INT8 fast inference' toggle (workbench + previews) to a bool."""
        try:
            return str(self.prefs_vars["inference_int8"].get()).strip() in ("1", "True", "true")
        except Exception:
            return False

    def _resolve_script(self, config: dict, script_key: str) -> str:
        """Resolve an absolute script path from an architecture config entry.

        Klein lives under FIZGIG_DIR — strip any legacy "FizgigIndependent/" prefix
        on the config value (back-compat with older config strings) and join onto
        FIZGIG_DIR.
        """
        rel = config[script_key]
        if rel.startswith("FizgigIndependent/"):
            rel = rel[len("FizgigIndependent/"):]
        return os.path.join(FIZGIG_DIR, rel)

    def _get_path(self, key: str) -> str:
        """Resolve a model/path setting from the current source of truth.

        Pulls from prefs_vars (model paths) or from the hidden _dataset_config_var
        (dataset config).
        """
        pref_map = {
            "VAE_MODEL": "vae",
            "DIT_MODEL": "base_dit",
            "TEXT_ENCODER": "text_encoder",
            "LORA_OUTPUT_DIR": "lora_output_dir",
        }
        pref_key = pref_map.get(key)
        if pref_key and pref_key in self.prefs_vars:
            return self.prefs_vars[pref_key].get()
        if key == "DATASET_CONFIG":
            return self._dataset_config_var.get() if hasattr(self, "_dataset_config_var") else ""
        return ""

    def _reset_adaptive_lr_defaults(self):
        """Reset Learning Rate, Min LR, and Max LR to adaptive-mode defaults."""
        # Learning Rate is a free-text entry; Min/Max LR are comboboxes.
        lr_entry = self.entries.get("LEARNING_RATE")
        if lr_entry is not None:
            lr_entry.delete(0, tk.END)
            lr_entry.insert(0, "4e-4")
        for key, value in (("ADAPTIVE_LR_MIN", "1e-5"), ("ADAPTIVE_LR_MAX", "4e-4")):
            entry = self.entries.get(key)
            if entry is not None:
                entry.config(state="readonly")
                entry.set(value)
        # Re-apply enabled/disabled state on the adaptive fields
        self._on_adaptive_lr_toggle()

    def _on_timestep_sampling_changed(self, event=None):
        """Enable/disable sigmoid_scale based on selected sampling method."""
        sampling = self.ts_sampling_var.get()
        uses_sigmoid = sampling in ("sigmoid", "shift")
        state = "normal" if uses_sigmoid else "disabled"
        color = COLORS["text_secondary"] if uses_sigmoid else COLORS["text_muted"]
        self.entries["SIGMOID_SCALE"].config(state=state)
        self.ts_sigmoid_label.config(fg=color)

    def _on_weighting_scheme_changed(self, event=None):
        """Enable/disable logit_mean/std and mode_scale based on weighting scheme."""
        scheme = self.weighting_scheme_var.get()

        # Logit Normal params
        is_logit = (scheme == "logit_normal")
        logit_state = "normal" if is_logit else "disabled"
        logit_color = COLORS["text_secondary"] if is_logit else COLORS["text_muted"]
        self.entries["LOGIT_MEAN"].config(state=logit_state)
        self.entries["LOGIT_STD"].config(state=logit_state)
        self.ts_logit_label.config(fg=logit_color)

        # Mode Scale param
        is_mode = (scheme == "mode")
        mode_state = "normal" if is_mode else "disabled"
        mode_color = COLORS["text_secondary"] if is_mode else COLORS["text_muted"]
        self.entries["MODE_SCALE"].config(state=mode_state)
        self.ts_mode_label.config(fg=mode_color)

    def _update_noise_range_label(self):
        """Update the dynamic noise range description label."""
        if not hasattr(self, 'noise_range_label'):
            return
        min_str = self.entries["MIN_TIMESTEP"].get().strip()
        max_str = self.entries["MAX_TIMESTEP"].get().strip()

        if not min_str and not max_str:
            self.noise_range_label.config(text="Full range (default) - All noise levels",
                                          fg=COLORS["accent"])
            return

        try:
            min_val = int(min_str) if min_str else 0
            max_val = int(max_str) if max_str else 1000
        except ValueError:
            self.noise_range_label.config(text="Invalid timestep values", fg=COLORS["error"])
            return

        if min_val == 0 and max_val >= 1000:
            self.noise_range_label.config(text=f"Full range ({min_val}-{max_val}) - All noise levels",
                                          fg=COLORS["accent"])
        elif max_val <= 300:
            self.noise_range_label.config(text=f"High noise ({min_val}-{max_val}) - Composition/structure",
                                          fg=COLORS["success"])
        elif min_val >= 700:
            self.noise_range_label.config(text=f"Low noise ({min_val}-{max_val}) - Details/textures",
                                          fg="#B388FF")  # purple
        elif min_val >= 300 and max_val <= 700:
            self.noise_range_label.config(text=f"Mid noise ({min_val}-{max_val}) - Features/characteristics",
                                          fg=COLORS["warning"])
        else:
            self.noise_range_label.config(text=f"Custom range ({min_val}-{max_val})",
                                          fg=COLORS["text_secondary"])

    def _ts_preset_full_range(self):
        """Timestep preset: Full Range"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_structure(self):
        """Timestep preset: Structure Focus (high noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "0")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "300")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_detail(self):
        """Timestep preset: Detail Focus (low noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "700")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "1000")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_sigmoid(self):
        """Timestep preset: Balanced Sigmoid"""
        self.ts_sampling_var.set("sigmoid")
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def show_row(self, key):
        """Show a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid()
            row_info["entry"].grid()
            if row_info["browse"]:
                row_info["browse"].grid()

    def hide_row(self, key):
        """Hide a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid_remove()
            row_info["entry"].grid_remove()
            if row_info["browse"]:
                row_info["browse"].grid_remove()

    def _network_type_is_lokr(self) -> bool:
        try:
            return str(self.entries["NETWORK_TYPE"].get()).startswith("LoKR")
        except (KeyError, tk.TclError):
            return False

    def _on_network_type_changed(self):
        """LoKR has no rank/alpha — a single Factor dial replaces them, so the rows swap.
        Only meaningful under Krea 2; the arch-visibility pass calls this on family switch."""
        if self._network_type_is_lokr():
            self.hide_row("NETWORK_DIM")
            self.hide_row("NETWORK_ALPHA")
            self.show_row("LOKR_FACTOR")   # hint lives inside the row frame, rides along
        else:
            self.show_row("NETWORK_DIM")
            self.show_row("NETWORK_ALPHA")
            self.hide_row("LOKR_FACTOR")
        self._save_last_used_paths()

    def toggle_scaled(self):
        """Enable or disable the Scaled checkbox based on FP8 checkbox state"""
        if self.fp8_var.get():
            self.scaled_check.config(state=tk.NORMAL)
        else:
            self.scaled_check.config(state=tk.DISABLED)
            self.scaled_var.set(False)

    def _refresh_perimage_toggle_state(self, *args):
        """Grey out the four per-image watch toggles whenever Batch Size > 1.

        Every per-image feature (detection, per-image LR, auto-recaption, look warm-up)
        rests on attributing one step's loss to one image; the trainer already disables
        the LR side loudly at batch > 1, but the tickboxes stayed live and looked like
        they'd work. Values are preserved — dropping batch back to 1 re-enables them."""
        try:
            bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
        except (ValueError, AttributeError):
            bs = 1
        state = tk.DISABLED if bs > 1 else tk.NORMAL
        for cb in (getattr(self, "_krea2_detect_cb", None),
                   getattr(self, "_krea2_perimglr_cb", None),
                   getattr(self, "_krea2_autorecap_cb", None),
                   getattr(self, "_krea2_warmuplook_cb", None)):
            if cb is not None:
                try:
                    cb.configure(state=state)
                except Exception:
                    pass
        note = getattr(self, "_krea2_perimage_batch_note", None)
        if note is not None:
            try:
                if bs > 1:
                    note.grid()
                else:
                    note.grid_remove()
            except Exception:
                pass

    # Base precision (Krea 2). Canonical key -> the label shown in the combobox. Stored as the
    # KEY so the saved value stays stable if the wording changes.
    _BASE_PRECISION_LABELS = {
        "auto": "Auto (recommended)",
        "int8": "INT8 — 8-bit, fastest",
        "nf4":  "4-bit NF4 — smallest",
        "fp8":  "fp8 — least compressed",
    }

    @classmethod
    def _normalize_base_precision(cls, value) -> str:
        """Canonical key from a stored value, a display label, or a legacy Auto/On/Off.

        Legacy migration preserves BEHAVIOUR, not the old label's wording. "Off" meant "not
        4-bit", which the strategy resolved to INT8 wherever it fits — so Off maps to int8,
        and int8 degrades to fp8 by itself on a card that cannot run it or has too little
        VRAM. Mapping Off to fp8 instead would silently drop 20 GB+ cards off the INT8 path,
        which is precisely the regression v2.8.7 had to hotfix.
        """
        v = str(value or "").strip()
        if v in cls._BASE_PRECISION_LABELS:
            return v
        for key, label in cls._BASE_PRECISION_LABELS.items():
            if v == label:
                return key
        return {"Auto": "auto", "On": "nf4", "Off": "int8", "no_4bit": "int8"}.get(v, "auto")

    def _base_precision(self) -> str:
        """Canonical base-precision key currently selected."""
        return self._normalize_base_precision(self.quant_4bit_mode_var.get())

    def _krea2_force_quant(self):
        """force_quant for recommend_krea2_strategy — None when Auto (let the ladder choose)."""
        key = self._base_precision()
        return None if key == "auto" else key

    def _on_quant_4bit_mode_changed(self):
        """Derive quant_4bit_var from the selected base precision. Only NF4 sets it True; Auto
        rests at False and the launch-time strategy sets it (Krea 2 + Blocks Swap on Auto)."""
        self.quant_4bit_var.set(self._base_precision() == "nf4")
        self._on_quant_4bit_toggle()

    def _on_quant_4bit_toggle(self):
        """4-bit (NF4) base forces block swap off (NF4 weights live in
        module._nf4_packed, not .weight, so they can't be swapped) and supersedes
        the fp8 Base options. Grey those controls while it's on."""
        on = self.quant_4bit_var.get()
        try:
            # Show what will actually run: the trainer forces swap to 0 under 4-bit, but
            # the greyed-out box kept displaying the old count. Remember the user's value
            # and restore it when 4-bit is toggled off.
            if on:
                _cur = self.entries["BLOCKS_SWAP"].get()
                if _cur != "0":
                    self._blocks_swap_before_4bit = _cur
                    self.entries["BLOCKS_SWAP"].set("0")
                self.entries["BLOCKS_SWAP"].configure(state="disabled")
            else:
                self.entries["BLOCKS_SWAP"].configure(state="normal")
                _prev = getattr(self, "_blocks_swap_before_4bit", None)
                if _prev is not None and self.entries["BLOCKS_SWAP"].get() == "0":
                    self.entries["BLOCKS_SWAP"].set(_prev)
                    self._blocks_swap_before_4bit = None
        except Exception:
            pass
        for chk in (getattr(self, "fp8_check", None), getattr(self, "scaled_check", None)):
            if chk is not None:
                try:
                    chk.configure(state=tk.DISABLED if on else tk.NORMAL)
                except Exception:
                    pass
        # 4-bit base REQUIRES gradient checkpointing. Its dequant forward (like the
        # old fp8 dequant) materializes a bf16 weight per matmul; without GC,
        # autograd pins all ~112 of them (~18 GB) for the backward — instant OOM on
        # the 10-12 GB cards NF4 targets. There's no frugal-dequant Function for NF4
        # (only fp8), so force GC on and lock the checkbox while NF4 is selected.
        gc_chk = getattr(self, "grad_checkpoint_check", None)
        if gc_chk is not None:
            try:
                if on:
                    self.grad_checkpoint_var.set(True)
                    gc_chk.configure(state=tk.DISABLED)
                else:
                    gc_chk.configure(state=tk.NORMAL)
            except Exception:
                pass
        if not on:
            # restore the Scaled checkbox's dependent-disabled state
            self.toggle_scaled()

    def _on_grad_checkpoint_toggle(self):
        """Warn (VRAM-aware) when gradient checkpointing is switched OFF. Turning
        it on is always safe; off greatly increases activation VRAM, so it only
        fits on big cards with no block swap."""
        if self.grad_checkpoint_var.get():
            return  # ON is always safe — no warning
        # Switched OFF — assess the card.
        try:
            import torch
            vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                       if torch.cuda.is_available() else 0.0)
        except Exception:
            vram_gb = 0.0
        # Is block swap likely active? Auto on a <24GB card resolves to >0; an
        # explicit non-zero value also means swapping.
        swap_raw = str(self.settings.get("BLOCKS_SWAP", "auto")).strip().lower()
        swap_active = False
        if swap_raw.startswith("auto"):
            swap_active = bool(vram_gb) and vram_gb < 23
        else:
            try:
                swap_active = int(''.join(ch for ch in swap_raw if ch.isdigit()) or "0") > 0
            except ValueError:
                swap_active = False

        if (vram_gb and vram_gb < 23) or swap_active:
            vram_txt = f"~{vram_gb:.0f} GB" if vram_gb else "this card"
            messagebox.showwarning(
                "Gradient checkpointing off — likely to run out of memory",
                f"Your GPU reports {vram_txt}"
                + (" and block swap is active" if swap_active else "") + ".\n\n"
                "With gradient checkpointing OFF, a 9B LoRA holds every block's activations for the "
                "backward pass — that usually won't fit under ~24 GB and training will likely hit CUDA "
                "out-of-memory.\n\nThis option is meant for 24 GB+ cards (ideally 32 GB) with Blocks Swap "
                "set to 0. On your setup, leave it ON unless you know you have the headroom.")
        else:
            messagebox.showinfo(
                "Gradient checkpointing off",
                "Gradient checkpointing is now OFF — training runs ~20–30% faster but uses much more VRAM.\n\n"
                "For it to fit, set Blocks Swap to 0 and keep resolution/batch modest. If you hit CUDA "
                "out-of-memory, switch it back on.")

    def _populate_other_options(self, parent, start_row=0):
        """Populate Attention / Logging / Memory / Metadata fields onto the given parent.
        Used to inline these into the Other Options section on the Training tab."""
        row = start_row

        # Attention Mechanism
        ttk.Label(parent, text="Attention Mechanism:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.attention_var = tk.StringVar(value=self.settings["ATTENTION_MECHANISM"])
        attention_options = ["sdpa", "flash3"]
        self.entries["ATTENTION_MECHANISM"] = ttk.Combobox(parent, textvariable=self.attention_var, values=attention_options, state="readonly")
        self.entries["ATTENTION_MECHANISM"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1
        ttk.Label(parent, text="sdpa works on all GPUs. flash3 requires pip install flash-attn and an "
                  "NVIDIA Hopper/Blackwell GPU (H100, RTX 5090, etc.).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        # Logging
        ttk.Label(parent, text="Logging Directory:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOGGING_DIR"] = ttk.Entry(parent, width=40)
        self.entries["LOGGING_DIR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.browse_directory("LOGGING_DIR")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent, text="Log With:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.log_with_var = tk.StringVar(value=self.settings["LOG_WITH"])
        log_with_options = ["none", "tensorboard", "wandb", "all"]
        self.entries["LOG_WITH"] = ttk.Combobox(parent, textvariable=self.log_with_var, values=log_with_options, state="readonly")
        self.entries["LOG_WITH"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Log Prefix:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOG_PREFIX"] = ttk.Entry(parent, width=40)
        self.entries["LOG_PREFIX"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # img_in/txt_in offloading — no-op for Klein 9B, widget kept for preset compat
        self.img_in_txt_in_offloading_var = tk.BooleanVar(value=self.settings["IMG_IN_TXT_IN_OFFLOADING"])
        self.entries["IMG_IN_TXT_IN_OFFLOADING"] = self.img_in_txt_in_offloading_var

        # Metadata
        ttk.Label(parent, text="Metadata Title:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TITLE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TITLE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Author:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_AUTHOR"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_AUTHOR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Description:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_DESCRIPTION"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_DESCRIPTION"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata License:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_LICENSE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_LICENSE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Tags:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TAGS"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TAGS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Trigger Phrase:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TRIGGER_PHRASE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TRIGGER_PHRASE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1
        ttk.Label(parent, text="Blank uses the Captions tab's trigger word.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent, text="Metadata Thumbnail:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_THUMBNAIL"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_THUMBNAIL"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.browse_file("METADATA_THUMBNAIL", "file")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        ttk.Label(parent, text="Blank auto-embeds the latest sample preview; type 'off' to disable.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        return row

    def create_samples_settings(self):
        """Create the Samples tab with sample generation settings (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.samples_tab)

        # Outer bg_deep container so the card stack sits on a consistent background.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Subtitle kept for reconfiguring per family — the "(Distilled 4-step)" parenthetical is
        # Klein's preview stack and means nothing on the other two.
        self._samples_banner = self._add_tab_banner(
            outer,
            "Sample Previews",
            "Preview prompts rendered periodically during training (Distilled 4-step). "
            "Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.",
        )
        _bkids = self._samples_banner.winfo_children()
        self._samples_banner_sub = _bkids[1] if len(_bkids) > 1 else None

        # === Base Model chooser (Peter) — the tab's options are family-shaped (Sample length
        # with sound, Turbo pace are MiniMax-only), so choose the family HERE rather than
        # round-tripping to the Training tab. Same StringVar as the Training-tab combobox, so
        # the two can never disagree; the bind is required because architecture changes ride
        # <<ComboboxSelected>>, not a var trace. Sits OUTSIDE sample_settings_frame on purpose:
        # it must survive the master-enable toggle and the enable/disable widget walk.
        if len(ARCHITECTURE_LIST) > 1:
            arch_card = self._start_section_card(
                outer, "Base Model",
                "Pick what you're training — the sample options below match it. Same setting "
                "as the Training tab.",
            )
            arch_row = tk.Frame(arch_card, bg=COLORS["bg_surface"])
            arch_row.pack(anchor=tk.W)
            tk.Label(
                arch_row, text="Model:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            samples_arch_combo = ttk.Combobox(
                arch_row, textvariable=self.architecture_var, state="readonly",
                width=28, values=ARCHITECTURE_LIST,
            )
            samples_arch_combo.pack(side=tk.LEFT)
            samples_arch_combo.bind("<<ComboboxSelected>>", self._on_architecture_selected)
            ToolTip(samples_arch_combo, "Model family to train (Klein 9B, Krea 2 or MiniMax H3)")
            self._samples_arch_combo = samples_arch_combo

        # Grid holder — video warning / master checkbox / settings block all row-managed
        # so update_samples_ui_for_architecture() can still .grid() / .grid_remove() them.
        grid_holder = tk.Frame(outer, bg=COLORS["bg_deep"])
        grid_holder.pack(fill=tk.X)
        grid_holder.grid_columnconfigure(0, weight=1)

        # --- Video model warning (hidden by default; grid_remove'd at the end) ---
        self.video_model_warning_frame = ttk.Frame(grid_holder)
        self.video_model_warning_frame.grid(row=0, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        ttk.Label(
            self.video_model_warning_frame,
            text="Sample generation is not available for video models (t2v, i2v).",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(
            self.video_model_warning_frame,
            text="Video sampling during training is too slow and memory-intensive.",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 15))
        video_viewer_frame = ttk.Frame(self.video_model_warning_frame)
        video_viewer_frame.pack(anchor=tk.W, pady=10)
        ttk.Button(video_viewer_frame, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_viewer_frame, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT, padx=5)
        self.video_model_warning_frame.grid_remove()

        # --- Master Enable card ---
        self.sample_enabled_var = tk.BooleanVar(value=self.settings["SAMPLE_ENABLED"])
        enable_card_outer = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        enable_card_outer.grid(row=1, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        enable_card = tk.Frame(enable_card_outer, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["border"], highlightthickness=1, bd=0)
        enable_card.pack(fill=tk.X)
        # Use ttk.Checkbutton — themed; sits on bg_surface via style inheritance
        self.sample_enabled_check = ttk.Checkbutton(
            enable_card, text="Enable Sample Generation", variable=self.sample_enabled_var,
            command=self.toggle_sample_settings,
        )
        self.sample_enabled_check.pack(anchor=tk.W, padx=20, pady=14)


        # --- Sample settings container (the 4 cards live inside this) ---
        self.sample_settings_frame = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        self.sample_settings_frame.grid(row=2, column=0, sticky=tk.EW)

        # Card 1: Prompt & Dimensions
        prompt_card = self._start_section_card(
            self.sample_settings_frame, "Prompt & Dimensions",
            "One prompt per line — every line renders its own sample each epoch. "
            "Output size, steps and seed apply to all of them.",
        )
        prompt_card.grid_columnconfigure(1, weight=1)

        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.NW, padx=(0, 10), pady=4)
        self.sample_prompt_text = tk.Text(
            prompt_card, height=3, width=50, bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"], font=(FONT_FAMILY, 10), wrap="word",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.sample_prompt_text.insert("1.0", self.last_used.get("sample_prompt", self.settings["SAMPLE_PROMPT"]))
        self.sample_prompt_text.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self.sample_prompt_text.bind("<KeyRelease>", lambda e: self._save_last_used_paths())
        # Issue #49: "Multi-line prompt" read as ONE prompt that may contain line breaks — two
        # users only discovered multiple prompts by accident. Say what a line actually does.
        tk.Label(prompt_card,
                 text="Each line is a SEPARATE prompt — press Enter to add another sample per "
                      "epoch. Keep a single prompt on one line (long ones wrap by themselves).",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=520, justify=tk.LEFT).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 6))

        ttk.Label(prompt_card, text="Width:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_width_var = tk.StringVar(value=str(self.settings["SAMPLE_WIDTH"]))
        self.sample_width_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_width_var,
            values=SAMPLE_RESOLUTIONS, state="readonly", width=10,
        )
        self.sample_width_combo.grid(row=2, column=1, sticky=tk.W, pady=4)

        ttk.Label(prompt_card, text="Height:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_height_var = tk.StringVar(value=str(self.settings["SAMPLE_HEIGHT"]))
        self.sample_height_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_height_var,
            values=SAMPLE_RESOLUTIONS, state="readonly", width=10,
        )
        self.sample_height_combo.grid(row=3, column=1, sticky=tk.W, pady=4)

        self.sample_steps_label = ttk.Label(prompt_card, text="Steps:")
        self.sample_steps_label.grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_STEPS"]))
        _steps_frame = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        _steps_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_steps_entry = ttk.Entry(_steps_frame, textvariable=self.sample_steps_var, width=10)
        self.sample_steps_entry.pack(side=tk.LEFT)
        self.sample_steps_note = tk.Label(_steps_frame, text="Base samples only — Distilled is locked at 4 steps",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.sample_steps_note.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(prompt_card, text="Seed:").grid(row=5, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_seed_var = tk.StringVar(value=str(self.settings["SAMPLE_SEED"]))
        self.sample_seed_entry = ttk.Entry(prompt_card, textvariable=self.sample_seed_var, width=10)
        self.sample_seed_entry.grid(row=5, column=1, sticky=tk.W, pady=4)
        tk.Label(prompt_card, text="(0 = random)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=5, column=2, sticky=tk.W, padx=(10, 0)
        )

        # Reference image (Klein edit conditioning) — the persistent default for
        # samples; the status-bar override can swap it live mid-run.
        self.sample_ref_label = ttk.Label(prompt_card, text="Reference:")
        self.sample_ref_label.grid(row=6, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_ref_image_var = tk.StringVar(value=self.last_used.get("sample_ref_image", ""))
        _ref_row = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        _ref_row.grid(row=6, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self._sample_ref_row = _ref_row        # hidden wholesale for MiniMax (no ref path)
        self.sample_ref_entry = ttk.Entry(_ref_row, textvariable=self.sample_ref_image_var, state="readonly")
        self.sample_ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sample_ref_browse_btn = ttk.Button(_ref_row, text="Browse…", command=self._browse_sample_ref)
        self.sample_ref_browse_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.sample_ref_clear_btn = ttk.Button(_ref_row, text="Clear", command=lambda: self.sample_ref_image_var.set(""))
        self.sample_ref_clear_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.sample_ref_image_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self.sample_ref_note = tk.Label(prompt_card,
                 text="Optional — Klein is an edit model, so samples can be conditioned on a real image (they edit "
                      "it rather than generate from scratch). Auto-resized to ~0.20 MP so any size is safe. Leave "
                      "empty for normal samples.",
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=560, justify=tk.LEFT)
        self.sample_ref_note.grid(row=7, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))

        # --- Sample length (MiniMax only) — still vs scrubbable clip -----------------------
        # H3 is a video model whose trained range is ~124-362 frames; a single still is out of
        # distribution and previews look worse than the same LoRA rendered as video in ComfyUI.
        # Shown/hidden by update_samples_ui_for_architecture.
        self.sample_frames_label = ttk.Label(prompt_card, text="Sample length:")
        self.sample_frames_label.grid(row=8, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        # Default: 56-FRAME CLIP WITH SOUND (Peter, 17 Aug — reversing the 11 Aug stills
        # call). What changed: the Turbo makes a 6-step clip render affordable, the sampler's
        # audio is finally trustworthy, and a clip IS the regime H3 was trained in — a
        # picture-and-sound preview is now the honest default heartbeat. Without the audio
        # VAE set it degrades to a silent clip with a console note; Still stays in the
        # dropdown for anyone who wants seconds-per-preview.
        self.sample_frames_var = tk.StringVar(
            value=self.last_used.get("sample_frames", "56 frames with sound (~2.3s)"))
        self.sample_frames_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_frames_var, state="readonly", width=34,
            values=["Still (1 frame)", "22 frames (~1s)", "56 frames (~2.3s)",
                    "124 frames (~5s — trained minimum)", "141 frames (~6s)",
                    "22 frames with sound (~1s)",
                    "56 frames with sound (~2.3s)",
                    "124 frames with sound (~5s)"])
        self.sample_frames_combo.grid(row=8, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_frames_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self._sample_frames_hint = tk.Label(prompt_card,
                 text="Samples render as short clips you can scrub in the gallery — the regime "
                      "H3 was trained in. Longer clips cost minutes each, so set the cadence "
                      "with 'Generate every N epochs'. On 16 GB cards previews cap at 768×640 "
                      "and 22 frames (sound kept) — longer or larger picks clamp automatically. "
                      "Full write-up in the README.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=560, justify=tk.LEFT)
        self._sample_frames_hint.grid(row=9, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))
        # hidden until a MiniMax family is selected
        for _w in (self.sample_frames_label, self.sample_frames_combo, self._sample_frames_hint):
            _w.grid_remove()

        # --- Turbo preview pace (MiniMax only) ---------------------------------------------
        # Live only when the Turbo LoRA is set in Preferences; 6 steps at 75% is the tested
        # recommendation (Peter). Shown/hidden with the sample-length row above.
        self.turbo_pace_label = ttk.Label(prompt_card, text="Turbo preview:")
        self.turbo_pace_label.grid(row=10, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._turbo_pace_row = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        self._turbo_pace_row.grid(row=10, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.turbo_steps_entry = ttk.Entry(self._turbo_pace_row, width=5)
        self.turbo_steps_entry.insert(0, str(self.settings.get("MINIMAX_TURBO_STEPS", 6)))
        self.turbo_steps_entry.pack(side=tk.LEFT)
        ttk.Label(self._turbo_pace_row, text="steps at").pack(side=tk.LEFT, padx=6)
        self.turbo_strength_entry = ttk.Entry(self._turbo_pace_row, width=5)
        self.turbo_strength_entry.insert(0, str(self.settings.get("MINIMAX_TURBO_STRENGTH", 75)))
        self.turbo_strength_entry.pack(side=tk.LEFT)
        ttk.Label(self._turbo_pace_row, text="% strength").pack(side=tk.LEFT, padx=(6, 0))
        self._turbo_pace_hint = tk.Label(prompt_card,
                 text="Used when the Turbo LoRA is set in Preferences: previews render in "
                      "these few steps with the Turbo at this strength on top of your "
                      "training LoRA — previews only, never the saved LoRA. 6 steps at 75% "
                      "is the tested recommendation; without the Turbo, the Steps box above "
                      "applies as before.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=560, justify=tk.LEFT)
        self._turbo_pace_hint.grid(row=11, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))
        for _w in (self.turbo_pace_label, self._turbo_pace_row, self._turbo_pace_hint):
            _w.grid_remove()

        # Card 2: Generation Frequency
        freq_card = self._start_section_card(
            self.sample_settings_frame, "Generation Frequency",
            "How often preview renders fire during training. Set either value to 0 to disable that cadence.",
        )
        freq_card.grid_columnconfigure(1, weight=1)

        ttk.Label(freq_card, text="Every N Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_epochs_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_EPOCHS"]))
        self.sample_every_n_epochs_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_epochs_var, width=10)
        self.sample_every_n_epochs_entry.grid(row=0, column=1, sticky=tk.W, pady=4)

        ttk.Label(freq_card, text="Every N Steps:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_STEPS"]))
        self.sample_every_n_steps_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_steps_var, width=10)
        self.sample_every_n_steps_entry.grid(row=1, column=1, sticky=tk.W, pady=4)
        tk.Label(freq_card, text="(0 = disabled)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=2, sticky=tk.W, padx=(10, 0)
        )

        self.sample_at_first_var = tk.BooleanVar(value=self.settings["SAMPLE_AT_FIRST"])
        ttk.Checkbutton(
            freq_card, text="Sample at Start", variable=self.sample_at_first_var
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        self.use_distilled_samples_var = tk.BooleanVar(value=True)
        self.use_distilled_check = ttk.Checkbutton(
            freq_card, text="Use Distilled model for samples (4-step, matches ComfyUI)",
            variable=self.use_distilled_samples_var,
            command=self._on_distilled_samples_toggled,
        )
        self.use_distilled_check.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        # Cache the Distilled sample model in CPU RAM between epochs (skip disk reload).
        self.cache_sample_model_label = ttk.Label(freq_card, text="Cache sample model in RAM:")
        self.cache_sample_model_label.grid(
            row=4, column=0, sticky=tk.W, padx=(0, 10), pady=(8, 0))
        # Allow-list on restore: a saved value outside auto/on/off lands in the readonly
        # combobox without complaint otherwise.
        _csm = str(self.settings.get("CACHE_SAMPLE_MODEL", "auto"))
        self.cache_sample_model_var = tk.StringVar(value=_csm if _csm in ("auto", "on", "off") else "auto")
        self.cache_sample_model_combo = ttk.Combobox(freq_card, textvariable=self.cache_sample_model_var,
                     values=["auto", "on", "off"], state="readonly", width=8)
        self.cache_sample_model_combo.grid(row=4, column=1, sticky=tk.W, pady=(8, 0))
        self.cache_sample_model_note = tk.Label(freq_card,
                 text="Keeps the ~10 GB Distilled model resident in system RAM between epochs so it isn't "
                      "re-read from disk every sample (~3–4 s/epoch saved). auto = only when free RAM is "
                      "comfortable; off = reload each time. Only applies when sampling isn't block-swapping the "
                      "Distilled — i.e. 24 GB+ cards, where the sample peaks around ~18 GB).",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self.cache_sample_model_note.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))

        # Krea 2 preview engine — lives HERE with the other sample-model choices (the Distilled
        # toggle above is Klein's equivalent choice). Shown only in Krea 2 mode, via
        # _apply_samples_klein_only. raw_lora: previews render on the resident training DiT
        # with the Turbo LoRA @1.0 — identical 8-step CFG-free settings, no ~13 GB Turbo load,
        # no parking the trainer to CPU per preview. turbo_model: classic checkpoint path.
        self._KREA2_ENGINE_LABELS = {
            "raw_lora": "RAW + Turbo LoRA (no model swapping — recommended)",
            "turbo_model": "Turbo model (classic — loads the fp8 Turbo each preview)",
        }
        _eng_saved = str(self.last_used.get("krea2_preview_engine", "raw_lora"))
        if _eng_saved not in self._KREA2_ENGINE_LABELS:
            _eng_saved = "raw_lora"
        self.krea2_preview_engine_var = tk.StringVar(value=self._KREA2_ENGINE_LABELS[_eng_saved])
        self.krea2_engine_frame = tk.Frame(freq_card, bg=COLORS["bg_surface"])
        tk.Label(self.krea2_engine_frame, text="Krea 2 preview engine:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))
        _eng_combo = ttk.Combobox(self.krea2_engine_frame, textvariable=self.krea2_preview_engine_var,
                                  state="readonly", width=52,
                                  values=list(self._KREA2_ENGINE_LABELS.values()))
        _eng_combo.pack(side=tk.LEFT)
        _eng_combo.bind("<<ComboboxSelected>>", lambda e: self._save_last_used_paths())
        ToolTip(_eng_combo,
                "RAW + Turbo LoRA renders previews on the training model itself with the Turbo\n"
                "distillation LoRA applied at 1.0 — same 8-step CFG-free settings as the Turbo\n"
                "model, but nothing is loaded or moved between epochs. The LoRA auto-downloads\n"
                "(~470 MB) if missing. The classic mode loads the fp8 Turbo checkpoint per preview.")
        self.krea2_engine_note = tk.Label(
            freq_card,
            text="Renders previews on the model already training, with the official Turbo LoRA "
                 "switched on just for the render — nothing loaded or moved between epochs. The "
                 "classic mode loads the ~13 GB Turbo checkpoint per preview instead.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=600, justify=tk.LEFT)
        # Gridded (rows 6-7) / removed by _apply_samples_klein_only; hidden by default (Klein).

        # Card 3: Architecture-Specific (Flow Shift / Guidance / Negative / CFG)
        arch_card = self._start_section_card(
            self.sample_settings_frame, "Advanced",
            "Architecture-specific knobs. Distilled models disable Negative Prompt; "
            "non-distilled models disable CFG Scale.",
        )
        self._samples_arch_card = arch_card       # description reworded per family
        # The whole card is hidden for MiniMax (every row in it is inapplicable), so keep the
        # OUTER frame — _start_section_card returns the inner content frame, and it is the outer
        # that was packed into sample_settings_frame.
        self._samples_arch_outer = arch_card.master.master
        arch_card.grid_columnconfigure(1, weight=1)

        def _arch_note(parent, text):
            return tk.Label(parent, text=text, font=(FONT_FAMILY, 9),
                            fg=COLORS["text_muted"], bg=COLORS["bg_surface"])

        self.sample_flow_shift_label = ttk.Label(arch_card, text="Flow Shift:")
        self.sample_flow_shift_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_flow_shift_var = tk.StringVar(value=str(self.settings["SAMPLE_FLOW_SHIFT"]))
        _flow_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _flow_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_flow_shift_entry = ttk.Entry(_flow_frame, textvariable=self.sample_flow_shift_var, width=10)
        self.sample_flow_shift_entry.pack(side=tk.LEFT)
        self._sample_flow_note = _arch_note(
            _flow_frame, "Base samples only — Distilled uses its own schedule")
        self._sample_flow_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_flow_shift_row = 0

        # (Guidance Scale removed — Klein Base has no guidance embed and Distilled
        # is locked at 1.0, so the field did nothing for Klein. CFG Scale is the
        # only real steering knob for Base samples.)

        self.sample_negative_label = ttk.Label(arch_card, text="Negative Prompt:")
        self.sample_negative_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_negative_var = tk.StringVar(value=self.settings["SAMPLE_NEGATIVE"])
        _neg_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _neg_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_negative_entry = ttk.Entry(_neg_frame, textvariable=self.sample_negative_var, width=50)
        self.sample_negative_entry.pack(side=tk.LEFT)
        self._sample_neg_note = _arch_note(_neg_frame, "Base samples only — Distilled ignores it")
        self._sample_neg_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_negative_row = 1

        self.sample_cfg_label = ttk.Label(arch_card, text="CFG Scale:")
        self.sample_cfg_label.grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_cfg_scale_var = tk.StringVar(value=str(self.settings["SAMPLE_CFG_SCALE"]))
        _cfg_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _cfg_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_cfg_scale_entry = ttk.Entry(_cfg_frame, textvariable=self.sample_cfg_scale_var, width=10)
        self.sample_cfg_scale_entry.pack(side=tk.LEFT)
        self._sample_cfg_note = _arch_note(_cfg_frame, "Base samples only — Distilled uses no CFG")
        self._sample_cfg_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_cfg_row = 2

        # Card 4: Viewer
        viewer_card = self._start_section_card(
            self.sample_settings_frame, "Viewer",
            "Browse generated samples without leaving the app, or open the folder in Explorer.",
        )
        # Anchor for re-showing the Advanced card: pack() alone would re-add it at the BOTTOM,
        # below Viewer, so it has to go back in before this one.
        self._samples_viewer_outer = viewer_card.master.master

        viewer_buttons = tk.Frame(viewer_card, bg=COLORS["bg_surface"])
        viewer_buttons.pack(anchor=tk.W)
        ttk.Button(viewer_buttons, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(viewer_buttons, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT)

        self.sample_output_label = tk.Label(
            viewer_card, text="Sample output: <output_dir>/sample/",
            font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
        )
        self.sample_output_label.pack(anchor=tk.W, pady=(10, 0))

        # Store entries for saving/loading
        self.entries["SAMPLE_ENABLED"] = self.sample_enabled_var
        self.entries["SAMPLE_WIDTH"] = self.sample_width_combo
        self.entries["SAMPLE_HEIGHT"] = self.sample_height_combo
        self.entries["SAMPLE_STEPS"] = self.sample_steps_entry
        self.entries["SAMPLE_SEED"] = self.sample_seed_entry
        self.entries["SAMPLE_EVERY_N_EPOCHS"] = self.sample_every_n_epochs_entry
        self.entries["SAMPLE_EVERY_N_STEPS"] = self.sample_every_n_steps_entry
        self.entries["SAMPLE_AT_FIRST"] = self.sample_at_first_var
        self.entries["SAMPLE_FLOW_SHIFT"] = self.sample_flow_shift_entry
        self.entries["SAMPLE_NEGATIVE"] = self.sample_negative_entry
        self.entries["SAMPLE_CFG_SCALE"] = self.sample_cfg_scale_entry
        self.entries["SAMPLE_FRAMES"] = self.sample_frames_combo
        self.entries["MINIMAX_TURBO_STEPS"] = self.turbo_steps_entry
        self.entries["MINIMAX_TURBO_STRENGTH"] = self.turbo_strength_entry

        # Initial UI state based on current architecture
        self.update_samples_ui_for_architecture()

        self._add_youtube_help_button(outer, "samples")

    def toggle_sample_settings(self):
        """Enable or disable sample settings based on the enable checkbox"""
        state = tk.NORMAL if self.sample_enabled_var.get() else tk.DISABLED

        def _apply(widget):
            try:
                if isinstance(widget, tk.Text):
                    widget.configure(state=state if state == tk.NORMAL else tk.DISABLED)
                else:
                    widget.configure(state=state)
            except tk.TclError:
                pass  # Some widgets don't support state

        def _walk(parent):
            for child in parent.winfo_children():
                _apply(child)
                if child.winfo_children():
                    _walk(child)

        _walk(self.sample_settings_frame)

        # The walk above re-enabled every child uniformly — re-assert the Klein-only
        # greying so the Distilled / Cache-model / Reference controls stay disabled in
        # Krea 2 mode (and the distilled-toggle-driven field states).
        if self.sample_enabled_var.get():
            try:
                cfg = ARCHITECTURES.get(self.architecture_var.get(), {})
                self._apply_samples_klein_only(cfg.get("is_krea2", False))
                self._on_distilled_samples_toggled()
            except Exception:
                pass

    def _on_distilled_samples_toggled(self):
        """Grey out fields that Distilled overrides when the checkbox is ticked."""
        use_distilled = self.use_distilled_samples_var.get()
        state = "disabled" if use_distilled else "normal"
        grey = COLORS["text_muted"] if use_distilled else COLORS["text_primary"]

        # Steps (overridden to 4)
        self.sample_steps_entry.configure(state=state)
        self.sample_steps_label.configure(foreground=grey)
        # Flow Shift (overridden to auto)
        self.sample_flow_shift_entry.configure(state=state)
        self.sample_flow_shift_label.configure(foreground=grey)
        # Negative Prompt (not used by Distilled)
        self.sample_negative_entry.configure(state=state)
        self.sample_negative_label.configure(foreground=grey)
        # CFG Scale (not used by Distilled)
        self.sample_cfg_scale_entry.configure(state=state)
        self.sample_cfg_label.configure(foreground=grey)

    def _lora_name_rename_blocked(self) -> bool:
        """True when the LoRA name must not be touched: a run is live, or one is paused.

        The name is a filename PREFIX, not a label — _detect_latest_state_dir rebuilds
        f'{name}-NNNNNN-state' from it to find a resumable run, and checkpoint discovery
        rebuilds '{name}-NNNNNN.safetensors'. Renaming the box would orphan a paused run in a
        way the user wouldn't notice until they tried to resume it."""
        proc = getattr(self, "current_process", None)
        try:
            if proc is not None and proc.poll() is None:
                return True
        except Exception:
            pass
        try:
            return os.path.exists(self._paused_sidecar_path())
        except Exception:
            return False

    def _apply_lora_name_suffix(self, arch: str):
        """Retag the LoRA name for `arch` — myface_k9b -> myface_krea2 and back.

        Only rewrites a trailing tag that is itself a known family suffix, so names that don't
        follow the convention (bobs_dog, portrait_v2) are never touched. Idempotent: a name that
        already carries the right suffix is left alone, which is what makes it safe to call both
        at startup and after every family switch."""
        entry = self.entries.get("LORA_NAME") if hasattr(self, "entries") else None
        want = ARCHITECTURES.get(arch, {}).get("lora_name_suffix")
        if entry is None or not want or self._lora_name_rename_blocked():
            return
        try:
            name = entry.get().strip()
        except (AttributeError, tk.TclError):
            return
        for suffix in LORA_NAME_SUFFIXES:
            if suffix != want and name.endswith("_" + suffix):
                new = name[: -len(suffix)] + want
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, new)
                except (AttributeError, tk.TclError):
                    pass
                return

    def _on_architecture_selected(self, event=None):
        """Model-family selector changed — refresh sample defaults + presets + persist."""
        # Snapshot the family we're LEAVING before anything reshapes the tab. The combobox
        # has already moved to the new value by the time this fires, so the outgoing family
        # is _arch_last_selected and the widgets still hold its values — but only until
        # update_ui_for_architecture() runs below. Hence: first thing, before any UI work.
        _arch_new = self.architecture_var.get()
        _arch_old = getattr(self, "_arch_last_selected", None)
        _arch_changed = _arch_new != _arch_old
        if _arch_changed and _arch_old:
            try:
                self._arch_settings_memory[_arch_old] = self._collect_preset_values()
                # Capture the preset LABEL here too — refresh_preset_combobox() below
                # rewrites it, so this is the last moment it still names the old family's.
                self._arch_preset_name_memory[_arch_old] = self.custom_preset_var.get()
            except Exception:
                pass

        try:
            self.update_samples_ui_for_architecture()
        except Exception:
            pass
        # Refresh Training-tab field/section visibility for the new architecture
        # (hides Krea 2-unsupported controls; re-shows them for Klein).
        try:
            self.update_ui_for_architecture()
        except Exception:
            pass
        # Pause/Resume availability is architecture-dependent (Krea 2: Start/Stop only).
        try:
            self._refresh_training_buttons()
        except Exception:
            pass
        # Swap the preset dropdown to this family's presets, and actually load values into
        # the fields — either what this family last had, or its default preset.
        #
        # Naming a preset without applying it is a lie the user acts on: switching to Krea 2
        # left Klein's 55 epochs / rank 16 sitting in the fields while the dropdown read
        # "Krea 2 Defaults (rank 32, full model)". Those values don't transfer — Klein's
        # rank/epoch/block-targeting recipe is meaningless for Krea 2.
        #
        # Per-family memory: first visit to a family gets its default preset, every later
        # visit gets that family's own settings back, so flipping over to check something
        # doesn't cost you your tuning. Session-scoped — a restart starts fresh.
        #
        # Only on a REAL family change: re-selecting the entry that's already active fires
        # <<ComboboxSelected>> too, and that must not touch the fields at all.
        try:
            self._arch_last_selected = _arch_new
            self.refresh_preset_combobox()
            builtins = self._builtins_for_arch(_arch_new)
            _default_name = next(iter(builtins)) if builtins else None
            if _arch_changed:
                _remembered = self._arch_settings_memory.get(_arch_new)
                if _remembered:
                    self._apply_preset_values(_remembered)
                    self.custom_preset_var.set(self._arch_preset_name_memory.get(_arch_new, ""))
                    self.update_console(f"[preset] {_arch_new} selected — restored your settings "
                                        f"from earlier this session\n")
                elif _default_name:
                    self._apply_preset_values(builtins[_default_name])
                    self.custom_preset_var.set(_default_name)
                    self.update_console(f"[preset] {_arch_new} selected — applied {_default_name}\n")
            elif _default_name and not self.custom_preset_var.get():
                self.custom_preset_var.set(_default_name)
        except Exception:
            pass
        # Retag the LoRA name LAST — _apply_preset_values above rewrites every field including
        # LORA_NAME, so doing this earlier would just be clobbered. On a return visit the
        # restored name already carries the right suffix and this is a no-op; on a first visit
        # it converts the outgoing family's name (which the built-in presets don't set).
        try:
            self._apply_lora_name_suffix(_arch_new)
        except Exception:
            pass
        try:
            self._save_last_used_paths()
        except Exception:
            pass
        # Audio-only greying depends on BOTH the folder and the family — re-check on a switch.
        try:
            self._refresh_audio_only_ui()
        except Exception:
            pass

    def update_samples_ui_for_architecture(self):
        """Update samples tab UI based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        supports_samples = config.get("supports_samples", False)

        if not supports_samples:
            # Show warning, hide settings
            self.video_model_warning_frame.grid()
            self.sample_enabled_check.grid_remove()
            self.sample_settings_frame.grid_remove()
        else:
            # Hide warning, show settings
            self.video_model_warning_frame.grid_remove()
            self.sample_enabled_check.grid()
            self.sample_settings_frame.grid()

            # Apply this architecture's sample defaults ONLY when the architecture actually
            # changed. This method fires on every Base Model combobox event — including
            # re-picking the same family — and unconditionally overwriting CFG/flow-shift/
            # steps/width/height silently reverted the user's preview config (e.g. a
            # 1024x1024 preview back to 768x768). On a real switch, the outgoing family's
            # values are stashed and restored when the user switches back.
            _prev = getattr(self, "_sample_defaults_arch", None)
            if _prev != arch:
                if not hasattr(self, "_arch_sample_stash"):
                    self._arch_sample_stash = {}
                if _prev is not None:
                    self._arch_sample_stash[_prev] = {
                        "cfg": self.sample_cfg_scale_var.get(),
                        "shift": self.sample_flow_shift_var.get(),
                        "steps": self.sample_steps_var.get(),
                        "w": self.sample_width_var.get(),
                        "h": self.sample_height_var.get(),
                    }
                stash = self._arch_sample_stash.get(arch)
                if stash is not None:
                    self.sample_cfg_scale_var.set(stash["cfg"])
                    self.sample_flow_shift_var.set(stash["shift"])
                    self.sample_steps_var.set(stash["steps"])
                    self.sample_width_var.set(stash["w"])
                    self.sample_height_var.set(stash["h"])
                else:
                    if config.get("sample_cfg_default") is not None:
                        self.sample_cfg_scale_var.set(str(config["sample_cfg_default"]))
                    if config.get("sample_flow_shift_default") is not None:
                        self.sample_flow_shift_var.set(str(config["sample_flow_shift_default"]))
                    if config.get("sample_steps_default") is not None:
                        self.sample_steps_var.set(str(config["sample_steps_default"]))
                    if config.get("sample_width_default") is not None:
                        self.sample_width_var.set(str(config["sample_width_default"]))
                    if config.get("sample_height_default") is not None:
                        self.sample_height_var.set(str(config["sample_height_default"]))
                self._sample_defaults_arch = arch

            # Enable/disable flow shift based on architecture
            if config.get("sample_flow_shift_default") is None:
                self.sample_flow_shift_entry.configure(state=tk.DISABLED)
                self.sample_flow_shift_label.configure(foreground="gray")
            else:
                self.sample_flow_shift_entry.configure(state=tk.NORMAL)
                self.sample_flow_shift_label.configure(foreground=FG_COLOR)

            # Handle distilled models (no negative prompts)
            if config.get("sample_is_distilled", False):
                self.sample_negative_entry.configure(state=tk.DISABLED)
                self.sample_negative_label.configure(foreground="gray")
            else:
                self.sample_negative_entry.configure(state=tk.NORMAL)
                self.sample_negative_label.configure(foreground=FG_COLOR)

            # Handle fixed steps/cfg for distilled models
            if config.get("sample_steps_fixed", False):
                self.sample_steps_entry.configure(state=tk.DISABLED)
            else:
                self.sample_steps_entry.configure(state=tk.NORMAL)

            if config.get("sample_cfg_fixed", False):
                self.sample_cfg_scale_entry.configure(state=tk.DISABLED)
            else:
                self.sample_cfg_scale_entry.configure(state=tk.NORMAL)

            # Grey out / relabel the Klein-only sample controls when Krea 2 is selected.
            self._apply_samples_klein_only(config.get("is_krea2", False))
            # ...then let MiniMax override the wording that is still Klein's. Runs AFTER, so the
            # Klein/Krea 2 paths above are untouched.
            self._apply_samples_minimax(bool(config.get("is_minimax")))

            # Sample length (clip) row — MiniMax only: the other families' preview stacks are
            # image pipelines with no frames axis.
            _mm = bool(config.get("is_minimax"))
            for _w in (getattr(self, "sample_frames_label", None),
                       getattr(self, "sample_frames_combo", None),
                       getattr(self, "_sample_frames_hint", None),
                       getattr(self, "turbo_pace_label", None),
                       getattr(self, "_turbo_pace_row", None),
                       getattr(self, "_turbo_pace_hint", None)):
                if _w is None:
                    continue
                (_w.grid if _mm else _w.grid_remove)()

        # Update sample output path label
        self.update_sample_output_label()

    def _apply_samples_minimax(self, is_minimax):
        """Reword the Samples tab for MiniMax H3, and hide what belongs to the other families.

        The tab was written for Klein and still says so: the banner advertises a Distilled
        4-step preview, and every Advanced row is annotated 'Base samples only — Distilled ...'.
        None of that exists on H3, which renders clips on the model being trained, CFG-free, on
        a fixed shift-12 schedule. Runs after _apply_samples_klein_only so Klein and Krea 2 keep
        exactly the text they had; this only rewrites when MiniMax is the selected family."""
        _MM = {
            "banner": "Preview prompts rendered periodically during training, as short clips on "
                      "the model being trained. Samples land in <output_dir>/sample/ and the "
                      "Gallery button below opens the viewer.",
            "advanced": "H3 renders CFG-free on a fixed schedule, so the knobs it does not use "
                        "are greyed out.",
            "flow": "Fixed at 12 for H3 — the schedule every shipped workflow uses",
            "neg": "Unused — H3 samples render without CFG",
            "cfg": "H3 renders without CFG; the shipped workflow does the same",
            "steps": "20 steps matches the shipped H3 workflow",
        }
        _KLEIN = {
            "banner": "Preview prompts rendered periodically during training (Distilled 4-step). "
                      "Samples land in <output_dir>/sample/ and the Gallery button below opens "
                      "the viewer.",
            "advanced": "Architecture-specific knobs. Distilled models disable Negative Prompt; "
                        "non-distilled models disable CFG Scale.",
            "flow": "Base samples only — Distilled uses its own schedule",
            "neg": "Base samples only — Distilled ignores it",
            "cfg": "Base samples only — Distilled uses no CFG",
        }
        _t = _MM if is_minimax else _KLEIN

        if getattr(self, "_samples_banner_sub", None) is not None:
            self._samples_banner_sub.configure(text=_t["banner"])
        _adv = getattr(self, "_samples_arch_card", None)
        if _adv is not None and getattr(_adv, "_desc_label", None) is not None:
            _adv._desc_label.configure(text=_t["advanced"])

        # The whole Advanced card goes for MiniMax: Flow Shift is fixed at 12, and Negative
        # Prompt and CFG Scale are both inert on a CFG-free family — three greyed rows under a
        # heading is just a card asking to be misread.
        _adv_outer = getattr(self, "_samples_arch_outer", None)
        if _adv_outer is not None:
            if is_minimax:
                _adv_outer.pack_forget()
            elif not _adv_outer.winfo_manager():
                _anchor = getattr(self, "_samples_viewer_outer", None)
                if _anchor is not None and _anchor.winfo_manager():
                    _adv_outer.pack(fill=tk.X, padx=36, pady=(0, 16), before=_anchor)
                else:
                    _adv_outer.pack(fill=tk.X, padx=36, pady=(0, 16))
        for _attr, _key in (("_sample_flow_note", "flow"), ("_sample_neg_note", "neg"),
                            ("_sample_cfg_note", "cfg")):
            _w = getattr(self, _attr, None)
            if _w is not None:
                _w.configure(text=_t[_key])
        # Steps note: _apply_samples_klein_only owns the Klein/Krea 2 wording, so only override.
        if is_minimax and hasattr(self, "sample_steps_note"):
            self.sample_steps_note.configure(text=_MM["steps"])

        # Klein's sample-model choice and its RAM cache have no MiniMax equivalent — previews
        # always render on the resident training DiT. The Reference row goes too: it exists
        # because Klein is an EDIT model (and Krea 2 has a vision path); H3 has neither, and its
        # own r2v reference conditioning is a training feature, not a sample one. Hide rather
        # than grey — a disabled control still reads as "something I could turn on".
        for _attr in ("use_distilled_check", "cache_sample_model_label",
                      "cache_sample_model_combo",
                      "sample_ref_label", "_sample_ref_row", "sample_ref_note"):
            _w = getattr(self, _attr, None)
            if _w is None:
                continue
            try:
                (_w.grid_remove if is_minimax else _w.grid)()
            except tk.TclError:
                pass          # packed, not gridded — leave it alone

    def _apply_samples_klein_only(self, is_krea2):
        """Mark the Klein-only sample controls when Krea 2 is selected.

        Krea 2 always renders previews on its fp8 Turbo (8-step, CFG-free, no edit
        reference, model reloaded per pass), so the 'Use Distilled model' toggle, the
        'Cache sample model in RAM' dropdown, and the Klein edit-Reference image don't
        apply. Disable them and relabel as 'Klein only' so it's clear, rather than
        silently leaving live controls that do nothing in Krea 2 mode."""
        muted = COLORS["text_muted"]
        secondary = COLORS["text_secondary"]
        label_fg = muted if is_krea2 else secondary

        # "Use Distilled model for samples" checkbox — Klein's sample-model choice. Krea 2's
        # equivalent choice is the Preview engine dropdown, shown right below in Krea 2 mode.
        if hasattr(self, "use_distilled_check"):
            self.use_distilled_check.configure(
                state=(tk.DISABLED if is_krea2 else tk.NORMAL),
                text=("Use Distilled model for samples — Klein only (Krea 2: pick a preview engine below)"
                      if is_krea2 else
                      "Use Distilled model for samples (4-step, matches ComfyUI)"))

        # Steps note
        if hasattr(self, "sample_steps_note"):
            self.sample_steps_note.configure(
                text=("Klein only — Krea 2 previews are 8-step either way (engine choice below)"
                      if is_krea2 else
                      "Base samples only — Distilled is locked at 4 steps"))

        # Krea 2 preview engine dropdown + note (rows 6-7 of the cadence card) — Krea 2 only.
        if hasattr(self, "krea2_engine_frame"):
            if is_krea2:
                self.krea2_engine_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
                self.krea2_engine_note.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
            else:
                self.krea2_engine_frame.grid_remove()
                self.krea2_engine_note.grid_remove()

        # Reference image — supported by BOTH families now (Klein: edit conditioning; Krea 2:
        # Qwen3-VL vision path). Always enabled; only the note differs. No strength dial on this
        # row in either mode (Krea 2's vision-path reference has no strength; Klein auto-caps).
        if hasattr(self, "sample_ref_entry"):
            self.sample_ref_entry.configure(state="readonly")
        for attr in ("sample_ref_browse_btn", "sample_ref_clear_btn"):
            w = getattr(self, attr, None)
            if w is not None:
                w.configure(state=tk.NORMAL)
        if hasattr(self, "sample_ref_label"):
            self.sample_ref_label.configure(foreground=secondary)
        if hasattr(self, "sample_ref_note"):
            self.sample_ref_note.configure(
                text=("Optional — fed through Krea 2's Qwen3-VL vision path so samples become visually "
                      "aware of it ('prompt from a picture', not a pixel edit). Downscaled to a cap; leave "
                      "empty for normal samples."
                      if is_krea2 else
                      "Optional — Klein is an edit model, so samples can be conditioned on a real image (they edit "
                      "it rather than generate from scratch). Auto-resized to ~0.20 MP so any size is safe. Leave "
                      "empty for normal samples."))

        # "Cache sample model in RAM" dropdown (not implemented for Krea 2)
        if hasattr(self, "cache_sample_model_combo"):
            self.cache_sample_model_combo.configure(state=(tk.DISABLED if is_krea2 else "readonly"))
        if hasattr(self, "cache_sample_model_label"):
            self.cache_sample_model_label.configure(
                text=("Cache sample model in RAM (Klein only):" if is_krea2 else "Cache sample model in RAM:"),
                foreground=label_fg)

    def update_sample_output_label(self):
        """Update the sample output path label to show actual path"""
        samples_dir = self.get_samples_dir()
        self.sample_output_label.config(text=f"Sample Output: {samples_dir}")

    def generate_sample_prompt_file(self):
        """Generate prompt file for sample generation"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        prompt_file = os.path.join(samples_dir, "prompts.txt")

        # Build prompt with options
        prompt = self.sample_prompt_text.get("1.0", tk.END).strip()

        # Add options if not already in prompt
        if "--w" not in prompt:
            prompt += f" --w {self.sample_width_var.get()}"
        if "--h" not in prompt:
            prompt += f" --h {self.sample_height_var.get()}"
        if "--f" not in prompt:
            prompt += " --f 1"  # Always 1 for images
        if "--s" not in prompt:
            prompt += f" --s {self.sample_steps_var.get()}"

        seed = self.sample_seed_var.get()
        if seed and seed != "0" and "--d" not in prompt:
            prompt += f" --d {seed}"

        # Add flow shift if set
        flow_shift = self.sample_flow_shift_var.get()
        if flow_shift and "--fs" not in prompt:
            prompt += f" --fs {flow_shift}"

        # Add negative prompt if set
        negative = self.sample_negative_var.get().strip()
        if negative and "--n" not in prompt:
            prompt += f" --n {negative}"

        # Always add CFG scale (omitting --l causes fallback to 4.0 in Z-Image)
        cfg = self.sample_cfg_scale_var.get()
        if cfg and "--l" not in prompt:
            prompt += f" --l {cfg}"

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-generated by Fizgig LoRA Trainer GUI\n{prompt}\n")

        return prompt_file

    def get_samples_dir(self):
        """Get the samples directory from output dir"""
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "")
        if output_dir:
            return os.path.join(output_dir, "sample")
        # Fallback to local samples folder
        return os.path.join(os.path.dirname(__file__), "output_loras", "sample")

    def find_free_port(self):
        """Find an available port for the HTTP server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_gallery_server(self):
        """Start HTTP server to serve samples directory (avoids CORS issues)"""
        if self.gallery_server is not None:
            return  # Already running

        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        # LoRA checkpoints live in the output dir (parent of sample/); serve them via /loras/.
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "") or os.path.dirname(samples_dir)

        # Find free port
        self.gallery_server_port = self.find_free_port()

        # Create handler that serves images from samples/ and checkpoints from /loras/ (output dir).
        app = self   # for the likeness endpoints (never touch Tk vars from handler threads)

        class SamplesHandler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=samples_dir, **kwargs)

            def translate_path(handler_self, path):
                # /loras/<file> -> the checkpoint in the output dir (basename-only, no traversal).
                # /dataset/<file> -> a training image (for the likeness baseline picker).
                clean = path.split('?', 1)[0].split('#', 1)[0]
                if clean.startswith('/loras/'):
                    import posixpath, urllib.parse
                    fname = posixpath.basename(urllib.parse.unquote(clean[len('/loras/'):]))
                    return os.path.join(output_dir, fname)
                if clean.startswith('/dataset/'):
                    import posixpath, urllib.parse
                    fname = posixpath.basename(urllib.parse.unquote(clean[len('/dataset/'):]))
                    return os.path.join(getattr(app, "_gal_dataset_dir", "") or "", fname)
                return super().translate_path(path)

            def do_POST(handler_self):
                # /set_baselines {"baselines": [3 names]} -> start CPU likeness scoring;
                # empty list clears it. Everything else is a 404.
                clean = handler_self.path.split('?', 1)[0].split('#', 1)[0]
                if clean != '/set_baselines':
                    handler_self.send_error(404)
                    return
                try:
                    ln = int(handler_self.headers.get('Content-Length') or 0)
                    data = json.loads(handler_self.rfile.read(ln) or b'{}')
                    ok, msg = app._gallery_set_baselines(data.get('baselines') or [])
                except Exception as e:
                    ok, msg = False, str(e)
                body = json.dumps({"ok": ok, "msg": msg}).encode("utf-8")
                handler_self.send_response(200 if ok else 400)
                handler_self.send_header('Content-Type', 'application/json')
                handler_self.send_header('Content-Length', str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(handler_self, format, *args):
                pass  # Suppress logging

        try:
            self.gallery_server = HTTPServer(('127.0.0.1', self.gallery_server_port), SamplesHandler)

            # Run server in background thread
            def serve_forever():
                self.gallery_server.serve_forever()

            self.gallery_server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.gallery_server_thread.start()

        except Exception as e:
            print(f"Failed to start gallery server: {e}")
            self.gallery_server = None
            self.gallery_server_port = None

    def stop_gallery_server(self):
        """Stop the HTTP server"""
        if self.gallery_server is not None:
            self.gallery_server.shutdown()
            self.gallery_server = None
            self.gallery_server_port = None

    def open_samples_gallery(self):
        """Open the samples gallery HTML viewer in browser via HTTP server"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        gallery_path = os.path.join(samples_dir, "gallery.html")

        # Snapshot the dataset folder for the likeness picker/scorer — the HTTP handler and
        # the scoring worker run on background threads and must never touch Tk variables.
        self._gal_dataset_dir = (self.image_folder_var.get().strip()
                                 if hasattr(self, "image_folder_var") else "")

        # Opening the gallery claims the samples dir for THIS session — any other running
        # Fizgig instance's watcher/scorer stands down instead of fighting over the sidecars.
        self._gallery_claim(samples_dir)

        # Always regenerate the template so template changes (e.g. the per-epoch download link) are
        # picked up — otherwise a stale gallery.html from an earlier run keeps the old JS forever.
        # The file is purely generated (static template + embedded data filled by update_gallery_html),
        # so overwriting it loses nothing.
        self.create_gallery_html(gallery_path)

        # Generate/update the gallery HTML with current files
        self.update_gallery_html()

        # If a previous session picked likeness baselines, resume scoring automatically.
        self._gallery_resume_likeness()

        # Start HTTP server if not running
        self.start_gallery_server()

        if self.gallery_server_port:
            # Open via HTTP (avoids CORS issues)
            webbrowser.open(f"http://127.0.0.1:{self.gallery_server_port}/gallery.html")
        else:
            # Fallback to file:// if server failed
            webbrowser.open(f"file://{os.path.abspath(gallery_path)}")

    def open_samples_folder(self):
        """Open the samples folder in file explorer"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        self._open_in_file_manager(samples_dir)

    def create_gallery_html(self, gallery_path):
        """Create the gallery HTML file if it doesn't exist"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }
        header { background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }
        header h1 { color: #ECF0F1; font-size: 24px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px; }
        .live-indicator { width: 12px; height: 12px; background-color: #27AE60; border-radius: 50%; animation: pulse 2s infinite; }
        .live-indicator.paused { background-color: #95A5A6; animation: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .controls { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
        .controls label { display: flex; align-items: center; gap: 8px; }
        .controls select { padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }
        .controls button { padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }
        .controls button:hover { background-color: #3498DB; }
        .status { color: #95A5A6; font-size: 14px; }
        main { padding: 20px; }
        #gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .gallery-item { background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .gallery-item:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
        .gallery-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight { 0%, 30% { box-shadow: 0 0 30px #27AE60; } 100% { box-shadow: none; } }
        .image-container { position: relative; }
        /* contain, not cover: a widescreen preview letterboxes instead of losing its sides —
           the grid is for JUDGING samples, and a cropped frame lies about the composition */
        .gallery-item img { width: 100%; height: 280px; object-fit: contain; display: block; background-color: #1B2A38; }
        .badge { position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .epoch-badge { background-color: #27AE60; color: white; }
        .clip-badge { background-color: #8E44AD; color: white; right: 8px; left: auto; }
        /* below the clip badge with clear air — the two overlapped at 34px (Peter) */
        .sound-badge { background-color: #16A085; color: white; right: 8px; left: auto; top: 40px; }
        #lightbox-vid { display: none; max-width: 90vw; max-height: 72vh; border-radius: 4px; }
        #lb-scrub-wrap { display: none; width: min(80vw, 640px); margin-top: 10px; text-align: center; }
        #lb-scrub-wrap.active { display: block; }
        #lb-scrub { width: 100%; }
        #lb-scrub-label { color: #95A5A6; font-size: 12px; margin-top: 2px; }
        .new-badge { position: absolute; top: 10px; right: 10px; background-color: #E74C3C; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .image-info { padding: 12px; }
        .lora-name { color: #9B59B6; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
        .meta-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
        .meta-item { font-size: 13px; color: #BDC3C7; }
        .lora-download { display: inline-block; margin-top: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600;
                         color: #fff; background: #9B59B6; border-radius: 6px; text-decoration: none; }
        .lora-download:hover { background: #8E44AD; }
        .final-lora-btn { padding: 6px 12px; font-size: 13px; font-weight: 600; color: #fff; background: #27AE60;
                          border-radius: 6px; text-decoration: none; }
        .final-lora-btn:hover { background: #219150; }
        .meta-item.seed { color: #3498DB; font-family: monospace; }
        .meta-item.time { color: #95A5A6; }
        .no-images { grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; }
        .no-images h2 { margin-bottom: 15px; color: #ECF0F1; }
        .stats { background-color: #2C3E50; padding: 8px 15px; border-radius: 4px; font-size: 14px; }
        #lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }
        #lightbox.active { display: flex; }
        #lightbox img { max-width: 90%; max-height: 80%; object-fit: contain; }
        #lightbox .close-btn { position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }
        #lightbox .close-btn:hover { color: #E74C3C; }
        #lightbox .nav-btn { position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }
        #lightbox .nav-btn:hover { color: #2980B9; }
        #lightbox .prev-btn { left: 20px; }
        #lightbox .next-btn { right: 20px; }
        #lightbox .image-details { margin-top: 15px; text-align: center; }
        #lightbox .image-name { color: #ECF0F1; font-size: 16px; }
        #lightbox .image-meta { color: #95A5A6; font-size: 14px; margin-top: 5px; }
        .lik-badge { position: absolute; bottom: 10px; left: 10px; padding: 4px 10px; border-radius: 4px;
                     font-weight: bold; font-size: 13px; color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .lik-good { background-color: #27AE60; } .lik-mid { background-color: #E67E22; }
        .lik-bad { background-color: #C0392B; } .lik-na { background-color: #5D6D7E; }
        #likeness-panel { display: none; background-color: #22303F; border-bottom: 1px solid #2C3E50; padding: 12px 20px; }
        #likeness-panel h3 { font-size: 15px; margin-bottom: 8px; color: #ECF0F1; }
        #lik-chart { background-color: #1B2A38; border-radius: 6px; width: 100%; height: 150px; display: block; }
        #likeness-panel .lik-note { color: #95A5A6; font-size: 12px; margin-top: 6px; }
        #basepicker { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                      background-color: rgba(0,0,0,0.93); z-index: 1100; overflow-y: auto; padding: 24px 30px; }
        #basepicker.active { display: block; }
        #bp-bar { position: sticky; top: -24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
                  background-color: rgba(0,0,0,0.93); padding: 12px 0; z-index: 1; }
        #bp-bar h2 { font-size: 20px; margin-right: 8px; }
        #bp-bar button { padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }
        #bp-bar button:disabled { background-color: #5D6D7E; cursor: default; }
        .bp-sub { color: #95A5A6; margin: 6px 0 14px 0; font-size: 13px; }
        #bp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
        .bp-item { border-radius: 6px; overflow: hidden; cursor: pointer; outline: 3px solid transparent; position: relative; }
        .bp-item img { width: 100%; height: 140px; object-fit: cover; display: block; background-color: #1B2A38; }
        .bp-item.selected { outline-color: #27AE60; }
        .bp-item .bp-num { position: absolute; top: 6px; left: 6px; background-color: #27AE60; color: #fff; font-weight: bold;
                           border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; }
        #runviz { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  background-color: rgba(0,0,0,0.95); z-index: 1150; overflow-y: auto; padding: 20px 30px; }
        #runviz.active { display: flex; flex-direction: column; align-items: center; }
        .rv-bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: center; padding: 6px 0; }
        .rv-bar h2 { font-size: 20px; }
        #runviz button, #runviz select { padding: 6px 12px; background-color: #2980B9; color: #ECF0F1;
                                         border: none; border-radius: 4px; cursor: pointer; }
        #runviz select { background-color: #1B2A38; border: 1px solid #2980B9; }
        #runviz label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
        #rv-stage { position: relative; margin: 10px 0; }
        #rv-img { max-width: min(85vw, 900px); max-height: 62vh; display: block; background-color: #1B2A38; border-radius: 6px; }
        #rv-epoch { position: absolute; bottom: 12px; left: 12px; background-color: rgba(0,0,0,0.65); color: #fff;
                    padding: 5px 12px; border-radius: 4px; font-weight: bold; font-size: 15px; }
        #rv-slider { width: min(85vw, 900px); }
        #rv-note { color: #95A5A6; font-size: 12px; margin-top: 10px; max-width: 720px; text-align: center; }
        #rv-status { color: #95A5A6; font-size: 13px; }
    </style>
</head>
<body>
    <header>
        <h1><span class="live-indicator" id="live-dot"></span> Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Show: <select id="run-select">
                <option value="all">All samples</option>
                <option value="current">Current run only</option>
            </select></label>
            <label>Sort: <select id="sort-select">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High-Low)</option>
                <option value="epoch-asc">Epoch (Low-High)</option>
            </select></label>
            <label>Refresh: <select id="refresh-select">
                <option value="3">3 sec</option>
                <option value="5">5 sec</option>
                <option value="10" selected>10 sec</option>
                <option value="30">30 sec</option>
                <option value="0">Off</option>
            </select></label>
            <button onclick="loadImages()">Refresh Now</button>
            <button onclick="openBaselinePicker()">🎯 Likeness scoring…</button>
            <button onclick="openRunViz()">🎞 Training Run Visualiser</button>
            <a id="final-lora-btn" class="final-lora-btn" href="#" download style="display:none">⬇ Download Final LoRA</a>
            <span class="stats" id="stats">0 images</span>
            <span class="status" id="status">Ready</span>
            <span class="status" id="lik-status"></span>
        </div>
    </header>
    <div id="likeness-panel">
        <h3>Likeness vs baselines — <span id="lik-run"></span></h3>
        <canvas id="lik-chart" width="940" height="150"></canvas>
        <div class="lik-note">Average ArcFace similarity of each epoch's samples to your 3 baseline photos
            (scoreable faces only — no-face samples are skipped). Same person typically lands 30–70%.
            This measures identity likeness ONLY: overbake / plastic skin still needs your eyes.</div>
    </div>
    <main>
        <div id="gallery">
            <div class="no-images">
                <h2>Loading...</h2>
            </div>
        </div>
    </main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <video id="lightbox-vid" controls preload="metadata"></video>
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div id="lb-scrub-wrap">
            <input type="range" id="lb-scrub" min="0" max="0" value="0"
                   oninput="lbScrub(parseInt(this.value))">
            <div id="lb-scrub-label"></div>
        </div>
        <div id="lb-audio-wrap" style="display:none; margin-top:8px; text-align:center;">
            <audio id="lb-audio" controls preload="none"
                   style="width:min(80vw,640px);"></audio>
        </div>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <div id="basepicker">
        <div id="bp-bar">
            <h2>🎯 Pick 3 baseline images</h2>
            <button id="bp-start" onclick="submitBaselines()" disabled>Start scoring</button>
            <button onclick="clearBaselines()">Clear scoring</button>
            <button onclick="closeBaselinePicker()">Cancel</button>
            <span class="status" id="bp-status"></span>
        </div>
        <div class="bp-sub">Choose the 3 training images that best nail the look you want — every sample is scored
            against all three and averaged, so no single photo's angle/lighting biases the result.
            Scoring runs on CPU with zero impact on training speed.<br>
            Listing: <span id="bp-folder" style="color:#3498DB">…</span> (the Start-tab training folder,
            snapshotted when the gallery was opened — reopen the gallery after changing it)</div>
        <div id="bp-grid"></div>
    </div>
    <div id="runviz">
        <div class="rv-bar">
            <h2>🎞 Training Run Visualiser</h2>
            <label>Sample slot: <select id="rv-slot" onchange="rvBuild()"></select></label>
            <button id="rv-play" onclick="rvTogglePlay()">▶ Play</button>
            <label>Speed: <select id="rv-speed">
                <option value="600">Slow</option>
                <option value="350" selected>Normal</option>
                <option value="180">Fast</option>
            </select></label>
            <button onclick="closeRunViz()">✕ Close</button>
        </div>
        <div class="rv-bar">
            <label><input type="checkbox" id="rv-pingpong" checked> Loop (ping-pong)</label>
            <label><input type="checkbox" id="rv-ticker" checked> Epoch ticker</label>
            <label><input type="checkbox" id="rv-tag" checked> Fizgig tag</label>
            <button onclick="rvExport()">⬇ Export clip (WebM)</button>
            <button onclick="rvSaveFrame()">⬇ Save frame (PNG)</button>
            <span id="rv-status"></span>
        </div>
        <div id="rv-stage">
            <img id="rv-img" src="" alt="">
            <div id="rv-epoch"></div>
        </div>
        <input type="range" id="rv-slider" min="0" max="0" value="0" oninput="rvShow(parseInt(this.value))">
        <div id="rv-note">Scrubbing this run's epochs, one sample slot at a time. Like it? The <b>LoRA Royale</b> tab
            in Fizgig does much more — checkpoint-vs-checkpoint battles, seed &amp; prompt travel, likeness scoring,
            and full MP4/GIF export with the same ticker and tag options.</div>
    </div>
    <!-- EMBEDDED_FILES_START -->
    <script id="files-data" type="application/json">[]</script>
    <!-- EMBEDDED_FILES_END -->
    <script>
        let images = [];
        let currentLightboxIndex = 0;
        let refreshTimer = null;
        let likeness = null;      // {baselines, status, scores} from likeness.json
        let bpSelected = [];      // baseline picker selection (max 3)

        document.getElementById('sort-select').value = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('refresh-select').value = localStorage.getItem('fizgig-refresh') || '10';
        document.getElementById('run-select').value = localStorage.getItem('fizgig-run') || 'all';

        document.getElementById('sort-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-sort', e.target.value);
            renderGallery();
        });

        document.getElementById('run-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-run', e.target.value);
            renderGallery();
        });

        function currentRunName() {
            // Current run = the LoRA name of the newest sample. Output folders are commonly
            // reused across trains, so old runs' samples share the folder — filter them out
            // by default rather than mixing subjects in one grid.
            let newest = null;
            images.forEach(im => { if (!newest || im.timestamp > newest.timestamp) newest = im; });
            return newest ? newest.loraName : null;
        }

        document.getElementById('refresh-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-refresh', e.target.value);
            setupTimer();
        });

        function setupTimer() {
            if (refreshTimer) clearInterval(refreshTimer);
            const sec = parseInt(document.getElementById('refresh-select').value);
            const dot = document.getElementById('live-dot');
            if (sec > 0) {
                refreshTimer = setInterval(loadImages, sec * 1000);
                dot.classList.remove('paused');
            } else {
                dot.classList.add('paused');
            }
        }

        function parseFilename(filename) {
            // Seed segment is OPTIONAL: samples generated with a random/unspecified seed are named
            // without a trailing _<seed> (the trainer omits it when seed is None). Requiring it made
            // those files fall back to epoch 0 / no timestamp — breaking order + epoch labels.
            const match = filename.match(/^(.+)_e(\\d{6})_(\\d{2})_(\\d{14})(?:_(\\d+))?\\.png$/i);
            if (match) {
                const ts = match[4];
                return {
                    filename,
                    loraName: match[1],
                    epoch: parseInt(match[2]),
                    idx: parseInt(match[3]),
                    timestamp: ts,
                    seed: match[5] || '',
                    time: `${ts.slice(8,10)}:${ts.slice(10,12)}:${ts.slice(12,14)}`
                };
            }
            return { filename, loraName: 'Unknown', epoch: 0, idx: 0, timestamp: '', seed: '', time: '' };
        }

        async function loadImages() {
            document.getElementById('status').textContent = 'Loading...';

            // Try fetch first (works with HTTP server), fallback to embedded
            try {
                const resp = await fetch('files.json?t=' + Date.now());
                if (resp.ok) {
                    const files = await resp.json();
                    images = files.map(f => parseFilename(f));
                    // Attach a per-epoch LoRA download link where a checkpoint exists (loras.json
                    // is an epoch -> filename map written by the trainer; served via /loras/).
                    try {
                        const lr = await fetch('loras.json?t=' + Date.now());
                        if (lr.ok) {
                            const lm = await lr.json();
                            images.forEach(im => { const ck = lm[String(im.epoch)]; if (ck) im.lora = 'loras/' + encodeURIComponent(ck); });
                            // Final LoRA ({name}.safetensors) -> header button (reserved "final" key).
                            const fb = document.getElementById('final-lora-btn');
                            if (fb) {
                                if (lm.final) { fb.href = 'loras/' + encodeURIComponent(lm.final); fb.style.display = 'inline-block'; }
                                else { fb.style.display = 'none'; }
                            }
                        }
                    } catch (e) {}
                    // Clip scrub data (MiniMax clip previews): filename -> frame list.
                    try {
                        const cj = await fetch('clips.json?t=' + Date.now());
                        if (cj.ok) { const cm = await cj.json(); images.forEach(im => { if (cm[im.filename]) im.clip = cm[im.filename]; }); }
                    } catch (e) {}
                    // Sample sound (previews with audio): filename -> wav. Never autoplays.
                    try {
                        const sj = await fetch('sounds.json?t=' + Date.now());
                        if (sj.ok) { const sm = await sj.json(); images.forEach(im => { if (sm[im.filename]) im.sound = sm[im.filename]; }); }
                    } catch (e) {}
                    // Playable clips (frames + sound muxed): filename -> mp4. Never autoplays.
                    try {
                        const vj = await fetch('videos.json?t=' + Date.now());
                        if (vj.ok) { const vm = await vj.json(); images.forEach(im => { if (vm[im.filename]) im.video = vm[im.filename]; }); }
                    } catch (e) {}
                    await loadLikeness();
                    renderGallery();
                    renderLikenessChart();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
                    return;
                }
            } catch (e) {
                // Fetch failed, try embedded data
            }

            // Fallback to embedded data
            const filesData = document.getElementById('files-data');
            if (filesData) {
                try {
                    const files = JSON.parse(filesData.textContent);
                    images = files.map(f => parseFilename(f));
                    renderGallery();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Loaded: ${new Date().toLocaleTimeString()}`;
                } catch (e) {
                    document.getElementById('status').textContent = 'Error loading files';
                }
            }
        }

        function renderGallery() {
            const gallery = document.getElementById('gallery');
            const sortBy = document.getElementById('sort-select').value;

            if (images.length === 0) {
                gallery.innerHTML = `<div class="no-images"><h2>No samples yet</h2><p>Start training with sample generation enabled.</p></div>`;
                return;
            }

            // Optional opt-in filter — default shows everything (output folders are often
            // shared across runs, and comparing runs side by side is a feature).
            let sorted = [...images];
            if (document.getElementById('run-select').value === 'current') {
                const run = currentRunName();
                if (run) sorted = sorted.filter(im => im.loraName === run);
            }
            switch (sortBy) {
                case 'newest': sorted.sort((a, b) => b.timestamp.localeCompare(a.timestamp)); break;
                case 'oldest': sorted.sort((a, b) => a.timestamp.localeCompare(b.timestamp)); break;
                case 'epoch-desc': sorted.sort((a, b) => b.epoch - a.epoch || b.timestamp.localeCompare(a.timestamp)); break;
                case 'epoch-asc': sorted.sort((a, b) => a.epoch - b.epoch || a.timestamp.localeCompare(b.timestamp)); break;
            }

            gallery.innerHTML = sorted.map(img => `
                <div class="gallery-item" onclick="openLightbox('${img.filename}')">
                    <div class="image-container">
                        <img src="${img.filename}" alt="${img.filename}" loading="lazy">
                        <span class="badge epoch-badge">Epoch ${img.epoch}</span>
                        ${likBadge(img)}
                        ${img.video ? `<span class="badge clip-badge">🎬 video</span>` : ''}
                        ${!img.video && img.clip ? `<span class="badge clip-badge">🎞 scrub</span>` : ''}
                        ${!img.video && img.sound ? `<span class="badge sound-badge">🔊 sound</span>` : ''}
                    </div>
                    <div class="image-info">
                        <div class="lora-name">${img.loraName}</div>
                        <div class="meta-row">
                            <span class="meta-item seed">Seed: ${img.seed || '—'}</span>
                            <span class="meta-item time">${img.time}</span>
                        </div>
                        ${img.lora ? `<a class="lora-download" href="${img.lora}" download onclick="event.stopPropagation()">⬇ Download LoRA (epoch ${img.epoch})</a>` : ''}
                    </div>
                </div>`).join('');
        }

        // ---------- Likeness scoring (CPU ArcFace vs 3 baselines, scored by Fizgig) ----------

        async function loadLikeness() {
            try {
                const r = await fetch('likeness.json?t=' + Date.now());
                if (r.ok) likeness = await r.json();
            } catch (e) {}
            const active = likeness && likeness.baselines && likeness.baselines.length === 3;
            document.getElementById('lik-status').textContent = active ? ('🎯 ' + (likeness.status || '')) : '';
        }

        function likBadge(img) {
            if (!likeness || !likeness.baselines || likeness.baselines.length !== 3) return '';
            const s = likeness.scores ? likeness.scores[img.filename] : undefined;
            if (s === undefined) return `<span class="lik-badge lik-na">…</span>`;
            if (s === null) return `<span class="lik-badge lik-na">no face</span>`;
            const cls = s >= 0.45 ? 'lik-good' : (s >= 0.30 ? 'lik-mid' : 'lik-bad');
            return `<span class="lik-badge ${cls}">${Math.round(s * 100)}%</span>`;
        }

        function renderLikenessChart() {
            const panel = document.getElementById('likeness-panel');
            const active = likeness && likeness.baselines && likeness.baselines.length === 3 && likeness.scores;
            if (!active || images.length === 0) { panel.style.display = 'none'; return; }
            // Current run = the LoRA name of the newest sample (old runs' samples share the
            // folder but must not pollute the trend).
            let newest = null;
            images.forEach(im => { if (!newest || im.timestamp > newest.timestamp) newest = im; });
            const byEpoch = {};
            images.forEach(im => {
                if (im.loraName !== newest.loraName) return;
                const s = likeness.scores[im.filename];
                if (typeof s === 'number') (byEpoch[im.epoch] = byEpoch[im.epoch] || []).push(s);
            });
            const epochs = Object.keys(byEpoch).map(Number).sort((a, b) => a - b);
            if (!epochs.length) { panel.style.display = 'none'; return; }
            const avgs = epochs.map(e => byEpoch[e].reduce((a, b) => a + b, 0) / byEpoch[e].length);
            let bestI = 0;
            avgs.forEach((v, i) => { if (v > avgs[bestI]) bestI = i; });
            document.getElementById('lik-run').textContent =
                `${newest.loraName} — best so far: epoch ${epochs[bestI]} (${Math.round(avgs[bestI] * 100)}%)`;
            panel.style.display = 'block';
            const cv = document.getElementById('lik-chart');
            // Size the backing store to the rendered width so the chart spans the page like the
            // thumbnail grid does (a fixed-width canvas stretched by CSS goes blurry instead).
            const cssW = cv.clientWidth || 940;
            if (cv.width !== cssW) cv.width = cssW;
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height, padL = 42, padR = 12, padT = 12, padB = 22;
            ctx.clearRect(0, 0, W, H);
            const ymax = Math.max(0.7, Math.max(...avgs) + 0.05);
            const x = i => epochs.length === 1 ? (padL + (W - padL - padR) / 2)
                                               : padL + (W - padL - padR) * i / (epochs.length - 1);
            const y = v => padT + (H - padT - padB) * (1 - v / ymax);
            ctx.font = '11px Segoe UI';
            ctx.lineWidth = 1;
            [0.30, 0.45].forEach(g => {   // the badge colour bands, for orientation
                ctx.strokeStyle = '#34495E'; ctx.fillStyle = '#7F8C8D';
                ctx.beginPath(); ctx.moveTo(padL, y(g)); ctx.lineTo(W - padR, y(g)); ctx.stroke();
                ctx.fillText(Math.round(g * 100) + '%', 8, y(g) + 4);
            });
            ctx.strokeStyle = '#3498DB'; ctx.lineWidth = 2; ctx.beginPath();
            avgs.forEach((v, i) => { i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)); });
            ctx.stroke();
            const labelEvery = Math.max(1, Math.ceil(epochs.length / 40));
            avgs.forEach((v, i) => {
                ctx.fillStyle = i === bestI ? '#27AE60' : '#3498DB';
                ctx.beginPath(); ctx.arc(x(i), y(v), i === bestI ? 5 : 3.5, 0, Math.PI * 2); ctx.fill();
                if (i % labelEvery === 0 || i === bestI) {
                    ctx.fillStyle = '#95A5A6';
                    ctx.fillText(epochs[i], x(i) - 6, H - 6);
                }
            });
        }

        async function openBaselinePicker() {
            const bp = document.getElementById('basepicker');
            bp.classList.add('active');
            document.body.style.overflow = 'hidden';
            const grid = document.getElementById('bp-grid');
            grid.innerHTML = '<div style="color:#95A5A6">Loading dataset…</div>';
            let names = [];
            try {
                const r = await fetch('dataset.json?t=' + Date.now());
                if (r.ok) {
                    const d = await r.json();
                    names = Array.isArray(d) ? d : (d.images || []);
                    document.getElementById('bp-folder').textContent = (d && d.folder) || 'unknown';
                }
            } catch (e) {}
            if (!names.length) {
                grid.innerHTML = '<div style="color:#E74C3C">No dataset images found — set the training ' +
                                 'image folder on the Start tab, then reopen the gallery from Fizgig.</div>';
                return;
            }
            bpSelected = (likeness && likeness.baselines && likeness.baselines.length === 3)
                         ? [...likeness.baselines] : [];
            grid.innerHTML = names.map(n => `
                <div class="bp-item" data-name="${n}" onclick="toggleBaseline(this)">
                    <img src="dataset/${encodeURIComponent(n)}" loading="lazy">
                </div>`).join('');
            refreshBpMarks();
        }

        function toggleBaseline(el) {
            const n = el.dataset.name;
            const i = bpSelected.indexOf(n);
            if (i >= 0) bpSelected.splice(i, 1);
            else { if (bpSelected.length >= 3) bpSelected.shift(); bpSelected.push(n); }
            refreshBpMarks();
        }

        function refreshBpMarks() {
            document.querySelectorAll('.bp-item').forEach(el => {
                const i = bpSelected.indexOf(el.dataset.name);
                el.classList.toggle('selected', i >= 0);
                let num = el.querySelector('.bp-num');
                if (i >= 0) {
                    if (!num) { num = document.createElement('div'); num.className = 'bp-num'; el.appendChild(num); }
                    num.textContent = i + 1;
                } else if (num) num.remove();
            });
            document.getElementById('bp-start').disabled = bpSelected.length !== 3;
            document.getElementById('bp-status').textContent = `${bpSelected.length}/3 selected`;
        }

        async function submitBaselines() {
            await postBaselines(bpSelected, true);
        }

        async function clearBaselines() {
            await postBaselines([], true);
            likeness = null;
            renderGallery();
            renderLikenessChart();
            document.getElementById('lik-status').textContent = '';
        }

        async function postBaselines(names, closeOnOk) {
            const st = document.getElementById('bp-status');
            st.textContent = 'Sending…';
            try {
                const r = await fetch('set_baselines', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ baselines: names })
                });
                const res = await r.json();
                if (res.ok) { if (closeOnOk) closeBaselinePicker(); loadImages(); }
                else st.textContent = '⚠ ' + res.msg;
            } catch (e) {
                st.textContent = '⚠ Fizgig not reachable — reopen the gallery from the app.';
            }
        }

        function closeBaselinePicker() {
            document.getElementById('basepicker').classList.remove('active');
            document.body.style.overflow = '';
        }

        // ---------- Training Run Visualiser (epoch carousel, Royale-style) ----------

        let rvFrames = [];      // [{epoch, filename, sim}] ascending epochs, one sample slot
        let rvIdx = 0;
        let rvTimer = null;
        let rvDir = 1;          // ping-pong direction while playing

        function rvCurrentRunImages() {
            const run = currentRunName();
            return run ? images.filter(im => im.loraName === run) : [];
        }

        function openRunViz() {
            const runImgs = rvCurrentRunImages();
            if (!runImgs.length) { alert('No samples yet — start a run with previews enabled.'); return; }
            const slots = [...new Set(runImgs.map(im => im.idx))].sort((a, b) => a - b);
            const sel = document.getElementById('rv-slot');
            const keep = sel.value;
            sel.innerHTML = slots.map(s => `<option value="${s}">${s}</option>`).join('');
            if (slots.map(String).includes(keep)) sel.value = keep;
            document.getElementById('runviz').classList.add('active');
            document.body.style.overflow = 'hidden';
            rvBuild();
        }

        function rvBuild() {
            const slot = parseInt(document.getElementById('rv-slot').value || '0');
            const byEpoch = {};
            rvCurrentRunImages().forEach(im => {
                if (im.idx !== slot) return;
                // Same epoch rendered twice (e.g. after a resume) -> keep the newest.
                if (!byEpoch[im.epoch] || im.timestamp > byEpoch[im.epoch].timestamp) byEpoch[im.epoch] = im;
            });
            rvFrames = Object.keys(byEpoch).map(Number).sort((a, b) => a - b).map(e => ({
                epoch: e,
                filename: byEpoch[e].filename,
                sim: (likeness && likeness.scores) ? likeness.scores[byEpoch[e].filename] : undefined,
            }));
            const slider = document.getElementById('rv-slider');
            slider.max = Math.max(0, rvFrames.length - 1);
            rvShow(rvFrames.length - 1);   // land on the newest epoch
        }

        function rvShow(i) {
            if (!rvFrames.length) return;
            rvIdx = Math.max(0, Math.min(i, rvFrames.length - 1));
            const fr = rvFrames[rvIdx];
            document.getElementById('rv-img').src = fr.filename;
            document.getElementById('rv-slider').value = rvIdx;
            let label = `Epoch ${fr.epoch}`;
            if (typeof fr.sim === 'number') label += `  ·  likeness ${Math.round(fr.sim * 100)}%`;
            document.getElementById('rv-epoch').textContent = label;
        }

        function rvTogglePlay() {
            if (rvTimer) { rvStop(); return; }
            if (rvFrames.length < 2) return;
            rvDir = 1;
            document.getElementById('rv-play').textContent = '⏸ Pause';
            const tick = () => {
                let next = rvIdx + rvDir;
                if (document.getElementById('rv-pingpong').checked) {
                    if (next >= rvFrames.length || next < 0) { rvDir = -rvDir; next = rvIdx + rvDir; }
                } else if (next >= rvFrames.length) next = 0;
                rvShow(next);
                rvTimer = setTimeout(tick, parseInt(document.getElementById('rv-speed').value));
            };
            rvTimer = setTimeout(tick, parseInt(document.getElementById('rv-speed').value));
        }

        function rvStop() {
            if (rvTimer) clearTimeout(rvTimer);
            rvTimer = null;
            document.getElementById('rv-play').textContent = '▶ Play';
        }

        function closeRunViz() {
            rvStop();
            document.getElementById('runviz').classList.remove('active');
            document.body.style.overflow = '';
        }

        function rvDrawFrame(ctx, imgEl, fr, W, H) {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, W, H);
            const sc = Math.min(W / imgEl.naturalWidth, H / imgEl.naturalHeight);
            const dw = imgEl.naturalWidth * sc, dh = imgEl.naturalHeight * sc;
            ctx.drawImage(imgEl, (W - dw) / 2, (H - dh) / 2, dw, dh);
            if (document.getElementById('rv-ticker').checked) {
                const label = 'Epoch ' + fr.epoch;
                ctx.font = 'bold 26px Segoe UI';
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0,0,0,0.65)';
                ctx.fillRect(16, H - 56, tw + 24, 40);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, 28, H - 28);
            }
            if (document.getElementById('rv-tag').checked) {
                // Scale with the frame — a fixed small px size vanished next to the epoch ticker
                // (and shrank to nothing on full-resolution saved frames).
                const fs = Math.max(28, Math.round(H * 0.055));
                ctx.font = `bold ${fs}px Segoe UI`;
                const tag = 'Fizgig';
                const tw = ctx.measureText(tag).width;
                ctx.fillStyle = 'rgba(255,255,255,0.7)';
                ctx.fillText(tag, W - tw - 20, H - 20);
            }
        }

        function rvPreload() {
            return Promise.all(rvFrames.map(fr => new Promise(res => {
                const im = new Image();
                im.onload = () => res({ fr, im });
                im.onerror = () => res(null);
                im.src = fr.filename;
            }))).then(list => list.filter(Boolean));
        }

        async function rvExport() {
            if (rvFrames.length < 2) { alert('Need at least 2 epochs to export a clip.'); return; }
            const st = document.getElementById('rv-status');
            st.textContent = 'Preparing frames…';
            rvStop();
            const loaded = await rvPreload();
            if (loaded.length < 2) { st.textContent = 'Could not load frames.'; return; }
            let seq = [...loaded];
            if (document.getElementById('rv-pingpong').checked) {
                seq = seq.concat([...loaded].reverse().slice(1, -1));
            }
            const first = loaded[0].im;
            const W = Math.min(1024, first.naturalWidth), H = Math.round(W * first.naturalHeight / first.naturalWidth);
            const cv = document.createElement('canvas');
            cv.width = W; cv.height = H;
            const ctx = cv.getContext('2d');
            const stream = cv.captureStream(30);
            let mime = 'video/webm;codecs=vp9';
            if (!MediaRecorder.isTypeSupported(mime)) mime = 'video/webm';
            const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
            const chunks = [];
            rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
            const doneRec = new Promise(res => { rec.onstop = res; });
            const stepMs = parseInt(document.getElementById('rv-speed').value);
            rec.start();
            for (let i = 0; i < seq.length; i++) {
                rvDrawFrame(ctx, seq[i].im, seq[i].fr, W, H);
                st.textContent = `Recording… ${i + 1}/${seq.length}`;
                await new Promise(r => setTimeout(r, stepMs));
            }
            await new Promise(r => setTimeout(r, 200));   // tail so the last frame lands
            rec.stop();
            await doneRec;
            const blob = new Blob(chunks, { type: 'video/webm' });
            const a = document.createElement('a');
            const run = currentRunName() || 'run';
            a.href = URL.createObjectURL(blob);
            a.download = `${run}_training_run.webm`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 5000);
            st.textContent = `Saved ${a.download} (${seq.length} frames). MP4/GIF export lives in LoRA Royale.`;
        }

        function rvSaveFrame() {
            if (!rvFrames.length) return;
            const fr = rvFrames[rvIdx];
            const im = new Image();
            im.onload = () => {
                const cv = document.createElement('canvas');
                cv.width = im.naturalWidth; cv.height = im.naturalHeight;
                rvDrawFrame(cv.getContext('2d'), im, fr, cv.width, cv.height);
                cv.toBlob(blob => {
                    const a = document.createElement('a');
                    const run = currentRunName() || 'run';
                    a.href = URL.createObjectURL(blob);
                    a.download = `${run}_epoch${fr.epoch}.png`;
                    a.click();
                    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
                }, 'image/png');
            };
            im.src = fr.filename;
        }

        // ---------- Lightbox ----------

        function openLightbox(filename) {
            const idx = images.findIndex(img => img.filename === filename);
            if (idx >= 0) { currentLightboxIndex = idx; showLightbox(images[idx]); }
        }

        let lbClip = null;              // active clip frame list, or null for a plain still

        function lbScrub(i) {
            if (!lbClip) return;
            i = Math.max(0, Math.min(lbClip.length - 1, i));
            document.getElementById('lightbox-img').src = lbClip[i];
            document.getElementById('lb-scrub-label').textContent =
                `frame ${i + 1} / ${lbClip.length} — drag to scrub (clips never autoplay)`;
        }

        function showLightbox(img) {
            const wrap = document.getElementById('lb-scrub-wrap');
            const slider = document.getElementById('lb-scrub');
            const aw = document.getElementById('lb-audio-wrap');
            const au = document.getElementById('lb-audio');
            const vid = document.getElementById('lightbox-vid');
            const imEl = document.getElementById('lightbox-img');
            au.pause();
            vid.pause();
            // A sample with a muxed mp4 plays as a REAL clip — controls, never autoplay —
            // replacing both the scrub slider and the separate audio player.
            if (img.video) {
                vid.src = img.video;
                vid.style.display = 'block';
                imEl.style.display = 'none';
                aw.style.display = 'none';
                au.removeAttribute('src');
                wrap.classList.remove('active');
                lbClip = null;
                document.getElementById('lightbox-name').textContent = img.filename;
                document.getElementById('lightbox-meta').textContent = `${img.loraName} | Epoch ${img.epoch} | Seed: ${img.seed} | ${img.time}`;
                document.getElementById('lightbox').classList.add('active');
                document.body.style.overflow = 'hidden';
                return;
            }
            vid.removeAttribute('src');
            vid.style.display = 'none';
            imEl.style.display = '';
            // The sample's generated sound, when it has one (wav without an mp4 — e.g. the
            // mux failed). A play CONTROL, never autoplay — scrubbing stays silent.
            if (img.sound) { au.src = img.sound; aw.style.display = 'block'; }
            else { au.removeAttribute('src'); aw.style.display = 'none'; }
            lbClip = img.clip || null;
            if (lbClip) {
                // Preload on OPEN, not up front — a 60-epoch gallery would otherwise pull
                // thousands of frames nobody asked for.
                lbClip.forEach(f => { const im = new Image(); im.src = f; });
                slider.max = lbClip.length - 1;
                const mid = Math.floor(lbClip.length / 2);
                slider.value = mid;
                wrap.classList.add('active');
                lbScrub(mid);
            } else {
                wrap.classList.remove('active');
                document.getElementById('lightbox-img').src = img.filename;
            }
            document.getElementById('lightbox-name').textContent = img.filename;
            document.getElementById('lightbox-meta').textContent = `${img.loraName} | Epoch ${img.epoch} | Seed: ${img.seed} | ${img.time}`;
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            document.getElementById('lb-audio').pause();
            document.getElementById('lightbox-vid').pause();
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }

        function navigateLightbox(dir) {
            if (images.length === 0) return;
            currentLightboxIndex = (currentLightboxIndex + dir + images.length) % images.length;
            showLightbox(images[currentLightboxIndex]);
        }

        document.addEventListener('keydown', (e) => {
            // Arrows on the clip scrub slider step the slider natively; without this guard the
            // same keydown bubbles here and ALSO jumps to the next image — two actions per key.
            if (e.target && e.target.id === 'lb-scrub') return;
            if (document.getElementById('runviz').classList.contains('active')) {
                if (e.key === 'Escape') closeRunViz();
                if (e.key === 'ArrowLeft') { rvStop(); rvShow(rvIdx - 1); }
                if (e.key === 'ArrowRight') { rvStop(); rvShow(rvIdx + 1); }
                if (e.key === ' ') { e.preventDefault(); rvTogglePlay(); }
                return;
            }
            if (e.key === 'Escape' && document.getElementById('basepicker').classList.contains('active')) {
                closeBaselinePicker();
                return;
            }
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        });

        document.getElementById('lightbox').addEventListener('click', (e) => {
            if (e.target.id === 'lightbox') closeLightbox();
        });

        let chartResizeTimer = null;
        window.addEventListener('resize', () => {
            if (chartResizeTimer) clearTimeout(chartResizeTimer);
            chartResizeTimer = setTimeout(renderLikenessChart, 150);
        });

        setupTimer();
        loadImages();
    </script>
</body>
</html>'''

        with open(gallery_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def update_gallery_html(self):
        """Update gallery HTML and files.json for live updates"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        if not self._gallery_owns(samples_dir):
            return   # another Fizgig session owns this gallery now — don't fight over sidecars

        # Find all images
        images = []
        if os.path.exists(samples_dir):
            for f in os.listdir(samples_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    images.append(f)

        # Sort alphabetically (which sorts by epoch due to naming convention)
        images.sort()

        # Write files.json for HTTP fetch (live updates)
        files_json_path = os.path.join(samples_dir, "files.json")
        try:
            with open(files_json_path, 'w', encoding='utf-8') as f:
                json.dump(images, f)
        except Exception:
            pass

        # Clip frames for the gallery scrubber: <stem>.clip/ dirs written by MiniMax clip
        # previews, mapped against their contract PNG. Additive — a gallery with no clips gets
        # an empty map and behaves exactly as before.
        try:
            clips = {}
            for f in os.listdir(samples_dir):
                if f.endswith(".clip") and os.path.isdir(os.path.join(samples_dir, f)):
                    frames = sorted(x for x in os.listdir(os.path.join(samples_dir, f))
                                    if x.lower().endswith((".jpg", ".jpeg", ".png")))
                    if frames:
                        clips[f[:-len(".clip")] + ".png"] = [f + "/" + x for x in frames]
            with open(os.path.join(samples_dir, "clips.json"), 'w', encoding='utf-8') as f:
                json.dump(clips, f)
            # Sample sound (previews with audio): <stem>.wav beside the contract PNG. Its own
            # map so clips.json keeps its shape for older galleries.
            sounds = {f[:-4] + ".png": f for f in os.listdir(samples_dir)
                      if f.lower().endswith(".wav")}
            with open(os.path.join(samples_dir, "sounds.json"), 'w', encoding='utf-8') as f:
                json.dump(sounds, f)
            # Playable clips (frames + sound muxed): <stem>.mp4 — the lightbox plays these
            # instead of the scrub slider + separate audio player.
            videos = {f[:-4] + ".png": f for f in os.listdir(samples_dir)
                      if f.lower().endswith(".mp4")}
            with open(os.path.join(samples_dir, "videos.json"), 'w', encoding='utf-8') as f:
                json.dump(videos, f)
        except Exception:
            pass

        # Map epoch -> checkpoint filename so each gallery entry can offer a "Download LoRA" link.
        # Computed here (Python knows the exact {name}-{epoch:06d}.safetensors naming) rather than
        # guessed in JS. Only epochs with a saved checkpoint appear — matches save_every_n_epochs.
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "") or os.path.dirname(samples_dir)
        lora_map = {}
        try:
            if output_dir and os.path.isdir(output_dir):
                import re as _re_ck
                # Key on THIS run's name, not epoch alone: in a reused output folder
                # (which the gallery supports) the last file listdir returned used to win,
                # serving another run's checkpoint from the Download button.
                _run_name = str(self.settings.get("LORA_NAME", "") or "").strip()
                if _run_name:
                    _ck_pat = _re_ck.compile(
                        r'^' + _re_ck.escape(_run_name) + r'-(\d{6})\.safetensors$')
                else:
                    _ck_pat = _re_ck.compile(r'-(\d{6})\.safetensors$')
                for f in os.listdir(output_dir):
                    if f.endswith(".safetensors"):
                        m = _ck_pat.search(f)
                        if m:
                            lora_map[str(int(m.group(1)))] = f
                # The final LoRA is {LORA_NAME}.safetensors (no epoch suffix) — surface it as a
                # header button under the reserved "final" key (epoch keys are numeric, so no clash).
                final_name = str(self.settings.get("LORA_NAME", "") or "").strip()
                if final_name and os.path.exists(os.path.join(output_dir, final_name + ".safetensors")):
                    lora_map["final"] = final_name + ".safetensors"
            with open(os.path.join(samples_dir, "loras.json"), 'w', encoding='utf-8') as f:
                json.dump(lora_map, f)
        except Exception:
            pass

        # Dataset image list for the likeness baseline picker (images served via /dataset/).
        # The folder path is included so the picker can SHOW which folder it's listing —
        # "why is that image here?" is answered by looking at the folder, not guessing.
        try:
            dfolder = getattr(self, "_gal_dataset_dir", "") or ""
            dimgs = []
            if dfolder and os.path.isdir(dfolder):
                dimgs = sorted(f for f in os.listdir(dfolder)
                               if os.path.splitext(f)[1].lower() in self._FF_EXTS)
            with open(os.path.join(samples_dir, "dataset.json"), 'w', encoding='utf-8') as f:
                json.dump({"folder": dfolder, "images": dimgs}, f)
        except Exception:
            pass

        # Also update embedded data in gallery.html (for fallback)
        gallery_path = os.path.join(samples_dir, "gallery.html")
        if os.path.exists(gallery_path):
            try:
                with open(gallery_path, 'r', encoding='utf-8') as f:
                    html = f.read()

                # Find and replace the embedded JSON. The replacement goes through a
                # FUNCTION, never the template parser: json.dumps escapes non-ASCII as
                # \uXXXX (one Chinese character in a sample filename), which re's template
                # parser rejects as "bad escape \u" — swallowed below, so the fallback
                # data silently stopped updating.
                import re
                new_json = json.dumps(images)
                pattern = r'(<script id="files-data" type="application/json">).*?(</script>)'
                new_html = re.sub(pattern, lambda m: m.group(1) + new_json + m.group(2),
                                  html, flags=re.DOTALL)

                if new_html != html:
                    with open(gallery_path, 'w', encoding='utf-8') as f:
                        f.write(new_html)
            except Exception:
                pass  # Don't fail if gallery update fails

    # region Gallery likeness scoring (CPU ArcFace vs 3 baselines)

    def _gallery_claim(self, samples_dir):
        """Claim gallery-sidecar ownership of this samples dir for THIS app session. Two Fizgig
        instances pointed at one output folder otherwise fight over likeness.json / dataset.json
        every 5s — caught live 2026-07-10: a stale instance's scorer (different baselines)
        alternated everyone's likeness scores down to ~5% each time a new sample landed."""
        if not getattr(self, "_gal_session_token", None):
            self._gal_session_token = f"{os.getpid()}-{int(time.time() * 1000)}"
        path = os.path.join(samples_dir, "gallery.owner")
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"token": self._gal_session_token}, f)
            os.replace(path + ".tmp", path)
        except Exception:
            pass

    def _gallery_owns(self, samples_dir):
        """True while this session still owns the samples dir (an unclaimed dir is claimed).
        The most recent session to open the gallery (or start scoring) wins; older sessions'
        watchers and scorers stand down the moment they see a foreign token."""
        path = os.path.join(samples_dir, "gallery.owner")
        try:
            with open(path, encoding="utf-8") as f:
                tok = json.load(f).get("token")
        except Exception:
            self._gallery_claim(samples_dir)
            return True
        return tok == getattr(self, "_gal_session_token", None)

    def _gallery_set_baselines(self, names):
        """Start (or clear, with an empty list) likeness scoring of the sample gallery.
        Called from the gallery HTTP server thread — plain attributes only, no Tk. Returns
        (ok, message) for the JSON response."""
        self._gal_gen = getattr(self, "_gal_gen", 0) + 1   # invalidates any running worker
        samples_dir = self.get_samples_dir()
        if not names:
            self._gal_baselines = []
            self._gallery_write_likeness(samples_dir, [], "cleared", {})
            return True, "cleared"
        if not FACE_DETECTION_AVAILABLE or FaceEmbedder is None:
            return False, "Face tools not installed — run install_fizgig.py."
        folder = getattr(self, "_gal_dataset_dir", "") or ""
        if not folder or not os.path.isdir(folder):
            return False, "Training image folder not set — set it on the Start tab, then reopen the gallery."
        if len(names) != 3:
            return False, f"Pick exactly 3 baseline images (got {len(names)})."
        paths = [os.path.join(folder, os.path.basename(n)) for n in names]
        missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
        if missing:
            return False, "Not in the dataset folder: " + ", ".join(missing)
        self._gal_baselines = paths
        self._gallery_claim(samples_dir)   # starting scoring (re)claims the dir for this session
        threading.Thread(target=self._gallery_likeness_worker,
                         args=(self._gal_gen, paths, samples_dir), daemon=True).start()
        return True, "scoring started"

    def _gallery_resume_likeness(self):
        """Resume scoring after a GUI restart — likeness.json persists the chosen baselines,
        so reopening the gallery picks up where the last session left off."""
        if getattr(self, "_gal_baselines", None):
            return   # already active this session
        try:
            with open(os.path.join(self.get_samples_dir(), "likeness.json"), encoding="utf-8") as f:
                names = json.load(f).get("baselines") or []
        except Exception:
            return
        if len(names) == 3:
            self._gallery_set_baselines(names)

    @staticmethod
    def _gallery_write_likeness(samples_dir, base_names, status, scores):
        payload = {"baselines": base_names, "status": status, "scores": scores}
        path = os.path.join(samples_dir, "likeness.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)   # atomic — the gallery polls this file
        except Exception:
            pass

    def _gallery_likeness_worker(self, gen, baselines, samples_dir):
        """Score every sample image's face against the 3 baselines (averaged) on CPU — zero
        GPU contention with training. Newest samples first, so the CURRENT run's epochs score
        immediately even when the folder holds hundreds of old samples; then keeps watching
        for new files as training produces them."""
        import numpy as np
        base_names = [os.path.basename(b) for b in baselines]
        scores = {}
        try:   # scores for the same baselines survive GUI restarts — no pointless rescoring
            with open(os.path.join(samples_dir, "likeness.json"), encoding="utf-8") as f:
                old = json.load(f)
            # Order-insensitive: the picker records baselines in CLICK order, and re-picking
            # the same 3 in a different order must not throw the whole score set away.
            if (sorted(old.get("baselines") or []) == sorted(base_names)
                    and isinstance(old.get("scores"), dict)):
                scores = old["scores"]
        except Exception:
            pass
        self._gallery_write_likeness(samples_dir, base_names, "loading face model…", scores)
        base_embs = [self._ff_embed_cached(b) for b in baselines]
        missing = [n for n, e in zip(base_names, base_embs) if e is None]
        if missing:
            self._gallery_write_likeness(samples_dir, base_names,
                                         "error: no face found in " + ", ".join(missing), scores)
            return
        last_status = None
        scored_mtimes = {}   # filename -> mtime at scoring time (rescore no-face on change)
        while gen == getattr(self, "_gal_gen", 0):
            if not self._gallery_owns(samples_dir):
                print("[likeness] another Fizgig session took over this samples folder — "
                      "this scorer is standing down.")
                return
            try:
                files = [f for f in os.listdir(samples_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            except OSError:
                files = []
            scores = {k: v for k, v in scores.items() if k in set(files)}   # drop deleted samples

            def _mt(f):
                try:
                    return os.path.getmtime(os.path.join(samples_dir, f))
                except OSError:
                    return 0.0
            # Settle guard: a sample the trainer is STILL WRITING decodes as a truncated image
            # and scores a spurious "no face" that then sticks. Skip anything modified in the
            # last few seconds — the next pass (~4 s) picks it up complete. A null that slipped
            # through anyway (e.g. scored by an older Fizgig) rescores when its mtime moves.
            now = time.time()
            todo = [f for f in files
                    if (now - _mt(f)) > 5.0
                    and (f not in scores
                         or (scores[f] is None and scored_mtimes.get(f) != _mt(f)))]
            todo.sort(key=_mt, reverse=True)   # current run scores first
            for i, f in enumerate(todo, 1):
                if gen != getattr(self, "_gal_gen", 0):
                    return
                scored_mtimes[f] = _mt(f)
                emb = self._ff_embed_cached(os.path.join(samples_dir, f))
                scores[f] = None if emb is None else round(
                    float(np.mean([float(np.dot(be, emb)) for be in base_embs])), 4)
                if i % 2 == 0 or i == len(todo):
                    self._gallery_write_likeness(samples_dir, base_names,
                                                 f"scoring… {len(scores)}/{len(files)}", scores)
            status = f"live — {len(scores)} sample(s) scored"
            if todo or status != last_status:
                self._gallery_write_likeness(samples_dir, base_names, status, scores)
                last_status = status
            for _ in range(16):   # ~4 s between folder checks, responsive to clear/re-pick
                if gen != getattr(self, "_gal_gen", 0):
                    return
                time.sleep(0.25)

    # endregion

    def start_samples_watcher(self):
        """Start background thread to update files.json for live gallery"""
        if self.samples_watcher_running:
            return

        self.samples_watcher_running = True

        def watcher_loop():
            while self.samples_watcher_running:
                try:
                    self.update_gallery_html()
                except Exception:
                    pass
                # Update every 5 seconds
                for _ in range(50):  # 5 seconds in 0.1s increments
                    if not self.samples_watcher_running:
                        break
                    time.sleep(0.1)

        self.samples_watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
        self.samples_watcher_thread.start()

    def stop_samples_watcher(self):
        """Stop the samples watcher thread"""
        was_running = self.samples_watcher_running
        self.samples_watcher_running = False
        if was_running:
            # One last refresh: the final epoch's samples and the final .safetensors land
            # seconds before the trainer exits, almost always inside the watcher's 5 s
            # sleep — without this the gallery never gains the final-epoch previews or
            # the Download Final LoRA button (loras.json "final" key).
            def _final_refresh():
                try:
                    self.update_gallery_html()
                except Exception:
                    pass
            threading.Thread(target=_final_refresh, daemon=True).start()

    def parse_sample_filename(self, filename):
        """Parse epoch, step, seed from sample filename"""
        import re
        info = {"epoch": None, "step": None, "seed": None, "prompt_idx": None}

        # Pattern: {name}_e{epoch:06d}_{promptIdx}_{timestamp}_{seed}.png
        # or: {name}_{step:06d}_{promptIdx}_{timestamp}_{seed}.png
        epoch_match = re.search(r'_e(\d{6})_', filename)
        if epoch_match:
            info["epoch"] = int(epoch_match.group(1))

        step_match = re.search(r'_(\d{6})_(\d{2})_\d{14}', filename)
        if step_match and not epoch_match:
            info["step"] = int(step_match.group(1))
            info["prompt_idx"] = int(step_match.group(2))
        elif epoch_match:
            prompt_match = re.search(r'_e\d{6}_(\d{2})_', filename)
            if prompt_match:
                info["prompt_idx"] = int(prompt_match.group(1))

        seed_match = re.search(r'_(\d+)\.\w+$', filename)
        if seed_match:
            info["seed"] = int(seed_match.group(1))

        return info

    def generate_gallery_html(self, images):
        """Generate the gallery HTML content"""
        import time

        # Parse image info and build items
        image_data = []
        for filename, mtime in images:
            info = self.parse_sample_filename(filename)
            info["filename"] = filename
            info["mtime"] = mtime
            info["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            image_data.append(info)

        image_items = ""
        for img in image_data:
            filename = img["filename"]
            timestamp = img["timestamp"]

            # Build epoch/step badge
            if img["epoch"] is not None:
                badge = f'<span class="epoch-badge">Epoch {img["epoch"]}</span>'
            elif img["step"] is not None:
                badge = f'<span class="step-badge">Step {img["step"]}</span>'
            else:
                badge = ''

            # Build seed info
            seed_info = f'<span class="seed">Seed: {img["seed"]}</span>' if img["seed"] else ''

            image_items += f'''
            <div class="gallery-item" onclick="openLightbox('{filename}')">
                <div class="image-container">
                    <img src="{filename}" alt="{filename}" loading="lazy">
                    {badge}
                </div>
                <div class="image-info">
                    <span class="filename">{filename}</span>
                    <div class="meta-row">
                        <span class="timestamp">{timestamp}</span>
                        {seed_info}
                    </div>
                </div>
            </div>'''

        if not image_items:
            image_items = '<div class="no-images">No sample images found yet. Start training to generate samples.</div>'

        # Build image data for JavaScript
        js_image_data = []
        for img in image_data:
            js_image_data.append({
                "filename": img["filename"],
                "epoch": img["epoch"],
                "step": img["step"],
                "seed": img["seed"]
            })

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }}
        header {{ background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }}
        header h1 {{ color: #ECF0F1; font-size: 24px; margin-bottom: 15px; }}
        .controls {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
        .controls label {{ display: flex; align-items: center; gap: 8px; }}
        .controls input[type="number"], .controls select {{ padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }}
        .controls button {{ padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background-color: #3498DB; }}
        #last-update {{ color: #95A5A6; font-size: 14px; }}
        main {{ padding: 20px; }}
        #gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .gallery-item {{ background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .gallery-item:hover {{ transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }}
        .image-container {{ position: relative; }}
        .gallery-item img {{ width: 100%; height: 250px; object-fit: contain; display: block; background-color: #1B2A38; }}
        .epoch-badge, .step-badge {{ position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .epoch-badge {{ background-color: #27AE60; color: white; }}
        .step-badge {{ background-color: #E67E22; color: white; }}
        .image-info {{ padding: 12px; display: flex; flex-direction: column; gap: 6px; }}
        .filename {{ font-weight: 500; font-size: 13px; word-break: break-all; color: #BDC3C7; }}
        .meta-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        .timestamp {{ color: #95A5A6; font-size: 12px; }}
        .seed {{ color: #3498DB; font-size: 12px; font-family: monospace; }}
        .no-images {{ grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; font-size: 18px; }}
        #lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }}
        #lightbox.active {{ display: flex; }}
        #lightbox img {{ max-width: 90%; max-height: 80%; object-fit: contain; }}
        #lightbox .close-btn {{ position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }}
        #lightbox .close-btn:hover {{ color: #E74C3C; }}
        #lightbox .nav-btn {{ position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }}
        #lightbox .nav-btn:hover {{ color: #2980B9; }}
        #lightbox .prev-btn {{ left: 20px; }}
        #lightbox .next-btn {{ right: 20px; }}
        #lightbox .image-details {{ margin-top: 15px; text-align: center; }}
        #lightbox .image-name {{ color: #ECF0F1; font-size: 16px; }}
        #lightbox .image-meta {{ color: #95A5A6; font-size: 14px; margin-top: 5px; }}
    </style>
</head>
<body>
    <header>
        <h1>Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Sort by: <select id="sort-select" onchange="sortGallery()">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High to Low)</option>
                <option value="epoch-asc">Epoch (Low to High)</option>
            </select></label>
            <label>Auto-refresh: <input type="number" id="refresh-interval" value="30" min="5" max="300"> sec</label>
            <button onclick="refreshGallery()">Refresh Now</button>
            <span id="last-update"></span>
        </div>
    </header>
    <main><div id="gallery">{image_items}</div></main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <script>
        const imageData = {json.dumps(js_image_data)};
        let refreshInterval = localStorage.getItem('fizgig-refresh') || 30;
        let refreshTimer = null;
        let currentImageIndex = 0;
        const images = imageData.map(d => d.filename);
        document.getElementById('refresh-interval').value = refreshInterval;
        updateLastRefresh();
        startRefreshTimer();
        const savedSort = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('sort-select').value = savedSort;
        document.getElementById('refresh-interval').addEventListener('change', (e) => {{
            refreshInterval = Math.max(5, Math.min(300, parseInt(e.target.value) || 30));
            e.target.value = refreshInterval;
            localStorage.setItem('fizgig-refresh', refreshInterval);
            startRefreshTimer();
        }});
        function startRefreshTimer() {{ if (refreshTimer) clearInterval(refreshTimer); refreshTimer = setInterval(refreshGallery, refreshInterval * 1000); }}
        function refreshGallery() {{ location.reload(); }}
        function updateLastRefresh() {{ document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString(); }}
        function sortGallery() {{
            const sortBy = document.getElementById('sort-select').value;
            localStorage.setItem('fizgig-sort', sortBy);
            const gallery = document.getElementById('gallery');
            const items = Array.from(gallery.querySelectorAll('.gallery-item'));
            items.sort((a, b) => {{
                const aFile = a.querySelector('img').alt;
                const bFile = b.querySelector('img').alt;
                const aData = imageData.find(d => d.filename === aFile) || {{}};
                const bData = imageData.find(d => d.filename === bFile) || {{}};
                switch(sortBy) {{
                    case 'newest': return images.indexOf(aFile) - images.indexOf(bFile);
                    case 'oldest': return images.indexOf(bFile) - images.indexOf(aFile);
                    case 'epoch-desc': return (bData.epoch || 0) - (aData.epoch || 0);
                    case 'epoch-asc': return (aData.epoch || 0) - (bData.epoch || 0);
                    default: return 0;
                }}
            }});
            items.forEach(item => gallery.appendChild(item));
        }}
        function getImageMeta(filename) {{
            const data = imageData.find(d => d.filename === filename);
            if (!data) return '';
            const parts = [];
            if (data.epoch !== null) parts.push('Epoch ' + data.epoch);
            if (data.step !== null) parts.push('Step ' + data.step);
            if (data.seed !== null) parts.push('Seed: ' + data.seed);
            return parts.join(' | ');
        }}
        function openLightbox(filename) {{
            currentImageIndex = images.indexOf(filename);
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }}
        function navigateLightbox(direction) {{
            if (images.length === 0) return;
            currentImageIndex = (currentImageIndex + direction + images.length) % images.length;
            const filename = images[currentImageIndex];
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
        }}
        document.addEventListener('keydown', (e) => {{
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        }});
        document.getElementById('lightbox').addEventListener('click', (e) => {{ if (e.target.id === 'lightbox') closeLightbox(); }});
        sortGallery();
    </script>
</body>
</html>'''
        return html

    def create_image_converter(self):
        """Create the Image Prep tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.image_converter_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Image Prep",
            "Resize, convert to PNG, and optionally face-crop your training images. "
            "Optional — skip straight to Captions if your images are already prepared.",
        )

        # Card 1: Training folder — display only. Everything happens IN this folder; the old
        # optional Output Folder is gone (it silently diverged from the training source set on
        # the Start tab, which is never what a training workflow wants).
        folders_card = self._start_section_card(
            outer, "Training Folder",
            "Everything below happens inside the training folder from the Start tab — "
            "prepared images land there, ready for the Captions tab and training.",
        )
        folders_card.grid_columnconfigure(1, weight=1)
        ttk.Label(folders_card, text="Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        tk.Label(folders_card, textvariable=self.image_folder_var,
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                 anchor="w").grid(row=0, column=1, sticky=tk.W, pady=4)
        tk.Label(folders_card, text="(set on the Start tab)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=1, sticky=tk.W, pady=(0, 4)
        )

        # Working from video or audio — sits ABOVE the prep steps because clips and voice
        # segments are cut first, and both then land in the same training folder as everything
        # else. Shown to everyone rather than only under MiniMax: the person who needs it is
        # the one who has not chosen a model yet, looking at an hour of footage or a voice
        # memo, wondering where to start.
        gizmo_card = self._start_section_card(
            outer, "Working from video or audio?",
            "Gizmo cuts training clips AND voice segments. Video: scrub to a moment, pick a "
            "length, save — frame rate, frame count, sizing and sound all come out on spec. "
            "Audio: open a recording (or a video, for just its sound), scrub the waveform, "
            "caption the voice — with optional Whisper transcription — and export ready "
            "training segments. Video and voice training are MiniMax H3 only; still images "
            "need none of this.",
        )
        _gz_row = tk.Frame(gizmo_card, bg=COLORS["bg_surface"])
        _gz_row.pack(anchor=tk.W)
        tk.Button(_gz_row, text="🎬🎙  Open Gizmo", command=self._launch_gizmo,
                  bg=COLORS["accent"], fg=COLORS["text_primary"],
                  activebackground=COLORS["accent_hover"],
                  activeforeground=COLORS["text_primary"], font=(FONT_FAMILY, 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=16, pady=8, cursor="hand2").pack(side=tk.LEFT)
        tk.Label(_gz_row, text="opens in its own window — Fizgig keeps running",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=12)

        # Card 2: What to do — one radio per outcome, plain-language hint under each. The radio
        # VALUES stay the historical mode strings so persistence and the convert pipeline are
        # untouched; only the visible labels changed.
        mode_card = self._start_section_card(outer, "1 · What to do", None)

        def _mode_radio(text, value, hint, hint_fg=None):
            rb = ttk.Radiobutton(mode_card, text=text, variable=self.prep_mode_var,
                                 value=value, command=self._on_prep_mode_changed)
            rb.pack(anchor=tk.W, pady=(6, 0))
            lbl = tk.Label(mode_card, text=hint, font=(FONT_FAMILY, 10),
                           fg=hint_fg or COLORS["text_explain"], bg=COLORS["bg_surface"],
                           wraplength=680, justify=tk.LEFT)
            lbl.pack(anchor=tk.W, padx=(24, 0))
            return rb

        _mode_radio(
            "Resize + face close-ups — recommended for people",
            "Auto Prep (Face Crops)",
            "Every photo is resized and saved as PNG, PLUS a zoomed-in copy of the face saved "
            "beside it — more detail shots for better likeness.\n"
            "\U0001F4A1 Works best on high-res originals: if your photos are already shrunk to "
            "training size, the face close-ups come out soft. Start from the biggest versions "
            "you have.")
        _mode_radio(
            "Resize only",
            "Resize Only",
            "Just resize + convert to PNG. Use for styles, objects, or already-cropped sets.")
        _mode_radio(
            "Face close-ups only",
            "Face Crop Only",
            "Keep only the cropped face from each photo — the full shot is not kept.")

        # Options row: max size always live; face options grey out in Resize Only (kept visible
        # so the layout doesn't jump and users learn they exist).
        opts_row = tk.Frame(mode_card, bg=COLORS["bg_surface"])
        opts_row.pack(anchor=tk.W, pady=(12, 0))
        ttk.Label(opts_row, text="Target megapixels:").pack(side=tk.LEFT, padx=(0, 4))
        _max_combo = ttk.Combobox(opts_row, textvariable=self.prep_megapixels_var,
                                  values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4",
                                          "3.0", "4.2"],
                                  state="readonly", width=6)
        _max_combo.pack(side=tk.LEFT)
        _max_combo.bind("<<ComboboxSelected>>", lambda e: self._update_prep_note())
        tk.Label(opts_row, text="MP  (larger images shrink to fit; smaller are left alone)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(
            side=tk.LEFT, padx=(4, 16))
        self._face_target_label = ttk.Label(opts_row, text="Face:")
        self._face_target_label.pack(side=tk.LEFT, padx=(0, 4))
        self._face_target_combo = ttk.Combobox(
            opts_row, textvariable=self.face_selection_var,
            values=["Largest Face", "Largest Male Face", "Largest Female Face"],
            state="readonly" if FACE_DETECTION_AVAILABLE else "disabled", width=18,
        )
        self._face_target_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._face_padding_label = ttk.Label(opts_row, text="Padding:")
        self._face_padding_label.pack(side=tk.LEFT, padx=(0, 4))
        self._face_padding_entry = ttk.Entry(opts_row, textvariable=self.face_padding_var, width=5)
        self._face_padding_entry.pack(side=tk.LEFT)
        self._face_pct_label = tk.Label(opts_row, text="% around the face",
                                        font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                                        bg=COLORS["bg_surface"])
        self._face_pct_label.pack(side=tk.LEFT, padx=(4, 0))
        if not FACE_DETECTION_AVAILABLE:
            self._face_unavail_label = ttk.Label(
                opts_row, text="(Run install_fizgig.py to enable)",
                foreground=COLORS["warning"],
            )
            self._face_unavail_label.pack(side=tk.LEFT, padx=(8, 0))
        else:
            self._face_unavail_label = None

        # Why this is megapixels and not a "max size" any more. Training picks its resolution by
        # AREA, so a longest-edge cap quietly shrank every non-square image below the training
        # target — a 3:4 photo kept only 75% of the pixels it could have trained at, 16:9 just 56%.
        tk.Label(mode_card,
                 text="Sizing is by target area (megapixels), not longest side — this matches how "
                      "training buckets your images, so prepping no longer costs you resolution. "
                      "Aspect ratio is preserved; nothing is cropped.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))
        # Only shown when prep is set BELOW the Training tab's target — the one harmful direction
        # (prepping higher is free: training simply downscales at cache time).
        self._prep_mp_warn_var = tk.StringVar(value="")
        self._prep_mp_warn_label = tk.Label(
            mode_card, textvariable=self._prep_mp_warn_var,
            font=(FONT_FAMILY, 9), fg=COLORS["warning"], bg=COLORS["bg_surface"],
            wraplength=760, justify=tk.LEFT)
        # packed/unpacked by _update_prep_note

        # Card 3: Your originals — the one real destination question, as an explicit choice
        # (replaces the old inverted "Replace originals" checkbox). Keep-safe is the default.
        orig_card = self._start_section_card(outer, "2 · Your originals", None)
        ttk.Radiobutton(
            orig_card, text="Keep them safe — moved to an 'originals' subfolder",
            variable=self.delete_originals_var, value=False,
            command=self._update_prep_note).pack(anchor=tk.W, pady=(4, 0))
        ttk.Radiobutton(
            orig_card, text="Replace them  ⚠ originals are gone after this",
            variable=self.delete_originals_var, value=True,
            command=self._update_prep_note).pack(anchor=tk.W, pady=(4, 2))

        # Card 4: What will happen — the single honest summary, computed from ALL the settings
        # (the old one-line note ignored half of them). Accent border so it reads as the answer.
        summary_card = self._start_section_card(outer, "\U0001F4CB What will happen", None,
                                                accent_border=True)
        self._prep_note_var = tk.StringVar()
        self._prep_note_label = tk.Label(
            summary_card, textvariable=self._prep_note_var,
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
            wraplength=700, justify=tk.LEFT,
        )
        self._prep_note_label.pack(anchor=tk.W)

        # Card 5: Run it — one unmistakable primary action (label carries the live image count,
        # set by _update_prep_note), with the face-detection test framed as the optional,
        # nothing-is-written side step it actually is.
        action_card = self._start_section_card(outer, "3 · Run it", None)

        self.prepare_images_btn = tk.Button(
            action_card, text="✨ Prepare Images Now", command=self.convert_images,
            font=(FONT_FAMILY, 12, "bold"),
            fg="#FFFFFF", bg=COLORS["accent"],
            activeforeground="#FFFFFF", activebackground=COLORS["accent_hover"],
            relief="flat", bd=0, padx=24, pady=8, cursor="hand2",
        )
        self.prepare_images_btn.pack(anchor=tk.W, pady=(4, 10))

        test_row = tk.Frame(action_card, bg=COLORS["bg_surface"])
        test_row.pack(anchor=tk.W)
        tk.Label(test_row, text="Want to check first?",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))
        self.preview_faces_btn = ttk.Button(
            test_row, text="Test face detection on one photo…", command=self.preview_faces,
            state="normal" if FACE_DETECTION_AVAILABLE else "disabled",
        )
        self.preview_faces_btn.pack(side=tk.LEFT)
        tk.Label(test_row, text="optional and safe — shows the crop, writes nothing",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_muted"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(8, 0))

        # Apply initial state (face-control greying + summary + button count)
        self._on_prep_mode_changed()

        # Card 5: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)

        self.convert_log = scrolledtext.ScrolledText(
            log_card, height=12, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.convert_log.pack(fill=tk.BOTH, expand=True)

        # Card 6: Look Consistency Filter — deliberately the LAST card: it scores the images as
        # they'll actually be trained, so it only makes sense after resize/crop/captioning is done.
        filter_card = self._start_section_card(
            outer, "Final Step: Look Consistency Filter (faces)",
            "Run this LAST, after every other prep stage — it scores the finished training folder. "
            "Pick THREE baseline images that nail the look you want; every image's face is scored "
            "against all three and averaged (ArcFace embedding similarity) — one baseline photo "
            "would bake its own angle/expression/lighting bias into every score, three cancel it "
            "out. Great for weeding out synthetic images that "
            "drifted off-look — the subtle near-misses a loss curve can never see. Click images to "
            "mark them, or let Auto-Suggest flag the statistical outliers, then move the marked "
            "ones out of the dataset in one go (they go to an 'excluded_by_look' subfolder — "
            "nothing is deleted). Real-but-unusual low scorers (tight angles, profiles) that you "
            "KEEP can ease into training gently — the scan saves its scores with the dataset, and "
            "the Training tab's 'Warm up look outliers' toggle (Krea 2) ramps their LR up over "
            "the first few epochs instead of letting them fight the forming identity.",
        )
        self._face_filter_btn = ttk.Button(
            filter_card, text="🔍 Open Look Filter…", command=self._open_face_filter_window,
            state="normal" if FACE_DETECTION_AVAILABLE else "disabled",
        )
        self._face_filter_btn.pack(anchor=tk.W)
        if not FACE_DETECTION_AVAILABLE:
            ttk.Label(filter_card, text="(Run install_fizgig.py to enable face tools)",
                      foreground=COLORS["warning"]).pack(anchor=tk.W, pady=(4, 0))

        self._add_youtube_help_button(outer, "image_prep")

    @property
    def face_detector(self):
        """Lazy-loaded face detector instance"""
        if self._face_detector is None and FACE_DETECTION_AVAILABLE:
            self._face_detector = FaceDetector()
        return self._face_detector

    def _on_prep_mode_changed(self, *args):
        """Grey out face-related controls in Resize Only (kept visible — layout doesn't jump,
        and users learn the options exist)."""
        mode = self.prep_mode_var.get()
        face_on = mode != "Resize Only"
        muted, secondary = COLORS["text_muted"], COLORS["text_secondary"]
        self._face_target_combo.configure(
            state=("readonly" if (face_on and FACE_DETECTION_AVAILABLE) else "disabled"))
        self._face_padding_entry.configure(state=("normal" if face_on else "disabled"))
        for lbl in (self._face_target_label, self._face_padding_label):
            try:
                lbl.configure(foreground=(secondary if face_on else muted))
            except tk.TclError:
                pass
        self._face_pct_label.configure(fg=(secondary if face_on else muted))
        self.preview_faces_btn.configure(
            state=("normal" if (face_on and FACE_DETECTION_AVAILABLE) else "disabled"))

        self._update_prep_note()

    def _prep_source_stats(self, max_sample=40):
        """(image_count, median_area_px, median_size) for the training folder's top-level images.

        AREA rather than longest edge, because that's what both prep and training size by.
        Read from image HEADERS only (PIL .size — no pixel decode), sampled at most
        `max_sample` files, so it's cheap enough to run on every settings change. Returns
        (0, None, None) when the folder is unset/empty."""
        folder = self.image_folder_var.get().strip()
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        try:
            files = [os.path.join(folder, f) for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in exts]
        except OSError:
            return 0, None, None
        sizes = []
        for p in files[:max_sample]:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                pass
        if not sizes:
            return len(files), None, None
        sizes.sort(key=lambda wh: wh[0] * wh[1])
        median_size = sizes[len(sizes) // 2]
        return len(files), median_size[0] * median_size[1], median_size

    def _update_prep_note(self, *args):
        """The 'What will happen' summary — computed from ALL the settings (mode, originals
        choice, target megapixels, live folder contents). The one-line note this replaces ignored the
        output folder entirely and taught users the wrong answer to 'does this touch my
        folder?'."""
        if not hasattr(self, '_prep_note_var'):
            return
        # A voice folder has nothing for this tab to do — resize, crop and face detection are
        # image operations. Say so instead of promising to process "your 0 images".
        if self._training_folder_audio_only():
            self._prep_note_var.set(
                "🎙 Audio-only training set — this tab prepares images, and voice recordings "
                "need none of it. Segments are cut, captioned and sized in Gizmo's audio tab; "
                "your files here are already ready to train.")
            return
        mode = self.prep_mode_var.get()
        replace = self.delete_originals_var.get()
        try:
            prep_mp = float(self.prep_megapixels_var.get())
        except (ValueError, tk.TclError):
            prep_mp = 1.0
        target_area = self._prep_target_area(prep_mp)
        n, median_area, median_size = self._prep_source_stats()

        # A worked example in the user's own aspect ratio, so "1.0 MP" is a concrete size.
        example = ""
        if median_size:
            _w, _h = self._prep_output_size(median_size, target_area)
            if (_w, _h) != tuple(median_size):
                example = f" (your typical {median_size[0]}×{median_size[1]} → {_w}×{_h})"

        count = f"your {n} images" if n else "your images"
        sized = f"resized to about {prep_mp:g} MP{example} and saved as PNG"
        if mode == "Auto Prep (Face Crops)":
            what = (f"{count} → {sized}, plus one face "
                    f"close-up each{f' (≈{n * 2} files)' if n else ''}")
        elif mode == "Face Crop Only":
            what = (f"{count} → replaced by just the cropped face from each photo, "
                    f"saved as PNG")
        else:
            what = f"{count} → {sized}"

        where = "Everything lands in your training folder."
        if replace:
            originals = "Your original files are replaced ⚠ there is no undo."
        else:
            originals = ("Your originals are moved to the 'originals' subfolder — "
                         "nothing is deleted.")

        lines = [f"{what}. {where} {originals}"]
        # Soft-crop warning: face modes cropping from images that are already training-size
        # produce small, blurry faces. Header-read median AREA across a sample of the folder.
        if mode != "Resize Only" and median_area is not None and median_area <= target_area:
            lines.append(f"⚠ Your images are already at or below {prep_mp:g} MP — face "
                         f"close-ups cut from them will be soft. If you have higher-res "
                         f"versions, prep from those instead.")
        if mode != "Resize Only":
            lines.append("Next: eyeball the face close-ups on the Captions tab and Remove any "
                         "blurry ones before captioning.")
        self._prep_note_var.set("\n".join(lines))

        # Prep BELOW the training target is the one harmful direction: training never upscales,
        # so those pixels are gone for good. Prepping higher is free — training just downscales.
        if hasattr(self, "_prep_mp_warn_label"):
            try:
                train_mp = float(self.dataset_megapixels_var.get())
            except (ValueError, tk.TclError):
                train_mp = prep_mp
            if prep_mp < train_mp:
                self._prep_mp_warn_var.set(
                    f"⚠ Training is set to {train_mp:g} MP but prep is set to {prep_mp:g} MP — "
                    f"your images would be shrunk below what training asks for, and training "
                    f"cannot get that detail back. Match them, or prep higher.")
                if not self._prep_mp_warn_label.winfo_manager():
                    self._prep_mp_warn_label.pack(anchor=tk.W, pady=(4, 0))
            else:
                self._prep_mp_warn_var.set("")
                if self._prep_mp_warn_label.winfo_manager():
                    self._prep_mp_warn_label.pack_forget()

        # The Run button carries the live count — "Prepare 34 Images Now" answers "run on what?"
        if hasattr(self, "prepare_images_btn"):
            self.prepare_images_btn.configure(
                text=(f"✨ Prepare {n} Image{'s' if n != 1 else ''} Now" if n
                      else "✨ Prepare Images Now"))

    def _get_face_selection_mode(self):
        """Parse face selection mode from Face Target dropdown."""
        mode_text = self.face_selection_var.get()
        if "Male" in mode_text:
            return "largest_male"
        elif "Female" in mode_text:
            return "largest_female"
        return "largest_face"

    def preview_faces(self):
        """Preview face detection on a single image"""
        if not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available. Run install_fizgig.py first.")
            return

        filepath = filedialog.askopenfilename(
            title="Select image to preview faces",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif")]
        )
        if not filepath:
            return

        try:
            # Detect faces
            faces = self.face_detector.detect_all(filepath)

            if not faces:
                messagebox.showinfo("No Faces", f"No faces detected in:\n{os.path.basename(filepath)}")
                return

            # Find the largest face (or by gender based on current mode)
            crop_mode = self._get_face_selection_mode()
            if crop_mode == "largest_male":
                selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            elif crop_mode == "largest_female":
                selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            else:
                selected = self.face_detector.get_largest(faces)

            # Get highlight index
            highlight_idx = faces.index(selected) if selected in faces else None

            # Load image and draw boxes
            with Image.open(filepath) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                preview_img = draw_face_boxes(img, faces, highlight_index=highlight_idx)

                # Create preview window
                self._show_face_preview_window(preview_img, faces, filepath, highlight_idx)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect faces:\n{str(e)}")

    def _show_face_preview_window(self, preview_img, faces, filepath, highlight_idx):
        """Show a popup window with the face detection preview"""
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"Face Preview - {os.path.basename(filepath)}")
        preview_window.configure(bg=BG_COLOR)

        # Resize for display if too large
        display_img = preview_img.copy()
        max_display = 800
        if display_img.width > max_display or display_img.height > max_display:
            ratio = min(max_display / display_img.width, max_display / display_img.height)
            new_size = (int(display_img.width * ratio), int(display_img.height * ratio))
            display_img = display_img.resize(new_size, Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)

        # Image label
        img_label = ttk.Label(preview_window, image=photo)
        img_label.image = photo  # Keep reference
        img_label.pack(padx=10, pady=10)

        # Info frame
        info_frame = ttk.Frame(preview_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        # Face count
        ttk.Label(info_frame, text=f"Faces detected: {len(faces)}", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        # Face details
        for i, face in enumerate(faces):
            marker = " [SELECTED]" if i == highlight_idx else ""
            gender = face.gender.capitalize() if face.gender != 'unknown' else '?'
            ttk.Label(
                info_frame,
                text=f"  Face {i+1}: {gender}, {face.area:,} px{marker}"
            ).pack(anchor=tk.W)

        # Legend
        legend_frame = ttk.Frame(preview_window)
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(legend_frame, text="Green = Selected for cropping", foreground="green").pack(side=tk.LEFT, padx=10)
        ttk.Label(legend_frame, text="Yellow = Other faces", foreground="yellow").pack(side=tk.LEFT, padx=10)

        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

    # region Look Consistency Filter (face-embedding drift)

    _FF_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    def _ff_verdict(self, sim):
        """(label, color, blurb) for a similarity score. Thresholds are ArcFace cosine-sim
        conventions: same person across varied photos usually lands 0.30-0.70 vs a single
        baseline; a different person rarely clears 0.25."""
        if sim is None:
            return ("no face", "#7F8C8D",
                    "No face detected — can't be scored. Back shots are fine; judge by eye.")
        if sim >= 0.45:
            return ("match", "#70AD47", "Solid match to the baseline look.")
        if sim >= 0.30:
            return ("borderline", "#E67E22",
                    "Same person territory, but drifting — worth an eyeball.")
        return ("drift", "#E74C3C",
                "Weak match — likely off-look (synthetic drift or a different subject).")

    def _open_face_filter_window(self):
        """Look Consistency Filter — score every training image's face against 3 chosen baselines
        (ArcFace embeddings averaged, CPU), mark drifters by click or auto-suggest, and move the
        marked ones to <folder>/excluded_by_look/. The Image Prep card places this LAST on
        purpose: it scores the finished dataset, so run it after resize/crop/captioning."""
        win = getattr(self, "_ff_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        win = tk.Toplevel(self.master)
        win.title("Look Consistency Filter — face embedding drift")
        win.geometry("860x720")
        win.configure(bg=COLORS["bg_deep"])
        self._ff_win = win
        self._ff_thumbs = {}          # path -> PhotoImage (kept alive)
        self._ff_scores = {}          # path -> float similarity or None (no face)
        self._ff_marked = set()       # paths marked for exclusion
        self._ff_row_ui = {}          # path -> row widgets (in-place repaints)
        self._ff_baselines = []       # exactly 3 baseline image paths (scores are averaged)
        self._ff_busy = False
        if not hasattr(self, "_ff_embed_cache"):
            self._ff_embed_cache = {}     # (path, mtime) -> embedding or None; survives reopens
        if not hasattr(self, "_ff_embedder"):
            self._ff_embedder = None      # lazy FaceEmbedder; model load is the slow part

        head = tk.Frame(win, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=14, pady=(12, 6))
        tk.Label(head, text="Look Consistency Filter", font=(FONT_FAMILY, 15, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(side=tk.LEFT)
        self._ff_apply_btn = ttk.Button(head, text="Move Marked Out of Dataset",
                                        command=self._ff_apply_moves, state="disabled")
        self._ff_apply_btn.pack(side=tk.RIGHT)
        self._ff_suggest_btn = ttk.Button(head, text="Auto-Suggest Drift",
                                          command=self._ff_auto_suggest, state="disabled")
        self._ff_suggest_btn.pack(side=tk.RIGHT, padx=(0, 8))

        base_row = tk.Frame(win, bg=COLORS["bg_deep"])
        base_row.pack(fill=tk.X, padx=14, pady=(0, 4))
        self._ff_base_slots = []
        for _ in range(3):
            holder = tk.Frame(base_row, width=72, height=72, bg=COLORS["bg_surface"],
                              highlightbackground=COLORS["border"], highlightthickness=1)
            holder.pack_propagate(False)
            holder.pack(side=tk.LEFT, padx=(0, 4))
            lbl = tk.Label(holder, text="no\nbaseline", font=(FONT_FAMILY, 8),
                           fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
            lbl.pack(expand=True)
            self._ff_base_slots.append(lbl)
        base_btns = tk.Frame(base_row, bg=COLORS["bg_deep"])
        base_btns.pack(side=tk.LEFT, padx=(10, 0))
        self._ff_base_name = tk.Label(base_btns, text="Pick the 3 images that best nail the look you "
                                                      "want (Ctrl-click to select all three at once).",
                                      font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"],
                                      bg=COLORS["bg_deep"], anchor="w")
        self._ff_base_name.pack(anchor="w")
        btns = tk.Frame(base_btns, bg=COLORS["bg_deep"])
        btns.pack(anchor="w", pady=(6, 0))
        ttk.Button(btns, text="Choose 3 Baselines…", command=self._ff_choose_baselines).pack(side=tk.LEFT)
        self._ff_scan_btn = ttk.Button(btns, text="Scan Folder", command=self._ff_scan, state="disabled")
        self._ff_scan_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._ff_status = tk.Label(win, text="Each score is the AVERAGE ArcFace similarity to your 3 "
                                             "baselines — averaging cancels the angle/expression/lighting "
                                             "bias any single photo carries. Same person typically lands "
                                             "30–70% (even the baselines themselves — each is scored "
                                             "against the other two as well as itself).",
                                   font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_deep"],
                                   justify=tk.LEFT, anchor="w", wraplength=820)
        self._ff_status.pack(fill=tk.X, padx=14)

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 12))
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vbar = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        rows_id = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(rows_id, width=e.width))
        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.
        self._ff_rows = rows

    def _ff_set_status(self, text):
        win = getattr(self, "_ff_win", None)
        if win is not None and win.winfo_exists():
            self._ff_status.config(text=text)

    def _ff_choose_baselines(self):
        """Exactly 3 baselines, scored by averaging — one photo bakes its own angle/expression/
        lighting bias into every score; three cancel it out."""
        folder = self.image_folder_var.get().strip()
        paths = filedialog.askopenfilenames(
            title="Choose 3 baseline images (the look you want)",
            initialdir=folder if folder and os.path.isdir(folder) else None,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if not paths:
            return
        if len(paths) != 3:
            messagebox.showwarning(
                "Look Filter",
                f"Pick exactly 3 baseline images (you picked {len(paths)}).\n\n"
                "Ctrl-click in the file dialog to select three. Three baselines average out "
                "the angle/expression/lighting bias any single photo carries.")
            return
        # normpath BOTH here and in the scan's file list: filedialog returns forward
        # slashes while the scan builds candidates with os.path.join (backslashes on
        # Windows), so every `p in self._ff_baselines` test was False — baselines were
        # never badged, embedded twice, and could be auto-suggested and MOVED OUT of the
        # dataset by "Move Marked".
        self._ff_baselines = [os.path.normpath(p) for p in paths]
        for slot, p in zip(self._ff_base_slots, paths):
            try:
                with Image.open(p) as im:
                    im.thumbnail((68, 68), Image.LANCZOS)
                    ph = ImageTk.PhotoImage(im)
                self._ff_thumbs[f"__baseline_{p}__"] = ph
                slot.config(image=ph, text="")
            except Exception:
                slot.config(image="", text="no\npreview")
        self._ff_base_name.config(
            text="Baselines: " + ", ".join(os.path.basename(p) for p in paths))
        self._ff_scan_btn.config(state="normal")
        if self._ff_scores:
            self._ff_set_status("Baselines changed — click Scan Folder to re-score "
                                "(cached embeddings make this fast).")

    _ff_lock = threading.Lock()   # one embed at a time: Look Filter scan + gallery scorer share the model

    def _ff_embed_cached(self, path):
        """Embedding via the (path, mtime) cache — model load + detection is the slow part,
        so re-scans and baseline swaps cost almost nothing. Self-initializing and locked:
        the Look Filter scan thread and the gallery likeness scorer share one embedder."""
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        if not hasattr(self, "_ff_embed_cache"):
            self._ff_embed_cache = {}
        if key not in self._ff_embed_cache:
            with self._ff_lock:
                if key not in self._ff_embed_cache:
                    if getattr(self, "_ff_embedder", None) is None:
                        self._ff_embedder = FaceEmbedder()
                    try:
                        self._ff_embed_cache[key] = self._ff_embedder.embed(path)
                    except Exception:
                        self._ff_embed_cache[key] = None
        return self._ff_embed_cache[key]

    def _repair_embed_pil(self, pil):
        """(embedding, bbox) for the largest face in an IN-MEMORY render — FaceEmbedder.embed
        is path-only, and the pop-out metrics score renders that never touch disk. Same lock
        and same lazily-created embedder as _ff_embed_cached, so the model loads once app-wide.

        The unpadded detection gives both the embedding and a usable bbox. The pad-retry
        fallback (frame-filling faces) returns coordinates in padded-image space, so it
        contributes the embedding only — texture then measures the whole frame."""
        if FaceEmbedder is None:
            return None, None
        try:
            import cv2
            import numpy as np
            bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            with self._ff_lock:
                if getattr(self, "_ff_embedder", None) is None:
                    self._ff_embedder = FaceEmbedder()
                self._ff_embedder._ensure_loaded()
                faces = self._ff_embedder._app.get(bgr)
                if faces:
                    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    x1, y1, x2, y2 = (int(v) for v in f.bbox)
                    h, w = bgr.shape[:2]
                    bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                    return np.asarray(f.normed_embedding, dtype=np.float32), bbox
                faces = self._ff_embedder._detect_with_pad_retry(bgr)
                if faces:
                    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    return np.asarray(f.normed_embedding, dtype=np.float32), None
        except Exception:
            pass
        return None, None

    def _ff_scan(self):
        if self._ff_busy:
            return
        folder = self.image_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Look Filter", "Set your training image folder on the Start tab first.")
            return
        if len(self._ff_baselines) != 3 or not all(os.path.exists(b) for b in self._ff_baselines):
            messagebox.showwarning("Look Filter", "Choose 3 baseline images first.")
            return
        # normpath to match self._ff_baselines (see the baseline picker) — the folder string
        # itself can carry forward slashes from a file dialog.
        files = sorted(
            os.path.normpath(os.path.join(folder, f)) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in self._FF_EXTS)
        if not files:
            messagebox.showinfo("Look Filter", "No images found in the training folder.")
            return
        self._ff_busy = True
        self._ff_scan_btn.config(state="disabled")
        self._ff_suggest_btn.config(state="disabled")
        self._ff_apply_btn.config(state="disabled")
        self._ff_set_status("Loading face model (first run downloads ~300 MB)…")
        win = self._ff_win

        def work():
            import numpy as np
            try:
                base_embs = [self._ff_embed_cached(b) for b in self._ff_baselines]
                missing = [os.path.basename(b) for b, e in zip(self._ff_baselines, base_embs)
                           if e is None]
                if missing:
                    self.master.after(0, lambda: self._ff_scan_done(
                        None, "No face found in baseline(s): " + ", ".join(missing) +
                        " — pick images with a clear face."))
                    return
                scores = {}
                for i, p in enumerate(files, 1):
                    if not win.winfo_exists():
                        return   # window closed mid-scan — drop the work silently
                    emb = self._ff_embed_cached(p)
                    # Average of the 3 similarities == similarity to the (unnormalized) centroid
                    # of the baselines — one photo's framing bias can't dominate the score.
                    scores[p] = None if emb is None else float(
                        np.mean([np.dot(be, emb) for be in base_embs]))
                    if i % 3 == 0 or i == len(files):
                        done, total = i, len(files)
                        self.master.after(0, lambda d=done, t=total:
                                          self._ff_set_status(f"Scoring faces… {d}/{t}"))
                self.master.after(0, lambda: self._ff_scan_done(scores, None))
            except Exception as e:
                self.master.after(0, lambda err=str(e): self._ff_scan_done(None, f"Scan failed: {err}"))

        threading.Thread(target=work, daemon=True).start()

    def _ff_scan_done(self, scores, error):
        self._ff_busy = False
        win = getattr(self, "_ff_win", None)
        if win is None or not win.winfo_exists():
            return
        self._ff_scan_btn.config(state="normal")
        if error:
            self._ff_set_status(error)
            return
        self._ff_scores = scores
        self._ff_marked &= set(scores)   # drop marks for files that vanished
        self._ff_suggest_btn.config(state="normal")
        scored = [s for s in scores.values() if s is not None]
        nf = sum(1 for s in scores.values() if s is None)
        # Persist for the trainer's "Warm up look outliers" toggle — the scores travel with the
        # dataset (same pattern as fizgig_excluded.json). Cutoff = the auto-suggest fence.
        try:
            folder = self.image_folder_var.get().strip()
            ss = sorted(scored)
            cutoff = None
            if len(ss) >= 4:
                n = len(ss)
                med, q1, q3 = ss[n // 2], ss[n // 4], ss[(3 * n) // 4]
                cutoff = max(med - 1.5 * (q3 - q1), 0.25)
            payload = {"baselines": [os.path.basename(b) for b in self._ff_baselines],
                       "cutoff": cutoff,
                       "scores": {os.path.splitext(os.path.basename(p))[0]: s
                                  for p, s in scores.items()}}
            with open(os.path.join(folder, "fizgig_look_scores.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        self._ff_set_status(f"{len(scores)} image(s) scored — {len(scored)} with faces, {nf} without. "
                            "Worst matches first. Click a row (or its button) to mark it, or use "
                            "Auto-Suggest. Scores saved with the dataset — low scorers you KEEP can "
                            "ease in gently: tick “Warm up look outliers” on the Training tab (Krea 2).")
        self._ff_build_rows()

    def _ff_toggle(self, path):
        if path in self._ff_marked:
            self._ff_marked.discard(path)
        else:
            self._ff_marked.add(path)
        # In-place row update — a full rebuild on every click froze the window for a beat
        # and yanked the scroll position back to the top.
        self._ff_update_row(path)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_update_row(self, path):
        """Repaint one row's marked/unmarked state without rebuilding the list."""
        ui = getattr(self, "_ff_row_ui", {}).get(path)
        if ui is None:
            return
        try:
            if not ui["frame"].winfo_exists():
                return
            marked = path in self._ff_marked
            ui["frame"].config(highlightbackground="#C0392B" if marked else ui["color"],
                               highlightthickness=3 if marked else 2)
            ui["mark"].config(text="  ❌ marked for exclusion" if marked else "")
            ui["btn"].config(text="Keep" if marked else "Mark")
        except Exception:
            pass

    def _ff_auto_suggest(self):
        """Mark statistical drift: below the dataset's own low outlier fence (median − 1.5·IQR)
        or below the 0.25 different-person floor. No-face rows are never suggested — unscoreable
        isn't the same as bad (think from-behind shots)."""
        scored = sorted(s for s in self._ff_scores.values() if s is not None)
        if len(scored) < 4:
            messagebox.showinfo("Look Filter", "Not enough scored faces for statistics — mark by eye.")
            return
        n = len(scored)
        med = scored[n // 2]
        q1, q3 = scored[n // 4], scored[(3 * n) // 4]
        cutoff = max(med - 1.5 * (q3 - q1), 0.25)   # dataset's low outlier fence, floored at the
        newly = {p for p, s in self._ff_scores.items()   # ~different-person similarity level
                 if s is not None and s < cutoff and p not in self._ff_baselines}
        self._ff_marked |= newly
        self._ff_set_status(f"Auto-suggest marked {len(newly)} image(s) "
                            f"(dataset median {med * 100:.0f}%, cutoff {cutoff * 100:.0f}%). "
                            "Review before moving — it flags statistical drift, not certainty.")
        for p in newly:
            self._ff_update_row(p)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_build_rows(self):
        win = getattr(self, "_ff_win", None)
        if win is None or not win.winfo_exists():
            return
        for w in self._ff_rows.winfo_children():
            w.destroy()
        self._ff_row_ui = {}   # path -> widgets for in-place mark/unmark repaints
        thumb_jobs = []
        # Worst match first; unscoreable (no face) at the bottom — they're a judgement call.
        items = sorted(self._ff_scores.items(),
                       key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0.0))
        for path, sim in items:
            label, color, blurb = self._ff_verdict(sim)
            marked = path in self._ff_marked
            row = tk.Frame(self._ff_rows, bg=COLORS["bg_surface"],
                           highlightbackground="#C0392B" if marked else color,
                           highlightthickness=3 if marked else 2)
            row.pack(fill=tk.X, pady=4)

            thumb_holder = tk.Frame(row, width=100, height=100, bg=COLORS["bg_surface"])
            thumb_holder.pack_propagate(False)
            thumb_holder.pack(side=tk.LEFT, padx=8, pady=8)
            ph = self._ff_thumbs.get(path)
            tl = tk.Label(thumb_holder, bg=COLORS["bg_surface"], cursor="hand2")
            if ph is not None:
                tl.config(image=ph)
            else:
                tl.config(text="…", font=(FONT_FAMILY, 8), fg=COLORS["text_muted"])
                thumb_jobs.append((path, tl))
            tl.pack(expand=True)
            tl.bind("<Button-1>", lambda e, p=path: self._ff_toggle(p))

            info = tk.Frame(row, bg=COLORS["bg_surface"])
            info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)
            name_row = tk.Frame(info, bg=COLORS["bg_surface"])
            name_row.pack(fill=tk.X, anchor="w")
            tk.Label(name_row, text=os.path.basename(path), font=(FONT_FAMILY, 10, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            tk.Label(name_row, text=f"  {label.upper()}", font=(FONT_FAMILY, 9, "bold"),
                     fg=color, bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            if path in self._ff_baselines:
                tk.Label(name_row, text="  ★ baseline", font=(FONT_FAMILY, 9),
                         fg="#F1C40F", bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            mark_lbl = tk.Label(name_row, text="  ❌ marked for exclusion" if marked else "",
                                font=(FONT_FAMILY, 9, "bold"),
                                fg="#C0392B", bg=COLORS["bg_surface"])
            mark_lbl.pack(side=tk.LEFT)
            btn = ttk.Button(name_row, text="Keep" if marked else "Mark", width=8,
                             command=lambda p=path: self._ff_toggle(p))
            btn.pack(side=tk.RIGHT, padx=(8, 0))
            self._ff_row_ui[path] = {"frame": row, "mark": mark_lbl, "btn": btn, "color": color}

            sim_txt = f"match {sim * 100:.0f}%" if sim is not None else "no face to score"
            tk.Label(info, text=sim_txt, font=(FONT_FAMILY, 9),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]).pack(anchor="w", pady=(2, 0))
            tk.Label(info, text=blurb, font=(FONT_FAMILY, 8, "italic"),
                     fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(anchor="w", pady=(2, 0))
        if thumb_jobs:
            self._load_thumbs_async(thumb_jobs, self._ff_thumbs)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_apply_moves(self):
        """Move marked images (+ their caption .txt) to <folder>/excluded_by_look/. Never deletes."""
        marked = [p for p in self._ff_marked if os.path.exists(p)]
        if not marked:
            return
        folder = self.image_folder_var.get().strip()
        dest_dir = os.path.join(folder, "excluded_by_look")
        if not messagebox.askyesno(
                "Move marked images",
                f"Move {len(marked)} image(s) (plus matching captions) out of the dataset "
                f"to:\n\n{dest_dir}\n\nNothing is deleted — move them back to re-admit them."):
            return
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        moved = 0
        for p in marked:
            target = None
            try:
                base = os.path.basename(p)
                target = os.path.join(dest_dir, base)
                n = 2
                while os.path.exists(target):
                    stem, ext = os.path.splitext(base)
                    target = os.path.join(dest_dir, f"{stem}_{n}{ext}")
                    n += 1
                shutil.move(p, target)
                cap = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(cap):
                    cap_target = os.path.splitext(target)[0] + ".txt"
                    shutil.move(cap, cap_target)
                moved += 1
                self._ff_scores.pop(p, None)
                self._ff_marked.discard(p)
                # Drop just this row — rebuilding the whole list is slow and loses scroll position.
                ui = self._ff_row_ui.pop(p, None)
                if ui is not None:
                    try:
                        ui["frame"].destroy()
                    except Exception:
                        pass
            except Exception as e:
                # A failed move can leave a half-state behind (shutil.move falls back to
                # copy+delete when the source is briefly locked, e.g. mid thumbnail decode;
                # the copy lands, the delete fails). Remove the orphan copy so the image
                # isn't duplicated in and out of the dataset.
                try:
                    if target and os.path.exists(p) and os.path.exists(target):
                        os.remove(target)
                except Exception:
                    pass
                self._ff_set_status(f"Could not move {os.path.basename(p)}: {e}")
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")
        self._ff_set_status(f"Moved {moved} image(s) to {dest_dir}. "
                            f"{len(self._ff_scores)} image(s) remain in the dataset.")

    # endregion

    # region Image Prep Helpers

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    def _launch_gizmo(self):
        """Open Gizmo, the clip and voice prep tool, as its own process.

        On a pod this button is the ONLY route — there is no desktop icon and a .bat is useless
        on Linux. It works because the container runs openbox and DISPLAY=:1 is set in the image,
        both of which this process already inherited, so the child does too.

        Not a "close Fizgig and open Gizmo" flow, deliberately: on a pod Fizgig is PID 1's
        successor and closing it would kill the pod.
        """
        script = os.path.join(FIZGIG_DIR, "gizmo.pyw" if os.name == "nt" else "gizmo.py")
        if not os.path.isfile(script):
            messagebox.showerror("Gizmo not found",
                                 f"{os.path.basename(script)} is missing from your Fizgig folder. "
                                 "Update Fizgig to get it.")
            return

        proc = getattr(self, "_gizmo_proc", None)
        if proc is not None and proc.poll() is None:
            messagebox.showinfo("Gizmo is already open",
                                "Gizmo is running — look for its window behind this one.")
            return

        exe = self._venv_python()
        if os.name == "nt":
            # pythonw, or the child inherits a console window Fizgig itself does not have.
            cand = os.path.join(FIZGIG_DIR, "venv", "Scripts", "pythonw.exe")
            if os.path.isfile(cand):
                exe = cand
        try:
            self._gizmo_proc = subprocess.Popen([exe, script], cwd=FIZGIG_DIR)
        except Exception as exc:
            messagebox.showerror("Gizmo could not start", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _atomic_png_save(img, output_path):
        """Write the PNG to a temp file then os.replace into place. In-place mode saves
        straight over the original — a crash, full disk or End Task mid-write used to
        truncate the source photo beyond recovery."""
        tmp = output_path + ".fizgig-tmp"
        try:
            img.save(tmp, "PNG")
            os.replace(tmp, output_path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _safe_output_path(self, filepath, output_path):
        """Never overwrite a DIFFERENT existing file.

        Every prep mode writes `<stem>.png`, so a folder holding photo.jpg AND an
        unrelated photo.png would have the .jpg's output destroy the .png (and
        _handle_original then deletes the .jpg — one photo gone, silently).
        In-place re-save of the same file is fine; a genuine collision gets a
        `_2`/`_3`... suffix instead, with a log line."""
        if not os.path.exists(output_path):
            return output_path
        try:
            if os.path.samefile(filepath, output_path):
                return output_path  # in-place re-save of itself
        except OSError:
            pass
        stem, ext = os.path.splitext(output_path)
        n = 2
        candidate = f"{stem}_{n}{ext}"
        while os.path.exists(candidate):
            n += 1
            candidate = f"{stem}_{n}{ext}"
        self._log(f"Name collision: {os.path.basename(output_path)} already exists — "
                  f"writing {os.path.basename(candidate)} instead\n")
        return candidate

    def _stash_original_if_inplace(self, filepath, output_path, output_folder, replace_originals):
        """Call BEFORE saving. Keep-safe mode + in-place PNG output means the save OVERWRITES
        the original — by the time _handle_original ran, there was nothing left to move
        (issue #43: PNG originals silently destroyed while JPGs were kept). A COPY rather
        than a move, because PIL may still hold the source file open at this point and a
        move of an open file fails on Windows; the end state is identical — original
        content in originals/, processed image at the original name."""
        if replace_originals or filepath != output_path or not os.path.exists(filepath):
            return
        if not hasattr(self, '_originals_dir_cache'):
            self._originals_dir_cache = {}
        if output_folder not in self._originals_dir_cache:
            self._originals_dir_cache[output_folder] = self._get_originals_dir(output_folder)
        originals_dir = self._originals_dir_cache[output_folder]
        os.makedirs(originals_dir, exist_ok=True)
        import shutil
        shutil.copy2(filepath, os.path.join(originals_dir, os.path.basename(filepath)))

    def _handle_original(self, filepath, output_path, output_folder, replace_originals):
        """Handle the original file: delete if replacing, move to subfolder if preserving.
        The in-place keep-safe case is covered by _stash_original_if_inplace BEFORE the save
        — by this point the overwrite has already happened, correctly for replace mode and
        harmlessly for keep-safe (the original is already copied out)."""
        if filepath == output_path:
            return  # Output overwrote original, nothing to do
        if replace_originals:
            os.remove(filepath)
        else:
            if not hasattr(self, '_originals_dir_cache'):
                self._originals_dir_cache = {}
            if output_folder not in self._originals_dir_cache:
                self._originals_dir_cache[output_folder] = self._get_originals_dir(output_folder)
            originals_dir = self._originals_dir_cache[output_folder]
            os.makedirs(originals_dir, exist_ok=True)
            import shutil
            dest = os.path.join(originals_dir, os.path.basename(filepath))
            shutil.move(filepath, dest)

    def _get_originals_dir(self, output_folder):
        """Find the next available originals folder (originals, originals_2, originals_3, etc.)."""
        candidate = os.path.join(output_folder, "originals")
        if not os.path.isdir(candidate):
            return candidate
        # Check if it has any images
        has_images = any(
            os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
            for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
        )
        if not has_images:
            return candidate
        # Find next numbered folder
        n = 2
        while True:
            candidate = os.path.join(output_folder, f"originals_{n}")
            if not os.path.isdir(candidate):
                return candidate
            has_images = any(
                os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
                for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
            )
            if not has_images:
                return candidate
            n += 1

    def _get_image_files(self, folder):
        """Scan folder for image files, return sorted list of full paths."""
        files = []
        for filename in sorted(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in self.IMAGE_EXTENSIONS:
                files.append(filepath)
        return files

    def _load_image(self, filepath):
        """Load an image and convert to RGB/RGBA."""
        img = Image.open(filepath)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
        return img

    @staticmethod
    def _bucket_step():
        """The resolution grid training buckets on (RESOLUTION_STEPS). Read from the dataset
        module so prep and bucketing can never drift apart; 16 if the import isn't available."""
        try:
            import sys as _sys
            _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
            if _src not in _sys.path:
                _sys.path.insert(0, _src)
            from fizgig.dataset.image_dataset import RESOLUTION_STEPS
            return int(RESOLUTION_STEPS)
        except Exception:
            return 16

    def _prep_target_area(self, mp):
        """Training's real target AREA for a megapixel setting.

        Derived exactly as the dataset TOML writer derives `resolution` — floor the square side
        to the bucket grid (1.0 MP -> 992x992 = 984 064 px, not 1 000 000). Matching it means
        prep lands at or just UNDER what training asks for, which keeps training's no-upscale
        path: the cache step then resamples and crops nothing at all."""
        step = self._bucket_step()
        side = max(step, int(math.sqrt(max(0.0, mp) * 1_000_000)) // step * step)
        return side * side

    def _prep_output_size(self, size, target_area):
        """The (w, h) `_resize_image` would produce for `size` — same maths, no pixels touched.
        Used by the summary card to show a worked example before anything is written."""
        width, height = size
        cur_area = width * height
        if cur_area <= target_area:
            return width, height
        step = self._bucket_step()
        scale = math.sqrt(target_area / cur_area)
        return (max(step, int(width * scale) // step * step),
                max(step, int(height * scale) // step * step))

    def _resize_image(self, img, target_area):
        """Resize to ~`target_area` PIXELS, preserving aspect ratio. Never upscales.
        Returns (img, resized_bool).

        Area, not longest edge: training chooses its resolution by area and — with No Upscale on,
        the default — leaves any image already at or under the target exactly as it is. A
        longest-edge cap therefore pushed every non-square image permanently below the training
        target (a 3:4 photo trained at 75% of the pixels it could have, 16:9 at 56%; issue #44).

        Both sides are floored to the bucket step (16). Training floors to that grid anyway, so
        doing it here makes the saved file exactly what trains, and lands just under the target
        area — which keeps training's no-upscale path and means the cache step resamples and
        crops nothing at all."""
        width, height = img.size
        cur_area = width * height
        if cur_area <= target_area:
            return img, False                      # never upscale — it would only invent detail
        step = self._bucket_step()
        scale = math.sqrt(target_area / cur_area)
        new_width = max(step, int(width * scale) // step * step)
        new_height = max(step, int(height * scale) // step * step)
        if (new_width, new_height) == (width, height):
            return img, False
        return img.resize((new_width, new_height), Image.LANCZOS), True

    def _select_face(self, faces, face_mode):
        """Select a face from detected faces based on face_mode. Returns (FaceInfo, note_str) or (None, note_str)."""
        if not faces:
            return None, ""
        if face_mode == "largest_male":
            selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            note = "  Note: No male face, using largest face\n" if (selected and selected.gender != "male") else ""
        elif face_mode == "largest_female":
            selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            note = "  Note: No female face, using largest face\n" if (selected and selected.gender != "female") else ""
        else:
            selected = self.face_detector.get_largest(faces)
            note = ""
        return selected, note

    def _get_next_facecrop_index(self, folder):
        """Find next available FaceCrop_NNN index in a folder."""
        import glob as glob_module
        existing = glob_module.glob(os.path.join(glob_module.escape(folder), "FaceCrop_*.png"))
        max_idx = 0
        for f in existing:
            basename = os.path.splitext(os.path.basename(f))[0]
            parts = basename.split("_")
            if len(parts) >= 2:
                try:
                    max_idx = max(max_idx, int(parts[1]))
                except ValueError:
                    pass
        return max_idx + 1

    def _log(self, text):
        """Append text to the convert log (preserves user scroll position). Marshals to the
        main thread — Tk widget writes from a worker are a hard crash, not an exception."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._log, text)
            return
        self._append_global_log(text)
        try:
            at_bottom = self.convert_log.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        self.convert_log.insert(tk.END, text)
        if at_bottom:
            self.convert_log.see(tk.END)
        self.convert_log.see(tk.END)
        self.master.update_idletasks()

    # endregion

    # region Dataset Analysis & Smart Defaults

    def _analyze_dataset(self, folder):
        """Analyze a dataset folder: count images, detect face crops, check captions."""
        if not folder or not os.path.isdir(folder):
            return None

        files = self._get_image_files(folder)
        face_crops = 0
        full_shots = 0
        for f in files:
            basename = os.path.splitext(os.path.basename(f))[0]
            if basename.startswith("FaceCrop_"):
                face_crops += 1
            else:
                full_shots += 1

        # Count caption files
        caption_count = 0
        for f in os.listdir(folder):
            if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f)):
                caption_count += 1

        return {
            "total_images": len(files),
            "face_crops": face_crops,
            "full_shots": full_shots,
            "has_captions": caption_count > 0,
            "caption_count": caption_count,
        }

    def _recommend_training_settings(self, analysis):
        """Recommend rank, LR, and epochs based on dataset analysis.
        Based on empirical findings from the Fizgig Expansion Vision document."""
        if analysis is None:
            return None

        total = analysis["total_images"]
        face_crops = analysis["face_crops"]

        if total >= 80 and face_crops >= 30:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0004, "epochs": 12,
                "tier": "optimal",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Strong dataset for rank 4:4. Fast convergence expected.",
            }
        elif total >= 40 and face_crops >= 15:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0003, "epochs": 16,
                "tier": "good",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Good dataset for rank 4:4. Slightly conservative LR recommended.",
            }
        elif total >= 20:
            warnings = []
            if face_crops < 15:
                warnings.append("Few face crops — use Auto Prep on the Image Prep tab to generate more.")
            return {
                "rank": 8, "alpha": 8, "lr": 0.0002, "epochs": 25,
                "tier": "caution",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Small dataset. Rank 8 recommended over rank 4.",
                "warnings": warnings,
            }
        else:
            return {
                "rank": 16, "alpha": 16, "lr": 0.0001, "epochs": 40,
                "tier": "limited",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Very small dataset. Higher rank needed to avoid underfitting.",
                "warnings": ["Very small dataset. Results may be inconsistent. Add more training images if possible."],
            }

    def _update_dataset_recommendation(self, *args):
        """Analyze dataset and update the recommendation panel on the Training tab."""
        if not hasattr(self, '_rec_summary_var'):
            return  # UI not built yet

        # Authoritative training folder lives on the Start tab (self.image_folder_var).
        folder = self.image_folder_var.get()
        analysis = self._analyze_dataset(folder)
        rec = self._recommend_training_settings(analysis)

        if rec is None:
            self._rec_summary_var.set("")
            self._rec_detail_var.set("")
            self._rec_warning_var.set("")
            self._last_recommendation = None
            return

        self._last_recommendation = rec

        # Tier colors
        tier_prefix = {"optimal": "Optimal", "good": "Good", "caution": "Caution", "limited": "Limited"}
        self._rec_summary_var.set(f"Dataset: {rec['summary']}")
        self._rec_detail_var.set(
            f"Recommended: rank {rec['rank']}:{rec['alpha']}, LR {rec['lr']}, ~{rec['epochs']} epochs  [{tier_prefix[rec['tier']]}]"
        )

        # Warnings
        warnings = rec.get("warnings", [])
        # Also check current rank vs recommendation
        try:
            current_rank = int(self.entries.get("NETWORK_DIM", tk.Entry()).get())
        except (ValueError, AttributeError):
            current_rank = 0
        if current_rank > 0 and current_rank <= 4 and rec["rank"] > 4:
            warnings.append(f"Current rank {current_rank} may be too low for this dataset size. Recommended: {rec['rank']}.")
        if analysis and not analysis["has_captions"]:
            warnings.append("No caption files (.txt) found — captions are required for training.")

        self._rec_warning_var.set("\n".join(warnings) if warnings else "")

    def _apply_recommendation(self):
        """Apply recommended settings to the training fields."""
        rec = getattr(self, '_last_recommendation', None)
        if rec is None:
            return

        if "NETWORK_DIM" in self.entries:
            self.entries["NETWORK_DIM"].delete(0, tk.END)
            self.entries["NETWORK_DIM"].insert(0, str(rec["rank"]))
        if "NETWORK_ALPHA" in self.entries:
            self.entries["NETWORK_ALPHA"].delete(0, tk.END)
            self.entries["NETWORK_ALPHA"].insert(0, str(rec["alpha"]))
        if "LEARNING_RATE" in self.entries:
            self.entries["LEARNING_RATE"].delete(0, tk.END)
            self.entries["LEARNING_RATE"].insert(0, str(rec["lr"]))
        if "MAX_TRAIN_EPOCHS" in self.entries:
            self.entries["MAX_TRAIN_EPOCHS"].delete(0, tk.END)
            self.entries["MAX_TRAIN_EPOCHS"].insert(0, str(rec["epochs"]))

        self._update_dataset_recommendation()  # Refresh warnings

    # endregion

    def convert_images(self):
        """Prepare images based on selected prep mode."""
        self._originals_dir_cache = {}  # Reset per run
        source_folder = self.image_folder_var.get()
        output_folder = self.convert_output_var.get() or source_folder
        # Target AREA in pixels, from the megapixel selector (see _resize_image for why area).
        try:
            target_area = self._prep_target_area(float(self.prep_megapixels_var.get()))
        except (ValueError, tk.TclError):
            target_area = self._prep_target_area(1.0)
        replace_originals = self.delete_originals_var.get()
        prep_mode = self.prep_mode_var.get()
        face_mode = self._get_face_selection_mode()

        try:
            face_padding = float(self.face_padding_var.get())
        except ValueError:
            face_padding = 20.0

        if not source_folder:
            messagebox.showerror("Error", "Please select a source folder.")
            return
        if not os.path.isdir(source_folder):
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        # Check face detection for modes that need it
        if prep_mode != "Resize Only" and not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available.\nRun install_fizgig.py to enable.")
            return

        os.makedirs(output_folder, exist_ok=True)

        if getattr(self, "_prep_running", False):
            messagebox.showinfo("Already running", "An image prep job is already running.")
            return

        # Clear log
        self.convert_log.configure(state="normal")
        self.convert_log.delete(1.0, tk.END)

        # Worker thread, NOT inline: face detection is ONNX inference per image plus full-size
        # PIL decode/encode, and running the batch on the Tk main thread froze the whole window
        # ("Not Responding") for minutes on a big folder. Every Tk read happened above; the
        # workers only touch the UI through _log, which already marshals via after() — it was
        # written for this thread and waiting for it. The button is disabled for the duration
        # so the job can't be double-started.
        self._prep_running = True
        try:
            self.prepare_images_btn.config(state="disabled", text="Preparing…")
        except Exception:
            pass

        def _prep_worker():
            try:
                if prep_mode == "Auto Prep (Face Crops)":
                    self._auto_prep_images(source_folder, output_folder, target_area, face_mode, face_padding, replace_originals)
                elif prep_mode == "Resize Only":
                    self._resize_only_images(source_folder, output_folder, target_area, replace_originals)
                elif prep_mode == "Face Crop Only":
                    self._face_crop_only_images(source_folder, output_folder, target_area, face_mode, face_padding, replace_originals)
            except Exception as e:
                self._log(f"\nERROR: prep failed — {type(e).__name__}: {e}\n")
            finally:
                self.master.after(0, self._prep_finished)

        threading.Thread(target=_prep_worker, daemon=True).start()

    def _prep_finished(self):
        """Main-thread epilogue for a prep run: finalize the log, re-arm the button."""
        self._prep_running = False
        try:
            self.prepare_images_btn.config(state="normal", text="✨ Prepare Images Now")
        except Exception:
            pass
        self.convert_log.configure(state="disabled")
        self.convert_log.see(tk.END)

    def _resize_only_images(self, source_folder, output_folder, target_area, replace_originals):
        """Resize Only mode: convert/resize images, no face detection."""
        self._log("Mode: Resize Only\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                img, resized = self._resize_image(img, target_area)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")

    def _face_crop_only_images(self, source_folder, output_folder, target_area, face_mode, face_padding, replace_originals):
        """Face Crop Only mode: face crop replaces the output."""
        self._log(f"Mode: Face Crop Only ({face_mode}, padding {face_padding}%)\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors, face_crops, no_face = 0, 0, 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                cropped = False
                crop_info = ""

                try:
                    faces = self.face_detector.detect_from_pil(img)
                    if faces:
                        selected, note = self._select_face(faces, face_mode)
                        if note:
                            self._log(note)
                        if selected:
                            img = crop_to_face(img, selected, face_padding)
                            cropped = True
                            face_crops += 1
                            crop_info = f" [face: {selected.gender}]"
                    else:
                        self._log(f"  No face in {filename}, skipping crop\n")
                        no_face += 1
                except Exception as fe:
                    self._log(f"  Face error ({filename}): {fe}\n")

                img, resized = self._resize_image(img, target_area)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized and not cropped:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if (resized or cropped) else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]{crop_info}\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops: {face_crops} | No face: {no_face}\n")

    def _auto_prep_images(self, source_folder, output_folder, target_area, face_mode, face_padding, replace_originals):
        """Auto Prep mode: resize originals + generate face crops from the HIGH-RES original
        (before it gets overwritten/moved), then handle originals."""
        self._log(f"Mode: Auto Prep (Face Crops)\n")
        self._log(f"Face target: {face_mode}, padding: {face_padding}%\n")
        self._log(f"Output: {output_folder}\n\n")

        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0
        face_crops_created, no_face = 0, 0
        crop_index = self._get_next_facecrop_index(output_folder)

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            base_name = os.path.splitext(filename)[0]

            # Skip existing FaceCrop derivatives
            if base_name.startswith("FaceCrop_"):
                self._log(f"Skipped (derivative): {filename}\n")
                skipped += 1
                continue

            try:
                # Load original at full resolution
                original_img = self._load_image(filepath)
                original_size = original_img.size

                # --- Face crop from the HIGH-RES original (before resize) ---
                try:
                    faces = self.face_detector.detect_from_pil(original_img)
                    if faces:
                        selected, note = self._select_face(faces, face_mode)
                        if note:
                            self._log(note)
                        if selected:
                            cropped = crop_to_face(original_img, selected, face_padding)
                            cropped, _ = self._resize_image(cropped, target_area)
                            crop_name = f"FaceCrop_{crop_index:03d}.png"
                            crop_path = os.path.join(output_folder, crop_name)
                            cropped.save(crop_path, "PNG")
                            cw, ch = cropped.size
                            self._log(f"Face crop: {crop_name} ({cw}x{ch}) from {filename} ({original_size[0]}x{original_size[1]}) [{selected.gender}]\n")
                            face_crops_created += 1
                            crop_index += 1
                            cropped.close()
                        else:
                            no_face += 1
                    else:
                        self._log(f"No face: {filename}\n")
                        no_face += 1
                except Exception as e:
                    self._log(f"Face crop error ({filename}): {e}\n")

                # --- Resize and save the main image ---
                resized_img, resized = self._resize_image(original_img, target_area)
                w, h = resized_img.size
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"OK (no changes): {filename}\n")
                    skipped += 1
                    resized_img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                # Keep-safe + in-place PNG means this save OVERWRITES the original and
                # _handle_original below has nothing left to move — the issue #43 failure, which
                # was fixed in the other two modes but not here, i.e. in the DEFAULT one.
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(resized_img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                resized_img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\n")
        self._log(f"Originals converted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops created: {face_crops_created} | No face: {no_face}\n")
        self._log(f"Total files in output: {len(self._get_image_files(output_folder))}\n")

    @staticmethod
    def _cache_dir_for(cache_root: str, image_dir: str) -> str:
        """`<cache_root>/<folder name>-<hash of full path>` — one cache dir per image folder.

        The trainer builds its item list by GLOBBING the cache directory, so two datasets sharing
        one folder would train on each other's leftovers; the dataset layer refuses outright
        (dataset/config.py: "cache_directory must be unique for each dataset"). The hash keeps it
        stable per folder and unique across same-named folders, and normalises case and trailing
        slash — which is also why the GUI must treat `C:\\A` and `c:/a/` as the SAME folder when
        validating Multi Concept."""
        import hashlib
        norm = image_dir.lower().replace("\\", "/").rstrip("/")
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
        nm = "".join(c if (c.isalnum() or c in "-_") else "_"
                     for c in os.path.basename(image_dir.rstrip("/\\"))) or "dataset"
        return os.path.join(cache_root, f"{nm}-{h}")

    def _dataset_folders(self) -> list:
        """Every image folder that should become a `[[datasets]]` block, in order.

        Normally just the Start-tab folder. Multi Concept (MiniMax only) appends the extra
        concept folders, so each subject gets its own block — which is what makes reference
        distillation pair each image only with OTHERS OF ITS OWN SUBJECT (the rotation in
        scripts/minimax_cache_text.py runs per dataset block).

        The Start folder stays the single source of truth for Captions, Image Prep, the Look
        filter and the gallery; only the TOML writer and validation ever see this list."""
        folders = [self.image_folder_var.get().strip()]
        if (getattr(self, "minimax_multiconcept_var", None) is not None
                and self.minimax_multiconcept_var.get() and self._is_minimax_arch()):
            for var in getattr(self, "_concept_folder_vars", []):
                extra = var.get().strip()
                # Skip blanks and duplicates — the dataset layer hard-fails on a repeated
                # cache_directory, and two spellings of one path hash to the same place.
                if not extra:
                    continue
                norm = extra.lower().replace("\\", "/").rstrip("/")
                if norm in [f.lower().replace("\\", "/").rstrip("/") for f in folders]:
                    continue
                folders.append(extra)
        return [f for f in folders if f]

    def auto_save_dataset_config_silent(self):
        """Write the dataset TOML on startup and on every relevant edit (no Save button)."""
        if _persist_disabled():
            return
        try:
            built = self._build_dataset_toml_text()
            if built is None:
                return
            dataset_name, toml_content = built
            output_path = os.path.join(DATASET_DIR, f"{dataset_name}.toml")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(toml_content)
            self._dataset_config_var.set(output_path)
            # Deliberately NOT settings["DATASET_CONFIG"] (#98): during a run that key
            # points at the run's frozen snapshot, and this writer fires on every edit —
            # re-pointing it mid-pipeline is exactly the dataset-swap race. The launch
            # collects the live path from _dataset_config_var itself.
        except Exception:
            pass  # Silently fail - user can manually save if needed

    _RUN_SNAPSHOT_DIRNAME = "run_snapshots"

    def _snapshot_dataset_config_for_run(self, live_path, resuming=False, prev_config=None):
        """Copy the dataset TOML to an immutable per-run file and return its path (#98).

        Each launched run trains from its own frozen copy of the config, so edits made on
        the Start tab while a run initialises (or trains) can never retarget it. On
        resume, the run's EXISTING snapshot (prev_config — captured by the launch BEFORE
        the settings collection overwrites the key with the live path) is kept: a paused
        run must finish on the dataset it started with, not whatever the Start tab shows
        now. Any failure falls back to the live path — the pre-#98 behaviour, never
        worse. Deliberately NOT _persist_disabled-guarded: snapshots are ephemeral,
        pruned, gitignored copies, and the guard would make this untestable — headless
        tests patch DATASET_DIR instead."""
        import shutil as _shutil
        import time as _time
        try:
            if resuming:
                prev = str(prev_config or "")
                if (os.path.basename(os.path.dirname(prev)) == self._RUN_SNAPSHOT_DIRNAME
                        and os.path.isfile(prev)):
                    return prev
            if not live_path or not os.path.isfile(live_path):
                return live_path
            snap_dir = os.path.join(DATASET_DIR, self._RUN_SNAPSHOT_DIRNAME)
            os.makedirs(snap_dir, exist_ok=True)
            _name = re.sub(r"[^A-Za-z0-9._-]+", "_",
                           str(self.settings.get("LORA_NAME", "") or "run")) or "run"
            snap = os.path.join(snap_dir, f"{_name}-{int(_time.time() * 1000)}.toml")
            _shutil.copyfile(live_path, snap)
        except Exception:
            return live_path
        # Prune AFTER the snapshot is secured, in its own guard — a prune hiccup must
        # never un-freeze the run (the copy above already succeeded). Rule: the newest 12
        # always stay, and older files go only once they are ALSO older than 30 days —
        # so a paused run's frozen config outlives any burst of launches, whatever
        # output dir its sidecar lives in (no sidecar lookup: at this point the settings
        # already describe the LAUNCHING run, not the paused one).
        try:
            import glob as _glob
            _cutoff = _time.time() - 30 * 86400
            olds = sorted(_glob.glob(os.path.join(_glob.escape(snap_dir), "*.toml")),
                          key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else 0),
                          reverse=True)
            for p in olds[12:]:
                try:
                    if (os.path.normpath(p) != os.path.normpath(snap)
                            and os.path.getmtime(p) < _cutoff):
                        os.remove(p)
                except OSError:
                    pass
        except Exception:
            pass
        return snap

    def _verify_frozen_dataset_config(self, path):
        """-> list of Start-tab folders MISSING from the frozen TOML, or None when all
        present (#98 follow-up). The auto-saver skips its rewrite silently when a dataset
        field fails to parse, so without this check a launch could freeze — and train —
        the PREVIOUS dataset under the new run's name. Unreadable/absent file returns
        None: existence is validate_inputs' job, and refusing here would double-report."""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return None

        def _norm(p):
            return str(p).strip().lower().replace("\\", "/").rstrip("/")

        listed = {_norm(m) for m in
                  re.findall(r'^\s*image_directory\s*=\s*"([^"]*)"', text, re.M)}
        if not listed:
            return None                    # not a TOML this writer produced — don't judge it
        missing = [f for f in self._dataset_folders() if f and _norm(f) not in listed]
        return missing or None

    def _build_dataset_toml_text(self):
        """-> (dataset_name, toml text), or None when the config is not writable yet.

        Split out of auto_save_dataset_config_silent so the CONTENT can be tested without
        touching the filesystem: the writer is guarded by _persist_disabled(), and defeating
        that guard in a test is how the real prefs got clobbered once already."""
        if True:
            dataset_name = self.dataset_name_var.get().strip()
            dataset_type = self.dataset_type_var.get()

            # Skip if no dataset name
            if not dataset_name:
                return

            # Check for invalid chars
            invalid_chars = '<>:"/\\|?*'
            if any(c in dataset_name for c in invalid_chars):
                return

            is_video = "Video" in dataset_type
            is_jsonl = "JSONL" in dataset_type

            # Check required fields exist
            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip()
                if not jsonl_file or not os.path.exists(jsonl_file):
                    return
            else:
                if is_video:
                    data_dir = self.dataset_video_dir_var.get().strip()
                else:
                    data_dir = self.image_folder_var.get().strip()
                if not data_dir or not os.path.exists(data_dir):
                    return

            # Validate numeric fields
            try:
                megapixels = float(self.dataset_megapixels_var.get())
                if megapixels <= 0:
                    return
                side = int(math.sqrt(megapixels * 1_000_000))
                side = (side // 16) * 16
                res_width = side
                res_height = side
                batch_size = int(self.dataset_batch_size_var.get())
                num_repeats = 1  # hardcoded — UI removed (Klein workflow always uses 1)
            except ValueError:
                return

            # Build TOML string
            toml_lines = ["[general]"]
            toml_lines.append(f"resolution = [{res_width}, {res_height}]")

            if not is_jsonl:
                caption_ext = self.dataset_caption_ext_var.get().strip()
                toml_lines.append(f'caption_extension = "{caption_ext}"')

            toml_lines.append(f"batch_size = {batch_size}")
            toml_lines.append(f"num_repeats = {num_repeats}")
            toml_lines.append(f"enable_bucket = {'true' if self.dataset_enable_bucket_var.get() else 'false'}")
            toml_lines.append(f"bucket_no_upscale = {'true' if self.dataset_no_upscale_var.get() else 'false'}")
            toml_lines.append("")
            toml_lines.append("[[datasets]]")

            # Cache directory is now sourced from Preferences (no longer a Dataset-tab field).
            # Each dataset gets its OWN subfolder: the trainer builds its item list by globbing
            # the cache directory, so two datasets sharing one folder would train on each other's
            # leftovers. <folder name>-<hash of full path> keeps it stable per dataset and unique
            # across same-named folders; switching datasets keeps both caches warm.
            cache_dir = self.prefs_vars["cache_dir"].get().strip() if "cache_dir" in self.prefs_vars else ""
            if cache_dir and not is_jsonl and not is_video:
                _cache_img_dir = self.image_folder_var.get().strip()
                if _cache_img_dir:
                    cache_dir = self._cache_dir_for(cache_dir, _cache_img_dir)

            # MiniMax uses ONE dataset at the Target Megapixels you set, exactly like Klein and
            # Krea 2. It briefly mirrored ai-toolkit's resolution: [512, 768, 1024], which copies
            # the dataset once per scale — every image trained three times per epoch. Dropped
            # (Peter, 4 Aug): with a bucketed dataset the scale variation is already there, and
            # tripling exposure to the same images per epoch is a much better way to overfit than
            # to teach scale invariance — most of all on tight face crops, where the extra copies
            # add no compositional diversity at all. It also silently tripled the work behind the
            # Epochs box, so "50 epochs" stopped meaning what it used to.
            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip().replace("\\", "/")
                if is_video:
                    toml_lines.append(f'video_jsonl_file = "{jsonl_file}"')
                else:
                    toml_lines.append(f'image_jsonl_file = "{jsonl_file}"')
            else:
                if is_video:
                    video_dir = self.dataset_video_dir_var.get().strip().replace("\\", "/")
                    toml_lines.append(f'video_directory = "{video_dir}"')
                else:
                    # One block per concept folder. Normally a single folder, so the output is
                    # byte-identical to the old single-block writer; Multi Concept adds a block
                    # per extra folder, each with its OWN cache directory (which is what keeps
                    # the reference rotation inside one subject).
                    _folders = self._dataset_folders()
                    _root = (self.prefs_vars["cache_dir"].get().strip()
                             if "cache_dir" in self.prefs_vars else "")
                    for _i, _folder in enumerate(_folders):
                        if _i:                       # the first block's header is already down
                            toml_lines.append("")
                            toml_lines.append("[[datasets]]")
                        toml_lines.append(
                            f'image_directory = "{_folder.replace(chr(92), "/")}"')
                        _cd = self._cache_dir_for(_root, _folder) if _root else ""
                        if _cd:
                            toml_lines.append(
                                f'cache_directory = "{_cd.replace(chr(92), "/")}"')
                    cache_dir = ""                   # emitted per block above

            if cache_dir:
                toml_lines.append(f'cache_directory = "{cache_dir.replace(chr(92), "/")}"')

            if is_video:
                try:
                    target_frames = [int(x.strip()) for x in self.dataset_target_frames_var.get().split(",")]
                    toml_lines.append(f"target_frames = [{', '.join(str(f) for f in target_frames)}]")
                    toml_lines.append(f'frame_extraction = "{self.dataset_frame_extraction_var.get()}"')
                    source_fps = float(self.dataset_source_fps_var.get())
                    toml_lines.append(f"source_fps = {source_fps}")
                except ValueError:
                    pass

            # Optional regularisation set (fine-tune only, per family): a second dataset
            # block marked is_reg, so the cache scripts pick it up for free and the trainer
            # can find its items. Only written when a folder is set — no folder, no block,
            # nothing changes. Fine-tune only: with FT off the block must not be written at
            # all, or the reg images would be cached and trained as ordinary subjects at
            # full LR. Arch-scoped: each family's reg row + FT toggle only speak for their
            # own family (a stale toggle from the other family must not leak a block in).
            if self._is_minimax_arch():
                reg_dir = (self.minimax_reg_dir_var.get().strip().replace("\\", "/")
                           if hasattr(self, "minimax_reg_dir_var") else "")
                reg_on = bool(getattr(self, "minimax_finetune_var", None)
                              and self.minimax_finetune_var.get())
            else:
                reg_dir = (self.krea2_reg_dir_var.get().strip().replace("\\", "/")
                           if hasattr(self, "krea2_reg_dir_var") else "")
                reg_on = bool(self._is_krea2_arch()
                              and getattr(self, "krea2_finetune_var", None)
                              and self.krea2_finetune_var.get())
            if reg_on and reg_dir and os.path.isdir(reg_dir) and not is_jsonl and not is_video:
                toml_lines.append("")
                toml_lines.append("[[datasets]]")
                toml_lines.append(f'image_directory = "{reg_dir}"')
                _reg_cache = self.prefs_vars["cache_dir"].get().strip() if "cache_dir" in self.prefs_vars else ""
                if _reg_cache:
                    # Its own subfolder for the same reason the subject set gets one: the
                    # trainer globs the cache dir, so a shared folder mixes the two sets.
                    _reg_cache = self._cache_dir_for(_reg_cache, reg_dir)
                    toml_lines.append(f'cache_directory = "{_reg_cache.replace(chr(92), "/")}"')
                toml_lines.append("is_reg = true")

            return dataset_name, "\n".join(toml_lines) + "\n"

    def run_subprocess(self, cmd, name, callback=None):
        """Run a subprocess and handle its output with UTF-8 encoding"""
        env = self._cuda_env_for_subprocess(os.environ.copy())
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"  # flush stdout/stderr line-by-line so log output streams live

        if os.name == 'nt':
            # BELOW_NORMAL: Windows weights GPU scheduling by process priority class, and the
            # desktop compositor renders on the same card that training saturates. Below-normal
            # gives DWM the preemption slices it needs (fixes juddery mouse/desktop during a run)
            # and costs training ~1% — it only yields when something else actually wants time.
            creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                             | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn
        )
        self.current_process = process
        if os.name == 'nt':
            self.process_group_id = process.pid

        def read_output(pipe, output_type):
            """Read subprocess output line by line"""
            while True:
                line = pipe.readline()
                if not line:
                    break
                self.master.after(0, self.update_console, line)
            pipe.close()

        threading.Thread(target=read_output, args=(process.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, "STDERR"), daemon=True).start()

        def check_process():
            """Check subprocess completion"""
            process.wait()
            self.master.after(0, self.update_console, f"{name} process completed.\n")
            self.current_process = None
            # Route pipeline exits through the pause/resume state machine (see the predicate
            # for which ones — a dead caching phase used to strand the app in "running").
            if self._pipeline_exit_routes_to_state_machine(name, process.returncode):
                self.master.after(0, self._on_training_subprocess_exited, process.returncode)
            if process.returncode != 0:
                self.master.after(0, self.update_console,
                    f"ERROR: {name} failed with exit code {process.returncode}. Pipeline stopped.\n")
                self.master.after(0, self.stop_samples_watcher)
                return
            if callback:
                # Marshal to the Tk main thread: pipeline-chain callbacks touch Tk widgets
                # (update_console etc.), and Tk calls off the main thread segfault on Linux.
                self.master.after(0, callback)

        threading.Thread(target=check_process, daemon=True).start()

    def start_training(self):
        """Start training with sequential cache process execution"""
        # Re-entrancy guard: the Start button stays enabled during a run, so a double-click
        # (or Start during caching) overwrote current_process and ORPHANED the first launch —
        # stop_training only ever kills the current one, and both runs wrote the same
        # checkpoints while the second's fresh-run wipe deleted the first's watch files.
        _proc = getattr(self, "current_process", None)
        try:
            if _proc is not None and _proc.poll() is None:
                # A run is active, so Start means QUEUE: capture the currently configured
                # run and append it. (The button already reads "Queue Train" in this state.)
                self._queue_current_run()
                return
        except Exception:
            pass

        # Validate inputs before starting
        if not self.validate_inputs():
            return

        # Resuming a state that's already at the final epoch trains nothing — the trainer's epoch
        # loop is empty and it just rewrites the final LoRA. A warning rather than a block,
        # because that fall-through is exactly how a run paused ON its last epoch gets completed.
        if not self._confirm_resume_has_epochs_left():
            return

        if not self._confirm_disk_headroom():
            return

        # Clear a stale pause sentinel from a previous session (window close / crash after
        # Pause left it on disk; the trainer would read it at epoch 1 and exit "cleanly").
        try:
            _stale_flag = self._pause_flag_path()
            if os.path.exists(_stale_flag) and getattr(self, "training_state", "idle") != "paused":
                os.remove(_stale_flag)
                self.update_console("[pause] removed stale .pause_requested from a previous session\n")
        except Exception:
            pass

        # Reset OOM warning flag for this run
        self._oom_warning_shown = False
        # Reset the VRAM/RAM peak markers so the status bar tracks THIS run.
        try:
            self.reset_status_peaks()
        except Exception:
            pass
        # Sync the sample-override sentinel to the current toggle (clears any
        # stale file from a previous session so it matches what the user sees).
        try:
            self._on_sample_override_changed()
        except Exception:
            pass

        # Auto-uncheck FP8 Base if the Base DiT file is already fp8-quantised (Klein only —
        # Krea 2 reads its own RAW DiT and dynamic-quantizes it, so this must not fire there).
        _is_krea2_run = ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_krea2", False)
        base_dit_path = self.prefs_vars.get("base_dit", tk.StringVar()).get()
        if not _is_krea2_run and "fp8" in os.path.basename(base_dit_path).lower() and self.fp8_var.get():
            self.fp8_var.set(False)
            self.scaled_var.set(False)
            self.toggle_scaled()

        # Snapshot current settings for the "Load Last Train" button
        self._save_last_train_settings()
        # ...and as the queue window's pinned "training now" card: editing a queued job loads
        # its settings into this tab, so the window needs a way back to the run in progress.
        self._active_run_item = self._queue_snapshot()

        if getattr(self, "_training_start_pending", False):
            return
        if self._caption_worker_alive():
            self._training_start_pending = True
            self._caption_worker_released_for_training = True
            try:
                self._start_training_btn.configure(state=tk.DISABLED)
            except Exception:
                pass
            self._stop_caption_worker_async(self._start_training_launch, graceful=False)
            return

        self._start_training_launch()

    def _start_training_launch(self):
        """Launch training after validations and any caption-worker VRAM release."""
        self._training_start_pending = False
        try:
            self._start_training_btn.configure(state=tk.NORMAL)
        except Exception:
            pass
        # ...and the tool-tab engines (Repair Studio / Explorer / Royale, 10-20 GB each).
        # A manual Start implies a switch to the Training tab, which unloads them via
        # on_tab_changed — but a queue auto-advance or the queue window's "Start next now"
        # involves NO tab switch, and training would otherwise launch against a full card.
        # All three are idle-guarded internally, so this is safe and idempotent.
        for _unl in ("_unload_repair_studio_models", "_unload_explorer_models", "_royale_unload"):
            try:
                getattr(self, _unl)()
            except Exception:
                pass

        # Start samples watcher for live gallery updates
        if self.sample_enabled_var.get():
            self.start_samples_watcher()

        # Clear cache directory before training — but NOT when resuming: the cache is already
        # built and we skip re-caching, so wiping it would leave the resumed run with no latents
        # /text. Read the resume path from the entry (the live source of truth at this point).
        _resume_entry = self.entries.get("RESUME_TRAINING")
        # An armed FT continuation is a resume too — same cache, same frozen dataset.
        _is_resuming_clear = bool((_resume_entry and _resume_entry.get().strip())
                                  or self._ft_resume_active())
        cache_dir = self.dataset_cache_dir_var.get().strip()
        if cache_dir and os.path.isdir(cache_dir) and not _is_resuming_clear:
            try:
                import shutil
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                self.update_console(f"Warning: Could not clear cache: {e}\n")


        # Get current architecture
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Validate blocks swap
        try:
            is_auto = self.entries["BLOCKS_SWAP"].get().strip().lower().startswith("auto")
            if config.get("is_minimax") and is_auto:
                # MiniMax resolves "auto" in the TRAINER from real free VRAM at run time (correct
                # for queued runs too) — the Klein/Krea2 tier tables here don't fit its NF4 base.
                blocks_swap = "auto"
                self.update_console("Block Swap: Auto — the trainer plans swap + checkpointing "
                                    "from free VRAM at launch\n")
            else:
                blocks_swap = self._parse_blocks_swap()
                if is_auto:
                    self.update_console(f"Block Swap: Auto detected → {blocks_swap} (based on GPU VRAM)\n")
                if blocks_swap > config["blocks_swap_max"]:
                    messagebox.showwarning(
                        "Warning",
                        f"Blocks Swap value ({blocks_swap}) exceeds maximum for {arch} ({config['blocks_swap_max']}). Using maximum value."
                    )
                    blocks_swap = config["blocks_swap_max"]
                    self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                    self.entries["BLOCKS_SWAP"].insert(0, str(blocks_swap))
        except ValueError:
            blocks_swap = config["blocks_swap_max"]

        # Update settings from entries
        # Path keys read via _get_path() (sourced from prefs_vars or hidden _dataset_config_var)
        # since the Model Paths section is no longer visible on the Training tab.
        # With Adaptive LR on the Learning Rate box is ignored (the trainer starts at the
        # geometric midpoint of Min/Max) — a stale value in the disabled box must not crash
        # collection, so fall back to a harmless placeholder the trainer will override.
        try:
            _lr_val = float(self.entries["LEARNING_RATE"].get())
        except ValueError:
            if hasattr(self, 'adaptive_lr_var') and self.adaptive_lr_var.get():
                _lr_val = 1e-4  # ignored by the trainer under adaptive
            else:
                raise
        # Captured BEFORE the update below overwrites DATASET_CONFIG with the live editor
        # path: on a resume this still holds the paused run's frozen snapshot (in-session
        # from the original launch, cross-restart from the startup sidecar restore), and
        # the freeze call needs it — reading settings AFTER the update made the resume
        # keep-rule dead code (three independent review agents, same finding).
        _prev_dataset_config = str(self.settings.get("DATASET_CONFIG", "") or "")
        self.settings.update({
            "ARCHITECTURE": arch,
            "MODEL_TYPE": self.entries["MODEL_TYPE"].get() if config["uses_model_type"] else "",
            "LEARNING_RATE": _lr_val,
            "LORA_LR_RATIO": int(self.entries["LORA_LR_RATIO"].get()),
            "NETWORK_DIM": int(self.entries["NETWORK_DIM"].get()),
            "NETWORK_ALPHA": float(self.entries["NETWORK_ALPHA"].get()),
            "NETWORK_TYPE": self.entries["NETWORK_TYPE"].get(),
            "LOKR_FACTOR": int(self.entries["LOKR_FACTOR"].get() or 8),
            "MAX_TRAIN_EPOCHS": int(self.entries["MAX_TRAIN_EPOCHS"].get()),
            "SAVE_EVERY_N_EPOCHS": int(self.entries["SAVE_EVERY_N_EPOCHS"].get()),
            "SEED": int(self.entries["SEED"].get()),
            "BLOCKS_SWAP": blocks_swap,
            # MiniMax-only; the widget exists (hidden) under every family, so read it unconditionally
            # and let the MiniMax command builder be the one that acts on it.
            # Stored as the percentage the user typed; the command builder converts it to the
            # trainer's shift. Keeping the percentage is what makes a saved preset mean the same
            # thing later, rather than a shift number nobody can interpret.
            "MINIMAX_LOWNOISE_PCT": str(self.entries["MINIMAX_LOWNOISE_PCT"].get() or "").strip(),
            "MINIMAX_HIGHNOISE_LR_PCT": str(
                self.entries["MINIMAX_HIGHNOISE_LR_PCT"].get() or "").strip(),
            # The hand-curated dict trap (5f20ba2): a control missing HERE silently never
            # reaches the trainer, however correct the widgets and the command builder are.
            "MIXED_STOP_CATEGORY": str(
                self.entries["MIXED_STOP_CATEGORY"].get() or "").strip(),
            "MIXED_STOP_EPOCH": str(
                self.entries["MIXED_STOP_EPOCH"].get() or "").strip(),
            "MIXED_STOP_MODE": str(
                self.entries["MIXED_STOP_MODE"].get() or "").strip(),
            # Likeness mode owns the block choice: the launch dict says "all" so the queue card,
            # snapshot and builder stay honest, while the combobox keeps the user's typed spec
            # for when they untick.
            "MINIMAX_BLOCKS": ("all" if self.entries["MINIMAX_LIKENESS_OPT"].get()
                               else minimax_block_spec(self.entries["MINIMAX_BLOCKS"].get())),
            "MINIMAX_LIKENESS_OPT": bool(self.entries["MINIMAX_LIKENESS_OPT"].get()),
            "MINIMAX_FT_CLIP_LIKENESS": bool(self.entries["MINIMAX_FT_CLIP_LIKENESS"].get())
            if "MINIMAX_FT_CLIP_LIKENESS" in self.entries else True,
            "MINIMAX_TRAIN_ADALN": bool(self.entries["MINIMAX_TRAIN_ADALN"].get()),
            "MINIMAX_DISTILL": bool(self.minimax_distill_var.get()),
            # Canonical key ("fl2va"/"ref2va"), never the display label. Preset-immune by
            # design — the var is outside self.entries and _collect_preset_values skips it.
            "MINIMAX_TRAIN_BASE": minimax_train_base(
                getattr(self, "minimax_train_base_var", None)
                and self.minimax_train_base_var.get()),
            "MINIMAX_MULTICONCEPT": bool(self.minimax_multiconcept_var.get()),
            "MINIMAX_CONCEPT_DIRS": [v.get().strip() for v in
                                     getattr(self, "_concept_folder_vars", [])],
            "MINIMAX_BASE_QUANT": self.entries["MINIMAX_BASE_QUANT"].get(),
            "MINIMAX_BLOCK_LIMIT": self.entries["MINIMAX_BLOCK_LIMIT"].get(),
            "MINIMAX_LR_WARMUP": self.entries["MINIMAX_LR_WARMUP"].get(),
            "MINIMAX_EMA": self.entries["MINIMAX_EMA"].get(),
            "MINIMAX_ADAPTER_RAMP": self.entries["MINIMAX_ADAPTER_RAMP"].get(),
            "MINIMAX_CAPTION_DROPOUT": self.entries["MINIMAX_CAPTION_DROPOUT"].get(),
            "MINIMAX_DISTILL_WEIGHT": str(self.entries["MINIMAX_DISTILL_WEIGHT"].get() or "0.8").strip(),
            "MINIMAX_DISTILL_REFS": str(self.entries["MINIMAX_DISTILL_REFS"].get() or "2").strip(),
            "MINIMAX_SLOW_BLOCKS": str(self.entries["MINIMAX_SLOW_BLOCKS"].get() or "").strip(),
            "MINIMAX_SLOW_LR_SCALE": str(self.entries["MINIMAX_SLOW_LR_SCALE"].get() or "1").strip(),
            "DATASET_CONFIG": self._get_path("DATASET_CONFIG"),
            "VAE_MODEL": self._get_path("VAE_MODEL"),
            "CLIP_MODEL": self._get_path("CLIP_MODEL"),
            "T5_MODEL": self._get_path("T5_MODEL"),
            "TEXT_ENCODER": self._get_path("TEXT_ENCODER"),
            "DIT_MODEL": self._get_path("DIT_MODEL"),
            "LORA_OUTPUT_DIR": self.entries["LORA_OUTPUT_DIR"].get(),
            "LORA_NAME": self.entries["LORA_NAME"].get(),
            "RESUME_TRAINING": self.entries["RESUME_TRAINING"].get(),
            "OPTIMIZER_TYPE": self.entries["OPTIMIZER_TYPE"].get(),
            "OPTIMIZER_ARGS": self.entries["OPTIMIZER_ARGS"].get(),
            "ATTENTION_MECHANISM": self.entries["ATTENTION_MECHANISM"].get(),
            "LOGGING_DIR": self.entries["LOGGING_DIR"].get(),
            "LOG_WITH": self.entries["LOG_WITH"].get(),
            "LOG_PREFIX": self.entries["LOG_PREFIX"].get(),
            "IMG_IN_TXT_IN_OFFLOADING": self.entries["IMG_IN_TXT_IN_OFFLOADING"].get(),
            "LR_SCHEDULER": self.entries["LR_SCHEDULER"].get(),
            "LR_WARMUP_STEPS": self.entries["LR_WARMUP_STEPS"].get(),
            "LR_DECAY_STEPS": self.entries["LR_DECAY_STEPS"].get(),
            "GRADIENT_ACCUMULATION": self.entries["GRADIENT_ACCUMULATION"].get(),
            "MAX_GRAD_NORM": self.entries["MAX_GRAD_NORM"].get(),
            "NETWORK_DROPOUT": self.entries["NETWORK_DROPOUT"].get(),
            "CONTEXT_LORA_PATH": self.entries["CONTEXT_LORA_PATH"].get(),
            "CONTEXT_LORA_STRENGTH": self.entries["CONTEXT_LORA_STRENGTH"].get(),
            "TIMESTEP_SAMPLING": self.ts_sampling_var.get(),
            "DISCRETE_FLOW_SHIFT": self.entries["DISCRETE_FLOW_SHIFT"].get(),
            "SIGMOID_SCALE": self.entries["SIGMOID_SCALE"].get(),
            "MIN_TIMESTEP": self.entries["MIN_TIMESTEP"].get(),
            "MAX_TIMESTEP": self.entries["MAX_TIMESTEP"].get(),
            "PRESERVE_DISTRIBUTION": self.preserve_dist_var.get(),
            "ADAPTIVE_LR": self.adaptive_lr_var.get(),
            "ADAPTIVE_LR_MIN": self.entries["ADAPTIVE_LR_MIN"].get(),
            "ADAPTIVE_LR_MAX": self.entries["ADAPTIVE_LR_MAX"].get(),
            "WEIGHTING_SCHEME": self.weighting_scheme_var.get(),
            "LOGIT_MEAN": self.entries["LOGIT_MEAN"].get(),
            "LOGIT_STD": self.entries["LOGIT_STD"].get(),
            "MODE_SCALE": self.entries["MODE_SCALE"].get(),
            "METADATA_TITLE": self.entries["METADATA_TITLE"].get(),
            "METADATA_AUTHOR": self.entries["METADATA_AUTHOR"].get(),
            "METADATA_DESCRIPTION": self.entries["METADATA_DESCRIPTION"].get(),
            "METADATA_LICENSE": self.entries["METADATA_LICENSE"].get(),
            "METADATA_TAGS": self.entries["METADATA_TAGS"].get(),
            "METADATA_TRIGGER_PHRASE": self.entries["METADATA_TRIGGER_PHRASE"].get(),
            "METADATA_THUMBNAIL": self.entries["METADATA_THUMBNAIL"].get(),
            "FP8": self.fp8_var.get(),
            "SCALED": self.scaled_var.get(),
            "QUANT_4BIT": self.quant_4bit_var.get(),
            "COMPILE_BLOCKS": self.compile_blocks_var.get(),
            "GRADIENT_CHECKPOINTING": self.grad_checkpoint_var.get(),
            "FP8_TEXT_ENCODER": self.fp8_text_encoder_var.get(),
            "SAVE_STATE": self.save_state_var.get(),
            "SAVE_STATE_ON_TRAIN_END": self.save_state_on_train_end_var.get(),
            "KEEP_LAST_N_STATES": self.entries["KEEP_LAST_N_STATES"].get(),
            "ENABLE_BUCKET": self.dataset_enable_bucket_var.get(),
            "BUCKET_NO_UPSCALE": self.dataset_no_upscale_var.get(),
        })

        # Freeze THIS run's dataset config (#98): the pipeline's stages each read the
        # dataset TOML at their own start, and the Start tab auto-saves that TOML on
        # every edit — so changing the dataset folder while run 1 initialised (e.g. to
        # queue run 2) retargeted run 1: dataset 2 trained under run 1's name and
        # settings. From here on, settings["DATASET_CONFIG"] is the run's immutable
        # snapshot; the live TOML belongs to the editor alone.
        self.settings["DATASET_CONFIG"] = self._snapshot_dataset_config_for_run(
            self.settings.get("DATASET_CONFIG", ""), resuming=_is_resuming_clear,
            prev_config=_prev_dataset_config)

        # The frozen config must describe the folders on the Start tab (#98 follow-up):
        # a dataset-field parse failure (e.g. Target Megapixels typed as "1,0") makes the
        # auto-saver skip its rewrite SILENTLY, so the launch would freeze a STALE toml —
        # and the previous dataset would train under this run's name. Never on a resume:
        # there the frozen config deliberately predates the Start tab.
        if not _is_resuming_clear:
            _missing = self._verify_frozen_dataset_config(self.settings["DATASET_CONFIG"])
            if _missing:
                self.stop_samples_watcher()
                _msg = ("The dataset config on disk does not include the training "
                        "folder(s) shown on the Start tab:\n\n"
                        + "\n".join(_missing)
                        + "\n\nThis usually means a dataset field failed to parse — check "
                        "Target Megapixels and Batch Size for typos — so the config was "
                        "never rewritten. Fix the value and press Start again.")
                self.update_console(f"[dataset] launch refused — {_msg}\n")
                messagebox.showerror("Dataset config out of date", _msg)
                return

        # Build training command based on architecture
        command = self.build_training_command(config)
        cache_latents_cmd = self.build_cache_latents_command(config)
        cache_text_cmd = self.build_cache_text_command(config)

        self.console_output.configure(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.configure(state="disabled")

        if getattr(self, "_caption_worker_released_for_training", False):
            self._caption_worker_released_for_training = False
            self.update_console("Caption model released.\n")

        def on_training_complete():
            """Called when training finishes - cleanup watchers"""
            self.stop_samples_watcher()

        # On resume, skip cache preparation entirely — the cache is already built from the
        # original launch. An armed FT continuation counts: it is the same run continuing.
        is_resuming = bool(self.settings.get("RESUME_TRAINING", "").strip()
                           or self._ft_resume_active())
        if self.enable_cache_var.get() and not is_resuming:
            self.update_console(f"Starting cache preparation for {arch}...\n")

            def on_text_encoder_caching_complete():
                self.update_console("Text encoder caching completed.\nStarting training...\n")
                self.run_subprocess(command, "Training", on_training_complete)

            def on_cache_preparation_complete():
                self.update_console("Cache preparation completed.\nStarting text encoder caching...\n")
                self.run_subprocess(cache_text_cmd, "Text Encoder Caching", on_text_encoder_caching_complete)

            self.run_subprocess(cache_latents_cmd, "Cache Preparation", on_cache_preparation_complete)
        else:
            if is_resuming:
                self.update_console("Resuming from saved state — skipping cache preparation (cache already built).\n")
            else:
                self.update_console(f"Starting {arch} training without caching...\n")
            self.run_subprocess(command, "Training", on_training_complete)
        # Mark as running for the pause/resume state machine
        self.training_state = "running"
        self._refresh_training_buttons()

    DISK_WARN_GB = 15

    def _confirm_disk_headroom(self):
        """True to proceed. Warns when the output drive is nearly full.

        A threshold plus the REAL figure rather than a predicted requirement: what a run actually
        writes depends on rank, epochs, save cadence and keep-N, and a confidently wrong estimate
        is worse than showing someone the number and letting them judge. Running out of disk four
        hours into a run costs the whole run."""
        out_dir = (self.settings.get("LORA_OUTPUT_DIR") or "").strip()
        if not out_dir:
            return True
        probe = out_dir
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                return True
            probe = parent
        try:
            import shutil as _sh
            free_gb = _sh.disk_usage(probe).free / 1024 ** 3
        except Exception:
            return True                      # never block a run over a failed disk probe
        if free_gb >= self.DISK_WARN_GB:
            return True
        return messagebox.askyesno(
            "Low disk space",
            f"Only {free_gb:.1f} GB free where your LoRAs are saved:\n{probe}\n\n"
            f"A run writes a checkpoint every few epochs, plus resumable state dirs (a few hundred "
            f"MB each) and sample images. Running out part-way through loses the run.\n\n"
            f"Free some space, or lower Save Every N Epochs and Keep Last.\n\n"
            f"Start training anyway?")

    def _confirm_resume_has_epochs_left(self):
        """True to proceed. Warns when the resume state is already at/past Max Train Epochs.

        Deliberately a warning, not a validation error: pausing on the final epoch exits before
        the final LoRA is written, and resuming that state — with zero epochs left to run — is
        precisely what completes it. Blocking would break that recovery."""
        resume_path = (self.entries["RESUME_TRAINING"].get() or "").strip()
        if not resume_path:
            return True
        m = re.search(r"-(\d{6})-state$", os.path.basename(resume_path.rstrip("/\\")))
        if not m:
            return True
        state_epoch = int(m.group(1))
        try:
            max_epochs = int(self.entries["MAX_TRAIN_EPOCHS"].get())
        except (TypeError, ValueError):
            return True
        if state_epoch < max_epochs:
            return True
        return messagebox.askyesno(
            "Nothing left to train",
            f"That state is already at epoch {state_epoch}, and Max Train Epochs is {max_epochs}.\n\n"
            f"Resuming it will not train anything — it will just write the final LoRA from the "
            f"restored state. That is what you want if you paused on the last epoch and are "
            f"finishing the run.\n\n"
            f"To train further, cancel and raise Max Train Epochs above {state_epoch} first.\n\n"
            f"Continue anyway?")

    def _state_flags(self):
        """Save-state CLI flags, shared by both families (the flag names are identical).

        Keep-N is clamped to >= 1 here as well as in the trainer: a blank or zero box must never
        reach a prune that would take the state just written with it."""
        flags = []
        if self.settings.get("SAVE_STATE", True):
            flags.append("--save_state")
        if self.settings.get("SAVE_STATE_ON_TRAIN_END", True):
            flags.append("--save_state_on_train_end")
        if flags:
            try:
                keep_n = max(1, int(str(self.settings.get("KEEP_LAST_N_STATES", 2)).strip()))
            except (TypeError, ValueError):
                keep_n = 2
            flags += ["--keep_last_n_states", str(keep_n)]
        return flags

    def build_training_command(self, config):
        """Build the training command based on architecture configuration"""
        # Stamp whether THIS launch is a rotation fine-tune (and which family), for the
        # pause exit-handler: an FT pause leaves a full checkpoint rather than a state dir,
        # and the Tk checkbox can be flipped mid-run, so the truth is recorded at launch.
        self._launched_ft_family = None
        if config.get("is_krea2"):
            if bool(getattr(self, "krea2_finetune_var", None) and self.krea2_finetune_var.get()):
                self._launched_ft_family = "krea2"
            return self._build_krea2_train_command()
        if config.get("is_minimax"):
            if bool(getattr(self, "minimax_finetune_var", None) and self.minimax_finetune_var.get()):
                self._launched_ft_family = "minimax"
            return self._build_minimax_train_command()
        arch = self.settings["ARCHITECTURE"]
        # Same reasoning as _venv_python: fall back to whatever is on PATH when the bundled venv
        # is not a sibling of the repo, rather than pointing at a file that is not there.
        accelerate_path = (os.path.join(FIZGIG_DIR, "venv", "Scripts", "accelerate.exe")
                           if os.name == 'nt'
                           else os.path.join(FIZGIG_DIR, "venv", "bin", "accelerate"))
        if not os.path.isfile(accelerate_path):
            import shutil as _shutil
            accelerate_path = _shutil.which("accelerate") or accelerate_path
        train_script_path = self._resolve_script(config, "train_script")

        # Auto-detect mixed precision from DiT model filename
        # fp16 model files require fp16 mixed precision, bf16 requires bf16
        dit_path = self.settings["DIT_MODEL"]
        dit_filename = os.path.basename(dit_path).lower()
        if "fp16" in dit_filename:
            mixed_precision = "fp16"
        else:
            mixed_precision = "bf16"

        command = [
            accelerate_path, "launch",
            "--num_cpu_threads_per_process", "2",
            "--mixed_precision", mixed_precision,
            train_script_path,
        ]

        # Architecture-specific parameters
        if arch.startswith("Wan"):
            command.extend(["--task", self.settings["MODEL_TYPE"]])
        elif config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        command.extend([
            "--dit", self.settings["DIT_MODEL"],
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--mixed_precision", mixed_precision,
        ])

        # VAE parameter (same flag for all architectures)
        command.extend(["--vae", self.settings["VAE_MODEL"]])

        # Text encoder parameters based on architecture
        if config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        # Base weight optimization — 4-bit NF4 supersedes fp8 (mutually exclusive).
        if self.settings.get("QUANT_4BIT", False):
            command.append("--quant_4bit")
        elif self.settings["FP8"]:
            command.append("--fp8_base")
            if self.settings["SCALED"]:
                command.append("--fp8_scaled")

        # FP8 text encoder
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        command.extend([
            "--blocks_to_swap", str(self.settings["BLOCKS_SWAP"]),
            "--optimizer_type", self.settings["OPTIMIZER_TYPE"],
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", config["network_module"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--timestep_sampling", self.settings["TIMESTEP_SAMPLING"],
        ])
        # --network_args is nargs="*": a SECOND occurrence would REPLACE the first, so all
        # network args must be emitted as one occurrence (loraplus + include_patterns below).
        network_args_tokens = [f"loraplus_lr_ratio={self.settings['LORA_LR_RATIO']}"]

        # Gradient checkpointing — on by default (recomputes activations in backward
        # to fit a 9B LoRA on most cards). Off trades ~20-30% faster steps for much
        # higher VRAM; only sensible on big cards with no block swap.
        if self.settings.get("GRADIENT_CHECKPOINTING", True):
            command.append("--gradient_checkpointing")

        # Target layers (selective layer training)
        # Block assignments based on empirical testing on Klein 9B:
        #   single_blocks 0-1: composition (layout, structure)
        #   single_blocks 2-11: identity/face (the core face signal)
        #   single_blocks 12-23: style (aesthetic, color, lighting)
        #   double_blocks 0-7: cross-attention (included in All Layers only)
        preset = self.training_preset_var.get() if hasattr(self, 'training_preset_var') else "Full Model"
        STYLE_COMP_PATTERNS = [r".*double_blocks\..*", r".*single_blocks\.[01]\..*"]
        IDENTITY_PATTERNS = [r".*single_blocks\.(1[0-6]|[1-9])\..*"]
        DETAILS_PATTERNS = [r".*single_blocks\.(1[2-9]|2[0-3])\..*"]

        patterns = None
        if preset == "Identity":
            patterns = IDENTITY_PATTERNS
        elif preset in ("Style", "Style+Composition"):
            patterns = STYLE_COMP_PATTERNS
        elif preset == "Details":
            patterns = DETAILS_PATTERNS
        elif preset == "Custom":
            patterns = self._build_custom_training_patterns()
            if patterns is None:
                # Visible warning — print() went to stdout only, invisible under the
                # windowed launcher, and the run silently trained the full model.
                self.update_console("[Warning] Model Area is Custom but no blocks are "
                                    "selected — training the FULL model.\n")
                messagebox.showwarning(
                    "Custom blocks empty",
                    "Model Area to Train is set to Custom but no blocks are ticked.\n\n"
                    "This run will train the FULL model. Tick blocks (or pick a preset) "
                    "if you wanted block targeting.")
        # "Full Model" → patterns stays None (train everything)

        if patterns:
            # Escape backslashes for the shell-parsed network_args value
            quoted = ",".join(f'"{p.replace(chr(92), chr(92) * 2)}"' for p in patterns)
            network_args_tokens.append(f"include_patterns=[{quoted}]")
        # Single --network_args occurrence carrying every token (see note above).
        command.extend(["--network_args"] + network_args_tokens)

        # Discrete flow shift (not for Flux 2 which uses flux2_shift automatic)
        if config.get("supports_discrete_flow_shift", True):
            command.extend(["--discrete_flow_shift", str(self.settings["DISCRETE_FLOW_SHIFT"])])

        # Sigmoid scale (only meaningful for sigmoid/shift sampling)
        ts_sampling = self.settings["TIMESTEP_SAMPLING"]
        sigmoid_scale = self.settings.get("SIGMOID_SCALE", "1.0")
        if ts_sampling in ("sigmoid", "shift") and sigmoid_scale and sigmoid_scale != "1.0":
            command.extend(["--sigmoid_scale", str(sigmoid_scale)])

        # Timestep range (from user settings, not hardcoded config)
        min_ts = self.settings.get("MIN_TIMESTEP", "")
        max_ts = self.settings.get("MAX_TIMESTEP", "")
        if min_ts:
            command.extend(["--min_timestep", str(min_ts)])
        if max_ts:
            command.extend(["--max_timestep", str(max_ts)])
        if self.settings.get("PRESERVE_DISTRIBUTION", False):
            command.append("--preserve_distribution_shape")

        command.extend([
            "--max_train_epochs", str(self.settings["MAX_TRAIN_EPOCHS"]),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--seed", str(self.settings["SEED"]),
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--pause_flag_path", os.path.join(self.settings["LORA_OUTPUT_DIR"], ".pause_requested"),
        ])

        # State saving. --save_state used to be passed unconditionally with no UI behind it, which
        # meant a 55-epoch run silently left 54 state dirs (hundreds of MB each) and never pruned.
        # Pause still saves state either way — the trainer forces it via --pause_flag_path.
        command.extend(self._state_flags())

        # Optional parameters
        if self.settings["OPTIMIZER_ARGS"]:
            # Klein's --optimizer_args is nargs='*' (one token per key=value). Passing the
            # whole box as ONE token made the trainer's key=value split fail with more than
            # one argument.
            command.extend(["--optimizer_args"] + self.settings["OPTIMIZER_ARGS"].split())

        # Gradient accumulation (effective batch = batch × this)
        gradient_accum = self.settings.get("GRADIENT_ACCUMULATION", 1)
        if isinstance(gradient_accum, str):
            gradient_accum = int(gradient_accum) if gradient_accum else 1
        if gradient_accum > 1:
            command.extend(["--gradient_accumulation_steps", str(gradient_accum)])

        # Max gradient norm (0 to disable clipping)
        max_grad_norm = self.settings.get("MAX_GRAD_NORM", 1.0)
        if isinstance(max_grad_norm, str):
            max_grad_norm = float(max_grad_norm) if max_grad_norm else 1.0
        if max_grad_norm > 0:
            command.extend(["--max_grad_norm", str(max_grad_norm)])

        # Network dropout (LoRA regularization)
        network_dropout = self.settings.get("NETWORK_DROPOUT", 0)
        if isinstance(network_dropout, str):
            network_dropout = float(network_dropout) if network_dropout else 0
        if network_dropout > 0:
            command.extend(["--network_dropout", str(network_dropout)])

        # Attention mechanism (user's choice, default is "sdpa")
        attention = self.settings["ATTENTION_MECHANISM"]
        if attention != "none":
            command.append(f"--{attention}")

        logging_dir = self.settings["LOGGING_DIR"]
        if logging_dir:
            command.extend(["--logging_dir", logging_dir])

        log_with = self.settings["LOG_WITH"]
        if log_with != "none":
            command.extend(["--log_with", log_with])

        log_prefix = self.settings["LOG_PREFIX"]
        if log_prefix:
            command.extend(["--log_prefix", log_prefix])

        if self.settings["IMG_IN_TXT_IN_OFFLOADING"]:
            command.append("--img_in_txt_in_offloading")

        # Adaptive LR overrides the step-based scheduler — force constant pre-phase.
        adaptive_on = bool(self.settings.get("ADAPTIVE_LR", False))
        lr_scheduler = "constant" if adaptive_on else self.settings["LR_SCHEDULER"]
        if lr_scheduler:
            command.extend(["--lr_scheduler", lr_scheduler])

        if not adaptive_on:
            lr_warmup_steps = self.settings["LR_WARMUP_STEPS"]
            if lr_warmup_steps:
                command.extend(["--lr_warmup_steps", lr_warmup_steps])

            lr_decay_steps = self.settings["LR_DECAY_STEPS"]
            if lr_decay_steps:
                command.extend(["--lr_decay_steps", lr_decay_steps])

        if adaptive_on:
            command.append("--adaptive_lr")
            min_lr = (self.settings.get("ADAPTIVE_LR_MIN", "1e-5") or "1e-5").split(" ")[0]
            max_lr = (self.settings.get("ADAPTIVE_LR_MAX", "4e-4") or "4e-4").split(" ")[0]
            command.extend(["--adaptive_lr_min", str(min_lr)])
            command.extend(["--adaptive_lr_max", str(max_lr)])

        # Context LoRA — train new LoRA with an existing one frozen + active
        ctx_path = self.settings.get("CONTEXT_LORA_PATH", "").strip()
        if ctx_path:
            command.extend(["--context_lora_path", ctx_path])
            ctx_strength = self.settings.get("CONTEXT_LORA_STRENGTH", "1.0") or "1.0"
            command.extend(["--context_lora_strength", str(ctx_strength)])

        weighting_scheme = self.settings["WEIGHTING_SCHEME"]
        if weighting_scheme != "none":
            command.extend(["--weighting_scheme", weighting_scheme])
            if weighting_scheme == "logit_normal":
                logit_mean = self.settings.get("LOGIT_MEAN", "0.0")
                logit_std = self.settings.get("LOGIT_STD", "1.0")
                if logit_mean and logit_mean != "0.0":
                    command.extend(["--logit_mean", str(logit_mean)])
                if logit_std and logit_std != "1.0":
                    command.extend(["--logit_std", str(logit_std)])
            elif weighting_scheme == "mode":
                mode_scale = self.settings.get("MODE_SCALE", "1.29")
                if mode_scale and mode_scale != "1.29":
                    command.extend(["--mode_scale", str(mode_scale)])

        # Metadata
        metadata_title = self.settings["METADATA_TITLE"]
        if metadata_title:
            command.extend(["--metadata_title", metadata_title])

        metadata_author = self.settings["METADATA_AUTHOR"]
        if metadata_author:
            command.extend(["--metadata_author", metadata_author])

        metadata_description = self.settings["METADATA_DESCRIPTION"]
        if metadata_description:
            command.extend(["--metadata_description", metadata_description])

        metadata_license = self.settings["METADATA_LICENSE"]
        if metadata_license:
            command.extend(["--metadata_license", metadata_license])

        metadata_tags = self.settings["METADATA_TAGS"]
        if metadata_tags:
            command.extend(["--metadata_tags", metadata_tags])

        metadata_trigger_phrase = self.settings["METADATA_TRIGGER_PHRASE"].strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if metadata_trigger_phrase and metadata_trigger_phrase.lower() != "trigger_word":
            command.extend(["--metadata_trigger_phrase", metadata_trigger_phrase])

        metadata_thumbnail = self.settings["METADATA_THUMBNAIL"].strip()
        if metadata_thumbnail:
            command.extend(["--metadata_thumbnail", metadata_thumbnail])

        if self.settings["RESUME_TRAINING"].strip():
            command.append(f"--resume={self.settings['RESUME_TRAINING']}")

        # Sample generation (only if enabled and architecture supports it)
        if self.sample_enabled_var.get() and config.get("supports_samples", False):
            # Generate prompt file
            prompt_file = self.generate_sample_prompt_file()
            command.extend(["--sample_prompts", prompt_file])

            # Frequency settings. Non-numeric text used to raise a bare ValueError out of
            # the command builder — treat it as "not set" instead.
            every_n_epochs = self.sample_every_n_epochs_var.get().strip()
            if every_n_epochs.isdigit() and int(every_n_epochs) > 0:
                command.extend(["--sample_every_n_epochs", every_n_epochs])

            every_n_steps = self.sample_every_n_steps_var.get().strip()
            if every_n_steps.isdigit() and int(every_n_steps) > 0:
                command.extend(["--sample_every_n_steps", every_n_steps])

            if self.sample_at_first_var.get():
                command.append("--sample_at_first")

            # Reference image for samples (Klein edit conditioning) — auto-capped
            # to ~0.20 MP in the trainer, so any size is safe.
            ref_img = getattr(self, "sample_ref_image_var", None)
            ref_img = ref_img.get().strip() if ref_img else ""
            if ref_img and os.path.exists(ref_img):
                command.extend(["--sample_ref_image", ref_img])

            # Use Distilled model for sample generation
            if getattr(self, 'use_distilled_samples_var', None) and self.use_distilled_samples_var.get():
                distilled_path = self.prefs_vars.get("distilled_dit", tk.StringVar()).get()
                if distilled_path and os.path.exists(distilled_path):
                    command.extend(["--sample_dit", distilled_path])
                    cache_mode = getattr(self, "cache_sample_model_var", None)
                    cache_mode = cache_mode.get() if cache_mode else self.settings.get("CACHE_SAMPLE_MODEL", "auto")
                    command.extend(["--cache_sample_model", cache_mode])
                    # INT8 fast preview matmul — same app-wide 'INT8 fast inference' toggle as the
                    # workbench + Krea 2 previews; applies to the Distilled sample DiT.
                    if self._get_inference_int8():
                        command.append("--sample_int8")
                    # Note: we deliberately do NOT forward the Preferences "DiT Block
                    # Swap (inference)" pref here. That setting governs the in-app
                    # inference tools (Repair Studio / Profiler / Extract / Explorer).
                    # The trainer auto-picks the Distilled sample swap from VRAM
                    # (_auto_distilled_sample_swap), so training samples manage their
                    # own memory independently. A power user can still force it via
                    # the raw --sample_blocks_to_swap flag.

        return command

    def build_cache_latents_command(self, config):
        """Build the cache latents command based on architecture"""
        if config.get("is_krea2"):
            return self._build_krea2_cache_command("krea2_cache_latents.py",
                                                   "--vae", self._krea2_pref("krea2_vae"))
        if config.get("is_minimax"):
            # --skip_existing: re-launching the same dataset should not re-encode every image.
            # Safe on LATENTS specifically because the skip validates the cached latent against
            # the CURRENT bucket, not just the filename — change Target Megapixels and it
            # re-encodes anyway. Deliberately NOT passed to text caching, where the skip is
            # filename-only and would silently reuse the embedding of an edited caption.
            cmd = self._build_krea2_cache_command(
                "minimax_cache_latents.py", "--vae", self._krea2_pref("minimax_vae")) + \
                ["--skip_existing"]
            # Passed whenever it's set — the script itself decides whether to load it, and only
            # does when the dataset actually contains clips. A stills folder never pays the
            # 605 MB, so there's nothing to gate on here.
            _avae = self._krea2_pref("minimax_audio_vae")
            if _avae:
                cmd += ["--audio_vae", _avae]
            return cmd
        arch = self.settings["ARCHITECTURE"]
        python_path = self._venv_python()
        cache_script_path = self._resolve_script(config, "cache_latents_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--vae", self.settings["VAE_MODEL"],
        ]

        # Wan needs CLIP for latent caching
        if config["uses_clip"]:
            command.extend(["--clip", self.settings["CLIP_MODEL"]])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    def build_cache_text_command(self, config):
        """Build the cache text encoder command based on architecture"""
        if config.get("is_krea2"):
            return self._build_krea2_cache_command("krea2_cache_text.py",
                                                   "--text_encoder", self._krea2_pref("krea2_text_encoder"))
        if config.get("is_minimax"):
            cmd = self._build_krea2_cache_command("minimax_cache_text.py",
                                                  "--text_encoder", self._krea2_pref("minimax_text_encoder"))
            # Reference distillation: the TEACHER's conditioning has to be built HERE, because
            # it needs the 15.7 GB vision-capable encoder and that can never be resident beside
            # the DiT at training time. Each image is paired with N others from this same
            # dataset — no picker, and no image is ever its own reference.
            if self.settings.get("MINIMAX_DISTILL"):
                cmd += ["--reference_count", str(self.settings.get("MINIMAX_DISTILL_REFS", "2"))]
            return cmd
        arch = self.settings["ARCHITECTURE"]
        python_path = self._venv_python()
        cache_script_path = self._resolve_script(config, "cache_text_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
        ]

        # Different text encoder parameters based on architecture
        if config["uses_t5"]:
            command.extend(["--t5", self.settings["T5_MODEL"]])
        elif config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        command.extend(["--batch_size", "16"])

        # FP8 text encoder flag
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    # === Krea 2 native command builders ===

    def _venv_python(self) -> str:
        """Python to launch training/caching subprocesses with.

        The bundled venv when it exists, otherwise whatever interpreter is running us. Without
        the fallback, any install where the venv is not a sibling of the repo — conda, a system
        install, the Docker image (venv lives at /opt/venv) — builds a command pointing at a
        file that is not there. The subprocess then fails to launch and the run dies silently
        right after "starting cache preparation", with nothing in the console to say why.
        """
        candidate = (os.path.join(FIZGIG_DIR, "venv", "Scripts", "python.exe") if os.name == 'nt'
                     else os.path.join(FIZGIG_DIR, "venv", "bin", "python"))
        return candidate if os.path.isfile(candidate) else sys.executable

    def _krea2_pref(self, key: str) -> str:
        """Read a Krea 2 model path from Preferences (krea2_raw_dit / krea2_turbo_dit / krea2_vae / krea2_text_encoder)."""
        var = self.prefs_vars.get(key)
        return var.get().strip() if var is not None else ""

    def _krea2_preview_engine(self) -> str:
        """Canonical Samples-tab preview engine for Krea 2: 'raw_lora' or 'turbo_model'.

        The combobox holds a display label; this maps it back. Unknown/missing -> 'raw_lora'
        (the default): renders previews on the resident training DiT with the Turbo LoRA @1.0
        instead of loading the Turbo checkpoint and parking the trainer to CPU."""
        var = getattr(self, "krea2_preview_engine_var", None)
        if var is not None:
            label = var.get()
            for key, text in self._KREA2_ENGINE_LABELS.items():
                if label == text:
                    return key
        return "raw_lora"

    def _krea2_script(self, name: str) -> str:
        return os.path.join(FIZGIG_DIR, "src", "fizgig", "scripts", name)

    def _minimax_reference_canvas(self):
        """The generation size the reference is scaled against — the square at Target Megapixels.

        The trainer sizes the reference against the largest training bucket; matching that here
        keeps the cached teacher conditioning and the training-time reference latent describing
        the same picture."""
        try:
            mp = float(str(self.dataset_megapixels_var.get()).strip())
        except (TypeError, ValueError, AttributeError):
            mp = 0.5
        side = int(round((mp * 1_000_000) ** 0.5 / 32) * 32) or 512
        return side, side

    def _build_krea2_cache_command(self, script_name: str, model_flag: str, model_path: str):
        """Krea 2 caching: a plain venv-python call to krea2_cache_latents.py / krea2_cache_text.py."""
        return [
            self._venv_python(),
            self._krea2_script(script_name),
            "--dataset_config", self.settings["DATASET_CONFIG"],
            model_flag, model_path,
        ]

    def _write_krea2_sample_prompts(self, filename="krea2_prompts.txt"):
        """Write the Samples-tab prompts as clean lines (one prompt per line) for krea2_train.

        Klein's prompt file carries inline flags (`--w`/`--h`/`--s`/...); krea2_train takes
        resolution as CLI args and reads each line as a literal prompt, so we strip any
        trailing ` --flag ...` group. Returns the file path, or None if no prompts."""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        lines = []
        for raw in self.sample_prompt_text.get("1.0", tk.END).splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            ln = ln.split(" --")[0].strip()  # drop Klein-style inline flags
            if ln:
                lines.append(ln)
        if not lines:
            return None
        path = os.path.join(samples_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def _build_krea2_train_command(self):
        """Build the native Krea 2 training command (RAW base, fp8 Turbo previews).

        Model paths come from Preferences (krea2_*); rank/alpha/lr/epochs/save/seed and the
        auto-resolved Blocks Swap come from the shared Training-tab knobs; sample resolution +
        frequency come from the Samples tab (same source Klein uses)."""
        # Armed fine-tune continuation (Resume after an FT pause): --dit becomes the pause
        # checkpoint — a one-run override, the preference is never touched — and the epoch
        # count is what's left of the original total. No --resume: FT has no state dirs.
        _fr = self._ft_resume_active()
        if getattr(self, "_ft_resume", None) and not _fr:
            self.update_console("[resume] armed fine-tune continuation IGNORED — the run being "
                                "launched is a different name or not a fine-tune.\n")
        _dit = _fr["checkpoint"] if _fr else self._krea2_pref("krea2_raw_dit")
        try:
            _epochs = int(str(self.settings["MAX_TRAIN_EPOCHS"]))
        except (KeyError, ValueError, TypeError):
            _epochs = 1
        if _fr:
            _epochs = max(1, _epochs - int(_fr.get("epochs_done", 0)))
            self.update_console(f"[resume] continuing fine-tune from "
                                f"{os.path.basename(_dit)} — {_epochs} epoch(s) to run\n")
        cmd = [
            self._venv_python(),
            self._krea2_script("krea2_train.py"),
            "--dit", _dit,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_train_epochs", str(_epochs),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--blocks_to_swap", str(self.settings["BLOCKS_SWAP"]),
            "--seed", str(self.settings["SEED"]),
            "--discrete_flow_shift", "2.5",
        ]
        # LoKR (Kronecker) — dim/alpha still ride along above but the trainer ignores them;
        # the factor is the dial. Klein's builder never reads NETWORK_TYPE (standard only).
        # Never emitted under base-model fine-tuning: the adapter is inert there, so a LoKR
        # would only burn VRAM (the trainer coerces too — belt and braces). Tk var read
        # directly, same rule as the FT flags below (ab3cca2).
        if (str(self.settings.get("NETWORK_TYPE", "")).startswith("LoKR")
                and not bool(self.krea2_finetune_var.get())):
            cmd += ["--network_type", "lokr",
                    "--lokr_factor", str(self.settings.get("LOKR_FACTOR", 8))]
        # State saving. Krea 2 previously wrote state ONLY on Pause, so a crash or a run that
        # finished too early meant starting over. Pause still saves regardless of these flags.
        cmd += self._state_flags()
        # Resume from a saved <name>-NNNNNN-state dir (set by the Resume button / pause flow).
        resume_path = (self.settings.get("RESUME_TRAINING") or "").strip()
        if resume_path:
            cmd += ["--resume", resume_path]
        # Context LoRA — train with an existing LoRA frozen + active on the base (model-agnostic).
        ctx_path = (self.settings.get("CONTEXT_LORA_PATH") or "").strip()
        if ctx_path:
            ctx_strength = (self.settings.get("CONTEXT_LORA_STRENGTH") or "1.0").strip() or "1.0"
            cmd += ["--context_lora_path", ctx_path, "--context_lora_strength", ctx_strength]
        # Adaptive LR — bi-directional plateau tracker (model-agnostic). Min/Max combo values
        # can carry a trailing note (e.g. "2e-4 - rank 4/8 only"); take the leading token.
        if self.settings.get("ADAPTIVE_LR"):
            min_lr = str(self.settings.get("ADAPTIVE_LR_MIN", "1e-5")).split(" ")[0]
            max_lr = str(self.settings.get("ADAPTIVE_LR_MAX", "4e-4")).split(" ")[0]
            cmd += ["--adaptive_lr", "--adaptive_lr_min", min_lr, "--adaptive_lr_max", max_lr]
        else:
            # LR scheduler + warmup (Other Options). Only when adaptive is OFF — adaptive owns the
            # LR, and the trainer would ignore the schedule anyway. These fields were visible under
            # Krea 2 but silently unwired before this.
            sched = (self.settings.get("LR_SCHEDULER") or "constant").strip() or "constant"
            if sched != "constant":
                cmd += ["--lr_scheduler", sched]
            warmup = str(self.settings.get("LR_WARMUP_STEPS", "") or "").strip()
            if warmup:
                try:
                    if int(float(warmup)) > 0:
                        cmd += ["--lr_warmup_steps", str(int(float(warmup)))]
                except ValueError:
                    pass
        # Gradient accumulation + grad clipping (Optimizer section — both wired for krea2).
        try:
            _accum = int(str(self.settings.get("GRADIENT_ACCUMULATION", 1) or 1).strip() or 1)
        except ValueError:
            _accum = 1
        if _accum > 1:
            cmd += ["--gradient_accumulation_steps", str(_accum)]
        _mgn = str(self.settings.get("MAX_GRAD_NORM", "") or "").strip()
        if _mgn:
            try:
                if abs(float(_mgn) - 1.0) > 1e-9:   # 1.0 is the trainer default
                    cmd += ["--max_grad_norm", str(float(_mgn))]
            except ValueError:
                pass
        # Optimizer family + free-form kwargs. Sent whenever set: the trainer's own default is
        # adamw8bit, so passing it explicitly is harmless and keeps the launched command a full
        # record of what the run actually used.
        _opt = str(self.settings.get("OPTIMIZER_TYPE", "") or "").strip()
        if _opt:
            cmd += ["--optimizer_type", _opt]
        _opt_args = str(self.settings.get("OPTIMIZER_ARGS", "") or "").strip()
        if _opt_args:
            cmd += ["--optimizer_args", _opt_args]
        _cb = str(self.settings.get("COMPILE_BLOCKS", "auto") or "auto").lower()
        # "outside" is a hand-set power value (settings JSON only — the dropdown offers
        # Auto/On/Off): the high-res compile boundary (#99). Passing it through beats
        # silently downgrading a stated choice to auto.
        if _cb in ("auto", "on", "off", "outside"):
            cmd += ["--compile_blocks", _cb]
        # torch.compile and block swap are mutually exclusive — compiled graphs assume their
        # weights stay put, and swap moves them every step, so the trainer ignores compile
        # whenever swap is active. It says so in its own log, but a user who set compile to On
        # sees the GUI still reading "On" and reasonably believes it is running.
        try:
            _blocks_swap = int(self.settings.get("BLOCKS_SWAP", 0) or 0)
        except (TypeError, ValueError):
            _blocks_swap = 0
        if _cb == "on" and _blocks_swap > 0:
            self.update_console(
                f"[compile] ignored this run — block swap is active ({_blocks_swap} blocks), and "
                "compiled graphs can't tolerate weights moving between CPU and GPU each step. "
                "Use 4-bit (NF4) instead of swapping if you want compile as well.\n")
        # Output metadata (Other Options → Metadata) — previously visible but never wired
        # for Krea 2; now recorded in the saved LoRA.
        for _mkey, _mflag in (("METADATA_TITLE", "--metadata_title"),
                              ("METADATA_AUTHOR", "--metadata_author"),
                              ("METADATA_DESCRIPTION", "--metadata_description"),
                              ("METADATA_LICENSE", "--metadata_license"),
                              ("METADATA_TAGS", "--metadata_tags")):
            _mval = str(self.settings.get(_mkey, "") or "").strip()
            if _mval:
                cmd += [_mflag, _mval]
        # Trigger phrase falls back to the Captions tab's trigger word — independent of
        # --trigger_word above, which is only ever sent when auto-recaption is on.
        _mtrig = self.settings.get("METADATA_TRIGGER_PHRASE", "").strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if _mtrig and _mtrig.lower() != "trigger_word":
            cmd += ["--metadata_trigger_phrase", _mtrig]
        _mthumb = self.settings.get("METADATA_THUMBNAIL", "").strip()
        if _mthumb:
            cmd += ["--metadata_thumbnail", _mthumb]
        # Base weight optimization. 4-bit NF4 supersedes fp8 (mutually exclusive): it quantizes the
        # frozen base to ~5.6 GB so a full LoRA trains on a 10-12 GB card with NO block swap (the
        # trainer forces blocks_to_swap=0 under 4-bit). Otherwise fp8 Base (the default) unless the
        # user unchecked it (bf16, 26 GB — big-card / heavy-swap only).
        # Explicit user choices FIRST — the auto branch used to be tested before them, so
        # unticking "FP8 Base" (an explicit bf16 request) did nothing when auto had chosen
        # INT8.
        # FP8 Base is hidden for Krea 2 and deliberately ignored here: --no_fp8 means a bf16
        # base (~28 GB) that no consumer card holds, the swap planner never accounted for it,
        # and this elif chain used to let it silently cancel the INT8 the planner had chosen.
        # A value persisted from Klein (or from before it was hidden) must not leak into a
        # Krea 2 run through a control the user can no longer see.
        _auto_i8 = getattr(self, "_auto_quant_int8", "")
        # An EXPLICIT INT8 pick must not depend on Blocks Swap being on Auto (#97): the auto
        # strategy is the only writer of _auto_quant_int8, and a manual swap value clears it
        # (the stale-leak guard in _parse_blocks_swap), so "Base Precision: INT8" plus a
        # manual swap silently fell back to the fp8 base — which Compile Blocks then dies on
        # for SM 8.6 cards (no fp8e4nv Triton support). At swap 0 the pick is honoured
        # directly. At swap N the fp8 fallback stays (INT8 weights don't ride the swap —
        # that pairing is the OOM the stale-leak guard exists for) but is now SAID, not
        # silent.
        try:
            _swap_now = int(str(self.settings.get("BLOCKS_SWAP", 0)).strip() or 0)
        except (TypeError, ValueError):
            _swap_now = 0
        try:
            _explicit_i8 = self._base_precision() == "int8"
        except Exception:
            _explicit_i8 = False
        if self.settings.get("QUANT_4BIT", False):
            cmd.append("--quantize_4bit")
        elif _explicit_i8 and _swap_now == 0:
            cmd += ["--quant_int8", "bf16"]
        elif _auto_i8:
            # Chosen by the auto strategy when there is VRAM for it: faster than NF4 and ~7x
            # more accurate, with exact gradients.
            cmd += ["--quant_int8", _auto_i8]
        elif _explicit_i8:
            self.update_console(
                f"[precision] INT8 needs Blocks Swap 0 — INT8 weights don't ride the swap. "
                f"Running the fp8 base with swap {_swap_now}; set Blocks Swap to 0 or Auto "
                f"to train on INT8.\n")

        # Per-image loss watch: detection logs/reports stuck images (Problem Images window);
        # per-image LR also throttles them (the trainer runs detection when either flag is on).
        # All four need Batch Size 1 (a batch-mean isn't a per-image signal) — the GUI greys
        # the toggles at batch > 1, and the flags are skipped here to match, with a note.
        try:
            _watch_bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
        except (ValueError, AttributeError):
            _watch_bs = 1
        _watch_ok = _watch_bs <= 1
        if not _watch_ok and (self.krea2_loss_watch_var.get() or self.krea2_per_image_lr_var.get()
                              or self.krea2_warmup_look_var.get() or self.krea2_auto_recaption_var.get()):
            self.update_console("[loss-watch] per-image features skipped — Batch Size is "
                                f"{_watch_bs}; they need Batch Size 1.\n")
        if _watch_ok and self.krea2_loss_watch_var.get():
            cmd.append("--log_per_image_loss")
        if _watch_ok and self.krea2_per_image_lr_var.get():
            cmd.append("--per_image_lr")
        if _watch_ok and self.krea2_warmup_look_var.get():
            cmd.append("--warmup_look_outliers")
        # Full base-model fine-tune (experimental): rotating trainable windows, full checkpoint out.
        # Read the Tk vars DIRECTLY, like every other krea2 toggle here — self.settings is only
        # refreshed when a preset is collected, so reading it made this silently never fire and
        # the run trained a LoRA instead.
        if bool(self.krea2_finetune_var.get()):
            mode = str(self.krea2_ft_mode_var.get() or "auto")
            if mode.lower().startswith("auto"):
                mode = "auto"   # the trainer resolves it from free VRAM at launch
            try:
                nblocks = int(str(self.krea2_ft_blocks_var.get()))
            except ValueError:
                nblocks = 14
            try:
                every = int(str(self.krea2_ft_every_var.get()))
            except ValueError:
                every = 1
            cmd += ["--finetune_rotation", str(max(1, nblocks)),
                    "--finetune_rotation_mode", mode,
                    "--finetune_rotate_every", str(max(1, every))]
            # Base precision under FINE-TUNE: NF4 is the trainer's default now, so an
            # explicit fp8 pick needs saying out loud. At the CLI an fp8 choice emits NO
            # flag (fp8_scaled = not --no_fp8, true by default), which is indistinguishable
            # from "Auto" — without this the dropdown's fp8 entry would silently produce
            # NF4 and the control would be lying. LoRA runs are untouched: this sits inside
            # the fine-tune branch.
            try:
                if self._base_precision() == "fp8":
                    cmd.append("--ft_base_fp8")
            except Exception:
                pass
            if _fr:
                # Continuation: pick the rotation cycle back up where the pause left it
                # (from the checkpoint's metadata) instead of restarting at window 0.
                cmd += ["--finetune_start_window", str(int(_fr.get("next_window", 0)))]
            if bool(self.krea2_ft_fused_var.get()):
                cmd.append("--finetune_fused_backward")
            if bool(self.krea2_fast_ft_var.get()):
                cmd.append("--fast_ft")
            _reg = self.krea2_reg_dir_var.get().strip()
            if _reg and os.path.isdir(_reg):
                try:
                    _rm = float(self.krea2_reg_mult_var.get())
                except ValueError:
                    _rm = 0.2
                cmd += ["--reg_lr_multiplier", str(max(0.0, _rm))]
        if _watch_ok and self.krea2_auto_recaption_var.get():
            cmd.append("--auto_recaption")
            # Trigger word from the Captions tab — appended (', <trigger>') to AI captions if
            # set. Reads the WIDGET-BOUND var (caption_text_var is an orphan that never
            # carried what the user typed). The placeholder guard stays in case an old
            # last_used.json seeded the literal "trigger_word".
            trig = (self.caption_trigger_var.get().strip()
                    if hasattr(self, "caption_trigger_var") else "")
            if trig and trig.lower() != "trigger_word":
                cmd += ["--trigger_word", trig]
            # Auto-recaption maps its two attempts onto two Captions-tab presets: attempt 1 uses
            # TRAINING CAPTION, attempt 2 uses EXHAUSTIVE DETAIL — your edited version of each
            # where you have one, the built-in otherwise. Deliberately not "whatever the tab is
            # set to": auto-recaption's job is fixed, so leaving the tab on "Short caption" must
            # not silently change what a training run writes mid-run.
            _ovr = self._caption_overrides()
            for _key, _flag in (("training", "--recaption_instruction"),
                                ("exhaustive", "--recaption_instruction_detailed")):
                _instr = str(_ovr.get(_key, "") or "").strip()
                if _instr:
                    cmd += [_flag, _instr]
        # Caption repair (manual edits from the Problem Images window AND auto-recaption)
        # re-encodes with the Qwen3-VL text encoder. --text_encoder used to be emitted only
        # inside the samples block, so with previews off the trainer had no TE path and every
        # caption fix bailed for the whole run, re-queueing forever. Emit it whenever any
        # watch toggle is on (a duplicate in the samples block is harmless — same value).
        if (self.krea2_loss_watch_var.get() or self.krea2_per_image_lr_var.get()
                or self.krea2_warmup_look_var.get() or self.krea2_auto_recaption_var.get()):
            _te = self._krea2_pref("krea2_text_encoder")
            if _te:
                cmd += ["--text_encoder", _te]

        # In-training previews: render the fp8 Turbo with the live LoRA. Resolution +
        # frequency come from the Samples tab; previews land in <output_dir>/sample, which
        # is exactly where the GUI samples watcher looks.
        if self.sample_enabled_var.get():
            prompt_file = self._write_krea2_sample_prompts()
            every = self.sample_every_n_epochs_var.get().strip()
            every_n = int(every) if every.isdigit() else 0
            # Krea 2 previews are per-EPOCH only. A steps-only config used to skip the
            # whole sample block in silence — say so instead.
            _steps_only = self.sample_every_n_steps_var.get().strip()
            if every_n <= 0 and _steps_only.isdigit() and int(_steps_only) > 0:
                self.update_console("[samples] Krea 2 previews are per-epoch — 'Every N Steps' "
                                    "has no effect. Set 'Every N Epochs' to enable previews.\n")
            ref_img = (getattr(self, "sample_ref_image_var", None).get().strip()
                       if getattr(self, "sample_ref_image_var", None) else "")
            ref_img = ref_img if (ref_img and os.path.exists(ref_img)) else ""
            _at_first = bool(getattr(self, "sample_at_first_var", None)
                             and self.sample_at_first_var.get())
            # Samples fire if there's a prompt OR a reference (ref-only = 'generate from this
            # picture' via the Qwen3-VL vision path). Sample-at-Start alone also counts.
            if (prompt_file or ref_img) and (every_n > 0 or _at_first):
                width = (self.sample_width_var.get().strip() or "1024")
                height = (self.sample_height_var.get().strip() or "1024")
                # Sample seed from the Samples tab (0 is a valid seed — don't let it fall through to
                # the trainer's default). Non-numeric/empty -> the SAMPLE_SEED default.
                try:
                    sample_seed = str(int(self.sample_seed_var.get().strip()))
                except (ValueError, AttributeError):
                    sample_seed = str(self.settings.get("SAMPLE_SEED", 1234))
                cmd += [
                    "--sample_every_n_epochs", str(every_n),
                    "--sample_width", width,
                    "--sample_height", height,
                    "--sample_seed", sample_seed,
                    "--turbo_dit", self._krea2_pref("krea2_turbo_dit"),
                    "--vae", self._krea2_pref("krea2_vae"),
                    "--text_encoder", self._krea2_pref("krea2_text_encoder"),
                    # Forward-only block swap on the preview Turbo, auto-detected for the Turbo's
                    # VRAM profile so previews fit the card — mirrors Klein's Distilled sample swap.
                    # (Ignored in raw_lora engine mode — that path uses the training placement.)
                    "--preview_blocks_to_swap", str(self._auto_krea2_inference_blocks_swap()),
                ]
                # Preview engine (Samples tab): raw_lora renders on the resident training DiT
                # with the Turbo LoRA @1.0 — no Turbo checkpoint load, no CPU parking. The
                # trainer prefers --turbo_lora over --turbo_dit when both are given, and falls
                # back to the Turbo checkpoint by itself if the LoRA file has gone missing.
                # Under a fine-tune the Turbo LoRA is REQUIRED for previews (the trained
                # weights live in the base, which the standalone Turbo can't show), so the
                # engine preference is overridden and the LoRA travels regardless.
                if (self._krea2_preview_engine() == "raw_lora"
                        or bool(self.krea2_finetune_var.get())):
                    _tlora = self._krea2_pref("krea2_turbo_lora")
                    if not _tlora or not os.path.isfile(_tlora):
                        # First use after an update: fetch it now (~470 MB, idempotent — the
                        # update script usually gets there first) and populate the pref.
                        try:
                            import sys as _sys
                            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
                            from fizgig.scripts.fetch_turbo_lora import ensure_turbo_lora
                            self.update_console("[preview] Turbo LoRA not set — downloading it "
                                                "now (one-time, ~470 MB)...\n")
                            _tlora = ensure_turbo_lora(
                                log=lambda m: self.update_console(f"[preview] {m}\n"),
                                require=True)
                            if _tlora and "krea2_turbo_lora" in self.prefs_vars:
                                self.prefs_vars["krea2_turbo_lora"].set(_tlora)
                        except Exception:
                            _tlora = None
                    if _tlora:
                        cmd += ["--turbo_lora", _tlora]
                    else:
                        self.update_console(
                            "[preview] Turbo LoRA unavailable (download failed?) — using the "
                            "classic Turbo model for previews this run. Set the path in "
                            "Preferences or re-run update_fizgig.bat.\n")
                # Steps / CFG / Negative / Sample-at-Start — previously visible on the Samples
                # tab but never wired into krea2_train.
                _st = self.sample_steps_var.get().strip()
                if _st.isdigit() and int(_st) > 0:
                    cmd += ["--sample_steps", _st]
                try:
                    _cfg = float(self.sample_cfg_scale_var.get().strip() or 1.0)
                except (ValueError, AttributeError):
                    _cfg = 1.0
                if _cfg > 0 and abs(_cfg - 1.0) > 1e-9:
                    cmd += ["--sample_cfg_scale", str(_cfg)]
                _negp = (self.sample_negative_var.get().strip()
                         if getattr(self, "sample_negative_var", None) else "")
                if _negp and _cfg > 1.0:
                    cmd += ["--sample_negative", _negp]
                if _at_first:
                    cmd.append("--sample_at_first")
                # INT8 fast preview matmul — same app-wide 'INT8 fast inference' toggle as the workbench.
                if self._get_inference_int8():
                    cmd.append("--preview_int8")
                if prompt_file:
                    cmd += ["--sample_prompts", prompt_file]
                if ref_img:
                    cmd += ["--sample_ref_image", ref_img]
        return cmd

    def _build_minimax_train_command(self):
        """Build the native MiniMax H3 training command — barebones image-only LoRA over an
        NF4-quantized frozen base. No samples, no block swap, no context LoRA, no LoKR, no
        per-image loss watch: just the core knobs (rank/alpha/lr/epochs/save/seed/optimizer) plus
        adaptive LR and output metadata. Model paths come from Preferences (minimax_*)."""
        # Armed fine-tune continuation (Resume after an FT pause): --dit becomes the pause
        # checkpoint — a one-run override that outranks the distill/ref2va choice too — and
        # the epoch count is what's left of the original total. No --resume under FT.
        _fr = self._ft_resume_active()
        if getattr(self, "_ft_resume", None) and not _fr:
            self.update_console("[resume] armed fine-tune continuation IGNORED — the run being "
                                "launched is a different name or not a fine-tune.\n")
        _dit = (_fr["checkpoint"] if _fr
                else (self._krea2_pref("minimax_ref_dit")
                      if ((self.settings.get("MINIMAX_DISTILL")
                           or self.settings.get("MINIMAX_TRAIN_BASE") == "ref2va")
                          and self._krea2_pref("minimax_ref_dit"))
                      else self._krea2_pref("minimax_dit")))
        try:
            _epochs = int(str(self.settings["MAX_TRAIN_EPOCHS"]))
        except (KeyError, ValueError, TypeError):
            _epochs = 1
        if _fr:
            _epochs = max(1, _epochs - int(_fr.get("epochs_done", 0)))
            self.update_console(f"[resume] continuing fine-tune from "
                                f"{os.path.basename(_dit)} — {_epochs} epoch(s) to run\n")
        cmd = [
            self._venv_python(),
            self._krea2_script("minimax_train.py"),
            # Distillation trains against ref2va — the teacher only exists on that model.
            # Otherwise the Training Base dropdown decides: ref2va when the user deploys on
            # the r2v workflow, the ordinary fl2va base by default.
            "--dit", _dit,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_train_epochs", str(_epochs),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--seed", str(self.settings["SEED"]),
        ]
        # Blocks Swap: "Auto (detect from GPU)" resolves in the TRAINER (it reads real free VRAM
        # at run time — correct for queued runs too); an explicit number passes through.
        _bs = str(self.settings.get("BLOCKS_SWAP", "auto") or "auto").strip()
        cmd += ["--blocks_to_swap", "auto" if _bs.lower().startswith("auto") else _bs]
        # Base Precision. Always sent, including "auto", so the launched command records which
        # base a run used rather than leaving it implicit — these get A/B'd against each other.
        cmd += ["--base_quant", minimax_base_quant(self.settings.get("MINIMAX_BASE_QUANT"))]
        # Per-step movement clip: RETIRED (Peter, 10 Aug) — the Adapter-relative LR ramp removes
        # the overshoot at its root rather than capping it after the fact. Never emitted, so an
        # old preset or a saved config cannot revive it.
        # Gradient Accumulation (Optimizer section). The field was visible under MiniMax but
        # never emitted, so it silently did nothing on this family.
        try:
            _accum = int(str(self.settings.get("GRADIENT_ACCUMULATION", 1) or 1).strip() or 1)
        except ValueError:
            _accum = 1
        if _accum > 1:
            cmd += ["--gradient_accumulation_steps", str(_accum)]
        # LR warmup: RETIRED alongside the clip — the ramp eases the first epochs in by
        # construction, and does not need an epoch count guessed up front. Never emitted.
        # EMA stays: "0.99 (recommended)" -> 0.99.
        _em = str(self.settings.get("MINIMAX_EMA", "Off") or "Off").split(" ")[0]
        if _em.replace(".", "", 1).isdigit():
            cmd += ["--ema_decay", _em]
        _ar = str(self.settings.get("MINIMAX_ADAPTER_RAMP", "Off") or "Off").split(" ")[0]
        if _ar.replace(".", "", 1).isdigit():
            cmd += ["--adapter_ramp", _ar]
        # Caption dropout. ALWAYS sent, including 0 — the trainer's own default is 0.05, so
        # "Off" has to be stated explicitly or it silently keeps dropping captions.
        # Whatever the box says, including under Multi Concept — the builder used to force it to
        # 0 there, which quietly made every multi-concept run a dropout-off run and confounded
        # the very comparison it was meant to help.
        _cd = str(self.settings.get("MINIMAX_CAPTION_DROPOUT", "0.05") or "0.05").split(" ")[0]
        cmd += ["--caption_dropout", _cd if _cd.replace(".", "", 1).isdigit() else "0"]
        # Gradient Checkpointing. The flag used to not be sent at all here, so the checkbox was
        # decorative on this family. Ticked (the default) means AUTO — the planner decides from
        # free VRAM, exactly like Blocks Swap and Base Precision, and in practice that is "on"
        # for anything short of a 36 GB+ card. Unticked is the explicit override that forces it
        # off. Deliberately no "force ON": it only differs from auto where there is memory to
        # spare, and there it just costs ~0.1 s/step for nothing.
        cmd += ["--gradient_checkpointing",
                "auto" if self.settings.get("GRADIENT_CHECKPOINTING", True) else "off"]
        # Detail Focus -> --shift. Sent ALWAYS, including the reference 12, so the launched
        # command (and the console line recording it) states which density a run used instead of
        # leaving it implicit — these are meant to be A/B'd against each other, often queued
        # back to back, and "which one was this?" has to be answerable from the record alone.
        # The trainer stamps the same thing into the LoRA as ss_timestep_density.
        # Low-noise share -> shift. Always sent, including the default, so the launched command
        # records which density ran instead of leaving it implicit.
        # Always the plain uniform-base shift. A saved preset or queue row carrying the retired
        # MINIMAX_LOGNORM is deliberately ignored rather than migrated — mid-concentrated is the
        # thing being removed, so honouring it here would keep shipping the fault.
        _shift = minimax_lownoise_to_shift(self.settings.get("MINIMAX_LOWNOISE_PCT"))
        if _shift is not None:
            cmd += ["--shift", f"{_shift:g}"]
        _hl = minimax_highnoise_lr(self.settings.get("MINIMAX_HIGHNOISE_LR_PCT"))
        _ft_now = bool(getattr(self, "minimax_finetune_var", None)
                       and self.minimax_finetune_var.get())
        # Not under FT: the band multiplier rewrites optimizer param-group LRs, which the
        # fused fine-tune doesn't have (and rotation rebuilds discard the stash). The GUI
        # hides the control under FT; a stale saved value must not resurrect the flag.
        if _hl is not None and abs(_hl - 1.0) > 1e-9 and not _ft_now:
            cmd += ["--highnoise_lr_scale", f"{_hl:g}"]
        # Per-category retirement (mixed visual+voice datasets). One category, one epoch —
        # sent only when the epoch is set: the flag's presence means the run used it.
        try:
            _n = int(str(self.settings.get("MIXED_STOP_EPOCH", "") or "").strip() or 0)
        except ValueError:
            _n = 0
        if _n > 0:
            _flag = ("visual" if "photo" in
                     str(self.settings.get("MIXED_STOP_CATEGORY", "")).lower() else "audio")
            # Under FT only "stop" exists (the anchor rides param-group LR machinery FT
            # doesn't have) and the trainer snaps the epoch to a rotation-cycle boundary.
            _mode = ("stop" if (_ft_now or "stop" in
                     str(self.settings.get("MIXED_STOP_MODE", "")).lower()) else "anchor")
            cmd += [f"--{_flag}_stop_epoch", str(_n), f"--{_flag}_stop_mode", _mode]
        # Blocks to Train — only sent when it's a real range; "all" is the trainer's own default,
        # and not sending it keeps the flag's presence meaning "this run was a block experiment".
        _mft_cmd_on = bool(getattr(self, "minimax_finetune_var", None)
                           and self.minimax_finetune_var.get())
        _blocks = minimax_block_spec(self.settings.get("MINIMAX_BLOCKS", "all"))
        if _blocks.lower() != "all" and not _mft_cmd_on:
            cmd += ["--train_blocks", _blocks]
        # Optimised Likeness Learning — photo steps train the identity blocks only, clips train
        # everything. The launch dict already forced MINIMAX_BLOCKS to "all" when this is on, so
        # the two flags never fight. The flag TRAVELS under fine-tune too: the trainer honours
        # the same semantics there (cycle-tighten on photo-only data, per-parameter photo
        # freezing on mixed). --train_blocks stays adapter-only and is never emitted under FT.
        if self.settings.get("MINIMAX_LIKENESS_OPT"):
            cmd += ["--photo_blocks", MINIMAX_LIKENESS_BLOCKS]
            # Restrict video to likeness blocks (FT only, on by default with likeness):
            # a confined overnight video run trained perfectly well (field, 29 Aug).
            # Unticked, clips keep the original whole-model behaviour.
            if _mft_cmd_on and self.settings.get("MINIMAX_FT_CLIP_LIKENESS", True):
                cmd += ["--clip_blocks", MINIMAX_LIKENESS_BLOCKS]
        # Voice routing — audio steps train only the measured voice zone (34-49): outside it
        # they corrupt the visual blocks (A/B, 24 Aug). Under FT it always travels (the
        # trainer also tightens the cycle to the union of what the dataset trains); in LoRA
        # mode it is part of Optimised Likeness Learning. Harmless without audio files.
        if _ft_now or self.settings.get("MINIMAX_LIKENESS_OPT"):
            cmd += ["--audio_blocks", MINIMAX_AUDIO_BLOCKS]
        # Reference distillation. Both flags travel together; the trainer also needs --vae to
        # encode the reference, which the sample block may already have added.
        if self.settings.get("MINIMAX_DISTILL"):
            cmd += ["--distill",
                    "--distill_weight", str(self.settings.get("MINIMAX_DISTILL_WEIGHT", "0.8"))]
            # Identity-first phase length. "Auto" -> -1 (the trainer sizes it from the dataset),
            # "Off" -> 0 (blended throughout), otherwise the leading number of epochs.
            _p1 = str(self.settings.get("MINIMAX_DISTILL_PHASE1", "Auto") or "Auto")
            _p1n = "-1" if _p1.startswith("Auto") else ("0" if _p1.startswith("Off")
                                                        else _p1.split(" ")[0])
            cmd += ["--distill_phase1_epochs", _p1n if _p1n.lstrip("-").isdigit() else "-1"]
        # AdaLN LOCKED off (Peter, 9 Aug): the pruned builds everyone deploys on cannot load
        # AdaLN LoRA keys, so training it only wastes capacity. Checkbox hidden; always opt out.
        cmd.append("--no_train_adaln")
        # Depth-split LR is RETIRED (Peter, 9 Aug): it was the manual precursor of the limiter
        # + governor, which target whoever actually runs hot instead of a guessed range. The
        # controls are hidden and a stale saved range is deliberately not sent.
        # Rotation fine-tune. Read from the Tk vars, not self.settings — the Krea builder
        # learned the hard way that reading settings made the flags silently never fire.
        _mft_on = bool(getattr(self, "minimax_finetune_var", None)
                       and self.minimax_finetune_var.get())
        if _mft_on:
            # Component is the only mode (24 Aug — block/numeric windows removed).
            cmd += ["--finetune_rotation", "1", "--finetune_rotation_mode", "component"]
            _mfte = str(self.minimax_ft_every_var.get()).strip()
            cmd += ["--finetune_rotate_every", _mfte if _mfte.isdigit() else "1"]
            if _fr:
                # Continuation: pick the rotation cycle back up where the pause left it
                # (from the checkpoint's metadata) instead of restarting at window 0.
                cmd += ["--finetune_start_window", str(int(_fr.get("next_window", 0)))]
            if str(self.minimax_ft_scope_var.get()).startswith("Photos"):
                cmd += ["--finetune_scope", "photo"]
            _mftspec = str(self.minimax_ft_blockspec_var.get()).strip()
            if _mftspec:
                cmd += ["--finetune_blocks", _mftspec]
            if not bool(self.minimax_ft_fused_var.get()):
                cmd += ["--no_finetune_fused_backward"]
            # Regularisation LR multiplier — only meaningful when the TOML carries the
            # is_reg block (same gate as the block writer: FT on + a real folder).
            _mreg = (self.minimax_reg_dir_var.get().strip()
                     if hasattr(self, "minimax_reg_dir_var") else "")
            if _mreg and os.path.isdir(_mreg):
                try:
                    _mrm = float(self.minimax_reg_mult_var.get())
                except ValueError:
                    _mrm = 0.2
                # Floor 0.01: 0.0 would still pay a full forward/backward per reg step
                # for a near-zero update — clearing the folder is how you disable this.
                cmd += ["--reg_lr_multiplier", str(max(0.01, _mrm))]
        # LoKR (Kronecker) — dim/alpha still ride along above but the trainer ignores them;
        # the factor is the dial. Same flags as the Krea 2 builder. Suppressed under FT: the
        # trainer builds no adapter at all there.
        if str(self.settings.get("NETWORK_TYPE", "")).startswith("LoKR") and not _mft_on:
            cmd += ["--network_type", "lokr",
                    "--lokr_factor", str(self.settings.get("LOKR_FACTOR", 8))]
        # In-training previews. Prompts come from the Samples tab (same widgets every family
        # uses); the trainer pre-encodes them with the 32B TE before the DiT loads.
        if self.sample_enabled_var.get():
            # plain one-prompt-per-line file (same writer Krea 2 uses; own filename so a
            # MiniMax output dir doesn't sprout a "krea2_" artefact)
            prompt_file = self._write_krea2_sample_prompts("minimax_prompts.txt")
            _every = self.sample_every_n_epochs_var.get().strip()
            _every_n = int(_every) if _every.isdigit() else 0
            _at_first = bool(getattr(self, "sample_at_first_var", None)
                             and self.sample_at_first_var.get())
            # H3 previews are per-EPOCH only; a steps-only config would otherwise silently
            # produce nothing (same constraint, and same warning, as Krea 2).
            _steps_only = self.sample_every_n_steps_var.get().strip()
            if _every_n <= 0 and _steps_only.isdigit() and int(_steps_only) > 0:
                self.update_console("[samples] MiniMax H3 previews are per-epoch — 'Every N Steps' "
                                    "has no effect. Set 'Every N Epochs' to enable previews.\n")
            _te = self._krea2_pref("minimax_text_encoder")
            if prompt_file and (_every_n > 0 or _at_first) and _te:
                try:
                    _seed = str(int(self.sample_seed_var.get().strip()))
                except (ValueError, AttributeError):
                    _seed = str(self.settings.get("SAMPLE_SEED", 42))
                cmd += [
                    "--sample_prompts", prompt_file,
                    "--sample_every_n_epochs", str(_every_n),
                    "--sample_width", (self.sample_width_var.get().strip() or "512"),
                    "--sample_height", (self.sample_height_var.get().strip() or "512"),
                    "--sample_seed", _seed,
                    "--text_encoder", _te,
                    "--vae", self._krea2_pref("minimax_vae"),
                ]
                # Sample length: "124 frames (~5s — trained minimum)" -> 124. Always sent so
                # the launched command records whether a run previewed stills or clips.
                _sf_raw = str(getattr(self, "sample_frames_var", None)
                              and self.sample_frames_var.get() or "")
                _sf = _sf_raw.split(" ")[0]
                cmd += ["--sample_frames", _sf if _sf.isdigit() else "1"]
                # "with sound" variants: the samples also carry their generated audio,
                # decoded through the audio VAE — same file the caching pass uses.
                if "with sound" in _sf_raw.lower():
                    _avae = self._krea2_pref("minimax_audio_vae")
                    if _avae:
                        cmd += ["--sample_audio", "--audio_vae", _avae]
                    else:
                        self.update_console("[samples] 'with sound' needs the Audio VAE path "
                                            "— set it in Preferences. Samples render "
                                            "silent.\n")
                # The Turbo LoRA (Preferences) takes over the preview pace when set: its own
                # steps + strength from the Samples tab (6 @ 75% recommended), nothing else
                # changed. Without it, the ordinary Steps box applies as before.
                _turbo = self._krea2_pref("minimax_turbo_lora")
                if _turbo and os.path.isfile(_turbo):
                    # read the WIDGETS, like the other sample fields — settings can lag them
                    _ts = str(getattr(self, "turbo_steps_entry", None)
                              and self.turbo_steps_entry.get() or "").strip()
                    _ts = _ts if _ts.isdigit() and int(_ts) > 0 else "6"
                    try:
                        _tstr = float(str(getattr(self, "turbo_strength_entry", None)
                                          and self.turbo_strength_entry.get()
                                          or "").strip() or 75) / 100.0
                    except ValueError:
                        _tstr = 0.75
                    _tstr = min(2.0, max(0.0, _tstr))
                    cmd += ["--turbo_lora_path", _turbo,
                            "--turbo_lora_strength", f"{_tstr:.3f}",
                            "--sample_steps", _ts]
                else:
                    _st = self.sample_steps_var.get().strip()
                    if _st.isdigit() and int(_st) > 0:
                        cmd += ["--sample_steps", _st]
                try:
                    _cfg = float(self.sample_cfg_scale_var.get().strip())
                except (ValueError, AttributeError):
                    _cfg = 1.0
                if abs(_cfg - 1.0) > 1e-9:
                    cmd += ["--sample_cfg_scale", str(_cfg)]
                    _neg = (self.sample_negative_var.get().strip()
                            if getattr(self, "sample_negative_var", None) else "")
                    if _neg:
                        cmd += ["--sample_negative", _neg]
                if _at_first:
                    cmd.append("--sample_at_first")
            elif prompt_file and (_every_n > 0 or _at_first) and not _te:
                self.update_console("[samples] previews need the Qwen3-VL-32B text encoder path "
                                    "— set it in Preferences. Training continues without them.\n")
        # Resumable state saving + resume — identical flag names across all three families.
        cmd += self._state_flags()
        resume_path = (self.settings.get("RESUME_TRAINING") or "").strip()
        if resume_path:
            cmd += ["--resume", resume_path]
        # Adaptive LR is RETIRED for MiniMax (Peter, 9 Aug): ticking it silently disabled the
        # governor and warmup (both defer to it), quietly dismantling the stability stack. The
        # control is hidden under this family and a stale saved ADAPTIVE_LR=True is deliberately
        # ignored here — the governor owns the schedule.
        # Grad clipping (Optimizer section). 1.0 is the trainer default — send only when the
        # user changed it, keeping the launched command a faithful record otherwise.
        _mgn = str(self.settings.get("MAX_GRAD_NORM", "") or "").strip()
        if _mgn:
            try:
                if abs(float(_mgn) - 1.0) > 1e-9:
                    cmd += ["--max_grad_norm", str(float(_mgn))]
            except ValueError:
                pass
        # Optimizer LOCKED to adamw (Peter, 9 Aug): full-precision state was the single biggest
        # likeness change measured on H3 — 8-bit state costs fine detail for 1.9 GB. The dropdown
        # is hidden under this family; whatever the shared setting holds is overridden here.
        cmd += ["--optimizer_type", "adamw"]
        _opt_args = str(self.settings.get("OPTIMIZER_ARGS", "") or "").strip()
        if _opt_args:
            cmd += ["--optimizer_args", _opt_args]
        # Output metadata (Other Options → Metadata) — recorded in the saved LoRA.
        for _mkey, _mflag in (("METADATA_TITLE", "--metadata_title"),
                              ("METADATA_AUTHOR", "--metadata_author"),
                              ("METADATA_DESCRIPTION", "--metadata_description"),
                              ("METADATA_LICENSE", "--metadata_license"),
                              ("METADATA_TAGS", "--metadata_tags")):
            _mval = str(self.settings.get(_mkey, "") or "").strip()
            if _mval:
                cmd += [_mflag, _mval]
        _mtrig = self.settings.get("METADATA_TRIGGER_PHRASE", "").strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if _mtrig and _mtrig.lower() != "trigger_word":
            cmd += ["--metadata_trigger_phrase", _mtrig]
        return cmd

    # === Pause / Resume support ===

    def _pause_flag_path(self) -> str:
        """Path to the pause sentinel file in the current output directory."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".pause_requested")

    def _paused_sidecar_path(self) -> str:
        """Path to the JSON sidecar that records paused-state metadata."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".fizgig_paused.json")

    def _refresh_training_buttons(self):
        """Show/hide Pause and Resume buttons based on self.training_state."""
        if not hasattr(self, "training_state"):
            self.training_state = "idle"
        # While a run is active the Start button queues instead of starting — say so on
        # the button itself rather than surprising the user with a popup.
        try:
            self._start_training_btn.config(
                text="Queue Train" if self.training_state in ("running", "pausing")
                else "Start Training")
        except Exception:
            pass
        # Every state transition passes through here, so it's the one hook that keeps an
        # OPEN queue window truthful (finish, advance, failure-hold, pause) — the render
        # no-ops when the window isn't up.
        try:
            self._render_queue_window()
        except Exception:
            pass
        # Pause: visible while running (Krea 2 now saves full state at the epoch boundary, so
        # graceful Pause/Resume works the same as Klein).
        if self.training_state == "running":
            try: self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._pause_training_btn.pack_forget()
            except Exception: pass
        # Resume: visible while paused
        if self.training_state == "paused":
            try: self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._resume_training_btn.pack_forget()
            except Exception: pass

    def _pause_training(self):
        """Request a graceful pause — trainer will save state at end of current epoch and exit."""
        if not getattr(self, "current_process", None) or self.current_process.poll() is not None:
            messagebox.showinfo("Not Running", "No active training to pause.")
            return
        if getattr(self, "training_state", "idle") != "running":
            return
        try:
            os.makedirs(os.path.dirname(self._pause_flag_path()) or ".", exist_ok=True)
            open(self._pause_flag_path(), "w").close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write pause flag:\n{e}")
            return
        self.update_console(
            "\n=== PAUSE REQUESTED — trainer will save full state and exit cleanly at end of current epoch. "
            "GPU memory will be freed. Click Resume Training afterwards to continue. ===\n\n"
        )
        messagebox.showinfo(
            "Pause Requested",
            "Pause queued. The trainer will finish the CURRENT epoch, save full state, "
            "and exit cleanly to free GPU memory.\n\n"
            "Click Resume Training afterwards to continue.",
        )
        self.training_state = "pausing"
        self._refresh_training_buttons()

    def _detect_latest_state_dir(self):
        """Find the highest-numbered <output_name>-NNNNNN-state/ directory in the output dir."""
        import re as _re
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        out_name = self.settings.get("LORA_NAME", "") or ""
        if not out_name or not os.path.isdir(out_dir):
            return None
        pattern = _re.compile(rf"^{_re.escape(out_name)}-(\d{{6}})-state$")
        candidates = []
        try:
            for entry in os.listdir(out_dir):
                m = pattern.match(entry)
                # training_state.json is the save's commit marker (written last) — a dir without
                # it is a partial save from a crashed write, not a state. Skipping it here means
                # Resume lands on the previous GOOD state instead of a refusal.
                if (m and os.path.isdir(os.path.join(out_dir, entry))
                        and os.path.isfile(os.path.join(out_dir, entry, "training_state.json"))):
                    candidates.append((int(m.group(1)), entry))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(out_dir, candidates[0][1])

    def _detect_latest_ft_checkpoint(self):
        """The FT twin of _detect_latest_state_dir: highest-numbered
        <output_name>-NNNNNN.safetensors in the output dir. Fine-tunes leave full
        checkpoints, never state dirs — the checkpoint IS the continuation point."""
        import re as _re
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        out_name = self.settings.get("LORA_NAME", "") or ""
        if not out_name or not os.path.isdir(out_dir):
            return None
        pattern = _re.compile(rf"^{_re.escape(out_name)}-(\d{{6}})\.safetensors$")
        candidates = []
        try:
            for entry in os.listdir(out_dir):
                m = pattern.match(entry)
                if m and os.path.isfile(os.path.join(out_dir, entry)):
                    candidates.append((int(m.group(1)), entry))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(out_dir, candidates[0][1])

    def _ft_resume_active(self):
        """The armed fine-tune continuation for THIS launch, or None.

        Guarded twice: the run being launched must still be a fine-tune of the same family
        (unticking the FT box means the user wants something else), and its output name must
        match the paused run's — a queued job launched via 'Start next now' while an FT
        continuation is armed must not consume another run's checkpoint."""
        _fr = getattr(self, "_ft_resume", None)
        if not _fr:
            return None
        if self._is_krea2_arch():
            _on = bool(getattr(self, "krea2_finetune_var", None) and self.krea2_finetune_var.get())
        elif self._is_minimax_arch():
            _on = bool(getattr(self, "minimax_finetune_var", None) and self.minimax_finetune_var.get())
        else:
            _on = False
        if not _on:
            return None
        if str(self.settings.get("LORA_NAME", "")) != str(_fr.get("output_name", "")):
            return None
        return _fr

    POD_STOP_COUNTDOWN = 120   # seconds

    def _maybe_stop_pod_after_training(self):
        """Offer to stop a rented pod once a run has finished on its own.

        Never silent and never immediate: the point is to stop billing on an UNATTENDED finish, so
        anyone actually sitting there must be able to stop it happening. A countdown they can
        cancel does both."""
        if not _running_on_pod():
            return
        if str(self.prefs_vars.get("runpod_stop_when_done", tk.StringVar()).get()).strip() != "1":
            return

        win = tk.Toplevel(self.master)
        win.title("Training finished — stopping pod")
        win.configure(bg=BG_COLOR)
        win.transient(self.master)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        win.resizable(False, False)

        # Cancel packed BOTTOM first so a long message can never push it off the edge (v2.8.5).
        row = ttk.Frame(win)
        row.pack(side=tk.BOTTOM, pady=(6, 14))

        tk.Label(win, text="Training finished", font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=BG_COLOR).pack(anchor=tk.W, padx=18, pady=(16, 2))
        msg = tk.Label(win, text="", font=(FONT_FAMILY, 10), fg=COLORS["text_explain"],
                       bg=BG_COLOR, wraplength=460, justify=tk.LEFT)
        msg.pack(anchor=tk.W, padx=18, pady=(0, 6))
        tk.Label(win,
                 text="Your LoRA and everything else under /workspace is on the persistent volume "
                      "and will still be there next time.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, padx=18)

        state = {"left": self.POD_STOP_COUNTDOWN, "cancelled": False}

        def cancel():
            state["cancelled"] = True
            self.update_console("[pod] auto-stop cancelled — pod left running.\n")
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(row, text="Keep the pod running", command=cancel).pack()

        def tick():
            if state["cancelled"]:
                return
            if state["left"] <= 0:
                try:
                    win.destroy()
                except Exception:
                    pass
                self._stop_this_pod()
                return
            msg.config(text=f"Stopping this pod in {state['left']}s to stop it billing.\n"
                            f"Cancel below if you want to keep working.")
            state["left"] -= 1
            win.after(1000, tick)

        tick()
        try:
            win.update_idletasks()
            win.grab_set()
        except Exception:
            pass

    def _stop_this_pod(self):
        """Stop this pod through RunPod's API. Reports what happened either way.

        Uses the GraphQL endpoint directly rather than runpodctl: runpodctl tries to sync SSH keys
        as a side effect of being configured, which fails noisily and has nothing to do with
        stopping a machine.

        The key RunPod injects as RUNPOD_API_KEY is POD-SCOPED and cannot manage pods — verified
        on a live pod, where `runpodctl pod list` returns 403 both before and after configuring
        with it. So this needs an account key the user supplies themselves."""
        key = self._pod_stop_key()
        pid = os.environ.get("RUNPOD_POD_ID", "").strip() or _pod_id()
        if not key:
            self.update_console(
                "[pod] auto-stop is on but no API key is set, so the pod is still running.\n"
                "[pod] Add RUNPOD_STOP_API_KEY to your template (RunPod > Settings > API Keys).\n"
                "[pod] The key RunPod provides automatically is pod-scoped and cannot stop pods.\n")
            return
        self.update_console(f"[pod] stopping pod {pid or '(unknown id)'}…\n")
        try:
            import json as _json
            import urllib.request as _u
            body = _json.dumps({
                "query": "mutation($id: String!) { podStop(input: {podId: $id}) "
                         "{ id desiredStatus } }",
                "variables": {"id": pid},
            }).encode()
            req = _u.Request(f"https://api.runpod.io/graphql?api_key={key}", data=body,
                             headers={"Content-Type": "application/json"})
            with _u.urlopen(req, timeout=45) as resp:
                payload = _json.loads(resp.read().decode())
            if payload.get("errors"):
                msg = payload["errors"][0].get("message", "unknown error")
                self.update_console(f"[pod] RunPod refused the stop: {msg}\n"
                                    f"[pod] Stop it from the dashboard to stop billing.\n")
            else:
                self.update_console("[pod] stop requested — this pod is shutting down.\n")
        except Exception as e:
            self.update_console(f"[pod] auto-stop failed ({type(e).__name__}: {e}). "
                                f"Stop it from the dashboard to stop billing.\n")

    def _on_training_subprocess_exited(self, return_code: int):
        """Called from check_process when the training subprocess ends. Routes to paused or idle."""
        # Captured BEFORE the branches below rewrite it — a pause is the one case that exits 0,
        # and telling it apart from a completed run is the whole basis of the auto-stop decision.
        was_state = getattr(self, "training_state", "idle")
        # Clean up pause flag if still present
        try:
            flag = self._pause_flag_path()
            if os.path.exists(flag):
                os.remove(flag)
        except Exception:
            pass
        # Only a run that finished on its own. Pause exits 0 too (state "pausing"); Stop and
        # crashes arrive non-zero. Each of the three conditions excludes a real case, and getting
        # it wrong shuts the machine down under someone who is still using it.
        # Every run exit invalidates whatever advance/retry timers were armed before it —
        # a stale tick must never fire into the state this exit is about to establish.
        self._cancel_pending_queue_advance()
        _queue_advancing = False
        if return_code == 0 and was_state == "running":
            if getattr(self, "training_queue", None):
                # Queue takes precedence over pod auto-stop: the pod must stay up until the
                # LAST queued run finishes — that final run's clean exit lands here with an
                # empty queue and fires the auto-stop as usual.
                _queue_advancing = True
                self._queue_busy_retries = 0
                self.update_console(f"\n[queue] run finished — next of "
                                    f"{len(self.training_queue)} queued run(s) starts in 5 s.\n")
                self._schedule_queue_advance(5000)
            else:
                self._maybe_stop_pod_after_training()
        elif getattr(self, "training_queue", None):
            # Failure, Stop, or Pause with runs still waiting: never cascade into the queue —
            # a crash loop through N queued runs would burn hours producing nothing. The queue
            # holds; the user restarts it from the queue window.
            if was_state == "pausing" and return_code == 0:
                self.update_console(f"[queue] run paused — {len(self.training_queue)} queued "
                                    f"run(s) are HELD. Resume the paused run from the Training "
                                    f"tab first; the queue continues after it finishes.\n")
            else:
                self.update_console(
                    f"[queue] run did not finish cleanly (exit {return_code}) — "
                    f"{len(self.training_queue)} queued run(s) are HELD. The FAILED run is not "
                    f"in the queue: its settings are still loaded in the Training tab (fix and "
                    f"Start Training to retry it), or open the queue (📋, bottom right) and "
                    f"'Start next now' to skip to the next job.\n")
        if getattr(self, "training_state", "idle") == "pausing" and return_code == 0:
            # Successful graceful exit — record paused state. A fine-tune leaves a full
            # checkpoint rather than a state dir (state dirs would hold only the inert
            # LoRA), so the paused artifact is detected per the launch's stamped mode.
            _ft_pause = bool(getattr(self, "_launched_ft_family", None))
            if _ft_pause:
                state_dir = self._detect_latest_ft_checkpoint()
            else:
                state_dir = self._detect_latest_state_dir()
            if state_dir is None:
                self.update_console(
                    "[pause] WARN: no fine-tune checkpoint found after pause exit. Treating as idle.\n"
                    if _ft_pause else
                    "[pause] WARN: no state directory found after pause exit. Treating as idle.\n")
                self.training_state = "idle"
            else:
                self.paused_state_path = state_dir
                self.paused_mode = "ft" if _ft_pause else "state"
                # Persist sidecar so paused state survives GUI restart
                try:
                    import json as _json
                    out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
                    sidecar = {
                        "mode": self.paused_mode,
                        "state_path": state_dir,
                        "output_name": self.settings.get("LORA_NAME", ""),
                        "dataset_config": self.settings.get("DATASET_CONFIG", ""),
                        "network_dim": str(self.settings.get("NETWORK_DIM", "")),
                        "network_alpha": str(self.settings.get("NETWORK_ALPHA", "")),
                        "max_train_epochs": str(self.settings.get("MAX_TRAIN_EPOCHS", "")),
                    }
                    with open(self._paused_sidecar_path(), "w") as f:
                        _json.dump(sidecar, f, indent=2)
                except Exception as e:
                    self.update_console(f"[pause] WARN: failed to write sidecar: {e}\n")
                self.training_state = "paused"
                self.update_console(
                    f"\n=== PAUSED — fine-tune checkpoint saved at {os.path.basename(state_dir)}. "
                    f"Click Resume Training to continue. ===\n\n"
                    if _ft_pause else
                    f"\n=== PAUSED — state saved at {state_dir}. Click Resume Training to continue. ===\n\n"
                )
        else:
            self.training_state = "idle"
        # A finished run must never leave its resume path armed: the next "fresh" Start
        # would silently continue the old LoRA from its saved state (restored optimizer/
        # RNG/scheduler) under a new output name, and skip re-caching a changed dataset.
        # The Resume button re-injects the right path itself when the user wants it.
        try:
            _entry = self.entries.get("RESUME_TRAINING")
            if _entry is not None and _entry.get().strip():
                _entry.delete(0, tk.END)
            self.settings["RESUME_TRAINING"] = ""
        except Exception:
            pass
        # Same hygiene for the FT twin: the armed continuation lives until its run ends
        # (it must survive the async caption-worker launch path), then dies here.
        self._ft_resume = None
        self._refresh_training_buttons()

    def _resume_training(self):
        """Re-launch training from the latest paused state (state dir, or FT checkpoint).

        A LoRA pause resumes via --resume <state dir>. A rotation fine-tune has no state
        dirs — its continuation is a fresh run whose --dit is the pause checkpoint, whose
        rotation cycle picks up at the window stamped in that checkpoint's metadata, and
        whose epoch count is what's left of the original total. The optimizer's second
        moments reset across that hop — the same reset every rotation boundary already
        performs, so it costs what one rotation costs."""
        if getattr(self, "training_state", "idle") != "paused":
            messagebox.showinfo("Not Paused", "No paused training to resume.")
            return
        state_path = getattr(self, "paused_state_path", None)
        if getattr(self, "paused_mode", "state") == "ft":
            if not state_path or not os.path.isfile(state_path):
                messagebox.showerror("Error", f"Paused fine-tune checkpoint not found:\n{state_path}")
                return
            next_window, epochs_done = ft_checkpoint_continuation(state_path)
            try:
                total = int(str(self.entries["MAX_TRAIN_EPOCHS"].get()).strip() or 0)
            except (KeyError, ValueError, TypeError):
                total = 0
            if total - epochs_done <= 0:
                messagebox.showinfo(
                    "Nothing left to train",
                    f"This fine-tune has already trained {epochs_done} epoch(s) — at or past "
                    f"Max Train Epochs ({total}).\n\n"
                    f"{os.path.basename(state_path)} IS the finished model — deploy it as-is.\n\n"
                    f"To train it further, raise Max Train Epochs above {epochs_done} and click "
                    "Resume Training again.")
                return
            # Arm the one-shot continuation. The command builders consume it (guarded by
            # family + output name); the exit handler clears it when the run ends.
            self._ft_resume = {"checkpoint": state_path, "next_window": next_window,
                               "epochs_done": epochs_done,
                               "output_name": self.settings.get("LORA_NAME", "")}
            self.update_console(
                f"\n=== RESUMING fine-tune from {os.path.basename(state_path)} — "
                f"{total - epochs_done} epoch(s) remaining, rotation window {next_window} ===\n\n")
        else:
            if not state_path or not os.path.isdir(state_path):
                messagebox.showerror("Error", f"Paused state directory not found:\n{state_path}")
                return
            # Inject resume path into settings + entry field, then reuse the standard start_training flow
            self.settings["RESUME_TRAINING"] = state_path
            try:
                entry = self.entries.get("RESUME_TRAINING")
                if entry is not None:
                    entry.delete(0, tk.END)
                    entry.insert(0, state_path)
            except Exception:
                pass
            self.update_console(f"\n=== RESUMING from {state_path} ===\n\n")
        self.start_training()
        # Only a start that actually LAUNCHED consumes the pause. start_training can decline
        # (validation, disk headroom, epochs-left) — destroying the sidecar and flipping the
        # state beforehand stranded a declinable resume with no way back to the paused run.
        _proc = getattr(self, "current_process", None)
        if getattr(self, "training_state", "idle") == "running" and _proc is not None and _proc.poll() is None:
            try:
                sidecar = self._paused_sidecar_path()
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            except Exception:
                pass
        else:
            self.training_state = "paused"
            self._refresh_training_buttons()
            self.update_console("[resume] start declined — the run is still PAUSED and can be "
                                "resumed once the issue above is fixed.\n")

    def _check_for_paused_state_on_startup(self):
        """On GUI launch, detect a leftover paused state and restore the Resume button."""
        try:
            sidecar = self._paused_sidecar_path()
            if not os.path.exists(sidecar):
                return
            import json as _json
            with open(sidecar, "r") as f:
                meta = _json.load(f)
            state_path = meta.get("state_path", "")
            # "ft" pauses point at a full checkpoint FILE (fine-tunes have no state dirs);
            # LoRA pauses point at a state DIRECTORY. Validate whichever this one is.
            _mode = str(meta.get("mode", "state") or "state")
            _ok = bool(state_path) and (os.path.isfile(state_path) if _mode == "ft"
                                        else os.path.isdir(state_path))
            if _ok:
                self.paused_state_path = state_path
                self.paused_mode = _mode
                # Restore the paused run's frozen dataset config (#98) so a cross-restart
                # resume trains the dataset it started with — the resume launch keeps an
                # existing snapshot instead of re-freezing whatever the Start tab shows.
                _dc = str(meta.get("dataset_config", "") or "")
                if (os.path.basename(os.path.dirname(_dc)) == self._RUN_SNAPSHOT_DIRNAME
                        and os.path.isfile(_dc)):
                    self.settings["DATASET_CONFIG"] = _dc
                self.training_state = "paused"
                self._refresh_training_buttons()
                self.update_console(
                    f"=== Paused {'fine-tune' if _mode == 'ft' else 'training'} detected: "
                    f"{meta.get('output_name','?')} "
                    f"at {os.path.basename(state_path)}. Click Resume Training to continue. ===\n"
                )
        except Exception:
            pass

    def _on_app_close(self):
        """WM_DELETE_WINDOW: never orphan a live training subprocess on window close."""
        proc = getattr(self, "current_process", None)
        try:
            running = proc is not None and proc.poll() is None
        except Exception:
            running = False
        if running:
            if not messagebox.askyesno(
                "Training in progress",
                "A training run is active.\n\n"
                "Close Fizgig and STOP the training run?\n\n"
                "(To keep training, click No — or use Pause Training first for a clean, "
                "resumable exit.)"
            ):
                return
            try:
                self.stop_training()
            except Exception:
                pass
        # Final settings snapshot — some fields only persist via debounced traces or other
        # tabs' events, so closing mid-edit would otherwise drop the last change.
        try:
            self._save_last_used_paths()
        except Exception:
            pass
        try:
            self._stop_caption_worker(silent=True, wait=False)
        except Exception:
            pass
        try:
            self.master.destroy()
        except Exception:
            pass

    def stop_training(self):
        """Stop the current running process"""
        # Stop samples watcher
        self.stop_samples_watcher()
        # A user Stop invalidates any armed queue-advance/retry timer immediately — the
        # exit handler bumps too, but for non-pipeline kills this is the only bump.
        try:
            self._cancel_pending_queue_advance()
        except Exception:
            pass
        # Snapshot: check_process's worker nulls self.current_process the instant the kill
        # lands, racing the .wait(timeout=5) below into an AttributeError on None.
        _proc = self.current_process
        if _proc and _proc.poll() is None:
            try:
                if os.name == 'nt':
                    # CREATE_NO_WINDOW prevents CTRL_BREAK_EVENT from working,
                    # so terminate the process tree via taskkill instead.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(_proc.pid)],
                        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
            except Exception as e:
                self.update_console("Error stopping process: " + str(e) + "\n")
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    _proc.kill()
                    _proc.wait()
                except Exception as e:
                    self.update_console("Error killing process: " + str(e) + "\n")
            self.current_process = None
            if self.training_thread:
                self.training_thread.join(timeout=1)
                self.training_thread = None
            self.update_console("Training stopped\n")
        else:
            self.update_console("No active process to stop\n")

    # (save_settings/load_settings removed: 160 lines of dead code with no
    #  callers, duplicating the preset system with a 4-key save/load asymmetry.)

if __name__ == "__main__":
    # Set unique app ID so Windows taskbar shows our icon, not Python's
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('fizgig.lora.studio')
    except Exception:
        pass
    root = tk.Tk()
    gui = LoRATrainerGUI(root)
    # Detect leftover paused training state from a prior session
    try:
        gui._check_for_paused_state_on_startup()
    except Exception:
        pass
    # Quiet update check a couple of seconds in — off the critical launch path, and it only
    # ever surfaces as the About-button label flipping to "Update Available".
    try:
        root.after(2500, gui._start_update_check)
    except Exception:
        pass
    root.mainloop()
