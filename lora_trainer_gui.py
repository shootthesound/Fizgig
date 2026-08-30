import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu, scrolledtext
import subprocess
import sys
import threading
import json
import os
import math
import re
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


from fizgig_gui.core.ui_base.widgets import _GUIWriter, ToolTip

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

from fizgig_gui.core.domain.minimax_math import MINIMAX_BASE_QUANT_OPTIONS

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
from fizgig_gui.core.tabs.samples_tab import SamplesTabMixin

from fizgig_gui.core.models.caption_model import CaptionModelMixin
from fizgig_gui.core.models.gallery_likeness import GalleryLikenessMixin

from fizgig_gui.core.ui_base.console_validation import ConsoleValidationMixin

from fizgig_gui.core.training.training_engine import TrainingEngineMixin
from fizgig_gui.core.training.training_preset import TrainingPresetMixin
from fizgig_gui.core.training.training_ui import TrainingUiMixin
from fizgig_gui.core.training.training_visibility import TrainingVisibilityMixin

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


class LoRATrainerGUI(
    StartTabMixin, MetadataTabMixin, PrefsTabMixin,
    ProfilerTabMixin, ExplorerTabMixin, ExtractTabMixin,
    RepairStudioTabMixin, SamplesTabMixin, StylingMixin,
    TabScaffoldMixin, TrainingEngineMixin, TrainingPresetMixin,
    TrainingUiMixin, TrainingVisibilityMixin, CaptionModelMixin,
    GalleryLikenessMixin, CaptionTabMixin, ConsoleValidationMixin
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
