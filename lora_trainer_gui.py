import tkinter as tk
from tkinter import ttk, messagebox, Menu
import subprocess
import sys
import threading
import json
import os
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

from fizgig_gui.core.dataset.dataset_analysis import DatasetAnalysisMixin
from fizgig_gui.core.dataset.dataset_config import DatasetConfigMixin
from fizgig_gui.core.dataset.image_converter import ImageConverterMixin
from fizgig_gui.core.dataset.image_prep import ImagePrepMixin
from fizgig_gui.core.dataset.look_consistency import LookConsistencyMixin

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
    GalleryLikenessMixin, CaptionTabMixin, ConsoleValidationMixin,
    DatasetAnalysisMixin, DatasetConfigMixin, ImageConverterMixin,
    ImagePrepMixin, LookConsistencyMixin
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
