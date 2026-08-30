import tkinter as tk
from tkinter import ttk, Menu
import sys
import threading
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

from fizgig_gui.core.config.constants import COLORS, BG_COLOR


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

from fizgig_gui.core.shell import ShellMixin
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

from fizgig_gui.core.config.last_used import DATASET_DIR, OUTPUT_LORAS_DIR, load_last_used
from fizgig_gui.core.config.prefs import DEFAULT_PREFS, _apply_cuda_device_pref, load_prefs


class LoRATrainerGUI(
    StartTabMixin, MetadataTabMixin, PrefsTabMixin,
    ProfilerTabMixin, ExplorerTabMixin, ExtractTabMixin,
    RepairStudioTabMixin, SamplesTabMixin, StylingMixin,
    TabScaffoldMixin, TrainingEngineMixin, TrainingPresetMixin,
    TrainingUiMixin, TrainingVisibilityMixin, CaptionModelMixin,
    GalleryLikenessMixin, CaptionTabMixin, ConsoleValidationMixin,
    DatasetAnalysisMixin, DatasetConfigMixin, ImageConverterMixin,
    ImagePrepMixin, LookConsistencyMixin, ShellMixin
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
