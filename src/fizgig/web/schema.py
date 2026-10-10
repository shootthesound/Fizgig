"""Schema extraction for Fizgig's schema-driven Web UI.

Extracts model family descriptions directly from `fizgig.families.registry` and
declarative CLI argument definitions.
"""
import dataclasses
from typing import Dict, Any, List
from fizgig.families.registry import training_families, get as get_family


def _jsonable(obj: Any) -> Any:
    """Turns anything a FamilyDescription holds (nested dataclasses, tuples, dicts) into plain JSON data.

    Walks dataclasses.fields() instead of naming fields, so a field added to FamilyDescription (or to the
    SamplingSettings / SpeedLoRA / ModelFile / LoRAFormat records inside it) reaches the web UI with no change here.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def describe_family(f) -> Dict[str, Any]:
    """One family's full declared description, ready for the page."""
    data = _jsonable(f)
    # Presets are declared as ((name, {setting key: value}), ...); the page wants an ordered list.
    data["presets"] = [{"name": name, "values": _jsonable(vals)} for name, vals in (f.presets or ())]
    data["training_ready"] = f.training_ready
    return data


def get_schema() -> Dict[str, Any]:
    """Generates the full schema used by the Web UI to render forms and preset chips.

    Families come from the registry on every call: a family that registers with a driver shows up in the page's
    selector, Preferences model paths, presets, run types and option lists without any web change.
    """
    families_data = {f.key: describe_family(f) for f in training_families()}

    # ---- Legacy families that haven't migrated to the registry yet ----
    # Klein, Krea 2 and MiniMax H3 still train via the desktop app's own code paths.  Until
    # their FamilyDescription+driver land in src/fizgig/families/, the web UI needs stub
    # entries so they appear in the selector, Preferences model paths, and preset chips.
    # When a family registers with a driver, training_families() returns it and the stub
    # is overwritten by the real description above — zero manual removal.
    _legacy = [
        {
            "key": "flux_klein",
            "display_name": "Flux 2 Klein Base 9B",
            "gui_label": "Flux 2 Klein Base 9B",
            "lora_name_suffix": "k9b",
            "training_ready": False,
            "experimental": False,
            "edit_training": False,
            "slider_training": False,
            "training_adapter": "",
            "training_adapter_note": "",
            "ema_default": "",
            "block_note": "Double blocks 0-7, Single blocks 0-23 (32 total)",
            "n_blocks": 32,
            "optimizers": ["adamw8bit", "adamw", "prodigy", "lion8bit", "schedule_free"],
            "network_types": ["lora"],
            "precisions": ["bf16"],
            "presets": [],
            "model_files": [
                {"pref_key": "base_dit", "label": "Base DiT", "required": True, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "Klein Base 9B .safetensors", "local_name": ""},
                {"pref_key": "distilled_dit", "label": "Distilled DiT", "required": False, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "For distilled-mode samples only", "local_name": ""},
                {"pref_key": "vae", "label": "AE Model (ae.safetensors)", "required": True, "role": "vae",
                 "repo": "", "path": "", "size_gb": 0, "note": "NOT the Diffusers subfolder VAE", "local_name": ""},
                {"pref_key": "text_encoder", "label": "Text Encoder (Qwen3-8B)", "required": True, "role": "text_encoder",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
            ],
            "workbench": ["repair", "profiler", "explorer", "extract", "royale"],
            "notes": [],
        },
        {
            "key": "krea2",
            "display_name": "Krea 2",
            "gui_label": "Krea 2 (RAW 12.9B)",
            "lora_name_suffix": "krea2",
            "training_ready": False,
            "experimental": False,
            "edit_training": False,
            "slider_training": False,
            "training_adapter": "",
            "training_adapter_note": "",
            "ema_default": "",
            "block_note": "Blocks 0-18 (19 total)",
            "n_blocks": 19,
            "optimizers": ["adamw8bit", "adamw", "prodigy"],
            "network_types": ["lora", "lokr"],
            "precisions": ["bf16", "int8"],
            "presets": [],
            "model_files": [
                {"pref_key": "krea2_raw_dit", "label": "RAW DiT", "required": True, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "Krea 2 RAW 12.9B", "local_name": ""},
                {"pref_key": "krea2_turbo_dit", "label": "Turbo DiT (fp8)", "required": False, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "For 8-step sample previews", "local_name": ""},
                {"pref_key": "krea2_vae", "label": "Qwen-Image VAE", "required": True, "role": "vae",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "krea2_text_encoder", "label": "Qwen3-VL Text Encoder", "required": True, "role": "text_encoder",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "krea2_turbo_lora", "label": "Turbo LoRA (rank 64)", "required": False, "role": "speed_lora",
                 "repo": "", "path": "", "size_gb": 0, "note": "Speeds up sample previews", "local_name": ""},
            ],
            "workbench": ["repair", "profiler", "extract", "royale"],
            "notes": [],
        },
        {
            "key": "minimax_h3",
            "display_name": "MiniMax H3",
            "gui_label": "MiniMax H3 (Video & Voice)",
            "lora_name_suffix": "mmh3",
            "training_ready": False,
            "experimental": False,
            "edit_training": False,
            "slider_training": False,
            "training_adapter": "minimax_training_adapter",
            "training_adapter_note": "",
            "ema_default": "",
            "block_note": "h3blk 0-49 (50 total)",
            "n_blocks": 50,
            "optimizers": ["adamw8bit", "adamw"],
            "network_types": ["lora"],
            "precisions": ["bf16", "nf4"],
            "presets": [],
            "model_files": [
                {"pref_key": "minimax_dit", "label": "DiT", "required": True, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "MiniMax H3 DiT", "local_name": ""},
                {"pref_key": "minimax_ref_dit", "label": "DiT (reference)", "required": False, "role": "dit",
                 "repo": "", "path": "", "size_gb": 0, "note": "RefMod reference DiT", "local_name": ""},
                {"pref_key": "minimax_text_encoder", "label": "Qwen3-VL-32B Text Encoder", "required": True, "role": "text_encoder",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_vae", "label": "Video VAE", "required": True, "role": "vae",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_audio_vae", "label": "Audio VAE", "required": False, "role": "vae",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_turbo_lora", "label": "Turbo LoRA", "required": False, "role": "speed_lora",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_circlestone_adapter", "label": "Training adapter (Circlestone)", "required": False, "role": "training_adapter",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_training_adapter", "label": "Training adapter (Ostris fl2va)", "required": False, "role": "training_adapter",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
                {"pref_key": "minimax_ref_training_adapter", "label": "Training adapter (Ostris ref2va)", "required": False, "role": "training_adapter",
                 "repo": "", "path": "", "size_gb": 0, "note": "", "local_name": ""},
            ],
            "workbench": ["repair", "profiler", "explorer", "extract", "royale"],
            "notes": [],
        },
    ]
    for stub in _legacy:
        if stub["key"] not in families_data:
            families_data[stub["key"]] = stub

    # Grouped arguments for Advanced section
    argument_sections = [
        {
            "id": "training_core",
            "title": "Training & Network Parameters",
            "fields": [
                {"name": "NETWORK_DIM", "label": "Network Dim (Rank)", "type": "int", "default": 16, "min": 1, "help": "LoRA rank/dimension"},
                {"name": "NETWORK_ALPHA", "label": "Network Alpha", "type": "float", "default": 16, "min": 0, "help": "Scaling factor (usually equal to rank)"},
                {"name": "LEARNING_RATE", "label": "Learning Rate", "type": "float", "default": 0.0001, "min": 0, "help": "Base learning rate (e.g. 1e-4)"},
                {"name": "MAX_TRAIN_EPOCHS", "label": "Max Train Epochs", "type": "int", "default": 30, "min": 1, "help": "Total epochs to train"},
                {"name": "SAVE_EVERY_N_EPOCHS", "label": "Save Every N Epochs", "type": "int", "default": 1, "min": 1, "help": "Checkpoint frequency"},
                {"name": "SEED", "label": "Seed", "type": "int", "default": 42, "help": "Random seed for reproducibility"},
            ]
        },
        {
            "id": "memory_perf",
            "title": "Memory & Performance",
            "fields": [
                {"name": "precision", "label": "Precision", "type": "select", "choices": ["Auto (fits your free VRAM)", "bf16 (full precision)", "INT8 (8-bit, fastest)", "4-bit NF4 (smallest)"], "default": "Auto (fits your free VRAM)", "help": "Quantization mode for the base DiT model"},
                {"name": "blocks_swap", "label": "Blocks to Swap", "type": "text", "default": "Auto (detect from GPU)", "help": "Offload transformer blocks between CPU and GPU"},
                {"name": "FAMILY_TRAINING_ADAPTER", "label": "Use Training Adapter", "type": "bool", "default": True, "help": "Keeps Qwen 2.1 LoRA training stable during optimization"},
                {"name": "SAVE_STATE", "label": "Save Resumable States", "type": "bool", "default": True, "help": "Save optimizer states for resuming training later"},
            ]
        },
        {
            "id": "adaptive_lr",
            "title": "Adaptive Learning Rate",
            "fields": [
                {"name": "ADAPTIVE_LR", "label": "Enable Adaptive LR", "type": "bool", "default": True, "help": "Automatically adjusts LR according to loss convergence"},
                {"name": "ADAPTIVE_LR_MIN", "label": "Adaptive Min LR", "type": "text", "default": "1e-4", "help": "Lower bound for adaptive learning rate"},
                {"name": "ADAPTIVE_LR_MAX", "label": "Adaptive Max LR", "type": "text", "default": "2e-4", "help": "Upper bound for adaptive learning rate"},
            ]
        },
        {
            "id": "loss_watch",
            "title": "Loss Watch & Outliers",
            "fields": [
                {"name": "KREA2_LOSS_WATCH", "label": "Log Per-Image Loss", "type": "bool", "default": True, "help": "Track individual image loss trajectory to catch bad samples"},
                {"name": "KREA2_PER_IMAGE_LR", "label": "Per-Image LR Scaling", "type": "bool", "default": False, "help": "Scale learning rate based on per-image loss spikes"},
                {"name": "KREA2_WARMUP_LOOK", "label": "Warm Up Look Outliers", "type": "bool", "default": False, "help": "Gentle LR ramp-up for statistical visual outliers"},
            ]
        },
        {
            "id": "samples_preview",
            "title": "In-Training Sample Previews",
            "fields": [
                {"name": "sample_enabled", "label": "Enable Previews", "type": "bool", "default": True, "help": "Generate sample image previews during training"},
                {"name": "sample_every", "label": "Sample Every N Epochs", "type": "int", "default": 1, "min": 1, "help": "Epoch interval for rendering sample previews"},
                {"name": "sample_width", "label": "Width", "type": "int", "default": 1024, "help": "Preview generation width"},
                {"name": "sample_height", "label": "Height", "type": "int", "default": 1024, "help": "Preview generation height"},
                {"name": "sample_steps", "label": "Steps", "type": "int", "default": 20, "help": "Denoising steps for preview"},
                {"name": "sample_prompts", "label": "Preview Prompts (one per line)", "type": "textarea", "default": "a photo of a person\na cinematic portrait in natural lighting", "help": "Prompts to render at each checkpoint"},
            ]
        }
    ]

    return {
        "families": families_data,
        "default_family": next(iter(families_data), ""),
        "sections": argument_sections,
    }
