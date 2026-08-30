import json
import os

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

# Directory for dataset configurations
DATASET_DIR = os.path.join(_REPO_ROOT, "dataset")

# Directory for cache (latents and text encoder outputs)
CACHE_DIR = os.path.join(_REPO_ROOT, "cache")

# Directory for output LoRAs
OUTPUT_LORAS_DIR = os.path.join(_REPO_ROOT, "output_loras")

# File for storing last-used folder paths
LAST_USED_FILE = os.path.join(_REPO_ROOT, ".last_used.json")


def load_last_used():
    """Load last-used folder paths from config file"""
    defaults = {
        "image_prep_source": "",
        "image_folder": "",  # Start tab: training image folder (shared with Captions)
        "image_folder2": "",  # Multi Concept (MiniMax): second subject, TRAINING ONLY
        "caption_trigger": "trigger_word",
        "dataset_cache_dir": CACHE_DIR,
        "sample_prompt": "A high quality photo",
    }
    if os.path.exists(LAST_USED_FILE):
        try:
            with open(LAST_USED_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    # Migrate pre-Start-tab last_used files: if image_folder isn't set but one
    # of the legacy keys (caption_folder / dataset_image_dir / image_prep_source)
    # has a value, seed image_folder from the best candidate.
    if not defaults.get("image_folder"):
        for legacy_key in ("caption_folder", "dataset_image_dir", "image_prep_source"):
            legacy_val = defaults.get(legacy_key, "")
            if legacy_val:
                defaults["image_folder"] = legacy_val
                break
    return defaults


def save_last_used(data):
    """Save last-used folder paths to config file.

    Atomic (tmp + os.replace): this file is rewritten on every traced-var edit, and its
    reader falls back to defaults on a JSONDecodeError — so a crash mid-write used to
    silently blank the remembered paths, and the next auto-save persisted the blanks."""
    try:
        tmp = LAST_USED_FILE + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, LAST_USED_FILE)
    except Exception:
        pass
