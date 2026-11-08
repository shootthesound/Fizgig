"""Fetch the small community LoRAs Fizgig leans on and populate their Preferences paths.

Runs from the update scripts (never the installer — a fresh install sets its models up from
the Preferences download button). Idempotent by design — safe to leave in update_fizgig.bat
forever, and it never fails the update: offline or a bad download is a logged line.

For each entry:
  - pref already set and the file exists ..... no-op ("already present")
  - file already in <repo>/models/ ........... just populates the pref
  - neither ................................... downloads with progress, verifies, populates

Entries: the Krea 2 Turbo LoRA (previews render on RAW + Turbo LoRA, no model swapping), the
Qwen Image 2.1 Turbo LoRA and Fizgig's Qwen Image 2.1 training adapter (v2), and the MiniMax H3 training adapters — Circlestone's (the default, one file for both bases) and
Ostris's two (per base) — which the Training tab's adapter dropdown chooses between. Always fetched when missing,
whatever family is configured (Peter, 2 Sep 2026: no "not a Krea 2 install" gate, ever).
The GUI calls ensure_turbo_lora() again at Krea 2 training start as a fallback.
"""
import json
import os
import struct
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _prefs_file(repo_dir):
    """prefs.json: a second Fizgig on another GPU keeps its own (FIZGIG_USER_DATA); the model files stay shared."""
    own = os.environ.get("FIZGIG_USER_DATA")
    if own and os.path.normcase(os.path.abspath(repo_dir)) == os.path.normcase(REPO_DIR):
        return os.path.join(own, "prefs.json")
    return os.path.join(repo_dir, "user_data", "prefs.json")


if os.path.join(REPO_DIR, "src") not in sys.path:      # the updater runs this file by path
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))
from fizgig import user_data  # noqa: E402
user_data.migrate()                                     # an update from Fizgig 7: its prefs into user_data first

# (pref key, filename, url, a key that must exist in the genuine file, minimum size, label)
# The probe key catches truncated/HTML-error downloads beyond what a size check can.
LORAS = [
    ("krea2_turbo_lora", "krea2_turbo_lora_rank_64_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/"
     "krea2_turbo_lora_rank_64_bf16.safetensors",
     "diffusion_model.blocks.0.attn.wq.lora_down.weight", 400 * 1024 * 1024,
     "Krea 2 Turbo LoRA (~470 MB)"),
    ("minimax_circlestone_adapter", "minimax_h3_image_training_adapter.safetensors",
     "https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter/resolve/main/"
     "minimax_h3_image_training_adapter.safetensors",
     "diffusion_model.blocks.0.attn.out_proj.lora_A.weight", 580 * 1024 * 1024,
     "MiniMax H3 training adapter, Circlestone (~620 MB, the default)"),
    ("minimax_training_adapter", "minimax_h3_training_adapter_v1.safetensors",
     "https://huggingface.co/ostris/minimax_h3_training_adapter/resolve/main/"
     "minimax_h3_training_adapter_v1.safetensors",
     "diffusion_model.blocks.0.attn.out_proj.lora_A.weight", 140 * 1024 * 1024,
     "MiniMax H3 training adapter, fl2va (~155 MB, Ostris)"),
    ("minimax_ref_training_adapter", "minimax_h3_ref2va_training_adapter_v1.safetensors",
     "https://huggingface.co/ostris/minimax_h3_training_adapter/resolve/main/"
     "minimax_h3_ref2va_training_adapter_v1.safetensors",
     "diffusion_model.blocks.0.attn.out_proj.lora_A.weight", 140 * 1024 * 1024,
     "MiniMax H3 training adapter, ref2va (~155 MB, Ostris)"),
    # Qwen Image 2.1's Turbo LoRA (Comfy-Org) replaced Viggle's in the same pref: an install still pointing at
    # another file gets this one (the old file stays on disk)
    ("qwen21_turbo_lora", "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/loras/"
     "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
     "diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight", 850 * 1024 * 1024,
     "Qwen Image 2.1 Turbo LoRA (~910 MB)", True),
    # Fizgig's Qwen Image 2.1 training adapter, v2: a pref on v1 (or none) moves to it, v1 stays on disk; another
    # adapter someone chose is kept
    ("qwen21_training_adapter", "fizgig_qwen_image_2.1_training_adapter_v2.safetensors",
     "https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter/resolve/main/"
     "fizgig_qwen_image_2.1_training_adapter_v2.safetensors",
     "transformer.transformer_blocks.0.attn.to_q.lora_A.weight", 75 * 1024 * 1024,
     "Qwen Image 2.1 Fizgig training adapter v2 (~84 MB)", ("fizgig_qwen_image_2.1_training_adapter.safetensors",)),
]

# Kept for callers that import the Krea 2 constants by name.
LORA_FILENAME = LORAS[0][1]
LORA_URL = LORAS[0][2]
PROBE_KEY = LORAS[0][3]
MIN_SIZE = LORAS[0][4]


def _valid_safetensors(path: str, probe_key: str = PROBE_KEY, min_size: int = MIN_SIZE) -> bool:
    """Header parses and the probe key is present — cheap, reads no tensor data."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100 * 1024 * 1024:      # a sane header is a few hundred KB
                return False
            hdr = json.loads(f.read(n))
        return probe_key in hdr and os.path.getsize(path) >= min_size
    except Exception:
        return False


def _download(url: str, dest: str, log, probe_key: str = PROBE_KEY,
              min_size: int = MIN_SIZE) -> bool:
    """Download to dest atomically (.tmp then replace). Returns success."""
    import urllib.request
    tmp = dest + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Fizgig-updater"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            next_pct = 10
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_pct:
                    log(f"  ... {done * 100 // total}% ({done // (1024*1024)} MB)")
                    next_pct += 10
        if not _valid_safetensors(tmp, probe_key, min_size):
            log("  downloaded file failed verification — discarded")
            os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except Exception as e:
        log(f"  download failed: {type(e).__name__}: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def _load_prefs(prefs_file):
    if os.path.exists(prefs_file):
        with open(prefs_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_pref(prefs_file, key, value):
    """Populate one pref atomically, preserving everything else in prefs.json."""
    prefs = _load_prefs(prefs_file)
    prefs[key] = value
    tmp = prefs_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    os.replace(tmp, prefs_file)


def ensure_lora(pref_key, filename, url, probe_key, min_size, label, replaces=False,
                repo_dir: str = REPO_DIR, log=print):
    """Make sure one LoRA exists locally and prefs.json points at it. Returns the path on
    success, None otherwise. Never raises. replaces: True = the pref must point at THIS file (it
    took over a pref that held another one); a tuple of file names = earlier versions of this file, moved to it
    (any other file there is the user's own choice and kept)."""
    try:
        prefs_file = _prefs_file(repo_dir)
        current = str(_load_prefs(prefs_file).get(pref_key) or "").strip()
        old = isinstance(replaces, (tuple, list)) and os.path.basename(current) in replaces
        if current and os.path.isfile(current) and not old and (replaces is not True
                                                                or os.path.basename(current) == filename):
            log(f"{label}: already present ({current})")
            return current
        models_dir = os.path.join(repo_dir, "user_data", "models")
        dest = os.path.join(models_dir, filename)
        if not (os.path.isfile(dest) and _valid_safetensors(dest, probe_key, min_size)):
            os.makedirs(models_dir, exist_ok=True)
            log(f"Downloading {label} -> {dest}")
            log(f"  from {url}")
            if not _download(url, dest, log, probe_key, min_size):
                return None
        else:
            log(f"{label} found at {dest} — linking it into Preferences.")
        _save_pref(prefs_file, pref_key, dest)
        log(f"Preferences updated: {pref_key} -> {dest}")
        return dest
    except Exception as e:
        log(f"{label}: setup skipped ({type(e).__name__}: {e})")
        return None


def ensure_turbo_lora(repo_dir: str = REPO_DIR, log=print, require: bool = False):
    """The Krea 2 Turbo LoRA only (the GUI's training-start fallback). `require` is accepted
    for older callers and ignored — every call always tries."""
    key, fn, url, probe, size, label = LORAS[0]
    return ensure_lora(key, fn, url, probe, size, label, repo_dir=repo_dir, log=log)


def ensure_all(repo_dir: str = REPO_DIR, log=print):
    """Every entry in LORAS, in order; a failure on one never stops the next."""
    return {e[0]: ensure_lora(*e, repo_dir=repo_dir, log=log) for e in LORAS}


def qwen_turbo_dit_hint(repo_dir: str = REPO_DIR, log=print):
    """A Qwen Image 2.1 install without the Turbo DiT gets told how to fetch it (7.3 GB - too big for the
    updater to pull unasked). Runs after the update's git pull, so it reaches this update, not the next."""
    try:
        prefs = _load_prefs(_prefs_file(repo_dir))
        qwen = str(prefs.get("qwen21_dit") or "").strip()
        turbo = str(prefs.get("qwen21_turbo_dit") or "").strip()
        if qwen and os.path.isfile(qwen) and not (turbo and os.path.isfile(turbo)):
            log("")
            log("Qwen Image 2.1: the new Turbo model (7.3 GB) for fast workbench previews isn't downloaded yet.")
            log("  Open Fizgig, go to the Preferences tab, and in the Qwen Image 2.1 section press")
            log("  'Download models for me'. Until then, previews work as before.")
    except Exception:
        pass


def ensure_taeh3(repo_dir: str = REPO_DIR, log=print):
    """MiniMax H3's quick-preview decoder (taeh3, 22.7 MB) for an install with H3 set up - so updating users get it
    here, before the first preview wants it. Never raises; without a connection the previews use the full decoder."""
    try:
        prefs = _load_prefs(_prefs_file(repo_dir))
        if not str(prefs.get("minimax_dit") or "").strip():
            return None
        from fizgig.minimax import taeh3
        return taeh3.fetch(os.path.join(repo_dir, "user_data", "models"), log=log)
    except Exception as e:
        log(f"H3 quick-preview decoder: skipped ({type(e).__name__}: {e})")
        return None


def ensure_moss_previews(repo_dir: str = REPO_DIR, log=print):
    """MOSS-Image's preview recipe (Turbo LoRA at 0.6 with the SFT look LoRA) for an install with MOSS-Image set up,
    so updating users' previews use it from the start. Never raises; without them previews render at full quality."""
    try:
        if not str(_load_prefs(_prefs_file(repo_dir)).get("moss_image_dit") or "").strip():
            return None
        hf = "https://huggingface.co/{}/resolve/main/{}"
        return [ensure_lora("moss_image_turbo_lora", "8-steps-dmd-lora.safetensors",
                            hf.format("lodestones/Kromoss", "8-steps-dmd-lora.safetensors"),
                            "diffusion_model.layers.0.proj.ffn_down.alpha", 1_000_000_000,
                            "MOSS-Image Turbo LoRA (previews)", repo_dir=repo_dir, log=log),
                ensure_lora("moss_image_sft_look_lora", "moss_image_sft_look_lora.safetensors",
                            hf.format("ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA",
                                      "moss_image_sft_look_lora.safetensors"),
                            "lora_unet_final_linear_video.alpha", 2_000_000_000,
                            "MOSS-Image SFT look LoRA (previews)", repo_dir=repo_dir, log=log)]
    except Exception as e:
        log(f"MOSS-Image preview LoRAs: skipped ({type(e).__name__}: {e})")
        return None


if __name__ == "__main__":
    ensure_all()
    ensure_taeh3()
    ensure_moss_previews()
    qwen_turbo_dit_hint()
