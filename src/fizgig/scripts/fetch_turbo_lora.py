"""Fetch the small community LoRAs Fizgig leans on and populate their Preferences paths.

Runs from the update scripts (never the installer — a fresh install sets its models up from
the Preferences download button). Idempotent by design — safe to leave in update_fizgig.bat
forever, and it never fails the update: offline or a bad download is a logged line.

For each entry:
  - pref already set and the file exists ..... no-op ("already present")
  - file already in <repo>/models/ ........... just populates the pref
  - neither ................................... downloads with progress, verifies, populates

Entries: the Krea 2 Turbo LoRA (previews render on RAW + Turbo LoRA, no model swapping) and
Ostris's two MiniMax H3 training adapters (ai-toolkit's "assistant LoRA" — the Training
tab's tickbox loads the one matching the selected base). Always fetched when missing,
whatever family is configured (Peter, 2 Sep 2026: no "not a Krea 2 install" gate, ever).
The GUI calls ensure_turbo_lora() again at Krea 2 training start as a fallback.
"""
import json
import os
import struct
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# (pref key, filename, url, a key that must exist in the genuine file, minimum size, label)
# The probe key catches truncated/HTML-error downloads beyond what a size check can.
LORAS = [
    ("krea2_turbo_lora", "krea2_turbo_lora_rank_64_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/"
     "krea2_turbo_lora_rank_64_bf16.safetensors",
     "diffusion_model.blocks.0.attn.wq.lora_down.weight", 400 * 1024 * 1024,
     "Krea 2 Turbo LoRA (~470 MB)"),
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


def ensure_lora(pref_key, filename, url, probe_key, min_size, label,
                repo_dir: str = REPO_DIR, log=print):
    """Make sure one LoRA exists locally and prefs.json points at it. Returns the path on
    success, None otherwise. Never raises."""
    try:
        prefs_file = os.path.join(repo_dir, "prefs.json")
        current = str(_load_prefs(prefs_file).get(pref_key) or "").strip()
        if current and os.path.isfile(current):
            log(f"{label}: already present ({current})")
            return current
        models_dir = os.path.join(repo_dir, "models")
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


if __name__ == "__main__":
    ensure_all()
