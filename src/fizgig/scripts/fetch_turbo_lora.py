"""Fetch the Krea 2 Turbo distillation LoRA and populate the Preferences path.

Idempotent by design — safe to leave in update_fizgig.bat forever:
  - pref already set and the file exists ..... no-op ("already present")
  - file already in <repo>/models/ ........... just populates the pref
  - neither ................................... downloads (~470 MB) with progress,
                                                verifies, then populates the pref

Always fetches when the file is missing — fresh installs and updates alike, whatever
family is configured (Peter, 2 Sep 2026: no "not a Krea 2 install" gate, ever). The GUI
calls ensure_turbo_lora() again at Krea 2 training start as a fallback for anyone who
skipped the scripts.
"""
import json
import os
import struct
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LORA_FILENAME = "krea2_turbo_lora_rank_64_bf16.safetensors"
LORA_URL = ("https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/"
            + LORA_FILENAME)
# A key that must exist in the genuine file — catches truncated/HTML-error downloads
# beyond what a size check can.
PROBE_KEY = "diffusion_model.blocks.0.attn.wq.lora_down.weight"
MIN_SIZE = 400 * 1024 * 1024   # genuine file is ~470 MB


def _valid_safetensors(path: str) -> bool:
    """Header parses and the probe key is present — cheap, reads no tensor data."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100 * 1024 * 1024:      # a sane header is a few hundred KB
                return False
            hdr = json.loads(f.read(n))
        return PROBE_KEY in hdr and os.path.getsize(path) >= MIN_SIZE
    except Exception:
        return False


def _download(url: str, dest: str, log) -> bool:
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
        if not _valid_safetensors(tmp):
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


def ensure_turbo_lora(repo_dir: str = REPO_DIR, log=print, require: bool = False):
    """Make sure the Turbo LoRA exists locally and prefs.json points at it.

    `require` is accepted for older callers and ignored — every call always tries.
    Returns the path on success, None otherwise. Never raises.
    """
    try:
        prefs_file = os.path.join(repo_dir, "prefs.json")
        prefs = {}
        if os.path.exists(prefs_file):
            with open(prefs_file, encoding="utf-8") as f:
                prefs = json.load(f)

        current = str(prefs.get("krea2_turbo_lora") or "").strip()
        if current and os.path.isfile(current):
            log(f"Turbo LoRA already present: {current}")
            return current

        models_dir = os.path.join(repo_dir, "models")
        dest = os.path.join(models_dir, LORA_FILENAME)
        if not (os.path.isfile(dest) and _valid_safetensors(dest)):
            os.makedirs(models_dir, exist_ok=True)
            log(f"Downloading Krea 2 Turbo LoRA (~470 MB) -> {dest}")
            log(f"  from {LORA_URL}")
            if not _download(LORA_URL, dest, log):
                return None
        else:
            log(f"Turbo LoRA found at {dest} — linking it into Preferences.")

        # Populate the pref atomically, preserving everything else in prefs.json.
        prefs["krea2_turbo_lora"] = dest
        tmp = prefs_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
        os.replace(tmp, prefs_file)
        log("Preferences updated: Krea 2 previews will use RAW + Turbo LoRA (no model swapping).")
        return dest
    except Exception as e:
        log(f"Turbo LoRA setup skipped ({type(e).__name__}: {e})")
        return None


if __name__ == "__main__":
    ensure_turbo_lora()
