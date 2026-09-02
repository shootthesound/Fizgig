"""Download the model files Fizgig needs and wire them into Preferences.

Replaces the hand-download: five files across four HuggingFace repos, then five paths pasted
into Preferences by hand. Run this instead and it fetches what's missing, verifies it, and
writes the paths into prefs.json for you.

    python -m fizgig.scripts.fetch_models --family krea2      # ~45 GB, no HF account needed
    python -m fizgig.scripts.fetch_models --family klein      # ~39 GB, needs an HF token
    python -m fizgig.scripts.fetch_models --family minimax    # ~47 GB, no HF account needed
    python -m fizgig.scripts.fetch_models --family tools      # ~1.6 GB helper models
    python -m fizgig.scripts.fetch_models --all

Two kinds of asset, handled differently:

  * WEIGHTS  — single .safetensors files. Downloaded into <repo>/models/ (or --models-dir)
               and written into prefs.json against the pref key the GUI reads.
  * TOOLS    — Florence-2, the InsightFace face model, the EN->ZH translator and Gizmo's
               Whisper transcriber. These are loaded BY NAME at runtime, not by path, so
               there's nothing to put in prefs — "fetching" them means warming the cache
               they'll be loaded from, which turns a mid-workflow stall into a one-off
               up-front download (and, for Whisper, guarantees Gizmo transcribes offline).

Idempotent throughout: anything already present and valid is skipped, so it's safe to re-run
and safe to call at container start. Resumable too — hf_hub_download picks a broken 26 GB
transfer back up rather than starting over.
"""
import argparse
import json
import os
import struct
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Machine-readable progress for the GUI. Off by default so a terminal user keeps
# huggingface_hub's own progress bar, which is nicer to watch than our polled byte counts.
EMIT_PROGRESS = False


class Weight:
    """One .safetensors file: where it lives on HF, where it goes, which pref points at it."""

    def __init__(self, pref_key, repo, path_in_repo, gb, note, optional=False, gated=False):
        self.pref_key = pref_key
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.filename = os.path.basename(path_in_repo)
        self.gb = gb
        self.note = note
        self.optional = optional
        self.gated = gated


# The Captions tab captions ANY dataset with the Krea 2 Qwen3-VL text encoder — it follows an
# editable instruction and writes better training captions than Florence-2. So every family's
# download button fetches it, not just Krea 2's: a Klein- or MiniMax-only user still captions.
# Defined once so all three lists point at the same file and pref key.
_CAPTION_TE = Weight("krea2_text_encoder", "Comfy-Org/Krea-2",
                     "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.2,
                     "Qwen3-VL text encoder — powers the Captions tab (shared with Krea 2)")

# Paths verified against the HuggingFace API, not the README — the two can drift.
FAMILIES = {
    "krea2": [
        Weight("krea2_raw_dit", "Comfy-Org/Krea-2",
               "diffusion_models/krea2_raw_bf16.safetensors", 26.0,
               "RAW DiT — what training runs on"),
        _CAPTION_TE,
        Weight("krea2_vae", "Comfy-Org/Krea-2",
               "vae/qwen_image_vae.safetensors", 0.25, "Qwen-Image VAE"),
        Weight("krea2_turbo_lora", "Comfy-Org/Krea-2",
               "loras/krea2_turbo_lora_rank_64_bf16.safetensors", 0.47,
               "Turbo LoRA — in-training previews"),
        Weight("krea2_turbo_dit", "Comfy-Org/Krea-2",
               "diffusion_models/krea2_turbo_fp8_scaled.safetensors", 13.0,
               "Turbo DiT — Repair Studio, Explorer, Royale and classic previews"),
    ],
    "minimax": [
        Weight("minimax_dit", "Comfy-Org/MiniMax-H3",
               "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 21.0,
               "H3 DiT (pruned int8) — the training base, the file ComfyUI runs"),
        Weight("minimax_text_encoder", "Comfy-Org/MiniMax-H3",
               "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.7,
               "Qwen3-VL-32B text encoder (nvfp4 — the compact ComfyUI file)"),
        Weight("minimax_vae", "Comfy-Org/MiniMax-H3",
               "vae/minimax_h3_video_vae_fp16.safetensors", 4.9,
               "Video VAE — caching + preview decode"),
        # Not behind --include-optional: 605 MB against this family's 47 GB is nothing, and a
        # user who adds video clips later would otherwise have to come back for one small file.
        Weight("minimax_audio_vae", "Comfy-Org/MiniMax-H3",
               "vae/minimax_h3_audio_vae_fp32.safetensors", 0.61,
               "Audio VAE — training on the sound in video clips"),
        # Same reasoning: 780 MB buys 6-step previews for every H3 run.
        Weight("minimax_turbo_lora", "larryvrh/MiniMax-H3-Turbo-Lora",
               "minimax_h3_turbo_v4_step600.safetensors", 0.78,
               "Turbo LoRA — fast 6-step in-training previews"),
        # Ostris's training adapters (ai-toolkit's "assistant LoRA"): 155 MB each, one per base.
        # The Training tab's tickbox loads the one matching the selected base at 1.0, on for
        # training and off for previews. Both fetched — the ref2va one is tiny next to its DiT.
        Weight("minimax_training_adapter", "ostris/minimax_h3_training_adapter",
               "minimax_h3_training_adapter_v1.safetensors", 0.16,
               "Training adapter (fl2va) — de-distills the base while your LoRA learns (Ostris)"),
        Weight("minimax_ref_training_adapter", "ostris/minimax_h3_training_adapter",
               "minimax_h3_ref2va_training_adapter_v1.safetensors", 0.16,
               "Training adapter (ref2va) — the same, for runs on the reference base (Ostris)"),
        # ref2va is a DIFFERENT fine-tune, needed only for reference distillation — 21 GB most
        # users don't want on a first setup, so it rides behind --include-optional like the
        # Krea 2 Turbo DiT does.
        Weight("minimax_ref_dit", "Comfy-Org/MiniMax-H3",
               "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 21.0,
               "Reference DiT (ref2va) — reference distillation only", optional=True),
        _CAPTION_TE,
    ],
    "klein": [
        Weight("base_dit", "black-forest-labs/FLUX.2-klein-base-9b-fp8",
               "flux-2-klein-base-9b-fp8.safetensors", 9.5,
               "Base DiT (fp8) — what training runs on", gated=True),
        Weight("text_encoder", "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
               "split_files/text_encoders/qwen_3_8b.safetensors", 15.0,
               "Qwen3-8B text encoder"),
        Weight("vae", "black-forest-labs/FLUX.2-dev",
               "ae.safetensors", 0.32,
               "VAE — the one from the repo ROOT, not the vae/ subfolder", gated=True),
        Weight("distilled_dit", "black-forest-labs/FLUX.2-klein-9b-fp8",
               "flux-2-klein-9b-fp8.safetensors", 9.0,
               "Distilled DiT — 4-step previews and the workbench", gated=True),
        _CAPTION_TE,
    ],
}

# Loaded by name at runtime, so there is no pref to write — see the module docstring.
TOOLS = [
    ("MiaoshouAI/Florence-2-base-PromptGen", 1.0, "Florence-2 captioner"),
    ("Helsinki-NLP/opus-mt-en-zh", 0.3, "EN->ZH translator (bilingual captions)"),
    ("insightface:buffalo_l", 0.3, "Face model — Look Filter + likeness scoring"),
    # hf-model: config + tokenizer + safetensors only. The repo also carries .bin/.msgpack/.h5
    # duplicates of the same weights — an unfiltered snapshot would pull 1.16 GB for a 0.3 GB
    # model. Gizmo checks this cache before loading (gizmo.py local_whisper_dir) and stays
    # offline once it's here; the pattern lists MUST match.
    ("hf-model:openai/whisper-base", 0.3, "Whisper — Gizmo's Transcribe button, offline"),
    # Tokenizer/processor configs only (hf-config: skips the weights — those load from the
    # user's own safetensors). A few MB each, and the difference between "captioning and text
    # encoding work offline" and "the first use needs internet for a vocab file".
    ("hf-config:Qwen/Qwen3-VL-4B-Instruct", 0.02, "Qwen3-VL tokenizer — Krea 2 offline"),
    ("hf-config:Qwen/Qwen3-8B", 0.02, "Qwen3 tokenizer — Klein offline"),
]


def _valid_safetensors(path, min_bytes):
    """Header parses and the file is a plausible size. Reads no tensor data.

    Catches the classic failure where an HTML error page or a truncated transfer lands at the
    destination filename and everything downstream fails with something incomprehensible.
    """
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100 * 1024 * 1024:          # a real header is a few hundred KB
                return False
            json.loads(f.read(n))
        return True
    except Exception:
        return False


def _load_prefs(prefs_file):
    if os.path.exists(prefs_file):
        try:
            with open(prefs_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_prefs(prefs_file, prefs):
    """Atomic — a half-written prefs.json would take the GUI down on next launch."""
    tmp = prefs_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    os.replace(tmp, prefs_file)


def _incomplete_bytes(models_dir):
    """Bytes written so far, read off huggingface_hub's partial-download file.

    Polling the file beats parsing hf_hub_download's tqdm: tqdm redraws with carriage returns
    rather than newlines, so a caller reading the stream line-by-line sees nothing at all for
    the entire transfer — which is exactly how a 9.5 GB download looks like a frozen app.
    """
    best = 0
    cache = os.path.join(models_dir, ".cache", "huggingface", "download")
    for root, _dirs, files in os.walk(cache):
        for f in files:
            if f.endswith(".incomplete"):
                try:
                    best = max(best, os.path.getsize(os.path.join(root, f)))
                except OSError:
                    pass
    return best


def _remote_size(w, token):
    """Exact byte size from HuggingFace, so the progress bar is honest. None if unavailable."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        md = get_hf_file_metadata(
            hf_hub_url(repo_id=w.repo, filename=w.path_in_repo),
            token=token or os.environ.get("HF_TOKEN") or None)
        return md.size or None
    except Exception:
        return None


def _download_with_progress(w, models_dir, token, log):
    """hf_hub_download on a worker thread, emitting [progress] lines while it runs.

    The GUI parses those to drive a real progress bar. Structured output rather than tqdm
    scraping keeps the parsing in one place and survives huggingface_hub changing its display.
    """
    import threading
    from huggingface_hub import hf_hub_download

    total = _remote_size(w, token) or int(w.gb * 1024 ** 3)
    held = {}

    def run():
        try:
            held["path"] = hf_hub_download(
                repo_id=w.repo, filename=w.path_in_repo, local_dir=models_dir,
                token=token or os.environ.get("HF_TOKEN") or None)
        except BaseException as e:      # re-raised on the calling thread below
            held["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    log(f"  [progress] 0 {total} {w.filename}")
    # High-water mark: the .incomplete file is renamed away the moment the transfer finishes,
    # while the thread is still verifying and moving it — so a naive poll reads 0 and the bar
    # snaps back to empty in the last second of a 26 GB download. Never go backwards.
    seen = 0
    while t.is_alive():
        t.join(1.5)
        seen = max(seen, _incomplete_bytes(models_dir))
        log(f"  [progress] {seen} {total} {w.filename}")
    if "err" in held:
        raise held["err"]
    log(f"  [progress] {total} {total} {w.filename}")
    return held["path"]


def fetch_weight(w, models_dir, prefs, token=None, log=print, dry_run=False):
    """Download one weight if needed and point its pref at it. Returns True if usable."""
    dest = os.path.join(models_dir, w.filename)
    # 80% of the nominal size — generous enough for rounding, tight enough to catch a truncation.
    min_bytes = int(w.gb * 0.8 * 1024 ** 3)

    # A pref the user already set, pointing at a file that exists, is LEFT ALONE — existence is
    # the only test. Deliberately not the size check below: that exists to catch a transfer WE
    # just made being truncated, and applying it here would second-guess legitimate choices.
    # Plenty are smaller than the file this manifest names — the fp8_scaled RAW DiT instead of
    # bf16, Klein's fp8mixed text encoder instead of the full one, a community fine-tune — and
    # every one of those would have been "too small", silently re-downloaded, and the user's
    # path overwritten with ours.
    current = str(prefs.get(w.pref_key) or "").strip()
    if current and os.path.isfile(current):
        if os.path.basename(current) != w.filename:
            log(f"  [keep] {w.pref_key} -> {os.path.basename(current)} (your own choice, not replacing it)")
        else:
            log(f"  [ok]   {w.filename} — already set up")
        return True

    if os.path.isfile(dest) and _valid_safetensors(dest, min_bytes):
        log(f"  [have] {w.filename} — on disk, linking into Preferences")
        prefs[w.pref_key] = dest
        return True

    if dry_run:
        log(f"  [get]  {w.filename} (~{w.gb:g} GB) — {w.note}")
        return True

    log(f"  [get]  {w.filename} (~{w.gb:g} GB) — {w.note}")
    try:
        if EMIT_PROGRESS:
            got = _download_with_progress(w, models_dir, token, log)
        else:
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(repo_id=w.repo, filename=w.path_in_repo, local_dir=models_dir,
                                  token=token or os.environ.get("HF_TOKEN") or None)
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            log(f"  [gated] {w.repo} needs you to accept its licence first:")
            log(f"          https://huggingface.co/{w.repo}")
            log(f"          Then re-run with --token hf_... (or set HF_TOKEN).")
        else:
            log(f"  [fail] {w.filename}: {name}: {e}")
        return False

    # hf_hub_download preserves the repo's folder structure under local_dir; the GUI just wants
    # a file, so flatten it into models/ and leave no stray nested dirs behind.
    got = os.path.abspath(got)
    if got != dest:
        os.replace(got, dest)
        d = os.path.dirname(got)
        while os.path.normpath(d) != os.path.normpath(models_dir):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)

    if not _valid_safetensors(dest, min_bytes):
        log(f"  [fail] {w.filename} downloaded but failed verification — leaving it for a re-run")
        return False

    prefs[w.pref_key] = dest
    log(f"  [done] {w.filename}")
    return True


def fetch_tool(spec, log=print, dry_run=False):
    """Warm a by-name model's cache. Never fatal — these all re-download on demand anyway."""
    model_id, gb, note = spec
    if model_id.startswith("insightface:"):
        # Cheap presence check first. Unlike the HF downloads, warming this one means actually
        # constructing FaceAnalysis and letting onnxruntime load the models — several seconds of
        # console noise. Both family buttons fetch the tools, so that would otherwise happen
        # twice for anyone who trains with both families.
        name = model_id.split(":", 1)[1]
        home = os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")
        if os.path.isdir(os.path.join(home, "models", name)):
            log(f"  [ok]   {model_id} — already present")
            return True
    log(f"  [get]  {model_id} (~{gb:g} GB) — {note}")
    if dry_run:
        return True
    try:
        if model_id.startswith("insightface:"):
            # No public download API — instantiating is what actually populates ~/.insightface,
            # and it proves onnxruntime can load the model rather than just that bytes arrived.
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=model_id.split(":", 1)[1],
                               allowed_modules=["detection", "genderage"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1)
        elif model_id.startswith("hf-config:"):
            # Tokenizer vocab + processor/config json only — NEVER the weights. The weights
            # repos here are multi-GB and the user already has them as local safetensors;
            # a bare snapshot_download would pull the whole thing.
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id.split(":", 1)[1],
                              allow_patterns=["*.json", "*.txt", "*.model"])
        elif model_id.startswith("hf-model:"):
            # Configs + tokenizer + the safetensors weights — skipping the .bin/.msgpack/.h5
            # duplicates these repos also carry. Keep in lockstep with the runtime's
            # local-cache check (gizmo.py _WHISPER_PATTERNS).
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id.split(":", 1)[1],
                              allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id)
        log(f"  [done] {model_id}")
        return True
    except Exception as e:
        log(f"  [skip] {model_id}: {type(e).__name__}: {e}")
        log("         Not fatal — it will download by itself the first time you use it.")
        return False


def fetch(families, models_dir=None, repo_dir=REPO_DIR, token=None, include_optional=False,
          log=print, dry_run=False):
    """Fetch every asset for the named families. Returns True if nothing failed."""
    models_dir = os.path.abspath(models_dir or os.path.join(repo_dir, "models"))
    prefs_file = os.path.join(repo_dir, "prefs.json")
    if not dry_run:
        os.makedirs(models_dir, exist_ok=True)
    prefs = _load_prefs(prefs_file)
    prefs_before = json.dumps(prefs, sort_keys=True)

    planned = [w for fam in families if fam in FAMILIES for w in FAMILIES[fam]
               if include_optional or not w.optional]
    total = sum(w.gb for w in planned) + (
        sum(g for _, g, _ in TOOLS) if "tools" in families else 0)
    log(f"Fetching: {', '.join(families)}  (~{total:.1f} GB)")
    log(f"Into: {models_dir}")
    if any(w.gated for w in planned) and not (token or os.environ.get("HF_TOKEN")):
        log("Note: some of these need a HuggingFace token — accept the licence on the model page,")
        log("      then pass --token hf_... or set HF_TOKEN.")
    log("")

    ok = True
    for fam in families:
        if fam == "tools":
            log("Helper models (Florence-2, face model, translator, Whisper):")
            for spec in TOOLS:
                fetch_tool(spec, log=log, dry_run=dry_run)   # never fails the run
            log("")
            continue
        if fam not in FAMILIES:
            log(f"Unknown family: {fam}")
            ok = False
            continue
        log(f"{fam}:")
        for w in FAMILIES[fam]:
            if w.optional and not include_optional:
                log(f"  [skip] {w.filename} (~{w.gb:g} GB) — optional: {w.note}")
                continue
            if not fetch_weight(w, models_dir, prefs, token=token, log=log, dry_run=dry_run):
                ok = False
        log("")

    # Only rewrite prefs.json if we actually changed something — a tools-only run has no paths
    # to record, and there's no reason to touch a user's settings file to say so.
    if not dry_run and json.dumps(prefs, sort_keys=True) != prefs_before:
        _save_prefs(prefs_file, prefs)
        log(f"Preferences updated: {prefs_file}")
    log("Done." if ok else "Finished with some items missing — re-run to retry just those.")
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Download Fizgig's model files and write them into Preferences.")
    p.add_argument("--family", action="append", choices=["krea2", "klein", "minimax", "tools"],
                   help="Repeatable. Krea 2 needs no HF account; Klein is gated.")
    p.add_argument("--all", action="store_true", help="Every family, including the helper models.")
    p.add_argument("--include-optional", action="store_true",
                   help="Also fetch the optional files (Krea 2 Turbo DiT ~13 GB, "
                        "MiniMax ref2va DiT ~21 GB).")
    p.add_argument("--models-dir", default=None, help="Default: <repo>/models")
    p.add_argument("--token", default=None, help="HuggingFace token for gated repos (or HF_TOKEN).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be fetched.")
    p.add_argument("--progress", action="store_true",
                   help="Emit machine-readable '[progress] done total name' lines instead of "
                        "huggingface_hub's progress bar. Used by the GUI.")
    a = p.parse_args()

    if a.progress:
        global EMIT_PROGRESS
        EMIT_PROGRESS = True
        try:
            # Otherwise tqdm's carriage-return redraws interleave with our lines and the GUI
            # sees one enormous unparseable blob.
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass

    families = ["krea2", "klein", "minimax", "tools"] if a.all else (a.family or [])
    if not families:
        p.error("pick --family krea2|klein|tools (repeatable), or --all")

    ok = fetch(families, models_dir=a.models_dir, token=a.token,
               include_optional=a.include_optional, dry_run=a.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
