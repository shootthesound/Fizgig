"""Download the model files Fizgig needs and wire them into Preferences.

Replaces the hand-download: five files across four HuggingFace repos, then five paths pasted
into Preferences by hand. Run this instead and it fetches what's missing, verifies it, and
writes the paths into prefs.json for you.

    python -m fizgig.scripts.fetch_models --family krea2      # ~45 GB, no HF account needed
    python -m fizgig.scripts.fetch_models --family klein      # ~39 GB, needs an HF token
    python -m fizgig.scripts.fetch_models --family minimax    # ~47 GB, no HF account needed
    python -m fizgig.scripts.fetch_models --family qwen_image21    # ~41 GB, no HF account needed
    python -m fizgig.scripts.fetch_models --family tools      # ~1.6 GB helper models
    python -m fizgig.scripts.fetch_models --all

Two kinds of asset, handled differently:

  * WEIGHTS  — single .safetensors files. Downloaded into <repo>/models/ (or --models-dir)
               and written into prefs.json against the pref key the GUI reads.
  * TOOLS    — Florence-2, the InsightFace face model, Make masks' person segmenter, the EN->ZH
               translator and Gizmo's Whisper transcriber. These are loaded BY NAME at runtime, not by path, so
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

    def __init__(self, pref_key, repo, path_in_repo, gb, note, optional=False, gated=False, local_name=None,
                 replaces=()):
        self.pref_key = pref_key
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.filename = local_name or os.path.basename(path_in_repo)   # local_name: see ModelFile.local_name
        self.gb = gb
        self.note = note
        self.optional = optional
        self.gated = gated
        self.replaces = tuple(replaces)          # earlier versions of this file: a pref on one moves to this one
        self.alt = None                          # the other build offered beside it (ModelFile.fetch_choice)
        self.switch_from = ""                    # picked over that build: a pref on it moves to this one
        self.same_as = ""                        # another family's pref for the same file (ModelFile.same_as)


# The Captions tab captions ANY dataset with the Krea 2 Qwen3-VL text encoder — it follows an
# editable instruction and writes better training captions than Florence-2. So every family's
# download button fetches it, not just Krea 2's: a Klein- or MiniMax-only user still captions.
# Defined once so all three lists point at the same file and pref key.
_CAPTION_TE = Weight("krea2_text_encoder", "Comfy-Org/Krea-2",
                     "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.2,
                     "Qwen3-VL text encoder — powers the Captions tab (shared with Krea 2)")

class Strangers:
    """The Subjects card's 32 base-model strangers (families/multi_person.py: a pinned Hugging Face dataset, into
    user_data, no pref). Fetched with every family's files, as the captioner is, so a multi-person run never has to
    download them at its start (a pod hit Hugging Face's rate limit there). Shaped like a Weight for the listing."""
    pref_key, repo, path_in_repo = "", "ShootTheSound/fizgig-multi-person-strangers", ""
    filename, gb, optional, gated, replaces = "multi_person_strangers/", 0.02, False, False, ()
    note = "Multi-person strangers - the Subjects card's side-by-side pictures (21 MB)"


_STRANGERS = Strangers()


def fetch_strangers(log=print, dry_run=False):
    if dry_run:
        log(f"  [plan] {_STRANGERS.filename} (~0.02 GB) — {_STRANGERS.note}")
        return True
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))
    from fizgig.families import multi_person
    return multi_person.download_strangers(log=log)


# Paths verified against the HuggingFace API, not the README — the two can drift.
FAMILIES = {
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
        # Training adapters, picked by the Training tab's dropdown: Circlestone (the default, one
        # 620 MB file for both bases) and Ostris's pair (155 MB each, one per base, for video-only
        # datasets). All fetched — small next to the DiT.
        Weight("minimax_circlestone_adapter", "circlestone-labs/MiniMax-H3-Image-Training-Adapter",
               "minimax_h3_image_training_adapter.safetensors", 0.62,
               "Training adapter (Circlestone) — the default, one file for both bases"),
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
        _STRANGERS,
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
        _STRANGERS,
    ],
}



def _described_families():
    """Families added through the standard layer (fizgig.families): their download lists come from their
    descriptions (model_files), never from a hand-written list here. Not part of --all (opt-in)."""
    try:
        sys.path.insert(0, os.path.join(REPO_DIR, "src"))
        from fizgig.families.registry import FAMILIES as _REG
    except Exception:
        return {}
    # Every family's button also fetches the Captions tab's Qwen3-VL captioner (Krea 2's own text encoder, so not
    # twice there), as the old families' lists do.
    out = {}
    for d in _REG.values():
        if not d.training_ready:
            continue
        ws = []
        for f in d.model_files:
            if not (f.repo and f.path and f.bulk_fetch):
                continue
            w = Weight(f.pref_key, f.repo, f.path, f.size_gb, f.label + (" — " + f.note if f.note else ""),
                       optional=f.fetch_is_optional, gated=f.gated, local_name=f.local_name or None,
                       replaces=f.replaces)
            w.same_as = f.same_as
            if f.fetch_choice and f.alt_repo and f.alt_path:
                w.alt = Weight(f.pref_key, f.alt_repo, f.alt_path, f.alt_size_gb or f.size_gb,
                               f"{f.label} — {f.fetch_choice[1]}", optional=w.optional, gated=f.gated,
                               replaces=f.replaces)
            ws.append(w)
        out[d.key] = ws + ([] if any(w.pref_key == _CAPTION_TE.pref_key for w in ws) else [_CAPTION_TE]) + [_STRANGERS]
    return out


def _described_helpers():
    """Small tokenizer/processor files the described families' text encoders load by name (description
    `helper_files`: (repo, allow_patterns) pairs), fetched with the other helpers so first use works offline."""
    try:
        from fizgig.families.registry import FAMILIES as _REG
    except Exception:
        return []
    return [(f"hf-files:{repo}|{','.join(pats)}", 0.02, f"{d.display_name} tokenizer / processor")
            for d in _REG.values() if d.training_ready for repo, pats in d.helper_files]


DESCRIBED = _described_families()
FAMILIES.update(DESCRIBED)

# Loaded by name at runtime, so there is no pref to write — see the module docstring.
TOOLS = [
    ("MiaoshouAI/Florence-2-base-PromptGen", 1.0, "Florence-2 captioner"),
    ("Helsinki-NLP/opus-mt-en-zh", 0.3, "EN->ZH translator (bilingual captions)"),
    ("insightface:buffalo_l", 0.3, "Face model — Look Filter + likeness scoring"),
    ("masks:segmenter", 0.02, "Person segmenter — Image Prep's Make masks"),
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
TOOLS += _described_helpers()


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
    from fizgig.utils import hf_public

    total = _remote_size(w, token) or int(w.gb * 1024 ** 3)
    held = {}

    def run():
        try:
            held["path"] = hf_public.download(
                repo_id=w.repo, filename=w.path_in_repo, local_dir=models_dir,
                token=token or os.environ.get("HF_TOKEN") or None, log=log)
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
    if current and w.switch_from and os.path.basename(current) == w.switch_from:
        log(f"  [pick] {w.pref_key}: {os.path.basename(current)} -> {w.filename} (your choice on the download card)")
        current = ""
    if current and os.path.basename(current) in w.replaces:
        log(f"  [new]  {w.pref_key}: {os.path.basename(current)} is replaced by {w.filename}")
        current = ""                             # an earlier version of ours: fetch the new one and point at it
    if current and os.path.isfile(current):
        if os.path.basename(current) != w.filename and w.switch_from:
            log(f"  [keep] {w.pref_key} -> {os.path.basename(current)}: your own file, so {w.filename} was not "
                "downloaded. To use the pick, clear that row in Preferences, then download again.")
        elif os.path.basename(current) != w.filename:
            log(f"  [keep] {w.pref_key} -> {os.path.basename(current)} (your own choice, not replacing it)")
        else:
            log(f"  [ok]   {w.filename} — already set up")
        return True

    # the same file set for another model (any name: the user's own copy, e.g. ComfyUI's renamed VAE): linked, not fetched
    same_as = getattr(w, "same_as", "")
    other = str(prefs.get(same_as) or "").strip() if same_as else ""
    if other and os.path.isfile(other):
        log(f"  [have] {w.filename} — already set for another model ({os.path.basename(other)}), linking into "
            "Preferences")
        prefs[w.pref_key] = other
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
            from fizgig.utils import hf_public
            got = hf_public.download(repo_id=w.repo, filename=w.path_in_repo, local_dir=models_dir,
                                     token=token or os.environ.get("HF_TOKEN") or None, log=log)
    except Exception as e:
        name = type(e).__name__
        from fizgig.utils import hf_public
        if isinstance(e, hf_public.LoginExpired):
            log(f"  [login] {w.filename}: {e}")
        elif "Gated" in name or "401" in str(e) or "403" in str(e):
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
    if model_id.startswith("masks:"):
        from fizgig.dataset import automask
        from fizgig.user_data import MODELS_DIR
        if os.path.isfile(automask.model_path(MODELS_DIR)):
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
        elif model_id.startswith("masks:"):
            # Make masks' segmenter: the same download and hash check as its first use (dataset/automask.py)
            from fizgig.dataset import automask
            from fizgig.user_data import MODELS_DIR
            automask.ensure_model(MODELS_DIR, log=lambda m: None)
        elif model_id.startswith("hf-config:"):
            # Tokenizer vocab + processor/config json only — NEVER the weights. The weights
            # repos here are multi-GB and the user already has them as local safetensors;
            # a bare snapshot_download would pull the whole thing.
            from fizgig.utils import hf_public
            hf_public.snapshot(log=log, repo_id=model_id.split(":", 1)[1],
                               allow_patterns=["*.json", "*.txt", "*.model"])
        elif model_id.startswith("hf-files:"):
            # A repo's small named files only (e.g. a processor/ folder), never its weights.
            from fizgig.utils import hf_public
            repo, _, pats = model_id[len("hf-files:"):].partition("|")
            hf_public.snapshot(log=log, repo_id=repo, allow_patterns=pats.split(","))
        elif model_id.startswith("hf-model:"):
            # Configs + tokenizer + the safetensors weights — skipping the .bin/.msgpack/.h5
            # duplicates these repos also carry. Keep in lockstep with the runtime's
            # local-cache check (gizmo.py _WHISPER_PATTERNS).
            from fizgig.utils import hf_public
            hf_public.snapshot(log=log, repo_id=model_id.split(":", 1)[1],
                               allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
        else:
            from fizgig.utils import hf_public
            hf_public.snapshot(log=log, repo_id=model_id)
        log(f"  [done] {model_id}")
        return True
    except Exception as e:
        log(f"  [skip] {model_id}: {type(e).__name__}: {e}")
        log("         Not fatal — it will download by itself the first time you use it.")
        return False


def _chosen(w, choices):
    """The build to fetch for a file with a choice: {pref_key: "main" | "alt"} picks one and lets it replace the other
    (so a pref on the other moves); without a pick, the main file as before, the user's own file kept."""
    pick = (choices or {}).get(getattr(w, "pref_key", ""))
    if getattr(w, "alt", None) is None or pick not in ("main", "alt"):     # the strangers pack has no build to pick
        return w
    use, other = (w.alt, w) if pick == "alt" else (w, w.alt)
    import copy
    use = copy.copy(use)
    use.switch_from = other.filename
    return use


def fetch(families, models_dir=None, repo_dir=REPO_DIR, token=None, include_optional=False,
          log=print, dry_run=False, choices=None):
    """Fetch every asset for the named families. Returns True if nothing failed. choices: {pref_key: "main" | "alt"}
    for files offered with an alternative build (ModelFile.fetch_choice)."""
    models_dir = os.path.abspath(models_dir or os.path.join(repo_dir, "user_data", "models"))
    if os.path.normcase(os.path.abspath(repo_dir)) == os.path.normcase(REPO_DIR):
        from fizgig import user_data
        user_data.migrate()
    prefs_file = os.path.join(repo_dir, "user_data", "prefs.json")
    if os.environ.get("FIZGIG_USER_DATA") and os.path.normcase(os.path.abspath(repo_dir)) == os.path.normcase(REPO_DIR):
        prefs_file = os.path.join(os.environ["FIZGIG_USER_DATA"], "prefs.json")    # a second Fizgig's own settings
    if not dry_run:
        os.makedirs(models_dir, exist_ok=True)
    prefs = _load_prefs(prefs_file)
    prefs_before = json.dumps(prefs, sort_keys=True)

    planned = [_chosen(w, choices) for fam in families if fam in FAMILIES for w in FAMILIES[fam]
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
            log("Helper models (Florence-2, face model, person segmenter, translator, Whisper):")
            for spec in TOOLS:
                fetch_tool(spec, log=log, dry_run=dry_run)   # never fails the run
            log("")
            continue
        if fam not in FAMILIES:
            log(f"Unknown family: {fam}")
            ok = False
            continue
        log(f"{fam}:")
        for w in (_chosen(x, choices) for x in FAMILIES[fam]):     # the download card's pick, where a file has one
            if w.optional and not include_optional:
                log(f"  [skip] {w.filename} (~{w.gb:g} GB) — optional: {w.note}")
                continue
            if isinstance(w, Strangers):
                if not fetch_strangers(log=log, dry_run=dry_run):
                    ok = False
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
    p.add_argument("--family", action="append", choices=["klein", "minimax", "tools", *DESCRIBED],
                   help="Repeatable. Krea 2 needs no HF account; Klein is gated.")
    p.add_argument("--all", action="store_true", help="Every family, including the helper models.")
    p.add_argument("--include-optional", action="store_true",
                   help="Also fetch the optional files (Krea 2 Turbo DiT ~13 GB, MiniMax ref2va DiT ~21 GB, "
                        "SDXL's separate VAE, Anima's Turbo LoRA).")
    p.add_argument("--models-dir", default=None, help="Default: <repo>/models")
    p.add_argument("--token", default=None, help="HuggingFace token for gated repos (or HF_TOKEN).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be fetched.")
    p.add_argument("--choose", action="append", default=[], metavar="PREF=main|alt",
                   help="For a file offered with another build (MOSS-Image Base or Kromoss): which one to fetch.")
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

    choices = dict(c.split("=", 1) for c in a.choose if "=" in c)
    ok = fetch(families, models_dir=a.models_dir, token=a.token,
               include_optional=a.include_optional, dry_run=a.dry_run, choices=choices)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
