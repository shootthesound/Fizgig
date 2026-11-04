"""The web UI's Captions tab: the desktop Captions tab's tools, run on the machine training.

AI captions use the desktop's own worker (fizgig.scripts.batch_caption --serve): one process that keeps its model
loaded between batches, so a single re-caption is quick, until Unload, a model change, or a run needs the GPU (it
quits before any run starts, on either side). Static captions, song captions and Find & Replace are file operations
here. Bilingual translation runs fizgig.scripts.translate_captions. One job at a time; status() is what the page
polls.

Model names, Florence pins and the Qwen3-VL task menu match the desktop app (lora_trainer_gui.py FLORENCE_*,
fizgig.krea2.caption_text.CAPTION_TASKS); the edited Qwen instructions are the same Preferences entry
(caption_qwen_instructions), so an edit in either app is used by both.
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
from fizgig.dataset.masks import is_mask_file  # noqa: E402
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JOB_DIR = user_file("caption_jobs")

def read_text(path) -> str:
    """A caption file's words, whatever wrote it: UTF-8 (a BOM dropped), UTF-16 with its BOM, else Windows-1252 (older
    Windows tools save that) - so one such file never empties a folder's caption list (#225). OSError if it isn't
    there."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


FLORENCE_DEFAULT_MODEL = "MiaoshouAI/Florence-2-base-PromptGen"
FLORENCE_MODELS = [FLORENCE_DEFAULT_MODEL, "microsoft/Florence-2-base", "microsoft/Florence-2-large"]
FLORENCE_REVISIONS = {
    "MiaoshouAI/Florence-2-base-PromptGen": "da7ac9f3deac56a928e2fd4d94d8bb985d231299",
    "microsoft/Florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
    "microsoft/Florence-2-large": "21a599d414c4d928c9032694c424fb94458e3594",
}
FLORENCE_CODE_REVISIONS = {"MiaoshouAI/Florence-2-base-PromptGen": "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"}
FLORENCE_TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
QWEN_MODEL = "Qwen3-VL (Krea 2 text encoder, or the secondary captioning model)"
# the secondary captioning model (#270): Preferences' optional row. Set, it captions everything (the Captions tab and
# auto-recaption); empty, the Krea 2 text encoder (Qwen3-VL-4B) does. Comfy-Org's Qwen3-VL-8B builds the loader reads
SECONDARY_PREF = "captioner_secondary"
SECONDARY_LINKS = (("fp8 (10.6 GB, about 10.5 GB of VRAM)", "Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
                   ("bf16 (17.5 GB, about 17 GB of VRAM)", "Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_8b_bf16.safetensors"))


def captioner_problem(path) -> str:
    """Why a file can't be the captioner, from its safetensors header alone ("" = it can): a Qwen3-VL 4B or 8B in a
    build the loader reads (bf16 or fp8 scaled). ComfyUI's int8 ConvRot / w4a8 builds (int8 / uint8 weights) don't load."""
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return "the file can't be read"
    if not any(k.startswith(("visual.", "model.visual.")) for k in head):
        return "it isn't a Qwen3-VL file (no vision tower)"
    if not any(k.endswith("embed_tokens.weight") and isinstance(v, dict) and v.get("shape", [0])[-1] in (2560, 4096)
               for k, v in head.items()):
        return "it isn't a Qwen3-VL 4B or 8B"
    if any(isinstance(v, dict) and k.endswith(".weight") and v.get("dtype") in ("I8", "U8") for k, v in head.items()):
        return "this build (int8 / w4a8) can't be read - use the bf16 or fp8 file"
    return ""


CUSTOM_TASK = "Custom…"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def caption_ext(ext) -> str:
    """The caption files' extension (Training > Caption Extension): .txt unless a plain one like .caption is set."""
    ext = str(ext or "").strip()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return ext if re.fullmatch(r"\.[A-Za-z0-9_]{1,12}", ext) else ".txt"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

_lock = threading.Lock()
_st = {"state": "idle", "job": "", "done": 0, "total": 0, "ok": 0, "fail": 0, "log": [], "model": "",
       "proc": None, "key": None, "ready": None, "job_proc": None}


def _log(line):
    with _lock:
        _st["log"].append(line)
        del _st["log"][:-400]


def status():
    with _lock:
        out = {k: _st[k] for k in ("state", "job", "done", "total", "ok", "fail", "model")}
        out["log"] = _st["log"][-200:]
        out["loaded"] = _st["proc"] is not None and _st["proc"].poll() is None
    return out


def _busy():
    with _lock:
        return _st["state"] not in ("idle", "loaded")


# ---- what the page offers ------------------------------------------------------------------------------------
def _tasks():
    from fizgig.krea2.caption_text import CAPTION_TASKS, DEFAULT_CAPTION_TASK   # text only: no torch to load
    return CAPTION_TASKS, DEFAULT_CAPTION_TASK


def _overrides(prefs):
    raw = prefs.get("caption_qwen_instructions")
    out = dict(raw) if isinstance(raw, dict) else {}
    legacy = str(prefs.get("caption_qwen_instruction", "") or "").strip()
    if legacy and "custom" not in out:
        out["custom"] = legacy
    return out


def options(prefs, captioner, secondary=""):
    """Models (Qwen3-VL first when its file is set), each model's tasks, and each Qwen task's instruction (the edited
    one if there is one), its shipped text and its suggested length. secondary: the secondary captioning model's file
    when it is the one captioning."""
    tasks, default = _tasks()
    ovr = _overrides(prefs)
    qwen = []
    for key, (label, text, tokens) in tasks.items():
        qwen.append({"key": key, "label": label, "tokens": tokens, "builtin": text,
                     "instruction": str(ovr.get(key) or "").strip() or text, "edited": bool(ovr.get(key))})
    custom_builtin = tasks[default][1]
    qwen.append({"key": "custom", "label": CUSTOM_TASK, "tokens": tasks[default][2], "builtin": custom_builtin,
                 "instruction": str(ovr.get("custom") or "").strip() or custom_builtin, "edited": bool(ovr.get("custom"))})
    models = FLORENCE_MODELS + ([QWEN_MODEL] if captioner else [])          # the desktop's order
    return {"models": models, "default_model": QWEN_MODEL if captioner else models[0], "qwen_model": QWEN_MODEL,
            "qwen_tasks": qwen,
            "default_qwen_task": tasks[default][0], "florence_tasks": FLORENCE_TASKS,
            "qwen_available": bool(captioner), "secondary": os.path.basename(secondary) if secondary else ""}


def instruction_override(prefs, key, text):
    """The new caption_qwen_instructions value with `key` set to `text` (dropped when it is the shipped text)."""
    tasks, default = _tasks()
    builtin = tasks[key][1] if key in tasks else tasks[default][1]
    ovr = _overrides(prefs)
    text = str(text or "").strip()
    if not text or text == builtin.strip():
        ovr.pop(key, None)
    else:
        ovr[key] = text
    return ovr


# ---- the files a job works on --------------------------------------------------------------------------------
def media_files(folder, clips, photos=True, audio=False):
    """The files a family captions: pictures (unless it trains on no pictures - a song folder's waveform previews
    are not training data), .mp4 clips when it trains on clips, and recordings when asked (Static captions)."""
    exts = (IMAGE_EXTS if photos else set()) | ({".mp4"} if clips else set()) | (AUDIO_EXTS if audio else set())
    try:
        return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                      if os.path.splitext(n)[1].lower() in exts and os.path.isfile(os.path.join(folder, n))
                      and not is_mask_file(n))
    except OSError:
        return []


def with_crops(folder):
    """[the folder] and, when it trains detail crops, its largest crop folder (what the Captions tab shows)."""
    from fizgig.dataset import detail_crops
    big = detail_crops.largest(folder)
    return [folder] + ([big] if big else [])


def static_captions(folder, trigger, overwrite, clips, photos=True, voice=False, ext=".txt"):
    """The trigger word as every file's caption (only files without one unless overwrite); a family that trains on
    recordings gets them too, and so do the folder's detail crops."""
    if not trigger.strip():
        raise ValueError("Type a trigger word first.")
    made = skipped = 0
    for p in [p for f in with_crops(folder) for p in media_files(f, clips, photos, audio=voice)]:
        cap = os.path.splitext(p)[0] + ext
        if os.path.exists(cap) and not overwrite:
            skipped += 1
            continue
        with open(cap, "w", encoding="utf-8") as f:
            f.write(trigger.strip())
        made += 1
    _mirror_crops([folder], ext)
    _log(f"Static captions: {made} written, {skipped} kept (they had one)")
    return {"written": made, "skipped": skipped}


def song_captions(folder, trigger, words, overwrite, ext=".txt"):
    """"<trigger>, <common words>" beside every song (as the desktop's Caption every song)."""
    caption = ", ".join(p for p in (trigger.strip().rstrip(","), " ".join(words.split()).strip(", ")) if p)
    if not caption:
        raise ValueError("Type a trigger word or the words every song shares first.")
    try:
        songs = sorted(n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
    except OSError:
        songs = []
    if not songs:
        raise ValueError("No songs (wav, mp3, flac, m4a) in the training folder.")
    made = skipped = 0
    for n in songs:
        cap = os.path.join(folder, os.path.splitext(n)[0] + ext)
        if os.path.exists(cap) and not overwrite:
            skipped += 1
            continue
        with open(cap, "w", encoding="utf-8") as f:
            f.write(caption)
        made += 1
    _log(f"Song captions: {made} written ({caption}), {skipped} kept (they had one)")
    return {"written": made, "skipped": skipped, "caption": caption}


def _caption_files(folder, ext):
    """The folder's caption files, then its largest detail crops' (the ones the Captions tab shows)."""
    return [fp for f in with_crops(folder) for fp in sorted(glob.glob(os.path.join(glob.escape(f), "*" + ext)))]


def _caption_name(folder, fp):
    """A caption's name in a change list: a detail crop's says so."""
    n = os.path.basename(fp)
    return n if os.path.dirname(fp) == folder else f"{n} (detail crop)"


def find_replace(folder, find, replace, apply, ext=".txt"):
    """Case-insensitive find & replace across the folder's .txt captions: the changes, and (apply) written."""
    if not find:
        raise ValueError("Type the text to find first.")
    pat = re.compile(re.escape(find), re.IGNORECASE)
    changes = []
    for fp in _caption_files(folder, ext):
        if fp.lower().endswith(".lyrics.txt"):          # a song's lyrics are the words sung, not a caption
            continue
        try:
            old = read_text(fp)
        except OSError:
            continue
        if not pat.search(old):
            continue
        new = pat.sub(lambda _m: replace, old)
        changes.append({"name": _caption_name(folder, fp), "old": old[:300], "new": new[:300]})
        if apply:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new)
    if apply:
        _mirror_crops([folder], ext)
        _log(f"Find & Replace: '{find}' -> '{replace}' in {len(changes)} captions")
    return {"changes": changes, "applied": bool(apply)}


def add_text(folder, start, end, trigger, apply, ext=".txt"):
    """Text added to the start and/or end of every .txt caption (not songs' lyrics): the changes, and (apply)
    written. The start goes after the trigger word when a caption opens with it, so captions stay trigger-first; a
    caption that already has the text is left as it is."""
    start, end, trigger = (str(x or "").strip().strip(",").strip() for x in (start, end, trigger))
    if not (start or end):
        raise ValueError("Type what to add to the start or the end first.")
    has = lambda cap, t: re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", cap, re.IGNORECASE)
    changes = []
    for fp in _caption_files(folder, ext):
        if fp.lower().endswith(".lyrics.txt"):
            continue
        try:
            old = read_text(fp)
        except OSError:
            continue
        new = old.strip()
        if start and not has(new, start):
            m = re.match(r"\s*" + re.escape(trigger) + r"(?!\w)\s*,?\s*", new, re.IGNORECASE) if trigger else None
            new = (new[:m.end()].rstrip(" ,") + ", " + start + (", " + new[m.end():] if new[m.end():] else "")
                   if m else start + (", " + new if new else ""))
        if end and not has(new, end):
            new = (new.rstrip(" .,") + ", " if new else "") + end
        if new == old.strip():
            continue
        changes.append({"name": _caption_name(folder, fp), "old": old[:300], "new": new[:300]})
        if apply:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new)
    if apply:
        _mirror_crops([folder], ext)
        _log(f"Added to {len(changes)} captions" + (f": '{start}' at the start" if start else "")
             + (f"{' and' if start else ':'} '{end}' at the end" if end else ""))
    return {"changes": changes, "applied": bool(apply)}


# ---- the AI worker -------------------------------------------------------------------------------------------
def _worker_config(model, captioner, clips):
    cfg = {"backend": "qwen" if model == QWEN_MODEL else "florence", "include_video": bool(clips)}
    if cfg["backend"] == "qwen":
        cfg["text_encoder"] = captioner
    else:
        cfg["florence_model"] = model
        if FLORENCE_REVISIONS.get(model):
            cfg["florence_revision"] = FLORENCE_REVISIONS[model]
        if FLORENCE_CODE_REVISIONS.get(model):
            cfg["florence_code_revision"] = FLORENCE_CODE_REVISIONS[model]
    return cfg


def _crop_images(images, overwrite, ext):
    """(the files to caption, the training folders whose detail crops then mirror their captions): a detail crop is
    captioned at the largest size it has, and that caption is copied over its other sizes (detail_crops.share_captions);
    without overwrite, one with a caption at its largest size is left out."""
    from fizgig.dataset import detail_crops
    out, roots = [], []
    for p in images:
        d = os.path.dirname(p)
        if not detail_crops.is_crop_dir(d):
            out.append(p)
            continue
        root = detail_crops.root_of(d)
        if root not in roots:
            roots.append(root)
        name = os.path.basename(p)
        big = next((os.path.join(o, name) for _l, o in detail_crops.sizes(root)[::-1]
                     if os.path.isfile(os.path.join(o, name))), p)
        if (overwrite or not os.path.isfile(os.path.splitext(big)[0] + ext)) and big not in out:
            out.append(big)
    return out, roots


def _mirror_crops(roots, ext):
    from fizgig.dataset import detail_crops
    n = sum(detail_crops.share_captions(r, ext) for r in roots)
    if n:
        _log(f"Copied {n} caption(s) to the same detail crops at their other sizes")
    return n


def _reader(proc):
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        with _lock:
            if line == "READY":
                _st["ready"].set()
                if _st["state"] == "loading":
                    _st["state"] = "running" if _st["job"] else "loaded"     # a queued job goes straight on
                continue
            m = re.match(r"PROGRESS: (\d+) (\d+)", line)
            if m:
                _st["done"], _st["total"] = int(m.group(1)), int(m.group(2))
                continue
            if line.startswith("OK: "):
                _st["ok"] += 1
            elif line.startswith("FAIL: "):
                _st["fail"] += 1
            roots = ()
            if line in ("DONE", "STOPPED"):
                _st["state"] = "loaded"
                _st["job"] = ""
                roots, _st["crop_roots"] = _st.get("crop_roots") or (), []
                ext = _st.get("crop_ext") or ".txt"
        _log(line)
        if roots:                                 # detail crops: the new captions over their other sizes
            _mirror_crops(roots, ext)
    with _lock:
        if _st["proc"] is proc:
            _st.update(proc=None, key=None, state="idle", job="", model="")
            if _st["ready"] is not None:
                _st["ready"].set()                # a worker that died loading wakes its waiter


def _env(prefs):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.path.join(BASE_DIR, "src"))
    from fizgig.web.server import gpu_env
    return gpu_env(env, prefs)                       # the same card as training


def _popen(cmd, prefs, stdin=False):
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
           | subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == "nt" else {"start_new_session": True})
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE if stdin else None, text=True, bufsize=1, encoding="utf-8",
                            errors="replace", env=_env(prefs), cwd=BASE_DIR, **kw)


def ai_captions(prefs, captioner, folder, model, task_label, instruction, max_tokens, trigger, overwrite, clips,
                only=None, photos=True, ext=".txt"):
    """Caption the folder's pictures (and clips) with Florence-2 or Qwen3-VL, or just `only` (a re-caption). The
    worker loads once and stays loaded for the next batch."""
    if _busy():
        raise ValueError("A captioning job is already running. Stop it first.")
    if model == QWEN_MODEL and not captioner:
        raise ValueError("Qwen3-VL needs the Krea 2 text encoder set in Preferences.")
    images = [p for p in (only or [p for f in with_crops(folder) for p in media_files(f, clips, photos)])
              if os.path.isfile(p)]
    if not images and not only:
        raise ValueError("Nothing here for the AI to caption: this model trains on recordings. Use Static Caption All "
                         "or Caption every song." if not photos and not clips else
                         "No pictures or clips in the training folder.")
    if not only and not overwrite:
        images = [p for p in images if not os.path.exists(os.path.splitext(p)[0] + ext)]
    images, roots = _crop_images(images, overwrite or bool(only), ext)    # detail crops: at their largest size
    if not images:
        raise ValueError("Every file already has a caption. Tick Overwrite to caption them again."
                         if not only else "That file isn't there any more.")
    os.makedirs(JOB_DIR, exist_ok=True)
    key = (model, captioner if model == QWEN_MODEL else "", bool(clips))
    with _lock:
        alive = _st["proc"] is not None and _st["proc"].poll() is None
        same = alive and _st["key"] == key
    if alive and not same:
        unload()
    with _lock:
        alive = _st["proc"] is not None and _st["proc"].poll() is None
    if not alive:
        cfg_path = os.path.join(JOB_DIR, "worker_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_worker_config(model, captioner, clips), f)
        proc = _popen([sys.executable, "-m", "fizgig.scripts.batch_caption", "--serve", "--config", cfg_path], prefs,
                      stdin=True)
        with _lock:
            _st.update(proc=proc, key=key, state="loading", model=model, ready=threading.Event())
        _log(f"Loading {model}…")
        threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    list_file, stop_file = os.path.join(JOB_DIR, "images.txt"), os.path.join(JOB_DIR, "stop")
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(images) + "\n")
    try:
        os.remove(stop_file)
    except OSError:
        pass
    job = {"list_file": list_file, "max_new_tokens": int(max_tokens or 120), "trigger": trigger.strip(),
           "stop_file": stop_file, "caption_ext": ext}
    if model == QWEN_MODEL:
        instr = os.path.join(JOB_DIR, "instruction.txt")
        with open(instr, "w", encoding="utf-8") as f:
            f.write(instruction.strip())
        job["instruction_file"] = instr
    else:
        job["florence_task"] = task_label if task_label in FLORENCE_TASKS else "<DETAILED_CAPTION>"
    job_path = os.path.join(JOB_DIR, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f)
    with _lock:
        _st.update(job="re-caption" if only else "captions", done=0, total=len(images), ok=0, fail=0,
                   crop_roots=roots, crop_ext=ext)
        if _st["state"] == "loaded":
            _st["state"] = "running"                  # busy from now: a second job waits for this one
        ready, proc = _st["ready"], _st["proc"]

    def send():
        while not ready.wait(5):                  # the model loads first (minutes the first time: a download)
            if proc is None or proc.poll() is not None:
                return
        with _lock:
            if proc is None or proc.poll() is not None or _st["proc"] is not proc:
                return
            _st["state"] = "running"
        try:
            proc.stdin.write(f"RUN {job_path}\n")
            proc.stdin.flush()
        except OSError:
            pass
    threading.Thread(target=send, daemon=True).start()
    _log(f"Captioning {len(images)} file{'s' if len(images) != 1 else ''} with {model}")
    return {"files": len(images)}


def translate(prefs, folder, skip_existing, ext=".txt"):
    """Add a Chinese translation to every caption (fizgig.scripts.translate_captions, its own process)."""
    if _busy():
        raise ValueError("A captioning job is already running. Stop it first.")
    os.makedirs(JOB_DIR, exist_ok=True)
    stop_file = os.path.join(JOB_DIR, "stop")
    try:
        os.remove(stop_file)
    except OSError:
        pass
    cmd = [sys.executable, "-m", "fizgig.scripts.translate_captions", folder, "--stop_file", stop_file, "--ext", ext]
    if skip_existing:
        cmd.append("--skip_existing")
    proc = _popen(cmd, prefs)
    with _lock:
        _st.update(state="translating", job="translation", done=0, total=0, ok=0, fail=0, job_proc=proc)

    def read():
        for line in proc.stdout:
            line = line.rstrip()
            m = re.match(r"PROGRESS: (\d+) (\d+)", line)
            with _lock:
                if m:
                    _st["done"], _st["total"] = int(m.group(1)), int(m.group(2))
                    continue
                if line.startswith("OK: "):
                    _st["ok"] += 1
                elif line.startswith("FAIL"):
                    _st["fail"] += 1
            if line:
                _log(line)
        proc.wait()
        with _lock:
            _st.update(job_proc=None, job="",
                       state="loaded" if _st["proc"] is not None and _st["proc"].poll() is None else "idle")
    threading.Thread(target=read, daemon=True).start()
    return {"ok": True}


def stop():
    """Stop the running job after the file it is on (the model stays loaded)."""
    os.makedirs(JOB_DIR, exist_ok=True)
    open(os.path.join(JOB_DIR, "stop"), "w").close()
    _log("Stopping after the current file…")
    return True


def unload():
    """Quit the worker: its GPU memory is freed. Called before any training run starts."""
    with _lock:
        proc = _st["proc"]
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    with _lock:
        if _st["proc"] is proc:
            _st.update(proc=None, key=None, state="idle", job="", model="")
    _log("Caption model unloaded.")
    return True


def model_loaded():
    with _lock:
        return _st["proc"] is not None and _st["proc"].poll() is None
