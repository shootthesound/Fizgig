"""The Samples gallery's Style score (from @Murf-Code's Fizgig Advanced): how close each preview is to the training
set's look as a whole - texture, palette, line and medium more than who is in the picture - for a style LoRA, where a
face likeness says nothing. Each picture's vector is the mean and spread of DINOv2-base's patch tokens (the pinned model
the anime / furry look match uses, downloaded once, on the CPU); a preview's score is the mean similarity of its 8
closest training pictures (up to 256 of the set, spread evenly), shown 0-1. Switched on per samples folder in the
gallery; scores and the on/off go to <samples>/style.json, the training pictures' vectors to .style_bank.npz beside it."""

import hashlib
import json
import os
import threading
import time

PICS = (".png", ".jpg", ".jpeg", ".webp")
MAX_IMAGES = 256        # the dataset side: up to this many training pictures, spread evenly through the list
TOP_K = 8               # a preview's score: the mean of its 8 closest training pictures (a varied set has no centre)
LO, HI = 0.75, 0.95     # raw -> shown, @Murf-Code's calibration (29 Oct 2026, one artist's 240 pictures): unrelated
                        # pictures ~0.78 raw (0.1 shown), other artists of the same kind ~0.88 (0.64), held-out
                        # pictures of the same artist ~0.93 (0.90)
_lock = threading.Lock()
_runs = {}              # samples dir -> generation
_model = {}


def _path(d):
    return os.path.join(d, "style.json")


def read(samples_dir):
    try:
        with open(_path(samples_dir), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(samples_dir, **kw):
    d = {**read(samples_dir), **kw}
    try:
        with open(_path(samples_dir) + ".tmp", "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(_path(samples_dir) + ".tmp", _path(samples_dir))
    except OSError:
        pass


def show(raw):
    return round(min(1.0, max(0.0, (float(raw) - LO) / (HI - LO))), 4)


def _embed(path):
    """A picture's style vector: DINOv2-base's patch tokens' mean and spread, L2-normalised."""
    import numpy as np
    import torch
    from PIL import Image
    from fizgig import face_utils as fu
    if "m" not in _model:
        from transformers import AutoModel
        _model["m"] = fu._hf_pretrained(AutoModel, fu._DinoV2.model_id, fu._DinoV2.revision).to("cpu",
                                                                                                   torch.float32).eval()
    x = fu._preprocess(Image.open(path), 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
        pt = _model["m"](pixel_values=torch.from_numpy(x)).last_hidden_state[0, 1:].float()
    v = torch.cat([pt.mean(0), pt.std(0)]).numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def _images(dataset_dir):
    from fizgig.dataset.masks import is_mask_file
    try:
        out = sorted(os.path.join(dataset_dir, n) for n in os.listdir(dataset_dir)
                     if n.lower().endswith(PICS) and not is_mask_file(n))
    except OSError:
        return []
    if len(out) > MAX_IMAGES:
        out = [out[round(i * (len(out) - 1) / (MAX_IMAGES - 1))] for i in range(MAX_IMAGES)]
    return out


def _bank(samples_dir, dataset_dir, say):
    """The training pictures' vectors, worked out once and kept beside the previews (a new set works them out again)."""
    import numpy as np
    imgs = _images(dataset_dir)
    if not imgs:
        return None
    key = hashlib.sha1("\n".join(imgs).encode("utf-8")).hexdigest()
    path = os.path.join(samples_dir, ".style_bank.npz")
    try:
        with np.load(path) as z:
            if str(z["key"]) == key:
                return z["bank"]
    except (OSError, KeyError, ValueError):
        pass
    vecs = []
    for i, f in enumerate(imgs, 1):
        try:
            vecs.append(_embed(f))
        except Exception:
            continue                   # an unreadable picture is left out
        if i % 8 == 0:
            say(f"reading the training pictures' look… {i}/{len(imgs)}")
    if not vecs:
        return None
    bank = np.stack(vecs).astype("float32")
    try:
        np.savez(path, key=np.array(key), bank=bank)
    except OSError:
        pass
    return bank


def state(samples_dir, dataset_dir):
    """style.json for the page, resuming the scorer when it was switched on and nothing scores this folder yet."""
    d = read(samples_dir)
    with _lock:
        running = samples_dir in _runs
    if d.get("on") and not running and dataset_dir and os.path.isdir(dataset_dir) and os.path.isdir(samples_dir):
        _start(samples_dir, dataset_dir)
    return {"on": bool(d.get("on")), "status": d.get("status") or "", "scores": d.get("scores") or {}}


def switch(samples_dir, dataset_dir, on):
    if not on:
        with _lock:
            _runs.pop(samples_dir, None)
        _write(samples_dir, on=False, status="")
        return {"on": False}
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise ValueError("Training image folder not set - set it on the Start tab.")
    if not os.path.isdir(samples_dir):
        raise ValueError("No samples folder yet - the run makes it with its first previews.")
    _write(samples_dir, on=True)
    _start(samples_dir, dataset_dir)
    return {"on": True}


def _start(samples_dir, dataset_dir):
    with _lock:
        gen = _runs.get(samples_dir, 0) + 1
        _runs[samples_dir] = gen
    threading.Thread(target=_worker, args=(samples_dir, dataset_dir, gen), daemon=True, name="gallery-style").start()


def _live(samples_dir, gen):
    with _lock:
        return _runs.get(samples_dir) == gen


def _worker(samples_dir, dataset_dir, gen):
    """The bank first (once), then every preview not yet scored, newest first; a look for new ones every ~4 s. A
    picture written in the last 5 s waits (half-written)."""
    import numpy as np
    say = lambda t: _write(samples_dir, status=t)
    say("getting DINOv2 (about 350 MB, once)…" if not _cached() else "loading DINOv2…")
    try:
        bank = _bank(samples_dir, dataset_dir, say)
    except Exception as e:
        _write(samples_dir, status=f"error: the Style score needs DINOv2, which couldn't be loaded ({str(e)[:120]}) - "
                                   "it tries again when you open the gallery")
        with _lock:
            if _runs.get(samples_dir) == gen:
                _runs.pop(samples_dir, None)
        return
    if bank is None:
        say("error: no training pictures to compare with")
        with _lock:
            if _runs.get(samples_dir) == gen:
                _runs.pop(samples_dir, None)
        return
    scores = dict(read(samples_dir).get("scores") or {})
    last = None
    while _live(samples_dir, gen):
        try:
            files = [f for f in os.listdir(samples_dir) if f.lower().endswith(PICS)]
        except OSError:
            files = []
        scores = {k: v for k, v in scores.items() if k in set(files)}
        now = time.time()
        todo = [f for f in files if f not in scores and now - _mtime(samples_dir, f) > 5.0]
        todo.sort(key=lambda f: _mtime(samples_dir, f), reverse=True)
        for i, f in enumerate(todo, 1):
            if not _live(samples_dir, gen):
                return
            try:
                sims = np.sort(bank @ _embed(os.path.join(samples_dir, f)).astype("float32"))[::-1]
                scores[f] = show(float(sims[:TOP_K].mean()))
            except Exception:
                scores[f] = None
            if i % 4 == 0 or i == len(todo):
                _write(samples_dir, scores=scores, status=f"scoring… {len(scores)}/{len(files)}")
        status = f"live - {len(scores)} preview(s) scored: the style % on each, bottom left"
        if todo or status != last:
            _write(samples_dir, scores=scores, status=status)
            last = status
        for _ in range(16):
            if not _live(samples_dir, gen):
                return
            time.sleep(0.25)


def _mtime(d, f):
    try:
        return os.path.getmtime(os.path.join(d, f))
    except OSError:
        return 0.0


def _cached():
    """DINOv2 is already in the Hugging Face cache (no download needed)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from fizgig import face_utils as fu
        return isinstance(try_to_load_from_cache(fu._DinoV2.model_id, "config.json", revision=fu._DinoV2.revision), str)
    except Exception:
        return False
