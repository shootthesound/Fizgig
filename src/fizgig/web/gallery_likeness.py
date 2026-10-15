"""The Samples gallery's likeness scoring (the desktop gallery's): every sample picture's largest face against three
photos of the subject from the dataset (on the CPU - training keeps the GPU), the mean of the three similarities. The
face embedder is Image Prep's look match; with Auto it is decided from the three photos (ArcFace for a person, DINOv2
for a furry / anime character), and likeness.json records which one made the scores, so a change rescores. Scores go to
<samples>/likeness.json in the desktop's format, so either app shows the other's scores and picks up its baselines;
the gallery.owner file keeps two sessions from scoring one folder against each other."""

import json
import os
import threading
import time

from fizgig.web import prep_jobs

PICS = (".png", ".jpg", ".jpeg", ".webp")
_lock = threading.Lock()
_runs = {}                     # samples dir -> {"gen", "dataset"}
_token = f"web-{os.getpid()}-{int(time.time() * 1000)}"


def _path(d):
    return os.path.join(d, "likeness.json")


def read(samples_dir):
    try:
        with open(_path(samples_dir), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(samples_dir, names, status, scores, meta=None):
    try:
        tmp = _path(samples_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"baselines": names, "status": status, "scores": scores, **(meta or {})}, f)
        os.replace(tmp, _path(samples_dir))       # atomic: the galleries poll it
    except OSError:
        pass


def _claim(samples_dir):
    try:
        p = os.path.join(samples_dir, "gallery.owner")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"token": _token}, f)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def _owns(samples_dir):
    """The latest session to start scoring a folder owns it; an older one's scorer stands down."""
    try:
        with open(os.path.join(samples_dir, "gallery.owner"), encoding="utf-8") as f:
            return json.load(f).get("token") == _token
    except (OSError, ValueError):
        _claim(samples_dir)
        return True


def state(samples_dir, dataset_dir):
    """likeness.json for the page, resuming the scorer when baselines were chosen (here or in the desktop) and nothing
    scores this folder in this server yet."""
    d = read(samples_dir)
    names = d.get("baselines") or []
    with _lock:
        running = samples_dir in _runs
    if len(names) == 3 and not running and dataset_dir and os.path.isdir(samples_dir)             and not str(d.get("status", "")).startswith("error"):     # a failed pick waits for a new one
        try:
            start(samples_dir, dataset_dir, names)    # opening the gallery claims it (the desktop's rule)
        except ValueError as e:
            d["status"] = f"error: {e}"
    thr = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else None
    return {"baselines": names, "status": d.get("status", ""), "scores": d.get("scores") or {},
            "available": prep_jobs.faces_available(), "embedder": d.get("embedder_label") or "",
            "thresholds": thr or {"match": 0.45, "borderline": 0.30}}


def start(samples_dir, dataset_dir, names):
    """Score this samples folder against three dataset photos (names in the dataset folder); none = clear."""
    names = [os.path.basename(str(n)) for n in names or []]
    with _lock:
        cur = _runs.get(samples_dir)
        gen = (cur["gen"] if cur else 0) + 1
        if not names:
            _runs.pop(samples_dir, None)
    if not names:
        _write(samples_dir, [], "cleared", {})
        return "cleared"
    if not prep_jobs.faces_available():
        raise ValueError("Face tools not installed - run install_fizgig.py.")
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise ValueError("Training image folder not set - set it on the Start tab.")
    if len(names) != 3:
        raise ValueError(f"Pick exactly 3 photos (got {len(names)}).")
    paths = [os.path.join(dataset_dir, n) for n in names]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise ValueError("Not in the training image folder: " + ", ".join(missing))
    if not os.path.isdir(samples_dir):
        raise ValueError("No samples folder yet - the run makes it with its first previews.")
    with _lock:
        _runs[samples_dir] = {"gen": gen, "dataset": dataset_dir}
    _claim(samples_dir)
    threading.Thread(target=_worker, args=(samples_dir, gen, paths), daemon=True).start()
    return "scoring started"


def _live(samples_dir, gen):
    with _lock:
        r = _runs.get(samples_dir)
        return r is not None and r["gen"] == gen


def _worker(samples_dir, gen, baselines):
    """Newest samples first (the current run's epochs score at once), then a look for new ones every ~4 s; a picture
    written in the last 5 s waits (a half-written file reads as no face), and a no-face rescores if its file changes.
    Scores for the same three photos survive a restart."""
    import numpy as np
    from fizgig.face_utils import same_embedder
    names = [os.path.basename(b) for b in baselines]
    _write(samples_dir, names, "loading the face model…", read(samples_dir).get("scores") or {})
    emb, msg = prep_jobs.session_embedder(baselines, log=lambda m: print(m.replace("[faces]", "[likeness]"), flush=True))
    meta = {"embedder": emb.cache_id, "embedder_label": emb.label, "thresholds": dict(emb.thresholds)}
    old = read(samples_dir)
    # the same three photos AND the same embedder keep their scores (a file without one was ArcFace)
    scores = old["scores"] if sorted(old.get("baselines") or []) == sorted(names) and isinstance(old.get("scores"), dict) \
        and same_embedder(old.get("embedder"), emb.cache_id) else {}
    _write(samples_dir, names, "loading the face model…", scores, meta)
    base = [prep_jobs._embed(b, emb) for b in baselines]
    missing = [n for n, e in zip(names, base) if e is None]
    if missing:
        _write(samples_dir, names, "error: no face found in " + ", ".join(missing), scores, meta)
        with _lock:
            if _runs.get(samples_dir, {}).get("gen") == gen:
                _runs.pop(samples_dir, None)
        return
    last, seen = None, {}

    def mtime(f):
        try:
            return os.path.getmtime(os.path.join(samples_dir, f))
        except OSError:
            return 0.0
    while _live(samples_dir, gen):
        if not _owns(samples_dir):
            print("[likeness] another Fizgig session took over this samples folder - this scorer is standing down.",
                  flush=True)
            with _lock:
                if _runs.get(samples_dir, {}).get("gen") == gen:
                    _runs.pop(samples_dir, None)
            return
        try:
            files = [f for f in os.listdir(samples_dir) if f.lower().endswith(PICS)]
        except OSError:
            files = []
        have = set(files)
        scores = {k: v for k, v in scores.items() if k in have}          # deleted samples drop out
        now = time.time()
        todo = [f for f in files if now - mtime(f) > 5.0
                and (f not in scores or (scores[f] is None and seen.get(f) != mtime(f)))]
        todo.sort(key=mtime, reverse=True)
        for i, f in enumerate(todo, 1):
            if not _live(samples_dir, gen):
                return
            seen[f] = mtime(f)
            e = prep_jobs._embed(os.path.join(samples_dir, f), emb)
            scores[f] = None if e is None else round(float(np.mean([float(np.dot(b, e)) for b in base])), 4)
            if i % 2 == 0 or i == len(todo):
                _write(samples_dir, names, f"scoring… {len(scores)}/{len(files)}", scores, meta)
        status = f"live - {len(scores)} sample(s) scored ({emb.label})"
        if todo or status != last:
            _write(samples_dir, names, status, scores, meta)
            last = status
        for _ in range(16):
            if not _live(samples_dir, gen):
                return
            time.sleep(0.25)
