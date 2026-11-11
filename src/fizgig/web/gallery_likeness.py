"""The Samples gallery's likeness scoring: each of a run's preview pictures, its largest face against three photos of
the subject from the run's training folder (on the CPU - training keeps the GPU), the mean of the three similarities.
The face embedder is Image Prep's look match; with Auto it is decided from the three photos (ArcFace for a person,
DINOv2 for a furry / anime character), and the scores file records which one made the scores, so a change rescores.

Per run and per training folder (11 Nov 2026): a run's scores go to <samples>/likeness_<run>.json, and the three photos
are remembered with the training folder (<dataset>/fizgig_likeness_baselines.json, beside fizgig_look_scores.json), so
every run of the same person picks them up and switching between runs never clears or mixes them. A
gallery_<run>.owner file keeps two sessions from scoring one run against each other."""

import json
import os
import threading
import time

from fizgig.web import prep_jobs

PICS = (".png", ".jpg", ".jpeg", ".webp")
_lock = threading.Lock()
_runs = {}                     # (samples dir, run) -> {"gen", "dataset"}
PICK_FILE = "fizgig_likeness_baselines.json"
_token = f"web-{os.getpid()}-{int(time.time() * 1000)}"


def _path(d, run):
    return os.path.join(d, f"likeness_{run}.json")


def _run_of(name):
    from fizgig.web.server import sample_info          # here, not at the top: server imports this module
    return (sample_info(name) or {}).get("run")


def dataset_pick(dataset_dir):
    """The three photos remembered for a training folder ([] = none yet, or one no longer there)."""
    try:
        with open(os.path.join(dataset_dir, PICK_FILE), encoding="utf-8") as f:
            names = [os.path.basename(str(n)) for n in (json.load(f).get("baselines") or [])]
    except (OSError, ValueError, AttributeError, TypeError):
        return []
    return names if len(names) == 3 and all(os.path.isfile(os.path.join(dataset_dir, n)) for n in names) else []


def _remember(dataset_dir, names):
    try:
        p = os.path.join(dataset_dir, PICK_FILE)
        if not names:
            if os.path.exists(p):
                os.remove(p)
            return
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"baselines": names}, f)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def read(samples_dir, run):
    try:
        with open(_path(samples_dir, run), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(samples_dir, run, names, status, scores, meta=None):
    try:
        tmp = _path(samples_dir, run) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"baselines": names, "status": status, "scores": scores, **(meta or {})}, f)
        os.replace(tmp, _path(samples_dir, run))       # atomic: the galleries poll it
    except OSError:
        pass


def _owner(samples_dir, run):
    return os.path.join(samples_dir, f"gallery_{run}.owner")


def _claim(samples_dir, run):
    try:
        p = _owner(samples_dir, run)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"token": _token}, f)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def _owns(samples_dir, run):
    """The latest session to start scoring a run owns it; an older one's scorer stands down."""
    try:
        with open(_owner(samples_dir, run), encoding="utf-8") as f:
            return json.load(f).get("token") == _token
    except (OSError, ValueError):
        _claim(samples_dir, run)
        return True


def state(samples_dir, run, dataset_dir):
    """A run's likeness scores for the page. Its photos: the ones it was scored with, while they're still in its
    training folder, else the ones remembered for that folder; the scorer resumes when there are three and nothing
    scores this run in this server yet."""
    d = read(samples_dir, run) if run else {}
    names = d.get("baselines") or []
    with _lock:
        running = (samples_dir, run) in _runs
    if names and dataset_dir and os.path.isdir(dataset_dir) \
            and not all(os.path.exists(os.path.join(dataset_dir, os.path.basename(str(n)))) for n in names):
        names = []                                    # photos no longer in the folder: not chosen
        d["status"] = ""
    if not names and run and dataset_dir:
        names = dataset_pick(dataset_dir)             # the training folder's photos, picked for another run of it
    if len(names) == 3 and run and not running and dataset_dir and os.path.isdir(samples_dir) \
            and not str(d.get("status", "")).startswith("error"):     # a failed pick waits for a new one
        try:
            start(samples_dir, run, dataset_dir, names, remember=False)   # opening the gallery claims it
        except ValueError as e:
            d["status"] = f"error: {e}"
    thr = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else None
    return {"baselines": names, "status": d.get("status", ""), "scores": d.get("scores") or {}, "run": run,
            "dataset": dataset_dir or "", "available": prep_jobs.faces_available(),
            "embedder": d.get("embedder_label") or "",
            "thresholds": thr or {"match": 0.45, "borderline": 0.30}}


def start(samples_dir, run, dataset_dir, names, remember=True):
    """Score a run's previews against three photos from its training folder (names in that folder), and remember
    them for the folder (remember=False: resuming with the folder's own or the run's own photos); none = clear."""
    names = [os.path.basename(str(n)) for n in names or []]
    if not run:
        raise ValueError("No run picked in the gallery yet.")
    key = (samples_dir, run)
    with _lock:
        cur = _runs.get(key)
        gen = (cur["gen"] if cur else 0) + 1
        if not names:
            _runs.pop(key, None)
    if not names:
        _write(samples_dir, run, [], "cleared", {})
        if remember and dataset_dir and os.path.isdir(dataset_dir):
            _remember(dataset_dir, [])
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
    if remember:
        _remember(dataset_dir, names)
    with _lock:
        _runs[key] = {"gen": gen, "dataset": dataset_dir}
    _claim(samples_dir, run)
    threading.Thread(target=_worker, args=(samples_dir, run, gen, paths), daemon=True).start()
    return "scoring started"


def _live(samples_dir, run, gen):
    with _lock:
        r = _runs.get((samples_dir, run))
        return r is not None and r["gen"] == gen


def _worker(samples_dir, run, gen, baselines):
    """The run's own previews, newest first (the current epochs score at once), then a look for new ones every ~4 s;
    a picture written in the last 5 s waits (a half-written file reads as no face), and a no-face rescores if its file
    changes. Scores for the same three photos survive a restart."""
    import numpy as np
    from fizgig.face_utils import same_embedder
    names = [os.path.basename(b) for b in baselines]
    key = (samples_dir, run)
    _write(samples_dir, run, names, "loading the face model…", read(samples_dir, run).get("scores") or {})
    emb, msg = prep_jobs.session_embedder(baselines, log=lambda m: print(m.replace("[faces]", "[likeness]"), flush=True))
    meta = {"embedder": emb.cache_id, "embedder_label": emb.label, "thresholds": dict(emb.thresholds)}
    old = read(samples_dir, run)
    # the same three photos AND the same embedder keep their scores (a file without one was ArcFace)
    scores = old["scores"] if sorted(old.get("baselines") or []) == sorted(names) and isinstance(old.get("scores"), dict) \
        and same_embedder(old.get("embedder"), emb.cache_id) else {}
    _write(samples_dir, run, names, "loading the face model…", scores, meta)
    base = [prep_jobs._embed(b, emb) for b in baselines]
    missing = [n for n, e in zip(names, base) if e is None]
    if missing:
        _write(samples_dir, run, names, "error: no face found in " + ", ".join(missing), scores, meta)
        with _lock:
            if _runs.get(key, {}).get("gen") == gen:
                _runs.pop(key, None)
        return
    last, seen = None, {}

    def mtime(f):
        try:
            return os.path.getmtime(os.path.join(samples_dir, f))
        except OSError:
            return 0.0
    while _live(samples_dir, run, gen):
        if not _owns(samples_dir, run):
            print(f"[likeness] another Fizgig session took over {run}'s scores - this scorer is standing down.",
                  flush=True)
            with _lock:
                if _runs.get(key, {}).get("gen") == gen:
                    _runs.pop(key, None)
            return
        try:
            files = [f for f in os.listdir(samples_dir) if f.lower().endswith(PICS) and _run_of(f) == run]
        except OSError:
            files = []
        have = set(files)
        scores = {k: v for k, v in scores.items() if k in have}          # deleted samples drop out
        now = time.time()
        todo = [f for f in files if now - mtime(f) > 5.0
                and (f not in scores or (scores[f] is None and seen.get(f) != mtime(f)))]
        todo.sort(key=mtime, reverse=True)
        for i, f in enumerate(todo, 1):
            if not _live(samples_dir, run, gen):
                return
            seen[f] = mtime(f)
            e = prep_jobs._embed(os.path.join(samples_dir, f), emb)
            scores[f] = None if e is None else round(float(np.mean([float(np.dot(b, e)) for b in base])), 4)
            if i % 2 == 0 or i == len(todo):
                _write(samples_dir, run, names, f"scoring… {len(scores)}/{len(files)}", scores, meta)
        status = f"live - {len(scores)} sample(s) scored" + ("" if emb.name == "arcface" else f" ({emb.label})")
        if todo or status != last:
            _write(samples_dir, run, names, status, scores, meta)
            last = status
        for _ in range(16):
            if not _live(samples_dir, run, gen):
                return
            time.sleep(0.25)
