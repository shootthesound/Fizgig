"""The Samples gallery's likeness scoring (the desktop gallery's): every sample picture's largest face against three
photos of the subject from the dataset (on the CPU - training keeps the GPU), the mean of the three similarities. The
face embedder is Image Prep's look match; with Auto it is decided from the three photos (AdaFace for a person, DINOv2
for a furry / anime character), and likeness.json records which one made the scores, so a change rescores. Scores go to
<samples>/likeness.json in the desktop's format, so either app shows the other's scores and picks up its baselines;
the gallery.owner file keeps two sessions from scoring one folder against each other."""

import json
import os
import threading
import time

from fizgig.web import prep_jobs

PICS = (".png", ".jpg", ".jpeg", ".webp")
BATCH = 20                     # previews scored between looks at the folder
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


def _run_of(name):
    from fizgig.web.server import sample_info          # here, not at the top: server imports this module
    return (sample_info(name) or {}).get("run")


def state(samples_dir, dataset_dir, run=""):
    """likeness.json for the page, resuming the scorer when baselines were chosen (here or in the desktop) and nothing
    scores this folder in this server yet. run: the run the gallery shows - only its previews are scored (the folder
    holds every run's; the others keep the scores they have); blank: the whole folder."""
    d = read(samples_dir)
    names = d.get("baselines") or []
    with _lock:
        running = samples_dir in _runs
        if running:
            _runs[samples_dir]["run"] = run or ""        # the scorer follows the run on show
    if names and not running and dataset_dir and os.path.isdir(dataset_dir) \
            and not all(os.path.exists(os.path.join(dataset_dir, os.path.basename(str(n)))) for n in names):
        names = []                                    # another run's pick (a different dataset): not chosen yet
        d["status"] = ""
    if len(names) == 3 and not running and dataset_dir and os.path.isdir(samples_dir)             and not str(d.get("status", "")).startswith("error"):     # a failed pick waits for a new one
        try:
            start(samples_dir, dataset_dir, names, run=run)    # opening the gallery claims it (the desktop's rule)
        except ValueError as e:
            d["status"] = f"error: {e}"
    thr = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else None
    real = d.get("real") if isinstance(d.get("real"), dict) and d["real"].get("n") else None
    return {"baselines": names, "status": d.get("status", ""), "scores": d.get("scores") or {},
            "available": prep_jobs.faces_available(), "embedder": d.get("embedder_label") or "",
            "thresholds": thr or {"match": 0.42, "borderline": 0.36}, "real": real}


def _summary(vals):
    """{n, median, p25, p75, best} of the training photos' scores (what a real photo of the person scores)."""
    import numpy as np
    v = np.asarray(sorted(vals), dtype=float)
    return {"n": int(len(v)), "median": round(float(np.median(v)), 4), "p25": round(float(np.percentile(v, 25)), 4),
            "p75": round(float(np.percentile(v, 75)), 4), "best": round(float(v.max()), 4)}


def _dataset_sig(dataset_dir, names):
    """What the training photos' measure depends on besides the model: the photos (names, sizes, times), not the 3."""
    from fizgig.dataset import prep
    out = []
    for p in prep.look_files(dataset_dir):
        if os.path.basename(p) in names:
            continue
        try:
            st = os.stat(p)
            out.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass
    return "|".join(out)


def real_photos(dataset_dir, names, base, emb, log=None):
    """How much the person's own training photos look like the 3 reference photos (each photo's mean similarity to
    them, the same measure as a sample's): the realistic top a preview can reach, since even real photos of the person
    never score 100%. The 3 themselves and photos with no face found are left out. Reuses a Look Filter scan made with
    the same 3 photos and the same model (fizgig_look_scores.json); otherwise embeds each photo once (CPU)."""
    import numpy as np
    from fizgig.dataset import prep
    from fizgig.face_utils import same_embedder
    files = [p for p in prep.look_files(dataset_dir) if os.path.basename(p) not in names]
    try:
        with open(os.path.join(dataset_dir, "fizgig_look_scores.json"), encoding="utf-8") as f:
            look = json.load(f)
        if sorted(look.get("baselines") or []) == sorted(names) and same_embedder(look.get("embedder"), emb.cache_id):
            stems = {os.path.splitext(os.path.basename(p))[0] for p in files}
            vals = [s for k, s in (look.get("scores") or {}).items() if k in stems and isinstance(s, (int, float))]
            if len(vals) >= 4:
                return _summary(vals)
    except (OSError, ValueError, AttributeError):
        pass
    vals = []
    for i, p in enumerate(files, 1):
        e = prep_jobs._embed(p, emb)
        if e is not None:
            vals.append(float(np.mean([float(np.dot(b, e)) for b in base])))
        if log and (i % 5 == 0 or i == len(files)):
            log(i, len(files))
    return _summary(vals) if len(vals) >= 4 else None


def start(samples_dir, dataset_dir, names, run=""):
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
        _runs[samples_dir] = {"gen": gen, "dataset": dataset_dir, "run": run or ""}
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
    same = sorted(old.get("baselines") or []) == sorted(names) and same_embedder(old.get("embedder"), emb.cache_id)
    scores = old["scores"] if same and isinstance(old.get("scores"), dict) else {}
    # the training photos' own range (real_photos), kept while the photos, the 3 and the model are the same
    dataset_dir = os.path.dirname(baselines[0])
    sig = _dataset_sig(dataset_dir, names)
    if same and isinstance(old.get("real"), dict) and old.get("real_sig") == sig:
        meta.update(real=old["real"], real_sig=sig)
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
    measuring, wrote_real = [], "real_sig" in meta      # the training photos' measure: started / written

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
            every = [f for f in os.listdir(samples_dir) if f.lower().endswith(PICS)]
        except OSError:
            every = []
        have = set(every)
        with _lock:
            want = (_runs.get(samples_dir) or {}).get("run") or ""
        files = [f for f in every if _run_of(f) == want] if want else every     # the run on show, not the folder
        scores = {k: v for k, v in scores.items() if k in have}          # deleted samples drop out
        now = time.time()
        todo = [f for f in files if now - mtime(f) > 5.0
                and (f not in scores or (scores[f] is None and seen.get(f) != mtime(f)))]
        todo.sort(key=mtime, reverse=True)
        # a batch at a time, the folder looked at again between batches: a preview the run writes meanwhile is next,
        # not behind a whole folder's rescore (a model change rescores every run's previews here)
        more = len(todo) > BATCH
        todo = todo[:BATCH]
        for i, f in enumerate(todo, 1):
            if not _live(samples_dir, gen):
                return
            seen[f] = mtime(f)
            e = prep_jobs._embed(os.path.join(samples_dir, f), emb)
            scores[f] = None if e is None else round(float(np.mean([float(np.dot(b, e)) for b in base])), 4)
            if i % 2 == 0 or i == len(todo):
                _write(samples_dir, names, f"scoring… {len(scores)}/{len(files)}", scores, meta)
        if not todo and "real_sig" not in meta and not measuring:
            # once the previews are scored: the training photos' own range, on a thread of its own, so a new preview
            # never waits for it (the two share the face model, its lock interleaving them a picture at a time)
            measuring.append(True)

            def measure():
                try:
                    real = real_photos(dataset_dir, names, base, emb)
                except Exception as e:                     # never stops the samples' scoring
                    print(f"[likeness] couldn't measure the training photos: {type(e).__name__}: {e}", flush=True)
                    real = None
                meta.update(real=real, real_sig=sig)       # the loop writes it with its next status
            threading.Thread(target=measure, daemon=True).start()
        if "real_sig" in meta and not wrote_real:
            wrote_real, last = True, None                  # written below, with the measure
        status = f"live - {len(scores)} sample(s) scored" + ("" if emb.name == "adaface" else f" ({emb.label})")
        if more:
            continue                                       # straight on with the next batch, the folder looked at again
        if todo or status != last:
            _write(samples_dir, names, status, scores, meta)
            last = status
        for _ in range(16):
            if not _live(samples_dir, gen):
                return
            time.sleep(0.25)
