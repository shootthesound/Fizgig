"""Per-image loss charts for the Runs page: each image's RESIDUAL by epoch - its loss minus the mean loss at the same
noise level (timestep bucket) over the whole log - from a run's loss_log/per_image_loss.jsonl, with the loss watch's
latest verdicts (problem_images.json). The residual is the loss watch's own measure (training/loss_logger.py
epoch_boundary: 40 buckets, means over every record), so a chart reads as the watch reads it: above zero = harder than
the dataset at that noise level, falling = being learned. A family whose description leaves loss_watch off never
writes the log, so it never has charts."""

import json
import os
import re

LOG = os.path.join("loss_log", "per_image_loss.jsonl")
REPORT = os.path.join("loss_log", "problem_images.json")
N_BUCKETS = 40              # = training/loss_logger._N_BUCKETS
_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif")

_cache = {}                 # log path -> ((mtime, size), result)


def log_path(output_dir):
    p = os.path.join(output_dir or "", LOG)
    return p if output_dir and os.path.isfile(p) else ""


def residuals(output_dir):
    """{"epochs": [..], "images": [{key, series: [[epoch, residual], ...], mean, recent, n}], "report": {..}} - a
    batch-mean key ("a|b") is no single image's and is left out, as the watch's report leaves it out."""
    p = log_path(output_dir)
    if not p:
        return None
    stamp = (os.path.getmtime(p), os.path.getsize(p))
    hit = _cache.get(p)
    if hit and hit[0] == stamp:
        return hit[1]
    recs, bsum, bcnt = [], [0.0] * N_BUCKETS, [0] * N_BUCKETS
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                t, loss = float(r["t"]), float(r["loss"])
            except (ValueError, KeyError, TypeError):
                continue                             # a line cut short by a crash / a write in progress
            b = min(int(min(max(t, 0.0), 0.999999) * N_BUCKETS), N_BUCKETS - 1)
            bsum[b] += loss
            bcnt[b] += 1
            recs.append((str(r.get("key", "")), int(r.get("epoch", 0)), b, loss))
    mean = [bsum[i] / bcnt[i] if bcnt[i] else 0.0 for i in range(N_BUCKETS)]
    per = {}
    for key, ep, b, loss in recs:
        if "|" in key:
            continue
        per.setdefault(key, {}).setdefault(ep, []).append(loss - mean[b])
    epochs = sorted({ep for _k, ep, _b, _l in recs})
    images = []
    for key, eps in per.items():
        series = [[e, round(sum(v) / len(v), 6)] for e, v in sorted(eps.items())]
        vals = [s[1] for s in series]
        recent = vals[-5:]                           # the watch's window (PerImageLossWatch window=5)
        images.append({"key": key, "series": series, "mean": round(sum(vals) / len(vals), 6),
                       "recent": round(sum(recent) / len(recent), 6), "n": sum(len(v) for v in eps.values())})
    report = {}
    try:
        with open(os.path.join(output_dir, REPORT), encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        pass
    verdicts = report.get("images") or {}
    for im in images:
        v = verdicts.get(im["key"]) or {}
        im["verdict"], im["multiplier"] = v.get("verdict"), v.get("multiplier")
    out = {"epochs": epochs, "images": images,
           "report": {k: report.get(k) for k in ("epoch", "plateaued", "best_epoch_estimate", "improving_count",
                                                 "apply_lr", "batched")}}
    _cache[p] = (stamp, out)
    return out


def caption_state(output_dir):
    """(keys waiting in the trainer's caption queue, {key: [{epoch, caption, attempt, auto}]} already re-encoded)."""
    d = os.path.join(output_dir or "", "loss_log")
    queued = set()
    for name in ("caption_updates.json", "caption_updates.json.processing"):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                queued.update(json.load(f).keys())
        except (OSError, ValueError, AttributeError):
            pass
    try:
        with open(os.path.join(d, "caption_updates_applied.json"), encoding="utf-8") as f:
            applied = {k: (v if isinstance(v, list) else [v]) for k, v in json.load(f).items()}
    except (OSError, ValueError, AttributeError):
        applied = {}
    return queued, applied


def _where(run, key):
    """(folder, name in it) of a training key, from the run as it is on this machine (a run moved here or back has
    its own paths): a sub-folder's key ("outfits/red/a") is a path inside the run's training folder, another
    subject's ("s2:a") is that subject's folder in the run's settings."""
    key, image_folder = str(key), run.get("image_folder") or ""
    m = re.match(r"s(\d+):(.*)$", key)
    if m:
        from fizgig.families.form import CONCEPT_ROWS
        n = int(m.group(1))
        rows = CONCEPT_ROWS[n - 2:n - 1] if n >= 2 else ()
        d = str((run.get("values") or {}).get(rows[0][0]) or "") if rows else ""
        return d, m.group(2)
    rel, _, name = key.rpartition("/")
    if rel == "fizgig_detail_crops":       # a detail crop (one key for every size): the size the run trains at
        from fizgig.dataset import detail_crops
        try:
            mp = float(str((run.get("values") or {}).get("DATASET_MEGAPIXELS") or "0").split(" ")[0])
        except ValueError:
            mp = 0.0
        return (detail_crops.size_dir(image_folder, mp) if mp > 0 else ""), name
    return (os.path.join(image_folder, *rel.split("/")) if rel else image_folder), name


def read_caption(run, key, ext):
    folder, name = _where(run, key)
    p = os.path.join(folder or "", name + ext)
    try:
        from fizgig.web.captions import read_text
        return read_text(p).strip(), True                 # a BOM dropped, Windows-1252 and UTF-16 read right
    except OSError:
        return "", os.path.isdir(folder or "")


def save_caption(run, key, caption, ext):
    """An image's caption fixed from the per-image loss view (the desktop's Problem Images editor): the caption file
    rewritten, and - while the run is going or paused - the edit queued for the trainer, which re-encodes it at the
    next epoch boundary and restarts that image's history (fizgig/families/loss_watch.py)."""
    caption = str(caption or "").strip()
    if not caption:
        raise ValueError("The caption is empty - not saved.")
    key = str(key)
    folder, name = _where(run, key)
    wrote, queued = False, False
    if os.path.isdir(folder):
        p = os.path.join(folder, name + ext)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(caption)
        os.replace(tmp, p)
        wrote = True
    if run.get("result") in ("running", "paused") and run.get("output_dir"):
        d = os.path.join(run["output_dir"], "loss_log")
        os.makedirs(d, exist_ok=True)
        q = os.path.join(d, "caption_updates.json")
        updates = {}
        if os.path.exists(q):
            try:
                with open(q, encoding="utf-8") as f:
                    updates = json.load(f)
            except (OSError, ValueError):
                # other edits are waiting in a file we can't read: writing just this one would lose them
                raise ValueError("The trainer's caption queue couldn't be read - not overwriting it. Try again in a "
                                 "moment.")
            if not isinstance(updates, dict):
                updates = {}
        updates[key] = caption
        with open(q + ".tmp", "w", encoding="utf-8") as f:
            json.dump(updates, f, indent=2)
        os.replace(q + ".tmp", q)                    # atomic: the trainer never reads half a file
        queued = True
    if not wrote and not queued:
        raise ValueError("The run's training folder isn't there any more - nothing saved.")
    return {"wrote": wrote, "queued": queued}


def image_paths(run, keys):
    """{key: picture path} - a key is its folder's prefix + the picture's name without its extension."""
    out, by = {}, {}
    for k in keys:
        folder, name = _where(run, k)
        by.setdefault(folder, {})[name] = k
    for folder, want in by.items():
        try:
            names = os.listdir(folder or "")
        except OSError:
            continue
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() in _IMAGES and stem in want and want[stem] not in out:
                out[want[stem]] = os.path.join(folder, n)
    return out


def owner_runs(runs):
    """{run id: True} for the run each log belongs to: a fresh run rotates the log away, so a folder's log is the
    newest run there that started before the log was last written."""
    by_dir = {}
    for r in runs:
        p = log_path(r.get("output_dir"))
        if p:
            by_dir.setdefault(os.path.normcase(os.path.abspath(p)), (p, []))[1].append(r)
    owners = {}
    for p, rs in by_dir.values():
        written = os.path.getmtime(p)
        fits = [r for r in rs if float(r.get("started_at") or 0) <= written + 60]
        if fits:
            owners[max(fits, key=lambda r: float(r.get("started_at") or 0))["id"]] = True
    return owners
