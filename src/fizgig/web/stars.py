"""Starred previews (Peter, 12 Nov 2026): a star on any preview in the Samples gallery or a run's card, kept by file
name in <samples>/fizgig_stars.json, so it lives with the previews; a checkpoint whose own epoch's preview is starred
shows a star in the checkpoint lists. A preview and its epoch's checkpoint are written together (the same second in
practice), so a checkpoint matches a starred preview of its run and epoch saved within MATCH_S of it - a later run
reusing the name never inherits an earlier one's stars."""
import json
import os
import re
import threading

FILE = "fizgig_stars.json"
MATCH_S = 600
_lock = threading.Lock()
_CKPT_EPOCH = re.compile(r"^(?P<run>.+?)(?:-(?P<epoch>\d{6}))?\.safetensors$", re.I)


def read(samples_dir) -> set:
    try:
        with open(os.path.join(samples_dir, FILE), encoding="utf-8") as f:
            return {str(n) for n in (json.load(f).get("stars") or [])}
    except (OSError, ValueError, AttributeError, TypeError):
        return set()


def toggle(path, on) -> bool:
    """Star (on) or unstar one preview, given its full path. Returns whether it's starred now."""
    folder, name = os.path.split(os.path.abspath(path))
    if not os.path.isfile(path):
        raise ValueError("That preview isn't there any more.")
    with _lock:
        stars = read(folder)
        (stars.add if on else stars.discard)(name)
        stars = {n for n in stars if os.path.exists(os.path.join(folder, n))}      # deleted previews drop out
        p = os.path.join(folder, FILE)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"stars": sorted(stars)}, f, indent=1)
        os.replace(p + ".tmp", p)
    return name in stars


def starred_checkpoints(out_dir, items, sample_info):
    """The names among items ({name, time} dicts of an output folder's .safetensors) whose epoch's preview is starred:
    the same run, the same epoch (a run's final file without a number: any of its previews) and saved within MATCH_S
    of each other."""
    samples = os.path.join(out_dir, "sample")
    stars = read(samples)
    if not stars:
        return set()
    marks = []                                   # (run, epoch, mtime) of each starred preview still there
    for n in stars:
        info = sample_info(n) or {}
        try:
            t = os.path.getmtime(os.path.join(samples, n))
        except OSError:
            continue
        if info.get("run") is not None:
            marks.append((info["run"], info.get("epoch"), t))
    out = set()
    for it in items:
        m = _CKPT_EPOCH.match(it["name"])
        if not m:
            continue
        run, epoch = m.group("run"), None if m.group("epoch") is None else int(m.group("epoch"))
        if any(r == run and (epoch is None or e == epoch) and abs(t - it["time"]) <= MATCH_S for r, e, t in marks):
            out.add(it["name"])
    return out
