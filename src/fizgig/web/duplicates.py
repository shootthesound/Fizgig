"""CPU-only near-duplicate picture finder (@Murf-Code's duplicate scan).

No training model, GPU, or network calls are used. Pictures are decoded one at
a time by a background thread. Duplicate candidates follow TagBench 1.2's 64-bit difference-hash and 4x4
RGB-grid matcher with its strictness slider 0-16. All matches require human review.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from fizgig.dataset.masks import is_mask_file

PHOTO_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".avif"}
MAX_FILES = 30000
MAX_PAIRS = 150000
MAX_DISTANCE = 16   # TagBench strictness slider: 0 identical-looking, 16 lenient
_JOBS: dict[str, dict] = {}
_LOCK = threading.RLock()
_WRITE_LOCK = threading.RLock()


def _root(folder):
    if not isinstance(folder, (str, os.PathLike)) or not str(folder).strip():
        raise ValueError("Choose an existing training dataset folder.")
    folder = os.path.realpath(os.path.abspath(str(folder)))
    if not os.path.isdir(folder):
        raise ValueError("Choose an existing training dataset folder.")
    return folder


def checked_path(root, path):
    """Only pictures inside a dataset, never masks, backups or removed files."""
    if not isinstance(path, str) or not path:
        raise ValueError("A picture path is missing.")
    p = os.path.realpath(os.path.abspath(
        path if os.path.isabs(path) else os.path.join(root, path)))
    if os.path.commonpath((root, p)) != root or p == root:
        raise ValueError("Picture is outside this dataset.")
    parts = os.path.relpath(p, root).split(os.sep)
    if any(x.lower() in ("removed", "removed_watermarks", ".git", "__pycache__")
           or x.startswith(".") for x in parts[:-1]):
        raise ValueError("Removed/hidden folders are not training pictures.")
    if os.path.splitext(p)[1].lower() not in PHOTO_EXTS or is_mask_file(os.path.basename(p)):
        raise ValueError("Only source photos may be compared or merged.")
    if not os.path.isfile(p):
        raise ValueError(f"Picture no longer exists: {os.path.basename(p)}")
    return p


def paths_for_scan(folder, paths):
    root = _root(folder)
    if not isinstance(paths, list) or len(paths) > MAX_FILES:
        raise ValueError(f"Send at most {MAX_FILES} dataset pictures.")
    result = {}
    for path in paths:
        try:
            p = checked_path(root, path)
            st = os.stat(p)
            result[p] = (st.st_size, st.st_mtime_ns)
        except (ValueError, OSError):
            continue  # an edited, removed or still-uploading file can vanish mid-scan
    return root, sorted(result), result


def fingerprint(path):
    """TagBench 1.2: 64-bit dHash + 4×4 RGB-grid + exact file check.

    Its original implementation is in TagBench's src/dupe-match.ts. Python
    computes the same grayscale-neighbor comparison and 40-point RGB distance
    for the web server; SHA-256 replaces the collision-prone 64-bit file FNV.
    """
    with Image.open(path) as src:
        src.seek(0)
        im = ImageOps.exif_transpose(src).convert("RGBA")
        # The TagBench canvas starts transparent; getImageData exposes
        # premultiplied RGB. Composite over black to match transparent pixels.
        rgb = Image.new("RGB", im.size, (0, 0, 0))
        rgb.paste(im, mask=im.getchannel("A"))
        width, height = rgb.size
        small = np.asarray(rgb.resize((9, 8), Image.Resampling.BILINEAR),
                           dtype=np.float64)
        luminance = small[:, :, 0] * .30 + small[:, :, 1] * .59 + small[:, :, 2] * .11
        differences = luminance[:, :-1] > luminance[:, 1:]
        # Row-major and low-bit first, as TagBench's dhash() produces.
        bitmask = sum(1 << i for i, yes in enumerate(differences.ravel()) if yes)
        color = np.asarray(rgb.resize((4, 4), Image.Resampling.BILINEAR),
                           dtype=np.uint8).reshape(48).tobytes()
    exact = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            exact.update(block)
    return bitmask, color, width, height, exact.hexdigest()


def hamming(a, b):
    return (a ^ b).bit_count()


def color_distance(left, right):
    # Count 4×4 cells whose RGB difference exceeds 40, same as TagBench.
    n = 0
    for i in range(0, 48, 3):
        if sum((left[i+k] - right[i+k]) ** 2 for k in range(3)) > 40 * 40:
            n += 1
    return n


def apart(a, b):
    """Original TagBench distance: 0 for equal bytes else max(dHash, color)."""
    if a[4] == b[4]:
        return 0
    return max(hamming(a[0], b[0]), color_distance(a[1], b[1]))


def distance(a, b):
    """Hamming metric for the BK-tree's inexpensive candidate screening."""
    return hamming(a, b)


class _BKTree:
    """Metric tree over Hamming bit distance; no quadratic all-pairs scan."""
    def __init__(self):
        self.root = None

    def insert(self, key, index):
        if self.root is None:
            self.root = [key, [index], {}]
            return
        node = self.root
        while True:
            d = distance(key, node[0])
            if not d:
                node[1].append(index)
                return
            child = node[2].get(d)
            if child is None:
                node[2][d] = [key, [index], {}]
                return
            node = child

    def matches(self, key, radius):
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = distance(key, node[0])
            if d <= radius:
                yield from node[1]
            stack.extend(child for k, child in node[2].items()
                         if d - radius <= k <= d + radius)


def _compute(job):
    tree = _BKTree()
    features = {}
    files = job["paths"]
    failures = []
    for i, path in enumerate(files):
        try:
            expected = job["stamps"][path]
            st = os.stat(path)
            if (st.st_size, st.st_mtime_ns) != expected:
                raise ValueError("changed during duplicate scan")
            f = fingerprint(path)
            st = os.stat(path)
            if (st.st_size, st.st_mtime_ns) != expected:
                raise ValueError("changed while being decoded")
            features[path] = f
        except (OSError, ValueError, Image.UnidentifiedImageError) as e:
            if len(failures) < 12:
                failures.append(os.path.basename(path) + ": " + str(e))
        if i % 12 == 0 or i + 1 == len(files):
            with _LOCK:
                job["processed"] = i + 1
    with _LOCK:
        job["stage"] = "matching"
    pairs = []
    truncated = False
    records = list(features.items())
    # Precompute TagBench-distance edges to 16 once; slider updates filter in memory.
    radius = MAX_DISTANCE
    for i, (_path, f) in enumerate(records):
        for j in tree.matches(f[0], radius):
            other_path, other = records[j]
            diff = apart(other, f)
            if diff <= MAX_DISTANCE:
                pairs.append((other_path, _path, diff, other[4] == f[4]))
                if len(pairs) >= MAX_PAIRS:
                    truncated = True
                    break
        if truncated:
            break
        tree.insert(f[0], i)
    with _LOCK:
        job.update(stage="done", files={p: {"path": p, "name": os.path.basename(p),
                                            "width": f[2], "height": f[3]}
                                        for p, f in features.items()},
                   pairs=pairs, failures=failures, truncated=truncated,
                   scanned=len(features))


def _worker(job):
    try:
        _compute(job)
    except Exception as exc:
        with _LOCK:
            job.update(stage="error", error=f"{type(exc).__name__}: {exc}")


def start(folder, paths):
    root, files, stamps = paths_for_scan(folder, paths)
    with _LOCK:
        old = _JOBS.get(root)
        if old and old["stage"] in ("scanning", "matching"):
            return status(root)
        job = {"stage": "scanning", "paths": files, "stamps": stamps,
               "total": len(files), "processed": 0, "files": {}, "pairs": [],
               "failures": [], "truncated": False, "scanned": 0}
        _JOBS[root] = job
        # Drop older folders' scans to bound retained fingerprints and matches.
        if len(_JOBS) > 3:
            for key in list(_JOBS):
                if key != root and _JOBS[key]["stage"] not in ("scanning", "matching"):
                    _JOBS.pop(key, None)
    threading.Thread(target=_worker, args=(job,), name="caption-duplicate-scan",
                     daemon=True).start()
    return status(root)


def status(folder, strictness=4, offset=0, page_size=32):
    root = _root(folder)
    try:
        strictness = int(strictness)
    except (TypeError, ValueError):
        raise ValueError("Strictness must be an integer from 0 through 16.") from None
    if not 0 <= strictness <= MAX_DISTANCE:
        raise ValueError(f"Strictness must be 0–{MAX_DISTANCE}.")
    try:
        offset = int(offset)
    except (ValueError, TypeError):
        offset = 0
    offset = max(0, offset)
    with _LOCK:
        j = _JOBS.get(root)
        if j is None:
            return {"stage": "idle", "processed": 0, "total": 0, "strictness": strictness}
        overview = {k: j[k] for k in ("stage", "processed", "total", "scanned",
                                      "failures", "truncated")}
        overview["strictness"] = strictness
        if j["stage"] == "error":
            overview["error"] = j.get("error", "Scan failed")
            return overview
        if j["stage"] != "done":
            return overview
        matches = [(a, b, diff, exact) for a, b, diff, exact in j["pairs"] if diff <= strictness]
        info = j["files"]
    parent = {p: p for a, b, _diff, _exact in matches for p in (a, b)}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for a, b, _diff, _exact in matches:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa
    members = defaultdict(set)
    for p in parent:
        members[find(p)].add(p)
    costs = defaultdict(list)
    links = defaultdict(list)
    for a, b, diff, exact in matches:
        key = find(a)
        costs[key].append(diff)
        links[key].append({"a": a, "b": b, "distance": diff, "identical": exact})
    groups = []
    for paths in members.values():
        sorted_paths = sorted(paths)
        root_key = find(sorted_paths[0])
        groups.append({"id": hashlib.sha1(sorted_paths[0].encode("utf-8")).hexdigest()[:16],
                       "distance": max(costs[root_key]), "identical": all(x["identical"] for x in links[root_key]),
                       "count": len(paths), "links": links[root_key][:200],
                       "items": [info[p] for p in sorted_paths]})
    groups.sort(key=lambda g: (g["distance"], -g["count"], g["items"][0]["path"]))
    overview.update(groups=groups[offset:offset + page_size],
                    group_count=len(groups), pair_count=len(matches),
                    offset=offset, next_offset=(offset + page_size if
                        offset + page_size < len(groups) else None))
    return overview
