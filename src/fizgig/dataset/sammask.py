"""Masks from words for masked training: Meta's SAM 3.1 (fizgig/sam3, ported from ComfyUI), as an alternative to
the person finder in automask.py. "Mask these" is a few short phrases ("person", "hair", "the dog"); each picture's
mask is everything they find, less what "Leave these out" finds ("glasses", "the phone").

The model is ComfyUI's single file (Comfy-Org/sam3.1, 1.75 GB), downloaded once into Fizgig's models folder from the
masks card. It runs on the GPU when there is one (a fraction of a second a picture), on the CPU otherwise (several seconds)."""
import os
import threading
import time

import numpy as np
from PIL import Image

from fizgig.dataset.automask import pictures
from fizgig.dataset.masks import MASK_SUFFIX, MIN_COVER, mask_for
from fizgig.utils.upright import upright
from fizgig.utils import hf_public

REPO, FILE = "Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors"
NAME = os.path.basename(FILE)
SIZE_GB = 1.75
IDLE_UNLOAD = 120                         # seconds the loaded model waits for the next preview or run

_lock = threading.Lock()
_use = threading.Lock()                  # one preview or mask run on the model at a time
_loaded = {"path": None, "seg": None, "used": 0.0}
_fetch = {"state": "", "error": "", "at": 0.0}


def split_words(text):
    """'person, hair' -> ['person', 'hair']: commas or new lines between phrases."""
    return [p.strip() for p in str(text or "").replace("\n", ",").split(",") if p.strip()]


def find_model(models_dir):
    """The downloaded SAM 3.1 file in the models folder, or "" before the download."""
    p = os.path.join(models_dir, NAME)
    return p if os.path.isfile(p) else ""


def download(models_dir):
    """Fetch the file into the models folder on a worker thread; fetch_state() follows it."""
    with _lock:
        if _fetch["state"] == "running":
            return
        _fetch.update(state="running", error="", at=time.time())

    def work():
        try:
            got = hf_public.download(repo_id=REPO, filename=FILE, local_dir=models_dir)
            dest = os.path.join(models_dir, NAME)
            if os.path.normcase(os.path.abspath(got)) != os.path.normcase(os.path.abspath(dest)):
                os.replace(got, dest)              # from the repo's checkpoints/ subfolder to the models folder
            with _lock:
                _fetch.update(state="done", at=time.time())
        except Exception as e:
            with _lock:
                _fetch.update(state="failed", error=hf_public.friendly_error(e), at=time.time())
    threading.Thread(target=work, daemon=True).start()


def fetch_state(models_dir):
    """{"state", "error", "gb", "of"}: how the download is going (gb arrived, from hf_hub_download's part file)."""
    with _lock:
        st = dict(_fetch)
    gb = 0.0
    if st["state"] == "running":
        d = os.path.join(models_dir, ".cache", "huggingface", "download", os.path.dirname(FILE))
        try:
            gb = max((os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".incomplete")),
                     default=0) / 2 ** 30
        except OSError:
            pass
    return {"state": st["state"], "error": st["error"], "gb": round(gb, 2), "of": SIZE_GB}


def segmenter(path, log=print):
    """The loaded model, kept a couple of minutes for the next preview or run, then freed."""
    with _lock:
        if _loaded["seg"] is not None and _loaded["path"] == path:
            _loaded["used"] = time.time()
            return _loaded["seg"]
    import torch
    from fizgig.sam3.segment import SAM3Segmenter
    unload()
    log("Loading SAM 3.1 ...\n")
    cuda = torch.cuda.is_available()
    seg = SAM3Segmenter(path, device="cuda" if cuda else "cpu", dtype=torch.float16 if cuda else torch.float32)
    with _lock:
        _loaded.update(path=path, seg=seg, used=time.time())
    threading.Thread(target=_unload_when_idle, daemon=True).start()
    return seg


def _unload_when_idle():
    while True:
        time.sleep(15)
        with _lock:
            if _loaded["seg"] is None:
                return
            if time.time() - _loaded["used"] < IDLE_UNLOAD:
                continue
        unload()
        return


def unload():
    with _lock:
        seg, _loaded["seg"], _loaded["path"] = _loaded["seg"], None, None
    if seg is not None:
        del seg
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def picture_mask(seg, img, include, exclude, subject=None, detector=None):
    """One picture's mask (H, W) float in [0, 1], and a note. subject / detector: as automask.make_folder - with
    several people, only the person whose face matches the baselines is kept (SAM's own "person" outline of them)."""
    arr = np.asarray(img)
    found = seg.masks(arr, list(dict.fromkeys(include + exclude)))
    h, w = arr.shape[:2]
    m = np.zeros((h, w), bool)
    for f in found:
        if f["prompt"] in include:
            m |= f["mask"]
    for f in found:
        if f["prompt"] in exclude:
            m &= ~f["mask"]
    note = ""
    if subject is not None and detector is not None:
        faces = [f.bbox for f in (detector.detect_from_pil(img) or [])]
        if len(faces) > 1:
            embed, bases = subject
            scores = []
            for x1, y1, x2, y2 in faces:
                pw, ph = (x2 - x1) * 0.6, (y2 - y1) * 0.6
                crop = img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                                 min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph))))
                v = embed(crop)
                scores.append(-1.0 if v is None else float(np.mean(
                    [np.dot(v, b) / (np.linalg.norm(v) * np.linalg.norm(b) + 1e-8) for b in bases])))
            best = int(np.argmax(scores))
            x1, y1, x2, y2 = (int(v) for v in faces[best])
            people = [f["mask"] for f in (seg.masks(arr, ["person"]) if "person" not in include + exclude else
                                          [f for f in found if f["prompt"] == "person"])]
            # the person most over the matching face: their outline bounds the mask
            cover = [p[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].mean() if p.size else 0 for p in people]
            if people and max(cover) > 0:
                import cv2
                own = cv2.dilate(people[int(np.argmax(cover))].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                m &= own
                note = f" (kept 1 of {len(faces)} people: the best match, {scores[best] * 100:.0f}%)"
    return m.astype(np.float32), note


def make_folder(folder, include, exclude, path, *, overwrite=False, subject=None, detector=None, log=print):
    """Write a mask for every picture in a folder, as automask.make_folder does. Returns its counts."""
    with _use:
        return _make_folder(segmenter(path, log), folder, include, exclude, overwrite, subject, detector, log)


def _make_folder(seg, folder, include, exclude, overwrite, subject, detector, log):
    counts = {"made": 0, "kept": 0, "skipped": 0, "no_person": 0}
    pics = pictures(folder)
    for i, p in enumerate(pics, 1):
        name = os.path.basename(p)
        out = os.path.join(folder, os.path.splitext(name)[0] + MASK_SUFFIX + ".png")
        old = mask_for(p)
        if old and not overwrite:
            counts["kept"] += 1
            continue
        try:
            with Image.open(p) as im:
                img = upright(im).convert("RGB")      # as training reads it
        except Exception as e:
            log(f"  {name}: can't be read ({e}) - skipped\n")
            counts["skipped"] += 1
            continue
        with _lock:
            _loaded["used"] = time.time()
        m, note = picture_mask(seg, img, include, exclude, subject, detector)
        if float(m.mean()) < MIN_COVER:
            log(f"  {name}: none of {', '.join(include)} found{note} - no mask written (it trains as normal)\n")
            counts["no_person"] += 1
            continue
        if old and os.path.normcase(old) != os.path.normcase(out):
            os.remove(old)                           # replaced: one mask per picture
        Image.fromarray((m * 255).astype(np.uint8), "L").save(out)
        counts["made"] += 1
        log(f"  {name} -> {os.path.basename(out)}{note}\n")
        if i % 20 == 0:
            log(f"{i} of {len(pics)} pictures\n")
    return counts


def point(img, x, y, path):
    """The mask editor's SAM click: the object at (x, y) of the upright picture `img` (PIL) -> bool (H, W)."""
    if not _use.acquire(blocking=False):
        raise ValueError("Masks are being made now. Wait for them to finish.")
    try:
        return segmenter(path).point_mask(np.asarray(img), [(float(x), float(y), 1)])
    finally:
        _use.release()


def preview_pictures(folder, shown=(), n=2):
    """(n random pictures not in `shown` - the names already previewed this round, the names that start the next
    round). When fewer than n are left, those go first and the rest come from a fresh round; the second list is then
    the fresh round's so far, else None."""
    import random
    pics = pictures(folder)
    seen = {s.lower() for s in shown}
    left = [p for p in pics if os.path.basename(p).lower() not in seen]
    if len(left) >= min(n, len(pics)):
        return random.sample(left, min(n, len(left))), None
    fill = random.sample([p for p in pics if p not in left], min(n, len(pics)) - len(left))
    return left + fill, [os.path.basename(p) for p in fill]


def preview(folder, include, exclude, path, shown=(), n=2, side=320):
    """n random pictures not previewed yet (preview_pictures), with their masks over them: ([(name, PIL picture,
    share masked), ...], the next round's names or None), nothing written."""
    if not _use.acquire(blocking=False):
        raise ValueError("Masks are being made now. Wait for them to finish.")
    try:
        pics, fresh = preview_pictures(folder, shown, n)
        return _preview(segmenter(path), pics, include, exclude, side), fresh
    finally:
        _use.release()


def _preview(seg, pics, include, exclude, side):
    out = []
    for p in pics:
        with Image.open(p) as im:
            img = upright(im).convert("RGB")
        m, _ = picture_mask(seg, img, include, exclude)
        a = np.asarray(img).astype(np.float32)
        tint = np.array([90, 170, 255], np.float32)
        shown = np.where(m[..., None] > 0.5, a * 0.45 + tint * 0.55, a * 0.3)
        pic = Image.fromarray(shown.clip(0, 255).astype(np.uint8))
        pic.thumbnail((side, side))
        out.append((os.path.basename(p), pic, float(m.mean())))
    return out
