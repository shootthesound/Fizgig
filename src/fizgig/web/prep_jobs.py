"""The web UI's Image Prep: a prep run (resize / PNG / face close-ups) and the Look Consistency Filter's scan, each
in a background thread on the Fizgig machine, with their state for the page to poll. The work itself is
fizgig/dataset/prep.py - the same code the desktop app's Image Prep tab runs. Face tools run on the CPU, with the face
finder and look match Image Prep's face settings pick (face_utils: InsightFace / YOLO / Auto, ArcFace / DINOv2).
"""
import base64
import io
import json
import os
import threading
import time

from fizgig.dataset import prep
from fizgig.utils.image import flatten
from fizgig.utils.upright import upright

_lock = threading.Lock()
_job = {"state": "idle", "log": "", "folder": "", "started": 0.0}
_look = {"state": "idle", "folder": "", "baselines": [], "scores": {}, "progress": [0, 0], "error": "",
         "embedder": "", "thresholds": None, "decision": ""}
_embed_cache = {}             # (path, mtime, embedder settings) -> embedding or None
_detector = None
_embedders = {}               # FaceEmbedder.settings_key -> FaceEmbedder (one per look match in use)
_model_lock = threading.Lock()


def faces_available() -> bool:
    """Whether the face finder Image Prep's face settings pick can run here (the default, InsightFace: InsightFace,
    OpenCV and onnxruntime installed, as before)."""
    try:
        from fizgig.face_utils import FaceDetector
        return bool(FaceDetector().available)
    except Exception:
        return False


def _face_detector():
    """The face finder for the current face settings; made again when they change."""
    global _detector
    from fizgig.face_utils import FaceDetector
    with _model_lock:
        fresh = FaceDetector()
        if _detector is None or _detector.settings_key != fresh.settings_key:
            _detector = fresh
    return _detector


def face_embedder(settings=None):
    """The FaceEmbedder for these face settings (default: prefs.json), one per look match: the model loads once."""
    from fizgig.face_utils import FaceEmbedder
    fresh = FaceEmbedder(settings=settings)
    with _model_lock:
        return _embedders.setdefault(fresh.settings_key, fresh)


def session_embedder(paths, log=None):
    """(embedder, decision message) for one scoring session over these pictures: with Auto look match, ArcFace for a
    human set and DINOv2 for a furry / anime / mixed one (face_utils.decide_look_match) - never a mix of the two."""
    from fizgig.face_utils import decide_look_match
    d = decide_look_match(list(paths), log=log)
    return face_embedder(d.settings), ("" if d.kind == "explicit" else d.message)   # only Auto's choice is news


FACE_DETECTOR_LABELS = {"insightface": "People (InsightFace, the default)",
                        "yolo": "Anime/furry/drawn (YOLO model)",
                        "auto": "Both, for sets with people and characters"}
FACE_IDENTITY_LABELS = {"arcface": "People (the default)", "dinov2": "Anime/furry/drawn",
                        "auto": "Auto (Decide for each set)"}


def face_settings():
    """Image Prep's face settings for the page: the saved values (prefs.json face_*), the choices, and what they come
    to (the YOLO model actually used, whether the finder can tell gender)."""
    from fizgig import face_utils as F
    st = F.load_face_settings()
    try:
        with open(F.PREFS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    det = F.FaceDetector(settings=st)
    return {"detector": st["face_detector"], "identity": st["face_identity"],
            "yolo_model": str(saved.get("face_yolo_model") or ""), "auto_threshold": st["face_auto_threshold"],
            "detectors": [[k, FACE_DETECTOR_LABELS[k]] for k in ("insightface", "yolo", "auto")],
            "identities": [[k, FACE_IDENTITY_LABELS[k]] for k in ("arcface", "dinov2", "auto")],
            "defaults": {"detector": F.FACE_SETTING_DEFAULTS["face_detector"],
                         "identity": F.FACE_SETTING_DEFAULTS["face_identity"]},
            "finder": det.describe(), "has_gender": det.has_gender,
            "insightface": F.is_insightface_available(), "local_model": F.LOCAL_YOLO_DEFAULT}


def save_face_settings(b):
    """Check and save the face settings sent by the page (only the keys given); the face finder and the look match
    pick them up on their next use."""
    from fizgig import face_utils as F
    ch = {}
    if "detector" in b:
        if b["detector"] not in F.DETECTORS:
            raise ValueError("Unknown face finder.")
        ch["face_detector"] = b["detector"]
    if "identity" in b:
        if b["identity"] not in F.IDENTITIES:
            raise ValueError("Unknown look match.")
        ch["face_identity"] = b["identity"]
    if "yolo_model" in b:
        m = str(b["yolo_model"] or "").strip().strip('"')
        full = m if not m or os.path.isabs(m) else os.path.join(F._HERE, m)
        if m and not (full.lower().endswith(".onnx") and os.path.isfile(full)):
            raise ValueError("Pick an .onnx face model on the Fizgig computer. docs/FACE_BACKENDS.md explains how to "
                             "make one from a .pt file.")
        ch["face_yolo_model"] = m
    if "auto_threshold" in b:
        try:
            v = float(b["auto_threshold"])
        except (TypeError, ValueError):
            raise ValueError("The Auto threshold is a number from 0 to 1.")
        if not 0.0 <= v <= 1.0:
            raise ValueError("The Auto threshold is a number from 0 to 1.")
        ch["face_auto_threshold"] = v
    return ch


def busy() -> bool:
    with _lock:
        return _job["state"] == "running" or _look["state"] == "running"


# ---- a prep run -------------------------------------------------------------------------------------------------
def status(since=0):
    with _lock:
        log = _job["log"]
        return {"state": _job["state"], "folder": _job["folder"], "log": log[since:], "size": len(log)}


def _log(text):
    with _lock:
        _job["log"] = (_job["log"] + text)[-400_000:]


def start(folder, mode, mp, face_label, padding, replace, crops_only=False):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if mode not in prep.MODES:
        raise ValueError("Pick what to do.")
    if mode in prep.FACE_MODES and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            det = _face_detector() if mode in prep.FACE_MODES else None
            prep.Prep(_log, det).run(mode, folder, float(mp), face_label, float(padding), bool(replace),
                                     bool(crops_only))
            _log("\nDone.\n")
            end = "done"
        except Exception as e:
            _log(f"\nERROR: prep failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with _lock:
            _job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def test_faces(path, face_label):
    """Test face detection on one photo: the picture with the faces boxed (the chosen one green) as a JPEG data URL,
    and each face found. Writes nothing."""
    from PIL import Image
    from fizgig.face_utils import draw_face_boxes
    if not os.path.isfile(path):
        raise ValueError("That picture isn't there.")
    det = _face_detector()
    faces = det.detect_all(path)
    if not faces:
        return {"faces": [], "image": None, "finder": det.describe()}
    job = prep.Prep(lambda _t: None, det)
    sel, _ = job.select_face(faces, prep.face_mode_of(face_label))
    hi = faces.index(sel) if sel in faces else None
    with Image.open(path) as img:
        img = upright(img)
        shown = draw_face_boxes(flatten(img), faces, highlight_index=hi).convert("RGB")
    shown.thumbnail((1400, 1400))
    buf = io.BytesIO()
    shown.save(buf, "JPEG", quality=88)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "finder": det.describe(),
            "faces": [{"gender": f.gender, "area": int(f.area), "selected": i == hi,
                       "source": getattr(f, "source", "insightface")} for i, f in enumerate(faces)]}


# ---- the Look Consistency Filter --------------------------------------------------------------------------------
def _embed(path, embedder=None):
    """A face embedding through the (path, mtime, embedder) cache: re-scans and baseline swaps cost almost nothing.
    embedder: a session's (session_embedder); default the face settings' own."""
    emb = embedder or face_embedder()
    try:
        key = (path, os.path.getmtime(path), emb.settings_key)
    except OSError:
        return None
    if key not in _embed_cache:
        try:
            v = emb.embed(path)
        except Exception as e:
            print(f"[faces] {os.path.basename(path)}: {emb.label} failed ({type(e).__name__}: {e})", flush=True)
            v = None
        with _model_lock:
            _embed_cache[key] = v
    return _embed_cache[key]


def embed_face_box(path, embedder=None):
    """(embedding, bbox) for the largest face in a picture file - the desktop's _repair_embed_pil - on the session's
    embedder (default the face settings' own). The box is None when only the padded retry found the face (its box is
    in padded space). (None, None) without a face."""
    from PIL import Image
    emb = embedder or face_embedder()
    try:
        if emb.name == "arcface":       # read as before (OpenCV, unicode-safe): the same pixels, the same numbers
            import cv2
            import numpy as np
            raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.ndim == 3 and raw.shape[-1] == 4:     # see-through: on white
                a = raw[..., 3:].astype(np.float32) / 255.0
                raw = (raw[..., :3] * a + 255.0 * (1 - a)).round().astype(np.uint8)
            bgr = None if raw is None else (cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw[..., :3])
            return (None, None) if bgr is None else emb.embed_bgr(bgr)
        with Image.open(path) as img:
            return emb.embed_pil(flatten(upright(img)))
    except Exception:
        return None, None


def look_status():
    with _lock:
        scores = dict(_look["scores"])
        base = list(_look["baselines"])
        state, err, prog, folder = _look["state"], _look["error"], list(_look["progress"]), _look["folder"]
        thr, emb_label, decision = _look["thresholds"], _look["embedder"], _look["decision"]
    floor = (thr or prep.ARCFACE)["floor"]
    med, cut = prep.cutoff(scores, floor) if scores else (None, None)
    rows = []
    for p, s in scores.items():
        label, colour, blurb = prep.verdict(s, thr)
        rows.append({"path": p, "name": os.path.basename(p), "score": s, "verdict": label, "colour": colour,
                     "blurb": blurb, "baseline": p in base, "suggest": bool(cut is not None and s is not None
                                                                          and s < cut and p not in base)})
    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))   # worst first
    return {"state": state, "error": err, "progress": prog, "folder": folder, "baselines": base, "rows": rows,
            "median": med, "cutoff": cut, "embedder": emb_label, "decision": decision,
            "thresholds": thr or prep.ARCFACE}


def look_scan(folder, baselines):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    base = [os.path.normpath(b) for b in baselines or []]
    if len(base) != 3 or not all(os.path.isfile(b) for b in base):
        raise ValueError("Pick exactly 3 baseline pictures.")
    files = prep.look_files(folder)
    if not files:
        raise ValueError("No pictures in the training folder.")
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is already going.")
        if _job["state"] == "running":
            raise ValueError("Image prep is running. Wait for it to finish.")
        _look.update(state="running", folder=folder, baselines=base, scores={}, progress=[0, len(files)], error="",
                     embedder="", thresholds=None, decision="")

    def progress(i, total):
        with _lock:
            _look["progress"] = [i, total]

    def work():
        try:
            # one look match for the whole folder (Auto: decided from a sample of it), shown with the scores
            emb, msg = session_embedder(files, log=lambda m: print(m, flush=True))
            with _lock:
                _look.update(embedder=emb.label, thresholds=dict(emb.thresholds), decision=msg)
            scores = prep.score(files, base, lambda p: _embed(p, emb), progress)
            try:
                prep.save_scores(folder, base, scores, emb)     # for training's "Warm up look outliers"
            except OSError:
                pass
            with _lock:
                _look.update(state="done", scores=scores)
        except Exception as e:
            with _lock:
                _look.update(state="failed", error=str(e))
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def look_move(folder, paths, caption_ext=".txt"):
    """Move the marked pictures (and their captions) to excluded_by_look/; they leave the scan's list."""
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is going. Wait for it to finish.")
        known = set(_look["scores"])
    paths = [os.path.normpath(p) for p in paths or [] if os.path.normpath(p) in known]   # a marked baseline goes too
    moved, errors = prep.move_out(folder, paths, (".txt", caption_ext or ".txt"))
    with _lock:
        for p in moved:
            _look["scores"].pop(p, None)
    return {"moved": len(moved), "errors": errors, "dest": os.path.join(folder, "excluded_by_look"),
            "left": len(_look["scores"])}


# ---- Make masks (masked training) -------------------------------------------------------------------------------
def masks_available() -> bool:
    from fizgig.dataset import automask
    return automask.available()


def look_baselines(folder):
    """The Look Filter's three baselines saved with this folder (fizgig_look_scores.json), as paths; [] without."""
    try:
        with open(os.path.join(folder, "fizgig_look_scores.json"), encoding="utf-8") as f:
            names = json.load(f).get("baselines") or []
    except (OSError, ValueError):
        return []
    paths = [os.path.join(folder, n) for n in names]
    return paths if len(paths) == 3 and all(os.path.isfile(p) for p in paths) else []


def start_masks(folder, kind, subject_only=False, overwrite=False, sam=None):
    """Masks for every picture in the training folder (dataset/automask.py), as a prep run: the same log and state.
    sam: {"path", "include", "exclude"} to make them with SAM 3.1 from words instead (dataset/sammask.py)."""
    from fizgig.dataset import automask
    from fizgig.user_data import MODELS_DIR
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if sam is not None:
        if not sam.get("path"):
            raise ValueError("SAM 3.1 isn't here yet: press Download on the masks card.")
        if not sam.get("include"):
            raise ValueError("Type what the masks should show in Mask these (e.g. person).")
    elif kind not in automask.KINDS:
        raise ValueError("Pick what the masks should show.")
    elif not automask.available():
        raise ValueError("Making masks needs the ai-edge-litert package: run the updater (update_fizgig) to add it.")
    bases = look_baselines(folder) if subject_only else []
    if subject_only and not bases:
        raise ValueError("Keeping only your subject needs the Look Filter's 3 baselines: choose them and scan the "
                         "folder (the last card on this tab) first.")
    if subject_only and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            what = (f"with SAM 3.1, {', '.join(sam['include'])}"
                    + (f" without {', '.join(sam['exclude'])}" if sam["exclude"] else "")) if sam is not None                 else automask.KINDS[kind].lower()
            _log(f"Making masks: {what}"
                 + (" - only your subject where there are several people" if subject_only else "")
                 + (" (replacing masks already there)" if overwrite else " (masks already there are kept)") + "\n\n")
            subject = det = None
            if subject_only:
                emb = face_embedder()
                vecs = [v for v in (emb.embed_pil(_open_rgb(b))[0] for b in bases) if v is not None]
                if not vecs:
                    raise ValueError("no face found in the Look Filter's baselines")
                subject, det = ((lambda crop: emb.embed_pil(crop)[0]), vecs), _face_detector()
            if sam is not None:
                from fizgig.dataset import sammask
                c = sammask.make_folder(folder, sam["include"], sam["exclude"], sam["path"], overwrite=overwrite,
                                        subject=subject, detector=det, log=_log)
            else:
                c = automask.make_folder(folder, kind, MODELS_DIR, overwrite=overwrite, subject=subject, detector=det,
                                         log=_log)
            _log(f"\nDone: {c['made']} mask(s) written"
                 + (f", {c['kept']} already there kept" if c["kept"] else "")
                 + (f", {c['no_person']} picture(s) with nothing to mask" if c["no_person"] else "")
                 + (f", {c['skipped']} skipped" if c["skipped"] else "")
                 + ". Check them on the Captions tab (Show mask); paint over any that are wrong.\n"
                 + ("To remake the ones kept, tick Replace masks already there and press Make masks again.\n"
                    if c["kept"] else ""))
            end = "done"
        except Exception as e:
            _log(f"\nERROR: making masks failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with _lock:
            _job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return flatten(im)
