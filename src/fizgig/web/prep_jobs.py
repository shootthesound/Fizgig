"""The web UI's Image Prep: a prep run (resize / PNG / face close-ups) and the Look Consistency Filter's scan, each
in a background thread on the Fizgig machine, with their state for the page to poll. The work itself is
fizgig/dataset/prep.py - the same code the desktop app's Image Prep tab runs. Face tools run on the CPU, with the face
finder and look match Image Prep's face settings pick (face_utils: InsightFace / YOLO / Auto, ArcFace / DINOv2 / CLIP).
"""
import base64
import io
import json
import os
import threading
import time

from fizgig.dataset import prep

_lock = threading.Lock()
_job = {"state": "idle", "log": "", "folder": "", "started": 0.0}
_look = {"state": "idle", "folder": "", "baselines": [], "scores": {}, "progress": [0, 0], "error": "",
         "embedder": "", "thresholds": None, "decision": ""}
_embed_cache = {}             # (path, mtime, embedder settings) -> embedding or None
_detector = None
_embedders = {}               # FaceEmbedder.settings_key -> FaceEmbedder (one per look match in use)
_model_lock = threading.Lock()


def faces_available() -> bool:
    try:
        from fizgig.face_utils import is_face_detection_available
        return bool(is_face_detection_available())
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
    return face_embedder(d.settings), d.message


FACE_DETECTOR_LABELS = {"auto": "Auto (InsightFace + YOLO)", "insightface": "InsightFace (people)",
                        "yolo": "YOLO model (furry / anime)"}
FACE_IDENTITY_LABELS = {"auto": "Auto (decided per dataset)", "arcface": "ArcFace (people)",
                        "dinov2": "DINOv2 (any face)", "clip": "CLIP (any face)"}


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
            "detectors": [[k, FACE_DETECTOR_LABELS[k]] for k in F.DETECTORS],
            "identities": [[k, FACE_IDENTITY_LABELS[k]] for k in F.IDENTITIES],
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
            raise ValueError("The YOLO model must be an .onnx file on the Fizgig computer (convert a .pt with "
                             "src/fizgig/scripts/convert_yolo_to_onnx.py).")
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


def start(folder, mode, mp, face_label, padding, replace):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if mode not in prep.MODES:
        raise ValueError("Pick what to do.")
    if mode != "Resize Only" and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            det = _face_detector() if mode != "Resize Only" else None
            prep.Prep(_log, det).run(mode, folder, float(mp), face_label, float(padding), bool(replace))
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
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        shown = draw_face_boxes(img, faces, highlight_index=hi).convert("RGB")
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
        with Image.open(path) as img:
            return emb.embed_pil(img.convert("RGB"))
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
