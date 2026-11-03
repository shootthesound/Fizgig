"""Score renders before one is used: likeness to the training pictures (ArcFace), eyes open (the 106-point
landmarks), head turn (the 3D landmarks' pose), face size, sharpness. All ONNX models (InsightFace buffalo_l) run by
onnxruntime: on the CPU here, on a Ryzen AI NPU's execution provider when one is installed. Scores rule out gross
failures (closed eyes, a stranger, a face turned away); between the pictures that pass, the eye decides."""
import os
import threading
import time

import numpy as np

from fizgig.bridge.mcp import ToolError

MODULES = ["detection", "recognition", "genderage", "landmark_2d_106", "landmark_3d_68"]
NPU = "VitisAIExecutionProvider"            # Ryzen AI
PICTURES = (".png", ".jpg", ".jpeg", ".webp")
EYES_OPEN = 0.20                            # eye height / width below this: closed or half-closed
TURNED = 35                                 # degrees of yaw or pitch beyond this: turned away
MAIN = 0.35                                 # another face this size of the main one: two people, not a crowd behind
LEFT_EYE, RIGHT_EYE = range(33, 43), range(87, 97)

_lock = threading.Lock()
_app = {}
_refs = {}                                  # folder -> (mtime, mean embedding, how many faces)


def _analysis():
    with _lock:
        if not _app:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            have = ort.get_available_providers()
            providers = ([NPU] if NPU in have else []) + ["CPUExecutionProvider"]
            t = time.time()
            app = FaceAnalysis(name="buffalo_l", allowed_modules=MODULES, providers=providers)
            app.prepare(ctx_id=-1)
            _app.update(app=app, provider=providers[0], load=round(time.time() - t, 2))
        return _app["app"]


def _read(path):
    import cv2
    from PIL import Image
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _faces(bgr):
    faces = _analysis().get(bgr)
    return sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _eye(pts):
    w = pts[:, 0].max() - pts[:, 0].min()
    return float((pts[:, 1].max() - pts[:, 1].min()) / max(w, 1e-6))


def reference(folder, most=8):
    """The mean ArcFace embedding of the largest face in up to `most` of the folder's pictures."""
    if not os.path.isdir(folder):
        raise ToolError(f"{folder} isn't a folder.")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(PICTURES)
                   and not n.lower().endswith("_mask.png"))
    if not names:
        raise ToolError(f"No pictures in {folder}.")
    stamp = max(os.path.getmtime(os.path.join(folder, n)) for n in names)
    hit = _refs.get(folder)
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    step = max(1, len(names) // most)
    embs = []
    for n in names[::step][:most]:
        f = _faces(_read(os.path.join(folder, n)))
        if f:
            embs.append(f[0].normed_embedding)
    if not embs:
        raise ToolError(f"No faces found in {folder}'s pictures.")
    m = np.mean(embs, axis=0)
    m = m / np.linalg.norm(m)
    _refs[folder] = (stamp, m, len(embs))
    return m, len(embs)


def score(paths, likeness_folder=""):
    """Each picture's scores and a pick: the best likeness among those with one clear face, eyes open and facing
    roughly forward (else the best likeness of all, flagged)."""
    import cv2
    paths = [p for p in paths if p]
    if not paths:
        raise ToolError("No pictures to score.")
    for p in paths:
        if not os.path.isfile(p):
            raise ToolError(f"{p} isn't there.")
    _analysis()
    t0 = time.time()
    ref, n_ref = reference(likeness_folder) if likeness_folder else (None, 0)
    t1 = time.time()
    out = []
    for p in paths:
        bgr = _read(p)
        faces = _faces(bgr)
        row = {"path": p, "faces": len(faces)}
        if faces:
            f = faces[0]
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            lm = f.landmark_2d_106
            eyes = min(_eye(lm[list(LEFT_EYE)]), _eye(lm[list(RIGHT_EYE)]))
            pitch, yaw, roll = [float(v) for v in f.pose]
            crop = cv2.cvtColor(bgr[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
            row.update(
                likeness=round(float(np.dot(f.normed_embedding, ref)), 3) if ref is not None else None,
                eyes_open=round(eyes, 3), eyes_ok=eyes >= EYES_OPEN,
                yaw=round(yaw, 1), pitch=round(pitch, 1), facing_ok=abs(yaw) <= TURNED and abs(pitch) <= TURNED,
                face_share=round((x2 - x1) * (y2 - y1) / float(w * h), 3),
                sharpness=round(float(cv2.Laplacian(crop, cv2.CV_64F).var()), 1) if crop.size else 0.0,
                age=int(f.age), sex=f.sex)
            area = lambda g: (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1])
            row["one_main_face"] = len(faces) == 1 or area(faces[1]) < MAIN * area(f)   # a crowd behind is fine
            row["ok"] = row["eyes_ok"] and row["facing_ok"] and row["one_main_face"]
        else:
            row["ok"] = False
        out.append(row)
    t2 = time.time()
    key = (lambda r: r.get("likeness") or 0) if ref is not None else (lambda r: r.get("sharpness") or 0)
    good = [r for r in out if r["ok"]]
    pool = good or [r for r in out if r["faces"]]
    best = max(pool, key=key) if pool else None
    return {"pictures": out,
            "pick": best["path"] if best else None,
            "pick_passed_checks": bool(good),
            "reference_faces": n_ref,
            "ran_on": _app["provider"],
            "seconds": {"model_load_once": _app["load"], "reference": round(t1 - t0, 2),
                        "per_picture": round((t2 - t1) / len(paths), 3)}}
