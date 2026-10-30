"""
Face finding and identity embedding for Fizgig
==============================================

Two pluggable parts, both CPU-only (they never compete with the DiT for VRAM) and lazily loaded. The defaults are
InsightFace + ArcFace, exactly what Fizgig always used: the YOLO, Auto and DINOv2 choices are opt-in, picked on
the Image Prep tab ("Anime/furry/drawn characters"). docs/FACE_BACKENDS.md has the details.

FACE DETECTOR (FaceDetector) - where the faces are. prefs.json "face_detector":
  * "insightface" - InsightFace buffalo_l (RetinaFace + gender/age). Human photos; the only one with gender. The
                    default, and exactly the detection Fizgig always ran (one pass, boxes as InsightFace gives them).
  * "yolo"        - a YOLO face model run through onnxruntime (no ultralytics needed): an ADetailer-style .onnx
                    the user picks in "face_yolo_model" (recommended for furry datasets: Fdetailer v1.1, converted
                    once with src/fizgig/scripts/convert_yolo_to_onnx.py - see docs/FACE_BACKENDS.md). Blank =
                    deepghs' MIT anime/furry head detector, downloaded from Hugging Face on first use. Bbox and
                    segmentation exports both work (only the boxes are used); multi-class models (Fdetailer: face,
                    pawpads, ...) are filtered to the "face" class.
  * "auto"        - BOTH on every image (~0.13 s + ~0.17 s CPU), boxes merged: where a confident InsightFace face
                    overlaps a YOLO box the InsightFace box (with gender) wins, every other YOLO box is kept - so human
                    photos, furry / anime characters and mixed sets all just work.

IDENTITY EMBEDDER (FaceEmbedder) - how alike two faces are (Look Filter, likeness scoring, Repair Studio metrics,
Block Profiler). prefs.json "face_identity":
  * "arcface" - InsightFace's ArcFace (512-d). Human faces only (it needs InsightFace's own detection/landmarks).
                The default.
  * "dinov2"  - DINOv2-base (facebook/dinov2-base, Apache-2.0) on the detector's face crop. Any face: anime, furry,
                human. Needs only transformers (already a dependency).
  * "clip"    - removed (tested and dropped: DINOv2 separates characters better); a saved "clip" reads as "dinov2".
  * "auto"    - "yolo" -> DINOv2, "insightface" -> ArcFace. With the "auto" detector it is decided PER DATASET /
                scoring session (decide_look_match): ArcFace only when InsightFace finds a human face in most of the
                probed images and few images have only non-human (YOLO) faces; otherwise DINOv2 for the whole session.
                A session never mixes embedding spaces.

Every embedder carries its own verdict thresholds (IDENTITY_THRESHOLDS) - cosine similarities from different
embedders are on different scales and must never be compared with each other's cut-offs - and a name that cached
scores record, so a cache made by another embedder is recognised and recomputed.
"""

import ast
import json
import os
import threading
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

# Suppress TensorFlow/ONNX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fizgig import user_data
from fizgig.utils.upright import upright

_HERE = user_data.BASE_DIR          # the Fizgig folder: a relative face_yolo_model is relative to it
PREFS_FILE = user_data.PREFS_FILE

# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------
DETECTORS = ("auto", "insightface", "yolo")
IDENTITIES = ("auto", "arcface", "dinov2")
IDENTITY_RENAMED = {"clip": "dinov2"}   # retired look matches -> what they read as now (old prefs / env)
FACE_SETTING_DEFAULTS = {          # what people get today: InsightFace + ArcFace; the rest is opt-in
    "face_detector": "insightface",
    "face_yolo_model": "",          # path to a YOLO .onnx; blank = DEFAULT_YOLO (auto-download)
    "face_yolo_conf": 0.35,         # box confidence; Fdetailer's true faces score ~0.6-0.9, misfires < 0.4
    "face_identity": "arcface",
    "face_auto_threshold": 0.5,     # Auto look match: share of probed images with a human face needed for ArcFace
}
AUTO_HUMAN_SCORE = 0.6     # an InsightFace face counts as human at this confidence (fursuits / art rarely reach it)
AUTO_MIXED_SHARE = 0.25    # ...and ArcFace also needs fewer than this share of images with only non-human faces
AUTO_PROBE_MAX = 24        # images probed per dataset (spread evenly over the sorted list)
# The fallback YOLO model: deepghs' anime head detector (MIT), YOLOv8s ONNX, one class "head". Head boxes include
# ears / hair, so crops from it run a little larger than from a face model.
DEFAULT_YOLO = {"repo": "deepghs/anime_head_detection", "file": "head_detect_v2.0_s/model.onnx",
                "revision": "06604feee81983792a57c21081e539c0ae229833", "classes": ("head",),
                "label": "deepghs anime head v2.0 s (MIT)"}
RECOMMENDED_FURRY_MODEL = "Fdetailer v1.1 (Civitai 1228695), converted to ONNX"
# Where a converted Fdetailer is looked for when face_yolo_model is blank. Present -> it is the YOLO model; absent ->
# DEFAULT_YOLO. Fizgig never downloads it (Civitai needs a login).
LOCAL_YOLO_DEFAULT = os.path.join(user_data.MODELS_DIR, "face", "fdetailer_v11.onnx")
YOLO_FACE_CLASSES = ("face", "head")   # classes taken from a multi-class model (the first one present wins)


def load_face_settings(prefs: Optional[dict] = None) -> dict:
    """The face settings: defaults, then prefs.json (or the given prefs dict), then FIZGIG_FACE_DETECTOR /
    FIZGIG_FACE_YOLO_MODEL / FIZGIG_FACE_IDENTITY environment overrides (tests, headless runs)."""
    out = dict(FACE_SETTING_DEFAULTS)
    src = prefs
    if src is None:
        try:
            with open(PREFS_FILE, encoding="utf-8") as f:
                src = json.load(f)
        except Exception:
            src = {}
    for k in FACE_SETTING_DEFAULTS:
        if src.get(k) not in (None, ""):
            out[k] = src[k]
    for k, env in (("face_detector", "FIZGIG_FACE_DETECTOR"), ("face_yolo_model", "FIZGIG_FACE_YOLO_MODEL"),
                   ("face_identity", "FIZGIG_FACE_IDENTITY"), ("face_auto_threshold", "FIZGIG_FACE_AUTO_THRESHOLD")):
        if os.environ.get(env):
            out[k] = os.environ[env]
    out["face_detector"] = str(out["face_detector"]).lower()
    out["face_identity"] = str(out["face_identity"]).lower()
    out["face_identity"] = IDENTITY_RENAMED.get(out["face_identity"], out["face_identity"])
    if out["face_detector"] not in DETECTORS:
        out["face_detector"] = FACE_SETTING_DEFAULTS["face_detector"]
    if out["face_identity"] not in IDENTITIES:
        out["face_identity"] = FACE_SETTING_DEFAULTS["face_identity"]
    try:
        out["face_yolo_conf"] = float(out["face_yolo_conf"])
    except (TypeError, ValueError):
        out["face_yolo_conf"] = FACE_SETTING_DEFAULTS["face_yolo_conf"]
    try:
        out["face_auto_threshold"] = min(1.0, max(0.0, float(out["face_auto_threshold"])))
    except (TypeError, ValueError):
        out["face_auto_threshold"] = FACE_SETTING_DEFAULTS["face_auto_threshold"]
    out["face_yolo_model"] = str(out["face_yolo_model"] or "").strip().strip('"')
    if not out["face_yolo_model"] and os.path.isfile(LOCAL_YOLO_DEFAULT):
        out["face_yolo_model"] = LOCAL_YOLO_DEFAULT
    return out


def resolve_identity(settings: dict) -> str:
    """The concrete identity embedder for these settings ("auto" follows the detector)."""
    ident = settings.get("face_identity", FACE_SETTING_DEFAULTS["face_identity"])
    ident = IDENTITY_RENAMED.get(ident, ident)
    if ident != "auto":
        return ident
    return "dinov2" if settings.get("face_detector") == "yolo" else "arcface"


class FaceInfo(NamedTuple):
    """Information about a detected face"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    gender: str  # 'male', 'female' or 'unknown' (YOLO models have no gender)
    gender_score: float  # Confidence (0-1); the detection score
    area: int  # Face area in pixels
    center: Tuple[int, int]  # Center point (x, y)
    score: float = 1.0  # detection confidence
    source: str = "insightface"  # which backend found it


def _face_info(x1, y1, x2, y2, gender="unknown", score=1.0, source="insightface") -> FaceInfo:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return FaceInfo(bbox=(x1, y1, x2, y2), gender=gender, gender_score=float(score),
                    area=int(max(0, x2 - x1) * max(0, y2 - y1)), center=((x1 + x2) // 2, (y1 + y2) // 2),
                    score=float(score), source=source)


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """The BGR array InsightFace / OpenCV expect: the same pixels cv2.cvtColor's GRAY2BGR / RGBA2BGR / RGB2BGR give
    (a channel copy or swap, nothing rounded), done in numpy."""
    a = np.array(pil_image)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    return np.ascontiguousarray(a[..., 2::-1])


def _load_pil(path: str) -> Image.Image:
    """PIL, not cv2.imread: handles unicode paths on Windows."""
    with Image.open(path) as im:
        return upright(im).convert("RGB")


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _box_cover(inner, outer) -> float:
    """Share of `inner`'s area inside `outer` (a face box inside a YOLO head box -> ~1)."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def _same_face(a, b, iou=0.3, cover=0.6) -> bool:
    return _box_iou(a, b) >= iou or _box_cover(a, b) >= cover or _box_cover(b, a) >= cover


def merge_faces(insight: List["FaceInfo"], yolo: List["FaceInfo"], human_score: float = AUTO_HUMAN_SCORE):
    """Auto detector: one face list from both backends. A confident InsightFace face (a real human face, with gender)
    wins over any YOLO box on the same face; every other YOLO box is kept (furry / anime faces InsightFace can't
    see); a low-confidence InsightFace box that a YOLO box also covers is dropped (RetinaFace half-recognising a
    fursuit), elsewhere it is kept."""
    strong = [f for f in insight if f.score >= human_score]
    out = list(strong)
    out += [y for y in yolo if not any(_same_face(y.bbox, f.bbox) for f in strong)]
    out += [w for w in insight if w.score < human_score and not any(_same_face(w.bbox, f.bbox) for f in out)]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Detector backends
# ---------------------------------------------------------------------------------------------------------------
def _importable(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


class InsightFaceBackend:
    """InsightFace buffalo_l (RetinaFace detection + gender/age, or + ArcFace recognition), CPU."""

    name = "insightface"
    has_gender = True
    _shared: Dict[tuple, object] = {}
    _lock = threading.Lock()

    def __init__(self, modules=("detection", "genderage")):
        self.modules = tuple(modules)

    @staticmethod
    def available() -> bool:
        return _importable("insightface", "cv2", "onnxruntime")

    @property
    def app(self):
        """The FaceAnalysis app, shared per module set across the process (the model load is the slow part)."""
        with self._lock:
            app = self._shared.get(self.modules)
            if app is None:
                if not self.available():
                    raise ImportError("InsightFace is not installed. Run install_fizgig.py to set up face detection.")
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name='buffalo_l', allowed_modules=list(self.modules),
                                   providers=['CPUExecutionProvider'])
                app.prepare(ctx_id=-1)      # ctx_id=-1 forces CPU
                self._shared[self.modules] = app
            return app

    def load(self):
        _ = self.app

    def raw_faces(self, bgr):
        """InsightFace Face objects, retrying on a padded copy when none are found: RetinaFace misses faces that
        FILL the frame (a close-up is beyond its anchor scales) and a border brings them back into range. Returns
        (faces, (px, py)) - the padding to subtract from their boxes."""
        import cv2
        faces = self.app.get(bgr)
        if faces:
            return faces, (0, 0)
        h, w = bgr.shape[:2]
        for pad in (0.25, 0.5):
            py, px = int(h * pad), int(w * pad)
            faces = self.app.get(cv2.copyMakeBorder(bgr, py, py, px, px, cv2.BORDER_REPLICATE))
            if faces:
                return faces, (px, py)
        return [], (0, 0)

    def detect_bgr(self, bgr, pad_retry: bool = False) -> List[FaceInfo]:
        """FaceInfo for every face. pad_retry=False (the InsightFace finder) is the detection Fizgig always ran: one
        pass, the boxes as InsightFace gives them (truncated to ints, not clamped), gender_score = the detection
        score. pad_retry=True (Auto) also tries a padded copy when nothing is found, its boxes mapped back and
        clamped to the picture."""
        if pad_retry:
            faces, (px, py) = self.raw_faces(bgr)
        else:
            faces, (px, py) = self.app.get(bgr), (0, 0)
        h, w = bgr.shape[:2]
        out = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in np.asarray(f.bbox).astype(int))
            if (px, py) != (0, 0):
                x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 - px), min(h, y2 - py)
            g = getattr(f, "gender", None)
            score = float(getattr(f, "det_score", 1.0))
            gender = "unknown" if g is None else ("male" if int(g) == 1 else "female")
            info = _face_info(x1, y1, x2, y2, gender, score, self.name)
            out.append(info if g is not None else info._replace(gender_score=0.0))
        return out


def decode_yolo(outputs, conf: float, iou: float, class_ids, *, n_classes: Optional[int] = None):
    """Raw YOLO ONNX outputs -> [(x1, y1, x2, y2, score, class_id)] in the network's input pixels, after NMS.

    Handles ultralytics' layouts: detect (1, 4+nc, N), segment (1, 4+nc+nm, N) with a second "protos" output (the
    mask coefficients are ignored), either orientation, and end-to-end exports (1, K, 6) = x1 y1 x2 y2 score cls.
    `class_ids`: the classes kept (None = all). `n_classes`: from the model's metadata when known."""
    import cv2
    out = np.asarray(outputs[0])
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape {np.asarray(outputs[0]).shape}")
    # End-to-end exports (YOLOv10 / YOLO26 style) give (max_det <= 1000, 6); plain detect/segment heads give
    # (4+nc[+32], N) with N in the thousands, so a short 6-wide table is unambiguous.
    if out.shape[1] == 6 and 6 < out.shape[0] <= 1000:
        boxes, scores, cls = out[:, :4], out[:, 4], out[:, 5].astype(int)     # end-to-end (NMS inside the model)
        keep = scores >= conf
        if class_ids is not None:
            keep &= np.isin(cls, list(class_ids))
        return [(*map(float, b), float(s), int(c)) for b, s, c in zip(boxes[keep], scores[keep], cls[keep])]
    if out.shape[0] < out.shape[1]:
        out = out.T                                                    # -> (N, C)
    n_ch = out.shape[1] - 4
    if n_classes is None:
        n_classes = n_ch - 32 if len(outputs) > 1 and n_ch > 32 else n_ch
    probs = out[:, 4:4 + n_classes]
    if class_ids is not None:
        ids = [c for c in class_ids if 0 <= c < n_classes]
        if not ids:
            return []
        sub = probs[:, ids]
        best = sub.argmax(1)
        scores = sub[np.arange(len(sub)), best]
        cls = np.asarray(ids)[best]
    else:
        cls = probs.argmax(1)
        scores = probs[np.arange(len(probs)), cls]
    keep = scores >= conf
    if not keep.any():
        return []
    b, s, c = out[keep, :4], scores[keep], cls[keep]
    xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
    res = []
    for k in np.unique(c):                                            # NMS per class
        m = c == k
        idx = cv2.dnn.NMSBoxes(xywh[m].tolist(), s[m].tolist(), conf, iou)
        for i in np.array(idx).flatten():
            x, y, w, h = xywh[m][i]
            res.append((float(x), float(y), float(x + w), float(y + h), float(s[m][i]), int(k)))
    return sorted(res, key=lambda t: -t[4])


def letterbox(rgb: np.ndarray, size: int):
    """Resize keeping aspect into a size x size canvas (grey 114, centred, as ultralytics). Returns (canvas, ratio,
    (left, top))."""
    import cv2
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else rgb
    dh, dw = (size - nh) / 2, (size - nw) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))      # ultralytics LetterBox's rounding
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = im
    return canvas, r, (left, top)


def _parse_names(meta: dict) -> Optional[Dict[int, str]]:
    raw = (meta or {}).get("names")
    if not raw:
        return None
    try:
        d = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(d, (list, tuple)):
            d = dict(enumerate(d))
        return {int(k): str(v) for k, v in d.items()}
    except Exception:
        return None


class YoloOnnxBackend:
    """An ultralytics-trained YOLO face model exported to ONNX, run by onnxruntime on the CPU."""

    name = "yolo"
    has_gender = False
    _sessions: Dict[str, object] = {}
    _lock = threading.Lock()

    def __init__(self, model_path: str = "", conf: float = 0.35, iou: float = 0.5, classes=YOLO_FACE_CLASSES,
                 providers=("CPUExecutionProvider",)):
        self.model_path = (model_path or "").strip().strip('"')
        if self.model_path and not os.path.isabs(self.model_path):
            here = os.path.join(_HERE, self.model_path)           # relative = relative to the Fizgig folder
            if os.path.exists(here) or not os.path.exists(self.model_path):
                self.model_path = here
        self.conf, self.iou, self.classes, self.providers = float(conf), float(iou), tuple(classes), list(providers)
        self._sess = None
        self.names: Optional[Dict[int, str]] = None
        self.class_ids = None
        self.size = 640
        self.label = os.path.basename(self.model_path) if self.model_path else DEFAULT_YOLO["label"]

    def available(self) -> bool:
        if not _importable("onnxruntime", "cv2"):
            return False
        if self.model_path:
            return os.path.isfile(self.model_path) and self.model_path.lower().endswith(".onnx")
        return True                                    # the default downloads on first use

    def resolve_path(self) -> str:
        if self.model_path:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"face model not found: {self.model_path}")
            if self.model_path.lower().endswith(".pt"):
                raise ValueError("a .pt face model is a pickle and needs ultralytics to open - convert it once to "
                                 ".onnx with src/fizgig/scripts/convert_yolo_to_onnx.py (docs/FACE_BACKENDS.md)")
            return self.model_path
        from huggingface_hub import hf_hub_download
        kw = dict(repo_id=DEFAULT_YOLO["repo"], filename=DEFAULT_YOLO["file"], revision=DEFAULT_YOLO["revision"])
        try:
            return hf_hub_download(local_files_only=True, **kw)
        except Exception:
            from fizgig.utils import hf_public
            return hf_public.download(**kw)

    def load(self):
        if self._sess is not None:
            return
        path = self.resolve_path()
        with self._lock:
            sess = self._sessions.get(path)
            if sess is None:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.log_severity_level = 3
                sess = ort.InferenceSession(path, sess_options=so, providers=self.providers)
                self._sessions[path] = sess
        self._sess = sess
        inp = sess.get_inputs()[0]
        self._input = inp.name
        shp = inp.shape
        self.size = int(shp[2]) if len(shp) == 4 and isinstance(shp[2], int) else 640
        meta = {}
        try:
            meta = sess.get_modelmeta().custom_metadata_map
        except Exception:
            pass
        self.names = _parse_names(meta)
        if not self.model_path:
            self.names = self.names or {0: DEFAULT_YOLO["classes"][0]}
        self.class_ids = self._pick_classes()

    def _pick_classes(self):
        """The face class(es): the first of self.classes the model names; a one-class model is taken as is."""
        if not self.names:
            return None
        if len(self.names) == 1:
            return [0]
        low = {v.lower(): k for k, v in self.names.items()}
        for c in self.classes:
            if c.lower() in low:
                return [low[c.lower()]]
        raise ValueError(f"{self.label}: none of the classes {self.classes} in this model ({self.names})")

    def detect_rgb(self, rgb: np.ndarray) -> List[FaceInfo]:
        self.load()
        h, w = rgb.shape[:2]
        canvas, r, (left, top) = letterbox(rgb, self.size)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outs = self._sess.run(None, {self._input: x})
        boxes = decode_yolo(outs, self.conf, self.iou, self.class_ids,
                            n_classes=len(self.names) if self.names else None)
        res = []
        for x1, y1, x2, y2, s, _c in boxes:
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            res.append(_face_info(max(0, x1), max(0, y1), min(w, x2), min(h, y2), "unknown", s, self.name))
        return res

    def detect_bgr(self, bgr) -> List[FaceInfo]:
        return self.detect_rgb(np.ascontiguousarray(bgr[..., ::-1]))


# ---------------------------------------------------------------------------------------------------------------
# FaceDetector - the API every caller uses
# ---------------------------------------------------------------------------------------------------------------
_DETECT_CACHE: Dict[tuple, list] = {}
_DETECT_LOCK = threading.Lock()


class FaceDetector:
    """
    Face detector with a pluggable backend (InsightFace / YOLO via onnxruntime / auto), lazily loaded, CPU-only.

    Usage:
        detector = FaceDetector()                 # backend from prefs.json (Image Prep's face settings)
        faces = detector.detect_all("image.jpg")
        largest = detector.get_largest(faces)
        male_face = detector.get_largest_by_gender(faces, "male")   # InsightFace only; others fall back
    """

    def __init__(self, backend: Optional[str] = None, settings: Optional[dict] = None):
        self.settings = load_face_settings(settings)
        if backend:
            self.settings["face_detector"] = backend
        self.backend_name = self.settings["face_detector"]
        self._insight = InsightFaceBackend()
        self._yolo = YoloOnnxBackend(self.settings["face_yolo_model"], conf=self.settings["face_yolo_conf"])
        self._available = None
        self._yolo_broken = False
        self.last_source = None      # which backend(s) produced the last detection

    # -- identity of this configuration (callers rebuild the detector when it changes)
    @property
    def settings_key(self) -> tuple:
        return (self.backend_name, self.settings["face_yolo_model"], self.settings["face_yolo_conf"])

    def describe(self) -> str:
        if self.backend_name == "insightface":
            return "InsightFace (people)"
        if self.backend_name == "yolo":
            return f"YOLO face model ({self._yolo.label})"
        return f"Both: InsightFace and a YOLO face model ({self._yolo.label})"

    @property
    def has_gender(self) -> bool:
        """True when the backend can tell male from female (InsightFace; auto when InsightFace is installed)."""
        if self.backend_name == "yolo":
            return False
        return self._insight.available()

    @property
    def available(self) -> bool:
        if self._available is None:
            if self.backend_name == "insightface":
                self._available = self._insight.available()
            elif self.backend_name == "yolo":
                self._available = self._yolo.available()
            else:
                self._available = self._insight.available() or self._yolo.available()
        return self._available

    @property
    def _app(self):
        """Back-compat: the InsightFace app (None when the backend isn't InsightFace)."""
        return self._insight.app if self.backend_name != "yolo" and self._insight.available() else None

    def _ensure_loaded(self):
        if not self.available:
            if self.backend_name == "yolo":
                raise ImportError(f"YOLO face model unavailable ({self._yolo.model_path or DEFAULT_YOLO['label']}): "
                                  "needs onnxruntime + opencv and an existing .onnx file")
            raise ImportError("InsightFace is not installed. Run install_fizgig.py to set up face detection.")
        if self.backend_name in ("insightface", "auto") and self._insight.available():
            self._insight.load()
        if self.backend_name == "yolo":
            self._yolo.load()

    def detect_bgr(self, bgr) -> List[FaceInfo]:
        self._ensure_loaded()
        if self.backend_name == "yolo":
            self.last_source = "yolo"
            return self._yolo.detect_bgr(bgr)
        if self.backend_name == "insightface":
            self.last_source = "insightface"
            return self._insight.detect_bgr(bgr)          # one pass, as Fizgig always did
        # auto: both on every image, merged (InsightFace keeps human faces + gender, YOLO adds the rest)
        insight, yolo = [], []
        if self._insight.available():
            insight = self._insight.detect_bgr(bgr, pad_retry=True)
        if self._yolo.available() and not self._yolo_broken:
            try:
                yolo = self._yolo.detect_bgr(bgr)
            except Exception as e:  # noqa: BLE001 - a missing download must not break human-photo prep
                self._yolo_broken = True
                print(f"[faces] Auto: YOLO model unavailable ({e}) - InsightFace only this session")
        faces = merge_faces(insight, yolo)
        srcs = sorted({f.source for f in faces})
        self.last_source = "+".join(srcs) if srcs else None
        return faces

    def detect_all(self, image_path: str) -> List[FaceInfo]:
        """Detect all faces in an image file. Cached per (file, mtime, size, settings): the Auto look-match probe,
        the look match's crops and the crop tools share one detection per image."""
        try:
            st = os.stat(image_path)
            key = (os.path.abspath(image_path), st.st_mtime, st.st_size, self.settings_key)
        except OSError:
            key = None
        if key is not None:
            with _DETECT_LOCK:
                hit = _DETECT_CACHE.get(key)
            if hit is not None:
                return list(hit)
        if self.backend_name == "insightface":
            # read as Fizgig always read it for InsightFace (OpenCV, which also applies a JPEG's EXIF rotation), so
            # Test face detection shows exactly what it showed before
            import cv2
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            try:
                bgr = _pil_to_bgr(_load_pil(image_path))
            except Exception as e:
                raise ValueError(f"Could not load image: {image_path} ({e})")
        faces = self.detect_bgr(bgr)
        if key is not None:
            with _DETECT_LOCK:
                _DETECT_CACHE[key] = list(faces)
                while len(_DETECT_CACHE) > 8192:
                    _DETECT_CACHE.pop(next(iter(_DETECT_CACHE)))
        return faces

    def detect_from_pil(self, pil_image: Image.Image) -> List[FaceInfo]:
        """Detect faces from a PIL Image object."""
        return self.detect_bgr(_pil_to_bgr(pil_image))

    def get_largest(self, faces: List[FaceInfo]) -> Optional[FaceInfo]:
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    def get_largest_by_gender(self, faces: List[FaceInfo], gender: str,
                              fallback_to_any: bool = True) -> Optional[FaceInfo]:
        """Largest face of a gender; with fallback_to_any, the largest face when none matches (always the case for
        YOLO faces, which have gender 'unknown')."""
        gender = gender.lower()
        matching = [f for f in faces if f.gender == gender]
        if matching:
            return max(matching, key=lambda f: f.area)
        if fallback_to_any and faces:
            return self.get_largest(faces)
        return None


def crop_to_face(
    image: Image.Image,
    face: FaceInfo,
    padding_percent: float = 20.0,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Crop an image to a detected face with padding.

    Args:
        image: PIL Image to crop
        face: FaceInfo object with bounding box
        padding_percent: Extra space around face as percentage (0-100)
        maintain_aspect: If True, make the crop region square

    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = face.bbox
    face_w = x2 - x1
    face_h = y2 - y1
    pad_w = int(face_w * padding_percent / 100)
    pad_h = int(face_h * padding_percent / 100)
    crop_x1 = x1 - pad_w
    crop_y1 = y1 - pad_h
    crop_x2 = x2 + pad_w
    crop_y2 = y2 + pad_h
    if maintain_aspect:
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w > crop_h:
            diff = crop_w - crop_h
            crop_y1 -= diff // 2
            crop_y2 += diff - diff // 2
        elif crop_h > crop_w:
            diff = crop_h - crop_w
            crop_x1 -= diff // 2
            crop_x2 += diff - diff // 2
    img_w, img_h = image.size
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def draw_face_boxes(
    image: Image.Image,
    faces: List[FaceInfo],
    highlight_index: Optional[int] = None
) -> Image.Image:
    """
    Draw bounding boxes on faces for preview (green = the one that would be used, yellow = the others). Labels show
    gender (M / F; ? for YOLO faces) and the detection score.
    """
    from PIL import ImageDraw, ImageFont

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        if highlight_index is not None and i == highlight_index:
            color, width = (0, 255, 0), 3
        else:
            color, width = (255, 255, 0), 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        gender_char = 'M' if face.gender == 'male' else 'F' if face.gender == 'female' else '?'
        label = f"{gender_char} ({face.area:,}px)"
        if face.source != "insightface":        # a YOLO face: its score instead of a gender to go by
            label = f"{gender_char} {face.score:.2f} ({face.area:,}px)"
        label_y = y1 - 20 if y1 > 20 else y2 + 5
        bbox = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((x1, label_y), label, fill=color, font=font)
    return preview


# ---------------------------------------------------------------------------------------------------------------
# Identity embedders
# ---------------------------------------------------------------------------------------------------------------
# Verdict thresholds per embedder (cosine similarity of a face to the baseline / reference):
#   match       >= "match":       solid match to the baseline look
#   borderline  >= "borderline":  same-subject territory, drifting
#   drift       below:            likely off-look / a different subject
#   floor:      the Look Filter's outlier cut-off never goes below it (a different subject rarely clears it)
# ArcFace: InsightFace's conventions (same person across varied photos 0.30-0.70 vs one baseline; a different
# person rarely clears 0.25). DINOv2: calibrated on face/head crops of anime, furry and human images
# (docs/FACE_BACKENDS.md has the numbers) - provisional, the Look Filter's dataset-relative cut-off
# (median - 1.5 IQR) does most of the work.
IDENTITY_THRESHOLDS = {
    "arcface": {"match": 0.45, "borderline": 0.30, "floor": 0.25},
    "dinov2": {"match": 0.62, "borderline": 0.48, "floor": 0.40},
}
IDENTITY_LABELS = {"arcface": "ArcFace (people)", "dinov2": "DINOv2 (any face)"}
CROP_MARGIN = 0.25      # the crop an embedder sees: the face box grown 25% each side (ears, muzzle, markings)


def same_embedder(recorded: Optional[str], cache_id: str) -> bool:
    """Whether cached scores recorded with embedder `recorded` can be reused under `cache_id`. Files written
    before the face backends existed carry no embedder and were ArcFace."""
    return (recorded or "arcface") == cache_id


def verdict_for(sim: Optional[float], thresholds: dict) -> str:
    """'no face' | 'match' | 'borderline' | 'drift' for a similarity under an embedder's thresholds."""
    if sim is None or sim != sim:
        return "no face"
    if sim >= thresholds["match"]:
        return "match"
    if sim >= thresholds["borderline"]:
        return "borderline"
    return "drift"


def look_cutoff(scores: List[float], floor: float) -> Optional[float]:
    """The Look Filter's low-outlier fence: median - 1.5 IQR, never below the embedder's floor. None for < 4."""
    ss = sorted(s for s in scores if s is not None)
    if len(ss) < 4:
        return None
    n = len(ss)
    med, q1, q3 = ss[n // 2], ss[n // 4], ss[(3 * n) // 4]
    return max(med - 1.5 * (q3 - q1), floor)


class _ArcFace:
    name = "arcface"

    def __init__(self, settings):
        self.backend = InsightFaceBackend(modules=("detection", "recognition"))

    def available(self) -> bool:
        return self.backend.available()

    def embed_bgr(self, bgr):
        faces, pad = self.backend.raw_faces(bgr)
        if not faces:
            return None, None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            return None, None
        bbox = None
        if pad == (0, 0):     # a padded detection's box is in padded-image space: embedding only
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        return np.asarray(emb, dtype=np.float32), bbox


class _CropEmbedder:
    """A general image model on the detector's face crop."""
    name = "crop"
    model_id = ""
    revision = None
    _models: Dict[str, tuple] = {}
    _lock = threading.Lock()

    def __init__(self, settings):
        self.detector = FaceDetector(settings=settings)

    def available(self) -> bool:
        return _importable("transformers", "torch") and self.detector.available

    def _model(self):
        with self._lock:
            m = self._models.get(self.model_id)
            if m is None:
                m = self._load()
                self._models[self.model_id] = m
            return m

    def _load(self):
        raise NotImplementedError

    def _features(self, crop: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def crop(pil: Image.Image, bbox, margin=CROP_MARGIN) -> Image.Image:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        s = max(w, h) * (1 + 2 * margin)                  # square, centred on the face
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        W, H = pil.size
        box = (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2)))
        return pil.crop(box)

    def embed_path(self, path):
        faces = self.detector.detect_all(path)
        f = self.detector.get_largest(faces)
        if f is None:
            return None, None
        return self._embed_face(_load_pil(path), f)

    def embed_bgr(self, bgr):
        faces = self.detector.detect_bgr(bgr)
        f = self.detector.get_largest(faces)
        if f is None:
            return None, None
        return self._embed_face(Image.fromarray(np.ascontiguousarray(bgr[..., ::-1])), f)

    def _embed_face(self, pil, f):
        v = self._features(self.crop(pil, f.bbox)).astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not n or n != n:
            return None, None
        return v / n, tuple(int(c) for c in f.bbox)


def _preprocess(img: Image.Image, short: int, size: int, mean, std) -> np.ndarray:
    """The models' own preprocessing (shortest side -> `short`, bicubic; centre crop `size`; normalise), done here so
    no image-processor class (whose fast variants need torchvision on newer transformers) is involved."""
    img = img.convert("RGB")
    w, h = img.size
    r = short / min(w, h)
    img = img.resize((max(size, round(w * r)), max(size, round(h * r))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    a = np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0
    a = (a - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return a.transpose(2, 0, 1)[None].copy()


def _hf_pretrained(cls, model_id, revision):
    """Cache-first load (works offline once downloaded), online otherwise."""
    try:
        return cls.from_pretrained(model_id, revision=revision, local_files_only=True)
    except Exception:
        return cls.from_pretrained(model_id, revision=revision)


class _DinoV2(_CropEmbedder):
    name = "dinov2"
    model_id = "facebook/dinov2-base"
    revision = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

    def _load(self):
        import torch
        from transformers import AutoModel
        model = _hf_pretrained(AutoModel, self.model_id, self.revision).to("cpu", torch.float32).eval()
        return None, model

    def _features(self, crop):
        import torch
        _, model = self._model()
        x = _preprocess(crop, 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(x))
        return out.pooler_output[0].float().numpy()       # CLS token after the final layernorm


_EMBEDDERS = {"arcface": _ArcFace, "dinov2": _DinoV2}


class FaceEmbedder:
    """
    Identity embedding of the LARGEST face in an image, with a pluggable backend (ArcFace / DINOv2), for
    look-consistency and likeness scoring. Embeddings are L2-normalised, so cosine similarity is a dot product -
    but only between embeddings of the SAME embedder (compare `name`).

    Usage:
        embedder = FaceEmbedder()                 # backend from prefs.json
        emb = embedder.embed("image.png")         # or None when no face
        emb, bbox = embedder.embed_pil(pil)       # in-memory renders
        embedder.verdict(float(np.dot(a, b)))     # per-embedder thresholds
    """

    def __init__(self, backend: Optional[str] = None, settings: Optional[dict] = None):
        self.settings = load_face_settings(settings)
        if backend:
            self.settings["face_identity"] = backend
        self.name = resolve_identity(self.settings)
        self._impl = _EMBEDDERS[self.name](self.settings)
        self.thresholds = dict(IDENTITY_THRESHOLDS[self.name])
        self.label = IDENTITY_LABELS[self.name]
        self._lock = threading.Lock()

    @property
    def settings_key(self) -> tuple:
        """Changes whenever an embedding would come out different (embedder, or the detector its crops come from)."""
        if self.name == "arcface":
            return (self.name,)
        return (self.name, self.settings["face_detector"], self.settings["face_yolo_model"],
                self.settings["face_yolo_conf"])

    @property
    def cache_id(self) -> str:
        """What cached scores record as their embedder (Look Filter / gallery JSONs)."""
        return self.name if self.name == "arcface" else f"{self.name}@{self.settings['face_detector']}:" \
            f"{os.path.basename(self.settings['face_yolo_model']) or 'default'}"

    @property
    def available(self) -> bool:
        return self._impl.available()

    def _ensure_loaded(self):
        if not self.available:
            raise ImportError(f"{self.label} is unavailable (missing InsightFace / onnxruntime / transformers or the "
                              f"face model). Run install_fizgig.py.")

    def embed_pil(self, pil: Image.Image):
        """(embedding, bbox) of the largest face; bbox None when unknown (a padded InsightFace retry), (None, None)
        when there is no face."""
        self._ensure_loaded()
        with self._lock:
            return self._impl.embed_bgr(_pil_to_bgr(pil.convert("RGB")))

    def embed_bgr(self, bgr):
        """(embedding, bbox) of the largest face in a BGR array, as embed_pil."""
        self._ensure_loaded()
        with self._lock:
            return self._impl.embed_bgr(bgr)

    def embed_image(self, pil: Image.Image) -> Optional[np.ndarray]:
        return self.embed_pil(pil)[0]

    def embed(self, image_path: str) -> Optional[np.ndarray]:
        """Embedding of the LARGEST detected face in the image file, or None if no face."""
        if hasattr(self._impl, "embed_path"):
            self._ensure_loaded()
            with self._lock:
                return self._impl.embed_path(image_path)[0]
        return self.embed_pil(_load_pil(image_path))[0]

    def verdict(self, sim: Optional[float]) -> str:
        return verdict_for(sim, self.thresholds)

    def cutoff(self, scores: List[float]) -> Optional[float]:
        return look_cutoff(scores, self.thresholds["floor"])


# ---------------------------------------------------------------------------------------------------------------
# Auto look match: one embedder per dataset / scoring session
# ---------------------------------------------------------------------------------------------------------------
class LookMatchDecision(NamedTuple):
    identity: str            # "arcface" | "dinov2"
    kind: str                # "human" | "furry" | "mixed" | "explicit" | "empty"
    human: int               # probed images with a confident InsightFace (human) face
    yolo_only: int           # probed images whose only faces came from the YOLO model
    probed: int
    settings: dict           # the session's settings: face_identity resolved, detector unchanged
    message: str

    def embedder(self) -> "FaceEmbedder":
        return FaceEmbedder(settings=self.settings)


_DECISIONS: Dict[tuple, LookMatchDecision] = {}
_DECISION_LOCK = threading.Lock()
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def probe_sample(paths, max_n: int = AUTO_PROBE_MAX) -> List[str]:
    """Up to max_n images spread evenly over the (sorted, de-duplicated) list - deterministic for a given folder."""
    from fizgig.dataset.masks import is_mask_file
    ps = sorted({os.path.normpath(p) for p in paths if str(p).lower().endswith(IMAGE_EXTS) and os.path.isfile(p)
                 and not is_mask_file(p)})
    if len(ps) <= max_n:
        return ps
    return [ps[round(i * (len(ps) - 1) / (max_n - 1))] for i in range(max_n)]


def decide_look_match(paths, settings: Optional[dict] = None, *, max_n: int = AUTO_PROBE_MAX, log=print,
                      detector: Optional["FaceDetector"] = None) -> LookMatchDecision:
    """Pick the look-match embedder for a dataset / scoring session (the Look Filter's folder, or the reference
    photos of a likeness session). Only "auto" + "auto" probes; an explicit face_identity, or an explicit detector
    with an auto identity, is used as is.

    Probe: the Auto detector (InsightFace + YOLO, merged) on up to max_n images. ArcFace when at least
    face_auto_threshold of them (default 50%) have a confident human face AND fewer than AUTO_MIXED_SHARE (25%) have
    only non-human faces; otherwise DINOv2 for the whole session - for a furry / anime set, and for a mixed one,
    since ArcFace can't embed a furry face and two embedders' scores can't be compared. Cached per (images, mtimes,
    settings) - the same folder decides the same way every time; detections are cached for the crops that follow."""
    st = load_face_settings(settings)
    if st["face_identity"] != "auto" or st["face_detector"] != "auto":
        ident = resolve_identity(st)
        return LookMatchDecision(ident, "explicit", 0, 0, 0, dict(st, face_identity=ident),
                                 f"[faces] Look match: {IDENTITY_LABELS[ident]} (set explicitly)")
    sample = probe_sample(paths, max_n)
    try:
        sig = tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in sample)
    except OSError:
        sig = tuple(sample)
    key = (sig, st["face_auto_threshold"], st["face_yolo_model"], st["face_yolo_conf"])
    with _DECISION_LOCK:
        hit = _DECISIONS.get(key)
    if hit is not None:
        if log:
            log(hit.message + " (cached)")
        return hit
    if not sample:
        d = LookMatchDecision("arcface", "empty", 0, 0, 0, dict(st, face_identity="arcface"),
                              "[faces] Auto look match: no images to probe - ArcFace")
        if log:
            log(d.message)
        return d
    det = detector or FaceDetector(backend="auto", settings=st)
    human = yolo_only = probed = 0
    for p in sample:
        try:
            faces = det.detect_all(p)
        except Exception:      # unreadable image: not part of the vote
            continue
        probed += 1
        if any(f.source == "insightface" and f.score >= AUTO_HUMAN_SCORE for f in faces):
            human += 1
        elif any(f.source == "yolo" for f in faces):
            yolo_only += 1
    thr = st["face_auto_threshold"]
    n = max(1, probed)
    if not InsightFaceBackend.available():
        kind, ident, why = "furry", "dinov2", "InsightFace isn't installed"
    elif human / n >= thr and yolo_only / n < AUTO_MIXED_SHARE:
        kind, ident, why = "human", "arcface", "a set of people"
    elif human / n >= thr or (human / n >= AUTO_MIXED_SHARE and yolo_only / n >= AUTO_MIXED_SHARE):
        kind, ident, why = "mixed", "dinov2", "people and characters mixed, so one look match for all"
    elif yolo_only / n >= AUTO_MIXED_SHARE:
        kind, ident, why = "furry", "dinov2", "anime, furry or drawn characters"
    else:
        kind, ident, why = "furry", "dinov2", "too few people's faces"
    msg = (f"[faces] Look match for this set: {IDENTITY_LABELS[ident]}, {why} (a person's face in {human}/{probed} "
           f"pictures looked at, only a character's face in {yolo_only}/{probed})")
    d = LookMatchDecision(ident, kind, human, yolo_only, probed, dict(st, face_identity=ident), msg)
    with _DECISION_LOCK:
        _DECISIONS[key] = d
    if log:
        log(msg)
    return d


def session_settings(paths, settings: Optional[dict] = None, log=print) -> dict:
    """Settings for one scoring session (face_identity resolved by decide_look_match)."""
    return decide_look_match(paths, settings, log=log).settings


def is_face_detection_available() -> bool:
    """True when some face detector can run: InsightFace, or onnxruntime + OpenCV for a YOLO model."""
    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def is_insightface_available() -> bool:
    return InsightFaceBackend.available()
