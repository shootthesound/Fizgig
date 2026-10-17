"""Face-likeness scoring for LoRA Royale.

Given a reference (training) image and a set of rendered epoch images, score how closely each render's face matches
the reference. The identity embedder is face_utils.FaceEmbedder - ArcFace for human faces, DINOv2 (or CLIP) on the
face crop for furry / anime faces - picked in Image Prep's face settings (prefs.json face_detector /
face_identity). Higher cosine similarity = more alike; scores are only comparable within one embedder.

CPU-only, so it never fights the DiT for VRAM.
"""

import os
import threading
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_embedder = None
_lock = threading.Lock()
last_embedder_label = ""      # the embedder the last score_renders() session used (shown beside the scores)


def get_embedder(settings: Optional[dict] = None):
    """The process-wide FaceEmbedder for the given face settings (default: prefs.json); rebuilt when they change."""
    global _embedder
    from fizgig.face_utils import FaceEmbedder, load_face_settings
    with _lock:
        fresh = FaceEmbedder(settings=load_face_settings(settings))
        if _embedder is None or _embedder.settings_key != fresh.settings_key:
            _embedder = fresh
        return _embedder


def available(settings: Optional[dict] = None) -> bool:
    """True if the configured identity embedder can run (InsightFace for ArcFace; a face detector + transformers for
    DINOv2 / CLIP)."""
    try:
        return get_embedder(settings).available
    except Exception:
        return False


def embedder_name(settings: Optional[dict] = None) -> str:
    try:
        return get_embedder(settings).label
    except Exception:
        return "face embedder"


def embedding_for_path(path: str) -> Optional[np.ndarray]:
    try:
        return get_embedder().embed(path)
    except Exception:
        return None


def embedding_for_image(pil_image) -> Optional[np.ndarray]:
    try:
        return get_embedder().embed_image(pil_image)
    except Exception:
        return None


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity of two normed embeddings (dot product). NaN if either
    is missing (no face detected)."""
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b))


def score_renders(ref_path: str, renders: List[Tuple], settings: Optional[dict] = None) -> List[Tuple[object, float]]:
    """`renders`: list of (label, PIL.Image). Returns [(label, score)] with
    score in [-1, 1] (NaN where the reference or a render has no detectable
    face). Reference is embedded once; one embedder for all of them - with Auto look match, decided from the
    reference photo (ArcFace for a human face, DINOv2 otherwise)."""
    global last_embedder_label
    from fizgig.face_utils import decide_look_match
    dec = decide_look_match([ref_path], settings)
    emb = get_embedder(dec.settings)
    last_embedder_label = emb.label

    def _e(fn, x):
        try:
            return fn(x)
        except Exception:
            return None
    ref = _e(emb.embed, ref_path)
    return [(label, cosine(ref, _e(emb.embed_image, img))) for label, img in renders]
