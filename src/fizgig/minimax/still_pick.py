"""Pick the frame of a clip to train as a still: the sharpest one that shows a face.

Used at CACHE time (see caching._encode_clip): the chosen frame is encoded once as an ordinary
still and stored beside the clip's own latent, so the training step that uses it loads a
finished latent and nothing is re-encoded. Frame 0 is the fallback whenever no frame shows a
face (or face detection is unavailable) — the causal VAE makes frame 0 the one frame that could
also be sliced straight out of the clip latent, which is what the feature did before this.

Scoring: variance of the Laplacian (a plain focus / motion-blur measure) on the FACE crop, so
subject motion blur is what decides, not background texture. Candidates are visited in order
of whole-frame sharpness and detection stops after `max_face_candidates` faces have been found
(typically the first few frames tried), so a 22-frame clip costs a handful of CPU detections.
"""
import logging
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_DETECTOR = None
_DETECTOR_FAILED = False


def _detector():
    """The app's insightface detector (CPU), loaded once; None when it can't be."""
    global _DETECTOR, _DETECTOR_FAILED
    if _DETECTOR is not None or _DETECTOR_FAILED:
        return _DETECTOR
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from face_utils import FaceDetector
        det = FaceDetector()
        if not det.available:
            raise ImportError("insightface not installed")
        det._ensure_loaded()
        _DETECTOR = det
    except Exception as e:  # noqa: BLE001 — any failure means "no face gate", not a crash
        _DETECTOR_FAILED = True
        logger.warning("[still] face detection unavailable (%s): clip stills fall back to "
                       "frame 0", e)
    return _DETECTOR


def laplacian_variance(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detect(det, frame_rgb: np.ndarray):
    """Largest face in the frame, retrying on a padded copy when none is found: RetinaFace
    misses faces that FILL the frame (a close-up is beyond its anchor scales), and a border
    brings it back into range — the Look Filter's fix (face_utils._detect_with_pad_retry).
    The bbox comes back in the frame's own coordinates."""
    import cv2
    from PIL import Image
    faces = det.detect_from_pil(Image.fromarray(frame_rgb))
    face = det.get_largest(faces)
    if face is not None:
        return face
    h, w = frame_rgb.shape[:2]
    for pad in (0.25, 0.5):
        py, px = int(h * pad), int(w * pad)
        padded = cv2.copyMakeBorder(frame_rgb, py, py, px, px, cv2.BORDER_REPLICATE)
        face = det.get_largest(det.detect_from_pil(Image.fromarray(padded)))
        if face is not None:
            x1, y1, x2, y2 = face.bbox
            return face._replace(bbox=(x1 - px, y1 - py, x2 - px, y2 - py))
    return None


def pick_still_frame(frames: Union[np.ndarray, List[np.ndarray]],
                     max_face_candidates: int = 6,
                     min_face_frac: float = 0.08,
                     detector=None) -> Tuple[int, dict]:
    """frames: (T, H, W, 3) uint8 RGB (or a list of such frames). Returns (index, info).

    info: {"face": bool, "score": face-crop Laplacian variance of the pick (or the whole-frame
    one when no face), "detections": how many frames were run through the detector,
    "candidates": how many frames with a usable face were scored}.
    """
    import cv2
    if isinstance(frames, list):
        frames = np.stack(frames)
    T, H, W = frames.shape[:3]
    if T <= 1:
        return 0, {"face": False, "score": 0.0, "detections": 0, "candidates": 0}

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    global_sharp = np.array([laplacian_variance(g) for g in grays])
    order = np.argsort(-global_sharp)                        # sharpest whole frame first

    det = detector if detector is not None else _detector()
    if det is None:
        return 0, {"face": False, "score": float(global_sharp[0]), "detections": 0, "candidates": 0}

    scored = []                                              # (face-crop score, index)
    detections = 0
    min_h = min_face_frac * H
    for i in order:
        i = int(i)
        face = _detect(det, frames[i])
        detections += 1
        if face is None:
            continue
        x1, y1, x2, y2 = face.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (y2 - y1) < min_h:
            continue                                         # too small to count as "visible"
        mx, my = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
        crop = grays[i][max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]
        if crop.size < 64:
            continue
        scored.append((laplacian_variance(crop), i))
        if len(scored) >= max_face_candidates:
            break
    if not scored:
        return 0, {"face": False, "score": float(global_sharp[0]), "detections": detections,
                   "candidates": 0}
    score, idx = max(scored)
    return idx, {"face": True, "score": float(score), "detections": detections,
                 "candidates": len(scored)}
