"""Overbake metrics for the Repair Studio pop-out — pure numpy/cv2, no GUI, no models.

The pop-out compares two renders of the SAME seed and prompt, differing only in the LoRA
blocks — the one setting where no-reference image metrics are honest: absolute quality scores
on a single image are noise, but the DELTA across a controlled pair is signal. Everything here
is cheap CPU work (<100 ms for a 768 pair); face likeness stays with the caller beside the
shared face embedder (face_utils.FaceEmbedder).

The metrics target overbake's known signatures:
  * patch grid       — what reads as "JPEG blockiness" in a fried render is usually the
                       model's own patch lattice showing through, and we know the pitch:
                       16 px for Klein and Krea 2 (8x VAE x 2x2 patch), 32 px for H3
                       (16x VAE x 2x2 patch).
  * texture energy   — plastic skin is a high-frequency collapse, fried skin a spike;
                       Laplacian variance in the face box tells which way a block pushes.
  * clipping + sat   — blown highlights and oversaturation arrive before anything else
                       is visible.
"""
from typing import Dict, Optional, Tuple

import numpy as np

# Family -> the pixel pitch of the model's patch lattice.
PATCH_PITCH = {"klein": 16, "krea2": 16, "minimax": 32}


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def patch_grid_score(rgb: np.ndarray, pitch: int) -> float:
    """On-grid vs off-grid edge energy. ~1.0 = no lattice; rising = the grid is emerging.

    First differences along each axis, compared at seams that land on the pitch against the
    mean everywhere else. A few pixels inside the border are ignored — real image edges at
    the frame would otherwise pollute both sides equally anyway, but cheaply avoiding them
    keeps small images honest."""
    g = _gray(rgb)
    h, w = g.shape
    if h < 2 * pitch or w < 2 * pitch:
        return 1.0
    dx = np.abs(np.diff(g, axis=1))          # [h, w-1]; dx[:, j] = edge between col j and j+1
    dy = np.abs(np.diff(g, axis=0))
    on, off = [], []
    for arr, n in ((dx, w - 1), (dy, h - 1)):
        seam = np.zeros(n, dtype=bool)
        # The seam between patch k and k+1 sits between columns pitch*k-1 and pitch*k —
        # dx index pitch*k-1. The neighbouring edge (index pitch*k) is counted too: ringing
        # and single-line artifacts straddle the boundary rather than landing on one side.
        idx = np.arange(pitch, n, pitch)
        seam[idx - 1] = True
        seam[idx[idx < n]] = True
        axis = 0 if arr is dx else 1
        line = arr.mean(axis=axis)            # mean edge energy per seam position
        on.append(line[seam])
        off.append(line[~seam])
    on_e = float(np.concatenate(on).mean()) if any(len(o) for o in on) else 0.0
    # Median, not mean: real image content (a hairline, a horizon) can land one huge edge
    # off-grid and drown the comparison; the off-grid MEDIAN is the honest noise floor.
    off_e = float(np.median(np.concatenate(off)))
    return on_e / max(off_e, 1e-6)


def texture_energy(rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """Laplacian variance — the classic detail/sharpness energy — over the face box when one
    is known, else the whole frame. Falls (plastic) or spikes (fried) under overbake."""
    import cv2
    g = _gray(rgb)
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(g.shape[1], x2), min(g.shape[0], y2)
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            g = g[y1:y2, x1:x2]
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def clip_and_saturation(rgb: np.ndarray) -> Tuple[float, float]:
    """(% of pixels with any channel blown to either rail, mean HSV saturation 0-255)."""
    import cv2
    clipped = float(np.mean((rgb >= 254).any(axis=-1) | (rgb <= 1).any(axis=-1)) * 100.0)
    sat = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean())
    return clipped, sat


def compare(baseline_rgb: np.ndarray, tweaked_rgb: np.ndarray, pitch: int,
            baseline_bbox=None, tweaked_bbox=None) -> Dict[str, float]:
    """Metrics 2-4 for both renders plus deltas. Likeness (metric 1) is computed by the
    caller, which owns the shared face embedder."""
    out: Dict[str, float] = {}
    out["grid_base"] = patch_grid_score(baseline_rgb, pitch)
    out["grid_tweak"] = patch_grid_score(tweaked_rgb, pitch)
    out["texture_base"] = texture_energy(baseline_rgb, baseline_bbox)
    out["texture_tweak"] = texture_energy(tweaked_rgb, tweaked_bbox)
    out["clip_base"], out["sat_base"] = clip_and_saturation(baseline_rgb)
    out["clip_tweak"], out["sat_tweak"] = clip_and_saturation(tweaked_rgb)
    for k in ("grid", "texture", "clip", "sat"):
        out[f"{k}_delta"] = out[f"{k}_tweak"] - out[f"{k}_base"]
    return out
