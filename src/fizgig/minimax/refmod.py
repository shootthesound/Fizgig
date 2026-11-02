"""MiniMax H3 RefMods — make one, and make it stronger than a plain encode.

A RefMod (the community node pack `ComfyUI-MiniMaxH3Mod`) is a reference for H3 saved as a
file: the reference run through the video VAE once, its normalized latent stored in a
`.safetensors` with a JSON header, and fed back at generation time as reference condition rows
(H3's own r2v `refs` payload). Nothing in it is learned — the model copies what the rows show.

This module writes that exact file, and then does the thing the extractor can't: with the
H3 base loaded and FROZEN, the mod's latent is the only parameter, and a few hundred steps of
the ordinary flow-matching loss on the user's dataset push it toward the values that make H3
reproduce the subject across the dataset's captions. Textual inversion, in the reference
channel. The output is still just a latent tensor in the node's own format, so the official
loader uses it unchanged — it simply carries more of the subject than the VAE alone did.

Layout contract (checked against the node pack's core.py, 9 Sep 2026, `_format_version` 2; the
pack writes 4 as of v0.2.6 and its reader takes ours unchanged, re-checked 16 Sep 2026):
  tensor  `latent`  [1, 24, T, H, W] fp16, normalized VAE units (ours are the same units —
                    the caches here are std ~1.0, and so is the node's example mod)
  header  `refmod_meta` = JSON {name, kind (image|video), latent_h, latent_w, latent_t, mode
                    (encode|training), source, source_shape, pool, optimize_steps, tags,
                    description, concept_type, _format_version}
A multi-reference mod is ONE video-kind block of T frames (kind "video", latent_t T) — the
node stacks references along time and ComfyUI packs the block on the video clock. The
optimiser feeds the DiT the same single block, so what it learns is what the node will show.
"""
from __future__ import annotations

import gc
import glob
import json
import logging
import math
import os
import random
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from fizgig.minimax.refmod_meta import NODE_META_KEY, NODE_TOKEN_CAP  # noqa: F401
from fizgig.utils.upright import upright

logger = logging.getLogger(__name__)

NODE_FORMAT_VERSION = 4          # what the pack writes as of v0.2.6; its reader takes 2 and 4 alike
BUNDLE_FORMAT_VERSION = 5        # the pack's one-file container: members ref_0, ref_1, … (BUNDLE_FORMAT.md)
ARCH = "minimaxh3"
MAX_REFS_DEFAULT = 16      # measured 10 Sep 2026: 16 references scored 70 on portraits vs 60 for 8


# ─── references from the caches ─────────────────────────────────────────────────────────────

def collect_refs(cache_dirs, max_refs: int = MAX_REFS_DEFAULT,
                 clips: str = "still") -> List[Tuple[str, torch.Tensor, str]]:
    """The dataset's own H3 latent caches -> up to `max_refs` (name, latent, kind).

    Photos come first (each a normalized still latent [24, h, w] at its bucket size), then the
    clips. clips="still": each clip is its sharpest-face frame (`still_latent`, [24, h, w]) —
    a clip cached without a still contributes nothing. clips="motion": each clip is ALL of its
    latent frames ([24, T, h, w]), the node pack's video reference — a run of frames the model
    reads as motion. Sorted by name so the pick is stable run to run."""
    from safetensors.torch import load_file
    photos, stills = [], []
    motion = str(clips or "still").lower().startswith("motion")
    seen = set()
    for d in cache_dirs:
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(glob.escape(d), f"*_{ARCH}.safetensors"))):
            base = os.path.basename(p)
            if base in seen:
                continue
            seen.add(base)
            try:
                sd = load_file(p)
            except Exception as exc:
                logger.warning(f"[refmod] skipped {base}: {exc}")
                continue
            stem = base[: -len(f"_{ARCH}.safetensors")]
            lat_keys = [k for k in sd if k.startswith("latent_")]
            zc = sd[lat_keys[0]] if lat_keys else None
            if motion and zc is not None and zc.dim() == 4:      # (C, T, H, W): the whole clip
                stills.append((stem, zc.float(), "clip motion"))
                continue
            if "still_latent" in sd:
                z = sd["still_latent"]
                if z.dim() == 3:
                    stills.append((stem, z.float(), "clip still"))
                continue
            if zc is None:
                continue
            if zc.dim() == 3:                      # (C, H, W): a still; clips are 4-D
                photos.append((stem, zc.float(), "photo"))
    refs = (photos + stills)[:max_refs]
    return refs


def exclude_refs_from_training(group, ref_stems) -> Tuple[int, int]:
    """Hold the reference stills OUT of the optimiser's training set (opt-in, --holdout_refs):
    by default the optimiser trains on every still, references included. Rebuilds each
    dataset's bucket manager without those items. A clip shares its still's cache file, so a
    clip whose still is a reference goes too (clips are skipped by the optimisers anyway).
    Returns (items removed, items remaining)."""
    from fizgig.dataset.image_dataset import BucketBatchManager
    # Compared without the _WxH size token: the references may be cached at another size.
    stems = {image_stem(s) for s in ref_stems}
    suffix = f"_{ARCH}.safetensors"
    removed = 0
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        if bm is None:
            continue
        kept = {}
        for reso, items in bm.buckets.items():
            keep = []
            for it in items:
                base = os.path.basename(getattr(it, "latent_cache_path", "") or "")
                stem = base[: -len(suffix)] if base.endswith(suffix) else base
                if image_stem(stem) in stems:
                    removed += 1
                else:
                    keep.append(it)
            if keep:
                kept[reso] = keep
        ds.batch_manager = BucketBatchManager(kept, bm.batch_size, num_timestep_buckets=bm.num_timestep_buckets)
        ds.num_train_items = sum(len(b) for b in kept.values())
    group.num_train_items = sum(getattr(ds, "num_train_items", 0) for ds in group.datasets)
    # The group is a torch ConcatDataset: its index table was built from the OLD batch counts
    # (the first held-out run indexed past the end). Recompute it.
    try:
        from torch.utils.data import ConcatDataset
        if isinstance(group, ConcatDataset):
            group.cumulative_sizes = ConcatDataset.cumsum(group.datasets)
    except Exception:
        pass
    return removed, group.num_train_items


def blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """The node pack's weakening target: a heavy spatial low-pass of the latent itself."""
    if z.dim() != 5:
        return z
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


def apply_ref_strength(z: torch.Tensor, strength: float) -> torch.Tensor:
    """Reference strength as the node pack defines it: 1.0 = the latent as stored; below that,
    mixed toward a blurred copy of itself (on-manifold, detail shed); <= 0 = no reference."""
    s = float(strength)
    if s >= 1.0:
        return z
    return s * z + (1.0 - s) * blur_latent(z)


def aspect_grid(pool: int, aspect_hw: float) -> Tuple[int, int]:
    """Even (h, w) latent grid whose long edge is `pool` and whose aspect matches the source —
    the node's own rule (a portrait pooled into a square grid comes out "fat")."""
    if aspect_hw >= 1.0:
        h, w = float(pool), pool / aspect_hw
    else:
        w, h = float(pool), pool * aspect_hw
    return max(2, round(h / 2) * 2), max(2, round(w / 2) * 2)


def _canvas_ref(refs) -> Tuple[int, int]:
    """The canvas every reference is fitted to: the MAJORITY aspect among the references (a
    portrait set on a portrait canvas, a square set on a square one), sized by the MEDIAN
    reference of that aspect (16 Sep 2026 — the largest used to set it, which interpolated
    every smaller reference UP in latent space and blurred it; at the median most references
    are used at native scale and only the biggest scale down). Whichever file sorts first is
    not a canvas policy."""
    from collections import Counter
    keys = [round(int(z.shape[-2]) / float(int(z.shape[-1])), 1) for _, z, _ in refs]
    top = Counter(keys).most_common(1)[0][0]
    cands = sorted([z for (_, z, _), k in zip(refs, keys) if k == top],
                   key=lambda z: int(z.shape[-2]) * int(z.shape[-1]))
    best = cands[len(cands) // 2]                       # upper-middle: lean larger on a tie
    return int(best.shape[-2]), int(best.shape[-1])


def crop_window(nh: int, nw: int, gh: int, gw: int, centre=None) -> Tuple[int, int]:
    """(top, left) of a gh x gw window inside an nh x nw latent. centre = (fy, fx) in [0, 1]
    (a face, normalised) puts the window around it, clamped to the edges; None = the middle."""
    if centre is None:
        return (nh - gh) // 2, (nw - gw) // 2
    fy, fx = float(centre[0]), float(centre[1])
    top = int(round(fy * nh - gh / 2.0))
    left = int(round(fx * nw - gw / 2.0))
    return max(0, min(nh - gh, top)), max(0, min(nw - gw, left))


def cover_crop(z4: torch.Tensor, gh: int, gw: int, centre=None) -> torch.Tensor:
    """[1, 24, h, w] -> [1, 24, gh, gw] with the ASPECT KEPT: scale so the latent covers the
    canvas, then crop — the node pack's `crop="center"` cover-crop, on latents, except that a
    face `centre` (normalised (y, x)) slides the window to keep the face (16 Sep 2026). A
    portrait latent on a square canvas loses its top/bottom instead of being squashed (the
    first version resized straight to the canvas and squashed faces — Peter, 10 Sep 2026)."""
    h, w = int(z4.shape[-2]), int(z4.shape[-1])
    s = max(gh / float(h), gw / float(w))
    nh, nw = max(gh, int(round(h * s))), max(gw, int(round(w * s)))
    if (nh, nw) != (h, w):
        z4 = F.interpolate(z4, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = crop_window(nh, nw, gh, gw, centre)
    return z4[..., top:top + gh, left:left + gw]


_STEM_RES_RX = None


def image_stem(cache_stem: str) -> str:
    """'shot (1)_1024x1024' (a cache file's stem) -> 'shot (1)' (the dataset image's stem)."""
    import re
    global _STEM_RES_RX
    if _STEM_RES_RX is None:
        _STEM_RES_RX = re.compile(r"_\d+x\d+$")
    return _STEM_RES_RX.sub("", str(cache_stem))


def reference_face_centres(refs, image_dirs, face_sizes: Optional[dict] = None) -> dict:
    """{cache stem: (fy, fx)} — the largest detected face in each reference's dataset image,
    normalised to the image. Only consulted when a reference has to be cropped. Empty when the
    detector is unavailable or an image can't be found; every failure is per-reference.
    `face_sizes`, when given, is filled with {stem: face box area / image area} from the same
    pass (the optimiser's subset pool wants the large-in-frame faces)."""
    out = {}
    dirs = [d for d in (image_dirs or []) if d and os.path.isdir(d)]
    if not dirs or not refs:
        return out
    try:
        import sys as _sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from fizgig.face_utils import FaceDetector
        from PIL import Image as _Image
        det = FaceDetector()
        _avail = getattr(det, "available", True)
        if not (_avail() if callable(_avail) else _avail):
            return out
    except Exception as exc:
        logger.info(f"[refmod] face-centred cropping unavailable ({exc}); centre crops")
        return out
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    for stem, _z, _kind in refs:
        istem = image_stem(stem)
        path = None
        for d in dirs:
            for e in exts:
                cand = os.path.join(d, istem + e)
                if os.path.isfile(cand):
                    path = cand
                    break
            if path:
                break
        if not path:
            continue
        try:
            # Opened with PIL, not cv2.imread: libpng's "iCCP: known incorrect sRGB profile"
            # line for every PNG with a stale colour profile came from OpenCV's reader.
            with _Image.open(path) as im:
                im = upright(im)
                W, H = im.size
                faces = det.detect_from_pil(im)
            face = det.get_largest(faces) if faces else None
            if face is None:
                continue
            cx, cy = face.center
            out[stem] = (float(max(0.0, min(1.0, cy / float(H)))), float(max(0.0, min(1.0, cx / float(W)))))
            if face_sizes is not None:
                x1, y1, x2, y2 = face.bbox
                face_sizes[stem] = float(max(0, x2 - x1) * max(0, y2 - y1)) / float(max(1, W * H))
        except Exception as exc:
            logger.info(f"[refmod] face detection skipped for {istem}: {exc}")
    return out


def large_face_pool(refs, face_sizes: dict, k: int, min_ratio: float = 0.5) -> Optional[list]:
    """Frame indices (in mod order) of the references the per-step subset may draw from: every
    reference whose face box is at least `min_ratio` of the largest face's area, and never
    fewer than `k` (topped up with the next-largest). None when no face was found anywhere —
    the subset then draws from every reference."""
    sized = [(float(face_sizes.get(stem, 0.0)), i) for i, (stem, _z, _k) in enumerate(refs)]
    if not sized or max(a for a, _ in sized) <= 0.0:
        return None
    biggest = max(a for a, _ in sized)
    pool = [i for a, i in sized if a >= min_ratio * biggest]
    if len(pool) < k:
        order = [i for _a, i in sorted(sized, key=lambda t: -t[0])]
        pool = order[:min(k, len(order))]
    return sorted(pool)


def build_mod(refs, grid: Optional[int], faces: Optional[dict] = None) -> Tuple[torch.Tensor, str]:
    """(name, [24, h, w], kind) refs -> the mod latent [1, 24, T, gh, gw] fp32 + a pool label.

    Every reference is cover-cropped (aspect kept) onto one canvas — the majority aspect among
    the references, sized by the largest of them. grid = None keeps that canvas at full
    resolution (the node's encode mode); an integer grid average-pools it to that many latent
    cells on the long edge (the node's training mode). Dims are always even: the DiT patches
    2x2 latent cells into one token."""
    if not refs:
        raise ValueError("no references — cache the dataset first (photos, or clips with "
                         "the sharpest-face still on)")
    ch, cw = _canvas_ref(refs)
    ch, cw = (ch // 2) * 2, (cw // 2) * 2
    if grid is None:
        gh, gw = ch, cw
        label = f"full-res {gw * 16}x{gh * 16}px"
    else:
        gh, gw = aspect_grid(int(grid), ch / float(cw))
        label = f"{len(refs)}x{gh}x{gw}"
    frames = []
    n_cropped = 0
    for stem, z, _ in refs:
        # a clip-as-motion reference is [24, T, h, w]: every frame goes in, cropped alike
        z_frames = [z] if z.dim() == 3 else [z[:, t] for t in range(int(z.shape[1]))]
        h, w = int(z.shape[-2]), int(z.shape[-1])
        sc = max(ch / float(h), cw / float(w))
        nh, nw = max(ch, int(round(h * sc))), max(cw, int(round(w * sc)))
        lost = 1.0 - (ch * cw) / float(nh * nw)
        centre = (faces or {}).get(stem)
        if lost > 0.005:
            n_cropped += 1
            _how = "centre — no face found"
            if centre:
                # The window slides within the overflow only — never a tighter crop. A face
                # near an edge lands the window at that edge; say so rather than crop more.
                _top, _left = crop_window(nh, nw, ch, cw, centre)
                _want_t, _want_l = int(round(centre[0] * nh - ch / 2.0)), int(round(centre[1] * nw - cw / 2.0))
                _how = ("around the face" if (_top, _left) == (_want_t, _want_l)
                        else "face kept, at the edge — the canvas can't centre it without cropping more")
            logger.info("[refmod] %s: %.0f%% cropped to the %dx%d canvas (%s)", stem, lost * 100,
                        cw * 16, ch * 16, _how)
        for zf in z_frames:
            z4 = cover_crop(zf.float().unsqueeze(0), ch, cw, centre)   # [1, 24, ch, cw], aspect kept
            if grid is not None:
                z4 = F.adaptive_avg_pool2d(z4, (gh, gw))
            frames.append(z4)
    if n_cropped:
        logger.info("[refmod] %d of %d references were cropped to the canvas; the rest are used "
                    "whole (a same-aspect set is never cropped)", n_cropped, len(refs))
    latent = torch.stack(frames, dim=2)                               # [1, 24, T, gh, gw]
    return latent.contiguous(), label


# ─── the file ────────────────────────────────────────────────────────────────────────────────

def visual_latent_for_file(latent: torch.Tensor) -> torch.Tensor:
    """fp16 [1, 24, T, H, W] on the CPU — the file's tensor."""
    latent = latent.detach().to("cpu", torch.float16).contiguous()
    if latent.dim() == 4:
        latent = latent.unsqueeze(2)
    assert latent.dim() == 5 and latent.shape[0] == 1, f"latent must be [1, 24, T, H, W], got {tuple(latent.shape)}"
    return latent


def visual_meta(latent: torch.Tensor, *, name: str, mode: str, pool: str, optimize_steps: int,
                source_shape: str = "", tags=None, description: str = "", concept_type: str = "identity") -> dict:
    """The pack's header block for a visual mod (standalone file or bundle member)."""
    T = int(latent.shape[2])
    return {
        "name": name,
        "kind": "video" if T > 1 else "image",
        "latent_h": int(latent.shape[3]),
        "latent_w": int(latent.shape[4]),
        "latent_t": T,
        "mode": mode,
        "source": "stack" if T > 1 else "image",
        "source_shape": source_shape,
        "pool": pool,
        "optimize_steps": int(optimize_steps),
        "tags": list(tags or []),
        "description": description or "",
        "concept_type": concept_type,
        "_format_version": NODE_FORMAT_VERSION,
    }


def save_bundle(path_no_ext: str, name: str, members, extra: Optional[dict] = None) -> str:
    """Write `<path>.safetensors` as the pack's one-file container: `members` is a list of
    (meta, latent) in order — visual metas from visual_meta(), audio metas from audio_meta() —
    stored as tensors ref_0, ref_1, … with the container block in the header. The pack's
    loader lists a bundle's members and lets a slot take All, Visual or Audio. Returns the path."""
    from safetensors.torch import save_file
    assert 1 <= len(members) <= 256, "a bundle holds 1-256 members"
    metas, tensors = [], {}
    for i, (meta, latent) in enumerate(members):
        kind = str(meta.get("kind", ""))
        if kind == "audio":
            latent = latent.detach().to("cpu", torch.float32).contiguous()
            assert latent.dim() == 4 and tuple(latent.shape[:3]) == (1, 32, 2), f"audio member must be [1, 32, 2, T], got {tuple(latent.shape)}"
        elif kind in ("image", "video"):
            latent = visual_latent_for_file(latent)
        else:
            raise ValueError(f"bundle member {i}: kind must be image, video or audio, got {kind!r}")
        metas.append(dict(meta))
        tensors[f"ref_{i}"] = latent
    container = {"_format_version": BUNDLE_FORMAT_VERSION, "kind": "bundle", "name": name, "members": metas}
    header = {NODE_META_KEY: json.dumps(container)}
    for k, v in (extra or {}).items():
        header[str(k)] = str(v)
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    out = path_no_ext + ".safetensors"
    save_file(tensors, out, metadata=header)
    return out


def save_refmod(path_no_ext: str, latent: torch.Tensor, *, name: str, mode: str, pool: str,
                optimize_steps: int, source_shape: str = "", tags=None, description: str = "",
                concept_type: str = "identity", extra: Optional[dict] = None) -> str:
    """Write `<path>.safetensors` in the node pack's layout (+ Fizgig's own `ss_*` keys, which
    the node ignores). Returns the path."""
    from safetensors.torch import save_file
    latent = visual_latent_for_file(latent)
    meta = visual_meta(latent, name=name, mode=mode, pool=pool, optimize_steps=optimize_steps,
                       source_shape=source_shape, tags=tags, description=description, concept_type=concept_type)
    header = {NODE_META_KEY: json.dumps(meta)}
    for k, v in (extra or {}).items():
        header[str(k)] = str(v)
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    out = path_no_ext + ".safetensors"
    save_file({"latent": latent}, out, metadata=header)
    return out


# ─── audio mods (encode only) ────────────────────────────────────────────────────────────────
# The pack's audio RefMod (v0.2.0+): the sound through the H3 audio VAE, one latent
# [1, 32, 2, T] at 40 latent frames a second, 2 tokens a frame, kind "audio". A plain encode —
# the pack offers no training for audio and Peter chose not to try the stepped recipe on it
# (16 Sep 2026). Their own tests: music carries over, a speaker's voice did not.

AUDIO_CONCEPT_TYPES = ("voice", "singing", "music_style", "sound_fx", "ambience")
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP = 800                      # samples per latent frame -> 40 a second
AUDIO_CHUNK_SECONDS = 10.0           # the pack's bounded encode chunk


def collect_audio(image_dirs, max_seconds: float):
    """The folder's sound in file order: every clip's soundtrack (muted clips skipped) and every
    audio file, joined until `max_seconds`. -> (waveform float32 [2, L] at 32 kHz, sources) or
    (None, []) when there is nothing to hear."""
    import numpy as np
    from fizgig.minimax.audio import AudioRejected, is_audio, read_audio_file
    from fizgig.minimax.clip import is_video, read_audio
    want = int(round(float(max_seconds) * AUDIO_SAMPLE_RATE))
    pieces, sources, total = [], [], 0
    for d in image_dirs:
        if not d or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d), key=str.lower):
            if total >= want:
                break
            p = os.path.join(d, fn)
            wav = None
            try:
                if is_video(p):
                    wav = read_audio(p)
                elif is_audio(p):
                    wav = read_audio_file(p)
            except AudioRejected as e:
                logger.info(f"[refmod] audio: {e}")
                continue
            except Exception as e:  # a broken file is skipped, not fatal
                logger.warning(f"[refmod] audio: {fn}: {e}")
                continue
            if wav is None or wav.size == 0:
                continue
            take = wav[:, :max(0, want - total)]
            pieces.append(np.asarray(take, dtype=np.float32))
            sources.append(fn)
            total += take.shape[1]
    if not pieces:
        return None, []
    return np.concatenate(pieces, axis=1), sources


def encode_audio_mod(audio_vae, wav, chunk_seconds: float = AUDIO_CHUNK_SECONDS) -> torch.Tensor:
    """[2, L] float32 at 32 kHz -> [1, 32, 2, T] fp32, encoded in bounded chunks of whole latent
    frames the way the pack does (10 s = 400 frames = 320 000 samples a chunk)."""
    chunk = max(AUDIO_HOP, int(round(float(chunk_seconds) * (AUDIO_SAMPLE_RATE // AUDIO_HOP))) * AUDIO_HOP)
    x = torch.as_tensor(wav, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)                                            # [1, 2, L]
    dev = next(audio_vae.parameters()).device if hasattr(audio_vae, "parameters") else "cpu"
    outs = []
    with torch.no_grad():
        for start in range(0, int(x.shape[-1]), chunk):
            z = audio_vae.encode(x[..., start:start + chunk].to(dev)).detach().float().cpu()
            if z.dim() != 4 or tuple(z.shape[:3]) != (1, 32, 2):
                raise ValueError(f"the H3 audio VAE returned an unexpected latent: {tuple(z.shape)}")
            outs.append(z)
    return torch.cat(outs, dim=-1).contiguous()


def audio_token_count(latent: torch.Tensor) -> int:
    return 2 * int(latent.shape[-1])


def audio_meta(latent: torch.Tensor, *, name: str, description: str = "", concept_type: str = "voice",
               tags=None) -> dict:
    """The pack's header block for an audio mod (standalone file or bundle member)."""
    T = int(latent.shape[-1])
    return {
        "name": name,
        "kind": "audio",
        "latent_h": 0,
        "latent_w": 0,
        "latent_t": T,
        "mode": "encode",
        "source": "audio",
        "source_shape": f"2x{T}",
        "pool": "",
        "optimize_steps": 0,
        "tags": list(tags or []),
        "description": description or "",
        "concept_type": concept_type if concept_type in AUDIO_CONCEPT_TYPES else "voice",
        "_format_version": NODE_FORMAT_VERSION,
        "sample_rate": AUDIO_SAMPLE_RATE,
    }


def encode_folder_audio(image_dirs, *, audio_vae_path: Optional[str], max_seconds: float = 30.0, device=None):
    """The folder's sound through the H3 audio VAE -> (latent [1, 32, 2, T] fp32, sources), or
    (None, []) when there is nothing to hear (logged, not an error)."""
    if not (float(max_seconds) > 0):
        raise ValueError(f"the audio mod's length must be above 0 seconds (got {max_seconds})")
    wav, sources = collect_audio(image_dirs, max_seconds)
    if wav is None:
        logger.warning("[refmod] audio: no sound in the dataset folder (no clip with a soundtrack, "
                       "no audio file) — no audio mod written")
        return None, []
    if not audio_vae_path or not os.path.isfile(audio_vae_path):
        raise RuntimeError("an audio mod needs the H3 audio VAE — set the Audio VAE path in "
                           "Preferences (Model Paths, MiniMax H3)")
    from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[refmod] audio: {wav.shape[1] / AUDIO_SAMPLE_RATE:.1f} s from {len(sources)} "
                f"source(s) — {', '.join(sources)}")
    vae = load_minimax_h3_audio_vae(audio_vae_path, device=device, dtype=torch.float32)
    try:
        latent = encode_audio_mod(vae, wav)
    finally:
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return latent, sources


def save_audio_refmod(path_no_ext: str, latent: torch.Tensor, *, name: str, description: str = "",
                      concept_type: str = "voice", tags=None, extra: Optional[dict] = None) -> str:
    """Write `<path>.safetensors` as the pack's audio RefMod: tensor `latent` [1, 32, 2, T] and the
    same header block, kind "audio", sample_rate 32000. Returns the path."""
    from safetensors.torch import save_file
    latent = latent.detach().to("cpu", torch.float32).contiguous()
    assert latent.dim() == 4 and tuple(latent.shape[:3]) == (1, 32, 2), \
        f"audio latent must be [1, 32, 2, T], got {tuple(latent.shape)}"
    meta = audio_meta(latent, name=name, description=description, concept_type=concept_type, tags=tags)
    header = {NODE_META_KEY: json.dumps(meta)}
    for k, v in (extra or {}).items():
        header[str(k)] = str(v)
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    out = path_no_ext + ".safetensors"
    save_file({"latent": latent}, out, metadata=header)
    return out


def make_audio_refmod(image_dirs, out_path_no_ext: str, *, name: str, audio_vae_path: Optional[str],
                      max_seconds: float = 30.0, concept_type: str = "voice", description: str = "",
                      device=None) -> Optional[str]:
    """The whole audio step: gather the folder's sound, encode it, write the file. Returns the
    path, or None when the folder has no sound (logged, not an error)."""
    latent, sources = encode_folder_audio(image_dirs, audio_vae_path=audio_vae_path,
                                          max_seconds=max_seconds, device=device)
    if latent is None:
        return None
    out = save_audio_refmod(out_path_no_ext, latent, name=name, description=description,
                            concept_type=concept_type,
                            tags=[f"{len(sources)} source(s), {latent.shape[-1] / 40:.1f} s", "fizgig"],
                            extra={"ss_refmod_audio_seconds": f"{max_seconds:g}"})
    logger.info(f"[refmod] audio mod saved {out} ({audio_token_count(latent)} tokens, "
                f"{int(latent.shape[-1])} latent frames) — a second file for the same loader slot "
                f"(components: Audio) or its own; plays in ComfyUI, RefMod Toolkit renders visual mods only")
    return out


def load_refmod(path: str):
    """-> (latent [1, 24, T, H, W] fp32, meta dict) — the node's reader, in miniature. A bundle
    gives its first visual member (meta carries `bundle_index`); the audio members are not
    what a maker or the Studio's renderer takes."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads((f.metadata() or {}).get(NODE_META_KEY, "{}"))
        if str(meta.get("kind", "")) == "bundle":
            members = meta.get("members") or []
            for i, m in enumerate(members):
                if str(m.get("kind", "")) in ("image", "video"):
                    latent = f.get_tensor(f"ref_{i}").float().clone()
                    m = dict(m); m["bundle_index"] = i; m["bundle_name"] = meta.get("name", "")
                    return latent, m
            raise ValueError(f"{path}: a bundle with no visual member")
        latent = f.get_tensor("latent").float().clone()
    return latent, meta


def base_model_kind(base_model: str, dit_path: str) -> str:
    """'ref2va' | 'fl2va' | the base file's name: the given kind when it is one of the two,
    else read from the file name, else the name itself."""
    k = str(base_model or "").strip().lower()
    if k in ("ref2va", "fl2va"):
        return k
    name = os.path.basename(str(dit_path or "")).lower()
    for kind in ("ref2va", "fl2va"):
        if kind in name:
            return kind
    return os.path.basename(str(dit_path or "")) or "unknown"


def token_count(latent: torch.Tensor) -> int:
    return int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)


def dedup_frame_indices(z: torch.Tensor, threshold: float = 0.02) -> List[int]:
    """Indices of the frames kept by the node pack's greedy near-duplicate rule.

    Each frame is compared with the last KEPT frame: mean-abs difference over the frames' own
    mean magnitude, dropped below `threshold`. A clip's static run (a held shot, a talking head)
    is mostly codec noise frame to frame and each frame still costs its tokens at every step;
    distinct photos land well above the threshold and all survive. `z` is [1, 24, T, h, w]."""
    t = int(z.shape[2])
    if t <= 1:
        return list(range(t))
    flat = z[0].float()
    kept = [0]
    prev = flat[:, 0]
    for i in range(1, t):
        cur = flat[:, i]
        denom = (cur.abs().mean() + prev.abs().mean()) / 2 + 1e-6
        diff = (cur - prev).abs().mean() / denom
        if float(diff) >= threshold:
            kept.append(i)
            prev = cur
    return kept


def thin_motion_frames(mod: torch.Tensor, refs, cap: int, label: str = "mod"):
    """The cap, applied to clip-as-motion frames ONLY (Peter, 16 Sep 2026: the thinning is for
    the video mode, it never goes near photos).

    Photos and clip stills are always kept, whatever the cap. Whatever budget the cap leaves after
    them is shared out among the motion clips in proportion to their frames, and each clip is
    thinned the way the pack thins a clip: near-duplicate frames first, then an even resample,
    never below one frame. `refs` is the list build_mod stacked, in order. Returns
    (mod, kept frame indices, frames per surviving ref or None when nothing changed)."""
    cap = int(cap or 0)
    owner = [i for i, r in enumerate(refs) for _ in range(int(r[1].shape[1]) if r[1].dim() == 4 else 1)]
    t = int(mod.shape[2])
    if cap <= 0 or t != len(owner):
        return mod, list(range(t)), None
    per_frame = (int(mod.shape[3]) // 2) * (int(mod.shape[4]) // 2)
    fixed = [k for k in range(t) if refs[owner[k]][2] != "clip motion"]
    clips = {}
    for k in range(t):
        if refs[owner[k]][2] == "clip motion":
            clips.setdefault(owner[k], []).append(k)
    if not clips or per_frame * t <= cap:
        return mod, list(range(t)), None
    budget = cap - per_frame * len(fixed)
    if budget < per_frame * len(clips):
        logger.warning(f"[refmod] {label}: the photos take {per_frame * len(fixed):,} of the {cap:,}-token "
                       f"cap, which leaves no room for the clips — the photos are kept whole (the cap "
                       f"only thins clips) and each clip keeps one frame")
    # pass 1: each clip loses its near-duplicates
    deduped = {i: [fr[j] for j in dedup_frame_indices(mod[:, :, fr])] for i, fr in clips.items()}
    n_dup = sum(len(clips[i]) - len(deduped[i]) for i in clips)
    if n_dup:
        logger.info(f"[refmod] {label}: dropped {n_dup} near-duplicate clip frame(s) "
                    f"({sum(len(v) for v in clips.values())} -> {sum(len(v) for v in deduped.values())})")
    # pass 2: what is left of the cap, shared by frame count, each clip resampled evenly
    total = sum(len(v) for v in deduped.values())
    fit_frames = max(len(clips), budget // per_frame)
    kept_m = []
    if total > fit_frames:
        shares = {i: max(1, int(fit_frames * len(fr) / float(total))) for i, fr in deduped.items()}
        # the one-frame floor can push the total over; trim the biggest shares until it fits
        # (fit_frames >= the clip count, so it always can)
        while sum(shares.values()) > fit_frames:
            big = max(shares, key=lambda i: shares[i])
            shares[big] -= 1
        for i, fr in deduped.items():
            share = shares[i]
            if share < len(fr):
                idx = torch.linspace(0, len(fr) - 1, share).round().long().tolist()
                fr = [fr[j] for j in idx]
            kept_m += fr
        logger.info(f"[refmod] {label}: clips resampled to {len(kept_m)} frame(s) to fit the "
                    f"{cap:,}-token cap beside {len(fixed)} photo(s)")
    else:
        for fr in deduped.values():
            kept_m += fr
    kept = sorted(fixed + kept_m)
    if len(kept) == t:
        return mod, kept, None
    per_ref = {}
    for k in kept:
        per_ref[owner[k]] = per_ref.get(owner[k], 0) + 1
    frames = [per_ref[i] for i in range(len(refs))]
    return mod[:, :, kept].contiguous(), kept, frames


# ─── the base model, planned like a training run ────────────────────────────────────────────

def plan_and_load_dit(dit_path: str, *, device, dtype, base_quant: str = "auto",
                      blocks_to_swap="auto", mp: float = 0.25, stills_per_step: int = 1):
    """The H3 base on the tier a LoRA run would get (int8 no-swap / int8 streamed / NF4),
    frozen. No adapter: the plan's adapter budget is zero. Gradient checkpointing follows the
    planner: `stills_per_step` is the step's token load in stills of `mp` (the references in
    the sequence plus the training still) — recompute is skipped only when that fits without
    it, as the trainer does; an explicit swap count keeps it on."""
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.common import (is_pruned_checkpoint, plan_base_quant, plan_vram,
                                        _INT8_TRANSIENT_GB)
    from fizgig.minimax import common as _tr
    pruned = is_pruned_checkpoint(dit_path)
    mode, n_swap = base_quant, 0
    _ckpt = True
    # the planner's activation term scales with tokens: the whole sequence, in stills of `mp`
    mp_plan = float(mp) * max(1, int(stills_per_step))
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available():
            from fizgig.utils.device import plannable_free_vram
            free_gb = plannable_free_vram()
            if base_quant == "auto":
                mode, n_swap, _ckpt, why = plan_base_quant(free_gb, pruned, mp=mp_plan, adapter_gb=0.0)
            else:
                mode = base_quant
                resident = (_tr._RESIDENT_INT8_GB if mode == "int8"
                            else (_tr._RESIDENT_PRUNED_GB if pruned else _tr._RESIDENT_GB))
                n_swap, _ckpt = plan_vram(free_gb, mp=mp_plan, resident_gb=resident,
                                          transient_gb=_INT8_TRANSIENT_GB if mode == "int8" else 0.0,
                                          adapter_gb=0.0)
                why = f"base precision pinned to {mode}"
            logger.info(f"[vram] refmod plan: free {free_gb:.1f} GB, step load {mp_plan:.2f} MP "
                        f"({stills_per_step} stills of {mp:.2f}), base {mode} -> "
                        f"blocks_to_swap={n_swap}, checkpointing {'on' if _ckpt else 'off'} ({why})")
        else:
            mode = "nf4" if base_quant == "auto" else base_quant
    else:
        n_swap = int(blocks_to_swap)
        mode = ("int8" if pruned else "nf4") if base_quant == "auto" else base_quant
    if not pruned and mode == "int8":
        mode = "nf4"
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=True,
                              blocks_to_swap=n_swap, base_quant=mode, adaln_fp32=True)
    dit.requires_grad_(False)
    if n_swap > 0:
        h2d = mode == "int8" or (mode in ("nf4", "hqq") and os.environ.get("FIZGIG_NO_NF4_H2D") != "1")
        n_swap = dit.enable_block_swap(n_swap, h2d_only=h2d, ring_size=2)
        _ckpt = True   # swapped blocks need recompute (autograd would pin their weights)
    if _ckpt:
        dit.enable_gradient_checkpointing()
    else:
        logger.info("[vram] gradient checkpointing off — the step fits without recompute")
    dit.eval()
    return dit, mode, n_swap


# ─── the optimisation ────────────────────────────────────────────────────────────────────────

def refmod_step_loss(dit, mod: torch.Tensor, latents: torch.Tensor, text: torch.Tensor, *,
                     device, dtype, shift=None, generator=None, seed: int = 0,
                     sigma_range=None):
    """One flow-matching loss with the mod riding as the reference block. `mod` is the
    parameter ([1, 24, T, gh, gw] fp32, requires_grad); grads reach it through the DiT's
    condition rows (the frozen base only supplies dX)."""
    from fizgig.minimax.common import sample_sigmas
    x0 = latents.float()
    _pt, _ph, _pw = getattr(dit, "patch_size", (1, 2, 2))
    H, W = x0.shape[-2], x0.shape[-1]
    Hc, Wc = (H // _ph) * _ph, (W // _pw) * _pw
    if (Hc, Wc) != (H, W):
        x0 = x0[..., :Hc, :Wc].contiguous()
    noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
    if sigma_range:
        lo, hi = float(sigma_range[0]), float(sigma_range[1])
        sigma = lo + (hi - lo) * torch.rand(1, device=device, generator=generator)
    else:
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=tokens)
    s = sigma.reshape(1, 1, 1, 1, 1)
    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)
    pred = dit(noised.to(dtype), t, text, ref_latents=[mod], seed=seed)
    return F.mse_loss(pred.float(), (x0 - noise).float()), float(sigma.reshape(-1)[0])


DEFAULT_LR = 5e-3   # trial (Peter, 16 Sep 2026): 5x the 10 Sep rate, flat, no warm-up, pull unchanged
DEFAULT_PULL = 2.0
DEFAULT_SIGMA_RANGE = (0.2, 0.8)


def optimize_refmod(dit, group, mod0: torch.Tensor, *, steps: int, lr: float = DEFAULT_LR,
                    pull: float = DEFAULT_PULL, device="cuda", dtype=torch.bfloat16, seed: int = 42,
                    uncond_text: Optional[torch.Tensor] = None, uncond_frac: float = 0.1,
                    warmup: int = 0, log_every: int = 10, on_step=None,
                    target: Optional[torch.Tensor] = None, shared_epoch=None,
                    sigma_range=DEFAULT_SIGMA_RANGE, ref_subset: int = 0,
                    ref_pool: Optional[list] = None) -> torch.Tensor:
    """Optimise the mod latent against the frozen H3 loss over the dataset's stills.

    ref_subset > 0: each step rides a random `ref_subset` of the mod's reference frames (in
    their saved order) instead of all of them. The references are ~90% of the step's tokens
    and attention is quadratic in the sequence, so 3 of 16 is several times faster per step;
    every frame still gets gradient every few steps, and the pull term always sees the whole
    mod. 0 = every reference every step (the 10 Sep 2026 measurement).

    Defaults are the measured recipe (mbacc photos, 10 Sep 2026, ref2va, Full canvas, 4 seeds,
    ArcFace vs the dataset): lr 1e-3, pull 2.0, noise window 0.2-0.8 put every seed at or
    above the raw encode on portrait prompts (70 vs 66) and +5.5 on four off-dataset scene
    prompts (58.7 vs 53.2). lr 5e-3 / pull 0.5 on H3's full shift-12 density LOST fidelity
    (55): at the top of that schedule the loss is about global structure, and pushing the
    reference rows to serve it costs the face detail they exist to carry.

    pull is the weight of an L2 term toward the initial encode: it keeps the mod a reference
    (on the VAE manifold, so the node's blur-toward-itself strength control still means what
    it means) rather than letting it walk off into an adversarial pattern that only works with
    the training captions. uncond_frac of the steps use the empty-prompt embedding when the
    cache has one, so the mod learns to carry the subject without a particular caption."""
    from torch.utils.data import DataLoader
    from fizgig.minimax.common import _Collator
    from multiprocessing import Value
    if steps <= 0:
        return mod0.detach().clone()
    torch.manual_seed(seed)
    random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    # the SAME counter the dataset group was built with — its buckets assert on it
    if shared_epoch is None:
        shared_epoch = getattr(group, "_fizgig_shared_epoch", None) or Value("i", 0)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=_Collator(shared_epoch, group),
                        num_workers=0)
    target = (target if target is not None else mod0).detach().to(device, torch.float32)
    param = torch.nn.Parameter(mod0.detach().to(device, torch.float32).clone())
    opt = torch.optim.AdamW([param], lr=lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    n_frames = int(param.shape[2])
    k_sub = int(ref_subset) if ref_subset and 0 < int(ref_subset) < n_frames else 0
    pool = [i for i in (ref_pool or []) if 0 <= int(i) < n_frames] or list(range(n_frames))
    if k_sub and len(pool) <= k_sub and len(pool) < n_frames:
        k_sub = len(pool)   # a pool no bigger than the subset: ride the whole pool every step
    if k_sub:
        logger.info(f"[refmod] each step rides {k_sub} of the {n_frames} references (random, saved "
                    f"order) drawn from {len(pool)} with a large face in frame")
    step = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    skipped_clips = 0
    while step < steps:
        shared_epoch.value += 1
        for batch in loader:
            if step >= steps:
                break
            lat = batch["latents"]
            if lat.dim() != 4:                     # a whole clip (or a voice placeholder)
                skipped_clips += 1
                continue
            latents = lat.to(device, dtype).unsqueeze(2)              # (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)
            if uncond_text is not None and random.random() < uncond_frac:
                text = uncond_text.to(device, dtype)
            # full rate from step 0 (Peter, 16 Sep 2026 — the 20-step warm-up of the 10 Sep
            # recipe is off; warmup > 0 brings it back)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / float(max(1, warmup)))
            if k_sub:
                _idx = torch.tensor(sorted(random.sample(pool, k_sub)), device=device)
                ride = param.index_select(2, _idx)      # grads flow back to the picked frames
            else:
                ride = param
            with torch.autocast("cuda", enabled=False):
                loss, sig = refmod_step_loss(dit, ride, latents, text, device=device, dtype=dtype,
                                             generator=gen, seed=seed, sigma_range=sigma_range)
                total = loss + pull * F.mse_loss(param, target) if pull > 0 else loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([param], 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            run_n += 1
            if on_step is not None:
                on_step(step, float(loss.detach()))
            if log_every and (step % log_every == 0 or step == steps):
                drift = float((param.detach() - target).pow(2).mean().sqrt())
                el = time.time() - t0
                print(f"[refmod] step {step}/{steps}  loss {run_loss / max(1, run_n):.4f}  "
                      f"drift {drift:.3f} (rms, latent units)  {el / step:.2f} s/step", flush=True)
                run_loss, run_n = 0.0, 0
    if skipped_clips:
        logger.info(f"[refmod] {skipped_clips} clip item(s) skipped — the optimiser trains on "
                    f"stills (photos and clip stills)")
    return param.detach().to("cpu", torch.float32)


# ─── previews ────────────────────────────────────────────────────────────────────────────────

def write_silent_mp4(path: str, frames: torch.Tensor, fps: int = 24) -> None:
    """Decoded frames [3, F, H, W] in [0, 1] -> a playable mp4 with no sound track (the mod is
    a visual reference; its previews carry no audio). Raises on any failure."""
    import subprocess
    from fizgig.minimax.common import _find_ffmpeg
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("no ffmpeg available")
    h, w = int(frames.shape[2]), int(frames.shape[3])
    raw = (frames.permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().cpu().numpy().tobytes()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    p = subprocess.run(cmd, input=raw, capture_output=True,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    if p.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:] or "ffmpeg failed")


def render_previews(dit, mod: torch.Tensor, encoded_prompts, *, out_dir: str, output_name: str,
                    epoch: int, width: int, height: int, steps: int, seed: int, device, dtype,
                    decoder=None, n_swap: int = 0, turbo=None, num_frames: int = 1):
    """One preview per prompt with the mod as the reference block — the way the node will use
    it (no <Picture> vision blocks: the file has no vision side). num_frames 1 = a still;
    above 1 = a clip (22 by default from the GUI): the middle frame is saved as the PNG and
    the whole clip as a silent mp4 beside it. PNG names follow the gallery's contract
    `<name>_e<epoch>_<i>_<ts>_<seed>.png`."""
    from PIL import Image
    from fizgig.minimax import sampling
    from fizgig.minimax.common import (park_dit_to_cpu, restore_parked_dit, turbo_adaln_patch,
                                        turbo_adaln_unpatch, cap_preview_res_small_card)
    os.makedirs(out_dir, exist_ok=True)
    width, height = cap_preview_res_small_card(width, height)
    ts = time.strftime("%Y%m%d%H%M%S")
    ref = mod.to(device, dtype)
    turbo_net, turbo_adaln = (turbo if turbo else (None, []))
    rendered = []
    try:
        if turbo_net is not None:
            turbo_net.to(device=device, dtype=dtype)
            for m in turbo_net.unet_loras:
                m.enabled = True
            turbo_adaln_patch(dit, turbo_adaln, device, dtype)
        with torch.no_grad():
            for i, txt in enumerate(encoded_prompts):
                _nf = max(1, int(num_frames))
                print(f"[preview] refmod preview {epoch}: prompt {i + 1}/{len(encoded_prompts)} "
                      f"({width}x{height}, {_nf} frame{'s' if _nf > 1 else ''}, seed {seed + i})", flush=True)
                lat, _ = sampling.sample_image(dit, txt.to(device, dtype), width=width, height=height,
                                               steps=steps, cfg_scale=1.0, seed=seed + i,
                                               device=device, dtype=dtype, log_steps=False,
                                               num_frames=_nf, ref_latents=[ref], return_audio=True)
                rendered.append((f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}", lat.to("cpu")))
                del lat
    finally:
        if turbo_net is not None:
            for m in turbo_net.unet_loras:
                m.enabled = False
            turbo_adaln_unpatch(turbo_adaln)
            turbo_net.to("cpu")
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    # decode: on 16 GB-class cards the whole base parks for the VAE (same rule as training)
    parked = False
    try:
        small = torch.cuda.get_device_properties(0).total_memory / 1e9 < 20.0
    except Exception:
        small = False
    if decoder is not None and small:
        park_dit_to_cpu(dit)
        parked = True
        gc.collect()
        torch.cuda.empty_cache()
    try:
        if decoder is not None:
            decoder = decoder.to(device)
        with torch.no_grad():
            for stem, lat in rendered:
                if decoder is not None and lat.shape[2] > 1:
                    # a clip: every frame to a silent mp4, the middle frame as the PNG
                    px = decoder.decode_clip(lat.to(device).float())[0]      # [3, F, H, W]
                    n_f = int(px.shape[1])
                    mid = (px[:, n_f // 2].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(mid)
                    try:
                        write_silent_mp4(os.path.join(out_dir, stem + ".mp4"), px.cpu())
                        print(f"[preview] wrote {n_f}-frame clip: {stem}.mp4", flush=True)
                    except Exception as _me:
                        logger.warning(f"[preview] mp4 skipped ({type(_me).__name__}: {_me}) — "
                                       f"the middle frame is saved as the PNG")
                    del px
                elif decoder is not None:
                    px = decoder.decode(lat.to(device).float())[0]
                    arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(arr)
                else:
                    arr = sampling.latent_to_rgb(lat[:, :, :1] if lat.shape[2] > 1 else lat)
                    img = Image.fromarray(arr).resize((width, height), Image.NEAREST)
                img.save(os.path.join(out_dir, stem + ".png"))
    finally:
        if decoder is not None:
            decoder.to("cpu")
        if parked:
            restore_parked_dit(dit, device, n_swap)
        gc.collect()
        torch.cuda.empty_cache()
    return [os.path.join(out_dir, s + ".png") for s, _ in rendered]


# ─── the run ─────────────────────────────────────────────────────────────────────────────────

def run_refmod(*, dataset_config: str, output_dir: str, output_name: str, dit_path: str,
               grid: Optional[int] = None, steps: int = 200, lr: float = DEFAULT_LR, pull: float = DEFAULT_PULL,
               max_refs: int = MAX_REFS_DEFAULT, seed: int = 42, base_quant: str = "auto",
               blocks_to_swap="auto", vae_path: Optional[str] = None,
               te_path: Optional[str] = None, sample_prompts: Optional[List[str]] = None,
               sample_width: int = 768, sample_height: int = 768, sample_steps: int = 20,
               sample_seed: int = 42, preview_every: int = 0, sample_frames: int = 1,
               turbo_lora_path: Optional[str] = None, turbo_lora_strength: float = 1.0,
               description: str = "", init_from: Optional[str] = None,
               sigma_range=DEFAULT_SIGMA_RANGE, exclude_refs: bool = False,
               ref_cache_dirs: Optional[List[str]] = None, ref_subset: int = 1,
               clips: str = "still", concept_type: str = "identity", token_cap: int = 0,
               audio: str = "off", audio_max_seconds: float = 30.0, audio_concept: str = "voice",
               audio_vae_path: Optional[str] = None, base_model: str = "") -> str:
    """Make the mod, optimise it, write it. Returns the output path.

    One file: <output_dir>/<output_name>.safetensors. Steps = 0 writes the plain encode (the
    node extractor's own result); otherwise the optimised mod. Previews: one set before any
    step (epoch 0) and one from the finished mod, plus every `preview_every` steps between."""
    import argparse
    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.dataset.image_dataset import ImageDataset
    from fizgig.training.metadata import ARCHITECTURE_MINIMAX
    from multiprocessing import Value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    if str(audio or "off").lower() != "off":
        # Refuse a bad audio setup here, before a run that may optimise for minutes: the
        # audio step comes last, and a bundle has nothing to write until it succeeds.
        if not (float(audio_max_seconds) > 0):
            raise ValueError(f"the audio mod's length must be above 0 seconds (got {audio_max_seconds})")
        if not audio_vae_path or not os.path.isfile(audio_vae_path):
            raise RuntimeError("an audio mod needs the H3 audio VAE — set the Audio VAE path in "
                               "Preferences (Model Paths, MiniMax H3), or set Audio support to off")

    # dataset (stills; clip stills as photos so every clip lends its sharpest face)
    ImageDataset.clip_still_as_photo = True
    # Steps 0 is a plain encode: nothing trains, so no captions and no text-encoder caches are
    # needed — the references are the latent caches alone. The training set (which needs both)
    # is only built when there are steps to run on it.
    trains = steps > 0
    if not trains:
        from fizgig.dataset.image_dataset import ImageDirectoryDatasource
        ImageDirectoryDatasource.captions_optional = True
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    shared_epoch = Value("i", 0)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=trains, num_timestep_buckets=None, shared_epoch=shared_epoch)
    group._fizgig_shared_epoch = shared_epoch
    if trains and group.num_train_items == 0:
        raise RuntimeError("No training items — run the MiniMax cache steps first.")
    cache_dirs = [getattr(ds, "cache_directory", "") for ds in group.datasets]
    # The references may come from their own caches (Target MP) while the optimiser's stills
    # stay at 0.25 MP in the dataset's caches: the mod's pixels are what the file carries, the
    # stills are only the loss target, and the recipe was measured at 0.25 (16 Sep 2026).
    if ref_cache_dirs:
        logger.info(f"[refmod] references from {', '.join(ref_cache_dirs)}")
    refs = collect_refs(ref_cache_dirs or cache_dirs, max_refs=max_refs, clips=clips)
    # Faces, for the references that end up cropped: the dataset is prepared framing, so a
    # crop keeps the face rather than the frame centre (Peter, 16 Sep 2026).
    face_sizes = {}
    faces = reference_face_centres(refs, [getattr(ds, "image_directory", "") for ds in group.datasets],
                                   face_sizes=face_sizes)
    if not refs:
        raise RuntimeError("No reference stills in the caches (photos, or clips cached with "
                           "'Also train the sharpest face still').")
    n_img = sum(1 for r in refs if r[2] == "photo")
    n_mo = sum(1 for r in refs if r[2] == "clip motion")
    n_st = len(refs) - n_img - n_mo
    mp = max((r[1].shape[-2] * 16) * (r[1].shape[-1] * 16) for r in refs) / 1e6
    if init_from:
        # start from an existing mod (re-preview it, or keep optimising it)
        mod0, _m0 = load_refmod(init_from)
        pool_label = str(_m0.get("pool", "")) or f"{mod0.shape[2]}x{mod0.shape[3]}x{mod0.shape[4]}"
        mode = str(_m0.get("mode", "training"))
        source_shape = str(_m0.get("source_shape", ""))
        logger.info(f"[refmod] starting from {init_from}: mod {tuple(mod0.shape)} "
                    f"({token_count(mod0)} tokens, {_m0.get('optimize_steps', 0)} prior steps)")
    else:
        logger.info(f"[refmod] {len(refs)} reference(s): {n_img} photo(s), {n_st} clip still(s) — "
                    + ", ".join(r[0] for r in refs))
        mod0, pool_label = build_mod(refs, grid, faces)
        mode = "training" if grid is not None else "encode"
        # Thinning: clips as motion only, never a photo (Peter, 16 Sep 2026). Every reference
        # survives; a clip keeps just the frames the cap allows it.
        mod0, _kept, _ref_frames = thin_motion_frames(mod0, refs, int(token_cap or 0), label=output_name)
        _tok = token_count(mod0)
        logger.info(f"[refmod] mod {tuple(mod0.shape)} ({pool_label}, {_tok} tokens, "
                    f"mode {mode}) — the node pack's extractor caps at {NODE_TOKEN_CAP} by default")
        if _tok > NODE_TOKEN_CAP:
            logger.warning(f"[refmod] {_tok} tokens is ABOVE the standard extractor's default cap "
                           f"({NODE_TOKEN_CAP}). The loaders don't refuse it, but every one of "
                           f"those tokens rides in the sequence at each sampling step — slower and "
                           f"more VRAM at generation. Fewer References or a pooled Grid brings it "
                           f"down.")
        source_shape = " +".join(
            f"{(_ref_frames[i] if _ref_frames else (r[1].shape[1] if r[1].dim() == 4 else 1))}x{r[1].shape[-2]}x{r[1].shape[-1]}"
            for i, r in enumerate(refs))

    # The per-step subset draws from the references with a large face in frame (at least half
    # the largest face's area), so every step's gradient comes from a face-sized signal.
    ref_pool = None
    if trains and ref_subset and int(ref_subset) > 0 and not init_from and len(refs) == int(mod0.shape[2]):
        ref_pool = large_face_pool(refs, face_sizes, int(ref_subset))
        if ref_pool is None:
            logger.info("[refmod] no faces measured — the per-step subset draws from every reference")
        else:
            _names = ", ".join(refs[i][0] for i in ref_pool)
            logger.info(f"[refmod] subset pool: {len(ref_pool)} of {len(refs)} references with a large "
                        f"face in frame (>= half the largest face's area) — {_names}")

    if exclude_refs and steps > 0:
        _rm, _left = exclude_refs_from_training(group, [r[0] for r in refs])
        if _left <= 0:
            logger.warning(f"[refmod] every still in the dataset is a reference — nothing would be "
                           f"left to train on, so the references stay in the training set")
            # rebuild is destructive; reload the group as it was
            group = generate_dataset_group_by_blueprint(
                blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
            group._fizgig_shared_epoch = shared_epoch
        else:
            logger.info(f"[refmod] {_rm} reference still(s) held out — the optimiser trains on the "
                        f"other {_left} item(s)")

    uncond_text = None
    for d in cache_dirs:
        f = os.path.join(d or "", f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors")
        if os.path.isfile(f):
            from safetensors.torch import load_file
            uncond_text = load_file(f)["hidden_states"].unsqueeze(0)
            break

    # prompts BEFORE the DiT (the TE never shares the card with it)
    encoded = None
    if sample_prompts and te_path:
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info(f"[preview] pre-encoding {len(sample_prompts)} sample prompt(s)...")
        encoded = encode_sample_prompts(te_path, sample_prompts, device=device, quantize=True)

    tags = [f"{n_img} img, {n_st} clip stills" + (f", {n_mo} clips as motion" if n_mo else ""), "fizgig"]

    if trains or encoded:
        _n_ride = (min(int(ref_subset), int(mod0.shape[2])) if ref_subset and int(ref_subset) > 0
                   else int(mod0.shape[2]))
        dit, base_mode, n_swap = plan_and_load_dit(dit_path, device=device, dtype=dtype,
                                                   base_quant=base_quant, blocks_to_swap=blocks_to_swap, mp=mp,
                                                   stills_per_step=_n_ride + 1 if trains else 1)
    else:
        # plain encode, no previews: the model is never touched
        dit, base_mode, n_swap = None, "none", 0
        logger.info("[refmod] plain encode — no steps and no previews, so the H3 base is not loaded")
    decoder = None
    if vae_path and encoded:
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
        decoder = MiniMaxH3VideoVAEDecoder()
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            decoder.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        decoder = decoder.to(torch.float16).eval()
    turbo = None
    if encoded and turbo_lora_path:
        from fizgig.minimax.common import load_preview_turbo
        turbo = load_preview_turbo(dit, turbo_lora_path, turbo_lora_strength)
    sample_dir = os.path.join(output_dir, "sample")

    def _preview(mod, epoch):
        if not encoded:
            return
        render_previews(dit, mod, encoded, out_dir=sample_dir, output_name=output_name, epoch=epoch,
                        width=sample_width, height=sample_height, steps=sample_steps, seed=sample_seed,
                        device=device, dtype=dtype, decoder=decoder, n_swap=n_swap, turbo=turbo,
                        num_frames=sample_frames)

    _preview(mod0, 0)
    mod = mod0
    if steps > 0:
        logger.info(f"[refmod] optimising {token_count(mod0)} tokens for {steps} steps "
                    f"(lr {lr:g}, pull {pull:g}, base {base_mode}, swap {n_swap})")
        mod = _optimize_with_previews(dit, group, mod0, steps=steps, lr=lr, pull=pull, device=device,
                                      dtype=dtype, seed=seed, uncond_text=uncond_text,
                                      preview_every=preview_every, preview_fn=_preview,
                                      sigma_range=sigma_range, ref_subset=ref_subset,
                                      ref_pool=ref_pool)
        _preview(mod, int(math.ceil(steps / float(preview_every))) if preview_every else 1)

    _audio = str(audio or "off").lower()
    # Which H3 model the mod was tuned against (Peter, 16 Sep 2026): the GUI passes Training
    # Base; a CLI run without it is read from the base file's name. A plain encode had no
    # model in the loop and says so. The tag is what the pack's Inspect node shows.
    _bm = base_model_kind(base_model, dit_path) if trains else "not trained"
    _base_tag = f"tuned on {_bm}" if trains else "plain encode"
    _save_kw = dict(name=output_name, mode=mode, pool=pool_label, optimize_steps=steps, source_shape=source_shape,
                    tags=tags + (["fizgig optimised"] if steps > 0 else []) + [_base_tag],
                    description=description, concept_type=(concept_type or "identity"))
    _extra = {"ss_refmod_steps": str(steps), "ss_refmod_lr": f"{lr:g}",
              "ss_refmod_ref_subset": str(int(ref_subset or 0)),
              "ss_refmod_pull": f"{pull:g}", "ss_refmod_refs": str(len(refs)),
              "ss_refmod_base": base_mode, "ss_refmod_grid": str(grid or "full"),
              "ss_refmod_token_cap": str(int(token_cap or 0)),
              "ss_refmod_base_model": _bm,
              "ss_refmod_base_checkpoint": (os.path.basename(dit_path or "") if trains else "")}
    _out_base = os.path.join(output_dir, output_name)
    if _audio == "off":
        out = save_refmod(_out_base, mod, extra=_extra, **_save_kw)
        mb = os.path.getsize(out) / 1024 / 1024
        logger.info(f"[refmod] saved {out} ({token_count(mod)} tokens, {mb:.2f} MB) — copy it to "
                    f"ComfyUI/models/refmods/ and load it with the ComfyUI-MiniMaxH3Mod nodes")
        return out
    if _audio == "folder":
        # Two files: the visual mod now, the audio mod once the card is clear.
        out = save_refmod(_out_base, mod, extra=_extra, **_save_kw)
        mb = os.path.getsize(out) / 1024 / 1024
        logger.info(f"[refmod] saved {out} ({token_count(mod)} tokens, {mb:.2f} MB) — copy it to "
                    f"ComfyUI/models/refmods/ and load it with the ComfyUI-MiniMaxH3Mod nodes")
    # The audio VAE (345 MB fp32 plus its chunk activations) must not land beside the DiT,
    # Turbo LoRA and decoder on a 16 GB card after an optimised run with previews.
    del dit, turbo, decoder, _preview
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _dirs = [getattr(ds, "image_directory", "") for ds in group.datasets]
    if _audio == "folder":
        make_audio_refmod(_dirs, _out_base + "_audio", name=output_name + "_audio",
                          audio_vae_path=audio_vae_path, max_seconds=float(audio_max_seconds),
                          concept_type=audio_concept, description=description, device=device)
        return out
    # One file (the pack's bundle): the visual member and the audio member together. No sound
    # in the folder -> the plain visual file, so the run still ends with a mod. Should the
    # audio step fail for any other reason, the visual mod is written first, then the error.
    try:
        _lat_a, _sources = encode_folder_audio(_dirs, audio_vae_path=audio_vae_path,
                                               max_seconds=float(audio_max_seconds), device=device)
    except Exception:
        out = save_refmod(_out_base, mod, extra=_extra, **_save_kw)
        logger.error(f"[refmod] the audio step failed — the visual mod was saved on its own as {out}")
        raise
    _stale = _out_base + "_audio.safetensors"
    if os.path.isfile(_stale):
        logger.warning(f"[refmod] {os.path.basename(_stale)} is left over from an earlier two-file run; the "
                       f"bundle carries its own audio member, so delete the old file or the loader will list both")
    if _lat_a is None:
        out = save_refmod(_out_base, mod, extra=_extra, **_save_kw)
        logger.info(f"[refmod] saved {out} ({token_count(mod)} tokens) as a plain visual mod — "
                    f"the folder had no sound to bundle")
        return out
    _vis = visual_latent_for_file(mod)
    _members = [(visual_meta(_vis, **_save_kw), _vis),
                (audio_meta(_lat_a, name=output_name + "_audio", description=description, concept_type=audio_concept,
                            tags=[f"{len(_sources)} source(s), {_lat_a.shape[-1] / 40:.1f} s", "fizgig"]), _lat_a)]
    _extra["ss_refmod_audio_seconds"] = f"{float(audio_max_seconds):g}"
    out = save_bundle(_out_base, output_name, _members, extra=_extra)
    mb = os.path.getsize(out) / 1024 / 1024
    logger.info(f"[refmod] saved {out} as one file: the visual mod ({token_count(mod)} tokens) and the audio "
                f"mod ({audio_token_count(_lat_a)} tokens, {int(_lat_a.shape[-1])} latent frames), {mb:.2f} MB — "
                f"the pack's loader lists both; a slot takes All, Visual or Audio")
    return out


def _optimize_with_previews(dit, group, mod0, *, steps, lr, pull, device, dtype, seed, uncond_text,
                            preview_every, preview_fn, sigma_range=None, ref_subset: int = 0,
                            ref_pool=None):
    """optimize_refmod in chunks so interim previews render from the live latent."""
    if not preview_every or preview_every >= steps:
        return optimize_refmod(dit, group, mod0, steps=steps, lr=lr, pull=pull, device=device,
                               dtype=dtype, seed=seed, uncond_text=uncond_text, sigma_range=sigma_range,
                               ref_subset=ref_subset, ref_pool=ref_pool)
    # chunked: each chunk restarts the optimizer state but keeps the latent — a small price,
    # and it keeps optimize_refmod itself simple. Warm-up only on the first chunk.
    mod = mod0
    done = 0
    k = 0
    while done < steps:
        n = min(preview_every, steps - done)
        mod = optimize_refmod(dit, group, mod, steps=n, lr=lr, pull=pull, device=device, dtype=dtype,
                              seed=seed + k, uncond_text=uncond_text, warmup=0,
                              target=mod0, sigma_range=sigma_range, ref_subset=ref_subset,
                              ref_pool=ref_pool)
        done += n
        k += 1
        if done < steps:
            preview_fn(mod, k)
    return mod
