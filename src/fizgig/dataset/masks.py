"""Masked training: a grayscale mask beside a training image weights each pixel's loss - white counts fully, grey
partly, black at the floor the Training tab sets. The picture itself stays whole.

A mask is named after its image with OneTrainer's suffix, so masks made there work here: photo.jpg -> photo-masklabel.png
(any image format). Mask files are never training images: every listing of a dataset's pictures leaves them out
(is_mask_file). Light on imports on purpose - the web server's folder listings use it.

load_mask brings a mask onto a cached latent's grid with the crop the picture got when it was cached (the same
resize_image_to_bucket, the same original size and bucket), so it is read at train time and never cached: editing a
mask needs no re-encode."""
import os

MASK_SUFFIX = "-masklabel"
_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ASPECT_TOLERANCE = 0.02          # a mask whose shape is further off than this from its picture's is not used


def is_mask_file(path) -> bool:
    return os.path.splitext(os.path.basename(str(path)))[0].lower().endswith(MASK_SUFFIX)


def mask_for(image_path, names=None):
    """The mask file beside an image, or None. names: the folder's file names (lower case -> real), to save a stat per
    extension when a whole folder is checked."""
    folder, base = os.path.split(str(image_path))
    return mask_for_stem(folder, os.path.splitext(base)[0], names)


def mask_for_stem(folder, stem, names=None):
    """The mask of the image called `stem` (no extension) in `folder`, or None."""
    stem = stem + MASK_SUFFIX
    for ext in _EXTS:
        if names is not None:
            real = names.get((stem + ext).lower())
            if real:
                return os.path.join(folder, real)
        elif os.path.isfile(os.path.join(folder, stem + ext)):
            return os.path.join(folder, stem + ext)
    return None


def folder_names(folder):
    """{lower-case name: real name} of a folder's files, for mask_for / mask_for_stem over a whole folder."""
    try:
        return {f.lower(): f for f in os.listdir(folder)}
    except OSError:
        return {}


def folder_masks(folder, image_exts):
    """{image name: mask path} for the pictures in a folder (top level) that have a mask."""
    try:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except OSError:
        return {}
    names = {f.lower(): f for f in files}
    exts = {e.lower() for e in image_exts}
    out = {}
    for f in files:
        if os.path.splitext(f)[1].lower() in exts and not is_mask_file(f):
            m = mask_for(os.path.join(folder, f), names)
            if m:
                out[f] = m
    return out


def mask_weight(m, floor):
    """A mask in [0, 1] -> the loss weight: never below the floor, 1 where it's white - OneTrainer's clamp, so a mask
    made there weights exactly as it did there."""
    return m.clamp(min=float(floor), max=1.0)


MIN_COVER = 0.01        # a mask with less white than this (an all-black one) leaves nothing to train on: not used


def aspect_ok(mask_size, image_size):
    """Is a mask (w, h) the shape of its picture (w, h)? A smaller mask is fine (it's stretched); one cropped,
    rotated or made for another picture is not."""
    (mw, mh), (iw, ih) = mask_size, image_size
    return mw > 0 and mh > 0 and iw > 0 and ih > 0 and abs((mw / mh) / (iw / ih) - 1.0) <= ASPECT_TOLERANCE


def read_mask(path):
    """A mask file -> grayscale PIL image (L): its brightness, white = train. A transparent part counts as black (a
    white subject painted on an empty layer), and a 16-bit grayscale file keeps its full range."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im, dtype=np.float64)
            return Image.fromarray(np.clip(a / (257.0 if a.max() > 255 else 1.0), 0, 255).astype(np.uint8), "L")
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            return Image.alpha_composite(black, rgba).convert("L")
        return im.convert("L")


def exif_rotated(image_path):
    """Does the picture carry a rotation flag (EXIF orientation other than 1)? Image Prep re-saves such a picture
    upright."""
    from PIL import Image
    try:
        with Image.open(image_path) as im:
            return int(im.getexif().get(0x0112, 1) or 1) != 1
    except Exception:
        return False


def mask_problem(path, image_size, image_path=None):
    """Why a mask can't be used for a picture of image_size (w, h), or "" - checked once as training starts (and by the
    Training tab's note), so a picture with such a mask trains unmasked and the log says which and why."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            size = im.size
    except Exception:
        return "can't be read"
    if not aspect_ok(size, image_size):
        return "isn't the shape of its picture (cropped or rotated after the mask was made?)"
    import numpy as np
    m = np.asarray(read_mask(path).reduce(4) if min(size) >= 64 else read_mask(path), dtype=np.float32) / 255.0
    if float(m.mean()) < MIN_COVER:
        return "is (almost) all black - it leaves nothing to train on"
    return ""


_MEMO = {}


def load_mask(path, image_size, bucket_reso, latent_hw):
    """A mask on a cached latent's grid: (h, w) float32 numpy in [0, 1], or None when the mask can't be used (the
    training prep checked it with mask_problem and logged any it won't use). image_size: the picture's (w, h) as cached;
    bucket_reso: its bucket (w, h); latent_hw: the latent's (h, w)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size, tuple(image_size), tuple(bucket_reso), tuple(latent_hw))
    if key in _MEMO:
        return _MEMO[key]
    import cv2
    import numpy as np
    from fizgig.dataset.image_dataset import resize_image_to_bucket
    m = read_mask(path)
    out = None
    if aspect_ok(m.size, image_size):
        if m.size != tuple(image_size):
            m = m.resize(tuple(image_size))           # a mask saved smaller (or larger) than its picture: stretched
        # the crop the picture got when it was cached (on the 8-bit image, so an upscale's ringing is clipped), then an
        # area average onto the latent grid
        px = resize_image_to_bucket(m, tuple(bucket_reso)).astype(np.float32) / 255.0
        out = cv2.resize(px, (int(latent_hw[1]), int(latent_hw[0])), interpolation=cv2.INTER_AREA)
        out = np.clip(out, 0.0, 1.0)
    if len(_MEMO) > 4096:
        _MEMO.clear()
    _MEMO[key] = out
    return out


_FACES = {}
_DETECTOR = []


def face_box(image_path):
    """Boost the face: the largest face in a picture as (x1, y1, x2, y2) in its upright pixels, or None (no face, no
    detector). Found once per file per run."""
    try:
        st = os.stat(image_path)
    except OSError:
        return None
    key = (os.path.abspath(image_path), st.st_mtime_ns, st.st_size)
    if key not in _FACES:
        box = None
        try:
            if not _DETECTOR:
                from fizgig.face_utils import FaceDetector
                _DETECTOR.append(FaceDetector())
            f = _DETECTOR[0].get_largest(_DETECTOR[0].detect_all(image_path))
            if f is not None:
                box = tuple(float(v) for v in f.bbox)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[face boost] no face detection for {image_path}: {e}")
        _FACES[key] = box
    return _FACES[key]


FACE_AREAS = ("face", "head")


def load_face(box, image_size, bucket_reso, latent_hw, area="face"):
    """A face box as a soft oval on a cached latent's grid: (h, w) float32 in [0, 1] (1 over the face, feathered to 0
    a little past it), cropped as the picture was when it was cached - the same path as load_mask. area "head": the
    oval grown sideways and upward to take in the hair (a mask trims it to the person)."""
    import cv2
    import numpy as np
    from PIL import Image
    from fizgig.dataset.image_dataset import resize_image_to_bucket
    w, h = (int(v) for v in image_size)
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if area == "head":                  # face and hair: wider, and up past the crown
        cy -= 0.2 * bh                  # full weight from ~0.45 face-heights above the hairline to past the chin
        ax, ay = max(1.0, bw * 1.0), max(1.0, bh * 1.05)
    else:
        # full weight over the whole box (chin, cheeks, hairline: r <= 0.7 at its edges), fading to 0 just past it
        ax, ay = max(1.0, bw * 0.72), max(1.0, bh * 0.75)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
    m = np.clip((1.0 - r) / 0.3, 0.0, 1.0)                            # full inside 0.7 of the oval, fading to its edge
    px = resize_image_to_bucket(Image.fromarray((m * 255).astype(np.uint8), "L"), tuple(bucket_reso))
    out = cv2.resize(px.astype(np.float32) / 255.0, (int(latent_hw[1]), int(latent_hw[0])),
                     interpolation=cv2.INTER_AREA)
    return np.clip(out, 0.0, 1.0)


_REPORTS = {}


def folder_report(folders, image_exts):
    """Masked training's readout for the training folder(s): {"with": pictures that have a mask, "total": pictures,
    "bad": ["photo-masklabel.png: why", ...] (masks that won't be used)}. Kept until a folder or a mask changes."""
    out = {"with": 0, "total": 0, "bad": []}
    exts = {e.lower() for e in image_exts}
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        names = folder_names(folder)
        pics = [n for low, n in names.items() if os.path.splitext(low)[1] in exts and not is_mask_file(low)]
        found = {n: mask_for(os.path.join(folder, n), names) for n in pics}
        try:
            key = (os.path.normcase(folder), tuple(sorted(
                (n, m, os.path.getmtime(m), os.path.getmtime(os.path.join(folder, n))) for n, m in found.items() if m)))
        except OSError:                               # a file went while the folder was read: the next look gets it
            continue
        if key not in _REPORTS:
            bad = []
            for n, m in found.items():
                if not m:
                    continue
                try:
                    from PIL import Image
                    from fizgig.utils.upright import upright_size
                    with Image.open(os.path.join(folder, n)) as im:
                        size = upright_size(im)         # as training reads it (a rotation flag applied)
                    why = mask_problem(m, size, os.path.join(folder, n))
                except Exception:
                    why = "its picture can't be read"
                if why:
                    bad.append(f"{os.path.basename(m)}: {why}")
            if len(_REPORTS) > 64:
                _REPORTS.clear()
            _REPORTS[key] = (sum(1 for m in found.values() if m) - len(bad), sorted(bad))
        good, bad = _REPORTS[key]
        out["with"] += good
        out["total"] += len(pics)
        out["bad"] += bad
    return out
