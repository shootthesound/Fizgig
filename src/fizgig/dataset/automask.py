"""Masks made for masked training: Google's MediaPipe multiclass selfie segmenter (Apache 2.0, 16 MB, downloaded on
first use) run on LiteRT, on the CPU. It labels every pixel background / hair / body skin / face skin / clothes /
other (accessories), so one pass gives:

- "person": everyone in the picture, hair to clothes (the default);
- "no_clothes": the person without their clothes and accessories (face, hair and skin);
- "head": the face and hair only;
- "body_clothes": the person without the face and hair (body, clothes and accessories);
- "body": the body's skin only - no face, hair, clothes or accessories;
- "clothes": the clothes and accessories only (an outfit LoRA);
- "not_faces": everything but the faces and hair (an outfit, a body, a style LoRA that mustn't learn faces);
- "background": the background only, every person left out (a place or scene LoRA).

With several people in a picture, "subject_only" keeps just the one whose face matches the Look Filter's three
baselines (AdaFace, as the filter scores them): the person outline is split into figures - each pixel goes to the face
nearest it inside its own connected shape - and the figure around the best-matching face is kept.

Masks are written beside their pictures as photo-masklabel.png (dataset/masks.py), never over an existing one unless
asked: a mask someone painted by hand stays."""
import hashlib
import os
import urllib.request

import numpy as np
from PIL import Image

from fizgig.dataset.masks import MASK_SUFFIX, MIN_COVER, folder_names, is_mask_file, mask_for
from fizgig.utils.image import flatten
from fizgig.utils.upright import upright

MODEL_NAME = "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/"
             "latest/" + MODEL_NAME)
MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
MODEL_MB = 16
KINDS = {
    "person": "The person (hair to clothes)",
    "no_clothes": "The person, not their clothes",
    "head": "Just the face and hair",
    "body_clothes": "The body and clothes, not the face or hair",
    "body": "The body's skin, not the face, hair or clothes",
    "clothes": "Just the clothes",
    "not_faces": "Everything but the faces",
    "background": "Just the background, not the people",
}
BACKGROUND, HAIR, BODY, FACE, CLOTHES, OTHER = range(6)
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def available():
    """Whether the segmenter's runtime (ai-edge-litert) is installed here."""
    try:
        import ai_edge_litert.interpreter  # noqa: F401
        return True
    except Exception:
        return False


def model_path(models_dir):
    return os.path.join(models_dir, MODEL_NAME)


def ensure_model(models_dir, log=print):
    """The segmenter file, downloaded (and its hash checked) the first time."""
    path = model_path(models_dir)
    if os.path.isfile(path):
        return path
    os.makedirs(models_dir, exist_ok=True)
    log(f"Downloading the person segmenter ({MODEL_MB} MB, once) ...\n")
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    with open(tmp, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("the downloaded segmenter isn't the expected file - try again later")
    os.replace(tmp, path)
    return path


class Segmenter:
    """The multiclass selfie model: a picture -> per-class probabilities at the picture's own size."""

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.h, self.w = int(self.inp["shape"][1]), int(self.inp["shape"][2])

    def classes(self, img):
        """(H, W, 6) float32 probabilities for a PIL RGB picture, at its own size."""
        x = np.asarray(img.resize((self.w, self.h), Image.BILINEAR), np.float32) / 255.0
        self.it.set_tensor(self.inp["index"], x[None])
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        if y.min() < 0 or y.max() > 1.0001:                        # logits: softmax them
            y = np.exp(y - y.max(-1, keepdims=True))
            y /= y.sum(-1, keepdims=True)
        planes = [np.asarray(Image.fromarray(y[..., c]).resize(img.size, Image.BILINEAR)) for c in range(y.shape[-1])]
        return np.stack(planes, -1)


def mask_from(classes, kind):
    """The mask (H, W) float in [0, 1] of a kind from the class probabilities."""
    if kind == "person":
        m = 1.0 - classes[..., BACKGROUND]
    elif kind == "no_clothes":
        m = classes[..., FACE] + classes[..., HAIR] + classes[..., BODY]
    elif kind == "body":
        m = classes[..., BODY]
    elif kind == "body_clothes":
        m = classes[..., BODY] + classes[..., CLOTHES] + classes[..., OTHER]
    elif kind == "head":
        m = classes[..., FACE] + classes[..., HAIR]
    elif kind == "clothes":
        m = classes[..., CLOTHES] + classes[..., OTHER]
    elif kind == "background":
        m = classes[..., BACKGROUND]
    elif kind == "not_faces":
        m = 1.0 - (classes[..., FACE] + classes[..., HAIR])
    else:
        raise ValueError(f"unknown mask kind {kind!r}")
    return np.clip(m, 0.0, 1.0)


def keep_figure(mask, person, faces, keep):
    """mask restricted to one figure: `person` (H, W) is everyone's outline, `faces` the face boxes (x1, y1, x2, y2),
    `keep` the index of the face to keep. Each person pixel goes to the nearest face centre within its own connected
    shape (people apart are separate shapes; people touching split between their faces)."""
    import cv2
    solid = (person > 0.5).astype(np.uint8)
    n, labels = cv2.connectedComponents(solid)
    h, w = person.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centres = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in faces]
    face_comp = [int(labels[min(h - 1, max(0, int(cy))), min(w - 1, max(0, int(cx)))]) for cx, cy in centres]
    own = np.zeros((h, w), bool)
    comp = face_comp[keep]
    if comp == 0 and n > 1:                          # the face centre fell off the outline: the shape most in its box
        x1, y1, x2, y2 = (int(v) for v in faces[keep])
        box = labels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        hits = np.bincount(box.ravel(), minlength=n)[1:] if box.size else np.zeros(n - 1)
        comp = int(np.argmax(hits) + 1) if hits.any() else 0
    if comp == 0:
        return mask * 0.0
    region = labels == comp
    rivals = [i for i, c in enumerate(face_comp) if c == comp and i != keep]
    if rivals:
        d_keep = (xx - centres[keep][0]) ** 2 + (yy - centres[keep][1]) ** 2
        d_rival = np.min([(xx - centres[i][0]) ** 2 + (yy - centres[i][1]) ** 2 for i in rivals], axis=0)
        region &= d_keep <= d_rival
    own |= region
    # the soft edge back: keep the mask's own values near the kept figure (a few pixels' grow)
    grown = cv2.dilate(own.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return mask * grown


def pictures(folder):
    names = folder_names(folder)
    return sorted(os.path.join(folder, real) for low, real in names.items()
                  if os.path.splitext(low)[1] in _IMAGE_EXTS and not is_mask_file(low))


def make_folder(folder, kind, models_dir, *, overwrite=False, subject=None, detector=None, log=print):
    """Write a mask for every picture in a folder. subject: (embed(PIL crop) -> vector or None, baseline vectors) to
    keep only the matching person in pictures with several faces; detector: a face finder (detect_from_pil).
    Returns {"made", "kept", "skipped", "no_person"}."""
    seg = Segmenter(ensure_model(models_dir, log))
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
                img = flatten(upright(im))            # a see-through picture on white, not its hidden colour
        except Exception as e:
            log(f"  {name}: can't be read ({e}) - skipped\n")
            counts["skipped"] += 1
            continue
        cls = seg.classes(img)
        person = mask_from(cls, "person")
        if float(person.mean()) < 0.01:
            log(f"  {name}: no person found - no mask written (it trains as normal)\n")
            counts["no_person"] += 1
            continue
        m = mask_from(cls, kind)
        note = ""
        if subject is not None and detector is not None and kind not in ("not_faces", "background"):
            faces = [f.bbox for f in (detector.detect_from_pil(img) or [])]
            if len(faces) > 1:
                embed, bases = subject
                scores = []
                for x1, y1, x2, y2 in faces:
                    pw, ph = (x2 - x1) * 0.6, (y2 - y1) * 0.6
                    crop = img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                                     min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph))))
                    v = embed(crop)
                    scores.append(-1.0 if v is None else float(np.mean([np.dot(v, b) / (np.linalg.norm(v) *
                                                                         np.linalg.norm(b) + 1e-8) for b in bases])))
                best = int(np.argmax(scores))
                m = keep_figure(m, person, faces, best)
                note = f" (kept 1 of {len(faces)} people: the best match, {scores[best] * 100:.0f}%)"
        if float(m.mean()) < MIN_COVER:
            log(f"  {name}: nothing of that kind found{note} - no mask written (it trains as normal)\n")
            counts["no_person"] += 1
            continue
        if old and os.path.normcase(old) != os.path.normcase(out):
            os.remove(old)                       # replaced: one mask per picture
        Image.fromarray((m * 255).round().astype(np.uint8), "L").save(out)
        counts["made"] += 1
        log(f"  {name} -> {os.path.basename(out)}{note}\n")
        if i % 20 == 0:
            log(f"{i} of {len(pics)} pictures\n")
    return counts
