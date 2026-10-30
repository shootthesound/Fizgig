"""Multi-person mode: a LoRA of one or more people where everyone else in a picture stays themselves.

Four parts, all model-agnostic (measured on Krea 2, 15 Oct 2026; branch experiment/dop):
- DOP every second step at twice the weight (the trainer's --dop): no subject in a prompt that doesn't name one.
- 16 side-by-side composites per subject - a training photo beside one of 32 fixed base-model strangers - trained at
  twice the LR on their own steps (--lr_boost): inside one picture, the other person stays a stranger.
- The composite captions name the stranger before the trigger: the text encoders are causal, so words after a
  trigger carry it; the stranger's described first are clean.
- Fused triggers: "zwxem, a woman" is read as "zwxemwoman" (the dataset's caption_replace, the user's .txt files
  untouched), so the subject never shares the generic " woman" token every other woman uses.

The strangers are a fixed, pinned set on Hugging Face, downloaded once into user_data on first use.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
from fizgig.utils.upright import open_upright

from fizgig.dataset.masks import is_mask_file

REPO = "ShootTheSound/fizgig-multi-person-strangers"
REVISION = "8aeb77502234781cb5dca524994c80efcaa51024"      # pinned: a run's composites never change underneath it
PREFIX = "fizgig_mp_"                                       # the composites' file names (the trainer's --lr_boost)
PER_SUBJECT = 16
LR_BOOST = 2.0
PANEL = 512                                                 # each half of a 1024 x 512 composite
TAIL = "two people side by side, two close-up portraits"
_IMG = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# what a subject can be, read from the captions when the folder's "replace it with" box is empty
CLASS_WORDS = ("woman", "man", "girl", "boy", "lady", "guy", "person", "child", "kid", "baby", "teenager", "dog", "cat",
               "puppy", "kitten", "horse")
# the class words that are people: only they get the side-by-side pictures with strangers (people portraits)
PEOPLE_WORDS = ("woman", "man", "girl", "boy", "lady", "guy", "person", "child", "kid", "baby", "teenager")
# the Subjects card's "Subject N is": what a subject is decides what the recipe gives it
KINDS = ("From its captions", "A person", "An animal or thing", "A style or look")
_KIND_KEYS = {"A person": "person", "An animal or thing": "thing", "A style or look": "style"}


def kind_key(label):
    """The card's choice as "person" / "thing" / "style", or "auto" (from its captions)."""
    return _KIND_KEYS.get(str(label or "").strip(), "auto")


def _style(trigger):
    """A style or look: its trigger kept as written (nothing joined) and dropped from the trigger-free caption, so
    the look stays off unless its word is used; no side-by-side pictures."""
    return {"trigger": trigger, "phrase": "", "fused": trigger, "rules": [], "kind": "style"}


def _kind(kind, noun):
    return kind if kind in ("person", "thing") else ("person" if noun.lower() in PEOPLE_WORDS else "thing")


def dop_caption(caption, pairs):
    """DOP's caption (the trainer's): every trigger in it swapped for its class word (dropped when that is empty), or
    None when the caption has none of the triggers - that image has nothing to keep apart."""
    if not caption:
        return None
    out, hit = caption, False
    for trigger, cls in pairs:
        pat = re.compile(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", re.I)
        if pat.search(out):
            hit = True
            out = pat.sub(lambda _m, c=cls: c, out)
    if not hit:
        return None
    out = re.sub(r"\s*,\s*(?:,\s*)+", ", ", out)            # a dropped trigger leaves ", ," or a stray comma behind
    out = re.sub(r"^\s*,\s*|\s*,\s*$", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def fuse(text, rules):
    """A caption or prompt read with a subject's fused trigger (the dataset's caption_replace, in order)."""
    for a, b in rules or ():
        if str(a):
            text = re.sub(r"(?<!\w)" + re.escape(str(a)) + r"(?!\w)", lambda _m, t=str(b): t, text, flags=re.I)
    return text


def caption_trigger(folder, ext=".txt"):
    """The trigger word the folder's captions open with: the word before the first comma that at least half of them
    start with ("zwxem, a woman smiling ..." -> "zwxem"), or "" when there is none."""
    caps = _captions(folder, ext)
    firsts = collections.Counter(m.group(1).lower() for c in caps
                                 if (m := re.match(r"^\s*([A-Za-z0-9_-]+)\s*,", c)))
    if not firsts:
        return ""
    word, n = firsts.most_common(1)[0]
    return word if n * 2 >= len(caps) else ""


def sample_caption(folder, trigger, ext=".txt"):
    """The folder's first caption (by file name) that holds the trigger, for the Subjects card's preview."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(ext.lower()) and not n.startswith(PREFIX))
    except OSError:
        return ""
    pat = re.compile(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", re.I) if trigger else None
    first = ""
    for n in names:
        try:
            with open(os.path.join(folder, n), encoding="utf-8", errors="replace") as f:
                c = f.read().strip()
        except OSError:
            continue
        first = first or c
        if pat is None or pat.search(c):
            return c
    return first


def stranger_example():
    """One stranger's description for the composite caption preview, when the set is already downloaded (the
    preview never downloads)."""
    from fizgig.user_data import user_file
    try:
        with open(os.path.join(user_file("multi_person_strangers"), "strangers.json"), encoding="utf-8") as f:
            return json.load(f)[0]["description"]
    except (OSError, ValueError, KeyError, IndexError):
        return ""


REVISION_FILE = "fizgig_revision.txt"                     # beside the strangers: the REVISION they were fetched at


def _strangers_dir():
    from fizgig.user_data import user_file
    return user_file("multi_person_strangers")


def _local_strangers(d):
    """[(path, description), ...] when the copy in d is complete AND the pinned REVISION (its marker file), else
    None - a set changed on Hugging Face (REVISION bumped) is fetched again, never used stale."""
    try:
        with open(os.path.join(d, REVISION_FILE), encoding="utf-8") as f:
            if f.read().strip() != REVISION:
                return None
        with open(os.path.join(d, "strangers.json"), encoding="utf-8") as f:
            rows = json.load(f)
        if rows and all(os.path.isfile(os.path.join(d, r["file"])) for r in rows):
            return [(os.path.join(d, r["file"]), r["description"]) for r in rows]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def strangers():
    """[(path, description), ...] - the 32 strangers: the copy on disk when it is this REVISION's (no network, so a
    rate limit or no internet never stops a run), else downloaded (about 21 MB) and its revision recorded."""
    d = _strangers_dir()
    got = _local_strangers(d)
    if got:
        return got
    from fizgig.utils import hf_public
    d = hf_public.snapshot(repo_id=REPO, repo_type="dataset", revision=REVISION, local_dir=d)
    with open(os.path.join(d, REVISION_FILE), "w", encoding="utf-8") as f:
        f.write(REVISION + "\n")
    got = _local_strangers(d)
    if not got:
        raise RuntimeError(f"the strangers download from {REPO} is incomplete")
    return got


def download_strangers(log=print):
    """The model downloaders' step (scripts/fetch_models, every family's list): the pinned strangers into
    user_data, so no run has to fetch them at its start. True when they're there afterwards."""
    d = _strangers_dir()
    got = _local_strangers(d)
    if got:
        log(f"  [ok]   multi-person strangers ({len(got)}) - already here: {d}")
        return True
    try:
        n = len(strangers())
        log(f"  [done] multi-person strangers ({n}) -> {d}")
        return True
    except Exception as e:
        log(f"  [fail] multi-person strangers: {e}")
        return False


def _captions(folder, ext):
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(ext.lower())
                 and not n.startswith(PREFIX)]   # Fizgig's side-by-sides in the folder aren't the subject's captions
    except OSError:
        return []
    out = []
    for n in names:
        try:
            with open(os.path.join(folder, n), encoding="utf-8", errors="replace") as f:
                out.append(f.read().strip())
        except OSError:
            pass
    return out


def subject(folder, trigger, cls="", ext=".txt", kind="auto"):
    """One subject's fused trigger: {"trigger", "phrase" ("a woman"), "fused" ("zwxemwoman"), "rules" (the dataset's
    caption_replace), "kind" ("person" / "thing" / "style")}, or a reason it can't be made. The class comes from `cls`
    (the folder's DOP "replace it with") when given, else from the captions' most common "<trigger>, a <word>". kind
    (kind_key): "style" keeps the trigger as written; "auto" is a person when the class is one (PEOPLE_WORDS)."""
    trigger = str(trigger or "").strip()
    if not trigger:
        return None, "no trigger word"
    if kind == "style":
        return _style(trigger), ""
    phrase = str(cls or "").strip()
    if not phrase:
        # the trigger, then "a" / "an", up to three describing words, then a known class word: "zwxem, a young woman"
        # gives "a woman" (never "a young")
        pat = re.compile(r"^\s*" + re.escape(trigger) + r"\s*,?\s*(an?)\s+(?:[A-Za-z'-]+\s+){0,3}?(" +
                         "|".join(CLASS_WORDS) + r")\b", re.I)
        caps = _captions(folder, ext)
        found = collections.Counter(f"{m.group(1).lower()} {m.group(2).lower()}" for c in caps if (m := pat.match(c)))
        if not found:        # no "<trigger>, a woman" opening: the class word the trigger's captions use most
            anywhere = re.compile(r"(?<!\w)(" + "|".join(CLASS_WORDS) + r")(?!\w)", re.I)
            trig = re.compile(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", re.I)
            words = collections.Counter(w.lower() for c in caps if trig.search(c) for w in anywhere.findall(c))
            found = collections.Counter({("an " if w[0] in "aeiou" else "a ") + w: n for w, n in words.items()})
        if not found:
            return None, (f"its captions never say what {trigger} is (woman, man, girl, dog...) - tick Correct a "
                          "trigger word on the Subjects card and type it in that subject's \"what it is\" box, e.g. "
                          "woman. For a style, choose A style or look in that subject's \"Subject ... is\" box.")
        phrase = found.most_common(1)[0][0]
    if not re.match(r"^an?\s", phrase, re.I):
        phrase = ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase
    noun = phrase.split()[-1]
    fused = re.sub(r"\s+", "", trigger) + noun.lower()
    return {"trigger": trigger, "phrase": phrase, "fused": fused, "kind": _kind(kind, noun),
            "rules": [[f"{trigger}, {phrase}", fused], [f"{trigger} {phrase}", fused], [trigger, fused]]}, ""


def subject_as_written(trigger, cls="", kind="auto"):
    """A subject whose captions already hold its joined word (zwxemwoman), used as written - nothing in the captions
    is rewritten. What it is (for the trigger-free caption) comes from `cls`, else from a class word the trigger ends
    with ("zwxemwoman" -> "a woman")."""
    trigger = str(trigger or "").strip()
    if not trigger:
        return None, "no trigger word"
    if kind == "style":
        return _style(trigger), ""
    phrase = str(cls or "").strip()
    if not phrase:
        end = next((w for w in sorted(CLASS_WORDS, key=len, reverse=True)
                    if trigger.lower().endswith(w) and len(trigger) > len(w)), "")
        if not end:
            return None, (f"say what {trigger} is - tick Correct a trigger word on the Subjects card and type it in "
                          "that subject's \"what it is\" box, e.g. woman. For a style, choose A style or "
                          "look in that subject's \"Subject ... is\" box.")
        phrase = end
    if not re.match(r"^an?\s", phrase, re.I):
        phrase = ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase
    return {"trigger": trigger, "phrase": phrase, "fused": trigger, "rules": [],
            "kind": _kind(kind, phrase.split()[-1])}, ""


def _square(im, size=PANEL, center=None):
    """The largest square, no zoom: centred on `center` (x, y) as far as the picture's edges allow, else the middle."""
    from PIL import Image
    w, h = im.size
    m = min(w, h)
    cx, cy = center or (w / 2, h / 2)
    left = int(min(max(cx - m / 2, 0), w - m))
    top = int(min(max(cy - m / 2, 0), h - m))
    return im.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)


FACECROP = "FaceCrop_"          # Image Prep's face crops (dataset/prep.py): already framed on the face, never re-cropped


class _Faces:
    """The largest face's centre in an upright picture, from Image Prep's own detector (CPU), or None when there is no
    face or no detector - then the picture is centre-cropped as before."""

    def __init__(self, log):
        self.log, self.det, self.broken = log, None, False

    def center(self, im):
        if self.broken:
            return None
        try:
            if self.det is None:
                from fizgig.face_utils import FaceDetector
                self.det = FaceDetector()
                if not self.det.available:
                    raise RuntimeError("no face detector installed")
            f = self.det.get_largest(self.det.detect_from_pil(im))
            return f.center if f else None
        except Exception as e:      # no detector: centre crops, as before
            self.broken = True
            self.log(f"[multi-person] face detection unavailable ({e}) - side-by-sides use centre crops")
            return None


def _picks(photos, rng, face_first, faces, load):
    """PER_SUBJECT (photo, face centre) picks, no photo twice until every usable one is used. face_first: Image Prep's
    face crops first, then photos with a face found, photos without one only if nothing else is left; off: every
    photo in one random order. A face crop is never cropped further (only trimmed square); others centre on the face."""
    crops = [n for n in photos if n.startswith(FACECROP)]
    rest = [n for n in photos if not n.startswith(FACECROP)]
    rng.shuffle(crops)
    rng.shuffle(rest)
    out, faceless = [(n, None) for n in crops] if face_first else [], []
    for n in (rest if face_first else rng.sample(photos, len(photos))):
        if len(out) >= PER_SUBJECT:     # enough: the rest are never opened (detection only on what is used)
            break
        if n.startswith(FACECROP):
            out.append((n, None))
            continue
        c = faces.center(load(n))
        (out if c or not face_first else faceless).append((n, c))
    out += faceless                     # face_first: photos without a face only as a last resort
    return [out[k % len(out)] for k in range(PER_SUBJECT)]


def build(subjects, ext=".txt", seed=42, log=print, face_first=True):
    """Each subject's composites: {"folder", "fused", "out", "index"} -> PER_SUBJECT side-by-sides in `out` (a
    training photo next to a stranger, side at random, the caption stranger-first with the fused trigger). The
    training photo's half: see _picks. Rebuilt only when what it was built from changes."""
    from PIL import Image
    people = strangers()
    faces = _Faces(log)
    for s in subjects:
        photos = sorted(n for n in os.listdir(s["folder"]) if n.lower().endswith(_IMG) and not is_mask_file(n))
        if not photos:
            raise ValueError(f"no photos in {s['folder']}")
        stamps = []                              # a photo edited or replaced under the same name rebuilds them
        for n in photos:
            st = os.stat(os.path.join(s["folder"], n))
            stamps.append([n, int(st.st_mtime), st.st_size])
        sig = {"photos": stamps, "fused": s["fused"], "n": PER_SUBJECT, "seed": seed, "index": s["index"],
               "revision": REVISION, "ext": ext, "face_first": bool(face_first), "picking": 2}
        mark = os.path.join(s["out"], "fizgig_mp_built.json")
        try:
            with open(mark, encoding="utf-8") as f:
                if json.load(f) == sig:
                    log(f"[multi-person] {s['fused']}: composites already built - {s['out']}")
                    continue
        except (OSError, ValueError):
            pass
        os.makedirs(s["out"], exist_ok=True)
        for n in os.listdir(s["out"]):
            if n.startswith(PREFIX):
                os.remove(os.path.join(s["out"], n))
        rng = random.Random(seed * 1000 + s["index"])
        order = list(range(len(people)))
        rng.shuffle(order)
        load = lambda n: open_upright(os.path.join(s["folder"], n)).convert("RGB")
        picks = _picks(photos, rng, face_first, faces, load)
        n_crops = sum(1 for n, _c in picks if n.startswith(FACECROP))
        log(f"[multi-person] {s['fused']}: side-by-sides from {n_crops} face crop(s) and "
            f"{PER_SUBJECT - n_crops} other photo(s)" + ("" if face_first else " (face crops not put first)"))
        for k in range(PER_SUBJECT):
            path, desc = people[order[k % len(order)]]
            mine, center = picks[k]
            left = rng.random() < 0.5
            a = _square(load(mine), center=center)
            b = _square(open_upright(path).convert("RGB"))
            sheet = Image.new("RGB", (2 * PANEL, PANEL))
            sheet.paste(a if left else b, (0, 0))
            sheet.paste(b if left else a, (PANEL, 0))
            name = f"{PREFIX}{k:02d}"
            sheet.save(os.path.join(s["out"], name + ".png"))
            me, them = ("left", "right") if left else ("right", "left")
            with open(os.path.join(s["out"], name + ext), "w", encoding="utf-8") as f:
                f.write(f"{desc} on the {them}, and {s['fused']} on the {me}, {TAIL}")
        with open(mark, "w", encoding="utf-8") as f:
            json.dump(sig, f)
        log(f"[multi-person] {s['fused']}: {PER_SUBJECT} composites -> {s['out']}")
