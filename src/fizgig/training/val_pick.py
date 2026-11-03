"""The Image Validation images you pick in the form (VALIDATION_PICKED: file names in the training folder) and how the
trainer finds them in the dataset. No torch here: the web form and the launcher use it too."""


MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jxl", ".mp4", ".webm",
              ".mov", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".ogg"}


def _stem(k):
    """A file name or item key -> its comparable stem: folder, a clip's "#still" and a media extension dropped
    (the cache's keys carry no extension; "img.v2" stays "img.v2")."""
    k = str(k).replace("\\", "/").rsplit("/", 1)[-1].split("#", 1)[0]
    base, dot, ext = k.rpartition(".")
    if dot and ("." + ext.lower()) in MEDIA_EXTS:
        k = base
    return k.lower()


def picked_list(value):
    """The VALIDATION_PICKED setting -> a clean list of names (a list, or text one name a line / ';'-separated)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(";", "\n").splitlines()
    out = []
    for v in value:
        v = str(v).strip()
        if v and v not in out:
            out.append(v)
    return out


def resolve_picked(picked, keys):
    """Picked file names/paths -> (the dataset's item keys for them, in pick order; the names not found). A name
    matches a key with the same file stem (img01.png <-> the cache's img01); an exact key match wins."""
    keys = sorted({str(k) for k in keys})
    exact = {k: k for k in keys}
    by_stem = {}
    for k in keys:
        by_stem.setdefault(_stem(k), []).append(k)
    found, missing = [], []
    for p in picked_list(picked):
        k = exact.get(p) or next(iter(by_stem.get(_stem(p), [])), None)
        if k is None:
            missing.append(p)
        elif k not in found:
            found.append(k)
    return found, missing


def validation_every_steps(value):
    """VALIDATION_EVERY_STEPS (How often to check) -> the trainer's --validation_every_steps: 0 = each epoch, N > 0 =
    every N steps. Anything not a whole number above 0 is each epoch (old runs saved 0 for each epoch)."""
    s = str(value if value is not None else "").strip()
    return int(s) if s.isdigit() and int(s) > 0 else 0
