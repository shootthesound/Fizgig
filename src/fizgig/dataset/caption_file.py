"""A caption file's words as training reads them, whatever wrote it. Light (no torch): Start's checks read captions
too, and must read them the way the training set does (image_dataset re-exports both)."""
import logging

logger = logging.getLogger(__name__)


def decode_caption(raw: bytes, caption_path: str, quiet: bool = False) -> str:
    """Decode a caption file's bytes, tolerating encodings other than UTF-8.

    Captions written by the Captions tab are UTF-8, but datasets assembled elsewhere
    routinely carry Windows-1252 text — one curly apostrophe (0x92) in one file out of a
    hundred used to abort the whole caching run with a bare UnicodeDecodeError deep in a
    worker thread that named no file at all.

    Order: UTF-16 if there's a BOM (Notepad's "Unicode" save), then UTF-8 (BOM tolerated),
    then cp1252, then latin-1 — which maps every byte and so cannot fail. Anything past
    UTF-8 is logged with the path, so a genuinely mis-encoded caption is still visible
    instead of being silently mangled into the training data. quiet: no warnings (a check that only looks).
    """
    warn = logger.debug if quiet else logger.warning
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16")
            warn("Caption %s is UTF-16 — decoded, but re-save it as UTF-8.", caption_path)
            return text
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue  # cp1252 leaves a handful of bytes undefined; latin-1 won't.
        warn("Caption %s is not valid UTF-8 — decoded as %s. Re-save it as UTF-8 to "
             "silence this warning.", caption_path, encoding)
        return text
    warn("Caption %s could not be decoded in any known encoding — undecodable bytes "
         "replaced.", caption_path)
    return raw.decode("utf-8", errors="replace")


def read_caption(path: str) -> str:
    """A caption file's words exactly as training reads them (decode_caption), without its warnings: for the checks
    that only look at captions (Start's). OSError if it isn't there."""
    with open(path, "rb") as f:
        return decode_caption(f.read(), path, quiet=True)
