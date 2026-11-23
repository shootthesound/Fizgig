"""Pictures as they are meant to be seen. A camera often stores the pixels sideways and writes a rotation flag (EXIF
orientation) that viewers apply; PIL reads the stored pixels as they are. OpenCV already applies the flag, so the
face finders see the picture upright: every PIL read of a training picture goes through here to match. The read
also gives 16-bit greyscale as 8-bit (eight_bit) and a picture with a colour profile in sRGB (srgb)."""
import struct

ORIENTATION = 0x0112

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
# what a PNG could hold an orientation in: its own eXIf chunk, or a text chunk (where XMP and ImageMagick's
# "Raw profile type exif" are carried)
_ORIENT_CHUNKS = (b"eXIf", b"tEXt", b"zTXt", b"iTXt")
_MAX_CHUNKS = 100_000         # a chunk-count ceiling: past it the walk stops and the picture is asked the slow way


def orientation(im):
    try:
        return int(im.getexif().get(ORIENTATION, 1) or 1)
    except Exception:
        return 1


def upright(im):
    """The open picture with its rotation flag applied (the flag dropped), or the same picture when it has none."""
    if orientation(im) in (2, 3, 4, 5, 6, 7, 8):
        from PIL import ImageOps
        return ImageOps.exif_transpose(im)
    return im


def open_upright(path):
    from PIL import Image
    return srgb(eight_bit(upright(Image.open(path))))


_TRANSFORMS = {}        # (profile hash, mode) -> transform, or None for an sRGB profile


def srgb(im):
    """A picture with an embedded colour profile other than sRGB (an iPhone's Display P3, an Adobe RGB or ProPhoto
    export, a CMYK file) converted to sRGB, the colours a viewer shows. PIL reads the numbers as stored, so such a
    picture trained, was captioned and went through Image Prep duller than it looks (ProPhoto much duller). The
    result carries no profile, so a re-save (Image Prep) isn't labelled with the old one. A picture without a
    profile, with an sRGB one, or that can't be converted is returned untouched (the same object)."""
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in ("RGB", "RGBA", "CMYK"):
        return im
    try:
        import hashlib
        import io
        from PIL import ImageCms
        key = (hashlib.sha1(icc).digest(), im.mode)
        if key not in _TRANSFORMS:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if "srgb" in (ImageCms.getProfileDescription(src) or "").lower().replace(" ", ""):
                _TRANSFORMS[key] = None
            else:
                _TRANSFORMS[key] = ImageCms.buildTransform(src, ImageCms.createProfile("sRGB"), im.mode,
                                                           "RGBA" if im.mode == "RGBA" else "RGB")
        if _TRANSFORMS[key] is None:
            return im
        out = ImageCms.applyTransform(im, _TRANSFORMS[key])
    except Exception:
        return im
    out.info = {k: v for k, v in im.info.items() if k != "icc_profile"}
    return out


def eight_bit(im):
    """A 16-bit (or 32-bit integer) greyscale picture as 8-bit greyscale over its full range; any other picture as it is.
    convert("RGB") clips every value above 255, so a 16-bit grey PNG (a black-and-white photo saved from Photoshop or
    GIMP) trained, captioned and went through Image Prep as pure white. 16-bit colour needs nothing: PIL reads it as
    8-bit RGB."""
    if im.mode not in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        return im
    import numpy as np
    from PIL import Image
    a = np.asarray(im)
    if im.mode == "I" and a.size and 0 <= a.min() and a.max() <= 255:     # 8-bit values held as 32-bit
        return Image.fromarray(a.astype(np.uint8), "L")
    return Image.fromarray((a.astype(np.float32) / 257.0 + 0.5).clip(0, 255).astype(np.uint8), "L")


def png_carries_orientation(path) -> bool:
    """Whether a PNG file holds anything a rotation flag could be read from.

    Pillow's PNG reader only puts exif in info when it read the chunk before the first IDAT, and
    PngImageFile.getexif() inflates the whole picture to find out - 17.8 ms a file against 44 us for the
    size itself (both measured over the 518 pictures of one training folder). Walking the chunk headers
    instead reads 8 bytes a chunk and seeks over the data, so the cost never follows the size of the
    picture: 137 us a file, median 20 chunks, and it sees a chunk written after the image data, which is
    exactly what open() misses.

    True whenever the walk can't answer for certain - an unreadable or unexpected file is asked the slow
    way rather than assumed to have no rotation."""
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIG:
                return True
            for _ in range(_MAX_CHUNKS):
                head = f.read(8)
                if len(head) < 8:
                    return True                                     # ended without IEND: no answer here
                length, typ = struct.unpack(">I4s", head)
                if typ in _ORIENT_CHUNKS:
                    return True
                if typ == b"IEND":
                    return False                                    # walked the whole file: nothing to read
                f.seek(length + 4, 1)                               # its data and its CRC
            return True                                             # chunk ceiling: the walk gave up
    except Exception:
        return True


def upright_size(im, path=None):
    """(width, height) as seen, from the header alone: flags 5-8 turn the picture a quarter.

    path: the picture's own file, which lets a PNG that carries no rotation anywhere answer without
    inflating itself (see png_carries_orientation). Omitted, every picture is asked the slow, certain
    way, so callers that only have an open file are exactly as they were."""
    w, h = im.size
    if path is not None and im.format == "PNG" and "exif" not in im.info and not png_carries_orientation(path):
        return (w, h)
    return (h, w) if orientation(im) in (5, 6, 7, 8) else (w, h)
