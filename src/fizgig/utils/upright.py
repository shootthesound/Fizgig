"""Pictures as they are meant to be seen. A camera often stores the pixels sideways and writes a rotation flag (EXIF
orientation) that viewers apply; PIL reads the stored pixels as they are. OpenCV already applies the flag, so the
face finders see the picture upright: every PIL read of a training picture goes through here to match."""

ORIENTATION = 0x0112


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
    return eight_bit(upright(Image.open(path)))


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


def upright_size(im):
    """(width, height) as seen, from the header alone: flags 5-8 turn the picture a quarter."""
    w, h = im.size
    return (h, w) if orientation(im) in (5, 6, 7, 8) else (w, h)
