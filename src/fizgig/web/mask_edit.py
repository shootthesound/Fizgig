"""Image Prep's mask editor (the masks card's Edit masks): each picture of the training folder with its mask, painted
in the page and saved back as photo-masklabel.png. Pictures go out upright and masks are saved upright, as training
reads them (a phone photo's rotation flag applied), so a mask never ends up sideways. Only the folder's own pictures
are served or written, by name."""
import base64
import io
import os

import numpy as np
from PIL import Image

from fizgig.dataset.automask import pictures
from fizgig.dataset.masks import MASK_SUFFIX, MIN_COVER, folder_names, mask_for, read_mask
from fizgig.utils.upright import upright


def _picture(folder, name):
    """The picture called `name` in the training folder (a mask file never), or ValueError."""
    for p in pictures(folder):
        if os.path.basename(p) == name:
            return p
    raise ValueError(f"{name} isn't a picture in the training folder.")


def listing(folder):
    """[{"name", "mask"}] for every picture in the folder, in name order."""
    names = folder_names(folder)
    return [{"name": os.path.basename(p), "mask": bool(mask_for(p, names))} for p in pictures(folder)]


def _open_upright(path):
    from fizgig.utils.image import flatten
    with Image.open(path) as im:
        return flatten(upright(im))


def picture_jpeg(folder, name):
    """The picture, upright, as JPEG bytes at its own size."""
    buf = io.BytesIO()
    _open_upright(_picture(folder, name)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def mask_png(folder, name):
    """The picture's mask as grayscale PNG bytes at the upright picture's size, or None when it has none."""
    pic = _picture(folder, name)
    m = mask_for(pic)
    if not m:
        return None
    with Image.open(pic) as im:
        size = upright(im).size
    mask = read_mask(m)
    if mask.size != size:
        mask = mask.resize(size, Image.BILINEAR)         # a smaller mask is stretched to fit, as training does
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return buf.getvalue()


def sam_click(folder, name, x, y, model_path):
    """The object SAM 3.1 finds at (x, y) on the upright picture -> grayscale PNG bytes at the picture's size."""
    from fizgig.dataset import sammask
    img = _open_upright(_picture(folder, name))
    m = sammask.point(img, x, y, model_path)
    buf = io.BytesIO()
    Image.fromarray(m.astype(np.uint8) * 255, "L").save(buf, "PNG")
    return buf.getvalue()


def save(folder, name, png_data_url):
    """Write the painted mask (a PNG data URL, white = train) as the picture's photo-masklabel.png, replacing any mask
    it had. An all-black mask removes the file instead (the picture then trains as normal). -> {"saved"|"removed"}."""
    pic = _picture(folder, name)
    raw = base64.b64decode(png_data_url.split(",", 1)[1] if "," in png_data_url else png_data_url)
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        mask = im.convert("RGBA")
    # the page's canvas is white on transparent black: brightness where painted
    black = Image.new("RGBA", mask.size, (0, 0, 0, 255))
    mask = Image.alpha_composite(black, mask).convert("L")
    with Image.open(pic) as im:
        size = upright(im).size
    if mask.size != size:
        mask = mask.resize(size, Image.BILINEAR)
    old = mask_for(pic)
    out = os.path.join(folder, os.path.splitext(name)[0] + MASK_SUFFIX + ".png")
    if old and old.lower().endswith(".png"):
        out = old                                         # same file, so a running run keeps finding it
    if float(np.asarray(mask, np.float32).mean()) / 255.0 < MIN_COVER:
        if old:
            os.remove(old)
        return {"removed": True}
    tmp = out + ".part"
    mask.save(tmp, "PNG")
    os.replace(tmp, out)
    if old and os.path.normcase(old) != os.path.normcase(out):
        os.remove(old)                                    # one mask per picture
    return {"saved": True}
