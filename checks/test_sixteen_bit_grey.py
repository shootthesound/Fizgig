"""A 16-bit greyscale picture reads as itself, not white. PIL opens a 16-bit grey PNG as mode I;16 and convert("RGB")
clips every value above 255, so such a picture (a black-and-white photo saved 16-bit from Photoshop or GIMP) trained,
was captioned and went through Image Prep as pure white (99.6% of a full-range gradient came out 255). open_upright,
the read every training picture goes through, now scales it to 8-bit over its full range; other pictures pass through
untouched."""
import numpy as np
from PIL import Image

from fizgig.utils.upright import eight_bit, open_upright


def _as_training_reads(path):
    img = open_upright(path)                      # image_dataset's own two lines
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if img.mode in ("LA", "PA") or "transparency" in img.info else "RGB")
    return np.asarray(img)


def test_a_16bit_grey_png_is_not_white(tmp_path):
    ramp = np.linspace(0, 65535, 256 * 256).reshape(256, 256).astype(np.uint16)
    p = tmp_path / "grey16.png"
    Image.fromarray(ramp).save(p)
    assert Image.open(p).mode == "I;16"
    a = _as_training_reads(p)
    assert a.shape == (256, 256, 3)
    assert abs(float(a.mean()) - 127.5) < 1.0 and a.min() == 0 and a.max() == 255
    assert float((a == 255).all(-1).mean()) < 0.01          # was 0.996


def test_other_pictures_pass_through(tmp_path):
    for mode, value in (("RGB", (10, 20, 30)), ("L", 77), ("RGBA", (1, 2, 3, 4)), ("P", 5)):
        im = Image.new(mode, (8, 8), value)
        assert eight_bit(im) is im
    small = Image.fromarray(np.full((4, 4), 200, dtype=np.int32), "I")       # 8-bit values held as 32-bit
    assert eight_bit(small).mode == "L" and np.asarray(eight_bit(small)).max() == 200
    p = tmp_path / "grey8.png"
    Image.new("L", (16, 16), 128).save(p)
    assert float(_as_training_reads(p).mean()) == 128.0
