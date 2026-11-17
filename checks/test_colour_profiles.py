"""A picture with a colour profile reads in sRGB. PIL reads a picture's numbers as stored, whatever profile is
embedded, so an Adobe RGB / ProPhoto export or an iPhone's Display P3 photo trained (and was captioned, and went
through Image Prep) duller than it looks. open_upright, the read every training picture goes through, now converts
it to sRGB and drops the old profile (a PNG re-save would otherwise label the new numbers with it). Pictures without a
profile or with an sRGB one are returned untouched."""
import struct

import numpy as np
from PIL import Image, ImageCms

from fizgig.utils.upright import open_upright, srgb


def _s15(x):
    return struct.pack(">i", round(x * 65536))


def _profile(desc, red, green, blue, gamma=563 / 256):
    """A small RGB matrix/TRC ICC v2 profile from its D50 colorants and one gamma."""
    xyz = lambda v: b"XYZ " + bytes(4) + b"".join(_s15(c) for c in v)
    curv = b"curv" + bytes(4) + struct.pack(">IH", 1, round(gamma * 256)) + bytes(2)
    text = desc.encode() + b"\0"
    dsc = b"desc" + bytes(4) + struct.pack(">I", len(text)) + text + struct.pack(">II", 0, 0) + bytes(3 + 67)
    dsc += bytes(-len(dsc) % 4)
    tags = [(b"desc", dsc), (b"wtpt", xyz((0.9642, 1.0, 0.8249))), (b"rXYZ", xyz(red)), (b"gXYZ", xyz(green)),
            (b"bXYZ", xyz(blue)), (b"rTRC", curv), (b"gTRC", curv), (b"bTRC", curv)]
    offset, table, data = 128 + 4 + 12 * len(tags), b"", b""
    for sig, body in tags:
        table += sig + struct.pack(">II", offset + len(data), len(body))
        data += body
    size = offset + len(data)
    header = (struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ " + bytes(12)
              + b"acsp" + bytes(24) + struct.pack(">I", 0) + _s15(0.9642) + _s15(1.0) + _s15(0.8249) + bytes(48))
    return header + struct.pack(">I", len(tags)) + table + data


ADOBE = _profile("Adobe RGB (1998) test", (0.6097, 0.3111, 0.0195), (0.2053, 0.6257, 0.0609),
                 (0.1492, 0.0632, 0.7446))
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _picture(mode, colour, icc):
    im = Image.new(mode, (8, 8), colour)
    if icc:
        im.info["icc_profile"] = icc
    return im


def test_an_adobe_rgb_picture_reads_as_srgb(tmp_path):
    for fmt in ("png", "jpeg"):
        p = tmp_path / f"adobe.{fmt}"
        Image.new("RGB", (16, 16), (60, 180, 60)).save(p, icc_profile=ADOBE, quality=100)
        im = open_upright(p)
        assert im.mode == "RGB" and "icc_profile" not in im.info
        r, g, b = np.asarray(im)[8, 8].astype(int)
        assert g - r > 180 - 60 + 10           # the green as a viewer shows it: more saturated than the stored numbers


def test_alpha_kept_and_other_pictures_untouched(tmp_path):
    rgba = srgb(_picture("RGBA", (60, 180, 60, 77), ADOBE))
    assert rgba.mode == "RGBA" and np.asarray(rgba)[0, 0, 3] == 77 and "icc_profile" not in rgba.info
    for im in (_picture("RGB", (60, 180, 60), None), _picture("RGB", (60, 180, 60), SRGB),
               _picture("L", 77, ADOBE), _picture("RGB", (1, 2, 3), b"not a profile")):
        assert srgb(im) is im
    p = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), (60, 180, 60)).save(p)
    assert tuple(np.asarray(open_upright(p))[0, 0]) == (60, 180, 60)


def test_a_re_save_is_not_labelled_with_the_old_profile(tmp_path):
    p, q = tmp_path / "adobe.png", tmp_path / "resaved.png"
    Image.new("RGB", (8, 8), (60, 180, 60)).save(p, icc_profile=ADOBE)
    open_upright(p).save(q)                     # as Image Prep re-saves a picture
    assert "icc_profile" not in Image.open(q).info
