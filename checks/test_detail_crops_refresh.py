"""Detail crops follow their picture's mask, and a picture replaced by one too small for the crop leaves nothing of its
old crop behind (both reported by @mabseyuk)."""
import os
import time

import numpy as np
import pytest
from PIL import Image

from fizgig.dataset import detail_crops
from fizgig.dataset.masks import MASK_SUFFIX


def _crops(folder, mp):
    return detail_crops.size_dir(folder, mp)


def _setup(tmp_path, mp=0.25):
    s = detail_crops.side(mp)
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (s * 2, s * 2, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    Image.new("L", (s * 2, s * 2), 255).save(tmp_path / f"a{MASK_SUFFIX}.png")
    detail_crops.make(str(tmp_path), mp, log=lambda t: None)
    out = _crops(str(tmp_path), mp)
    (open(os.path.join(out, "a.txt"), "w")).write("zwxem, a crop")
    return s, out


def _later(path):
    t = os.path.getmtime(path) + 5
    os.utime(path, (t, t))


def test_an_edited_mask_remakes_the_mask_crop_and_keeps_the_caption(tmp_path):
    s, out = _setup(tmp_path)
    Image.new("L", (s * 2, s * 2), 0).save(tmp_path / f"a{MASK_SUFFIX}.png")
    _later(tmp_path / f"a{MASK_SUFFIX}.png")
    detail_crops.make(str(tmp_path), 0.25, log=lambda t: None)
    assert np.asarray(Image.open(os.path.join(out, "a-masklabel.png"))).max() == 0
    assert open(os.path.join(out, "a.txt")).read() == "zwxem, a crop"         # same crop: its caption still fits


def test_a_deleted_mask_removes_the_mask_crop(tmp_path):
    _s, out = _setup(tmp_path)
    os.remove(tmp_path / f"a{MASK_SUFFIX}.png")
    detail_crops.make(str(tmp_path), 0.25, log=lambda t: None)
    assert not os.path.exists(os.path.join(out, "a-masklabel.png"))
    assert os.path.exists(os.path.join(out, "a.png"))


def test_a_picture_now_too_small_leaves_no_crop(tmp_path):
    s, out = _setup(tmp_path)
    Image.new("RGB", (s // 2, s // 2), (9, 9, 9)).save(tmp_path / "a.png")
    _later(tmp_path / "a.png")
    detail_crops.make(str(tmp_path), 0.25, log=lambda t: None)
    for n in ("a.png", "a-masklabel.png", "a.txt"):
        assert not os.path.exists(os.path.join(out, n)), n
