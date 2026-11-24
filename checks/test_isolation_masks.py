"""LoRA Isolation's side-by-sides (#407): two pasted photos, two backgrounds and a seam, taught that two people in one
picture come split down the middle. Each side-by-side now has a mask: the stranger's half from the strangers set (the
person white, the background black), the subject's half whole; it trains at floor 0 whatever the masked-training
setting."""
import os

import numpy as np
import pytest
from PIL import Image

from fizgig.families import multi_person as mp


def _portrait(path, size, person_box):
    """A picture with a grey background and a 'person' block, and its mask beside it when asked."""
    im = Image.new("RGB", size, (120, 120, 120))
    x0, y0, x1, y1 = person_box
    im.paste((200, 150, 120), (x0, y0, x1, y1))
    im.save(path)


def _strangers(tmp_path):
    d = tmp_path / "strangers"
    d.mkdir()
    out = []
    for i in range(3):
        p = d / f"stranger_{i:02d}.png"
        _portrait(p, (768, 768), (200, 150, 568, 768))
        m = Image.new("L", (768, 768), 0)
        m.paste(255, (200, 150, 568, 768))
        m.save(d / f"stranger_{i:02d}-masklabel.png")
        out.append((str(p), f"stranger {i}"))
    return out


@pytest.fixture
def built(tmp_path, monkeypatch):
    people = _strangers(tmp_path)
    monkeypatch.setattr(mp, "strangers", lambda: people)
    monkeypatch.setattr(mp._Faces, "center", lambda self, im: None)
    folder = tmp_path / "subject"
    folder.mkdir()
    for i in range(4):
        _portrait(folder / f"p{i}.png", (600, 800), (100, 200, 500, 800))
        (folder / f"p{i}.txt").write_text("zwxa, a girl", encoding="utf-8")
    own = Image.new("L", (600, 800), 0)                  # p0 has a mask of its own: the side-by-side doesn't use it
    own.paste(255, (250, 350, 350, 450))
    own.save(folder / "p0-masklabel.png")
    out = tmp_path / "out"
    logs = []
    mp.build([{"folder": str(folder), "fused": "zwxagirl", "out": str(out), "index": 1}], log=logs.append)
    return out, logs


def test_every_side_by_side_has_a_mask_of_its_shape(built):
    out, _ = built
    pics = sorted(n for n in os.listdir(out) if n.startswith(mp.PREFIX) and n.endswith(".png")
                  and not n.endswith("-masklabel.png"))
    assert len(pics) == mp.PER_SUBJECT
    for n in pics:
        m = Image.open(out / n.replace(".png", "-masklabel.png"))
        assert m.size == Image.open(out / n).size == (2 * mp.PANEL, mp.PANEL) and m.mode == "L"


def test_the_strangers_half_leaves_out_its_background(built):
    out, _ = built
    for k in range(mp.PER_SUBJECT):
        cap = (out / f"{mp.PREFIX}{k:02d}.txt").read_text(encoding="utf-8")
        m = np.asarray(Image.open(out / f"{mp.PREFIX}{k:02d}-masklabel.png"), dtype=np.float32) / 255
        them = slice(mp.PANEL, None) if "on the right, and" in cap else slice(0, mp.PANEL)
        half = m[:, them]
        assert half[:40, :40].max() == 0.0                # its background's corner: out
        assert half[mp.PANEL - 20:, mp.PANEL // 2 - 10:mp.PANEL // 2 + 10].min() == 1.0   # the person: in


def test_the_subjects_half_trains_whole(built):
    out, _ = built
    for k in range(mp.PER_SUBJECT):
        cap = (out / f"{mp.PREFIX}{k:02d}.txt").read_text(encoding="utf-8")
        m = np.asarray(Image.open(out / f"{mp.PREFIX}{k:02d}-masklabel.png"), dtype=np.float32) / 255
        mine = slice(0, mp.PANEL) if "on the right, and" in cap else slice(mp.PANEL, None)
        assert m[:, mine].min() == 1.0


def test_the_caption_is_as_before_and_masks_rebuild_older_side_by_sides(built):
    out, _ = built
    assert (out / f"{mp.PREFIX}00.txt").read_text(encoding="utf-8").endswith(
        ", two people side by side, two close-up portraits")
    import json
    sig = json.load(open(out / "fizgig_mp_built.json", encoding="utf-8"))
    assert sig["masks"] == 1 and sig["revision"] == mp.REVISION


def test_side_by_sides_weigh_at_floor_zero_whatever_the_setting():
    import torch
    from fizgig.dataset.image_dataset import SIDE_BY_SIDE, side_by_side
    from fizgig.dataset.masks import weights_with_side_by_sides
    assert SIDE_BY_SIDE == mp.PREFIX
    assert side_by_side("mp1:fizgig_mp_03") and side_by_side("fizgig_mp_00") and not side_by_side("s2:holiday")
    m = torch.zeros(2, 1, 4, 4)
    m[:, :, 1:3, 1:3] = 1.0
    off = weights_with_side_by_sides(m, [True, False], 0.1, masked_training=False)
    assert off[0].min() == 0.0 and off[0].max() == 1.0    # the side-by-side: its people only
    assert off[1].min() == 1.0                            # masked training off: the other picture whole
    on = weights_with_side_by_sides(m, [True, False], 0.1, masked_training=True)
    assert on[0].min() == 0.0 and abs(float(on[1].min()) - 0.1) < 1e-6   # on: the other at the run's floor
