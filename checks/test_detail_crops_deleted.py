"""Detail crops you delete stay deleted: Start (detail_crops.make) never makes them again, at any size, while the
full pictures keep training; put one back and it trains again."""
import json
import os

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from fizgig.dataset import detail_crops as dc  # noqa: E402


def _folder(tmp_path):
    d = tmp_path / "pics"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)).save(d / f"p{i}.png")
        (d / f"p{i}.txt").write_text(f"picture {i}")
    dc.switch(str(d), True)
    return str(d)


def _crops(folder, mp):
    d = dc.size_dir(folder, mp)
    return sorted(n for n in os.listdir(d) if n.endswith(".png"))


def test_a_deleted_crop_is_never_made_again(tmp_path):
    f = _folder(tmp_path)
    q = lambda *a: None                                          # noqa: E731
    dc.make(f, 0.25, log=q)
    dc.make(f, 0.5, log=q)
    assert _crops(f, 0.25) == _crops(f, 0.5) == ["p0.png", "p1.png", "p2.png"]
    os.remove(os.path.join(dc.size_dir(f, 0.5), "p1.png"))        # deleted at 0.5 MP (in Explorer, say)
    assert dc.deleted(f, apply=False) == {"p1"}                   # the check before Start: nothing moved
    assert os.path.isfile(os.path.join(dc.size_dir(f, 0.25), "p1.png"))
    todo = dc.make(f, 0.25, log=q)                                # Start, at another size
    assert _crops(f, 0.25) == ["p0.png", "p2.png"]                # gone there too: moved aside, never deleted
    assert os.path.isfile(os.path.join(dc.size_dir(f, 0.25), dc.REMOVED, "p1.png"))
    assert not any(p.endswith("p1.png") for p in todo)            # and never sent to the captioner
    dc.make(f, 0.5, log=q)
    assert _crops(f, 0.5) == ["p0.png", "p2.png"]                 # not remade where it was deleted
    with open(os.path.join(f, dc.DIR, dc.DELETED), encoding="utf-8") as fh:
        assert json.load(fh) == {"p1": "0.5mp"}
    assert "p1.png" in dc.pictures(f)                             # the full picture still trains
    dc.make(f, 0.36, log=q)                                       # a new size: never made there either
    assert _crops(f, 0.36) == ["p0.png", "p2.png"]


def test_the_captions_tab_remove_and_its_undo(tmp_path):
    f = _folder(tmp_path)
    q = lambda *a: None                                          # noqa: E731
    dc.make(f, 0.25, log=q)
    dc.make(f, 0.5, log=q)
    big = dc.size_dir(f, 0.5)
    os.makedirs(os.path.join(big, dc.REMOVED))
    os.replace(os.path.join(big, "p2.png"), os.path.join(big, dc.REMOVED, "p2.png"))   # Remove, in the Captions tab
    dc.make(f, 0.25, log=q)
    assert dc.deleted(f, apply=False) == {"p2"} and "p2.png" not in _crops(f, 0.25)
    os.replace(os.path.join(big, dc.REMOVED, "p2.png"), os.path.join(big, "p2.png"))   # Undo
    dc.make(f, 0.25, log=q)
    assert dc.deleted(f, apply=False) == set()
    assert _crops(f, 0.25) == ["p0.png", "p1.png", "p2.png"]      # made again at the other size
