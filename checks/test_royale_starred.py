"""LoRA Royale from starred previews (#401): Max renders "Starred" and the comparison sheet's Epochs "starred" take the
epochs whose preview was starred in the Samples gallery - the checkpoint saved with that preview."""
import os

import pytest

from fizgig.web import royale, stars
from fizgig.web.server import sample_info


def _file(p, t):
    p.write_bytes(b"x")
    os.utime(p, (t, t))
    return str(p)


def _run(tmp_path):
    s = tmp_path / "sample"
    s.mkdir()
    items = []
    for i, ep in enumerate((5, 10, 15, 20)):
        t = 10000 + i * 1000
        items.append([ep, _file(tmp_path / f"one-{ep:06d}.safetensors", t)])
        _file(s / f"one_e{ep:06d}_00_2026111{i}120000_1.png", t + 2)
    return s, items


def test_starred_items_are_the_epochs_with_a_starred_preview_in_run_order(tmp_path):
    s, items = _run(tmp_path)
    stars.toggle(str(s / "one_e000015_00_20261112120000_1.png"), True)
    stars.toggle(str(s / "one_e000005_00_20261110120000_1.png"), True)
    assert [lbl for lbl, _p in royale.starred_items(str(tmp_path), items, sample_info)] == [5, 15]
    assert royale.starred_or_say(str(tmp_path), items, sample_info) == [items[0], items[2]]


def test_no_stars_says_how_to_get_them(tmp_path):
    _s, items = _run(tmp_path)
    assert royale.starred_items(str(tmp_path), items, sample_info) == []
    with pytest.raises(ValueError, match="Samples gallery"):
        royale.starred_or_say(str(tmp_path), items, sample_info)


def test_the_box_words(tmp_path):
    assert all(royale.is_starred(w) for w in ("starred", " Starred ", "favourites", "Favorites", "★"))
    assert not any(royale.is_starred(w) for w in ("", "12", "All", "5,11", "every 4"))
    assert royale.listed_epochs("starred") == ["starred"]       # why the server checks is_starred first


def test_starred_within_a_range(tmp_path):
    s, items = _run(tmp_path)
    for ep, i in ((5, 0), (15, 2), (20, 3)):
        stars.toggle(str(s / f"one_e{ep:06d}_00_2026111{i}120000_1.png"), True)
    got = royale.epoch_range(royale.starred_items(str(tmp_path), items, sample_info), 10, 20)
    assert [lbl for lbl, _p in got] == [15, 20]


def test_a_starred_epoch_without_its_checkpoint_is_named(tmp_path):
    s, items = _run(tmp_path)
    p = s / "one_e000012_00_20261112093128_1.png"                  # epoch 12: never saved as a checkpoint
    p.write_bytes(b"x")
    stars.toggle(str(p), True)
    with pytest.raises(ValueError, match="one epoch 12"):
        royale.starred_or_say(str(tmp_path), items, sample_info)
