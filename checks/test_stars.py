"""Starred previews (web/stars.py): a star is kept by file name in the samples folder, and a checkpoint shows one when
its own epoch's preview is starred - the same run and epoch, saved within MATCH_S of each other, so a later run that
reuses the name never inherits an earlier run's stars."""
import json
import os

import pytest

from fizgig.web import stars
from fizgig.web.server import sample_info


def _preview(d, name, t):
    p = d / name
    p.write_bytes(b"x")
    os.utime(p, (t, t))
    return str(p)


def test_toggle_keeps_stars_by_name_and_drops_deleted_ones(tmp_path):
    s = tmp_path / "sample"
    s.mkdir()
    a = _preview(s, "three_e000026_00_20261112120000_1.png", 1000)
    b = _preview(s, "three_e000027_00_20261112120100_1.png", 1100)
    assert stars.toggle(a, True) and stars.toggle(b, True)
    assert stars.read(str(s)) == {os.path.basename(a), os.path.basename(b)}
    assert stars.toggle(a, False) is False and stars.read(str(s)) == {os.path.basename(b)}
    os.remove(b)
    stars.toggle(_preview(s, "three_e000028_00_x_1.png", 1200), True)
    assert stars.read(str(s)) == {"three_e000028_00_x_1.png"}                  # the deleted preview dropped out
    assert json.loads((s / stars.FILE).read_text())["stars"] == ["three_e000028_00_x_1.png"]
    with pytest.raises(ValueError):
        stars.toggle(str(s / "gone.png"), True)


def test_a_checkpoint_is_starred_by_its_own_epochs_preview(tmp_path):
    s = tmp_path / "sample"
    s.mkdir()
    stars.toggle(_preview(s, "one_e000010_00_20261112093128_1.png", 5000), True)
    items = [{"name": "one-000010.safetensors", "time": 5001},                # its epoch, the same moment: starred
             {"name": "one-000011.safetensors", "time": 5050},                # another epoch
             {"name": "one-000010.safetensors".replace("one", "two"), "time": 5000},   # another run
             {"name": "one.safetensors", "time": 5030}]                       # the run's final file, near it: starred
    assert stars.starred_checkpoints(str(tmp_path), items, sample_info) == {"one-000010.safetensors", "one.safetensors"}


def test_a_reused_run_name_doesnt_inherit_the_old_runs_stars(tmp_path):
    s = tmp_path / "sample"
    s.mkdir()
    stars.toggle(_preview(s, "one_e000010_00_20261112093128_1.png", 5000), True)        # the first "one"
    later = [{"name": "one-000010.safetensors", "time": 5000 + 3 * 3600}]               # a new "one", hours later
    assert stars.starred_checkpoints(str(tmp_path), later, sample_info) == set()


def test_no_stars_file_means_no_stars(tmp_path):
    assert stars.read(str(tmp_path)) == set()
    assert stars.starred_checkpoints(str(tmp_path), [{"name": "a-000001.safetensors", "time": 1}], sample_info) == set()
