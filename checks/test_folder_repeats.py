"""Repeats from folder names (#300): under Also train on its sub-folders, a tick (off by default) reads a sub-folder
named "10_Laffey_base" as 10 repeats an epoch. Off, a number in a name means nothing and the run is as before."""
import os

import pytest
from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family


def _pic(path, caption="a photo"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(caption)


@pytest.fixture
def tree(tmp_path):
    d = tmp_path / "data"
    for i in range(4):
        _pic(str(d / "10_base" / f"b{i}.png"))
    for i in range(8):
        _pic(str(d / "2_outfit" / f"o{i}.png"))
    _pic(str(d / "plain" / "p.png"))
    return d


def _st(d, **kw):
    return {"image_folder": str(d), "LORA_OUTPUT_DIR": str(d / "out"), "megapixels": 0.25,
            "cache_root": str(d.parent / "cache"), "subfolders": True, **kw}


@pytest.mark.parametrize("name,want", [("10_base", (10, 10)), ("base", (1, None)), ("0_x", (1, 0)),
                                       ("500_x", (100, 500)), ("10_", (1, None)), ("a/b/3_c", (3, 3)),
                                       ("3_a/b", (1, None)), ("10-base", (1, None))])
def test_the_number_comes_from_the_folders_own_name(name, want):
    assert launch.folder_repeats(name) == want


def test_off_by_default_and_off_is_unchanged(tree):
    assert form.DEFAULTS["TRAIN_SUBFOLDER_REPEATS"] is False
    desc = get_family("krea2")
    off = launch.dataset_toml(desc, _st(tree))
    assert "num_repeats = 10" not in off and "num_repeats = 2" not in off
    assert off == launch.dataset_toml(desc, _st(tree, subfolder_repeats=False))
    assert launch._training_items(desc, _st(tree)) == 13
    assert launch.repeats_line(desc, _st(tree)) == ""


def test_on_each_block_gets_its_repeats(tree):
    desc = get_family("krea2")
    toml = launch.dataset_toml(desc, _st(tree, subfolder_repeats=True))
    blocks = toml.split("[[datasets]]")[1:]
    reps = {b.split('item_prefix = "')[1].split('"')[0]: ("num_repeats = 10" in b, "num_repeats = 2" in b,
                                                            "num_repeats" in b) for b in blocks}
    assert reps["10_base/"] == (True, False, True)
    assert reps["2_outfit/"] == (False, True, True)
    assert reps["plain/"] == (False, False, False)
    assert launch._training_items(desc, _st(tree, subfolder_repeats=True)) == 4 * 10 + 8 * 2 + 1
    line = launch.repeats_line(desc, _st(tree, subfolder_repeats=True))
    assert line.startswith("[repeats] 3 folders, 57 steps an epoch from 13 pictures: 10_base x10 (70%)")


def test_needs_the_subfolder_tick(tree):
    desc = get_family("krea2")
    assert not launch.repeats_on(desc, _st(tree, subfolders=False, subfolder_repeats=True))
    assert not launch.repeats_on(desc, _st(tree, subfolder_repeats=True, multiconcept=True))


def test_row_shows_only_under_the_ticked_subfolder_row(tree):
    desc = get_family("klein")
    assert "TRAIN_SUBFOLDER_REPEATS" in form.state(desc, {}, str(tree))["hidden"]
    assert "TRAIN_SUBFOLDER_REPEATS" in form.state(desc, {"TRAIN_SUBFOLDER_REPEATS": True}, str(tree))["hidden"]
    st = form.state(desc, {"TRAIN_SUBFOLDERS": True}, str(tree))
    assert "TRAIN_SUBFOLDER_REPEATS" not in st["hidden"]
    assert "look like repeat counts" in st["notes"]["TRAIN_SUBFOLDERS"]      # the hint, tick still off
    assert "TRAIN_SUBFOLDERS" not in st["actions"]                            # the tick is right below
    st = form.state(desc, {"TRAIN_SUBFOLDERS": True, "TRAIN_SUBFOLDER_REPEATS": True}, str(tree))
    assert "look like repeat counts" not in st["notes"]["TRAIN_SUBFOLDERS"]
    assert "TRAIN_SUBFOLDERS" not in st["actions"]                            # the table lists the folders
    note = st["notes"]["TRAIN_SUBFOLDER_REPEATS"]
    assert note.startswith("Each epoch now shows the model 57 pictures: the 13 in the folders")
    assert "| 10_base | 4 | 10x | 40 | 70% |" in note
    assert "|! plain | 1 | 1x | 1 | 2% |" in note and "Marked:" in note      # under 3%: flagged
    assert "An epoch is now 4.4x as long. Your 12 epochs (Max Epochs) show 684 pictures instead of 156" in note and "set Max Epochs to 3" in note


def test_unticked_numbered_folders_offer_both_ticks(tree):
    st = form.state(get_family("klein"), {}, str(tree))
    assert "Their names look like repeat counts" in st["notes"]["TRAIN_SUBFOLDERS"]
    assert st["actions"]["TRAIN_SUBFOLDERS"]["values"] == {"TRAIN_SUBFOLDERS": True, "TRAIN_SUBFOLDER_REPEATS": True}


def test_plain_names_get_a_concrete_rename(tmp_path):
    d = tmp_path / "d"
    for i in range(20):
        _pic(str(d / "red" / f"{i}.png"))
    for i in range(8):
        _pic(str(d / "blue" / f"{i}.png"))
    note = form.state(get_family("klein"), {"TRAIN_SUBFOLDERS": True, "TRAIN_SUBFOLDER_REPEATS": True},
                      str(d))["notes"]["TRAIN_SUBFOLDER_REPEATS"]
    assert "rename the folder blue to 2_blue" in note and "(8 x 2 = 16, close to red's 20)" in note


def test_hidden_without_subfolders(tmp_path):
    d = tmp_path / "flat"
    _pic(str(d / "a.png"))
    assert "TRAIN_SUBFOLDER_REPEATS" in form.state(get_family("klein"), {"TRAIN_SUBFOLDERS": True}, str(d))["hidden"]
