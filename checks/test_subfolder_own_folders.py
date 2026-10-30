"""Training on sub-folders never trains the folders Fizgig makes inside a training folder: Image Prep's Keep them
safe (originals, and originals_2, _3... on later runs), Exclude by look and the desktop's Remove, in any case. The run log names what it left out in one line."""
import os
import sys

import pytest
from PIL import Image

from fizgig.dataset.prep import Prep
from fizgig.families import form, launch
from fizgig.families.registry import get as get_family

def _pic(path, caption="a photo", size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (90, 40, 20)).save(path)
    if caption is not None:
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(caption)


def _st(d, **kw):
    return {"image_folder": str(d), "LORA_OUTPUT_DIR": str(d / "out"), "megapixels": 0.25,
            "cache_root": str(d.parent / "cache"), "subfolders": True, **kw}


@pytest.mark.parametrize("name", ["originals", "originals_2", "originals_17", "Originals", "ORIGINALS_3",
                                  "excluded_by_look", "Excluded_By_Look", "removed", "Removed"])
def test_fizgigs_own_folders_by_name_any_case(name):
    assert launch.is_own_subfolder(name)


@pytest.mark.parametrize("name", ["originals_x", "my_originals", "originals2", "outfits", "removed_bg", "red"])
def test_other_folders_are_not_fizgigs(name):
    assert not launch.is_own_subfolder(name)


def test_numbered_originals_removed_and_nested_ones_are_skipped_and_named(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "a.png"))
    _pic(str(d / "outfits" / "red" / "r.png"))
    _pic(str(d / "outfits" / "red" / "Originals" / "r.png"))     # a sub-folder's own keep-safe folder, any case
    for own in ("originals", "originals_2", "Removed", "excluded_by_look"):
        _pic(str(d / own / "x.png"), caption=None)                  # originals have no captions
    skipped = []
    got = [rel for rel, _f in launch.subfolders(str(d), (), skipped)]
    assert got == ["outfits/red"]
    assert skipped == ["Removed", "excluded_by_look", "originals", "originals_2", "outfits/red/Originals"]
    toml = launch.dataset_toml(get_family("krea2"), _st(d))
    dirs = [ln for ln in toml.splitlines() if ln.startswith("image_directory")]
    assert len(dirs) == 2 and dirs[1].endswith('/data/outfits/red"')
    probs = launch.start_problems(get_family("krea2"), _st(d))          # uncaptioned originals don't block Start
    assert not any("originals" in p.lower() or "removed" in p.lower() or "sub-folder" in p for p in probs)
    line = launch.own_skipped_line(get_family("krea2"), _st(d))
    assert line.startswith("[sub-folders] Not trained (Fizgig's own folders): Removed, excluded_by_look, originals")
    assert "\n" not in line
    assert launch.own_skipped_line(get_family("krea2"), _st(d, subfolders=False)) == ""      # tick off: no line


def test_a_real_keep_safe_prep_twice_then_remove_trains_none_of_them(tmp_path):
    d = tmp_path / "data"
    for n in "abc":
        _pic(str(d / f"{n}.jpg"), size=(1600, 1200))
    _pic(str(d / "outfits" / "red" / "r.jpg"))
    Prep(lambda s: None).run("Resize Only", str(d), 1.0, replace=False)      # Keep them safe -> originals/
    for n in "de":
        _pic(str(d / f"{n}.jpg"), size=(1600, 1200))
    Prep(lambda s: None).run("Resize Only", str(d), 1.0, replace=False)      # the next prep -> originals_2/
    from fizgig.web import server
    server.remove_from_dataset(str(d), "a.png", ".txt")                      # the desktop's Remove -> removed/
    assert sorted(os.listdir(d / "originals")) == ["a.jpg", "b.jpg", "c.jpg"]
    assert sorted(os.listdir(d / "originals_2")) == ["d.jpg", "e.jpg"]
    assert os.path.isfile(d / "removed" / "a.png")
    assert [r for r, _f in launch.run_subfolders(get_family("krea2"), _st(d))] == ["outfits/red"]


def test_masks_and_captions_in_an_included_subfolder_still_count(tmp_path):
    from fizgig.dataset.masks import MASK_SUFFIX
    d = tmp_path / "data"
    _pic(str(d / "a.png"))
    _pic(str(d / "outfits" / "red" / "r.png"), caption="zwxem in red")
    Image.new("L", (32, 32), 255).save(d / "outfits" / "red" / f"r{MASK_SUFFIX}.png")
    _pic(str(d / "originals" / "o.png"))
    Image.new("L", (32, 32), 255).save(d / "originals" / f"o{MASK_SUFFIX}.png")
    st = _st(d)
    folders = launch.mask_folders(get_family("krea2"), st)
    assert str(d / "outfits" / "red") in folders and not any("originals" in f for f in folders)
    assert launch._media_count(str(d / "outfits" / "red")) == 1        # the mask isn't a picture
    toml = launch.dataset_toml(get_family("krea2"), st)
    assert 'item_prefix = "outfits/red/"' in toml


def test_the_run_log_gets_one_line_naming_what_was_left_out(tmp_path):
    krea = get_family("krea2")
    d = tmp_path / "imgs"
    _pic(str(d / "a.png"), caption="zwxem a photo")
    _pic(str(d / "outfits" / "r.png"), caption="zwxem in red")
    _pic(str(d / "originals" / "a.jpg"), caption=None)
    _pic(str(d / "removed" / "b.png"))
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)
    inp = form.inputs(krea, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "TRAIN_SUBFOLDERS": True},
                      models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(d),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    p = launch.plan(krea, inp)
    assert p.problems == []
    lines = [ln for ln in p.console if ln.startswith("[sub-folders]")]
    assert lines == ["[sub-folders] Not trained (Fizgig's own folders): originals, removed"]
    assert "/imgs/outfits" in p.dataset_toml and "/imgs/originals" not in p.dataset_toml


def test_every_keep_safe_folder_prep_can_make_is_skipped(tmp_path):
    """Image Prep's originals, originals_2, _3..."""
    d = tmp_path / "data"
    os.makedirs(d)
    p = Prep(lambda s: None)
    for n in range(1, 4):
        o = p._originals_dir(str(d))
        assert launch.is_own_subfolder(o)
        _pic(os.path.join(o, "x.png"))              # a picture and its caption backup
    assert sorted(os.listdir(d)) == ["originals", "originals_2", "originals_3"]
    assert launch.subfolders(str(d)) == []
