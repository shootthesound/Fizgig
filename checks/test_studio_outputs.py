"""Studio outputs: Save and Save every render write into Preferences' Studio outputs folder (images / videos / songs),
named <model>_<lora>_<original|tweaked>_seed<seed>_<time>, never over another file; Save every render saves the
Baseline only when it rendered again; the three folders ship (their .gitkeep) and saved renders are git-ignored."""
import os
import subprocess

import pytest

studio = pytest.importorskip("fizgig.web.studio")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_names_are_unique_and_by_kind(tmp_path):
    a = studio.output_path(str(tmp_path), "image", "Klein 9B", "my lora", "tweaked", 42, ".png", when=0)
    b = studio.output_path(str(tmp_path), "image", "Klein 9B", "my lora", "tweaked", 42, ".png", when=0)
    assert os.path.dirname(a) == str(tmp_path / "images")
    assert os.path.basename(a).startswith("Klein-9B_my-lora_tweaked_seed42_") and a.endswith(".png")
    open(a, "w").close()
    b = studio.output_path(str(tmp_path), "image", "Klein 9B", "my lora", "tweaked", 42, ".png", when=0)
    assert b != a and b.endswith("-2.png")
    assert os.path.dirname(studio.output_path(str(tmp_path), "video", "H3", "l", "original", 1, ".mp4")) == str(tmp_path / "videos")
    assert os.path.dirname(studio.output_path(str(tmp_path), "song", "ACE", "l", "tweaked", 1, ".wav")) == str(tmp_path / "songs")


def _pane(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_autosave_skips_an_unchanged_baseline(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(studio, "outputs_dir", lambda: str(out))
    monkeypatch.setattr(studio, "autosave_on", lambda fam: True)
    studio._S.update(family="klein", primary="x/lora.safetensors", out_saved={}, gens={1: {"seed": 7}, 2: {"seed": 7}},
                     same=False, baseline_media="", tweaked_media="")
    studio._S.update(shown=1, baseline=_pane(tmp_path, "b1.png", b"base"), tweaked=_pane(tmp_path, "t1.png", b"one"))
    studio.autosave(1)
    studio._S.update(shown=2, baseline=_pane(tmp_path, "b2.png", b"base"), tweaked=_pane(tmp_path, "t2.png", b"two"))
    studio.autosave(2)                                       # only the Tweaked one changed
    names = sorted(os.listdir(out / "images"))
    assert len(names) == 3 and sum("_original_" in n for n in names) == 1
    assert studio.save_output("baseline")["already"]           # Save on it says it's there


def test_nothing_moved_saves_once(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(studio, "outputs_dir", lambda: str(out))
    monkeypatch.setattr(studio, "autosave_on", lambda fam: True)
    studio._S.update(family="klein", primary="lora.safetensors", out_saved={}, gens={}, same=True, shown=3,
                     baseline=_pane(tmp_path, "b.png", b"same"), tweaked=_pane(tmp_path, "t.png", b"same"),
                     baseline_media="", tweaked_media="")
    studio.autosave(3)
    assert len(os.listdir(out / "images")) == 1


def test_off_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "outputs_dir", lambda: str(tmp_path / "out"))
    monkeypatch.setattr(studio, "autosave_on", lambda fam: False)
    studio._S.update(family="klein", shown=4, tweaked=_pane(tmp_path, "t.png", b"x"))
    studio.autosave(4)
    assert not (tmp_path / "out").exists()


def test_folders_ship_and_renders_are_ignored():
    for d in ("images", "videos", "songs"):
        assert os.path.isfile(os.path.join(ROOT, "Studio_Outputs", d, ".gitkeep"))
    r = subprocess.run(["git", "check-ignore", "-q", "Studio_Outputs/videos/a.mp4"], cwd=ROOT)
    keep = subprocess.run(["git", "check-ignore", "-q", "Studio_Outputs/videos/.gitkeep"], cwd=ROOT)
    if r.returncode == 128:
        pytest.skip("not a git checkout")
    assert r.returncode == 0 and keep.returncode == 1
