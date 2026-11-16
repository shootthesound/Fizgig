"""Caption tools share an explicitly resolved folder scope; no model is loaded."""
import json
import os
import sys
import types
from pathlib import Path

import pytest

from fizgig.web import captions as c


def put(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "data"
    a = put(root / "red" / "same.png")
    b = put(root / "blue" / "same.png")
    put(root / "red" / "same-masklabel.png")
    put(root / "red" / "clip.mp4")
    put(root / "excluded" / "other.png")
    return root, [str(root), str(root / "red"), str(root / "blue")], a, b


def test_static_scoped_sidecars_extensions_and_overwrite(tree):
    root, folders, a, b = tree
    assert c.static_captions(str(root), "off", False, True)["written"] == 0
    assert c.static_captions(str(root), "first", False, True, ext=".caption", folders=folders)["written"] == 3
    assert c.static_captions(str(root), "kept", False, True, ext=".caption", folders=folders)["skipped"] == 3
    assert c.static_captions(str(root), "new", True, False, ext=".caption", folders=folders)["written"] == 2
    for path in (a, b):
        assert open(os.path.splitext(path)[0] + ".caption", encoding="utf-8").read() == "new"
    assert not (root / "excluded" / "other.caption").exists()
    assert not (root / "red" / "same-masklabel.caption").exists()
    assert not (root / "same.caption").exists()


def test_song_and_caption_text_tools_use_scope(tree):
    root, folders, a, b = tree
    put(root / "red" / "song.wav")
    assert c.song_captions(str(root), "z", "music", False, folders=folders)["written"] == 1
    c.static_captions(str(root), "old", False, False, folders=folders)
    put(root / "red" / "song.lyrics.txt", "old")
    result = c.find_replace(str(root), "old", "new", True, folders=folders)
    assert len(result["changes"]) == 2
    assert len({x["name"] for x in result["changes"]}) == 2
    c.add_text(str(root), "start", "end", "", True, folders=folders)
    assert (root / "red" / "same.txt").read_text() == "start, new, end"
    assert (root / "red" / "song.lyrics.txt").read_text() == "old"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    desc = c._captioners.get("florence2_base")
    proc = types.SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(c, "JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(c, "_busy", lambda: False)
    monkeypatch.setattr(c, "_st", {"proc": proc, "key": (desc.key, json.dumps({}), True),
                                   "state": "loaded", "ready": None, "log": []})
    monkeypatch.setattr(c.threading, "Thread", lambda **kw: types.SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(c, "_popen", lambda *a, **kw: pytest.fail("No model process should start"))
    return desc


def run_ai(root, folders, worker, **kw):
    return c.ai_captions({}, "", str(root), worker.key, "", "caption", 120, "z", False, True,
                         ext=".caption", folders=folders, **kw)


def test_ai_batch_and_single_selection_keep_full_paths(tree, worker):
    root, folders, a, b = tree
    assert run_ai(root, folders, worker)["files"] == 3
    paths = (open(os.path.join(c.JOB_DIR, "images.txt"), encoding="utf-8").read().splitlines())
    assert a in paths and b in paths and not any("masklabel" in p for p in paths)
    put(root / "red" / "same.caption", "existing")
    assert run_ai(root, folders, worker)["files"] == 2
    assert run_ai(root, folders, worker, only=[a])["files"] == 1
    assert open(os.path.join(c.JOB_DIR, "images.txt"), encoding="utf-8").read().splitlines() == [a]
    assert run_ai(root, folders, worker, photos=False)["files"] == 1
    assert open(os.path.join(c.JOB_DIR, "images.txt"), encoding="utf-8").read().splitlines() == [str(root / "red" / "clip.mp4")]


def test_single_selection_rejects_excluded_masks_and_mixed_invalid_paths(tree, worker):
    root, folders, a, _ = tree
    for paths in ([str(root / "excluded" / "other.png")], [str(root / "red" / "same-masklabel.png")],
                  [a, str(root / "missing.png")]):
        with pytest.raises(ValueError):
            run_ai(root, folders, worker, only=paths)
    assert not os.path.exists(c.JOB_DIR)


def test_single_selection_does_not_scan_unrelated_media(tree, worker, monkeypatch):
    root, folders, a, _ = tree
    monkeypatch.setattr(c, "media_files", lambda *a, **kw: pytest.fail("Single selection scanned the dataset"))
    assert run_ai(root, folders, worker, only=[a])["files"] == 1


@pytest.mark.parametrize("choice", [[], ["outside.png"]])
def test_explicit_empty_or_invalid_selection_never_becomes_batch(tree, worker, choice):
    root, folders, *_ = tree
    with pytest.raises(ValueError):
        run_ai(root, folders, worker, only=choice)
    assert not os.path.exists(c.JOB_DIR)


def test_crop_expansion_deduplicates_and_mirrors_each_root(tree):
    root, folders, *_ = tree
    for base in folders[1:]:
        put(Path(base) / "fizgig_detail_crops.json", '{"on": true}')
        put(Path(base) / "fizgig_detail_crops" / "0.25mp" / "same.png")
        put(Path(base) / "fizgig_detail_crops" / "1mp" / "same.png")
    expanded = c.with_crops(str(root), folders + folders[1:])
    assert len(expanded) == 5 and len(set(expanded)) == 5
    c.static_captions(str(root), "crop", False, False, folders=folders)
    for base in folders[1:]:
        assert open(os.path.join(base, "fizgig_detail_crops", "0.25mp", "same.txt"), encoding="utf-8").read() == "crop"
    c.find_replace(str(root), "crop", "edited", True, folders=folders)
    for base in folders[1:]:
        assert open(os.path.join(base, "fizgig_detail_crops", "0.25mp", "same.txt"), encoding="utf-8").read() == "edited"


def test_translation_job_passes_only_scoped_caption_paths(tree, monkeypatch, tmp_path):
    root, folders, *_ = tree
    put(root / "red" / "same.caption", "red")
    put(root / "blue" / "same.caption", "blue")
    put(root / "excluded" / "other.caption", "untouched")
    commands = []
    monkeypatch.setattr(c, "JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(c, "_busy", lambda: False)
    monkeypatch.setattr(c, "_st", dict(c._st))
    monkeypatch.setattr(c.threading, "Thread", lambda **kw: types.SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(c, "_popen", lambda cmd, prefs: commands.append(cmd))
    c.translate({}, str(root), True, ".caption", folders=folders)
    cmd = commands[0]
    assert "--skip_existing" in cmd
    paths = Path(cmd[cmd.index("--list_file") + 1]).read_text(encoding="utf-8").splitlines()
    assert set(paths) == {str(root / "red" / "same.caption"), str(root / "blue" / "same.caption")}


def test_ai_remembers_every_scoped_crop_root_for_mirroring(tree, worker):
    root, folders, *_ = tree
    for base in folders[1:]:
        put(Path(base) / "fizgig_detail_crops.json", '{"on": true}')
        put(Path(base) / "fizgig_detail_crops" / "1mp" / "same.png")
    assert run_ai(root, folders, worker)["files"] == 5
    assert set(c._st["crop_roots"]) == set(folders[1:])


@pytest.mark.parametrize("explicit", [False, True])
def test_translator_consumes_explicit_list_and_preserves_folder_cli(tmp_path, monkeypatch, explicit):
    from fizgig.scripts import translate_captions as script
    root = tmp_path / "data"
    top = put(root / "same.txt", "top")
    nested = put(root / "red" / "same.txt", "nested")
    lyrics = put(root / "red" / "song.lyrics.txt", "sung words")
    args = ["translate_captions", str(root)]
    if explicit:
        listing = put(tmp_path / "files.txt", "\n".join([nested, lyrics, nested]) + "\n")
        args += ["--list_file", listing]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(script, "_load", lambda: (None, None, "cpu"))
    monkeypatch.setattr(script, "_translate", lambda *args: "translated")
    assert script.main() == 0
    assert Path(top).read_text() == ("top" if explicit else "top - translated")
    assert Path(nested).read_text() == ("nested - translated" if explicit else "nested")
    assert Path(lyrics).read_text() == "sung words"
