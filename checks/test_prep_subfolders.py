"""Prep and the caption gallery use the same opted-in dataset folders as training."""
import json
import os
from urllib.parse import urlencode

import pytest
from PIL import Image

from fizgig.dataset import prep
from fizgig.web import prep_jobs, server


def _pic(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), (40, 80, 120)).save(path)
    return str(path)


def test_prep_summary_counts_the_selected_subfolders(tmp_path):
    _pic(tmp_path / "red" / "same.jpg")
    _pic(tmp_path / "blue" / "deep" / "same.jpg")
    folders = [str(tmp_path), str(tmp_path / "red"), str(tmp_path / "blue" / "deep")]
    assert prep.summary(str(tmp_path), "Resize Only", False, 1.0)["count"] == 0
    result = prep.summary(str(tmp_path), "Resize Only", False, 1.0, folders=folders)
    assert result["count"] == 2
    assert "sub-folder" in " ".join(result["lines"]).lower()


@pytest.mark.parametrize("mode", ["Resize Only", "Auto Prep (Face Crops)", "Face Crop Only"])
def test_prep_keeps_same_names_and_originals_in_their_own_folder(tmp_path, mode):
    class NoFaces:
        def detect_from_pil(self, image):
            return []

    roots = [tmp_path / "red", tmp_path / "blue" / "deep"]
    for d in roots:
        _pic(d / "same.jpg")
        (d / "same.txt").write_text(d.name, encoding="utf-8")
    logs = []
    prep.Prep(logs.append, NoFaces()).run(mode, str(tmp_path), 1.0,
                                          folders=[str(tmp_path), *map(str, roots)])
    for d in roots:
        assert (d / "same.png").is_file()
        assert (d / "originals" / "same.jpg").is_file()
        assert (d / "same.txt").read_text(encoding="utf-8") == d.name
        assert not (d / "originals" / "same.png").exists()
    assert not (tmp_path / "same.png").exists()


def test_scope_uses_training_exclusions_and_off_stays_flat(tmp_path, monkeypatch):
    for d in ("red", "blue/deep", "originals", "originals_2", "Removed", "excluded_by_look",
              ".hidden", "red/originals", "out/sample", "cache", "reg"):
        _pic(tmp_path / d / "a.png")
    values = {"TRAIN_SUBFOLDERS": True, "LORA_OUTPUT_DIR": str(tmp_path / "out"),
              "FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(tmp_path / "reg")}
    monkeypatch.setattr(server, "load_prefs", lambda: {"cache_dir": str(tmp_path / "cache")})
    result = server._tool_folders(str(tmp_path), "krea2", values)
    assert {os.path.relpath(d, tmp_path).replace("\\", "/") for d in result} == {".", "red", "blue/deep"}
    assert server._tool_folders(str(tmp_path), "krea2", {}) == [str(tmp_path)]


@pytest.mark.parametrize("values", [{"FAMILY_MULTICONCEPT": True}, {"FAMILY_DOP": True},
                                    {"FAMILY_MULTI_PERSON": True}, {"FAMILY_KIND": "edit"}])
def test_inapplicable_saved_subfolder_tick_stays_flat(tmp_path, values):
    _pic(tmp_path / "nested" / "a.png")
    assert server._tool_folders(str(tmp_path), "klein", {"TRAIN_SUBFOLDERS": True, **values}) == [str(tmp_path)]


@pytest.mark.parametrize("output", ["", "   "])
def test_scope_excludes_the_default_output_folder_from_preferences(tmp_path, monkeypatch, output):
    _pic(tmp_path / "photos" / "a.png")
    _pic(tmp_path / "outputs" / "sample" / "preview.png")
    monkeypatch.setattr(server, "load_prefs", lambda: {"lora_output_dir": str(tmp_path / "outputs")})
    assert server._tool_folders(str(tmp_path), "krea2", {"TRAIN_SUBFOLDERS": True, "LORA_OUTPUT_DIR": output}) == [
        str(tmp_path), str(tmp_path / "photos")]


def _handler(path=""):
    handler = object.__new__(server.Handler)
    handler.path = path
    handler.json = lambda value, *args: value
    handler._host_ok = lambda: True
    handler._authed = lambda: True
    handler._api_ok = lambda *args: True
    return handler


def test_gallery_keeps_basename_and_full_path_for_nested_saves(tmp_path):
    paths = {_pic(tmp_path / d / "same.png") for d in ("red", "blue/deep")}
    query = urlencode({"folder": str(tmp_path), "family": "krea2", "ext": ".txt",
                       "values": json.dumps({"TRAIN_SUBFOLDERS": True})})
    handler = _handler("/api/captions?" + query)
    result = handler._get()
    assert {it["path"] for it in result["items"]} == paths
    assert all(it["name"] == "same.png" for it in result["items"])
    assert {it["relative"] for it in result["items"]} == {"red/same.png", "blue/deep/same.png"}


def test_background_prep_passes_the_complete_scope(tmp_path, monkeypatch):
    _pic(tmp_path / "nested" / "a.jpg")
    folders = [str(tmp_path), str(tmp_path / "nested")]
    monkeypatch.setitem(prep_jobs._job, "state", "idle")
    monkeypatch.setitem(prep_jobs._look, "state", "idle")
    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()
    monkeypatch.setattr(prep_jobs.threading, "Thread", ImmediateThread)
    prep_jobs.start(str(tmp_path), "Resize Only", 1.0, "Largest Face", 20, False, folders=folders)
    assert prep_jobs.status()["state"] == "done"
    assert (tmp_path / "nested" / "a.png").exists()


def test_recursive_scope_never_switches_subfolders_to_untrained_detail_crops(tmp_path, monkeypatch):
    from fizgig.dataset import detail_crops
    _pic(tmp_path / "a.png")
    _pic(tmp_path / "nested" / "b.png")
    switched = []
    monkeypatch.setattr(detail_crops, "switch", lambda folder, on: switched.append(folder))
    monkeypatch.setattr(detail_crops, "make", lambda *args, **kwargs: None)
    monkeypatch.setattr(detail_crops, "archive", lambda *args, **kwargs: 0)
    monkeypatch.setattr(detail_crops, "share_captions", lambda *args: 0)
    monkeypatch.setattr(detail_crops, "sizes", lambda *args: [])
    folders = [str(tmp_path), str(tmp_path / "nested")]
    prep.Prep(lambda text: None).run("Detail Crops (Style)", str(tmp_path), 0.25, folders=folders)
    assert switched == [str(tmp_path)]
    assert (tmp_path / "nested" / "b.png").exists()


def test_ai_route_rejects_an_explicit_null_selection(tmp_path, monkeypatch):
    _pic(tmp_path / "a.png")
    handler = _handler("/api/captions/ai")
    handler._body = lambda: {"family": "krea2", "folder": str(tmp_path), "model": "florence2_base", "only": None}
    monkeypatch.setattr(server, "load_prefs", lambda: {})
    monkeypatch.setattr(server, "_caption_gpu_check", lambda *args: None)
    monkeypatch.setattr(server.captions, "ai_captions", lambda *args, **kwargs: {"batch_started": True})
    result = handler._post()
    assert "error" in result
    assert "batch_started" not in result
