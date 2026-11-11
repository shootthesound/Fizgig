"""The Samples gallery's likeness photos are remembered with the training folder, and each run keeps its own scores
(web/gallery_likeness.py): switching between runs never clears or mixes them. CPU only, no face model."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.web import gallery_likeness as gl  # noqa: E402
from fizgig.web import history  # noqa: E402


def _dataset(tmp_path, name, pics=("a.jpg", "b.jpg", "c.jpg", "d.jpg")):
    d = tmp_path / name
    d.mkdir()
    for p in pics:
        (d / p).write_bytes(b"x")
    return str(d)


def _no_scoring(monkeypatch):
    """start() without the face model or a scoring thread: records what would have started."""
    started = []
    monkeypatch.setattr(gl.prep_jobs, "faces_available", lambda: True)
    monkeypatch.setattr(gl, "_worker", lambda samples_dir, run, gen, paths: started.append((run, [os.path.basename(p) for p in paths])))
    gl._runs.clear()
    return started


def test_pick_is_remembered_with_the_training_folder(tmp_path, monkeypatch):
    started = _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    gl.start(str(samples), "run_one", ds, ["a.jpg", "b.jpg", "c.jpg"])
    assert gl.dataset_pick(ds) == ["a.jpg", "b.jpg", "c.jpg"]
    assert json.loads((tmp_path / "nicola" / gl.PICK_FILE).read_text(encoding="utf-8"))["baselines"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert started[-1] == ("run_one", ["a.jpg", "b.jpg", "c.jpg"])


def test_another_run_of_the_same_folder_picks_them_up(tmp_path, monkeypatch):
    started = _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    gl.start(str(samples), "run_one", ds, ["a.jpg", "b.jpg", "d.jpg"])
    gl._runs.clear()
    st = gl.state(str(samples), "run_two", ds)           # never picked for run_two
    assert st["baselines"] == ["a.jpg", "b.jpg", "d.jpg"] and st["run"] == "run_two"
    assert started[-1] == ("run_two", ["a.jpg", "b.jpg", "d.jpg"])   # its own scorer, the folder's photos


def test_runs_of_different_people_keep_their_own(tmp_path, monkeypatch):
    _no_scoring(monkeypatch)
    one, two, samples = _dataset(tmp_path, "nicola"), _dataset(tmp_path, "peter", ("p1.png", "p2.png", "p3.png")), tmp_path / "sample"
    samples.mkdir()
    gl.start(str(samples), "nicola_run", one, ["a.jpg", "b.jpg", "c.jpg"])
    gl.start(str(samples), "peter_run", two, ["p1.png", "p2.png", "p3.png"])
    gl._runs.clear()
    assert gl.state(str(samples), "nicola_run", one)["baselines"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert gl.state(str(samples), "peter_run", two)["baselines"] == ["p1.png", "p2.png", "p3.png"]


def test_a_runs_own_photos_win_over_the_folders_newer_pick(tmp_path, monkeypatch):
    """Two runs of one person: re-picking for the second run doesn't change what the first was scored against."""
    _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    gl._write(str(samples), "first", ["a.jpg", "b.jpg", "c.jpg"], "live - 4 sample(s) scored", {"x_e000001_00_1_1.png": 0.5})
    gl.start(str(samples), "second", ds, ["b.jpg", "c.jpg", "d.jpg"])      # the folder now remembers b, c, d
    gl._runs.clear()
    st = gl.state(str(samples), "first", ds)
    assert st["baselines"] == ["a.jpg", "b.jpg", "c.jpg"] and st["scores"] == {"x_e000001_00_1_1.png": 0.5}
    assert gl.state(str(samples), "second", ds)["baselines"] == ["b.jpg", "c.jpg", "d.jpg"]


def test_a_run_never_scored_and_a_folder_never_picked_shows_none(tmp_path, monkeypatch):
    started = _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    st = gl.state(str(samples), "fresh", ds)
    assert st["baselines"] == [] and not started


def test_photos_gone_from_the_folder_are_not_used(tmp_path, monkeypatch):
    _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    gl.start(str(samples), "r", ds, ["a.jpg", "b.jpg", "c.jpg"])
    os.remove(os.path.join(ds, "c.jpg"))
    gl._runs.clear()
    assert gl.dataset_pick(ds) == []
    assert gl.state(str(samples), "r", ds)["baselines"] == []


def test_clearing_forgets_the_folders_pick(tmp_path, monkeypatch):
    _no_scoring(monkeypatch)
    ds, samples = _dataset(tmp_path, "nicola"), tmp_path / "sample"
    samples.mkdir()
    gl.start(str(samples), "r", ds, ["a.jpg", "b.jpg", "c.jpg"])
    assert gl.start(str(samples), "r", ds, []) == "cleared"
    assert gl.dataset_pick(ds) == [] and not (tmp_path / "nicola" / gl.PICK_FILE).exists()


def test_a_run_scores_only_its_own_previews():
    assert gl._run_of("nicola_run_e000012_00_20261111170000_42.png") == "nicola_run"
    assert gl._run_of("peter_run_e000003_01_20261111170000_42_s050.png") == "peter_run"
    assert gl._run_of("not_a_preview.png") is None


def test_a_runs_training_folder_comes_from_its_record(monkeypatch, tmp_path):
    out = str(tmp_path / "loras")
    monkeypatch.setattr(history, "_read", lambda: [
        {"lora": "nicola_run", "output_dir": out, "image_folder": "D:/old"},
        {"lora": "nicola_run", "output_dir": out, "image_folder": "D:/nicola"},
        {"lora": "other", "output_dir": out, "image_folder": "D:/other"}])
    assert history.image_folder_of("nicola_run", out) == "D:/nicola"      # the newest record
    assert history.image_folder_of("missing", out) == ""


def test_profiler_prefill_uses_the_folders_photos(monkeypatch, tmp_path):
    from fizgig.web import tools
    ds = _dataset(tmp_path, "nicola")
    with open(os.path.join(ds, gl.PICK_FILE), "w", encoding="utf-8") as f:
        json.dump({"baselines": ["a.jpg", "b.jpg", "c.jpg"]}, f)
    out = tmp_path / "loras"
    out.mkdir()
    monkeypatch.setattr(history, "_read", lambda: [{"lora": "nicola_run", "output_dir": str(out), "image_folder": ds}])
    monkeypatch.setattr(tools, "_run_sample_prompt", lambda lora, trig: ("p", "c"))
    r = tools.profile_prefill(str(out / "nicola_run-000012.safetensors"))
    assert [os.path.basename(p) for p in r["baselines"]] == ["a.jpg", "b.jpg", "c.jpg"]
