"""Studio renders only the panes on screen: Baseline hidden (◀) renders no Baseline, Tweaked hidden (▶) no Tweaked,
for pictures and songs; nothing moved is still one render. Save every render saves only what rendered."""
import types

import pytest

w = pytest.importorskip("fizgig.scripts.studio_worker")
studio = pytest.importorskip("fizgig.web.studio")


class Pic:
    def save(self, path):
        open(path, "wb").write(b"png")


class Eng:
    primary_network = object()

    def __init__(self, same=False):
        self.same, self.calls = same, []

    def block_ids(self):
        return []

    def clear_cancel(self):
        pass

    def same_as_baseline(self, st):
        return self.same

    def generate_baseline(self, st):
        self.calls.append("baseline")
        return Pic()

    def generate_preview(self, st):
        self.calls.append("tweaked")
        return Pic()


def _run(monkeypatch, tmp_path, panes, same=False):
    ev = []
    monkeypatch.setattr(w, "emit", lambda **d: ev.append(d))
    monkeypatch.setattr(w, "_apply_mix", lambda eng, mix: {})
    monkeypatch.setattr(w, "_state", lambda d: types.SimpleNamespace(blocks={}))
    eng = Eng(same)
    w.S.update(engine=eng, desc=types.SimpleNamespace(), notes={})
    w._render({"gen": 1, "state": {}, "out_dir": str(tmp_path), "panes": panes})
    done = next(e for e in ev if e.get("ev") == "rendered")
    return eng.calls, done


def test_baseline_hidden_renders_no_baseline(monkeypatch, tmp_path):
    calls, done = _run(monkeypatch, tmp_path, "tweaked")
    assert calls == ["tweaked"] and done["baseline"] == "" and done["tweaked"]


def test_tweaked_hidden_renders_no_tweaked(monkeypatch, tmp_path):
    calls, done = _run(monkeypatch, tmp_path, "baseline")
    assert calls == ["baseline"] and done["tweaked"] == "" and done["baseline"]


def test_both_and_nothing_moved(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, "both")[0] == ["baseline", "tweaked"]
    calls, done = _run(monkeypatch, tmp_path, "tweaked", same=True)      # nothing moved: one render for both
    assert calls == ["baseline"] and done["tweaked"]


def test_songs_render_only_the_shown_pane(monkeypatch, tmp_path):
    class Song:
        wave, activity = [0.0], None

        def picture(self):
            return Pic()
    eng = Eng()
    eng.generate_baseline = lambda st: (eng.calls.append("baseline"), Song())[1]
    eng.generate_preview = lambda st: (eng.calls.append("tweaked"), Song())[1]
    eng.driver = types.SimpleNamespace(set_options=lambda o: None)
    eng.primary_path = "l.safetensors"
    monkeypatch.setattr(w, "_audio_inputs", lambda *a: {})
    monkeypatch.setattr(w, "_write_activity", lambda *a: "")
    import fizgig.minimax.common as mc
    monkeypatch.setattr(mc, "write_wav", lambda p, wave, sample_rate: open(p, "wb").close())
    w.S["notes"] = {}
    st = types.SimpleNamespace(preview_width=30, seed=1, prompt="p")
    paths, _ = w._render_songs(eng, st, {"audio": {}, "panes": "tweaked"}, str(tmp_path), 1)
    assert eng.calls == ["tweaked"] and paths["baseline"] == "" and paths["tweaked_media"]


def test_save_every_render_saves_only_what_rendered(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "outputs_dir", lambda: str(tmp_path / "out"))
    monkeypatch.setattr(studio, "autosave_on", lambda fam: True)
    b = tmp_path / "b.png"
    b.write_bytes(b"base")
    studio._S.update(family="klein", primary="l.safetensors", out_saved={}, gens={}, same=False, shown=7,
                     baseline=str(b), tweaked="", baseline_media="", tweaked_media="")
    assert len(studio.autosave(7)) == 1
