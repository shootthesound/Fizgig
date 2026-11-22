import importlib.util
import os

import pytest

# loaded by its file: the module is stdlib-only, so this runs without torch
_p = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "native_pick.py")
_spec = importlib.util.spec_from_file_location("native_pick", _p)
native_pick = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native_pick)


def test_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(native_pick, "_avail", False)
    assert native_pick.pick("x", True) == {"unavailable": True}


def test_cancel_and_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(native_pick, "_avail", True)
    class R: stdout = '{"path": ""}\n'
    monkeypatch.setattr(native_pick.subprocess, "run", lambda *a, **k: R)
    assert native_pick.pick("", True) == {"cancelled": True}
    R.stdout = '{"path": "%s"}' % str(tmp_path).replace("\\", "/")
    assert native_pick.pick("", False) == {"path": os.path.normpath(str(tmp_path))}


def test_start_dir(tmp_path):
    f = tmp_path / "a.onnx"; f.write_text("")
    assert native_pick.start_dir(str(f)) == str(tmp_path)
    assert native_pick.start_dir("/no/such/place/x") == ""


def test_local_viewer_only():
    Handler = pytest.importorskip("fizgig.web.server").Handler
    class H: headers = {}
    for ip, ok in (("127.0.0.1", True), ("::1", True), ("192.168.1.5", False)):
        H.client_address = (ip, 0)
        assert Handler._local_viewer(H) is ok
    H.client_address = ("127.0.0.1", 0); H.headers = {"X-Forwarded-For": "1.2.3.4"}
    assert not Handler._local_viewer(H)


def test_save_as_window(monkeypatch, tmp_path):
    """Studio's Download buttons: a Save window opening in the folder saved to last, the file name filled in."""
    import json
    monkeypatch.setattr(native_pick, "_avail", True)
    seen = {}
    class R: stdout = '{"path": "%s"}' % str(tmp_path / "x.png").replace("\\", "/")
    def run(cmd, **k):
        seen.update(json.loads(cmd[-1]))
        return R
    monkeypatch.setattr(native_pick.subprocess, "run", run)
    assert native_pick.save_as(str(tmp_path), "lora_tweaked_seed42.png") == {"path": os.path.normpath(str(tmp_path / "x.png"))}
    assert seen["save"] == "lora_tweaked_seed42.png" and seen["ext"] == ".png" and seen["start"] == str(tmp_path)
    compile(native_pick._SCRIPT, "<native_pick>", "exec")


def test_off_by_default_on_windows(monkeypatch):
    """#387: Choose... opens Fizgig's chooser on Windows unless Preferences turns the computer's own window on; the
    route answers unavailable while it's off, and Reset to defaults puts the default back."""
    import sys
    server = pytest.importorskip("fizgig.web.server")
    from fizgig.web import native_pick as real
    assert server.PREF_DEFAULTS["native_file_window"] == ("0" if sys.platform == "win32" else "1")
    assert "native_file_window" in server.SETTING_PREFS
    for on, called in (("0", False), ("1", True)):
        seen, out = [], {}
        monkeypatch.setattr(server, "load_prefs", lambda on=on: {"native_file_window": on})
        monkeypatch.setattr(real, "pick", lambda *a: seen.append(a) or {"cancelled": True})

        class H:
            _local_viewer = lambda self: True
            json = lambda self, d: out.update(d)
        server.Handler._native_pick(H(), {"path": [""], "dir": ["1"]})
        assert bool(seen) == called and out == ({"cancelled": True} if called else {"unavailable": True})
