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
