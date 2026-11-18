"""Unexpected Studio exits retain their native error and exit code."""
from types import SimpleNamespace
from unittest.mock import Mock

from fizgig.web import studio


def test_unexpected_worker_exit_keeps_cause(monkeypatch):
    proc = SimpleNamespace(stdout=iter(["Memory access fault\n"]), wait=Mock(return_value=-6))
    monkeypatch.setattr(studio, "_S", {"proc": proc, "log": [], "state": "ready", "rendering": True})
    monkeypatch.setattr(studio, "_R", {"state": "idle"})
    monkeypatch.setattr(studio, "_M", {"state": "idle"})
    monkeypatch.setattr(studio, "_T", {})
    studio._reader(proc)
    state = studio._S
    assert state["proc"] is None and state["state"] == "idle" and not state["rendering"]
    assert "exit code -6" in state["error"]
    assert state["log"] == ["Memory access fault", state["error"]]


def test_deliberately_stopped_worker_does_not_report_failure(monkeypatch):
    proc = SimpleNamespace(stdout=iter([]), wait=Mock(return_value=-15))
    state = {"proc": None, "log": [], "status": "Models unloaded.", "error": ""}
    monkeypatch.setattr(studio, "_S", state)
    studio._reader(proc)
    assert state["status"] == "Models unloaded." and state["error"] == ""
