"""#380: INT8 fast inference is off by default on an AMD launch (the stored fp8 file is faster there), on elsewhere;
a saved choice wins either way."""
import importlib

import pytest


def _server(monkeypatch, backend):
    monkeypatch.setenv("FIZGIG_GPU_BACKEND", backend)
    server = pytest.importorskip("fizgig.web.server")
    return importlib.reload(server)


def test_amd_launch_defaults_int8_off(monkeypatch):
    s = _server(monkeypatch, "rocm")
    assert s.PREF_DEFAULTS["inference_int8"] == "0" and s.ROCM_LAUNCH


def test_other_launch_keeps_int8_on(monkeypatch):
    s = _server(monkeypatch, "cuda")
    assert s.PREF_DEFAULTS["inference_int8"] == "1" and not s.ROCM_LAUNCH
