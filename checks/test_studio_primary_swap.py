"""Studio: another primary LoRA goes on the loaded model in place, never a model reload (#394).

Choosing a new primary and pressing Render used to reload the whole model (minutes on a big one), because the page
only knew two paths: render on what's loaded, or a fresh load. Now a change of primary alone swaps the LoRA on the
loaded model, a LoRA that can't load leaves the old one on, and the page waits for the worker's answer.
"""
import os

import pytest

W = pytest.importorskip("fizgig.scripts.studio_worker")
from fizgig.web import studio  # noqa: E402

APP = os.path.join(os.path.dirname(studio.__file__), "static", "app.js")


class _Eng:
    primary_block_ids = donor_block_ids = ()
    primary_hash = ""

    def __init__(self):
        self.primary_network, self.primary_path, self.calls = object(), "old.safetensors", []

    def unload_primary(self):
        self.calls.append("unload")
        self.primary_network, self.primary_path = None, None

    def load_primary(self, path):
        self.calls.append(("load", path))
        if path == "other_model.safetensors":
            raise ValueError("other_model.safetensors adapts nothing in MOSS-Image")
        self.primary_network, self.primary_path = object(), path


def _run(monkeypatch, path):
    eng, out = _Eng(), []
    monkeypatch.setitem(W.S, "engine", eng)
    monkeypatch.setattr(W, "emit", lambda **d: out.append(d))
    W.cmd_primary({"path": path})
    return eng, out


def test_a_new_primary_swaps_in_place(monkeypatch):
    eng, out = _run(monkeypatch, "new.safetensors")
    assert eng.calls == ["unload", ("load", "new.safetensors")]
    assert out[-1]["ev"] == "primary" and out[-1]["primary"] == "new.safetensors" and not out[-1].get("error")


def test_a_lora_that_cant_load_leaves_the_old_one_on(monkeypatch):
    eng, out = _run(monkeypatch, "other_model.safetensors")
    assert eng.primary_path == "old.safetensors" and eng.primary_network is not None
    assert out[-1]["primary"] == "old.safetensors" and "adapts nothing" in out[-1]["error"]


def test_the_page_hears_each_swap_and_its_error(monkeypatch):
    monkeypatch.setattr(studio, "_S", dict(studio._S))
    seq = studio.status()["primary_seq"]
    studio._event({"ev": "primary", "primary": "old.safetensors", "error": "adapts nothing"})
    st = studio.status()
    assert st["primary_seq"] == seq + 1 and st["error"] == "adapts nothing" and "adapts nothing" in st["status"]
    studio._event({"ev": "primary", "primary": "new.safetensors"})
    st = studio.status()
    assert st["primary_seq"] == seq + 2 and st["error"] == "" and st["primary"] == "new.safetensors"


def test_render_swaps_the_primary_before_it_would_reload():
    js = open(APP, encoding="utf-8").read()
    start = js[js.index("async function studioStart"):js.index("async function studioPoll")]
    assert "studioSwapPrimary(" in start
    assert start.index("studioSwapPrimary(") < start.index('api("/api/studio/load"')
    # only when nothing but the primary differs from what's loaded, and never for a video model or Frankenstein mode
    assert 'studioLoadKey(L, fast, st.primary || "")' in start
    assert 'L.kind !== "video"' in start and "studioFrankAsked()" in start
    swap = js[js.index("async function studioSwapPrimary"):]
    assert 'api("/api/studio/primary", {path: p})' in swap and "primary_seq" in swap
