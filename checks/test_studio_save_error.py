"""#413: a Save repaired LoRA that failed left its card on "Saving…" for ever - the worker's error went to the Studio
status line only. The card now says the save failed, and why."""
import os

from fizgig.web import studio


def test_a_failed_save_reaches_the_save_card(monkeypatch):
    monkeypatch.setattr(studio, "_send", lambda cmd: None)
    monkeypatch.setitem(studio._S, "state", "ready")
    monkeypatch.setitem(studio._S, "primary", "a.safetensors")
    studio.save("out.safetensors", {}, False)
    assert studio.status()["save_error"] is None
    studio._event({"ev": "error", "where": "save", "msg": "out of memory"})
    st = studio.status()
    assert st["save_error"]["msg"] == "out of memory" and st["status"] == "Couldn't save: out of memory"
    first = st["save_error"]["n"]
    studio.save("out.safetensors", {}, False)            # the next try starts clean, and its failure is new to the page
    assert studio.status()["save_error"] is None
    studio._event({"ev": "error", "where": "save", "msg": "out of memory"})
    assert studio.status()["save_error"]["n"] != first


def test_the_page_shows_it_in_the_card():
    js = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
              encoding="utf-8").read()
    assert "const se = st.save_error;" in js and "Couldn't save: ${se.msg}" in js
