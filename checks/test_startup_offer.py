"""A run cut off when Fizgig closed is offered again at the next start: carry it on from its last save, start the next
queued run, or clear the queue."""
import os

import pytest

from fizgig.web import history, server
from fizgig.web import queue as webqueue


@pytest.fixture
def cut(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    monkeypatch.setattr(server, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(webqueue, "QUEUE_FILE", str(tmp_path / "queue.json"))
    webqueue._q.update(items=[], held=False, reason="", waiting="")
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "mine-000004-state"))
    open(os.path.join(out, "mine-000004-state", "training_state.json"), "w").close()
    rid = history.start("web", "krea2", "mine", out, str(tmp_path / "data"),
                        {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": out, "MAX_TRAIN_EPOCHS": "10"})
    os.utime(os.path.join(out, "mine-000004-state", "training_state.json"))      # saved after the run started
    history.end(rid, "stopped")
    server.save_state({"interrupted": rid})
    server._startup["offer"] = None
    return {"id": rid, "out": out}


def test_offer_carry_on(cut):
    o = server._startup_offer()
    assert (o["lora"], o["can_continue"], o["done"], o["of"], o["queued"]) == ("mine", True, 4, 10, 0)


def test_nothing_saved_and_no_queue_is_no_question(cut, tmp_path):
    import shutil
    shutil.rmtree(os.path.join(cut["out"], "mine-000004-state"))
    assert server._startup_offer() is None and not server.load_state().get("interrupted")


def test_nothing_saved_but_a_queue(cut):
    import shutil
    shutil.rmtree(os.path.join(cut["out"], "mine-000004-state"))
    webqueue.add("krea2", {"LORA_NAME": "next"}, "", "next", cut["out"])
    o = server._startup_offer()
    assert o["can_continue"] is False and o["queued"] == 1


def test_fresh_clears_the_queue(cut):
    webqueue.add("krea2", {"LORA_NAME": "next"}, "", "next", cut["out"])
    server._startup["offer"] = server._startup_offer()
    assert server.startup_choice("fresh") == {"ok": True}
    assert not webqueue.status()["items"] and server._startup["offer"] is None
    assert not server.load_state().get("interrupted")
    assert server.startup_choice("fresh").get("error")              # answered: nothing left to decide


def test_a_run_that_came_by_a_move_is_never_offered(cut):
    recs = history._read()
    recs[-1]["values"]["__moved_in"] = "1"          # as helpers.move_in marks it
    history._write(recs)
    assert server._startup_offer() is None and not server.load_state().get("interrupted")


def test_the_bar_place_is_kept_with_the_window(tmp_path, monkeypatch):
    import json
    from fizgig.web import app_window
    prefs = tmp_path / "prefs.json"
    prefs.write_text(json.dumps({"web_port": "8081"}), encoding="utf-8")
    monkeypatch.setattr(app_window, "PREFS_FILE", str(prefs))
    app_window._remember_bar([0.25, 300])
    assert json.loads(prefs.read_text(encoding="utf-8")) == {"web_port": "8081", "app_bar_place": [0.25, 300.0]}
    assert "[0.25, 300.0]" in app_window._bar("On a pod", "Back", "t", "home")
    app_window._remember_bar(None)                  # a double-click: back to the top middle
    assert "app_bar_place" not in json.loads(prefs.read_text(encoding="utf-8"))
    assert app_window._bar("On a pod", "Back", "t", "home").rstrip().endswith("null);")


def test_a_run_stopped_on_purpose_is_never_offered(cut):
    server.save_state({"interrupted": ""})
    assert server._startup_offer() is None
