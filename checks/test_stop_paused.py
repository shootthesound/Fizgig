"""Stop paused run (#348): a paused run held the queue with no way to set it aside but Resume then Stop. Stop now ends
the pause: the queue can start, the saves stay, and its record reads stopped so Continue carries it on."""
import os

import pytest

from fizgig.web import history, server
from fizgig.web import queue as webqueue


@pytest.fixture
def paused(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    monkeypatch.setattr(server, "append_log", lambda *a: None)
    out = str(tmp_path / "out")
    state = os.path.join(out, "mine-000006-state")
    os.makedirs(state)
    open(os.path.join(state, "training_state.json"), "w").close()
    rid = history.start("web", "krea2", "mine", out, str(tmp_path / "data"),
                        {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": out, "MAX_TRAIN_EPOCHS": "100"})
    history.end(rid, "paused")
    side = os.path.join(out, server.PAUSED_FILE)
    open(side, "w").write("{}")
    ended = []
    monkeypatch.setattr(webqueue, "on_run_ended", lambda result, lora, starter: ended.append((result, lora)))
    # as after a restart: the restored pause carries no record id
    server._run.update(proc=None, state="paused", lora="mine", output_dir=out, resume=state, resume_mode="state",
                       history_id=None)
    yield {"id": rid, "side": side, "state": state, "ended": ended}
    server._run.update(state="idle", resume="", resume_mode="", lora="", output_dir="")


def test_stop_ends_the_pause_and_keeps_the_saves(paused):
    assert server._stop() is False                  # nothing is training: the plain Stop has nothing to do
    assert server._stop_paused() is True
    assert server._run["state"] == "stopped" and not server._run["resume"]
    assert not os.path.exists(paused["side"])       # no Resume at the next start-up
    assert os.path.isdir(paused["state"])           # the saves stay
    assert history.get(paused["id"])["result"] == "stopped"
    assert paused["ended"] == [("stopped", "mine")]  # the queue is told, as for any stopped run
    pt = server._continue_point(history.get(paused["id"]))
    assert pt and pt["path"] == paused["state"]      # Continue on its card carries it on


def test_nothing_paused_nothing_done(paused):
    server._run.update(state="completed")
    assert server._stop_paused() is False
    assert os.path.exists(paused["side"]) and history.get(paused["id"])["result"] == "paused"
