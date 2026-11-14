"""Another run of the same thing takes the next free name: name_2, then _3, past the runs in the history, the queue,
the run training and whatever of that name is in the output folder already."""
import os

from fizgig.web import history, server
from fizgig.web import queue as webqueue


def test_next_free_name(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    monkeypatch.setattr(webqueue, "status", lambda: {"items": [{"lora": "mine_3"}]})
    server._run.update(lora="mine")
    out = str(tmp_path / "out")
    os.makedirs(out)
    assert server.next_run_name("mine", out) == "mine_2"
    open(os.path.join(out, "mine_2.safetensors"), "w").close()         # a finished LoRA of that name
    assert server.next_run_name("mine", out) == "mine_4"               # _3 is queued
    os.makedirs(os.path.join(out, "mine_4-000002-state"))              # a saved state counts too
    assert server.next_run_name("mine", out) == "mine_5"
    assert server.next_run_name("mine_4", out) == "mine_5"             # a numbered name counts on from its number
    history.end(history.start("web", "krea2", "mine_5", out, "", {}), "finished")
    assert server.next_run_name("mine", out) == "mine_6"               # a past run's name
    assert server.next_run_name("other", str(tmp_path / "missing")) == "other_2"
