"""A new run can't take a name that already has files in the same output folder; carrying that run on can, and a
name that made nothing is free."""
import os

from fizgig.web import history, server
from fizgig.web import queue as webqueue


def test_name_taken(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    monkeypatch.setattr(webqueue, "status", lambda: {"items": []})
    server._run.update(lora="")
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "sample"))
    inputs = {"LORA_OUTPUT_DIR": out, "LORA_NAME": "mine"}
    assert server.name_taken({}, inputs) == ""                              # nothing of that name there
    open(os.path.join(out, "mine_2-000004.safetensors"), "w").close()       # a longer name's files are not this name's
    assert server.name_taken({}, inputs) == ""
    open(os.path.join(out, "mine-000004.safetensors"), "w").close()
    why = server.name_taken({}, inputs)
    assert "already has files" in why and "mine_3 is free" in why
    assert server.name_taken({"RESUME_TRAINING": os.path.join(out, "mine-000004-state")}, inputs) == ""   # carrying on
    assert server.name_taken({"__continue": "abc"}, inputs) == ""
    os.remove(os.path.join(out, "mine-000004.safetensors"))
    open(os.path.join(out, "sample", "mine_e0002_00.png"), "w").close()     # previews count too
    assert server.name_taken({}, inputs)
    assert server.name_taken({}, {"LORA_OUTPUT_DIR": out, "LORA_NAME": "other"}) == ""
