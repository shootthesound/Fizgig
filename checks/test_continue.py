"""Continue on Runs & outputs: a run trains on from its newest save, with its own settings and into its own record; a
run that trained all its epochs asks how far to go first; a run training here puts it on the queue."""
import os
import time

import pytest

from fizgig.web import history, server
from fizgig.web import queue as webqueue


def _state(out, name, epoch, done=True, age=0):
    d = os.path.join(out, f"{name}-{epoch:06d}-state")
    os.makedirs(d)
    if done:                                    # the commit marker, written last
        marker = os.path.join(d, "training_state.json")
        open(marker, "w").close()
        if age:                                 # saved that many seconds ago
            os.utime(marker, (time.time() - age, time.time() - age))
    return d


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    out = str(tmp_path / "out")
    os.makedirs(out)
    _state(out, "mine", 50, age=3600)           # an earlier run with the same name and folder: not this run's save
    rid = history.start("web", "krea2", "mine", out, str(tmp_path / "data"),
                        {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": out, "MAX_TRAIN_EPOCHS": "10"})
    _state(out, "mine", 4)
    newest = _state(out, "mine", 6)
    _state(out, "mine", 8, done=False)          # still being written: not a save to continue from
    history.end(rid, "stopped")
    started = []
    monkeypatch.setattr(server, "_start", lambda fam, values, folder, **kw: started.append((fam, values, kw))
                        or {"ok": True})
    queued = []
    monkeypatch.setattr(webqueue, "add", lambda *a: queued.append(a))
    monkeypatch.setattr(webqueue, "resumed", lambda: None)
    server._run.update(proc=None, state="idle")
    return {"id": rid, "out": out, "newest": newest, "started": started, "queued": queued}


def test_continue_point(run):
    pt = server._continue_point(history.get(run["id"]))
    assert (pt["kind"], pt["path"], pt["done"], pt["of"]) == ("state", run["newest"], 6, 10)
    assert server._continue_point({**history.get(run["id"]), "result": "running"}) is None


def test_continues_from_the_newest_save_into_its_own_record(run):
    assert server._continue_run(run["id"])["ok"]
    fam, values, kw = run["started"][0]
    assert (fam, kw["resume"], kw["carry"], values["MAX_TRAIN_EPOCHS"]) == ("krea2", run["newest"], run["id"], "10")
    # the record carries on, however that run ended (this one was stopped)
    rid = history.start("web", "krea2", "mine", run["out"], "", values, resume=True, carry=run["id"])
    assert rid == run["id"] and len(history.runs()) == 1 and history.get(rid)["result"] == "running"


def test_finished_run_asks_how_far(run):
    rec = history.get(run["id"])
    rec["values"]["MAX_TRAIN_EPOCHS"] = "6"
    history._write([rec])
    r = server._continue_run(run["id"])
    assert r["need"] == "MAX_TRAIN_EPOCHS" and r["done"] == 6 and not run["started"]
    assert server._continue_run(run["id"], "5").get("need")              # not past where it is
    assert server._continue_run(run["id"], "9")["ok"]
    assert run["started"][0][1]["MAX_TRAIN_EPOCHS"] == "9"


def test_queues_behind_a_run_training_here(run):
    server._run.update(state="running")
    try:
        assert server._continue_run(run["id"])["queued"]
        fam, values, folder, lora, out = run["queued"][0]
        assert values["__continue"] == run["id"] and not run["started"]
        server._run.update(state="idle")
        ok, _why = server._queue_starter({"family": fam, "values": values, "image_folder": folder})
        assert ok and run["started"][0][2]["resume"] == run["newest"]
    finally:
        server._run.update(state="idle")


def test_queued_continue_edited_on_the_queue(run):
    # Edit on the queue: its settings, saved back with __continue, are what it carries on with
    rec = history.get(run["id"])
    item = {"family": "krea2", "image_folder": rec["image_folder"],
            "values": {**rec["values"], "MAX_TRAIN_EPOCHS": "20", "__continue": run["id"]}}
    ok, _why = server._queue_starter(item)
    assert ok and run["started"][0][1]["MAX_TRAIN_EPOCHS"] == "20" and "__continue" not in run["started"][0][1]
    assert run["started"][0][2]["resume"] == run["newest"]


def test_nothing_left_to_continue_from(run, tmp_path):
    import shutil
    for d in os.listdir(run["out"]):
        shutil.rmtree(os.path.join(run["out"], d))
    assert "isn't there" in server._continue_run(run["id"])["error"]


def test_two_starts_at_once_make_one_run(monkeypatch):
    # checking a run takes a few seconds: a second Start in that time must find the first going, not start a copy
    import threading
    import time
    from types import SimpleNamespace
    calls = []

    def slow_plan(*a, **kw):
        time.sleep(0.3)
        calls.append(1)
        return None, {"LORA_OUTPUT_DIR": "", "LORA_NAME": "x"}, SimpleNamespace(problems=["stop here"])

    monkeypatch.setattr(server, "_plan", slow_plan)
    monkeypatch.setattr(server.captions, "status", lambda: {"state": "idle"})
    monkeypatch.setattr(server.captions, "unload", lambda: None)
    monkeypatch.setattr(server.studio, "holding_gpu", lambda: False)
    server._run.update(proc=None, state="idle")
    real_one = server._start_one

    def one(*a):                                 # the first gets as far as starting; its plan is slow
        r = real_one(*a)
        server._run["state"] = "starting"
        return r
    monkeypatch.setattr(server, "_start_one", one)
    out = []
    ts = [threading.Thread(target=lambda: out.append(server._start("krea2", {}, ""))) for _ in range(3)]
    try:
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert len(calls) == 1 and sum(1 for r in out if "already going" in str(r.get("error"))) == 2
    finally:
        server._run.update(state="idle")
