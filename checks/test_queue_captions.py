"""To-do #14: an AI caption job in the queue - it starts like a run, the queue waits while it captions, and its end
(DONE / STOPPED / the worker dying) moves the queue on the same way a run's does. CPU only: the caption worker and
the run start are stand-ins."""
import pytest

from fizgig.web import captions, queue


@pytest.fixture
def q(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_save", lambda: None)
    monkeypatch.setattr(queue, "schedule", lambda delay, starter: queue.start_next(starter))   # no 10 s timer
    queue._q.update(items=[], held=False, reason="", waiting="", for_pause=False, failed=[])
    folder = tmp_path / "set"
    folder.mkdir()
    return str(folder)


def test_caption_job_then_run(q):
    started = []

    def starter(item):
        started.append(item.get("kind") or "run")
        if item.get("kind") == "captions":
            captions.when_done(lambda result: queue.on_run_ended(result, item["lora"], starter))
        return True, ""

    queue.add_captions({"folder": q, "model": "x"}, q, "Captions: set")
    queue.add("klein", {}, q, "lora1", q)
    st = queue.status()
    assert [i.get("kind") for i in st["items"]] == ["captions", None]
    queue.start_next(starter)
    assert started == ["captions"]                    # the run waits for the captions
    captions._ended("finished")                       # the worker printed DONE
    assert started == ["captions", "run"]
    captions._ended("finished")                       # called once only
    assert started == ["captions", "run"]


def test_save_to_queue_keeps_its_place(q):
    a = queue.add_captions({"folder": q, "model": "x"}, q, "Captions: set")
    queue.add("klein", {}, q, "lora1", q)
    assert queue.update_captions(a["id"], {"folder": q, "model": "y"}, q, "Captions: set (y)")
    assert queue.get(a["id"])["body"]["model"] == "y"
    assert [i["lora"] for i in queue.status()["items"]] == ["Captions: set (y)", "lora1"]
    assert not queue.update_captions("gone", {}, q, "x")


def test_caption_job_survives_a_restart(q, tmp_path, monkeypatch):
    import json
    f = tmp_path / "web_queue.json"
    item = {"id": "a", "kind": "captions", "body": {"folder": q}, "family": "", "values": {}, "image_folder": q,
            "lora": "Captions: set", "output_dir": "", "queued_at": ""}
    f.write_text(json.dumps([item]), encoding="utf-8")
    monkeypatch.setattr(queue, "QUEUE_FILE", str(f))
    queue.load()
    st = queue.status()
    assert [i.get("kind") for i in st["items"]] == ["captions"] and st["held"]


def test_stopped_caption_job_holds_the_queue(q):
    def starter(item):
        captions.when_done(lambda result: queue.on_run_ended(result, item["lora"], starter))
        return True, ""

    queue.add_captions({"folder": q}, q, "Captions: set")
    queue.add("klein", {}, q, "lora1", q)
    queue.start_next(starter)
    captions._ended("stopped")
    st = queue.status()
    assert st["held"] and len(st["items"]) == 1       # the run is still queued, waiting for Start queue
