"""Studio's own queue: jobs load their model when it differs, render one at a time, are saved, and fail on their own;
it pauses (never resumes by itself) and waits while training has the GPU; a page edit waits for a job instead of
cancelling it; a training start waits for a Studio render and pauses the Studio queue."""
import pytest

server = pytest.importorskip("fizgig.web.server")
from fizgig.web import queue as webqueue  # noqa: E402
from fizgig.web import studio, studio_queue  # noqa: E402


@pytest.fixture
def q(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_queue, "QUEUE_FILE", str(tmp_path / "studio_queue.json"))
    studio_queue._q.update(items=[], paused=False, reason="", waiting="", job="", loaded=True)
    world = {"state": "idle", "rendering": False, "key": None, "loads": [], "renders": [], "busy": ""}

    def load(body):
        world["loads"].append(body)
        world["state"] = "ready"
        world["key"] = None                      # studio.load forgets the key; the queue sets it after

    def render_job(state, extra, job):
        if world["state"] != "ready" or world["rendering"]:
            return False
        world["rendering"] = job
        world["renders"].append(job)
        return True
    monkeypatch.setattr(studio_queue, "_hooks", {"load": load, "busy": lambda: world["busy"]})
    monkeypatch.setattr(studio, "status", lambda: {"state": world["state"], "error": ""})
    monkeypatch.setattr(studio, "generating", lambda: bool(world["rendering"]))
    monkeypatch.setattr(studio, "current_job", lambda: world["rendering"] or "")
    monkeypatch.setattr(studio, "load_key", lambda: world["key"])
    monkeypatch.setattr(studio, "set_load_key", lambda k: world.__setitem__("key", k))
    monkeypatch.setattr(studio, "render_job", render_job)

    def finish(ok=True):
        job, world["rendering"] = world["rendering"], False
        studio_queue.job_ended(job, ok, ["/out/a.png"] if ok else [], "" if ok else "boom")
    world["finish"] = finish
    return world


def _job(lora="a"):
    return studio_queue.add({"family": "klein", "primary": lora}, {"state": {"seed": 1}, "extra": {}}, "p")


def test_jobs_load_render_and_share_a_load(q):
    a, b = _job("a"), _job("a")
    studio_queue._tick()                         # loads the model
    assert len(q["loads"]) == 1 and studio_queue.status()["items"][0]["status"] == "loading"
    studio_queue._tick()                         # ready: renders
    assert q["renders"] == [a["id"]]
    q["finish"]()
    studio_queue._tick()                         # same load: no second load
    assert len(q["loads"]) == 1 and q["renders"] == [a["id"], b["id"]]
    q["finish"](ok=False)
    st = {i["id"]: i for i in studio_queue.status()["items"]}
    assert st[a["id"]]["status"] == "done" and st[a["id"]]["paths"] == ["/out/a.png"]
    assert st[b["id"]]["status"] == "failed" and st[b["id"]]["error"] == "boom"


def test_another_lora_loads_again(q):
    _job("a")
    _job("b")
    for _ in range(2):
        studio_queue._tick()
    q["finish"]()
    studio_queue._tick()
    assert [x["primary"] for x in q["loads"]] == ["a", "b"]


def test_pause_never_resumes_by_itself_and_training_waits(q):
    _job()
    assert studio_queue.pause("You left Studio.")
    for _ in range(3):
        studio_queue._tick()
    assert not q["loads"] and studio_queue.status()["paused"]
    studio_queue.resume()
    q["busy"] = "Waiting for the training run to finish."
    studio_queue._tick()
    assert not q["loads"] and studio_queue.status()["waiting"] == q["busy"]
    q["busy"] = ""
    studio_queue._tick()
    assert q["loads"]


def test_pause_needs_waiting_jobs(q):
    assert not studio_queue.pause("x")           # an empty queue isn't marked paused


def test_a_page_edit_waits_for_a_job(monkeypatch):
    sent = []
    monkeypatch.setattr(studio, "_send", lambda m: sent.append(m))
    studio._S.update(state="ready", rendering=True, gen=5, pending=None, nolora_busy=False,
                     gens={5: {"job": "j1"}})
    r = studio.render({"seed": 1})
    assert r.get("after_job") and not sent and studio._S["pending"]
    studio._S["gens"] = {5: {}}                  # a page render going: the edit cancels it, as before
    studio.render({"seed": 2})
    assert sent == [{"cmd": "cancel"}]
    studio._S.update(rendering=False, pending=None, gens={})


def test_training_start_waits_for_a_studio_render(q, monkeypatch):
    _job()
    monkeypatch.setattr(studio, "generating", lambda: True)
    monkeypatch.setattr(server.captions, "status", lambda: {"state": "idle"})
    with server._lock:
        server._run.update(proc=None, state="idle")
    r = server._start_one("klein", {}, "", "", None, "", None, "")
    assert r.get("wait") and studio_queue.status()["paused"]


def test_waiting_on_studio_looks_again_soon(monkeypatch):
    calls = []
    monkeypatch.setattr(webqueue, "schedule", lambda d, s: calls.append(d))
    webqueue._q.update(items=[{"id": "x", "image_folder": ".", "lora": "l"}], held=False)
    webqueue.start_next(lambda item: (False, "wait-soon: Studio is finishing a render."))
    assert calls == [3] and webqueue._q["waiting"] == "Studio is finishing a render."
    webqueue._q.update(items=[], waiting="")
