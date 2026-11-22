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


def _job(lora="a", family="klein"):
    return studio_queue.add({"family": family, "primary": lora}, {"state": {"seed": 1}, "extra": {}}, "p")


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


def test_another_model_loads_again(q):
    # another LoRA on the same model swaps in place instead (test_studio_queue_swap)
    _job("a")
    _job("b", family="zimage")
    for _ in range(2):
        studio_queue._tick()
    q["finish"]()
    studio_queue._tick()
    assert [(x["family"], x["primary"]) for x in q["loads"]] == [("klein", "a"), ("zimage", "b")]


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


def _after_job(monkeypatch, load_key, page_key):
    """A queue job's render ends while a page render waits: what happens to the waiting one."""
    sent = []
    monkeypatch.setattr(studio, "_send", lambda m: sent.append(m))
    monkeypatch.setattr(studio, "autosave", lambda gen: [])
    monkeypatch.setattr(studio, "on_job_end", lambda *a, **k: None)
    studio._S.update(state="ready", rendering=True, gen=9, gens={9: {"job": "j"}}, pending=None, nolora_busy=False,
                     load_key=load_key, page_key=page_key, page_waiting=0.0, proc=object())
    assert studio.render({"seed": 3}).get("after_job")
    studio._event({"ev": "rendered", "gen": 9})
    return sent


def test_your_waiting_render_never_runs_on_a_jobs_lora(monkeypatch):
    sent = _after_job(monkeypatch, load_key='{"primary": "B"}', page_key='{"primary": "A"}')
    assert not [m for m in sent if m.get("cmd") == "render"] and studio.page_waiting()
    studio._S.update(rendering=False, page_waiting=0.0)


def test_your_waiting_render_runs_on_your_own_load(monkeypatch):
    sent = _after_job(monkeypatch, load_key='{"primary": "A"}', page_key='{"primary": "A"}')
    assert [m for m in sent if m.get("cmd") == "render"] and not studio.page_waiting()
    studio._S.update(rendering=False, pending=None)


def test_a_page_render_on_a_jobs_lora_reloads_yours(monkeypatch):
    monkeypatch.setattr(studio, "_send", lambda m: None)
    studio._S.update(state="ready", rendering=False, gens={}, load_key='{"primary": "B"}', page_key='{"primary": "A"}',
                     page_waiting=0.0)
    assert studio.render({"seed": 1}).get("reload")
    studio._S.update(load_key=None, page_key=None, page_waiting=0.0)


def test_a_page_that_went_away_doesnt_hold_the_queue(monkeypatch):
    import time
    studio._S["page_waiting"] = time.time() - studio.PAGE_WAIT_S - 1
    assert not studio.page_waiting()
    studio._S["page_waiting"] = 0.0


def test_a_dead_workers_waiting_render_holds_nothing():
    studio._S.update(state="idle", rendering=False, pending=({"seed": 1}, None))
    assert not studio.generating()
    studio._S["pending"] = None


def test_resume_and_continue_wait_then_start_by_themselves(monkeypatch):
    import time
    calls = []

    def one(*a):
        calls.append(a)
        return {"wait": "Studio is finishing a render.", "error": "x"} if len(calls) < 3 else {"ok": True}
    monkeypatch.setattr(server, "_start_one", one)
    monkeypatch.setattr(server, "GPU_WAIT_S", 0.01)
    r = server._start("klein", {}, "", resume="state")
    assert r.get("ok") and r.get("queued") and server._training_waits()
    for _ in range(200):
        if len(calls) >= 3 and not server._gpu_wait["n"]:
            break
        time.sleep(0.01)
    assert len(calls) == 3 and not server._gpu_wait["n"]
    assert server._start("klein", {}, "", raw_wait=True).get("ok")     # Start Training takes the wait as it is


def test_studio_starts_nothing_while_a_run_waits(monkeypatch):
    monkeypatch.setattr(server, "_training_waits", lambda: True)
    with pytest.raises(ValueError):
        server._studio_post("render", {"state": {}})
    assert "training run" in server._studio_queue_busy()
