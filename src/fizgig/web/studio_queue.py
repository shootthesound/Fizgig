"""Studio's own queue: renders lined up to run one after another on Studio's GPU worker, separate from the training
queue (web/queue.py).

A job is everything Studio needs for one render, taken when it was queued: the model load (model, LoRAs, preview
settings) and the render (sliders, prompt, seed, size, a clip's settings). Editing Studio afterwards doesn't change
it. Each finished job is saved into the Studio outputs folder, whatever Save every render says (the panes show only
the newest render). A job naming another model or LoRA loads it first.

When it waits or pauses:
- Paused by leaving the Studio tab, by a training run starting, or by a restart; nothing resumes it but Resume (the
  page asks when you come back to Studio). The job rendering when it pauses finishes.
- Waits (not paused) while a training run, caption job or the Profiler has the GPU, and while a render you asked
  for on the page (or LoRA Royale's) is going.
- A job that fails is marked failed, and the next one runs.

server.py calls configure() with the load it does for Studio's Start and the GPU check; studio.py reports the end of
each job's render through on_job_end.
"""
import json
import os
import threading
import time
import uuid

from fizgig.user_data import user_file
from fizgig.web import studio

QUEUE_FILE = user_file("studio_queue.json")
DONE_KEEP = 30                                 # finished / failed jobs kept in the list

_lock = threading.Lock()
_q = {"items": [], "paused": False, "reason": "", "waiting": "", "job": "", "loaded": False}
_hooks = {"load": None, "busy": None}
_wake = threading.Event()
_thread = []


def _save():
    tmp = QUEUE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_q["items"], f, default=str)
        os.replace(tmp, QUEUE_FILE)
    except OSError:
        pass


def configure(load, busy):
    """load(body): Studio's load as the Start button does it; busy() -> "" or why the GPU isn't free (training...)."""
    _hooks.update(load=load, busy=busy)
    studio.on_job_end = job_ended
    with _lock:
        if not _q["loaded"]:
            _q["loaded"] = True
            try:
                with open(QUEUE_FILE, encoding="utf-8") as f:
                    items = [i for i in json.load(f) if isinstance(i, dict) and i.get("id")]
            except (OSError, ValueError):
                items = []
            for i in items:                      # a job cut off by the restart runs again
                if i.get("status") in ("loading", "rendering"):
                    i["status"] = "waiting"
            _q["items"] = items
            if any(i.get("status") == "waiting" for i in items):
                _q.update(paused=True, reason="Fizgig restarted.")
    if not _thread:
        t = threading.Thread(target=_runner, daemon=True)
        _thread.append(t)
        t.start()


def _waiting():
    return [i for i in _q["items"] if i.get("status") == "waiting"]


def status():
    with _lock:
        return {"items": [{k: i.get(k) for k in ("id", "label", "status", "paths", "error", "queued_at", "lora",
                                                 "model")} for i in _q["items"]],
                "paused": _q["paused"], "reason": _q["reason"], "waiting": _q["waiting"],
                "left": len(_waiting()), "job": _q["job"]}


def add(load, render, label, lora="", model=""):
    if not isinstance(load, dict) or not isinstance(render, dict) or not isinstance(render.get("state"), dict):
        raise ValueError("Not a Studio render.")
    item = {"id": uuid.uuid4().hex[:10], "load": load, "render": render, "label": str(label or "")[:200],
            "lora": str(lora or ""), "model": str(model or ""), "status": "waiting", "paths": [], "error": "",
            "queued_at": time.strftime("%Y-%m-%d %H:%M")}
    with _lock:
        _q["items"].append(item)
        _save()
    _wake.set()
    return item


def remove(item_id):
    """Takes a job off the list (a waiting one, or a finished one's line). The job rendering now can't be removed:
    Cancel stops it."""
    with _lock:
        it = next((i for i in _q["items"] if i["id"] == item_id), None)
        if it is None:
            return
        if it.get("status") in ("loading", "rendering"):
            raise ValueError("That job is rendering now: Cancel stops it.")
        _q["items"] = [i for i in _q["items"] if i["id"] != item_id]
        _save()


def clear_done():
    with _lock:
        _q["items"] = [i for i in _q["items"] if i.get("status") not in ("done", "failed")]
        _save()


def move(item_id, delta):
    """A waiting job earlier (-1) or later (+1) among the waiting ones."""
    with _lock:
        w = _waiting()
        ids = [i["id"] for i in w]
        if item_id not in ids:
            return
        a = ids.index(item_id)
        b = a + (1 if delta > 0 else -1)
        if not 0 <= b < len(w):
            return
        items = _q["items"]
        ia, ib = items.index(w[a]), items.index(w[b])
        items[ia], items[ib] = items[ib], items[ia]
        _save()


def pause(reason):
    """Paused (only when jobs are waiting): the job rendering now finishes, nothing else starts until Resume."""
    with _lock:
        if _waiting() and not _q["paused"]:
            _q.update(paused=True, reason=str(reason or ""), waiting="")
            return True
    return False


def resume():
    with _lock:
        _q.update(paused=False, reason="")
    _wake.set()


def running():
    """The id of the job loading or rendering now ("" for none)."""
    with _lock:
        return _q["job"]


def job_ended(job_id, ok, paths=None, error=""):
    """studio.py: a job's render finished (its saved files), failed or was cancelled."""
    with _lock:
        it = next((i for i in _q["items"] if i["id"] == job_id), None)
        if it is not None:
            it.update(status="done" if ok else "failed", paths=list(paths or []), error=str(error or ""),
                      ended_at=time.strftime("%Y-%m-%d %H:%M"))
        if _q["job"] == job_id:
            _q["job"] = ""
        done = [i for i in _q["items"] if i.get("status") in ("done", "failed")]
        if len(done) > DONE_KEEP:                # the oldest finished lines go
            drop = {i["id"] for i in done[:len(done) - DONE_KEEP]}
            _q["items"] = [i for i in _q["items"] if i["id"] not in drop]
        _save()
    _wake.set()


def _fail(it, why):
    with _lock:
        it.update(status="failed", error=why)
        if _q["job"] == it["id"]:
            _q["job"] = ""
        _save()


def _runner():
    while True:
        _wake.wait(1.0)
        _wake.clear()
        try:
            _tick()
        except Exception as e:                   # the queue never dies with one bad job
            print(f"[studio queue] {e}", flush=True)


def _say(why):
    with _lock:
        _q["waiting"] = why


def _tick():
    with _lock:
        if _q["job"]:
            it = next((i for i in _q["items"] if i["id"] == _q["job"]), None)
        else:
            it = None
            if not _q["paused"]:
                w = _waiting()
                it = w[0] if w else None
            if it is None:
                _q["waiting"] = ""
                return
    if it is None:                               # the job went (removed): free again
        with _lock:
            _q["job"] = ""
        return
    st = studio.status()
    if it.get("status") == "loading":            # a job's load: wait for the model, then render
        if st["state"] == "loading":
            return
        if st["state"] != "ready":
            _fail(it, st.get("error") or "The model didn't load.")
            return
        _render(it)
        return
    if it.get("status") == "rendering":
        if not studio.generating() and studio.current_job() != it["id"]:     # the worker stopped under it
            _fail(it, st.get("error") or "The render didn't finish.")
        return
    why = _hooks["busy"]() if _hooks["busy"] else ""
    if why:
        _say(why)
        return
    if studio.generating():
        _say("Waiting for the render going now to finish.")
        return
    _say("")
    key = json.dumps(it["load"], sort_keys=True)
    with _lock:
        _q["job"] = it["id"]
    if st["state"] == "ready" and studio.load_key() == key:
        _render(it)
        return
    try:
        _hooks["load"](it["load"])
        studio.set_load_key(key)
        with _lock:
            it["status"] = "loading"
    except Exception as e:
        _fail(it, f"Couldn't load: {e}")


def _render(it):
    r = it["render"]
    started = studio.render_job(r.get("state") or {}, r.get("extra") or {}, it["id"])
    if not started:                              # something else took the worker meanwhile: next time round
        with _lock:
            it["status"] = "waiting"
            _q["job"] = ""
        return
    with _lock:
        it["status"] = "rendering"
        _save()
