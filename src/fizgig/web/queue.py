"""The web UI's training queue: runs lined up to start one after another, kept by the web server.

It is the web UI's own (user_data/web_queue.json in the Fizgig folder, git-ignored) - the desktop app keeps its own
queue in presets/training_queue.json, and neither app edits the other's. One run trains at a time across both: the
next queued run waits while the desktop app is training or making captions (and the desktop's queue waits for this
one the same way).

The queue moves on only after a run finishes. A run that fails, is stopped or pauses holds it (with the reason), and
so does a restart of the web UI - nothing launches by surprise; Start queue carries on. A paused run is waited for:
the queue carries on after it is resumed and finishes, as the desktop's does. server.py calls on_run_ended()
when a run ends and start_next() to launch; this module never touches the run itself.
"""
import json
import os
import threading
import time
import uuid

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
QUEUE_FILE = user_file("web_queue.json")
WAIT_S = 30                                   # how often a waiting queue looks again

_lock = threading.Lock()
_q = {"items": [], "held": False, "reason": "", "waiting": "", "timer": None, "for_pause": False, "failed": []}


def _save():
    tmp = QUEUE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_q["items"], f, default=str)
        os.replace(tmp, QUEUE_FILE)
    except OSError:
        pass


def load():
    """At start-up: the queue as it was, held if it holds anything (a restart never launches a run by itself)."""
    try:
        with open(QUEUE_FILE, encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, ValueError):
        items = []
    items = [i for i in items if isinstance(i, dict) and i.get("family") and isinstance(i.get("values"), dict)]
    with _lock:
        _q.update(items=items, held=bool(items), waiting="",
                  reason=("Your queued runs from Fizgig 7 are here: press Start queue when you want them to run."
                          if any(i.get("from_fizgig_7") for i in items) else
                          "The web UI restarted: press Start queue to carry on.") if items else "")


def status():
    with _lock:
        return {"items": [{**{k: i.get(k) for k in ("id", "family", "lora", "image_folder", "queued_at", "output_dir")},
                           "continues": bool((i.get("values") or {}).get("__continue"))}     # Continue on Runs & outputs
                          for i in _q["items"]],
                "held": _q["held"], "reason": _q["reason"], "waiting": _q["waiting"],
                # the runs that failed (an out-of-memory recovery that gave up too) - the queue moved on past them
                "failed": (f"{', '.join(_q['failed'][-3:])} failed - see the log on its card; the queue moved on"
                           if _q["failed"] else "")}


def add(family, values, image_folder, lora, output_dir):
    item = {"id": uuid.uuid4().hex[:10], "family": family, "values": dict(values), "image_folder": image_folder,
            "lora": lora, "output_dir": output_dir, "queued_at": time.strftime("%Y-%m-%d %H:%M")}
    with _lock:
        _q["items"].append(item)
        _save()
    return item


def clashes(output_dir, lora, skip=""):
    """A queued run that would save into the same files (same output folder and LoRA name), or None. skip: the item
    being edited, which may of course keep its own name."""
    key = (os.path.normcase(os.path.abspath(output_dir or "")), (lora or "").lower())
    with _lock:
        for n, i in enumerate(_q["items"], 1):
            if i["id"] != skip and (os.path.normcase(os.path.abspath(i.get("output_dir") or "")),
                                    (i.get("lora") or "").lower()) == key:
                return n
    return None


def get(item_id):
    with _lock:
        return next((dict(i) for i in _q["items"] if i["id"] == item_id), None)


def update(item_id, family, values, image_folder, lora, output_dir):
    """A queued run's settings replaced (Edit on the queue): it keeps its place. False when it has gone (started)."""
    with _lock:
        for i in _q["items"]:
            if i["id"] == item_id:
                i.update(family=family, values=dict(values), image_folder=image_folder, lora=lora,
                         output_dir=output_dir)
                _save()
                return True
    return False


def remove(item_id):
    with _lock:
        _q["items"] = [i for i in _q["items"] if i["id"] != item_id]
        if not _q["items"]:
            _q.update(held=False, reason="", waiting="")
        _save()


def move(item_id, delta):
    with _lock:
        items = _q["items"]
        i = next((n for n, x in enumerate(items) if x["id"] == item_id), None)
        j = None if i is None else i + delta
        if i is not None and 0 <= j < len(items):
            items[i], items[j] = items[j], items[i]
            _save()


def hold(reason, for_pause=False):
    """for_pause: held for a paused run, which the queue waits for - it carries on after that run finishes."""
    with _lock:
        _cancel_timer()
        if _q["items"]:
            _q.update(held=True, reason=reason, waiting="", for_pause=for_pause)


def _cancel_timer():
    t = _q.get("timer")
    if t is not None:
        t.cancel()
    _q["timer"] = None


def schedule(delay_s, starter):
    """Look again (and start the next run if it can) in delay_s seconds. One timer at a time."""
    with _lock:
        _cancel_timer()
        t = threading.Timer(delay_s, lambda: start_next(starter))
        t.daemon = True
        _q["timer"] = t
        t.start()


def unhold():
    """No longer held, without starting anything: a run is about to start, and the queue follows it as any run."""
    with _lock:
        _q.update(held=False, reason="", waiting="", for_pause=False)


def release(starter):
    """Start queue: no longer held; the next run starts now if it can."""
    with _lock:
        _q.update(held=False, reason="", waiting="", for_pause=False)
    start_next(starter)


def resumed():
    """The paused run the queue was held for is training again: the queue waits for it as for any run, and carries on
    after it finishes (not held, no Start queue)."""
    with _lock:
        if _q["held"] and _q["for_pause"]:
            _q.update(held=False, reason="", for_pause=False)


def on_run_ended(result, lora, starter):
    """A web run ended: the next one starts shortly after a finish; anything else holds the queue, saying why."""
    if result == "finished":
        with _lock:
            if _q["for_pause"]:              # the paused run (resumed) is done: the queue carries on
                _q.update(held=False, reason="", for_pause=False)
        schedule(10, starter)
    elif result == "failed":            # an overnight queue doesn't stall on one run: it's named, and the next starts
        with _lock:
            _q["failed"] = (_q["failed"] + [str(lora or "a run")])[-10:]
        schedule(10, starter)
    elif result == "stopped":
        hold("You stopped the last run. Press Start queue to carry on with the next.")
    elif result == "paused":
        hold("The last run is paused. Resume it: the queue carries on after it finishes.", for_pause=True)


def start_next(starter):
    """The head of the queue, started through starter(item) -> (started, why). Waits (and says so) while the GPU is
    busy elsewhere; a run that won't start goes back at the head and holds the queue."""
    with _lock:
        _q["timer"] = None
        if _q["held"] or not _q["items"]:
            _q["waiting"] = ""
            return
        item = _q["items"][0]
    if not os.path.isdir(item.get("image_folder") or ""):
        hold(f"The next run's training folder isn't there any more: {item.get('image_folder')}. Restore it or "
             "remove the run, then press Start queue.")
        return
    started, why = starter(item)
    if started:
        with _lock:
            _q["items"] = [i for i in _q["items"] if i["id"] != item["id"]]
            _q["waiting"] = ""
            _save()
        return
    if why.startswith("wait:"):                  # the GPU is busy elsewhere: look again soon
        with _lock:
            _q["waiting"] = why[5:].strip()
        schedule(WAIT_S, starter)
        return
    hold(f"The next run ({item.get('lora')}) couldn't start: {why}")
