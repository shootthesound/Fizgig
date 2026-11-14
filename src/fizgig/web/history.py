"""Run history: every training run the web UI or the desktop app starts, newest first - for the web UI's Runs &
outputs tab (what ran, on what, how it ended, its previews and checkpoints, and its settings to start again with).

One file both apps write (user_data/runs.json in the Fizgig folder, git-ignored): each change re-reads it under a lock
file and writes it whole, so the two apps never lose each other's records. A record:

    {id, source: "web" | "desktop", family, lora, output_dir, image_folder, values (the settings), started_at,
     ended_at, result: "running" | "finished" | "failed" | "stopped" | "paused", progress (the last figures)}
"""
import json
import os
import time
import uuid

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HISTORY_FILE = user_file("runs.json")
KEEP = 300                                  # the newest runs kept


class _Lock:
    """A lock file the two apps share; a stale one (a crashed writer) is taken over after a few seconds."""

    def __init__(self):
        self.path = HISTORY_FILE + ".lock"
        self.fd = None

    def __enter__(self):
        for _ in range(100):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 5:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        return self                         # could not lock: carry on rather than lose the record

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.remove(self.path)
            except OSError:
                pass


def _read():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _write(runs):
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(runs[-KEEP:], f, default=str)
    os.replace(tmp, HISTORY_FILE)


def _same_run(r, lora, output_dir):
    norm = lambda p: os.path.normcase(os.path.abspath(str(p or ".")))
    return (str(r.get("lora") or "").lower() == str(lora or "").lower()
            and norm(r.get("output_dir")) == norm(output_dir))


def last_values(lora, output_dir):
    """The settings the newest run with this LoRA name and folder last trained with, or None (not a run Fizgig
    knows: a state folder from elsewhere)."""
    try:
        for r in reversed(_read()):
            if _same_run(r, lora, output_dir):
                return dict(r.get("values") or {})
    except OSError:
        pass
    return None


def image_folder_of(lora, output_dir):
    """The training folder the newest run with this LoRA name and output folder trained on, or "" (a run Fizgig has
    no record of)."""
    try:
        for r in reversed(_read()):
            if _same_run(r, lora, output_dir):
                return str(r.get("image_folder") or "")
    except OSError:
        pass
    return ""


def start(source, family, lora, output_dir, image_folder, values, resume=False, carry=""):
    """Record a run that is starting; returns its id. A resume carries on the paused run's own record (the newest
    paused one with the same LoRA name and folder, from either app) instead of listing the run twice; Continue on
    Runs & outputs carries on the record it was pressed on (carry: its id), however that run ended."""
    if resume:
        try:
            with _Lock():
                runs = _read()
                for r in reversed(runs):
                    if (r.get("id") == carry) if carry else _same_run(r, lora, output_dir):
                        if r.get("result") != "paused" and not carry:
                            break
                        r.update(result="running", ended_at=None, resumed_at=time.time(), server_pid=os.getpid(),
                                 values=values or r.get("values") or {},    # e.g. a new Target Megapixels
                                 # where its pictures are now (a run moved to a pod again unpacks somewhere new)
                                 image_folder=image_folder or r.get("image_folder"))
                        _write(runs)
                        return r["id"]
        except OSError:
            pass
    rec = {"id": uuid.uuid4().hex[:12], "source": source, "family": family, "lora": lora, "output_dir": output_dir,
           "image_folder": image_folder, "values": values or {}, "started_at": time.time(), "ended_at": None,
           "result": "running", "progress": None, "server_pid": os.getpid()}
    try:
        with _Lock():
            runs = _read()
            runs.append(rec)
            _write(runs)
    except OSError:
        pass
    return rec["id"]


def set_values(run_id, changes):
    """Settings changed while the run trains (a prompt slider's live prompts): its record's, so Start again uses
    them."""
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id and isinstance(r.get("values"), dict):
                    r["values"].update(changes)
            _write(runs)
    except OSError:
        pass


def set_field(run_id, key, value):
    """One field of a run's record (the out-of-memory recovery's tries, "oom")."""
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id:
                    r[key] = value
            _write(runs)
    except OSError:
        pass


def set_notes(run_id, text):
    """The run's notes (Runs & outputs): what you tried, what you thought of it."""
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id:
                    r["notes"] = str(text or "")[:20000]
            _write(runs)
    except OSError:
        pass


_curves = {}                                # run id -> [epoch on show, its last loss, [[epoch, loss], ...]]


def _set_curve(run_id, points):
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id:
                    r["curve"] = points[-2000:]
            _write(runs)
    except OSError:
        pass


def feed_loss(run_id, epoch, loss_text):
    """The run's average loss as the trainer reports it: each epoch's last value is kept on its record (written once
    an epoch, when the next begins, and at the end), for the loss curve on Runs & outputs."""
    if not run_id:
        return
    try:
        loss, epoch = float(loss_text), int(epoch)
    except (TypeError, ValueError):
        return
    c = _curves.get(run_id)
    if c is None:                            # a resumed run carries on its own curve
        rec = get(run_id) or {}
        c = _curves[run_id] = [None, None, [p for p in (rec.get("curve") or []) if isinstance(p, list)]]
    if c[0] is not None and epoch != c[0] and c[1] is not None:
        c[2] = [p for p in c[2] if p[0] != c[0]] + [[c[0], c[1]]]
        _set_curve(run_id, c[2])
    c[0], c[1] = epoch, loss


def _flush_curve(run_id):
    c = _curves.pop(run_id, None)
    if c and c[0] is not None and c[1] is not None:
        _set_curve(run_id, [p for p in c[2] if p[0] != c[0]] + [[c[0], c[1]]])


def end(run_id, result, progress=None):
    """A run ended (finished / failed / stopped / paused), with its last figures."""
    if not run_id:
        return
    _flush_curve(run_id)
    try:
        with _Lock():
            runs = _read()
            for r in runs:
                if r.get("id") == run_id:
                    r.update(result=result, ended_at=time.time(), progress=progress or r.get("progress"))
            _write(runs)
    except OSError:
        pass


def remove(run_id):
    """Take a run off the list (Runs & outputs' Delete run; its files are deleted by the caller)."""
    try:
        with _Lock():
            runs = _read()
            _write([r for r in runs if r.get("id") != run_id])
    except OSError:
        pass


def _server_alive(pid):
    """Whether the Fizgig server with this pid still runs (another one, not this process)."""
    if not pid or pid == os.getpid():
        return False
    try:
        import psutil
        return "python" in psutil.Process(int(pid)).name().lower()
    except Exception:
        return False


last_orphan = [""]          # the newest run close_orphans closed (the one Fizgig was training when it went away)


def close_orphans():
    """At the server's start: a run still recorded as running has nothing behind it now (its server stopped or
    crashed, and the training process went with it), so it is recorded as stopped - unless the server that started it
    is still running (a second Fizgig server on another port must not close a live run). Returns how many."""
    n = 0
    try:
        with _Lock():
            rs = _read()
            for r in rs:
                if r.get("result") == "running" and not _server_alive(r.get("server_pid")):
                    r["result"] = "stopped"
                    r["ended_at"] = r.get("ended_at") or time.time()
                    last_orphan[0] = str(r.get("id") or "")
                    n += 1
            if n:
                _write(rs)
    except OSError:
        pass
    return n


def runs():
    """Newest first. A run still marked running whose app is gone for good (a crash) reads as ended unknown, a week on."""
    out = list(reversed(_read()))
    for r in out:
        if r.get("result") == "running" and time.time() - float(r.get("started_at") or 0) > 7 * 86400:
            r["result"] = "unknown"
    return out


def get(run_id):
    return next((r for r in _read() if r.get("id") == run_id), None)
