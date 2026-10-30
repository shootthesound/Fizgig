"""A pod runs the same Fizgig as the computer using it. A pod takes the newest Fizgig (master) at every start; when
this computer's Fizgig (the app window on a pod's page, or a pod it has as a helper) finds the pod on another build,
it asks the pod to switch to its own: the pod checks out that commit and starts its web UI again (the pod's start-up
script runs it in a loop, so it comes straight back). The commit is kept on the volume (PIN), so every later start
of the pod comes back to it - the start-up script still updates to master first, and this file's boot() moves it on
to the pinned commit before the web UI binds its port. A pod set to a release of its own (FIZGIG_REF v8.9.92) is left
alone. Only a pod switches: another computer's Fizgig is never changed.

Nothing switches while the pod is busy (a run, a download, captions, a helper's job, a move): a restart would cut it
off. The computer asks again on its next look."""
import json
import os
import re
import subprocess
import sys
import threading
import time

from fizgig.web.port import on_pod

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PIN = "/workspace/.fizgig_pin"
_st = {"state": "idle", "commit": "", "error": "", "at": 0.0}     # idle / switching / failed
_lock = threading.Lock()


def _git(*args, timeout=120):
    r = subprocess.run(["git", "-C", BASE, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or r.stderr).strip()


def head() -> str:
    code, out = _git("rev-parse", "HEAD", timeout=10)
    return out if code == 0 else ""


def allowed() -> bool:
    """A pod following a branch (the template's master): one set to a release of its own (FIZGIG_REF v8.9.92) keeps
    it."""
    return on_pod() and not re.fullmatch(r"v\d+(\.\d+)*", str(os.environ.get("FIZGIG_REF") or "master").strip())


def pinned() -> dict:
    try:
        with open(PIN, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and re.fullmatch(r"[0-9a-f]{40}", str(d.get("commit") or "")) else {}
    except (OSError, ValueError):
        return {}


def busy() -> str:
    """Why the pod can't start its web UI again now, or ""."""
    from fizgig.web import captions, helpers, prep_jobs, server as s, studio
    with s._lock:
        if s._run["proc"] is not None and s._run["proc"].poll() is None or s._run["state"] in ("starting", "running"):
            return "a run is going on it"
        if s._fetch["proc"] is not None and s._fetch["proc"].poll() is None:
            return "it is downloading models"
    if prep_jobs.busy():
        return "Image Prep is working on it"
    if captions._busy():
        return "its captioner is loaded or working"
    if str(studio.status().get("state") or "idle") not in ("idle", ""):
        return "Studio has a model loaded on it"
    if helpers._worker_state().get("state") or helpers._out.get("url"):
        return "it is helping a computer"
    if any(w["proc"].poll() is None for w in list(helpers._sw.values())):
        return "it is sharing a run"
    if helpers.move_busy() or any(not m.get("gone") for m in helpers._mvin.values()):
        return "a run is moving to or from it"
    return ""


def status() -> dict:
    with _lock:
        d = dict(_st)
    pin = pinned()
    return {**d, "head": head(), "pinned": pin.get("commit", ""), "by": pin.get("by", ""), "allowed": allowed()}


def _deps(before):
    """The venv topped up when the switch changed requirements.txt (the start-up script did master's)."""
    try:
        with open(os.path.join(BASE, "requirements.txt"), encoding="utf-8") as f:
            after = f.read()
    except OSError:
        return
    if after == before:
        return
    print("[fizgig] requirements.txt differs in this version: installing its packages", flush=True)
    subprocess.run([sys.executable, os.path.join(BASE, "uv_install_deps.py"), os.path.join(BASE, "requirements.txt"),
                    os.environ.get("VIRTUAL_ENV") or os.path.dirname(os.path.dirname(sys.executable))])


def _checkout(commit) -> str:
    """The checkout moved to `commit` (fetched from GitHub when the shallow clone hasn't it): "" or why not."""
    if head() == commit:
        return ""
    try:
        with open(os.path.join(BASE, "requirements.txt"), encoding="utf-8") as f:
            reqs = f.read()
    except OSError:
        reqs = ""
    if _git("cat-file", "-e", commit + "^{commit}", timeout=10)[0] != 0:
        code, out = _git("fetch", "--depth", "1", "origin", commit, timeout=300)
        if code != 0:
            return ("that version isn't on GitHub (it may be changes that haven't been pushed), so the pod can't "
                    "get it" if "not our ref" in out or "couldn't find remote ref" in out or "unadvertised" in out
                    else f"getting it from GitHub failed: {out.splitlines()[-1] if out else 'no answer'}")
    code, out = _git("reset", "--hard", commit, timeout=60)
    if code != 0:
        return f"switching failed: {out.splitlines()[-1] if out else ''}"
    _deps(reqs)
    return ""


def match(commit, by="", auto=False) -> dict:
    """The page's or a computer's ask: run `commit`. {"same"} / {"switching"} / {"busy": why} / {"error": why}.
    auto: a computer's Pods list asking by itself, not someone opening the pod."""
    commit = str(commit or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return {"error": "That isn't a Fizgig version."}
    if not on_pod():
        return {"error": "Only a pod switches its Fizgig to match: update this computer's Fizgig with its updater."}
    if not allowed():
        return {"error": f"This pod is set to run {os.environ.get('FIZGIG_REF')} (its FIZGIG_REF), so it stays on it."}
    if head() == commit:
        return {"same": True}
    why = busy()
    if why:
        return {"busy": f"It switches once it's free: {why}."}
    pin = pinned()
    if auto and pin.get("by") and pin["by"] != by and time.time() - float(pin.get("at") or 0) < 30 * 60:
        mins = max(1, int((time.time() - float(pin["at"])) / 60))
        return {"busy": f"{pin['by']} put it on its Fizgig {mins} minute{'s' if mins > 1 else ''} ago: "
                        "open the pod from this computer to switch it to this one's."}
    with _lock:
        if _st["state"] == "switching" and time.time() - _st["at"] < 15 * 60:   # else it never came back: again
            return {"switching": True}
        _st.update(state="switching", commit=commit, error="", at=time.time())
    threading.Thread(target=_switch, args=(commit, by), daemon=True, name="pod-version").start()
    return {"switching": True}


def _switch(commit, by):
    from fizgig.web import server as s
    print(f"[fizgig] Switching this pod to the Fizgig of {by or 'the computer using it'} ({commit[:7]})", flush=True)
    why = _checkout(commit)
    if why:
        print(f"[fizgig] Couldn't switch: {why}", flush=True)
        with _lock:
            _st.update(state="failed", error=why[:1].upper() + why[1:] + ".")
        return
    try:
        with open(PIN, "w", encoding="utf-8") as f:
            json.dump({"commit": commit, "by": by, "at": time.time()}, f)
    except OSError as e:
        print(f"[fizgig] Couldn't keep the version for the next start ({e})", flush=True)
    print("[fizgig] Starting the web UI again on that version", flush=True)
    threading.Timer(0.5, s.shutdown_for_switch).start()


def newest() -> dict:
    """The pod back on the newest Fizgig: the pin is dropped and the web UI starts again on master."""
    if not on_pod():
        return {"error": "Only a pod does this."}
    why = busy()
    if why:
        return {"error": f"Not now: {why}."}
    try:
        os.remove(PIN)
    except OSError:
        pass
    with _lock:
        _st.update(state="switching", commit="", error="", at=time.time())

    def work():
        from fizgig.web import server as s
        ref = str(os.environ.get("FIZGIG_REF") or "master").strip() or "master"
        print(f"[fizgig] Back to the newest Fizgig ({ref})", flush=True)
        code, out = _git("fetch", "--depth", "1", "origin", ref, timeout=300)
        fetched = code == 0 and _git("rev-parse", "FETCH_HEAD", timeout=10)[1]
        why = "" if fetched else f"getting it from GitHub failed: {out.splitlines()[-1] if out else 'no answer'}"
        if not why:
            why = _checkout(fetched)
        if why:
            print(f"[fizgig] Couldn't: {why}", flush=True)
            with _lock:
                _st.update(state="failed", error=why[:1].upper() + why[1:] + ".")
            return
        threading.Timer(0.5, s.shutdown_for_switch).start()

    threading.Thread(target=work, daemon=True, name="pod-version").start()
    return {"switching": True}


def boot():
    """At the web UI's start on a pod: the start-up script put it on master; a pinned version comes back first, then
    this same command runs again on it. Any trouble: say so and carry on with what's checked out."""
    if not allowed():
        return
    pin = pinned().get("commit", "")
    if not pin or head() == pin:
        return
    print(f"[fizgig] This pod runs the Fizgig of the computer using it ({pin[:7]}): switching to it", flush=True)
    why = _checkout(pin)
    if why:
        print(f"[fizgig] Couldn't ({why}): starting on the newest Fizgig instead", flush=True)
        return
    os.execv(sys.executable, [sys.executable] + sys.argv)
