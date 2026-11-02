"""Stop a rented pod once a run has finished on its own (the desktop's Preferences > RunPod switch): a two-minute
countdown anyone on the page can cancel, then RunPod's API stops - never terminates - the pod, so the files under
/workspace are still there when it starts again. Runs in the server, so it works with no page open: the point is the
unattended finish."""

import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request

COUNTDOWN = 120                # seconds
_lock = threading.Lock()
_s = {"deadline": 0.0, "gen": 0, "note": ""}


def on_pod() -> bool:
    """The Docker pod image (its own marker), or RunPod's id."""
    return os.environ.get("FIZGIG_POD", "0") not in ("0", "") or bool(os.environ.get("RUNPOD_POD_ID"))


def tunnels() -> dict:
    """The Vast image's HTTPS addresses (docker/entrypoint_vast.sh: Cloudflare quick tunnels), {"web": url, "files":
    url} - new each time the instance starts; {} without them."""
    try:
        with open("/tmp/fizgig_tunnels.json", encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if isinstance(v, str) and v.startswith("https://")}
    except (OSError, ValueError, AttributeError):
        return {}


def file_manager():
    """The pod's file manager (filebrowser on port 8080, started by the pod image's entrypoint) -> {url, login}, or
    None off a pod or without it. Its login is admin with the pod's password: "exact", "padded" (filebrowser
    refused it as too common, so the entrypoint added 0s up to 12 characters) or "none" (refused both ways). The
    entrypoint writes which; an older image wrote nothing and padded any password under 12 characters."""
    if not on_pod() or not shutil.which("filebrowser"):
        return None
    vast = os.environ.get("VAST_TCP_PORT_8080") and os.environ.get("PUBLIC_IPADDR")
    pid = os.environ.get("RUNPOD_POD_ID", "")    # not pod_id(): its HOSTNAME fallback would make up a RunPod address
    if not (pid or vast):
        return None
    try:
        with open("/tmp/fizgig_filebrowser_login", encoding="utf-8") as f:
            login = f.read().strip()
    except OSError:
        login = "padded" if len(os.environ.get("VNC_PASSWORD", "")) < 12 else "exact"
    # a Vast.ai instance: its https tunnel (the Vast image's), else its public address and the port Vast mapped 8080
    # to (plain http, as Vast serves ports)
    url = ((tunnels().get("files", "").rstrip("/") + "/" if tunnels().get("files") else
            f"http://{os.environ['PUBLIC_IPADDR']}:{os.environ['VAST_TCP_PORT_8080']}/") if vast and not
           os.environ.get("RUNPOD_POD_ID") else f"https://{pid}-8080.proxy.runpod.net/")
    return {"url": url, "login": login if login in ("exact", "padded", "none") else "exact"}


def pod_id() -> str:
    for k in ("RUNPOD_POD_ID", "HOSTNAME"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def api_key(prefs) -> str:
    """The key saved in Preferences, else RUNPOD_STOP_API_KEY. Never RunPod's injected RUNPOD_API_KEY: that one is
    pod-scoped and can't stop pods."""
    return str(prefs.get("runpod_api_key") or "").strip() or (os.environ.get("RUNPOD_STOP_API_KEY") or "").strip()


def status():
    with _lock:
        left = max(0, int(round(_s["deadline"] - time.time()))) if _s["deadline"] else 0
        return {"counting": bool(_s["deadline"]), "left": left, "note": _s["note"]}


def cancel(log):
    with _lock:
        was = bool(_s["deadline"])
        _s.update(deadline=0.0, gen=_s["gen"] + 1, note="")
    if was:
        log("[pod] auto-stop cancelled - pod left running.\n")
    return was


def _vast() -> bool:
    """A Vast.ai instance: it stops itself with its own key (CONTAINER_API_KEY), no key of the renter's needed."""
    return on_pod() and bool(os.environ.get("CONTAINER_API_KEY") and os.environ.get("CONTAINER_ID"))


def maybe_start(prefs, log):
    """After a run that finished on its own with nothing queued: the countdown, when the switch is on (a RunPod pod,
    through RunPod's API with the renter's key, or a Vast.ai instance, with its own)."""
    if not (os.environ.get("RUNPOD_POD_ID") or _vast()) or str(prefs.get("runpod_stop_when_done") or "0").strip() != "1":
        return
    with _lock:
        _s["gen"] += 1
        gen = _s["gen"]
        _s.update(deadline=time.time() + COUNTDOWN, note="")
    log(f"[pod] training finished - this pod stops in {COUNTDOWN} s to stop it billing (Keep the pod running cancels).\n")

    def wait():
        while True:
            with _lock:
                if _s["gen"] != gen:
                    return
                left = _s["deadline"] - time.time()
            if left <= 0:
                break
            time.sleep(min(1.0, left))
        with _lock:
            if _s["gen"] != gen:
                return
            _s["deadline"] = 0.0
        _stop(prefs, log)
    threading.Thread(target=wait, daemon=True).start()


def _stop_vast(log):
    """Vast's API, as the instance itself: PUT instances/<its id> {state: stopped}. The GPU stops billing; its disk
    bills until the instance is destroyed on Vast's site."""
    iid, key = os.environ.get("CONTAINER_ID", ""), os.environ.get("CONTAINER_API_KEY", "")
    log(f"[pod] stopping Vast.ai instance {iid}…\n")
    try:
        req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{iid}/",
                                     data=json.dumps({"state": "stopped"}).encode(), method="PUT",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            ok = json.loads(resp.read().decode() or "{}").get("success")
        msg = ("Stop requested - this instance is stopping. Its disk still bills until you destroy it on Vast's site "
               "(bring your LoRAs home first)." if ok else "Vast didn't confirm the stop: stop it on Vast's site.")
    except Exception as e:
        msg = f"Auto-stop failed ({type(e).__name__}: {e}). Stop it on Vast's site to stop billing."
    log("[pod] " + msg + "\n")
    with _lock:
        _s["note"] = msg


def _stop(prefs, log):
    if _vast() and not os.environ.get("RUNPOD_POD_ID"):
        return _stop_vast(log)
    key, pid = api_key(prefs), pod_id()
    if not key:
        msg = ("Auto-stop is on but no RunPod API key is set, so the pod is still running. Add one in Preferences > "
               "RunPod (RunPod > Settings > API Keys).")
        log("[pod] " + msg + "\n")
        with _lock:
            _s["note"] = msg
        return
    log(f"[pod] stopping pod {pid or '(unknown id)'}…\n")
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pid}/action", data=b'{"action": "stop"}',
                                 method="POST", headers={"Content-Type": "application/json", "User-Agent": "Fizgig",
                                                         "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=45):
            msg = "Stop requested - this pod is shutting down."
    except urllib.error.HTTPError as e:
        try:
            why = json.loads(e.read().decode() or "{}").get("detail") or f"HTTP {e.code}"
        except ValueError:
            why = f"HTTP {e.code}"
        msg = f"RunPod refused the stop: {why}. Stop it from the dashboard to stop billing."
    except Exception as e:
        msg = f"Auto-stop failed ({type(e).__name__}: {e}). Stop it from the dashboard to stop billing."
    log("[pod] " + msg + "\n")
    with _lock:
        _s["note"] = msg
