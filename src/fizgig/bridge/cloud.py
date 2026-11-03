"""Training on a rented RunPod pod, start to finish: rent the cheapest card that can train the model under a price
cap, wait for it with a deadline (a pod that doesn't come up is deleted and the next offer tried), send the training
folder, train there, bring every LoRA file home, delete the pod. The GPU here stays free all the while (a local
language model keeps its place on it).

The limits are kept here, not by whoever asked: a watchdog thread deletes the job's pods when the money cap or the
time limit is reached, whatever the job's own thread is doing. Pods this creates are written to
user_data/bridge_cloud.json until they're gone, so a Fizgig that stopped mid-job deletes them at its next start.

One job at a time. Torch-free."""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import zipfile

from fizgig.user_data import user_file
from fizgig.web import pods
from fizgig.web.helpers import Client

POD_FILE = "bridge_cloud.json"
TRAIN_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt", ".json")
_lock = threading.Lock()
_job: dict = {}


class Stop(Exception):
    """The job ends here (a cap, a cancel, nothing left to try): the reason is the message."""


def _pods_file() -> str:
    return user_file(POD_FILE)


def _remember(pod_id, gone=False):
    with _lock:
        try:
            with open(_pods_file(), encoding="utf-8") as f:
                ids = set(json.load(f))
        except (OSError, ValueError):
            ids = set()
        (ids.discard if gone else ids.add)(pod_id)
        with open(_pods_file(), "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)


def leftovers(key) -> list:
    """Delete the pods an earlier job made and never deleted (Fizgig stopped mid-job). Only those: never another pod."""
    try:
        with open(_pods_file(), encoding="utf-8") as f:
            ids = list(json.load(f))
    except (OSError, ValueError):
        return []
    done = []
    for pid in ids:
        if _delete(key, pid):
            done.append(pid)
    return done


def _delete(key, pod_id) -> bool:
    """The pod deleted and checked gone (RunPod says no such pod)."""
    for _ in range(4):
        try:
            pods.terminate(key, pod_id)
        except ValueError:
            pass                         # gone already, or a passing refusal: the check below decides
        try:
            pods._rest(key, "GET", f"/pods/{urllib.parse.quote(pod_id)}", timeout=15)
        except LookupError:
            _remember(pod_id, gone=True)
            return True
        except ValueError:
            pass
        time.sleep(5)
    return False


def candidates(key, max_price, min_vram) -> list:
    """The cards to try, cheapest for the work first (price over speed: a card twice as fast at 1.5x the price
    costs less per run): whole NVIDIA cards with at least `min_vram` GB, in stock, at or under `max_price` an hour
    (Secure and Community Cloud both, each its own offer)."""
    out = []
    for g in pods.offers(key).get("gpus") or []:
        if g.get("vram", 0) < min_vram or not g.get("id"):
            continue
        for cloud in ("secure", "community"):
            o = g.get(cloud)
            if o and o["price"] <= max_price:
                out.append({"gpu": g["gpu"], "id": g["id"], "vram": g["vram"], "cloud": cloud.upper(),
                            "price": o["price"], "stock": o.get("stock") or "", "speed": g.get("speed") or 0.3})
    rank = {"High": 0, "Medium": 1, "Low": 2}
    out.sort(key=lambda o: (round(o["price"] / o["speed"], 2), rank.get(o["stock"], 3)))
    return out


def status() -> dict:
    with _lock:
        j = {k: v for k, v in _job.items() if not k.startswith("_")}
    if j:
        j["spent"] = round(_spent(), 3)
        j["minutes"] = round(((j.get("ended") or time.time()) - j["started"]) / 60, 1)
    return j


def cancel() -> dict:
    with _lock:
        if _job.get("state") not in ("renting", "booting", "sending", "training", "fetching"):
            return {"cancelled": False}
        _job["_abort"] = "Cancelled."
    return {"cancelled": True}


def _spent() -> float:
    """Dollars so far: each pod's hourly price for the time it has existed (RunPod bills from creation)."""
    now = time.time()
    return sum(p["price"] * ((p.get("ended") or now) - p["created"]) / 3600 for p in _job.get("pods", []))


def _note(text, **kw):
    with _lock:
        _job.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {text}")
        _job.update(kw)


def _check():
    """Raise Stop when the watchdog (or a cancel) has ended the job."""
    reason = _job.get("_abort")
    if reason:
        raise Stop(reason)


def start(key, folder, family, *, preset="", overrides=None, lora_name="", trigger="", dest_root="",
          max_price=2.0, max_total=4.0, min_vram=32, boot_minutes=15, stall_minutes=8, max_minutes=75,
          attempts=3) -> dict:
    if not key:
        raise ValueError("No RunPod API key: add one on the Pods tab (read and write).")
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} isn't a folder.")
    with _lock:
        if _job.get("state") in ("renting", "booting", "sending", "training", "fetching"):
            raise ValueError("A cloud training job is already going: wait for it or cancel it.")
        lora_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", lora_name or os.path.basename(folder.rstrip("\\/")))[:60]
        _job.clear()
        _job.update(state="renting", started=time.time(), folder=folder, family=family, lora=lora_name, pods=[],
                    log=[], caps={"max_price": max_price, "max_total": max_total, "boot_minutes": boot_minutes,
                                  "max_minutes": max_minutes}, files=[], error="")
    args = dict(key=key, folder=folder, family=family, preset=preset, overrides=overrides or {}, lora=lora_name,
                trigger=trigger, dest=os.path.join(dest_root or user_file("cloud"), lora_name),
                max_price=max_price, min_vram=min_vram, boot=boot_minutes * 60, stall=stall_minutes * 60,
                attempts=attempts)
    threading.Thread(target=_work, kwargs=args, daemon=True, name="bridge-cloud").start()
    threading.Thread(target=_watchdog, args=(key, max_total, max_minutes * 60), daemon=True,
                     name="bridge-cloud-watchdog").start()
    return status()


def _watchdog(key, max_total, max_seconds):
    """The caps, kept whatever the job is doing: over the money or the time, every live pod of the job is deleted."""
    t0 = _job["started"]
    while _job.get("state") in ("renting", "booting", "sending", "training", "fetching"):
        reason = ""
        if _spent() >= max_total:
            reason = f"The ${max_total:.2f} spending cap was reached."
        elif time.time() - t0 >= max_seconds:
            reason = f"The {max_seconds // 60:.0f}-minute limit was reached."
        if reason:
            _job["_abort"] = reason
            for p in list(_job.get("pods", [])):
                if not p.get("ended"):
                    _delete(key, p["id"])
                    p["ended"] = time.time()
            return
        time.sleep(10)


def _work(key, folder, family, preset, overrides, lora, trigger, dest, max_price, min_vram, boot, stall, attempts):
    pod = None
    try:
        gone = leftovers(key)
        if gone:
            _note(f"Deleted {len(gone)} pod{'s' if len(gone) != 1 else ''} an earlier job left behind.")
        offers = candidates(key, max_price, min_vram)
        if not offers:
            raise Stop(f"No card with {min_vram} GB or more is free at ${max_price:.2f}/hr or less right now.")
        _note(f"{len(offers)} offers at or under ${max_price:.2f}/hr; best value {offers[0]['gpu']} "
              f"({offers[0]['cloud'].lower()}) ${offers[0]['price']:.2f}/hr.")
        password = secrets.token_urlsafe(12)
        tried = 0
        for offer in offers:
            if tried >= attempts:
                break
            _check()
            pod = _rent(key, offer, password, tried + 1)
            if pod is None:                  # refused (sold out since the list): no money spent, not a try
                continue
            tried += 1
            if _boot(key, pod, boot):
                break
            _end(key, pod)
            pod = None
        if pod is None:
            raise Stop(f"No pod came up after {tried} tr{'y' if tried == 1 else 'ies'}.")
        c = Client(pods.pod_url(pod["id"]))
        c.login(password)
        _note("Sending the training folder.", state="sending")
        remote = _send(c, folder, lora)
        _train(c, remote, family, preset, overrides, lora, trigger)
        _watch(key, c, pod, stall)
        files = _bring_home(c, dest)
        _end(key, pod)
        pod = None
        _note(f"Done: {len(files)} LoRA file{'s' if len(files) != 1 else ''} in {dest}. The pod is deleted.",
              state="done", files=files, dest=dest, newest=files[0] if files else "")
    except Stop as e:
        _note(str(e), state="failed", error=str(e))
    except Exception as e:                                       # noqa: BLE001 - whatever it is, the pod goes
        _note(f"Failed: {e}", state="failed", error=str(e) or type(e).__name__)
    finally:
        for p in _job.get("pods", []):
            if not p.get("ended"):
                _end(key, p)
        if _job.get("state") == "failed":
            _note("The pods of this job are deleted. To train here instead, start training on this computer.",
                  fallback="local")
        _job["ended"] = time.time()


def _rent(key, offer, password, n):
    _note(f"Try {n}: renting {offer['gpu']} ({offer['cloud'].lower()}) at ${offer['price']:.2f}/hr.",
          state="renting")
    body = {"name": f"fizgig-cloud-{time.strftime('%m%d-%H%M')}", "templateId": pods.TEMPLATE,
            "gpu": {"id": offer["id"], "count": 1, "minCudaVersion": pods.MIN_CUDA},
            "cloud": offer["cloud"], "env": {"FIZGIG_PASSWORD": password}}
    try:
        r = pods._rest(key, "POST", "/pods", body, timeout=60)
    except ValueError as e:
        _note(f"RunPod refused: {e}")
        return None
    pod = {"id": r.get("id"), "gpu": offer["gpu"], "price": offer["price"], "created": time.time()}
    if not pod["id"]:
        _note("RunPod answered without a pod id.")
        return None
    _remember(pod["id"])
    with _lock:
        _job["pods"].append(pod)
        _job["pod"] = pod["id"]
    return pod


def _boot(key, pod, deadline) -> bool:
    """Wait for the pod's Fizgig to answer, up to `deadline` seconds. False: it didn't (or the pod died)."""
    _note(f"Pod {pod['id']} rented; waiting for Fizgig on it (up to {deadline // 60:.0f} min).", state="booting")
    end = time.time() + deadline
    url = pods.pod_url(pod["id"])
    while time.time() < end:
        _check()
        try:
            st = str(pods._rest(key, "GET", f"/pods/{pod['id']}", timeout=15).get("status") or "").upper()
        except LookupError:
            _note("The pod vanished while starting.")
            return False
        except ValueError:
            st = ""
        if st in ("EXITED", "TERMINATED"):
            _note(f"The pod stopped while starting ({st.lower()}).")
            return False
        if pods.probe(url) == "ready":
            pod["ready"] = time.time()
            _note(f"Fizgig answered on the pod after {(pod['ready'] - pod['created']) / 60:.1f} min.")
            return True
        time.sleep(10)
    _note(f"No answer from the pod in {deadline // 60:.0f} min: deleting it.")
    return False


def _end(key, pod):
    ok = _delete(key, pod["id"])
    pod["ended"] = time.time()
    if not ok:
        _note(f"Couldn't confirm pod {pod['id']} is deleted: check RunPod's console.")


def _send(c, folder, name) -> str:
    """The folder's training files (top level) zipped and sent; the pod unpacks it into its training data."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for e in sorted(os.scandir(folder), key=lambda e: e.name):
            if e.is_file() and e.name.lower().endswith(TRAIN_EXTS):
                z.write(e.path, e.name)
    data = buf.getvalue()
    for attempt in range(3):
        _check()
        try:
            r = c.call("/api/upload?" + urllib.parse.urlencode({"folder": name, "name": name + ".zip"}),
                       raw=data, timeout=600)
            break
        except (OSError, ValueError) as e:
            if attempt == 2:
                raise
            _note(f"Sending again ({e}).")
    _note(f"Sent {len(data) / 1e6:.1f} MB: {len(r.get('saved') or [])} files.")
    return r.get("folder") or ""


def _train(c, folder, family, preset, overrides, lora, trigger):
    if trigger:
        c.call("/api/trigger", {"folder": folder, "word": trigger})
    values = dict(((c.call("/api/state").get("values") or {}).get(family)) or {})
    if preset:
        values = c.call("/api/preset", {"family": family, "name": preset, "values": values}).get("values", values)
    values.update(overrides or {})
    values["LORA_NAME"] = lora
    r = c.call("/api/start", {"family": family, "values": values, "image_folder": folder, "anyway": True},
               timeout=120)
    if r.get("problems"):
        raise Stop("The pod wouldn't start the run: " + "; ".join(r["problems"]))
    _note("Downloading the model on the pod, then training." if r.get("fetching") else "Training on the pod.",
          state="training")


def _watch(key, c, pod, stall):
    """Until the pod's run ends. A pod that stops answering for `stall` seconds is checked on RunPod: dead or gone
    ends the job (deleted); answering again carries on."""
    seen, quiet_since, last_epoch, fetch_seen = False, None, None, False
    while True:
        _check()
        time.sleep(15)
        try:
            run = c.call("/api/run", timeout=30)
            fetch = c.call("/api/fetch", timeout=30)
            quiet_since = None
        except PermissionError:
            raise Stop("The pod logged this computer out.")
        except (OSError, ValueError) as e:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since < stall:
                continue
            try:
                st = str(pods._rest(key, "GET", f"/pods/{pod['id']}", timeout=15).get("status") or "").upper()
            except LookupError:
                st = "gone"
            except ValueError:
                st = "unknown"
            if st != "RUNNING":
                raise Stop(f"The pod died mid-run ({st.lower()}; last error: {e}).")
            raise Stop(f"The pod has been silent for {stall // 60:.0f} min: given up on.")
        fetching = fetch.get("state") == "running"
        if fetching:
            fetch_seen = True
            pct = 100 * fetch.get("done", 0) / max(1, fetch.get("total") or 1)
            with _lock:
                _job["phase"] = f"downloading the model on the pod ({pct:.0f}%)"
            continue
        if fetch_seen and fetch.get("state") == "failed":
            raise Stop("The model download on the pod failed.")
        running = run.get("running") or run.get("state") in ("starting", "running", "caching")
        if running:
            seen = True
            stage = run.get("stage") or run.get("state")
            if stage != last_epoch:
                last_epoch = stage
                with _lock:
                    _job["phase"] = f"on the pod: {stage}"
            continue
        if seen or (not fetch.get("then_start") and run.get("state") in ("done", "failed", "stopped", "paused")
                    and run.get("lora") == _job.get("lora")):
            state = run.get("state")
            _note(f"The run on the pod ended ({state}).", state="fetching")
            _job["remote_out"] = run.get("output_dir") or ""
            if state not in ("done", "completed"):
                _job["remote_state"] = state
            return


def _bring_home(c, dest) -> list:
    out = _job.get("remote_out") or ""
    items = (c.call("/api/checkpoints?dir=" + urllib.parse.quote(out)).get("items") or []) if out else []
    lora = _job.get("lora") or ""
    items = [i for i in items if i["name"].endswith(".safetensors") and i.get("run") in (lora, lora + ".safetensors")]
    if not items:
        raise Stop("The run on the pod saved no LoRA" + (f" (it {_job['remote_state']})."
                                                         if _job.get("remote_state") else "."))
    os.makedirs(dest, exist_ok=True)
    got = []
    for i in items:                                  # newest first, as the pod lists them
        _check()
        path = os.path.join(dest, os.path.basename(i["name"]))
        if not (os.path.isfile(path) and os.path.getsize(path) == i["size"]):
            c.download("/api/download?path=" + urllib.parse.quote(i["path"]), path, lambda n: None)
        got.append(path)
    _note(f"Brought home {len(got)} LoRA file{'s' if len(got) != 1 else ''}.")
    return got
