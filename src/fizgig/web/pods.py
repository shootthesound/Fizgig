"""Your RunPod pods, from the Fizgig on your own computer (the Pods tab): the ones you added by ID or address and,
with a RunPod API key, every Fizgig pod on your account, each with whether its Fizgig is answering yet. Start and
Stop go through RunPod's API. Opening a pod is the page's job: the app window goes there, a browser opens a tab.

Torch-free, like the rest of the server; every call to RunPod or a pod has a short timeout, so a slow pod never
holds the page up."""

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.runpod.io/graphql"
TEMPLATE = "bdrdew7hxw"
REF = "vkb387ep"                    # Fizgig's RunPod referral: every Deploy link carries it


def deploy_url(gpu="RTX 5090", count=1) -> str:
    """RunPod's deploy page with Fizgig's template and this GPU (one card or two) picked: the person presses Deploy
    there."""
    return "https://console.runpod.io/deploy?" + urllib.parse.urlencode(
        {"type": "GPU", "gpu": gpu, "count": int(count or 1), "template": TEMPLATE, "ref": REF})


MIN_CUDA = "12.8"                   # the pod image's PyTorch (cu128) needs a driver this new: stock on such hosts only


DEPLOY = deploy_url()
UA = "Fizgig"                       # RunPod's proxy refuses Python's default User-Agent
POD_ID = re.compile(r"^[a-z0-9]{8,24}$")
PROXY = re.compile(r"^https?://([a-z0-9]+)-\d+\.proxy\.runpod\.net", re.I)
MAX_SAVED = 20

# runpodctl's own fields; FIELDS_MIN when RunPod rejects one of them
FIELDS = "id name desiredStatus imageName costPerHr gpuCount ports env machine { gpuDisplayName }"
FIELDS_MIN = "id name desiredStatus imageName"


def pod_url(pod_id: str) -> str:
    return f"https://{pod_id}-8081.proxy.runpod.net/"


def parse(text: str) -> dict:
    """What someone pasted: a pod ID, any of the pod's proxy addresses (another port's is fine), RunPod's own page
    for the pod (…?id=…), or the address of a Fizgig somewhere else (another computer on the network) -> {id, url}."""
    t = (text or "").strip()
    if not t:
        raise ValueError("Paste the pod's ID or its address.")
    if POD_ID.match(t.lower()):
        return {"id": t.lower(), "url": pod_url(t.lower())}
    m = PROXY.match(t)
    if m:
        return {"id": m.group(1).lower(), "url": pod_url(m.group(1).lower())}
    if not re.match(r"^https?://", t, re.I) and re.match(r"^[\w.-]+(:\d+)?(/.*)?$", t):
        t = "http://" + t                                   # 192.168.0.20:8081
    u = urllib.parse.urlparse(t)
    if u.scheme in ("http", "https") and u.netloc:
        q = urllib.parse.parse_qs(u.query)
        pid = (q.get("id") or q.get("podId") or [""])[0].strip().lower()
        if "runpod.io" in u.netloc.lower():                 # the console's page for the pod
            if POD_ID.match(pid):
                return {"id": pid, "url": pod_url(pid)}
            raise ValueError("That's a RunPod page, not a pod. Paste the pod's ID (under its name on the Pods page).")
        return {"id": "", "url": f"{u.scheme}://{u.netloc}/"}
    raise ValueError("That isn't a pod ID or an address. The ID is the code under the pod's name on RunPod's Pods page.")


def saved(prefs) -> list:
    out = []
    for p in prefs.get("app_pods") or []:
        if isinstance(p, dict) and p.get("url"):
            out.append({"id": str(p.get("id") or ""), "url": str(p["url"]), "name": str(p.get("name") or ""),
                        "last": float(p.get("last") or 0), "lan_id": str(p.get("lan_id") or ""),
                        # False: a pod pasted in that the account's key didn't list (another account's); else ours
                        "ours": p.get("ours") is not False})
    return out


def add(prefs, text, name="") -> list:
    p = parse(text)
    rows = [r for r in saved(prefs) if r["url"] != p["url"]]
    rows.insert(0, {**p, "name": (name or "").strip()[:60], "last": time.time()})
    return rows[:MAX_SAVED]


def forget(prefs, url) -> list:
    return [r for r in saved(prefs) if r["url"] != url]


def touch(prefs, url, name="", lan_id="") -> list:
    """Opened just now: first in the list next time (and saved, if it came from the account's list or the
    network)."""
    rows = saved(prefs)
    hit = next((r for r in rows if r["url"] == url), None)
    if hit is None:
        p = parse(url)
        hit = {**p, "name": (name or "").strip()[:60]}
    rows = [r for r in rows if r["url"] != url]
    hit["last"] = time.time()
    if name and not hit.get("name"):
        hit["name"] = name.strip()[:60]
    if lan_id:
        hit["lan_id"] = lan_id[:32]
    return ([hit] + rows)[:MAX_SAVED]


def probe(url: str, timeout=6.0) -> str:
    """'ready' (Fizgig's page answers), 'waiting' (the address answers but Fizgig isn't up yet: still pulling the
    image or starting), or 'off' (nothing answers)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return "ready" if b"fizgig-token" in r.read(8192) else "waiting"
    except urllib.error.HTTPError:
        return "waiting"                                   # RunPod's proxy answers 404 / 502 until the port is up
    except Exception:
        return "off"


def _gql(key, query, variables=None, timeout=20):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "User-Agent": UA,
                                                         "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError("RunPod didn't accept that API key. Make one at RunPod > Settings > API Keys "
                             "(read and write).") from None
        raise ValueError(f"RunPod's API answered {e.code}. Try again in a moment.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ValueError(f"Couldn't reach RunPod ({getattr(e, 'reason', e)}).") from None
    if payload.get("errors"):
        raise ValueError(str(payload["errors"][0].get("message") or "RunPod refused that."))
    return payload.get("data") or {}


def is_fizgig(pod) -> bool:
    ports = str(pod.get("ports") or "")
    return "fizgig" in str(pod.get("imageName") or "").lower() or "8081" in ports


def account(key) -> list:
    """The account's Fizgig pods (made from a Fizgig template: its image, or port 8081 open), running ones first."""
    q = "query { myself { pods { %s } } }"
    try:
        data = _gql(key, q % FIELDS)
    except ValueError as e:
        if "Cannot query field" not in str(e):
            raise
        data = _gql(key, q % FIELDS_MIN)
    pods = ((data.get("myself") or {}).get("pods")) or []
    out = []
    for p in pods:
        if not is_fizgig(p):
            continue
        out.append({"id": p.get("id") or "", "name": p.get("name") or "", "url": pod_url(p.get("id") or ""),
                    "running": str(p.get("desiredStatus") or "").upper() == "RUNNING",
                    "status": str(p.get("desiredStatus") or "").upper(),
                    "gpu": ((p.get("machine") or {}).get("gpuDisplayName")) or "",
                    "gpus": int(p.get("gpuCount") or 1), "cost": float(p.get("costPerHr") or 0),
                    "auto_login": bool(_env_password(p))})       # the app window logs you in (never the password itself)
    out.sort(key=lambda r: (not r["running"], r["name"].lower()))
    return out


def login_password(key, url) -> str:
    """The password a pod of this account logs in with, read from its settings on RunPod (FIZGIG_PASSWORD, else
    VNC_PASSWORD - as the pod picks it), or "" when there's none to read: no key, not this account's pod, a RunPod
    secret (the API gives its name, not its value), or one the pod made up itself. Asked fresh each time, never kept."""
    pod_id = (parse(url) or {}).get("id") if url else ""
    if not key or not pod_id:
        return ""
    try:
        pods = ((_gql(key, "query { myself { pods { id env runtime { uptimeInSeconds } } } }", timeout=10)
                 .get("myself") or {}).get("pods")) or []
    except ValueError:
        return ""
    pod = next((p for p in pods if p.get("id") == pod_id), None)
    if pod is None:
        return ""
    # none in its settings (deployed from the template link: the pod made one up; or a RunPod secret): the one its
    # start-up banner printed, from its log
    return _env_password(pod) or _log_password(key, pod_id, (pod.get("runtime") or {}).get("uptimeInSeconds"))


_BANNER_PW = re.compile(r"^\[fizgig\]\s+password:\s*(\S+)\s*$")


def _log_password(key, pod_id, uptime=None, wait=4.0) -> str:
    """The password in the "Ready" banner the pod printed when it started, read from its container log (RunPod's v2
    log stream), or "". Read from the start of this run of the pod, so a long training log can't push the banner out
    of reach; the stream then goes quiet and is closed."""
    if uptime:
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - float(uptime) - 120))
        q = "since=" + urllib.parse.quote(since)
    else:
        q = "tail=5000"
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{urllib.parse.quote(pod_id)}/logs?source=container"
                                 f"&{q}", headers={"Authorization": f"Bearer {key}", "User-Agent": UA,
                                                   "Accept": "text/event-stream"})
    found, end = "", time.time() + wait
    try:
        with urllib.request.urlopen(req, timeout=wait) as r:
            while time.time() < end:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    m = _BANNER_PW.match(str(json.loads(line[5:]).get("line") or "").strip())
                except ValueError:
                    continue
                if m:
                    found = m.group(1)
                    if uptime:                       # this start's own banner: done
                        break
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass
    return found


def _env_password(pod) -> str:
    env = {}
    for e in pod.get("env") or []:
        if isinstance(e, str) and "=" in e:
            k, _, v = e.partition("=")
            env[k.strip()] = v
        elif isinstance(e, dict):
            env[str(e.get("key") or "")] = str(e.get("value") or "")
    for name in ("FIZGIG_PASSWORD", "VNC_PASSWORD"):
        v = str(env.get(name) or "").strip()
        if v:
            return "" if "{{" in v else v          # a secret's reference: the pod uses it, the API won't say it
    return ""


def start(key, pod_id, gpus=1):
    """podResume - RunPod finds the pod's GPU on its own machine again; when that machine's GPUs are all rented, it
    says so and the pod stays stopped."""
    try:
        _gql(key, "mutation($id: String!, $n: Int!) { podResume(input: {podId: $id, gpuCount: $n}) "
                  "{ id desiredStatus } }", {"id": pod_id, "n": int(gpus or 1)})
    except ValueError as e:
        msg = str(e)
        if "not enough free" in msg.lower() or "no longer any instances" in msg.lower():
            raise ValueError("The machine this pod lives on has no free GPU right now. Try again in a while, or "
                             "deploy a new pod (this pod's files stay on its volume).") from None
        raise


def stop(key, pod_id):
    _gql(key, "mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }", {"id": pod_id})


def terminate(key, pod_id):
    """podTerminate: the pod and its disk gone for good (once its run's checkpoints are home, helpers._home_delete)."""
    _gql(key, "mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})


def _lan_rows(rows, found, mine, save):
    """The Fizgigs that answered on the network, into the list: a saved computer whose address changed takes its new
    one (and keeps it), the rest are listed as found, ready to open."""
    moved = False
    port = lambda u: u.rstrip("/").rsplit(":", 1)[-1]
    for f in found:
        mine_ = [r for r in rows.values() if f.get("id") and r.get("lan_id") == f["id"]]
        old = next((r for r in mine_ if port(r["url"]) == port(f["url"])), None)
        if not old and len(mine_) == 1 and sum(g.get("id") == f["id"] for g in found) == 1:
            old = mine_[0]                      # its only Fizgig, on another port now (the old one was taken)
        if old and old["url"] != f["url"]:
            rows.pop(old["url"])
            old["url"] = f["url"]
            moved = True
        base = old or rows.get(f["url"]) or {"id": "", "url": f["url"], "name": "", "last": 0, "saved": False,
                                            "account": False}
        gpu = str(f.get("gpu") or "").replace("NVIDIA GeForce ", "") + (f" ({f['vram_gb']} GB)" if f.get("vram_gb") else "")
        rows[f["url"]] = {**base, "name": base.get("name") or f.get("name") or "", "lan": True,
                          "lan_id": f.get("id") or "", "gpu": gpu, "busy": f.get("busy") or "",
                          "queued": int(f.get("queued") or 0), "queue_held": bool(f.get("queue_held")),
                          "version": f.get("version") or "", "build": _build_word(mine, f.get("commit")),
                          "state": "ready"}
    if moved and save:
        save({"app_pods": [{k: r.get(k) for k in ("id", "url", "name", "last", "lan_id", "ours")}
                           for r in rows.values() if r.get("saved")]})


def _build_word(mine, theirs):
    """'' when both run the same build; else which one to update (git knows which build came first)."""
    theirs = str(theirs or "")
    if not mine or not theirs or mine == theirs:
        return ""
    import subprocess
    import os
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    try:
        known = subprocess.run(["git", "cat-file", "-e", theirs + "^{commit}"], cwd=base, capture_output=True,
                               timeout=5, **kw).returncode == 0
        older = known and subprocess.run(["git", "merge-base", "--is-ancestor", theirs, mine], cwd=base,
                                         capture_output=True, timeout=5, **kw).returncode == 0
    except Exception:
        return "other"
    return "theirs" if older else "mine"


def listing(prefs, key, save=None, mine=""):
    """Everything the Pods tab shows: saved pods and the account's, merged by address, each probed in parallel, and
    the Fizgigs that answer on this network."""
    from fizgig.web import lan
    finding = (concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(lan.discover, 1.0)
               if lan.enabled(prefs) else None)          # Preferences > Network can switch the search off
    rows, err, listed = {}, "", False
    for r in saved(prefs):
        rows[r["url"]] = {**r, "saved": True, "account": False}
    if key:
        try:
            for a in account(key):
                hit = rows.get(a["url"])
                rows[a["url"]] = {**(hit or {"last": 0, "saved": False}), **a,
                                  "name": (hit or {}).get("name") or a["name"], "account": True}
            listed = True
        except ValueError as e:
            err = str(e)
    try:
        me = str(prefs.get("lan_id") or "")     # this computer's other Fizgig (a second launch, another port) isn't a computer to open
        _lan_rows(rows, [f for f in (finding.result(timeout=5) if finding else []) if not me or f.get("id") != me], mine, save)
    except Exception:
        pass
    items = list(rows.values())
    # a saved pod of this account's that the account no longer has was deleted: it can never come back, so it's
    # forgotten (RunPod's proxy still answers its address, which read as starting). One pasted in from another
    # account (never listed by this key) is said to be not on your account, and kept.
    if listed:
        dead = [r for r in items if r.get("saved") and not r.get("account") and PROXY.match(r["url"]) and r.get("ours")]
        if dead:
            items = [r for r in items if r not in dead]
            if save:
                save({"app_pods": [{k: r.get(k) for k in ("id", "url", "name", "last", "lan_id", "ours")}
                                   for r in items if r.get("saved")]})
        for r in items:
            if not r.get("account") and PROXY.match(r["url"]):
                r["gone"] = True
    live = [r for r in items if not r.get("gone") and (not r.get("account") or r.get("running"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        states = dict(zip([r["url"] for r in live],
                          ex.map(lambda r: probe(r["url"], 3.0 if r.get("lan") else 6.0), live)))
    for r in items:
        st = "gone" if r.get("gone") else states.get(r["url"], "stopped" if r.get("account") and not r.get("running")
                                                      else "off")
        # found on the network but its page doesn't answer: its firewall lets the question in, not the web UI
        r["state"] = ("ready" if st == "ready" else "blocked") if r.get("lan") else st
        if r["state"] == "off" and r.get("lan_id"):  # a computer of yours that didn't answer: off, asleep or Fizgig closed
            r["state"] = "away"
    rank = {"ready": 0, "blocked": 1, "waiting": 1, "stopped": 2, "away": 3, "off": 3, "gone": 4}
    items.sort(key=lambda r: (rank.get(r["state"], 3), -(r.get("last") or 0), (r.get("name") or "").lower()))
    return {"pods": items, "error": err, "deploy": DEPLOY}


# How fast each card trains next to an RTX 5090: a rough guide from public diffusion-training benchmarks, not a
# measurement of Fizgig (real runs vary with the model and settings). It is also the list the picker offers: exactly
# the whole cards on RunPod's deploy page (its All tab, 14 Oct 2026). RunPod's API names cards that page doesn't
# (the RTX PRO 5000, RTX 5080...), and a Deploy link to one of those opens with no card chosen.
SPEED = {"RTX 5090": 1.0, "RTX 4090": 0.75, "RTX 3090": 0.4, "RTX A6000": 0.45, "RTX A5000": 0.35,
         "RTX A4500": 0.3, "RTX A4000": 0.25, "RTX 6000 Ada": 0.8, "RTX 4000 Ada": 0.3, "RTX 2000 Ada": 0.15,
         "L4": 0.3, "L40": 0.6, "L40S": 0.8, "A40": 0.45, "A100 PCIe": 0.8, "A100 SXM": 0.85,
         "H100 PCIe": 1.2, "H100 SXM": 1.5, "H100 NVL": 1.35, "H200 SXM": 1.6, "H200 NVL": 1.5, "B200": 2.5,
         "B300": 2.7, "RTX PRO 4000": 0.45, "RTX PRO 4500": 0.65, "RTX PRO 4500 SE": 0.65, "RTX PRO 6000": 1.2,
         "RTX PRO 6000 WK": 1.2}
# without an API key RunPod shares no prices or stock: these, to pick from on its own page
SHORTLIST = [("RTX 5090", 32), ("RTX 4090", 24), ("L40S", 48), ("RTX PRO 6000", 96), ("A100 SXM", 80),
             ("H100 SXM", 80)]
_offers_cache = {}


def _trains(g) -> bool:
    """A card Fizgig's pod image trains on: NVIDIA (the image is CUDA), bf16-capable (not a V100), a whole GPU."""
    gid, name = str(g.get("id") or ""), str(g.get("displayName") or "")
    return gid.startswith("NVIDIA") and "V100" not in gid and "MIG" not in name


def offers(key, dc="", count=1) -> dict:
    """Every card Fizgig can train on, with its live price and stock in Secure and Community Cloud (in one data
    centre when `dc` is set: where a network volume lives), for pods of `count` cards, on hosts whose driver runs the
    pod image (MIN_CUDA: RunPod's plain stock counts every host, so a card could show free and then not deploy), and
    the account's network volumes. Asked again at most once a minute."""
    count = 2 if str(count) == "2" else 1
    if not key:
        return {"key": False, "volumes": [], "count": count, "gpus": [
            {"gpu": n, "vram": v, "speed": SPEED.get(n), "url": deploy_url(n, count)} for n, v in SHORTLIST]}
    hit = _offers_cache.get((key, dc, count))
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    where = f', dataCenterId: "{dc}"' if re.match(r"^[A-Z0-9-]{3,20}$", dc or "") else ""
    d = _gql(key, "{ myself { networkVolumes { id name size dataCenterId } } gpuTypes { id displayName memoryInGb "
                  f's: lowestPrice(input: {{gpuCount: {count}, secureCloud: true, minCudaVersion: "{MIN_CUDA}"{where}}}) '
                  "{ uninterruptablePrice stockStatus } "
                  f'c: lowestPrice(input: {{gpuCount: {count}, secureCloud: false, minCudaVersion: "{MIN_CUDA}"{where}}}) '
                  "{ uninterruptablePrice stockStatus } } }")
    gpus = []
    for g in d.get("gpuTypes") or []:
        if not _trains(g):
            continue
        name = g["displayName"]
        if name not in SPEED:                    # not on RunPod's deploy page
            continue
        row = {"gpu": name, "vram": int(g.get("memoryInGb") or 0), "speed": SPEED.get(name),
               "url": deploy_url(name, count)}
        for cloud, k in (("secure", "s"), ("community", "c")):
            lp = g.get(k) or {}
            if lp.get("uninterruptablePrice"):
                row[cloud] = {"price": float(lp["uninterruptablePrice"]), "stock": lp.get("stockStatus") or ""}
        gpus.append(row)
    vols = [{"id": v["id"], "name": v.get("name") or v["id"], "size": v.get("size"), "dc": v.get("dataCenterId") or ""}
            for v in (d.get("myself") or {}).get("networkVolumes") or []]
    out = {"key": True, "gpus": gpus, "volumes": vols, "dc": dc if where else "", "count": count}
    _offers_cache[(key, dc, count)] = (time.time(), out)
    return out
