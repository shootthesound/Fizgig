"""Helpers: another Fizgig that takes work off this one's GPU. Both halves live here.

As a helper, any Fizgig answers /api/helper/*: what it is (GPU, build, busy), a file put a piece at a time (the
epoch's LoRA), a job (a round of previews) rendered by its preview worker (fizgig.families.preview_worker, the run's
model kept loaded), the job's state and pictures. A job needing a model this computer doesn't have starts its own
downloads (the regular downloader; the SFT look LoRA's first-use fetch) and waits for them.

As the lead, a Fizgig keeps its helpers (prefs "helpers": url, name, login, which jobs each takes), checks them
every few seconds, and while a run trains with a helper taking its previews: tells the trainer so
(<output_dir>/.helper_previews.json, refreshed), takes each round the trainer hands over (<output_dir>/.helper_rounds/
<epoch>/: the epoch's LoRA and its preview setup), sends it, and brings the pictures back into the run's samples
folder. If the helper goes away the note goes stale and the trainer renders its previews itself again.

The lead logs in to a helper with its web password once and keeps the login cookie (prefs), never the password.
"""
import glob
import hashlib
import http.client
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse

from fizgig.dataset.masks import folder_masks, is_mask_file

logger = logging.getLogger("fizgig.web.helpers")

PIECE = 8 << 20                                     # bytes per put (a 300 MB LoRA = 38 pieces)
# fizgig.families.train's HELPER_PREVIEWS / HELPER_ROUNDS, named here so the server never imports torch
HELPER_PREVIEWS, HELPER_ROUNDS = ".helper_previews.json", ".helper_rounds"
_NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}


def _srv():
    from fizgig.web import server
    return server


def _dir(*p):
    from fizgig import user_data
    return os.path.join(user_data.USER_DATA, "helper", *p)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    for n in range(20):          # Windows refuses the replace while another process has the file open for a moment
        try:
            os.replace(path + ".tmp", path)
            return
        except PermissionError:
            if n == 19:
                raise
            time.sleep(0.05)


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# =====================================================================================================================
# the helper's side
# =====================================================================================================================
_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_SAFE_FILE = re.compile(r"^(?!\.+$)[^/\\:*?\"<>|]{1,200}$")      # a file name a dataset may hold (spaces, brackets), never a path


def _inside(root, rel):
    """root/rel for a relative name another computer sent ("a/b.png"), or None when it isn't one: an absolute path,
    a drive, "..", or anything that would land outside root."""
    parts = str(rel or "").replace("\\", "/").split("/")
    if not rel or any(p in ("", ".", "..") or ":" in p for p in parts):
        return None
    root = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root, *parts))
    return p if p.startswith(root + os.sep) else None
_w = {"proc": None, "key": "", "dir": "", "log": None}
_wlock = threading.Lock()


def info():
    """What this Fizgig tells a lead: name, GPU, build, and whether it's busy (training here, or helping)."""
    s = _srv()
    about = s.about_info()
    with s._lock:
        vt = s._stats["vram"][1]
        gpu = s._stats["device_name"]
        busy = s._run["lora"] if s._run["proc"] is not None and s._run["proc"].poll() is None else ""
    st = _worker_state()
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    vast = os.environ.get("CONTAINER_ID") if os.environ.get("VAST_CONTAINERLABEL") else ""
    name = os.environ.get("FIZGIG_HELPER_NAME") or (f"pod {pod_id}" if pod_id else f"Vast.ai instance {vast}" if vast
                                                    else socket.gethostname())
    return {"name": name, "gpu": str(gpu or "").replace("NVIDIA GeForce ", "") if vt else "",
            "vram": round(vt / 2 ** 30) if vt else 0, "version": about["version"], "commit": about["commit"],
            "training": busy, "worker": st, "pod": bool(s.on_pod()), "helping": _out.get("name", ""),
            "sharing": [w.get("lead", "") for w in list(_sw.values()) if w["proc"].poll() is None]}


def put(name, offset, data):
    """A piece of a file a lead sends (the epoch's LoRA, a context LoRA), at its offset; offset 0 starts it again."""
    if not _SAFE.match(name or "") or name.startswith("."):
        raise ValueError("Bad file name.")
    path = _dir("files", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r+b" if offset and os.path.exists(path) else "wb") as f:
        f.seek(int(offset))
        f.write(data)
        f.truncate()
    return {"size": os.path.getsize(path)}


def have(name):
    p = _dir("files", name) if _SAFE.match(name or "") else ""
    return {"size": os.path.getsize(p) if p and os.path.isfile(p) else -1}


def _needs(desc, job, prefs):
    """The model files a job needs here: [(label, path or "", ModelFile)] in the order they're loaded."""
    s = _srv()
    from fizgig.families import launch
    out = []
    for role in ("dit", "vae") if job.get("conds") else ("dit", "vae", "text_encoder"):
        f = next((m for m in desc.model_files if m.role == role), None)
        if f is not None:
            out.append((f.label, s.pref_path(prefs, f.pref_key), f))
    sp = desc.preview_speed()
    if sp is not None and float(job.get("turbo") or 0) > 0 and sp.pref_key:
        f = next((m for m in desc.model_files if m.pref_key == sp.pref_key), None)
        out.append((sp.name, s.pref_path(prefs, sp.pref_key), f))
    lk = desc.preview_look()
    if lk is not None and float(job.get("look") or 0) > 0:
        out.append((lk.label, launch.look_path(desc, {"models": {lk.pref_key: s.pref_path(prefs, lk.pref_key)}}), lk))
    return out


def models(family, job=None, fetch=False):
    """Which of a job's model files this computer has, and (fetch) its own downloads started for the rest:
    {have: [labels], missing: [labels], fetching: {what, got, total} or None, error}."""
    s = _srv()
    desc = s.get_family(family)
    if desc is None:
        raise ValueError(f"This Fizgig doesn't know {family!r}: update it.")
    prefs = s.load_prefs()
    needs = _needs(desc, job or {"turbo": 1, "look": 1}, prefs)
    ok = lambda p: bool(p) and os.path.isfile(p)
    missing = [(lab, f) for lab, p, f in needs if not ok(p)]
    res = {"family": desc.display_name, "have": [lab for lab, p, _f in needs if ok(p)],
           "missing": [lab for lab, _f in missing], "fetching": None, "error": ""}
    if not missing:
        return res
    fs = s._fetch_status()
    if fs.get("state") == "running":
        res["fetching"] = {"what": os.path.basename(fs.get("file") or "") or "models",
                           "got": (fs.get("done") or 0) / 2 ** 30, "total": (fs.get("total") or 0) / 2 ** 30}
    for lab, f in missing:
        if f is not None and getattr(f, "fetch_on_use", False):
            st = s._fetch_on_use(f) if fetch else ""
            if st == "running":
                res["fetching"] = {"what": lab, "got": s._on_use_gb(f), "total": f.size_gb}
            elif st == "failed":
                res["error"] = f"The {lab} download failed here."
    regular = [f for _lab, f in missing if f is not None and not getattr(f, "fetch_on_use", False)]
    if fetch and regular and fs.get("state") != "running":
        try:     # the regular downloader, as Preferences' "Download models for me" (what's there already is kept)
            s._start_fetch(desc.key, "", optional=any(f.role == "speed_lora" for f in regular))
            res["fetching"] = res["fetching"] or {"what": regular[0].label, "got": 0.0, "total": regular[0].size_gb}
        except ValueError as e:
            res["error"] = (f"{e} Add a Hugging Face token on the helper (HF_TOKEN) or download them there once."
                            if "token" in str(e).lower() else str(e))
    return res


def _worker_state():
    with _wlock:
        p, d = _w["proc"], _w["dir"]
    st = _read(os.path.join(d, "worker.json"), {}) if d else {}
    alive = p is not None and p.poll() is None
    return {"state": st.get("state", "") if alive or st.get("state") == "failed" else "", "text": st.get("text", ""),
            "epoch": st.get("epoch")}


def _ensure_worker(setup):
    """The preview worker for this setup: the running one when it matches, else a new one (the old one stops)."""
    key = hashlib.sha1(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16]
    with _wlock:
        p = _w["proc"]
        if p is not None and p.poll() is None and _w["key"] == key:
            return _w["dir"]
        if p is not None and p.poll() is None:
            open(os.path.join(_w["dir"], "stop"), "w").close()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        if _w.get("log") is not None:       # the last worker's log, closed (Windows won't clear an open file)
            _w["log"].close()
            _w["log"] = None
        d = _dir("work", key)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        _write(os.path.join(d, "setup.json"), setup)
        s = _srv()
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(s.BASE_DIR, "src"))
        log = open(os.path.join(d, "worker.log"), "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "fizgig.families.preview_worker", "--setup",
                                 os.path.join(d, "setup.json"), "--jobs", d, "--idle", "900"], stdout=log,
                                stderr=subprocess.STDOUT, cwd=s.BASE_DIR, env=env, **_NO_WINDOW)
        _w.update(proc=proc, key=key, dir=d, log=log)
        _write(os.path.join(d, "worker.json"), {"state": "loading", "text": "Starting", "at": time.time()})
        return d


def submit(body):
    """A round of previews: {family, job (the trainer's round), lora (a put file), context (a put file or "")}.
    Returns {job: id} once it's queued, or {models: ...} while this computer fetches what it needs first."""
    s = _srv()
    if s._gpu_training():
        raise ValueError("This computer is training: it can't render another computer's previews now.")
    job = dict(body.get("job") or {})
    desc = s.get_family(str(job.get("family") or ""))
    if desc is None:
        raise ValueError("This Fizgig doesn't know that model: update it.")
    m = models(desc.key, job, fetch=True)
    if m["missing"]:
        return {"models": m}
    for k in ("lora", "context", "conds"):
        if body.get(k) and have(body[k])["size"] < 0:
            raise ValueError(f"The {k} file didn't arrive: send it again.")
    prefs = s.load_prefs()
    paths = {lab: p for lab, p, _f in _needs(desc, job, prefs)}
    roles = {f.role if f is not None else "": p for _lab, p, f in _needs(desc, job, prefs)}
    sp, lk = desc.preview_speed(), desc.preview_look()
    setup = {"family": desc.key, "options": job.get("options") or {}, "precision": job.get("precision") or "auto",
             "network": job.get("network") or {},
             "paths": {"dit": roles.get("dit"), "vae": roles.get("vae"), "text_encoder": roles.get("text_encoder"),
                       "speed_lora": paths.get(sp.name) if sp is not None else None,
                       "look_lora": paths.get(lk.label) if lk is not None else None,
                       "context": _dir("files", body["context"]) if body.get("context") else None},
             "context_strength": (job.get("context") or [None, 1.0])[1]}
    if not setup["paths"]["speed_lora"] and sp is not None:      # this round has no Turbo: keep the worker's
        setup["paths"]["speed_lora"] = s.pref_path(prefs, sp.pref_key) if sp.pref_key and os.path.isfile(
            s.pref_path(prefs, sp.pref_key) or "") else None
    if not setup["paths"]["look_lora"] and lk is not None:
        from fizgig.families import launch
        p = launch.look_path(desc, {"models": {lk.pref_key: s.pref_path(prefs, lk.pref_key)}})
        setup["paths"]["look_lora"] = p if p and os.path.isfile(p) else None
    _tidy()
    d = _ensure_worker(setup)
    jid = secrets.token_hex(6)
    out = _dir("out", jid)
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(d, jid + ".job.json"), {**job, "lora": _dir("files", body["lora"]), "out": out,
                                                 "conds_file": _dir("files", body["conds"]) if body.get("conds") else ""})
    return {"job": jid}


def _tidy(hours=6):
    """Files and pictures of jobs whose lead went away mid-round, once they're old."""
    old = time.time() - hours * 3600
    for p in glob.glob(_dir("files", "*")) + glob.glob(_dir("out", "*")):
        try:
            if os.path.getmtime(p) < old:
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def status(jid):
    if not _SAFE.match(jid or ""):
        raise ValueError("Bad job.")
    with _wlock:
        d = _w["dir"]
    done = _read(os.path.join(d, jid + ".done.json")) if d else None
    if done is not None:
        return {"state": "done" if done.get("ok") else "failed", **done}
    w = _worker_state()
    with _wlock:
        dead = _w["proc"] is None or _w["proc"].poll() is not None
    if d and dead and any(os.path.exists(os.path.join(d, jid + x)) for x in (".working", ".job.json")):
        return {"state": "failed", "error": w["text"] if w["state"] == "failed" else
                "the preview worker stopped" + (f": {_tail(os.path.join(d, 'worker.log'), 300)}" if d else "")}
    if d and os.path.exists(os.path.join(d, jid + ".working")):
        return {"state": "rendering", "worker": w}
    if d and os.path.exists(os.path.join(d, jid + ".job.json")):
        if w["state"] == "failed":
            return {"state": "failed", "error": w["text"]}
        return {"state": "loading" if w["state"] == "loading" else "queued", "worker": w}
    return {"state": "gone"}


def file_path(jid, name):
    if not (_SAFE.match(jid or "") and _SAFE.match(name or "")):
        raise ValueError("Bad name.")
    return _dir("out", jid, name)


def drop(jid, lora=""):
    """A finished job's pictures and its LoRA, once the lead has them."""
    if _SAFE.match(jid or ""):
        shutil.rmtree(_dir("out", jid), ignore_errors=True)
        with _wlock:
            d = _w["dir"]
        if d:
            for p in glob.glob(os.path.join(d, jid + ".*")):
                try:
                    os.remove(p)
                except OSError:
                    pass
    if lora and _SAFE.match(lora):
        try:
            os.remove(_dir("files", lora))
        except OSError:
            pass
    return {"ok": True}


# =====================================================================================================================
# the lead's side
# =====================================================================================================================
_live = {}                 # helper id -> {state, ping_ms, info, activity, at}
_lead = {"thread": None, "checker": None, "flag_dir": "", "mine": ""}
_out = {}                  # this Fizgig helping another (the pod case): name, url, text, gen
_gen = [0]                 # the helping loop's generation: a loop runs while _out["gen"] is its own


class Client:
    """A helper's API, as its own web page talks to it: the page's token (fetched again when the helper restarts),
    the login cookie."""

    def __init__(self, url, cookie=""):
        u = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        self.https, self.host = u.scheme == "https", u.netloc
        self.cookie, self.token = cookie, ""

    def _conn(self, timeout):
        c = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return c(self.host, timeout=timeout)

    def _raw(self, method, path, body=None, headers=None, timeout=20):
        c = self._conn(timeout)
        try:
            h = {"Cookie": f"fizgig_session={self.cookie}" if self.cookie else "", "X-Fizgig-Token": self.token,
                 "User-Agent": "Fizgig-helper/1 (curl-compatible)", **(headers or {})}   # RunPod's proxy turns
            # away Python's own
            c.request(method, path, body=body, headers=h)
            r = c.getresponse()
            return r.status, r.getheader("Set-Cookie") or "", r.read()
        finally:
            c.close()

    def page_token(self):
        code, _c, page = self._raw("GET", "/", timeout=10)
        m = re.search(rb'name="fizgig-token"\s+content="([^"]+)"', page)
        if not m:
            raise ValueError("That address doesn't answer as Fizgig.")
        self.token = m.group(1).decode()

    def call(self, path, body=None, raw=None, timeout=20):
        """JSON in/out (raw: bytes out, the body sent as-is). A restarted helper's new token is fetched once."""
        for attempt in (0, 1):
            if not self.token:
                self.page_token()
            data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            hdr = {"Content-Type": "application/octet-stream" if raw is not None else "application/json"}
            code, _c, out = self._raw("POST" if data is not None else "GET", path, data, hdr, timeout)
            if code == 403 and attempt == 0:
                self.token = ""
                continue
            if code == 401:
                raise PermissionError("login")
            if path.startswith(("/api/helper/file", "/api/helper/speed?", "/api/helpers/lora", "/api/helpers/share_cache",
                                "/api/helpers/share_startfile", "/api/helper/move_file")):
                if code != 200:
                    raise ValueError(f"helper answered {code}")
                return out
            d = json.loads(out or b"{}")
            if code >= 400:
                raise ValueError(d.get("error") or f"helper answered {code}")
            return d
        raise ValueError("The helper keeps refusing this computer's requests.")

    def download(self, path, dest, progress):
        """A GET streamed to a file (a checkpoint can be hundreds of MB), via a .part file; progress(bytes) as it goes."""
        for attempt in (0, 1):
            if not self.token:
                self.page_token()
            c = self._conn(120)
            try:
                c.request("GET", path, headers={"Cookie": f"fizgig_session={self.cookie}" if self.cookie else "",
                                                "X-Fizgig-Token": self.token, "User-Agent": "Fizgig-helper/1"})
                r = c.getresponse()
                if r.status == 403 and attempt == 0:
                    r.read()
                    self.token = ""
                    continue
                if r.status == 401:
                    raise PermissionError("login")
                if r.status != 200:
                    raise ValueError(f"it answered {r.status}")
                with open(dest + ".part", "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        progress(len(chunk))
                os.replace(dest + ".part", dest)
                return
            finally:
                c.close()

    def login(self, password):
        self.page_token()
        code, cookie, out = self._raw("POST", "/api/login", json.dumps({"password": password, "remember": True}).encode(),
                                      {"Content-Type": "application/json"})
        if code != 200:
            raise ValueError((json.loads(out or b"{}").get("error")) or "Wrong password.")
        m = re.search(r"fizgig_session=([^;]+)", cookie)
        self.cookie = m.group(1) if m else ""


def _helpers():
    return list(_srv().load_prefs().get("helpers") or [])


def _save(rows):
    _srv().update_prefs({"helpers": rows})


def _norm(url):
    url = str(url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Give its address.")
    if "://" not in url:
        url = ("https://" if "proxy.runpod.net" in url else "http://") + url
    u = urllib.parse.urlparse(url)
    if not u.port and u.scheme == "http":
        url += ":8081"
    return url


def add(url, password):
    url = _norm(url)
    c = Client(url)
    prefs = _srv().load_prefs()
    key = ""
    if not password and "proxy.runpod.net" in url:
        # a pod of this account: its password from its RunPod settings, else from its log
        from fizgig.web import pod_stop, pods
        key = pod_stop.api_key(prefs)
        password = pods.login_password(key, url)
    if not password:                     # else the login this computer already has there (helping it, its checkpoints)
        c.cookie = (prefs.get("helper_out_cookies") or {}).get(url, "")
        if not c.cookie and "proxy.runpod.net" in url:
            raise ValueError("Fizgig couldn't find this pod's password. Copy it from the pod's log on RunPod (the "
                             "line under \"Ready\") and paste it into the box beside the pod." if key else
                             "Type the pod's password into the box beside it. Or save your RunPod API key on the "
                             "Pods tab, and Fizgig finds it for you.")
    try:
        if password or not c.cookie:
            c.login(password or "")
        i = c.call("/api/helper/info")
    except (OSError, http.client.HTTPException) as e:
        logger.info(f"[helpers] adding {url}: {e}")
        raise ValueError(f"Nothing answers at {url}: check Fizgig is running there and its firewall lets this "
                         "computer in (Fizgig's port is 8081 unless it was changed).")
    except PermissionError:
        raise ValueError("Wrong password.")
    name = i.get("name") or url
    old = [h for h in _helpers() if h.get("url") == url or h.get("name") == name]    # added before, or it moved
    rows = [h for h in _helpers() if h not in old]
    hid = hashlib.sha1(url.encode()).hexdigest()[:10]
    for h in old:                        # the same Fizgig at a new address (a Vast.ai instance started again)
        if _norm(h.get("url") or url) != url:
            _readdress(_norm(h["url"]), url, c.cookie)
    rows.append({"id": hid, "url": url, "name": name, "cookie": c.cookie, "where": _where(url),
                 "previews": any(h.get("previews") for h in old), "share": any(h.get("share") for h in old)})
    for h in old:
        _live.pop(h["id"], None)
    _save(rows)
    _live.pop(hid, None)
    ensure_lead()
    threading.Thread(target=measure_link, args=(url, c.cookie), daemon=True).start()
    return {"say": f"Added {i.get('name') or url}" + (f" ({i['gpu']})" if i.get("gpu") else "") + "."}


def _readdress(old, new, cookie):
    """A Fizgig this computer knows came back at a new address (a Vast.ai instance gets a new IP or port when it starts
    again): its Pods tab entry, its login, its bring-home row and a run of this computer's moved there follow it."""
    s = _srv()
    prefs = s.load_prefs()
    ch = {}
    rows = list(prefs.get("app_pods") or [])
    if any(_norm(r.get("url") or "") == old for r in rows):
        ch["app_pods"] = [{**r, "url": new + "/"} if _norm(r.get("url") or "") == old else r for r in rows]
    cookies = dict(prefs.get("helper_out_cookies") or {})
    if old in cookies:
        cookies[new] = cookie or cookies[old]
        cookies.pop(old)
        ch["helper_out_cookies"] = cookies
    home = dict(prefs.get("helper_bring_home") or {})
    if old in home:
        home[new] = home.pop(old)
        _home[new] = _home.pop(old, {})
        ch["helper_bring_home"] = home
    if ch:
        s.update_prefs(ch)
    if _norm(_mv.get("url") or "x") == old:
        _mv["url"] = new
        if cookie:
            _mv["cookie"] = cookie
        mf = os.path.join(_mv.get("out_dir") or "", MOVED_FILE)
        m = _read(mf)
        if isinstance(m, dict):
            _write(mf, {**m, "url": new})
    logger.info(f"[helpers] {old} is at {new} now: followed")


def remove(hid):
    _save([h for h in _helpers() if h.get("id") != hid])
    _live.pop(hid, None)
    return {"say": "Removed."}


def fetch_models(hid):
    """Get them now (a helper's tile): it downloads the model files the last job found missing."""
    h = next((x for x in _helpers() if x.get("id") == hid), None)
    ask = (_live.get(hid) or {}).get("models_ask") if h else None
    if not ask:
        raise ValueError("Nothing has asked it for models yet.")
    m = Client(h["url"], h.get("cookie", "")).call("/api/helper/models", {**ask, "fetch": True})
    if m.get("error"):
        raise ValueError(m["error"])
    return {"say": f"{h.get('name')} is downloading {', '.join(m.get('missing') or []) or 'nothing: it has them'}."}


def set_jobs(hid, previews=None, share=None, every=None, send_cache=None, wait=None):
    if every is not None:
        _srv().update_prefs({"helper_share_every": max(1, min(10, int(every)))})
    if send_cache is not None:
        _srv().update_prefs({"helper_share_send_cache": bool(send_cache)})
    if wait is not None:
        _srv().update_prefs({"helper_share_wait": bool(wait)})
    rows = _helpers()
    for h in rows:
        if h.get("id") == hid:
            if previews is not None:
                h["previews"] = bool(previews)
                if previews:                       # one helper renders a run's previews: the others step down
                    for o in rows:
                        if o is not h:
                            o["previews"] = False
            if share is not None:
                h["share"] = bool(share)
                _share["fail"] = None
    _save(rows)
    return {"ok": True}


_relogged = {}            # helper id -> when its login was last tried again


def _relogin(h, c):
    """A pod helper whose login ran out (it restarted on a new session key): logged in again with its password,
    found as Add finds it, and its info - or None (not a pod, no password to find, tried in the last 5 minutes)."""
    if time.time() - _relogged.get(h["id"], 0) < 300:
        return None
    _relogged[h["id"]] = time.time()
    pw = _pod_password(h["url"])
    if not pw:
        return None
    try:
        c.login(pw)
        i = c.call("/api/helper/info", timeout=6)
    except Exception:
        return None
    rows = _helpers()
    for r in rows:
        if r.get("id") == h["id"]:
            r["cookie"] = c.cookie
    _save(rows)
    h["cookie"] = c.cookie
    return i


def _check(h, mine):
    """One helper's state now: ready / away / login / elsewhere, its GPU and build, and how far away it is."""
    c = Client(h["url"], h.get("cookie", ""))
    t = time.time()
    try:
        i = c.call("/api/helper/info", timeout=6)
    except PermissionError:
        i = _relogin(h, c)
        if i is None:
            return {"state": "login"}
    except Exception:
        return {"state": "away"}
    from fizgig.web import pods
    build = pods._build_word(mine, i.get("commit")) or "same"
    other = next((n for n in i.get("sharing") or [] if n != socket.gethostname()), "")     # another lead's share
    state = "elsewhere" if i.get("helping") or other else "busy" if i.get("training") else "ready"
    return {"state": state, "ping_ms": int((time.time() - t) * 1000), "gpu": i.get("gpu"), "vram": i.get("vram"),
            "build": build, "worker": i.get("worker") or {}, "elsewhere": i.get("helping") or other,
            "training": i.get("training"), "pod": bool(i.get("pod"))}


def _match_pod(h, lv, mine):
    """A pod of the list on another build, free: asked to run this computer's (pod_version.py). Asked again a minute
    and a half later while it differs (busy then, or switching: it is back on the new build in under a minute)."""
    if time.time() - lv.get("match_at", 0) < 90:
        return
    lv["match_at"] = time.time()
    try:
        d = Client(h["url"], h.get("cookie", "")).call("/api/pod/match", {"commit": mine, "by": socket.gethostname(),
                                                                          "auto": True}, timeout=15)
    except Exception as e:
        lv["match"] = ""                       # a pod from before this: it says nothing, the build note stays
        logger.info(f"[helpers] matching {h.get('name')}: {e}")
        return
    lv["match"] = ("Switching to this computer's Fizgig" if d.get("switching") else d.get("busy") or d.get("error")
                   or "")


def listing(look=False):
    """The Helpers tab: each helper with its state and jobs, computers it could add, pods this one could help. `look`:
    the tab is open, so the network is searched for Fizgigs to add."""
    rows = []
    for h in _helpers():
        lv = _live.get(h["id"]) or {"state": "away"}
        st = lv.get("state")
        rows.append({"id": h["id"], "name": h.get("name"), "url": h.get("url"), "where": h.get("where"),
                     "previews": h.get("previews"), "share": h.get("share"),
                     "state": st, "training": lv.get("training"), "gpu": lv.get("gpu"), "vram": lv.get("vram"),
                     "ping_ms": lv.get("ping_ms"), "build": lv.get("build"), "elsewhere": lv.get("elsewhere"),
                     "match": lv.get("match", ""),
                     "activity": lv.get("activity"), "previews_stat": lv.get("previews_stat", ""),
                     "check_stat": lv.get("check_stat", ""),
                     "share_stat": lv.get("share_stat", "") or ("Its share follows the two GPUs' speeds, round by "
                                                                "round" if h.get("share") else ""),
                     "share_plan": lv.get("share_plan", "") if h.get("share") else "",
                     "share_how": lv.get("share_how", "") if h.get("share") else "", "models": lv.get("models"),
                     "link": _link_line(_link(h["url"])), "link_mbps": _link_mbps(_link(h["url"]))})
    # before a share's first round: its estimate, from the link and this run's speed so far
    out_dir, run = _run_now()
    setup = _setup(out_dir, run) if out_dir else None
    s = _srv()
    with s._lock:
        prog, my_gpu = dict(s._run.get("progress") or {}), s._stats.get("device_name")
    if setup and not setup.get("off"):
        for r in rows:
            h = next((x for x in _helpers() if x.get("id") == r["id"]), None)
            if h and not r.get("share_plan") and (r.get("share") or r.get("state") == "ready"):
                lk = _link(h["url"])
                r["share_plan"] = _share_estimate(setup, lk, my_gpu, r.get("gpu"), prog, _share_every(),
                                                  lk.get("overhead"))
                if r["share_plan"] and not (_share["run"] and _share["run"].get("hid") == r["id"]):
                    r["share_plan"] += _setup_guess(out_dir, run, setup, lk)
    now = time.time()
    for iid, i in list(_inbound.items()):
        if now - i["at"] < 20:
            rows.append({"id": iid, "name": i.get("name"), "where": "", "inbound": True, "previews": True,
                         "share": False, "state": "ready", "build": "same", "gpu": i.get("gpu"), "vram": i.get("vram"),
                         "activity": i.get("activity"), "previews_stat": i.get("stat", ""),
                         "check_stat": i.get("check_stat", "")})
    for iid, i in list(_share_in.items()):
        if now - i["at"] < 20:
            row = next((r for r in rows if r.get("inbound") and r["name"] == i["name"]), None)
            if row is None:
                row = {"id": iid, "name": i.get("name"), "where": "", "inbound": True, "previews": False,
                       "state": "ready", "build": "same", "gpu": i.get("gpu"), "activity": i.get("activity")}
                rows.append(row)
            lk = i.get("link") or {}
            plan = i.get("plan", "") or (_share_estimate(setup, lk, my_gpu, i.get("gpu"), prog, i.get("every"),
                                                         i.get("overhead"), lead_up=False) if setup else "")
            row.update(link=_link_line(lk) and _link_line(lk).replace("Link:", "Its link:"))
            row.update(share=True, share_how=i.get("how", ""), share_plan=plan, share_stat=i.get("stat", "") or ("Ready to train" if i.get("ready") else
                                                                    "Getting ready for the share"))
            if i.get("activity"):
                row["activity"] = i["activity"]
    found = _found(look)                 # first: it refreshes what's on the network, which the help list reads
    out = _pods_to_help()
    for o in out:
        o.update(link=_link_line(_link(o.get("url"))), link_mbps=_link_mbps(_link(o.get("url"))),
                 home=home_status(o.get("url")), pod="proxy.runpod.net" in str(o.get("url")))
    return {"helpers": rows, "found": found, "out": out, "share_every": _share_every(), "home_done": _home_done[-3:],
            "share_off": _off_line(setup) if setup and setup.get("off") else "", "running": bool(out_dir),
            "previews_off": _previews_off(out_dir, run),
            "share_send_cache": _send_cache(), "share_wait": _share_wait(), "local": local_status(),
            "helping": {k: _out.get(k) for k in ("name", "url", "text", "mode")} if _out.get("url") else None}


def _pods_to_help():
    """The Fizgigs this computer could help right now: RunPod pods running on your account (a deleted or stopped one
    left out; without a key, the Pods tab's saved ones), computers on this network that are on now, and the Pods tab's
    saved addresses out on the internet (a Vast.ai instance: there's no telling without trying). The one it's helping
    always stays."""
    try:
        from fizgig.web import pod_stop, pods
        s = _srv()
        prefs = s.load_prefs()
        cookies = prefs.get("helper_out_cookies") or {}
        saved = [p for p in (prefs.get("app_pods") or []) if p.get("url")]
        key = pod_stop.api_key(prefs)
        acct = _account_rows(key) if key else None          # None: not known (no key, refused, not answered yet)
        running = {_norm(a["url"]): a for a in acct or []}
        on_lan = {_norm(f["url"]) for f in _lan_seen["rows"] if f.get("url")}
        helping = _norm(_out["url"]) if _out.get("url") else ""
        out, seen = [], set()

        def add(name, url):
            u = _norm(url)
            if u not in seen:
                seen.add(u)
                out.append({"name": name or url, "url": u, "training": "", "logged_in": bool(cookies.get(u))})
        for p in saved:
            u, where = _norm(p["url"]), _where(p["url"])
            if u == helping or (where == "pod" and (acct is None or u in running)) or where == "address" or \
                    (where in ("lan", "local", "tailscale") and u in on_lan):
                add(p.get("name") or p.get("id") or p["url"], p["url"])
        for u, a in running.items():                          # a running pod of yours not saved on the Pods tab
            add(a.get("name") or a.get("id"), a["url"])
        return out
    except Exception as e:
        logger.info(f"[helpers] listing computers to help: {_plain(e)}")
        return []


_acct = {"at": 0.0, "rows": None, "key": "", "busy": False}
_lan_seen = {"at": 0.0, "rows": []}
_found_lock = threading.Lock()


def _where(url):
    """What kind of address a helper is at: a RunPod pod, this computer, Tailscale, this network, or anywhere else
    (a Vast.ai instance: a public address or a name on the internet)."""
    import ipaddress
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if "proxy.runpod.net" in url:
        return "pod"
    if host in ("127.0.0.1", "localhost", "::1"):
        return "local"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:                   # a name: a computer's on this network, or one out on the internet
        return "lan" if host.endswith((".local", ".lan", ".home")) or "." not in host else "address"
    if ip in ipaddress.ip_network("100.64.0.0/10"):
        return "tailscale"
    return "lan" if ip.is_private or ip.is_link_local else "address"


def _account_rows(key):
    """The RunPod account's running pods, from a minute-old copy; a newer one is asked for in the background (the
    Helpers tab polls every few seconds: it never waits on RunPod). None until the first answer, or for a key RunPod
    refuses."""
    with _found_lock:
        fresh = _acct["key"] == key and time.time() - _acct["at"] < 60
        if fresh or _acct["busy"]:
            return _acct["rows"] if _acct["key"] == key else None
        _acct["busy"] = True

    def ask():
        from fizgig.web import pods
        try:
            rows = [p for p in pods.account(key) if p.get("running")]
        except ValueError:
            rows = None                  # a refused key: the saved pods are listed instead
        with _found_lock:
            _acct.update(rows=rows, at=time.time(), key=key, busy=False)
    threading.Thread(target=ask, daemon=True, name="helpers-account").start()
    return _acct["rows"] if _acct["key"] == key else None


def _found(look=False):
    """The Fizgigs that could be added: on this network (as the Pods tab finds them, asked at most every 10 s and only
    while the Helpers tab is open - `look` - so a page in the background sends nothing), the pods running on your
    RunPod account (else the Pods tab's saved ones), and the Pods tab's saved addresses out on the internet (a Vast.ai
    instance), leaving out the helpers already added."""
    mine = {_norm(h["url"]) for h in _helpers()}
    out = []
    seen = lambda u: _norm(u) in mine or _norm(u) in {_norm(x["url"]) for x in out}
    try:
        from fizgig.web import lan
        if not lan.enabled(_srv().load_prefs()):
            _lan_seen.update(rows=[], at=0.0)
        elif look and time.time() - _lan_seen["at"] > 10:
            _lan_seen.update(rows=lan.discover(0.6), at=time.time())
        out += [{"name": f.get("name"), "url": f.get("url"), "where": "lan",
                 "gpu": str(f.get("gpu") or "").replace("NVIDIA GeForce ", ""), "vram": f.get("vram_gb")}
                for f in _lan_seen["rows"] if f.get("url") and not seen(f["url"])]
    except Exception:
        pass
    try:
        from fizgig.web import pod_stop, pods
        prefs = _srv().load_prefs()
        cookies = prefs.get("helper_out_cookies") or {}
        key = pod_stop.api_key(prefs)
        rows = _account_rows(key) if key else None
        if rows is None:                 # no key, a refused one, or not answered yet: the Pods tab's saved pods
            rows = [{"name": p.get("name"), "url": p["url"]} for p in pods.saved(prefs) if "proxy.runpod.net" in p["url"]]
        for p in rows:
            if not seen(p["url"]):
                out.append({"name": p.get("name") or (pods.parse(p["url"]).get("id") or p["url"]), "url": p["url"],
                            "where": "pod", "gpu": str(p.get("gpu") or "").replace("NVIDIA GeForce ", ""),
                            "auto_login": bool(p.get("auto_login")), "logged_in": bool(cookies.get(_norm(p["url"])))})
        for p in pods.saved(prefs):      # an address out on the internet (Vast.ai): typed once, on the Pods tab
            if _where(p["url"]) == "address" and not seen(p["url"]):
                out.append({"name": p.get("name") or urllib.parse.urlparse(p["url"]).hostname, "url": p["url"],
                            "where": "address", "gpu": "", "logged_in": bool(cookies.get(_norm(p["url"])))})
    except Exception as e:
        logger.info(f"[helpers] listing computers to add: {_plain(e)}")
    return out


def ensure_lead():
    """The lead's two threads: one checks the helpers and keeps the trainer's note fresh, one sends the rounds (a
    round can take minutes; the note must not go stale meanwhile)."""
    for key, fn in (("checker", _check_loop), ("thread", _lead_loop)):
        if _lead[key] is None or not _lead[key].is_alive():
            _lead[key] = threading.Thread(target=fn, daemon=True, name=f"helpers-{key}")
            _lead[key].start()
    ensure_share()


def _preview_helper(hs):
    """The helper taking this computer's previews, when it's there and can (else None)."""
    h = next((x for x in hs if x.get("previews")), None)
    lv = _live.get(h["id"]) if h else None
    return (h, lv) if h and lv and lv.get("state") == "ready" and lv.get("build") == "same" else (None, None)


def _check_loop():
    while True:
        try:
            if not _lead["mine"]:
                _lead["mine"] = _srv().about_info().get("commit") or ""
            hs = _helpers()
            for h in hs:
                lv, r = _live.setdefault(h["id"], {}), _check(h, _lead["mine"])     # in place: the sender holds it
                if r["state"] == "away" and lv.get("state") not in (None, "away"):
                    # one slow answer (a pod's proxy, Wi-Fi) isn't a helper gone: three in a row, about 15 s
                    lv["misses"] = lv.get("misses", 0) + 1
                    if lv["misses"] < 3:
                        continue
                lv["misses"] = 0
                lv.update(r)
                if r.get("pod") and r.get("state") == "ready" and r.get("build") not in ("", "same", None) and \
                        _lead["mine"]:
                    _match_pod(h, lv, _lead["mine"])
                elif r.get("pod") and r.get("state") == "busy" and r.get("build") not in ("", "same", None):
                    lv["match"] = "It switches to this computer's Fizgig once its run is done"
                elif r.get("build") in ("", "same"):
                    lv["match"] = ""
            if _local.get("uuid") and _local_clash():
                local_stop()                 # Preferences put this Fizgig on that card: one Fizgig per card
                _local["text"] = "Stopped: Preferences put this Fizgig on that GPU"
            out_dir, _run = _run_now()
            h, _lv = _preview_helper(hs)
            inbound = next((i for i in _inbound.values() if time.time() - i["at"] < 20), None)
            _flag(out_dir, inbound["name"] if inbound else h.get("name") if h else "",
                  (inbound or {}).get("mb_s") if inbound else (_lv or {}).get("mb_s"))
        except Exception:
            logger.exception("[helpers] checker")
        time.sleep(4.0)


def _lead_loop():
    while True:
        try:
            _dispatch(_helpers())
        except Exception:
            logger.exception("[helpers] dispatcher")
        time.sleep(1.0)


def _run_now():
    s = _srv()
    with s._lock:
        live = s._run["proc"] is not None and s._run["proc"].poll() is None
        return (s._run.get("output_dir") or "", s._run.get("lora") or "") if live else ("", "")


def check_picture(hid):
    """Check picture: the next preview round renders here and on the helper, and the two are compared."""
    _lead["compare"] = round(time.time(), 3)      # the request's id: the trainer answers each once
    if hid in _inbound:
        _inbound[hid]["check_stat"] = "Check picture: the next preview round renders here and there, then they're compared"
    lv = _live.setdefault(hid, {})
    lv["check_stat"] = "Check picture: the next preview round renders here and there, then they're compared"
    return {"say": "The next preview round renders on both, then the pictures are compared."}


def _compare(lv, job, got):
    """How alike the helper's pictures are to this GPU's (PSNR in dB, the worst pair). Both are kept side by side in
    <output_dir>/helper_check/, to judge by eye."""
    import numpy as np
    from PIL import Image
    worst = None
    for mine, theirs in zip(job.get("compare") or [], got):
        keep = os.path.join(os.path.dirname(os.path.dirname(mine)), "helper_check")
        os.makedirs(keep, exist_ok=True)
        stem = os.path.splitext(os.path.basename(mine))[0]
        shutil.copyfile(mine, os.path.join(keep, stem + "_here.png"))
        shutil.copyfile(theirs, os.path.join(keep, stem + "_helper.png"))
        a = np.asarray(Image.open(mine).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(theirs).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            worst = 0.0
            break
        mse = float(((a - b) ** 2).mean())
        p = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
        worst = p if worst is None else min(worst, p)
    # measured 20 Oct: the same setup on another GPU type 27-34 dB (rounding builds up over the steps), a different
    # setup (the SFT look on one side only) 15-16 dB
    where = "both are in the run's helper_check folder"
    if worst is None:
        lv["check_stat"] = "Check picture: nothing to compare"
    elif worst >= 30:
        lv["check_stat"] = (f"Check picture: the same picture as this GPU (epoch {job['epoch']}, PSNR {worst:.0f} dB: "
                            f"rounding only; {where})")
    elif worst >= 22:
        lv["check_stat"] = (f"Check picture: the same picture, with the small differences another GPU type gives "
                            f"(epoch {job['epoch']}, PSNR {worst:.0f} dB; {where})")
    else:
        lv["check_stat"] = (f"Check picture: a different picture from this GPU's (epoch {job['epoch']}, PSNR "
                            f"{worst:.0f} dB): the two aren't set up alike - compare them by eye ({where})")
    logger.info(f"[helpers] {lv['check_stat']}")


# below this the trainer hands rounds over in bf16 (half the size; 34 dB vs fp32 measured); the env sets it for tests
SLOW_MB_S = float(os.environ.get("FIZGIG_HELPER_SLOW_MB_S") or 20.0)


def _flag(out_dir, name, mb_s=None):
    """The note that tells the trainer a helper takes its previews (name) - or none (the trainer renders)."""
    if _lead["flag_dir"] and _lead["flag_dir"] != out_dir:
        try:
            os.remove(os.path.join(_lead["flag_dir"], HELPER_PREVIEWS))
        except OSError:
            pass
        _lead["flag_dir"] = ""
    if not out_dir:
        return
    p = os.path.join(out_dir, HELPER_PREVIEWS)
    if name:
        _write(p, {"on": True, "name": name, "at": time.time(), "compare": _lead.get("compare") or 0,
                   "dtype": "bf16" if mb_s is not None and mb_s < SLOW_MB_S else "fp32"})
        _lead["flag_dir"] = out_dir
    else:
        try:
            os.remove(p)
        except OSError:
            pass


def _dispatch(hs):
    """Send the run's handed-over rounds to its preview helper, one at a time, and bring the pictures back - the
    last run's too after it finishes, until its queue is done."""
    out_dir, run = _rounds_run()
    h, lv = _preview_helper(hs)
    if any(time.time() - i["at"] < 20 for i in _inbound.values()):
        return                              # a computer that connected in takes the rounds itself (pull)
    rounds = sorted(glob.glob(os.path.join(out_dir, HELPER_ROUNDS, "*", "round.json"))) if out_dir else []
    if rounds and not h:
        # a round already handed to a helper still goes to it once it's ready, its Previews switch off or not: the
        # switch is for new rounds, and the trainer went on without this one, so nothing else would render it
        job = _read(rounds[0], {}) or {}           # its run (maybe an earlier one in this folder) and epoch
        owed = (_read(os.path.join(out_dir, "sample", f".previews_{job.get('output_name') or run}.json"), {}) or {}
                ).get("rounds") or {}
        name = (owed.get(str(job.get("epoch"))) or {}).get("helper")
        h = next((x for x in hs if name and x.get("name") == name), None)
        lv = _live.get(h["id"]) if h else None
        if not (lv and lv.get("state") == "ready" and lv.get("build") == "same"):
            h, lv = None, lv
    if not (h and rounds):
        if lv is not None and not rounds:
            lv["activity"] = None
        return
    rd = os.path.dirname(rounds[0])
    try:
        _send_round(h, lv, rd, out_dir, run)
    except PermissionError:
        lv["state"] = "login"
    except Exception as e:
        logger.warning(f"[helpers] round {os.path.basename(rd)} on {h.get('name')}: {e}")
        lv["activity"] = {"kind": "text", "text": f"Waiting to try again: {_plain(e)}"}
        time.sleep(5)


def _send_round(h, lv, rd, out_dir, run):
    job = _read(os.path.join(rd, "round.json"))
    epoch = int(job["epoch"])
    if job.get("compare"):
        _lead["compare"] = False
    c = Client(h["url"], h.get("cookie", ""))
    queued = len(glob.glob(os.path.join(os.path.dirname(rd), "*", "round.json"))) - 1
    act = lambda **a: lv.__setitem__("activity", {"run": run, "epoch": epoch, "queued": queued, **a})
    tag = hashlib.sha1(f"{socket.gethostname()}|{out_dir}|{job['output_name']}".encode()).hexdigest()[:8]
    # the model files first: a helper without them fetches them itself, and the round waits for them
    m = c.call("/api/helper/models", {"family": job["family"], "job": job, "fetch": True})
    lv["models"] = {"family": m.get("family"), "have": m.get("have") or [], "missing": m.get("missing") or []}
    lv["models_ask"] = {"family": job["family"], "job": job}
    if m.get("missing"):
        if m.get("error"):
            raise ValueError(m["error"])
        f = m.get("fetching") or {}
        act(kind="models", what=f.get("what"), got=float(f.get("got") or 0), total=float(f.get("total") or 1) or 1,
            **{"for": f"{run}'s previews"})
        time.sleep(2)
        return
    lora = f"{tag}-e{epoch:06d}.safetensors"
    t_up = time.time()
    sent = _send_file(c, os.path.join(rd, "lora.safetensors"), lora, lambda frac: act(kind="previews", progress=frac * .3,
                                                                                       text="Sending the LoRA"))
    if sent > 50e6:                                  # long enough to time the link
        lv["mb_s"] = sent / 1e6 / max(.1, time.time() - t_up)
    ctx = ""
    if job.get("context"):
        cp = job["context"][0]
        ctx = f"ctx-{hashlib.sha1(f'{cp}|{os.path.getmtime(cp)}'.encode()).hexdigest()[:10]}.safetensors"
        if c.call(f"/api/helper/have?name={ctx}")["size"] != os.path.getsize(cp):
            _send_file(c, cp, ctx, lambda frac: act(kind="previews", progress=0.0, text="Sending the context LoRA"))
    conds = ""
    if os.path.isfile(os.path.join(rd, "conds.pt")):
        conds = f"{tag}-e{epoch:06d}-conds.pt"
        _send_file(c, os.path.join(rd, "conds.pt"), conds, lambda frac: None)
    r = c.call("/api/helper/submit", {"job": job, "lora": lora, "context": ctx, "conds": conds}, timeout=60)
    if r.get("models"):
        return
    jid, t0 = r["job"], time.time()
    ahead = False
    while True:
        st = c.call(f"/api/helper/status?job={jid}")
        if st["state"] == "done":
            break
        if st["state"] == "rendering" and not ahead:     # the next round's files go up while this one renders
            ahead = True
            nxt = sorted(p for p in glob.glob(os.path.join(os.path.dirname(rd), "*", "round.json"))
                         if os.path.dirname(p) > rd)
            if nxt:
                nd = os.path.dirname(nxt[0])
                ne = int(os.path.basename(nd))
                try:
                    _send_file(c, os.path.join(nd, "lora.safetensors"), f"{tag}-e{ne:06d}.safetensors", lambda f: None)
                    if os.path.isfile(os.path.join(nd, "conds.pt")):
                        _send_file(c, os.path.join(nd, "conds.pt"), f"{tag}-e{ne:06d}-conds.pt", lambda f: None)
                except Exception as e:
                    logger.info(f"[helpers] sending epoch {ne} ahead: {e} (it goes in its turn)")
            continue
        if st["state"] in ("failed", "gone"):
            c.call("/api/helper/drop", {"job": jid, "lora": lora})
            raise ValueError(st.get("error") or "the helper lost the job")
        w = st.get("worker") or {}
        act(kind="previews", progress=.3 if st["state"] != "rendering" else min(.95, .3 + (time.time() - t0) / 120),
            text=(f"Loading the model there, the first round only ({w.get('text') or 'starting'})"
                  if st["state"] == "loading" else "Waiting its turn there" if st["state"] == "queued" else ""))
        if not os.path.isdir(rd):
            return
        time.sleep(1.5)
    sample_dir = os.path.join(out_dir, "sample") if not job.get("compare") else os.path.join(rd, "theirs")
    os.makedirs(sample_dir, exist_ok=True)
    got = []
    for name in st.get("files") or []:
        if not _SAFE.match(str(name)) or not name.endswith(".png"):
            continue                         # only a plain picture name, never a path
        data = c.call(f"/api/helper/file?job={jid}&name={urllib.parse.quote(name)}", timeout=120)
        tmp = os.path.join(sample_dir, name + ".part")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(sample_dir, name))
        got.append(os.path.join(sample_dir, name))
    c.call("/api/helper/drop", {"job": jid, "lora": lora})
    if conds:
        c.call("/api/helper/drop", {"job": "", "lora": conds})
    if job.get("compare"):
        _compare(lv, job, got)
    else:
        lv["previews_stat"] = f"Last round: epoch {epoch} in {st.get('seconds', 0):.0f} s on its GPU"
    shutil.rmtree(rd, ignore_errors=True)
    lv["activity"] = None
    logger.info(f"[helpers] epoch {epoch}: {len(st.get('files') or [])} preview(s) from {h.get('name')}")


def _send_file(c, path, name, progress):
    """Send a file a piece at a time (what's there already is kept). Returns the bytes sent."""
    size = os.path.getsize(path)
    at = c.call(f"/api/helper/have?name={name}")["size"]
    at = at if 0 <= at <= size else 0
    start = at
    with open(path, "rb") as f:
        f.seek(at)
        while at < size:
            piece = f.read(PIECE)
            for n in range(5):                      # a stalled put is sent again, not waited out
                try:
                    c.call(f"/api/helper/put?name={name}&offset={at}", raw=piece, timeout=60)
                    break
                except (OSError, http.client.HTTPException):
                    if n == 4:
                        raise
                    time.sleep(2)
            at += len(piece)
            progress(at / size)
    return at - start



# =====================================================================================================================
# pull: a computer this one can't reach (home, behind its router) helps a run here (a pod) by connecting in itself
# =====================================================================================================================
_inbound = {}              # helper name -> {name, gpu, vram, at, activity, stat}: computers that pulled lately


def _rounds_run():
    """The run whose rounds go to helpers: the one training, else the last one until its queue is done."""
    out_dir, run = _run_now()
    if out_dir:
        _lead["last_run"] = (out_dir, run)
        return out_dir, run
    if not _lead.get("last_run") and time.time() - _lead.get("looked", 0) > 60:
        # after a restart here: a recent run's folder that still has rounds a helper owes it
        _lead["looked"] = time.time()
        try:
            for r in (_srv().history.runs() or [])[:20]:
                d = r.get("output_dir") or ""
                if d and glob.glob(os.path.join(d, HELPER_ROUNDS, "*", "round.json")):
                    _lead["last_run"] = (d, r.get("lora") or "")
                    break
        except Exception:
            pass
    return _lead.get("last_run") or ("", "")


def _round_dir(name):
    out_dir, _run = _rounds_run()
    if not out_dir or not re.match(r"^\d{6}$", str(name or "")):
        raise ValueError("No such round.")
    return out_dir, os.path.join(out_dir, HELPER_ROUNDS, name)


def pull(body):
    """A helper asks for work (and says it's there): the oldest round nobody has, or none."""
    name = str(body.get("name") or "helper")[:60]
    i = _inbound.setdefault(name, {"name": name})
    i.update(at=time.time(), gpu=body.get("gpu"), vram=body.get("vram"))
    if body.get("activity") is not None:
        i["activity"] = body.get("activity") or None
    if body.get("stat"):
        i["stat"] = body["stat"]
    if body.get("mb_s"):
        i["mb_s"] = float(body["mb_s"])
    ensure_lead()
    out_dir, run = _rounds_run()
    if not out_dir:
        return {"run": ""}
    for rj in sorted(glob.glob(os.path.join(out_dir, HELPER_ROUNDS, "*", "round.json"))):
        rd = os.path.dirname(rj)
        taken = _read(os.path.join(rd, "taken.json"), {})
        if taken and (taken.get("by") != name and time.time() - taken.get("at", 0) < 600):
            continue
        _write(os.path.join(rd, "taken.json"), {"by": name, "at": time.time()})
        return {"run": run, "round": os.path.basename(rd), "job": _read(rj),
                "size": os.path.getsize(os.path.join(rd, "lora.safetensors"))}
    return {"run": run}


def lora_file(name, offset=0, n=PIECE, which="lora"):
    _out_dir, rd = _round_dir(name)
    with open(os.path.join(rd, "conds.pt" if which == "conds" else "lora.safetensors"), "rb") as f:
        f.seek(int(offset))
        return f.read(max(0, min(int(n), PIECE)))


def deliver(name, picture, data):
    """A picture the helper made for a round, into the run's samples folder."""
    out_dir, rd = _round_dir(name)
    if not _SAFE.match(picture or "") or not picture.endswith(".png"):
        raise ValueError("Bad picture name.")
    compare = (_read(os.path.join(rd, "round.json"), {}) or {}).get("compare")
    sd = os.path.join(rd, "theirs") if compare else os.path.join(out_dir, "sample")
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, picture + ".part"), "wb") as f:
        f.write(data)
    os.replace(os.path.join(sd, picture + ".part"), os.path.join(sd, picture))
    return {"ok": True}


def finish(name, ok, error=""):
    _out_dir, rd = _round_dir(name)
    if ok:
        job = _read(os.path.join(rd, "round.json"), {}) or {}
        if job.get("compare"):
            _lead["compare"] = False
            lv = {}
            _compare(lv, job, sorted(glob.glob(os.path.join(rd, "theirs", "*.png"))))
            for i in _inbound.values():
                if time.time() - i["at"] < 20:
                    i["check_stat"] = lv.get("check_stat", "")
        shutil.rmtree(rd, ignore_errors=True)
    else:
        try:
            os.remove(os.path.join(rd, "taken.json"))
        except OSError:
            pass
        logger.warning(f"[helpers] round {name} came back failed: {error}")
    return {"ok": True}


def _pod_password(url):
    """A pod of this account's password, found as Add does (its RunPod settings, else its start-up log), or ""."""
    if "proxy.runpod.net" not in url:
        return ""
    from fizgig.web import pod_stop, pods
    return pods.login_password(pod_stop.api_key(_srv().load_prefs()), url)


def help_start(url, password="", mode="previews"):
    """This computer helps the run at `url` with its previews, or (mode "share") trains its share of each epoch;
    it connects, so the other end needn't reach it."""
    url = _norm(url)
    s = _srv()
    cookies = dict(s.load_prefs().get("helper_out_cookies") or {})
    c = Client(url, cookies.get(url, ""))
    password = password or ("" if c.cookie else _pod_password(url))
    if password:
        c.login(password)
        cookies[url] = c.cookie
        s.update_prefs({"helper_out_cookies": cookies})
    try:
        i = c.call("/api/helper/info")
    except PermissionError:
        raise ValueError("Its password is needed: type it into the box beside it.")
    from fizgig.web import pods
    if (pods._build_word(s.about_info().get("commit") or "", i.get("commit")) or "same") != "same":
        raise ValueError("It runs a different Fizgig version: update both, then try again.")
    name = i.get("name") or url
    if s._gpu_training():                # this GPU has its own run: it can't help another
        raise ValueError(f"This computer is training its own run, so its GPU can't help {name} now. To have {name} "
                         "make this run's previews, turn on Previews on its tile under Your helpers.")
    idle = not (c.call("/api/run") or {}).get("running")      # set up ahead of its run: it waits for one
    _out.clear()                         # a loop still going for the last start sees a new generation, and ends
    _gen[0] += 1
    share = mode == "share"
    _out.update(url=url, name=i.get("name") or url, gen=_gen[0], mode=mode,
                text="Connecting for its training share" if share else "Waiting for its next preview round")
    threading.Thread(target=_out_share_loop if share else _out_loop, args=(c, dict(_out)), daemon=True,
                     name="helpers-out").start()
    if idle:
        return {"say": f"Nothing is training on {name} yet. This GPU starts helping when a run starts there."}
    return {"say": f"Training a share of {_out['name']}'s run (experimental)." if share else
            f"Helping {_out['name']} with its previews."}


def _again(fn, tries=5):
    """A call that a dropped connection or a proxy's error page shouldn't end: asked again a few times."""
    for n in range(tries):
        try:
            return fn()
        except PermissionError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as e:
            if n == tries - 1:
                raise
            logger.info(f"[helpers] asking again ({_plain(e)})")
            time.sleep(2)


def _piece(c, path, tries=5):
    """One piece of a pulled file: a proxy that stalls a request gets it asked again soon, not waited out."""
    for n in range(tries):
        try:
            return c.call(path, timeout=30)
        except (OSError, http.client.HTTPException) as e:
            if n == tries - 1:
                raise
            logger.info(f"[helpers] a piece stalled ({e}); asking again")
            time.sleep(2)


def help_stop():
    _out.clear()
    _gen[0] += 1
    return {"say": "Stopped helping."}


def _out_loop(c, me):
    s = _srv()
    mine = info()
    say = lambda t: _out.__setitem__("text", t) if _out.get("gen") == me["gen"] else None
    stat, mb_s = "", None
    while _out.get("gen") == me["gen"]:
        try:
            if s._gpu_training():
                say("Paused: this computer is training")
                c.call("/api/helpers/pull", {"name": mine["name"], "gpu": mine["gpu"], "vram": mine["vram"],
                                             "activity": {"kind": "text", "text": "Paused: training its own run"}})
                time.sleep(10)
                continue
            r = c.call("/api/helpers/pull", {"name": mine["name"], "gpu": mine["gpu"], "vram": mine["vram"],
                                             "activity": None, "stat": stat, "mb_s": mb_s})
            if not r.get("round"):
                say(f"Waiting for {r['run']}'s next preview round" if r.get("run") else "Nothing training there")
                time.sleep(3)
                continue
            job, rnd = r["job"], r["round"]
            act = {"kind": "previews", "run": r["run"], "epoch": job["epoch"], "queued": 0}
            push = lambda **a: c.call("/api/helpers/pull", {"name": mine["name"], "activity": {**act, **a}})
            lora = f"in-{hashlib.sha1(me['url'].encode()).hexdigest()[:8]}-{rnd}.safetensors"
            size = int(r["size"])
            at = max(0, min(have(lora)["size"], size))      # a round tried before carries on where it stopped
            t_dl, start = time.time(), at
            while at < size:
                piece = _piece(c, f"/api/helpers/lora?round={rnd}&offset={at}&n={PIECE}")
                put(lora, at, piece)
                at += len(piece)
                say(f"Getting {r['run']}'s epoch {job['epoch']} LoRA ({at * 100 // size}%)")
                push(progress=.3 * at / size, text="Sending the LoRA")
            conds = ""
            if job.get("conds"):
                conds, at = lora[:-len(".safetensors")] + "-conds.pt", 0
                while True:
                    piece = _piece(c, f"/api/helpers/lora?round={rnd}&which=conds&offset={at}&n={PIECE}")
                    put(conds, at, piece) if piece or at == 0 else None
                    at += len(piece)
                    if len(piece) < PIECE:
                        break
            if at - start > 50e6:                           # the link's speed, for the lead's LoRA size choice
                mb_s = (at - start) / 1e6 / max(.1, time.time() - t_dl)
            while True:
                sub_ = submit({"job": job, "lora": lora, "context": "", "conds": conds})
                if not sub_.get("models"):
                    break
                m = sub_["models"]
                if m.get("error"):
                    raise ValueError(m["error"])
                f = m.get("fetching") or {}
                say(f"Getting {f.get('what') or 'models'} for {r['run']}'s previews")
                push(kind="models", what=f.get("what"), got=float(f.get("got") or 0),
                     total=float(f.get("total") or 1) or 1, **{"for": f"{r['run']}'s previews"})
                time.sleep(3)
            jid, t0 = sub_["job"], time.time()
            while True:
                st = status(jid)
                if st["state"] in ("done", "failed", "gone"):
                    break
                say(f"Rendering {r['run']}'s previews for epoch {job['epoch']}")
                push(progress=min(.95, .3 + (time.time() - t0) / 120) if st["state"] == "rendering" else .3)
                time.sleep(2)
            if st["state"] != "done":
                c.call("/api/helpers/finish", {"round": rnd, "ok": False, "error": st.get("error") or ""})
                drop(jid, lora)
                say(f"Epoch {job['epoch']} failed here: {st.get('error') or 'lost'}")
                time.sleep(5)
                continue
            for name in st.get("files") or []:
                with open(file_path(jid, name), "rb") as f:
                    c.call(f"/api/helpers/deliver?round={rnd}&name={urllib.parse.quote(name)}", raw=f.read(),
                           timeout=120)
            c.call("/api/helpers/finish", {"round": rnd, "ok": True})
            drop(jid, lora)
            if conds:
                drop("", conds)
            stat = f"Last round: epoch {job['epoch']} in {st.get('seconds', 0):.0f} s on its GPU"
            say(f"Sent {r['run']}'s epoch {job['epoch']} previews ({st.get('seconds', 0):.0f} s)")
        except PermissionError:
            say("Its login ran out: start helping again with its password")
            return
        except Exception as e:
            logger.warning(f"[helpers] helping {me['url']}: {e}")
            say("Can't reach it right now: trying again every few seconds")
            time.sleep(8)


# =====================================================================================================================
# trying prompts on another computer's run: its checkpoints, fetched here, for LoRA Royale on this GPU
# =====================================================================================================================
_remote = {"state": "", "done": 0, "total": 0, "folder": "", "error": "", "name": "", "te_note": ""}


def _client_for(url):
    """A logged-in client for a Fizgig this one helps or has as a helper (its saved login)."""
    url = _norm(url)
    s = _srv()
    cookie = (s.load_prefs().get("helper_out_cookies") or {}).get(url) or next(
        (h.get("cookie") for h in _helpers() if h.get("url") == url), "")
    return url, Client(url, cookie)


def remote_checkpoints(url, password=""):
    """The run at `url`: its name and its saved checkpoints, newest first (password: log in to it first, and keep
    the login)."""
    url, c = _client_for(url)
    password = password or ("" if c.cookie else _pod_password(url))
    if password:
        c.login(password)
        s = _srv()
        cookies = dict(s.load_prefs().get("helper_out_cookies") or {})
        cookies[url] = c.cookie
        s.update_prefs({"helper_out_cookies": cookies})
    try:
        run = c.call("/api/run")
    except PermissionError:
        raise ValueError("Its password is needed: help it first (Help it with previews) or add it as a helper.")
    out = run.get("output_dir") or ""
    if not out:                          # nothing going (or it just restarted): its newest run in Runs & outputs
        last = next((x for x in (c.call("/api/runs").get("runs") or []) if x.get("output_dir")), None)
        if last is None:
            return {"run": "", "family": "", "items": []}
        run = {"output_dir": last["output_dir"], "lora": last.get("lora"), "family": last.get("family")}
        out = run["output_dir"]
    items = c.call("/api/checkpoints?dir=" + urllib.parse.quote(out)).get("items") or []
    name = run.get("lora") or ""
    items = [i for i in items if not name or i.get("run") in (name, name + ".safetensors")]   # + the final LoRA
    return {"run": name, "family": run.get("family") or "", "items": [
        {"name": i["name"], "path": i["path"], "size": i["size"], "time": i["time"]} for i in items]}


def remote_get(url, items, run="", family=""):
    """Fetch the picked checkpoints into user_data/helper/remote/<their run>/ (what's there already is kept), on a
    thread; remote_status() follows it. Returns the folder they land in."""
    if _remote.get("state") == "running":
        raise ValueError("Checkpoints are already on their way.")
    url, c = _client_for(url)
    run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run or "run"))[:60]
    folder = _dir("remote", hashlib.sha1(url.encode()).hexdigest()[:6] + "-" + run)
    os.makedirs(folder, exist_ok=True)
    _remote.update(state="running", te_note="", done=0, total=sum(int(i.get("size") or 0) for i in items), folder=folder,
                   error="", name="", family=family)

    def work():
        try:
            for i in items:
                name = os.path.basename(str(i["path"]))
                if not _SAFE.match(name) or not name.endswith(".safetensors"):
                    continue
                dest = os.path.join(folder, name)
                if os.path.isfile(dest) and os.path.getsize(dest) == int(i.get("size") or -1):
                    _remote["done"] += os.path.getsize(dest)
                    continue
                _remote["name"] = name
                c.download("/api/download?path=" + urllib.parse.quote(str(i["path"])), dest,
                           lambda n: _remote.__setitem__("done", _remote["done"] + n))
            _light_text_encoder(family)
            _remote["state"] = "done"
        except Exception as e:
            logger.warning(f"[helpers] fetching checkpoints from {url}: {e}")
            _remote.update(state="failed", error=str(e) or type(e).__name__)
    threading.Thread(target=work, daemon=True, name="helpers-remote").start()
    return {"folder": folder}


def remote_status():
    st = dict(_remote)
    if st.get("te_dir"):                   # the text encoder's download: its part file in the models folder's cache
        try:
            st["done"] = max((os.path.getsize(os.path.join(st["te_dir"], n)) for n in os.listdir(st["te_dir"])
                              if n.endswith(".incomplete")), default=st["done"])
        except OSError:
            pass
    return st


def _light_text_encoder(family):
    """Prompts are encoded here when you try them on another computer's epochs: a computer without the family's text
    encoder fetches its lighter build when the family names one (the lightest close one in text_encoder_builds, else
    alt_*: MOSS-Image's int8 Qwen3.5-9B, 9.7 GB not 19.3), into the models folder, and points Preferences at it. Its own download, the regular downloader untouched.
    The lighter build conditions a little differently; it's this computer's own trial, not the run's previews."""
    s = _srv()
    desc = s.get_family(family) if family else None
    f = next((m for m in (desc.model_files if desc else ()) if m.role == "text_encoder"), None)
    if f is None:
        return
    have = s.pref_path(s.load_prefs(), f.pref_key)
    if have and os.path.isfile(have):
        return
    # the lightest build that conditions all but the same (cosine 0.99 and up, not a variant), else the family's alt
    light = sorted((b for b in getattr(desc, "text_encoder_builds", ()) if not b.variant and b.cosine >= 0.99
                    and b.size_gb < f.size_gb), key=lambda b: b.size_gb)
    if light:
        repo, path, label, gb = light[0].repo, light[0].path, light[0].label, light[0].size_gb
    else:
        repo, path = (f.alt_repo, f.alt_path) if f.alt_repo and f.alt_path else (f.repo, f.path)
        label = f.alt_label or "full"
        gb = (f.alt_size_gb or f.size_gb) if f.alt_path else f.size_gb
    from fizgig import user_data
    _remote.update(name=f"{f.label} ({label} build) for trying prompts here", done=0, total=int(gb * 1e9),
                   te_dir=os.path.join(user_data.MODELS_DIR, ".cache", "huggingface", "download", os.path.dirname(path)))
    try:
        from fizgig.utils import hf_public
        got = hf_public.download(repo_id=repo, filename=path, local_dir=user_data.MODELS_DIR)
    finally:
        _remote["te_dir"] = ""
    dest = os.path.join(user_data.MODELS_DIR, os.path.basename(path))
    if os.path.normcase(os.path.abspath(got)) != os.path.normcase(os.path.abspath(dest)):
        os.replace(got, dest)
    s.update_prefs({f.pref_key: dest})
    _remote["te_note"] = (f"Preferences' {f.label} row had no file, so it now points at {os.path.basename(dest)} "
                          f"(the {label} build, fetched for trying prompts here)")
    logger.info(f"[helpers] {os.path.basename(dest)} fetched for trying prompts on another computer's run")


# =====================================================================================================================
# this computer's other GPU: a second Fizgig on it, added as a helper (previews, a training share, a move)
# =====================================================================================================================
_local = {"proc": None, "index": None, "url": "", "text": "", "gpus": None, "at": 0.0}


def local_gpus():
    """This computer's GPUs other than the one Fizgig uses (Preferences' GPU, else the first), for Use this
    computer's other GPU; none inside that second Fizgig itself. nvidia-smi's list, asked once a minute."""
    if os.environ.get("FIZGIG_USER_DATA"):
        return []
    s = _srv()
    if _local["gpus"] is None or time.time() - _local["at"] > 60:
        _local.update(gpus=s.list_gpus(), at=time.time())
    gpus = [g for g in _local["gpus"] if g.get("uuid")]
    mine = s._chosen_gpu(gpus, s.load_prefs().get("cuda_device"))
    return [{"index": g["index"], "uuid": g["uuid"], "name": g["name"].replace("NVIDIA GeForce ", ""),
             "vram": round(g["total_gb"])} for g in gpus if not mine or g["uuid"] != mine["uuid"]]


def _local_clash():
    """Preferences' GPU for this Fizgig is the card the second one has (a failed nvidia-smi says nothing)."""
    s = _srv()
    local_gpus()                                     # the list, fresh once a minute
    gpus = [g for g in _local["gpus"] or [] if g.get("uuid")]
    mine = s._chosen_gpu(gpus, s.load_prefs().get("cuda_device")) if gpus else None
    return bool(mine and mine["uuid"] == _local.get("uuid"))


def local_status():
    p = _local["proc"]
    on = p is not None and p.poll() is None
    return {"gpus": [{k: g[k] for k in ("index", "name", "vram")} for g in local_gpus()],
            "on": _local["index"] if on else None, "text": _local["text"]}


def local_restore():
    """At start-up: the other GPU was in use as a helper, so its Fizgig starts again."""
    idx = _srv().load_prefs().get("helper_local_gpu")
    if idx not in (None, ""):
        try:
            local_start(idx)
        except Exception as e:
            logger.warning(f"[helpers] this computer's other GPU: {_plain(e)}")


def local_start(index):
    """A second Fizgig on GPU `index`: its own settings folder (user_data/other_gpu_N, with this computer's model
    files), on 127.0.0.1 only, with a fresh password; added as a helper once it answers. It ends with this one."""
    s = _srv()
    g = next((x for x in local_gpus() if x["index"] == int(index)), None)
    if g is None:
        raise ValueError("That GPU isn't one this computer can lend: Fizgig itself uses it, or it isn't there.")
    local_stop(keep=True)
    from fizgig import user_data
    ud = user_data.user_file(f"other_gpu_{g['index']}")
    os.makedirs(ud, exist_ok=True)
    pp = os.path.join(ud, "prefs.json")
    theirs = _read(pp, {}) or {}
    for k, v in s.load_prefs().items():          # the model files (and folders) this computer uses: the same ones
        if isinstance(v, str) and v and os.path.isabs(v) and os.path.exists(v) and k != "lora_output_dir":
            theirs[k] = v
    for k in ("cuda_device", "web_password", "web_port", "helpers", "helper_local_gpu", "helper_bring_home",
              "cache_dir", "profiles_dir", "input_dataset_dir"):     # its caches and uploads: its own
        theirs.pop(k, None)
    _write(pp, theirs)
    # the same port each time it starts (a run moved to it keeps its address across a restart), else a free one
    port = int(s.load_prefs().get("helper_local_port") or 0)
    with socket.socket() as sk:
        try:
            sk.bind(("127.0.0.1", port))
        except OSError:
            sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    s.update_prefs({"helper_local_port": port})
    pw = secrets.token_urlsafe(18)
    env = {k: v for k, v in os.environ.items() if k not in ("FIZGIG_PASSWORD", "FIZGIG_APP_KEY", "FIZGIG_RESTARTED")}
    env.update(CUDA_VISIBLE_DEVICES=g["uuid"], FIZGIG_USER_DATA=ud, FIZGIG_PARENT_PID=str(os.getpid()),
               FIZGIG_HELPER_NAME=f"This computer's {g['name']} (GPU {g['index']})", PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(s.BASE_DIR, "src"))
    try:
        os.remove(os.path.join(ud, "stop"))
    except OSError:
        pass
    log = open(os.path.join(ud, "server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "fizgig.web", "--host", "127.0.0.1", "--port", str(port),
                             "--password", pw], stdout=log, stderr=subprocess.STDOUT, cwd=s.BASE_DIR, env=env,
                            **_NO_WINDOW)
    url = f"http://127.0.0.1:{port}"
    _local.update(proc=proc, index=g["index"], uuid=g["uuid"], url=url, text=f"Starting Fizgig on the {g['name']}", log=log, dir=ud)
    s.update_prefs({"helper_local_gpu": g["index"]})

    def join():
        for _ in range(90):
            if proc.poll() is not None:
                _local["text"] = f"Its Fizgig stopped as it started: see {os.path.join(ud, 'server.log')}"
                return
            try:
                add(url, pw)
                _local["text"] = ""
                return
            except ValueError:
                time.sleep(2)
        _local["text"] = f"Its Fizgig didn't answer: see {os.path.join(ud, 'server.log')}"
    threading.Thread(target=join, daemon=True, name="helpers-local").start()
    return {"say": f"Starting a second Fizgig on the {g['name']}: it shows below as a helper in a few seconds."}


def local_stop(keep=False):
    """The second Fizgig ends (its runs and share trainers with it) and leaves the helpers; keep: starting again."""
    p, ud = _local.get("proc"), _local.get("dir")
    if p is not None and p.poll() is None:
        open(os.path.join(ud, "stop"), "w").close()      # it ends its own runs and share trainers, as Ctrl+C does
        try:
            p.wait(45)
        except subprocess.TimeoutExpired:
            p.kill()
    if _local.get("log") is not None:
        _local["log"].close()
    if _local.get("url"):
        rows = [h for h in _helpers() if _norm(h.get("url")) != _norm(_local["url"])]
        _save(rows)
    _local.update(proc=None, index=None, uuid="", url="", text="", log=None)
    if not keep:
        _srv().update_prefs({"helper_local_gpu": None})
        return {"say": "Stopped using the other GPU."}
    return {}


def shutdown():
    """This Fizgig is going (a second one on another GPU, told to stop): its share trainers and preview worker go
    with it."""
    share_stop_all()
    p = _w.get("proc")
    if p is not None and p.poll() is None:
        p.terminate()


# =====================================================================================================================
# a pod's run brought home: its checkpoints (and previews) fetched here as they're saved, the pod deleted after
# =====================================================================================================================
HOME_DIR = "From pods"
HOME_GIVE_UP = 900          # seconds after its run finished: still not all home, so the pod is stopped, not deleted
_home = {}                  # url -> what has come home from it: text, run, finished_at
_home_done = []             # pods deleted since this Fizgig started: what was said
_home_thread = [None]


def _home_rows():
    return {u: r for u, r in (_srv().load_prefs().get("helper_bring_home") or {}).items() if isinstance(r, dict)}


def home_folders():
    """Every folder a pod's run came home to (server.output_folders lists them: Runs & outputs, LoRA Royale)."""
    return [f for f in (_srv().load_prefs().get("helper_home_folders") or []) if isinstance(f, str)]


def checkpoint_sha(path):
    """A checkpoint's SHA-256 (the files /api/download serves), for the computer that brought it home to check its
    copy before it deletes this pod."""
    if not str(path).lower().endswith(".safetensors") or not os.path.isfile(path):
        raise ValueError("not a checkpoint")
    return {"sha": _sha(path)}


def home_set(url, on=None, previews=None, delete=None, password=""):
    """A pod's options on Help another Fizgig with this GPU: bring its run's checkpoints home as they're saved
    (its previews too), and delete the pod once its run has finished and everything is home and checked."""
    url, c = _client_for(url)
    s = _srv()
    password = password or ("" if c.cookie else _pod_password(url))
    if password:
        c.login(password)
        cookies = dict(s.load_prefs().get("helper_out_cookies") or {})
        cookies[url] = c.cookie
        s.update_prefs({"helper_out_cookies": cookies})
    rows = _home_rows()
    row = dict(rows.get(url) or {})
    for k, v in (("on", on), ("previews", previews), ("delete", delete)):
        if v is not None:
            row[k] = bool(v)
    if row.get("delete"):
        from fizgig.web import pod_stop, pods
        if not pods.parse(url).get("id"):
            raise ValueError("Only a RunPod pod can be deleted: this is another computer.")
        if not pod_stop.api_key(s.load_prefs()):
            raise ValueError("Deleting it needs your RunPod API key: add it on the Pods tab.")
        row["on"] = True
    if row.get("on"):
        try:
            c.call("/api/run")
        except PermissionError:
            raise ValueError("Its password is needed: type it, then tick this again.")
        from fizgig.web import pods
        pid = pods.parse(url).get("id") or ""
        name = next((p["name"] for p in _pods_to_help() if _norm(p["url"]) == url), "") or pid or \
            urllib.parse.urlparse(url).netloc
        name = f"{name} ({pid})" if pid and pid not in name else name   # pods from one template share a name
        row.setdefault("dest", os.path.join(s.default_output_dir(), HOME_DIR,
                                            re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")[:80] or "pod"))
        rows[url] = row
        folders = home_folders()
        s.update_prefs({"helper_bring_home": rows,
                        "helper_home_folders": folders + [row["dest"]] if row["dest"] not in folders else folders})
    else:
        rows.pop(url, None)
        _home.pop(url, None)
        s.update_prefs({"helper_bring_home": rows})
    ensure_home()
    if not row.get("on"):
        return {"say": "Its checkpoints stay on it."}
    return {"say": f"Its checkpoints come home to {row['dest']} as they're saved" +
            (", and the pod is deleted once its run has finished and they're all here." if row.get("delete") else ".")}


def _home_row(url, **kw):
    """One pod's bring-home row, changed and saved."""
    rows = _home_rows()
    if url in rows:
        rows[url] = {**rows[url], **kw}
        _srv().update_prefs({"helper_bring_home": rows})


def ensure_home():
    if _home_rows() and (_home_thread[0] is None or not _home_thread[0].is_alive()):
        _home_thread[0] = threading.Thread(target=_home_loop, daemon=True, name="helpers-home")
        _home_thread[0].start()


def home_status(url):
    st = _home.get(_norm(url)) or {}
    return {**{k: v for k, v in (_home_rows().get(_norm(url)) or {}).items() if k in ("on", "previews", "delete")},
            "text": st.get("text") or ""}


def _home_loop():
    nxt = {}
    while True:
        for url, row in _home_rows().items():
            if not row.get("on") or time.time() < nxt.get(url, 0):
                continue
            st = _home.setdefault(url, {"text": "Looking for its checkpoints"})
            try:
                _home_active[url] = True
                nxt[url] = time.time() + _home_once(url, row, st)
            except PermissionError:
                st["text"] = "Its login ran out: type its password, then tick this again"
                nxt[url] = time.time() + 60
            except Exception as e:
                logger.info(f"[helpers] bringing {url}'s checkpoints home: {_plain(e)}")
                st["text"] = f"Couldn't bring them all home just now ({_plain(e)}): trying again in a minute"
                nxt[url] = time.time() + 60
            finally:
                _home_active[url] = False
            done_at = float(_home_rows().get(url, {}).get("finished_at") or 0)
            if row.get("delete") and done_at and time.time() - done_at > HOME_GIVE_UP and not st.get("deleted"):
                _home_give_up(url, st, "Not everything came home within 15 minutes of its run finishing")
        time.sleep(5)


def _home_once(url, row, st):
    """One look at the pod: what's new comes home. Returns the seconds until the next look."""
    url, c = _client_for(url)
    if st.get("back"):                   # a moved run of this computer's, being brought back before the pod goes
        if _mv.get("state") == "back":
            return 10
        if _mv.get("state") == "moved":      # the bring-back didn't work: the run is still there, tried again
            st.pop("back", None)
            raise ValueError(f"bringing it back didn't work ({_mv.get('error') or 'stopped'})")
        _home_delete(url, row, st, *st["count"])
        st.pop("back", None)
        st.pop("count", None)
        return 60
    r = c.call("/api/run", timeout=20)
    out, name, state = r.get("output_dir") or "", r.get("lora") or "", r.get("state") or ""
    if not out:                          # nothing going (or it restarted): its newest run in Runs & outputs
        last = next((x for x in (c.call("/api/runs").get("runs") or []) if x.get("output_dir")), None)
        if last is None:
            st["text"] = "Nothing trained there yet"
            return 60
        # a pod restarted after its run finished still counts as finished (else it would bill on, never deleted)
        out, name = last["output_dir"], last.get("lora") or ""
        state = "completed" if last.get("result") == "finished" else ""
    if st.get("run") != f"{out}|{name}":
        st.update(run=f"{out}|{name}", finished_at=0)
    finished = state == "completed" and not (r.get("web_queue") or {}).get("items")
    # this computer's own run, moved to that pod: it comes home into its own output folder, and at the end it's
    # brought back as Bring its LoRAs here does, which also closes the move
    moved = (_mv.get("state") in ("moved", "back") and _norm(_mv.get("url") or "x") == url
             and _mv.get("run") == name and _mv.get("remote_out") == out)
    if _mv.get("state") == "back" and _norm(_mv.get("url") or "x") == url:
        return 10                            # being brought back: the bring-back fetches the rest
    from fizgig.web import pod_stop, pods
    key, pid = pod_stop.api_key(_srv().load_prefs()), pods.parse(url).get("id")
    # the pod's own stop is held while this computer works on it: 30 minutes, asked again each look, so if this
    # computer goes (asleep, restarted, logged out) the pod's stop goes on by itself (server._hold_stop)
    if finished and row.get("delete"):
        if not row.get("finished_at") or row.get("finished_run") != st["run"]:     # kept across a restart here
            _home_row(url, finished_at=time.time(), finished_run=st["run"])
        c.call("/api/pod/keep", {"hold": 1800})
    elif finished and key and pid and ((r.get("pod_stop") or {}).get("counting") or st.get("stop_after") == st["run"]):
        c.call("/api/pod/keep", {"hold": 1800})  # its own stop would beat the last checkpoint home: done from here
        st["stop_after"] = st["run"]
    if not finished and row.get("finished_at"):  # it trains again (another run there): no give-up clock running
        _home_row(url, finished_at=0, finished_run="")
    want = [i for i in c.call("/api/checkpoints?dir=" + urllib.parse.quote(out)).get("items") or []
            if i.get("run") in (name, name + ".safetensors") and _SAFE_FILE.match(str(i.get("name") or "")) and i["name"].endswith(".safetensors")]
    dest = _mv["out_dir"] if moved else _moved_dest(url, name, out) or row["dest"]
    st["folder"] = dest
    os.makedirs(dest, exist_ok=True)
    for n, i in enumerate(want):
        if _mv.get("state") == "back" and _norm(_mv.get("url") or "x") == url:
            return 10                        # the bring-back started: it fetches the rest
        d = os.path.join(dest, i["name"])
        # its saved time travels with it: a re-run there under the same name comes home again
        if os.path.isfile(d) and os.path.getsize(d) == int(i["size"]) and abs(os.path.getmtime(d) - float(i["time"])) < 2:
            continue
        st["text"] = f"Bringing {i['name']} home ({n} of {len(want)} here)"
        c.download("/api/download?path=" + urllib.parse.quote(i["path"]), d, lambda _n: None)
        if os.path.getsize(d) != int(i["size"]):
            os.remove(d)
            raise ValueError(f"{i['name']} came home the wrong size")
        os.utime(d, (float(i["time"]), float(i["time"])))
    pics = 0
    if row.get("previews"):
        sdir = out.rstrip("/\\") + "/sample"
        items = c.call(f"/api/samples?dir={urllib.parse.quote(sdir)}&run={urllib.parse.quote(name)}").get("items") or []
        os.makedirs(os.path.join(dest, "sample"), exist_ok=True)
        for p in {p for it in items for p in (it.get("path"), it.get("image")) if p}:
            fname = p.replace("\\", "/").rsplit("/", 1)[-1]
            d = os.path.join(dest, "sample", fname)
            if _SAFE_FILE.match(fname) and not os.path.isfile(d):
                st["text"] = f"Bringing {name}'s previews home"
                c.download("/api/media?path=" + urllib.parse.quote(p), d, lambda _n: None)
            pics += 1
    st["text"] = (f"{len(want)} of {name}'s checkpoints home" + (f" and {pics} previews" if pics else "") +
                  f", in {dest}") if want else f"{name or 'Its run'} hasn't saved a checkpoint yet"
    if st.get("stop_after") == st["run"] and not row.get("delete"):
        st.pop("stop_after")
        pods.stop(key, pid)              # as the pod's own Stop the pod when the run ends would have
        st["text"] += ", and the pod was stopped (its own setting), its files still on it"
        return 300
    if not (finished and row.get("delete") and want):
        return 300 if state in ("completed", "", "idle", "stopped", "failed") else (15 if row.get("move") else 60)   # a moved run: little left for Bring back
    # A: never while the pod still works for a computer (previews, a share, a move on its way), or trains anything
    i = c.call("/api/helper/info", timeout=20)
    if (i.get("sharing") or i.get("helping") or i.get("training") or (i.get("worker") or {}).get("state") in
            ("loading", "rendering", "busy") or _mv.get("state") == "moving" and _norm(_mv.get("url") or "x") == url):
        st["text"] = f"{name} has finished; the pod is deleted once it stops working for a computer"
        return 60
    # T: the pod's other runs too: every LoRA and checkpoint on it comes home before it goes
    others = [f for run in c.call("/api/outputs?all=1").get("runs") or [] for f in run.get("files") or []
              if f.get("path") not in {i["path"] for i in want} and _SAFE_FILE.match(str(f.get("name") or ""))]
    for f in others:
        sub = re.sub(r'[\\/:*?"<>|]+', "_", os.path.basename(os.path.dirname(f["path"].replace("\\", "/"))))
        f["home"] = d = os.path.join(row["dest"], "other runs", sub or "pod", f["name"])
        if not (os.path.isfile(d) and os.path.getsize(d) == int(f["size"])):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            st["text"] = f"Bringing the pod's other LoRAs home first: {f['name']}"
            c.download("/api/download?path=" + urllib.parse.quote(f["path"]), d, lambda _n: None)
    want = want + others
    st["text"] = f"{name} has finished: checking every checkpoint came home whole before deleting the pod"
    for i in want:
        theirs = c.call("/api/helper/checkpoint_sha?path=" + urllib.parse.quote(i["path"]), timeout=600).get("sha")
        mine = i.get("home") or os.path.join(dest, i["name"])
        if not theirs or theirs != _sha(mine):
            os.remove(mine)
            raise ValueError(f"{i['name']} didn't match its copy on the pod: fetching it again")
    if moved:
        # checked while the pod still has them; then the move's own bring-back (its loss log, state and captions
        # too), which ends the run there and closes the move here; the pod goes once that's done (_home_once's top)
        if _mv.get("state") == "moved":
            move_back(False)
        st.update(back=st["run"], count=(len(want), pics), text=f"{name} has finished there: bringing it back here")
        return 10
    i = c.call("/api/helper/info", timeout=20)
    if i.get("sharing") or i.get("helping") or i.get("training"):
        st["text"] = f"{name} has finished; the pod is deleted once it stops working for a computer"
        return 60
    _home_delete(url, row, st, len(want), pics)
    return 60


def _home_delete(url, row, st, n, pics):
    from fizgig.web import pod_stop, pods
    s = _srv()
    pods.terminate(pod_stop.api_key(s.load_prefs()), pods.parse(url)["id"])
    name = next((p["name"] for p in _pods_to_help() if _norm(p["url"]) == url), "") or url
    rows = _home_rows()
    rows.pop(url, None)
    s.update_prefs({"helper_bring_home": rows, "app_pods": [p for p in pods.saved(s.load_prefs())
                                                            if _norm(p["url"]) != url]})
    st["deleted"] = True
    st["text"] = (f"{name} was deleted once its run had finished and its {n} checkpoint{'s' if n != 1 else ''}"
                  + (f" and {pics} previews" if pics else "") + f" were home, in {st.get('folder') or row['dest']}")
    _home_done.append(st["text"])
    logger.info(f"[helpers] {st['text']}")


def _home_give_up(url, st, why):
    """Deleting the pod is off for good: it's stopped instead, so its files are still there."""
    from fizgig.web import pod_stop, pods
    s = _srv()
    rows = _home_rows()
    if url in rows:
        rows[url] = {**rows[url], "delete": False}
        s.update_prefs({"helper_bring_home": rows})
    try:
        pods.stop(pod_stop.api_key(s.load_prefs()), pods.parse(url)["id"])
        st["text"] = f"{why}, so the pod was stopped, not deleted: its files are still on it"
    except Exception as e:
        st["text"] = f"{why}, and stopping the pod failed too ({_plain(e)}): stop it on the Pods tab"
    logger.warning(f"[helpers] {url}: {st['text']}")


# =====================================================================================================================
# training share: a helper trains its share of each epoch's images (EXPERIMENTAL)
# =====================================================================================================================
HELPER_SHARE, SHARE_DIR = ".helper_share.json", ".helper_share"     # fizgig.families.train's, without torch
HELPER_SHARE_WAIT = ".helper_share_wait.json"
_DROP = ("--sample", "--speed_lora", "--preview", "--save_state", "--keep_last", "--ema", "--adaptive", "--metadata",
         "--log_per_image", "--per_image", "--auto_recaption", "--warmup_look", "--trigger", "--recaption",
         "--captioner", "--resume", "--text_encoder", "--save_every", "--save_until", "--share_worker",
         "--gradient_checkpointing")
_sw = {}               # tag -> {proc, dir, log}: this computer's share workers


def _cache_key(tag, i, name):
    """The name a cache file travels under (safe for put), from its own name."""
    return f"sh{tag}-{i}-{hashlib.sha1(name.encode('utf-8')).hexdigest()[:20]}.bin"


# ---------------------------------------------------------------------------------------------------------------------
# The share's cache. By default the helper makes it itself from the training folder (a few hundred MB, where the cache
# can be GBs: Krea 2's text cache is 31.5 MB a caption), with the lead's own VAE and text encoder - the same files by
# hash, its own or the family's download - so both halves train on the same numbers. When it can't (an encoder it
# can't download, a cache step with a file of the lead's own), or "Send this computer's cache" is ticked, the lead's
# cache goes over, its text cache trimmed to the real tokens where the family drops the padding.
# ---------------------------------------------------------------------------------------------------------------------
CACHE_PLAN = "cache.json"         # <output_dir>/.helper_share/cache.json: the run's cache steps, from its launch
_TRIM_META = "fizgig_trim"
_hash_lock = threading.Lock()
_prep = {}                         # share tag -> the lead's side of a run's cache (hashes, trimmed copies)
_probes = {}                       # (family, hashes) -> this computer's answer: can it make the cache itself
_made = {}                         # share tag -> this computer making a lead's cache: {state, stage, done, total}


def share_cache_plan(out_dir, run, cmds):
    """At a run's launch: its cache steps (families/cache.py's arguments), for a helper to make the same cache."""
    stages = [{"stage": c[c.index("--stage") + 1], "args": [str(a) for a in c[2:]]} for c in cmds
              if "--stage" in c and "--model" in c]
    try:
        _write(os.path.join(out_dir, SHARE_DIR, CACHE_PLAN), {"output_name": run, "stages": stages})
    except OSError:
        pass


def _send_cache():
    """This computer's tick: send its cache rather than have the helper make one (a fast link)."""
    return bool(_srv().load_prefs().get("helper_share_send_cache"))


def _sha(path):
    """A file's SHA-256, remembered by path, size and time (a text encoder is GBs: hashed once)."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    store = _dir("hashes.json")
    with _hash_lock:
        known = _read(store, {}) or {}
    if key in known:
        return known[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    with _hash_lock:
        known = _read(store, {}) or {}
        known[key] = h.hexdigest()
        _write(store, known)
    return known[key]


def _hub_sha(repo, path):
    """The SHA-256 of a file on Hugging Face (its LFS etag), or "" if it can't be had."""
    if not (repo and path):
        return ""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        e = (get_hf_file_metadata(hf_hub_url(repo, path), timeout=20).etag or "").strip('"').lower()
        return e if re.fullmatch(r"[0-9a-f]{64}", e) else ""
    except Exception:
        return ""


def _trim(src, dst):
    """A text cache file with only the tokens its masks mark, the original lengths kept to pad it back. False for a
    file with no mask (a latent)."""
    from safetensors import safe_open
    with safe_open(src, "pt") as f:
        keys = list(f.keys())
        masks = [k for k in keys if k.endswith("__attention_mask")]
        if not masks:
            return False
        sd, meta = {k: f.get_tensor(k) for k in keys}, dict(f.metadata() or {})
    from safetensors.torch import save_file
    seqs = {}
    for mk in masks:
        base, m = mk[:-len("__attention_mask")], sd[mk].bool()
        seqs[base] = int(m.shape[0])
        for k in keys:
            if k.startswith(base + "__") and sd[k].dim() >= 1 and sd[k].shape[0] == m.shape[0]:
                sd[k] = sd[k][m].contiguous()
    meta[_TRIM_META] = json.dumps(seqs)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_file(sd, dst + ".tmp", meta)
    os.replace(dst + ".tmp", dst)
    return True


def _untrim(src, dst):
    """A trimmed text cache file padded back to its length: the real tokens first, then masked-off zeros (the
    trainer takes the marked tokens in order, so it trains on exactly what the lead does). Anything else is moved."""
    from safetensors import safe_open
    with safe_open(src, "pt") as f:
        meta = dict(f.metadata() or {})
        seqs = json.loads(meta.pop(_TRIM_META)) if _TRIM_META in meta else None
        sd = {k: f.get_tensor(k) for k in f.keys()} if seqs else None
    if not seqs:
        os.replace(src, dst)
        return
    import torch
    from safetensors.torch import save_file
    for base, n in seqs.items():
        mk = base + "__attention_mask"
        have = int(sd[mk].shape[0])
        for k in [k for k in sd if k.startswith(base + "__") and sd[k].dim() >= 1 and sd[k].shape[0] == have]:
            pad = torch.zeros((n - have, *sd[k].shape[1:]), dtype=sd[k].dtype)
            sd[k] = torch.cat([sd[k] if k != mk else torch.ones(have, dtype=torch.bool), pad])
    save_file(sd, dst + ".tmp", meta)
    os.replace(dst + ".tmp", dst)
    os.remove(src)


def _cache_prep(out_dir, run, setup):
    """The lead's side, worked out once a run in the background: its cache steps and their models' hashes, and the
    trimmed text cache to send. None while it works."""
    tag = _share_tag(out_dir, run, setup)
    p = _prep.get(tag)
    if p is None:
        p = _prep[tag] = {"state": "working"}
        threading.Thread(target=_cache_prep_work, args=(p, out_dir, run, setup), daemon=True).start()
    return None if p["state"] == "working" else p


def _cache_prep_work(p, out_dir, run, setup):
    why, stages, shas, trim, bundles = "", [], {}, "", {}
    try:
        plan = _read(os.path.join(out_dir, SHARE_DIR, CACHE_PLAN), {}) or {}
        stages = (plan.get("stages") or []) if plan.get("output_name") == run else []
        if not {"latents", "text"} <= {s["stage"] for s in stages}:
            why = "this run's cache wasn't made at its start"
        for s in stages:
            a = s["args"]
            if any(os.path.isabs(v.partition("=")[2]) for k, v in zip(a, a[1:]) if k == "--aux"):
                why = "its cache step uses a file of this computer's own"
            shas[s["stage"]] = _sha(a[a.index("--model") + 1])
        args = list(setup.get("args") or [])
        if "--dit" in args[:-1]:                  # the model the run trains on: the helper's must be the same file
            shas["dit"] = _sha(args[args.index("--dit") + 1])
        desc = _srv().get_family(str(setup.get("family") or ""))
        if desc is not None and getattr(desc, "text_cache_trim", False):
            trim = os.path.join(out_dir, SHARE_DIR, "trimmed")
            shutil.rmtree(trim, ignore_errors=True)
            for i, cd in enumerate(setup.get("cache_dirs") or []):
                for src in glob.glob(os.path.join(cd, "*.safetensors")):
                    _trim(src, os.path.join(trim, str(i), os.path.basename(src)))
        # each set in one file (sent in pieces like a LoRA): hundreds of small files would cost two round trips each
        for kind, items in (("folder", _share_folder_files(setup)), ("cache", _share_cache_files(setup, {"trim": trim}))):
            path = os.path.join(out_dir, SHARE_DIR, f"{kind}.tar")
            with tarfile.open(path + ".tmp", "w") as tf:
                for i, f in items:
                    tf.add(f, arcname=f"{i}/{os.path.basename(f)}")
            os.replace(path + ".tmp", path)
            bundles[kind] = path
    except Exception as e:
        logger.warning(f"[share] couldn't get the run's cache ready to share: {e}")
        why = why or "its cache couldn't be read"
    p.update(state="done", why=why, stages=stages, shas=shas, trim=trim, bundles=bundles)


def _share_folder_files(setup):
    """The training folder(s) a helper makes the cache from: [(index "m<n>", path)] - pictures and captions."""
    import toml
    exts = _media_exts()
    out = []
    for n, ds in enumerate(toml.loads(setup.get("dataset_config") or "").get("datasets") or []):
        d, cap = ds.get("image_directory") or "", (ds.get("caption_extension") or ".txt").lower()
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts | {cap}:
                out.append((f"m{n}", p))
    return out


def _share_cache_files(setup, prep):
    """The lead's cache to send: [(index, path)], a trimmed copy in place of each text cache file that has one. Only
    the items still in the training folder: the trainer skips a stale one here, and the helper (without the folder)
    would train it, so the two would split different sets."""
    import toml
    norm = lambda p: os.path.normcase(os.path.abspath(p))
    sets = {norm(ds["cache_directory"]): ds for ds in toml.loads(setup.get("dataset_config") or "").get("datasets") or []
            if ds.get("cache_directory")}
    out = []
    for i, cd in enumerate(setup.get("cache_dirs") or []):
        ds = sets.get(norm(cd)) or {}
        d, cap = ds.get("image_directory") or "", (ds.get("caption_extension") or ".txt").lower()
        keys = {os.path.splitext(f)[0] for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() not in (cap, ".json")} if d and os.path.isdir(d) else None
        for p in sorted(glob.glob(os.path.join(cd, "*"))):
            # cache names: <item>_<WxH>_<arch>.safetensors, <item>_<arch>_te.safetensors
            if keys is not None and "_".join(os.path.basename(p).split("_")[:-2]) not in keys:
                continue
            if os.path.isfile(p):
                t = os.path.join(prep["trim"], str(i), os.path.basename(p)) if prep.get("trim") else ""
                out.append((i, t if t and os.path.isfile(t) else p))
        if "--masked_training" in (setup.get("args") or []) and d and os.path.isdir(d):
            # masked training: the pictures' masks go with the cache (the helper has no training folder)
            # (only those this computer uses, so both halves train the same pictures masked)
            from PIL import Image
            from fizgig.dataset.masks import mask_problem
            from fizgig.utils.upright import upright_size
            for pic, m in sorted(folder_masks(d, _media_exts()).items()):
                try:
                    with Image.open(os.path.join(d, pic)) as im:
                        ok = not mask_problem(m, upright_size(im), os.path.join(d, pic))
                except Exception:
                    ok = False
                if ok:
                    out.append((f"k{i}", m))
    return out


def _share_send_list(setup, prep, make):
    """What goes to the helper for a run, as one file: the training folder (it makes the cache) or the cache."""
    path = (prep.get("bundles") or {}).get("folder" if make else "cache")
    if not path:
        raise ValueError("the run's files couldn't be packed to send")
    ctx = _share_context(setup)
    return [("b", path)] + ([("c", ctx)] if ctx else [])


def _share_context(setup):
    """The run's context LoRA (--context_lora_path), which the helper's half trains with too, or ""."""
    a = list(setup.get("args") or [])
    p = a[a.index("--context_lora_path") + 1] if "--context_lora_path" in a[:-1] else ""
    if p and not os.path.isfile(p):
        raise ValueError(f"the run's context LoRA isn't at {p}")
    return p


def _unbundle(tag):
    """A set sent as one file, back into this computer's sent files (sh<tag>-<i>-<name>), as if sent one by one."""
    for bp in glob.glob(_dir("files", f"sh{tag}-b-*")):
        with tarfile.open(bp) as tf:
            for m in tf.getmembers():
                i, _, base = m.name.partition("/")
                if not (m.isfile() and re.fullmatch(r"[mk]?\d+", i) and base and os.path.basename(base) == base
                        and not base.startswith(".")):
                    continue
                with tf.extractfile(m) as src, open(_dir("files", f"sh{tag}-{i}-{base}"), "wb") as dst:
                    shutil.copyfileobj(src, dst)
        os.remove(bp)


_CACHE_WORDS = {"latents": "the pictures", "text": "the captions"}


def _share_how(prep, setup, probe, say):
    """Whether the helper makes the run's cache (True) or this computer sends it, and the line saying so. `probe`
    asks the helper share_probe's question; `say` shows a line while it checks."""
    while True:
        r = probe({"family": setup.get("family"), "shas": prep.get("shas") or {}})
        if not r.get("checking"):
            break
        say("Checking the helper has this run's model files")
        time.sleep(3)
    if r.get("fatal"):
        raise ValueError(f"It can't train this run's share: {r.get('why')}")
    prep["files"] = r.get("files") or {}       # its copies of the run's files (the model always), by hash
    if _send_cache():
        return False, "This computer's cache goes over (Send this computer's cache is ticked)"
    if prep.get("why"):
        return False, f"This computer's cache goes over: {prep['why']}"
    if r.get("ok"):
        te = prep["files"].get("text") or {}
        got = f" (it gets {te['label']} first)" if te.get("label") and not te.get("path") else ""
        return True, f"The helper makes the cache itself, from the training folder{got}"
    return False, f"This computer's cache goes over: {r.get('why') or 'the helper cannot make it'}"


def share_probe(family, shas):
    """Can this computer make a lead's cache itself: {checking} while it hashes, then {ok, fetch} or {ok: False, why}.
    It needs the lead's VAE and text encoder exactly: its own file if the hash matches, or the family's download if
    that's the same file and it has none."""
    key = (str(family), json.dumps(shas or {}, sort_keys=True))
    r = _probes.get(key)
    if r is None:
        r = _probes[key] = {"checking": True}

        def work():
            try:
                r.update(_probe(str(family), shas or {}))
            except Exception as e:
                r.update(ok=False, why=str(e))
            r.pop("checking", None)
        threading.Thread(target=work, daemon=True).start()
    if not r.get("checking"):
        _probes.pop(key, None)         # answered: the next share asks afresh (a file put right since counts)
    return dict(r)


def _builds(desc, role):
    """The builds of a model a computer could fetch: the family's own file (its hash asked of Hugging Face) and, for
    the text encoder, the description's other builds (their hashes recorded). [(label, repo, path, gb, sha)]"""
    f = next((m for m in desc.model_files if m.role == role), None)
    out = [(f.label, f.repo, f.path, f.size_gb, "")] if f is not None and f.repo else []
    if role == "text_encoder":
        out += [(b.label, b.repo, b.path, b.size_gb, b.sha256) for b in getattr(desc, "text_encoder_builds", ())]
    return out


def _probe(family, shas):
    """For each cache step, the file this computer would use: its own when it hashes to the lead's, else the build
    of it the family lists with that hash, from the models folder or to fetch there (Preferences untouched)."""
    s = _srv()
    desc = s.get_family(family)
    if desc is None:
        return {"ok": False, "why": "this Fizgig doesn't know that model"}
    from fizgig import user_data
    prefs, files = s.load_prefs(), {}

    def pick(role, want):
        """(this computer's file for it, or a build to fetch, or None) - by hash."""
        f = next((m for m in desc.model_files if m.role == role), None)
        if f is None:
            return f, None
        own = (desc.model_path(role, lambda k: s.pref_path(prefs, k)) if hasattr(desc, "model_path")
               else s.pref_path(prefs, f.pref_key))
        if own and os.path.isfile(own) and _sha(own) == want:
            return f, {"path": own}
        hit = next((b for b in _builds(desc, role) if (b[4] or _hub_sha(b[1], b[2])) == want), None)
        if hit is None:
            return f, None
        if role == "dit" and not (own and os.path.isfile(own)):
            return f, {"path": ""}     # none here yet: the regular download (Preferences' file) is the run's file
        local = os.path.join(user_data.MODELS_DIR, os.path.basename(hit[2]))
        return f, {"path": local if os.path.isfile(local) and _sha(local) == want else "", "label": hit[0],
                   "repo": hit[1], "file": hit[2], "gb": hit[3], "sha": want}
    if shas.get("dit"):     # the model both halves train on: the same file, or the two halves don't add up
        f, got = pick("dit", shas["dit"])
        if got is None:
            return {"ok": False, "fatal": True,
                    "why": f"its {f.label if f else 'model'} isn't the file this run trains on, and that file isn't "
                           f"one it can download: point its Preferences at the same file"}
        files["dit"] = got
    for stage, role in (("latents", "vae"), ("text", "text_encoder")):
        want = shas.get(stage)
        if not want:
            return {"ok": False, "why": "the run has no cache step for it", "files": files}
        f, got = pick(role, want)
        if got is None:
            return {"ok": False, "files": files, "why": f"this run's {f.label if f else role} isn't one it can "
                                                        f"download (not in {desc.display_name}'s list of builds)"}
        files[stage] = got
    return {"ok": True, "files": files}


_gets = {}                         # repo/path -> this computer fetching a build for a share: {state, error}


def _get_build(spec):
    """A build a share needs, fetched into the models folder: its path once it's there, else a models-style answer
    with how far the download has got."""
    from fizgig import user_data
    local = os.path.join(user_data.MODELS_DIR, os.path.basename(spec["file"]))
    aside = _inside(os.path.join(user_data.MODELS_DIR, ".share_builds"), spec["file"])
    g = _gets.get(spec["file"])
    for p in (local, aside):
        if g is None and p and os.path.isfile(p) and _sha(p) == spec["sha"]:
            return p, None
    if g is None:
        g = _gets[spec["file"]] = {"state": "running", "error": ""}

        def work():
            try:
                from fizgig.utils import hf_public
                stage = os.path.join(user_data.MODELS_DIR, ".share_builds")      # never over a file of yours
                got = hf_public.download(repo_id=spec["repo"], filename=spec["file"], local_dir=stage)
                if _sha(got) != spec["sha"]:
                    os.remove(got)
                    g.update(state="failed", error="the download isn't the file the run used")
                    return
                if not os.path.exists(local):    # a different file of yours by that name stays; this one stays aside
                    os.replace(got, local)
                    got = local
                g.update(state="done", error="", path=got)
            except Exception as e:
                g.update(state="failed", error=str(e))
        threading.Thread(target=work, daemon=True).start()
    if g["state"] == "done":
        _gets.pop(spec["file"], None)
        return g["path"], None
    if g["state"] == "failed":
        _gets.pop(spec["file"], None)
        return None, {"missing": [spec["label"]], "fetching": None, "error": f"Getting {spec['label']} failed: "
                                                                             f"{g['error']}"}
    part = sum(os.path.getsize(os.path.join(r, f)) for d in (".cache", ".share_builds")
               for r, _d, fs in os.walk(os.path.join(user_data.MODELS_DIR, d)) for f in fs if f.endswith(".incomplete"))
    return None, {"missing": [spec["label"]], "error": "",
                  "fetching": {"what": spec["label"], "got": part / 2 ** 30, "total": float(spec.get("gb") or 1)}}


def _make_cache(tag, d, desc, cfg, stages, paths=None):
    """This computer makes a lead's cache from its training folder, with its own copies of the lead's VAE and text
    encoder: the run's own cache steps, run here."""
    import toml
    s = _srv()
    m = _made[tag]
    prefs = s.load_prefs()
    role = lambda r: desc.model_path(r, lambda k: s.pref_path(prefs, k))
    try:
        with open(os.path.join(d, "dataset.toml"), "w", encoding="utf-8") as f:
            toml.dump(cfg, f)
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(s.BASE_DIR, "src"))
        for st in sorted(stages, key=lambda x: x["stage"] != "latents"):
            a = list(st["args"])
            a[a.index("--dataset_config") + 1] = os.path.join(d, "dataset.toml")
            a[a.index("--model") + 1] = (paths or {}).get(st["stage"]) or role("vae" if st["stage"] == "latents"
                                                                             else "text_encoder")
            m.update(stage=st["stage"], done=0)
            logp = os.path.join(d, f"cache_{st['stage']}.log")
            with open(logp, "w", encoding="utf-8") as log:
                p = subprocess.Popen([sys.executable, os.path.join(s.BASE_DIR, desc.cache_script), *a], stdout=log,
                                     stderr=subprocess.STDOUT, cwd=s.BASE_DIR, env=env, **_NO_WINDOW)
                while p.poll() is None:
                    pat = "*_te.safetensors" if st["stage"] == "text" else "*[0-9]x[0-9]*.safetensors"
                    m["done"] = sum(len(glob.glob(os.path.join(ds["cache_directory"], pat))) for ds in cfg["datasets"])
                    time.sleep(2)
            if p.returncode:
                with open(logp, encoding="utf-8", errors="replace") as f:
                    raise ValueError(f"its {st['stage']} cache step failed: {_plain(f.read())}")
        m["state"] = "done"
    except Exception as e:
        m.update(state="failed", error=str(e))


def _pairs(args):
    """A command line as [(flag, [values])], in order (a flag may repeat: --family_option)."""
    out = []
    for a in args:
        if a.startswith("--"):
            out.append([a, []])
        elif out:
            out[-1][1].append(a)
    return out


_share_lock = threading.Lock()     # one share start at a time: two calls at once would make two caches or workers


def share_start(body):
    with _share_lock:
        return _share_start(body)


def _share_start(body):
    """Start this computer's share worker for a lead's run: the run's own command with this computer's model
    paths, no previews, saves or EMA. Its cache is made here from the training folder (body "make": the run's cache
    steps; the folder sent before as files sh<tag>-m<n>-<name>), or was sent (sh<tag>-<i>-<name>). Answers {models}
    while models download, {caching} while the cache is made, then {ok}."""
    import toml
    s = _srv()
    if s._gpu_training():
        raise ValueError("This computer is training: it can't train another computer's share now.")
    tag = str(body.get("tag") or "")
    if not re.match(r"^[a-f0-9]{6,16}$", tag):
        raise ValueError("Bad tag.")
    setup = body.get("setup") or {}
    desc = s.get_family(str(setup.get("family") or ""))
    if desc is None:
        raise ValueError("This Fizgig doesn't know that model: update it.")
    make = (body.get("make") or {}).get("stages")
    paths = {}
    specs = body.get("files") or (body.get("make") or {}).get("files") or {}       # the builds the probe picked
    for stage, spec in specs.items():
        if stage != "dit" and not make:
            continue                       # the cache came over: no VAE or text encoder needed here
        if spec.get("path") and os.path.isfile(spec["path"]):
            paths[stage] = spec["path"]
            continue
        if not spec.get("file"):
            continue                       # Preferences' own file, downloaded the regular way below

        got, wait = _get_build(spec)
        if wait is not None:
            return {"models": wait}
        paths[stage] = got
    # the DiT (and the VAE unless picked above); the text encoder only when making the cache without a picked build
    m = models(desc.key, {"conds": not make or "text" in paths, "turbo": 0, "look": 0}, fetch=True)
    if m["missing"]:
        return {"models": m}
    old = _sw.get(tag)
    if old and old["proc"].poll() is None:
        return {"ok": True, "dir": old["dir"]}
    lead = str(body.get("lead") or "")
    for t, w in list(_sw.items()):        # another share's worker: this lead's earlier one goes, another lead's stays
        if t != tag and w["proc"].poll() is None:
            if w.get("lead") and w.get("lead") != lead:
                raise ValueError(f"It's training a share for {w['lead']} now.")
            share_stop(t)
    for t, m in list(_made.items()):      # another lead's cache, still being made here (no worker yet)
        if t != tag and m.get("state") == "working" and m.get("lead") and m.get("lead") != lead:
            raise ValueError(f"It's making the cache for {m['lead']}'s share now.")
    _share_tidy(tag)
    _unbundle(tag)
    d = _dir("share", tag)
    made = _made.get(tag)
    names = body.get("names") or {}
    cache_dirs = list(setup.get("cache_dirs") or [])
    cfg = toml.loads(setup.get("dataset_config") or "")

    def take(prefix, into, fix=os.replace):
        """The files sent as sh<tag>-<prefix>-*, under their own names, into a folder here."""
        os.makedirs(into, exist_ok=True)
        for p in glob.glob(_dir("files", f"sh{tag}-{prefix}-*")):
            real = names.get(os.path.basename(p)) or os.path.basename(p)[len(f"sh{tag}-{prefix}-"):]
            if os.path.basename(real) != real or real.startswith("."):
                continue
            fix(p, os.path.join(into, real))
    if make and made is None:           # the training folder came: make the cache here, as the lead's steps did
        shutil.rmtree(d, ignore_errors=True)
        for n, ds in enumerate(cfg.get("datasets") or []):
            take(f"m{n}", os.path.join(d, f"images{n}"))
            ds["image_directory"] = os.path.join(d, f"images{n}").replace("\\", "/")
            ds["cache_directory"] = os.path.join(d, f"cache{n}").replace("\\", "/")
            os.makedirs(ds["cache_directory"], exist_ok=True)
        made = _made[tag] = {"state": "working", "stage": "latents", "done": 0, "cfg": cfg, "lead": lead,
                             "total": sum(len(glob.glob(os.path.join(ds["image_directory"], "*"))) for ds in
                                          cfg.get("datasets") or []) // 2 or 1}
        threading.Thread(target=_make_cache, args=(tag, d, desc, cfg, make, paths), daemon=True).start()
    if made is not None:
        if made["state"] == "working":
            return {"caching": {k: made.get(k) for k in ("stage", "done", "total")}}
        if made["state"] == "failed":
            _made.pop(tag, None)
            raise ValueError(f"Making the cache here failed: {_plain(made.get('error', ''))}")
        cfg = made["cfg"]
    else:
        # the dataset from the lead's cache, sent over: no image folder, so the stale-cache check stands aside
        shutil.rmtree(d, ignore_errors=True)
        for ds in cfg.get("datasets") or []:
            cd = ds.get("cache_directory")
            i = cache_dirs.index(cd) if cd in cache_dirs else 0
            take(i, os.path.join(d, f"cache{i}"), _untrim)
            ds["cache_directory"] = os.path.join(d, f"cache{i}").replace("\\", "/")
            ds["image_directory"] = os.path.join(d, "no_images").replace("\\", "/")
            if glob.glob(_dir("files", f"sh{tag}-k{i}-*")):        # masked training: the masks came with the cache
                take(f"k{i}", os.path.join(d, f"masks{i}"))
                ds["mask_directory"] = os.path.join(d, f"masks{i}").replace("\\", "/")
    jobs = os.path.join(d, "jobs")
    shutil.rmtree(jobs, ignore_errors=True)
    os.makedirs(jobs)
    with open(os.path.join(d, "dataset.toml"), "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    prefs = s.load_prefs()
    role = lambda r: desc.model_path(r, lambda k: s.pref_path(prefs, k))
    args = []
    for flag, vals in _pairs(setup.get("args") or []):
        if flag.startswith(_DROP):
            continue
        if flag == "--dit":
            vals = [paths.get("dit") or role("dit")]
        elif flag == "--precision" and setup.get("precision"):
            vals = [setup["precision"]]     # the lead's pick (Auto decided there): the same base numbers
        elif flag == "--blocks_to_swap":
            vals = ["-1"]                   # the fewest swapped blocks that fit this card
        elif flag == "--vae":
            vals = [role("vae")]
        elif flag == "--dataset_config":
            vals = [os.path.join(d, "dataset.toml")]
        elif flag == "--output_dir":
            vals = [os.path.join(d, "out")]
        elif flag == "--training_adapter":
            vals = [role("training_adapter")]
        elif flag == "--context_lora_path":       # sent with the run's files (sh<tag>-c-*), kept in the share folder
            ctx = glob.glob(_dir("files", f"sh{tag}-c-*"))
            if ctx:
                os.replace(ctx[0], os.path.join(d, "context.safetensors"))
            if not os.path.isfile(os.path.join(d, "context.safetensors")):
                raise ValueError("The run's context LoRA didn't arrive: start the share again.")
            vals = [os.path.join(d, "context.safetensors")]
        args += [flag, *vals]
    if "--blocks_to_swap" not in args:
        args += ["--blocks_to_swap", "-1"]
    if setup.get("precision") and "--precision" not in args:
        args += ["--precision", setup["precision"]]
    args += ["--gradient_checkpointing", "on", "--share_worker", jobs]   # as the lead trains when it shares
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(s.BASE_DIR, "src"))
    log = open(os.path.join(d, "worker.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "fizgig.families.train", *args], stdout=log, stderr=subprocess.STDOUT,
                            cwd=s.BASE_DIR, env=env, **_NO_WINDOW)
    _sw[tag] = {"proc": proc, "dir": d, "log": log, "lead": lead}
    logger.info(f"[share] worker for {setup.get('output_name')} started")
    return {"ok": True, "dir": d}


def share_job(body):
    """One epoch's share: the start LoRA (a put file), the share, the seed and the LR, into the worker's queue."""
    w = _sw.get(str(body.get("tag") or ""))
    if not w or w["proc"].poll() is not None:
        raise ValueError("No share worker is running here for that run.")
    ep = int(body["epoch"])
    job = {"epoch": ep, "epochs": [int(e) for e in body.get("epochs") or [ep]], "frac": float(body["frac"]),
           "seed": body.get("seed"), "lrs": body.get("lrs") or [], "sched": body.get("sched"),
           "start": _dir("files", body["start"]), "out": os.path.join(w["dir"], f"{ep:06d}.theirs.safetensors")}
    _write(os.path.join(w["dir"], "jobs", f"{ep:06d}.job.json"), job)
    return {"ok": True}


def share_cancel(tag, epoch):
    """A round the lead stopped waiting for (too slow): dropped from the worker's queue, or cut short if it's
    training, so the next round doesn't wait behind it."""
    w = _sw.get(str(tag or ""))
    if w:
        jobs = os.path.join(w["dir"], "jobs")
        try:
            os.remove(os.path.join(jobs, f"{int(epoch):06d}.job.json"))
        except OSError:
            pass
        open(os.path.join(jobs, f"{int(epoch):06d}.cancel"), "w").close()
    return {"ok": True}


def share_state(tag, epoch=0):
    """The worker: starting / ready / an epoch's result."""
    w = _sw.get(tag)
    if not w:
        return {"state": "gone"}
    alive = w["proc"].poll() is None
    if epoch:
        done = _read(os.path.join(w["dir"], "jobs", f"{int(epoch):06d}.done.json"))
        if done is not None:
            return {"state": "done", **done}
        if not alive:
            return {"state": "failed", "error": _tail(os.path.join(w["dir"], "worker.log"))}
        return {"state": "working"}
    if not alive:
        return {"state": "failed", "error": _tail(os.path.join(w["dir"], "worker.log"))}
    return {"state": "ready" if os.path.exists(os.path.join(w["dir"], "jobs", "ready.json")) else "loading"}


def _tail(path, n=600):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def share_file(tag, epoch):
    w = _sw.get(tag)
    if not w:
        raise ValueError("No such share.")
    return os.path.join(w["dir"], f"{int(epoch):06d}.theirs.safetensors")


def _share_tidy(keep):
    """An earlier share's sent files and folders, gone (a share that restarted left them): all but `keep`'s and
    those of a worker still running."""
    live = ({keep} | {t for t, w in _sw.items() if w["proc"].poll() is None}
            | {t for t, m in _made.items() if m.get("state") == "working"})
    for p in glob.glob(_dir("files", "sh*")):
        m = re.match(r"sh([a-f0-9]{6,16})-", os.path.basename(p))
        if m and m.group(1) not in live:
            try:
                os.remove(p)
            except OSError:
                pass
    for d in glob.glob(_dir("share", "*")):
        if os.path.basename(d) not in live:
            shutil.rmtree(d, ignore_errors=True)
            _made.pop(os.path.basename(d), None)


def share_stop_all():
    """This computer's own run starts: its share workers stop (their leads carry on alone)."""
    for t in list(_sw):
        share_stop(t)


def share_stop(tag):
    w = _sw.pop(tag, None)
    _made.pop(tag, None)              # its cache lived in the folder removed below: a restart makes it again
    if not w:
        shutil.rmtree(_dir("share", tag), ignore_errors=True)
    if w:
        open(os.path.join(w["dir"], "jobs", "stop"), "w").close()
        try:
            w["proc"].wait(30)
        except subprocess.TimeoutExpired:
            w["proc"].kill()
        try:                       # the worker's log is kept (the last few), for when a share went wrong
            w["log"].close()
            os.makedirs(_dir("share_logs"), exist_ok=True)
            shutil.copyfile(os.path.join(w["dir"], "worker.log"), _dir("share_logs", f"{tag}.log"))
            for old in sorted(glob.glob(_dir("share_logs", "*.log")), key=os.path.getmtime)[:-5]:
                os.remove(old)
        except OSError:
            pass
        shutil.rmtree(w["dir"], ignore_errors=True)
    return {"ok": True}


# ---- the lead's side of the share
_share = {"thread": None, "run": None, "trying": None, "fail": None, "wait": None, "wait_thread": None}
_wait_lock = threading.Lock()


def _share_wait():
    """Start together (Helpers tab, on unless unticked): a run holds its first epoch until its helper is ready, so
    both sides train from the first epoch."""
    v = _srv().load_prefs().get("helper_share_wait")
    return True if v is None else bool(v)


def _wait_set(out_dir=None, run="", launch="", h=None):
    """The run's hold while its share gets ready, kept fresh by _wait_loop (the trainer lets go of a stale one);
    no out_dir: let go."""
    with _wait_lock:
        old = _share.get("wait")
        _share["wait"] = ({"out_dir": out_dir, "run": run, "launch": launch, "hid": (h or {}).get("id"),
                           "name": (h or {}).get("name") or "the helper"}
                          if out_dir and _share_wait() else None)
        if old and (_share["wait"] is None or old["out_dir"] != out_dir):
            try:
                os.remove(os.path.join(old["out_dir"], HELPER_SHARE_WAIT))
            except OSError:
                pass
        if _share["wait"] and not old:
            _wait_write(_share["wait"])
        if _share["wait"] and (_share["wait_thread"] is None or not _share["wait_thread"].is_alive()):
            _share["wait_thread"] = threading.Thread(target=_wait_loop, daemon=True, name="helpers-share-wait")
            _share["wait_thread"].start()


def _wait_loop():
    while True:
        time.sleep(3)
        with _wait_lock:
            w = _share.get("wait")
            if w is None:
                return
            p = os.path.join(w["out_dir"], HELPER_SHARE_WAIT)
            if not _share_wait():                  # unticked while it waits: the run starts on its own
                _share["wait"] = None
                try:
                    os.remove(p)
                except OSError:
                    pass
                return
            _wait_write(w)


def _wait_write(w):
    lv = _live.get(w["hid"]) or {}
    try:
        _write(os.path.join(w["out_dir"], HELPER_SHARE_WAIT), {
            "run": w["run"], "launch": w["launch"], "at": time.time(), "name": w["name"],
            "text": (lv.get("activity") or {}).get("text") or ""})
    except OSError:
        pass


def _share_helper(hs):
    """The helper training this run's share: the first with the share ticked that's there and can (one at a time)."""
    for h in hs:
        lv = _live.get(h["id"]) if h.get("share") else None
        if lv and lv.get("state") == "ready" and lv.get("build") == "same":
            return h, lv
    return None, None


def ensure_share():
    if _share["thread"] is None or not _share["thread"].is_alive():
        _share["thread"] = threading.Thread(target=_share_loop, daemon=True, name="helpers-share")
        _share["thread"].start()


def _setup_guess(out_dir, run, setup, link):
    """A very rough window for getting a share ready, before its first round: the training folder over the link,
    and about 0.1-0.5 s a picture for the helper to make its cache, plus loading the model. Display only."""
    try:
        p = _prep.get(_share_tag(out_dir, run, setup)) or {}
        folder = (p.get("bundles") or {}).get("folder")
        mb = os.path.getsize(folder) / 1e6 if folder and os.path.isfile(folder) else 0
        n = sum(1 for _i, f in _share_folder_files(setup) if not f.lower().endswith(".txt")
                and not is_mask_file(f)) if mb else 0
        if not (mb and n and link.get("up")):
            return ""
        send = mb / link["up"]
        lo, hi = send + 0.1 * n + 20, 1.5 * send + 0.5 * n + 60
        span = lambda v: f"{v / 60:.0f} min" if v >= 90 else f"{v:.0f} s"
        rng = f"{lo / 60:.0f} to {hi / 60:.0f} min" if lo >= 90 else f"{span(lo)} to {span(hi)}"
        return f". Getting it ready first: roughly {rng} (a rough guess)"
    except Exception:
        return ""


def _first_frac(their_gpu):
    """The first round's share: by the two GPUs' speeds when both are known, else half."""
    s = _srv()
    with s._lock:
        my_gpu = s._stats.get("device_name")
    mine, theirs = _gpu_speed(my_gpu), _gpu_speed(their_gpu)
    return max(0.1, min(0.8, theirs / (mine + theirs))) if mine and theirs else 0.5


def _share_every():
    """Combine every N epochs (Helpers tab), 1-10; the trainer also combines before any epoch that previews or
    saves."""
    try:
        return max(1, min(10, int(_srv().load_prefs().get("helper_share_every") or 3)))
    except (TypeError, ValueError):
        return 1


def _share_busy(out_dir, name, frac, ep, _last=[0.0]):
    """The flag kept fresh (and busy with round `ep`) during a transfer: at most every 2 s, and never the
    transfer's undoing."""
    if time.time() - _last[0] < 2.0:
        return
    _last[0] = time.time()
    try:
        _share_flag(out_dir, name, frac, busy=ep)
    except OSError:
        pass


def _share_flag(out_dir, name="", frac=0.0, every=None, busy=0):
    p = os.path.join(out_dir, HELPER_SHARE) if out_dir else ""
    if not p:
        return
    if name:
        _write(p, {"on": True, "name": name, "frac": round(frac, 3), "at": time.time(), "run": _run_now()[1],
                   "every": every or _share_every(), "busy": int(busy or 0)})
    else:
        try:
            os.remove(p)
        except OSError:
            pass


def _plain(e, n=200):
    """An error's last line, short: what went wrong, not the traceback."""
    lines = [x.strip() for x in str(e).splitlines() if x.strip()]
    return (lines[-1] if lines else type(e).__name__)[:n]


def _previews_off(out_dir, run):
    """The line saying why the run going now keeps its previews on this GPU (the trainer's note), or ""."""
    n = _read(os.path.join(out_dir, ".helper_previews_off.json"), {}) if out_dir else {}
    if not (n and n.get("why") and n.get("run") == run):
        return ""
    return f"This run's previews render on this GPU: helpers don't render {n['why']} yet"


def _off_line(setup):
    return f"This run trains on this GPU only: the training share is off for runs with {setup['off']}"


def _share_loop():
    while True:
        try:
            _share_tick()
        except PermissionError:
            pass
        except Exception as e:
            logger.exception("[share] lead")
            k = _share.get("trying")
            if k and _share["run"] is None:          # the share's start failed: counted per run and helper
                f = _share["fail"] if (_share.get("fail") or {}).get("key") == k else {"key": k, "n": 0}
                f.update(n=f["n"] + 1, error=_plain(e))
                _share["fail"] = f
                if f["n"] >= 3:              # given up: the run starts alone rather than waiting on it
                    _wait_set()
                lv = _live.get(k[1])
                if lv is not None:
                    lv["activity"] = {"kind": "text", "text": f"Its share couldn't start: {f['error']}" + (
                        " - trying again" if f["n"] < 3 else ". Switch Training share off and on to try again.")}
            time.sleep(5)
        time.sleep(1.0)


def _share_tick():
    out_dir, run = _run_now()
    st = _share["run"]
    inbound = _share_inbound()
    if inbound is not None:                # a computer that connected in trains the share: it does the sending
        _wait_set()
        if out_dir:
            _share_flag(out_dir, inbound["name"], inbound.get("frac", 0.5), inbound.get("every"),
                        (inbound.get("activity") or {}).get("epoch") if inbound.get("activity") else 0)
        return
    hs = _helpers()
    h, lv = _share_helper(hs)
    setup = _setup(out_dir, run)
    # the run ended, a new launch began (a queued run in the same folder, or a re-run) or the helper left
    if st and (st["out_dir"] != out_dir or st.get("launch") != (setup or {}).get("launch") or not h
               or h["id"] != st["hid"]):
        _share_flag(st["out_dir"])
        try:
            Client(st["url"], st["cookie"]).call("/api/helper/share_stop", {"tag": st["tag"]})
        except Exception:
            pass
        if lv is not None:
            lv["activity"] = None
        _share["run"] = st = None
    if not (out_dir and h and setup):
        _wait_set()                        # no helper to wait for (none ticked, or it left): the run starts alone
        return
    if setup.get("off"):                   # this run keeps to one GPU: no share to start
        _wait_set()
        lv["activity"] = {"kind": "text", "text": _off_line(setup)}
        return
    key = (setup.get("launch"), h["id"])
    fl = _share.get("fail") or {}
    if st is None and fl.get("key") == key and fl.get("n", 0) >= 3:
        _wait_set()
        lv["activity"] = {"kind": "text", "text": f"Its share couldn't start: {fl['error']}. Switch Training share "
                                                  "off and on to try again."}
        return
    if st is None or not st["ready"]:      # Start together: the run's first epoch waits for this
        _wait_set(out_dir, run, setup.get("launch"), h)
    if st is None:
        _share["trying"] = key
    c = Client(h["url"], h.get("cookie", ""))
    if st is None:                         # the start: the training folder (or the cache) over, then the worker
        for k in ("share_stat", "share_plan", "share_how"):
            lv.pop(k, None)
        prep = _cache_prep(out_dir, run, setup)
        if prep is None:
            lv["activity"] = {"kind": "share", "text": "Getting the run ready to share", "progress": 0}
            return
        lv["activity"] = {"kind": "share", "text": "Measuring the link", "progress": 0}
        _fresh_link(h["url"], h.get("cookie", ""))
        make, how = _share_how(prep, setup, lambda s: c.call("/api/helper/share_probe", s, timeout=60),
                               lambda t: lv.__setitem__("activity", {"kind": "share", "text": t, "progress": 0}))
        lv["share_how"] = how
        tag = hashlib.sha1(f"{socket.gethostname()}|{out_dir}|{run}|{time.time()}".encode()).hexdigest()[:10]
        files = _share_send_list(setup, prep, make)
        total = sum(os.path.getsize(p) for _i, p in files) or 1
        sent, names = 0, {}
        what = "the training folder" if make else "the run's cache"
        for i, p in files:
            size = os.path.getsize(p)
            show = lambda f: lv.__setitem__("activity", {"kind": "share", "progress": .2 * (sent + f * size) / total,
                                                        "text": f"Sending {what} ({int((sent + f * size) * 100 / total)}%)"})
            show(0)
            key = _cache_key(tag, i, os.path.basename(p))
            names[key] = os.path.basename(p)
            _send_file(c, p, key, show)
            sent += size
        while True:
            r = c.call("/api/helper/share_start", {"tag": tag, "setup": setup, "names": names,
                                                   "lead": socket.gethostname(),
                                                   "files": prep.get("files"),
                                                   "make": {"stages": prep["stages"], "files": prep.get("files")}
                                                   if make else None}, timeout=120)
            if r.get("caching"):
                k = r["caching"]
                lv["activity"] = {"kind": "share", "text": f"Making the cache there: {_CACHE_WORDS.get(k.get('stage'), '')}"
                                  f" ({k.get('done') or 0} of {k.get('total') or 0})",
                                  "progress": .2 + .1 * (k.get("done") or 0) / max(1, k.get("total") or 1)}
                time.sleep(3)
                continue
            if not r.get("models"):
                break
            f = (r["models"] or {}).get("fetching") or {}
            if r["models"].get("error"):
                raise ValueError(r["models"]["error"])
            lv["activity"] = {"kind": "models", "what": f.get("what"), "got": float(f.get("got") or 0),
                              "total": float(f.get("total") or 1) or 1, "for": f"{run}'s training share"}
            time.sleep(3)
        st = _share["run"] = {"out_dir": out_dir, "hid": h["id"], "url": h["url"], "cookie": h.get("cookie", ""),
                              "tag": tag, "frac": _first_frac(lv.get("gpu")), "sent": set(), "ready": False,
                              "launch": setup.get("launch")}
    if not st["ready"]:
        s = c.call(f"/api/helper/share_state?tag={st['tag']}")
        if s["state"] in ("failed", "gone"):       # its worker stopped while loading: a start that failed
            _share["run"] = None
            try:
                c.call("/api/helper/share_stop", {"tag": st["tag"]})
            except Exception:
                pass
            raise ValueError(f"its worker stopped ({_plain(s.get('error') or 'no reason given')})")
        if s["state"] != "ready":
            lv["activity"] = {"kind": "share", "text": "Loading the model for its share", "progress": .25}
            return
        st["ready"] = True
        _share["trying"], _share["fail"] = None, None
    _share_flag(out_dir, h.get("name"), st["frac"])
    _wait_set()                            # the flag is up: the trainer's first epoch goes to both
    lv["share_frac"] = st["frac"]
    # an epoch handed over and not yet sent: its start LoRA up, the job, then its result back
    for jp in sorted(glob.glob(os.path.join(out_dir, SHARE_DIR, "*", "job.json"))):
        ed = os.path.dirname(jp)
        if ed in st["sent"] or any(os.path.exists(os.path.join(ed, n)) for n in (
                "theirs.safetensors", "closed.json", "failed.json")):
            continue                       # back already, or a round the lead has closed: nothing to send
        job = _read(jp)
        if not job or job.get("launch") != st.get("launch"):
            continue
        st["sent"].add(ed)
        ep = int(job["epoch"])
        try:
            lv["activity"] = {"kind": "share", "epoch": ep, "text": "Sending this round's LoRA", "progress": .3}
            name = f"sh{st['tag']}-e{ep:06d}.safetensors"
            t_round = time.time()
            _send_file(c, os.path.join(ed, "start.safetensors"), name,
                       lambda f: _share_busy(out_dir, h.get("name"), st["frac"], ep))
            c.call("/api/helper/share_job", {"tag": st["tag"], "start": name, **job})
            t0 = time.time()
            while True:
                s = _again(lambda: c.call(f"/api/helper/share_state?tag={st['tag']}&epoch={ep}"))
                if s["state"] == "done":
                    break
                if s["state"] in ("failed", "gone"):
                    raise ValueError(_plain(s.get("error") or "its share stopped"))
                if os.path.exists(os.path.join(ed, "closed.json")) or _run_now()[0] != out_dir:
                    try:
                        c.call("/api/helper/share_cancel", {"tag": st["tag"], "epoch": ep}, timeout=20)
                    except Exception:
                        pass               # an older helper, or gone: its worker finishes the round unused
                    raise ValueError("this computer stopped waiting for it (too slow for this round)")
                _share_flag(out_dir, h.get("name"), st["frac"], busy=ep)     # fresh while the round trains
                lv["activity"] = {"kind": "share", "epoch": ep, "progress": min(.9, .3 + (time.time() - t0) / 120)}
                time.sleep(1.0)
            tmp = os.path.join(ed, "theirs.safetensors.part")
            _again(lambda: c.download(f"/api/helper/share_file?tag={st['tag']}&epoch={ep}", tmp,
                                      lambda n: _share_busy(out_dir, h.get("name"), st["frac"], ep)))
            t_back = time.time()               # the round, to its result here (not this GPU's own half after it)
            if os.path.exists(os.path.join(ed, "closed.json")) or (_read(jp) or {}).get("launch") != st.get("launch"):
                os.remove(tmp)             # the lead stopped waiting for it, or a new launch took the folder: too late
            else:
                os.replace(tmp, os.path.join(ed, "theirs.safetensors"))
            c.call("/api/helper/drop", {"job": "", "lora": name})
            lead = None
            for k in range(240):
                lead = _read(os.path.join(ed, "lead.json"))
                if lead:
                    break
                if k % 20 == 0:            # still sharing, while this computer finishes its half
                    _share_flag(out_dir, h.get("name"), st["frac"])
                time.sleep(0.5)
            if lead and s.get("steps") and s.get("seconds") and lead.get("steps") and lead.get("seconds"):
                # the share follows the two speeds and the transfers, smoothed
                wall = t_back - t_round
                st["frac"], lv["share_plan"] = _share_plan(
                    s["steps"], s["seconds"], lead["steps"], lead["seconds"], wall,
                    len(job.get("epochs") or [ep]), _share_every(), st["frac"])
                lk, lb = _link(h["url"]), setup.get("lora_bytes") or 0
                if lk.get("up") and lk.get("down") and lb:     # the round's cost beyond its bytes, for the next estimate
                    _set_link(h["url"], overhead=round(max(0.0, wall - s["seconds"] - lb / 1e6 / lk["up"]
                                                           - lb / 1e6 / lk["down"]), 1))
                lv["share_stat"] = (f"Epoch {ep}: {s['steps']} images in {s['seconds']:.0f} s there, "
                                    f"{lead['steps']} here in {lead['seconds']:.0f} s; its share now "
                                    f"{st['frac'] * 100:.0f}%")
            lv["activity"] = None
        except Exception as e:
            _write(os.path.join(ed, "failed.json"), {"error": str(e)})
            lv["activity"] = {"kind": "text", "text": f"Epoch {ep}'s share failed ({_plain(e)}): that epoch kept "
                                                      "this computer's images only"}
            try:
                ws = c.call(f"/api/helper/share_state?tag={st['tag']}", timeout=20)["state"]
            except Exception:
                ws = "gone"
            if ws in ("failed", "gone"):   # its worker is gone (a restart there, a crash): no more epochs split
                _share_flag(out_dir)       # until it's set up again, which counts as a start (three tries)
                _share["run"] = None
                _share["trying"] = (st.get("launch"), st["hid"])
            raise


# ---- training share, pulled: a computer this one can't reach (home, for a run on a pod) trains its share by
# connecting in - it asks for the run's setup and cache, then each epoch's start, and hands its LoRA back
_share_in = {}           # helper name -> {name, gpu, at, ready, tag, activity, stat}


# ---------------------------------------------------------------------------------------------------------------------
# The link to a helper: measured (a few pieces each way, as the share's own transfers go) or typed in, kept per
# address. With the run's speed, the LoRA's size and the GPUs, the share tile says before its first round whether
# the share should pay; after it, the measured verdict takes over.
# ---------------------------------------------------------------------------------------------------------------------
_SPEED_PIECES = 2                  # 8 MB pieces each way: 16 MB, a few seconds on a home upload
_ROUND_OVERHEAD = 20.0             # a round's fixed cost before one's been measured (load, hand-off, polling), s


def speed_piece(n=PIECE):
    """Bytes for a link test (the other side times how long they take to come)."""
    return os.urandom(max(1, min(int(n), PIECE)))


def _links():
    return _read(_dir("links.json"), {}) or {}


def _link(url):
    return _links().get((url or "").rstrip("/"), {})


def _set_link(url, **kw):
    rows = _links()
    key = (url or "").rstrip("/")
    rows[key] = {**rows.get(key, {}), **kw}
    _write(_dir("links.json"), rows)
    return rows[key]


def measure_link(url, cookie=""):
    """This computer's link to another Fizgig, in MB/s each way: pieces sent and fetched as a share sends them."""
    c = Client(url, cookie)
    try:
        t = time.time()
        for _ in range(_SPEED_PIECES):
            c.call("/api/helper/speed", raw=os.urandom(PIECE), timeout=60)
        up = _SPEED_PIECES * PIECE / 1e6 / max(1e-3, time.time() - t)
        t = time.time()
        got = sum(len(c.call(f"/api/helper/speed?n={PIECE}", timeout=60)) for _ in range(_SPEED_PIECES))
        down = got / 1e6 / max(1e-3, time.time() - t)
        return _set_link(url, up=round(up, 3), down=round(down, 3), at=time.time(), how="measured", error="")
    except Exception as e:
        logger.info(f"[helpers] measuring the link to {url}: {e}")
        return _set_link(url, up=None, down=None, at=time.time(), how="",
                         error="The link test didn't finish: type your speeds in, or Measure again")


def set_link(url, up_mbps, down_mbps):
    """Speeds typed in, in Mb/s as broadband is sold (8 bits a byte); blank goes back to measuring."""
    try:
        up, down = float(up_mbps), float(down_mbps)
    except (TypeError, ValueError):
        raise ValueError("Type both speeds in Mb/s, or leave both empty to measure again.")
    if up <= 0 or down <= 0:
        raise ValueError("Speeds must be above 0 Mb/s.")
    return _set_link(url, up=round(up / 8, 2), down=round(down / 8, 2), at=time.time(), how="typed", error="")


def _link_mbps(link):
    """A link's speeds in Mb/s, to fill the boxes for typing them."""
    return {k: _mb(link[k] * 8) for k in ("up", "down") if link.get(k)}


def _mb(v):
    """Mb/s as people read them: whole numbers, one decimal under 10."""
    return round(v, 1) if v < 10 else round(v)


def _link_line(link):
    """The tile's line for a link: up and down in Mb/s, and how they were had."""
    if not link:
        return ""
    if link.get("error") and not link.get("up"):
        return link["error"]
    how = "typed in" if link.get("how") == "typed" else "measured"
    return f"Link: {_mb(link['up'] * 8)} Mb/s up, {_mb(link['down'] * 8)} Mb/s down ({how})"


# home cards next to a 5090, from their tensor throughput (a rough guide, as pods.SPEED is)
_HOME_SPEED = {"RTX 5080": 0.62, "RTX 5070 Ti": 0.5, "RTX 5070": 0.35, "RTX 5060 Ti": 0.25, "RTX 4080 SUPER": 0.58,
               "RTX 4080": 0.55, "RTX 4070 Ti SUPER": 0.45, "RTX 4070 Ti": 0.4, "RTX 4070 SUPER": 0.35,
               "RTX 4070": 0.3, "RTX 4060 Ti": 0.2, "RTX 4060": 0.15, "RTX 3090 Ti": 0.45, "RTX 3080 Ti": 0.37,
               "RTX 3080": 0.33, "RTX 3070 Ti": 0.25, "RTX 3070": 0.23, "RTX 3060 Ti": 0.2, "RTX 3060": 0.17}
# the names torch reports for pod cards, to the table's
_GPU_ALIASES = [(r"H100.*(HBM3|SXM)", "H100 SXM"), (r"H100.*NVL", "H100 NVL"), (r"H100.*PCIe", "H100 PCIe"),
                (r"H200.*NVL", "H200 NVL"), (r"H200", "H200 SXM"), (r"A100.*SXM", "A100 SXM"),
                (r"A100.*PCI", "A100 PCIe"), (r"A100", "A100 SXM")]


def _gpu_speed(name):
    """A GPU's training speed next to an RTX 5090 (pods.SPEED, a rough guide); a laptop card at about 45% of its
    desktop namesake (a 4090 laptop measured at about a third of a 5090). None for a card the table doesn't know."""
    from fizgig.web.pods import SPEED
    SPEED = {**_HOME_SPEED, **SPEED}
    n = re.sub(r"^(NVIDIA )?(GeForce )?", "", str(name or "")).replace(" Laptop GPU", "").strip()
    n = next((k for pat, k in _GPU_ALIASES if re.search(pat, n)), n)
    base = SPEED.get(n) or next((v for k, v in sorted(SPEED.items(), key=lambda kv: -len(kv[0])) if k in n), None)
    return base * (0.45 if "Laptop" in str(name or "") else 1.0) if base else None


def _combine_every(setup, every):
    """How often a run's share really combines: every N epochs, and before any epoch that saves or previews."""
    a, out = setup.get("args") or [], [max(1, int(every or 1))]
    for flag in ("--save_every_n_epochs", "--sample_every_n_epochs"):
        if flag in a:
            try:
                out.append(max(1, int(a[a.index(flag) + 1])))
            except (ValueError, IndexError):
                pass
    return min(out)


def _share_estimate(setup, link, my_gpu, their_gpu, progress, every, overhead=None, lead_up=True):
    """Before a share's first round: whether it should pay, from this run's speed so far (its it/s), the LoRA's size,
    the two GPUs and the link. lead_up: the lead sends the start over its upload (it connected out); else the helper
    fetched it (its download) and sends its change over its upload."""
    try:                                                     # "1.88 it/s", or "2.40 s/it" on a slow card
        v, unit = str((progress or {}).get("speed") or "").split()[:2]
        sps = float(v) if unit.startswith("it") else 1 / float(v)
    except (ValueError, ZeroDivisionError):
        sps = 0.0
    steps = (progress or {}).get("total_steps") or 0
    epochs = (progress or {}).get("total_epochs") or 0
    if int((progress or {}).get("epoch") or 0) < 3 and epochs >= 3:
        # the first two epochs run slow while the GPU warms up (kernels planned, caches filled): a speed read then
        # would promise far more than the share saves
        return "Expected: an estimate from the 3rd epoch, once this run's speed has settled" if sps > 0 else ""
    lb = setup.get("lora_bytes") or 0
    if not (sps > 0 and steps and epochs and lb and link.get("up") and link.get("down")):
        return ""
    mine, theirs = _gpu_speed(my_gpu), _gpu_speed(their_gpu)
    r = theirs / mine if mine and theirs else 1.0
    e = _combine_every(setup, every)
    m = e * steps / epochs                                   # a round's steps, both sides together
    there, back = (link["up"], link["down"]) if lead_up else (link["down"], link["up"])
    xfer = lb / 1e6 / there + lb / 1e6 / back + (overhead if overhead is not None else _ROUND_OVERHEAD)
    half = m / 2
    _f, line = _share_plan(half, half / (sps * r), half, half / sps, half / (sps * r) + xfer, e, every, 0.5)
    if not line:
        return ""
    note = "" if mine and theirs else " (its GPU's speed guessed as this one's)"
    often = f"every {e} epochs" if e > 1 else "every epoch"
    forced = f"; this run saves or previews {often}, so it combines that often" if e < max(1, int(every or 1)) else ""
    return f"Expected before its first round{note}: {line[0].lower()}{line[1:]}{forced}"


def _cookie_for(url):
    """This computer's login at another Fizgig: as its helper, or as one it helps."""
    url = (url or "").rstrip("/")
    h = next((h for h in _helpers() if (h.get("url") or "").rstrip("/") == url), None)
    if h:
        return h.get("cookie", "")
    return next((v for k, v in (_srv().load_prefs().get("helper_out_cookies") or {}).items() if k.rstrip("/") == url), "")


def link(url, up_mbps=None, down_mbps=None):
    """The Helpers tab: speeds typed in, or (none given) measured again."""
    if up_mbps in (None, "") and down_mbps in (None, ""):
        lk = measure_link(url, _cookie_for(url))
    else:
        lk = set_link(url, up_mbps, down_mbps)
    return {**lk, "say": lk.get("error") or _link_line(lk)}


def _fresh_link(url, cookie):
    """The link for a share about to start: typed speeds stand; a measure older than 6 hours is taken again."""
    lk = _link(url)
    if lk.get("how") == "typed" or (lk.get("up") and time.time() - (lk.get("at") or 0) < 6 * 3600):
        return lk
    return measure_link(url, cookie)


def _share_plan(hs, hsec, ls, lsec, wall, epochs, every, frac):
    """Whether the share pays, from a round's measurements: the helper's images and seconds training (hs, hsec), the
    lead's (ls, lsec), the helper's whole round with its transfers (wall), the round's epochs. Returns the share the
    helper should take next (its speed AND the transfers: both sides then finish together) and a verdict line."""
    if not (hs and hsec and ls and lsec):
        return frac, ""
    m, xfer = hs + ls, max(0.0, wall - hsec)            # images in the round; the round's transfers (both ways)
    a, b = m * lsec / ls, m * hsec / hs                 # the round's images on the lead alone / on the helper alone
    best = (a - xfer) / (a + b)                         # both finish together: (1 - f)·a = f·b + xfer
    f = max(0.05, min(0.9, 0.3 * frac + 0.7 * max(0.05, best)))
    per_ep = m / max(1, epochs)
    # the images an epoch from which the helper takes at least 15% (combining every `every` epochs)
    need = lsec / ls - 0.15 * (lsec / ls + hsec / hs)
    from_n = int(xfer / (every * need)) + 1 if need > 0 else 0
    if best >= 0.15:
        return f, (f"Worth it: about {best * a / max(1, epochs):.0f} s saved an epoch "
                   f"(the transfers take {xfer:.0f} s a round)")
    tail = f"; it pays from about {from_n} images an epoch, this run has {per_ep:.0f}" if from_n else ""
    if every < 5 and need > 0 and (5 * per_ep * lsec / ls - xfer) / (5 * per_ep * (lsec / ls + hsec / hs)) >= 0.15:
        tail += ", or combining every 5 epochs"
    return f, (f"Not worth it on this link: a round's transfers take {xfer:.0f} s and "
               + (f"its {epochs} epochs take" if epochs > 1 else "its epoch takes") + f" {a:.0f} s on this GPU alone{tail}")


def _setup(out_dir, run):
    """The run's share setup, once its trainer has written it (one left by an earlier run in the same folder
    names that run, and is skipped)."""
    s = _read(os.path.join(out_dir, SHARE_DIR, "setup.json")) if out_dir else None
    return s if s and s.get("output_name") == run else None


def _share_tag(out_dir, run, setup=None):
    """A run's share, by its folder, name and launch: a re-run of the same name is a new share (its own files,
    cache and worker)."""
    return hashlib.sha1(f"{out_dir}|{run}|{(setup or {}).get('launch', '')}".encode()).hexdigest()[:10]


def share_hello(body):
    """A helper that connected in, for the training share: it says it's there (and whether its worker is ready);
    the answer is the run going now, its setup and its cache files, or nothing."""
    name = str(body.get("name") or "helper")[:60]
    i = _share_in.setdefault(name, {"name": name})
    i.update(at=time.time(), gpu=body.get("gpu"), ready=bool(body.get("ready")), tag=body.get("tag"),
             every=int(body.get("every") or 1))
    if body.get("activity") is not None:
        i["activity"] = body.get("activity") or None
    if body.get("how"):
        i["how"] = str(body["how"])[:200]
    if isinstance(body.get("link"), dict):
        i["link"] = {k: body["link"].get(k) for k in ("up", "down", "how")}
    ensure_lead()
    out_dir, run = _run_now()
    setup = _setup(out_dir, run)
    if not setup:
        return {"run": ""}
    if setup.get("off"):
        return {"run": "", "off": _off_line(setup)}
    if i.get("run_tag") != _share_tag(out_dir, run, setup):     # a new run: its own split and verdict, from scratch
        for k in ("frac", "plan", "stat", "how"):
            i.pop(k, None)
        i["frac"] = _first_frac(i.get("gpu"))
        i["run_tag"] = _share_tag(out_dir, run, setup)
    prep = _cache_prep(out_dir, run, setup)
    if prep is None:
        return {"run": run, "tag": _share_tag(out_dir, run, setup), "preparing": True}
    closed = [int(os.path.basename(os.path.dirname(p))) for p in glob.glob(os.path.join(out_dir, SHARE_DIR, "*", "closed.json"))
              if os.path.basename(os.path.dirname(p)).isdigit()]
    entry = lambda make: [{"i": i, "name": os.path.basename(p), "size": os.path.getsize(p)}
                          for i, p in _share_send_list(setup, prep, make)]
    return {"run": run, "tag": _share_tag(out_dir, run, setup), "setup": setup, "files": entry(False), "folder": entry(True),
            "closed": closed,
            "prep": {k: prep.get(k) for k in ("why", "shas", "stages")}}


def share_cache_piece(i, name, offset=0, n=PIECE):
    """A piece of one of the files share_hello offered: the training folder's ("m<n>") or the cache's."""
    out_dir, run = _run_now()
    setup = _setup(out_dir, run)
    prep = _prep.get(_share_tag(out_dir, run, setup)) or {}
    files = [p for make in (True, False) for _k, p in _share_send_list(setup, prep, make)] if setup and prep else []
    path = next((p for p in files if os.path.basename(p) == name), None)
    if not name or path is None:
        raise ValueError("No such file.")
    with open(path, "rb") as f:
        f.seek(int(offset))
        return f.read(max(0, min(int(n), PIECE)))


def _share_epoch_dir(ep):
    out_dir, _run = _run_now()
    if not out_dir:
        raise ValueError("Nothing is training here.")
    return os.path.join(out_dir, SHARE_DIR, f"{int(ep):06d}")


def share_next(body):
    """The next epoch handed over that nobody has: its job and its start's size."""
    name = str(body.get("name") or "helper")[:60]
    out_dir, run = _run_now()
    launch = (_setup(out_dir, run) or {}).get("launch")
    for jp in sorted(glob.glob(os.path.join(out_dir, SHARE_DIR, "*", "job.json"))) if out_dir else []:
        ed = os.path.dirname(jp)
        if any(os.path.exists(os.path.join(ed, n)) for n in ("theirs.safetensors", "closed.json", "failed.json"))                 or not os.path.exists(os.path.join(ed, "start.safetensors")):
            continue                       # back already, closed by the lead, or not this run's
        if (_read(jp, {}) or {}).get("launch") != launch:
            continue
        taken = _read(os.path.join(ed, "taken.json"), {}) or {}
        if taken and taken.get("by") != name and time.time() - taken.get("at", 0) < 600:
            continue
        _write(os.path.join(ed, "taken.json"), {"by": name, "at": time.time()})
        return {"job": _read(jp), "size": os.path.getsize(os.path.join(ed, "start.safetensors"))}
    return {}


def share_start_piece(ep, offset=0, n=PIECE):
    with open(os.path.join(_share_epoch_dir(ep), "start.safetensors"), "rb") as f:
        f.seek(int(offset))
        return f.read(max(0, min(int(n), PIECE)))


def share_deliver(ep, offset, data, done=False, steps=0, seconds=0.0, name="", launch=""):
    """A piece of the helper's LoRA for an epoch; the last call (done) hands it to the trainer."""
    ed = _share_epoch_dir(ep)
    part, theirs = os.path.join(ed, "theirs.safetensors.part"), os.path.join(ed, "theirs.safetensors")
    if os.path.exists(os.path.join(ed, "closed.json")) or (
            launch and (_read(os.path.join(ed, "job.json"), {}) or {}).get("launch") != launch):
        return {"ok": True, "late": True}          # the lead stopped waiting, or it's a round of an earlier launch
    if done and os.path.exists(theirs) and not os.path.exists(part):
        return {"ok": True}                        # the last piece again (its answer was lost): already in place
    if int(offset) and not os.path.exists(part):
        raise ValueError("The start of that file never arrived: send it again from the beginning.")
    with open(part, "r+b" if int(offset) else "wb") as f:
        f.seek(int(offset))
        f.write(data)
        f.truncate()
    if done:
        os.replace(part, theirs)
        t_back = time.time()
        taken = _read(os.path.join(ed, "taken.json"), {}) or {}
        job = _read(os.path.join(ed, "job.json"), {}) or {}
        lead = None
        i = _share_in.get(name)
        for _ in range(180):    # the share follows the two speeds; answered inside the helper's 120 s and a proxy's
            lead = _read(os.path.join(ed, "lead.json"))
            if lead:
                break
            if i is not None:   # it's here, waiting on this call: still the share's helper
                i["at"] = time.time()
            time.sleep(0.5)
        if i is not None and lead and steps and seconds and lead.get("steps") and lead.get("seconds"):
            wall = t_back - float(taken.get("at") or t_back)
            i["frac"], i["plan"] = _share_plan(
                int(steps), float(seconds), lead["steps"], lead["seconds"], wall, len(job.get("epochs") or [ep]),
                int(i.get("every") or 1), i.get("frac", 0.5))
            lk, lb = i.get("link") or {}, (_setup(*_run_now()) or {}).get("lora_bytes") or 0
            if lk.get("up") and lk.get("down") and lb:
                i["overhead"] = max(0.0, wall - float(seconds) - lb / 1e6 / lk["up"] - lb / 1e6 / lk["down"])
            i["stat"] = (f"Epoch {ep}: {steps} images in {float(seconds):.0f} s there, {lead['steps']} here in "
                         f"{lead['seconds']:.0f} s; its share now {i['frac'] * 100:.0f}%")
            return {"ok": True, "plan": i["plan"]}
    return {"ok": True}


def share_fail(ep, error=""):
    _write(os.path.join(_share_epoch_dir(ep), "failed.json"), {"error": str(error)[:400]})
    return {"ok": True}


def _share_inbound():
    """The helper that connected in for the share and whose worker is ready for the run going now, or None."""
    out_dir, run = _run_now()
    tag = _share_tag(out_dir, run, _setup(out_dir, run)) if out_dir else ""
    return next((i for i in _share_in.values() if time.time() - i["at"] < 20 and i.get("ready")
                 and tag and i.get("tag") == tag), None)


def _out_share_loop(c, me):
    """This computer trains its share of the run at me["url"]: setup and cache in, its worker up, then each epoch's
    start in and its LoRA back out."""
    mine = info()
    say = lambda t: _out.__setitem__("text", t) if _out.get("gen") == me["gen"] else None
    tag, ready, restarts = "", False, 0
    say("Measuring the link")
    lk = _fresh_link(me["url"], c.cookie)
    hello = lambda **kw: c.call("/api/helpers/share_hello", {
        "name": mine["name"], "gpu": mine["gpu"], "ready": ready, "tag": tag, "every": _share_every(),
        "link": {k: lk.get(k) for k in ("up", "down", "how")} if lk.get("up") else None, **kw})
    try:
        while _out.get("gen") == me["gen"]:
            try:
                if _srv()._gpu_training():          # this computer's own run: it comes first
                    if tag:
                        share_stop(tag)
                        tag, ready = "", False
                    say("Training its own run: it helps again when that finishes")
                    hello()
                    time.sleep(10)
                    continue
                r = hello()
                if not r.get("run"):
                    if tag:
                        share_stop(tag)
                        tag, ready = "", False
                    say(r.get("off") or "Nothing training there yet")
                    time.sleep(5)
                    continue
                if r.get("preparing"):
                    say(f"Getting ready for {r['run']}'s training share")
                    time.sleep(3)
                    continue
                # a new run, or its worker went or failed: its training folder or cache here, then the worker
                ws = share_state(tag) if tag else {}
                if r["tag"] != tag or ws.get("state") in ("gone", "failed"):
                    if r["tag"] == tag and ws.get("state") == "failed":
                        restarts += 1
                        if restarts >= 3:
                            say(f"Its share keeps failing here ({_plain(ws.get('error') or '')}): stop helping, and "
                                "see this computer's log")
                            share_stop(tag)
                            return
                    if tag:
                        share_stop(tag)
                    tag, ready, new = "", False, r["tag"]
                    make, how = _share_how(r.get("prep") or {}, r["setup"],
                                           lambda s: share_probe(s["family"], s["shas"]), say)
                    _out["how"] = how
                    hello(how=how)
                    files = r["folder"] if make else r["files"]
                    what = "training folder" if make else "cache"
                    total = sum(f["size"] for f in files) or 1
                    got, names, seen = 0, {}, 0.0
                    for f in files:
                        name = _cache_key(new, f["i"], f["name"])
                        names[name] = f["name"]
                        at = max(0, min(have(name)["size"], f["size"]))
                        while at < f["size"]:
                            piece = _piece(c, f"/api/helpers/share_cache?i={f['i']}&name={urllib.parse.quote(f['name'])}"
                                              f"&offset={at}&n={PIECE}")
                            put(name, at, piece)
                            at += len(piece)
                            t = f"Getting {r['run']}'s {what} ({(got + at) * 100 // total}%)"
                            say(t)
                            if time.time() - seen > 8:      # the lead's tab keeps this computer's row meanwhile
                                hello(activity={"kind": "share", "text": t, "progress": .2 * (got + at) / total})
                                seen = time.time()
                        got += f["size"]
                    while True:
                        st = share_start({"tag": new, "setup": r["setup"], "names": names, "lead": me["name"],
                                          "files": (r.get("prep") or {}).get("files"),
                                          "make": {"stages": r["prep"]["stages"],
                                                   "files": (r.get("prep") or {}).get("files")} if make else None})
                        if st.get("caching"):
                            k = st["caching"]
                            t = (f"Making {r['run']}'s cache here: {_CACHE_WORDS.get(k.get('stage'), '')} "
                                 f"({k.get('done') or 0} of {k.get('total') or 0})")
                            say(t)
                            hello(activity={"kind": "share", "text": t, "progress": .2})
                            time.sleep(3)
                            continue
                        if not st.get("models"):
                            break
                        m = st["models"]
                        if m.get("error"):
                            raise ValueError(m["error"])
                        say(f"Getting {(m.get('fetching') or {}).get('what') or 'models'} for {r['run']}'s training share")
                        hello(activity={"kind": "models", **(m.get("fetching") or {}), "for": f"{r['run']}'s share"})
                        time.sleep(3)
                    tag = new
                if not ready:
                    s = share_state(tag)
                    if s["state"] == "failed":
                        raise ValueError("the share worker stopped: " + _plain(s.get("error") or ""))
                    if s["state"] != "ready":
                        say(f"Loading the model for {r['run']}'s training share")
                        time.sleep(3)
                        continue
                    ready = True
                    hello()
                n = c.call("/api/helpers/share_next", {"name": mine["name"]})
                if not n.get("job"):
                    say(f"Ready for {r['run']}'s next epoch")
                    time.sleep(2)
                    continue
                job = n["job"]
                ep = int(job["epoch"])
                start = f"sh{tag}-e{ep:06d}.safetensors"
                at = 0
                while at < int(n["size"]):
                    piece = _piece(c, f"/api/helpers/share_startfile?epoch={ep}&offset={at}&n={PIECE}")
                    put(start, at, piece)
                    at += len(piece)
                    hello(activity={"kind": "share", "epoch": ep, "text": "Getting this round's LoRA",
                                    "progress": .3 * at / int(n["size"])})
                share_job({"tag": tag, "start": start, **job})
                say(f"Training its share of {r['run']}'s epoch {ep}")
                late = False
                while True:
                    s = share_state(tag, ep)
                    if s["state"] == "done":
                        break
                    if s["state"] in ("failed", "gone"):
                        c.call("/api/helpers/share_fail", {"epoch": ep, "error": s.get("error") or ""})
                        raise ValueError(s.get("error") or "the share worker stopped")
                    hi = hello(activity={"kind": "share", "epoch": ep})
                    if ep in ((hi or {}).get("closed") or []):     # the lead stopped waiting: the next round goes
                        share_cancel(tag, ep)
                        late = True
                        break
                    time.sleep(1.5)
                if late:
                    drop("", start)
                    say(f"{r['run']}'s epoch {ep} went on without it (too slow for that round)")
                    hello(activity=None)
                    continue
                theirs = share_file(tag, ep)
                size, at, plan = os.path.getsize(theirs), 0, ""
                with open(theirs, "rb") as f:
                    while at < size:
                        piece = f.read(PIECE)
                        last = at + len(piece) >= size
                        q = (f"/api/helpers/share_deliver?epoch={ep}&offset={at}&name={urllib.parse.quote(mine['name'])}"
                             f"&launch={urllib.parse.quote(str(job.get('launch') or ''))}"
                             + (f"&done=1&steps={s.get('steps', 0)}&seconds={s.get('seconds', 0)}" if last else ""))
                        for k in range(5):
                            try:
                                plan = (c.call(q, raw=piece, timeout=120) or {}).get("plan") or plan
                                break
                            except (OSError, http.client.HTTPException):
                                if k == 4:
                                    raise
                                time.sleep(2)
                        at += len(piece)
                        hello(activity={"kind": "share", "epoch": ep, "text": "Sending its half back",
                                        "progress": .7 + .3 * at / size})
                drop("", start)
                try:
                    os.remove(theirs)
                except OSError:
                    pass
                say(f"Handed back {r['run']}'s epoch {ep} ({s.get('steps', 0)} images in {s.get('seconds', 0):.0f} s)"
                    + (f". {plan}" if plan else ""))
                hello(activity=None)
            except PermissionError:
                say("Its login ran out: start helping again with its password")
                return
            except Exception as e:
                logger.warning(f"[share] helping {me['url']}: {e}")
                say(f"Its share hit a problem ({_plain(e)}): trying again")
                time.sleep(8)
    finally:
        if tag:
            share_stop(tag)



# =====================================================================================================================
# move: this computer's run carries on on another one (a pod, the laptop) while this computer is wanted for something
# else. Paused here at the end of an epoch (or stopped now, from its last save), its saved state, training folder and
# loss log go over as one file, and it resumes there: the same model files (by hash) at the same precision, its cache
# made there from the pictures, block swap planned for that card. Back the same way: paused there, its new
# checkpoints, previews, logs and state brought here, and resumed here.
# =====================================================================================================================
MOVED_FILE = ".fizgig_moved.json"     # in the run's output folder here while it trains elsewhere: where it went
# the folders and files a run names besides its training folder: they go with it (dirs/<key>/<its name>/...,
# files/<key>/<name>) and come back the same way, so every LoRA kind moves
_MOVE_DIRS = ("FAMILY_CONCEPT_DIR", "FAMILY_CONCEPT_DIR3", "FAMILY_CONCEPT_DIR4", "FAMILY_EDIT_DIR",
              "FAMILY_EDIT_EXTRAS_DIR", "FAMILY_SLIDER_DIR")
_MOVE_FILES = ("FAMILY_EDIT_REF", "SAMPLE_REFERENCE", "METADATA_THUMBNAIL")


def _media_exts():
    """What a training folder trains on: pictures, clips and songs (lower case)."""
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    return {e.lower() for e in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}


def _caption_ext(values):
    return (str(values.get("DATASET_CAPTION_EXT") or ".txt").strip() or ".txt").lower()


def _move_extras(values):
    """The run's other folders and files that are there: ({key: folder}, {key: file})."""
    dirs = {k: str(values.get(k) or "").strip() for k in _MOVE_DIRS}
    files = {k: str(values.get(k) or "").strip() for k in _MOVE_FILES}
    files = {k: f for k, f in files.items() if f.lower() != "off"}      # Thumbnail "off": none, not a file
    return ({k: d for k, d in dirs.items() if d and os.path.isdir(d)},
            {k: f for k, f in files.items() if f and os.path.isfile(f)})


def _run_subfolders(folder, values):
    """[(relative path, folder)] of the run's sub-folders when it trains them (Also train on its sub-folders)."""
    if not (values or {}).get("TRAIN_SUBFOLDERS"):
        return []
    from fizgig.families import launch
    return launch.subfolders(folder, (str(values.get("LORA_OUTPUT_DIR") or ""),
                                      str(values.get("FAMILY_FT_REG_DIR") or "")))


def _safe_rel(rel):
    """A file's path inside a training folder: plain names joined by /, never a way out of it."""
    parts = str(rel or "").split("/")
    return bool(parts) and all(_SAFE_FILE.match(p) for p in parts)


def _dir_name(d):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", os.path.basename(os.path.normpath(d))).strip(" .")[:80] or "images"


_mv = {"state": ""}       # this computer's run: state (moving / moved / back / failed), text, progress, to, url, tag ...
_mvin = {}                # tag -> a run moved here: {run, folder, output_dir, base, started, gone}


def moved_in_waiting(out_dir):
    """A run moved here that finished here, not yet taken back by its computer: a pod doesn't stop itself on it
    (Bring its LoRAs here would find it stopped)."""
    return any(r.get("output_dir") == out_dir and not r.get("gone") for r in _mvin_load().values())


def move_status():
    if not _mv.get("state"):
        return None
    out = {k: v for k, v in _mv.items() if k not in ("cookie", "values")}
    row = _home_rows().get(_norm(_mv.get("url") or "x")) if _mv.get("url") else None
    if row and row.get("move"):              # its files coming home as they're saved: how far that has got
        out["home_text"] = (_home.get(_norm(_mv["url"])) or {}).get("text") or ""
    from fizgig.web import pods
    out["pod"] = bool(pods.parse(_mv.get("url") or "").get("id"))
    return out


def move_busy():
    """Why nothing else can start or resume here now, or "": a run of this computer's is moving or trains elsewhere."""
    st = _mv.get("state")
    if st in ("moving", "back"):
        return "A run is moving between computers: wait until it has gone or come back."
    if st == "moved":
        return f"This computer's run trains on {_mv.get('to')}: bring it back, or Carry on here, first."
    return ""


def move_clear(here=False):
    """The note of a move that didn't work goes (the run is paused here, as it was). here: a run that went to a
    computer it can't come back from (a pod deleted, a laptop gone) carries on here from its last save here - the
    epochs it trained there are lost, and that computer is told to stop it if it still answers."""
    if _mv.get("state") == "failed":
        _mv.clear()
        _mv["state"] = ""
    elif here and _mv.get("state") == "moved":
        s = _srv()
        out_dir, name, tag = _mv.get("out_dir"), _mv.get("run"), _mv.get("tag")
        c = _mv_client()
        try:
            c.call("/api/helper/move_out", {"tag": tag, "abandon": True}, timeout=15)
        except Exception:                    # gone, or away just now: told again until it hears (or is gone for good),
            threading.Thread(target=_abandon_later, args=(c, tag), daemon=True,   # or it trains on there too
                             name="helpers-move-abandon").start()
        state = s.launch.latest_state(out_dir, name)
        try:
            os.remove(os.path.join(out_dir, MOVED_FILE))
        except OSError:
            pass
        fam, values, folder = _mv.get("family"), dict(_mv.get("values") or {}), _mv.get("image_folder") or ""
        _mv.clear()
        _mv["state"] = ""
        with s._lock:
            s._run.update(state="paused" if state else "stopped", resume=state or "", resume_mode="state",
                          family=fam, values=values, image_folder=folder, lora=name, output_dir=out_dir)
        if state:
            s._record_pause(out_dir)
        s.append_log(f"[move] {name} carries on here from {os.path.basename(state) if state else 'nothing saved'}"
                     ": Resume it.\n")
    return {"ok": True}


_home_active = {}         # pod url -> True while the bring-home loop is fetching from it


def _moved_record(url, run, remote_out, out_dir):
    """Which run of this computer's went to `url`, so what comes home from it later (Help another Fizgig with
    this GPU, after the move is over too) goes into its own output folder, not a pod's folder of its own."""
    rec = dict(_srv().load_prefs().get("helper_moved_runs") or {})
    rec[_norm(url)] = {"run": run, "remote_out": remote_out, "out_dir": out_dir, "at": time.time()}
    _srv().update_prefs({"helper_moved_runs": rec})


def _moved_dest(url, run, remote_out):
    """This computer's output folder for a run that was moved to `url` from here, or ""."""
    r = (_srv().load_prefs().get("helper_moved_runs") or {}).get(_norm(url)) or {}
    return r.get("out_dir") or "" if r.get("run") == run and r.get("remote_out") == remote_out else ""


def _home_for_move(url, out_dir):
    """A moved run's checkpoints and previews, home as they're saved (the bring-home loop)."""
    rows = _home_rows()
    rows[_norm(url)] = {**rows.get(_norm(url), {}), "on": True, "previews": True, "move": True, "dest": out_dir}
    _srv().update_prefs({"helper_bring_home": rows})
    ensure_home()


def _back_done(url, to, delete, ok):
    """A run brought back: its move's bring-home row goes, and with Delete the pod ticked, the pod - everything
    of the run is here now."""
    u = _norm(url) if url else ""
    rows = _home_rows()
    if u in rows and rows[u].get("move"):
        rows.pop(u, None)
        _home.pop(u, None)
        _srv().update_prefs({"helper_bring_home": rows})
    if not (delete and ok and u):
        return
    from fizgig.web import pod_stop, pods
    s = _srv()
    try:
        pods.terminate(pod_stop.api_key(s.load_prefs()), pods.parse(u)["id"])
    except Exception as e:
        s.append_log(f"[move] Everything is back, but {to} couldn't be deleted ({_plain(e)}): delete it on RunPod.\n")
        return
    s.update_prefs({"app_pods": [p for p in pods.saved(s.load_prefs()) if _norm(p["url"]) != u],
                    "helpers": [h for h in _helpers() if _norm(h.get("url") or "x") != u]})
    s.append_log(f"[move] Everything is back from {to}, so the pod was deleted.\n")


def _abandon_later(c, tag, hours=12):
    """Carry on here while the other computer was away: it's told to stop its copy once it answers again, so the
    run never trains in two places (a deleted pod never answers: given up after `hours`)."""
    end = time.time() + hours * 3600
    while time.time() < end:
        time.sleep(30)
        try:
            c.call("/api/helper/move_out", {"tag": tag, "abandon": True}, timeout=30)
            _srv().append_log("[move] The other computer is back: its copy of the run was stopped.\n")
            return
        except json.JSONDecodeError:
            pass                             # a proxy's error page, not Fizgig's answer
        except ValueError:
            return                           # it answered, and that run isn't there (any more)
        except Exception:
            pass


def _mv_say(text, progress=None, log=True):
    _mv.update(text=text, progress=progress)
    if log:
        _srv().append_log(f"[move] {text}\n")


def _mv_client():
    h = next((x for x in _helpers() if x.get("url") == _mv.get("url")), None)
    return Client(_mv["url"], (h or {}).get("cookie", "") or _mv.get("cookie", ""))


def move_start(hid, now=False, home=True):
    if _mv.get("state") in ("moving", "back", "moved"):
        raise ValueError("This run is already moving, or training elsewhere.")
    h = next((x for x in _helpers() if x.get("id") == hid), None)
    if h is None:
        raise ValueError("That helper isn't in the list any more.")
    lv = _live.get(hid) or {}
    why = {"busy": "it's training a run of its own", "away": "it isn't answering", "elsewhere":
           "it's helping another computer", "login": "its login ran out: add it again"}.get(lv.get("state"), "")
    if lv.get("state") != "ready":
        raise ValueError(f"{h.get('name')} can't take it: {why or 'it is not ready yet'}.")
    if lv.get("build") not in (None, "", "same"):
        word = {"mine": "a newer Fizgig than this computer", "theirs": "an older Fizgig than this computer"}.get(
            lv["build"], "another Fizgig version")
        how = (lv.get("match") or "") if lv.get("pod") else "Update the older one, then Move."
        raise ValueError(f"{h.get('name')} runs {word}, and both need the same one. {how}".strip())
    s = _srv()
    with s._lock:
        r = {k: s._run.get(k) for k in ("state", "stage", "family", "values", "lora", "output_dir", "resume_mode")}
        running = s._run["proc"] is not None and s._run["proc"].poll() is None
    if not (running and r["stage"] == "Training") and r["state"] != "paused":
        raise ValueError("Nothing is training here to move (a run still making its cache can move once it trains).")
    desc = s.get_family(r["family"] or "")
    v = dict(r["values"] or {})
    if desc is None or r["family"] == s.refmod.KEY or r["resume_mode"] == "ft" or s.form._kind(desc, v) == "finetune":
        raise ValueError("A fine-tune or a RefMod can't move: LoRA runs move (standard, edit and slider).")
    gone = [k for k in _MOVE_DIRS + _MOVE_FILES if str(v.get(k) or "").strip().lower() not in ("", "off") and not (
        os.path.isdir(str(v[k]).strip()) if k in _MOVE_DIRS else os.path.isfile(str(v[k]).strip()))]
    if gone:
        raise ValueError(f"It names {str(v[gone[0]]).strip()}, which isn't on this computer any more, so it can't "
                         "go with the run.")
    if now and running and not s.launch.latest_state(r["output_dir"], r["lora"]):
        raise ValueError("It hasn't saved yet, so it can't move now: move it at the end of this epoch instead.")
    _mv.clear()
    _mv.update(state="moving", hid=hid, to=h.get("name") or h["url"], url=h["url"], run=r["lora"],
               out_dir=r["output_dir"], error="", text="Getting ready to move", progress=None, home=bool(home))
    threading.Thread(target=_move_work, args=(h, now), daemon=True, name="helpers-move").start()
    return {"ok": True}


def _move_shas(out_dir, name, setup):
    """The run's model files by hash: the one it trains on, and the VAE and text encoder its cache was made with."""
    shas = {}
    plan = _read(os.path.join(out_dir, SHARE_DIR, CACHE_PLAN), {}) or {}
    for st in (plan.get("stages") or []) if plan.get("output_name") == name else []:
        a = st["args"]
        if "--model" in a[:-1]:
            shas[st["stage"]] = _sha(a[a.index("--model") + 1])
    args = list(setup.get("args") or [])
    if "--dit" in args[:-1]:
        shas["dit"] = _sha(args[args.index("--dit") + 1])
    return shas


_LIVE_KEYS = {"seed": "SAMPLE_SEED", "width": "SAMPLE_WIDTH", "height": "SAMPLE_HEIGHT", "steps": "SAMPLE_STEPS",
              "cfg": "SAMPLE_CFG_SCALE", "turbo": "FAMILY_TURBO_STRENGTH", "every": "SAMPLE_EVERY_N_EPOCHS",
              "until": "SAMPLE_UNTIL_EPOCH", "after": "SAMPLE_EVERY_AFTER"}


def _with_live_previews(out_dir, values):
    """The run's settings with the Samples tab's live change in them (Change the previews of the run going now:
    <out_dir>/.sample_override.json), so its previews look the same wherever it goes next - the file itself is
    this computer's (its paths) and a start clears it."""
    ov = _read(os.path.join(out_dir or "", ".sample_override.json"), {}) or {}
    if not ov.get("prompts"):
        return values
    v = dict(values, SAMPLE_PROMPT="\n".join(ov["prompts"]))
    for k, key in _LIVE_KEYS.items():
        if ov.get(k) is not None:
            v[key] = str(ov[k])
    if "until" in ov or "after" in ov:       # Then preview less often: until / after 0 = off
        v["SAMPLE_LATER"] = bool(int(ov.get("until") or 0) and int(ov.get("after") or 0))
    if "look" in ov:                         # the SFT look tick: a path here, so only whether it's on
        v["FAMILY_PREVIEW_LOOK"] = bool(ov["look"])
    if "checkpoint" in ov:                   # the preview-model tick (Qwen's Turbo)
        v["SAMPLE_CHECKPOINT"] = bool(ov["checkpoint"])
    return v


def _move_pack(out_dir, name, state, folder, values):
    """The run as one file: its training folder (pictures, captions, the dataset's own fizgig_*.json), the other
    folders and files it names (_MOVE_DIRS, _MOVE_FILES), its saved state, loss log, loss graph and previews, and a
    context LoRA."""
    cap = _caption_ext(values)
    exts = _media_exts() | {cap}
    path = os.path.join(out_dir, SHARE_DIR, "move.tar")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path + ".tmp", "w") as tf:
        for p in sorted(glob.glob(os.path.join(folder, "*"))):
            b = os.path.basename(p)
            if os.path.isfile(p) and (os.path.splitext(b)[1].lower() in exts or
                                      (b.startswith("fizgig_") and b.endswith(".json"))):
                tf.add(p, arcname=f"images/{b}")
        for rel, sub in _run_subfolders(folder, values):      # its sub-folders, when the run trains them
            for p in sorted(glob.glob(os.path.join(glob.escape(sub), "*"))):
                b = os.path.basename(p)
                if os.path.isfile(p) and (os.path.splitext(b)[1].lower() in exts or
                                          (b.startswith("fizgig_") and b.endswith(".json"))):
                    tf.add(p, arcname=f"images/{rel}/{b}")
        dirs, files = _move_extras(values)
        for k, d in dirs.items():
            for p in sorted(glob.glob(os.path.join(d, "*"))):
                b = os.path.basename(p)
                if os.path.isfile(p) and (os.path.splitext(b)[1].lower() in exts or
                                          (b.startswith("fizgig_") and b.endswith(".json"))):
                    tf.add(p, arcname=f"dirs/{k}/{_dir_name(d)}/{b}")
        for k, f in files.items():
            tf.add(f, arcname=f"files/{k}/{os.path.basename(f)}")
        tf.add(state, arcname=f"out/{os.path.basename(state)}")
        for extra in ("loss_log", "sample", f".fizgig_loss_{name}.jsonl"):
            if os.path.exists(os.path.join(out_dir, extra)):
                tf.add(os.path.join(out_dir, extra), arcname=f"out/{extra}")
        ctx = str(values.get("CONTEXT_LORA_PATH") or "").strip()
        if ctx and os.path.isfile(ctx):
            tf.add(ctx, arcname="context.safetensors")
    os.replace(path + ".tmp", path)
    return path


def _move_wait_paused(s, out_dir):
    """Ask the run going here to pause at the end of this epoch, and wait until it has."""
    with open(os.path.join(out_dir, ".pause_requested"), "w", encoding="utf-8") as f:
        f.write("pause\n")
    with s._lock:
        s._run["pause"] = True
    while True:
        with s._lock:
            st = s._run["state"]
        if st == "paused":
            return
        if st in ("failed", "stopped", "completed"):
            raise ValueError(f"the run {'finished' if st == 'completed' else st} before it could move")
        time.sleep(1)


def _move_work(h, now):
    s = _srv()
    to, tag, out_dir = _mv["to"], "", None
    try:
        with s._lock:
            running = s._run["proc"] is not None and s._run["proc"].poll() is None
            out_dir, name = s._run["output_dir"], s._run["lora"]
        if running and now:
            _mv_say(f"Stopping the run to move it to {to}")
            s._stop_and_wait(60)
            if not s._record_pause(out_dir):
                raise ValueError("it has no saved state to carry on from")
            with s._lock:
                s._run["state"] = "paused"
        elif running:
            _mv_say(f"Moving to {to} at the end of this epoch")
            _move_wait_paused(s, out_dir)
        with s._lock:
            fam, values, folder, state = (s._run["family"], dict(s._run["values"] or {}), s._run["image_folder"],
                                          s._run["resume"])
        values = _with_live_previews(out_dir, values)    # its previews as they are now, not as it started
        if not state or not os.path.isdir(state):
            raise ValueError("its saved state isn't there")
        setup = _read(os.path.join(out_dir, SHARE_DIR, "setup.json"), {}) or {}
        setup = setup if setup.get("output_name") == name else {}
        c = _mv_client()
        _mv_say(f"Checking {to} has this run's model files (the first time hashes them)")
        shas = _move_shas(out_dir, name, setup)
        while True:
            pr = _again(lambda: c.call("/api/helper/share_probe", {"family": fam, "shas": shas}, timeout=30))
            if not pr.get("checking"):
                break
            time.sleep(3)
        if not pr.get("ok"):
            raise ValueError(f"{to} can't train it: {pr.get('why') or 'its model files are not the same'}")
        _mv_say("Packing the run to send")
        tar = _move_pack(out_dir, name, state, folder, values)
        size = os.path.getsize(tar)
        tag = hashlib.sha1(f"{out_dir}|{name}|{os.path.basename(state)}|{time.time()}".encode()).hexdigest()[:10]
        _mv_say(f"Sending the run to {to} ({size / 2 ** 20:.0f} MB)")
        _send_file(c, tar, f"mv{tag}.tar", lambda f: _mv_say(
            f"Sending the run to {to}: {size * f / 2 ** 20:.0f} of {size / 2 ** 20:.0f} MB", f, log=False))
        body = {"tag": tag, "family": fam, "values": values, "run": name, "state": os.path.basename(state),
                "folder": os.path.basename(os.path.normpath(folder)) or "images", "lead": socket.gethostname(),
                "precision": setup.get("precision") or "", "files": pr.get("files") or {},
                "dirs": {k: _dir_name(d) for k, d in _move_extras(values)[0].items()},
                "extra_files": {k: os.path.basename(f) for k, f in _move_extras(values)[1].items()}}
        _mv_say(f"{to} is getting the run ready")
        said = ""
        while True:
            a = _again(lambda: c.call("/api/helper/move_in", body, timeout=120))
            if a.get("ok"):
                break
            m = a.get("models") or {}
            if m.get("error"):
                raise ValueError(m["error"])
            f = m.get("fetching") or {}
            t = (f"{to} is downloading {f.get('what') or 'the model files'}" +
                 (f": {f['got']:.1f} of {f['total']:.1f} GB" if f.get("total") else "") if f else
                 f"{to} is getting the model files ready")
            _mv_say(t, (f["got"] / f["total"]) if f.get("total") else None, log=t.split(":")[0] != said)
            said = t.split(":")[0]
            time.sleep(5)
        _write(os.path.join(out_dir, MOVED_FILE), {"url": _mv["url"], "to": to, "tag": tag, "run": name,
                                                   "remote_out": a.get("output_dir") or "", "at": time.time(),
                                                   "family": fam, "values": values, "image_folder": folder,
                                                   "home": bool(_mv.get("home"))})
        _moved_record(_mv["url"], name, a.get("output_dir") or "", out_dir)
        if _mv.get("home"):                  # its checkpoints and previews come home as they're saved, queued
            _home_for_move(_mv["url"], out_dir)
        with s._lock:
            s._run["state"] = "moved"
        _mv.update(state="moved", tag=tag, remote_out=a.get("output_dir") or "", remote=None, family=fam,
                   values=values, image_folder=folder)
        _mv_say(f"Training on {to} now, from {os.path.basename(state)}. Bring it back from the run card when you "
                f"want it here again")
        threading.Thread(target=_move_watch, daemon=True, name="helpers-move-watch").start()
    except PermissionError:
        _mv.update(state="failed", error="login")
        _mv_say(f"The move didn't work: {to}'s login ran out (add it again on the Helpers tab). The run is paused "
                "here: Resume carries on here")
    except Exception as e:
        logger.warning(f"[move] to {to}: {e}")
        if tag:                              # it may have started there with only the answer lost: stop it there
            try:
                _mv_client().call("/api/helper/move_out", {"tag": tag, "abandon": True}, timeout=30)
            except Exception:
                pass
        _mv.update(state="failed", error=_plain(e))
        _mv_say(f"The move didn't work: {_plain(e)}. The run is paused here: Resume carries on here")
    finally:
        for p in (os.path.join(out_dir, SHARE_DIR, "move.tar"), os.path.join(out_dir, SHARE_DIR, "move.tar.tmp")):
            try:
                os.remove(p)
            except (OSError, TypeError):
                pass


def _move_watch():
    """While the run trains elsewhere: how far it has got there, every few seconds."""
    c, fails = _mv_client(), 0
    while _mv.get("state") == "moved":
        try:
            r = c.call("/api/run", timeout=15)
            fails = 0
            mine = r.get("output_dir") == _mv.get("remote_out") and r.get("lora") == _mv.get("run")
            p = (r.get("progress") or {}) if mine else {}
            _mv["remote"] = {"state": r.get("state") if mine else "gone", "epoch": p.get("epoch"),
                             "epochs": p.get("total_epochs"), "percent": p.get("percent"), "eta": p.get("eta"),
                             "stage": r.get("stage") if mine else ""}
        except PermissionError:
            _mv["remote"] = {"state": "login"}
        except Exception:
            fails += 1
            if fails >= 3:
                _mv["remote"] = {"state": "away"}
        time.sleep(8)


def move_restore(out_dir, name):
    """At start-up: a paused run whose output folder says it went to another computer is that, not paused."""
    m = _read(os.path.join(out_dir, MOVED_FILE)) if out_dir else None
    if not m or m.get("run") != name:
        return False
    _mv.clear()
    _mv.update(state="moved", to=m.get("to"), url=m.get("url"), tag=m.get("tag"), run=name, out_dir=out_dir,
               remote_out=m.get("remote_out"), remote=None, text=f"Training on {m.get('to')}", progress=None,
               family=m.get("family"), values=m.get("values") or {}, image_folder=m.get("image_folder") or "")
    threading.Thread(target=_move_watch, daemon=True, name="helpers-move-watch").start()
    return True


# ---- the computer a run moves to ------------------------------------------------------------------------------------
def _mvin_load():
    if not _mvin:
        _mvin.update(_read(_dir("moved", "moved.json"), {}) or {})
    return _mvin


def _mvin_save():
    _write(_dir("moved", "moved.json"), _mvin)


def _unpack_move(tar, base, folder, out_dir):
    """The moved run's file: images/ into its training folder here, out/ into its output folder, the context LoRA
    beside them. Only plain files and folders, never a path out of those."""
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            parts = m.name.replace("\\", "/").split("/")
            if not (m.isfile() or m.isdir()) or ".." in parts or m.name.startswith("/") or ":" in m.name:
                continue
            if m.name == "context.safetensors":
                dest = os.path.join(base, m.name)
            elif parts[0] in ("images", "out") and len(parts) > 1:
                dest = os.path.join(folder if parts[0] == "images" else out_dir, *parts[1:])
            elif parts[0] in ("dirs", "files") and len(parts) > 2:     # the run's other folders and files
                dest = os.path.join(base, *parts)
            else:
                continue
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def move_in(body):
    """A run moving here: its model files checked (fetched where they're missing), its file unpacked, and it resumed
    as this computer's own run. Answers {models} while files download, then {ok, output_dir}."""
    s = _srv()
    tag = str(body.get("tag") or "")
    if not re.match(r"^[a-f0-9]{10}$", tag):
        raise ValueError("Bad tag.")
    rec = _mvin_load().get(tag)
    if rec and rec.get("started"):
        return {"ok": True, "output_dir": rec["output_dir"], "run": rec["run"]}
    with s._lock:
        busy = s._run["proc"] is not None and s._run["proc"].poll() is None or s._run["state"] in ("starting", "running")
        paused = s._run["state"] == "paused"
    if busy or s._gpu_training():
        raise ValueError("It's training a run of its own now.")
    if paused:
        raise ValueError("It has a paused run of its own: resume or stop that there first.")
    desc = s.get_family(str(body.get("family") or ""))
    if desc is None:
        raise ValueError("This Fizgig doesn't know that model: update it.")
    v = dict(body.get("values") or {})
    paths = {}
    for stage, spec in (body.get("files") or {}).items():      # the files the run was checked on (share_probe)
        if spec.get("path") and os.path.isfile(spec["path"]):
            paths[stage] = spec["path"]
        elif spec.get("file"):
            got, wait = _get_build(spec)
            if wait is not None:
                return {"models": wait}
            paths[stage] = got
    try:
        turbo = float(str(v.get("FAMILY_TURBO_STRENGTH") or 0).strip() or 0)
    except ValueError:
        turbo = 0.0
    m = models(desc.key, {"conds": "text" in paths, "turbo": turbo, "look": 1 if v.get("FAMILY_PREVIEW_LOOK") else 0},
               fetch=True)
    if m["missing"]:
        return {"models": m}
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("run") or "run")).strip("._")[:80] or "run"
    for old in glob.glob(_dir("files", "mv*.tar")):      # an earlier attempt's file that never became a run
        if os.path.basename(old) != f"mv{tag}.tar" and time.time() - os.path.getmtime(old) > 3600:
            try:
                os.remove(old)
            except OSError:
                pass
    if rec is None or not rec.get("unpacked"):
        tar = _dir("files", f"mv{tag}.tar")
        if not os.path.isfile(tar):
            raise ValueError("The run's files didn't arrive: move it again.")
        base = _dir("moved", f"{name[:60]}-{tag}")
        folder = os.path.join(base, re.sub(r"[^A-Za-z0-9_. -]+", "_", str(body.get("folder") or ""))[:80] or "images")
        out_dir = os.path.join(s.default_output_dir(), name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            out_dir = os.path.join(s.default_output_dir(), f"{name}-moved-{tag[:4]}")
        _unpack_move(tar, base, folder, out_dir)
        os.remove(tar)
        # the run's own name (its files are named by it: a space and all); `name` is only for folders
        rec = _mvin[tag] = {"run": str(v.get("LORA_NAME") or body.get("run") or name), "folder": folder,
                            "output_dir": out_dir, "base": base, "unpacked": True, "lead": str(body.get("lead") or ""),
                            "cap": _caption_ext(v), "values": {"TRAIN_SUBFOLDERS": bool(v.get("TRAIN_SUBFOLDERS"))}}
        _mvin_save()
    state = os.path.join(rec["output_dir"], os.path.basename(str(body.get("state") or "")))
    if not os.path.isdir(state):
        raise ValueError("The run's saved state didn't arrive whole: move it again.")
    v.update(LORA_OUTPUT_DIR=rec["output_dir"], BLOCKS_SWAP=s.form.DEFAULTS["BLOCKS_SWAP"], RESUME_TRAINING="")
    labels = {**s.launch.PRECISION_LABELS, **dict(desc.precision_labels or ())}
    if body.get("precision") in labels:          # the precision it trained at there (Auto decided there)
        v["FAMILY_PRECISION"] = labels[body["precision"]]
    rec["dirs"] = {}
    for k, n in (body.get("dirs") or {}).items():         # its other folders and files: where they landed here
        if k in _MOVE_DIRS:
            d = os.path.join(rec["base"], "dirs", k, _dir_name(str(n)))
            if not os.path.isdir(d):
                raise ValueError("One of the run's folders didn't arrive: move it again.")
            v[k] = rec["dirs"][k] = d
    for k, n in (body.get("extra_files") or {}).items():
        if k in _MOVE_FILES:
            f = os.path.join(rec["base"], "files", k, os.path.basename(str(n)))
            if not os.path.isfile(f):
                raise ValueError("One of the run's files didn't arrive: move it again.")
            v[k] = f
    _mvin_save()
    if str(v.get("CONTEXT_LORA_PATH") or "").strip():
        v["CONTEXT_LORA_PATH"] = os.path.join(rec["base"], "context.safetensors")
        if not os.path.isfile(v["CONTEXT_LORA_PATH"]):
            raise ValueError("The run's context LoRA didn't arrive: move it again.")
    over = {}
    for stage, role in (("dit", "dit"), ("latents", "vae"), ("text", "text_encoder")):
        f = next((mf for mf in desc.model_files if mf.role == role), None)
        if f is not None and paths.get(stage):
            over[f.pref_key] = paths[stage]
    r = s._start(desc.key, v, rec["folder"], resume=state, models=over,
                 note=f"=== Moved here from {rec.get('lead') or 'another computer'}: carrying on from "
                      f"{os.path.basename(state)} ===")
    if not r.get("ok"):
        raise ValueError(r.get("error") or "; ".join(r.get("problems") or ["it couldn't start"]))
    rec["started"] = True
    _mvin_save()
    logger.info(f"[move] {rec['run']} moved here from {rec.get('lead')}: resuming from {os.path.basename(state)}")
    return {"ok": True, "output_dir": rec["output_dir"], "run": rec["run"]}


def move_out(body):
    """A run moved here going back: paused here (at the end of this epoch, or now from its last save), then {files}:
    everything in its output folder the computer it came from needs, and the state to carry on from. done: it has
    gone back, so it can't be resumed here as well."""
    s = _srv()
    rec = _mvin_load().get(str(body.get("tag") or ""))
    if rec is None:
        raise ValueError("That run isn't on this computer.")
    out, name = rec["output_dir"], rec["run"]
    with s._lock:
        mine = s._run.get("output_dir") == out
        running = mine and s._run["proc"] is not None and s._run["proc"].poll() is None
        st = s._run["state"] if mine else ""
    if body.get("done") or body.get("abandon"):
        finished_here = st == "completed"
        if body.get("abandon") and running:
            s._stop_and_wait(60)
        with s._lock:
            if mine and s._run["state"] in ("paused", "completed", "stopped", "failed"):
                s._run["state"] = "moved"
        rec["gone"] = True
        _mvin_save()
        try:
            os.remove(os.path.join(out, s.PAUSED_FILE))
        except OSError:
            pass
        if body.get("done"):                 # everything is back there: this computer's copies go
            shutil.rmtree(rec.get("base") or "", ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
            with s._lock:
                idle = s._run["proc"] is None or s._run["proc"].poll() is not None
            if finished_here and idle and not s.webqueue.status()["items"]:
                from fizgig.web import pod_stop   # the stop it held back for this (moved_in_waiting) can go now
                pod_stop.maybe_start(s.load_prefs(), s.append_log)
        s.append_log(f"[move] {name} " + ("went back to " if body.get("done") else "was taken back by ") +
                     f"{rec.get('lead') or 'the computer it came from'}.\n")
        return {"ok": True}
    if running:
        if body.get("now") and s.launch.latest_state(out, name):
            s.append_log(f"[move] Stopping {name}: it's going back to {rec.get('lead')}.\n")
            s._stop_and_wait(60)
            s._record_pause(out)
            with s._lock:
                s._run["state"] = "paused"
        else:
            if not os.path.exists(os.path.join(out, ".pause_requested")):
                with open(os.path.join(out, ".pause_requested"), "w", encoding="utf-8") as f:
                    f.write("pause\n")
                with s._lock:
                    s._run["pause"] = True
                s.append_log(f"[move] {name} goes back to {rec.get('lead')} at the end of this epoch.\n")
            return {"text": "It goes back at the end of this epoch"}
    state = s.launch.latest_state(out, name)
    with s._lock:
        finished = mine and s._run["state"] == "completed"
    # after a restart here the run record is gone: its final LoRA (<name>.safetensors) says it finished
    finished = finished or (not running and os.path.isfile(os.path.join(out, name + ".safetensors")))
    keep = os.path.basename(state) if state else ""
    files = []
    for root, _dirs, fs in os.walk(out):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace("\\", "/")
            top = rel.split("/")[0]
            if top.startswith(".helper") or rel in (s.PAUSED_FILE, ".pause_requested") or f.endswith((".tmp", ".part")):
                continue
            if re.fullmatch(re.escape(name) + r"-\d{6}-state", top) and top != keep:
                continue                     # only the newest state: the one it carries on from
            files.append({"name": rel, "size": os.path.getsize(p)})
    # captions and exclusions changed here: the training folder's, and the run's other folders'
    subs = [(f"images/{rel}/", d) for rel, d in _run_subfolders(rec["folder"], rec.get("values") or {})]
    for prefix, d in [("images/", rec["folder"])] + subs + [(f"dirs/{k}/", d) for k, d in (rec.get("dirs") or {}).items()]:
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            b = os.path.basename(p)
            if os.path.isfile(p) and (b.lower().endswith(rec.get("cap") or ".txt")
                                      or (b.startswith("fizgig_") and b.endswith(".json"))):
                files.append({"name": prefix + b, "size": os.path.getsize(p), "folder": True})
    return {"files": files, "state": "" if finished else keep, "finished": finished}


def move_piece(tag, name, offset=0, n=PIECE):
    """A piece of one of a moved run's files, for the computer it goes back to."""
    rec = _mvin_load().get(str(tag or ""))
    if rec is None:
        raise ValueError("That run isn't on this computer.")
    name = str(name or "")
    if name.startswith("images/"):          # a caption or the dataset's own json, from its training folder here
        p = os.path.join(rec["folder"], *name[7:].split("/")) if _safe_rel(name[7:]) else None
    elif name.startswith("dirs/"):          # ... or from one of its other folders here
        _d, k, b = (name.split("/", 2) + ["", ""])[:3]
        d = (rec.get("dirs") or {}).get(k)
        p = os.path.join(d, b) if d and _SAFE_FILE.match(b) else None
    else:
        p = _inside(rec["output_dir"], name)
    if not p or not os.path.isfile(p):
        raise ValueError("Not one of the run's files.")
    with open(p, "rb") as f:
        f.seek(int(offset))
        return f.read(max(1, min(int(n), PIECE)))


# ---- bringing it back -----------------------------------------------------------------------------------------------
def move_back(now=False, delete=False):
    if _mv.get("state") != "moved":
        raise ValueError("No run of this computer's is training elsewhere.")
    if delete:
        from fizgig.web import pod_stop, pods
        if not pods.parse(_mv.get("url") or "").get("id"):
            delete = False                   # another computer: never deleted
        elif not pod_stop.api_key(_srv().load_prefs()):
            raise ValueError("Deleting the pod needs your RunPod API key: add it on the Pods tab, or untick Delete "
                             "the pod.")
    s = _srv()
    with s._lock:
        busy = s._run["proc"] is not None and s._run["proc"].poll() is None
    if busy:
        raise ValueError("Something is training here: stop it first.")
    _mv.update(state="back", error="", delete_after=bool(delete))
    threading.Thread(target=_back_work, args=(now,), daemon=True, name="helpers-move-back").start()
    return {"ok": True}


def _back_work(now):
    s = _srv()
    to, tag, out_dir, name = _mv["to"], _mv["tag"], _mv["out_dir"], _mv["run"]
    folder_here = _mv.get("image_folder") or ""
    url, delete = _mv.get("url") or "", bool(_mv.get("delete_after"))
    state = ""
    try:
        c = _mv_client()
        for _n in range(600):                # the file it's bringing home now (it stops after that one)
            if not _home_active.get(_norm(url)):
                break
            _mv_say(f"Finishing the checkpoint on its way home from {to}", log=False)
            time.sleep(1)
        _mv_say(f"Asking {to} to pause it" + (" now" if now else " at the end of this epoch"))
        while True:
            a = _again(lambda: c.call("/api/helper/move_out", {"tag": tag, "now": now}, timeout=90))
            if a.get("files") is not None:
                break
            _mv_say(f"{to}: {a.get('text') or 'pausing it'}", log=False)
            time.sleep(4)
        dirs_here = {k: str((_mv.get("values") or {}).get(k) or "").strip() for k in _MOVE_DIRS}

        def dest_of(n):
            if n.startswith("images/"):
                return (os.path.join(folder_here, *n[7:].split("/")) if folder_here and _safe_rel(n[7:])
                        else None)
            if n.startswith("dirs/"):            # a caption from one of its other folders: back into that folder
                _d, k, b = (n.split("/", 2) + ["", ""])[:3]
                return os.path.join(dirs_here[k], b) if dirs_here.get(k) and _SAFE_FILE.match(b) else None
            return _inside(out_dir, n)
        files = [f for f in a["files"] if dest_of(f["name"]) and not f["name"].endswith("_dop_targets.pt")]
        if a.get("state") and not re.fullmatch(re.escape(name) + r"-\d{6}-state", str(a["state"])):
            raise ValueError(f"{to} named a state that isn't this run's")
        total, got = sum(int(f["size"]) for f in files) or 1, 0
        for f in files:
            dest = dest_of(f["name"])
            size = int(f["size"])
            if (os.path.isfile(dest) and os.path.getsize(dest) == size and not f["name"].endswith(".json")
                    and not f["name"].startswith(("images/", "dirs/"))):
                got += size
                continue                     # a preview or checkpoint already here
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest + ".part", "wb") as fh:
                at = 0
                while at < size:
                    q = urllib.parse.quote(f["name"])
                    piece = _piece(c, f"/api/helper/move_file?tag={tag}&name={q}&offset={at}&n={PIECE}")
                    if not piece:
                        raise ValueError(f"{f['name']} came back short")
                    fh.write(piece)
                    at += len(piece)
                    got += len(piece)
                    _mv_say(f"Bringing it back from {to}: {got / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB",
                            got / total, log=False)
            os.replace(dest + ".part", dest)
        state = os.path.join(out_dir, a["state"]) if a.get("state") else ""
        _again(lambda: c.call("/api/helper/move_out", {"tag": tag, "done": True}, timeout=30))
        try:
            os.remove(os.path.join(out_dir, MOVED_FILE))
        except OSError:
            pass
        fam, values, folder = _mv.get("family"), dict(_mv.get("values") or {}), _mv.get("image_folder") or ""
        remote = dict(_mv.get("remote") or {})
        _mv.clear()
        _mv["state"] = ""
        if not state:
            for f in (s.PAUSED_FILE, ".pause_requested"):     # its pause from when it left: a restart here would
                try:                                          # offer a Resume of a run that has finished
                    os.remove(os.path.join(out_dir, f))
                except OSError:
                    pass
            with s._lock:
                if s._run.get("output_dir") == out_dir:
                    s._run["state"] = "completed"
                hid = s._run.get("history_id") if s._run.get("output_dir") == out_dir else ""
            hid = hid or next((r.get("id") for r in s.history.runs() if r.get("lora") == name      # after a
                               and os.path.normcase(r.get("output_dir") or "") == os.path.normcase(out_dir)), "")  # restart
            rem = remote or {}                   # Runs & outputs: finished, at the epoch it reached there
            s.history.end(hid, "finished", {"epoch": rem.get("epochs") or rem.get("epoch"),
                                            "total_epochs": rem.get("epochs")} if rem.get("epoch") else None)
            s.append_log(f"[move] {name} finished on {to}: its LoRAs and previews are here now.\n")
            _back_done(url, to, delete, True)
            return
        with s._lock:
            s._run.update(state="paused", resume=state, resume_mode="state", family=fam, values=values,
                          image_folder=folder, lora=name, output_dir=out_dir)
        r = s._start(fam, values, folder, resume=state,
                     note=f"=== Back from {to}: carrying on from {os.path.basename(state)} ===")
        if not r.get("ok"):
            s.append_log(f"[move] It's back, but couldn't start here: "
                         f"{r.get('error') or '; '.join(r.get('problems') or [])}. Resume carries on.\n")
            s._record_pause(out_dir)
            with s._lock:
                s._run["state"] = "paused"
        _back_done(url, to, delete, True)
    except Exception as e:
        logger.warning(f"[move] back from {to}: {e}")
        _mv.update(state="moved", error=_plain(e))
        _mv_say(f"Bringing it back didn't work ({_plain(e)}): it's still on {to}. Try again, or Carry on here")
        threading.Thread(target=_move_watch, daemon=True, name="helpers-move-watch").start()
