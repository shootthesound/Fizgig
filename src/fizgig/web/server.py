"""Fizgig's web UI server: the training workflow in a browser, for cloud GPU pods and headless machines.

Standard library only. The page is built from what Fizgig declares: every registered family's description and its
form (fizgig.families.form), so a new family, option or preset reaches the web UI with no change here. A run starts
through fizgig.families.launch, exactly as the desktop app starts it.

Access: on a pod (a password from --password, VNC_PASSWORD or FIZGIG_PASSWORD) it listens on every address and asks
for the password; without one it listens on localhost only. Either way every API call carries the page's session
token (in the served page, unreadable to other sites) and a localhost server checks the Host header, so a web page
elsewhere cannot drive it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families
from fizgig.web import captions, history
from fizgig.web import queue as webqueue
from fizgig.web import prep_jobs
from fizgig.families import refmod
from fizgig.web import studio
from fizgig.web.port import PORT_DEFAULT, PORT_WALK, listening, on_pod, saved_port
from fizgig.web.schema import get_schema

logger = logging.getLogger("fizgig.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
from fizgig import user_data  # noqa: E402
PREFS_FILE = user_data.PREFS_FILE
STATE_FILE = user_file("web_state.json")      # the web UI's own memory (git-ignored)
DATASET_DIR = user_file("dataset")          # each run's dataset TOML (written at Start)
OUTPUT_DEFAULT = os.path.join(BASE_DIR, "output_loras")
MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif", ".wav", ".mp3", ".flac", ".m4a", ".mp4",
         ".webm"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

TOKEN = secrets.token_urlsafe(24)       # this server's session token, placed in the page it serves
_SESSION_KEY = [""]          # the logins' key, kept in prefs.json so a login outlives a restart; Log out everywhere
                             # replaces it

_lock = threading.Lock()
_login_lock = threading.Lock()
# Open on your phone: one-time passes in the QR code -> when made. Each opens the page logged in once, for a minute.
_phone_passes: dict = {}
PHONE_PASS_SECONDS = 60


def _phone_pass_new() -> str:
    now = time.time()
    with _login_lock:
        for k in [k for k, t in _phone_passes.items() if now - t > PHONE_PASS_SECONDS]:
            del _phone_passes[k]
        p = secrets.token_urlsafe(18)
        _phone_passes[p] = now
        return p


def _phone_pass_take(p) -> bool:
    """True once for a pass made in the last minute; it is gone after."""
    with _login_lock:
        t = _phone_passes.pop(p, None)
    return t is not None and time.time() - t <= PHONE_PASS_SECONDS


_login_fails: dict = {}    # client address -> (wrong guesses, locked until, last miss)
_run = {"proc": None, "state": "idle", "stage": "", "lora": "", "output_dir": "", "family": "", "values": {},
        "image_folder": "", "pause": False}
_log: list[tuple[int, str]] = []        # (line number, text), the last 5000
_log_n = [0]
_log_event = threading.Condition(_lock)
_stats = {"vram": [0, 0], "ram": [0, 0], "vram_peak": 0, "ram_peak": 0, "device_name": "GPU"}


# no console window for helper commands: started from the desktop app the web UI has none, so each
# nvidia-smi / taskkill would flash one up on Windows
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# ---- hardware meters (as the desktop status bar) ---------------------------------------------------------------
def list_gpus():
    """[{index, name, total_gb, used_gb, uuid}] from nvidia-smi (every card, whatever CUDA_VISIBLE_DEVICES says)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,uuid,utilization.gpu,"
                              "temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4, **_NO_WINDOW)

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        rows = []
        for line in out.stdout.strip().splitlines():
            i, n, t, u, uid, *more = [x.strip() for x in line.split(",")]
            util, temp, pw, pl = (more + [""] * 4)[:4]
            rows.append({"index": int(i), "name": n, "total_gb": int(t) / 1024, "used_gb": int(u) / 1024,
                         "uuid": uid, "util": num(util), "temp": num(temp), "power": num(pw), "power_limit": num(pl)})
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def _chosen_gpu(rows, want):
    """The row for Preferences' cuda_device (an index or a GPU-… uuid), else the first card."""
    want = str(want or os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    for r in rows:
        if want and (want == str(r["index"]) or want == r["uuid"]):
            return r
    return rows[0] if rows else None


def _poll_stats_loop():
    try:
        import psutil
    except ImportError:
        psutil = None
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
    except Exception:
        pass
    while True:
        vu = vt = 0
        name = "GPU"
        extra = {}
        want = str(os.environ.get("CUDA_VISIBLE_DEVICES") or load_prefs().get("cuda_device") or "").split(",")[0]
        want = want.strip()
        try:
            if nvml is not None:
                h = (nvml.nvmlDeviceGetHandleByUUID(want) if want.startswith("GPU-") else
                     nvml.nvmlDeviceGetHandleByIndex(int(want) if want.isdigit() else 0))
                m = nvml.nvmlDeviceGetMemoryInfo(h)
                vu, vt = int(m.used), int(m.total)
                name = nvml.nvmlDeviceGetName(h)
                name = name.decode("utf-8", "ignore") if isinstance(name, bytes) else name
        except Exception:
            pass
        if not vt:                                   # no pynvml: nvidia-smi, as the desktop status bar falls back
            r = _chosen_gpu(list_gpus(), want)
            if r:
                name, vu, vt = r["name"], int(r["used_gb"] * 2 ** 30), int(r["total_gb"] * 2 ** 30)
                extra = {k: r.get(k) for k in ("util", "temp", "power", "power_limit")}
        ru = rt = 0
        if psutil is not None:
            vm = psutil.virtual_memory()
            ru, rt = vm.total - vm.available, vm.total
        with _lock:
            _stats.update(vram=[vu, vt], ram=[ru, rt], device_name=name, vram_peak=max(_stats["vram_peak"], vu),
                          ram_peak=max(_stats["ram_peak"], ru), **extra)
            mine = {"running": _run["proc"] is not None and _run["proc"].poll() is None, "state": _run["state"],
                    "lora": _run["lora"], "output_dir": _run["output_dir"]}
        time.sleep(1.0)


def append_log(text: str):
    with _lock:
        _log_n[0] += 1
        _log.append((_log_n[0], text))
        del _log[:-5000]
        _log_event.notify_all()


# ---- prefs and the web UI's own state ---------------------------------------------------------------------------
def _read_json(path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# the desktop's DEFAULT_PREFS that aren't empty: a web-only install starts where the desktop app would
PREF_DEFAULTS = {"lora_output_dir": "output_loras", "profiles_dir": "user_data/profiles", "cache_dir": "user_data/cache",
                 "runpod_stop_when_done": "0", "inference_blocks_to_swap": "Auto (detect from GPU)",
                 "inference_int8": "1"}


LAST_USED_FILE = user_data.LAST_USED_FILE           # the training folder, each folder's trigger word


def load_last_used() -> dict:
    """The desktop's .last_used.json: the training folder both apps show (from an old key if that is all there is,
    as the desktop reads it) and each folder's trigger word."""
    d = _read_json(LAST_USED_FILE)
    if not d.get("image_folder"):
        for k in ("caption_folder", "dataset_image_dir", "image_prep_source"):
            if d.get(k):
                d["image_folder"] = d[k]
                break
    return d


def update_last_used(changes: dict):
    with _lock:
        d = _read_json(LAST_USED_FILE)
        d.update(changes)
        try:
            _write_json(LAST_USED_FILE, d)
        except OSError:
            pass


def _trigger_key(folder):
    return os.path.normcase(os.path.abspath(folder)) if str(folder or "").strip() else ""


def folder_trigger(folder) -> str:
    """The training folder's trigger word (one per folder, shared with the desktop's Captions tab)."""
    m = load_last_used().get("folder_triggers")
    return str((m if isinstance(m, dict) else {}).get(_trigger_key(folder), "") or "")


def subject_trigger(folder, ext=".txt") -> str:
    """A subject folder's trigger word: the one saved for it on the Captions tab, else the word its captions open
    with (a folder never opened on the Captions tab)."""
    from fizgig.families import multi_person as mp
    return folder_trigger(folder).rstrip(",").strip() or (mp.caption_trigger(folder, ext) if folder else "")


def set_folder_trigger(folder, word):
    k = _trigger_key(folder)
    if not k:
        return
    with _lock:
        d = _read_json(LAST_USED_FILE)
        m = d.get("folder_triggers") if isinstance(d.get("folder_triggers"), dict) else {}
        m[k] = str(word or "").strip()
        d["folder_triggers"] = m
        try:
            _write_json(LAST_USED_FILE, d)
        except OSError:
            pass


# the Preferences settings Reset to defaults puts back (the model files are never reset)
SETTING_PREFS = ("lora_output_dir", "profiles_dir", "cache_dir", "input_lora_dir", "input_ref_dir",
                 "inference_blocks_to_swap", "inference_int8")


def load_prefs() -> dict:
    return {**PREF_DEFAULTS, **_read_json(PREFS_FILE)}


_skey_lock = threading.Lock()


def _session_key() -> str:
    with _skey_lock:
        if not _SESSION_KEY[0]:
            k = str(load_prefs().get("web_session_key") or "")
            if len(k) < 24:
                k = secrets.token_urlsafe(32)
                try:
                    update_prefs({"web_session_key": k})
                except OSError:
                    pass                     # a read-only folder: logins last until the server stops
            _SESSION_KEY[0] = k
        return _SESSION_KEY[0]


def update_prefs(changes: dict):
    """Only the given keys, read fresh and written atomically: the desktop app may be open on the same files."""
    with _lock:
        p = load_prefs()
        p.update(changes)
        _write_json(PREFS_FILE, p)


def pref_path(prefs, key) -> str:
    """A Preferences path as the desktop resolves it (a folder inside Fizgig is stored relative)."""
    v = str(prefs.get(key) or "").strip()
    return v if not v or os.path.isabs(v) else os.path.normpath(os.path.join(BASE_DIR, v))


PRESETS_DIR = user_data.PRESETS_DIR


class _RefModPresets:
    gui_label = refmod.LABEL
    presets = tuple(refmod.REFMOD_BUILT_IN_PRESETS.items())


def _preset_family(key):
    """What the preset calls need of a model: its name (the folder) and built-in presets."""
    return _RefModPresets if key == refmod.KEY else get_family(key)


def write_all_captions(text, folder, ext, overwrite=False):
    """The desktop's Write captions: the typed line (an edit's instruction, a slider's shared description) as the
    caption of every photo in the training folder. {exists, total} first when some already have one."""
    if not text:
        raise ValueError('Type the line first, e.g. "Apply my concert grade."')
    if not os.path.isdir(folder):
        raise ValueError("Set the training folder on the Start tab first.")
    photos = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in
                    (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if not photos:
        raise ValueError(f"No photos in {folder}.")
    if not ext.startswith("."):
        ext = "." + ext
    existing = [f for f in photos if os.path.exists(os.path.join(folder, os.path.splitext(f)[0] + ext))]
    if existing and not overwrite:
        return {"exists": len(existing), "total": len(photos)}
    for f in photos:
        with open(os.path.join(folder, os.path.splitext(f)[0] + ext), "w", encoding="utf-8") as fh:
            fh.write(text)
    append_log(f"[captions] wrote \"{text}\" as the caption of {len(photos)} photos in {folder}\n")
    return {"ok": True, "written": len(photos)}


def user_presets(desc) -> list:
    """The presets saved for this model, in either app (presets/<model's name>/<name>.json), by name."""
    d = os.path.join(PRESETS_DIR, desc.gui_label)
    try:
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
    except OSError:
        return []


def _code_stamp():
    """The newest change to the web UI's server code (and the shared modules it runs): a page served after an update
    while this server still runs the old code is told to restart it."""
    newest = 0.0
    for d in (os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "families")):
        try:
            newest = max([newest] + [os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".py")])
        except OSError:
            pass
    return newest


_STARTED_CODE = _code_stamp()


def default_output_dir() -> str:
    """The LoRA output folder in Preferences (the desktop's Preferences > Output Directories > LoRA output, the same
    prefs.json key), else output_loras inside Fizgig: where a model's LoRAs go until a folder is picked for it."""
    return pref_path(load_prefs(), "lora_output_dir") or OUTPUT_DEFAULT


def load_state() -> dict:
    return _read_json(STATE_FILE)


def save_state(changes: dict):
    if not changes:
        return
    with _lock:
        s = load_state()
        s.update(changes)
        _write_json(STATE_FILE, s)


# ---- runs -------------------------------------------------------------------------------------------------------
STUDIO_PRESETS = os.path.join(user_data.PRESETS_DIR, "repair_studio")


def _studio_desc(key):
    d = get_family(key)
    if d is None or d not in studio.families():
        raise ValueError("Repair Studio doesn't offer that model here yet.")
    return d


def _studio_samples(desc):
    """The web UI's Samples settings for this model (a workbench that follows the Samples tab previews with them). A
    value the tab never saved is what the tab shows: the Turbo's own steps and CFG while it renders (its file set,
    strength above 0 - Qwen's: 8 steps, CFG 1), else the plain model's (as the Samples tab's Turbo follow does)."""
    v = (load_state().get("values") or {}).get(desc.key) or {}
    blank = lambda x: x is None or str(x).strip() == ""
    sp, spd = desc.preview_speed(), desc.preview_speed_defaults()
    turbo = v.get("FAMILY_TURBO_STRENGTH")
    if blank(turbo):
        turbo = spd[1] if spd else 0
    f = pref_path(load_prefs(), sp.pref_key) if sp is not None and sp.pref_key else ""
    try:
        live = bool(f and os.path.isfile(f) and float(turbo) > 0)
    except (TypeError, ValueError):
        live = False
    steps, cfg = v.get("SAMPLE_STEPS"), v.get("SAMPLE_CFG_SCALE")
    if blank(steps):
        steps = spd[0] if live else desc.preview_steps
    if blank(cfg):
        cfg = f"{sp.settings.cfg:g}" if live else f"{desc.preview_cfg:g}"
    neg = v.get("SAMPLE_NEGATIVE")
    if neg is None:                     # never saved: the Samples tab's own default, as its box shows it
        neg = desc.preview_negative or ""
    return {"steps": str(steps), "cfg": str(cfg), "turbo": str(turbo), "negative": neg,
            "turbo_live": live, "turbo_name": sp.name if sp is not None else ""}


def _gpu_training():
    """A training run is using the GPU: Repair Studio waits for it."""
    with _lock:
        return _run["proc"] is not None and _run["proc"].poll() is None


STUDIO_SETUPS = os.path.join(STUDIO_PRESETS, "setups")      # the desktop's Save / Load Setup folder


def _setup_path(name):
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip()).strip(". ")
    if not name:
        raise ValueError("Give the setup a name.")
    return os.path.join(STUDIO_SETUPS, name if name.lower().endswith(".json") else name + ".json")


def _setups():
    """The saved setups (either app's): [{name, family}], newest first."""
    out = []
    if os.path.isdir(STUDIO_SETUPS):
        for f in os.listdir(STUDIO_SETUPS):
            p = os.path.join(STUDIO_SETUPS, f)
            if not f.lower().endswith(".json"):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and d.get("fizgig_repair_setup"):
                out.append((os.path.getmtime(p), {"name": f[:-5], "family": d.get("family", "")}))
    return [x for _t, x in sorted(out, key=lambda t: -t[0])]


def _pods_post(what, b):
    from fizgig.web import pod_stop, pods
    if on_pod():
        raise ValueError("The Pods tab is for the Fizgig on your own computer.")
    p = load_prefs()
    if what == "add":
        rows = pods.add(p, str(b.get("text") or ""), str(b.get("name") or ""))
        update_prefs({"app_pods": rows})
        return {"ok": True, "pod": rows[0], "state": pods.probe(rows[0]["url"])}
    if what == "forget":
        update_prefs({"app_pods": pods.forget(p, str(b.get("url") or ""))})
        return {"ok": True}
    if what == "opened":
        update_prefs({"app_pods": pods.touch(p, str(b.get("url") or ""), str(b.get("name") or ""),
                                             str(b.get("lan_id") or ""))})
        return {"ok": True}
    if what == "key":                        # checked against RunPod before it is kept
        key = str(b.get("key") or "").strip()
        if key:
            found = pods.account(key)
            update_prefs({"runpod_api_key": key})
            return {"ok": True, "found": len(found)}
        update_prefs({"runpod_api_key": ""})
        return {"ok": True, "found": 0}
    key = pod_stop.api_key(p)
    if what in ("start", "stop"):
        if not key:
            raise ValueError("Starting and stopping pods from here needs your RunPod API key (below).")
        pid = str(b.get("id") or "")
        if not pods.POD_ID.match(pid):
            raise ValueError("No such pod.")
        if what == "start":
            pods.start(key, pid, int(b.get("gpus") or 1))
        else:
            pods.stop(key, pid)
        return {"ok": True}
    raise ValueError("Unknown pods action.")


def _studio_post(what, b):
    """Repair Studio's actions (fizgig/web/studio.py runs them in its worker)."""
    if what == "setup_save":                 # the whole session in one file, the desktop's format (Save Setup)
        data = b.get("data") or {}
        if not isinstance(data, dict) or not data.get("fizgig_repair_setup"):
            raise ValueError("Not a setup.")
        p = _setup_path(b.get("name"))
        if os.path.exists(p) and not b.get("overwrite"):
            return {"exists": True, "name": os.path.basename(p)[:-5]}
        os.makedirs(STUDIO_SETUPS, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, p)
        return {"ok": True, "name": os.path.basename(p)[:-5]}
    if what == "load":
        if _gpu_training():
            raise ValueError("A training run is using the GPU. Repair Studio can load when it finishes.")
        from fizgig.web import tools
        if tools.holding_gpu():
            raise ValueError("The Profiler is rendering on the GPU. Studio can load when it finishes (or Stop it).")
        desc = _studio_desc(b.get("family", ""))
        prefs = load_prefs()
        if desc.video_workbench:
            settings = None
            kw, status, problems = studio.plan_video(desc, prefs, pref_path, checkpoint=str(b.get("h3_model") or ""),
                                                     base=str(b.get("base") or ""))
        else:
            kw, settings, status, problems = studio.plan(desc, prefs, pref_path, fast=b.get("fast", True) is not False,
                                                         samples=_studio_samples(desc), negative=b.get("negative"))
        if problems:
            raise ValueError(" ".join(problems))
        captions.unload()                       # the caption model leaves the GPU too
        studio.load(desc, str(b.get("primary") or "").strip(), str(b.get("donor") or "").strip(), kw, settings,
                    status, b.get("turbo", True), canvas=b.get("canvas") if desc.video_workbench else None)
        return {"ok": True}
    if what == "keep":
        return {"path": studio.keep_tweaked()}
    if what == "render":
        return studio.render(b.get("state") or {}, extra={k: b.get(k) for k in ("video", "audio", "keyframes", "ref_mode", "mix")
                                                          if b.get(k) is not None})
    if what == "cancel":
        studio._send({"cmd": "cancel"})
        return {"ok": True}
    if what == "donor":
        studio.donor(str(b.get("path") or "").strip())
        return {"ok": True}
    if what == "settings":
        desc = _studio_desc(b.get("family", ""))
        ps = None
        if desc.workbench_follows_samples and "negative" in b:
            ps = studio.plan(desc, load_prefs(), pref_path, samples=_studio_samples(desc), negative=b.get("negative"))[1]
        studio.settings(turbo=b.get("turbo"), preview_settings=ps)
        return {"ok": True}
    if what == "save":
        out = str(b.get("out") or "").strip()
        if not out.lower().endswith(".safetensors"):
            raise ValueError("Give the file a name ending in .safetensors.")
        if not os.path.isdir(os.path.dirname(os.path.abspath(out))):
            raise ValueError("That folder isn't there.")
        if os.path.exists(out) and not b.get("overwrite"):
            return {"exists": True}
        studio.save(out, b.get("state") or {}, b.get("include_donor"))
        return {"ok": True}
    if what == "reset":
        studio.reset()
        return {"ok": True}
    if what in ("preset_save", "preset_load"):
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(b.get("family") or ""))
        name = str(b.get("name") or "").strip()        # any name a file can have, as the desktop's
        if not name or name.startswith("✨") or any(c in name for c in '<>:"/\\|?*') or name in (".", ".."):
            raise ValueError('Give the preset a name (not < > : " / \\ | ? *).')
        path_ = os.path.join(STUDIO_PRESETS, key, name + ".json")
        if what == "preset_load":
            with open(path_, encoding="utf-8") as f:
                return {"blocks": (json.load(f) or {}).get("blocks") or {}}
        if os.path.exists(path_) and not b.get("overwrite"):
            return {"exists": True}
        os.makedirs(os.path.dirname(path_), exist_ok=True)
        with open(path_, "w", encoding="utf-8") as f:     # the desktop's format: {"blocks": {...}, "family": ...}
            json.dump({"blocks": b.get("blocks") or {}, "family": b.get("family")}, f, indent=2)
        return {"ok": True}
    raise ValueError("unknown action")


def _rm_metas(b):
    """RefMod Toolkit: the folder's mods (and the LoRA's, a ReFLoRA's) by their picker names."""
    from fizgig.web import refmod_studio as rms
    return {m["name"]: m for m in rms.scan(str(b.get("folder") or "").strip(), str(b.get("lora") or "").strip())["mods"]}


def _rm_family(b):
    from fizgig.web import refmod_studio as rms
    fams = {d.key: d for d in rms.families()}
    desc = fams.get(str(b.get("family") or "")) or next(iter(fams.values()), None)
    if desc is None:
        raise ValueError("No model offers RefMod Toolkit.")
    return desc


def _rm_plan(b):
    """(desc, load plan, audio VAE there) for the model and base the page picked."""
    desc = _rm_family(b)
    kw, _status, problems = studio.plan_video(desc, load_prefs(), pref_path, checkpoint=str(b.get("model") or ""),
                                              base=str(b.get("base") or ""), refmod=True)
    if problems:
        raise ValueError(" ".join(problems))
    return desc, kw, bool(kw.get("audio_vae_path"))


def _refmod_post(what, b):
    """RefMod Toolkit's actions (fizgig/web/refmod_studio.py; renders on the Studio worker)."""
    from fizgig.web import refmod_studio as rms
    st = b.get("state") or {}
    if what == "calc":                       # the tokens readout, reference map, ComfyUI readout, prompt hints
        metas = _rm_metas(st)
        desc = _rm_family(st)
        try:
            audio_ok = bool(_rm_plan(st)[2])
        except ValueError:
            audio_ok = False
        j = rms.job(st, metas, audio_ok, desc.video_controls)
        return {"tokens": rms.tokens(st.get("rows"), metas, st.get("retention")), "ref_map": j["ref_map"],
                "readout": j["readout"], "hints": rms.prompt_hints(st.get("rows"), metas), "describe": j["describe"]}
    if what in ("render", "sweep"):
        if _gpu_training():
            raise ValueError("A training run is using the GPU. RefMod Toolkit can render when it finishes.")
        from fizgig.web import tools
        if tools.holding_gpu():
            raise ValueError("The Profiler is rendering on the GPU - wait for it, or Stop it.")
        desc, kw, audio_ok = _rm_plan(st)
        metas = _rm_metas(st)
        vc = desc.video_controls
        if what == "render":
            j = rms.job(st, metas, audio_ok, vc)
            jobs, label = [j["job"]], j["describe"]
        else:
            kind = str(b.get("sweep") or rms.SWEEPS[0])
            js = [rms.job(st, metas, audio_ok, vc, dict(ov, frames=vc.sweep_frames, label=lbl))
                  for lbl, ov in rms.sweep_items(kind, st)]
            for x in js:
                x["job"]["sound"] = False
            j, jobs, label = js[0], [x["job"] for x in js], kind
        if not j["job"]["prompt"]:
            raise ValueError("Enter a prompt.")
        n_refs = sum(r["copies"] for r in j["job"]["rows"] if r["on"] and r["value"] > 0)
        canvas = [j["job"]["width"], j["job"]["height"], j["job"]["frames"], 0, n_refs]
        captions.unload()
        got = studio.refmod_render(desc, kw, canvas, what, jobs, j["lora"], j["compare"], j["base_key"], label=label)
        return dict(got, active=j["active"])
    if what == "cancel":
        return {"ok": studio.refmod_cancel()}
    if what == "lora":
        return rms.lora_info(str(b.get("path") or ""), str(b.get("folder") or ""))
    if what == "curve_save":
        return rms.curve_preset_save(b.get("name"), st.get("frame_curve"), st.get("step_curve"), st.get("step_on"),
                                     st.get("retention"), overwrite=bool(b.get("overwrite")))
    if what == "curve_load":
        return rms.curve_preset_load(b.get("name"))
    if what == "curve_delete":
        return rms.curve_preset_delete(b.get("name"))
    if what == "graph":
        return {"curve": rms.graph_preset(str(b.get("path") or ""))}
    if what == "setup_save":
        return rms.setup_save(b.get("name"), b.get("data") or {}, overwrite=bool(b.get("overwrite")))
    if what == "setup_load":
        return {"data": rms.setup_load(b.get("name"))}
    if what == "save_preview":
        it = studio.refmod_item(b.get("gen", 0), b.get("side") or "tweaked", b.get("i", 0))
        if it is None:
            raise ValueError("Nothing rendered yet.")
        return {"path": rms.save_item(it, str(b.get("out") or ""), "refmod_with_mods" if it["side"] == "tweaked"
                                      else "refmod_" + re.sub(r"[^A-Za-z0-9_-]+", "_", it.get("label") or it["side"]))}
    if what == "save_strip":
        return {"path": rms.save_strip(studio.refmod_chips(), str(b.get("out") or ""), studio.refmod_status()["sweep"])}
    if what == "bake":
        rows = st.get("rows") or []
        i = int(b.get("row", 0))
        if not 0 <= i < len(rows):
            raise ValueError("Pick the row to bake.")
        return rms.bake(rows[i], _rm_metas(st), st.get("retention"), st.get("frame_curve"), str(b.get("name") or ""))
    if what == "reflora":
        return rms.save_reflora(st.get("lora"), st.get("rows"), _rm_metas(st), st.get("retention"),
                                str(b.get("name") or ""))
    raise ValueError("unknown action")


def _files_family(key):
    """The family whose files a model runs on: RefMod runs on MiniMax H3's."""
    return get_family("minimax" if key == refmod.KEY else key)


def _refmod_plan(values, image_folder):
    """MiniMax H3 RefMod's launch (fizgig/families/refmod.py, the desktop's own): H3's files, the card's values."""
    prefs = load_prefs()
    h3 = get_family("minimax")
    values = dict(values or {})
    name = str(values.get("LORA_NAME") or "").strip() or "my_refmod"
    out_dir = str(values.get("LORA_OUTPUT_DIR") or "").strip() or default_output_dir()
    base = next((o for o in h3.options if o.key == "H3_TRAIN_BASE"), None)
    st = {**refmod.REFMOD_DEFAULTS, "MINIMAX_REFMOD_GRID": refmod.REFMOD_GRID_OPTIONS[0], "SEED": 42,
          "BLOCKS_SWAP": "auto", "DATASET_MEGAPIXELS": "0.25",
          **{k: v for k, v in values.items() if not str(k).startswith("__")},
          "LORA_NAME": name, "LORA_OUTPUT_DIR": out_dir,
          "DATASET_CONFIG": os.path.join(DATASET_DIR, f"{_safe(name)}_web.toml"),
          "MINIMAX_TRAIN_BASE": refmod.minimax_train_base(values.get("H3_TRAIN_BASE") or (base.default if base else ""))}
    models = {k: pref_path(prefs, k) for k in form.model_keys(h3)}
    p = refmod.plan(st, models, python=sys.executable, repo_dir=BASE_DIR, image_folder=str(image_folder or "").strip(),
                    cache_root=pref_path(prefs, "cache_dir"), cache_dir_for=launch.cache_dir_for,
                    caching=values.get("ENABLE_CACHE", True) is not False)
    return None, st, p


def _multi_person_notes(vals, image_folder, multi_person=True):
    """The Subjects card's line under each subject: its trigger word, or (other people keeping their faces) the word
    it trains as and what it was read as - the subject's own boxes override - or why it can't be made."""
    from fizgig.families import multi_person as mp
    ext = str(vals.get("DATASET_CAPTION_EXT") or ".txt").strip() or ".txt"
    rows = [(str(image_folder or "").strip(), "FAMILY_DOP_TRIGGER", "FAMILY_DOP_CLASS")]
    if vals.get("FAMILY_MULTICONCEPT"):
        rows += [(str(vals.get(d) or "").strip(), t, c)
                 for d, t, c in form.CONCEPT_ROWS[:form.concept_count(vals) - 1]]
    notes, classes = {}, []
    dop = bool(vals.get("FAMILY_DOP"))
    fused_rules = []
    found = {}                     # the correction boxes' greyed text: the trigger word and class Fizgig read
    for n, (folder, tkey, ckey) in enumerate(rows, start=1):
        if not folder:
            notes[f"_subject{n}_as"] = ("Pick the training folder on the Start tab to see it here." if n == 1 else
                                        f"Pick subject {n}'s folder above to see it here.")
            continue
        trig = str(vals.get(tkey) or "").strip() or subject_trigger(folder, ext)
        cls = str(vals.get(ckey) or "").strip()
        auto_trig = subject_trigger(folder, ext)
        if auto_trig:
            found[tkey] = auto_trig
            got, _w = mp.subject(folder, auto_trig, "", ext)
            if got:
                found[ckey] = got["phrase"].split()[-1]
        cap = mp.sample_caption(folder, trig.split(",")[0].strip() if trig else "", ext)
        lines = [f"Trigger word: {trig}" if trig else "Trigger word: none found in this folder's captions",
                 f"One of its captions: {cap}" if cap else "One of its captions: none found - caption the folder first"]
        if not multi_person:
            if dop and cap and trig:
                pairs = list(zip([t.strip() for t in trig.split(",") if t.strip()],
                                 [c.strip() for c in cls.split(",")] * 8))
                kept = mp.dop_caption(cap, pairs)
                lines += [f"Trains on: {cap}", f"And keeps as the model made it: {kept}" if kept else
                          "And keeps as the model made it: nothing - this caption has no trigger word"]
            elif cap:
                lines.append(f"Trains on: {cap} (as written)")
            notes[f"_subject{n}_as"] = "\n".join(lines)
            continue
        manual = bool(vals.get("FAMILY_MP_MANUAL"))
        kind = mp.kind_key(vals.get("FAMILY_SUBJECT_KIND" + ("" if n == 1 else str(n))))
        s, why = mp.subject_as_written(trig, cls, kind) if manual else mp.subject(folder, trig, cls, ext, kind)
        if not s:
            notes[f"_subject{n}_as"] = "\n".join(lines + [f"Can't keep other people's faces with this folder: {why}"])
            continue
        if s["phrase"]:
            classes.append(s["phrase"])
        fused_rules += s["rules"]
        lines[0] = (f"Trains as: {s['fused']} - a style or look, its trigger word kept as written. Use {s['fused']} "
                    "in your prompts; without it, pictures stay as the model made them." if s["kind"] == "style" else
                    f"Trains as: {s['fused']} - used as your captions write it ({s['phrase']}). Use "
                    f"{s['fused']} in your prompts." if manual else
                    f"Trains as: {s['fused']} - Fizgig joins the trigger word to what the subject is "
                    f"(\"{s['trigger']}, {s['phrase']}\" becomes \"{s['fused']}\"), which keeps the subjects apart. "
                    f"Your caption files aren't changed. Use {s['fused']} in your prompts.")
        if manual and cap and not re.search(r"(?<!\w)" + re.escape(s["fused"]) + r"(?!\w)", cap, re.I):
            lines.append(f"Check your captions: this one never uses {s['fused']} - with Use my captions exactly as "
                         f"written on, write {s['fused']} into them (see How to caption for this).")
        if cap:
            fused = mp.fuse(cap, s["rules"])
            kept = mp.dop_caption(fused, [(s["fused"], s["phrase"])])
            who = mp.stranger_example() or "one of 32 example people, described"
            lines += [f"Trains on: {fused}",
                      f"And keeps as the model made it: {kept}" if kept else
                      "And keeps as the model made it: nothing - this caption has no trigger word"]
            lines.append(f"Plus {mp.PER_SUBJECT} side-by-side pictures of {s['fused']} next to a stranger, e.g.: {who} "
                         f"on the left, and {s['fused']} on the right, {mp.TAIL}" if s["kind"] == "person" else
                         f"No side-by-side pictures: {s['fused']} isn't a person"
                         + (" (Subject {} is: From its captions found {})".format(n, s["phrase"])
                            if kind == "auto" and s["phrase"] else "") + ".")
        notes[f"_subject{n}_as"] = "\n".join(lines)
    notes["__placeholders"] = found
    if multi_person and fused_rules:          # the previews' prompts, as they will be read
        prompts = [ln.strip() for ln in str(vals.get("SAMPLE_PROMPT") or "").splitlines() if ln.strip()]
        read = [mp.fuse(p, fused_rules) for p in prompts]
        if any(a != b for a, b in zip(prompts, read)):
            notes["SAMPLE_PROMPT"] = "Previews will use: " + " | ".join(read)
    same = sorted({c for c in classes if classes.count(c) > 1})
    if same:                 # two women measured: right at epoch 16 where a woman and a man were at 10
        plural = lambda w: {"woman": "women", "man": "men", "child": "children", "person": "people"}.get(w, w + "s")
        words = {2: "Two", 3: "Three", 4: "Four"}
        kinds_ = " and ".join(f"{words.get(classes.count(c), classes.count(c))} of your subjects are "
                              f"{plural(c.split()[-1])}" for c in same)      # "Two of your subjects are women"
        notes["MAX_TRAIN_EPOCHS"] = f"{kinds_}: subjects of the same kind can need about 1.6x the epochs."
    return notes


def _plan(family_key, values, image_folder, resume="", ft_continue=None):
    if family_key == refmod.KEY:
        return _refmod_plan(values, image_folder)
    desc = get_family(family_key)
    if desc is None:
        raise ValueError(f"Unknown family: {family_key}")
    prefs = load_prefs()
    values = dict(values or {})
    if resume:
        values["RESUME_TRAINING"] = resume
    trig = folder_trigger(image_folder)
    if trig:                                         # the folder's own trigger word (Captions tab)
        values["CAPTION_TRIGGER_WORD"] = trig.rstrip(",").strip()
    # DOP's word to remove when a folder's box is empty: that folder's own trigger word
    ext = str(values.get("DATASET_CAPTION_EXT") or ".txt").strip() or ".txt"
    values["DOP_FOLDER_TRIGGER"] = subject_trigger(image_folder, ext) if image_folder else ""
    for i, (d, _t, _c) in enumerate(form.CONCEPT_ROWS, start=2):
        concept = str(values.get(d) or "").strip()
        values[f"DOP_FOLDER_TRIGGER{i}"] = subject_trigger(concept, ext) if concept else ""
    name = str(values.get("LORA_NAME") or "").strip() or "my_lora"
    out_dir = str(values.get("LORA_OUTPUT_DIR") or "").strip() or default_output_dir()
    values.update(LORA_OUTPUT_DIR=out_dir, DATASET_CONFIG=os.path.join(DATASET_DIR, f"{_safe(name)}_web.toml"))
    captioner = pref_path(prefs, "krea2_text_encoder")
    raw_ovr = prefs.get("caption_qwen_instructions")
    inputs = form.inputs(
        desc, values, models={k: pref_path(prefs, k) for k in form.model_keys(desc)}, python=sys.executable,
        repo_dir=BASE_DIR, image_folder=str(image_folder or "").strip(),
        cache_root=pref_path(prefs, "cache_dir"), samples_dir=os.path.join(out_dir, "sample"),
        inference_int8=str(prefs.get("inference_int8", "")).strip() in ("1", "True", "true"),
        captioner=captioner if captioner and os.path.isfile(captioner) else "",
        caption_overrides=dict(raw_ovr) if isinstance(raw_ovr, dict) else {})
    if ft_continue:          # a paused fine-tune: from its checkpoint, for the rotations left (the cache is built)
        inputs.update(FAMILY_FT_CONTINUE=ft_continue["continue"], FAMILY_FT_ROTATIONS=str(ft_continue["rotations"]),
                      resuming=True)
    if resume or ft_continue or str(values.get("RESUME_TRAINING") or "").strip():
        inputs["recache"] = _size_changed(name, out_dir, values)
    return desc, inputs, launch.plan(desc, inputs)


_SIZE_KEYS = ("DATASET_MEGAPIXELS", "CLIP_MEGAPIXELS")


# a prompt slider's live prompts: the request the web UI writes, and what the trainer took (fizgig.families.train's
# SLIDER_LIVE / SLIDER_LIVE_APPLIED - named here so the server never imports torch)
SLIDER_LIVE, SLIDER_LIVE_APPLIED = ".slider_prompts.json", ".slider_prompts_applied.json"
_SLIDER_LIVE_KEYS = {"base": "FAMILY_SLIDER_BASE", "pos": "FAMILY_SLIDER_POS", "neg": "FAMILY_SLIDER_NEG",
                     "guidance": "FAMILY_SLIDER_GUIDANCE"}


def _prompt_slider_run(run) -> bool:
    """The run going is a prompt slider (its prompts can change while it trains)."""
    v = run.get("values") or {}
    desc = get_family(run.get("family") or "")
    if desc is None or not getattr(desc, "slider_training", False) or form._kind(desc, v) != "slider":
        return False                                  # the run's values hold the Kind label, as the form does
    srcs = form.slider_sources(desc)
    return srcs == ["prompts"] or (len(srcs) > 1 and str(v.get("FAMILY_SLIDER_SOURCE") or "") == "Prompts")


def _slider_live_status(run):
    """{base, pos, neg, guidance, epoch (when the trainer last took new ones, else None), waiting} for the Training
    tab's Slider prompts panel, or None for any other run."""
    if not _prompt_slider_run(run):
        return None
    v, out = run.get("values") or {}, run.get("output_dir") or ""
    want = _read_json(os.path.join(out, SLIDER_LIVE)) if out else {}
    got = _read_json(os.path.join(out, SLIDER_LIVE_APPLIED)) if out else {}
    same = lambda a, b: all(str(a.get(k) or "") == str(b.get(k) or "") for k in ("base", "pos", "neg")) and         float(a.get("guidance") or 0) == float(b.get("guidance") or 0)
    return {**{k: str(v.get(key) or "") for k, key in _SLIDER_LIVE_KEYS.items()}, "epoch": got.get("epoch"),
            "waiting": bool(want) and not (got and same(want, got))}


def _size_changed(lora, out_dir, values):
    """A resume that trains at another size from now on (Target Megapixels, a clip family's clip size) caches again
    first; a run Fizgig has no record of does too, to be safe. Unchanged: the cache is used as it is."""
    from fizgig.web import history
    old = history.last_values(lora, out_dir)
    if old is None:
        return True

    def num(v):
        try:
            return round(float(str(v).strip()), 4)
        except ValueError:
            return str(v or "").strip()
    return any(num(old.get(k, "")) != num(values.get(k, "")) for k in _SIZE_KEYS)


def _ref_strength(v):
    """Royale's reference strength: 0 to 2 (0 = the reference off), 1 when there is none - as the desktop's."""
    try:
        return max(0.0, min(2.0, float(str(v).strip())))
    except (TypeError, ValueError):
        return 1.0


def _seed(v, default):
    """A seed from the page: its number (0 is a seed too), the default when there is none."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80] or "run"


_meta_cache: dict = {}         # path -> (mtime, metadata) - a folder of LoRAs is read once, not on every look


def _safetensors_meta(path) -> dict:
    """A .safetensors file's metadata header (a few KB, never the tensors)."""
    try:
        mtime = os.path.getmtime(path)
        hit = _meta_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            meta = json.loads(f.read(n)).get("__metadata__") or {} if 0 < n < (64 << 20) else {}
    except (OSError, ValueError):
        meta = {}
    _meta_cache[path] = (mtime if "mtime" in locals() else 0, meta)
    return meta


def _fizgig_made(meta) -> bool:
    """Trained by Fizgig: its LoRAs name it as their network module, its fine-tunes carry fizgig_finetune."""
    return "fizgig" in str(meta.get("ss_network_module", "")).lower() or bool(meta.get("fizgig_finetune"))


def output_folders() -> list:
    """Every folder Fizgig saves LoRAs and fine-tunes to: the web UI's per-model output folders, the desktop app's
    (its .last_used.json lora_output_dirs), the default output_loras, and the runs going now."""
    found = [OUTPUT_DEFAULT, default_output_dir()]
    for vals in (load_state().get("values") or {}).values():
        if isinstance(vals, dict) and vals.get("LORA_OUTPUT_DIR"):
            found.append(vals["LORA_OUTPUT_DIR"])
    desk = _read_json(LAST_USED_FILE).get("lora_output_dirs")      # the old desktop app's per-model folders
    if isinstance(desk, dict):
        found += [v for v in desk.values() if isinstance(v, str) and v]
    with _lock:
        found.append(_run.get("output_dir") or "")
    out, seen = [], set()
    for f in found:
        f = pref_path({"x": f}, "x") if f else ""
        key = os.path.normcase(os.path.abspath(f)) if f else ""
        if f and key not in seen and os.path.isdir(f):
            seen.add(key)
            out.append(os.path.abspath(f))
    return out


_CKPT = re.compile(r"^(.*)-(?:(\d{6})|step(\d{7}))$")   # an epoch's checkpoint, or a Save now (-step0001234)


def list_outputs(fizgig_only=True) -> dict:
    """Runs, newest first: {name, folder, files (the final first, then its checkpoints newest first), time,
    samples (how many previews)}; `hidden` counts the other LoRAs left out when fizgig_only."""
    from concurrent.futures import ThreadPoolExecutor
    runs, hidden, now = {}, 0, time.time()
    for folder in output_folders():
        try:
            entries = [e for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(".safetensors")]
        except OSError:
            continue
        entries = [e for e in entries if now - e.stat().st_mtime >= 5]       # not one still being written
        # each file's header, read side by side (a ComfyUI loras folder can hold a thousand LoRAs; one at a time
        # that is a long wait on a slow disk), then remembered until the file changes
        with ThreadPoolExecutor(16) as pool:
            metas = dict(zip((e.path for e in entries), pool.map(lambda e: _safetensors_meta(e.path), entries)))
        previews = _folder_previews(folder)
        for e in entries:
            st = e.stat()
            meta = metas[e.path]
            if fizgig_only and not _fizgig_made(meta):
                hidden += 1
                continue
            stem = e.name[:-len(".safetensors")]
            m = _CKPT.match(stem)
            base, epoch = (m.group(1), int(m.group(2)) if m.group(2) else None) if m else (stem, None)
            r = runs.setdefault((folder, base), {"name": base, "folder": folder, "files": [], "time": 0})
            r["files"].append({"name": e.name, "path": e.path, "size": st.st_size, "time": st.st_mtime,
                               "epoch": epoch, "kind": "finetune" if meta.get("fizgig_finetune") else "lora"})
            r["time"] = max(r["time"], st.st_mtime)
            r["samples"], r["samples_bytes"] = previews.get(base, (0, 0))
    for r in runs.values():
        r["files"].sort(key=lambda f: (f["epoch"] is not None, -(f["epoch"] or 0), -f["time"]))
    return {"runs": sorted(runs.values(), key=lambda r: -r["time"]), "hidden": hidden,
            "folders": output_folders()}


def _folder_previews(folder) -> dict:
    """{run name: (how many previews, their bytes)} for an output folder's sample/, scanned once: a preview is named
    <run>_e<epoch>_…, so its run is everything before the last "_e<digits>_"."""
    out = {}
    try:
        for e in os.scandir(os.path.join(folder, "sample")):
            if e.is_file() and os.path.splitext(e.name)[1].lower() in MEDIA:
                m = re.match(r"^(.*)_e\d+_", e.name)
                if m:
                    n, b = out.get(m.group(1), (0, 0))
                    out[m.group(1)] = (n + 1, b + e.stat().st_size)
    except OSError:
        pass
    return out


def _run_samples(folder, name):
    """A run's preview files in <output>/sample (they start with the run's name)."""
    sdir = os.path.join(folder, "sample")
    try:
        return sorted(e.path for e in os.scandir(sdir) if e.is_file()
                      and os.path.splitext(e.name)[1].lower() in MEDIA
                      and (re.match(r"^(.*)_e\d+_", e.name) or [None, None])[1] == name)  # not a longer-named run's
    except OSError:
        return []


def _zip_file(prefix, files) -> str:
    """files [(path, name in the zip)] in one stored zip under user_data/zips (a day-old leftover cleared first)."""
    import tempfile
    import zipfile
    zdir = user_file("zips")
    os.makedirs(zdir, exist_ok=True)
    for old in os.scandir(zdir):
        try:
            if time.time() - old.stat().st_mtime > 86400:
                os.remove(old.path)
        except OSError:
            pass
    fd, out = tempfile.mkstemp(prefix=prefix, suffix=".zip", dir=zdir)
    os.close(fd)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p, arc in files:
            z.write(p, arc)
    return out


def dataset_zip(folder, captions_only, ext) -> str:
    """A training folder in one zip, for taking it home from a pod: its pictures, clips and songs with their captions
    (and Fizgig's own notes on it - look scores, exclusions), or the captions alone. The folder's own files only:
    removed/ and excluded_by_look/ stay behind."""
    try:
        entries = [e for e in os.scandir(folder) if e.is_file()]
    except OSError:
        raise ValueError("That folder isn't there.")
    ext = ext.lower()
    keep = [e for e in entries if e.name.lower().endswith(ext) or (not captions_only and (
        os.path.splitext(e.name)[1].lower() in MEDIA or re.fullmatch(r"fizgig_\w+\.json", e.name)))]
    if not keep:
        raise ValueError(f"No {ext} captions in that folder." if captions_only else "That folder is empty.")
    return _zip_file("dataset_", [(e.path, e.name) for e in sorted(keep, key=lambda e: e.name)])


def samples_zip(folder, name) -> str:
    """The run's previews in one zip (stored, not compressed: pictures and songs don't shrink), for downloading."""
    import zipfile
    files = _run_samples(folder, name)
    if not files:
        raise ValueError("This run has no previews.")
    import tempfile
    zdir = user_file("zips")
    os.makedirs(zdir, exist_ok=True)
    for old in os.scandir(zdir):                  # one left by a download that never finished, a day on
        try:
            if time.time() - old.stat().st_mtime > 86400:
                os.remove(old.path)
        except OSError:
            pass
    fd, out = tempfile.mkstemp(prefix="previews_", suffix=".zip", dir=zdir)   # each download its own file
    os.close(fd)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, os.path.basename(p))
    return out


def _checkpoint_kind(path) -> str:
    """'finetune' (a whole model, the source file with the trained tensors replaced) or 'lora', from the file's
    metadata header - a few KB read, never the tensors."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < (64 << 20):
                return "lora"
            meta = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError):
        return "lora"
    return "finetune" if meta.get("fizgig_finetune") else "lora"


def run_outputs(folder, name) -> dict:
    """Everything a run made in its output folder, for Delete run: its LoRAs / checkpoints (<name>.safetensors,
    <name>-NNNNNN.safetensors), its resumable states (<name>-NNNNNN-state, <name>-state) and its previews in sample/.
    Only names that are exactly this run's: a longer-named run in the same folder (name2) is never caught."""
    ckpts, states = [], []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        entries = []
    for e in entries:
        if e.is_file() and e.name.lower().endswith(".safetensors"):
            stem = e.name[:-len(".safetensors")]
            m = _CKPT.match(stem)
            if (m.group(1) if m else stem) == name:
                ckpts.append(e.path)
        elif e.is_dir() and re.fullmatch(re.escape(name) + r"(-\d{6})?-state", e.name):
            states.append(e.path)
    previews = _run_samples(folder, name)

    def size(p):
        if os.path.isdir(p):
            return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
        return os.path.getsize(p)
    total = 0
    for p in ckpts + states + previews:
        try:
            total += size(p)
        except OSError:
            pass
    return {"checkpoints": ckpts, "states": states, "previews": previews, "bytes": total}


def delete_run(run_id, confirm=False) -> dict:
    """Delete run: what it would delete (confirm False), or delete it - its checkpoints, states and previews, and its
    line in the run history. The training folder is never touched. Refused while that run is training."""
    import shutil
    rec = history.get(run_id)
    if rec is None:
        return {"error": "That run isn't in the list any more."}
    folder, name = str(rec.get("output_dir") or ""), str(rec.get("lora") or "")
    same = lambda d, n: (os.path.normcase(os.path.abspath(d or ".")) == os.path.normcase(os.path.abspath(folder or "."))
                         and (n or "").lower() == name.lower())
    with _lock:
        web_live = _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running")
        if web_live and same(_run["output_dir"], _run["lora"]):
            return {"error": "That run is training now. Stop it first."}
    found = run_outputs(folder, name) if folder and name else {"checkpoints": [], "states": [], "previews": [],
                                                              "bytes": 0}
    if not confirm:
        return {"ok": True, "checkpoints": len(found["checkpoints"]), "states": len(found["states"]),
                "previews": len(found["previews"]), "bytes": found["bytes"], "folder": folder}
    failed = []
    for p in found["checkpoints"] + found["previews"]:
        try:
            os.remove(p)
        except OSError:
            failed.append(p)
    for p in found["states"]:
        try:
            shutil.rmtree(p)
        except OSError:
            failed.append(p)
    try:                                       # its loss graph's steps
        from fizgig.training import step_log
        os.remove(step_log.loss_path(folder, name))
    except OSError:
        pass
    try:                                       # a pause record for this run would offer a Resume that can't work
        side = os.path.join(folder, ".fizgig_paused.json")
        with open(side, encoding="utf-8") as f:
            paused = json.load(f)
        if str(paused.get("output_name") or "") == name:
            os.remove(side)
    except (OSError, ValueError, AttributeError):
        pass
    if failed:
        return {"error": f"{len(failed)} file(s) couldn't be deleted (in use?), e.g. {failed[0]}. The run stays "
                         "in the list; try again."}
    history.remove(run_id)
    return {"ok": True}


PAUSED_FILE = ".fizgig_paused.json"       # in the run's output folder, as the desktop writes it


def _record_pause(out_dir):
    """A run that paused: its continuation point (a fine-tune's checkpoint, else its newest complete state folder),
    written as the desktop's .fizgig_paused.json so the Resume survives a restart (of either app). False if there is
    nothing to resume from."""
    with _lock:
        fam, values, folder, name = _run["family"], dict(_run["values"]), _run["image_folder"], _run["lora"]
    desc = get_family(fam)
    ft = bool(desc is not None and desc.finetune and form._kind(desc, values) == "finetune")
    path = launch.latest_ft_checkpoint(out_dir, name) if ft else launch.latest_state(out_dir, name)
    if not path:
        return False
    side = {"mode": "ft" if ft else "state", "state_path": path, "output_name": name,
            "dataset_config": os.path.join(DATASET_DIR, f"{_safe(name)}_web.toml"),
            "network_dim": str(values.get("NETWORK_DIM", "")), "network_alpha": str(values.get("NETWORK_ALPHA", "")),
            "max_train_epochs": str(values.get("MAX_TRAIN_EPOCHS", "")), "run": None,
            "web": {"family": fam, "values": values, "image_folder": folder}}
    try:
        _write_json(os.path.join(out_dir, PAUSED_FILE), side)
    except OSError:
        pass
    with _lock:
        _run.update(resume=path, resume_mode=side["mode"])
    return True


def _restore_pause():
    """At start-up: the last web run, if it paused (its record is still beside its saves), is paused again here."""
    lr = load_state().get("last_run") or {}
    values = lr.get("values") or {}
    name = str(values.get("LORA_NAME") or "").strip()
    out_dir = str(values.get("LORA_OUTPUT_DIR") or "").strip() or default_output_dir()
    try:
        with open(os.path.join(out_dir, PAUSED_FILE), encoding="utf-8") as f:
            side = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(side, dict):
        return
    path, mode = str(side.get("state_path") or ""), str(side.get("mode") or "state")
    if (side.get("output_name") != name or not lr.get("family")
            or not (os.path.isfile(path) if mode == "ft" else os.path.isdir(path))):
        return
    with _lock:
        _run.update(state="paused", family=lr["family"], values=values, image_folder=lr.get("image_folder", ""),
                    lora=name, output_dir=out_dir, resume=path, resume_mode=mode)
    webqueue.hold("The last run is paused. Resume it: the queue carries on after it finishes.", for_pause=True)


LOG_CLEAR = "\x00clear\n"      # in the log where a run starts: the page's console begins again from there


def _folder_problem(e, out_dir) -> str:
    """Why a run's folder couldn't be made, and where to change it: a drive that isn't reachable is the usual one (a
    network drive disconnected, an external drive unplugged, or a path from another computer)."""
    where = getattr(e, "filename", None) or out_dir
    drive = os.path.splitdrive(str(where))[0]
    if drive and not os.path.exists(drive + os.sep):
        where = out_dir                         # the folder asked for, not just the missing drive's root
        why = (f"drive {drive} isn't reachable from here (a network drive that's disconnected, an external drive that "
               "isn't plugged in, or a path from another computer)")
    else:
        why = e.strerror or str(e)
    return (f"Couldn't make the run's folder {where}: {why}. Choose another LoRA output folder on the Training tab "
            f"(Output), or set the default in Preferences > LoRA output.")


def _start(family_key, values, image_folder, resume="", ft_continue=None, note=""):
    if studio.holding_gpu():                    # Repair Studio's model leaves the GPU for the run
        append_log("Unloading Repair Studio's model to free the GPU for this run.\n")
        studio.stop_worker()
    with _lock:
        if _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running"):
            return {"error": "A run is already going. Stop or pause it first."}
    if captions.status()["state"] in ("loading", "running", "translating"):
        return {"error": "Captions are being made (Captions tab). Stop that first, then start the run."}
    captions.unload()                                     # the run gets the GPU memory the caption model held
    desc, inputs, plan = _plan(family_key, values, image_folder, resume, ft_continue)
    if plan.problems:
        return {"problems": plan.problems}
    out_dir = inputs["LORA_OUTPUT_DIR"]
    # as the desktop's Start clears them; a resume's pause record is used up once its run launches
    for stale in (".pause_requested", ".sample_override.json", ".sample_now", ".save_now") + (
            (PAUSED_FILE,) if resume or ft_continue else ()):
        try:
            os.remove(os.path.join(out_dir, stale))
        except OSError:
            pass
    try:
        for d in plan.dirs:
            os.makedirs(d, exist_ok=True)
        for path, text in plan.files:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    except OSError as e:
        return {"error": _folder_problem(e, out_dir)}
    for f in (SLIDER_LIVE, SLIDER_LIVE_APPLIED):     # an earlier run's live slider prompts are not this one's (a resume
        try:                                         # has the latest in its own values); cleared before it launches, so
            os.remove(os.path.join(out_dir, f))      # an Apply while it loads waits for its first epoch, not deleted
        except OSError:
            pass
    with _lock:
        _run.update(state="starting", stage="", lora=inputs["LORA_NAME"], output_dir=out_dir, family=family_key,
                    values=dict(values), image_folder=image_folder, pause=False, resume="", resume_mode="")
    save_state({"last_run": {"family": family_key, "values": values, "image_folder": image_folder}})
    hid = history.start("web", family_key, inputs["LORA_NAME"], out_dir, image_folder, dict(values),
                        resume=bool(resume or ft_continue))
    with _lock:
        _run["history_id"] = hid
    append_log(LOG_CLEAR)                    # this run's log on a clean console
    if note:
        append_log(note + "\n")
    for line in plan.console:
        append_log(line + "\n")
    threading.Thread(target=_run_stages, args=(plan.stages, out_dir), daemon=True).start()
    return {"ok": True, "stages": [s.name for s in plan.stages]}


def _queue_starter(item):
    """For webqueue.start_next: (True, "") once the item's run has started; (False, "wait: ...") while the GPU is
    busy elsewhere (this run, a caption job); (False, why) when it can't start."""
    with _lock:
        if _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running"):
            return False, "wait: a run is training here"
    if captions.status()["state"] in ("loading", "running", "translating"):
        return False, "wait: captions are being made - the next run starts when they finish"
    try:
        r = _start(item["family"], item["values"], item.get("image_folder") or "")
    except Exception as e:
        return False, str(e)
    if r.get("ok"):
        return True, ""
    if r.get("error") and "Captions are" in r["error"]:
        return False, "wait: " + r["error"]
    return False, r.get("error") or "; ".join(r.get("problems") or ["it couldn't start"])


def _popen(cmd):
    """As the desktop's run_subprocess: UTF-8, unbuffered, the chosen GPU, its own process group (and below-normal
    priority on Windows, so the desktop stays smooth)."""
    env = gpu_env(dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                               | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env, cwd=BASE_DIR, **kw)


def pod_storage():
    """The pod volume's free space, as the desktop's RunPod card says it: (text, warn)."""
    try:
        u = shutil.disk_usage("/workspace")
    except OSError:
        return ["Storage: unavailable", False]
    free, total = u.free / 1024 ** 3, u.total / 1024 ** 3
    if total > 10000:                # a network volume shows the host's whole pool, not its quota
        return ["Storage: /workspace is a network volume. Its size is set in RunPod and isn't visible from in "
                "here - check usage in the RunPod dashboard.", False]
    if total < 60:                   # container disk: the volume never mounted, and RunPod wipes it at a stop
        return [f"Storage: only {free:.0f} GB free of {total:.0f} GB - that looks like container disk, not your "
                "volume. Check the template's volume mount path is /workspace, or your files will vanish when the "
                "pod stops.", True]
    return [f"Storage: {free:.0f} GB free of {total:.0f} GB on /workspace", free < 40]


def gpu_env(env, prefs=None):
    """env with the GPU chosen in Preferences (cuda_device), as the desktop app runs everything: one set where Fizgig
    was started wins. Training, captions, Studio, Profiler and Extract all use it."""
    want = str((prefs if prefs is not None else load_prefs()).get("cuda_device") or "").strip()
    if want and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = want
    if env.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip().isdigit():
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")      # a card number means the card nvidia-smi numbers so
    return env


def _progress_dict(upd):
    """The tracker's update as the page shows it (the desktop app sends the same from its own tracker)."""
    step, total = int(upd.get("step") or 0), int(upd.get("total_steps") or 0)
    return {"epoch": upd.get("epoch"), "total_epochs": upd.get("total_epochs"), "step": step, "total_steps": total,
            "speed": upd.get("speed_text") or "", "loss": upd.get("average_loss_text") or "",
            "eta": upd.get("eta_text") or "", "percent": 100.0 * min(step, total) / total if total else 0.0,
            "note": ""}


def _run_stages(stages, out_dir):
    from fizgig.training.progress import TrainingProgressTracker
    append_log(f"\n=== Fizgig run, {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    with _lock:
        _run["state"] = "running"
        _run["progress"] = None
        tracker = TrainingProgressTracker((_run.get("values") or {}).get("MAX_TRAIN_EPOCHS", 1))
        _run["started_at"] = time.time()
    for stage in stages:
        with _lock:
            if _run["state"] == "stopped":
                history.end(_run.get("history_id"), "stopped", _run.get("progress"))
                webqueue.on_run_ended("stopped", _run.get("lora"), _queue_starter)
                return
            _run["stage"] = stage.name
        append_log(f"\n▶ {stage.name}\n")
        try:
            proc = _popen(stage.cmd)
        except OSError as e:
            append_log(f"✖ {stage.name} could not start: {e}\n")
            with _lock:
                _run.update(state="failed", proc=None, ended_at=time.time())
            history.end(_run.get("history_id"), "failed", _run.get("progress"))
            webqueue.on_run_ended("failed", _run.get("lora"), _queue_starter)
            return
        with _lock:
            _run["proc"] = proc
        for line in proc.stdout:
            append_log(line)
            try:
                upd = tracker.consume(line)
            except Exception:
                upd = None
            if upd:
                with _lock:
                    if upd.get("kind") == "preview":
                        p = dict(_run.get("progress") or _progress_dict({}))
                        p["note"] = {"start": "Making previews…", "complete": ""}.get(upd.get("phase"), "")
                        _run["progress"] = p
                    else:
                        _run["progress"] = _progress_dict(upd)
                        if upd.get("kind") == "training":
                            history.feed_loss(_run.get("history_id"), upd.get("epoch"), upd.get("average_loss_text"))
        ret = proc.wait()
        with _lock:
            stopped = _run["state"] == "stopped"
        if stopped:
            append_log(f"\n■ Stopped during {stage.name}.\n")
            history.end(_run.get("history_id"), "stopped", _run.get("progress"))
            webqueue.on_run_ended("stopped", _run.get("lora"), _queue_starter)
            return
        if ret != 0:
            append_log(f"\n✖ {stage.name} failed (exit code {ret}).\n")
            with _lock:
                _run.update(state="failed", proc=None, ended_at=time.time())
            history.end(_run.get("history_id"), "failed", _run.get("progress"))
            webqueue.on_run_ended("failed", _run.get("lora"), _queue_starter)
            return
    paused = ((os.path.exists(os.path.join(out_dir, ".pause_requested")) or _run["pause"])
              and _run.get("family") != refmod.KEY)                  # a RefMod finishes in one go
    if paused and not _record_pause(out_dir):
        append_log("\n[pause] No checkpoint or saved state was found after the pause, so there is nothing to resume "
                   "from: treating the run as ended.\n")
        paused = False
    with _lock:
        _run.update(state="paused" if paused else "completed", proc=None, stage="", ended_at=time.time())
    history.end(_run.get("history_id"), "paused" if paused else "finished", _run.get("progress"))
    webqueue.on_run_ended("paused" if paused else "finished", _run.get("lora"), _queue_starter)
    append_log("\n⏸ Paused. Resume continues from the saved state.\n" if paused else "\n✔ Run complete.\n")
    if not paused and not webqueue.status()["items"]:      # the last run done: a pod can stop billing
        from fizgig.web import pod_stop
        pod_stop.maybe_start(load_prefs(), append_log)


def _stop_and_wait(timeout=20.0):
    """Stop the run (if one is training) and wait until its record says so - before the server goes away."""
    if not _stop():
        return False
    t = time.time()
    while time.time() - t < timeout:
        with _lock:
            if _run["proc"] is None:
                break
        time.sleep(0.25)
    return True


def _stop():
    with _lock:
        proc = _run["proc"]
        if proc is None or proc.poll() is not None:
            return False
        _run["state"] = "stopped"
        _run["ended_at"] = time.time()
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, **_NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    return True


# ---- uploads: a training folder sent from the browser's computer ------------------------------------------------
# what the trainers read (dataset/image_dataset.py: pictures, .mp4 clips, songs) + captions, ACE-Step's .json song
# details and zips. Not .gif / .webm: nothing trains on them.
UPLOAD_EXTS = (MEDIA - {".gif", ".webm"}) | {".txt", ".caption", ".json", ".zip"}
UPLOAD_DEFAULT = os.path.join(BASE_DIR, "training_data")     # when Preferences has no input folder set
UPLOAD_FILES = os.path.join(BASE_DIR, "uploads")              # single files: a context LoRA, a test photo
UPLOAD_FILE_EXTS = (UPLOAD_EXTS - {".zip"}) | {".safetensors"}


def uploads_root() -> str:
    """Where uploaded folders go: the desktop's Preferences > Input Directories folder (a pod sets it to
    /workspace/datasets), else training_data/ inside Fizgig."""
    return pref_path(load_prefs(), "input_dataset_dir") or UPLOAD_DEFAULT


def _tidy_upload_name(name) -> str:
    """The name with what a file can't be called on every system replaced (Act 1: Intro.wav -> Act 1_ Intro.wav),
    the same way for a song and its caption / lyrics / .json files, so they still pair up."""
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", str(name or "")).strip()
    return name.rstrip(". ")


def _upload_name(name, what) -> str:
    """One plain name, kept as it is ("Don't Stop (Live) #1.wav" is fine): no folders, no '..', nothing hidden, and
    nothing Windows can't store, so every upload stays inside uploads_root()."""
    name = str(name or "").strip()
    bad = (not name or name.startswith(".") or name.endswith((".", " "))
           or re.search(r'[/\\<>:"|?*\x00-\x1f]', name)
           or os.path.splitext(name)[0].upper() in _WIN_RESERVED)
    if bad:
        raise ValueError(f"{what} {name!r}: names can't contain / \\ < > : \" | ? *, start with a dot, or end "
                         f"with a dot or space.")
    return name


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


_PIC = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
_SONG = {".wav", ".mp3", ".flac", ".m4a"}


_AUDIO = {".wav", ".mp3", ".flac", ".m4a"}
_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif"}


def _audio_seconds(path):
    """A song's length from its header (soundfile when installed, else a WAV's own header), or None."""
    try:
        import soundfile
        return float(soundfile.info(path).duration)
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        try:
            import wave
            with wave.open(path) as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return None
    return None


# a preview: <run>_e<epoch>_<prompt>_<time>_<seed>[_s<strength %>].<ext> (families/train.py)
_SAMPLE_NAME = re.compile(r"^(?P<run>.+)_e(?P<epoch>\d{6})_(?P<idx>\d{2})_\d+_\d+(?:_s(?P<s>\d{3}))?\.\w+$")


def sample_info(name) -> dict:
    """What a preview's name says: its run, epoch, prompt number and strength (None = full)."""
    m = _SAMPLE_NAME.match(name)
    if not m:
        return {}
    return {"run": m.group("run"), "epoch": int(m.group("epoch")), "idx": int(m.group("idx")),
            "strength": int(m.group("s")) / 100.0 if m.group("s") else None}


def run_prompts(out_dir, name) -> list:
    """The preview prompts the newest run of this name in this folder used (its recorded settings), in order."""
    norm = lambda p: os.path.normcase(os.path.abspath(p or "."))
    for r in history.runs():
        if (r.get("lora") or "") == name and norm(r.get("output_dir")) == norm(out_dir):
            text = str((r.get("values") or {}).get("SAMPLE_PROMPT") or "")
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return []


def pair_songs(entries):
    """[(name, path, extra), ...] with each song given the picture of the same name beside it (its waveform) and
    its length, and those pictures left out as items of their own. A clip preview (MiniMax H3: <stem>.mp4 with its
    middle frame as <stem>.png and its sound as <stem>.wav) is the one video, the picture its poster."""
    stems_with_clip = {os.path.splitext(n)[0] for n, _, _ in entries if os.path.splitext(n)[1].lower() == ".mp4"}
    stems_with_song = {os.path.splitext(n)[0] for n, _, _ in entries if os.path.splitext(n)[1].lower() in _AUDIO}
    pics = {os.path.splitext(n)[0]: p for n, p, _ in entries if os.path.splitext(n)[1].lower() in _IMAGES}
    out = []
    for n, p, extra in entries:
        stem, ext = os.path.splitext(n)
        ext = ext.lower()
        if stem in stems_with_clip and ext != ".mp4":
            continue                                 # the clip's own still and sound: inside its video
        if ext in _IMAGES and stem in stems_with_song:
            continue                                 # shown on its song's card
        item = {"name": n, "path": p, **extra}
        if ext == ".mp4" and pics.get(stem):
            item["image"] = pics[stem]               # the clip's poster
        if ext in _AUDIO:
            item["image"] = pics.get(stem)
            item["duration"] = _audio_seconds(p)
        out.append(item)
    return out


_sizes: dict = {}             # (path, mtime) -> [width, height] or None


def _media_size(path):
    """[width, height] of a picture (its header only) or a clip (its first stream), None for a song or an unreadable
    file - for the Captions cards and their size sort."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _AUDIO:
        return None
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _sizes:
        wh = None
        try:
            if ext in _IMAGES:
                from PIL import Image
                with Image.open(path) as im:
                    wh = [int(im.width), int(im.height)]
            else:
                import cv2
                cap = cv2.VideoCapture(path)
                try:
                    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    wh = [w, h] if w and h else None
                finally:
                    cap.release()
        except Exception:
            wh = None
        if len(_sizes) > 20000:
            _sizes.clear()
        _sizes[key] = wh
    return _sizes[key]


REMOVED_DIR = "removed"           # the desktop's Remove folder


def remove_from_dataset(folder, name, ext):
    """Take a file out of the training set without deleting it: it, its caption (and a song's waveform picture) move
    into <folder>/removed/. Returns the names moved, for Undo."""
    folder = os.path.abspath(folder)
    src = os.path.join(folder, os.path.basename(name))
    if not os.path.isfile(src) or os.path.dirname(src) != folder:
        raise ValueError("That file isn't in the training folder.")
    stem, fext = os.path.splitext(os.path.basename(name))
    names = [os.path.basename(name)]
    for cext in {ext or ".txt", ".txt"}:
        if os.path.isfile(os.path.join(folder, stem + cext)):
            names.append(stem + cext)
    if fext.lower() in _AUDIO:
        names += [f for f in os.listdir(folder) if os.path.splitext(f)[0] == stem and os.path.splitext(f)[1].lower() in _IMAGES]
    dest = os.path.join(folder, REMOVED_DIR)
    os.makedirs(dest, exist_ok=True)
    moved = []
    for n in names:
        target, k = os.path.join(dest, n), 2
        while os.path.exists(target):                # never overwrite an earlier removal
            s, e = os.path.splitext(n)
            target, k = os.path.join(dest, f"{s} ({k}){e}"), k + 1
        os.replace(os.path.join(folder, n), target)
        moved.append([n, os.path.basename(target)])
    return moved


def restore_to_dataset(folder, moved):
    """Undo a removal: [[original name, name in removed]] back where they were (never over a file there)."""
    folder = os.path.abspath(folder)
    back = []
    for orig, there in moved or ():
        src = os.path.join(folder, REMOVED_DIR, os.path.basename(there))
        dst = os.path.join(folder, os.path.basename(orig))
        if os.path.isfile(src) and not os.path.exists(dst):
            os.replace(src, dst)
            back.append(orig)
    return back


_thumbs: dict = {}            # (path, mtime) -> a small JPEG; the oldest go past 600


_thumbs_lock = threading.Lock()
_thumb_slots = threading.BoundedSemaphore(4)   # a page of tiles decodes a few big photos at a time, not dozens at once


def _thumb(path, size=320):
    """A small JPEG of a picture for a tile (a phone shouldn't fetch four full photos per folder), kept in memory.
    The cache is shared by the request threads, so it is locked (an unlocked trim crashed requests once it was full,
    and those tiles showed broken)."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    with _thumbs_lock:
        hit = _thumbs.get(key)
    if hit is not None:
        return hit
    try:
        import io
        from PIL import Image
        with _thumb_slots:
            with Image.open(path) as im:
                im.draft("RGB", (size * 2, size * 2))      # a JPEG decodes at a reduced size: much faster
                im.thumbnail((size, size))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=82)
        data = buf.getvalue()
    except Exception:
        return None
    with _thumbs_lock:
        _thumbs[key] = data
        while len(_thumbs) > 600:
            _thumbs.pop(next(iter(_thumbs)))
    return data


_RELEASE = []


def _newest_release(git):
    """(tag, commit) of the newest vX.Y.Z tag on origin, asked once per run; None when it can't be reached. A pod
    clones one commit with no tags, so its About box would otherwise show only a branch and a commit."""
    if not _RELEASE:
        best = None
        for line in git("ls-remote", "--tags", "origin").splitlines():
            sha, _, ref = line.partition("\t")
            m = re.fullmatch(r"refs/tags/v(\d+)\.(\d+)\.(\d+)(\^\{\})?", ref.strip())
            if not m:
                continue
            key = tuple(int(x) for x in m.group(1, 2, 3))
            if best is None or key > best[0] or (key == best[0] and m.group(4)):   # ^{} = the tag's own commit
                best = (key, f"v{key[0]}.{key[1]}.{key[2]}", sha)
        _RELEASE.append(best[1:] if best else None)
    return _RELEASE[0]


def about_info() -> dict:
    """The About box's facts, as the desktop's: the running version and how this install updates."""
    def git(*a):
        try:
            r = subprocess.run(["git", *a], cwd=BASE_DIR, capture_output=True, text=True, timeout=5, **_NO_WINDOW)
            return r.stdout.strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""
    v = git("describe", "--tags", "--always")
    if v and "v" not in v.split("-")[0]:              # a tagless checkout (a pod's shallow clone): say what it is
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch and branch != "HEAD":
            v = f"{branch} @ {v}"
        rel = _newest_release(git)                    # the release number, from GitHub's tags (the clone has none)
        if rel:
            v = rel[0] if rel[1] == git("rev-parse", "HEAD") else f"{rel[0]} + later changes ({v})"
    rocm = os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm"
    if on_pod():
        how = "On RunPod, Fizgig updates itself every time you restart the pod - just stop and start it to get the latest."
    elif os.name == "nt":
        how = f"To update: close Fizgig and run {'update_fizgig_rocm.bat' if rocm else 'update_fizgig.bat'}."
    else:
        how = ("To update: stop Fizgig (and its service, if it runs as one), then in the Fizgig folder run "
               "venv/bin/python update_fizgig.py" + (" --rocm" if rocm else "") + ", and start it again.")
    return {"version": v or "unknown", "update": how, "commit": git("rev-parse", "HEAD")}  # the app window: same code?


def _sweep_parts(folder, entries):
    """An upload abandoned a day ago leaves a hidden .name.part (it can be GBs): cleared when the folder is listed."""
    for e in entries:
        if e.name.startswith(".") and e.name.endswith(".part"):
            try:
                if time.time() - e.stat().st_mtime > 86400:
                    os.remove(e.path)
            except OSError:
                pass


def _settled(entry, names) -> bool:
    """A file whole on disk, not one still arriving (an upload's .part beside it, or written in the last few seconds):
    a tile's thumbnail of a half-written picture comes out broken."""
    if f".{entry.name}.part" in names or f"{entry.name}.part" in names:
        return False
    try:
        return time.time() - entry.stat().st_mtime > 10
    except OSError:
        return False


RECENT_KEEP = 12                            # the training folders from elsewhere the Start tab remembers


def remember_folder(path):
    """A training folder you picked: first in the list of recent ones (last_used recent_folders), so the Start tab's
    Your training data shows it even when it lives outside the training data folder."""
    path = str(path or "").strip()
    if not path:
        return
    d = load_last_used()
    key = os.path.normcase(os.path.abspath(path))
    old = [p for p in (d.get("recent_folders") or []) if isinstance(p, str)
           and os.path.normcase(os.path.abspath(p)) != key]
    update_last_used({"recent_folders": [path] + old[:RECENT_KEEP - 1]})


def forget_folder(path):
    key = os.path.normcase(os.path.abspath(str(path or "")))
    d = load_last_used()
    update_last_used({"recent_folders": [p for p in (d.get("recent_folders") or []) if isinstance(p, str)
                                         and os.path.normcase(os.path.abspath(p)) != key]})


def _elsewhere_folders(root) -> list:
    """Training folders you've used that aren't inside the training data folder: the one picked now, the recent ones,
    and those of past runs (a Fizgig 7 install's datasets, wherever they are)."""
    d = load_last_used()
    cand = [d.get("image_folder"), d.get("image_folder2")] + list(d.get("recent_folders") or [])
    try:
        cand += [r.get("image_folder") for r in history.runs()[:60]]
    except Exception:
        pass
    inside = os.path.normcase(os.path.abspath(root)) + os.sep
    seen, out = set(), []
    for p in cand:
        if not isinstance(p, str) or not p.strip():
            continue
        k = os.path.normcase(os.path.abspath(p))
        if k in seen or (k + os.sep).startswith(inside) or not os.path.isdir(p):
            continue
        seen.add(k)
        out.append(p)
    return out


def list_datasets() -> dict:
    """The folders in the training data folder (uploads_root()), and the training folders you've used from anywhere
    else (marked elsewhere), newest first: what each holds (pictures, clips, songs, how many have a caption), when
    it last changed, and up to four pictures for its tile."""
    root = uploads_root()
    out = []
    try:
        folders = [(e.path, e.name, False) for e in os.scandir(root) if e.is_dir() and not e.name.startswith(".")]
    except OSError:
        folders = []
    folders += [(p, os.path.basename(os.path.normpath(p)) or p, True) for p in _elsewhere_folders(root)]
    for path, name, elsewhere in folders:
        pics = clips = songs = captioned = 0
        thumbs, newest = [], 0.0
        try:
            entries = list(os.scandir(path))
        except OSError:
            continue
        names = {e.name for e in entries}
        song_stems = {os.path.splitext(n)[0] for n in names if os.path.splitext(n)[1].lower() in _SONG}
        waves = []
        for e in entries:
            if not e.is_file():
                continue
            stem, ext = os.path.splitext(e.name)
            ext = ext.lower()
            if ext in _PIC and stem in song_stems:     # a song's waveform: its tile picture, not a training picture
                if len(waves) < 4:
                    waves.append(e.path)
                continue
            kind = "pic" if ext in _PIC else "clip" if ext == ".mp4" else "song" if ext in _SONG else ""
            if not kind:
                continue
            pics += kind == "pic"
            clips += kind == "clip"
            songs += kind == "song"
            captioned += any(stem + c in names for c in (".txt", ".caption"))
            try:
                newest = max(newest, e.stat().st_mtime)
            except OSError:
                pass
            if kind == "pic" and len(thumbs) < 4 and _settled(e, names):
                thumbs.append(e.path)
        if not elsewhere:
            _sweep_parts(path, entries)
        if pics + clips + songs:
            out.append({"name": name, "path": path, "pictures": pics, "clips": clips, "songs": songs,
                        "captioned": captioned, "time": newest, "thumbs": thumbs or waves, "elsewhere": elsewhere})
    out.sort(key=lambda x: -x["time"])
    return {"root": root, "datasets": out[:300], "total": len(out)}


def _upload_target(folder, name, single):
    """(name, ext, the folder it goes into, its half-written .part) - every check an upload goes through."""
    name = _upload_name(name, "File name")
    ext = os.path.splitext(name)[1].lower()
    if single:
        if ext not in UPLOAD_FILE_EXTS:
            raise ValueError(f"{name}: not a LoRA (.safetensors), picture, clip or song.")
        dest = UPLOAD_FILES
    else:
        folder = _upload_name(folder, "Folder name")
        if ext not in UPLOAD_EXTS:
            raise ValueError(f"{name}: Fizgig doesn't train on {ext or 'this'} files. It takes pictures, .mp4 clips, "
                             f"songs (.wav .mp3 .flac .m4a), caption .txt files, .json song details and .zip files.")
        dest = os.path.join(uploads_root(), folder)
    return name, ext, dest, os.path.join(dest, f".{name}.part")


def upload_status(folder, name, single, size, mtime=None):
    """For a resumed upload: how much of the file is already here (its .part), or done (the file itself: the same
    size and the same modified time - an edited caption of the same length is sent again), so the page sends only
    the rest - or nothing."""
    name, ext, dest, tmp = _upload_target(folder, name, single)
    final = os.path.join(dest, name)
    same_time = mtime is not None and os.path.isfile(final) and abs(os.path.getmtime(final) - mtime) < 2
    if ext != ".zip" and os.path.isfile(final) and os.path.getsize(final) == size and same_time:
        return {"have": size, "done": True, "path": final, "folder": dest}
    have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    return {"have": have if have <= size else 0, "done": False}


def save_chunk(folder, name, rfile, length, offset, total, single=False, mtime=None):
    """One piece of a file at `offset` (a big upload goes up in pieces, so a dropped connection costs one piece, not
    the file); the last piece finishes it exactly as a whole-file upload. A piece that doesn't follow on from what
    is here is refused with what is here, so the page carries on from there."""
    name, ext, dest, tmp = _upload_target(folder, name, single)
    os.makedirs(dest, exist_ok=True)
    have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    if offset != (0 if offset == 0 else have):
        return {"resync": have}
    left = length
    with open(tmp, "wb" if offset == 0 else "r+b") as f:
        f.seek(offset)
        f.truncate()
        while left > 0:
            chunk = rfile.read(min(left, 1 << 20))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        return {"resync": os.path.getsize(tmp)}       # cut off mid-piece: what arrived stays, the page resends the rest
    if offset + length < total:
        return {"have": offset + length}
    done = _finish_upload(name, ext, dest, tmp, single)
    if mtime is not None and ext != ".zip":            # it keeps its own modified time, as a copied file does
        try:
            os.utime(os.path.join(dest, name), (mtime, mtime))
        except OSError:
            pass
    return done


def _save_upload(folder, name, rfile, length, single=False) -> dict:
    """A whole file in one request, into a folder under uploads_root() (training data), or `single`: one file into
    <Fizgig>/uploads."""
    name, ext, dest, tmp = _upload_target(folder, name, single)
    os.makedirs(dest, exist_ok=True)
    left = length
    try:
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
    except OSError:
        left = left or 1
    if left:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise ValueError(f"{name}: the upload was cut off. Upload it again.")
    return _finish_upload(name, ext, dest, tmp, single)


def _finish_upload(name, ext, dest, tmp, single):
    """The arrived file put in place: a zip unpacked (what it held), anything else renamed from its .part."""
    saved, skipped, replaced, badname = [], [], [], []
    if ext == ".zip" and not single:
        import zipfile
        try:
            with zipfile.ZipFile(tmp) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if info.is_dir() or not base or "__MACOSX" in info.filename or base.startswith("."):
                        continue
                    if os.path.splitext(base)[1].lower() not in UPLOAD_EXTS - {".zip"}:
                        skipped.append(base)
                        continue
                    try:
                        base = _upload_name(_tidy_upload_name(base), "File name")
                    except ValueError:
                        badname.append(base)
                        continue
                    if base in saved or os.path.exists(os.path.join(dest, base)):
                        replaced.append(base)
                    with z.open(info) as src, open(os.path.join(dest, base), "wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
                    saved.append(base)
        except zipfile.BadZipFile:
            raise ValueError(f"{name}: not a readable zip.")
        finally:
            os.remove(tmp)
    else:
        if os.path.exists(os.path.join(dest, name)):
            replaced.append(name)
        os.replace(tmp, os.path.join(dest, name))
        saved.append(name)
    return {"folder": dest, "saved": sorted(set(saved)), "skipped": skipped, "replaced": replaced,
            "badname": badname}


# ---- model downloads: the desktop's "Download models for me" (fizgig.scripts.fetch_models) ----------------------
_fetch = {"proc": None, "state": "idle", "family": "", "file": "", "done": 0, "total": 0, "lines": []}


def _fetch_status() -> dict:
    with _lock:
        out = {k: v for k, v in _fetch.items() if k not in ("proc", "lines")}
        out["lines"] = _fetch["lines"][-40:]
    out["env_token"] = bool(os.environ.get("HF_TOKEN"))
    return out


def _start_fetch(family, token="", optional=False):
    """The family's model files plus the helper models, into <repo>/models (a pod links that to its volume), paths
    written to Preferences by the fetcher. A token is passed to that one download only, never written to disk."""
    desc = _files_family(family)
    family = desc.key if desc is not None else family
    if desc is None:
        raise ValueError(f"Unknown family: {family}")
    token = (token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if any(f.gated and f.repo for f in desc.model_files) and not token:
        raise ValueError(f"{desc.display_name}'s files need a Hugging Face token.")
    with _lock:
        if _fetch["proc"] is not None and _fetch["proc"].poll() is None:
            raise ValueError("A model download is already running.")
    cmd = [sys.executable, "-m", "fizgig.scripts.fetch_models", "--progress", "--family", "tools", "--family", family]
    if optional or not desc.fetch_optional_label:
        cmd.append("--include-optional")          # no optional tick on the family: its whole list, as the desktop
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=os.path.join(BASE_DIR, "src"))
    if token:
        env["HF_TOKEN"] = token
    kw = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env, cwd=BASE_DIR, **kw)
    with _lock:
        _fetch.update(proc=proc, state="running", family=family, file="", done=0, total=0, lines=[])
    threading.Thread(target=_fetch_watch, args=(proc,), daemon=True).start()


def _fetch_watch(proc):
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = re.match(r"\s*\[progress\] (\d+) (\d+) (.+)$", line)
        with _lock:
            if m:
                _fetch.update(done=int(m.group(1)), total=int(m.group(2)), file=m.group(3))
            else:
                _fetch["lines"].append(line)
                del _fetch["lines"][:-200]
    code = proc.wait()
    with _lock:
        if _fetch["state"] == "running":
            _fetch["state"] = "done" if code == 0 else "failed"


def _caption_gpu_check(need_gb):
    """With a run training, a caption job only starts when the card has room for it beside the run (what the job
    needs + 3 GB)."""
    with _lock:
        mine = _run["proc"] is not None and _run["proc"].poll() is None
    if not mine:
        return
    g = _chosen_gpu(list_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES") or load_prefs().get("cuda_device"))
    free = (g["total_gb"] - g["used_gb"]) if g else 0.0
    if free < need_gb + 3.0:
        raise ValueError(f"A run is training and the card has {free:.0f} GB free; this needs about {need_gb + 3:.0f} GB "
                         f"beside it. Wait for the run to finish, or pause it.")


def _caption_args(b):
    prefs = load_prefs()
    desc = _files_family(b.get("family", ""))
    media = set(desc.media) if desc is not None else {"photo"}
    folder = str(b.get("folder") or "")
    if not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    return prefs, folder, "clip" in media, "photo" in media, "voice" in media


def _cancel_fetch():
    """Partial files resume next time; Preferences are only written when a file is complete."""
    with _lock:
        proc = _fetch["proc"]
        if proc is None or proc.poll() is not None:
            return False
        _fetch["state"] = "cancelled"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, **_NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    return True


# ---- the HTTP side ----------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    password: str | None = None
    local_only = True

    def log_message(self, fmt, *args):            # quiet: the console is for the run
        pass

    # -- replies
    def _send(self, code, body: bytes, ctype, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, code=200, extra=()):
        self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _file(self, path, ctype=None):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return self.json({"error": "not found"}, 404)
        self._send(200, data, ctype or mimetypes.guess_type(path)[0] or "application/octet-stream")

    def _stream(self, path, attachment=True, filename=None, keep=False):
        """A file sent in pieces (a fine-tune checkpoint is ~10 GB; it never sits in memory), honouring a byte
        range: a player dragged along a song or clip asks for "Range: bytes=N-" and gets just that (206), and a
        dropped download can carry on where it stopped. attachment: a download (else played in the page). keep: the
        address names this exact file (a render's gen, a file's modified time), so the browser may keep it - a clip
        played, looped or played again is not fetched again."""
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            return self.json({"error": "not found"}, 404)
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:                                         # "bytes=-N": the last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                f.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            code = 206
        name = (filename or os.path.basename(path)).replace('"', "")
        with f:
            f.seek(start)
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if attachment
                             else mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Cache-Control", "private, max-age=86400" if keep and not attachment else "no-store")
            self.end_headers()
            left = end - start + 1
            try:
                while left > 0 and (chunk := f.read(min(left, 8 << 20))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    # -- access
    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
        return not self.local_only or host in LOCAL_HOSTS

    def _session(self):
        return hmac.new(_session_key().encode(), (self.password or "").encode(), hashlib.sha256).hexdigest()

    def _cookie(self, remember=False):
        """The login cookie: for 30 days on this device with Keep me logged in, else until the browser closes.
        Secure through a pod's HTTPS proxy."""
        return ("Set-Cookie", f"fizgig_session={self._session()}; Path=/; HttpOnly; SameSite=Strict"
                              + ("; Max-Age=2592000" if remember else "") + ("; Secure" if on_pod() else ""))

    def _logged_in(self):
        if not self.password:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "fizgig_session" and hmac.compare_digest(v, self._session()):
                return True
        return False

    def _api_ok(self, qs=None):
        token = self.headers.get("X-Fizgig-Token") or ((qs or {}).get("t") or [""])[0]
        if not hmac.compare_digest(token, TOKEN):
            self.json({"error": "Reload the page."}, 403)
            return False
        if not self._logged_in():
            self.json({"error": "Log in first.", "login": True}, 401)
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    # -- GET
    def _get(self):
        if not self._host_ok():
            return self.json({"error": "bad host"}, 403)
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
                page = f.read().replace("__FIZGIG_TOKEN__", TOKEN)
            # the Fizgig app window (fizgig/web/app_window.py) started this server: its one key logs it in, so the
            # web password guards the network, not the window on this computer
            key = os.environ.get("FIZGIG_APP_KEY", "")
            extra = ([self._cookie()]
                     if key and self.password and hmac.compare_digest((qs.get("app_key") or [""])[0], key) else [])
            phone = (qs.get("phone") or [""])[0]
            if phone and self.password and _phone_pass_take(phone):
                extra = [self._cookie(remember=True)]
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8", extra=extra)
        if path == "/welcome.mp4":                  # the welcome popup's tour (it ships with Fizgig: no network)
            return self._stream(os.path.join(STATIC_DIR, "welcome_v8.mp4"), attachment=False)  # ranged: it seeks
        if path in ("/app.js", "/style.css", "/qrcode.js"):
            return self._file(os.path.join(STATIC_DIR, path[1:]),
                              "application/javascript; charset=utf-8" if path.endswith(".js") else "text/css")
        if path == "/icon.png":
            return self._file(os.path.join(STATIC_DIR, "icon.png"))
        if path == "/logo.jpg":
            return self._file(os.path.join(BASE_DIR, "docs", "assets", "logo.jpg"))
        if path in ("/manifest.webmanifest", "/icon-192.png", "/icon-512.png"):   # Add to Home Screen (no login:
            return self._file(os.path.join(STATIC_DIR, path[1:]),                 # a phone fetches it without one)
                              "application/manifest+json" if path.endswith(".webmanifest") else None)
        if not path.startswith("/api/") or not self._api_ok(qs):
            return None if path.startswith("/api/") else self.json({"error": "not found"}, 404)

        if path == "/api/schema":
            return self.json({**get_schema(ref_dit_set=bool(pref_path(load_prefs(), "minimax_ref_dit"))),
                              "logged_in": True, "outdated": _code_stamp() > _STARTED_CODE + 1})
        if path == "/api/prefs":
            from fizgig.web import pod_stop
            p = load_prefs()
            return self.json({"models": {k: str(p.get(k) or "") for d in training_families()
                                         for k in form.model_keys(d)},
                              "cache_dir": str(p.get("cache_dir") or ""),
                              "settings": {k: str(p.get(k) or "") for k in SETTING_PREFS},
                              "abs": {k: pref_path(p, k) for k in ("profiles_dir", "cache_dir")},
                              "lora_output_dir": str(p.get("lora_output_dir") or ""), "output_default": default_output_dir(),
                              "web_port": saved_port(p), "port_now": self.server.server_address[1], "on_pod": on_pod(), "file_manager": pod_stop.file_manager(), "host_name": socket.gethostname(), "port_fixed": getattr(self, "port_fixed", False),
                              "password_on": bool(self.password),
                              "password_saved": bool(str(p.get("web_password") or "").strip()),
                              "welcome_hidden": str(p.get("welcome_v8_hidden") or "") == "1",
                              "password_env": bool(os.environ.get("FIZGIG_PASSWORD")) or getattr(self, "password_arg", False),
                              "lan_ip": __import__("fizgig.web.port", fromlist=["x"]).lan_ip(),
                              "cuda_device": str(p.get("cuda_device") or ""), "gpus": list_gpus(),
                              "pod": {"stop_when_done": str(p.get("runpod_stop_when_done") or "0") == "1",
                                      "key_saved": bool(str(p.get("runpod_api_key") or "").strip()),
                                      "key_env": bool((os.environ.get("RUNPOD_STOP_API_KEY") or "").strip()),
                                      "pod_id": __import__("fizgig.web.pod_stop", fromlist=["x"]).pod_id(),
                                      "storage": pod_storage()}
                              if on_pod() else None,
                              "gpu_env_locked": bool(os.environ.get("CUDA_VISIBLE_DEVICES"))})
        if path == "/api/state":
            st = load_state()
            shared = load_last_used().get("image_folder")
            if shared:                               # the desktop's choice, if it made one since
                st["image_folder"] = shared
            return self.json(st)
        if path == "/api/trigger":                   # the training folder's trigger word
            return self.json({"word": folder_trigger((qs.get("folder") or [""])[0])})
        if path == "/api/fetch":
            return self.json(_fetch_status())
        if path == "/api/captions/options":
            p = load_prefs()
            cap = pref_path(p, "krea2_text_encoder")
            desc = get_family((qs.get("family") or [""])[0])
            return self.json({**captions.options(p, cap if cap and os.path.isfile(cap) else ""),
                              "song_words": getattr(desc, "song_captions", "") if desc is not None else ""})
        if path == "/api/presets":               # a model's presets: the built-in ones, then the ones you saved
            desc = _preset_family((qs.get("family") or [""])[0])
            if desc is None:
                return self.json({"error": "unknown family"}, 400)
            built = [n for n, _ in (desc.presets or ())]
            return self.json({"builtin": built, "user": [n for n in user_presets(desc) if n not in built]})
        if path == "/api/studio/families":
            return self.json({"families": [{"key": d.key, "name": d.display_name} for d in studio.families()]})
        if path == "/api/studio/layout":
            try:
                desc = _studio_desc((qs.get("family") or [""])[0])
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
            return self.json(studio.layout(desc, load_prefs(), pref_path, samples=_studio_samples(desc)))
        if path == "/api/studio/status":
            return self.json(studio.status())
        if path == "/api/refmod/info":              # RefMod Toolkit: its choices, the desktop's last setup, presets
            from fizgig.web import refmod_studio as rms
            fams = rms.families()
            want = (qs.get("family") or [""])[0]
            desc = next((d for d in fams if d.key == want), fams[0] if fams else None)
            return self.json(rms.info(desc, _read_json(LAST_USED_FILE)))
        if path == "/api/refmod/scan":
            from fizgig.web import refmod_studio as rms
            return self.json(rms.scan((qs.get("folder") or [""])[0].strip(), (qs.get("lora") or [""])[0].strip()))
        if path == "/api/refmod/graphs":            # the pack's graph presets in <mods folder>/graph_presets
            from fizgig.web import refmod_studio as rms
            return self.json({"presets": rms.graph_presets((qs.get("folder") or [""])[0].strip())})
        if path == "/api/refmod/status":
            return self.json(studio.refmod_status())
        if path == "/api/refmod/media":             # a render: its PNG, or (mp4=1) its clip
            it = studio.refmod_item((qs.get("gen") or ["0"])[0], (qs.get("side") or [""])[0], (qs.get("i") or ["0"])[0])
            p = (it or {}).get("mp4" if (qs.get("mp4") or [""])[0] else "png") or ""
            if not p or not os.path.isfile(p):
                return self.json({"error": "no picture"}, 404)
            return self._stream(p, attachment=False)
        if path == "/api/studio/image":
            p = studio.image((qs.get("which") or [""])[0], (qs.get("gen") or [""])[0])
            if not p:
                return self.json({"error": "no picture yet"}, 404)
            dl = re.sub(r'[\\/:*?"<>|]+', "_", (qs.get("dl") or [""])[0])        # a Save / Download button
            if dl:                                    # the name the page asks for, with the file's own extension
                dl = os.path.splitext(dl)[0] + os.path.splitext(p)[1]
            return self._stream(p, attachment=bool(dl), filename=dl or None, keep=bool((qs.get("gen") or [""])[0]))
        if path == "/api/studio/peaks":             # a song's waveform for the player's strip
            p = studio.image((qs.get("which") or [""])[0], (qs.get("gen") or [""])[0])
            if not p or not p.lower().endswith(".wav"):
                return self.json({"error": "no song"}, 404)
            return self.json(studio.song_peaks(p))
        if path == "/api/extract/info":             # Extract: the models, a model's presets and block picker
            from fizgig.web import tools
            fams = tools.families()
            want = (qs.get("family") or [""])[0]
            desc = next((d for d in fams if d.key == want), fams[0] if fams else None)
            if desc is None:
                return self.json({"error": "No model offers Extract."}, 404)
            out_dir = str(((load_state().get("values") or {}).get(desc.key) or {}).get("LORA_OUTPUT_DIR") or "").strip()
            return self.json({"families": [{"key": d.key, "name": d.display_name} for d in fams],
                              **tools.info(desc, out_dir or default_output_dir())})
        if path == "/api/extract/status":
            from fizgig.web import tools
            return self.json(tools.status("extract"))
        if path == "/api/royale/info":              # LoRA Royale: the models that offer it, and what each takes
            from fizgig.web import royale
            fams = royale.families()
            want = (qs.get("family") or [""])[0]
            d = next((x for x in fams if x.key == want), fams[0] if fams else None)
            out_dir = str(((load_state().get("values") or {}).get(d.key) or {}).get("LORA_OUTPUT_DIR") or "").strip()                 if d is not None else ""
            return self.json({"families": [{"key": x.key, "name": x.display_name, "video": bool(x.video_workbench)}
                                           for x in fams], "family": d.key if d else "",
                              "reference_kind": (d.reference_kind or "") if d else "",
                              "reference_strength": bool(d and d.reference_strength),
                              "out_dir": out_dir or default_output_dir()})
        if path == "/api/royale/scan":
            from fizgig.web import royale
            return self.json(royale.scan((qs.get("folder") or [""])[0]))
        if path == "/api/royale/status":
            return self.json(studio.royale_status())
        if path == "/api/royale/travel/status":
            return self.json(studio.travel_status())
        if path == "/api/royale/travel/image":      # a travel frame, or the comparison sheet (no i)
            i = (qs.get("i") or [None])[0]
            p = studio.travel_file((qs.get("mode") or [""])[0], None if i in (None, "") else int(i))
            if not p:
                return self.json({"error": "no picture"}, 404)
            return self._stream(p, attachment=False)
        if path == "/api/royale/pt":                # prompt travel: its presets, subjects and a travel's waypoints
            from fizgig.lora_royale import SEED_TRAVEL_PRESETS
            from fizgig.lora_royale import prompt_travel as pt
            dim, subj = (qs.get("dim") or [""])[0], (qs.get("subject") or ["Woman"])[0]
            preset = pt.PRESETS_BY_NAME.get((qs.get("preset") or [""])[0])
            words = (pt.parse_custom((qs.get("custom") or [""])[0]) if dim == "Custom" else
                     pt.waypoints_for(dim, subj) if dim else [])
            return self.json({"presets": pt.PRESET_NAMES, "subjects": pt.SUBJECT_LABELS, "words": words,
                              "seed_presets": SEED_TRAVEL_PRESETS,
                              "preset": dict(preset, prompt=pt.fill_subject(preset["prompt"], subj)) if preset else None})
        if path == "/api/royale/image":             # one render (by its place in the run)
            p = studio.royale_png((qs.get("i") or ["-1"])[0])
            if not p:
                return self.json({"error": "no picture"}, 404)
            return self._stream(p, attachment=False)
        if path == "/api/profile/info":             # Profiler: the models that offer it
            from fizgig.web import tools
            return self.json({"families": [{"key": d.key, "name": d.display_name} for d in tools.families("profiler")]})
        if path == "/api/profile/prefill":          # prompts from the LoRA's trigger, its run's subject photos
            from fizgig.web import tools
            p = (qs.get("path") or [""])[0]
            return self.json(tools.profile_prefill(p) if os.path.isfile(p) else {})
        if path == "/api/profile/status":
            from fizgig.web import tools
            return self.json(tools.status("profile"))
        if path == "/api/profile/report":           # a Profiler report (the profiles folder's HTML)
            p = (qs.get("path") or [""])[0]
            root = os.path.abspath(pref_path(load_prefs(), "profiles_dir") or user_file("profiles"))
            if not p.lower().endswith(".html") or not os.path.isfile(p) or                     os.path.commonpath([root, os.path.abspath(p)]) != root:
                return self.json({"error": "not a report"}, 404)
            with open(p, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/metadata":                  # Metadata: any .safetensors file's metadata (header only)
            p = (qs.get("path") or [""])[0]
            if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
                return self.json({"error": "Choose a .safetensors file (that file isn't there)."}, 404)
            from fizgig.utils.st_meta import read_metadata
            try:
                return self.json({"meta": read_metadata(p), "size": os.path.getsize(p)})
            except (OSError, ValueError) as e:
                return self.json({"error": f"Could not read metadata: {e}"}, 400)
        if path == "/api/studio/setups":
            return self.json({"setups": _setups()})
        if path == "/api/studio/setup":
            try:
                with open(_setup_path((qs.get("name") or [""])[0]), encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError) as e:
                return self.json({"error": f"Could not read the setup: {e}"}, 404)
            if not isinstance(d, dict) or not d.get("fizgig_repair_setup"):
                return self.json({"error": "That file isn't a Studio setup."}, 404)
            return self.json(d)
        if path == "/api/studio/metrics":            # the pair on screen, measured (and likeness to a reference)
            return self.json(studio.metrics((qs.get("ref") or [""])[0]))
        if path == "/api/studio/epochs":             # the run a LoRA belongs to (the mix's Up to dial)
            p = (qs.get("path") or [""])[0]
            if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
                return self.json({"error": "That LoRA isn't there."}, 404)
            from fizgig.families.epochs import run_epochs
            run = run_epochs(p)
            own = next((e for e, f in run if os.path.normcase(os.path.abspath(f)) == os.path.normcase(os.path.abspath(p))), None)
            return self.json({"epochs": [e for e, _f in run], "own": own})
        if path == "/api/studio/lorafamily":         # which model a LoRA is for (the mix only takes this model's)
            from fizgig.families.registry import family_of_lora
            try:
                d = family_of_lora((qs.get("path") or [""])[0])
            except Exception:
                d = None
            return self.json({"key": d.key if d else "", "name": d.display_name if d else ""})
        if path == "/api/studio/report":             # the Profiler's HTML report for the loaded LoRA
            st = studio.status()
            rep = (st.get("profile") or {}).get("report") or ""
            if not rep or not os.path.isfile(rep):
                return self.json({"error": "no report"}, 404)
            with open(rep, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/studio/presets":            # your saved slider presets for this model
            key = (qs.get("family") or [""])[0]
            d = os.path.join(STUDIO_PRESETS, re.sub(r"[^A-Za-z0-9_.-]+", "_", key))
            try:
                names = sorted(f[:-5] for f in os.listdir(d) if f.lower().endswith(".json"))
            except OSError:
                names = []
            return self.json({"user": names})
        if path == "/api/prep/options":            # Image Prep: its choices, and whether face tools are here
            from fizgig.dataset import prep as _prep
            return self.json({"modes": list(_prep.MODES), "megapixels": _prep.MEGAPIXELS,
                              "faces": _prep.FACE_TARGETS, "face_tools": prep_jobs.faces_available(),
                              "face_settings": prep_jobs.face_settings()})
        if path == "/api/prep/summary":            # the What will happen card
            from fizgig.dataset import prep as _prep
            g = lambda k, d="": (qs.get(k) or [d])[0]
            folder = g("folder")
            try:
                mp = float(g("mp", "1.0"))
                train = float(g("train_mp")) if g("train_mp") else None
            except ValueError:
                mp, train = 1.0, None
            r = _prep.summary(folder, g("mode", _prep.MODES[0]), g("replace") == "1", mp, train)
            try:
                names = os.listdir(folder)
            except OSError:
                names = []
            exts = {os.path.splitext(n)[1].lower() for n in names}
            r["audio_only"] = not r["count"] and bool(exts & {".wav", ".mp3", ".flac", ".m4a"})
            r["clips"] = sum(1 for n in names if n.lower().endswith(".mp4"))
            return self.json(r)
        if path == "/api/prep/status":
            return self.json(prep_jobs.status(int((qs.get("since") or ["0"])[0] or 0)))
        if path == "/api/look/status":
            return self.json(prep_jobs.look_status())
        if path == "/api/captions/status":
            return self.json(captions.status())
        if path == "/api/runs":
            runs = history.runs()[:200]
            from fizgig.web import image_loss
            owners = image_loss.owner_runs(runs)
            from fizgig.training import step_log
            newest = set()                       # a run's loss file is its name's: only the newest such run owns it
            for r in runs:
                key = (os.path.normcase(os.path.abspath(r.get("output_dir") or ".")), (r.get("lora") or "").lower())
                d = get_family(r.get("family") or "")
                r["image_loss"] = bool(owners.get(r.get("id")) and d is not None and getattr(d, "loss_watch", False))
                r["loss_graph"] = key not in newest and os.path.isfile(
                    step_log.loss_path(r.get("output_dir") or "", r.get("lora") or ""))
                newest.add(key)
            return self.json({"runs": runs})
        if path == "/api/queue/item":                # a queued run's settings, for Edit
            it = webqueue.get((qs.get("id") or [""])[0])
            if it is None:
                return self.json({"error": "That run has left the queue."}, 404)
            return self.json({k: it.get(k) for k in ("id", "family", "values", "image_folder", "lora")})
        if path == "/api/runs/loss":                 # a run's loss by step (the loss graph); live=1: the run going now
            from fizgig.training import step_log
            if (qs.get("live") or [""])[0] == "1":
                with _lock:
                    folder, name = _run["output_dir"], _run["lora"]
            else:
                rec = history.get((qs.get("id") or [""])[0]) or {}
                folder, name = rec.get("output_dir") or "", rec.get("lora") or ""
            data = step_log.read(step_log.loss_path(folder, name)) if folder and name else None
            return self.json(data or {"points": [], "epochs": [], "steps": 0})
        if path == "/api/runs/image_loss":           # one run's per-image residuals by epoch (the loss watch's)
            from fizgig.web import image_loss
            run = history.get((qs.get("id") or [""])[0])
            d = get_family((run or {}).get("family") or "")
            res = image_loss.residuals(run.get("output_dir")) if run and d is not None and d.loss_watch else None
            if res is None:
                return self.json({"error": "This run has no per-image loss log."}, 404)
            pics = image_loss.image_paths(run.get("image_folder"), [im["key"] for im in res["images"]])
            queued, applied = image_loss.caption_state(run.get("output_dir"))
            return self.json({**res, "live": run.get("result") in ("running", "paused"),
                              "images": [{**im, "path": pics.get(im["key"], ""), "queued": im["key"] in queued,
                                          "applied": (applied.get(im["key"]) or [{}])[-1].get("epoch")}
                                         for im in res["images"]]})
        if path == "/api/runs/caption":              # one training image's caption, for the per-image editor
            from fizgig.web import image_loss
            run = history.get((qs.get("id") or [""])[0]) or {}
            text, ok = image_loss.read_caption(run.get("image_folder"), (qs.get("key") or [""])[0],
                                               captions.caption_ext((qs.get("ext") or [""])[0]))
            return self.json({"caption": text, "folder_ok": ok})
        if path == "/api/about":                     # ?check=1: look for a newer release again (the About box's)
            from fizgig.web import update_check
            if (qs.get("check") or [""])[0] == "1":
                update_check.start()
            return self.json({**about_info(), "release": update_check.status()})
        if path == "/api/datasets":
            return self.json(list_datasets())
        if path == "/api/upload/status":
            try:
                mt = (qs.get("mtime") or [""])[0]
                return self.json(upload_status((qs.get("folder") or [""])[0], (qs.get("name") or [""])[0],
                                               bool((qs.get("single") or [""])[0]), int((qs.get("size") or ["0"])[0]),
                                               float(mt) if mt else None))
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/upload/root":
            return self.json({"root": uploads_root(), "set": bool(pref_path(load_prefs(), "input_dataset_dir")),
                              "local": self._local_viewer()})
        if path == "/api/pods":                      # the Pods tab: saved pods + the account's, each probed
            from fizgig.web import pod_stop, pods
            if on_pod():
                return self.json({"error": "The Pods tab is for the Fizgig on your own computer."}, 400)
            p = load_prefs()
            key = pod_stop.api_key(p)
            return self.json({**pods.listing(p, key, update_prefs, about_info()["commit"]), "key_saved": bool(key)})
        if path == "/api/lan/firewall":              # Preferences > Web UI: is Fizgig let through (Windows)?
            from fizgig.web import lan
            if not self._local_viewer():
                return self.json({"error": "Only on the computer Fizgig runs on."}, 403)
            return self.json(lan.firewall_status(self.server.server_address[1]))
        if path == "/api/pods/gpus":                 # the Pods tab's GPU picker: RunPod's cards, prices and stock
            from fizgig.web import pod_stop, pods
            if on_pod():
                return self.json({"error": "The Pods tab is for the Fizgig on your own computer."}, 400)
            try:
                return self.json(pods.offers(pod_stop.api_key(load_prefs()), (qs.get("dc") or [""])[0]))
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/pods/probe":                # one pod, while it starts
            from fizgig.web import pods
            url = (qs.get("url") or [""])[0]
            if on_pod() or not url.startswith(("http://", "https://")):
                return self.json({"error": "no"}, 400)
            return self.json({"state": pods.probe(pods.parse(url)["url"])})
        if path == "/api/stats":
            with _lock:
                return self.json(dict(_stats))
        if path == "/api/run":
            with _lock:
                r = {k: v for k, v in _run.items() if k != "proc"}
                r["running"] = _run["proc"] is not None and _run["proc"].poll() is None
            r["resume"] = r.get("resume") if r["state"] == "paused" else ""
            r["web_queue"] = webqueue.status()
            from fizgig.web import pod_stop
            r["pod_stop"] = pod_stop.status()
            r["slider_live"] = _slider_live_status(r) if r["running"] else None
            return self.json(r)
        if path == "/api/log":
            return self._log_stream(int((qs.get("since") or ["0"])[0] or 0))
        if path == "/api/browse":
            return self._browse((qs.get("path") or [""])[0])
        if path == "/api/samples":
            return self._samples((qs.get("dir") or [""])[0], (qs.get("run") or [""])[0])
        if path == "/api/likeness":                 # the gallery's likeness scores (resumes the scorer)
            from fizgig.web import gallery_likeness
            return self.json(gallery_likeness.state((qs.get("dir") or [""])[0], (qs.get("dataset") or [""])[0]))
        if path == "/api/thumb":
            p = (qs.get("path") or [""])[0]
            if os.path.splitext(p)[1].lower() not in _PIC or not os.path.isfile(p):
                return self.json({"error": "not a picture"}, 404)
            data = _thumb(p)
            if data is None:
                return self._stream(p, attachment=False)
            return self._send(200, data, "image/jpeg", extra=[("Cache-Control", "private, max-age=3600")])
        if path == "/api/media":
            p = (qs.get("path") or [""])[0]
            if os.path.splitext(p)[1].lower() not in MEDIA or not os.path.isfile(p):
                return self.json({"error": "not a picture, sound or clip"}, 404)
            # in pieces, with byte ranges: songs and clips seek; kept by the browser when the page names its time
            return self._stream(p, attachment=False, keep=bool((qs.get("v") or [""])[0]))
        if path == "/api/checkpoints":
            return self._checkpoints((qs.get("dir") or [""])[0])
        if path == "/api/outputs":
            return self.json(list_outputs(fizgig_only=(qs.get("all") or ["0"])[0] != "1"))
        if path == "/api/outputs/previews":
            folder, name = (qs.get("dir") or [""])[0], (qs.get("name") or [""])[0]
            if os.path.abspath(folder) not in output_folders():
                return self.json({"error": "not an output folder"}, 404)
            try:
                zpath = samples_zip(folder, name)
            except ValueError as e:
                return self.json({"error": str(e)}, 404)
            try:
                return self._stream(zpath, filename=re.sub(r'[\\/:*?"<>|]+', "_", name) + "_previews.zip")
            finally:
                try:
                    os.remove(zpath)             # sent (or abandoned): it was only ever for this download
                except OSError:
                    pass
        if path == "/api/datasets/zip":            # a training folder (or its captions) to download
            folder = (qs.get("folder") or [""])[0]
            caps = (qs.get("captions") or ["0"])[0] == "1"
            try:
                zpath = dataset_zip(folder, caps, captions.caption_ext((qs.get("ext") or [""])[0]))
            except ValueError as e:
                return self.json({"error": str(e)}, 404)
            stem = re.sub(r'[\\/:*?"<>|]+', "_", os.path.basename(os.path.normpath(folder)) or "dataset")
            try:
                return self._stream(zpath, filename=f"{stem}_{'captions' if caps else 'dataset'}.zip")
            finally:
                try:
                    os.remove(zpath)
                except OSError:
                    pass
        if path == "/api/download":
            p = (qs.get("path") or [""])[0]
            if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
                return self.json({"error": "not a checkpoint"}, 404)
            return self._stream(p)
        if path == "/api/captions":
            return self._captions((qs.get("folder") or [""])[0], (qs.get("ext") or [".txt"])[0])
        return self.json({"error": "not found"}, 404)

    def _log_stream(self, since):
        """Server-sent events: every console line after `since`, then new ones as they arrive."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = since
        try:
            while True:
                with _lock:
                    fresh = [(n, t) for n, t in _log if n > last]
                    if not fresh:
                        _log_event.wait(timeout=15)
                        fresh = [(n, t) for n, t in _log if n > last]
                if fresh:
                    for n, t in fresh:
                        self.wfile.write(f"id: {n}\ndata: {json.dumps(t)}\n\n".encode("utf-8"))
                    last = fresh[-1][0]
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _local_viewer(self):
        """The page is open on the machine running Fizgig (a browser there needs no uploads: Choose reaches the
        same files). Through a proxy (a pod) it never is."""
        return (self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")
                and not self.headers.get("X-Forwarded-For"))

    def _browse(self, target):
        from fizgig.web import drives
        if target == drives.DRIVES:                  # This computer: every drive / mounted disk
            return self.json({"path": drives.DRIVES, "label": "This computer", "parent": "",
                              "dirs": drives.listing(), "files": [], "places": self._places()})
        target = target or BASE_DIR
        if not os.path.isdir(target):
            target = os.path.dirname(target) if os.path.isdir(os.path.dirname(target)) else BASE_DIR
        dirs, files = [], []
        try:
            for e in sorted(os.scandir(target), key=lambda e: e.name.lower()):
                if e.name.startswith("."):
                    continue
                (dirs if e.is_dir() else files).append({"name": e.name, "path": e.path})
        except OSError:
            pass
        parent = os.path.dirname(target)
        return self.json({"path": target, "parent": parent if parent != target else drives.DRIVES,   # a root: Up = the drives
                          "dirs": dirs, "files": files[:2000], "places": self._places()})

    def _places(self):
        from fizgig.web import drives
        last_out = (load_state().get("values") or {}).get("LORA_OUTPUT_DIR") or default_output_dir()
        places = [("Training data", uploads_root()), ("Trained LoRAs", last_out), ("Uploaded files", UPLOAD_FILES),
                  ("Fizgig", BASE_DIR)]
        return [{"name": n, "path": p} for n, p in places if os.path.isdir(p)] + [{"name": "This computer",
                                                                                   "path": drives.DRIVES}]

    def _samples(self, folder, run=""):
        """A sample folder's previews, newest first, each with its run, epoch, prompt number and strength; `runs` the
        runs that have previews here (newest first). With run: only that run's, and the prompts it used."""
        try:
            items = [e for e in os.scandir(folder) if os.path.splitext(e.name)[1].lower() in MEDIA]
        except OSError:
            return self.json({"items": [], "runs": [], "prompts": []})
        items.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        infos = {e.name: sample_info(e.name) for e in items}
        runs = list(dict.fromkeys(infos[e.name]["run"] for e in items if infos[e.name]))
        if run:
            items = [e for e in items if infos[e.name].get("run") == run]
        out = pair_songs([(e.name, e.path, {"time": e.stat().st_mtime, **infos[e.name]}) for e in items[:1500]])
        return self.json({"items": out[:1000] if run else out[:300], "runs": runs,
                          "prompts": run_prompts(os.path.dirname(folder.rstrip("\\/")), run) if run else []})

    def _checkpoints(self, folder):
        """The run's saved LoRAs / checkpoints in its output folder, newest first."""
        try:
            items = [e for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(".safetensors")]
        except OSError:
            return self.json({"items": []})
        now = time.time()
        items = [e for e in items if now - e.stat().st_mtime > 5]       # not one still being written
        items.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        base = lambda n: (lambda m: m.group(1) if m else n)(_CKPT.match(n[:-len(".safetensors")]))
        return self.json({"items": [{"name": e.name, "path": e.path, "size": e.stat().st_size,
                                     "time": e.stat().st_mtime, "kind": _checkpoint_kind(e.path),
                                     "run": base(e.name)} for e in items]})

    def _captions(self, folder, ext):
        out = []
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return self.json({"items": []})
        for n in names:
            stem, e = os.path.splitext(n)
            if e.lower() not in MEDIA:
                continue
            cap = os.path.join(folder, stem + ext)
            try:
                with open(cap, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                text = None
            p = os.path.join(folder, n)
            out.append((n, p, {"caption": text, "size": _media_size(p)}))
        return self.json({"items": pair_songs(out)})

    # -- POST
    def do_POST(self):
        self._guarded(self._post)

    def do_GET(self):
        self._guarded(self._get)

    def _guarded(self, handle):
        """A request that fails inside Fizgig still gets an answer - the error, said - so the page shows what went
        wrong instead of the browser's bare "Failed to fetch"."""
        try:
            return handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                  # the page went away mid-answer: nobody to tell
        except Exception as e:
            logger.exception(f"{self.command} {self.path}")
            try:
                self.json({"error": f"Fizgig hit an error doing that ({type(e).__name__}: {e})."}, 500)
            except Exception:
                pass

    def _app_window(self):
        """A call from the Fizgig window that started this server: it carries that window's key."""
        key = os.environ.get("FIZGIG_APP_KEY", "")
        return bool(key) and hmac.compare_digest(self.headers.get("X-Fizgig-App-Key") or "", key)

    def _post(self):
        if not self._host_ok():
            return self.json({"error": "bad host"}, 403)
        path = urlparse(self.path).path
        if path in ("/api/app/busy", "/api/app/stop_run"):   # the window, closing: is a run training? stop it
            if not self._app_window():
                return self.json({"error": "no"}, 403)
            with _lock:
                busy = _run["proc"] is not None and _run["proc"].poll() is None
                lora = _run.get("lora") or ""
            if path == "/api/app/busy":
                return self.json({"training": busy, "lora": lora})
            return self.json({"stopped": _stop_and_wait()})
        if path == "/api/login":
            if self.headers.get("X-Fizgig-Token") != TOKEN:
                return self.json({"error": "Reload the page."}, 403)
            self._body_cache = self._body()
            pwd = str(self._body_cache.get("password") or "")
            if self.password:
                # wrong guesses lock out only the device making them, for longer each time (1 s, 2 s, ... 60 s),
                # forgotten after 15 minutes; nothing waits while holding the lock, so the owner's own login on
                # another device is never held up. Bytes, so a non-ASCII password compares.
                # on a pod everyone arrives through RunPod's proxy (one address), so the visitor is the address the
                # proxy adds - the LAST in X-Forwarded-For (earlier ones are whatever the visitor sent). Off a pod the
                # header could be forged outright, so the connection's own address counts.
                fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[-1].strip() if on_pod() else ""
                ip, now = fwd or self.client_address[0], time.time()
                with _login_lock:
                    n, until, last = _login_fails.get(ip, (0, 0.0, 0.0))
                    if now - last > 900:
                        n, until = 0, 0.0
                    if now < until:
                        return self.json({"error": f"Too many wrong passwords. Try again in {int(until - now) + 1} s."},
                                         429)
                    if not hmac.compare_digest(pwd.encode("utf-8"), self.password.encode("utf-8")):
                        n += 1
                        _login_fails[ip] = (n, now + min(60.0, float(n)), now)
                        if len(_login_fails) > 1000:                  # many addresses: keep the recent ones
                            for k in sorted(_login_fails, key=lambda k: _login_fails[k][2])[:500]:
                                _login_fails.pop(k, None)
                        return self.json({"error": "Wrong password."}, 401)
                    _login_fails.pop(ip, None)
            return self.json({"ok": True}, extra=[self._cookie(remember=bool(self._body_cache.get("remember")))])
        if not self._api_ok():
            return None
        if path == "/api/upload":                    # a raw file body, not JSON
            qs = parse_qs(urlparse(self.path).query)
            try:
                d = _save_upload((qs.get("folder") or [""])[0], (qs.get("name") or [""])[0], self.rfile,
                                 int(self.headers.get("Content-Length") or 0), single=bool(qs.get("single")))
                if qs.get("single"):
                    d["path"] = os.path.join(d["folder"], d["saved"][0])
                return self.json(d)
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/upload/chunk":               # one piece of a resumable upload
            qs = parse_qs(urlparse(self.path).query)
            g = lambda k: (qs.get(k) or [""])[0]
            try:
                d = save_chunk(g("folder"), g("name"), self.rfile, int(self.headers.get("Content-Length") or 0),
                               int(g("offset") or 0), int(g("total") or 0), single=bool(g("single")),
                               mtime=float(g("mtime")) if g("mtime") else None)
                if d.get("saved") and g("single"):
                    d["path"] = os.path.join(d["folder"], d["saved"][0])
                return self.json(d)
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        b = self._body()
        try:
            if path == "/api/form/state":
                if b.get("family") == refmod.KEY:
                    return self.json(form.refmod_state(b.get("values") or {}, b.get("image_folder") or ""))
                desc = get_family(b.get("family", ""))
                if desc is None:
                    return self.json({"error": "unknown family"}, 400)
                prefs = load_prefs()
                cap = pref_path(prefs, "krea2_text_encoder")
                sp = desc.preview_speed()
                spf = pref_path(prefs, sp.pref_key) if sp is not None and sp.pref_key else ""
                ck = desc.preview_checkpoint()
                ckf = pref_path(prefs, ck[0].pref_key) if ck else ""
                vals = b.get("values") or {}
                st = form.state(desc, vals, b.get("image_folder") or "",
                                captioner=cap if cap and os.path.isfile(cap) else "",
                                speed_file=bool(spf and os.path.isfile(spf)),
                                checkpoint_file=bool(ckf and os.path.isfile(ckf)))
                if st["kind"] in ("standard", "finetune"):     # the Subjects card: what each folder trains as
                    mpn = _multi_person_notes(vals, b.get("image_folder") or "",
                                              multi_person=st["kind"] == "standard" and
                                              bool(vals.get("FAMILY_MULTI_PERSON")))
                    st["placeholders"] = mpn.pop("__placeholders", {})
                    st["notes"].update(mpn)
                if st["kind"] == "finetune" and desc.finetune:      # the fine-tune card's plan line (the desktop's)
                    from fizgig.web import ft_plan
                    st["notes"]["FAMILY_FT_ROTATIONS"], st["pending"] = ft_plan.line(
                        desc, vals, pref_path(prefs, desc.pref_for("dit")),
                        gpu_env(dict(os.environ, PYTHONPATH=os.path.join(BASE_DIR, "src")), prefs))
                return self.json(st)
            if path == "/api/captions/write_all":     # Edit / Slider pairs: one line as every photo's caption
                return self.json(write_all_captions(str(b.get("text") or "").strip(), str(b.get("folder") or ""),
                                                    str(b.get("ext") or ".txt").strip() or ".txt",
                                                    bool(b.get("overwrite"))))
            if path == "/api/preset/save":            # Save preset: the desktop's format and folder
                desc = _preset_family(b.get("family", ""))
                name = str(b.get("name") or "").strip()
                if desc is None:
                    return self.json({"error": "unknown family"}, 400)
                if not name or any(c in name for c in '<>:"/\\|?*'):
                    return self.json({"error": 'A preset name can\'t be empty or hold < > : " / \\ | ? *'}, 400)
                d = os.path.join(PRESETS_DIR, desc.gui_label)
                path_ = os.path.join(d, name + ".json")
                if os.path.exists(path_) and not b.get("overwrite"):
                    return self.json({"exists": True})
                os.makedirs(d, exist_ok=True)
                with open(path_, "w", encoding="utf-8") as f:
                    vals = b.get("values") or {}
                    json.dump(form.to_preset(desc, vals) if desc is not _RefModPresets else
                              {k: v for k, v in vals.items() if not str(k).startswith("__") and k != "H3_TRAIN_BASE"},
                              f, indent=4)
                return self.json({"ok": True, "user": user_presets(desc)})
            if path == "/api/preset":
                desc = _preset_family(b.get("family", ""))
                preset = dict(desc.presets or ()).get(b.get("name")) if desc else None
                if preset is None and desc is not None and b.get("name") in user_presets(desc):
                    try:                             # one you saved, in either app
                        with open(os.path.join(PRESETS_DIR, desc.gui_label, b["name"] + ".json"), encoding="utf-8") as f:
                            preset = json.load(f)
                    except (OSError, ValueError):
                        return self.json({"error": "That preset's file can't be read."}, 400)
                if preset is None:
                    return self.json({"error": "unknown preset"}, 400)
                if desc is _RefModPresets:
                    return self.json({"values": form.refmod_preset(b.get("values") or {}, preset)})
                return self.json({"values": form.apply_preset(desc, b.get("values") or {}, preset)})
            if path == "/api/mkdir":                     # the folder picker's New folder
                parent, name = str(b.get("parent") or ""), str(b.get("name") or "").strip()
                if not os.path.isdir(parent):
                    return self.json({"error": "That folder isn't there any more."}, 400)
                name = _upload_name(_tidy_upload_name(name), "Folder name")
                new = os.path.join(parent, name)
                if os.path.exists(new):
                    return self.json({"error": f"{name} is already there."}, 400)
                os.makedirs(new)
                return self.json({"ok": True, "path": new})
            if path in ("/api/queue/add", "/api/queue/update"):   # Add to queue (or a queued run edited): checked
                fam, vals, folder = b.get("family", ""), b.get("values") or {}, b.get("image_folder") or ""  # like Start
                editing = str(b.get("id") or "") if path == "/api/queue/update" else ""
                if path == "/api/queue/update" and webqueue.get(editing) is None:
                    return self.json({"error": "That run has left the queue (it started, or was removed) - add "
                                               "this as a new one instead."}, 400)
                desc, inputs, plan = _plan(fam, vals, folder)
                if plan.problems:
                    return self.json({"problems": plan.problems})
                out_dir, lora = inputs["LORA_OUTPUT_DIR"], inputs["LORA_NAME"]
                n = webqueue.clashes(out_dir, lora, skip=editing)
                with _lock:
                    live_clash = (_run["proc"] is not None and _run["proc"].poll() is None
                                  and os.path.normcase(os.path.abspath(_run["output_dir"] or "")) ==
                                  os.path.normcase(os.path.abspath(out_dir)) and _run["lora"].lower() == lora.lower())
                if n or live_clash:
                    return self.json({"error": f"Run {n} in the queue already saves as {lora} in that folder - give this "
                                      "one another LoRA name, or they'd overwrite each other's files." if n else
                                      f"The run going now saves as {lora} in that folder - give this one another LoRA "
                                      "name."}, 400)
                if editing:
                    if not webqueue.update(editing, fam, vals, folder, lora, out_dir):
                        return self.json({"error": "That run has left the queue - add this as a new one instead."},
                                         400)
                    ids = [i["id"] for i in webqueue.status()["items"]]
                    return self.json({"ok": True, "id": editing, "position": ids.index(editing) + 1
                                      if editing in ids else 0})
                item = webqueue.add(fam, vals, folder, lora, out_dir)
                with _lock:
                    idle = not (_run["proc"] is not None and _run["proc"].poll() is None) and \
                        _run["state"] not in ("starting", "running")
                if idle and not webqueue.status()["held"]:
                    webqueue.start_next(_queue_starter)          # nothing running: it starts now (or waits, saying why)
                return self.json({"ok": True, "id": item["id"], "position": len(webqueue.status()["items"])})
            if path == "/api/runs/notes":
                history.set_notes(str(b.get("id") or ""), str(b.get("text") or ""))
                return self.json({"ok": True})
            if path == "/api/queue/remove":
                webqueue.remove(str(b.get("id") or ""))
                return self.json({"ok": True})
            if path == "/api/queue/move":
                webqueue.move(str(b.get("id") or ""), -1 if int(b.get("delta") or 0) < 0 else 1)
                return self.json({"ok": True})
            if path == "/api/queue/start":
                with _lock:
                    paused = _run["state"] == "paused"
                if paused:                              # as the desktop's Start next now: it would leave the run
                    return self.json({"error": "A run is paused. Resume it first: the queue carries on after it "
                                               "finishes."}, 400)
                webqueue.release(_queue_starter)
                return self.json({"ok": True})
            if path == "/api/queue/hold":
                webqueue.hold("Held by you. Press Start queue to carry on.")
                return self.json({"ok": True})
            if path == "/api/runs/delete":              # Delete run: asks first (confirm false = what it'd delete)
                r = delete_run(str(b.get("id") or ""), confirm=bool(b.get("confirm")))
                return self.json(r, 400 if r.get("error") else 200)
            if path == "/api/runs/settings":            # Start again with these settings
                rec = history.get(str(b.get("id") or ""))
                if rec and rec.get("family") in (refmod.KEY, refmod.LABEL):
                    return self.json({"family": refmod.KEY, "values": {k: v for k, v in (rec.get("values") or {}).items()
                                                                        if not str(k).startswith("__")},
                                      "image_folder": rec.get("image_folder") or ""})
                desc = get_family(rec.get("family", "")) if rec else None
                if desc is None:
                    return self.json({"error": "That run's model isn't available here."}, 400)
                vals = rec.get("values") or {}
                if rec.get("source") == "desktop":       # the desktop's settings, as a preset applies them, with the
                    mk = {o.key for o in desc.options if o.tab == "model"}   # Base Model card's options beside them
                    vals = {**form.apply_preset(desc, {}, vals, whole=True),
                            **{k: v for k, v in vals.items() if k in mk}}
                return self.json({"family": desc.key, "values": vals, "image_folder": rec.get("image_folder") or ""})
            if path == "/api/plan":
                _, _, plan = _plan(b.get("family", ""), b.get("values") or {}, b.get("image_folder") or "")
                return self.json({"problems": plan.problems, "stages": [{"name": s.name, "cmd": s.cmd}
                                                                         for s in plan.stages],
                                  "dataset_toml": plan.dataset_toml})
            if path == "/api/start":
                return self.json(_start(b.get("family", ""), b.get("values") or {}, b.get("image_folder") or ""))
            if path == "/api/resume":                    # as the desktop's Resume Training
                with _lock:
                    last = {k: _run.get(k) for k in ("state", "family", "values", "image_folder", "resume",
                                                     "resume_mode")}
                if last["state"] != "paused" or not last["resume"]:
                    return self.json({"error": "No paused run to resume."}, 400)
                last["values"] = dict(last["values"] or {})
                for k in _SIZE_KEYS:                     # Resume at the size the form has now: caches again first
                    if str(b.get(k) or "").strip():
                        last["values"][k] = str(b[k]).strip()
                if last["resume_mode"] == "ft":
                    if not os.path.isfile(last["resume"]):
                        return self.json({"error": "The paused fine-tune's checkpoint isn't there any more: "
                                                   + last["resume"]}, 400)
                    v = last["values"] or {}
                    left, done, cont = launch.ft_rotations_left(last["resume"], v.get("FAMILY_FT_ROTATIONS"),
                                                                v.get("FAMILY_FT_ROTATE_EVERY"))
                    if left <= 0:
                        return self.json({"error": f"This fine-tune has already trained {done} rotation(s), and Train "
                                                   f"for is {v.get('FAMILY_FT_ROTATIONS')}: "
                                                   f"{os.path.basename(last['resume'])} is the finished model. To "
                                                   f"train it further, raise Train for and press Resume again."}, 400)
                    return self.json(_start(last["family"], v, last["image_folder"],
                                            ft_continue={"continue": cont, "rotations": left},
                                            note=f"=== Resuming the fine-tune from {os.path.basename(last['resume'])}: "
                                                 f"{left} rotation(s) left ==="))
                if not os.path.isdir(last["resume"]):
                    return self.json({"error": "The paused run's saved state isn't there any more: "
                                               + last["resume"]}, 400)
                return self.json(_start(last["family"], last["values"], last["image_folder"], resume=last["resume"],
                                        note=f"=== Resuming from {last['resume']} ==="))
            if path == "/api/run/slider_prompts":    # a prompt slider's new ends, taken at its next epoch boundary
                with _lock:
                    running = _run["proc"] is not None and _run["proc"].poll() is None
                    run = {k: _run.get(k) for k in ("family", "values", "output_dir", "history_id")}
                if not running or not _prompt_slider_run(run):
                    return self.json({"error": "No prompt slider is training right now."}, 400)
                new = {k: str(b.get(k) or "").strip() for k in ("base", "pos", "neg")}
                try:
                    g = float(str(b.get("guidance") or "").strip())
                except ValueError:
                    g = 0.0
                if not all(new.values()):
                    return self.json({"error": "Fill in all three: what the picture is, and what each end adds."}, 400)
                if g <= 0:
                    return self.json({"error": "Push strength must be a number above 0."}, 400)
                _write_json(os.path.join(run["output_dir"], SLIDER_LIVE), {**new, "guidance": g})
                changes = {_SLIDER_LIVE_KEYS[k]: v for k, v in {**new, "guidance": f"{g:g}"}.items()}
                with _lock:                              # a Pause and Resume, or Start again, keeps them
                    if isinstance(_run.get("values"), dict):
                        _run["values"].update(changes)
                if run.get("history_id"):
                    history.set_values(run["history_id"], changes)
                return self.json({"ok": True})
            if path == "/api/run/sample_override":   # the Samples tab's live previews: the run going now's override
                with _lock:
                    out = _run["output_dir"] if _run["proc"] is not None and _run["proc"].poll() is None else ""
                if not out:
                    return self.json({"error": "Nothing is training right now."}, 400)
                side = os.path.join(out, ".sample_override.json")
                prompts = [ln.strip() for ln in str(b.get("prompts") or "").splitlines() if ln.strip()]
                if not b.get("on") or not prompts:
                    try:
                        os.remove(side)
                    except OSError:
                        pass
                    return self.json({"ok": True, "on": False})
                _write_json(side, {"prompts": prompts, "seed": _seed(b.get("seed"), 1234),
                                   "width": _seed(b.get("width"), 1024), "height": _seed(b.get("height"), 1024)})
                return self.json({"ok": True, "on": True})
            if path == "/api/run/now":               # Sample now / Save now: a request the trainer takes after a step
                what = str(b.get("what") or "")
                if what not in ("sample", "save"):
                    raise ValueError("Unknown request.")
                with _lock:
                    running = _run["proc"] is not None and _run["proc"].poll() is None and _run["stage"] == "Training"
                    out, fam, vals = _run["output_dir"], _run["family"], dict(_run["values"])
                if not running or not out:
                    return self.json({"error": "Nothing is training right now."}, 400)
                desc = get_family(fam)
                if desc is not None and vals and form._kind(desc, vals) == "finetune":
                    return self.json({"error": "A fine-tune saves and previews at whole rotations only."}, 400)
                open(os.path.join(out, f".{what}_now"), "w").close()
                return self.json({"ok": True})
            if path == "/api/pause":
                with _lock:
                    out, running = _run["output_dir"], _run["proc"] is not None and _run["proc"].poll() is None
                if not running:
                    return self.json({"error": "Nothing is running."}, 400)
                if _run.get("family") == refmod.KEY:
                    return self.json({"error": "A RefMod can't pause: it is made in one go. Stop ends it."}, 400)
                with open(os.path.join(out, ".pause_requested"), "w", encoding="utf-8") as f:
                    f.write("pause\n")
                with _lock:
                    _run["pause"] = True
                append_log("\n⏸ Pause requested: the run stops cleanly at the end of this epoch.\n")
                return self.json({"ok": True})
            if path == "/api/fetch":
                _start_fetch(b.get("family", ""), b.get("token", ""), bool(b.get("optional")))
                return self.json({"ok": True})
            if path == "/api/fetch/cancel":
                return self.json({"ok": _cancel_fetch()})
            if path == "/api/stop":
                return self.json({"ok": _stop()})
            if path == "/api/logout":                     # this device: its login cookie goes
                return self.json({"ok": True}, extra=[("Set-Cookie", "fizgig_session=; Path=/; Max-Age=0; HttpOnly; "
                                                                     "SameSite=Strict")])
            if path == "/api/logout_all":                 # every device: a new key makes every login stale
                _SESSION_KEY[0] = secrets.token_urlsafe(32)
                update_prefs({"web_session_key": _SESSION_KEY[0]})
                return self.json({"ok": True})
            if path == "/api/prefs":
                allowed = {k for d in training_families() for k in form.model_keys(d)} | set(SETTING_PREFS) | {
                    "input_dataset_dir", "welcome_v8_hidden"}
                ch = b.get("changes") or {}
                if b.get("reset"):                      # Reset to defaults: the settings, never the model files
                    ch = {**{k: "" for k in SETTING_PREFS}, **{k: v for k, v in PREF_DEFAULTS.items()
                                                                  if k in SETTING_PREFS}}
                if "inference_int8" in ch:
                    ch["inference_int8"] = "1" if ch["inference_int8"] in (True, "1", 1) else "0"
                upd = {k: str(v) for k, v in ch.items() if k in allowed}
                if "cuda_device" in ch:
                    v = str(ch["cuda_device"] or "").strip()
                    if v and v not in {g["uuid"] for g in list_gpus()} | {str(g["index"]) for g in list_gpus()}:
                        raise ValueError("That GPU isn't in this machine.")
                    upd["cuda_device"] = v
                if "runpod_stop_when_done" in ch:
                    upd["runpod_stop_when_done"] = "1" if ch["runpod_stop_when_done"] in (True, "1", 1) else "0"
                if "runpod_api_key" in ch:
                    upd["runpod_api_key"] = str(ch["runpod_api_key"] or "").strip()
                if "web_port" in ch:
                    port = int(ch["web_port"])
                    if not 1024 <= port <= 65535:
                        raise ValueError("The port must be between 1024 and 65535.")
                    upd["web_port"] = str(port)
                update_prefs(upd)
                return self.json({"ok": True})
            if path == "/api/state":
                one = b.get("model_values")
                if isinstance(one, dict) and one.get("family") and isinstance(one.get("values"), dict):
                    with _lock:                         # one model's settings, merged into the saved ones
                        st = load_state()
                        vals = st.get("values") if isinstance(st.get("values"), dict) else {}
                        vals[str(one["family"])] = one["values"]
                        st["values"] = vals
                        _write_json(STATE_FILE, st)
                save_state({k: v for k, v in b.items() if k in ("family", "image_folder", "tab", "captions", "prep")})
                if isinstance(b.get("image_folder"), str) and b["image_folder"].strip():
                    update_last_used({"image_folder": b["image_folder"].strip()})     # the desktop shows it too
                    remember_folder(b["image_folder"])
                return self.json({"ok": True})
            if path == "/api/trigger":
                set_folder_trigger(str(b.get("folder") or ""), str(b.get("word") or ""))
                return self.json({"ok": True})
            if path == "/api/captions/ai":
                prefs, folder, clips, photos, _ = _caption_args(b)
                cap = pref_path(prefs, "krea2_text_encoder")
                model = str(b.get("model") or "")
                _caption_gpu_check(8.0 if model == captions.QWEN_MODEL else 2.5)
                only = [p for p in (b.get("only") or []) if os.path.dirname(os.path.abspath(p)) ==
                        os.path.abspath(folder)]
                return self.json(captions.ai_captions(
                    prefs, cap if cap and os.path.isfile(cap) else "", folder, model, str(b.get("task") or ""),
                    str(b.get("instruction") or ""), int(b.get("max_tokens") or 120), str(b.get("trigger") or ""),
                    bool(b.get("overwrite")), clips, only=only or None, photos=photos,
                    ext=captions.caption_ext(b.get("ext"))))
            if path.startswith("/api/studio/"):
                try:
                    return self.json(_studio_post(path[len("/api/studio/"):], b))
                except (ValueError, RuntimeError) as e:
                    return self.json({"error": str(e)}, 400)
            if path.startswith("/api/refmod/"):
                try:
                    return self.json(_refmod_post(path[len("/api/refmod/"):], b))
                except (ValueError, RuntimeError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
            if path == "/api/phone_link":           # Open on your phone: the QR code's address, with a one-time pass
                if not self.password:
                    return self.json({"error": "Set a password first (Preferences > Web UI)."}, 400)
                from fizgig.web import pod_stop
                from fizgig.web.port import lan_ip
                if on_pod():
                    pid = pod_stop.pod_id()
                    base = f"https://{pid}-8081.proxy.runpod.net/" if pid else ""
                else:
                    ip = lan_ip()
                    base = f"http://{ip}:{self.server.server_address[1]}/" if ip else ""
                if not base:
                    return self.json({"error": "This computer has no address on a network for a phone to open."}, 400)
                return self.json({"url": base + "?phone=" + _phone_pass_new(), "seconds": 30})
            if path == "/api/restart":              # Restart now: a new password (or none) takes effect
                if on_pod():
                    return self.json({"error": "A pod's password comes from its template: restart the pod."}, 400)
                with _lock:
                    busy = _run["proc"] is not None and _run["proc"].poll() is None
                if busy:
                    return self.json({"error": "A run is going: Fizgig restarts once it's done (pause it, or "
                                               "restart after it finishes)."}, 400)
                _RESTART[0] = True
                threading.Timer(0.3, self.server.shutdown).start()
                return self.json({"ok": True})
            if path == "/api/datasets/forget":      # a folder from elsewhere off Your training data (files untouched)
                forget_folder(b.get("path"))
                return self.json({"ok": True})
            if path == "/api/prefs/password":       # the web UI's password: used from the next start
                pwd = str(b.get("password") or "").strip()
                if pwd and len(pwd) < 8:
                    return self.json({"error": f"Too short: {len(pwd)} of at least 8 characters."}, 400)
                update_prefs({"web_password": pwd})
                return self.json({"ok": True, "saved": bool(pwd)})
            if path == "/api/gizmo/open":           # Gizmo, the clip / voice / music prep app, on this computer
                if on_pod() or not self._local_viewer():
                    return self.json({"error": "Gizmo opens on the computer running Fizgig - use it there."}, 400)
                script = os.path.join(BASE_DIR, "src", "gizmo", "gizmo.pyw" if os.name == "nt" else "gizmo.py")
                if not os.path.isfile(script):
                    return self.json({"error": "Gizmo is missing from your Fizgig folder - update Fizgig."}, 400)
                exe = sys.executable
                if os.name == "nt" and os.path.basename(exe).lower() == "python.exe":
                    exe = os.path.join(os.path.dirname(exe), "pythonw.exe")    # no console window beside it
                subprocess.Popen([exe, script], cwd=BASE_DIR, **({"creationflags": 0x08000000} if os.name == "nt" else {}))
                return self.json({"ok": True})
            if path == "/api/lan/firewall":          # Allow on my network: Windows asks for an administrator's OK
                from fizgig.web import lan
                if not self._local_viewer():
                    return self.json({"error": "Only on the computer Fizgig runs on."}, 403)
                try:
                    return self.json(lan.firewall_allow(self.server.server_address[1], saved_port(load_prefs())))
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
            if path.startswith("/api/pods/"):        # the Pods tab: add, forget, opened, API key, start, stop
                return self.json(_pods_post(path[len("/api/pods/"):], b))
            if path == "/api/pod/keep":             # the pod's auto-stop countdown cancelled
                from fizgig.web import pod_stop
                return self.json({"ok": pod_stop.cancel(append_log)})
            if path == "/api/runs/caption":         # a caption fixed from the per-image loss view
                from fizgig.web import image_loss
                run = history.get(str(b.get("id") or ""))
                if not run:
                    return self.json({"error": "No such run."}, 404)
                try:
                    return self.json(image_loss.save_caption(run, b.get("key"), b.get("caption"),
                                                             captions.caption_ext(b.get("ext"))))
                except (ValueError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
            if path == "/api/likeness":            # three dataset photos to score the samples against; [] clears
                from fizgig.web import gallery_likeness
                try:
                    return self.json({"ok": True, "msg": gallery_likeness.start(
                        str(b.get("dir") or ""), str(b.get("dataset") or ""), b.get("baselines") or [])})
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
            if path in ("/api/prep/run", "/api/look/scan", "/api/look/move"):
                folder = str(b.get("folder") or "")
                norm = lambda p: os.path.normcase(os.path.abspath(p or "."))
                with _lock:
                    training_here = (_run["proc"] is not None and _run["proc"].poll() is None
                                     and norm(_run.get("image_folder")) == norm(folder))
                if training_here:
                    return self.json({"error": "A run is training on this folder now. Wait for it to finish, or "
                                               "stop it first."}, 400)
                try:
                    if path == "/api/prep/run":
                        return self.json(prep_jobs.start(folder, str(b.get("mode") or ""), b.get("mp") or 1.0,
                                                         str(b.get("face") or "Largest Face"),
                                                         b.get("padding") or 20, bool(b.get("replace"))))
                    if path == "/api/look/scan":
                        return self.json(prep_jobs.look_scan(folder, b.get("baselines") or []))
                    return self.json(prep_jobs.look_move(folder, b.get("paths") or [],
                                                         captions.caption_ext(b.get("ext"))))
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
            if path == "/api/prep/face_settings":     # the face finder / YOLO model / look match (prefs.json face_*)
                try:
                    update_prefs(prep_jobs.save_face_settings(b))
                    return self.json(prep_jobs.face_settings())
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
            if path == "/api/prep/test":              # Test face detection on one photo (writes nothing)
                try:
                    return self.json(prep_jobs.test_faces(str(b.get("path") or ""), str(b.get("face") or "")))
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
            if path == "/api/captions/transcribe":   # the desktop's Append Transcription: a clip's speech
                _, folder, *_ = _caption_args(b)
                name = os.path.basename(str(b.get("name") or ""))
                clip = os.path.join(folder, name)
                if not (name.lower().endswith(".mp4") and os.path.isfile(clip)):
                    return self.json({"error": "That clip isn't in the training folder."}, 400)
                if os.path.splitext(name)[0].lower().endswith("_mute"):
                    return self.json({"error": "This clip is muted (_mute): it trains without its sound."}, 400)
                try:
                    p = subprocess.run([sys.executable, "-m", "fizgig.scripts.transcribe_clip", clip],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=900, cwd=BASE_DIR, env=dict(os.environ, PYTHONIOENCODING="utf-8",
                                       PYTHONPATH=os.path.join(BASE_DIR, "src")),
                                       **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
                    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), "")
                    r = json.loads(line) if line else {"error": (p.stderr or "Whisper didn't answer").strip()[-300:]}
                except subprocess.TimeoutExpired:
                    r = {"error": "Whisper took too long on this clip."}
                return self.json(r, 400 if r.get("error") else 200)
            if path == "/api/captions/trigger_check":     # a trigger word against the model's own tokenizer
                from fizgig.families import trigger_check
                try:
                    r = trigger_check.check(str(b.get("family") or ""), str(b.get("trigger") or ""),
                                            str(b.get("cls") or ""), [str(o) for o in b.get("others") or []])
                except Exception as e:                     # the tokenizer couldn't be fetched (offline, first use)
                    r = {"error": f"Couldn't load this model's tokenizer: {e}"}
                return self.json(r, 400 if r.get("error") else 200)
            if path == "/api/captions/static":
                _, folder, clips, photos, voice = _caption_args(b)
                return self.json(captions.static_captions(folder, str(b.get("trigger") or ""),
                                                          bool(b.get("overwrite")), clips, photos, voice,
                                                          captions.caption_ext(b.get("ext"))))
            if path == "/api/captions/songs":
                _, folder, *_ = _caption_args(b)
                return self.json(captions.song_captions(folder, str(b.get("trigger") or ""),
                                                        str(b.get("words") or ""), bool(b.get("overwrite")),
                                                        captions.caption_ext(b.get("ext"))))
            if path == "/api/captions/replace":
                _, folder, *_ = _caption_args(b)
                return self.json(captions.find_replace(folder, str(b.get("find") or ""), str(b.get("replace") or ""),
                                                       bool(b.get("apply")), captions.caption_ext(b.get("ext"))))
            if path == "/api/captions/translate":
                prefs, folder, *_ = _caption_args(b)
                _caption_gpu_check(1.0)
                return self.json(captions.translate(prefs, folder, bool(b.get("skip_existing")),
                                                    captions.caption_ext(b.get("ext"))))
            if path == "/api/captions/stop":
                return self.json({"ok": captions.stop()})
            if path == "/api/captions/unload":
                return self.json({"ok": captions.unload()})
            if path == "/api/captions/instruction":
                ovr = captions.instruction_override(load_prefs(), str(b.get("key") or "custom"),
                                                    str(b.get("text") or ""))
                update_prefs({"caption_qwen_instructions": ovr})
                return self.json({"ok": True})
            if path == "/api/royale/render":
                from fizgig.web import royale
                fams = {d.key: d for d in royale.families()}
                scan = royale.scan(str(b.get("folder") or "").strip())
                items = royale.pick(scan["items"], b.get("max") or "12")
                if not items:
                    raise ValueError("No checkpoints found - pick a folder with trained LoRAs.")
                prompt = str(b.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("Enter a prompt (include your trigger word).")
                key = royale.family_of([p for _l, p in items], fams) or str(b.get("family") or "")
                desc = fams.get(key)
                if desc is None:
                    raise ValueError("Pick a model first.")
                if _gpu_training():
                    raise ValueError("A training run is using the GPU. Royale can render when it finishes.")
                from fizgig.web import tools
                if tools.holding_gpu():
                    raise ValueError("The Profiler is rendering on the GPU - wait for it, or Stop it.")
                prefs = load_prefs()
                if desc.video_workbench:
                    kw, _status, problems = studio.plan_video(desc, prefs, pref_path)
                    kw["audio_vae_path"] = ""
                    settings = None
                else:
                    kw, settings, _status, problems = studio.plan(desc, prefs, pref_path, fast=True,
                                                                  samples=_studio_samples(desc))
                if problems:
                    raise ValueError(" ".join(problems))
                captions.unload()
                gen = studio.royale_render(desc, kw, settings, items, prompt, _seed(b.get("seed"), 42),
                                           int(b.get("width") or 512), int(b.get("height") or 512),
                                           ref=str(b.get("ref") or "") if desc.reference_kind else "",
                                           ref_strength=_ref_strength(b.get("ref_strength")),
                                           meta={"run": scan["run"], "family": desc.key, "folder": b.get("folder")})
                return self.json({"ok": True, "gen": gen, "family": desc.key, "total": len(items)})
            if path == "/api/royale/travel":        # seed / strength / prompt travel, or the comparison sheet
                from fizgig.web import royale
                mode = str(b.get("mode") or "")
                if mode not in ("seed", "strength", "prompt", "sheet"):
                    raise ValueError("Unknown travel.")
                fams = {d.key: d for d in royale.families()}
                rendered = studio.royale_rendered()
                single = str(b.get("source") or "folder") == "single"
                if single:
                    lpath = str(b.get("single") or "").strip()
                    label = os.path.splitext(os.path.basename(lpath))[0]
                elif rendered:
                    i = max(0, min(int(b.get("i") or 0), len(rendered) - 1))
                    label, lpath, epoch_png = rendered[i]
                else:
                    label, lpath = None, ""
                p = {k: b.get(k) for k in ("prompt", "seed", "width", "height", "frames")}
                if mode == "sheet":
                    prompts = [ln.strip() for ln in str(b.get("prompts") or "").splitlines()
                               if ln.strip() and not ln.strip().startswith("#")]
                    if not prompts:
                        raise ValueError("Enter at least one prompt - each line becomes a row of the sheet.")
                    sheet_mode = str(b.get("sheet_mode") or "Without / with LoRA")
                    if sheet_mode == "Every epoch":
                        cols = [[l, pth] for l, pth, _png in rendered if os.path.isfile(pth)]
                        if not cols:
                            raise ValueError('"Every epoch" reuses the epochs from the render at the top of this tab: '
                                             'pick the training folder and Render epochs first (or use "Without / '
                                             'with LoRA", which needs no epoch render).')
                        cols = royale.pick_columns(cols, b.get("epochs"))
                    else:
                        if not lpath or not os.path.isfile(lpath):
                            raise ValueError("Pick a LoRA file (Single LoRA), or render epochs and slide to the one "
                                             "you want to showcase.")
                        cols = [[None, lpath], [label, lpath]]       # column 0 renders at strength 0 (the base model)
                    total = len(prompts) * len(cols)
                    if total > 24 and not b.get("confirm"):       # the desktop's Large sheet question
                        return self.json({"confirm": f"That's {total} renders ({len(prompts)} prompts x {len(cols)} "
                                                     "columns). It will take a while."})
                    p.update(cols=cols, prompts=prompts, sheet_mode=sheet_mode, trigger=str(b.get("trigger") or ""),
                             row_labels=bool(b.get("row_labels", True)), brand=bool(b.get("brand", True)))
                    paths = [c[1] for c in cols]
                else:
                    if not lpath or not os.path.isfile(lpath):
                        raise ValueError("Pick a LoRA file (Single LoRA), or render epochs and slide to the one you "
                                         "want, then travel.")
                    paths = [lpath]
                    p["path"] = lpath
                    if mode != "prompt" and not str(b.get("prompt") or "").strip():
                        raise ValueError("Enter a prompt (include your trigger word).")
                    ref = ""
                    if mode in ("seed", "prompt"):           # the card's reference: the shown epoch, its file, Setup's
                        ref = (epoch_png if b.get("use_epoch") and not single and rendered else
                               str(b.get("ref") or "").strip() or str(b.get("setup_ref") or "").strip())
                    def num(k, d, lo, hi):           # as the desktop: a box without a number is its default
                        try:
                            return max(lo, min(hi, float(str(b.get(k)).strip())))
                        except (TypeError, ValueError):
                            return d
                    p.update(ref=ref, ref_strength=num("ref_strength", 1.0, 0, 2), ref_mp=num("ref_mp", 0.2, 0.05, 1),
                             sequential=bool(b.get("sequential")), anchor=bool(b.get("anchor")) and mode == "prompt",
                             anchor_str=num("anchor_str", 1.0, 0, 2),
                             label_disp=(f"Epoch {label}" if str(label).isdigit() else str(label)))
                    if mode == "seed":
                        p.update(seed_a=_seed(b.get("seed_a"), 42), seed_b=_seed(b.get("seed_b"), 4242),
                                 waypoints=max(2, int(b.get("waypoints") or 2)))
                        if p["seed_a"] == p["seed_b"] and p["waypoints"] <= 2:
                            raise ValueError("Start and end seed are the same - pick two different seeds, or raise "
                                             "Waypoints for a longer journey.")
                    elif mode == "strength":
                        p.update(s_start=float(b.get("s_start") or 0), s_end=float(b.get("s_end") or 1))
                        if p["s_start"] == p["s_end"]:
                            raise ValueError("Start and end strength are the same - pick two different values.")
                    else:
                        words = [str(w) for w in (b.get("words") or [])]
                        if len(words) < 2:
                            raise ValueError("Prompt travel needs at least two waypoints (pick a travel, or enter 2+ "
                                             "custom words).")
                        if not str(b.get("base") or "").strip():
                            raise ValueError("Enter a base prompt (include your trigger word, and {x} where the travel "
                                             "word goes).")
                        p.update(base=str(b["base"]).strip(), words=words, interp=str(b.get("interp") or "Linear"),
                                 drift=num("drift", 0.0, 0, 1), vary_seed=bool(b.get("vary_seed")))
                key = royale.family_of(paths, fams) or str(b.get("family") or "")
                desc = fams.get(key)
                if desc is None:
                    raise ValueError("Pick a model first.")
                if not desc.reference_strength:
                    p["sequential"] = False                  # no latent chain: each frame takes the reference itself
                if _gpu_training():
                    raise ValueError("A training run is using the GPU. Royale can render when it finishes.")
                from fizgig.web import tools
                if tools.holding_gpu():
                    raise ValueError("The Profiler is rendering on the GPU - wait for it, or Stop it.")
                prefs = load_prefs()
                if desc.video_workbench:
                    kw, _status, problems = studio.plan_video(desc, prefs, pref_path)
                    kw["audio_vae_path"] = ""
                    settings = None
                else:
                    kw, settings, _status, problems = studio.plan(desc, prefs, pref_path, fast=True,
                                                                  samples=_studio_samples(desc))
                if problems:
                    raise ValueError(" ".join(problems))
                captions.unload()
                gen = studio.travel_render(mode, desc, kw, settings, p)
                return self.json({"ok": True, "gen": gen, "family": desc.key})
            if path in ("/api/royale/travel/save", "/api/royale/travel/frame"):
                from fizgig.web import royale
                mode = str(b.get("mode") or "")
                folder = str(b.get("out_dir") or "").strip() or default_output_dir()
                run = (studio.royale_status().get("meta") or {}).get("run") or "lora"
                prefix = {"seed": "seedtravel", "strength": "strengthtravel", "prompt": "prompttravel"}.get(mode, mode)
                if mode == "sheet":
                    src = studio.travel_file("sheet")
                    if not src:
                        raise ValueError("Render the sheet first.")
                    return self.json({"path": royale.copy_png(src, folder, "fizgig_comparison.png")})
                if path.endswith("/frame"):
                    i = int(b.get("i") or 0)
                    src = studio.travel_file(mode, i)
                    if not src:
                        raise ValueError("Render the frames first.")
                    return self.json({"path": royale.copy_png(src, folder, f"{prefix}-frame{i + 1:03d}.png")})
                return self.json(royale.save_travel(studio.travel_frames(mode), folder, run, prefix,
                                                    b.get("format") or "MP4", b.get("speed") or "Normal",
                                                    bool(b.get("loop", True)), bool(b.get("tag", True)),
                                                    bool(b.get("badge", True)), str(b.get("deflicker") or "None")))
            if path == "/api/royale/cancel":
                return self.json({"ok": studio.royale_cancel()})
            if path in ("/api/royale/score", "/api/royale/export", "/api/royale/stills", "/api/royale/promote"):
                from fizgig.web import royale
                st = studio.royale_status()
                rendered = studio.royale_rendered()
                run = (st.get("meta") or {}).get("run") or "lora"
                if not rendered:
                    raise ValueError("Render some epochs first.")
                if path == "/api/royale/score":
                    return self.json(royale.score(st["gen"], rendered, str(b.get("ref") or "")))
                folder = str(b.get("out_dir") or "").strip() or default_output_dir()
                if path == "/api/royale/export":
                    return self.json(royale.export(rendered, folder, run, b.get("format") or "MP4",
                                                   b.get("speed") or "Normal", bool(b.get("loop", True)),
                                                   bool(b.get("ticker", True)), bool(b.get("tag", True)),
                                                   likeness_clip=bool(b.get("likeness")), gen=st["gen"]))
                if path == "/api/royale/stills":
                    return self.json({"saved": royale.save_stills(rendered, folder, run)})
                i = int(b.get("i") or 0)
                label, lpath, _png = rendered[max(0, min(i, len(rendered) - 1))]
                return self.json({"path": royale.promote(lpath, label, run, folder, str(b.get("name") or ""))})
            if path == "/api/profile/run":
                from fizgig.web import tools
                desc = next((d for d in tools.families("profiler") if d.key == b.get("family")), None)
                if desc is None:
                    raise ValueError("Pick a model first.")
                mode = str(b.get("mode") or "quick")
                prefs = load_prefs()
                job = {"family": desc.key, "lora": str(b.get("lora") or "").strip(), "mode": mode,
                       "prompt": str(b.get("prompt") or ""), "class_prompt": str(b.get("class_prompt") or ""),
                       "baselines": [str(x) for x in (b.get("baselines") or [])][:3], "size": int(b.get("size") or 768),
                       "profiles_dir": pref_path(prefs, "profiles_dir") or user_file("profiles"),
                       "plan": None, "settings": None}
                if job["baselines"] and len(job["baselines"]) != 3:
                    raise ValueError("Pick exactly 3 photos of the subject (one alone carries its own framing into every score).")
                if mode != "weights":
                    if _gpu_training():
                        raise ValueError("A training run is using the GPU. Profile when it finishes.")
                    if desc.video_workbench:
                        kw, _status, problems = studio.plan_video(desc, prefs, pref_path)
                        kw["audio_vae_path"] = ""            # the profile scores pictures (the desktop's audio=False)
                    else:
                        kw, job["settings"], _status, problems = studio.plan(desc, prefs, pref_path, fast=True,
                                                                             samples=_studio_samples(desc))
                    if problems:
                        raise ValueError(" ".join(problems))
                    job["plan"] = kw
                    studio.stop_worker()                     # Studio's model off the card first (as a training run)
                    captions.unload()
                tools.run_profile(job)
                return self.json({"ok": True})
            if path == "/api/profile/stop":
                from fizgig.web import tools
                return self.json({"ok": tools.stop("profile")})
            if path == "/api/extract/finetune":     # From a fine-tune: the LoRA a full fine-tune amounts to
                from fizgig.web import tools
                if _gpu_training():
                    raise ValueError("A training run is using the GPU. Extract can run when it finishes.")
                tools.run_finetune(str(b.get("base") or "").strip(), str(b.get("tuned") or "").strip(),
                                   str(b.get("out_dir") or "").strip() or default_output_dir(),
                                   str(b.get("name") or ""), b.get("ranks") or [])
                return self.json({"ok": True})
            if path == "/api/extract/run":
                from fizgig.web import tools
                desc = next((d for d in tools.families() if d.key == b.get("family")), None)
                if desc is None:
                    raise ValueError("Pick a model first.")
                name = str(b.get("name") or "").strip()
                if not name:
                    raise ValueError("Give the extraction a name.")
                if not name.lower().endswith(".safetensors"):
                    name += ".safetensors"
                folder = str(b.get("out_dir") or "").strip() or default_output_dir()
                out = tools.run(desc, str(b.get("source") or "").strip(), os.path.join(folder, os.path.basename(name)),
                                int(b.get("rank") or 4), str(b.get("preset") or ""), [str(x) for x in (b.get("blocks") or [])])
                return self.json({"ok": True, "output": out})
            if path == "/api/metadata/save":         # the header rewritten, every tensor byte as it was
                from fizgig.utils.st_meta import write_metadata
                src = str(b.get("path") or "")
                dest = str(b.get("dest") or "").strip() or src
                if not src.lower().endswith(".safetensors") or not os.path.isfile(src):
                    raise ValueError("Load a .safetensors file first.")
                if not dest.lower().endswith(".safetensors"):
                    dest += ".safetensors"
                if os.path.abspath(dest) != os.path.abspath(src) and os.path.exists(dest) and not b.get("overwrite"):
                    return self.json({"exists": True})
                meta = {str(k): str(v) for k, v in (b.get("meta") or {}).items() if str(v).strip()}
                for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
                    meta.pop(stale, None)                  # the hashes were of the old bytes (as the desktop's save)
                write_metadata(src, dest, meta)
                return self.json({"ok": True, "dest": dest})
            if path == "/api/metadata/thumbnail":    # a picture as the thumbnail ComfyUI shows (a data URI)
                from fizgig.training.metadata import thumbnail_data_uri
                uri = thumbnail_data_uri(str(b.get("path") or ""))
                if not uri:
                    raise ValueError("Could not read that picture.")
                return self.json({"uri": uri})
            if path == "/api/captions/remove":       # out of the training set, into removed/ (never deleted)
                return self.json({"moved": remove_from_dataset(str(b.get("folder") or ""), str(b.get("name") or ""),
                                                               captions.caption_ext(b.get("ext")))})
            if path == "/api/captions/restore":
                return self.json({"restored": restore_to_dataset(str(b.get("folder") or ""), b.get("moved") or [])})
            if path == "/api/captions/save":
                folder, name, text = b.get("folder", ""), b.get("name", ""), b.get("text", "")
                stem = os.path.splitext(os.path.basename(name))[0]
                ext = captions.caption_ext(b.get("ext"))     # the Caption Extension, as every caption route reads it
                if not stem or not os.path.isdir(folder):
                    return self.json({"error": "bad folder or file"}, 400)
                with open(os.path.join(folder, stem + ext), "w", encoding="utf-8") as f:
                    f.write(text)
                return self.json({"ok": True})
        except ValueError as e:
            return self.json({"error": str(e)}, 400)
        return self.json({"error": "not found"}, 404)


class _Server(ThreadingHTTPServer):
    # Windows' SO_REUSEADDR lets a second program bind a port another is already serving on, so a taken port would
    # look free there; only POSIX needs it (a quick restart rebinding a port still in TIME_WAIT).
    allow_reuse_address = os.name != "nt"
    daemon_threads = True
    # every request is its own connection (HTTP/1.0), and a page of thumbnails asks for dozens at once: the default
    # backlog of 5 made Windows refuse the overflow, and those pictures showed broken
    request_queue_size = 128


def _bind(host, start, walk):
    """The server on `start`, or the nearest free port above it when `walk`. Returns (server, port)."""
    last = None
    for port in range(start, min(65536, start + (PORT_WALK if walk else 1))):
        if listening(port):
            continue
        try:
            return _Server((host, port), Handler), port
        except OSError as e:
            last = e
    if walk:
        sys.exit(f"No free port from {start} to {start + PORT_WALK - 1}; set one with --port.")
    sys.exit(f"Port {start} is in use{f' ({last})' if last else ''}; pick another with --port.")


def _follow_parent(pid):
    """Started for the Fizgig app window: stop when that window's process is gone (closed, crashed or killed), so no
    web UI is left running with nobody's window on it. Studio's worker goes too."""
    try:
        import psutil
    except ImportError:
        return
    while True:
        time.sleep(5)
        try:
            alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except Exception:
            alive = False
        if not alive:
            try:
                _stop_and_wait()                  # a run doesn't train on, orphaned, with no server to stop it
            except Exception:
                pass
            try:
                studio.stop_worker()
            except Exception:
                pass
            os._exit(0)


def _lan_announce(port):
    """What this Fizgig tells another computer's Pods tab: its name, GPU, port, build and whether it's training."""
    from fizgig.web import lan
    mid = lan.machine_id(load_prefs(), update_prefs)
    about = about_info()

    def info():
        with _lock:
            vt = _stats["vram"][1]
            gpu = _stats["device_name"]
            busy = _run["lora"] if _run["proc"] is not None and _run["proc"].poll() is None else ""
        q = webqueue.status()
        return {"id": mid, "name": socket.gethostname(), "port": port, "gpu": gpu if vt else "",
                "vram_gb": round(vt / 2 ** 30) if vt else 0, "version": about["version"], "commit": about["commit"],
                "busy": busy, "queued": len(q["items"]), "queue_held": bool(q["held"])}
    lan.start(info)


def run_server(host, port=None, password=None, password_arg=False, open_browser=False):
    """`port` given = that port only. Otherwise the Preferences port (8081 until one is saved); when it is taken, the
    nearest free one above it, saved to Preferences so every later launch gets that same port. A pod always uses
    8081: it is the port its proxy forwards."""
    parent = os.environ.get("FIZGIG_APP_PARENT", "")
    if parent.isdigit():
        threading.Thread(target=_follow_parent, args=(int(parent),), daemon=True).start()
    Handler.password = password or None
    Handler.local_only = not password
    Handler.password_arg = password_arg
    if os.environ.pop("FIZGIG_RESTARTED", None):     # Restart now: the old server may still be letting go
        want = port or (PORT_DEFAULT if on_pod() else saved_port(load_prefs()))
        for _ in range(60):
            if not listening(want):
                break
            time.sleep(0.25)
    Handler.port_fixed = port is not None and not os.environ.get("FIZGIG_APP_KEY")   # the app window passes its saved port
    if port is not None or on_pod():
        srv, port = _bind(host, port or PORT_DEFAULT, walk=False)
    else:
        want = saved_port(load_prefs())
        srv, port = _bind(host, want, walk=True)
        if port != want:
            logger.info(f"Port {want} is in use; the web UI takes {port} and keeps it (Preferences > Web UI)")
        if str(load_prefs().get("web_port") or "") != str(port):
            update_prefs({"web_port": str(port)})
    if history.close_orphans():                        # a run left "running" by a stopped or crashed server
        logger.info("A run from before was still recorded as running: it is recorded as stopped now")
    webqueue.load()                                    # held if it holds anything: a restart never launches a run
    _restore_pause()                                   # a run paused before the restart can still be resumed
    __import__("fizgig.web.update_check", fromlist=["x"]).start()   # a newer release? (the About button says)
    threading.Thread(target=_poll_stats_loop, daemon=True).start()
    if password and not on_pod() and host not in ("127.0.0.1", "localhost", "::1"):
        _lan_announce(port)                    # other computers' Pods tabs find this one by themselves
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    logger.info(f"Fizgig web UI on http://{shown}:{port}/" + (" (password required)" if password else ""))
    if open_browser:                           # Fizgig in browser: the page, once this is listening
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(f"http://127.0.0.1:{port}/",)).start()
    def _term(*_a):                            # a service manager's stop: end the run with it, as Ctrl+C does
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _term)
    except (ValueError, OSError):
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
        captions.unload()
        srv.server_close()
        if _RESTART[0]:
            _respawn()


_RESTART = [False]


def _respawn():
    """Restart now: the same command again, once this one has let go of its port (the app window's key and parent
    travel in the environment, so the window stays logged in)."""
    args = [sys.executable, os.path.abspath(sys.argv[0])] + [a for a in sys.argv[1:] if a != "--open"]
    kw = {}
    if os.name == "nt":
        import ctypes
        if not ctypes.windll.kernel32.GetConsoleWindow():      # no console (the app window's): none for it either
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    subprocess.Popen(args, cwd=os.getcwd(), env=dict(os.environ, FIZGIG_RESTARTED="1"), **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fizgig's web UI")
    ap.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default=None, help="Address to listen on (default: localhost; every address with a "
                                                 "password)")
    ap.add_argument("--port", type=int, default=None, help="Port to serve on (default: the one in Preferences, "
                                                            "8081 at first; a taken one moves to the next free one)")
    ap.add_argument("--open", action="store_true", help="Open the web UI in the browser once it is up (and if "
                                                         "Fizgig is running already, just open it)")
    ap.add_argument("--password", default=None, help="Login password, 8 characters or more (default: "
                                                     "FIZGIG_PASSWORD, then on a pod VNC_PASSWORD, then the one in "
                                                     "Preferences)")
    a = ap.parse_args(argv)
    user_data.migrate()                       # Fizgig 7's loose files into user_data, before anything reads them
    # a pod: its VNC_PASSWORD, the remote desktop's, so one password opens both. Elsewhere VNC_PASSWORD is ignored (a
    # stray one would open the web UI to the network while Preferences says no password).
    saved = str(load_prefs().get("web_password") or "").strip()
    pwd = (a.password or os.environ.get("FIZGIG_PASSWORD")
           or ((os.environ.get("VNC_PASSWORD") or saved) if on_pod() else saved))
    pod_pwd = on_pod() and not a.password and not os.environ.get("FIZGIG_PASSWORD") and pwd == os.environ.get(
        "VNC_PASSWORD")
    if pwd and len(pwd) < 8 and not pod_pwd:          # a pod's own VNC password is taken at any length
        sys.exit("The web UI password needs at least 8 characters (Preferences > Web UI, web_password in "
                 "prefs.json, FIZGIG_PASSWORD or --password).")
    host = a.host or ("0.0.0.0" if pwd else "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not pwd:
        sys.exit("Listening beyond localhost needs a password: set one in Preferences > Web UI (web_password in "
                 "prefs.json), or use FIZGIG_PASSWORD or --password.")
    if a.open and not on_pod():
        from fizgig.web.port import is_fizgig
        running = a.port or saved_port(load_prefs())
        if is_fizgig(running):                 # Fizgig is up already (the app window's, say): the browser on it
            import webbrowser
            print(f"Fizgig is already running on port {running}: opening it.")
            webbrowser.open(f"http://127.0.0.1:{running}/")
            return
    run_server(host, a.port, password=pwd, password_arg=bool(a.password), open_browser=a.open and not on_pod())


if __name__ == "__main__":
    main()
