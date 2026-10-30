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

from fizgig.dataset.masks import is_mask_file, mask_for_stem
from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families
from fizgig.web import captions, history
from fizgig.web import queue as webqueue
from fizgig.web import prep_jobs
from fizgig.families import refmod
from fizgig.web import studio
from fizgig.web.port import PORT_DEFAULT, PORT_WALK, listening, on_pod, on_runpod, on_vast, saved_port
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
LOGS_DIR = user_file("logs")            # each run's log as a file too (the console's only lasts until a restart)
LOGS_KEPT = 20
_run_log = {"file": None, "path": "", "bar_at": 0.0}
_BAR = re.compile(r"\d+%\|")             # a progress bar's redraw: one a minute in the file, and its 100%
_stats = {"vram": [0, 0], "ram": [0, 0], "vram_peak": 0, "ram_peak": 0, "device_name": "GPU"}


# no console window for helper commands: started from the desktop app the web UI has none, so each
# nvidia-smi / taskkill would flash one up on Windows
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# ---- hardware meters (as the desktop status bar) ---------------------------------------------------------------
def _rocm_launch() -> bool:
    """True only on the AMD launcher path. NVIDIA Launch Fizgig.bat must never enter the AMD VRAM reader."""
    return os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm"


def _amd_gpu_row():
    """One synthetic GPU row from the ROCm VRAM reader (#188). Never imports torch / HIP."""
    if not _rocm_launch():
        return None
    try:
        from fizgig.utils.vram_monitor import amd_gpu_name, read_amd_gpu_vram
        hit = read_amd_gpu_vram()
        if not hit:
            return None
        used_b, total_b = hit
        if total_b <= 0:
            return None
        return {"index": 0, "name": amd_gpu_name() or "AMD GPU",
                "total_gb": total_b / 2 ** 30, "used_gb": max(0, used_b) / 2 ** 30,
                "uuid": "", "util": None, "temp": None, "power": None, "power_limit": None}
    except Exception:
        return None


def list_gpus():
    """[{index, name, total_gb, used_gb, uuid}] from nvidia-smi (unchanged). On ROCm launch only, AMD fallback (#188)."""
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
        if rows:
            return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    row = _amd_gpu_row()
    return [row] if row else []


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
                extra = {k: r.get(k) for k in ("util", "temp", "power", "power_limit") if r.get(k) is not None}
        # ROCm launch only - never typeperf/amd-smi on the NVIDIA path even if nvidia-smi is missing (#188).
        if not vt and _rocm_launch():
            try:
                from fizgig.utils.vram_monitor import amd_gpu_name, read_amd_gpu_vram
                hit = read_amd_gpu_vram()
                if hit:
                    vu, vt = int(hit[0]), int(hit[1])
                    name = amd_gpu_name() or name
            except Exception:
                pass
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
        f = _run_log["file"]
        if f is None or text == LOG_CLEAR:
            return
        if _BAR.search(text) and "100%|" not in text:
            if time.time() - _run_log["bar_at"] < 60:
                return
            _run_log["bar_at"] = time.time()
        try:
            f.write(text)
            f.flush()
        except (OSError, ValueError):
            _run_log["file"] = None


def _open_run_log(lora):
    """A new run's log file, user_data/logs/<date>_<time>_<lora>.log; only the newest LOGS_KEPT are kept."""
    with _lock:
        if _run_log["file"] is not None:
            try:
                _run_log["file"].close()
            except OSError:
                pass
        _run_log.update(file=None, path="", bar_at=0.0)
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", lora or "run")[:80]
            path = os.path.join(LOGS_DIR, f"{datetime.datetime.now():%Y-%m-%d_%H%M%S}_{name}.log")
            _run_log.update(file=open(path, "w", encoding="utf-8"), path=path)
        except OSError:
            return
    for p in sorted((p for p in os.listdir(LOGS_DIR) if p.endswith(".log")), reverse=True)[LOGS_KEPT:]:
        try:
            os.remove(os.path.join(LOGS_DIR, p))
        except OSError:
            pass


def _log_where():
    """The line naming this run's log file, for a failure's message."""
    return f"Full log: {_run_log['path']}\n" if _run_log["path"] else ""


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
                 "inference_int8": "1", "lan_discovery": "1", "update_check": "1"}


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
                 "inference_blocks_to_swap", "inference_int8", "lan_discovery", "update_check")


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


def _sam_info():
    """The masks card's SAM 3.1 line: the downloaded file ("" before the download) and how a download is going."""
    from fizgig.dataset import sammask
    return {"path": sammask.find_model(user_data.MODELS_DIR), "fetch": sammask.fetch_state(user_data.MODELS_DIR),
            "gb": sammask.SIZE_GB}


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
                    (".png", ".jpg", ".jpeg", ".webp", ".bmp") and not is_mask_file(f))
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


PROMPT_SETS_DIR = os.path.join(PRESETS_DIR, "sample_prompts")


def prompt_sets() -> list:
    """The Samples tab's saved prompt sets, shared by every model (presets/sample_prompts/<name>.txt, one prompt a
    line), by name (#267)."""
    try:
        return sorted((f[:-4] for f in os.listdir(PROMPT_SETS_DIR) if f.endswith(".txt")), key=str.lower)
    except OSError:
        return []


def prompt_set_path(name) -> str:
    name = str(name or "").strip()
    if not name or any(c in name for c in '<>:"/\\|?*') or name.startswith("."):
        raise ValueError('A set\'s name can\'t be empty or hold < > : " / \\ | ? *')
    return os.path.join(PROMPT_SETS_DIR, name + ".txt")


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
    # the Samples tab's preview-checkpoint tick (MOSS-Image's Use the SFT model for samples): the workbench renders
    # where training previews do - on the checkpoint while it's ticked, on the training model when it isn't
    ck = v.get("SAMPLE_CHECKPOINT")
    if blank(ck):
        ck = desc.preview_checkpoint_on
    ck = ck if isinstance(ck, bool) else str(ck).strip().lower() in ("1", "true", "yes", "on")
    # the preview look LoRA the Samples tab puts on Base previews (its tick and file), and the workbench's render
    # choice while that tab is on the training model (description.workbench_render_choice)
    lk = desc.preview_look()
    look = ""
    if lk is not None and v.get("FAMILY_PREVIEW_LOOK"):
        look = launch.look_path(desc, {"models": {lk.pref_key: pref_path(load_prefs(), lk.pref_key)}})
    render = str(load_prefs().get(f"workbench_render_{desc.key}") or "checkpoint")
    return {"steps": str(steps), "cfg": str(cfg), "turbo": str(turbo), "negative": neg,
            "turbo_live": live, "turbo_name": sp.name if sp is not None else "", "checkpoint": ck,
            "look": look, "render": "samples" if render == "samples" else "checkpoint"}


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
        key = pod_stop.api_key(p)
        if key and rows[0].get("id"):            # a pod this account's key doesn't list: another account's, kept
            try:
                rows[0]["ours"] = any(a.get("id") == rows[0]["id"] for a in pods.account(key))
            except ValueError:
                pass
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


def _plan(family_key, values, image_folder, resume="", ft_continue=None, models=None):
    """models: {pref key: path} in place of Preferences' (a run moved here trains on the files it was checked on)."""
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
    captioner = captioner_for(prefs)
    raw_ovr = prefs.get("caption_qwen_instructions")
    inputs = form.inputs(
        desc, values, models={**{k: pref_path(prefs, k) for k in form.model_keys(desc)}, **(models or {})},
        python=sys.executable,
        repo_dir=BASE_DIR, image_folder=str(image_folder or "").strip(),
        cache_root=pref_path(prefs, "cache_dir"), samples_dir=os.path.join(out_dir, "sample"),
        inference_int8=str(prefs.get("inference_int8", "")).strip() in ("1", "True", "true"),
        captioner=captioner if captioner and os.path.isfile(captioner) else "",
        caption_overrides=dict(raw_ovr) if isinstance(raw_ovr, dict) else {},
        caption_concept=str((load_state().get("captions") or {}).get("concept") or ""))
    # auto-recaption's trigger word per folder: a sub-folder's or another subject's own (the Captions tab's for it, else
    # the word its captions open with), never the training folder's
    trig = {f"{rel}/": subject_trigger(f, ext) for rel, f in launch.run_subfolders(desc, inputs)}
    if inputs.get("multiconcept"):
        trig.update({f"s{i}:": str(values.get(f"DOP_FOLDER_TRIGGER{i}") or "")
                     for i, f in enumerate(inputs.get("extra_folders") or [], start=2) if f})
    inputs["recaption_triggers"] = trig
    # a helper set to train a share: both halves keep gradient checkpointing, so they train with the same options
    inputs["keep_checkpointing"] = any(h.get("share") for h in (prefs.get("helpers") or []))
    if ft_continue:          # a paused fine-tune: from its checkpoint, for the rotations left (the cache is built)
        inputs.update(FAMILY_FT_CONTINUE=ft_continue["continue"], FAMILY_FT_ROTATIONS=str(ft_continue["rotations"]),
                      resuming=True)
    if resume or ft_continue or str(values.get("RESUME_TRAINING") or "").strip():
        inputs["recache"] = _size_changed(name, out_dir, values)
    return desc, inputs, launch.plan(desc, inputs)


_SIZE_KEYS = ("DATASET_MEGAPIXELS", "CLIP_MEGAPIXELS")
# the settings a Resume takes from the form as it is now: the Samples tab's, block swap (memory only, the results
# are the same), the save schedule (when checkpoints are written) and the learning rate (the run trains on at the
# new one, logged); the rest trains on as it was
_RESUME_SAMPLE_KEYS = ("SAMPLE_PROMPT", "SAMPLE_SEED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
                       "SAMPLE_CFG_SCALE", "FAMILY_TURBO_STRENGTH", "SAMPLE_CHECKPOINT",   # + previews on the checkpoint
                       "FAMILY_PREVIEW_LOOK",                                              # + the SFT look tick
                       "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_LATER", "SAMPLE_UNTIL_EPOCH", "SAMPLE_EVERY_AFTER",
                       "BLOCKS_SWAP", "SAVE_EVERY_N_EPOCHS", "SAVE_LATER", "SAVE_UNTIL_EPOCH", "SAVE_EVERY_AFTER",
                       "LEARNING_RATE", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX")


# a prompt slider's live prompts: the request the web UI writes, and what the trainer took (fizgig.families.train's
# SLIDER_LIVE / SLIDER_LIVE_APPLIED - named here so the server never imports torch)
SLIDER_LIVE, SLIDER_LIVE_APPLIED = ".slider_prompts.json", ".slider_prompts_applied.json"
_SLIDER_LIVE_KEYS = {"base": "FAMILY_SLIDER_BASE", "pos": "FAMILY_SLIDER_POS", "neg": "FAMILY_SLIDER_NEG",
                     "guidance": "FAMILY_SLIDER_GUIDANCE", "anchors": "FAMILY_SLIDER_ANCHORS"}


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
    lines = lambda x: [s.strip() for s in (x.splitlines() if isinstance(x, str) else (x or [])) if str(s).strip()]
    same = lambda a, b: all(str(a.get(k) or "") == str(b.get(k) or "") for k in ("base", "pos", "neg")) and \
        lines(a.get("anchors")) == lines(b.get("anchors")) and         float(a.get("guidance") or 0) == float(b.get("guidance") or 0)
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
    found += __import__("fizgig.web.helpers", fromlist=["x"]).home_folders()     # pods' runs brought home
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
    from fizgig.web import helpers
    moved = helpers.move_restore(out_dir, name)     # it went on to train on another computer: no Resume here
    with _lock:
        _run.update(state="moved" if moved else "paused", family=lr["family"], values=values,
                    image_folder=lr.get("image_folder", ""), lora=name, output_dir=out_dir, resume=path,
                    resume_mode=mode)
    if moved:
        return
    webqueue.hold("The last run is paused. Resume it: the queue carries on after it finishes.", for_pause=True)


def _continue_point(rec):
    """Where Continue on a run's card carries on from: {kind "state" | "ft", path, done, of, key, unit} - a LoRA's
    newest saved state (its epoch, of Max Train Epochs) or a fine-tune's newest checkpoint (its rotations, of Train
    for) - or None when there's nothing to continue from (no save left, a run training or moved away)."""
    if not rec or rec.get("source") != "web" or rec.get("result") in ("running", "moved"):
        return None
    desc = get_family(rec.get("family") or "")
    values, name, out = rec.get("values") or {}, rec.get("lora") or "", rec.get("output_dir") or ""
    if desc is None or not name or not out:
        return None
    if desc.finetune and form._kind(desc, values) == "finetune":
        path = launch.latest_ft_checkpoint(out, name)
        if not path:
            return None
        left, done, _c = launch.ft_rotations_left(path, values.get("FAMILY_FT_ROTATIONS"),
                                                  values.get("FAMILY_FT_ROTATE_EVERY"))
        return {"kind": "ft", "path": path, "done": done, "of": done + left, "key": "FAMILY_FT_ROTATIONS",
                "unit": "rotation"}
    path = launch.latest_state(out, name)
    m = re.search(r"-(\d{6})-state$", path)
    if not m:
        return None
    try:
        of = int(str(values.get("MAX_TRAIN_EPOCHS") or "0").strip())
    except ValueError:
        of = 0
    return {"kind": "state", "path": path, "done": int(m.group(1)), "of": of, "key": "MAX_TRAIN_EPOCHS",
            "unit": "epoch"}


def _oom_note(run_id):
    """What to do after a run ran out of GPU memory: Continue from its newest save (fresh memory - memory that
    fragments over hours is the usual cause late in a run), or, with nothing saved yet, a lighter setup."""
    pt = _continue_point(history.get(run_id))
    if pt:
        return (f"\nThe GPU ran out of memory. The run is saved at {pt['unit']} {pt['done']}: press Continue on its "
                f"card in Runs & outputs to carry on from there, with fresh memory. Closing other programs that use "
                f"the GPU (a browser, a game) helps too.\n")
    return ("\nThe GPU ran out of memory before the run saved anything. Try a smaller picture size or more block "
            "swap (Memory & Precision, on the Training tab), and close other programs that use the GPU.\n")


def _run_form(rec, values=None):
    """A run's model, training folder and settings as the Training tab loads them (Start again; a Continue that
    started, with the settings it carries on with, so the tab shows the run training and not the one before)."""
    if rec and rec.get("family") in (refmod.KEY, refmod.LABEL):
        return {"family": refmod.KEY, "values": {k: v for k, v in (values or rec.get("values") or {}).items()
                                                 if not str(k).startswith("__")},
                "image_folder": rec.get("image_folder") or ""}
    desc = get_family(rec.get("family", "")) if rec else None
    if desc is None:
        return {"error": "That run's model isn't available here."}
    vals = values or rec.get("values") or {}
    if rec.get("source") == "desktop":       # the desktop's settings, as a preset applies them, with the
        mk = {o.key for o in desc.options if o.tab == "model"}   # Base Model card's options beside them
        vals = {**form.apply_preset(desc, {}, vals, whole=True), **{k: v for k, v in vals.items() if k in mk}}
    return {"family": desc.key, "values": vals, "image_folder": rec.get("image_folder") or ""}


def _continue_run(run_id, more="", queue_ok=True, values=None):
    """Continue on Runs & outputs: the run trains on from its newest save, with its own settings and into its own
    record. more: the epochs (a fine-tune: rotations) to train to, for a run that already trained all it was set to.
    values: the settings to carry on with in place of the run's (a queued Continue edited on the queue). A run
    training here now: it goes on the queue instead, and continues when its turn comes."""
    rec = history.get(run_id)
    pt = _continue_point(rec)
    if pt is None:
        return {"error": "Nothing to continue from: this run's saved state isn't there any more."}
    values = {k: v for k, v in (values or rec.get("values") or {}).items() if not str(k).startswith("__")}
    if str(more or "").strip():
        try:
            values[pt["key"]] = str(int(str(more).strip()))
        except ValueError:
            return {"error": f"Train to: a whole number of {pt['unit']}s, please."}
    unit = pt["unit"] + "s"
    if pt["kind"] == "ft":
        left, done, cont = launch.ft_rotations_left(pt["path"], values.get("FAMILY_FT_ROTATIONS"),
                                                    values.get("FAMILY_FT_ROTATE_EVERY"))
        target = done + left
    else:
        done, target = pt["done"], int(str(values.get("MAX_TRAIN_EPOCHS") or "0").strip() or 0)
    if done >= target:
        return {"need": pt["key"], "done": done, "unit": pt["unit"],
                "error": f"This run has trained all {target} {unit}. Choose how many {unit} to train to, then "
                         f"Continue."}
    with _lock:
        busy = _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running")
    if busy:
        if not queue_ok:
            return {"error": "A run is already going. Stop or pause it first."}
        webqueue.add(rec["family"], {**values, "__continue": run_id, "__more": str(more or "")},
                     rec.get("image_folder") or "", rec.get("lora") or "", rec.get("output_dir") or "")
        return {"ok": True, "queued": True}
    note = f"=== Continuing {rec.get('lora')} from {os.path.basename(pt['path'])}: to {target} {unit} ==="
    if pt["kind"] == "ft":
        r = _resumed(_start(rec["family"], values, rec.get("image_folder") or "", ft_continue={
            "continue": cont, "rotations": left}, note=note, carry=run_id))
    else:
        r = _resumed(_start(rec["family"], values, rec.get("image_folder") or "", resume=pt["path"], note=note,
                            carry=run_id))
    if isinstance(r, dict) and not r.get("error") and not r.get("problems"):
        f = _run_form(rec, values)            # the Training tab then shows this run's settings (it kept the last ones)
        if not f.get("error"):
            r = {**r, "form": f}
    return r


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


def _resumed(result):
    """A resume that started: the queue held for that paused run stops showing as held."""
    if isinstance(result, dict) and not result.get("error") and not result.get("problems"):
        webqueue.resumed()
    return result


_claim = threading.Lock()                   # one Start at a time: checking and planning a run takes a few seconds


def _start(family_key, values, image_folder, resume="", ft_continue=None, note="", models=None, carry=""):
    # a second Start while the first is still being checked (a double click, two tabs, the queue) waits here and then
    # finds the first one going - before, each passed the check and three copies of one run trained at once
    with _claim:
        return _start_one(family_key, values, image_folder, resume, ft_continue, note, models, carry)


def _start_one(family_key, values, image_folder, resume, ft_continue, note, models, carry):
    with _lock:
        if _run["proc"] is not None and _run["proc"].poll() is None or _run["state"] in ("starting", "running"):
            return {"error": "A run is already going. Stop or pause it first."}
    if studio.holding_gpu():                    # Repair Studio's model leaves the GPU for the run
        append_log("Unloading Repair Studio's model to free the GPU for this run.\n")
        studio.stop_worker()
    if captions.status()["state"] in ("loading", "running", "translating"):
        return {"error": "Captions are being made (Captions tab). Stop that first, then start the run."}
    from fizgig.web import tools
    if tools.holding_gpu():                     # a profile part-way: refused like captions, not thrown away
        return {"error": "The Profiler is rendering on the GPU. Wait for it, or Stop it, then start the run."}
    captions.unload()                                     # the run gets the GPU memory the caption model held
    from fizgig.web import helpers
    helpers.share_stop_all()       # this computer's own run comes first: a share it trains for another stops there
    desc, inputs, plan = _plan(family_key, values, image_folder, resume, ft_continue, models)
    if plan.problems:
        return {"problems": plan.problems}
    lk = desc.preview_look() if desc is not None else None
    if lk is not None and lk.fetch_on_use and (values or {}).get("FAMILY_PREVIEW_LOOK"):
        # a run started with the SFT look ticked (from the queue, a script or a helper's lead) fetches it as the
        # Samples tab's tick does: the run picks it up at a preview once it's there
        _look_fetch(desc, values, launch.look_path(desc, {"models": {lk.pref_key: pref_path(load_prefs(), lk.pref_key)}}))
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
    helpers.share_cache_plan(                # a training share can make the cache on the helper the same way
        out_dir, inputs["LORA_NAME"], [s.cmd for s in plan.stages])
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
                        resume=bool(resume or ft_continue), carry=carry)
    with _lock:
        _run["history_id"] = hid
    append_log(LOG_CLEAR)                    # this run's log on a clean console
    _open_run_log(inputs["LORA_NAME"])
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
        cid = (item.get("values") or {}).get("__continue")
        r = (_continue_run(cid, item["values"].get("__more") or "", queue_ok=False, values=item["values"]) if cid else
             _start(item["family"], item["values"], item.get("image_folder") or ""))
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
    if on_vast() and not on_runpod():  # Vast's container disk: kept through a stop, gone when the instance is destroyed
        return [f"Storage: {free:.0f} GB free of {total:.0f} GB on /workspace", free < 40]
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
    p = prefs if prefs is not None else load_prefs()
    want = str(p.get("cuda_device") or "").strip()
    if not want and p.get("helper_local_gpu") not in (None, ""):
        want = "0"                     # a second Fizgig has this computer's other GPU: this one keeps the first
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
            if str((_run.get("progress") or {}).get("note") or "").startswith("Caching"):
                _run["progress"] = None          # the last stage's cache count isn't this stage's
        append_log(f"\n▶ {stage.name}\n")
        try:
            proc = _popen(stage.cmd)
        except OSError as e:
            append_log(f"✖ {stage.name} could not start: {e}\n" + _log_where())
            with _lock:
                _run.update(state="failed", proc=None, ended_at=time.time())
            history.end(_run.get("history_id"), "failed", _run.get("progress"))
            webqueue.on_run_ended("failed", _run.get("lora"), _queue_starter)
            return
        with _lock:
            _run["proc"] = proc
        oom = False
        for line in proc.stdout:
            append_log(line)
            oom = oom or "out of memory" in line.lower()
            try:
                upd = tracker.consume(line)
            except Exception:
                upd = None
            if upd:
                with _lock:
                    if upd.get("kind") == "cache":      # a cache stage's count: its own bar and line
                        n, total = upd["n"], upd["total"]
                        _run["progress"] = {**_progress_dict({}), "percent": 100.0 * n / total if total else 0.0,
                                            "note": f"Caching {upd['what']}: {n} of {total}"}
                    elif upd.get("kind") == "preview":
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
            with _lock:
                _run["proc"] = None              # gone: a stop-and-wait sees it at once
            append_log(f"\n■ Stopped during {stage.name}.\n")
            history.end(_run.get("history_id"), "stopped", _run.get("progress"))
            webqueue.on_run_ended("stopped", _run.get("lora"), _queue_starter)
            return
        if ret != 0:
            append_log(f"\n✖ {stage.name} failed (exit code {ret}).\n" + _log_where())
            with _lock:
                _run.update(state="failed", proc=None, ended_at=time.time())
            history.end(_run.get("history_id"), "failed", _run.get("progress"))
            if oom:
                append_log(_oom_note(_run.get("history_id")))
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
    from fizgig.web import helpers as _h
    if not paused and not webqueue.status()["items"] and not _h.moved_in_waiting(out_dir):
        from fizgig.web import pod_stop                    # the last run done: a pod can stop billing (a run moved
        pod_stop.maybe_start(load_prefs(), append_log)     # here waits until its computer has taken it back)


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
UPLOAD_FILE_EXTS = (UPLOAD_EXTS - {".zip"}) | {".safetensors", ".onnx"}      # .onnx: a YOLO face model (Image Prep)


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


def preview_labels(mode) -> dict:
    """{epoch: the line over that epoch's enlarged previews} from the run's record (rounds) and its family's words
    (description preview_labels); {} for a family that declares none or a run from before the record."""
    if not isinstance(mode, dict):
        return {}
    desc = get_family(mode.get("family", ""))
    words = dict(getattr(desc, "preview_labels", ()) or ())
    out = {}
    for ep, r in (mode.get("rounds") or {}).items():
        line = words.get("checkpoint" if r.get("checkpoint") else "look" if r.get("look") else "base", "")
        if r.get("turbo") and words.get("turbo"):
            line = (words["turbo"] + " " + line).strip()
        if r.get("helper"):                      # a helper rendered the round (web/helpers.py)
            line = (line.rstrip(".") + f" · rendered on {r['helper']}" if line else f"Rendered on {r['helper']}")
        if line:
            out[str(ep)] = line
    return out


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


def epoch_prompts(sample_dir, name) -> dict:
    """The prompts each epoch's previews used, as the trainer recorded them ({"4": [prompt, ...]}); {} for runs from
    before it recorded them, which show the run's settings instead."""
    try:
        with open(os.path.join(sample_dir, f".prompts_{name}.json"), encoding="utf-8") as f:
            data = json.load(f)
        return ({str(k): list(v) for k, v in data.items() if str(k).isdigit() and isinstance(v, list)}
                if isinstance(data, dict) else {})
    except (OSError, ValueError):
        return {}


def _caption_media_filter(names, family):
    """Which of a folder's files the family trains on: an image family's captions list leaves out the folder's
    recordings and clips. A song's waveform picture and a clip's still and sound stay, for their cards."""
    try:
        desc = _files_family(family) if family else None
    except Exception:
        desc = None
    if desc is None:
        return lambda n: True
    media = set(desc.media)
    stems = lambda exts: {os.path.splitext(n)[0] for n in names if os.path.splitext(n)[1].lower() in exts}
    clip_stems = stems({".mp4"}) if "clip" in media else set()
    song_stems = stems(_AUDIO) if "voice" in media else set()

    def keep(n):
        stem, e = os.path.splitext(n)
        e = e.lower()
        if stem in clip_stems:
            return True
        if e in _AUDIO:
            return "voice" in media
        if e in (".mp4", ".webm"):
            return "clip" in media
        return "photo" in media or stem in song_stems
    return keep


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
                    from fizgig.utils.upright import upright_size
                    wh = list(upright_size(im))
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
    mask = mask_for_stem(folder, stem)                # masked training: a picture's mask goes with it
    if mask:
        names.append(os.path.basename(mask))
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
                from fizgig.utils.upright import upright
                im = upright(im)
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
        how = ("Fizgig updates itself every time this rented GPU starts again - stop and start the pod (RunPod) or "
               "restart the instance (Vast.ai) to get the latest.")
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
    update_last_used({"recent_folders": [path] + old[:RECENT_KEEP - 1],      # picked again: on the list again
                      "hidden_folders": [h for h in (d.get("hidden_folders") or []) if h != key]})


def forget_folder(path):
    """The tile's x: off Your training data (its files stay). Hidden, not only dropped from the recent ones - a past
    run's training folder came straight back from the run history (#266) - until it's picked again."""
    key = os.path.normcase(os.path.abspath(str(path or "")))
    d = load_last_used()
    update_last_used({"recent_folders": [p for p in (d.get("recent_folders") or []) if isinstance(p, str)
                                         and os.path.normcase(os.path.abspath(p)) != key],
                      "hidden_folders": [h for h in (d.get("hidden_folders") or []) if h != key] + [key]})


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
    seen, out = set(d.get("hidden_folders") or []), []          # the ones taken off the list (forget_folder)
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
            if not kind or is_mask_file(e.name):          # photo-masklabel.png: photo.png's mask, not a picture
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
        # its sub-folders' pictures: trained on only with the Training tab's Also train on its sub-folders
        subs = launch.subfolders(path) if any(e.is_dir() and not e.name.startswith(".") for e in entries) else []
        sub_items = sum(launch._media_count(f) for _r, f in subs)
        if pics + clips + songs + sub_items:
            out.append({"name": name, "path": path, "pictures": pics, "clips": clips, "songs": songs,
                        "captioned": captioned, "time": newest, "thumbs": thumbs or waves, "elsewhere": elsewhere,
                        "sub_items": sub_items, "sub_folders": len(subs)})
    out.sort(key=lambda x: -x["time"])
    return {"root": root, "datasets": out[:300], "total": len(out)}


def _upload_top(dest, single):
    """The training folder an upload went into: its top folder when a file went into one of its sub-folders."""
    if single:
        return dest
    root = uploads_root()
    rel = os.path.relpath(dest, root)
    return os.path.join(root, rel.split(os.sep)[0]) if not rel.startswith("..") else dest


def _upload_target(folder, name, single):
    """(name, ext, the folder it goes into, its half-written .part) - every check an upload goes through."""
    name = _upload_name(name, "File name")
    ext = os.path.splitext(name)[1].lower()
    if single:
        if ext not in UPLOAD_FILE_EXTS:
            raise ValueError(f"{name}: not a LoRA (.safetensors), face model (.onnx), picture, clip or song.")
        dest = UPLOAD_FILES
    else:
        # "name/sub/folder": a folder upload that keeps its sub-folders (Also train on its sub-folders)
        folder = "/".join(_upload_name(p, "Folder name") for p in str(folder or "").split("/"))
        if ext not in UPLOAD_EXTS:
            raise ValueError(f"{name}: Fizgig doesn't train on {ext or 'this'} files. It takes pictures, .mp4 clips, "
                             f"songs (.wav .mp3 .flac .m4a), caption .txt files, .json song details and .zip files.")
        dest = os.path.join(uploads_root(), *folder.split("/"))
    return name, ext, dest, os.path.join(dest, f".{name}.part")


def upload_status(folder, name, single, size, mtime=None):
    """For a resumed upload: how much of the file is already here (its .part), or done (the file itself: the same
    size and the same modified time - an edited caption of the same length is sent again), so the page sends only
    the rest - or nothing."""
    name, ext, dest, tmp = _upload_target(folder, name, single)
    final = os.path.join(dest, name)
    same_time = mtime is not None and os.path.isfile(final) and abs(os.path.getmtime(final) - mtime) < 2
    if ext != ".zip" and os.path.isfile(final) and os.path.getsize(final) == size and same_time:
        return {"have": size, "done": True, "path": final, "folder": _upload_top(dest, single)}
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
    return {"folder": _upload_top(dest, single), "saved": sorted(set(saved)), "skipped": skipped,
            "replaced": replaced,
            "badname": badname}


# ---- model downloads: the desktop's "Download models for me" (fizgig.scripts.fetch_models) ----------------------
_fetch = {"proc": None, "state": "idle", "family": "", "file": "", "done": 0, "total": 0, "lines": []}


def _fetch_status() -> dict:
    with _lock:
        out = {k: v for k, v in _fetch.items() if k not in ("proc", "lines")}
        out["lines"] = _fetch["lines"][-40:]
    out["env_token"] = bool(os.environ.get("HF_TOKEN"))
    return out


def _start_fetch(family, token="", optional=False, choices=None):
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
    offered = {f.pref_key for f in desc.model_files if f.fetch_choice}
    for key, pick in (choices or {}).items():   # the card's pick for a file offered with another build
        if key in offered and pick in ("main", "alt"):
            cmd += ["--choose", f"{key}={pick}"]
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


_on_use = {}          # pref key -> {"state": "running" | "failed" | "done", "at": time}: files fetched on first use


def _fetch_on_use(f):
    """One model file declared fetch_on_use, downloaded the first time its feature is used: into the models folder
    on a worker thread, its Preferences row set once it is complete (hf_hub_download writes the final name only
    then). A failed download is tried again 30 s on, not on every form change. Returns "running" / "failed" / ""."""
    with _lock:
        st = _on_use.get(f.pref_key)
        if st and (st["state"] == "running" or (st["state"] == "failed" and time.time() - st["at"] < 30)):
            return st["state"]
        _on_use[f.pref_key] = {"state": "running", "at": time.time()}

    def work():
        try:
            from fizgig.utils import hf_public
            got = hf_public.download(repo_id=f.repo, filename=f.path, local_dir=user_data.MODELS_DIR)
            dest = os.path.join(user_data.MODELS_DIR, f.local_name or os.path.basename(f.path))
            if os.path.normcase(os.path.abspath(got)) != os.path.normcase(os.path.abspath(dest)):
                os.replace(got, dest)             # a file in a repo subfolder: to the models folder itself
            update_prefs({f.pref_key: dest})
            with _lock:
                _on_use[f.pref_key] = {"state": "done", "at": time.time()}
            logger.info(f"Downloaded {os.path.basename(dest)} on first use")
        except Exception as e:
            with _lock:
                _on_use[f.pref_key] = {"state": "failed", "at": time.time()}
            logger.warning(f"Couldn't download {f.path} from {f.repo}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return "running"


def _on_use_gb(f):
    """How much of a first-use download has arrived, in GB: hf_hub_download's part file in the models folder's
    .cache (a hashed name ending .incomplete) until it is complete - the largest there, as a resumed one may sit
    beside an older leftover."""
    d = os.path.join(user_data.MODELS_DIR, ".cache", "huggingface", "download", os.path.dirname(f.path))
    try:
        return max((os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".incomplete")),
                   default=0) / 2 ** 30
    except OSError:
        return 0.0


def _look_fetch(desc, values, path):
    """The preview look LoRA (MOSS-Image's SFT look LoRA), ticked on the Samples tab: its own download
    (_fetch_on_use) when it is not on disk. Returns (state, GB arrived of GB): state "running" / "failed" / "done"
    (just finished: the note says so) / "" (nothing to say)."""
    look = desc.preview_look()
    if look is None or not look.fetch_on_use or not values.get("FAMILY_PREVIEW_LOOK"):
        return "", None
    if path and os.path.isfile(path):
        with _lock:
            st = _on_use.get(look.pref_key) or {}
        return ("done" if st.get("state") == "done" and time.time() - st["at"] < 120 else ""), None
    state = _fetch_on_use(look)
    return state, ((_on_use_gb(look), look.size_gb) if state == "running" else None)


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


def captioner_for(prefs) -> str:
    """The captioner (the Captions tab's Qwen3-VL and auto-recaption): the secondary captioning model when that row is
    set to a file it can read (#270), else the Krea 2 text encoder."""
    sec = pref_path(prefs, captions.SECONDARY_PREF)
    if sec and os.path.isfile(sec) and not captions.captioner_problem(sec):
        return sec
    return pref_path(prefs, "krea2_text_encoder")


def secondary_captioner_note(prefs) -> str:
    """The secondary captioning row's line for Preferences: what captions, or why the file set there can't."""
    sec = pref_path(prefs, captions.SECONDARY_PREF)
    if not sec:
        return ""
    if not os.path.isfile(sec):
        return "That file isn't there: captioning uses the Krea 2 text encoder."
    why = captions.captioner_problem(sec)
    return (f"Not used - {why}. Captioning uses the Krea 2 text encoder." if why
            else f"Captioning (the Captions tab and auto-recaption) uses {os.path.basename(sec)}.")


def _caption_gpu_check(need_gb):
    """With a run training, a caption job only starts when the card has room for it beside the run (what the job
    needs + 3 GB)."""
    with _lock:
        mine = _run["proc"] is not None and _run["proc"].poll() is None
    if not mine:
        return
    g = _chosen_gpu(list_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES") or load_prefs().get("cuda_device"))
    if g is None:
        free = 0.0
        # ROCm launch only (#188). NVIDIA path keeps free=0 when list_gpus is empty (same as before).
        if _rocm_launch():
            try:
                from fizgig.utils.vram_monitor import read_amd_gpu_vram
                hit = read_amd_gpu_vram()
                if hit and hit[1] > 0:
                    free = (hit[1] - hit[0]) / 2 ** 30
            except Exception:
                pass
    else:
        free = g["total_gb"] - g["used_gb"]
        # ROCm synthetic row (uuid ""): typeperf may still report used=0 before its first sample - refuse
        # captioning beside a run until a real used reading exists.
        if _rocm_launch() and g.get("uuid") == "" and g["used_gb"] <= 0 and free >= need_gb + 3.0:
            try:
                from fizgig.utils.vram_monitor import read_amd_gpu_vram
                hit = read_amd_gpu_vram()
                if hit is None or hit[0] <= 0:
                    free = 0.0
                else:
                    free = (hit[1] - hit[0]) / 2 ** 30
            except Exception:
                free = 0.0
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
        Secure through a RunPod pod's HTTPS proxy (a Vast.ai instance's ports are plain http: a Secure cookie would
        never be kept there)."""
        return ("Set-Cookie", f"fizgig_session={self._session()}; Path=/; HttpOnly; SameSite=Strict"
                              + ("; Max-Age=2592000" if remember else "") + ("; Secure" if on_runpod() else ""))

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
                              "captioner": {"path": str(p.get(captions.SECONDARY_PREF) or ""),
                                            "note": secondary_captioner_note(p),
                                            "links": [{"label": lab, "href": f"https://huggingface.co/{r}/blob/main/{f}"}
                                                      for lab, r, f in captions.SECONDARY_LINKS]},
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
                                      "storage": pod_storage(), "provider": "vast" if on_vast() and not on_runpod()
                                      else "runpod", "https": __import__("fizgig.web.pod_stop",
                                                                         fromlist=["x"]).tunnels().get("web", "")}
                              if on_runpod() or on_vast() else None,     # the rented GPU's own card
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
        if path.startswith(("/api/helper/", "/api/helpers")):   # helpers: another Fizgig's GPU (web/helpers.py)
            from fizgig.web import helpers
            g = lambda k: (qs.get(k) or [""])[0]
            try:
                if path == "/api/helpers":
                    helpers.ensure_lead()
                    return self.json(helpers.listing(look=g("look") == "1"))
                if path == "/api/helper/info":
                    return self.json(helpers.info())
                if path == "/api/helper/checkpoint_sha":   # a computer that brought a checkpoint home checks it
                    return self.json(helpers.checkpoint_sha(g("path")))
                if path == "/api/helpers/remote_status":
                    return self.json(helpers.remote_status())
                if path == "/api/helper/move_file":       # a run moved here: a file of it, for the computer it came from
                    return self._send(200, helpers.move_piece(g("tag"), g("name"), int(g("offset") or 0),
                                                              int(g("n") or helpers.PIECE)), "application/octet-stream")
                if path == "/api/helpers/share_cache":     # the run's cache, a piece at a time, for a helper pulling
                    return self._send(200, helpers.share_cache_piece(g("i") or 0, g("name"), int(g("offset") or 0),
                                                                     int(g("n") or helpers.PIECE)), "application/octet-stream")
                if path == "/api/helpers/share_startfile":
                    return self._send(200, helpers.share_start_piece(g("epoch"), int(g("offset") or 0),
                                                                     int(g("n") or helpers.PIECE)), "application/octet-stream")
                if path == "/api/helper/share_state":
                    return self.json(helpers.share_state(g("tag"), int(g("epoch") or 0)))
                if path == "/api/helper/share_file":
                    return self._file(helpers.share_file(g("tag"), int(g("epoch") or 0)))
                if path == "/api/helper/have":
                    return self.json(helpers.have(g("name")))
                if path == "/api/helper/speed":          # a link test's download: bytes to time
                    return self._send(200, helpers.speed_piece(int(g("n") or helpers.PIECE)), "application/octet-stream")
                if path == "/api/helper/status":
                    return self.json(helpers.status(g("job")))
                if path == "/api/helper/file":
                    return self._file(helpers.file_path(g("job"), g("name")))
                if path == "/api/helpers/lora":          # a round's LoRA, a piece at a time, for a helper pulling it
                    return self._send(200, helpers.lora_file(g("round"), int(g("offset") or 0),
                                                             int(g("n") or helpers.PIECE), g("which") or "lora"),
                                      "application/octet-stream")
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/captions/options":
            p = load_prefs()
            cap = captioner_for(p)
            sec = pref_path(p, captions.SECONDARY_PREF)
            desc = get_family((qs.get("family") or [""])[0])
            return self.json({**captions.options(p, cap if cap and os.path.isfile(cap) else "",
                                                 cap if cap and cap == sec else ""),
                              "song_words": getattr(desc, "song_captions", "") if desc is not None else ""})
        if path == "/api/presets":               # a model's presets: the built-in ones, then the ones you saved
            desc = _preset_family((qs.get("family") or [""])[0])
            if desc is None:
                return self.json({"error": "unknown family"}, 400)
            built = [n for n, _ in (desc.presets or ())]
            return self.json({"builtin": built, "user": [n for n in user_presets(desc) if n not in built]})
        if path == "/api/prompt_sets":           # the Samples tab's saved prompt sets
            return self.json({"sets": prompt_sets()})
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
        if path == "/api/studio/render_choice":     # the workbench's render choice, and Studio's note with it
            desc = get_family((qs.get("family") or [""])[0])
            if desc is None:
                return self.json({"render_choice": None})
            sm = _studio_samples(desc) if desc.workbench_follows_samples else None
            L = studio.layout(desc, load_prefs(), pref_path, samples=sm)
            return self.json({k: L.get(k) for k in ("key", "render_choice", "follow_ck", "follow_look", "follow_turbo",
                                                     "follow_steps", "follow_cfg", "follows_samples")})
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
        if path in ("/api/masks/edit/list", "/api/masks/edit/picture", "/api/masks/edit/mask"):   # the mask editor
            from fizgig.web import mask_edit
            folder, name = (qs.get("folder") or [""])[0], (qs.get("name") or [""])[0]
            if not folder or not os.path.isdir(folder):
                return self.json({"error": "Choose the training folder on the Start tab first."}, 400)
            try:
                if path == "/api/masks/edit/list":
                    return self.json({"pictures": mask_edit.listing(folder)})
                if path == "/api/masks/edit/picture":
                    return self._send(200, mask_edit.picture_jpeg(folder, name), "image/jpeg")
                data = mask_edit.mask_png(folder, name)
                return self.json({"error": "no mask"}, 404) if data is None else self._send(200, data, "image/png")
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/masks/info":              # Image Prep's Make masks card: what's there for this folder
            from fizgig.dataset import automask
            folder = (qs.get("folder") or [""])[0]
            pics = automask.pictures(folder) if folder and os.path.isdir(folder) else []
            have = sum(1 for p in pics if mask_for_stem(folder, os.path.splitext(os.path.basename(p))[0]))
            return self.json({"kinds": automask.KINDS, "available": automask.available(),
                              "model_here": os.path.isfile(automask.model_path(user_data.MODELS_DIR)),
                              "model_mb": automask.MODEL_MB, "pictures": len(pics), "with_mask": have,
                              "baselines": bool(prep_jobs.look_baselines(folder)) if folder else False,
                              "faces": prep_jobs.faces_available(), "sam": _sam_info()})
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
                # Continue: the newest run with its name and folder owns its saves (an older one's are overwritten)
                r["continue"] = _continue_point(r) if key not in newest else None
                newest.add(key)
            return self.json({"runs": runs})
        if path == "/api/queue/item":                # a queued run's settings, for Edit
            it = webqueue.get((qs.get("id") or [""])[0])
            if it is None:
                return self.json({"error": "That run has left the queue."}, 404)
            return self.json({**{k: it.get(k) for k in ("id", "family", "image_folder", "lora")},
                              "values": {k: v for k, v in (it.get("values") or {}).items() if not str(k).startswith("__")},
                              "continues": bool((it.get("values") or {}).get("__continue"))})
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
            pics = image_loss.image_paths(run, [im["key"] for im in res["images"]])
            queued, applied = image_loss.caption_state(run.get("output_dir"))
            return self.json({**res, "live": run.get("result") in ("running", "paused"),
                              "images": [{**im, "path": pics.get(im["key"], ""), "queued": im["key"] in queued,
                                          "applied": (applied.get(im["key"]) or [{}])[-1].get("epoch")}
                                         for im in res["images"]]})
        if path == "/api/runs/caption":              # one training image's caption, for the per-image editor
            from fizgig.web import image_loss
            run = history.get((qs.get("id") or [""])[0]) or {}
            text, ok = image_loss.read_caption(run, (qs.get("key") or [""])[0],
                                               captions.caption_ext((qs.get("ext") or [""])[0]))
            return self.json({"caption": text, "folder_ok": ok})
        if path == "/api/about":                     # ?check=1: look for a newer release again (the About box's)
            from fizgig.web import update_check
            if (qs.get("check") or [""])[0] == "1":
                update_check.start()
            from fizgig.web import pod_version
            return self.json({**about_info(), "release": update_check.status(),
                              "pod_version": pod_version.status() if on_pod() else None})
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
                return self.json(pods.offers(pod_stop.api_key(load_prefs()), (qs.get("dc") or [""])[0],
                                             (qs.get("count") or ["1"])[0]))
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
            from fizgig.web import helpers
            r["move"] = helpers.move_status()
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
            return self._captions((qs.get("folder") or [""])[0], (qs.get("ext") or [".txt"])[0],
                                  (qs.get("family") or [""])[0])
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
                isdir = e.is_dir()
                item = {"name": e.name, "path": e.path}
                try:                                 # dates and size for the picker's columns (stat is cached on Windows)
                    st = e.stat()
                    item["mtime"] = st.st_mtime
                    item["ctime"] = getattr(st, "st_birthtime", None) or (st.st_ctime if os.name == "nt" else None)
                    if not isdir:
                        item["size"] = st.st_size
                except OSError:
                    pass
                (dirs if isdir else files).append(item)
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
        by_epoch = epoch_prompts(folder, run) if run else {}
        try:
            with open(os.path.join(folder, f".previews_{run}.json"), encoding="utf-8") as f:
                mode = json.load(f) if run else None
        except (OSError, ValueError):
            mode = None
        newest = by_epoch[max(by_epoch, key=int)] if by_epoch else None
        return self.json({"items": out[:1000] if run else out[:300], "runs": runs, "epoch_prompts": by_epoch,
                          "loras": self._epoch_loras(os.path.dirname(folder.rstrip("\\/")), run, infos) if run else {},
                          "preview_mode": mode if isinstance(mode, dict) else None,
                          "preview_labels": preview_labels(mode),
                          "prompts": newest or (run_prompts(os.path.dirname(folder.rstrip("\\/")), run) if run else [])})

    @staticmethod
    def _epoch_loras(out_dir, run, infos):
        """{epoch: the LoRA file saved at it} for a run's previews: <run>-NNNNNN.safetensors, and the run's final
        <run>.safetensors for its last previewed epoch when that one has no numbered file (fine-tunes left out)."""
        found = {}
        try:
            for e in os.scandir(out_dir):
                m = _CKPT.match(e.name[:-len(".safetensors")]) if e.name.lower().endswith(".safetensors") else None
                if m and m.group(1) == run and m.group(2):
                    found[int(m.group(2))] = e.path
        except OSError:
            return {}
        epochs = [i["epoch"] for i in infos.values() if i and i.get("run") == run and i.get("epoch") is not None]
        final = os.path.join(out_dir, run + ".safetensors")
        if epochs and max(epochs) not in found and os.path.isfile(final):
            found[max(epochs)] = final
        return {str(k): p for k, p in found.items() if _checkpoint_kind(p) != "finetune"}

    def _checkpoints(self, folder):
        """The run's saved LoRAs / checkpoints in its output folder, newest first."""
        try:
            items = [e for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(".safetensors")]
        except OSError:
            return self.json({"items": []})
        now, stats = time.time(), []
        for e in items:
            try:                                 # one deleted while this reads (a run moved away clears its files)
                st = e.stat()
            except OSError:
                continue
            if now - st.st_mtime > 5:            # not one still being written
                stats.append((e, st))
        stats.sort(key=lambda x: x[1].st_mtime, reverse=True)
        base = lambda n: (lambda m: m.group(1) if m else n)(_CKPT.match(n[:-len(".safetensors")]))
        return self.json({"items": [{"name": e.name, "path": e.path, "size": st.st_size, "time": st.st_mtime,
                                     "kind": _checkpoint_kind(e.path), "run": base(e.name)} for e, st in stats]})

    def _captions(self, folder, ext, family=""):
        out = []
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return self.json({"items": []})
        keep = _caption_media_filter(names, family)
        lower = {n.lower(): n for n in names}
        for n in names:
            stem, e = os.path.splitext(n)
            if e.lower() not in MEDIA or not keep(n) or is_mask_file(n):
                continue
            cap = os.path.join(folder, stem + ext)
            try:
                text = captions.read_text(cap)          # any encoding: one odd file never empties the list (#225)
            except OSError:
                text = None
            p = os.path.join(folder, n)
            mask = mask_for_stem(folder, stem, lower)        # masked training: the picture's mask, shown on its row
            out.append((n, p, {"caption": text, "size": _media_size(p),
                               **({"mask": os.path.basename(mask)} if mask else {})}))
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
                fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[-1].strip() if on_runpod() else ""
                if on_vast() and self.client_address[0] == "127.0.0.1":   # through the Vast image's https tunnel:
                    fwd = (self.headers.get("Cf-Connecting-Ip") or "").strip()  # Cloudflare names the visitor
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
        if path == "/api/helpers/deliver":            # a picture a helper made for one of this run's rounds
            from fizgig.web import helpers
            qs = parse_qs(urlparse(self.path).query)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                return self.json(helpers.deliver((qs.get("round") or [""])[0], (qs.get("name") or [""])[0],
                                                 self.rfile.read(n)))
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/helpers/share_deliver":      # a piece of a pulling helper's share LoRA for an epoch
            from fizgig.web import helpers
            qs = parse_qs(urlparse(self.path).query)
            g = lambda k: (qs.get(k) or [""])[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                return self.json(helpers.share_deliver(g("epoch"), int(g("offset") or 0), self.rfile.read(n),
                                                       done=g("done") == "1", steps=int(g("steps") or 0),
                                                       seconds=float(g("seconds") or 0), name=g("name"),
                                                       launch=g("launch")))
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/helper/speed":               # a link test's upload: read and dropped
            n = int(self.headers.get("Content-Length") or 0)
            while n > 0:
                n -= len(self.rfile.read(min(n, 1 << 20)) or b"x" * n)
            return self.json({"ok": True})
        if path == "/api/helper/put":                 # a piece of a file a lead sends this helper (web/helpers.py)
            from fizgig.web import helpers
            qs = parse_qs(urlparse(self.path).query)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                return self.json(helpers.put((qs.get("name") or [""])[0], int((qs.get("offset") or ["0"])[0]),
                                             self.rfile.read(n)))
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        b = self._body()
        if path.startswith("/api/helper/") or path.startswith("/api/helpers/"):
            from fizgig.web import helpers
            try:
                if path == "/api/helper/models":
                    return self.json(helpers.models(str(b.get("family") or ""), b.get("job"), bool(b.get("fetch"))))
                if path == "/api/helper/submit":
                    return self.json(helpers.submit(b))
                if path == "/api/helper/drop":
                    return self.json(helpers.drop(str(b.get("job") or ""), str(b.get("lora") or "")))
                if path == "/api/helpers/add":
                    return self.json(helpers.add(b.get("url"), b.get("password")))
                if path == "/api/helpers/remove":
                    return self.json(helpers.remove(str(b.get("id") or "")))
                if path == "/api/helpers/fetch":
                    return self.json(helpers.fetch_models(str(b.get("id") or "")))
                if path == "/api/helpers/set":
                    return self.json(helpers.set_jobs(str(b.get("id") or ""), b.get("previews"), b.get("share"),
                                                      b.get("every"), b.get("send_cache"), b.get("wait")))
                if path == "/api/helpers/check":
                    return self.json(helpers.check_picture(str(b.get("id") or "")))
                if path == "/api/helpers/remote":         # another computer's run's checkpoints, to try prompts here
                    return self.json(helpers.remote_checkpoints(b.get("url"), str(b.get("password") or "")))
                if path == "/api/helper/share_start":
                    return self.json(helpers.share_start(b))
                if path == "/api/helpers/move":           # this computer's run: carry on training on a helper
                    return self.json(helpers.move_start(str(b.get("id") or ""), bool(b.get("now")), b.get("home", True) is not False))
                if path == "/api/helpers/move_back":      # a run moved to a helper: back here, to carry on training
                    return self.json(helpers.move_back(bool(b.get("now")), bool(b.get("delete"))))
                if path == "/api/helpers/move_clear":
                    return self.json(helpers.move_clear(bool(b.get("here"))))
                if path == "/api/helper/move_in":         # a run moving here from another computer
                    return self.json(helpers.move_in(b))
                if path == "/api/helper/move_out":        # a run moved here going back: paused, its files listed
                    return self.json(helpers.move_out(b))
                if path == "/api/helpers/link":          # a helper's link: speeds typed in (Mb/s), or measured again
                    return self.json(helpers.link(b.get("url"), b.get("up"), b.get("down")))
                if path == "/api/helper/share_probe":     # can this computer make a lead's cache itself
                    return self.json(helpers.share_probe(b.get("family"), b.get("shas") or {}))
                if path == "/api/helper/share_job":
                    return self.json(helpers.share_job(b))
                if path == "/api/helper/share_cancel":
                    return self.json(helpers.share_cancel(b.get("tag"), b.get("epoch") or 0))
                if path == "/api/helper/share_stop":
                    return self.json(helpers.share_stop(str(b.get("tag") or "")))
                if path == "/api/helpers/local":          # this computer's other GPU as a helper, or not
                    return self.json(helpers.local_start(b.get("index")) if b.get("on") else helpers.local_stop())
                if path == "/api/helpers/home":           # a pod's checkpoints home as they're saved; delete it after
                    return self.json(helpers.home_set(b.get("url"), b.get("on"), b.get("previews"), b.get("delete"),
                                                      str(b.get("password") or "")))
                if path == "/api/helpers/remote_get":
                    return self.json(helpers.remote_get(b.get("url"), b.get("items") or [], str(b.get("run") or ""),
                                                        str(b.get("family") or "")))
                if path == "/api/helpers/share_hello":
                    return self.json(helpers.share_hello(b))
                if path == "/api/helpers/share_next":
                    return self.json(helpers.share_next(b))
                if path == "/api/helpers/share_fail":
                    return self.json(helpers.share_fail(b.get("epoch"), str(b.get("error") or "")))
                if path == "/api/helpers/pull":
                    return self.json(helpers.pull(b))
                if path == "/api/helpers/finish":
                    return self.json(helpers.finish(str(b.get("round") or ""), bool(b.get("ok")),
                                                    str(b.get("error") or "")))
                if path == "/api/helpers/help":
                    return self.json(helpers.help_start(b.get("url"), b.get("password") or "",
                                                        str(b.get("mode") or "previews")) if b.get("on")
                                     else helpers.help_stop())
                return self.json({"error": "That isn't built yet."}, 400)
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        try:
            if path == "/api/form/state":
                if b.get("family") == refmod.KEY:
                    return self.json(form.refmod_state(b.get("values") or {}, b.get("image_folder") or ""))
                desc = get_family(b.get("family", ""))
                if desc is None:
                    return self.json({"error": "unknown family"}, 400)
                prefs = load_prefs()
                cap = captioner_for(prefs)
                sp = desc.preview_speed()
                spf = pref_path(prefs, sp.pref_key) if sp is not None and sp.pref_key else ""
                ck = desc.preview_checkpoint()
                ckf = pref_path(prefs, ck[0].pref_key) if ck else ""
                vals = b.get("values") or {}
                lk = desc.preview_look()
                lkf = launch.look_path(desc, {"models": {lk.pref_key: pref_path(prefs, lk.pref_key)}}) if lk else ""
                dit = next((f.pref_key for f in desc.model_files if f.role == "dit"), "")
                st = form.state(desc, vals, b.get("image_folder") or "",
                                captioner=cap if cap and os.path.isfile(cap) else "",
                                speed_file=bool(spf and os.path.isfile(spf)),
                                checkpoint_file=bool(ckf and os.path.isfile(ckf)),
                                base_file=pref_path(prefs, dit) if dit else "",
                                checkpoint_path=ckf if ckf and os.path.isfile(ckf) else "",
                                look_file=bool(lkf and os.path.isfile(lkf)), look_fetch=_look_fetch(desc, vals, lkf))
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
            if path in ("/api/prompt_set", "/api/prompt_set/save", "/api/prompt_set/delete"):
                try:
                    p = prompt_set_path(b.get("name"))
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
                if path == "/api/prompt_set/save":
                    lines = [ln.rstrip() for ln in str(b.get("prompts") or "").splitlines()]
                    if not any(lines):
                        return self.json({"error": "Type a prompt first."}, 400)
                    if os.path.exists(p) and not b.get("overwrite"):
                        return self.json({"exists": True})
                    os.makedirs(PROMPT_SETS_DIR, exist_ok=True)
                    with open(p, "w", encoding="utf-8") as f:
                        f.write("\n".join(ln for ln in lines if ln.strip()) + "\n")
                elif path == "/api/prompt_set/delete":
                    if os.path.isfile(p):
                        os.remove(p)
                else:
                    try:
                        with open(p, encoding="utf-8") as f:
                            return self.json({"prompts": "\n".join(ln.rstrip() for ln in f.read().splitlines()
                                                                   if ln.strip())})
                    except OSError:
                        return self.json({"error": "That set's file can't be read."}, 400)
                return self.json({"ok": True, "sets": prompt_sets()})
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
                    was = (webqueue.get(editing) or {}).get("values") or {}
                    if was.get("__continue"):            # a queued Continue stays one: on from the same run's save
                        rec = history.get(was["__continue"])
                        pt = _continue_point(rec)
                        if pt is None:
                            return self.json({"error": "That run's saved state isn't there any more, so it can't "
                                                       "continue. Remove it from the queue."}, 400)
                        if (fam != rec.get("family") or lora.lower() != str(rec.get("lora") or "").lower()
                                or os.path.normcase(os.path.abspath(out_dir)) !=
                                os.path.normcase(os.path.abspath(rec.get("output_dir") or "."))):
                            return self.json({"error": "This queued run continues " + str(rec.get("lora")) + ": keep "
                                              "its model, LoRA name and output folder. For a new run, use Add to "
                                              "queue instead."}, 400)
                        try:
                            to = int(str(vals.get(pt["key"]) or "0").strip())
                        except ValueError:
                            to = 0
                        if pt["kind"] == "state" and to <= pt["done"]:
                            return self.json({"error": f"It's saved at epoch {pt['done']}: set Max Train Epochs above "
                                                       f"{pt['done']} for it to train on."}, 400)
                        vals = {**vals, "__continue": was["__continue"]}
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
            if path in ("/api/start", "/api/resume", "/api/queue/start"):
                from fizgig.web import helpers as _helpers
                if _helpers.move_busy():                # a run of this computer's is moving
                    return self.json({"error": _helpers.move_busy()}, 400)
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
            if path == "/api/runs/continue":            # Continue: on from the run's newest save
                from fizgig.web import helpers as _helpers
                if _helpers.move_busy():                # a run of this computer's is moving
                    return self.json({"error": _helpers.move_busy()}, 400)
                r = _continue_run(str(b.get("id") or ""), str(b.get("more") or ""))
                return self.json(r, 400 if r.get("error") and not r.get("need") else 200)
            if path == "/api/runs/settings":            # Start again with these settings
                r = _run_form(history.get(str(b.get("id") or "")))
                return self.json(r, 400 if r.get("error") else 200)
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
                samples = b.get("samples") if isinstance(b.get("samples"), dict) else {}
                for k in _RESUME_SAMPLE_KEYS:            # the previews as the Samples tab has them now
                    if k in samples and samples[k] is not None and str(samples[k]).strip() != "":
                        last["values"][k] = samples[k]
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
                    return self.json(_resumed(_start(last["family"], v, last["image_folder"],
                                            ft_continue={"continue": cont, "rotations": left},
                                            note=f"=== Resuming the fine-tune from {os.path.basename(last['resume'])}: "
                                                 f"{left} rotation(s) left ===")))
                if not os.path.isdir(last["resume"]):
                    return self.json({"error": "The paused run's saved state isn't there any more: "
                                               + last["resume"]}, 400)
                return self.json(_resumed(_start(last["family"], last["values"], last["image_folder"],
                                                 resume=last["resume"], note=f"=== Resuming from {last['resume']} ===")))
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
                anchors = [a.strip() for a in str(b.get("anchors") or "").splitlines() if a.strip()]
                _write_json(os.path.join(run["output_dir"], SLIDER_LIVE), {**new, "guidance": g, "anchors": anchors})
                changes = {_SLIDER_LIVE_KEYS[k]: v for k, v in {**new, "guidance": f"{g:g}",
                                                                 "anchors": "\n".join(anchors)}.items()}
                with _lock:                              # a Pause and Resume, or Start again, keeps them
                    if isinstance(_run.get("values"), dict):
                        _run["values"].update(changes)
                if run.get("history_id"):
                    history.set_values(run["history_id"], changes)
                return self.json({"ok": True})
            if path == "/api/run/sample_override":   # the Samples tab's live previews: the run going now's override
                with _lock:
                    out = _run["output_dir"] if _run["proc"] is not None and _run["proc"].poll() is None else ""
                    fam = _run["family"]
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
                live = {"prompts": prompts, "seed": _seed(b.get("seed"), 1234),
                        "width": _seed(b.get("width"), 1024), "height": _seed(b.get("height"), 1024)}
                for k in ("steps", "cfg", "turbo", "every", "until", "after"):   # Steps, CFG, Turbo strength and
                    # the preview cadence (until / after 0 = Then preview less often off), when given
                    try:
                        if str(b.get(k, "")).strip() != "":
                            live[k] = float(b[k]) if k in ("cfg", "turbo") else int(float(b[k]))
                    except (TypeError, ValueError):
                        pass
                desc = get_family(fam or "")
                lk = desc.preview_look() if desc is not None and "look" in b else None
                if lk is not None:      # the SFT look tick: its file's path when ticked ("" = off), at its strength
                    live["look"] = launch.look_path(desc, {"models": {lk.pref_key: pref_path(load_prefs(),
                                                                                            lk.pref_key)}}) \
                        if b.get("look") else ""
                    live["look_strength"] = desc.preview_look_strength
                ck = desc.preview_checkpoint() if desc is not None and desc.train_preview_checkpoint \
                    and "checkpoint" in b else None
                if ck:                  # the preview-model tick (Qwen's Turbo): its file when ticked, "" = off
                    path = pref_path(load_prefs(), ck[0].pref_key) if b.get("checkpoint") else ""
                    live["checkpoint"] = path if (not path or os.path.isfile(path)) else ""
                _write_json(side, live)
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
                _start_fetch(b.get("family", ""), b.get("token", ""), bool(b.get("optional")),
                             b.get("choices") if isinstance(b.get("choices"), dict) else None)
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
                # the workbench's render choice per model that offers it (description.workbench_render_choice)
                render_keys = {f"workbench_render_{d.key}" for d in training_families()
                               if getattr(d, "workbench_render_choice", False)}
                allowed = {k for d in training_families() for k in form.model_keys(d)} | set(SETTING_PREFS) | {
                    "input_dataset_dir", "welcome_v8_hidden", captions.SECONDARY_PREF} | render_keys
                ch = b.get("changes") or {}
                ch = {k: v for k, v in ch.items() if k not in render_keys or v in ("checkpoint", "samples")}
                if b.get("reset"):                      # Reset to defaults: the settings, never the model files
                    ch = {**{k: "" for k in SETTING_PREFS}, **{k: v for k, v in PREF_DEFAULTS.items()
                                                                  if k in SETTING_PREFS}}
                if "inference_int8" in ch:
                    ch["inference_int8"] = "1" if ch["inference_int8"] in (True, "1", 1) else "0"
                for k in ("lan_discovery", "update_check"):
                    if k in ch:
                        ch[k] = "1" if ch[k] in (True, "1", 1) else "0"
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
                if "lan_discovery" in upd:              # Preferences > Network: answering other computers, now
                    from fizgig.web import lan
                    if upd["lan_discovery"] == "0":
                        lan.stop()
                    elif _LAN_PORT[0]:
                        _lan_announce(_LAN_PORT[0])
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
                cap = captioner_for(prefs)
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
                if on_runpod():
                    pid = pod_stop.pod_id()
                    base = f"https://{pid}-8081.proxy.runpod.net/" if pid else ""
                elif on_pod():                      # a Vast.ai instance: its https tunnel, else its public address
                    ip, vp = os.environ.get("PUBLIC_IPADDR"), os.environ.get("VAST_TCP_PORT_8081")
                    web = pod_stop.tunnels().get("web", "")
                    base = web.rstrip("/") + "/" if web else f"http://{ip}:{vp}/" if ip and vp else ""
                else:
                    ip = lan_ip()
                    base = f"http://{ip}:{self.server.server_address[1]}/" if ip else ""
                if not base:
                    return self.json({"error": "This computer has no address on a network for a phone to open."}, 400)
                return self.json({"url": base + "?phone=" + _phone_pass_new(), "seconds": 30})
            if path == "/api/pod/match":            # a pod: run this computer's Fizgig (pod_version.py)
                from fizgig.web import pod_version
                return self.json(pod_version.match(b.get("commit"), str(b.get("by") or "")[:80], bool(b.get("auto"))))
            if path == "/api/pod/newest":           # a pod: back on the newest Fizgig
                from fizgig.web import pod_version
                return self.json(pod_version.newest())
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
                ok = pod_stop.cancel(append_log)
                hold = min(7200, max(0, int(b.get("hold") or 0)))
                if hold:                            # a computer bringing its files home: only for so long
                    _hold_stop(hold)
                return self.json({"ok": ok})
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
            if path == "/api/masks/sam_fetch":      # SAM 3.1 into the models folder (1.75 GB, once)
                from fizgig.dataset import sammask
                sammask.download(user_data.MODELS_DIR)
                return self.json({"ok": True, "sam": _sam_info()})
            if path == "/api/masks/edit/sam_click":  # the mask editor: the object SAM 3.1 finds where you clicked
                from fizgig.web import mask_edit
                import base64
                where = _sam_info()["path"]
                if not where:
                    return self.json({"error": "SAM 3.1 isn't here yet: choose SAM 3.1 on this card and press "
                                               "Download first."}, 400)
                try:
                    png = mask_edit.sam_click(str(b.get("folder") or ""), str(b.get("name") or ""),
                                              float(b.get("x") or 0), float(b.get("y") or 0), where)
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
                except Exception as e:
                    logger.exception("SAM click")
                    return self.json({"error": f"SAM 3.1 couldn't run: {type(e).__name__}: {e}"}, 400)
                return self.json({"ok": True, "png": "data:image/png;base64," + base64.b64encode(png).decode()})
            if path == "/api/masks/sam_preview":    # the first two pictures with their SAM masks, nothing written
                from fizgig.dataset import sammask
                import base64
                import io
                folder, where = str(b.get("folder") or ""), _sam_info()["path"]
                if not where:
                    return self.json({"error": "SAM 3.1 isn't here yet: press Download first."}, 400)
                include, exclude = sammask.split_words(b.get("include")), sammask.split_words(b.get("exclude"))
                if not include:
                    return self.json({"error": "Type what the masks should show in Mask these (e.g. person)."}, 400)
                try:
                    shots, fresh = sammask.preview(folder, include, exclude, where, shown=b.get("shown") or [])
                except ValueError as e:
                    return self.json({"error": str(e)}, 400)
                except Exception as e:
                    logger.exception("SAM preview")
                    return self.json({"error": f"SAM 3.1 couldn't run: {type(e).__name__}: {e}"}, 400)
                pics = []
                for name, pic, cover in shots:
                    buf = io.BytesIO()
                    pic.save(buf, "JPEG", quality=88)
                    pics.append({"name": name, "cover": round(cover, 3),
                                 "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
                return self.json({"ok": True, "pictures": pics, "fresh": fresh, "of": len(sammask.pictures(folder))})
            if path in ("/api/prep/run", "/api/look/scan", "/api/look/move", "/api/masks/make", "/api/masks/edit/save"):
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
                    if path == "/api/masks/edit/save":
                        from fizgig.web import mask_edit
                        return self.json(mask_edit.save(folder, str(b.get("name") or ""), str(b.get("png") or "")))
                    if path == "/api/masks/make":
                        sam = None
                        if b.get("sam"):
                            from fizgig.dataset import sammask
                            sam = {"path": _sam_info()["path"], "include": sammask.split_words(b.get("include")),
                                   "exclude": sammask.split_words(b.get("exclude"))}
                        return self.json(prep_jobs.start_masks(folder, str(b.get("kind") or ""),
                                                               bool(b.get("subject_only")), bool(b.get("overwrite")),
                                                               sam=sam))
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
                    return self.json(royale.score(st["gen"], rendered, [str(r) for r in b.get("refs") or []]))
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


_LAN_PORT = [0]          # the port other computers are told, when this Fizgig can be found at all (a password, not local)


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
    _SRV[0] = srv
    if history.close_orphans():                        # a run left "running" by a stopped or crashed server
        logger.info("A run from before was still recorded as running: it is recorded as stopped now")
    webqueue.load()                                    # held if it holds anything: a restart never launches a run
    _restore_pause()                                   # a run paused before the restart can still be resumed
    __import__("fizgig.web.helpers", fromlist=["x"]).ensure_home()   # pods' checkpoints still coming home
    __import__("fizgig.web.helpers", fromlist=["x"]).local_restore()  # this computer's other GPU, as a helper
    if os.environ.get("FIZGIG_PARENT_PID"):          # this is that second Fizgig: it goes when the first does
        threading.Thread(target=_follow_first, daemon=True).start()
    if str(load_prefs().get("update_check", "1")) != "0":     # Preferences > Network: Check for new releases
        __import__("fizgig.web.update_check", fromlist=["x"]).start()   # a newer release? (the About button says)
    threading.Thread(target=_poll_stats_loop, daemon=True).start()
    # the page's first ask is the schema, and its first build imports torch (block maps, the optimizer list): seconds,
    # more on a slow disk. Built now, while the window or browser opens, the page has it sooner (#249)
    def _warm():
        try:
            get_schema(ref_dit_set=False)
        except Exception:                      # the page's own ask says what's wrong
            pass
    threading.Thread(target=_warm, daemon=True).start()
    if password and not on_pod() and host not in ("127.0.0.1", "localhost", "::1"):
        _LAN_PORT[0] = port                    # Preferences > Network can switch the answering on and off
        if __import__("fizgig.web.lan", fromlist=["x"]).enabled(load_prefs()):
            _lan_announce(port)                # other computers' Pods tabs find this one by themselves
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
        if _SWITCH[0]:                         # a pod on another version: its start-up script starts it again
            logging.shutdown()
            os._exit(0)


_RESTART = [False]
_SWITCH = [False]
_SRV = [None]


def shutdown_for_switch():
    """A pod switching version (pod_version.py): its helpers' workers go, then the server, then the process."""
    from fizgig.web import helpers
    helpers.shutdown()
    _SWITCH[0] = True
    if _SRV[0] is not None:
        _SRV[0].shutdown()
_held = [0.0]


def _hold_stop(seconds):
    """A computer bringing a finished run's files home holds this pod's own stop for `seconds`, asking again while it
    works; if it stops asking (asleep, restarted, logged out), the stop starts again by itself, so the pod never
    bills on for nobody."""
    first = _held[0] == 0.0
    _held[0] = time.time() + seconds

    def watch():
        from fizgig.web import helpers, pod_stop
        while time.time() < _held[0]:
            time.sleep(10)
        _held[0] = 0.0
        with _lock:
            busy = _run["proc"] is not None and _run["proc"].poll() is None
            out = _run.get("output_dir") or ""
        if not busy and not webqueue.status()["items"] and not helpers.moved_in_waiting(out):
            append_log("[pod] the computer bringing this run home stopped asking: this pod's own stop goes on.\n")
            pod_stop.maybe_start(load_prefs(), append_log)
    if first:
        threading.Thread(target=watch, daemon=True).start()


def _follow_first():
    """A second Fizgig on this computer's other GPU ends with the Fizgig that started it, or when that one asks
    (its stop file): its runs and share trainers end with it, as Ctrl+C ends them."""
    try:
        import psutil
    except ImportError:
        psutil = None
    pid, stop = int(os.environ["FIZGIG_PARENT_PID"]), user_file("stop")
    # Linux: still its child (a killed parent can linger as a zombie, so its pid alone says nothing); Windows: its pid
    alive = (lambda: os.getppid() == pid) if os.name != "nt" else (lambda: psutil is None or psutil.pid_exists(pid))
    while not os.path.exists(stop) and alive():
        time.sleep(3)
    try:
        os.remove(stop)
    except OSError:
        pass
    from fizgig.web import helpers
    helpers.shutdown()
    signal.raise_signal(signal.SIGTERM)


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
    if on_pod():                              # a pod kept on the version of the computer using it: that one first
        from fizgig.web import pod_version
        pod_version.boot()
    user_data.migrate()                       # Fizgig 7's loose files into user_data, before anything reads them
    # a pod: its VNC_PASSWORD, the remote desktop's, so one password opens both. Elsewhere VNC_PASSWORD is ignored (a
    # stray one would open the web UI to the network while Preferences says no password).
    saved = str(load_prefs().get("web_password") or "").strip()
    pwd = (a.password or os.environ.get("FIZGIG_PASSWORD")
           or ((os.environ.get("VNC_PASSWORD") or saved) if on_pod() else saved))
    # a RunPod pod's own VNC password is taken at any length (it opens its remote desktop too); anywhere else the
    # image runs (Vast.ai: plain http on a public address) a password is 8 characters or more, as at home
    pod_pwd = on_runpod() and not a.password and not os.environ.get("FIZGIG_PASSWORD") and pwd == os.environ.get(
        "VNC_PASSWORD")
    if pwd and len(pwd) < 8 and not pod_pwd:
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
