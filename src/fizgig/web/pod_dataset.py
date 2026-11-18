"""Start tab, on a pod: download a training folder from a link (to-do #13, @girzai #308). Runs
scripts/fetch_dataset.py as its own process (Stop ends it; a second try picks up where it stopped), checks the disk
first, and reports progress from the folder's size."""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from fizgig.scripts.fetch_dataset import ARCHIVES, parse

_lock = threading.Lock()
_job = {"proc": None, "folder": "", "total": 0, "start": 0.0, "state": "idle", "message": "", "link": ""}


def root():
    """Where downloaded training folders go: /workspace/datasets on a pod, else Fizgig's user_data/datasets."""
    from fizgig.user_data import USER_DATA
    base = "/workspace" if os.path.isdir("/workspace") else USER_DATA
    return os.path.join(base, "datasets")


def size_of(link):
    """(bytes, is_archive) the download will take, or (None, None) when the source doesn't say."""
    kind, target = parse(link)
    if kind == "hf":
        from huggingface_hub import HfApi
        repo, sub = target
        info = HfApi().dataset_info(repo, files_metadata=True, token=os.environ.get("HF_TOKEN") or None)
        files = [f for f in info.siblings if not sub or f.rfilename.startswith(sub.rstrip("/") + "/")]
        return sum(int(f.size or 0) for f in files), any(f.rfilename.lower().endswith(ARCHIVES) for f in files)
    if kind == "url":
        import requests
        r = requests.head(target, allow_redirects=True, timeout=30)
        n = int(r.headers.get("content-length") or 0)
        return (n or None), target.split("?")[0].lower().endswith(ARCHIVES)
    return None, True                             # Drive: no size without downloading


def disk_problem(need, is_archive, free):
    """A plain refusal when the folder (and, for an archive, its unpacked copy) won't fit with 10% to spare, else ""."""
    if not need:
        return ""
    want = need * (2 if is_archive else 1) * 1.1
    if want <= free:
        return ""
    gb = lambda b: f"{b / 1e9:.1f} GB"
    return (f"Not enough disk on this pod: the download is {gb(need)}" + (" and unpacking it takes as much again"
            if is_archive else "") + f", but {gb(free)} is free. Rent the pod with a bigger disk, or a network "
            "volume, and try again.")


def _name(link):
    kind, target = parse(link)
    if kind == "hf":
        raw = target[1].split("/")[-1] or target[0].split("/")[-1]
    elif kind == "drive":
        raw = f"drive_{target[:8]}"
    else:
        raw = os.path.basename(target.split("?")[0])
    for ext in ARCHIVES:
        if raw.lower().endswith(ext):
            raw = raw[:-len(ext)]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "dataset"


def start(link, name=""):
    with _lock:
        if _job["proc"] is not None and _job["proc"].poll() is None:
            raise ValueError("A download is already running.")
    parse(link)                                    # a link it can't take: the reason, before anything starts
    folder = os.path.join(root(), re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) if name.strip() else _name(link))
    os.makedirs(folder, exist_ok=True)
    try:
        total, is_archive = size_of(link)
    except Exception as e:
        raise ValueError(f"Couldn't read that link: {e}")
    problem = disk_problem(total, is_archive, shutil.disk_usage(folder).free)
    if problem:
        raise ValueError(problem)
    log = open(os.path.join(folder, ".fizgig_download.log"), "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "fizgig.scripts.fetch_dataset", link, folder], stdout=log,
                            stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    with _lock:
        _job.update(proc=proc, folder=folder, total=total or 0, start=time.time(), state="running", message="",
                    link=link)
    return status()


def _folder_bytes(folder):
    n = 0
    for dp, _dn, fn in os.walk(folder):
        for f in fn:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n


def status():
    with _lock:
        j = dict(_job)
    p = j["proc"]
    if p is not None and p.poll() is not None and j["state"] == "running":
        try:
            lines = open(os.path.join(j["folder"], ".fizgig_download.log"), encoding="utf-8").read().splitlines()
        except OSError:
            lines = []
        err = next((ln[6:] for ln in reversed(lines) if ln.startswith("ERROR ")), "")
        state = "done" if p.returncode == 0 else ("failed" if err else "stopped")
        with _lock:
            _job.update(state=state, message=err or ("" if state == "done" else "The download stopped."))
            j = dict(_job)
    have = _folder_bytes(j["folder"]) if j["folder"] and os.path.isdir(j["folder"]) else 0
    secs = max(1e-6, time.time() - j["start"]) if j["start"] else 0
    speed = have / secs if secs and j["state"] == "running" else 0
    left = (j["total"] - have) / speed if speed and j["total"] > have else None
    return {"state": j["state"], "folder": j["folder"], "bytes": have, "total": j["total"], "speed": speed,
            "seconds_left": left, "message": j["message"], "link": j["link"]}


def stop():
    with _lock:
        p = _job["proc"]
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
        with _lock:
            _job.update(state="stopped", message="Stopped. Download again with the same link and folder name to carry on.")
    return status()
