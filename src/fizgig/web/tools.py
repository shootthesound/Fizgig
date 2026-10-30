"""The web UI's Extract and Profiler tabs: each the desktop tab's job, run in its own process so the server never loads
torch - Extract the extract CLI (exact weight SVD from the file, no model loaded), Profiler scripts/profile_run.py (the
weights report, or the measured profile through the workbench engine). Their progress and logs come back for the page.
One run of each at a time."""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_lock = threading.Lock()


def _job():
    return {"state": "idle", "log": [], "progress": None, "text": "", "output": "", "outputs": [], "error": "",
            "proc": None, "started": 0.0, "result": None}


_J = {"extract": _job(), "profile": _job()}


def families(kind="extract"):
    """The models a tab offers: every family whose description lists that workbench (the desktop's lists)."""
    from fizgig.families.registry import training_families
    return [d for d in training_families() if kind in (d.workbench or ()) and d.training_ready]


def info(desc, out_dir):
    groups = [{"label": g.label, "blocks": [{"id": b.id, "label": b.label} for b in g.blocks]}
              for g in desc.load_driver().block_map()] if desc.extract_presets else []
    return {"family": desc.key, "presets": [{"name": n, "blocks": sorted(dict(b))} for n, b in desc.extract_presets],
            "groups": groups, "out_dir": out_dir}


def free_name(path):
    """`path`, or path_2, path_3... - an extraction never overwrites a file."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def _start(kind, cmd, first_line, output=""):
    with _lock:
        if _J[kind]["state"] == "running":
            raise ValueError(f"A {'profile' if kind == 'profile' else 'extraction'} is running - wait for it to finish.")
    from fizgig.web.server import gpu_env
    env = gpu_env(dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",           # the Preferences GPU
                       PYTHONPATH=os.path.join(BASE_DIR, "src") + os.pathsep + BASE_DIR))
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kw)
    with _lock:
        _J[kind] = {**_job(), "state": "running", "log": [first_line], "output": output, "proc": proc,
                    "started": time.time()}
    threading.Thread(target=_follow, args=(kind, proc), daemon=True).start()


def _line(kind, line):
    j = _J[kind]
    m = re.match(r"^(\w+): (\d+)/(\d+)$", line)
    if kind == "extract" and m:                  # the extract CLI's "\rSVD: i/n"
        j["progress"] = [int(m.group(2)), int(m.group(3))]
        return
    if line.startswith("@@"):                    # the profile run's events
        tag, _, body = line[2:].partition(" ")
        try:
            d = json.loads(body)
        except ValueError:
            return
        if tag == "OUT":                         # a file the fine-tune extraction wrote (one per rank)
            j["outputs"].append(d)
            j["output"] = j["output"] or d
            return
        if tag == "P":
            j["text"] = d.get("text", j["text"])
            if d.get("total"):
                j["progress"] = [d.get("done", 0), d["total"]]
        elif tag == "DONE":
            j["result"] = d
            j["output"] = d.get("report", "")
        elif tag == "FAIL":
            j["error"] = "Stopped." if d.get("stopped") else (d.get("error") or "failed")
        return
    j["log"] = (j["log"] + [line])[-400:]


def _follow(kind, proc):
    buf = b""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in (b"\r", b"\n"):
            line = buf.decode("utf-8", "replace").strip()
            buf = b""
            if line:
                with _lock:
                    _line(kind, line)
        else:
            buf += ch
    code = proc.wait()
    with _lock:
        j = _J[kind]
        if j["proc"] is not proc:
            return
        ok = code == 0 and (kind != "extract" or os.path.isfile(j["output"]))
        stopped = j.get("stopping")
        j.update(state="done" if ok else "stopped" if stopped else "failed", proc=None)
        if not ok and not j["error"]:
            j["error"] = "Stopped." if stopped else (j["log"][-1] if j["log"] else f"exit code {code}")
        j["log"].append(f"Done in {time.time() - j['started']:.0f} s." if ok else
                        "Stopped." if stopped else f"The {'profile' if kind == 'profile' else 'extraction'} failed.")


def status(kind="extract"):
    with _lock:
        return {k: _J[kind][k] for k in ("state", "log", "progress", "text", "output", "outputs", "error", "result")}


def holding_gpu():
    with _lock:
        return _J["profile"]["state"] == "running"


def stop(kind):
    with _lock:
        j = _J[kind]
        p = j["proc"]
        if p is None:
            return False
        j["stopping"] = True
    try:
        p.kill()
    except OSError:
        pass
    return True


# ---- Extract ----------------------------------------------------------------------------------------------------
def run(desc, source, output, rank, preset="", blocks=None):
    if not source or not os.path.isfile(source):
        raise ValueError("Choose a source LoRA (that file isn't there).")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output = free_name(output)
    cmd = [sys.executable, os.path.join(BASE_DIR, "src", "fizgig", "scripts", "extract_lora.py"), "--family", desc.key,
           "--source", source, "--output", output, "--rank", str(int(rank))]
    if blocks:
        cmd += ["--blocks", ",".join(blocks)]
    elif preset:
        cmd += ["--preset", preset]
    _start("extract", cmd, f"Extracting {os.path.basename(source)} at rank {rank} -> {output}", output)
    return output


def run_finetune(base, tuned, out_dir, name, ranks):
    """A full fine-tune as ordinary LoRAs: its difference from the base it started from, one file per rank from one
    SVD pass (scripts/extract_finetune.py, the old Checkpoint to LoRA app's job). Never over a file: each name carries
    the time."""
    for path, what in ((base, "the base checkpoint"), (tuned, "the trained checkpoint")):
        if not path or not os.path.isfile(path):
            raise ValueError(f"Choose {what} (that file isn't there).")
    if os.path.normcase(os.path.abspath(base)) == os.path.normcase(os.path.abspath(tuned)):
        raise ValueError("The base and the trained checkpoint are the same file.")
    ranks = sorted({int(r) for r in ranks if int(r) >= 1})
    if not ranks:
        raise ValueError("Tick at least one rank.")
    os.makedirs(out_dir, exist_ok=True)
    name = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip()) or "extracted"
    cmd = [sys.executable, "-m", "fizgig.scripts.extract_finetune", "--base", base, "--tuned", tuned, "--out-dir", out_dir,
           "--name", name, "--ranks", ",".join(map(str, ranks))]
    _start("extract", cmd, f"Extracting {os.path.basename(tuned)} against {os.path.basename(base)} at rank "
                           f"{', '.join(map(str, ranks))} -> {out_dir}")


# ---- Profiler ---------------------------------------------------------------------------------------------------
def profile_prefill(lora):
    """The desktop's _profiler_prefill_from_lora: prompts from the LoRA's trigger, and the run's likeness baselines
    (its sample gallery's three photos) when it has them."""
    from fizgig.utils.st_meta import read_metadata
    try:
        md = read_metadata(lora)
    except Exception:
        md = {}
    trig = (md.get("modelspec.trigger_phrase") or md.get("ss_trigger_word") or "").strip()
    baselines = []
    try:
        sdir = os.path.join(os.path.dirname(lora), "sample")
        with open(os.path.join(sdir, "likeness.json"), encoding="utf-8") as f:
            lk = json.load(f)
        with open(os.path.join(sdir, "dataset.json"), encoding="utf-8") as f:
            ds = json.load(f).get("folder", "")
        paths = [os.path.join(ds, b) for b in lk.get("baselines", [])]
        if len(paths) == 3 and all(os.path.isfile(p) for p in paths):
            baselines = paths
    except (OSError, ValueError, AttributeError):
        pass
    prompt, class_prompt = _run_sample_prompt(lora, trig)
    if not prompt:                       # no run on record (a downloaded LoRA): a portrait, as most profiled LoRAs are
        prompt = f"{trig + ', ' if trig else ''}a portrait photo of a person, natural light"
        class_prompt = "a portrait photo of a person, natural light"
    return {"prompt": prompt, "class_prompt": class_prompt, "baselines": baselines}


def _run_sample_prompt(lora, trig):
    """(prompt, the same prompt without the trigger) from the run's first Samples prompt - what its owner set up to
    look at this LoRA, a person or a style - or ("", "") when the run isn't in history or that prompt doesn't hold
    the trigger (the untouched default prompt). A Fizgig LoRA Isolation run's prompts use joined
    words (zwxemwoman) a bleed check can't strip, and Edit and Slider runs don't preview with these prompts, so
    those keep the portrait."""
    from fizgig.web import history
    from fizgig.families import registry
    from fizgig.families.form import _kind
    name = re.sub(r"-\d{6}$", "", os.path.splitext(os.path.basename(lora))[0])       # an epoch file: its run's name
    try:
        run = next((r for r in history.runs() if history._same_run(r, name, os.path.dirname(lora))), None)
    except Exception:
        run = None
    vals = (run or {}).get("values") or {}
    if not run or vals.get("FAMILY_MULTI_PERSON"):
        return "", ""
    desc = registry.get(run.get("family") or "")
    if desc and _kind(desc, vals) in ("edit", "slider"):
        return "", ""
    first = next((ln.strip() for ln in str(vals.get("SAMPLE_PROMPT") or "").splitlines() if ln.strip()), "")
    if not first:
        return "", ""
    words = [trig, str(vals.get("CAPTION_TRIGGER_WORD") or "").strip()]      # a phrase trigger: also each part
    words += [w.strip() for w in trig.split(",") if w.strip()] if "," in trig else []
    if run.get("image_folder"):                                      # the word the captions start with
        try:
            from fizgig.families.multi_person import caption_trigger
            words.append(caption_trigger(run["image_folder"], vals.get("DATASET_CAPTION_EXT") or ".txt") or "")
        except Exception:
            pass
    word = next((w for w in words if w and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", first, re.I)), "")
    if not word:
        return "", ""
    bare = re.sub(r"(?<!\w)" + re.escape(word) + r"(?!\w)", "", first, flags=re.I)    # as sample_caption
    bare = re.sub(r"\s+([,.])", r"\1", bare)
    bare = re.sub(r"\s*,\s*(,\s*)*", ", ", bare).strip(" ,")            # the comma the trigger leaves behind
    return first, bare


def run_profile(job):
    lora = job.get("lora") or ""
    if not lora or not os.path.isfile(lora):
        raise ValueError("Choose a LoRA (that file isn't there).")
    if job.get("mode") != "weights" and not (job.get("prompt") or "").strip():
        raise ValueError("Enter a prompt (with the LoRA's trigger word, if it has one).")
    fd, path = tempfile.mkstemp(prefix="fizgig_profile_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(job, f)
    _start("profile", [sys.executable, "-m", "fizgig.scripts.profile_run", path],
           f"Profiling {os.path.basename(lora)} ({job.get('mode')})")
