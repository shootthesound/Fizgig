"""Fizgig's own files - its settings, state, caches, presets and profiles - live in one folder, user_data, in the
Fizgig folder, so the Fizgig folder itself holds only what ships with it.

Fizgig 7 kept them loose in the Fizgig folder. migrate() moves them in once: whatever reads them first calls it (the
app's start, the updater's model fetch, Gizmo, a pod's start-up), so after the update to 8 the models stay linked and
the presets stay listed. Nothing is deleted except leftovers of what Fizgig 8 no longer has; if both places hold the
same file, the old one goes to user_data/from_fizgig_7 instead.

Standalone on purpose: scripts run by path (the updater's) import it without the rest of Fizgig.
"""
import json
import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# a second Fizgig on this computer's other GPU (web/helpers.py local_start) keeps its own settings and state there;
# the model files stay shared
OWN_USER_DATA = os.environ.get("FIZGIG_USER_DATA") or ""
USER_DATA = os.path.abspath(OWN_USER_DATA) if OWN_USER_DATA else os.path.join(BASE_DIR, "user_data")
PREFS_FILE = os.path.join(USER_DATA, "prefs.json")
LAST_USED_FILE = os.path.join(USER_DATA, "last_used.json")       # the training folder and its trigger words
PRESETS_DIR = os.path.join(USER_DATA, "presets")
MODELS_DIR = os.path.join(BASE_DIR, "user_data", "models")  # where Download models for me puts them

# what Fizgig 7 kept in its folder -> where it lives now
_MOVES = (
    ("prefs.json", "prefs.json"), (".last_used.json", "last_used.json"), ("models", "models"), ("presets", "presets"),
    ("profiles", "profiles"), ("cache", "cache"), ("dataset", "dataset"), ("gizmo_settings.json", "gizmo_settings.json"),
    ("rocm_env.bat", "rocm_env.bat"), ("rocm_env.sh", "rocm_env.sh"), ("web_state.json", "web_state.json"),
    (".fizgig_web_queue.json", "web_queue.json"), (".fizgig_runs.json", "runs.json"),
    (".fizgig_app_profile", "app_profile"), (".fizgig_web_studio", "studio"), (".fizgig_web_caption", "caption_jobs"),
)
# leftovers of what Fizgig 8 no longer has (the classic desktop app, Diff to LoRA) and of passing state
_GONE = (
    ".diff_to_lora.json", ".fizgig_web_zips", ".fizgig_live_desktop.json", ".fizgig_live_web.json",
    ".fizgig_desktop_cmd.json", ".fizgig_desktop_samples.json", ".fizgig_desktop_console.log",
    "repair_studio_last_preview.log", "launch_error.log", "gizmo_error.log", "gpu_detect_debug.log",
)
# a Preferences folder still on its old default follows its folder in
_PREF_DIRS = (("cache_dir", "cache"), ("profiles_dir", "profiles"))
_done = []


def user_file(name):
    os.makedirs(USER_DATA, exist_ok=True)
    return os.path.join(USER_DATA, name)


def pref(key, default=""):
    """One Preferences setting as text, read from prefs.json each call (a training run or worker reads the choice the
    page saved), or `default` when it isn't set or the file can't be read."""
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            v = json.load(f).get(key)
    except (OSError, ValueError):
        return default
    return default if v is None or v == "" else str(v)


def _park(src):
    """Both places hold it: the old one is kept, out of the way and where it was (its folders too), not lost."""
    dest = os.path.join(USER_DATA, "from_fizgig_7", os.path.relpath(src, BASE_DIR))
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)


def _move(src, dest):
    if not os.path.exists(src):
        return False
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)                         # a rename on the same drive: a big cache moves at once
        return True
    if os.path.isdir(src) and os.path.isdir(dest):     # merge a folder in, item by item
        for name in os.listdir(src):
            _move(os.path.join(src, name), os.path.join(dest, name))
        try:
            os.rmdir(src)
        except OSError:
            _park(src)
        return True
    _park(src)
    return True


def _repoint_prefs():
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            prefs = json.load(f)
    except (OSError, ValueError):
        return
    changed = False
    # a model downloaded into the old models folder: its path follows the folder in
    old_models = os.path.normcase(os.path.join(BASE_DIR, "models"))
    for key, v in list(prefs.items()):
        if not isinstance(v, str) or not v.strip():
            continue
        full = os.path.normpath(os.path.join(BASE_DIR, v.strip()))
        if os.path.normcase(full).startswith(old_models + os.sep):
            prefs[key] = os.path.join(MODELS_DIR, full[len(old_models) + 1:]).replace("\\", "/")
            changed = True
    for key, old in _PREF_DIRS:
        v = str(prefs.get(key) or "").strip()
        if v and os.path.normcase(os.path.normpath(os.path.join(BASE_DIR, v))) == \
                os.path.normcase(os.path.join(BASE_DIR, old)):
            prefs[key] = "user_data/" + old
            changed = True
    if changed:
        tmp = PREFS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
        os.replace(tmp, PREFS_FILE)


def _carry_over():
    """What Fizgig 7 knew that Fizgig 8 keeps elsewhere, taken across once: the model it was on, with its last
    training settings, for the web app's first start (only while the web app hasn't chosen one of its own), and the
    runs in its queue, into the web app's queue (held: nothing starts by itself). Its queue file is then renamed so
    the runs come across once."""
    def read(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    try:
        from fizgig.families import registry
    except Exception:
        return
    key_of = lambda label: getattr(registry.by_gui_label(str(label or "")), "key", "")
    state_file = os.path.join(USER_DATA, "web_state.json")
    state = read(state_file)
    state = state if isinstance(state, dict) else {}
    if not state.get("family"):
        key = key_of((read(LAST_USED_FILE) or {}).get("architecture"))
        if key:
            state["family"] = key
            last = read(os.path.join(PRESETS_DIR, ".last_train_settings.json"))
            values = state.get("values") if isinstance(state.get("values"), dict) else {}
            if isinstance(last, dict) and key not in values:
                values[key] = {k: v for k, v in last.items() if isinstance(v, (str, int, float, bool))}
                state["values"] = values
            _write(state_file, state)
    runs_file = os.path.join(USER_DATA, "runs.json")
    if not os.path.exists(runs_file):
        runs = _past_runs(read(LAST_USED_FILE) or {}, registry, read(PREFS_FILE) or {})
        if runs:
            _write(runs_file, runs)
    old_queue = os.path.join(PRESETS_DIR, "training_queue.json")
    items = read(old_queue)
    if not isinstance(items, list):
        return
    web_file = os.path.join(USER_DATA, "web_queue.json")
    web = read(web_file)
    web = web if isinstance(web, list) else []
    for i in items:
        values = i.get("preset") if isinstance(i, dict) else None
        key = key_of(i.get("architecture")) if isinstance(i, dict) else ""
        if not key or not isinstance(values, dict):
            continue
        web.append({"id": str(i.get("id") or "")[-10:] or os.urandom(5).hex(), "family": key,
                    "values": {k: v for k, v in values.items() if isinstance(v, (str, int, float, bool))},
                    "image_folder": str(i.get("image_folder") or ""), "lora": str(values.get("LORA_NAME") or ""),
                    "output_dir": str(values.get("LORA_OUTPUT_DIR") or ""),
                    "queued_at": str(i.get("queued_at") or ""), "from_fizgig_7": True})
    _write(web_file, web)
    os.replace(old_queue, old_queue[:-5] + ".fizgig7.json")


def _past_runs(last_used, registry, prefs=None) -> list:
    """Fizgig 7 kept no run list, but each LoRA it trained carries its run in its header (the model, epochs, rank, the
    learning rate, when): one finished run per final LoRA in its output folders, oldest first, for Runs & outputs."""
    import re
    import struct
    # the desktop kept its output folder in last_used; a Fizgig 7 pod set it in prefs (/workspace/output_loras)
    dirs = [last_used.get("lora_output_dir"), (prefs or {}).get("lora_output_dir"), os.path.join(BASE_DIR, "output_loras")]
    dirs += list((last_used.get("lora_output_dirs") or {}).values()) if isinstance(last_used.get("lora_output_dirs"), dict) else []
    seen, out = set(), []
    for d in dirs:
        if not isinstance(d, str) or not os.path.isdir(d) or os.path.normcase(os.path.abspath(d)) in seen:
            continue
        seen.add(os.path.normcase(os.path.abspath(d)))
        for name in os.listdir(d):
            stem, ext = os.path.splitext(name)
            if ext.lower() != ".safetensors" or re.search(r"-\d{6}$", stem):
                continue                                  # an epoch's save: the run is its final file
            path = os.path.join(d, name)
            try:
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    meta = json.loads(f.read(n)).get("__metadata__") or {} if 0 < n < (64 << 20) else {}
                when = os.path.getmtime(path)
            except (OSError, ValueError, struct.error):
                continue
            if "fizgig" not in str(meta.get("ss_network_module", "")).lower():
                continue                                  # someone else's LoRA in the folder
            desc = registry.by_arch_id(str(meta.get("ss_architecture") or ""))
            if desc is None:
                continue
            epochs = str(meta.get("ss_epoch") or "")
            values = {"LORA_NAME": stem, "LORA_OUTPUT_DIR": d, "MAX_TRAIN_EPOCHS": epochs,
                      "NETWORK_DIM": str(meta.get("ss_network_dim") or ""),
                      "NETWORK_ALPHA": str(meta.get("ss_network_alpha") or "").replace(".0", ""),
                      "LEARNING_RATE": str(meta.get("ss_learning_rate") or "")}
            out.append({"id": os.urandom(6).hex(), "source": "desktop", "family": desc.key, "lora": stem,
                        "output_dir": d, "image_folder": "", "values": {k: v for k, v in values.items() if v},
                        "started_at": when, "ended_at": when, "result": "finished",
                        "progress": {"epoch": int(epochs) if epochs.isdigit() else None,
                                     "total_epochs": int(epochs) if epochs.isdigit() else None, "percent": 100.0}})
    out.sort(key=lambda r: r["ended_at"])
    return out


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def migrate(log=print):
    """Fizgig 7's files into user_data, once per process (cheap when there is nothing to move). Never raises: a file
    another program holds open stays where it is, and the next start tries again."""
    if _done or OWN_USER_DATA:                 # a second Fizgig's folder has no Fizgig 7 files to take in
        return
    _done.append(True)
    moved = []
    for old, new in _MOVES:
        try:
            if _move(os.path.join(BASE_DIR, old), os.path.join(USER_DATA, new)):
                moved.append(old)
        except OSError:
            pass
    for old in _GONE:
        p = os.path.join(BASE_DIR, old)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    try:
        _carry_over()
    except Exception:
        pass
    if moved:
        try:
            _repoint_prefs()
        except OSError:
            pass
        if log:
            log(f"[fizgig] moved Fizgig 7's files into user_data: {', '.join(moved)}")
