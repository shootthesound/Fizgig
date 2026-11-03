"""What the demo setup and launchers share: where things live, the config, starting and finding the servers."""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LLM_DIR = os.path.join(HERE, "llm")
PHOTOS = os.path.join(HERE, "my photos")
LOGS = os.path.join(HERE, "logs")
CONFIG = os.path.join(HERE, "config.json")
PY = os.path.join(REPO, "venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(REPO, "venv", "bin", "python")
PORT = 8095                                         # the demos' own Fizgig, apart from one you may run on 8081
FIZGIG = f"http://127.0.0.1:{PORT}"
LLM_PORT = 38100
LLM = f"http://127.0.0.1:{LLM_PORT}"
LLM_NAME = "demo-llm"                               # the model's name in the router's presets.ini
LLAMA_BUILD = "b10666"                              # pinned: the build the demos were made with

sys.path.insert(0, os.path.join(REPO, "src"))


def config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(c):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=1)


def up(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for(cond, seconds, what):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(2)
    raise SystemExit(f"{what} didn't come up in {seconds} s.")


def _detached(cmd, cwd, log):
    os.makedirs(LOGS, exist_ok=True)
    out = open(os.path.join(LOGS, log), "a", encoding="utf-8")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} if os.name == "nt" \
        else {"start_new_session": True}
    subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)


def llama_server():
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    for root, _dirs, files in os.walk(os.path.join(LLM_DIR, "llama.cpp")):
        if exe in files:
            return os.path.join(root, exe)
    return ""


def start_llm():
    """The language model's server (a llama.cpp router, so Fizgig can take it off the GPU while it renders)."""
    if up(LLM + "/health"):
        return
    exe = llama_server()
    if not exe:
        raise SystemExit("The language model isn't set up: run '0 - Set up everything' first.")
    print("Starting the language model's server ...")
    _detached([exe, "--models-preset", os.path.join(LLM_DIR, "presets.ini"), "--models-max", "1",
               "--port", str(LLM_PORT), "--jinja"], os.path.dirname(exe), "llm.log")
    wait_for(lambda: up(LLM + "/health"), 120, "The language model's server")


def start_fizgig():
    if up(FIZGIG + "/"):
        return
    if not os.path.isfile(PY):
        raise SystemExit("Fizgig isn't installed: run '0 - Set up everything' first.")
    print("Starting Fizgig ...")
    _detached([PY, os.path.join(REPO, "fizgig.py"), "--web", "--port", str(PORT)], REPO, "fizgig.log")
    wait_for(lambda: up(FIZGIG + "/"), 180, "Fizgig")


class Fizgig:
    """Fizgig's web API, as its page uses it."""

    def __init__(self, base=FIZGIG):
        self.base = base
        page = urllib.request.urlopen(base + "/", timeout=30).read().decode()
        m = re.search(r'name="fizgig-token" content="([^"]+)"', page)
        self.token = m.group(1) if m else ""

    def __call__(self, path, body=None, timeout=60):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"X-Fizgig-Token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")


def prefs_update(**values):
    """Merge values into Fizgig's Preferences file (Fizgig not running, or restarted after)."""
    from fizgig.user_data import user_file
    p = user_file("prefs.json")
    try:
        with open(p, encoding="utf-8") as f:
            prefs = json.load(f)
    except (OSError, ValueError):
        prefs = {}
    prefs.update(values)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    return prefs
