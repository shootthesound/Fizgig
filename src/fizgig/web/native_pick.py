"""The computer's own Open / Choose folder window, for a page open on the machine running Fizgig.

tkinter's dialog runs in a fresh `python -c` (its own main thread, nothing left over in the server), with a hidden,
topmost root so the window comes up in front. No tkinter or no screen: unavailable, and the page uses its own picker.
"""
import json
import os
import subprocess
import sys

_SCRIPT = r"""
import json, sys
a = json.loads(sys.argv[1])
import tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw()
try: root.attributes("-topmost", True)
except Exception: pass
root.update()
kw = {"parent": root, "initialdir": a["start"] or None, "title": a["title"]}
p = filedialog.askdirectory(mustexist=True, **kw) if a["dir"] else filedialog.askopenfilename(
    filetypes=[("All files", "*.*")], **kw)
root.destroy()
print(json.dumps({"path": p or ""}))
"""

_avail = None


def available():
    """tkinter is here and there is a screen to show it on (checked once, cheaply)."""
    global _avail
    if _avail is None:
        try:
            import importlib.util
            ok = importlib.util.find_spec("tkinter") is not None
        except Exception:
            ok = False
        if ok and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            ok = False
        _avail = ok
    return _avail


def start_dir(start):
    """The folder the window opens in: start itself, else its folder, else none (the OS picks)."""
    start = (start or "").strip()
    if start and os.path.isdir(start):
        return start
    if start and os.path.isdir(os.path.dirname(start)):
        return os.path.dirname(start)
    return ""


def pick(start="", dir_only=True, title=""):
    """{"path": chosen} / {"cancelled": True} / {"unavailable": True} (the page falls back to its own picker)."""
    if not available():
        return {"unavailable": True}
    args = json.dumps({"start": start_dir(start), "dir": bool(dir_only),
                       "title": title or ("Choose a folder" if dir_only else "Choose a file")})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run([sys.executable, "-c", _SCRIPT, args], capture_output=True, text=True,
                           timeout=3600, creationflags=flags)
        out = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {"unavailable": True}
    p = out.get("path") or ""
    return {"path": os.path.normpath(p)} if p else {"cancelled": True}
