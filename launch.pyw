"""Fizgig launcher — double-click to run without a console window.

The .pyw extension tells Windows to use pythonw.exe, which hides the
console.  Console output is captured by the GUI's built-in log — click
the status indicator in the top-right corner to view it.

A hidden console hides FAILURES too: if the GUI dies on import, pythonw
discards the traceback and the app simply never appears — no window, no
error, no clue.  (A user hit exactly this with a Python installed without
Tkinter.)  So every startup failure is reported through two channels that
cannot themselves depend on what broke: a native Windows message box via
ctypes — deliberately not a Tkinter dialog, since missing Tkinter is the
classic cause — and launch_error.log beside this file.
"""
import os
import sys
import subprocess
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHONW = os.path.join(HERE, "venv", "Scripts", "pythonw.exe")
LOG = os.path.join(HERE, "launch_error.log")


def _report(title, message, detail=""):
    """The failure path. Must not use Tkinter, and must not raise."""
    logged = False
    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{message}\n\n{detail}\n"
                    f"\npython: {sys.executable}\nversion: {sys.version}\n")
        logged = True
    except Exception:
        pass
    body = message + (f"\n\nDetails saved to:\n{LOG}" if logged else "")
    try:
        import ctypes
        # 0x10 = error icon, 0x10000 = bring the box to the foreground
        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10010)
    except Exception:
        # Not Windows, or no usable GUI session. Guarded because stderr is
        # None under pythonw, and the reporter must never raise.
        if getattr(sys, "stderr", None):
            sys.stderr.write(f"{title}\n{body}\n{detail}\n")
    sys.exit(1)


# If the venv exists and we're not already running from it, re-launch
if os.path.exists(VENV_PYTHONW) and os.path.normcase(sys.executable) != os.path.normcase(VENV_PYTHONW):
    try:
        subprocess.Popen([VENV_PYTHONW, os.path.abspath(__file__)])
    except Exception as exc:
        _report("Fizgig could not start",
                "The bundled Python failed to launch:\n"
                f"{VENV_PYTHONW}\n\n"
                "Re-run install_fizgig.bat to repair the environment.",
                f"{type(exc).__name__}: {exc}")
    sys.exit(0)

# Checked HERE, before the GUI is touched, so the report names the actual
# problem instead of surfacing as a traceback from line 1 of a 20,000-line
# file.  Tkinter ships with Python itself — it is NOT pip-installable (the
# `tk` package on PyPI is unrelated), so requirements.txt cannot supply it;
# the fix is a Python installed with it.
try:
    import tkinter  # noqa: F401
except Exception as exc:
    _report(
        "Fizgig — Python is missing Tkinter",
        "This Python was installed without Tkinter, which Fizgig needs for "
        "its interface.\n\n"
        "Reinstall Python from python.org and tick “tcl/tk and IDLE” "
        "during setup, then run install_fizgig.bat again.",
        f"{type(exc).__name__}: {exc}")

try:
    if os.path.isfile(LOG):
        os.remove(LOG)          # stale report from an earlier failed start
except Exception:
    pass

sys.path.insert(0, HERE)


def _start_with_splash():
    """Put the splash up before the 27k-line GUI file is even compiled (#115).

    The Tk root is created HERE, withdrawn, and handed to the GUI — one interpreter, so
    every image and variable the GUI creates lands on the right root. The GUI module is
    imported on a worker thread while this thread keeps the splash animating: the import
    is where a slow drive spends its minutes (torch, insightface, transformers), and it
    touches no Tk objects at module level. The build itself then runs on the main thread
    with the constructor reporting each tab to the splash."""
    import importlib.util
    import threading
    import time
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        from fizgig_splash import Splash
        splash = Splash(root, "Loading modules… (the first launch on a slow drive can "
                              "take a few minutes)")
    except Exception:                       # no splash is a cosmetic loss, never a failure
        class _NoSplash:
            def status(self, _text): pass
            def pump(self): pass
            def close(self): pass
        splash = _NoSplash()

    result = {}

    def _load():
        try:
            spec = importlib.util.spec_from_file_location(
                "lora_trainer_gui", os.path.join(HERE, "lora_trainer_gui.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["lora_trainer_gui"] = mod
            spec.loader.exec_module(mod)
            result["mod"] = mod
            # Pre-warm what the constructor imports lazily (the caption-task list pulls in
            # transformers; likeness pulls insightface). On a slow drive those are the
            # minutes — better spent here, where the splash is animating, than mid-build
            # where nothing can repaint. Best effort: the constructor imports them itself
            # regardless, so a failure here changes nothing.
            src = os.path.join(HERE, "src")           # the GUI inserts this itself, later
            if src not in sys.path:
                sys.path.insert(0, src)
            for name in ("torch", "safetensors", "fizgig.krea2.embedder"):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
        except BaseException as exc:            # reported on the main thread below
            result["err"] = exc

    worker = threading.Thread(target=_load, name="fizgig-import", daemon=True)
    worker.start()
    while worker.is_alive():
        splash.pump()
        time.sleep(0.03)
    if "err" in result:
        splash.close()
        raise result["err"]              # carries its own traceback into _report
    result["mod"].main(root=root, splash=splash)


try:
    _start_with_splash()
except (SystemExit, KeyboardInterrupt):
    raise                       # deliberate exits are not errors
except BaseException as exc:
    _report("Fizgig could not start",
            f"Fizgig hit an error while starting:\n\n{type(exc).__name__}: {exc}\n\n"
            "If this keeps happening, please open a GitHub issue and attach "
            "the log below.",
            traceback.format_exc())
