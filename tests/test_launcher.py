"""The launcher's every path, actually executed — not read.

launch.pyw runs under pythonw, where a startup failure has no console: the
app just never appears (a user with a Tkinter-less Python reported exactly
that).  The fix reports through a native message box + launch_error.log, and
this suite runs each path in a sandboxed copy with a stub GUI so the real
app never opens.

Two testing tricks worth knowing:
 * A real MessageBoxW BLOCKS until clicked, so failure paths run with a
   recorder ctypes shadowing the stdlib via PYTHONPATH.  Any run that pops a
   real dialog therefore hangs and fails by timeout — which doubles as proof
   the happy path never shows one.
 * Missing Tkinter is simulated the same way: a poisoned tkinter.py that
   raises on import, shadowing the real one.

Run: venv/Scripts/python.exe tests/test_launcher.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = r"W:/Peter/Documents/Development/Fizgig"
PY = os.path.join(REPO, "venv", "Scripts", "python.exe")
PYW = os.path.join(REPO, "venv", "Scripts", "pythonw.exe")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


BASE = tempfile.mkdtemp(prefix="fizgig_launch_sim_")
SB = os.path.join(BASE, "app")
os.makedirs(SB)
shutil.copy(os.path.join(REPO, "launch.pyw"), os.path.join(SB, "launch.pyw"))
shutil.copy(os.path.join(REPO, "fizgig_splash.py"), os.path.join(SB, "fizgig_splash.py"))
LAUNCH = os.path.join(SB, "launch.pyw")
LOGF = os.path.join(SB, "launch_error.log")
MARK = os.path.join(SB, "started.txt")

# --- the stand-ins ------------------------------------------------------------------------------
FAKECT = os.path.join(BASE, "fakectypes")
os.makedirs(FAKECT)
MSGBOX = os.path.join(FAKECT, "msgbox.json")
with open(os.path.join(FAKECT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(f"""
        import json
        class _User32:
            def MessageBoxW(self, hwnd, text, title, flags):
                with open({MSGBOX!r}, "w", encoding="utf-8") as fh:
                    json.dump({{"title": title, "text": text, "flags": flags}}, fh)
                return 1
        class _WinDLL:
            user32 = _User32()
        windll = _WinDLL()
        """))

POISON = os.path.join(BASE, "poison_tk")
os.makedirs(POISON)
with open(os.path.join(POISON, "tkinter.py"), "w", encoding="utf-8") as f:
    f.write("raise ImportError('simulated: Python built without Tkinter')\n")

BROKENCT = os.path.join(BASE, "broken_ct")
os.makedirs(BROKENCT)
with open(os.path.join(BROKENCT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write("raise ImportError('simulated: no ctypes on this platform')\n")

# Linux is the other shape: `import ctypes` SUCCEEDS but there is no windll
# attribute at all.  (The pod itself runs lora_trainer_gui.py directly and
# never touches this file, but nothing stops a Linux user double-entry via
# `python launch.pyw`, so it must degrade cleanly.)
LINUXCT = os.path.join(BASE, "linux_ct")
os.makedirs(LINUXCT)
with open(os.path.join(LINUXCT, "ctypes.py"), "w", encoding="utf-8") as f:
    f.write("# imports fine; ctypes.windll does not exist off Windows\n")

# The shadowing trick only works if the interpreter has not already imported
# the real ctypes during startup.  Verify rather than assume.
r = subprocess.run([PY, "-c", "import sys; print('ctypes' in sys.modules)"],
                   capture_output=True, text=True)
ck("interpreter does not preload ctypes (shadowing is sound)",
   r.stdout.strip() == "False", r.stdout.strip())


def run_launcher(gui_src, *, extra_path=(), exe=PY, pre_log=None, timeout=40):
    with open(os.path.join(SB, "lora_trainer_gui.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(gui_src))
    for p in (MSGBOX, MARK):
        if os.path.isfile(p):
            os.remove(p)
    if os.path.isfile(LOGF):
        os.remove(LOGF)
    if pre_log is not None:
        with open(LOGF, "w", encoding="utf-8") as f:
            f.write(pre_log)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(extra_path)
    try:
        return subprocess.run([exe, LAUNCH], capture_output=True, text=True,
                              env=env, cwd=SB, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None  # a real dialog appeared, or the launcher hung — both are failures


def box():
    if os.path.isfile(MSGBOX):
        with open(MSGBOX, encoding="utf-8") as f:
            return json.load(f)
    return None


def log_text():
    if os.path.isfile(LOGF):
        with open(LOGF, encoding="utf-8") as f:
            return f.read()
    return ""


# The launcher imports the GUI as a module and calls its main() — the splash needs the root
# before the module exists (#115) — so a fake GUI must offer one.
HAPPY = (f"with open({MARK!r}, 'w') as f: f.write('started')\n"
         "def main(root=None, splash=None): pass\n")

# --- 1. the happy path is untouched -------------------------------------------------------------
r = run_launcher(HAPPY, extra_path=[FAKECT])
ck("happy path: exit 0", r is not None and r.returncode == 0,
   None if r is None else (r.returncode, r.stderr[-200:]))
ck("  GUI actually ran", os.path.isfile(MARK))
ck("  no error log", not os.path.isfile(LOGF))
ck("  no message box", box() is None)

r = run_launcher(HAPPY, extra_path=[FAKECT], pre_log="old failure")
ck("a stale error log is removed on a clean start",
   r is not None and r.returncode == 0 and not os.path.isfile(LOGF))

r = run_launcher(HAPPY, extra_path=[FAKECT], exe=PYW)
ck("happy path under pythonw (the real deployment)",
   r is not None and r.returncode == 0 and os.path.isfile(MARK))

# --- 2. missing Tkinter: the reported case ------------------------------------------------------
r = run_launcher(HAPPY, extra_path=[FAKECT, POISON])
b = box()
ck("missing tkinter: exit 1", r is not None and r.returncode == 1,
   None if r is None else r.returncode)
ck("  GUI was never started", not os.path.isfile(MARK))
ck("  message box appeared", b is not None)
ck("  box names Tkinter", b is not None and "Tkinter" in b["title"], b and b["title"])
ck("  box says how to fix it", b is not None and "tcl/tk and IDLE" in b["text"])
ck("  box points at the log", b is not None and "launch_error.log" in b["text"])
ck("  log written with the cause", "Tkinter" in log_text() and "simulated" in log_text())

r = run_launcher(HAPPY, extra_path=[FAKECT, POISON], exe=PYW)
ck("missing tkinter under pythonw still reports (exit 1 + box)",
   r is not None and r.returncode == 1 and box() is not None)

# --- 3. any other startup crash -----------------------------------------------------------------
r = run_launcher("raise RuntimeError('boom at startup')", extra_path=[FAKECT])
b = box()
ck("GUI crash: exit 1", r is not None and r.returncode == 1)
ck("  message box appeared", b is not None)
ck("  box carries the exception", b is not None and "RuntimeError: boom at startup" in b["text"])
ck("  log holds the full traceback",
   "Traceback" in log_text() and "boom at startup" in log_text())

# --- 4. deliberate exits are not errors ---------------------------------------------------------
r = run_launcher("import sys; sys.exit(7)", extra_path=[FAKECT])
ck("SystemExit passes through untouched (code 7, no box, no log)",
   r is not None and r.returncode == 7 and box() is None and not os.path.isfile(LOGF),
   None if r is None else r.returncode)

r = run_launcher("raise KeyboardInterrupt", extra_path=[FAKECT])
ck("KeyboardInterrupt is not dressed up as a crash (no box, no log)",
   r is not None and r.returncode != 0 and box() is None and not os.path.isfile(LOGF))

# --- 5. the reporter itself must never die ------------------------------------------------------
os.mkdir(LOGF)  # a directory at the log path makes open(..., 'w') fail
try:
    r = run_launcher("raise RuntimeError('boom')", extra_path=[FAKECT], pre_log=None)
    b = box()
    ck("unwritable log: box still shown, clean exit 1",
       r is not None and r.returncode == 1 and b is not None)
    ck("  box does not promise a log that does not exist",
       b is not None and "Details saved" not in b["text"])
finally:
    os.rmdir(LOGF)

r = run_launcher("raise RuntimeError('boom')", extra_path=[BROKENCT])
ck("no ctypes at all: falls back to stderr + log, exit 1",
   r is not None and r.returncode == 1 and "could not start" in r.stderr
   and "boom" in log_text(), None if r is None else r.stderr[-150:])

r = run_launcher("raise RuntimeError('boom')", extra_path=[LINUXCT])
ck("Linux shape (ctypes imports, no windll): stderr + log, exit 1",
   r is not None and r.returncode == 1 and "could not start" in r.stderr
   and "boom" in log_text(), None if r is None else r.stderr[-150:])

# --- 6. the venv re-launch hop ------------------------------------------------------------------
SB2 = os.path.join(BASE, "app2")
os.makedirs(SB2)
shutil.copy(os.path.join(REPO, "launch.pyw"), os.path.join(SB2, "launch.pyw"))
shutil.copy(os.path.join(REPO, "fizgig_splash.py"), os.path.join(SB2, "fizgig_splash.py"))
MARK2 = os.path.join(SB2, "started.txt")
with open(os.path.join(SB2, "lora_trainer_gui.py"), "w", encoding="utf-8") as f:
    f.write(f"with open({MARK2!r}, 'w') as fh: fh.write('started')\n")
r = subprocess.run([PY, "-m", "venv", "--without-pip", os.path.join(SB2, "venv")],
                   capture_output=True, text=True, timeout=120)
if r.returncode == 0:
    r = subprocess.run([PY, os.path.join(SB2, "launch.pyw")], capture_output=True,
                       text=True, cwd=SB2, timeout=40)
    ck("outer python hands off to the venv and exits 0", r.returncode == 0, r.stderr[-200:])
    end = time.time() + 25
    while time.time() < end and not os.path.isfile(MARK2):
        time.sleep(0.3)
    ck("  the venv child actually started the GUI", os.path.isfile(MARK2))
else:
    ck("venv creation for the re-launch test", False, r.stderr[-200:])

# --- 7. static guards ---------------------------------------------------------------------------
import py_compile
try:
    py_compile.compile(os.path.join(REPO, "launch.pyw"), doraise=True)
    ck("launch.pyw compiles", True)
except py_compile.PyCompileError as e:
    ck("launch.pyw compiles", False, e)

src = open(os.path.join(REPO, "launch.pyw"), encoding="utf-8").read()
ck("stderr fallback is guarded (stderr is None under pythonw)",
   'getattr(sys, "stderr", None)' in src)
ck("deliberate exits re-raised before the catch-all",
   "except (SystemExit, KeyboardInterrupt):" in src)
ck("the reporter never imports tkinter", "tkinter" not in src.split("def _report")[1].split("\n\n")[0])

shutil.rmtree(BASE, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
