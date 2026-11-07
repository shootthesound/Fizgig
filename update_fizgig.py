"""Fizgig updater (Windows). Run through update_fizgig.bat (NVIDIA) or update_fizgig_rocm.bat (AMD ROCm, --rocm).

All update logic lives here rather than in the .bat files. cmd reads a running .bat by byte offset, so a git
pull that rewrites the .bat mid-run used to resume at a garbage position; the old fix copied the .bat to %TEMP%
and re-ran the copy, which antivirus heuristics flag as dropper behaviour (#157). Python compiles this whole file
before running it, so the pull below can rewrite anything, this file included, without affecting the running
update. The .bat launchers stay a few static lines.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LINUX = sys.platform.startswith("linux")
UPDATER = "./update_fizgig.sh" if LINUX else "update_fizgig.bat"
ROCM_UPDATER = "./install_fizgig_rocm.sh (run it again)" if LINUX else "update_fizgig_rocm.bat"

# Which torch the venv has, probed in a child so this process never imports torch: exit 10 = a ROCm/HIP build,
# 11 = anything else (CUDA/CPU), 12 = torch missing or broken. Distinct codes so a failed import is never mistaken
# for either vendor (the old .bat read "exit 1" as ROCm, so a broken torch sent people to the wrong updater).
TORCH_ROCM, TORCH_OTHER, TORCH_BROKEN = 10, 11, 12
_TORCH_KIND = ("import sys\ntry:\n    import torch\nexcept Exception:\n    sys.exit(12)\n"
               "v=getattr(torch,'__version__','') or ''; r=getattr(getattr(torch,'version',None),'rocm',None); "
               "h=getattr(getattr(torch,'version',None),'hip',None); sys.exit(10 if (r or h or '+rocm' in v.lower()) else 11)")


def run(cmd, **kw):
    try:
        return subprocess.call(cmd, cwd=HERE, **kw)
    except OSError as e:            # e.g. git not on PATH: say so and carry on, as the old .bat did
        print(f"'{cmd[0]}' could not be run: {e}")
        return 127


def quiet(cmd):
    try:
        return subprocess.call(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return 127


def fail(msg):
    print()
    print("ERROR: " + msg)
    print("Aborting update.")
    sys.exit(1)


def ensure_uv():
    if quiet([PY, "-m", "uv", "--version"]) != 0:
        if run([PY, "-m", "pip", "install", "--upgrade", "uv"]) != 0:
            fail("Failed to install the uv package.")


def torch_kind():
    rc = quiet([PY, "-c", _TORCH_KIND])
    return rc if rc in (TORCH_ROCM, TORCH_OTHER) else TORCH_BROKEN


def wrong_updater(this_is_rocm_updater, kind=None):
    """True when the venv's torch belongs to the other GPU vendor (that updater would break it).
    A missing/broken torch is not "the other vendor": see cuda_deps / rocm_deps."""
    kind = torch_kind() if kind is None else kind
    if kind == TORCH_BROKEN:
        return False
    return (kind == TORCH_ROCM) != this_is_rocm_updater


def cuda_deps():
    kind = torch_kind()      # one torch import per update
    if kind == TORCH_BROKEN:
        # Nothing to protect: the CUDA requirements install below puts a working CUDA torch back.
        print("NOTE: PyTorch could not be imported in this venv - reinstalling it from requirements.txt.")
    elif wrong_updater(False, kind):
        print()
        print("ERROR: This looks like an AMD ROCm install.")
        print()
        print(f"  {UPDATER} installs CUDA packages from requirements.txt and would")
        print("  overwrite your ROCm PyTorch / bitsandbytes stack.")
        print()
        print(f"  Use instead:  {ROCM_UPDATER}")
        sys.exit(1)
    if sys.version_info < (3, 11):
        # PyTorch 2.15 has no Python 3.10 wheels, so the install below can't resolve: say what to do instead.
        print()
        print(f"ERROR: Fizgig now needs Python 3.11 or newer; this install's Python is {sys.version_info[0]}."
              f"{sys.version_info[1]}.")
        print()
        print("  1. Install Python 3.12 from https://www.python.org/downloads/")
        print("     (on Windows, tick \"Add Python to PATH\").")
        print("  2. Run install_fizgig.bat (Linux: python3.12 install_fizgig.py). It rebuilds Fizgig's")
        print("     Python environment on 3.12. Your models, datasets, LoRAs and settings are kept.")
        sys.exit(1)
    ensure_uv()
    if run([PY, "uv_install_deps.py", "requirements.txt", "venv"]) != 0:
        fail("Failed to install/update dependencies using uv. See output above.")


def rocm_deps():
    kind = torch_kind()
    if kind == TORCH_BROKEN:
        print()
        print("ERROR: PyTorch could not be imported in this venv.")
        print()
        print("  The ROCm updater only refreshes the shared packages and keeps your ROCm PyTorch,")
        print("  so it cannot repair a missing or broken one. Re-run install_fizgig_rocm.bat.")
        print("  (On an NVIDIA card, use update_fizgig.bat instead.)")
        sys.exit(1)
    if wrong_updater(True, kind):
        print()
        print("ERROR: This looks like an NVIDIA / CUDA install.")
        print()
        print("  update_fizgig_rocm.bat installs the AMD ROCm bitsandbytes wheel and would")
        print("  overwrite your CUDA PyTorch / bitsandbytes stack.")
        print()
        print("  Use instead:  update_fizgig.bat")
        sys.exit(1)
    ensure_uv()
    # Shared deps, the same path as install_fizgig_rocm.bat (not uv_install_deps.py, which is CUDA).
    fd, reqs = tempfile.mkstemp(prefix="fizgig_rocm_shared_reqs_", suffix=".txt")
    os.close(fd)
    try:
        if run([PY, "filter_requirements_rocm.py", "requirements.txt", reqs]) != 0:
            fail("Failed to build ROCm-safe requirements. ROCm torch left untouched.")
        print("Installing shared dependencies (CUDA torch/bnb lines stripped)...")
        # hqq (4-bit HQQ base) builds an optional CUDA kernel from its sdist unless told not to.
        env = dict(os.environ, DISABLE_CUDA="1")
        if run([PY, "-m", "uv", "pip", "install", "--index-strategy", "unsafe-best-match", "-r", reqs], env=env) != 0:
            fail("Failed to install shared dependencies.")
    finally:
        try:
            os.remove(reqs)
        except OSError:
            pass
    # bitsandbytes: the one pin, read from install_fizgig_rocm.bat (never duplicated here).
    wheel = None
    try:
        text = open(os.path.join(HERE, "install_fizgig_rocm.bat"), encoding="utf-8", errors="replace").read()
        m = re.search(r'BNB_WHEEL=([^"\r\n]+)', text, re.IGNORECASE)
        wheel = m.group(1).strip() if m else None
    except OSError:
        pass
    if not wheel:
        print("WARNING: BNB_WHEEL not found in install_fizgig_rocm.bat - skipping bitsandbytes.")
    else:
        print("Syncing bitsandbytes from installer pin:")
        print("  " + wheel)
        if run([PY, "-m", "uv", "pip", "install", wheel]) != 0:
            fail("Failed to install the bitsandbytes wheel.")
    # Refresh the launcher env. Stay --experimental unless rocm_env.bat pins BNB_ROCM_VERSION.
    experimental = True
    env_bat = os.path.join(HERE, "user_data", "rocm_env.bat")
    if os.path.exists(env_bat):
        experimental = "bnb_rocm_version=" not in open(env_bat, encoding="utf-8", errors="replace").read().lower()
    if experimental:
        print("Refreshing rocm_env.bat (experimental: BNB_ROCM_VERSION unset)...")
        run([PY, "write_rocm_env.py", "--experimental"])
    else:
        print("Refreshing rocm_env.bat...")
        run([PY, "write_rocm_env.py"])


def msvc_check():
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    found = False
    if os.path.exists(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                                  "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                                 capture_output=True, text=True).stdout
            found = bool(out.strip())
        except OSError:
            pass
    if not found:
        print()
        print("------------------------------------------------------------------------")
        print(" OPTIONAL: MSVC C++ Build Tools not detected.")
        print()
        print(" The torch.compile training speedup needs them. Training works fine")
        print(" without - you just skip the compile speedup.")
        print()
        print(" Direct download (exact installer, no hunting on the MS site):")
        print("   https://aka.ms/vs/17/release/vs_BuildTools.exe")
        print(' During install, tick the "Desktop development with C++" workload.')
        print()
        print(" Or install unattended from a terminal:")
        print('   winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"')
        print()
        print(" Install it any time - no need to re-run this update. The next training")
        print(" run detects it automatically.")
        print("------------------------------------------------------------------------")


def clear_old_stage2():
    """The old updaters left a copy of themselves in %TEMP% (the file antivirus flagged); remove it."""
    for name in ("fizgig_update_stage2.bat", "fizgig_update_rocm_stage2.bat"):
        try:
            os.remove(os.path.join(tempfile.gettempdir(), name))
        except OSError:
            pass


# The venv's own health, probed in a child (this process never imports torch): torch imports, sees the GPU, its
# triton pairs with it, and bitsandbytes loads its CUDA library. A half-finished upgrade can leave torch reporting the
# new version while one of the others is broken, so the version alone is not enough.
_HEALTH = r"""
import json, sys
sys.path.insert(0, "src")
r = {}
try:
    import torch
    r["torch"] = torch.__version__.split("+")[0]
    r["gpu"] = bool(torch.cuda.is_available())
except Exception as e:
    r["err"] = f"PyTorch won't import ({type(e).__name__}: {e})"[:200]
else:
    if not r["gpu"]:
        r["err"] = "PyTorch can't see the GPU"
        r["driver"] = True
    else:
        try:
            from fizgig.utils.capabilities import triton_matches_torch
            ok, why = triton_matches_torch()
            if not ok:
                r["err"] = why
        except Exception:
            pass
        if "err" not in r and "--rocm" not in sys.argv:
            try:
                import bitsandbytes.functional as F
                F.quantize_4bit(torch.zeros(64, device="cuda", dtype=torch.float16))
            except Exception as e:
                r["err"] = f"bitsandbytes can't load its CUDA library ({type(e).__name__})"
print(json.dumps(r))
"""
COUNTDOWN_S = 10


def venv_health(rocm=False):
    """{"torch", "gpu", "err"?, "driver"?} from the venv, or {"err": ...} when even the probe fails."""
    import json
    try:
        out = subprocess.run([PY, "-c", _HEALTH] + (["--rocm"] if rocm else []), cwd=HERE, capture_output=True,
                             text=True, timeout=300).stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {"err": "the Python environment didn't answer"}
    except Exception as e:
        return {"err": f"the Python environment didn't answer ({type(e).__name__})"}


def pinned_torch():
    try:
        m = re.search(r"^torch==([\w.]+)", open(os.path.join(HERE, "requirements.txt"), encoding="utf-8").read(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def interactive():
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def read_key(seconds):
    """One key press (lower-case), or None after `seconds`. Windows: any key, no Enter. Elsewhere: the key, then Enter."""
    import time
    end = time.time() + seconds
    if os.name == "nt":
        import msvcrt
        while time.time() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                return " " if ch in ("\r", "\n") else ch.lower()
            time.sleep(0.05)
        return None
    import select
    r, _, _ = select.select([sys.stdin], [], [], seconds)
    if not r:
        return None
    line = sys.stdin.readline()
    return (line.strip()[:1] or " ").lower()


def countdown(prompt, seconds):
    """Show `prompt` counting down; return the key pressed, or None when it runs out."""
    import time
    end = time.time() + seconds
    while True:
        left = int(round(end - time.time()))
        if left <= 0:
            print()
            return None
        sys.stdout.write(f"\r{prompt.format(s=left)}   ")
        sys.stdout.flush()
        k = read_key(1 if os.name == "nt" else min(1, left))
        if k is not None:
            print()
            return k


def confirm_rebuild():
    print()
    print("Rebuild deletes the Python environment (the venv folder) and installs it again from scratch:")
    print("several GB to download. Your models, datasets, LoRAs and settings are not touched.")
    k = countdown("Delete and rebuild?  [Y] yes   [N] no   (no in {s} s)", 30)
    return k == "y"


def rebuild(rocm):
    """Hand over to the installer to delete the venv and install fresh. This process runs on the venv's own python,
    which Windows won't let it delete, so there the installer starts in its own window and this one exits."""
    base = getattr(sys, "_base_executable", None) or PY
    print()
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        if rocm:
            # the ROCm installer sets the venv up itself; give this window a moment to let go of the venv first
            subprocess.Popen(["cmd", "/c", "timeout /t 4 /nobreak >nul & rmdir /s /q venv & install_fizgig_rocm.bat"],
                             cwd=HERE, creationflags=flags)
        else:
            subprocess.Popen([base, "install_fizgig.py", "--rebuild"], cwd=HERE, creationflags=flags)
        print("The rebuild continues in a new window. You can close this one.")
        sys.exit(0)
    if rocm:
        fail("On Linux, rebuild an AMD ROCm install by deleting the venv folder and running ./install_fizgig_rocm.sh.")
    sys.exit(run([base, "install_fizgig.py", "--rebuild"]))


def choose(rocm):
    """Before the update: say what this update does to PyTorch, and whether the venv looks broken; Space (or waiting)
    updates as normal, R rebuilds the venv from scratch. Without a keyboard (pods, scripts, Fizgig's own Update
    button) it goes straight to the normal update."""
    h = venv_health(rocm)
    want = pinned_torch()
    print()
    if h.get("err"):
        print(f"  Your Python environment looks broken: {h['err']}.")
        if h.get("driver"):
            print("  If this started after an update, the NVIDIA driver may be too old: this Fizgig's PyTorch needs a")
            print("  driver for CUDA 13 (580 or newer). A rebuild won't fix a driver.")
        else:
            print("  Rebuilding it (R) is the likely fix.")
    elif want and h.get("torch") and h["torch"] != want and not rocm:
        print(f"  This update upgrades PyTorch ({h['torch']} -> {want}, about 3 GB). It should go smoothly.")
        print("  If Fizgig won't start or train afterwards, run this updater again and press R.")
    if not interactive():
        print("  Updating as normal (no keyboard to ask).")
        return
    print()
    print("  [Space] Update as normal")
    print("  [R]     Rebuild: delete the Python environment and install it fresh")
    print("          (if Fizgig stopped starting or training; models, datasets, LoRAs and settings are kept)")
    k = countdown("Updating as normal in {s} s ...", COUNTDOWN_S)
    if k == "r" and confirm_rebuild():
        rebuild(rocm)


def check_after(rocm):
    """After the update: the same health check. A failure is said plainly, with the rebuild offered on the spot."""
    h = venv_health(rocm)
    if not h.get("err"):
        return
    print()
    print(f"WARNING: after the update, {h['err']}.")
    if h.get("driver"):
        print("The NVIDIA driver may be too old: this Fizgig's PyTorch needs a driver for CUDA 13 (580 or newer).")
        return
    print("Rebuilding the Python environment from scratch usually fixes this.")
    if interactive():
        k = countdown("[R] Rebuild now   [Space] Leave it   (leaving it in {s} s)", 30)
        if k == "r" and confirm_rebuild():
            rebuild(rocm)
    else:
        print(f"To rebuild: run {ROCM_UPDATER if rocm else UPDATER} again and press R.")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)     # keep our lines in order with the child processes' output in a log
    except (AttributeError, ValueError):
        pass
    rocm = "--rocm" in sys.argv[1:]
    if rocm and LINUX:            # the ROCm path below is Windows's (its bitsandbytes wheel, rocm_env.bat)
        fail("On Linux, update an AMD ROCm install by running ./install_fizgig_rocm.sh again.")
    print("Updating Fizgig (AMD ROCm)..." if rocm else "Updating Fizgig...")
    clear_old_stage2()
    # Older installers overwrote the tracked launchers; restore them so they never block the pull.
    for launcher in (("Launch Fizgig AMD.bat", "run_fizgig.bat") if rocm else ("run_fizgig.bat",)):
        quiet(["git", "checkout", "--", launcher])
    run(["git", "pull"])
    choose(rocm)                  # after the pull: requirements.txt now says which PyTorch this version wants
    print()
    print("Installing/updating dependencies...")
    if not os.path.exists(os.path.join(HERE, "requirements.txt")):
        print("WARNING: requirements.txt not found. Skipping dependency installation.")
        print()
    else:
        rocm_deps() if rocm else cuda_deps()
        if LINUX:                                # Qt for the app window (install_fizgig.py's install_linux_window)
            if run([PY, "-m", "uv", "pip", "install", "--python", PY, "pywebview[qt]>=5,<7"]) != 0:
                print("NOTE: Qt for the app window didn't install; Fizgig opens in your browser instead.")
    # Community LoRAs + H3 training adapters. Idempotent, and a failure (offline) never aborts the update.
    run([PY, os.path.join("src", "fizgig", "scripts", "fetch_turbo_lora.py")])
    if not rocm:
        msvc_check()
    check_after(rocm)
    print()
    print("Update complete! Start Fizgig with ./launch_fizgig.sh (its own window) or ./launch_fizgig_browser.sh"
          if LINUX else "Update complete! Start Fizgig with Launch Fizgig AMD.bat" if rocm else
          "Update complete! Start Fizgig with Launch Fizgig.bat (its own window) or Launch Fizgig in browser.bat")


if __name__ == "__main__":
    main()
