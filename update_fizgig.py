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
        print("  update_fizgig.bat installs CUDA packages from requirements.txt and would")
        print("  overwrite your ROCm PyTorch / bitsandbytes stack.")
        print()
        print("  Use instead:  update_fizgig_rocm.bat")
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
    env_bat = os.path.join(HERE, "rocm_env.bat")
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


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)     # keep our lines in order with the child processes' output in a log
    except (AttributeError, ValueError):
        pass
    rocm = "--rocm" in sys.argv[1:]
    print("Updating Fizgig (AMD ROCm)..." if rocm else "Updating Fizgig...")
    clear_old_stage2()
    # Older installers overwrote the tracked launchers; restore them so they never block the pull.
    for launcher in (("run_fizgig_rocm.bat", "run_fizgig.bat") if rocm else ("run_fizgig.bat",)):
        quiet(["git", "checkout", "--", launcher])
    run(["git", "pull"])
    print()
    print("Installing/updating dependencies...")
    if not os.path.exists(os.path.join(HERE, "requirements.txt")):
        print("WARNING: requirements.txt not found. Skipping dependency installation.")
        print()
    else:
        rocm_deps() if rocm else cuda_deps()
    # Community LoRAs + H3 training adapters. Idempotent, and a failure (offline) never aborts the update.
    run([PY, os.path.join("src", "fizgig", "scripts", "fetch_turbo_lora.py")])
    if not rocm:
        msvc_check()
    print()
    print("Update complete! Launch with run_fizgig_rocm.bat" if rocm else "Update complete! Run Fizgig with run_fizgig.bat")


if __name__ == "__main__":
    main()
