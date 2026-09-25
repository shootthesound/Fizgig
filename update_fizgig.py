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

# A torch that reports a ROCm/HIP build -> exit 1. Run in a child so this process never imports torch.
_IS_ROCM = ("import torch; v=getattr(torch,'__version__','') or ''; r=getattr(getattr(torch,'version',None),'rocm',None); "
            "h=getattr(getattr(torch,'version',None),'hip',None); raise SystemExit(1 if (r or h or '+rocm' in v.lower()) else 0)")


def run(cmd, **kw):
    return subprocess.call(cmd, cwd=HERE, **kw)


def quiet(cmd):
    return subprocess.call(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fail(msg):
    print()
    print("ERROR: " + msg)
    print("Aborting update.")
    sys.exit(1)


def ensure_uv():
    if quiet([PY, "-m", "uv", "--version"]) != 0:
        if run([PY, "-m", "pip", "install", "--upgrade", "uv"]) != 0:
            fail("Failed to install the uv package.")


def wrong_updater(this_is_rocm_updater):
    """True when the venv's torch belongs to the other GPU vendor (that updater would break it)."""
    venv_is_rocm = quiet([PY, "-c", _IS_ROCM]) == 1
    return venv_is_rocm != this_is_rocm_updater


def cuda_deps():
    if wrong_updater(False):
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
    if wrong_updater(True):
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


def main():
    rocm = "--rocm" in sys.argv[1:]
    print("Updating Fizgig (AMD ROCm)..." if rocm else "Updating Fizgig...")
    # Older installers overwrote the tracked launcher; restore it so it never blocks the pull.
    quiet(["git", "checkout", "--", "run_fizgig_rocm.bat" if rocm else "run_fizgig.bat"])
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
