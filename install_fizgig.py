#!/usr/bin/env python3
"""
Fizgig Installer
================
Sets up Fizgig with its own Python virtual environment.
Installs PyTorch built against CUDA 12.8 — required for Klein 9B and Krea 2 training,
preview rendering, profiling, extraction, and Repair Studio on NVIDIA GPUs.

Features:
- Creates isolated venv for Fizgig dependencies
- Installs CUDA 12.8 PyTorch (supports RTX 30xx / 40xx / 50xx Blackwell)
- Installs InsightFace face detection (runs on CPU for GPU independence)
- Downloads face detection models automatically
- Installs Florence-2 AI captioning (transformers library; runs on GPU)
- Creates launcher scripts for Windows/Linux/Mac
- Verifies CUDA is visible to PyTorch after install

Note: Florence-2 models are auto-downloaded from Hugging Face on first use (~500MB-1.5GB)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 10)

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / "venv"
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements.txt"


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 40)


def check_python_version():
    """Check if Python version meets minimum requirements"""
    version = sys.version_info[:2]
    if version < MIN_PYTHON_VERSION:
        print(f"Error: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required.")
        print(f"Current version: Python {version[0]}.{version[1]}")
        return False
    print(f"Python version: {version[0]}.{version[1]} (OK)")
    return True


def conda_reason():
    """Why this Python must not build the venv, or None. Conda ships its own Tk that cannot
    see the system's fonts, so a venv made from it draws the GUI with missing text (#141)."""
    env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")
    if env:
        return f"a conda environment is active ({env})"
    if any(tag in sys.version.lower() for tag in ("conda", "anaconda", "continuum")):
        return "this Python was built by conda"
    for prefix in (getattr(sys, "base_prefix", ""), sys.prefix):
        if prefix and os.path.isdir(os.path.join(prefix, "conda-meta")):
            return "this Python was built by conda"
    return None


def check_not_conda():
    """Refuse to build the venv from a conda Python (override: FIZGIG_ALLOW_CONDA=1)."""
    if os.environ.get("FIZGIG_ALLOW_CONDA"):
        print("FIZGIG_ALLOW_CONDA set — continuing inside conda (the GUI's fonts may render wrong).")
        return True
    reason = conda_reason()
    if not reason:
        print("Not a conda Python (OK)")
        return True
    print(f"Error: {reason}.")
    print("Fizgig has to be installed from your system Python, not conda: conda bundles its own")
    print("Tk, which cannot see the system's fonts, so the app comes up with missing or")
    print("substituted text. Leave conda, then run the installer again:")
    print("    conda deactivate")
    print("    python3 install_fizgig.py      (Windows: py install_fizgig.py)")
    print("If 'python3' is still conda's after that (conda on your PATH for good), run the")
    print("system interpreter by its full path instead, e.g. /usr/bin/python3 install_fizgig.py")
    print("(Windows: a python.org install, then py install_fizgig.py).")
    print("To go ahead inside conda anyway, set FIZGIG_ALLOW_CONDA=1.")
    return False


def _is_conda_home(home):
    """Was a venv's pyvenv.cfg `home` a conda Python? Off under FIZGIG_ALLOW_CONDA, or the
    venv the override just created would be condemned by the health check straight after."""
    if os.environ.get("FIZGIG_ALLOW_CONDA"):
        return False
    h = Path(home)
    for p in (h, h.parent):
        if (p / "conda-meta").is_dir():
            return True
    # Whole path COMPONENTS only: a parent folder that merely contains the word (a user
    # called conda-dev, a project dir named conda_tools) must not condemn a plain venv.
    for part in (p.lower() for p in h.parts):
        if part in ("conda", "anaconda", "miniconda", "miniforge", "mambaforge") or any(
                part.startswith(tag) and part[len(tag):].isdigit()
                for tag in ("anaconda", "miniconda", "miniforge", "mambaforge")):
            return True
    return False


def venv_health(venv_dir=None):
    """Why an existing venv can't be trusted — None when it's fine, else (kind, reason).

    kind 'broken' means reuse cannot work (missing interpreter, orphaned base, won't
    boot) — declining recreation has to abort. kind 'mismatch' means the venv boots
    fine but was built from a different Python than the one running this installer —
    the shape that produced #111, where a venv from an original 3.10 install was
    retried against 3.10-3.13, reused every time because the recreate prompt defaults
    to No, and a leftover 32-bit Python 3.7 then booted inside uv's interpreter probe
    and died at 'No module named encodings'. Recreation is the safe default there, but
    a mismatched venv that has been working is allowed to keep working (review round:
    every dependency install runs through the venv's OWN python, so "reuse would fail"
    is not a promise this function can make) — which is why the boot probe runs FIRST
    and outranks the version string.

    The probe uses '-I' (isolated mode) deliberately: that is how uv queries the
    interpreter, and #111's venv passed a plain spawn while failing exactly that."""
    venv_dir = Path(venv_dir) if venv_dir else VENV_DIR
    python_path = venv_dir / ("Scripts/python.exe" if platform.system() == "Windows"
                              else "bin/python")
    if not python_path.exists():
        return ("broken", "its Python executable is missing")

    home = version = None
    cfg = venv_dir / "pyvenv.cfg"
    if cfg.exists():
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                key, _, value = line.partition("=")
                key, value = key.strip().lower(), value.strip()
                if key == "home":
                    home = value
                elif key in ("version", "version_info"):
                    version = value
        except OSError:
            pass
    if home and not Path(home).exists():
        # Only the whole home directory vanishing counts — probing for a specific
        # executable name inside it would false-positive on Unix layouts where the
        # binary is python3, and a wrongly-condemned venv is worse than a missed one.
        return ("broken", f"it was built from a Python that is no longer installed ({home})")
    if home and _is_conda_home(home):
        return ("broken", f"it was built from a conda Python ({home}) — conda's Tk cannot see "
                          "the system's fonts, so Fizgig needs a venv from the system Python")

    try:
        result = subprocess.run(
            [str(python_path), "-I", "-c", "import encodings"],
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return ("broken", "its Python cannot start in isolated mode — the way the "
                              "package installer runs it — usually because another "
                              "Python installation is interfering")
    except (OSError, subprocess.TimeoutExpired):
        return ("broken", "its Python cannot be started at all")

    if version:
        parts = version.split(".")
        # Digits-only guard: a hand-mangled cfg ('version = ..') must not condemn a
        # venv the probe just proved bootable with a nonsense comparison.
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            built = f"{parts[0]}.{parts[1]}"
            running = f"{sys.version_info[0]}.{sys.version_info[1]}"
            if built != running:
                return ("mismatch", f"it was built from Python {built}, but this "
                                    f"installer is running Python {running}")
    return None


def _remove_venv(venv_dir=None):
    """rmtree with a plain-language failure instead of a traceback — on Windows a
    locked file (Fizgig still open, a terminal with the venv activated, an antivirus
    scan) is the common way this dies, and it can leave a half-deleted venv behind."""
    venv_dir = Path(venv_dir) if venv_dir else VENV_DIR
    try:
        shutil.rmtree(venv_dir)
        return True
    except OSError as e:
        print(f"Could not remove the venv: {e}")
        print("Close Fizgig and any terminal windows using this venv, then run the "
              "installer again. (A partially removed venv is fine — the next run "
              "detects it and finishes the job.)")
        return False


def create_venv():
    """Create virtual environment"""
    if VENV_DIR.exists():
        print(f"Virtual environment already exists at: {VENV_DIR}")
        health = venv_health()
        if health:
            # An untrusted venv must not be the default choice — the y/N prompt below
            # kept #111's venv alive through every retry the user made.
            kind, reason = health
            print(f"But {reason}.")
            if kind == "broken":
                print("Reusing it cannot work, so it needs recreating.")
                response = input("Delete and recreate it now? (Y/n): ").strip().lower()
                if response in ("n", "no"):
                    print("Cannot continue with this venv. Delete the folder manually "
                          "(or answer Y here) and run the installer again.")
                    return False
            else:
                # mismatch: it boots, and dependency installs run through the venv's
                # own python — so it MAY keep working. Recreating is still the safe
                # default (a version-straddling venv is exactly the #111 trap), but
                # the cost is honest and declining continues rather than aborts.
                print("Recreating it is the safe choice — a venv reused across Python "
                      "versions is a common source of confusing install failures — but "
                      "it does mean re-downloading all packages (several GB). If this "
                      "venv has been working for you, answering n keeps it.")
                response = input("Delete and recreate it now? (Y/n): ").strip().lower()
                if response in ("n", "no"):
                    print("Keeping the existing venv.")
                    return True
            print("Removing venv...")
            if not _remove_venv():
                return False
        else:
            response = input("Delete and recreate? (y/N): ").strip().lower()
            if response == 'y':
                print("Removing existing venv...")
                if not _remove_venv():
                    return False
            else:
                print("Using existing venv.")
                return True

    print(f"Creating virtual environment at: {VENV_DIR}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating venv: {e}")
        return False
    health = venv_health()
    if health and health[0] == "broken":
        # A fresh venv can only be 'broken' here (its cfg version matches by
        # construction) — but guard on kind anyway so a future check can't turn
        # a working new venv into a hard stop.
        print(f"The new virtual environment was created, but {health[1]}.")
        if platform.system() == "Windows":
            print("This usually means a stray older Python is getting in the way — "
                  "check for old installations (especially 32-bit ones, e.g. in "
                  "'C:\\Program Files (x86)') and uninstall them or remove them from "
                  "PATH, then run this installer again from a new terminal.")
        else:
            print("This usually means another Python installation is getting in the "
                  "way — check PATH for stray or partial Python installs, then run "
                  "this installer again from a new shell.")
        return False
    print("Virtual environment created successfully.")
    return True


def get_pip_path():
    """Get the path to pip in the venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_python_path():
    """Get the path to python in the venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_dependencies():
    """Install dependencies from requirements.txt"""
    python_path = get_python_path()

    if not python_path.exists():
        print(f"Error: python not found at {python_path}")
        return False

    print("Upgrading pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("Installing uv...")
    # shell=False (list form) with internal Path constants — not injectable
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "uv"], check=True)

    print(f"Installing dependencies from: {REQUIREMENTS_FILE} (using uv)")
    print("(This may take a few minutes for PyTorch download...)")

    from uv_install_deps import install_requirements
    if install_requirements(REQUIREMENTS_FILE, VENV_DIR, python_path):
        print("Dependencies installed successfully.")
        return True
    return False


def verify_cuda():
    """Probe torch.cuda after install so users learn about driver mismatches
    at install time rather than mid-training."""
    python_path = get_python_path()
    probe_script = '''
import sys
try:
    import torch
except Exception as e:
    print(f"FAIL: torch import failed: {e}")
    sys.exit(2)

if torch.cuda.is_available():
    try:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"OK  CUDA visible: {name} ({vram_gb:.1f} GB VRAM)")
        print(f"    PyTorch {torch.__version__}  |  CUDA runtime {torch.version.cuda}")
        sys.exit(0)
    except Exception as e:
        print(f"WARN CUDA reported available but device query failed: {e}")
        sys.exit(1)
else:
    print("WARN PyTorch installed but torch.cuda.is_available() is False.")
    print("     Fizgig training (Klein 9B / Krea 2) needs a CUDA-capable GPU.")
    print("     If you have one, update your NVIDIA driver to 555+ and re-run this installer.")
    sys.exit(1)
'''
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe_script],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        return result.returncode == 0
    except Exception as e:
        print(f"Note: CUDA probe skipped: {e}")
        return True  # Don't fail installation


def download_insightface_models():
    """Trigger InsightFace model download by running a test detection"""
    python_path = get_python_path()

    print("Downloading InsightFace models (buffalo_l)...")
    print("(First-time download, ~300MB)")

    # Python script to download models
    download_script = '''
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from insightface.app import FaceAnalysis

    print("Initializing FaceAnalysis (this downloads models)...")
    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'genderage', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1)  # -1 = CPU only
    print("Models downloaded successfully!")
    sys.exit(0)
except Exception as e:
    print(f"Warning: Could not download models: {e}")
    print("Models will be downloaded on first use.")
    sys.exit(0)  # Don't fail installation
'''

    try:
        result = subprocess.run(
            [str(python_path), "-c", download_script],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            # Filter out common warnings
            for line in result.stderr.split('\n'):
                if line and 'WARNING' not in line and 'deprecated' not in line.lower():
                    print(line)
        return True
    except Exception as e:
        print(f"Note: Model pre-download skipped: {e}")
        print("Models will download automatically on first use.")
        return True  # Don't fail installation


def create_launcher_scripts():
    """Verify/create launcher scripts.

    Windows: the launchers ship WITH the repo (Launch Fizgig.bat, the app window; Launch Fizgig in browser.bat, the web UI
    with the browser opened on it; Launch Gizmo.bat). This step used to overwrite it with an ancient console-attached
    version (python + pause), which (a) gave every fresh install a lingering console window
    that ended in 'Press any key to continue', and (b) dirtied a tracked file so the next
    update_fizgig.bat's `git pull` refused to run. Never write it here.
    """
    # Gizmo's launcher is the same story and gets the same treatment: tracked, verified, never
    # written. It needs no install of its own — Tkinter is stdlib, PIL and imageio-ffmpeg are
    # already pinned — so it simply arrives with an ordinary update.
    for name in ("Launch Fizgig.bat", "Launch Fizgig in browser.bat", "Launch Gizmo.bat"):
        bat_path = SCRIPT_DIR / name
        if bat_path.exists():
            print(f"Launcher present: {bat_path} (ships with the repo — not modified)")
        else:
            print(f"WARNING: {bat_path} is missing — restore it with "
                  f'`git checkout -- "{name}"`')

    # Linux/Mac: launch_fizgig.sh, launch_fizgig_browser.sh and launch_gizmo.sh ship with the repo too
    for name in ("launch_fizgig.sh", "launch_fizgig_browser.sh", "launch_gizmo.sh"):
        sh_path = SCRIPT_DIR / name
        if sh_path.exists() and platform.system() != "Windows":
            os.chmod(sh_path, 0o755)
    return True


def print_summary():
    """Print installation summary and next steps"""
    print_header("Installation Complete!")

    print("\nFizgig has been installed successfully.")
    print("\nTo launch Fizgig:")

    if platform.system() == "Windows":
        print(f"  Double-click: {SCRIPT_DIR / 'Launch Fizgig.bat'}  (its own window)")
        print(f"  or {SCRIPT_DIR / 'Launch Fizgig in browser.bat'}  (in your web browser)")
    else:
        print("  Run: ./launch_fizgig.sh  (its own window)  or  ./launch_fizgig_browser.sh  (in your web browser)")

    print("\nFace Detection:")
    print("  - InsightFace models will download on first use (~300MB)")
    print("  - Uses CPU for detection (works on any GPU)")
    print("  - Supports gender classification for face filtering")
    print("  - Furry / anime: pick a YOLO face model on the Image Prep tab (docs/FACE_BACKENDS.md)")

    print("\nAI Captioning (Florence-2):")
    print("  - Florence models auto-download from Hugging Face on first use")
    print("  - Model sizes: base (~500MB), large (~1.5GB)")
    print("  - Runs on GPU via CUDA 12.8 PyTorch (fast)")

    print("\n" + "=" * 60)


def check_msvc_build_tools():
    """Windows: detect the MSVC C++ Build Tools torch.compile's inductor needs.

    triton installs via requirements.txt, but inductor also compiles host-side C++ —
    without MSVC, Compile Blocks silently runs eager (a console note at training time).
    The MS download site is ambiguous, so print the EXACT installer link and the one
    workload to tick. Informational only; training works fine without."""
    if os.name != "nt":
        print("  Not Windows — torch.compile uses the system toolchain, nothing to check.")
        return
    import glob as _glob
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    found = False
    if os.path.exists(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=30)
            found = bool(out.stdout.strip())
        except Exception:
            pass
    if not found:
        # Fallback sweep for cl.exe in the standard install roots (covers odd setups).
        for root in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if _glob.glob(os.path.join(root, "*", "*", "VC", "Tools", "MSVC", "*",
                                       "bin", "Hostx64", "x64", "cl.exe")):
                found = True
                break
    if found:
        print("  MSVC C++ Build Tools found — torch.compile speedups are available.")
        return
    print("  OPTIONAL: MSVC C++ Build Tools not detected.")
    print("  The torch.compile training speedup needs them; everything else works without.")
    print()
    print("  Direct download (exact installer, no hunting on the MS site):")
    print("    https://aka.ms/vs/17/release/vs_BuildTools.exe")
    print('  During install, tick the "Desktop development with C++" workload.')
    print()
    print("  Or install unattended from a terminal:")
    print('    winget install Microsoft.VisualStudio.2022.BuildTools --override '
          '"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"')
    print()
    print("  Install it any time — no need to re-run this installer. The next training run")
    print("  detects it automatically.")


def _windows_has_amd_radeon() -> bool:
    """Best-effort AMD detect for the hand-off message (PowerShell CIM)."""
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            text = result.stdout or ""
            return "AMD" in text and "Radeon" in text
    except Exception:
        pass
    return False


def redirect_amd_to_rocm_installer():
    """AMD support is additive: leave the CUDA path untouched and hand off early."""
    if platform.system() == "Linux":
        has_nvidia = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
        if os.path.exists("/dev/kfd") and not has_nvidia:
            print("AMD ROCm detected. This installer is NVIDIA CUDA only.")
            print("Use the separate AMD installer (Linux AMD is highly experimental):")
            print(f"  chmod +x {SCRIPT_DIR / 'install_fizgig_rocm.sh'}")
            print(f"  {SCRIPT_DIR / 'install_fizgig_rocm.sh'}")
            sys.exit(1)
        return

    if platform.system() == "Windows":
        if shutil.which("nvidia-smi"):
            return
        if _windows_has_amd_radeon():
            print("AMD GPU detected. This installer is NVIDIA CUDA only.")
            print("Use the separate AMD installer:")
            print(f"  {SCRIPT_DIR / 'install_fizgig_rocm.bat'}")
            sys.exit(1)


def main():
    print_header("Fizgig Installer — Klein 9B & Krea 2 LoRA Workbench")
    print(f"Installation directory: {SCRIPT_DIR}")

    redirect_amd_to_rocm_installer()

    # Step 1: Check Python version, and that it is not conda's (#141)
    print_step(1, "Checking Python")
    if not check_python_version() or not check_not_conda():
        sys.exit(1)

    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_venv():
        sys.exit(1)

    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        sys.exit(1)

    # Step 4: Verify CUDA is visible to PyTorch
    print_step(4, "Verifying CUDA availability")
    verify_cuda()  # Don't fail on this — user may be on a CPU-only machine for prep only

    # Step 5: Download InsightFace models
    print_step(5, "Downloading face detection models")
    download_insightface_models()  # Don't fail on this

    # Step 6: Create launcher scripts
    print_step(6, "Creating launcher scripts")
    if not create_launcher_scripts():
        sys.exit(1)

    # Step 7: MSVC C++ Build Tools check (torch.compile's inductor backend needs them on
    # Windows; triton itself comes from requirements.txt). Informational — never fails.
    print_step(7, "Checking for MSVC C++ Build Tools (torch.compile)")
    check_msvc_build_tools()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
