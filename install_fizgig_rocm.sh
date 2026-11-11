#!/usr/bin/env bash
# Fizgig ROCm Linux installer — AMD pip wheels (Path B).
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only (driver resets, gfx gaps,
# desktop+compute contention). Use Windows ROCm or NVIDIA Linux for production workloads.
# Detects gfx target, installs PyTorch/ROCm from AMD multi-arch wheels into venv,
# then Fizgig deps from requirements.txt (CUDA torch/bnb lines filtered out).
# Prerequisites: amdgpu driver, /dev/kfd, user in render/video groups; sudo for libnuma-dev / pythonX.Y-dev.
set -euo pipefail

FIZGIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default stable = AMD's stable whl-next (torch 2.14.0+rocm10.1.0 today), same family as Windows.
# Nightly / experimental = floating builds from nightly.repo.amd.com (ROCM_CHANNEL=nightly).
ROCM_INDEX="${ROCM_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
ROCM_NIGHTLY_INDEX="${ROCM_NIGHTLY_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}"
ROCM_CHANNEL="${ROCM_CHANNEL:-stable}"
if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    ROCM_SDK_PIN="${ROCM_SDK_PIN:-10.1.0}"
    TORCH_PIN="${TORCH_PIN:-2.14.0+rocm10.1.0}"
    TORCHVISION_PIN="${TORCHVISION_PIN:-0.29.0a0+rocm10.1.0}"
else
    ROCM_SDK_PIN="${ROCM_SDK_PIN:-}"
    TORCH_PIN="${TORCH_PIN:-}"
    TORCHVISION_PIN="${TORCHVISION_PIN:-}"
fi

# python3 used to create venv — not conda; override with FIZGIG_PYTHON=/usr/bin/python3.12
_fizgig_install_python() {
    if [[ -n "${FIZGIG_PYTHON:-}" ]]; then
        echo "$FIZGIG_PYTHON"
        return 0
    fi
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] || [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "ERROR: Conda environment is active (${CONDA_DEFAULT_ENV:-unknown})." >&2
        echo "       Fizgig builds a local venv from system python3 — deactivate conda first:" >&2
        echo "         conda deactivate" >&2
        exit 1
    fi
    local py_path py
    py_path="$(command -v python3)"
    if [[ "$py_path" == *"/conda/"* ]] || [[ "$py_path" == *"/miniconda"* ]] || [[ "$py_path" == *"/anaconda"* ]]; then
        if [[ -x /usr/bin/python3 ]]; then
            local sys_py
            sys_py="$(readlink -f /usr/bin/python3 2>/dev/null || echo /usr/bin/python3)"
            if [[ "$sys_py" != *conda* ]] && [[ "$sys_py" != *miniconda* ]] && [[ "$sys_py" != *anaconda* ]]; then
                echo "NOTE: python3 on PATH is conda; using /usr/bin/python3 for the Fizgig venv." >&2
                py="/usr/bin/python3"
            else
                echo "ERROR: python3 and /usr/bin/python3 both appear to be conda." >&2
                echo "       Deactivate conda or set FIZGIG_PYTHON to a system interpreter." >&2
                exit 1
            fi
        else
            echo "ERROR: python3 is conda and /usr/bin/python3 was not found." >&2
            exit 1
        fi
    else
        py="python3"
    fi
    echo "$py"
}

_ensure_python_dev_headers() {
    local py="$1"
    local py_mm header
    py_mm="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    header="/usr/include/python${py_mm}/Python.h"
    if [[ -f "$header" ]]; then
        return 0
    fi
    echo "Installing python${py_mm}-dev (headers for $($py --version | cut -d' ' -f2) — Triton/torch.compile needs Python.h)..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y "python${py_mm}-dev" || sudo apt-get install -y python3-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "python${py_mm}-devel" || sudo dnf install -y python3-devel
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y "python${py_mm}-devel" || sudo yum install -y python3-devel
    else
        echo "ERROR: Python.h not found (${header}). Install python${py_mm}-dev, then re-run."
        exit 1
    fi
    if [[ ! -f "$header" ]]; then
        echo "ERROR: Python.h still missing at ${header} after installing python${py_mm}-dev." >&2
        exit 1
    fi
}

_fizgig_rocm_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${FIZGIG_ROOT}/venv/bin/python" ]]; then
        echo "${FIZGIG_ROOT}/venv/bin/python"
    else
        echo python3
    fi
}

# Device kernels come from torch[device-${ARCH}] / torchvision[device-${ARCH}] extras
# on AMD's whl-next indexes (stable default, nightly via ROCM_CHANNEL=nightly).

verify_torch_device_wheel() {
    local py
    py="$(_fizgig_rocm_python)"
    if [[ "${STABLE_DEVICE_WHEEL:-0}" != "1" ]]; then
        return 0
    fi
    "$py" - <<PY
import importlib.metadata as md
import sys

import torch
import torchvision

arch = "${ARCH}"
checks = [
    (f"amd-torch-device-{arch}", torch.__version__),
    (f"amd-torchvision-device-{arch}", torchvision.__version__),
]
for pkg_prefix, want_ver in checks:
    needle = pkg_prefix.lower()
    found = []
    for d in md.distributions():
        name = (d.metadata.get("Name") or "").lower()
        if name == needle:
            found.append(d.metadata.get("Name", ""))
            if d.version != want_ver:
                print(
                    f"ERROR: {d.metadata.get('Name')} {d.version} != expected {want_ver}",
                    file=sys.stderr,
                )
                sys.exit(1)
    if not found:
        print(f"ERROR: missing {pkg_prefix}=={want_ver}", file=sys.stderr)
        sys.exit(1)
    print(f"OK  {found[0]} (matches {want_ver})")
PY
}

install_rocm_torch_wheels() {
    : "${ARCH:?ARCH must be set before installing torch}"
    local index="$1"
    shift
    local -a uv_extra=("$@")
    local -a uv_pkgs=()
    local py
    py="$(_fizgig_rocm_python)"
    STABLE_DEVICE_WHEEL=1

    if [[ -n "${TORCH_PIN:-}" && -n "${TORCHVISION_PIN:-}" ]]; then
        uv_pkgs=(
            "torch[device-${ARCH}]==${TORCH_PIN}"
            "torchvision[device-${ARCH}]==${TORCHVISION_PIN}"
        )
        if [[ -n "${ROCM_SDK_PIN:-}" ]]; then
            uv_pkgs+=("rocm-sdk-devel==${ROCM_SDK_PIN}")
        else
            uv_pkgs+=("rocm-sdk-devel")
        fi
        echo "Installing torch[device-${ARCH}]==${TORCH_PIN} + torchvision[device-${ARCH}]==${TORCHVISION_PIN} + rocm-sdk-devel${ROCM_SDK_PIN:+==${ROCM_SDK_PIN}} (uv)..."
    else
        uv_pkgs=(
            "torch[device-${ARCH}]"
            "torchvision[device-${ARCH}]"
            "rocm-sdk-devel"
        )
        echo "Installing torch[device-${ARCH}] + torchvision[device-${ARCH}] + rocm-sdk-devel (floating, uv)..."
        echo "  Override: TORCH_PIN=… TORCHVISION_PIN=… ROCM_SDK_PIN=…"
    fi

    echo "Source: ${index}"
    "$py" -m uv pip install --index-strategy unsafe-best-match \
        --prerelease allow --upgrade --reinstall --index-url "$index" \
        "${uv_pkgs[@]}" \
        "${uv_extra[@]}"
}

verify_torch_rocm_pin() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<PY
import re
import sys

import torch

pin = "${ROCM_SDK_PIN}"
want = ".".join(pin.split(".")[:2])
got = None

rocm = getattr(torch.version, "rocm", None)
if rocm:
    m = re.match(r"(\d+\.\d+)", str(rocm))
    if m:
        got = m.group(1)

if got is None:
    m = re.search(r"\+rocm(\d+\.\d+)", torch.__version__, re.I)
    if m:
        got = m.group(1)

if got != want:
    print(
        f"ERROR: PyTorch ROCm {got or '?'} != required {want} "
        f"(ROCM_SDK_PIN={pin})",
        file=sys.stderr,
    )
    print(f"       torch {torch.__version__}", file=sys.stderr)
    sys.exit(1)

print(f"OK  torch ROCm {got} matches pin (ROCM_SDK_PIN={pin})")
PY
}

report_installed_rocm_torch() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import torch

rocm = getattr(torch.version, "rocm", "n/a")
hip = getattr(torch.version, "hip", "n/a")
print(f"OK  torch {torch.__version__}  rocm={rocm}  hip={hip}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
PY
}

verify_bitsandbytes_rocm_lib() {
    local py bnb_suffix
    py="$(_fizgig_rocm_python)"
    bnb_suffix="$(_bnb_rocm_suffix_from_torch)" || bnb_suffix="714"
    BNB_ROCM_SUFFIX="$bnb_suffix"
    "$py" - <<PY
import sys
from pathlib import Path

import bitsandbytes

suffix = "${bnb_suffix}"
so = Path(bitsandbytes.__file__).resolve().parent / f"libbitsandbytes_rocm{suffix}.so"
if so.is_file():
    print(f"OK  {so.name} ({so.stat().st_size:,} bytes) — set BNB_ROCM_VERSION={suffix}")
    sys.exit(0)

candidates = sorted(Path(bitsandbytes.__file__).resolve().parent.glob("libbitsandbytes_rocm*.so"))
print(f"WARN: libbitsandbytes_rocm{suffix}.so missing (torch ROCm {suffix}).", file=sys.stderr)
if candidates:
    print("      Available in bitsandbytes wheel:", file=sys.stderr)
    for p in candidates:
        print(f"        {p.name}", file=sys.stderr)
    print(
        f"      8-bit optimizers may fail until bitsandbytes ships rocm{suffix}; "
        f"write_rocm_env falls back to an available lib when possible.",
        file=sys.stderr,
    )
else:
    print("      No libbitsandbytes_rocm*.so found.", file=sys.stderr)
PY
}

verify_torch_gpu_kernel() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print("WARN GPU kernel test skipped (cuda not available)", file=sys.stderr)
    sys.exit(1)
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    y = x @ x
    torch.cuda.synchronize()
    del x, y
except Exception as exc:
    print(f"ERROR: GPU kernel smoke test failed: {exc}", file=sys.stderr)
    sys.exit(1)
print("OK  GPU kernel smoke test passed")
PY
}

verify_torchvision() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import sys

import torch
import torchvision

print(f"torch {torch.__version__}  torchvision {torchvision.__version__}")
try:
    from torchvision.transforms import InterpolationMode  # noqa: F401
except Exception as exc:
    print(f"ERROR: torchvision incompatible with torch: {exc}", file=sys.stderr)
    sys.exit(1)
print("OK  torchvision import (torch/torchvision ABI match)")
PY
}

install_rocm_torch_pinned() {
    local index="$1"
    shift

    install_rocm_torch_wheels "$index" "$@" || return 1
    _fizgig_export_rocm_runtime_path
    _rocm_sdk_init_post_torch || true
    if [[ -n "${ROCM_SDK_PIN:-}" ]]; then
        verify_torch_rocm_pin || return 1
    fi
    verify_torch_device_wheel || return 1
    verify_torch_gpu_kernel || return 1
    verify_torchvision || return 1
    report_installed_rocm_torch || return 1
    return 0
}

install_torch_stable() {
    install_rocm_torch_pinned "$ROCM_INDEX" || return 1
}

install_torch_nightly() {
    install_rocm_torch_pinned "$ROCM_NIGHTLY_INDEX" || return 1
}

cd "$FIZGIG_ROOT"

echo "============================================================"
echo "  Fizgig Installer — AMD ROCm (Linux, pip wheels)"
echo "  LoRA and Fine-tune Studio"
echo "  *** HIGHLY EXPERIMENTAL — Linux AMD is best-effort only ***"
echo "============================================================"
echo
echo "PyTorch / ROCm wheels are from AMD indexes — not built by Fizgig."
echo "  Channel: ${ROCM_CHANNEL}  (default stable → ${ROCM_INDEX})"
if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    echo "  Stable pin: torch==${TORCH_PIN}  torchvision==${TORCHVISION_PIN}  rocm-sdk-devel==${ROCM_SDK_PIN}"
    echo "  Prefer nightly? ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh"
else
    echo "  Nightly index: ${ROCM_NIGHTLY_INDEX} (floating torch[device-\${ARCH}] unless TORCH_PIN is set)"
    echo "  Stable instead: omit ROCM_CHANNEL or use ROCM_CHANNEL=stable ./install_fizgig_rocm.sh"
    echo "  Override: TORCH_PIN=…  TORCHVISION_PIN=…  ROCM_SDK_PIN=…"
fi
echo
echo "  Docs: https://github.com/ROCm/TheRock/blob/main/RELEASES.md"
echo
echo "Shared deps come from requirements.txt with CUDA torch/bnb and nvidia-ml-py filtered out."
echo "bitsandbytes>=0.50.0 is installed separately (--no-deps; lib must match torch ROCm major.minor)."
echo
echo "Optional status-bar VRAM: sudo apt install amdrocm-amdsmi  (or dnf equivalent)."
echo

if [[ ! -e /dev/kfd ]]; then
    echo "ERROR: /dev/kfd not found — AMD ROCm kernel driver is not loaded."
    echo "Install the amdgpu/ROCm stack first (see AMD ROCm install docs), then re-run."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found."
    exit 1
fi

INSTALL_PYTHON="$(_fizgig_install_python)"

"$INSTALL_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "ERROR: Python 3.10+ required."
    exit 1
}

echo "Python: $("$INSTALL_PYTHON" --version) ($("$INSTALL_PYTHON" -c 'import sys; print(sys.executable)'))"
echo

_ensure_python_dev_headers "$INSTALL_PYTHON"

if [[ ! -e /usr/lib/x86_64-linux-gnu/libnuma.so ]] && [[ ! -e /lib/x86_64-linux-gnu/libnuma.so ]]; then
    echo "Installing libnuma-dev (PyTorch rocSHMEM needs unversioned libnuma.so)..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y libnuma-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y numactl-devel
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y numactl-devel
    else
        echo "ERROR: libnuma.so not found and no supported package manager (apt/dnf/yum)."
        echo "Install libnuma-dev (Debian/Ubuntu) or numactl-devel (Fedora/RHEL), then re-run."
        exit 1
    fi
fi

if [[ -d venv ]]; then
    read -r -p "Virtual environment already exists at venv/. Delete and recreate? (y/N): " RECREATE
    if [[ "${RECREATE,,}" == "y" ]]; then
        rm -rf venv
    fi
fi

if [[ ! -d venv ]]; then
    echo "Creating virtual environment..."
    "$INSTALL_PYTHON" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade uv

echo
echo "Detecting AMD GPU architecture..."
ARCH=""
mkdir -p user_data
if [[ ! -f src/install/detect_gpu_linux.py ]]; then
    echo "ERROR: src/install/detect_gpu_linux.py not found."
    exit 1
fi

if ! ARCH="$(python src/install/detect_gpu_linux.py 2>user_data/gpu_detect_debug.log)"; then
    echo "ERROR: GPU detection failed or unsupported AMD GPU."
    cat gpu_detect_debug.log 2>/dev/null || true
    exit 1
fi
ARCH="${ARCH//$'\r'/}"
ARCH="${ARCH//$'\n'/}"
if [[ -z "$ARCH" ]]; then
    echo "ERROR: Empty gfx code from detect_gpu_linux.py"
    exit 1
fi
echo "Detected GPU architecture: $ARCH"
if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    echo "ROCm channel: stable (torch ${TORCH_PIN}, device-${ARCH})"
else
    echo "ROCm channel: nightly (floating device-${ARCH} unless TORCH_PIN is set)"
fi
echo

if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    echo "Installing ROCm PyTorch ${TORCH_PIN} (stable whl-next) for ${ARCH}..."
    install_torch_stable || {
        echo "ERROR: Stable ROCm PyTorch install failed for ${ARCH}." >&2
        echo "       Try nightly: ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh" >&2
        echo "       Override: TORCH_PIN=… TORCHVISION_PIN=… ROCM_SDK_PIN=…" >&2
        exit 1
    }
else
    echo "Installing ROCm PyTorch (nightly whl-next) for ${ARCH}..."
    echo "  pip install --index-url ${ROCM_NIGHTLY_INDEX} \\"
    echo "      \"torch[device-${ARCH}]\" \"torchvision[device-${ARCH}]\" \"rocm-sdk-devel\""
    install_torch_nightly || {
        echo "ERROR: Nightly ROCm PyTorch install failed for ${ARCH}." >&2
        echo "       See https://github.com/ROCm/TheRock/blob/main/RELEASES.md" >&2
        echo "       Fallback: ROCM_CHANNEL=stable ./install_fizgig_rocm.sh" >&2
        exit 1
    }
fi

echo
echo "Installing Fizgig dependencies from requirements.txt (CUDA torch/bnb + nvidia-ml-py stripped)..."
SHARED_REQS="$(mktemp)"
python filter_requirements_rocm.py requirements.txt "$SHARED_REQS"
# hqq (4-bit HQQ base) builds an optional CUDA kernel from its sdist unless told not to.
DISABLE_CUDA=1 python -m uv pip install --index-strategy unsafe-best-match \
    -r "$SHARED_REQS"
rm -f "$SHARED_REQS"

echo
echo "Installing Qt for Fizgig's app window..."
# pywebview finds no window engine inside the venv otherwise (the distro's GTK bindings aren't visible there)
python -m uv pip install "pywebview[qt]>=5,<7" \
    || echo "NOTE: Qt for the app window didn't install; launch_fizgig_amd.sh opens Fizgig in your browser instead."

echo
echo "Installing bitsandbytes (ROCm — matched to installed torch when possible)..."
# PyPI bnb declares torch>=2.4 — would pull CUDA torch and nvidia-* libs over our ROCm wheel.
python -m uv pip install --no-deps -U "bitsandbytes>=0.50.0"

echo
echo "Verifying bitsandbytes ROCm library..."
verify_bitsandbytes_rocm_lib || true

echo
echo "Verifying ROCm / HIP..."
_fizgig_export_rocm_runtime_path
python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"HIP: {getattr(torch.version, 'hip', 'n/a')}")
print(f"ROCm: {getattr(torch.version, 'rocm', 'n/a')}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
PY
if [[ -n "${ROCM_SDK_PIN:-}" ]]; then
    verify_torch_rocm_pin || exit 1
fi
"$(_fizgig_rocm_python)" - <<'PY'
import torch
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    _ = x @ x
    torch.cuda.synchronize()
    print("OK  GPU kernel smoke test passed")
except Exception as exc:
    import sys
    print(f"ERROR: GPU kernel smoke test failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Checking VRAM monitor (amd-smi)..."
python - <<'PY'
import sys
sys.path.insert(0, "src")
from fizgig.utils.vram_monitor import _read_vram_amd_smi_cli
hit = _read_vram_amd_smi_cli()
if hit:
    used, total = hit
    print(f"OK  VRAM monitor: {used / (1024**3):.1f} / {total / (1024**3):.1f} GB in use")
else:
    print("WARN VRAM monitor: amd-smi unavailable or returned no data.")
    print("     Optional: sudo apt install amdrocm-amdsmi (see README)")
PY

echo
echo "Downloading InsightFace models (CPU, ~300 MB)..."
python - <<'PY'
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "genderage", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1)
print("Models ready.")
PY

echo
echo "Writing ROCm launcher config (BNB_ROCM_VERSION, rocm_env.sh)..."
python write_rocm_env.py || true
chmod +x launch_fizgig_amd.sh

echo
echo "============================================================"
echo "  Installation complete!"
echo
echo "  Launch with: ./launch_fizgig_amd.sh"
echo "  (launch_fizgig.sh / launch_fizgig_browser.sh are the upstream launchers — no ROCm env; do not use for AMD training)"
echo
echo "  NOTE: Linux AMD ROCm is highly experimental — crashes and GPU resets are common."
echo "============================================================"
