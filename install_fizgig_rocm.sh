#!/usr/bin/env bash
# Fizgig ROCm Linux installer — AMD pip wheels (Path B).
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only (driver resets, gfx gaps,
# desktop+compute contention). Use Windows ROCm or NVIDIA Linux for production workloads.
# Detects gfx target, installs PyTorch/ROCm from AMD multi-arch wheels into venv,
# then Fizgig deps from requirements.txt (CUDA torch/bnb lines filtered out).
# Prerequisites: amdgpu driver, /dev/kfd, user in render/video groups; sudo for libnuma-dev / pythonX.Y-dev.
set -euo pipefail

FIZGIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROCM_INDEX="${ROCM_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"
ROCM_NIGHTLY_INDEX="${ROCM_NIGHTLY_INDEX:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
# nightly (default) = TheRock multi-arch + [device-gfx*]; pinned to ROCm 7.14 (not 7.16+).
# stable = repo.amd.com ROCm 7.14 wheels (torch 2.12.0+rocm7.14.0 today).
ROCM_CHANNEL="${ROCM_CHANNEL:-nightly}"
if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    ROCM_SDK_PIN="${ROCM_SDK_PIN:-7.14.0}"
    TORCH_PIN="${TORCH_PIN:-2.12.0+rocm7.14.0}"
else
    ROCM_SDK_PIN="${ROCM_SDK_PIN:-7.14.0}"
    TORCH_PIN="${TORCH_PIN:-}"
    ROCM_META_PIN="${ROCM_META_PIN:-}"
    TORCH_NIGHTLY_MINOR="${TORCH_NIGHTLY_MINOR:-2.12}"
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

# Nightly: resolve torch + matching rocm meta on 7.14 (bitsandbytes 714); stable uses _resolve_stable_stack.
_resolve_nightly_stack() {
    local index="$1"
    local py
    py="$(_fizgig_rocm_python)"
    TORCH_VER=""
    ROCM_META_VER=""
    VISION_VER=""
    while IFS= read -r line; do
        case "$line" in
            TORCH_VER=*) TORCH_VER="${line#TORCH_VER=}" ;;
            ROCM_META_VER=*) ROCM_META_VER="${line#ROCM_META_VER=}" ;;
            VISION_VER=*) VISION_VER="${line#VISION_VER=}" ;;
        esac
    done < <("$py" - <<PY
import json
import re
import subprocess
import sys

index = """${index}"""
sdk_pin = """${ROCM_SDK_PIN:-7.14.0}"""
torch_pin = """${TORCH_PIN:-}"""
rocm_meta_pin = """${ROCM_META_PIN:-}"""
prefer_minor = """${TORCH_NIGHTLY_MINOR:-2.12}"""


def semver_tuple(v: str) -> tuple[int, ...]:
    base = v.split("+", 1)[0]
    return tuple(int(p) for p in re.split(r"[.\-]", base) if p.isdigit())


def pip_versions(package: str) -> list[str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "index", "versions", package,
            "--index-url", index, "--pre", "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)["versions"]


sdk_major_minor = ".".join(sdk_pin.split(".")[:2])
sdk_prefix = sdk_pin if sdk_pin.endswith(".0") else f"{sdk_pin.split('.')[0]}.{sdk_pin.split('.')[1]}.0"


def rocm714_meta_versions() -> list[str]:
    return [
        v for v in pip_versions("rocm")
        if v.startswith(sdk_prefix) or v.startswith(f"{sdk_major_minor}.")
    ]


def torch714_versions() -> list[str]:
    out = []
    for v in pip_versions("torch"):
        if not re.search(rf"\+rocm{re.escape(sdk_major_minor)}", v, re.I):
            continue
        if re.search(r"\+rocm7\.15|7\.16|rocm10", v, re.I):
            continue
        out.append(v)
    return out


def alpha_key(v: str) -> tuple:
    m = re.search(r"\.0a(\d+)$", v)
    if m:
        return (int(m.group(1)),)
    m2 = re.match(r"(\d+)\.(\d+)\.(\d+)$", v)
    if m2:
        return (0, int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    return (0,)


def torch_sort_key(v: str) -> tuple:
    m = re.search(r"\+rocm7\.14\.0a(\d+)", v, re.I)
    alpha = int(m.group(1)) if m else 0
    return (*semver_tuple(v), alpha)


def vision714_versions() -> list[str]:
    out = []
    for v in pip_versions("torchvision"):
        if not re.search(rf"\+rocm{re.escape(sdk_major_minor)}", v, re.I):
            continue
        if re.search(r"\+rocm7\.15|7\.16|rocm10", v, re.I):
            continue
        out.append(v)
    return out


def vision_for_torch(torch_ver: str) -> str:
    """TheRock matrix: torch 2.12 -> 0.27, 2.13 -> 0.28, 2.14 -> 0.29, etc."""
    base = torch_ver.split("+", 1)[0]
    rocm_tag = torch_ver.split("+", 1)[1] if "+" in torch_ver else ""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(.*)$", base)
    if not m:
        print(f"ERROR: cannot parse torch version {torch_ver!r}", file=sys.stderr)
        sys.exit(1)
    _major, minor, patch, rest = m.groups()
    vision_base = f"0.{int(minor) + 15}.{patch}{rest}"
    exact = f"{vision_base}+{rocm_tag}" if rocm_tag else vision_base
    versions = vision714_versions()
    if exact in versions:
        return exact
    prefix = f"{vision_base}+"
    cands = [v for v in versions if v.startswith(prefix)]
    if cands:
        return max(cands, key=semver_tuple)
    print(
        f"ERROR: no torchvision match for torch {torch_ver} on {index} "
        f"(expected ~{vision_base}+rocm…)",
        file=sys.stderr,
    )
    sys.exit(1)


if torch_pin:
    torch_ver = torch_pin
elif rocm_meta_pin:
    meta = rocm_meta_pin
    suffix = f"+rocm{meta}"
    cands = [v for v in torch714_versions() if v.lower().endswith(suffix.lower())]
    if not cands:
        cands = [v for v in torch714_versions() if f"+rocm{meta.split('a')[0]}" in v.lower()]
    if not cands:
        print(
            f"ERROR: no torch +rocm{meta} on {index}",
            file=sys.stderr,
        )
        sys.exit(1)
    pref = [v for v in cands if v.startswith(f"{prefer_minor}.")]
    torch_ver = max(pref or cands, key=torch_sort_key)
    meta = re.search(r"\+rocm(7\.14\.0(?:a\d+)?)", torch_ver, re.I)
    meta = meta.group(1) if meta else rocm_meta_pin
else:
    meta_vers = rocm714_meta_versions()
    if not meta_vers:
        print(f"ERROR: no rocm {sdk_prefix}* on {index}", file=sys.stderr)
        sys.exit(1)
    meta = max(meta_vers, key=alpha_key)
    suffix = f"+rocm{meta}"
    cands = [v for v in torch714_versions() if v.lower().endswith(suffix.lower())]
    if not cands:
        cands = [v for v in torch714_versions() if re.search(r"\+rocm7\.14", v, re.I)]
    if not cands:
        print(f"ERROR: no torch +rocm{sdk_major_minor} on {index}", file=sys.stderr)
        sys.exit(1)
    pref = [v for v in cands if v.startswith(f"{prefer_minor}.")]
    torch_ver = max(pref or cands, key=torch_sort_key)
    m = re.search(r"\+rocm(7\.14\.0(?:a\d+)?)", torch_ver, re.I)
    if m:
        meta = m.group(1)

print(f"TORCH_VER={torch_ver}")
print(f"ROCM_META_VER={meta}")
print(f"VISION_VER={vision_for_torch(torch_ver)}")
PY
)
    if [[ -z "$TORCH_VER" || -z "$ROCM_META_VER" || -z "$VISION_VER" ]]; then
        echo "ERROR: failed to resolve nightly torch/vision/rocm ${ROCM_SDK_PIN} from ${index}" >&2
        return 1
    fi
}

_rocm_sdk_init_post_torch() {
    _fizgig_export_rocm_runtime_path
    if ! command -v rocm-sdk >/dev/null 2>&1; then
        echo "WARN: rocm-sdk not on PATH — skipping rocm-sdk init (torch may still work)."
        return 0
    fi
    echo "Running rocm-sdk init (TheRock devel tree)..."
    if rocm-sdk init; then
        echo "OK  rocm-sdk init"
    else
        echo "WARN: rocm-sdk init failed."
        return 1
    fi
    if rocm-sdk targets 2>/dev/null | head -1; then
        :
    fi
}

_fizgig_export_rocm_runtime_path() {
    local _bin="${FIZGIG_ROOT}/venv/bin"
    [[ -d "$_bin" ]] && PATH="${_bin}${PATH:+:$PATH}"
    for _d in /opt/rocm/core-*/bin /opt/rocm/bin; do
        [[ -d "$_d" ]] && PATH="${_d}${PATH:+:$PATH}"
    done
    export PATH
    for _lib in "${FIZGIG_ROOT}"/venv/lib/python*/site-packages/_rocm_sdk_core/lib \
                "${FIZGIG_ROOT}"/venv/lib/python*/site-packages/_rocm_sdk/lib \
                "${FIZGIG_ROOT}"/venv/lib/python*/site-packages/_rocm_sdk_libraries/lib \
                /opt/rocm/lib /opt/rocm/lib64; do
        [[ -d "$_lib" ]] && LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
}

_bnb_rocm_suffix_from_torch() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import re
import sys

import torch

rocm = getattr(torch.version, "rocm", None)
if rocm:
    m = re.match(r"(\d+)\.(\d+)", str(rocm))
    if m:
        print(f"{m.group(1)}{int(m.group(2))}")
        sys.exit(0)

m = re.search(r"\+rocm(\d+)\.(\d+)", torch.__version__, re.I)
if m:
    print(f"{m.group(1)}{int(m.group(2))}")
    sys.exit(0)

print("714")
PY
}

# Stable: never use torch[device-*] extras (2.13+ warns and skips device wheels). Install
# amd-torch-device-{arch} explicitly when that package exists on the index.
_resolve_stable_stack() {
    local index="$1"
    local py
    py="$(_fizgig_rocm_python)"
    TORCH_VER=""
    VISION_VER=""
    STABLE_DEVICE_WHEEL=0
    if [[ -n "${TORCH_PIN:-}" ]]; then
        TORCH_VER="$TORCH_PIN"
    fi
    while IFS= read -r line; do
        case "$line" in
            TORCH_VER=*) TORCH_VER="${line#TORCH_VER=}" ;;
            VISION_VER=*) VISION_VER="${line#VISION_VER=}" ;;
            STABLE_DEVICE_WHEEL=*) STABLE_DEVICE_WHEEL="${line#STABLE_DEVICE_WHEEL=}" ;;
        esac
    done < <("$py" - <<PY
import json
import re
import subprocess
import sys

index = """${index}"""
arch = """${ARCH}"""
torch_pin = """${TORCH_PIN:-}"""
torch_device_pkg = f"amd-torch-device-{arch}"
vision_device_pkg = f"amd-torchvision-device-{arch}"


def semver_tuple(v: str) -> tuple[int, ...]:
    base = v.split("+", 1)[0]
    return tuple(int(p) for p in re.split(r"[.\-]", base) if p.isdigit())


def pip_versions(package: str) -> list[str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "index", "versions", package,
            "--index-url", index, "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return json.loads(proc.stdout)["versions"]


def rocm714_only(versions: list[str]) -> list[str]:
    return [
        v for v in versions
        if re.search(r"\+rocm7\.14", v, re.I) and not re.search(r"\+rocm7\.15", v, re.I)
    ]


def resolve_torch_pin(requested: str, available: list[str]) -> str:
    if requested in available:
        return requested
    parts = requested.split("+", 1)[0].split(".")
    if len(parts) >= 2:
        prefix = f"{parts[0]}.{parts[1]}."
        cands = [v for v in available if v.startswith(prefix)]
        if cands:
            chosen = max(cands, key=semver_tuple)
            print(f"NOTE: {requested} not on index; using {chosen}", file=sys.stderr)
            return chosen
    print(f"ERROR: {requested} not on {index}", file=sys.stderr)
    print(
        "       Stable ships torch==2.12.0+rocm7.14.0 today. To try torch 2.14 nightly:",
        file=sys.stderr,
    )
    print(
        "         ROCM_CHANNEL=nightly TORCH_NIGHTLY_MINOR=2.14 ./install_fizgig_rocm.sh",
        file=sys.stderr,
    )
    print(
        "       Example pin (check index for newer builds): "
        "TORCH_PIN=2.14.0a0+rocm7.14.0a20260625",
        file=sys.stderr,
    )
    sys.exit(1)


def vision_for_torch(torch_ver: str) -> str:
    base = torch_ver.split("+", 1)[0]
    rocm_tag = torch_ver.split("+", 1)[1] if "+" in torch_ver else ""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(.*)$", base)
    if not m:
        print(f"ERROR: cannot parse torch version {torch_ver!r}", file=sys.stderr)
        sys.exit(1)
    _major, minor, patch, rest = m.groups()
    vision_base = f"0.{int(minor) + 15}.{patch}{rest}"
    exact = f"{vision_base}+{rocm_tag}" if rocm_tag else vision_base
    versions = rocm714_only(pip_versions("torchvision"))
    if exact in versions:
        return exact
    prefix = f"{vision_base}+"
    cands = [v for v in versions if v.startswith(prefix)]
    if cands:
        return max(cands, key=semver_tuple)
    print(f"ERROR: no torchvision match for torch {torch_ver} on {index}", file=sys.stderr)
    sys.exit(1)


torch_device_vers = rocm714_only(pip_versions(torch_device_pkg))
vision_device_vers = rocm714_only(pip_versions(vision_device_pkg))
torch_vers = rocm714_only(pip_versions("torch"))
if not torch_device_vers or not vision_device_vers:
    if torch_pin:
        torch_ver = resolve_torch_pin(torch_pin, torch_vers)
        print(f"TORCH_VER={torch_ver}")
        print(f"VISION_VER={vision_for_torch(torch_ver)}")
    print("STABLE_DEVICE_WHEEL=0")
    sys.exit(0)

if torch_pin:
    torch_ver = resolve_torch_pin(torch_pin, torch_vers or torch_device_vers)
else:
    torch_ver = max(torch_device_vers, key=semver_tuple)
vision_ver = vision_for_torch(torch_ver)
vision_prefix = vision_ver.split("+", 1)[0] + "+"
if vision_ver not in vision_device_vers and not any(v.startswith(vision_prefix) for v in vision_device_vers):
    print(f"TORCH_VER={torch_ver}")
    print(f"VISION_VER={vision_ver}")
    print("STABLE_DEVICE_WHEEL=0")
    sys.exit(0)

print(f"TORCH_VER={torch_ver}")
print(f"VISION_VER={vision_ver}")
print("STABLE_DEVICE_WHEEL=1")
PY
)
}

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

    local -a uv_pre=()
    local -a uv_upgrade=(--reinstall)
    STABLE_DEVICE_WHEEL=0

    if [[ "$index" == *nightlies* ]]; then
        _resolve_nightly_stack "$index"
        uv_pkgs=(
            "rocm[libraries,devel,device-${ARCH}]==${ROCM_META_VER}"
            "torch[device-${ARCH}]==${TORCH_VER}"
            "torchvision[device-${ARCH}]==${VISION_VER}"
        )
        echo "Installing rocm[libraries,devel,device-${ARCH}]==${ROCM_META_VER} + torch[device-${ARCH}]==${TORCH_VER} + torchvision[device-${ARCH}]==${VISION_VER} (nightly 7.14 pin, uv)..."
        echo "  Override: TORCH_PIN=… ROCM_META_PIN=… TORCH_NIGHTLY_MINOR=2.12 (default minor when unpinned)"
        uv_pre=(--prerelease allow)
        uv_upgrade=(--upgrade --reinstall)
        STABLE_DEVICE_WHEEL=1
    else
        _resolve_stable_stack "$index"
        if [[ "${STABLE_DEVICE_WHEEL}" == "1" ]]; then
            uv_pkgs=(
                "torch==${TORCH_VER}"
                "torchvision==${VISION_VER}"
                "amd-torch-device-${ARCH}==${TORCH_VER}"
                "amd-torchvision-device-${ARCH}==${VISION_VER}"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch==${TORCH_VER} + torchvision==${VISION_VER} + amd-torch-device-${ARCH} + amd-torchvision-device-${ARCH} + rocm-sdk-devel==${ROCM_SDK_PIN} (uv)..."
        elif [[ -n "${TORCH_VER:-}" ]]; then
            uv_pkgs=(
                "torch==${TORCH_VER}"
                "torchvision==${VISION_VER}"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch==${TORCH_VER} + torchvision==${VISION_VER} + rocm-sdk-devel==${ROCM_SDK_PIN} (stable — TORCH_PIN, no device wheel, uv)..."
        else
            uv_pkgs=(
                "torch"
                "torchvision"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch + torchvision + rocm-sdk-devel==${ROCM_SDK_PIN} (stable, unpinned, uv)..."
        fi
    fi

    echo "Source: ${index}"
    "$py" -m uv pip install --index-strategy unsafe-best-match \
        "${uv_pre[@]}" "${uv_upgrade[@]}" --index-url "$index" \
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
        f"(stable stack expects libbitsandbytes_rocm714.so / BNB_ROCM_VERSION=714)",
        file=sys.stderr,
    )
    print(f"       torch {torch.__version__}", file=sys.stderr)
    sys.exit(1)

print(f"OK  torch ROCm {got} matches stable pin (BNB_ROCM_VERSION=714)")
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
        f"      8-bit optimizers may fail until bitsandbytes ships rocm{suffix} "
        f"or you use ROCM_CHANNEL=stable (7.14).",
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
    verify_torch_rocm_pin || return 1
    verify_torch_device_wheel || return 1
    verify_torch_gpu_kernel || return 1
    verify_torchvision || return 1
    return 0
}

install_torch_stable() {
    install_rocm_torch_pinned "$ROCM_INDEX" || return 1
}

install_torch_nightly() {
    install_rocm_torch_wheels "$ROCM_NIGHTLY_INDEX" || return 1
    _fizgig_export_rocm_runtime_path
    _rocm_sdk_init_post_torch || return 1
    verify_torch_rocm_pin || return 1
    verify_torch_device_wheel || return 1
    verify_torch_gpu_kernel || return 1
    verify_torchvision || return 1
    report_installed_rocm_torch || return 1
    return 0
}

cd "$FIZGIG_ROOT"

echo "============================================================"
echo "  Fizgig Installer — AMD ROCm (Linux, pip wheels)"
echo "  Klein 9B and Krea 2 LoRA Studio"
echo "  *** HIGHLY EXPERIMENTAL — Linux AMD is best-effort only ***"
echo "============================================================"
echo
echo "PyTorch / ROCm wheels are from AMD indexes — not built by Fizgig."
echo "  Channel: ${ROCM_CHANNEL}  (default nightly → ${ROCM_NIGHTLY_INDEX})"
if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    echo "  Stable pin: torch==${TORCH_PIN}  rocm-sdk==${ROCM_SDK_PIN}"
    echo "  Prefer nightly? Omit ROCM_CHANNEL or use: ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh"
else
    echo "  Nightly pin: ROCm ${ROCM_SDK_PIN} — torch[device-\${ARCH}] minor ${TORCH_NIGHTLY_MINOR} when unpinned"
    echo "  Stable instead: ROCM_CHANNEL=stable ./install_fizgig_rocm.sh  (torch==2.12.0+rocm7.14.0)"
    echo "  Try torch 2.14 nightly: TORCH_NIGHTLY_MINOR=2.14  (or TORCH_PIN=2.14.0a0+rocm7.14.0a20260625)"
    echo "  Override: TORCH_PIN=…  ROCM_META_PIN=…  TORCH_NIGHTLY_MINOR=…"
fi
echo
echo "  Stable install: ROCM_CHANNEL=stable ./install_fizgig_rocm.sh"
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
if [[ ! -f detect_gpu_linux.py ]]; then
    echo "ERROR: detect_gpu_linux.py not found."
    exit 1
fi

if ! ARCH="$(python detect_gpu_linux.py 2>gpu_detect_debug.log)"; then
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
    echo "ROCm SDK pin: ${ROCM_SDK_PIN} (bitsandbytes libbitsandbytes_rocm714.so)"
else
    echo "ROCm channel: nightly (pinned to ${ROCM_SDK_PIN} for bitsandbytes 714; device-${ARCH})"
fi
echo

if [[ "$ROCM_CHANNEL" == "stable" ]]; then
    echo "Installing ROCm PyTorch ${ROCM_SDK_PIN} (stable multi-arch from repo.amd.com) for ${ARCH}..."
    install_torch_stable || {
        echo "ERROR: Stable ROCm PyTorch install failed for ${ARCH}." >&2
        echo "       Try nightly: ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh" >&2
        echo "       Override: TORCH_PIN=… (must install amd-torch-device-${ARCH})." >&2
        exit 1
    }
else
    echo "Installing ROCm PyTorch (nightly multi-arch) for ${ARCH}..."
    echo "  pip install --index-url ${ROCM_NIGHTLY_INDEX} \\"
    echo "      \"rocm[libraries,devel,device-${ARCH}]==\${ROCM_META_VER}\" \\"
    echo "      \"torch[device-${ARCH}]==\${TORCH_VER}\" \"torchvision[device-${ARCH}]==\${VISION_VER}\""
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
verify_torch_rocm_pin || exit 1
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
chmod +x run_fizgig_rocm.sh

echo
echo "============================================================"
echo "  Installation complete!"
echo
echo "  Launch with: ./run_fizgig_rocm.sh"
echo "  (run_fizgig.sh is the upstream launcher — no ROCm env; do not use for AMD training)"
echo
echo "  NOTE: Linux AMD ROCm is highly experimental — crashes and GPU resets are common."
echo "============================================================"
