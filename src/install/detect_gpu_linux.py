#!/usr/bin/env python3
"""Detect AMD GPU gfx target on Linux for ROCm PyTorch wheel installs.

Prints a single gfx code (e.g. gfx1100) to stdout for install_fizgig_rocm.sh.
Diagnostics go to stderr. Prefers discrete GPU agents over integrated/APU when both
are present.
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys
from typing import Iterable, Optional

# PCI device IDs whose Windows-reported names are generic — same mapping as detect_gpu.py.
# Source: pci.ids + AMD ROCm gfx-target docs.
PCI_DEV_TO_GFX: dict[str, str] = {
    "7590": "gfx1200",
    "7550": "gfx1201",
    "7551": "gfx1201",
    "7580": "gfx1201",
    "7581": "gfx1201",
    "7591": "gfx1201",
    "75a1": "gfx1201",
    "75b0": "gfx1201",
    "150e": "gfx1150",
    "1586": "gfx1151",
    "1114": "gfx1152",
    "1590": "gfx1150",
    "1591": "gfx1150",
    "15d0": "gfx1152",
    "744c": "gfx1100",
    "747e": "gfx1100",
    "7480": "gfx1100",
    "7489": "gfx1100",
    "748a": "gfx1100",
    "748b": "gfx1100",
    "748d": "gfx1100",
    "7499": "gfx1100",
    "74a0": "gfx1101",
    "74a1": "gfx1101",
    "74a2": "gfx1102",
    "74a3": "gfx1102",
    "73ef": "gfx1032",
    "73ff": "gfx1032",
    "73bf": "gfx1030",
    "7420": "gfx1103",
    "7421": "gfx1103",
    "7422": "gfx1103",
    "7423": "gfx1103",
    "7424": "gfx1103",
    "164e": "gfx1153",
    "15bf": "gfx1152",
    "15c8": "gfx1150",
    "73df": "gfx1031",
}


def log(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _run(cmd: list[str], timeout: float = 8) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception as exc:
        log(f"command failed ({' '.join(cmd)}): {exc}")
    return None


def _gfx_from_rocminfo() -> list[str]:
    text = _run(["rocminfo"])
    if not text:
        return []
    agents: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip().startswith("Agent"):
            if current:
                agents.append(current)
            current = {}
            continue
        m = re.match(r"\s*([^:]+):\s*(.+)", line)
        if m:
            current[m.group(1).strip().lower()] = m.group(2).strip()
    if current:
        agents.append(current)

    gfx_targets: list[str] = []
    for agent in agents:
        name = agent.get("name", "")
        device_type = agent.get("device type", "")
        m = re.search(r"(gfx\d+[a-z0-9]*)", name, re.I)
        if not m:
            continue
        gfx = m.group(1).lower()
        if device_type.upper() == "GPU":
            gfx_targets.append(gfx)
    return gfx_targets


def _gfx_from_kfd_topology() -> list[str]:
    hits: list[str] = []
    for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        m = re.search(r"gfx_target_version\s+\d+\s+(gfx\d+[a-z0-9]*)", text, re.I)
        if m:
            hits.append(m.group(1).lower())
    return hits


def _gfx_from_drm_pci() -> list[str]:
    hits: list[str] = []
    for dev_path in glob.glob("/sys/class/drm/card*/device"):
        vendor_path = f"{dev_path}/vendor"
        device_path = f"{dev_path}/device"
        try:
            with open(vendor_path, encoding="utf-8") as fh:
                vendor = fh.read().strip().lower()
            if vendor not in ("0x1002", "4098"):
                continue
            with open(device_path, encoding="utf-8") as fh:
                dev_id = fh.read().strip().lower().removeprefix("0x")
            gfx = PCI_DEV_TO_GFX.get(dev_id)
            if gfx:
                hits.append(gfx)
        except OSError:
            continue
    return hits


def _pick_gfx(candidates: Iterable[str]) -> Optional[str]:
    ordered = []
    seen = set()
    for gfx in candidates:
        g = gfx.lower()
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    if not ordered:
        return None
    # Prefer higher-end discrete targets when multiple gfx codes appear (e.g. APU + dGPU).
    priority = (
        "gfx950", "gfx942", "gfx1201", "gfx1200",
        "gfx1100", "gfx1101", "gfx1102", "gfx1103",
        "gfx1030", "gfx1031", "gfx1032",
        "gfx1151", "gfx1150", "gfx1152", "gfx1153",
    )
    for pref in priority:
        if pref in ordered:
            return pref
    return ordered[0]


def detect_gfx() -> Optional[str]:
    candidates: list[str] = []
    candidates.extend(_gfx_from_rocminfo())
    candidates.extend(_gfx_from_kfd_topology())
    candidates.extend(_gfx_from_drm_pci())
    if not candidates:
        log("No AMD gfx target found (rocminfo, kfd topology, drm PCI).")
        return None
    gfx = _pick_gfx(candidates)
    log(f"gfx candidates: {', '.join(dict.fromkeys(candidates))} -> {gfx}")
    return gfx


def main() -> int:
    if not glob.glob("/dev/kfd"):
        log("ERROR: /dev/kfd not found — AMD ROCm driver not loaded.")
        return 1
    gfx = detect_gfx()
    if not gfx:
        return 1
    print(gfx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
