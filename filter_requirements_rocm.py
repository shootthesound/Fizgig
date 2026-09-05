#!/usr/bin/env python3
"""Build a ROCm-safe requirements file from the shared requirements.txt.

Leaves requirements.txt untouched (CUDA path). Skips packages the AMD installers
install separately (torch / torchvision / bitsandbytes), NVIDIA-only deps
(nvidia-ml-py — ROCm uses amd-smi / rocm-smi fallbacks), and the CUDA
--extra-index-url line so a ROCm torch install is not replaced by cu128 wheels.

Fails loudly if the number of stripped lines does not match what the CUDA
requirements.txt is known to contain — e.g. a second --extra-index-url would
otherwise slip through or be over-stripped without anyone noticing until an
AMD install breaks.

Usage:
  python filter_requirements_rocm.py [requirements.txt] [output.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Installed by install_fizgig_rocm.bat / install_fizgig_rocm.sh instead.
SKIP_PACKAGES = frozenset({"torch", "torchvision", "bitsandbytes", "nvidia-ml-py", "comfy-kitchen"})

# Exact shape of today's requirements.txt CUDA/NVIDIA block. Bump deliberately when
# upstream changes that block — silent drift is how AMD installs go weird.
EXPECTED_SKIPPED_PACKAGES = frozenset({"torch", "torchvision", "bitsandbytes", "nvidia-ml-py", "comfy-kitchen"})
EXPECTED_CUDA_INDEX_LINES = 1  # --extra-index-url …/whl/cu128


def _package_name(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("-"):
        return None
    # Environment markers: "triton-windows ; sys_platform == 'win32'"
    req = s.split(";", 1)[0].strip()
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in req:
            return req.split(sep, 1)[0].strip().lower().split("[")[0]
    return req.split()[0].lower().split("[")[0]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "requirements.txt")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "requirements-rocm-shared.txt")
    if not src.is_file():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    kept: list[str] = []
    skipped_packages: list[str] = []
    skipped_index_lines: list[str] = []

    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("--extra-index-url") or s.startswith("--index-url") or s.startswith("-f ") or s.startswith("--find-links"):
            if s.startswith("--extra-index-url") and "pytorch.org" in s:
                skipped_index_lines.append(s)
                continue
            print(
                f"ERROR: {src} has an unexpected pip index/find-links option that "
                f"filter_requirements_rocm.py does not know how to handle:\n  {s}\n"
                f"Update the filter (EXPECTED_CUDA_INDEX_LINES / skip rules) before shipping.",
                file=sys.stderr,
            )
            return 1

        name = _package_name(line)
        if name in SKIP_PACKAGES:
            skipped_packages.append(name)
            continue
        kept.append(line)

    skipped_set = frozenset(skipped_packages)
    errors: list[str] = []

    if skipped_set != EXPECTED_SKIPPED_PACKAGES:
        errors.append(
            f"skipped packages {sorted(skipped_set)} != expected "
            f"{sorted(EXPECTED_SKIPPED_PACKAGES)} "
            f"(raw removals: {skipped_packages})"
        )
    elif len(skipped_packages) != len(EXPECTED_SKIPPED_PACKAGES):
        errors.append(
            f"expected each of {sorted(EXPECTED_SKIPPED_PACKAGES)} exactly once, "
            f"got {skipped_packages}"
        )

    if len(skipped_index_lines) != EXPECTED_CUDA_INDEX_LINES:
        errors.append(
            f"skipped {len(skipped_index_lines)} CUDA index line(s), expected "
            f"{EXPECTED_CUDA_INDEX_LINES}: {skipped_index_lines or '(none)'}"
        )

    if errors:
        print(f"ERROR: {src} does not match the ROCm filter's expected CUDA block:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "Update EXPECTED_SKIPPED_PACKAGES / EXPECTED_CUDA_INDEX_LINES in "
            "filter_requirements_rocm.py if the change is intentional.",
            file=sys.stderr,
        )
        return 1

    out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(
        f"{out}  (stripped {len(skipped_packages)} packages + "
        f"{len(skipped_index_lines)} CUDA index line)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
