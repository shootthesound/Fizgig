#!/usr/bin/env python3
"""Shared uv-install logic for every Fizgig entry point that installs requirements.txt into a
venv: install_fizgig.py, update_fizgig.bat, docker/Dockerfile, and docker/entrypoint.sh. One
file so the two fixes below live in exactly one place instead of four near-duplicates that
drift out of sync (Windows batch, POSIX shell, and Dockerfile RUN steps all just invoke this
as a subprocess via the venv's own python).

1. --link-mode is decided up front by checking whether the uv cache and the venv actually
   share a volume, instead of always forcing copy mode. uv's own hardlink fast path only works
   when they do; forcing copy unconditionally (the previous behavior, added purely to silence
   uv's fallback warning) throws that away even in the common same-drive case. When they don't
   share a volume, we say so ourselves with one calm line before uv ever gets a chance to print
   its own alarming "Failed to hardlink files..." warning.

2. torch/torchvision install from PyTorch's own CUDA wheel index via
   --index-strategy unsafe-best-match, but that flag is scoped to ONLY that one call instead of
   the whole requirements.txt. Applying it file-wide is the more dangerous form: PyTorch's index
   mirrors a chunk of the ordinary PyPI ecosystem (tqdm among them) at whatever older versions
   its own wheels were built against, and unsafe-best-match means every other package in
   requirements.txt effectively takes suggestions from that second index too, not just torch.
   (The narrower failure mode — file-wide *first-index*, uv's safe default — is just as broken:
   it locks onto whichever index it sees a package name on first, at ANY version, and never
   falls through to PyPI even when that first hit doesn't satisfy the pin. Confirmed via
   `uv pip install --dry-run` against a clean venv: the shared file fails outright on
   tqdm==4.67.1 because PyTorch's mirror only carries an older one.)

Usable two ways:
    Imported:    from uv_install_deps import install_requirements
    As a script: <venv-python> uv_install_deps.py [requirements_file] [venv_dir]
                 (both default to requirements.txt / venv next to this file)
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TORCH_ECOSYSTEM = ("torch", "torchvision", "torchaudio")


def _uv_cache_dir(python_path):
    """Ask uv for its actual cache directory (respects UV_CACHE_DIR/XDG overrides and any
    project config) rather than guessing at its platform defaults ourselves."""
    try:
        # shell=False (list form) — python_path is always a caller-supplied venv interpreter
        # path, never external/untrusted input; no shell is involved to inject into.
        result = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [str(python_path), "-m", "uv", "cache", "dir"],
            capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())
    except Exception:
        return None


def _same_volume(path_a: Path, path_b: Path):
    """True/False once both sides can be resolved to a real device; None if that can't be
    determined — callers treat None as 'don't know, let uv decide for itself'."""
    if os.name == "nt":
        # normpath first so two spellings of the same path (trailing slash, mixed \\ and /,
        # case) still compare equal — splitdrive is a plain string op with no normalization
        # of its own. This does NOT unify a mapped drive letter with a UNC path pointing at
        # the same share (Z:\ vs \\server\share\ for the same target reads as "different" and
        # just costs a --link-mode copy we didn't strictly need, never a wrong answer) — that
        # needs an actual volume-identity API this module doesn't otherwise depend on, and the
        # failure mode is "unnecessarily cautious," not silently incorrect.
        da = os.path.splitdrive(os.path.normpath(str(path_a)))[0].lower()
        db = os.path.splitdrive(os.path.normpath(str(path_b)))[0].lower()
        return (da == db) if (da and db) else None

    def _existing_ancestor_dev(p: Path):
        # Neither side has to exist yet (a first-run uv cache, a not-yet-created venv) — walk
        # up to the nearest ancestor that does.
        p = Path(p).resolve()
        while not p.exists():
            if p.parent == p:
                return None
            p = p.parent
        return os.stat(p).st_dev

    dev_a = _existing_ancestor_dev(path_a)
    dev_b = _existing_ancestor_dev(path_b)
    return (dev_a == dev_b) if (dev_a is not None and dev_b is not None) else None


def _link_mode_args(python_path, venv_dir):
    """Decide --link-mode before running uv, instead of letting it discover mid-install that
    hardlinking is impossible and print its own scary fallback warning."""
    cache_dir = _uv_cache_dir(python_path)
    if cache_dir is None:
        return []  # couldn't ask uv — let it pick its own default and handle its own fallback
    if _same_volume(cache_dir, Path(venv_dir)) is False:
        print(f"Note: the uv package cache ({cache_dir}) and the venv ({venv_dir}) are on "
              "different drives, so uv will copy packages in rather than link them. Takes a "
              "little longer — nothing's wrong.")
        return ["--link-mode", "copy"]
    return []  # same volume (or undetermined) — let uv use its fast hardlink path


def _parse_requirements(requirements_path):
    """Split requirements.txt into (extra_index_url, torch_ecosystem_specs, other_lines).
    other_lines is the file's content minus the --extra-index-url line, unpinned lines and all
    — torch/torchvision stay in it deliberately, so the PyPI-only call still sees their exact
    pins and doesn't let some other package's loose "torch>=X" drag in a different, non-CUDA
    build from PyPI once the pinned build is no longer the only option resolution considers."""
    extra_index_url = None
    torch_specs = []
    other_lines = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for raw in f:
            code = raw.split("#", 1)[0].strip()
            if code.startswith("--extra-index-url"):
                if extra_index_url is not None:
                    # Silently keeping only the last one (or only the first) would drop an
                    # index some other package actually needs, with no error to point at why —
                    # this function only knows how to scope ONE extra index to the torch call
                    # and strip it everywhere else, so a second one has to be a loud failure,
                    # not a silent one, until this parsing is generalized to handle it.
                    raise ValueError(
                        f"{requirements_path} declares more than one --extra-index-url — "
                        "uv_install_deps.py only knows how to scope a single extra index to "
                        "the torch/torchvision install; update _parse_requirements to handle "
                        "multiple indexes before adding a second one.")
                parts = code.split(None, 1)
                if len(parts) == 2:
                    extra_index_url = parts[1]
                continue  # dropped — the PyPI-only call never sees this index at all
            other_lines.append(raw.rstrip("\n"))
            if code:
                # maxsplit by keyword: 3.13 deprecates passing it positionally to re.split, and
                # this runs on every update, so positionally it prints a DeprecationWarning above
                # the install output on every user's console.
                pkg = re.split(r"[=<>!~\s\[;]", code, maxsplit=1)[0]
                if pkg in TORCH_ECOSYSTEM:
                    torch_specs.append(code)
    return extra_index_url, torch_specs, other_lines


def install_requirements(requirements_path, venv_dir, python_path=None):
    """Install everything in requirements_path into the venv at venv_dir via uv. Returns True
    on success. Streams uv's own output straight to the console (nothing is captured) so a
    multi-GB torch download still shows live progress."""
    requirements_path = Path(requirements_path)
    venv_dir = Path(venv_dir)
    if python_path is None:
        python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python_path = Path(python_path)

    link_mode_args = _link_mode_args(python_path, venv_dir)
    extra_index_url, torch_specs, other_lines = _parse_requirements(requirements_path)

    # hqq ships as an sdist whose setup.py runs an `os.system` CUDA-kernel build during
    # egg_info unless DISABLE_CUDA=1 — slow where nvcc exists (pods), a harmless error
    # everywhere else, and Fizgig only uses its PyTorch path. uv's isolated build inherits
    # this environment, so the one variable covers every caller of this function.
    build_env = {**os.environ, "DISABLE_CUDA": "1"}

    try:
        # shell=False (list form) throughout — extra_index_url/torch_specs/link_mode_args all
        # come from this repo's own requirements.txt or uv's own `cache dir` output, not from
        # anything a caller passes through unsanitized.
        if extra_index_url and torch_specs:
            subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                [str(python_path), "-m", "uv", "pip", "install", *link_mode_args,
                 "--extra-index-url", extra_index_url, "--index-strategy", "unsafe-best-match",
                 *torch_specs],
                check=True)

        fd, filtered_path = tempfile.mkstemp(prefix="fizgig_requirements_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(other_lines) + "\n")
            subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                [str(python_path), "-m", "uv", "pip", "install", *link_mode_args,
                 "-r", filtered_path],
                check=True, env=build_env)
        finally:
            try:
                os.remove(filtered_path)
            except OSError:
                pass
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def main():
    here = Path(__file__).parent
    requirements_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "requirements.txt"
    venv_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "venv"
    # Run as a script, we're always invoked BY the target venv's own interpreter (every caller
    # does this) — sys.executable IS that interpreter, no need to reconstruct its path from
    # venv_dir and guess at Scripts/ vs bin/ naming.
    ok = install_requirements(requirements_path, venv_dir, python_path=sys.executable)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
