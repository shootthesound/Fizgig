"""Centralized path resolution for the Fizgig repo.

Instead of hardcoding a fixed number of "os.path.dirname" calls (which breaks
silently when a module is moved to a deeper sub-package), we walk up from this
file until we find the directory that contains "lora_trainer_gui.py" — the
canonical marker of the repository root.
"""
import os


def _find_repo_root() -> str:
    """Walk up from this file's directory until "lora_trainer_gui.py" is found."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(current, "lora_trainer_gui.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root without finding the marker
            raise RuntimeError(
                f"""Cannot locate repo root: no "lora_trainer_gui.py" found walking up from {current}"""
            )
        current = parent


REPO_ROOT: str = _find_repo_root()

# Backward-compatible alias used throughout the codebase.
FIZGIG_DIR: str = REPO_ROOT