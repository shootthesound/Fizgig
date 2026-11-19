"""#379: Fizgig's workers start in its folder (python -m fizgig.scripts.…), where `import fizgig` finds the launcher
fizgig.py before the package. Its top must run there on Linux ROCm too: an import through fizgig.utils failed with
"'fizgig' is not a package" and stopped every caption, cache and training worker."""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(sys.platform == "win32", reason="the ROCm branch is Linux-only (and needs a symlink)")
def test_launcher_imported_as_fizgig_on_rocm(tmp_path):
    shutil.copy(os.path.join(ROOT, "fizgig.py"), tmp_path / "fizgig.py")     # a copy: its user_data is the temp one
    os.symlink(os.path.join(ROOT, "src"), tmp_path / "src")
    env = dict(os.environ, FIZGIG_GPU_BACKEND="rocm", PYTHONPATH=str(tmp_path / "src"))
    env.pop("FIZGIG_NO_EXPANDABLE", None)
    # a worker's start: -m a module under fizgig.scripts (a light one, no torch), in the folder fizgig.py is in
    r = subprocess.run([sys.executable, "-m", "fizgig.scripts.analyze_loss_log", "--help"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert "not a package" not in r.stderr and r.returncode == 0, r.stderr
