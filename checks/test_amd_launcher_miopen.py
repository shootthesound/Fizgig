"""Launch Fizgig AMD.bat points MIOpen at its kernel database: _rocm_sdk_devel\\bin, where the Windows ROCm wheels keep
it (#389, @scryptio), or a share\\miopen\\db tree where one exists (as on Linux, #384). gfx1100 skips the block."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

BAT = Path(__file__).resolve().parents[1] / "Launch Fizgig AMD.bat"


def _lines():
    return BAT.read_text(encoding="utf-8").splitlines()


def test_miopen_points_at_bin_or_a_database_folder():
    vals = [m.group(1) for ln in _lines() for m in [re.search(r'set "MIOPEN_SYSTEM_DB_PATH=([^"]*)"', ln)] if m]
    assert vals, "the launcher no longer sets MIOPEN_SYSTEM_DB_PATH"
    assert vals[0] == "!ROCM_DEVEL_BIN!", "the Windows wheels' bin folder first, so a share tree overrides it"
    for v in vals[1:]:
        assert v.lower().endswith(r"\share\miopen\db"), v
    assert "_rocm_sdk_libraries" in vals[-1], "the libraries database should win (set last)"


def _block():
    """The launcher's database block: from the site-packages line to the end of the gfx1100 if."""
    lines = _lines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('set "ROCM_SP='))
    end = next(i for i in range(start, len(lines)) if lines[i] == ")")
    return lines[start:end + 1]


@pytest.mark.skipif(sys.platform != "win32", reason="runs the batch file's own lines in cmd")
@pytest.mark.parametrize("have,arch,want", [
    (("libraries",), "gfx1201", "_rocm_sdk_libraries"),
    (("devel",), "gfx1201", "_rocm_sdk_devel"),
    (("libraries", "devel"), "gfx1201", "_rocm_sdk_libraries"),
    ((), "gfx1201", "bin"),         # the Windows wheels' layout: no share tree, the database in bin
    (("libraries", "devel"), "gfx1100", None),
])
def test_miopen_path_on_venv_trees(tmp_path, have, arch, want):
    sp = tmp_path / "venv" / "Lib" / "site-packages"
    (sp / "_rocm_sdk_devel" / "bin" / "rocblas" / "library").mkdir(parents=True)
    for h in have:
        (sp / f"_rocm_sdk_{h}" / "share" / "miopen" / "db").mkdir(parents=True)
    bat = tmp_path / "probe.bat"
    bat.write_text("\r\n".join(["@echo off", "setlocal enabledelayedexpansion", "set MIOPEN_SYSTEM_DB_PATH=",
                                f'set "GPU_ARCH={arch}"', *_block(),
                                "echo MIOPEN=[!MIOPEN_SYSTEM_DB_PATH!]", "echo TENSILE=[!ROCBLAS_TENSILE_DB_PATH!]"]) + "\r\n",
                   encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MIOPEN_", "ROCBLAS_"))}
    out = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True, env=env, timeout=30).stdout
    got = re.search(r"MIOPEN=\[(.*)\]", out).group(1)
    if want is None:
        assert got == "", out
    elif want == "bin":
        assert Path(got) == sp / "_rocm_sdk_devel" / "bin", out
    else:
        assert Path(got) == sp / f"{want}" / "share" / "miopen" / "db", out
    tensile = re.search(r"TENSILE=\[(.*)\]", out).group(1)
    if arch != "gfx1100":
        assert Path(tensile) == sp / "_rocm_sdk_devel" / "bin" / "rocblas", out
