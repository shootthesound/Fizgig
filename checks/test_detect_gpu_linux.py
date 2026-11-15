"""The Linux AMD installer's GPU detection reads the card's own gfx target from rocminfo, never the generic ISA
family listed under it (an MI350X read as "gfx9" and the torch install failed)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "detect_gpu_linux", Path(__file__).resolve().parents[1] / "src" / "install" / "detect_gpu_linux.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _rocminfo(name, generic):
    return f"""HSA Agents
*******
Agent 1
*******
  Name:                    AMD EPYC 9755 128-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    {name}
  Marketing Name:          Some AMD GPU
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--{name}:sramecc+:xnack-
    ISA 2
      Name:                    amdgcn-amd-amdhsa--{generic}
*** Done ***
"""


def test_card_target_not_the_generic_isa(monkeypatch):
    for name, generic in (("gfx950", "gfx9-4-generic:sramecc+:xnack-"), ("gfx1201", "gfx12-generic"),
                          ("gfx1100", "gfx11-generic"), ("gfx90a", "gfx9-generic:xnack-")):
        monkeypatch.setattr(D, "_run", lambda cmd, timeout=8, t=_rocminfo(name, generic): t)
        assert D._gfx_from_rocminfo() == [name]


def test_generic_names_never_match():
    for s in ("gfx9-4-generic", "gfx12-generic", "gfx11-generic", "gfx9"):
        assert D._GFX_RE.search(s) is None
    for s in ("gfx950", "gfx942", "gfx90a", "gfx1201", "gfx1036", "amdgcn-amd-amdhsa--gfx1100"):
        assert D._GFX_RE.search(s)
