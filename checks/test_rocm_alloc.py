"""AMD expandable segments: on for datacenter Instinct cards only; off on consumer Radeons, where they corrupt memory
(#363), on an unknown card, and with FIZGIG_NO_EXPANDABLE=1."""
from fizgig.utils import rocm_alloc as R


def test_kernel_version_names():
    assert R._arch(90402) == "gfx942"
    assert R._arch(90010) == "gfx90a"
    assert R._arch(120001) == "gfx1201"
    assert R._arch(110000) == "gfx1100"
    assert R._arch(90500) == "gfx950"


def test_datacenter_only(monkeypatch):
    monkeypatch.delenv("FIZGIG_NO_EXPANDABLE", raising=False)
    for arch in ("gfx942", "gfx950", "gfx90a", "gfx908"):
        assert R.expandable_ok([arch])
    for arch in ("gfx1201", "gfx1200", "gfx1100", "gfx1030", "gfx1151", "gfx906"):
        assert not R.expandable_ok([arch])
    assert not R.expandable_ok([])                       # a card it can't name
    assert not R.expandable_ok(["gfx942", "gfx1151"])   # an Instinct beside a Radeon / APU: off


def test_in_effect_reads_the_names_rocm_reads(monkeypatch):
    for k in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(k, raising=False)
    assert not R.expandable_in_effect()                  # Windows AMD, a Linux Radeon: the tight-fit warning shows
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert not R.expandable_in_effect()                  # ROCm ignores the CUDA name
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")
    assert R.expandable_in_effect()


def test_opt_out_and_installer_name(monkeypatch):
    monkeypatch.setenv("FIZGIG_AMD_GFX", "gfx942")
    monkeypatch.delenv("FIZGIG_NO_EXPANDABLE", raising=False)
    assert R.amd_archs() == ["gfx942"] and R.expandable_ok()
    monkeypatch.setenv("FIZGIG_NO_EXPANDABLE", "1")
    assert not R.expandable_ok()
    monkeypatch.setenv("FIZGIG_AMD_GFX", "gfx1201")
    monkeypatch.delenv("FIZGIG_NO_EXPANDABLE")
    assert not R.expandable_ok()
