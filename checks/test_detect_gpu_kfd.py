"""The Linux AMD installer's fallback card reading: the kernel driver's topology file holds gfx_target_version as a bare
number, which becomes the card's gfx name (it used to look for a gfx name after the number, which the file never has)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "detect_gpu_linux", os.path.join(os.path.dirname(__file__), "..", "src", "install", "detect_gpu_linux.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def test_kernel_numbers_become_gfx_names():
    assert D._gfx_name(90402) == "gfx942"
    assert D._gfx_name(90010) == "gfx90a"
    assert D._gfx_name(110001) == "gfx1101"
    assert D._gfx_name(120001) == "gfx1201"
    assert D._gfx_name(100300) == "gfx1030"


def test_topology_files(tmp_path, monkeypatch):
    for n, version in enumerate((0, 120001)):          # a CPU node (0), then an RX 9070 XT
        node = tmp_path / str(n)
        node.mkdir()
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\nvendor_id 4098\n")
    real = D.glob.glob
    monkeypatch.setattr(D.glob, "glob", lambda p: real(str(tmp_path / "*" / "properties"))
                        if p.startswith("/sys/class/kfd") else real(p))
    assert D._gfx_from_kfd_topology() == ["gfx1201"]
