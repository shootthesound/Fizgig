"""A model path that isn't a .safetensors file (a GGUF quant, say) is refused from its first eight bytes. A GGUF's magic
and version read as a header length of ~14 GB, and every header read - the memory plans, int8 / pruned checks, the
loaders - asked for that much before failing: seconds and gigabytes on a real 15-20 GB file, at each plan."""
import builtins
import json
import struct

import pytest
import torch
from safetensors.torch import save_file

GGUF_HEAD = b"GGUF" + struct.pack("<I", 3)          # read as <Q: 14,064,895,815


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "model-Q4_K_M.gguf"
    p.write_bytes(GGUF_HEAD + b"\0" * 4096)
    return str(p)


@pytest.fixture
def asked(monkeypatch, gguf):
    """Every read size requested from the GGUF file."""
    sizes, real_open = [], builtins.open

    class Rec:
        def __init__(self, f):
            self.f = f

        def read(self, n=-1):
            sizes.append(n)
            return self.f.read(n)

        def __getattr__(self, a):
            return getattr(self.f, a)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    def opener(file, *a, **k):
        f = real_open(file, *a, **k)
        return Rec(f) if str(file) == gguf else f

    monkeypatch.setattr(builtins, "open", opener)
    return sizes


def _small(sizes):
    assert sizes, "the file was never read"
    assert max(sizes) <= 8, f"asked for {max(sizes):,} bytes of a file that isn't a safetensors file"


def test_planning_reads_refuse_a_gguf(gguf, asked):
    from fizgig.families import launch
    from fizgig.minimax import common
    from fizgig.minimax_image.model import PDD_SIGMAS, file_pdd_sigmas
    assert common.adapter_param_count(gguf, []) == 0
    assert common.frozen_lora_vram_gb(gguf) == 0.0
    assert common.is_pruned_checkpoint(gguf) is False
    assert common.has_int8_codes(gguf) is False
    assert common.int8_loadable(gguf) is False
    assert launch.ft_continuation(gguf) == (0, 0, 0)
    assert file_pdd_sigmas(gguf) == PDD_SIGMAS
    _small(asked)


def test_target_shapes_refuse_a_gguf(gguf, asked):
    from fizgig.families import quant

    class Driver:
        def block_map(self, _):
            return []

    assert quant._target_shapes(Driver(), gguf) == []
    _small(asked)


@pytest.mark.parametrize("mod", ["fizgig.utils.safetensors", "fizgig.krea2.safetensors_utils"])
def test_loaders_refuse_a_gguf(gguf, asked, mod):
    import importlib
    with pytest.raises(ValueError):
        importlib.import_module(mod).MemoryEfficientSafeOpen(gguf)
    _small(asked)


def test_a_length_past_the_end_is_refused(tmp_path):
    """A header length under the 100 MB cap but longer than the file: a cut-off download, or another format."""
    from fizgig.utils.st_meta import read_header
    p = tmp_path / "cut.safetensors"
    p.write_bytes(struct.pack("<Q", 5000) + b"{}")
    with pytest.raises(ValueError):
        read_header(str(p))


def test_real_files_still_read(tmp_path):
    from fizgig.minimax import common
    from fizgig.minimax_image.model import file_pdd_sigmas
    from fizgig.utils.safetensors import MemoryEfficientSafeOpen
    from fizgig.utils.st_meta import read_header
    p = str(tmp_path / "m.safetensors")
    save_file({"adaln_t_table": torch.zeros(2, 3), "x.weight": torch.zeros(4, 4)}, p,
              metadata={"config": json.dumps({"transformer": {"pdd_sigmas": [1.0, 0.5, 0.0]}})})
    assert set(read_header(p)) >= {"adaln_t_table", "x.weight", "__metadata__"}
    assert common.is_pruned_checkpoint(p) is True
    assert file_pdd_sigmas(p) == (1.0, 0.5, 0.0)
    with MemoryEfficientSafeOpen(p) as f:
        assert sorted(f.keys()) == ["adaln_t_table", "x.weight"]
        assert f.get_tensor("x.weight").shape == (4, 4)
