"""#342: on AMD, the launcher's HIP_VISIBLE_DEVICES pick becomes the current device when the ROCm build ignores it."""
import torch

from fizgig.utils import gpu_backend


def _fake(monkeypatch, n, rocm=True):
    calls = []
    monkeypatch.setattr(gpu_backend, "is_rocm", lambda: rocm)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: f"dev{i}")
    return calls


def test_ignored_variable_pins_the_chosen_card(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    calls = _fake(monkeypatch, 2)
    gpu_backend.pin_rocm_device()
    assert calls == [1]


def test_applied_variable_leaves_the_only_device(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    calls = _fake(monkeypatch, 1)
    gpu_backend.pin_rocm_device()
    assert calls == []


def test_nvidia_and_unset_or_lists_are_untouched(monkeypatch):
    calls = _fake(monkeypatch, 2, rocm=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    gpu_backend.pin_rocm_device()
    assert calls == []
    calls = _fake(monkeypatch, 2)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    gpu_backend.pin_rocm_device()
    assert calls == []


class _Props:
    def __init__(self, name, gb, integrated=False):
        self.name, self.total_memory, self.is_integrated = name, int(gb * 2 ** 30), integrated


def _unset(monkeypatch):
    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)


def test_nothing_set_picks_the_card_over_built_in_graphics(monkeypatch):
    """#342: the launcher's pick never ran (its for /f lost its quotes), so torch's device 0 was the built-in graphics."""
    _unset(monkeypatch)
    calls = _fake(monkeypatch, 2)
    props = [_Props("AMD Radeon(TM) Graphics", 0.5), _Props("AMD Radeon RX 7900 XT", 20)]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props[i])
    gpu_backend.pin_rocm_device()
    assert calls == [1]


def test_nothing_set_leaves_a_card_already_first(monkeypatch):
    _unset(monkeypatch)
    calls = _fake(monkeypatch, 2)
    props = [_Props("AMD Radeon RX 7900 XT", 20), _Props("AMD Radeon(TM) Graphics", 0.5, integrated=True)]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props[i])
    gpu_backend.pin_rocm_device()
    assert calls == []


def test_a_user_pick_by_another_variable_wins(monkeypatch):
    _unset(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = _fake(monkeypatch, 2)
    gpu_backend.pin_rocm_device()
    assert calls == []


def test_out_of_range_index_is_ignored(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    calls = _fake(monkeypatch, 2)
    gpu_backend.pin_rocm_device()
    assert calls == []


def test_every_amd_process_says_what_it_sees(monkeypatch, capsys):
    """#342: each process's line (stderr, so a stage's is in its run log): the variables, the devices, the one used."""
    _fake(monkeypatch, 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu_backend.report_rocm_devices()
    err = capsys.readouterr().err
    assert "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=-" in err and "2 devices (0: dev0, 1: dev1); using 1, dev1" in err
    _fake(monkeypatch, 2, rocm=False)
    gpu_backend.report_rocm_devices()
    assert capsys.readouterr().err == ""                 # NVIDIA: nothing
