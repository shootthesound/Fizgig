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
    for v in ("", "0,1"):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", v)
        gpu_backend.pin_rocm_device()
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    gpu_backend.pin_rocm_device()
    assert calls == []


def test_out_of_range_index_is_ignored(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    calls = _fake(monkeypatch, 2)
    gpu_backend.pin_rocm_device()
    assert calls == []
