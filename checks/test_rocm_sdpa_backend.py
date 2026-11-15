"""Backend selection only; GPU speed and peak memory need a ROCm benchmark."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend

from fizgig.modules import sdpa


@pytest.mark.parametrize("grad", [True, False])
def test_rocm_keeps_default_backends_without_probing_cudnn(monkeypatch, grad):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    monkeypatch.setattr(sdpa, "_init_ctx", lambda: pytest.fail("cuDNN probed on ROCm"))
    with torch.set_grad_enabled(grad):
        with sdpa.sdpa_backend_ctx(), sdpa.masked_ctx(SimpleNamespace(is_cuda=True)):
            pass


def test_nvidia_masked_priority_is_unchanged(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("FIZGIG_SDPA_BACKEND", "auto")
    calls = []

    def kernel(backends, *, set_priority):
        calls.append((backends, set_priority))
        from contextlib import nullcontext
        return nullcontext()

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", kernel)
    with sdpa.masked_ctx(SimpleNamespace(is_cuda=True)):
        pass
    assert calls == [([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], True)]


def test_nvidia_inference_still_uses_resolved_backend(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("FIZGIG_SDPA_BACKEND", "auto")
    from contextlib import nullcontext
    calls = []

    def context():
        calls.append(True)
        return nullcontext()

    monkeypatch.setattr(sdpa, "_SDPA_CTX", context)
    with torch.no_grad(), sdpa.sdpa_backend_ctx():
        pass
    assert calls == [True]
