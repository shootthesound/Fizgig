"""#404 and the VAEs on AMD: modules/rocm_conv computes a large convolution in bands of output rows, each band given
exactly the input rows it reads, so the result is the same convolution with the work buffer bounded. Checked against
PyTorch's own F.conv2d / F.conv3d over strides, paddings, kernels, groups and the edges; installed only on ROCm."""
import itertools

import pytest
import torch
import torch.nn.functional as F

from fizgig.modules import rocm_conv


@pytest.fixture
def tiny_bands(monkeypatch):
    monkeypatch.setattr(rocm_conv, "BAND_BYTES", 2048)       # every case below splits into several bands


CASES_2D = list(itertools.product((1, 2), (0, 1, (2, 1)), (1, 3, 4), (1, 2), (True, False)))


@pytest.mark.parametrize("stride,padding,k,groups,bias", CASES_2D)
def test_conv2d_bands_match(tiny_bands, stride, padding, k, groups, bias):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 23, 17, dtype=torch.float64)
    w = torch.randn(6, 4 // groups, k, k, dtype=torch.float64)
    b = torch.randn(6, dtype=torch.float64) if bias else None
    want = F.conv2d(x, w, b, stride, padding, 1, groups)
    got = rocm_conv._banded(F.conv2d, 2)(x, w, b, stride, padding, 1, groups)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("stride,padding", [(1, 1), (1, (2, 1, 1)), (2, 1), ((1, 2, 2), 0)])
def test_conv3d_bands_match(tiny_bands, stride, padding):
    torch.manual_seed(1)
    x = torch.randn(1, 3, 3, 19, 13, dtype=torch.float64)
    w = torch.randn(5, 3, 3, 3, 3, dtype=torch.float64)
    want = F.conv3d(x, w, None, stride, padding)
    got = rocm_conv._banded(F.conv3d, 3)(x, w, None, stride, padding)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


def test_small_convolutions_untouched(monkeypatch):
    calls = []
    fn = rocm_conv._banded(lambda *a: calls.append(a[0].shape) or F.conv2d(*a), 2)
    fn(torch.randn(1, 2, 8, 8), torch.randn(2, 2, 3, 3), None, 1, 1, 1, 1)
    assert calls == [torch.Size([1, 2, 8, 8])]                    # one call, the whole input


def test_installed_only_on_rocm(monkeypatch):
    rocm_conv.uninstall()
    monkeypatch.setattr(torch.version, "hip", None)
    assert not rocm_conv.install() and not getattr(F.conv2d, "_fizgig_banded", False)
    monkeypatch.setattr(torch.version, "hip", "7.15")
    monkeypatch.setenv("FIZGIG_ROCM_BANDS", "0")
    assert not rocm_conv.install()
    monkeypatch.setenv("FIZGIG_ROCM_BANDS", "1")
    try:
        assert rocm_conv.install() and F.conv2d._fizgig_banded and F.conv3d._fizgig_banded
        conv = torch.nn.Conv2d(3, 4, 3, padding=1)                 # nn.Conv2d goes through it
        x = torch.randn(1, 3, 9, 9)
        torch.testing.assert_close(conv(x), rocm_conv._orig["conv2d"](x, conv.weight, conv.bias, 1, 1))
    finally:
        rocm_conv.uninstall()
    assert not getattr(F.conv2d, "_fizgig_banded", False)


def test_every_family_driver_installs_it():
    import inspect
    from fizgig.families.description import FamilyDescription
    assert "rocm_conv.install()" in inspect.getsource(FamilyDescription.load_driver)
