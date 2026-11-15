"""AMD (ROCm): a key-padding mask that broadcasts over the queries is expanded before SDPA (ROCm's fused kernel
attends wrongly with the broadcast one: #341, #343). Simulated here: the wrapper's expansion and its no-op elsewhere."""
import importlib

import torch
import torch.nn.functional as F

from fizgig.utils import gpu_backend


def _install(monkeypatch):
    seen = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(torch.version, "hip", "test", raising=False)

    def fake(q, k, v, attn_mask=None, *a, **kw):
        seen.append(None if attn_mask is None else tuple(attn_mask.shape))
        return real(q, k, v, attn_mask=attn_mask, *a, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake)
    gpu_backend.fix_rocm_sdpa_masks()
    return seen


def test_query_broadcast_mask_is_expanded_and_result_unchanged(monkeypatch):
    q, k, v = (torch.randn(2, 4, 9, 8) for _ in range(3))
    m = torch.ones(2, 1, 1, 9, dtype=torch.bool)
    m[0, ..., 6:] = False
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    seen = _install(monkeypatch)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    assert seen == [(2, 1, 9, 9)]
    torch.testing.assert_close(got, want)


def test_full_masks_none_and_installed_once(monkeypatch):
    seen = _install(monkeypatch)
    q = torch.randn(1, 2, 5, 4)
    F.scaled_dot_product_attention(q, q, q)
    F.scaled_dot_product_attention(q, q, q, attn_mask=torch.ones(1, 1, 5, 5, dtype=torch.bool))
    first = F.scaled_dot_product_attention
    gpu_backend.fix_rocm_sdpa_masks()
    assert F.scaled_dot_product_attention is first
    assert seen == [None, (1, 1, 5, 5)]


def test_nvidia_untouched(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    before = F.scaled_dot_product_attention
    gpu_backend.fix_rocm_sdpa_masks()
    assert F.scaled_dot_product_attention is before
