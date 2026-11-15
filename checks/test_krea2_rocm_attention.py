"""Krea 2 on AMD (ROCm): attention never sends a key-padding mask to SDPA there (#341 / #343: masked attention
attended wrongly on ROCm). A uniform batch trims exactly; a ragged batch goes per sample. Same result as the mask."""
import torch
import torch.nn.functional as F

from fizgig.krea2 import attention as A


def _ref(q, k, v, lens):
    out = torch.zeros(q.shape[0], q.shape[1], q.shape[2] * q.shape[3])
    for i, n in enumerate(lens):
        o = F.scaled_dot_product_attention(q[i:i + 1, :n].transpose(1, 2), k[i:i + 1, :n].transpose(1, 2),
                                           v[i:i + 1, :n].transpose(1, 2))
        out[i, :n] = o.transpose(1, 2).flatten(2)[0]
    return out


def test_ragged_batch_goes_per_sample_on_rocm(monkeypatch):
    monkeypatch.setattr(A, "_HIP", True)
    calls = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        calls.append(attn_mask is None)
        return real(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    torch.manual_seed(3)
    img, txt = 6, 5
    q, k, v = (torch.randn(2, img + txt, 4, 8) for _ in range(3))
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    p = A.AttentionParams.create_attention_params_from_mask("torch", False, img, mask)
    with torch.no_grad():
        got = A.attention(q, k, v, p)
    assert calls and all(calls)                       # no mask ever reached SDPA
    want = _ref(q, k, v, (img + 2, img + 4))
    for i, n in enumerate((img + 2, img + 4)):
        torch.testing.assert_close(got[i, :n], want[i, :n])


def test_nvidia_keeps_the_masked_path(monkeypatch):
    monkeypatch.setattr(A, "_HIP", False)
    seen = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                        lambda q, k, v, attn_mask=None, **kw: (seen.append(attn_mask is not None),
                                                               real(q, k, v, attn_mask=attn_mask, **kw))[1])
    q, k, v = (torch.randn(2, 11, 4, 8) for _ in range(3))
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    with torch.no_grad():
        A.attention(q, k, v, A.AttentionParams.create_attention_params_from_mask("torch", False, 6, mask))
    assert seen == [True]
