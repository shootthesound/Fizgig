"""The LoKR forward (two GEMMs over all tokens, #307) equals x @ kron(w1, w2)ᵀ, for any leading shape, with gradients."""
import torch

from fizgig.networks.lora import _lokr_forward_update


def _ref(x, w1, w2):
    return x @ torch.kron(w1, w2).t()


def test_matches_kron_any_leading_shape():
    torch.manual_seed(0)
    a, b, c, d = 3, 4, 5, 6
    w1, w2 = torch.randn(a, b, dtype=torch.float64), torch.randn(c, d, dtype=torch.float64)
    for lead in [(), (7,), (2, 7), (2, 3, 7)]:
        x = torch.randn(*lead, b * d, dtype=torch.float64)
        assert torch.allclose(_lokr_forward_update(x, w1, w2, a, b, c, d), _ref(x, w1, w2))


def test_gradients_match_kron():
    torch.manual_seed(1)
    a, b, c, d = 2, 3, 4, 5
    args = [torch.randn(2, 6, b * d, dtype=torch.float64), torch.randn(a, b, dtype=torch.float64),
            torch.randn(c, d, dtype=torch.float64)]
    g = []
    for f in (lambda x, w1, w2: _lokr_forward_update(x, w1, w2, a, b, c, d), _ref):
        ts = [t.clone().requires_grad_() for t in args]
        (f(*ts) ** 2).sum().backward()
        g.append([t.grad for t in ts])
    for p, q in zip(*g):
        assert torch.allclose(p, q)


def test_non_contiguous_input():
    a, b, c, d = 2, 2, 3, 3
    w1, w2 = torch.randn(a, b, dtype=torch.float64), torch.randn(c, d, dtype=torch.float64)
    x = torch.randn(b * d, 5, dtype=torch.float64).t()               # (5, b·d), not contiguous
    assert torch.allclose(_lokr_forward_update(x, w1, w2, a, b, c, d), _ref(x, w1, w2))
