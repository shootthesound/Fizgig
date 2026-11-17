"""#353: Automagic v4's shown rate is its working rate - its group rate times its mean brake (balance), read from its
state - not the flat cap; other optimizers show what they did before."""
import torch

from fizgig.training import optimizers as O
from fizgig.training.automagic4 import Automagic4


def _trained(steps=12):
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(16, 8) * 0.1)
    b = torch.nn.Parameter(torch.zeros(8))
    opt = Automagic4([a, b], lr=2.5e-4)
    for _ in range(steps):
        loss = ((torch.randn(4, 16) @ a + b) ** 2).mean() * torch.tensor([1.0, 50.0]).repeat(4)[:8].mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return opt


def test_brakes_are_a_share_and_the_shown_rate_is_under_the_rate():
    opt = _trained()
    brakes = O.automagic4_brakes(opt)
    assert len(brakes) == 1
    bal, braked = brakes[0]
    assert 0.0 < bal <= 1.0 and 0.0 <= braked <= 1.0
    rate = opt.get_avg_learning_rate()
    assert O.shown_lr(opt) == rate * bal
    assert "mean brake" in O.automagic4_detail(opt)


def test_other_optimizers_are_unchanged():
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p], lr=1e-4)
    assert O.automagic4_brakes(opt) is None and O.shown_lr(opt) == O.optimizer_lr(opt) == 1e-4
    assert O.automagic4_detail(opt) == ""
