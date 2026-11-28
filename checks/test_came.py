"""CAME is back (#416): the vendored copy builds and steps, it is in the Optimizer Type list, the Learning Rate box /
Adaptive LR / the scheduler set its rate (it owns none), and Muon's args left in the box from a preset are dropped."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from fizgig.families import form, registry  # noqa: E402
from fizgig.training.came import CAME  # noqa: E402
from fizgig.training.optimizers import (  # noqa: E402
    MUON_DEFAULT_ARGS,
    available_optimizers,
    create_optimizer,
    optimizer_lr,
    owns_its_rate,
)


def test_came_listed_and_offered_on_the_form():
    assert "came" in available_optimizers() and "came" in available_optimizers(probe=False)
    rows = {r["key"]: r for s in form.form(registry.get("moss_image"))["sections"] for r in s["rows"] if r}
    assert "came" in rows["OPTIMIZER_TYPE"]["choices"]


def test_came_is_the_vendored_copy_and_steps():
    p = nn.Parameter(torch.zeros(64, 8))
    opt, label = create_optimizer("came", [p], 5e-5)
    assert isinstance(opt, CAME) and label == "came"
    for _ in range(3):
        p.grad = torch.randn(64, 8)
        opt.step()
    assert p.abs().sum() > 0 and torch.isfinite(p).all()
    q = nn.Parameter(torch.zeros(5))                     # a 1-D parameter takes the unfactored path
    o1, _ = create_optimizer("came", [q], 1e-4)
    q.grad = torch.randn(5)
    o1.step()
    assert q.abs().sum() > 0


def test_came_rate_is_the_one_given():
    p = nn.Parameter(torch.zeros(16, 4))
    opt, _ = create_optimizer("came", [p], 1e-4)
    assert not owns_its_rate(opt) and optimizer_lr(opt) == pytest.approx(1e-4)
    opt.param_groups[0]["lr"] = 2e-4                    # what Adaptive LR / the scheduler write
    p.grad = torch.ones(16, 4)
    before = p.detach().clone()
    opt.step()
    zero_rate = nn.Parameter(torch.zeros(16, 4))
    o0, _ = create_optimizer("came", [zero_rate], 1e-4)
    o0.param_groups[0]["lr"] = 0.0
    zero_rate.grad = torch.ones(16, 4)
    o0.step()
    assert not torch.equal(p.detach(), before) and torch.equal(zero_rate.detach(), torch.zeros(16, 4))


def test_came_drops_muons_leftover_args_and_takes_its_own():
    p = nn.Parameter(torch.zeros(8, 8))
    opt, label = create_optimizer("came", [p], 1e-4, MUON_DEFAULT_ARGS)
    assert isinstance(opt, CAME) and label == "came"
    opt, _ = create_optimizer("came", [nn.Parameter(torch.zeros(8, 8))], 1e-4, "weight_decay=0.01")
    assert isinstance(opt, CAME) and opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)
