"""Adaptive LR's learned ceiling: a cut soon after a probe up, from the rate that probe set, caps later probes one
step below it - replayed on the epoch losses of Peter's Qwen Image 2.1 Muon run (6 Nov 2026: 2.83e-4 -> 3.54e-4,
plateau, cut to the 2e-4 floor), then a run of improving epochs that would otherwise climb back past 3.54e-4."""
import pytest

torch = pytest.importorskip("torch")

from fizgig.training.adaptive_lr import AdaptiveLR  # noqa: E402


def _run(losses, lo=2e-4, hi=4e-4, state=None):
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=(lo * hi) ** 0.5)
    a = AdaptiveLR(lo, hi)
    if state:
        a.load_state_dict(state["a"])
        for g in opt.param_groups:
            g["lr"] = state["lr"]
    lrs = []
    for e, loss in enumerate(losses, start=state["epoch"] if state else 0):
        a.epoch_boundary(e, loss, net, opt)
        lrs.append(opt.param_groups[0]["lr"])
    return a, opt, lrs


QWEN = [0.2269, 0.2134, 0.2093, 0.2122, 0.2191, 0.2270]          # epochs 1-6 of the run
BETTER = [0.199 - 0.004 * i for i in range(14)]                    # then steady improvement


def test_cut_after_probe_learns_a_ceiling():
    a, _, lrs = _run(QWEN)
    assert lrs[2] == pytest.approx(3.536e-4, rel=1e-3)            # epoch 3 probes up
    assert lrs[5] == pytest.approx(2e-4)                           # epoch 6 cuts to the floor
    assert a.ceiling == pytest.approx(3.536e-4 / 1.25, rel=1e-3)  # 2.83e-4


def test_later_probes_stop_at_the_ceiling():
    a, _, lrs = _run(QWEN + BETTER)
    assert max(lrs[6:]) == pytest.approx(a.ceiling, rel=1e-3)
    assert max(lrs[6:]) < 3.5e-4


def test_without_a_cut_nothing_changes():
    a, _, lrs = _run([0.30 - 0.01 * i for i in range(20)])        # always improving: climbs to max_lr as before
    assert a.ceiling is None
    assert max(lrs) == pytest.approx(4e-4)


def test_ceiling_survives_resume():
    a, opt, _ = _run(QWEN)
    state = {"a": a.state_dict(), "lr": opt.param_groups[0]["lr"], "epoch": len(QWEN)}
    b, _, lrs = _run(BETTER, state=state)
    assert b.ceiling == pytest.approx(a.ceiling)
    assert max(lrs) == pytest.approx(a.ceiling, rel=1e-3)
