"""Adaptive LR judges plateau / improving on the mean loss of the last 3 epochs: an epoch's loss jitters, and read
one epoch at a time that jitter halved the LR down to Min (a Muon Krea 2 run, 28 Oct 2026, floored by epoch 9)."""
import torch

from fizgig.training.adaptive_lr import AdaptiveLR

RUN = [.1498, .1542, .1445, .1477, .1400, .1383, .1362, .1409, .1384, .1310, .1378, .1360, .1324, .1312, .1370,
       .1240, .1301, .1226, .1236, .1274, .1210, .1273, .1254, .1279, .1194, .1162, .1327, .1181, .1269, .1185]


def _replay(losses, state=None):
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=2.828e-4)
    a = AdaptiveLR(1e-4, 8e-4)
    lrs = []
    for e, loss in enumerate(losses):
        a.epoch_boundary(e, loss, net, opt)
        lrs.append(opt.param_groups[0]["lr"])
    return a, lrs


def test_jittery_but_falling_loss_does_not_walk_the_lr_to_min():
    _a, lrs = _replay(RUN)
    assert min(lrs[:20]) > 1e-4 and max(lrs) > 2.828e-4          # it climbed, and wasn't floored early


def test_the_recent_losses_survive_a_resume():
    a, _ = _replay(RUN[:5])
    b = AdaptiveLR(1e-4, 8e-4)
    b.load_state_dict(a.state_dict())
    assert b.recent_losses == a.recent_losses and len(b.recent_losses) == 3
