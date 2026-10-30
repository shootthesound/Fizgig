"""Anchor prompts for prompt sliders (#252): with none the slider step is exactly what it was, and an anchor adds its
own target - the frozen model's prediction for it - at +1 and -1, so training pulls the dial's effect on the anchor
toward zero. A toy model stands in for a DiT: prediction = x + c + multiplier * (w * c)."""
import json

import torch

from fizgig.families.train import _anchor_lines, _prompt_slider_step, _read_slider_live


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.5]))
        self.m = 1.0

    def set_trainable_multiplier(self, m):
        self.m = m


class _Driver:
    def __init__(self, net):
        self.net = net

    def noise_latents(self, latents, gen, min_t=None, max_t=None):
        return {"x": latents, "t": torch.tensor([0.5])}

    def predict(self, dit, state, c):
        return state["x"] + c["e"] + self.net.m * (self.net.w * c["e"])


def _enc(*vals):
    return [{"e": torch.tensor(v)} for v in vals]


def _run(anchor=None, weight=1.0):
    net = _Net()
    drv = _Driver(net)
    out = _prompt_slider_step(drv, None, net, torch.zeros(3), _enc([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]), None,
                              guidance=2.0, min_t=0, max_t=1, anchor=anchor, anchor_weight=weight)
    return out, net.w.grad.clone(), net.m


def test_no_anchor_is_the_old_step():
    (loss, t, a), grad, m = _run()
    assert a is None and m == 1.0
    # the old step, by hand: v_n + m*delta targets at m = +1 and -1, each loss halved
    w = torch.tensor([0.3, -0.2, 0.5], requires_grad=True)
    n, p, g = torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])
    delta = 2.0 * ((p) - (g))                     # frozen (m = 0) predictions: x + c
    ref = sum(0.5 * torch.nn.functional.mse_loss(n + m * (w * n), n + m * delta) for m in (1.0, -1.0))
    ref.backward()
    assert abs(loss - ref.item()) < 1e-6
    assert torch.allclose(grad, w.grad)


def test_anchor_pulls_its_change_toward_zero():
    anchor = {"e": torch.tensor([0., 0., 2.])}
    (loss, _t, a_loss), grad, m = _run(anchor)
    (_l0, _t0, _), grad0, _ = _run()
    assert m == 1.0
    # the anchor's change at +-1 is w * anchor: its loss is the mean of that squared, and its gradient on w[2]
    # (the only part the anchor touches) points to shrink it
    assert abs(a_loss - float(((torch.tensor([0.3, -0.2, 0.5]) * anchor["e"]) ** 2).mean())) < 1e-6
    extra = grad - grad0
    assert extra[0] == 0 and extra[1] == 0 and extra[2] > 0          # w[2] = 0.5 > 0: descent lowers it
    # weight scales the anchor's pull and nothing else
    (_l2, _t2, a2), grad2, _ = _run(anchor, weight=2.0)
    assert abs(a2 - a_loss) < 1e-9
    assert torch.allclose(grad2 - grad0, 2 * extra)


def test_anchor_lines():
    assert _anchor_lines("a photo of a man\n\n  a dog  \n") == ["a photo of a man", "a dog"]
    assert _anchor_lines(["x", " ", "y"]) == ["x", "y"]
    assert _anchor_lines(None) == []


def test_live_prompts_carry_anchors(tmp_path):
    (tmp_path / ".slider_prompts.json").write_text(json.dumps(
        {"base": "a photo of a woman,", "pos": "long hair", "neg": "short hair", "guidance": 3,
         "anchors": ["a photo of a man"]}), encoding="utf-8")
    got = _read_slider_live(str(tmp_path))
    assert got["anchors"] == ["a photo of a man"] and got["guidance"] == 3.0
    (tmp_path / ".slider_prompts.json").write_text(json.dumps(
        {"base": "b", "pos": "p", "neg": "n", "guidance": 1}), encoding="utf-8")
    assert _read_slider_live(str(tmp_path))["anchors"] == []      # an older panel's request: no anchors
