"""MiniMax H3 Image Speed boost (Samples tab and Studio's own pick): the first N steps run on a half-size latent, the
rest at full size, at least one step always at full size; Off renders exactly as before. Run on a stand-in DiT."""
import torch
from torch import nn

from fizgig.families.registry import get
from fizgig.minimax_image import sampling as S
from fizgig.minimax_image.model import PDD_SIGMAS


class _Head:
    pdd_sigmas = list(PDD_SIGMAS)
    span_override = None


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.final_layer = _Head()
        self.seen = []

    def forward(self, x, t, txt, **kw):
        self.seen.append(tuple(x.shape[-2:]))
        return -0.5 * x.float()


def _run(boost, sampler="euler", steps=6):
    dit = _DiT()
    z = S.sample(dit, torch.zeros(4, 8), 256, 256, steps=steps, seed=3, sampler=sampler, boost=boost)
    return z, dit.seen


def test_the_first_steps_run_at_half_size():
    for sampler in ("euler", "lcm"):
        for k in (1, 2, 4):
            z, seen = _run(k, sampler)
            assert seen == [(8, 8)] * k + [(16, 16)] * (6 - k), (sampler, k)
            assert z.shape[-2:] == (16, 16) and torch.isfinite(z).all()


def test_at_least_one_full_size_step_and_off_is_unchanged():
    assert S.boost_steps(6, 9) == 5 and S.boost_steps(6, 0) == 0 and S.boost_steps(1, 3) == 0
    assert S.boost_steps(6, "2") == 2 and S.boost_steps(6, "x") == 0
    z0, seen = _run(0)
    assert seen == [(16, 16)] * 6
    dit = _DiT()
    assert torch.equal(z0, S.sample(dit, torch.zeros(4, 8), 256, 256, steps=6, seed=3))   # no boost argument


def test_on_is_two_steps_or_a_third_whichever_is_more():
    assert [S.boost_steps(n, "auto") for n in (2, 4, 6, 8, 9, 10, 12, 30)] == [1, 2, 2, 2, 3, 3, 4, 10]
    z, seen = _run("auto")                                      # 6 steps: 2 at half size
    assert seen == [(8, 8)] * 2 + [(16, 16)] * 4


def test_the_choice_is_on_or_off_on_the_samples_tab_and_in_studio():
    from fizgig.web import studio
    d = get("minimax_image")
    o = next(x for x in d.options if x.key == "MINIMAX_IMAGE_SPEED_BOOST")
    assert o.tab == "samples" and o.studio and list(o.choice_labels()) == ["Off", "On"]
    assert "boost_steps" not in studio.studio_tokens(d, {})                         # off unless picked: nothing sent
    assert studio.studio_tokens(d, {"studio_opt_MINIMAX_IMAGE_SPEED_BOOST": "On"})["boost_steps"] == "auto"
