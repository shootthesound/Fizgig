"""MiniMax H3 Image's LCM sampler (ComfyUI's sample_lcm, as a community text-to-image recipe uses it: s_noise 1.1
falling to 0.9): each step goes to the head's clean picture, then back to the next sigma with fresh noise of that
strength; Euler stays the default."""
import torch

from fizgig.minimax_image import sampling as S


class _Head(torch.nn.Module):
    """A DiT stand-in: returns x0 - noise for a fixed x0, so the clean picture at every step is x0."""

    def __init__(self, x0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.x0 = x0
        self.final_layer = type("FL", (), {"pdd_sigmas": [1.0, 0.85, 0.7, 0.55, 0.4, 0.232, 0.0],
                                            "span_override": None})()
        self.seen = []

    def forward(self, x, t, txt, **kw):
        s = 1.0 - float(t)
        self.seen.append(s)
        noise = (x.float() - (1.0 - s) * self.x0) / s       # the noise in x at this sigma
        return (self.x0 - noise).to(x.dtype)


def _run(sampler, x0):
    head = _Head(x0)
    z = S.sample(head, torch.zeros(4, 8), 32, 32, steps=6, seed=3, sampler=sampler)
    return z, head.seen


def test_lcm_steps_through_clean_pictures_with_fresh_noise():
    x0 = torch.randn(1, 24, 1, 2, 2, generator=torch.Generator().manual_seed(0))
    z, seen = _run("lcm", x0)
    assert len(seen) == 6
    assert torch.allclose(z, x0, atol=2e-2)                 # the last step lands on the clean picture
    # by hand: the same steps, noise from ComfyUI's generator (the seed + 1 on the CPU), strength 1.1 -> 0.9
    sig = S.schedule(6, [1.0, 0.85, 0.7, 0.55, 0.4, 0.232, 0.0])
    g = torch.Generator("cpu").manual_seed(4)
    x = S.initial_noise(3, 32, 32) * sig[0]
    xs = []
    for i in range(6):
        x = x0.clone()
        if sig[i + 1] > 0:
            k = 1.1 + (0.9 - 1.1) * i / 5
            x = sig[i + 1] * (k * torch.randn(x.size(), generator=g)) + (1.0 - sig[i + 1]) * x
        xs.append(x)
    assert torch.allclose(z, xs[-1], atol=2e-2)


def test_euler_is_unchanged_by_default():
    x0 = torch.randn(1, 24, 1, 2, 2, generator=torch.Generator().manual_seed(1))
    head = _Head(x0)
    a = S.sample(head, torch.zeros(4, 8), 32, 32, steps=6, seed=3)
    b = S.sample(_Head(x0), torch.zeros(4, 8), 32, 32, steps=6, seed=3, sampler="euler")
    assert torch.equal(a, b)
