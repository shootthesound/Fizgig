"""H3 Studio render speed: the int8 linear switch stays out of everything but a CUDA render, a LoRA's one-pass add is
the same maths as the two-pass one, and the base plan for a long clip (no reload to NF4, no over-asking past 56 frames)."""
import torch
import torch.nn as nn

from fizgig.modules import int8_linear
from fizgig.networks.lora import LoRAInfModule
from fizgig.repair_studio.h3_engine import H3RepairEngine as E


def test_int8_linear_switch_is_scoped_and_declines_off_the_card():
    assert not int8_linear.wanted()
    x, q, s = torch.randn(4, 16), torch.zeros(8, 16, dtype=torch.int8), torch.ones(8, 1)
    with torch.no_grad(), int8_linear.renders(True):
        assert int8_linear.wanted()
        assert int8_linear.linear(x, q, s, None, 1, torch.float32) is None      # CPU tensors: the caller's own path
    assert not int8_linear.wanted()
    with int8_linear.renders(True):
        assert int8_linear.linear(x, q, s, None, 1, torch.float32) is None      # a gradient is being taken


def _lora(lin, mult=0.7):
    m = LoRAInfModule("lora_unet_x", lin, multiplier=mult, lora_dim=4, alpha=2.0)
    nn.init.normal_(m.lora_up.weight)
    m.apply_to()
    return m


def test_lora_one_pass_add_matches_the_two_pass_one():
    torch.manual_seed(0)
    lin = nn.Linear(32, 48)
    m = _lora(lin)
    x = torch.randn(2, 5, 32)
    want = m.org_forward(x) + m.lora_up(m.lora_down(x)) * m.multiplier * m.scale
    with torch.no_grad():
        got = lin(x)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(lin(x), want, atol=1e-5)                              # with a gradient: the old path
    m.enabled = False
    with torch.no_grad():
        assert torch.equal(lin(x), m.org_forward(x))


def test_lora_one_pass_add_chains_two_loras():
    torch.manual_seed(1)
    lin = nn.Linear(16, 24)
    a = _lora(lin, 1.0)
    b = _lora(lin, -0.5)
    x = torch.randn(3, 16)
    want = (a.org_forward(x) + a.lora_up(a.lora_down(x)) * a.scale
            + b.lora_up(b.lora_down(x)) * -0.5 * b.scale)
    with torch.no_grad():
        assert torch.allclose(lin(x), want, atol=1e-5)


def _old_need(w, h, f, k=2, r=0):
    return (4.5 + (w * h / (768 * 640)) * (0.92 + 0.081 * f + 0.4 * k + 0.75 * r)) * 2 ** 30 / 1e9


def test_clip_need_is_unchanged_up_to_56_frames_and_lower_past_it():
    for c in [(768, 640, 22, 2, 0), (768, 640, 56, 2, 0), (768, 640, 22, 0, 2), (512, 512, 22, 0, 0),
              (1024, 1024, 56, 2, 0)]:
        assert abs(E.clip_need_gb(*c) - _old_need(*c)) < 1e-6
    long_clip = E.clip_need_gb(512, 512, 124, 0, 0)
    assert E.clip_need_gb(512, 512, 56, 0, 0) < long_clip < _old_need(512, 512, 124, 0, 0)
    assert E.clip_need_gb(512, 512, 124, 0, 0) < E.clip_need_gb(512, 512, 226, 0, 0)


def test_a_reload_for_room_keeps_the_int8_base():
    need = E.clip_need_gb(512, 512, 124, 0, 0)
    assert E.plan_base(12.0, "auto", need) == ("nf4", 0)                        # a first load on a small card
    quant, n = E.plan_base(12.0, "auto", need, keep_int8=True)                  # a reload mid-session
    assert quant == "int8" and 0 < n <= 40
    assert E.plan_base(31.0, "auto", need, keep_int8=True)[0] == "int8"
    assert E.plan_base(12.0, "nf4", need, keep_int8=True) == ("nf4", 0)         # the user's own pick stands
