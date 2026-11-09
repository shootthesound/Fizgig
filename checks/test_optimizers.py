"""Prodigy, Muon and Automagic v4 (#292): the vendored optimizers build and step, Prodigy owns its rate (Adaptive LR
and the scheduler stand down, a fine-tune keeps its own optimizer and rate), they never fall back to AdamW, no
preset picks them - and every other optimizer is exactly as before (arguments, rates, fallback)."""

import logging
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from fizgig.families import form, launch, registry  # noqa: E402
from fizgig.training.muon import Muon as FallbackMuon  # noqa: E402
from fizgig.training.optimizers import (  # noqa: E402
    MUON_DEFAULT_ARGS,
    PRODIGY_DEFAULT_ARGS,
    available_optimizers,
    create_optimizer,
    optimizer_lr,
    owns_its_rate,
)
from fizgig.training.prodigy import Prodigy as VendoredProdigy  # noqa: E402

NEW = ("prodigy", "muon", "automagic4")


def test_new_optimizers_listed():
    opts = available_optimizers()
    for name in NEW + ("automagic3", "adamw", "adamw8bit"):
        assert name in opts


# presets that pick a new optimizer by decision (Peter, 6 Nov 2026 - Muon's source-matching detail)
CHOSEN = {k: "muon" for k in ("qwen_image21", "krea2", "moss_image", "vimoo", "zimage", "zimage_base",
                                    "klein", "sdxl", "flux3", "anima", "fibo2")}


def test_no_preset_picks_a_new_optimizer():
    """No preset picks a new optimizer, except the families in CHOSEN, whose presets all use their chosen one with
    its default arguments. A Fine-tune preset trains with Adafactor (#312)."""
    for desc in registry.training_families():
        for name, p in desc.presets or ():
            opt = str(p.get("OPTIMIZER_TYPE") or "").lower()
            if p.get("FAMILY_FT"):
                assert opt == "adafactor", (desc.key, name)
            elif desc.key in CHOSEN:
                assert opt == CHOSEN[desc.key], (desc.key, name)
                if opt == "muon":
                    assert p.get("OPTIMIZER_ARGS") == MUON_DEFAULT_ARGS, (desc.key, name)
            else:
                assert opt not in NEW, (desc.key, name)


def test_prodigy_is_the_vendored_copy_with_its_defaults():
    p = [nn.Parameter(torch.randn(8, 8))]
    opt, _ = create_optimizer("prodigy", p, 1.0)
    assert isinstance(opt, VendoredProdigy) and owns_its_rate(opt)
    g = opt.param_groups[0]
    assert g["lr"] == 1.0 and g["decouple"] is True and g["safeguard_warmup"] is True
    assert g["weight_decay"] == pytest.approx(0.01)
    assert optimizer_lr(opt) == pytest.approx(1e-6)          # d0 x lr before the first step
    opt2, label = create_optimizer("prodigy", [nn.Parameter(torch.randn(8, 8))], 1.0, "d_coef=0.5 slice_p=11")
    assert opt2.param_groups[0]["d_coef"] == pytest.approx(0.5) and opt2.param_groups[0]["slice_p"] == 11
    assert "d_coef=0.5" in label


def test_prodigy_finds_a_rate():
    torch.manual_seed(42)
    w, target = nn.Parameter(torch.randn(16, 16)), torch.randn(16, 16)
    opt, _ = create_optimizer("prodigy", [w], 1.0)
    rates = []
    for _ in range(14):
        opt.zero_grad()
        ((w - target) ** 2).mean().backward()
        opt.step()
        rates.append(optimizer_lr(opt))
    assert rates[0] == pytest.approx(1e-6) and 1e-6 < max(rates) < 0.1


def test_muon_steps_and_its_fallback():
    torch.manual_seed(7)
    down, up = nn.Parameter(torch.randn(16, 64)), nn.Parameter(torch.zeros(64, 16))
    opt, _ = create_optimizer("muon", [down, up], 1e-4)
    assert not owns_its_rate(opt)                             # Adaptive LR and the scheduler still drive it
    g = opt.param_groups[0]
    assert g["adjust_lr_fn"] == "match_rms_adamw" and g["momentum"] == pytest.approx(0.95) and g["ns_steps"] == 5
    x, target, before = torch.randn(4, 64), torch.randn(4, 64), down.detach().clone()
    opt.zero_grad()
    (((x @ down.T) @ up.T - target) ** 2).mean().backward()
    opt.step()
    assert not torch.equal(down, before)
    conv = [nn.Parameter(torch.randn(8, 4, 3, 3)), nn.Parameter(torch.randn(8))]
    opt_fb, _ = create_optimizer("muon", conv, 1e-4)
    assert isinstance(opt_fb, FallbackMuon)


def test_automagic4_owns_its_rate():
    opt, _ = create_optimizer("automagic4", [nn.Parameter(torch.randn(8, 8))], 1e-6)
    assert owns_its_rate(opt)


def test_new_optimizers_never_fall_back_to_adamw():
    for name in NEW:
        with pytest.raises(ValueError):
            create_optimizer(name, [nn.Parameter(torch.randn(4, 4))], 1.0, "no_such_arg=1")


def test_other_optimizers_unchanged(caplog):
    # the master behaviour: a bad argument falls back to AdamW at the rate given, nothing stripped or clamped
    with caplog.at_level(logging.WARNING):
        opt, label = create_optimizer("adamw", [nn.Parameter(torch.randn(4, 4))], 1.0, "momentum=0.9")
    assert label == "adamw (fallback)" and opt.param_groups[0]["lr"] == 1.0
    opt, _ = create_optimizer("adamw", [nn.Parameter(torch.randn(4, 4))], 1e-4, "weight_decay=0.02")
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.02)

    class D(torch.optim.SGD):                                  # a module-path optimizer with a "d" in its group
        pass
    d = D([nn.Parameter(torch.randn(2, 2))], lr=0.1)
    d.param_groups[0]["d"] = 5.0
    assert not owns_its_rate(d) and optimizer_lr(d) == pytest.approx(0.1)


def test_form_leaves_other_optimizers_alone():
    krea = registry.get("krea2")
    for opt in ("adamw8bit", "adamw", "lion8bit", "automagic3"):
        st = form.state(krea, {**form.DEFAULTS, "OPTIMIZER_TYPE": opt, "LEARNING_RATE": "1.0",
                               "OPTIMIZER_ARGS": "momentum=0.9 " + PRODIGY_DEFAULT_ARGS})
        assert "OPTIMIZER_ARGS" not in st["force"] and st["force"].get("LEARNING_RATE") in (None,), (opt, st["force"])


def test_form_prodigy_and_muon():
    krea = registry.get("krea2")
    vals = {**form.DEFAULTS, "OPTIMIZER_TYPE": "prodigy", "LEARNING_RATE": "1e-4", "OPTIMIZER_ARGS": "",
            "ADAPTIVE_LR": True, "KREA2_PER_IMAGE_LR": True, "KREA2_WARMUP_LOOK": True}
    st = form.state(krea, vals)
    assert st["force"]["LEARNING_RATE"] == "1.0" and st["force"]["OPTIMIZER_ARGS"] == PRODIGY_DEFAULT_ARGS
    assert st["force"]["ADAPTIVE_LR"] is False and st["force"]["KREA2_WARMUP_LOOK"] is False
    assert "KREA2_PER_IMAGE_LR" not in st["force"] and "LEARNING_RATE" in st["disabled"]
    keep = form.state(krea, {**vals, "OPTIMIZER_ARGS": "d_coef=0.5"})       # typed args stay
    assert "OPTIMIZER_ARGS" not in keep["force"]
    mu = form.state(krea, {**vals, **st["force"], "OPTIMIZER_TYPE": "muon"})
    assert mu["force"]["OPTIMIZER_ARGS"] == MUON_DEFAULT_ARGS and "LEARNING_RATE" not in mu["force"]
    assert "ADAPTIVE_LR" not in mu["disabled"]


def test_switching_off_muon_clears_its_args():
    """Every picture preset is on Muon: switching to AdamW / AdamW8bit must not keep Muon's args (they made the new
    optimizer fail and fall back to plain, unfused AdamW)."""
    krea = registry.get("krea2")
    for opt in ("adamw", "adamw8bit"):
        st = form.state(krea, {**form.DEFAULTS, "OPTIMIZER_TYPE": opt, "OPTIMIZER_ARGS": MUON_DEFAULT_ARGS})
        assert st["force"].get("OPTIMIZER_ARGS") == ""
    typed = form.state(krea, {**form.DEFAULTS, "OPTIMIZER_TYPE": "adamw", "OPTIMIZER_ARGS": "weight_decay=0.05"})
    assert "OPTIMIZER_ARGS" not in typed["force"]                            # typed args stay


def test_create_optimizer_drops_leftover_muon_args():
    p = [nn.Parameter(torch.randn(16, 64))]
    opt, label = create_optimizer("adamw", p, 1e-4, MUON_DEFAULT_ARGS)
    assert isinstance(opt, torch.optim.AdamW) and "fallback" not in label


def test_fine_tune_with_prodigy_keeps_its_rate():
    moss = registry.get("moss_image")
    st = form.state(moss, {**form.DEFAULTS, "FAMILY_KIND": "finetune", "OPTIMIZER_TYPE": "prodigy",
                           "LEARNING_RATE": "1e-5"})
    assert "LEARNING_RATE" not in st["force"] and "LEARNING_RATE" not in st["disabled"]
    inputs = {**form.DEFAULTS, "FAMILY_FT": True, "OPTIMIZER_TYPE": "prodigy", "LEARNING_RATE": "1e-5"}
    if launch.ft_on(moss, inputs):
        launch_inputs = dict(inputs)
        launch_inputs["ADAPTIVE_LR"] = False
        assert not (launch.prodigy(launch_inputs) and not launch.ft_on(moss, launch_inputs))


def test_trainer_composes_step_scales_into_prodigy():
    src = open(launch.__file__.replace("launch.py", "train.py"), encoding="utf-8").read()
    assert 'optimizer.__class__.__name__ in ("Automagic3", "Automagic4")' in src
    assert "None if global_step % 20 else" in src                # the rate logged now and then, as before


def test_lr_band_check_skips_prodigy_lora():
    krea = registry.get("krea2")
    if not krea.start_lr_bands:
        pytest.skip("no bands")
    st = {**form.DEFAULTS, "OPTIMIZER_TYPE": "prodigy", "LEARNING_RATE": "1.0", "ADAPTIVE_LR": False}
    assert launch.prodigy(st) and not launch.ft_on(krea, st)


# BatchedMuon (#221): torch.optim.Muon's update with the per-matrix work batched - a LoRA's hundreds of small matrices
# made the per-parameter loop a third of a MOSS step. It must stay the same maths.
import copy  # noqa: E402

from fizgig.training.muon import BatchedMuon  # noqa: E402

_SHAPES = [(16, 64), (64, 16), (16, 64), (48, 16), (8, 8), (16, 96), (96, 16)]


def _params(seed):
    g = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(s, generator=g) * 0.02) for s in _SHAPES]


def _grads(step):
    g = torch.Generator().manual_seed(1000 + step)
    return [torch.randn(s, generator=g) for s in _SHAPES]


def test_muon_is_batched_for_lora_matrices():
    opt, _ = create_optimizer("muon", _params(0), 1e-4)
    assert isinstance(opt, BatchedMuon)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="needs torch.optim.Muon")
@pytest.mark.parametrize("kw", [dict(nesterov=True, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"),
                                dict(nesterov=False, weight_decay=0.0, adjust_lr_fn="original"),
                                dict(nesterov=True, weight_decay=0.1, adjust_lr_fn=None)])
def test_batched_muon_matches_torch_muon(kw):
    pa, pb = _params(1), _params(1)
    oa = torch.optim.Muon(pa, lr=1e-3, momentum=0.95, ns_steps=5, **kw)
    ob = BatchedMuon(pb, lr=1e-3, momentum=0.95, ns_steps=5, **kw)
    for step in range(6):
        for p, q, g in zip(pa, pb, _grads(step)):
            p.grad, q.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb):
        assert (a - b).abs().max() <= 1e-3 * a.abs().max(), (tuple(a.shape), (a - b).abs().max().item())


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="needs torch.optim.Muon")
def test_batched_muon_resumes_a_torch_muon_state():
    """A run paused on torch.optim.Muon goes on under BatchedMuon as it would have on torch's."""
    pa, pb = _params(2), _params(2)
    kw = dict(lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    oa = torch.optim.Muon(pa, **kw)
    for step in range(3):
        for p, g in zip(pa, _grads(step)):
            p.grad = g.clone()
        oa.step()
    with torch.no_grad():
        for p, q in zip(pa, pb):
            q.copy_(p)
    ob = BatchedMuon(pb, **kw)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))   # as a resume loads it from a file: nothing shared
    for step in range(3, 6):
        for p, q, g in zip(pa, pb, _grads(step)):
            p.grad, q.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb):
        assert (a - b).abs().max() <= 1e-3 * a.abs().max()


def test_batched_muon_leaves_params_without_a_grad_alone():
    ps = _params(3)
    opt = BatchedMuon(ps, lr=1e-3, weight_decay=0.01)
    before = [p.detach().clone() for p in ps]
    for i, (p, g) in enumerate(zip(ps, _grads(0))):
        p.grad = g if i % 2 == 0 else None
    opt.step()
    for i, (p, b) in enumerate(zip(ps, before)):
        assert torch.equal(p, b) == (i % 2 == 1)
