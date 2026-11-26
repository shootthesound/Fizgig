"""#411: Studio's save of an "Epochs of this run" mix (or a LoRA mix) rank-concatenated every file in it, so twenty
r128 epochs saved at r2560 (~4.5 GB). The mix now comes back to its files' rank by an SVD of its product, more only
where a mix of unlike LoRAs needs it; a donor stays concatenated beside the primary, as before."""
import types

import torch
import torch.nn as nn

from fizgig.families import lora as L


def _pair(out, inp, r, base_a=None, base_b=None, noise=0.0):
    a, b = nn.Linear(inp, r, bias=False), nn.Linear(r, out, bias=False)
    with torch.no_grad():
        if base_a is not None:
            a.weight.copy_(base_a + noise * torch.randn_like(base_a))
            b.weight.copy_(base_b + noise * torch.randn_like(base_b))
        else:
            a.weight.normal_()
            b.weight.normal_()
    return (a, b)


def _net(adapters, scales):
    w = types.SimpleNamespace(adapters=adapters, scales=scales)
    return types.SimpleNamespace(wrapped={"m": w}, _keys=lambda full: ("A", "B", "alpha"))


def _delta(sd):
    return sd["B"].float() @ sd["A"].float()


def test_epochs_of_one_run_save_at_one_files_rank():
    torch.manual_seed(0)
    a0, b0 = torch.randn(16, 96), torch.randn(64, 16)
    eps = {f"e{i}": _pair(64, 96, 16, a0, b0, noise=0.0) for i in range(6)}
    sc = {n: 1 / 6 for n in eps}
    want = sum(sc[n] * eps[n][1].weight @ eps[n][0].weight for n in eps)
    names = list(eps)
    rep = {}
    sd, ranks = L.FamilyLoRA.bake(_net(eps, sc), names, dtype=torch.float32, mixes=[names], report=rep)
    assert ranks["m"] == 16 and sd["A"].shape == (16, 96) and float(sd["alpha"]) == 16
    assert torch.allclose(_delta(sd), want, atol=1e-3, rtol=1e-3)
    assert rep["from"] == 96 and rep["to"] == 16 and rep["kept"] > 0.9999


def test_unlike_loras_keep_the_rank_they_need():
    torch.manual_seed(1)
    mix = {f"l{i}": _pair(64, 96, 8) for i in range(3)}
    sc = {n: 1.0 for n in mix}
    names = list(mix)
    sd, ranks = L.FamilyLoRA.bake(_net(mix, sc), names, dtype=torch.float32, mixes=[names], report={})
    assert 8 < ranks["m"] <= 24
    want = sum(mix[n][1].weight @ mix[n][0].weight for n in mix)
    err = (_delta(sd) - want).norm() / want.norm()
    assert err < 0.04


def test_without_a_mix_the_bake_is_concatenated_as_before():
    torch.manual_seed(2)
    ads = {"primary": _pair(32, 40, 8), "donor": _pair(32, 40, 4)}
    sc = {"primary": 1.0, "donor": 0.5}
    sd, ranks = L.FamilyLoRA.bake(_net(ads, sc), ["primary", "donor"], dtype=torch.float32,
                                  mixes=[["primary"], ["donor"]])
    assert ranks["m"] == 12
    want = ads["primary"][1].weight @ ads["primary"][0].weight + 0.5 * ads["donor"][1].weight @ ads["donor"][0].weight
    assert torch.allclose(_delta(sd), want, atol=1e-4)


def test_conv_factors_squeeze_too():
    torch.manual_seed(3)
    a0, b0 = torch.randn(8, 12, 3, 3), torch.randn(20, 8, 1, 1)
    As, Bs = [a0.clone() for _ in range(4)], [b0 / 4 for _ in range(4)]
    (A,), (B,) = L._squeeze_mix(As, Bs, 0.999)
    assert A.shape == (8, 12, 3, 3) and B.shape == (20, 8, 1, 1)
    want = b0.reshape(20, 8) @ a0.reshape(8, -1)
    assert torch.allclose(B.reshape(20, 8) @ A.reshape(8, -1), want, atol=1e-3, rtol=1e-3)


def test_studio_save_passes_each_loras_mix():
    import inspect
    from fizgig.families import workbench
    src = inspect.getsource(workbench.WorkbenchEngine.save_repaired)
    assert "mixes=mixes" in src and "mix_rank" in src
