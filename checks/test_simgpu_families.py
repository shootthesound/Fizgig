"""Each family's own block swap on the simulated GPU (checks/simgpu.py): a tiny model of the family, set up as the
trainer sets it up (the base with block swap on, gradient checkpointing, then Fizgig's LoRA wrapped round its Linears)
and trained a few steps through the family driver's own training loss - one picture a step, or the driver's batch
path (several forwards, one backward) - with an optimizer step between, so each step starts from where the last left
the blocks. Against the same model with nothing swapped: the same losses and gradients, and no operation ever meets a
block left on the CPU."""
import copy

import pytest
import torch

pytest.importorskip("fizgig.families.lora")
from simgpu import on_card, simulated_gpu  # noqa: E402

from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402

N = 6                                         # blocks in each tiny model: swaps of 1 .. N-2


def _moss(monkeypatch):
    from fizgig.moss_image.model import MOSSImageDiT
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=N, text_dim=24, freq_dim=32, latent_channels=8)

    def item(g):
        return torch.randn(1, 8, 4, 6, generator=g), {"hidden_states": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


def _fibo(monkeypatch):
    from fizgig.fibo2.model import Fibo2DiT
    dit = Fibo2DiT(dim=64, n_layers=N, n_refiner_layers=1, n_heads=4, cap_feat_dim=24,
                   injection_layer_ids=tuple(range(N)), perceiver_num_layers=1, min_num_gist_tokens=4,
                   max_num_gist_tokens=8, gist_step=2, gist_min_text_len=2, gist_max_text_len=12, axes_dims=(4, 6, 6),
                   axes_lens=(64, 32, 32))

    def item(g):
        n = 6
        return (torch.randn(1, 32, 8, 8, generator=g),
                {"tokens": torch.randn(1, 1, N + 1, n, 24, generator=g), "mask": torch.ones(1, 1, n, dtype=torch.bool)})
    return dit.float(), item


def _flux3(monkeypatch):
    from fizgig.flux3 import model as M
    for k, v in (("HIDDEN", 64), ("HEADS", 4), ("DEPTH", 1), ("DEPTH_SINGLE", N), ("AXES_DIM", (4, 4, 4, 4))):
        monkeypatch.setattr(M, k, v)
    monkeypatch.setitem(M.IN_CHANNELS, "text", 24)
    dit = M.Flux3DiT()

    def item(g):
        return torch.randn(1, 128, 4, 6, generator=g), {"tokens": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


def _qwen(monkeypatch):
    from fizgig.qwen_image21.model import QwenImage21DiT
    dit = QwenImage21DiT(num_layers=N, attention_head_dim=16, num_attention_heads=2, context_in_dim=24,
                         axes_dims_rope=(4, 6, 6))

    def item(g):
        return torch.randn(1, 64, 4, 4, generator=g), {"hidden_states": torch.randn(1, 5, 24, generator=g)}
    return dit.to(torch.bfloat16), item                  # the driver feeds it bf16, as the real base is


def _zimage(monkeypatch):
    from fizgig.zimage.model import ZImageTransformer2DModel
    dit = ZImageTransformer2DModel(in_channels=16, dim=64, n_layers=N, n_refiner_layers=1, n_heads=4, n_kv_heads=4,
                                   cap_feat_dim=24, axes_dims=(8, 4, 4), axes_lens=(64, 32, 32))
    for prm in dit.parameters():                         # its pad tokens are torch.empty until a checkpoint loads
        torch.nn.init.normal_(prm, std=0.05)

    def item(g):
        return torch.randn(1, 16, 8, 8, generator=g), {"cap": torch.randn(1, 5, 24, generator=g)}
    return dit.to(torch.bfloat16), item                  # the driver runs it in bf16


def _klein(monkeypatch):
    from fizgig.klein.model import Flux2Params, KleinDiT
    # Klein 9B's 1:3 double / single ratio (8 / 24), so a swap splits as it does there - a swap of 1 is double 0,
    # single 1, the case that once left every double block on the CPU
    dit = KleinDiT(Flux2Params(in_channels=16, context_in_dim=24, hidden_size=64, num_heads=4, depth=4,
                               depth_single_blocks=12, axes_dim=[4, 4, 4, 4]), attn_mode="torch")

    def item(g):
        return torch.randn(1, 16, 4, 4, generator=g), {"ctx_vec": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


FAMILIES = {"moss_image": _moss, "fibo2": _fibo, "flux3": _flux3, "qwen_image21": _qwen, "zimage": _zimage,
            "klein": _klein}


def _train(key, build, monkeypatch, swap, batch, steps=3):
    """Losses and LoRA gradients of `steps` training steps, swapping `swap` blocks (0: all on the card)."""
    drv = get_family(key).load_driver()
    torch.manual_seed(0)
    dit, item = build(monkeypatch)
    g = torch.Generator().manual_seed(1)
    data = [[item(g) for _ in range(batch)] for _ in range(steps)]
    out = []
    with simulated_gpu():
        if swap:
            drv.enable_block_swap(dit, swap, "cuda")
        else:
            dit.to("cuda")
        drv.enable_gradient_checkpointing(dit, True)
        torch.manual_seed(2)
        net = FamilyLoRA(dit, drv, device="cuda")
        net.add_trainable(4, 4)
        for p in net.parameters():
            p.data = p.data.to("cuda")
            torch.nn.init.normal_(p, std=0.05)          # both poles non-zero, so every gradient path is live
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        dit.train()
        for s, items in enumerate(data):
            gen = torch.Generator().manual_seed(100 + s)
            lats = [lat.to("cuda") for lat, _ in items]
            conds = [{k: v.to("cuda") for k, v in c.items()} for _, c in items]
            if batch > 1:
                loss, _ = drv.training_loss_batch(dit, torch.cat(lats), drv.stack_conds(conds), gen)
            else:
                loss, _ = drv.training_loss(dit, lats[0], conds[0], gen)
            loss.backward()
            out.append((float(loss.detach().to("cpu")),
                        [None if p.grad is None else p.grad.detach().to("cpu").clone() for p in net.parameters()]))
            opt.step()
            opt.zero_grad(set_to_none=True)
    return out


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("swap", [1, 3, 4])
@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_family_block_swap_trains_as_with_no_swap(key, swap, batch, monkeypatch):
    want = _train(key, FAMILIES[key], monkeypatch, 0, batch)
    got = _train(key, FAMILIES[key], monkeypatch, swap, batch)
    for step, ((wl, wg), (gl, gg)) in enumerate(zip(want, got)):
        assert wl == pytest.approx(gl, rel=1e-5, abs=1e-6), (key, step)
        for i, (a, b) in enumerate(zip(wg, gg)):
            assert (a is None) == (b is None), (key, step, i)       # an adapter with no gradient has none either way
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (key, step, i, (a - b).abs().max())


def test_klein_swap_of_one_puts_every_double_block_on_the_card(monkeypatch):
    """Found by these checks (20 Nov 2026), confirmed on a 5090: a swap of 1 splits as double 0 / single 1 on Klein 9B,
    and moving the model left both block lists off the card for their offloaders - but an offloader with 0 blocks does
    nothing, so every double block stayed on the CPU and the first step met "mat2 is on cpu"."""
    drv = get_family("klein").load_driver()
    torch.manual_seed(0)
    dit, _ = _klein(monkeypatch)
    with simulated_gpu():
        drv.enable_block_swap(dit, 1, "cuda")
        assert dit.offloader_double.blocks_to_swap == 0 and dit.offloader_single.blocks_to_swap == 1
        assert all(on_card(b.img_attn.qkv.weight) for b in dit.double_blocks)
