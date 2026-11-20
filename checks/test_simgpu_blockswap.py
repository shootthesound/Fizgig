"""Block swap on a simulated GPU (checks/simgpu.py): the real ModelOffloader on a tiny model wired as the families
wire it (MOSS-Image / Qwen Image 2.1: wait_for_block, the block - checkpointed when training - then
submit_move_blocks_forward), frozen weights with a small trainable adapter as in LoRA training.

Across batch sizes (several forwards before one backward), swap depths, checkpointing on and off and several steps:
no operation ever mixes card and CPU weights, no more blocks are on the card than the swap allows, and the losses and
gradients are exactly those of the same model with nothing swapped. #335 is the first regression test: the
offloader as it was before its fix fails in simulation the way the issue's run failed."""
import copy
import os
import subprocess
import sys
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

pytest.importorskip("fizgig.modules.offloading")
from simgpu import SimDeviceMismatch, blocks_on_card, on_card, simulated_gpu  # noqa: E402

N_BLOCKS, WIDTH = 6, 16


class Adapter(nn.Module):
    """A LoRA-like trainable pair on a frozen Linear (named so the offloader streams the base, not the pair)."""

    def __init__(self, base, rank=2):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.randn(rank, base.in_features) * 0.1)
        self.up = nn.Parameter(torch.randn(base.out_features, rank) * 0.1)

    def forward(self, x):
        return self.base(x) + x @ self.down.t() @ self.up.t()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Adapter(nn.Linear(WIDTH, WIDTH))
        self.fc2 = Adapter(nn.Linear(WIDTH, WIDTH))

    def forward(self, x):
        return x + self.fc2(torch.tanh(self.fc1(x)))


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(WIDTH, WIDTH)
        self.layers = nn.ModuleList(Block() for _ in range(N_BLOCKS))
        self.out = nn.Linear(WIDTH, WIDTH)
        self.blocks_to_swap, self.offloader, self.gc = 0, None, False
        for n, p in self.named_parameters():
            p.requires_grad_(n.endswith((".down", ".up")))

    def enable_block_swap(self, k, device, offloader_cls):
        self.blocks_to_swap = k
        self.offloader = offloader_cls("tiny", list(self.layers), len(self.layers), k, True, torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        layers = self.layers
        self.layers = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.layers = layers
        for b in self.layers:                      # the adapters live on the card (they train); the bases stream
            for m in b.modules():
                if isinstance(m, Adapter):
                    m.down.data, m.up.data = m.down.data.to(device), m.up.data.to(device)
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.layers))
        else:
            self.layers.to(device)

    def forward(self, x):
        x = self.inp(x)
        layers = list(self.layers) if self.blocks_to_swap else None
        for i, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            x = checkpoint(layer, x, use_reentrant=False) if self.gc and self.training else layer(x)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(layers, i)
        return self.out(x)


def _step(model, xs, device):
    """One training step as the family trainer runs a batch: a forward per item, one backward over their sum."""
    loss = sum(((model(x.to(device)) - 1.0) ** 2).mean() for x in xs)
    loss.backward()
    grads = {n: p.grad.detach().to("cpu").clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return loss.detach().to("cpu").item(), grads


def _run(offloader_cls, k, batch, gc, steps=3, seed=0):
    torch.manual_seed(seed)
    ref = TinyDiT()
    data = [[torch.randn(3, WIDTH) for _ in range(batch)] for _ in range(steps)]
    swap = copy.deepcopy(ref)
    ref.gc = swap.gc = gc
    want = [_step(ref, xs, "cpu") for xs in data]   # nothing swapped, on the CPU: the answer
    with simulated_gpu():
        if k:
            swap.enable_block_swap(k, "cuda", offloader_cls)
        swap.move_to_device_except_swap_blocks("cuda")
        swap.train()
        got, peak = [], 0
        for xs in data:
            if k:
                swap.offloader.prepare_block_devices_before_forward(list(swap.layers))
            got.append(_step(swap, xs, "cuda"))
            peak = max(peak, sum(1 for b in blocks_on_card([lay.fc1.base for lay in swap.layers]) if b))
    return want, got, peak


def _same(want, got):
    for (wl, wg), (gl, gg) in zip(want, got):
        assert wl == pytest.approx(gl, rel=1e-6, abs=1e-6)
        assert wg.keys() == gg.keys()
        for n in wg:
            assert torch.allclose(wg[n], gg[n], atol=1e-6), n


def _offloader():
    from fizgig.modules.offloading import ModelOffloader
    return ModelOffloader


def test_the_simulated_card_catches_a_block_left_on_the_cpu():
    with simulated_gpu():
        lin = nn.Linear(4, 4).to("cuda")
        x = torch.randn(2, 4).to("cuda")
        assert x.device.type == "cuda" and on_card(lin.weight)
        lin.weight.data = lin.weight.data.to("cpu")          # swapped out, then used: the bug class
        with pytest.raises(SimDeviceMismatch):
            lin(x)


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_block_swap_matches_no_swap(k, batch):
    """Training with checkpointing, as every swapped run does (below): exact losses and gradients, and never more
    blocks on the card than the swap allows."""
    want, got, peak = _run(_offloader(), k, batch, gc=True)
    _same(want, got)
    assert peak <= N_BLOCKS - k


def test_swap_without_checkpointing_is_never_planned():
    """Found by this harness: without checkpointing, the backward reads the weights a forward saved, but the swap
    reuses that card memory for other blocks - with several forwards a step (batch above 1) the gradients come out
    silently wrong. So a swapped run must always checkpoint: Auto's planner keeps it on whenever blocks are swapped."""
    want, got, _ = _run(_offloader(), 1, 2, gc=False)
    assert any(not torch.allclose(w, g, atol=1e-6) for (_, wg), (_, gg) in zip(want, got)
               for w, g in ((wg[n], gg[n]) for n in wg)), "swapping without checkpointing came out right: re-check"
    from fizgig.families import quant
    from fizgig.families.registry import FAMILIES
    for d in FAMILIES.values():
        on, why = quant.plan_checkpointing(d, "int8", 4, 0.25, free_gb=999, compiled=True)
        assert on, (d.key, why)


def _offloader_before(commit):
    """ModelOffloader as it was at a commit (git history), or None without git."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        src = subprocess.run(["git", "show", f"{commit}:src/fizgig/modules/offloading.py"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod = types.ModuleType("offloading_before")
    mod.__dict__["__name__"] = "fizgig.modules.offloading"     # so simgpu's inline mover patch reaches it
    exec(compile(src, "offloading_before.py", "exec"), mod.__dict__)
    return mod


def test_issue_335_reproduces_before_the_fix_and_is_fixed_now(monkeypatch):
    """Batch size above 1 with block swap and checkpointing: the offloader before 6da94f1c left the last blocks on
    the CPU for the earlier pictures' backward ("mat2 is on cpu"); today's doesn't."""
    old = _offloader_before("6da94f1c^")
    if old is None:
        pytest.skip("no git history here")
    from simgpu import _Inline
    monkeypatch.setattr(old, "ThreadPoolExecutor", _Inline)
    with pytest.raises(SimDeviceMismatch):
        _run(old.ModelOffloader, 2, 2, True)
    want, got, _ = _run(_offloader(), 2, 2, True)
    _same(want, got)


def test_more_than_half_swapped_with_a_batch_reproduces_before_the_fix_and_is_fixed_now(monkeypatch):
    """Found by this harness (20 Nov 2026): with more than half the blocks swapped and Batch size above 1, the #335
    fix paired early block i with late block n-k+i, the pairs overlapped, and the earlier pictures' backward met a
    block on the CPU ("mat2 is on cpu"). The offloader at 49028c81 (before the fix) fails; today's matches no swap."""
    old = _offloader_before("49028c81")
    if old is None:
        pytest.skip("no git history here")
    from simgpu import _Inline
    monkeypatch.setattr(old, "ThreadPoolExecutor", _Inline)
    with pytest.raises(SimDeviceMismatch):
        _run(old.ModelOffloader, 4, 2, True)
    want, got, _ = _run(_offloader(), 4, 2, True)
    _same(want, got)
