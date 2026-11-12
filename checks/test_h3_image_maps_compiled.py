"""MiniMax H3 Image's trigger-word map hooks find each block's attention inside Compile Blocks' wrappers
(CheckpointedBlock, torch.compile's module): before, the maps failed at epoch 0 on a compiled run and turned off."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.families.compile import CheckpointedBlock  # noqa: E402
from fizgig.minimax_image.driver import MiniMaxH3ImageDriver  # noqa: E402


class _Attn(torch.nn.Module):
    def forward(self, x, cos=None, sin=None, segs=None):
        return x


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()


def test_hooks_reach_wrapped_blocks():
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([CheckpointedBlock(_Block(), True), _Block()])
    attns = [dit.blocks[0].block.attn, dit.blocks[1].attn]
    before = [a.forward for a in attns]
    drv = MiniMaxH3ImageDriver.__new__(MiniMaxH3ImageDriver)
    with drv.trigger_attention_hooks(dit):
        assert all(a.forward is not f for a, f in zip(attns, before))
    assert all(a.forward == f for a, f in zip(attns, before))
