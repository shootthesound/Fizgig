"""Block swap over blocks that don't all hold the same Linears (Fibo 2: layers 6, 14, 22, 30, 38 carry a
cross-attention). Issue #2: INT8 + swap 28 crashed with "mat2 is on cpu" in _int_mm, swap 30 trained - at 28 two
injection blocks swap with each other (44 - 28 = 16, a multiple of their spacing of 8), and the staging buffers,
made for the first plain pair, were zipped against that longer job list, so the cross-attention weights never moved."""
import torch
import torch.nn as nn

from fizgig.modules.offloading import Offloader, _swap_jobs


class Block(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.q = nn.Linear(32, 32)
        self.out = nn.Linear(32, 32)
        self.cross_k = nn.Linear(48, 32) if cross else None


def test_each_pair_gets_buffers_for_its_own_weights():
    plain, inj = Block(), Block(cross=True)
    off = Offloader("t", 4, 2, torch.device("cpu"))
    make = lambda v: torch.empty_like(v)                      # noqa: E731 - pin_memory needs CUDA
    jobs_pp, _ = _swap_jobs(Block(), plain, torch.device("cpu"))
    jobs_ii, _ = _swap_jobs(Block(cross=True), inj, torch.device("cpu"))
    assert len(jobs_ii) == len(jobs_pp) + 1
    a_pp, _b = off._staging_for(jobs_pp, make)               # the first swap: a plain pair
    a_ii, b_ii = off._staging_for(jobs_ii, make)              # then two injection blocks: every weight has a buffer
    assert len(a_ii) == len(b_ii) == len(jobs_ii)
    assert [t.shape for t in a_ii] == [j[2].shape for j in jobs_ii]
    assert off._staging_for(jobs_pp, make)[0] is a_pp         # made once per set of weights


def test_an_outgoing_cross_attention_without_a_partner_leaves_the_gpu():
    jobs, unpaired = _swap_jobs(Block(cross=True), Block(), torch.device("cpu"))
    assert len(jobs) == 2 and len(unpaired) == 1 and unpaired[0].weight.shape == (32, 48)
