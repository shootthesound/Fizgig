"""A frozen LoRA file can adapt a layer the model reads a weight dtype off (H3's audio_patch_proj, from another
trainer's Context LoRA): the read goes through the wrapper, and the wrapper itself stays without a .weight, so block
swap's Linear pick (a class name ending in "Linear" with a weight) never takes a layer twice (#298, @spaghettified26)."""
import torch
import torch.nn as nn

from fizgig.families.lora import LoRALinear
from fizgig.minimax.model import _wdtype


def test_weight_dtype_reads_through_the_wrapper():
    lin = nn.Linear(32, 64).to(torch.bfloat16)
    w = LoRALinear(lin)
    w.add("context", 4, 4.0, False, A=torch.randn(4, 32), B=torch.randn(64, 4))
    assert _wdtype(w) == torch.bfloat16 and _wdtype(lin) == torch.bfloat16
    out = w(torch.randn(2, 32).to(_wdtype(w)))
    assert out.shape == (2, 64)


def test_block_swap_picks_each_layer_once():
    from fizgig.modules.offloading import _streamed_linears
    block = nn.Module()
    block.audio_patch_proj = LoRALinear(nn.Linear(8, 8))
    block.plain = nn.Linear(8, 8)
    assert not hasattr(block.audio_patch_proj, "weight")
    picked = _streamed_linears(block)
    assert sorted(picked) == ["audio_patch_proj", "plain"]
    assert isinstance(picked["audio_patch_proj"], nn.Linear)
