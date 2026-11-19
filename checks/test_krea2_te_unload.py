"""#380: Krea 2's text encoder unload frees its weights in place (meta) instead of copying them to RAM first."""
import torch

from fizgig.families.registry import get


def test_unload_goes_to_meta_not_ram():
    drv = get("krea2").load_driver()
    te = torch.nn.Sequential(torch.nn.Linear(4, 4))
    drv.unload_text_encoder(te)
    assert all(p.device.type == "meta" for p in te.parameters())
