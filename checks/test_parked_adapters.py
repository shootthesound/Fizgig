"""#405: a frozen adapter parked on the CPU between previews (the speed LoRA, MOSS-Image's look LoRA) stays there
through block swap's moves. Training maps, DOP and masked prior run a forward with block swap streamed as inference and
then put the training layout back; that put every block on the card with a plain .to(), and the parked LoRAs inside
its Linears went with it: ~2.3 GB of MOSS-Image's Turbo and look LoRAs on a 10 GB card with 14 of 42 blocks swapped."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simgpu import on_card, simulated_gpu  # noqa: E402
from test_simgpu_families import FAMILIES  # noqa: E402

from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402

LOOK = "preview_look"


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_parked_adapter_stays_off_the_card(key, monkeypatch, tmp_path):
    drv = get_family(key).load_driver()
    torch.manual_seed(0)
    built = FAMILIES[key](monkeypatch)
    if len(built) > 2:
        pytest.skip("its model sets its own swap up")
    dit, item = built[0], built[1]
    dit.requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    with simulated_gpu():
        drv.enable_block_swap(dit, 2, "cuda")
        drv.enable_gradient_checkpointing(dit, True)
        net = FamilyLoRA(dit, drv, device="cuda")
        net.add_trainable(4, 4)
        for p in net.parameters():
            p.data = p.data.to("cuda")
            torch.nn.init.normal_(p, std=0.05)
        path = str(tmp_path / "look.safetensors")
        net.save(path)
        assert net.add_file(path, LOOK, 1.0) > 0
        net.set_enabled(LOOK, False)
        net.move_adapter(LOOK, "cpu")
        parked = [t for w in net.wrapped.values() if LOOK in w.adapters for t in w.adapters[LOOK].parameters()]
        assert parked and not any(on_card(t) for t in parked)
        dit.train()
        for s in range(2):
            lat, cond = item(g)
            lat, cond = lat.to("cuda"), {k: v.to("cuda") for k, v in cond.items()}
            gen = torch.Generator().manual_seed(100 + s)
            drv.block_swap_mode(dit, inference=True)          # a training-maps / DOP forward
            with torch.no_grad():
                drv.predict(dit, drv.noise_latents(lat, gen), cond)
            drv.block_swap_mode(dit, inference=False)         # the training layout back
            loss, _ = drv.training_loss(dit, lat, cond, gen)
            loss.backward()
            moved = sum(on_card(t) for t in parked)
            assert moved == 0, f"{key}: {moved} of {len(parked)} parked adapter tensors on the card after step {s}"
