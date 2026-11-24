"""Krea 2's block swap keeps the swapped blocks in pinned RAM when there is room to spare (copies to and from the card
then skip a staging pass: 3.06 -> 2.01 s/it on a 5090, INT8, 16 swapped, 1 MP), and in ordinary RAM otherwise - pinning
with RAM nearly full has failed before (#175), and on a 32 GB laptop it left 0.2 GB free for 4% more speed."""
import pytest

from fizgig.families import quant


@pytest.mark.parametrize("pinned_gb, free_gb, want", [(7.0, 100, True), (7.0, 11.0, True), (7.0, 10.9, False),
                                                      (7.0, 5.3, False), (2.0, 6.0, True), (2.0, 5.9, False)])
def test_pins_only_with_room_to_spare(pinned_gb, free_gb, want, monkeypatch):
    monkeypatch.delenv("FIZGIG_PIN_SWAP", raising=False)
    assert quant.pin_swap_ok(pinned_gb, free_gb) is want


def test_switch_overrides(monkeypatch):
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "0")
    assert quant.pin_swap_ok(1.0, 500) is False
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "1")
    assert quant.pin_swap_ok(50.0, 1) is True


def test_krea2_asks_the_guard(monkeypatch):
    import torch
    from fizgig.krea2.driver import Krea2Driver
    got, asked = {}, []
    monkeypatch.setattr(quant, "pin_swap_ok", lambda gb: asked.append(gb) or True)

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(10, 10) for _ in range(4))

        def enable_block_swap(self, n, cfg):
            got["cfg"] = cfg

        def move_to_device_except_swap_blocks(self, d):
            pass

        def switch_block_swap_for_training(self):
            pass

    Krea2Driver.enable_block_swap(object.__new__(Krea2Driver), DiT(), 2, "cuda", True)
    assert got["cfg"].use_pinned_memory is True and not got["cfg"].h2d_only
    assert asked == [pytest.approx(2 * 110 * 4 / 1e9)]       # the last 2 blocks' weights and biases, fp32
