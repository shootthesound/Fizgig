"""Krea 2's block swap keeps the swapped blocks in pinned RAM only when asked (FIZGIG_PIN_SWAP=1) and only with room
to spare: pinning the existing swap was 1.5x on a 5090 with 128 GB, 4% on a 32 GB laptop (free RAM down to 0.2 GB) and
5.5% slower on a 4070 Ti with 64 GB (#403), and pinning with RAM nearly full has failed before (#175)."""
import os
import sys

import pytest

from fizgig.families import quant


def test_off_by_default(monkeypatch, tmp_path):
    from fizgig import user_data
    monkeypatch.setattr(user_data, "PREFS_FILE", str(tmp_path / "prefs.json"))     # no tick saved
    monkeypatch.delenv("FIZGIG_PIN_SWAP", raising=False)
    assert quant.pin_swap_ok(1.0, 500) is False
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "0")
    assert quant.pin_swap_ok(1.0, 500) is False


@pytest.mark.parametrize("pinned_gb, free_gb, want", [(7.0, 100, True), (7.0, 11.0, True), (7.0, 10.9, False),
                                                      (7.0, 5.3, False), (2.0, 6.0, True), (2.0, 5.9, False)])
def test_asked_pins_only_with_room_to_spare(pinned_gb, free_gb, want, monkeypatch):
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "1")
    assert quant.pin_swap_ok(pinned_gb, free_gb) is want


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


@pytest.mark.parametrize("swap", [1, 3, 4])
def test_pinned_swap_trains_as_with_no_swap(swap, monkeypatch):
    """The pinned path, asked for, still trains exactly as with no swap (the simulated card)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_simgpu_families as sim
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "1")
    monkeypatch.setattr(quant, "pin_swap_ok", lambda gb, available_gb=None: True)
    sim.test_family_block_swap_trains_as_with_no_swap("krea2", swap, 1, monkeypatch)
