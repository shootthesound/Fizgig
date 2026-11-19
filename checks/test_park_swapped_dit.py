"""#377: a block-swapped DiT's resident part parks on the CPU while the text encoder or VAE runs (a 12 GB card can't
hold both), and comes back in its streaming layout only when it was parked."""
import torch

from fizgig.families import quant
from fizgig.families.workbench import WorkbenchEngine


class _DiT(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.calls = []

    def move_to_device_except_swap_blocks(self, device):
        self.calls.append(("resident", str(device)))


class _Driver:
    def __init__(self):
        self.modes = []

    def block_swap_mode(self, dit, inference):
        self.modes.append(inference)


def _engine(monkeypatch, swapped, free=4.0):
    from fizgig.families import workbench
    monkeypatch.setattr(workbench, "_free_vram_gb", lambda: free)
    moves = []
    monkeypatch.setattr(quant, "move", lambda dit, where: moves.append(str(where)))
    e = WorkbenchEngine.__new__(WorkbenchEngine)
    e.dit, e.driver, e.swapped, e._swap_parked = _DiT(), _Driver(), swapped, False
    return e, moves


def test_swapped_dit_parks_and_comes_back(monkeypatch):
    e, moves = _engine(monkeypatch, 16)
    e._park_dit("cpu")
    e._park_dit("cpu")                                  # parked already: nothing more
    assert moves == ["cpu"]
    e._park_dit("cuda")
    assert e.dit.calls == [("resident", "cuda")] and e.driver.modes == [True] and not e._swap_parked


def test_not_parked_is_left_alone(monkeypatch):
    e, moves = _engine(monkeypatch, 16)
    e._park_dit("cuda")                                 # after a decode that didn't park: no re-layout
    assert moves == [] and e.dit.calls == [] and e.driver.modes == []


def test_unswapped_moves_as_before(monkeypatch):
    e, moves = _engine(monkeypatch, 0)
    e._park_dit("cpu")
    e._park_dit("cuda")
    assert moves == ["cpu", "cuda"] and e.dit.calls == []


def test_room_enough_stays_put(monkeypatch):
    e, moves = _engine(monkeypatch, 16, free=9.0)      # a 16 GB card: the encoder fits beside it, no 8-16 s move
    e._park_dit("cpu")
    e._park_dit("cuda")
    assert moves == [] and e.dit.calls == []
