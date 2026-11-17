"""#372: a preview helper loads a family that plans itself (MiniMax H3, own_plan) as that family plans for the
helper's own card, not from the shared train_memory table H3 doesn't have (where an int8 run "fit" a 16 GB card with
nothing streamed, and the helper ran out of memory loading it). CPU only: free VRAM and the card size are faked."""
from types import SimpleNamespace

import pytest
import torch

from fizgig.families import preview_worker, quant
from fizgig.families.registry import get
from fizgig.minimax import common
from fizgig.utils import device as dev


def _driver(monkeypatch, free_gb, total_gb, frames="56", int8_ok=True):
    monkeypatch.setattr(dev, "plannable_free_vram", lambda *a, **k: free_gb)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=total_gb * 1e9))
    monkeypatch.setattr(common, "int8_loadable", lambda path: int8_ok)
    desc = get("minimax")
    d = desc.load_driver()
    d.set_options({"preview_frames": frames})
    return desc, d


def test_a_16gb_helper_loads_the_4bit_base_with_the_small_card_caps(monkeypatch):
    desc, d = _driver(monkeypatch, free_gb=14.7, total_gb=16.3)
    precision, swap, why = preview_worker.plan(desc, d, "int8", "h3.safetensors")
    assert (precision, swap) == ("nf4", 0)
    assert int(d.options["preview_frames"]) == 22 and d._small_card


def test_a_32gb_helper_keeps_int8_with_room_streamed_for_the_clip(monkeypatch):
    desc, d = _driver(monkeypatch, free_gb=30.5, total_gb=32.6, frames="22")
    precision, swap, _ = preview_worker.plan(desc, d, "int8", "h3.safetensors")
    assert precision == "int8" and swap >= 1          # 22 GB at rest + a clip + margin is more than 30.5


def test_never_more_precise_than_the_run_and_never_int8_from_a_file_that_cannot(monkeypatch):
    desc, d = _driver(monkeypatch, free_gb=30.5, total_gb=32.6)
    assert preview_worker.plan(desc, d, "nf4", "h3.safetensors")[:2] == ("nf4", 0)
    desc, d = _driver(monkeypatch, free_gb=30.5, total_gb=32.6, int8_ok=False)
    assert preview_worker.plan(desc, d, "int8", "h3_full.safetensors")[:2] == ("nf4", 0)


@pytest.mark.parametrize("key", ["krea2", "minimax_image"])
def test_families_with_figures_keep_the_shared_planner(monkeypatch, key):
    desc = get(key)
    assert not desc.own_plan
    seen = []
    monkeypatch.setattr(quant, "plan", lambda *a, **k: seen.append(a) or ("int8", 3, "shared"))
    assert preview_worker.plan(desc, desc.load_driver(), "int8", "x.safetensors") == ("int8", 3, "shared")
    assert seen
