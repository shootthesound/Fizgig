"""#372: a preview helper asks a model that plans itself (MiniMax H3, own_plan) for its base the way the Batch size note
does (driver.plan_pictures), at the run's precision, and applies the trainer's preview caps; every other model plans
from its measured figures as before. CPU only: the planner's answer and the card size are faked."""
from types import SimpleNamespace

import pytest
import torch

from fizgig.families import preview_worker, quant
from fizgig.families.registry import get

SETUP = {"paths": {"dit": "h3.safetensors", "context": None}, "network": {"type": "lora", "dim": 8, "factor": 8}}


def test_h3_asks_its_own_planner_and_takes_the_16gb_preview_caps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=16.3e9))
    desc = get("minimax")
    d = desc.load_driver()
    d.set_options({"preview_frames": "56"})
    asked = []
    monkeypatch.setattr(d, "plan_pictures", lambda p, swap, **kw: asked.append((p, swap, kw)) or ("int8", 34))
    assert preview_worker.plan(desc, d, SETUP, "int8")[:2] == ("int8", 34)
    p, swap, kw = asked[0]
    assert (p, swap, kw["pictures"]) == ("int8", -1, 1)                 # the run's precision, swap planned
    assert kw["run"]["dit_path"] == "h3.safetensors" and kw["run"]["network_dim"] == 8
    assert int(d.options["preview_frames"]) == 22                       # the trainer's 16 GB preview caps


@pytest.mark.parametrize("key", ["krea2", "minimax_image", "klein"])
def test_models_with_figures_plan_as_before(monkeypatch, key):
    desc = get(key)
    assert not desc.own_plan
    monkeypatch.setattr(quant, "plan", lambda *a, **k: ("int8", 3, "shared"))
    assert preview_worker.plan(desc, desc.load_driver(), SETUP, "int8") == ("int8", 3, "shared")
