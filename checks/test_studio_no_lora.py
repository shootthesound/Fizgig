"""Studio with no LoRA (the base model on its own): the zero adapter that stands in for the primary changes nothing at
any strength or slider, touches no block (every slider greys), and the server takes an empty primary but refuses
Save repaired LoRA."""
import types

import pytest
import torch
import torch.nn as nn

lora = pytest.importorskip("fizgig.families.lora")
studio = pytest.importorskip("fizgig.web.studio")


class Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)) for _ in range(2)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _net():
    dit = Dit()
    desc = types.SimpleNamespace(display_name="Test", lora=types.SimpleNamespace(kohya=True, file_prefix="", down="lora_down",
                                                                                 up="lora_up", alpha_key="{prefix}.alpha"))
    drv = types.SimpleNamespace(description=desc, lora_target_names=lambda d: [n for n, m in d.named_modules()
                                                                                  if isinstance(m, nn.Linear)],
                                block_of=lambda full: f"b{full.split('.')[1]}")
    return dit, lora.FamilyLoRA(dit, drv)


def test_zero_adapter_changes_nothing():
    torch.manual_seed(0)
    dit, net = _net()
    x = torch.randn(3, 8)
    with torch.no_grad():
        before = dit(x)
        assert net.add_zero("primary") == 1
        for mult in (1.0, 3.0, -2.0):
            for full, w in net.wrapped.items():
                if "primary" in w.scales:
                    w.scales["primary"] = mult
            assert torch.equal(dit(x), before)


def test_server_takes_an_empty_primary_and_refuses_the_save(monkeypatch):
    with pytest.raises(ValueError, match="isn't there"):
        studio.load(types.SimpleNamespace(key="x"), "Z:/no/such.safetensors", "", {}, None, "", True)
    studio._S.update(state="ready", primary="")
    with pytest.raises(ValueError, match="No LoRA is loaded"):
        studio.save("out.safetensors", {}, False)
    studio._S.update(state="idle")


def test_unload_primary_goes_to_the_worker_and_queue_jobs_load_again(monkeypatch):
    sent = []
    monkeypatch.setattr(studio, "_send", lambda m: sent.append(m))
    studio.primary("")
    assert sent == [{"cmd": "primary", "path": ""}]
    studio._S.update(load_key='{"primary": "A"}', page_key='{"primary": "A"}', primary="A")
    studio._event({"ev": "primary", "primary": "", "primary_blocks": [], "donor_blocks": [], "hash": "", "eq": {}})
    assert studio._S["primary"] == "" and studio._S["load_key"] is None and studio._S["page_key"] is None
