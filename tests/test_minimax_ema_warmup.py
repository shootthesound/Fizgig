"""High-LR smoothing (warmup + EMA) — pin the paths that fail SILENTLY.

The dangerous failures here are all quiet ones: an EMA whose swap never fires saves raw
weights while claiming to smooth; a preset string that stops parsing simply emits no flag;
a settings key missing from start_training's hand-curated dict launches with the feature
off while the GUI shows it on (the exact failure the limiter shipped with, twice).

Run: venv/Scripts/python.exe tests/test_minimax_ema_warmup.py
"""
import io as _io
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from fizgig.minimax.trainer import EMAWeights

_fail = []


def ck(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        _fail.append(name)


# --- EMA math + swap exactness ---------------------------------------------------------------
class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)


torch.manual_seed(0)
net = _Net()
net.b.weight.requires_grad_(False)          # frozen params must NOT be tracked
net.b.bias.requires_grad_(False)
ema = EMAWeights(net, 0.99)
ck("ema tracks only trainable params", len(ema.params) == 2)

# manual reference: same ramped decay min(d, (1+n)/(10+n))
ref = [p.detach().clone().float() for p in ema.params]
n = 0
for step in range(25):
    with torch.no_grad():
        for p in ema.params:
            p.add_(torch.randn_like(p) * 0.1)
    ema.update()
    n += 1
    d = min(0.99, (1 + n) / (10 + n))
    for r, p in zip(ref, ema.params):
        r.mul_(d).add_(p.detach().float(), alpha=1 - d)
ck("ema update matches the manual reference exactly",
   all(torch.equal(r, s) for r, s in zip(ref, ema.shadow)))

raw = [p.detach().clone() for p in ema.params]
ema.swap_in()
ck("swap_in puts the shadow into the live params",
   all(torch.allclose(p.detach().float(), s, atol=1e-3)      # via param dtype round-trip
       for p, s in zip(ema.params, ema.shadow)))
ck("swap_in actually changed the weights (feature not a no-op)",
   not all(torch.equal(p.detach(), r) for p, r in zip(ema.params, raw)))
ema.swap_out()
ck("swap_out restores the raw weights bit-exactly",
   all(torch.equal(p.detach(), r) for p, r in zip(ema.params, raw)))

sd = ema.state_dict()
ema2 = EMAWeights(net, 0.99)
ema2.load_state_dict(sd)
ck("state round-trip preserves shadow + step count",
   ema2.n == ema.n and all(torch.equal(a, b) for a, b in zip(ema.shadow, ema2.shadow)))

# --- warmup schedule arithmetic (the exact loop expression) ----------------------------------
warmup_steps = 76           # 2 epochs x 38 steps
base = 1e-4
lrs = [base * min(1.0, (g + 1) / warmup_steps) for g in range(warmup_steps + 5)]
ck("warmup starts near zero, not at full LR", lrs[0] < 2e-6)
ck("warmup hits exactly the configured LR on the last ramp step", lrs[warmup_steps - 1] == base)
ck("warmup never overshoots", max(lrs) == base)

# --- GUI: launch dict + builder flags + preset strings ---------------------------------------
_gui = _io.open(os.path.join(REPO, "lora_trainer_gui.py"), encoding="utf-8").read()
_i = _gui.index("def _start_training_launch")   # the launch dict lives in the helper split out of start_training
_j = _gui.index("def ", _i + 10)
for key in ("MINIMAX_LR_WARMUP", "MINIMAX_EMA"):
    ck(f"start_training's settings dict carries {key}",
       f'"{key}": self.entries["{key}"].get()' in _gui[_i:_j])
# LR warmup was RETIRED (11 Aug) - the row is hidden and forced Off, and the builder must never
# emit the flag, or a stale preset could revive a control that is no longer shown anywhere.
# The warmup MATH above is still exercised because the trainer keeps the code path for the CLI.
ck("builder no longer emits --lr_warmup_epochs (retired)", "--lr_warmup_epochs" not in _gui)
ck("builder emits --ema_decay", "--ema_decay" in _gui)

os.environ["FIZGIG_NO_PERSIST"] = "1"
import lora_trainer_gui as G
G.save_prefs = lambda *a, **k: None
for name, preset in G.MINIMAX_BUILT_IN_PRESETS.items():
    wu = str(preset["MINIMAX_LR_WARMUP"]).split(" ")[0]
    em = str(preset["MINIMAX_EMA"]).split(" ")[0]
    ck(f"preset {name!r}: warmup/EMA strings parse ({wu!r}/{em!r})",
       (wu == "Off" or wu.replace(".", "", 1).isdigit())
       and (em == "Off" or em.replace(".", "", 1).isdigit()))
# Warmup is RETIRED/OFF in the shipped preset (11 Aug): the Adapter-relative LR ramp eases the
# run in by construction. EMA ships ON at 0.98 (9 Sep A/B: +5 pts late likeness, half the
# spread). Looked up by POSITION - the preset has been renamed twice and the name is not what
# matters.
_defaults = list(G.MINIMAX_BUILT_IN_PRESETS.values())[0]
ck("Defaults preset ships MINIMAX_LR_WARMUP Off", str(_defaults["MINIMAX_LR_WARMUP"]).split(" ")[0] == "Off")
for _n, _p in G.MINIMAX_BUILT_IN_PRESETS.items():
    ck(f"preset {_n!r} ships EMA 0.98", str(_p["MINIMAX_EMA"]).split(" ")[0] == "0.98", _p["MINIMAX_EMA"])

print()
if _fail:
    raise SystemExit(f"FAILURES: {_fail}")
print("ALL PASS")
