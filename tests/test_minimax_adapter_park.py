"""The training adapter is parked on the CPU for previews (its modules are disabled for the
render, so it was dead weight on the card) and restored for training. Headless.

Run: venv/Scripts/python.exe tests/test_minimax_adapter_park.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


from fizgig.minimax.trainer import park_frozen_lora, restore_frozen_lora
from fizgig.networks.lora import LoRAInfModule

cuda = torch.cuda.is_available()
dev = torch.device("cuda" if cuda else "cpu")

# a base Linear with one frozen LoRA module wrapped round it, the way the adapter wraps the DiT
base = torch.nn.Linear(16, 16).to(dev, torch.bfloat16)
mod = LoRAInfModule("lora_unet_x", base, multiplier=1.0, lora_dim=4, alpha=4)
mod.to(dev, torch.bfloat16)
mod.apply_to()
net = torch.nn.Module(); net.unet_loras = [mod]; net.add_module("m0", mod)
calls = []
mod.lora_down.register_forward_hook(lambda *a: calls.append(1))
A = torch.randn(3, 8, device=dev, dtype=torch.bfloat16); B = torch.randn(8, 3, device=dev, dtype=torch.bfloat16)
pairs = [(base, A, B)]

x = torch.randn(2, 16, device=dev, dtype=torch.bfloat16)
mod.enabled = False
ref = base(x)                                              # the adapter off = the base alone
moved = park_frozen_lora(net, pairs)
ck("parking moves every LoRA parameter and the AdaLN rows to the CPU and counts the bytes",
   all(not p.is_cuda for p in net.parameters()) and not pairs[0][1].is_cuda and not pairs[0][2].is_cuda
   and (moved > 0 if cuda else moved == 0), moved)
calls.clear()
out = base(x)
ck("with the modules disabled a forward never touches the parked weights and equals the base",
   torch.equal(out, ref) and calls == [])
restore_frozen_lora(net, dev)
ck("restore puts the parameters back on the device in their dtype",
   all(p.device.type == dev.type and p.dtype == torch.bfloat16 for p in net.parameters()))
mod.enabled = True
calls.clear(); _ = base(x)
ck("...and the module runs again once enabled", calls == [1])

os.environ["FIZGIG_NO_ADAPTER_PARK"] = "1"
ck("FIZGIG_NO_ADAPTER_PARK=1 leaves everything where it is",
   park_frozen_lora(net, pairs) == 0 and all(p.device.type == dev.type for p in net.parameters()))
os.environ.pop("FIZGIG_NO_ADAPTER_PARK", None)
ck("None network / no rows is a no-op", park_frozen_lora(None, None) == 0)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
