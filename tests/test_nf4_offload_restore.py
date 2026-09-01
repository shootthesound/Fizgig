"""NF4 park/restore must bring BOTH halves of the model back.

Regression for issue #17 — training died at the first step after an auto-recaption with

    RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
    cuda:0, different from other tensors on cpu

The trainer parks the DiT on CPU to make room for Qwen3-VL. An NF4 model is split across
two storage mechanisms, so parking takes two calls:

    dit.to("cpu")                    # ordinary params/buffers
    move_nf4_to_device(dit, "cpu")   # _nf4_packed / _nf4_state — plain attrs .to() can't see

The restore had them as `if nf4: ... elif swap: ... else: dit.to(device)` — mutually
exclusive — so on a 4-bit run only the packed weights returned and every ordinary
parameter stayed on CPU. compute_loss read its device from the first parameter it found,
sent the whole batch to CPU, and the first CUDA-resident tensor it met raised.

The two calls are complementary, never alternatives. That is what this pins down.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import torch.nn as nn

from fizgig.modules.nf4 import move_nf4_to_device

DEV = "cuda"
TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "src", "fizgig", "krea2", "trainer.py")


class FakeNF4Linear(nn.Module):
    """Mimics the storage split: an ordinary bias .to() moves, plus packed data it cannot see."""

    def __init__(self, n=8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))                     # .to() moves this
        self._is_nf4 = True
        self._nf4_packed = torch.zeros(n, n // 2, dtype=torch.uint8)  # .to() cannot see this
        self._nf4_state = None


class FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)   # the module that actually stranded in the report
        self.blocks = nn.ModuleList([FakeNF4Linear() for _ in range(3)])
        self._nf4_quantized = True


def fresh():
    dit = FakeDiT().to(DEV)
    move_nf4_to_device(dit, DEV)
    return dit


def park(dit):
    dit.to("cpu")
    move_nf4_to_device(dit, "cpu")


def devices(dit):
    ordinary = {p.device.type for p in dit.parameters()}
    packed = {m._nf4_packed.device.type for m in dit.modules()
              if getattr(m, "_is_nf4", False)}
    return ordinary, packed


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -> ' + detail) if detail and not cond else ''}")
    return bool(cond)


def main():
    if not torch.cuda.is_available():
        print("no CUDA available — skipped")
        return 0
    ok = True
    print("NF4 park / restore")

    # 1. parking moves both halves
    dit = fresh()
    park(dit)
    o, p = devices(dit)
    ok &= check("park moves ordinary params to CPU", o == {"cpu"}, str(o))
    ok &= check("park moves packed weights to CPU", p == {"cpu"}, str(p))

    # 2. the bug — restoring only the NF4 half strands everything else
    dit = fresh()
    park(dit)
    move_nf4_to_device(dit, DEV)                    # the old, exclusive restore
    o, p = devices(dit)
    ok &= check("packed-only restore returns the packed weights", p == {"cuda"}, str(p))
    ok &= check("packed-only restore STRANDS ordinary params (the bug)", o == {"cpu"},
                f"{o} — if this fails, move_nf4_to_device now moves them too "
                "and the trainer fix can be simplified")

    # 3. the fix — placement first, then the packed weights
    dit = fresh()
    park(dit)
    dit.to(DEV)                                     # the call the NF4 branch used to skip
    move_nf4_to_device(dit, DEV)
    o, p = devices(dit)
    ok &= check("both-halves restore returns ordinary params", o == {"cuda"}, str(o))
    ok &= check("both-halves restore returns packed weights", p == {"cuda"}, str(p))

    # 4. static guard so the shape can't regress at any of the three park/restore pairs
    print("trainer restore sites")
    with open(TRAINER, encoding="utf-8") as f:
        src = f.read()
    exclusive = ('if getattr(dit, "_nf4_quantized", False):\n'
                 '            from fizgig.modules.nf4 import move_nf4_to_device\n'
                 '            move_nf4_to_device(dit, device)\n'
                 '        elif blocks_to_swap > 0:')
    ok &= check("no restore gates .to(device) behind an NF4 elif", exclusive not in src,
                "an NF4 restore is exclusive again — see issue #17")
    n = src.count("move_nf4_to_device(dit, device)")
    ok &= check("all three restore sites present", n == 3, f"found {n}")
    ok &= check("compute_loss takes an explicit device",
            re.search(r"def compute_loss\([^)]*device=None", src) is not None)

    print()
    print("all passed" if ok else "FAILURES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
