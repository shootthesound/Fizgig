"""The H3 VRAM plan must count everything that is resident for the whole run.

Until 10 Sep 2026 it counted only the trainable adapter's weights + optimizer state. Three other
things sit on the card from setup to the last step and were invisible to it: the frozen training
adapter (on by default in every H3 preset — 0.155 GB for the v1 file Fizgig downloads, twice
that if a user points the path at upstream's rank-32 v2), a Context LoRA, and the EMA shadow — which is FP32, so four bytes per trainable parameter, 1.25 GB on LoKR factor
8. That last one alone is most of the planner's 1.5 GB reserve.

These pin the arithmetic and the gating, both of which fail silently: a plan that is 1.5 GB
optimistic does not raise, it OOMs mid-run on someone else's card.

Run: venv\\Scripts\\python.exe tests\\test_minimax_vram_terms.py
"""
import os
import struct
import sys
import json
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.minimax.trainer import (          # noqa: E402
    frozen_lora_vram_gb, ema_shadow_gb, adapter_vram_gb, plan_base_quant,
    plan_adapter_gb)

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def fake_safetensors(path, shapes, dtype="F32"):
    """A header-only safetensors file: the planner never reads past the header, and building one
    keeps this test free of the 155 MB real adapter."""
    hdr, off = {}, 0
    for i, shape in enumerate(shapes):
        n = 1
        for d in shape:
            n *= d
        nbytes = n * (4 if dtype == "F32" else 2)
        hdr[f"t{i}"] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(hdr).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        # the tensor data itself is never read by the header parser
    return path


tmp = tempfile.mkdtemp(prefix="fizgig_vram_")

# --- 1. frozen_lora_vram_gb: elements x the TRAINING dtype, not the file's -----------------------
# load_context_lora does net.to(dtype=dtype), so an fp32 file still lands as bf16 on the card.
p32 = fake_safetensors(os.path.join(tmp, "fp32.safetensors"), [(1000, 1000)], dtype="F32")
ck("counts elements at the training dtype (2 bytes), not the file's 4",
   abs(frozen_lora_vram_gb(p32) - 1000 * 1000 * 2 / 1e9) < 1e-12, frozen_lora_vram_gb(p32))
ck("bytes_per_elem is honoured when asked",
   abs(frozen_lora_vram_gb(p32, 4) - 1000 * 1000 * 4 / 1e9) < 1e-12)
multi = fake_safetensors(os.path.join(tmp, "multi.safetensors"),
                         [(32, 7168), (7168, 32), (16, 512)], dtype="BF16")
ck("sums every tensor in the file",
   abs(frozen_lora_vram_gb(multi) - (32 * 7168 + 7168 * 32 + 16 * 512) * 2 / 1e9) < 1e-12)

# --- 2. it must never take the plan down with it -------------------------------------------------
ck("no path -> 0.0", frozen_lora_vram_gb(None) == 0.0 and frozen_lora_vram_gb("") == 0.0)
ck("missing file -> 0.0", frozen_lora_vram_gb(os.path.join(tmp, "nope.safetensors")) == 0.0)
junk = os.path.join(tmp, "junk.safetensors")
open(junk, "wb").write(b"not a safetensors file at all")
ck("unreadable header -> 0.0 (plan as before, never raise)", frozen_lora_vram_gb(junk) == 0.0)
ck("a directory -> 0.0", frozen_lora_vram_gb(tmp) == 0.0)

# --- 3. ema_shadow_gb: FP32, four bytes ----------------------------------------------------------
ck("EMA shadow is fp32 (4 bytes/param)", ema_shadow_gb(39_000_000) == 39_000_000 * 4 / 1e9)
ck("LoKR factor 8's shadow is 1.25 GB", abs(ema_shadow_gb(313_000_000) - 1.252) < 1e-9,
   ema_shadow_gb(313_000_000))
ck("zero params -> 0.0", ema_shadow_gb(0) == 0.0)

# --- 4. the terms are big enough to move a real plan ---------------------------------------------
# Sized from the adapter Fizgig actually ships (Preferences downloads ostris v1, rank 16). A user
# who points the path at upstream's rank-32 v2 gets twice this; nothing here may assume either, so
# the file on disk is measured when it is present and the shipped figure is used when it is not.
_ADAPTER = os.path.join(REPO, "models", "minimax_h3_training_adapter_v1.safetensors")
_adapter_gb = frozen_lora_vram_gb(_ADAPTER) if os.path.isfile(_ADAPTER) else 0.155
ck("the shipped training adapter is ~0.155 GB resident", 0.14 < _adapter_gb < 0.17, f"{_adapter_gb:.3f} GB")

# The EMA shadow ALONE is most of the planner's 1.5 GB reserve on a big trainable adapter. That is
# the claim that does not depend on which adapter file the user has.
ck("LoKR factor 8's EMA shadow alone is most of the 1.5 GB reserve",
   ema_shadow_gb(313_000_000) > 0.8 * 1.5, f"{ema_shadow_gb(313_000_000):.2f} GB")

_lokr = adapter_vram_gb(313_000_000, "adamw")
_missing = ema_shadow_gb(313_000_000) + _adapter_gb
before = plan_base_quant(14.6, True, mp=0.25, adapter_gb=_lokr)
after = plan_base_quant(14.6, True, mp=0.25, adapter_gb=_lokr + _missing)
ck("LoKR at 16 GB: counting them changes the BASE PRECISION, not just the swap count",
   before[0] == "int8" and after[0] == "nf4", f"{before[0]} -> {after[0]}")

# The shipped default is LoRA, not LoKR — every built-in H3 preset and the CLI default. So the
# honest headline for most users is "one more swapped block", and that must stay true.
_r8 = adapter_vram_gb(39_000_000, "adamw")
_r8_missing = ema_shadow_gb(39_000_000) + _adapter_gb
ck("rank 8's uncounted total is well under the reserve", _r8_missing < 0.5, f"{_r8_missing:.2f} GB")
for free in (10.8, 14.6, 22.4):
    a = plan_base_quant(free, True, mp=0.25, adapter_gb=_r8)
    b = plan_base_quant(free, True, mp=0.25, adapter_gb=_r8 + _r8_missing)
    ck(f"rank 8 at {free} GB free: same base, swap +1 at most",
       a[0] == b[0] and 0 <= b[1] - a[1] <= 1, f"{a[0]} {a[1]} -> {b[0]} {b[1]}")

# --- 5. the gating, exercised rather than grepped -------------------------------------------------
# This used to assert that certain lines EXIST in trainer.py, which passes whether or not they
# ever run — and it was the only coverage the gates had. plan_adapter_gb exists so the same
# decisions can be called directly.
P = 39_000_000
_base = adapter_vram_gb(P, "adamw")
_ema = ema_shadow_gb(P)

t, fr, em = plan_adapter_gb(P, "adamw")
ck("nothing configured -> the trainable adapter only", (t, fr, em) == (_base, 0.0, 0.0), (t, fr, em))

t, fr, em = plan_adapter_gb(P, "adamw", training_adapter_path=multi)
ck("a training adapter is counted", abs(fr - frozen_lora_vram_gb(multi)) < 1e-12 and em == 0.0, (fr, em))

t, fr, em = plan_adapter_gb(P, "adamw", context_lora_path=multi)
ck("a Context LoRA is counted on its own", abs(fr - frozen_lora_vram_gb(multi)) < 1e-12, fr)

t, fr, em = plan_adapter_gb(P, "adamw", training_adapter_path=multi, context_lora_path=p32)
ck("both frozen files are counted, not just the first",
   abs(fr - (frozen_lora_vram_gb(multi) + frozen_lora_vram_gb(p32))) < 1e-12, fr)

t, fr, em = plan_adapter_gb(P, "adamw", ema_decay=0.98)
ck("EMA on -> the fp32 shadow is counted", abs(em - _ema) < 1e-12 and fr == 0.0, (fr, em))
ck("...and it lands in the total", abs(t - (_base + _ema)) < 1e-12, t)
for off in (0.0, 0, None, ""):
    _, _, em = plan_adapter_gb(P, "adamw", ema_decay=off)
    ck(f"EMA off ({off!r}) -> no shadow", em == 0.0, em)

# Under FT rotation none of the three is resident: the frozen pair are refused at load time and
# the FT coercion block forces ema_decay to 0 before the plan runs. Counting them there would
# plan for memory nothing will use.
t, fr, em = plan_adapter_gb(P, "adamw", training_adapter_path=multi, context_lora_path=p32,
                            ema_decay=0.98, ft_rotation=4)
ck("FT rotation -> frozen LoRAs and EMA are BOTH excluded",
   (t, fr, em) == (_base, 0.0, 0.0), (t, fr, em))

# And the call site really uses it, with the run's own values.
src = open(os.path.join(REPO, "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
ck("the planner calls plan_adapter_gb with the run's paths, decay and FT state",
   "_adapter, _frozen_gb, _ema_gb = plan_adapter_gb(" in src
   and "training_adapter_path=training_adapter_path" in src
   and "context_lora_path=context_lora_path" in src
   and "ema_decay=ema_decay" in src and "ft_rotation=ft_rotation" in src)
ck("the plan uses the same include_patterns the run will (user's first)",
   "_pat = list(user_include_patterns or" in src)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL PASS")
