"""triton must pair with torch: the requirements pin and the compile guard's version check.

Run: venv/Scripts/python.exe tests/test_triton_pin.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


from fizgig.utils.capabilities import triton_matches_torch

ck("3.5.1 on torch 2.10: ok (field-proven)", triton_matches_torch("3.5.1", "2.10.0+cu128")[0])
ck("3.6.0 on torch 2.10: ok (the Linux wheel's own pair)", triton_matches_torch("3.6.0", "2.10.0")[0])
ok, why = triton_matches_torch("3.8.0", "2.10.0+cu128")
ck("3.8.0 on torch 2.10: refused, with the fix in the note (the fresh-install case)",
   not ok and "3.5 or 3.6" in why and "pip install" in why, why)
ck("3.4.0 on torch 2.10: refused", not triton_matches_torch("3.4.0", "2.10.0")[0])
ck("an unknown torch is not gated", triton_matches_torch("9.9.9", "3.1.0")[0])
ck("post-release suffixes are ignored", triton_matches_torch("3.6.0.post26", "2.10.0")[0])

req = open(os.path.join(REPO, "requirements.txt"), encoding="utf-8").read()
m = re.search(r"^triton-windows([^;\n]*);", req, re.M)
ck("requirements pins triton-windows below 3.7 and at least 3.5.1",
   m is not None and "<3.7" in m.group(1) and ">=3.5.1" in m.group(1), m.group(0) if m else None)
tm = re.search(r"^torch==(\d+\.\d+)", req, re.M)
ck("...and that pin is the one the helper wants for the pinned torch",
   tm is not None and triton_matches_torch("3.6.0", tm.group(1) + ".0")[0]
   and triton_matches_torch("3.5.1", tm.group(1) + ".0")[0])

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
