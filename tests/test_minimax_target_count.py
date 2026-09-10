"""The H3 trainer's wrap-count guard and its VRAM planner must count targeted Linears the way the
network builder matches them (re.fullmatch on the module path). With re.search, an unanchored
'blocks\\.\\d+\\.attn' pattern also catches 'token_refiner.blocks.0.attn', so a run with the text
token refiner off (the default since v5.6.0) counted 208 targets, wrapped 200, and refused to
start: "only 200 of 208 targeted Linears were wrapped" (Peter, 10 Sep 2026)."""
import os, re, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))
from fizgig.minimax.trainer import DEFAULT_INCLUDE_PATTERNS

fails = 0
def ck(label, ok, got=None):
    global fails
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  ({got!r})"))
    fails += (not ok)

names = ([f"blocks.{i}.attn.qkv" for i in range(50)] + [f"blocks.{i}.mlp.fc1" for i in range(50)]
         + [f"token_refiner.blocks.{i}.attn.qkv" for i in range(2)]
         + [f"token_refiner.blocks.{i}.mlp.fc1" for i in range(2)])
off = [p for p in DEFAULT_INCLUDE_PATTERNS if "token_refiner" not in p]
fm = sum(any(re.fullmatch(p, n) for p in off) for n in names)
se = sum(any(re.search(p, n) for p in off) for n in names)
ck("refiner off: fullmatch excludes the refiner's Linears", fm == 100, fm)
ck("refiner off: search would have over-counted them (the bug)", se == 104, se)

src = open(os.path.join(REPO, "src", "fizgig", "minimax", "trainer.py"), encoding="utf-8").read()
ck("wrap-count guard uses fullmatch", "any(_re.fullmatch(p, n) for p in include_patterns)" in src)
ck("VRAM planner uses fullmatch", "any(r.fullmatch(name) for r in rx)" in src)
ck("no search-based target count remains",
   "any(_re.search(p, n) for p in include_patterns)" not in src and "any(r.search(name) for r in rx)" not in src)
print("\nALL PASS" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
