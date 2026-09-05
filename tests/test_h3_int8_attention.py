"""The H3 int8-attention switch (comfy-kitchen) and its three fallbacks — headless, CPU.

Run: venv/Scripts/python.exe tests/test_h3_int8_attention.py
"""
import os
import sys
import types
import importlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn.functional as F

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


from fizgig.minimax import model as mm

q, k, v = (torch.randn(1, 4, 16, 32) for _ in range(3))
ref = F.scaled_dot_product_attention(q, k, v)


def reset(fake=None):
    """Fresh switch state; `fake` installs a stand-in comfy_kitchen module (None = absent)."""
    mm._INT8_ATTN.update({"wanted": False, "checked": False, "fn": None})
    sys.modules.pop("comfy_kitchen", None)
    if fake is not None:
        sys.modules["comfy_kitchen"] = fake


# 1. off by default: plain SDPA, the package is never even imported
reset()
ck("off by default: PyTorch attention, comfy-kitchen not imported",
   torch.equal(mm.h3_attention(q, k, v), ref) and not mm._INT8_ATTN["checked"])

# 2. wanted, package absent -> PyTorch attention, active() says so
_blocker = types.ModuleType("comfy_kitchen")
del _blocker  # (an absent module is simulated by making the import fail)
reset()
_real_import = importlib.import_module
sys.modules["comfy_kitchen"] = None          # `import comfy_kitchen` raises ImportError
mm.set_int8_attention(True)
ck("wanted but the package is missing: PyTorch attention, not 'active'",
   torch.equal(mm.h3_attention(q, k, v), ref) and mm.int8_kernel_available() is False
   and mm._INT8_ATTN["checked"])

# 3. package present, kernel unavailable on this GPU (AMD / old card) -> PyTorch attention
fake = types.ModuleType("comfy_kitchen")
fake.int8_attention_is_available = lambda: False
fake.int8_attention = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called"))
reset(fake); mm.set_int8_attention(True)
ck("package present, no kernel for this GPU: PyTorch attention, kernel never called",
   torch.equal(mm.h3_attention(q, k, v), ref) and mm.int8_kernel_available() is False)

# 4. kernel present -> it is what runs (under no_grad), and 'active' is True
calls = []
fake = types.ModuleType("comfy_kitchen")
fake.int8_attention_is_available = lambda: True
fake.int8_attention = lambda q, k, v, **kw: (calls.append(1), ref * 0.5)[1]
reset(fake); mm.set_int8_attention(True)
with torch.no_grad():
    out = mm.h3_attention(q, k, v)
ck("kernel present + wanted: the kernel runs and 'active' is True",
   len(calls) == 1 and torch.equal(out, ref * 0.5) and mm.int8_kernel_available())
with torch.enable_grad():
    out_g = mm.h3_attention(q.requires_grad_(True), k, v)
ck("...but never under grad (training is never touched)", len(calls) == 1 and torch.allclose(out_g, ref))
q = q.detach()

# 5. a call raising at run time -> PyTorch attention for the rest of the run
fake = types.ModuleType("comfy_kitchen")
fake.int8_attention_is_available = lambda: True
fake.int8_attention = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
reset(fake); mm.set_int8_attention(True)
with torch.no_grad():
    o1 = mm.h3_attention(q, k, v); o2 = mm.h3_attention(q, k, v)
ck("kernel raises: falls back to PyTorch attention for the rest of the run",
   torch.equal(o1, ref) and torch.equal(o2, ref) and mm._INT8_ATTN["fn"] is None
   and mm.int8_kernel_available() is False)

# 6. the engine sets the switch only for its render and clears it after
reset()
from fizgig.repair_studio.h3_engine import H3RepairEngine as _H3E
class _E:
    int8_attention = True
seen = []
src = _H3E.render_latent.__code__
ck("engine: the _render wrapper sets the switch and clears it in a finally",
   "set_int8_attention" in src.co_names or any("set_int8_attention" in (c.co_names if hasattr(c, "co_names") else ())
                                              for c in src.co_consts if hasattr(c, "co_names")))
reset()
mm.set_int8_attention(True); mm.set_int8_attention(False)
ck("switch off again after a render", not mm.int8_attention_wanted())
fake = types.ModuleType("comfy_kitchen")
fake.int8_attention_is_available = lambda: True
fake.int8_attention = lambda q, k, v, **kw: ref
reset(fake)
ck("kernel availability reads True with the per-render switch OFF (the clip dict is tagged after the render)",
   mm.int8_kernel_available() and not mm.int8_attention_wanted())

# 6b. the engine wants it by default — no switch anywhere
ck("the engine asks for int8 attention by default (silently PyTorch's where the kernel is missing)",
   _H3E().int8_attention is True)

# 6c. the REAL clip_from_cache (library peeks, history views, the pinned baseline) builds
# its dict without a NameError — v5.3.0 shipped one that only render_clip had imported, and
# every peek died on it; the GUI battery's stub engine hid it (Peter's console, 5 Sep).
from PIL import Image as _Image
class _CacheStub:
    def get(self, sig): return (torch.zeros(1, 24, 2, 4, 4), None)
    def info(self, sig): return {"label": "Block 3 off"}
class _PeekEng:
    int8_attention = True
    clip_from_cache = _H3E.clip_from_cache
    _int8_tag = _H3E._int8_tag
    def regime_params(self, regime, steps, turbo): return 4, 1.0
    def decode_clip_frames(self, lat): return [_Image.new("RGB", (8, 8)) for _ in range(5)]
    def decode_audio(self, aud): return None
reset()
_clip = _PeekEng().clip_from_cache(_CacheStub(), "off:h3blk_3", regime="dial", with_audio=True)
ck("the real clip_from_cache builds a peek's clip dict (no NameError), tagged and labelled",
   _clip is not None and _clip["frames_n"] == 5 and _clip["label"] == "Block 3 off"
   and _clip["cached"] is True and _clip["int8_attention"] in (True, False))

# 7. the render-cache setup key changes with the flag (an int8 library never serves an exact render)
from fizgig.repair_studio.h3_render_cache import setup_key, CACHE_FORMAT
kw = dict(primary_hash="p", donor_hash="", prompt="x", seed=1, frames=22, width=768, height=640,
          steps=4, turbo_strength=1.0, keyframe_sig=())
ck("setup key differs with int8 attention on (CACHE_FORMAT 5)",
   setup_key(**kw) != setup_key(int8_attention=True, **kw) and CACHE_FORMAT == 5)

reset()
print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
