"""Exact render cache (repair_studio.h3_render_cache) — signatures, store, resume, clear.

Synthetic latents, no models. Pins: the signature rules (base / off:<bid> / hashed; a
disabled block is "off"; a donor row makes it a hash; block order doesn't matter); put/get
round-trips (fp16 latent, fp32 audio) with the jpg thumb; the manifest is written last and a
stray .tmp or a vanished file is never vouched for; a fresh instance on the same dir resumes;
done/missing/complete track the LoRA's blocks; build_order is ascending with refiners last;
clear removes only cached setups.

Run: venv/Scripts/python.exe tests/test_h3_render_cache.py
"""
import os
import shutil
import sys
import tempfile

import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.repair_studio import h3_render_cache as RC                   # noqa: E402
from fizgig.repair_studio.state import SliderState                        # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


root = tempfile.mkdtemp(prefix="fizgig_rcache_")
try:
    # --- signatures ---------------------------------------------------------------------
    st = SliderState.default_h3()
    ck("default state -> base", RC.signature(st) == RC.BASE_SIG)
    st.blocks["h3blk_30"].primary_strength = 0.0
    ck("one block at 0 -> off:<bid>", RC.signature(st) == "off:h3blk_30")
    st.blocks["h3blk_30"].primary_strength = 1.0
    st.blocks["h3blk_30"].primary_enabled = False
    ck("one block disabled -> the same off:<bid> (same render)", RC.signature(st) == "off:h3blk_30")
    st.blocks["h3blk_30"].primary_enabled = True
    st.blocks["h3blk_30"].primary_strength = 0.5
    h1 = RC.signature(st)
    ck("one block at 0.5 -> a hash", len(h1) == 16 and h1 not in (RC.BASE_SIG, "off:h3blk_30"))
    st.blocks["h3blk_3"].primary_strength = 2.0
    h2 = RC.signature(st)
    st2 = SliderState.default_h3()
    st2.blocks["h3blk_3"].primary_strength = 2.0          # set in the other order
    st2.blocks["h3blk_30"].primary_strength = 0.5
    ck("hash is order-independent and state-specific", RC.signature(st2) == h2 and h2 != h1)
    st3 = SliderState.default_h3()
    st3.blocks["h3blk_30"].primary_strength = 0.0
    st3.blocks["h3blk_30"].donor_strength = 0.7
    ck("a donor row turns off:<bid> into a hash", RC.signature(st3) not in ("off:h3blk_30", RC.BASE_SIG))
    st4 = SliderState.default_h3()
    st4.blocks["h3_rf_1"].primary_strength = 0.0
    ck("refiner off -> off:h3_rf_1", RC.signature(st4) == "off:h3_rf_1")

    # --- store --------------------------------------------------------------------------
    ids = ["h3blk_0", "h3blk_7", "h3blk_30", "h3_rf_0"]
    key = RC.setup_key(primary_hash="abc", donor_hash="", prompt="p", seed=7, frames=22,
                       width=512, height=416, steps=4, turbo_strength=1.0, keyframe_sig=())
    key_full = RC.setup_key(primary_hash="abc", donor_hash="", prompt="p", seed=7, frames=22,
                            width=768, height=640, steps=4, turbo_strength=1.0, keyframe_sig=())
    key_conf = RC.setup_key(primary_hash="abc", donor_hash="", prompt="p", seed=7, frames=22,
                            width=512, height=416, steps=6, turbo_strength=0.75, keyframe_sig=())
    ck("setup key: canvas and regime each make a different setup",
       len({key, key_full, key_conf}) == 3 and len(key) == 20)
    c = RC.RenderCache(root, key, ids)
    ck("fresh: nothing cached", not c.complete() and c.missing() == ids and c.n_entries() == 0)
    lat = torch.randint(-16, 16, (1, 24, 7, 8, 8)).float() * 0.25   # fp16-exact, and so are +1/+2
    aud = torch.randn(74, 32)
    thumb = Image.new("RGB", (768, 640), "orange")
    c.put(RC.BASE_SIG, lat, aud, middle=thumb, regime="dial", label="Baseline")
    c.put("off:h3blk_7", lat + 1, aud + 1, middle=thumb, regime="dial", label="Block 7 off")
    c.put("off:h3blk_30", lat + 2, None, regime="dial", label="Block 30 off")
    ck("has/get round-trip (fp16 latent exact for these values, fp32 audio)",
       c.has("off:h3blk_7") and torch.equal(c.get("off:h3blk_7")[0], lat + 1)
       and torch.allclose(c.get("off:h3blk_7")[1], aud + 1))
    ck("entry without audio -> None audio", c.get("off:h3blk_30")[1] is None)
    ck("thumb written as a <=256 px jpg",
       os.path.isfile(c.thumb_path("off:h3blk_7"))
       and max(Image.open(c.thumb_path("off:h3blk_7")).size) <= 256)
    ck("done/missing follow the LoRA's blocks",
       c.done_ids() == ["h3blk_7", "h3blk_30"] and c.missing() == ["h3blk_0", "h3_rf_0"])
    ck("info + entries carry the meta", c.info("off:h3blk_7").get("label") == "Block 7 off"
       and list(c.entries())[-1] == "off:h3blk_30")
    ck("a never-rendered state is a miss", c.get("deadbeefdeadbeef") is None)

    # --- resume from disk ---------------------------------------------------------------
    c2 = RC.RenderCache(root, key, ids)
    ck("fresh instance resumes every entry",
       c2.n_entries() == 3 and torch.equal(c2.get("off:h3blk_30")[0], lat + 2)
       and c2.info(RC.BASE_SIG).get("label") == "Baseline")
    open(os.path.join(c2.dir, "off__h3blk_0.safetensors.tmp"), "wb").close()
    c3 = RC.RenderCache(root, key, ids)
    ck("a stray .tmp is not an entry", not c3.has("off:h3blk_0"))
    os.remove(c3._path("off:h3blk_7"))
    c4 = RC.RenderCache(root, key, ids)
    ck("manifest entry whose file vanished is dropped", not c4.has("off:h3blk_7") and c4.has(RC.BASE_SIG))
    c4.put("off:h3blk_7", lat, None); c4.put("off:h3blk_0", lat, None); c4.put("off:h3_rf_0", lat, None)
    ck("complete once base + every block-off entry exist", c4.complete())
    ck("size reported", c4.size_bytes() > 0)

    ck("build_order: main blocks ascending, refiners last",
       RC.build_order(["h3blk_30", "h3_rf_0", "h3blk_7", "h3blk_0"])
       == ["h3blk_0", "h3blk_7", "h3blk_30", "h3_rf_0"])

    os.makedirs(os.path.join(root, "not_a_cache"), exist_ok=True)
    n, freed = RC.clear_render_cache(root)
    ck("clear removes cached setups only", n == 1 and freed > 0 and not os.path.isdir(c4.dir)
       and os.path.isdir(os.path.join(root, "not_a_cache")))
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
