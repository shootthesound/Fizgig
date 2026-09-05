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
    # banks: five main blocks off, stride four, the last absorbs 49; refiners never in one
    ck("12 banks: 0-4, 4-8 ... 40-44, 44-49 (overlap of one, the last takes the remainder)",
       [b[0] for b in RC.all_banks()] == [f"bank:{a}-{a + 4}" for a in range(0, 44, 4)] + ["bank:44-49"]
       and RC.all_banks()[-1][2] == [f"h3blk_{i}" for i in range(44, 50)]
       and RC.all_banks()[1][1] == "Blocks 4–8 off")
    st5 = SliderState.default_h3()
    for _i in range(4, 9):
        st5.blocks[f"h3blk_{_i}"].primary_enabled = False
    ck("blocks 4-8 unticked -> bank:4-8", RC.signature(st5) == "bank:4-8")
    st5.blocks["h3blk_9"].primary_enabled = False
    ck("blocks 4-9 off -> a hash (not a bank)", len(RC.signature(st5)) == 16)
    st6 = SliderState.default_h3()
    for _i in range(44, 50):
        st6.blocks[f"h3blk_{_i}"].primary_strength = 0.0
    ck("blocks 44-49 at 0 -> bank:44-49 (strength 0 and unticked are the same render)",
       RC.signature(st6) == "bank:44-49")
    ck("bank_ids parses a bank signature and nothing else",
       RC.bank_ids("bank:8-12") == [f"h3blk_{i}" for i in range(8, 13)] and RC.bank_ids("off:h3blk_3") is None)

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
    BANKS = ["bank:0-4", "bank:4-8", "bank:28-32"]        # the banks holding a touched main block
    ck("the LoRA's banks: those holding a block it touches (refiners in none)",
       c.bank_sigs() == BANKS and c.bank_blocks("bank:4-8") == [f"h3blk_{i}" for i in range(4, 9)]
       and c.bank_label("bank:28-32") == "Blocks 28–32 off")
    ck("fresh: nothing cached", not c.complete() and c.missing() == BANKS and c.n_entries() == 0)
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
    ck("hand renders (single-block off) are entries but not banks",
       c.done_banks() == [] and c.missing() == BANKS and c.n_entries() == 3)
    c.put("bank:4-8", lat + 3, None, regime="dial", label="Blocks 4–8 off")
    ck("done/missing follow the banks",
       c.done_banks() == ["bank:4-8"] and c.missing() == ["bank:0-4", "bank:28-32"])
    ck("info + entries carry the meta", c.info("off:h3blk_7").get("label") == "Block 7 off"
       and list(c.entries())[-1] == "bank:4-8")
    ck("a never-rendered state is a miss", c.get("deadbeefdeadbeef") is None)

    # --- resume from disk ---------------------------------------------------------------
    c2 = RC.RenderCache(root, key, ids)
    ck("fresh instance resumes every entry",
       c2.n_entries() == 4 and torch.equal(c2.get("off:h3blk_30")[0], lat + 2)
       and c2.info(RC.BASE_SIG).get("label") == "Baseline" and c2.has("bank:4-8"))
    open(os.path.join(c2.dir, "off__h3blk_0.safetensors.tmp"), "wb").close()
    c3 = RC.RenderCache(root, key, ids)
    ck("a stray .tmp is not an entry", not c3.has("off:h3blk_0"))
    os.remove(c3._path("off:h3blk_7"))
    c4 = RC.RenderCache(root, key, ids)
    ck("manifest entry whose file vanished is dropped", not c4.has("off:h3blk_7") and c4.has(RC.BASE_SIG))
    c4.put("bank:0-4", lat, None); c4.put("bank:28-32", lat, None)
    ck("complete once base + every bank entry exist", c4.complete())
    ck("size reported", c4.size_bytes() > 0)

    ck("build_order: banks ascending by first block",
       RC.build_order(["bank:28-32", "bank:0-4", "bank:4-8"]) == ["bank:0-4", "bank:4-8", "bank:28-32"])

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
