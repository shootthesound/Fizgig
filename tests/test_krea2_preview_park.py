"""Issue #123 (9 Sep 2026): Krea 2 on-DiT previews on a 16 GB card. The training DiT (NF4 packed
weights included) parks for the VAE decode and comes back afterwards, the preview canvas caps at
768 px on 16 GB-class cards, and [preview-vram] waypoints are logged. Source + signature pins;
no GPU needed.

Run: venv/Scripts/python.exe tests/test_krea2_preview_park.py
"""
import inspect
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


from fizgig.krea2 import sampling, trainer  # noqa: E402

# --- sampling.sample: hooks around the decode, in the right order -----------------------------
sig = inspect.signature(sampling.sample)
ck("sample() takes before_decode / after_decode (keyword, default None)",
   all(k in sig.parameters and sig.parameters[k].default is None for k in ("before_decode", "after_decode")))
src = inspect.getsource(sampling.sample)
i_before = src.find("before_decode()")
i_to = src.find("ae = ae.to(img.device)")
i_dec = src.find("ae.decode_to_pixels(")
i_cpu = src.find('ae = ae.to("cpu")')
i_after = src.find("after_decode()")
ck("before_decode fires BEFORE the VAE moves to the GPU", 0 < i_before < i_to < i_dec)
ck("after_decode fires AFTER the VAE is back on CPU", i_dec < i_cpu < i_after)

# --- the gate: 16 GB-class cards only, simulator- and override-aware --------------------------
_saved = {k: os.environ.get(k) for k in ("FIZGIG_PREVIEW_LOWMEM", "FIZGIG_SIM_VRAM_GB")}
try:
    import torch
    has_cuda = torch.cuda.is_available()
    os.environ.pop("FIZGIG_PREVIEW_LOWMEM", None)
    os.environ["FIZGIG_SIM_VRAM_GB"] = "16"
    ck("simulated 16 GB card → low-memory previews on", trainer._small_card_previews() == has_cuda)
    os.environ["FIZGIG_SIM_VRAM_GB"] = "32"
    ck("simulated 32 GB card → unchanged (off)", trainer._small_card_previews() is False)
    os.environ["FIZGIG_PREVIEW_LOWMEM"] = "1"
    ck("FIZGIG_PREVIEW_LOWMEM=1 forces it on regardless", trainer._small_card_previews() is True)
    os.environ["FIZGIG_PREVIEW_LOWMEM"] = "0"
    os.environ["FIZGIG_SIM_VRAM_GB"] = "16"
    ck("FIZGIG_PREVIEW_LOWMEM=0 forces it off regardless", trainer._small_card_previews() is False)
finally:
    for k, v in _saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# --- sample_previews_on_dit: park for the decode, restore in the right order -------------------
s = inspect.getsource(trainer.sample_previews_on_dit)
ck("the on-DiT path builds a park closure that moves the DiT AND its NF4 packed weights to CPU",
   'dit.to("cpu")' in s and 'move_nf4_to_device(dit, "cpu")' in s and "def _park_for_decode" in s)
ck("the park also moves the Turbo LoRA off the card and empties the cache",
   'turbo_net.to(device="cpu")' in s.split("def _restore_after_decode")[0].split("def _park_for_decode")[1]
   and "empty_cache()" in s.split("def _restore_after_decode")[0].split("def _park_for_decode")[1])
r = s.split("def _restore_after_decode")[1].split("try:")[0]
ck("restore honours block swap (move_to_device_except_swap_blocks) or plain .to(device)",
   "move_to_device_except_swap_blocks" in r and "dit.to(device)" in r)
ck("restore does placement FIRST, then the NF4 packed weights (not an elif)",
   r.find("dit.to(device)") < r.find("move_nf4_to_device(dit, device)")
   and not any(ln.strip().startswith("elif") for ln in r.splitlines()))
ck("hooks are handed to the render only on 16 GB-class cards (else None)",
   "before_decode=_park_for_decode if _lowmem else None" in s
   and "after_decode=_restore_after_decode if _lowmem else None" in s)
ck("_lowmem comes from the shared gate", "_lowmem = _small_card_previews()" in s)
ck("[preview-vram] waypoints: start, before decode, parked, restored, after cleanup",
   all(t in s for t in ('"preview start"', '"before decode"', '"DiT parked for the decode"',
                        '"after decode, DiT restored"', '"after preview cleanup"')))
ck("_render_prompt_set threads the hooks through to sampling.sample",
   "before_decode=before_decode, after_decode=after_decode" in inspect.getsource(trainer._render_prompt_set))

# --- the preview canvas cap and the first-step-after waypoint ------------------------------------
t = inspect.getsource(trainer.train_krea2)
cap = t.split("sample_ae = load_vae(")[1][:2500]
ck("canvas cap lives right after the preview VAE is created, gated on the small-card check",
   "_small_card_previews()" in cap and "_long > 768" in cap and "capped to" in cap)
ck("tiled VAE decode is NOT enabled (seam risk; the park alone removes the spike)",
   "enable_tiling" not in t)
ck("the first training step after a preview logs its VRAM",
   "_vram_after_preview = True" in t and "first training step after the preview" in t)

# the rounding math the cap uses, replicated: 1024x1024 → 768x768, 1024x1536 → 512x768, 832x1216 → 528x768
for (w, h), want in (((1024, 1024), (768, 768)), ((1024, 1536), (512, 768)), ((832, 1216), (528, 768))):
    _long = max(w, h); _sc = 768.0 / _long
    got = (max(256, int(round(w * _sc / 16.0)) * 16), max(256, int(round(h * _sc / 16.0)) * 16))
    ck(f"cap math {w}x{h} → {want[0]}x{want[1]}", got == want, got)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
