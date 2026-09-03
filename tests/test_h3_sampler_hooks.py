"""MiniMax H3 sampler hooks for the Repair Studio fast loop — exactness pins on the tiny model.

  1. BlockCacheContext(record_steps={0}) records ONLY step 0 (the other passes are never
     parked), and a render with an empty cache equals the plain render bit for bit;
  2. exact pass-1 resume: after a change to block k, a render that resumes step 0 from the
     previous render's cache at k equals a plain full render of the changed model — allclose
     at fp32 roundoff — and still equals it when the change is to the LAST block;
  3. a resume with entries only for step 0 leaves steps 1..n running in full (their inputs
     already carry the change) — pinned by the same equality;
  4. on_denoised fires once per evaluation with (step, n_eval, x0 estimate of the clip's
     latent shape), and a callback that raises never takes the render down.

Run: venv/Scripts/python.exe tests/test_h3_sampler_hooks.py   (CPU)
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.minimax import sampling                                          # noqa: E402
from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT              # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


torch.manual_seed(0)
tiny = MiniMaxH3Config(hidden_size=64, num_layers=4, token_refiner_num_layers=1,
                       num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=64,
                       latents_dim=24, audio_latents_dim=6, patch_size=(1, 2, 2), text_dim=24,
                       time_embed_dim=8, rope_inv_freq_len=2)
dit = MiniMaxH3DiT(tiny).eval()
with torch.no_grad():
    for p in dit.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
txt = torch.randn(1, 5, 24)
W, H, FRAMES, STEPS = 128, 128, 22, 4
KW = dict(width=W, height=H, steps=STEPS, seed=3, device="cpu", dtype=torch.float32,
          num_frames=FRAMES, return_audio=True)


def render(block_cache=None, on_denoised=None):
    torch.manual_seed(11)     # the audio silence rows draw from the global RNG
    with torch.no_grad():
        return sampling.sample_image(dit, txt, block_cache=block_cache,
                                     on_denoised=on_denoised, **KW)


# --- 1. record_steps={0} ------------------------------------------------------------------
plain, plain_a = render()
ctx = sampling.BlockCacheContext(entries={}, resume_from=None, cache_device="cpu",
                                 record_steps={0})
cached, cached_a = render(ctx)
ck("empty cache + record_steps={0}: render is bit-identical to plain",
   torch.equal(plain, cached) and torch.equal(plain_a, cached_a))
ck("only step 0 recorded", sorted(ctx.new_entries) == [0], sorted(ctx.new_entries))
e0 = ctx.new_entries[0]
ck("step-0 entry holds one input per block",
   len(e0.block_inputs) == 4 and all(b is not None for b in e0.block_inputs))
ctx_all = sampling.BlockCacheContext(entries={}, resume_from=None, cache_device="cpu")
render(ctx_all)
ck("record_steps=None still records every step", sorted(ctx_all.new_entries) == [0, 1, 2, 3])


# --- 2/3. exact pass-1 resume after a block change ----------------------------------------
def perturb(k, scale=1.5):
    with torch.no_grad():
        for p in dit.blocks[k].parameters():
            p.mul_(scale)


for k in (2, 3):     # a middle block and the LAST block
    perturb(k)
    full, full_a = render()
    ctx_r = sampling.BlockCacheContext(entries={0: e0}, resume_from=k, cache_device="cpu",
                                       record_steps={0})
    res, res_a = render(ctx_r)
    ck(f"resume at block {k} == full render of the changed model (video)",
       torch.allclose(res, full, atol=1e-5, rtol=1e-5),
       f"max|diff|={float((res - full).abs().max()):.2e}")
    ck(f"resume at block {k} == full render (audio rows)",
       torch.allclose(res_a, full_a, atol=1e-5, rtol=1e-5))
    ck(f"resume at block {k}: the change is real (differs from the pre-change render)",
       not torch.allclose(res, plain, atol=1e-4))
    e0 = ctx_r.new_entries[0]                    # roll the cache forward like the engine
    perturb(k, 1.0 / 1.5)
    # (undo so the next k starts from the same model; e0 now describes the perturbed model —
    # re-record a fresh step-0 entry for the restored model before the next round)
    ctx0 = sampling.BlockCacheContext(entries={}, resume_from=None, cache_device="cpu",
                                      record_steps={0})
    render(ctx0)
    e0 = ctx0.new_entries[0]

# resume_from beyond the cached prefix falls back to a full pass (no crash, still exact)
perturb(1)
full1, _ = render()
ctx_b = sampling.BlockCacheContext(entries={0: e0}, resume_from=1, cache_device="cpu",
                                   record_steps={0})
res1, _ = render(ctx_b)
ck("resume at block 1 (early block) == full render", torch.allclose(res1, full1, atol=1e-5))
perturb(1, 1.0 / 1.5)

# --- 4. on_denoised -----------------------------------------------------------------------
seen = []
def _cb(step, n, x0):
    seen.append((step, n, tuple(x0.shape), x0.dtype))
render(None, _cb)
ck("on_denoised fires once per evaluation", [s[0] for s in seen] == [1, 2, 3, 4], seen)
ck("...with n_eval and the clip latent shape in fp32",
   all(s[1] == 4 and s[2] == tuple(plain.shape) and s[3] == torch.float32 for s in seen))
def _boom(step, n, x0):
    raise RuntimeError("early-look decode failed")
plain_now, _ = render()          # (the perturb/undo above isn't a bitwise identity)
out_b, _ = render(None, _boom)
ck("a raising callback never takes the render down", torch.equal(out_b, plain_now))

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
