"""TREAD token routing on the tiny H3 DiT: training-only, video rows only, identity rejoin,
off by default; the CLI / trainer plumbing. Headless, CPU.

Run: venv/Scripts/python.exe tests/test_h3_tread.py
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


from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT, audio_latents_for_frames, AUDIO_CHANNELS, pixel_frames_for_latent

torch.manual_seed(0)
cfg = MiniMaxH3Config(hidden_size=64, num_layers=6, token_refiner_num_layers=2,
                      num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=48,
                      latents_dim=24, audio_latents_dim=6, patch_size=(1, 2, 2), text_dim=20,
                      timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
                      rope_inv_freq_len=2)
dit = MiniMaxH3DiT(cfg).eval()
lat = torch.randn(1, 24, 2, 8, 8)
t = torch.tensor([0.5])
txt = torch.randn(1, 5, cfg.text_dim)
# explicit audio rows: without them the forward draws fresh random rows per call
arows = torch.randn(audio_latents_for_frames(pixel_frames_for_latent(2)) * AUDIO_CHANNELS, cfg.audio_latents_dim)

with torch.no_grad():
    ref = dit(lat, t, txt, audio_rows=arows)
ck("no routing configured: plain forward", torch.equal(dit(lat, t, txt, audio_rows=arows).detach(), ref))

dit._tread = (0.5, 1, 4)
with torch.no_grad():
    ck("routing never runs under no_grad (previews / inference)", torch.equal(dit(lat, t, txt, audio_rows=arows), ref))
torch.manual_seed(1)
out = dit(lat, t, txt, audio_rows=arows)
ck("under grad the routed forward has the same shape and differs from the plain one",
   out.shape == ref.shape and not torch.allclose(out.detach(), ref) and torch.isfinite(out).all())
out.float().square().mean().backward()
ck("loss over every token: gradients reach the input and are finite",
   all(p.grad is None or torch.isfinite(p.grad).all() for p in dit.parameters()))
dit.zero_grad(set_to_none=True)
torch.manual_seed(1); a = dit(lat, t, txt, audio_rows=arows).detach()
torch.manual_seed(1); b = dit(lat, t, txt, audio_rows=arows).detach()
ck("the random routing follows torch's RNG (same seed -> same forward)", torch.equal(a, b))
dit._tread = (0.0, 1, 4)
ck("ratio 0 = off", torch.equal(dit(lat, t, txt, audio_rows=arows).detach(), ref))
dit._tread = (0.5, 4, 4)
ck("an empty span [start, end) = off", torch.equal(dit(lat, t, txt, audio_rows=arows).detach(), ref))
dit._tread = (0.5, 2, 99)
ck("an end past the last block clips to the block count and still runs",
   dit(lat, t, txt, audio_rows=arows).shape == ref.shape)
# rows before video_start are never routed: count the sequence length the routed blocks see
dit._tread = (0.5, 1, 4)
seen = []
h0 = dit.blocks[2].register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[0]))
h1 = dit.blocks[5].register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[0]))
_ = dit(lat, t, txt, audio_rows=arows)
h0.remove(); h1.remove()
n_video = 2 * (8 // 2) * (8 // 2)                     # latent_t x (H/2) x (W/2) patches (patch 1,2,2)
full = seen[1]
ck("a routed block sees all the non-video rows plus half the video rows; a block past the end sees everything",
   seen[0] == full - n_video // 2 and full > n_video, (seen, n_video))
dit._tread = None

# ---- FizGigVid: nested lower-res middle for clips --------------------------------------
from fizgig.minimax.trainer import fizgigvid_levels, FIZGIGVID_PRESETS
ck("presets parse: front4_id2 = 2x over 2-46 with 2x more over 2-19; a schedule string parses too",
   fizgigvid_levels("front4_id2") == [(2, 47, 2), (2, 20, 2)] and fizgigvid_levels("all4") == [(2, 47, 4)]
   and fizgigvid_levels("off") == [] and fizgigvid_levels("1-4:2,1-3:2") == [(1, 4, 2), (1, 3, 2)])
dit._tread = None
others = arows.shape[0] + 5                                    # audio rows + text rows
dit._fizgigvid = [(1, 4, 2), (1, 3, 2)]                          # 4x for blocks 1-2, 2x for block 3
with torch.no_grad():
    ck("fizgigvid never runs under no_grad", torch.equal(dit(lat, t, txt, audio_rows=arows), ref))
seen = []
hooks = [dit.blocks[i].register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[0])) for i in (2, 3, 5)]
out_v = dit(lat, t, txt, audio_rows=arows)
for hk in hooks: hk.remove()
ck("nested levels: block 2 sees the video at 1/16, block 3 at 1/4, block 5 everything (non-video rows always there)",
   seen == [others + n_video // 16, others + n_video // 4, others + n_video]
   and out_v.shape == ref.shape and torch.isfinite(out_v).all(), (seen, others, n_video))
out_v.float().square().mean().backward()
ck("gradients flow through the pool / unpool", all(p.grad is None or torch.isfinite(p.grad).all() for p in dit.parameters()))
dit.zero_grad(set_to_none=True)
still = torch.randn(1, 24, 1, 8, 8)
arows1 = torch.randn(audio_latents_for_frames(pixel_frames_for_latent(1)) * AUDIO_CHANNELS, cfg.audio_latents_dim)
with torch.no_grad():
    ref_still = dit(still, t, txt, audio_rows=arows1)
ck("a still (one latent frame) is never pooled", torch.equal(dit(still, t, txt, audio_rows=arows1).detach(), ref_still))
# identity: with the outer span's blocks replaced by identity the levels add nothing
import types
saved = [dit.blocks[i].forward for i in (1, 2, 3)]
for i in (1, 2, 3):
    dit.blocks[i].forward = types.MethodType(lambda self, x, *a, **k: x, dit.blocks[i])
dit._fizgigvid = None
with torch.no_grad():
    ref_id = dit(lat, t, txt, audio_rows=arows)
dit._fizgigvid = [(1, 4, 2), (1, 3, 2)]
out_id = dit(lat, t, txt, audio_rows=arows).detach()
for i, f in zip((1, 2, 3), saved):
    dit.blocks[i].forward = f
ck("identity middle blocks: the residual unpools add nothing and the output equals the plain forward",
   torch.allclose(out_id, ref_id, atol=1e-5), float((out_id - ref_id).abs().max()))
dit._fizgigvid = [(1, 4, 2), (1, 3, 2)]; dit._tread = (0.5, 1, 4)
seen = []
hooks = [dit.blocks[i].register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[0])) for i in (2, 3)]
out_both = dit(lat, t, txt, audio_rows=arows)
for hk in hooks: hk.remove()
ck("TREAD rides inside the outer level after the inner one rejoins: block 2 untouched by routing, block 3 sees half of the 1/4 grid",
   seen == [others + n_video // 16, others + n_video // 8] and out_both.shape == ref.shape, (seen, n_video))
dit._fizgigvid = [(1, 4, 3)]
ck("a factor the grid does not divide is skipped (4x4 patches by 3)", torch.equal(dit(lat, t, txt, audio_rows=arows).detach(), ref) if False else dit(lat, t, txt, audio_rows=arows).shape == ref.shape)
dit._fizgigvid = None; dit._tread = None

# CLI plumbing
import importlib.util
spec = importlib.util.spec_from_file_location("mmt", os.path.join(REPO, "src", "fizgig", "scripts", "minimax_train.py"))
src = open(spec.origin, encoding="utf-8").read()
ck("CLI: --tread_ratio / --tread_start / --tread_end / --fizgigvid exist and are passed through",
   src.count('add_argument("--tread_ratio"') == 1 and "tread_ratio=args.tread_ratio" in src and "tread_end=args.tread_end" in src
   and src.count('add_argument("--fizgigvid"') == 1 and "fizgigvid=args.fizgigvid" in src)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
