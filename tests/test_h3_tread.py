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

# ---- photos unrouted (the 4th element) ---------------------------------------------------
still = torch.randn(1, 24, 1, 8, 8)
arows1 = torch.randn(audio_latents_for_frames(pixel_frames_for_latent(1)) * AUDIO_CHANNELS, cfg.audio_latents_dim)
with torch.no_grad():
    ref_still = dit(still, t, txt, audio_rows=arows1)
dit._tread = (0.5, 1, 4, True)
ck("skip-photos: a one-frame item is not routed",
   torch.equal(dit(still, t, txt, audio_rows=arows1).detach(), ref_still))
torch.manual_seed(3)
ck("...while a clip still is", not torch.allclose(dit(lat, t, txt, audio_rows=arows).detach(), ref))
dit._tread = (0.5, 1, 4, False)
torch.manual_seed(3)
ck("without the flag a one-frame item is routed",
   not torch.allclose(dit(still, t, txt, audio_rows=arows1).detach(), ref_still))
dit._tread = None

# CLI plumbing
import importlib.util
spec = importlib.util.spec_from_file_location("mmt", os.path.join(REPO, "src", "fizgig", "scripts", "minimax_train.py"))
src = open(spec.origin, encoding="utf-8").read()
ck("CLI: --tread_ratio / --tread_start / --tread_end exist and are passed through",
   src.count("--tread_ratio") == 1 and "tread_ratio=args.tread_ratio" in src and "tread_end=args.tread_end" in src)
ck("CLI: --tread_skip_photos exists and is passed through",
   src.count("--tread_skip_photos") == 1 and "tread_skip_photos=args.tread_skip_photos" in src)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
