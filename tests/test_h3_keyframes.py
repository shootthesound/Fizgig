"""MiniMax H3 keyframe (fl2va first/last-frame) conditioning — layout + identity pins.

On the tiny CPU config:
  1. keyframes=None is BIT-IDENTICAL to the pre-change forward (same seed, same inputs);
  2. with keyframes the sequence grows by frame_rows per keyframe, the video output keeps its
     shape, and the cond rows sit right after the text (before refs) with VIDEO tags;
  3. their positions are origin + FRAME_RESCALE * index on the target clock (last frame = the
     clip's last pixel frame), and the cond timestep label is max(t, 0.999);
  4. sample_image threads them through and clamps an over-long index to the snapped clip;
     forward_cached refuses them out loud.

Run: venv/Scripts/python.exe tests/test_h3_keyframes.py   (CPU)
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.minimax import model as M                                       # noqa: E402
from fizgig.minimax.model import (MiniMaxH3Config, MiniMaxH3DiT, image_position_ids,  # noqa: E402
                                  FRAME_RESCALE, _frame_grid, _video_t_spans,
                                  pixel_frames_for_latent)

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


torch.manual_seed(0)
tiny = MiniMaxH3Config(hidden_size=64, num_layers=2, token_refiner_num_layers=1,
                       num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=64,
                       latents_dim=24, audio_latents_dim=6, patch_size=(1, 2, 2), text_dim=24,
                       time_embed_dim=8, rope_inv_freq_len=2)
dit = MiniMaxH3DiT(tiny).eval()
with torch.no_grad():
    for p in dit.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
T, H, W = 7, 8, 8                      # 22-frame clip at a 128x128 canvas (latent 8x8)
text_len = 5
lat = torch.randn(1, 24, T, H, W)
txt = torch.randn(1, text_len, 24)
t = torch.tensor([0.4])
frame_rows = _frame_grid(H, W).shape[0]
frames = pixel_frames_for_latent(T)
kf_first = torch.randn(1, 24, 1, H, W)
kf_last = torch.randn(1, 24, 1, H, W)

# --- 1. identity without keyframes --------------------------------------------------------
with torch.no_grad():
    torch.manual_seed(1); y0 = dit(lat, t, txt)
    torch.manual_seed(1); y1 = dit(lat, t, txt, keyframes=None)
ck("keyframes=None is bit-identical to the plain forward", torch.equal(y0, y1))

# --- 2/3. positions -----------------------------------------------------------------------
pos_plain = image_position_ids(text_len, H, W, 0, latent_t=T)
pos_kf = image_position_ids(text_len, H, W, 0, latent_t=T, keyframes=[0, frames - 1])
ck("two keyframes add 2 x frame_rows position rows",
   pos_kf.shape[0] == pos_plain.shape[0] + 2 * frame_rows)
ck("text rows unchanged", torch.equal(pos_kf[:text_len], pos_plain[:text_len]))
a = text_len; b = a + frame_rows; c = b + frame_rows
ck("first-frame rows sit right after text at t = origin",
   torch.allclose(pos_kf[a:b, 0], torch.full((frame_rows,), float(text_len), dtype=torch.float64)))
last_t = float(text_len) + FRAME_RESCALE * (frames - 1)
ck("last-frame rows at origin + 5/3 * (frames-1)",
   torch.allclose(pos_kf[b:c, 0], torch.full((frame_rows,), last_t, dtype=torch.float64)))
ck("...which equals the span the video rows cover (lands on the last pixel frame)",
   abs(sum(_video_t_spans(T)) - FRAME_RESCALE * frames) < 1e-9)
ck("keyframe rows carry the target's own h/w grid",
   torch.equal(pos_kf[a:b, 1:], _frame_grid(H, W)) and torch.equal(pos_kf[b:c, 1:], _frame_grid(H, W)))
ck("the video rows after them are the plain layout shifted by the inserted rows",
   torch.equal(pos_kf[c:], pos_plain[text_len:]))
# with a ref present the keyframe clock is the POST-ref cursor
pos_ref = image_position_ids(text_len, H, W, 0, refs=[(H, W)], latent_t=T, keyframes=[0])
ck("with a reference the keyframe sits on the post-ref origin (text_len + 1)",
   float(pos_ref[text_len, 0]) == float(text_len) + 1.0
   and float(pos_ref[text_len + frame_rows, 0]) == float(text_len))    # the ref row itself

# --- forward with keyframes ----------------------------------------------------------------
# (the audio rows draw fresh silence noise from the global RNG every forward — reseed so the
# only thing that differs between runs is what the pin says differs)
with torch.no_grad():
    torch.manual_seed(1); y_kf = dit(lat, t, txt, keyframes=[(0, kf_first), (frames - 1, kf_last)])
ck("video output shape unchanged with keyframes", tuple(y_kf.shape) == tuple(lat.shape))
ck("keyframes change the prediction", not torch.allclose(y_kf, y0))
with torch.no_grad():
    torch.manual_seed(1); y_kf2 = dit(lat, t, txt, keyframes=[(0, kf_first), (frames - 1, kf_last)])
ck("keyframe noise is role-seeded, so the forward is deterministic", torch.equal(y_kf, y_kf2))
# (no "swap first/last changes the output" pin: with rope_inv_freq_len=2 the tiny model's
# positional signal is ~nil, so swapping two cond rows is a near-permutation of an attention
# set — invisible here; the positions themselves are pinned above)
try:
    with torch.no_grad():
        dit(lat, t, txt, keyframes=[(0, torch.randn(1, 24, 1, H // 2, W // 2))])
    ck("a keyframe at the wrong size raises", False)
except ValueError as e:
    ck("a keyframe at the wrong size raises", "target grid" in str(e))
try:
    with torch.no_grad():
        dit(lat, t, txt, keyframes=[(frames, kf_first)])
    ck("an out-of-clip index raises", False)
except ValueError as e:
    ck("an out-of-clip index raises", "outside" in str(e))

# --- cond timestep label: cond rows share max(t, 0.999) -----------------------------------
seen = {}
_orig = dit._time_embedding
def _spy(uniq):
    seen["uniq"] = uniq.detach().clone()
    return _orig(uniq)
dit._time_embedding = _spy
with torch.no_grad():
    dit(lat, t, txt, keyframes=[(0, kf_first)])
dit._time_embedding = _orig
ck("a cond timestep row max(t, 0.999) exists for the keyframe rows",
   bool((seen["uniq"] - 0.999).abs().min() < 1e-6))

# --- sampler threading + forward_cached refusal -------------------------------------------
from fizgig.minimax import sampling
calls = []
_real_fwd = dit.forward
def _fwd(*a, **k):
    calls.append(sorted(k.get("keyframes", []) and [i for i, _ in k["keyframes"]]))
    return _real_fwd(*a, **k)
dit.forward = _fwd
with torch.no_grad():
    out = sampling.sample_image(dit, txt, width=W * 16, height=H * 16, steps=3, seed=1,
                                device="cpu", dtype=torch.float32, num_frames=22,
                                keyframes=[(0, kf_first), (99, kf_last)])
dit.forward = _real_fwd
ck("sample_image passes keyframes on every evaluation",
   len(calls) == 3 and all(c == [0, frames - 1] for c in calls), calls)
ck("...clamping an over-long last-frame index to the snapped clip's last frame",
   calls and calls[0][-1] == frames - 1)
ck("latent shape from a keyframed sample", tuple(out.shape) == (1, 24, T, H, W))
try:
    dit.forward_cached(lat, t, txt, keyframes=[(0, kf_first)])
    ck("forward_cached refuses keyframes", False)
except ValueError as e:
    ck("forward_cached refuses keyframes", "keyframe" in str(e))

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
