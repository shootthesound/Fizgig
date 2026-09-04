"""MiniMax H3 clip sampling — grid math, position ids, tiny-model forward, decode chunk plan.

The trained range is ~124-362 frames (ComfyUI's own tooltip); a lone still is out of
distribution, which is why clip previews exist. These pin the 17n+5 <-> 5n+2 grid, the
(1,4,4,4,4)x5/3 temporal RoPE, pixel-vs-latent audio sizing, snapping, and the ported
temporal-chunked decode's frame accounting — everything that can be checked without a GPU.

Run: venv/Scripts/python.exe tests/test_minimax_clip.py
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.minimax.model import (MiniMaxH3Config, MiniMaxH3DiT, FRAME_RESCALE,  # noqa: E402
                                  _video_t_spans, audio_latents_for_frames,
                                  image_position_ids, latent_frames_for_pixels,
                                  pixel_frames_for_latent)
from fizgig.minimax import sampling  # noqa: E402
from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# --- 1. the grid ----------------------------------------------------------------------------
for px, lat in ((1, 1), (5, 2), (22, 7), (56, 17), (107, 32), (124, 37), (141, 42)):
    ck(f"grid: {px} px <-> {lat} latents",
       latent_frames_for_pixels(px) == lat and pixel_frames_for_latent(lat) == px)
ck("off-grid snaps DOWN (30 -> 22, 123 -> 107)",
   latent_frames_for_pixels(30) == 7 and latent_frames_for_pixels(123) == 32)
ck("audio latents are PIXEL-frame based (124f -> 207)", audio_latents_for_frames(124) == 207)
ck("partial temporal group is exact arithmetic (3 -> 9, 4 -> 13, 6 -> 18)",
   [pixel_frames_for_latent(t) for t in (3, 4, 6)] == [9, 13, 18])
ck("exact=True keeps a token-grid count (5 -> 2, 9 -> 3, 13 -> 4, 22 -> 7)",
   [latent_frames_for_pixels(f, exact=True) for f in (1, 5, 9, 13, 22)] == [1, 2, 3, 4, 7])
try:
    latent_frames_for_pixels(10, exact=True)
    ck("exact=True refuses a count off the token grid (10)", False)
except ValueError:
    ck("exact=True refuses a count off the token grid (10)", True)

# --- 2. position ids ------------------------------------------------------------------------
fr = (8 // 2) * (8 // 2)
p1 = image_position_ids(4, 8, 8, 2)
p7 = image_position_ids(4, 8, 8, 2, latent_t=7)
ck("T=1 layout unchanged (clip support must not move a still's rows)",
   torch.equal(p1, image_position_ids(4, 8, 8, 2, latent_t=1)))
ck("T=7 adds exactly 6 more frame-row blocks", p7.shape[0] == p1.shape[0] + 6 * fr)
vid = p7[-7 * fr:]
tv = vid[:, 0].reshape(7, fr)
ck("each frame's rows share one t", bool((tv == tv[:, :1]).all()))
got = [float(x) for x in tv[:, 0]]
exp = [4.0]
for s_ in _video_t_spans(7)[:-1]:
    exp.append(exp[-1] + s_)
ck("t follows the (1,4,4,4,4) x 5/3 span grid", all(abs(a - b) < 1e-9 for a, b in zip(got, exp)),
   [round(g, 3) for g in got])
ck("one 5-latent group spans 17 x 5/3 rotary units",
   abs((got[5] - got[0]) - 17 * FRAME_RESCALE) < 1e-9)

# --- 3. tiny-model forward + sampler --------------------------------------------------------
CFG = dict(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=4,
           attention_head_dim=16, ffn_hidden_size=48, latents_dim=24, audio_latents_dim=6,
           patch_size=(1, 2, 2), text_dim=32, timestep_input_dim=16, time_embed_hidden_size=64,
           time_embed_dim=32, rope_inv_freq_len=2)
torch.manual_seed(0)
dit = MiniMaxH3DiT(MiniMaxH3Config(**CFG)).eval()
txt = torch.randn(1, 5, 32)
with torch.no_grad():
    out, aud = dit(torch.randn(1, 24, 7, 8, 8), torch.tensor([0.5]), txt, return_audio=True)
ck("tiny forward at T=7 returns matching video shape", out.shape == (1, 24, 7, 8, 8))
ck("...and pixel-frame-sized audio rows (22f -> 74)", aud.shape[0] == round(22 / 24 * 40) * 2)
with torch.no_grad():
    x22 = sampling.sample_image(dit, txt, width=128, height=128, steps=3, seed=1,
                                device="cpu", dtype=torch.float32, num_frames=22)
    x30 = sampling.sample_image(dit, txt, width=128, height=128, steps=3, seed=1,
                                device="cpu", dtype=torch.float32, num_frames=30)
    x1 = sampling.sample_image(dit, txt, width=128, height=128, steps=3, seed=1,
                               device="cpu", dtype=torch.float32)
ck("sample_image 22 frames -> T=7 latent", x22.shape == (1, 24, 7, 8, 8))
ck("sample_image snaps 30 -> 22", x30.shape[2] == 7)
ck("sample_image default is still T=1", x1.shape[2] == 1)

# --- 4. temporal-chunked decode: frame accounting with a token-tracing fake decoder ---------
dec = MiniMaxH3VideoVAEDecoder.__new__(MiniMaxH3VideoVAEDecoder)
torch.nn.Module.__init__(dec)
for n, v in (("latents_mean", torch.zeros(24)), ("latents_std", torch.ones(24)),
             ("pixel_mean", torch.zeros(1, 3, 1, 1, 1)), ("pixel_std", torch.ones(1, 3, 1, 1, 1))):
    dec.register_buffer(n, v)
dec.post_quant_conv = torch.nn.Conv3d(24, 24, 1, bias=False)
with torch.no_grad():
    dec.post_quant_conv.weight.zero_()
    for c in range(24):
        dec.post_quant_conv.weight[c, c, 0, 0, 0] = 1.0


class _FakeViT(torch.nn.Module):
    def forward(self, zz):
        B, C, T, H, W = zz.shape
        vals = zz[:, 0, :, 0, 0]
        return (vals.repeat_interleave(4, dim=1)
                .view(B, 1, 4 * T, 1, 1).expand(B, 3, 4 * T, 16 * H, 16 * W).clone())


dec.decoder = _FakeViT()
for T in (2, 3, 4, 7, 12, 37, 42):          # 3, 4 = partial temporal group (Repair Studio short clips)
    z = (torch.arange(T, dtype=torch.float32) / max(T - 1, 1)).view(1, 1, T, 1, 1)
    z = z.expand(1, 24, T, 4, 4).clone()
    px = dec.decode_clip(z)
    want = pixel_frames_for_latent(T)
    seq = px[0, 0, :, 0, 0] * max(T - 1, 1)
    ok = (px.shape[2] == want and bool((seq[1:] >= seq[:-1] - 1e-4).all())
          and abs(float(seq[-1]) - (T - 1)) < 1e-3)
    ck(f"decode_clip T={T}: {want} frames, monotone token sources, ends on the last token", ok,
       f"got {px.shape[2]} frames, last src {float(seq[-1]):.2f}")
# the (1,4,4,4,4) structure itself: token k's frame count within one group
z = (torch.arange(7, dtype=torch.float32)).view(1, 1, 7, 1, 1).expand(1, 24, 7, 4, 4).clone()
dec2 = dec
px = dec2.decode_clip(z / 6)
counts = torch.bincount((px[0, 0, :, 0, 0] * 6).round().long(), minlength=7).tolist()
ck("frame counts per token follow (1,4,4,4,4 | 1,4,...)", counts == [1, 4, 4, 4, 4, 1, 4], counts)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
