"""✨Prompt Enhancer — the Krea 2 text conditioning re-derived through a gain profile.

Ported from `ComfyUI-Krea2T-Enhancer` by Capitan01R
(https://github.com/K3nt3L/ComfyUI-Krea2-Enhancer, MIT; the repo is a licensed re-host after the
original node was taken down). The node exists because Krea 2's text conditioning is 12 stacked
encoder taps rather than one vector: the last taps carry the words and the early ones carry almost
nothing, so a prompt can be followed loosely even when it is in the context. The fix is to run the
text-fusion stack a second time on a per-chunk gain-scaled conditioning and add the difference
back, with the difference capped so no token can be rewritten wholesale.

The maths, for strength `s` (0..2, 0 = off):

* 24 chunks of 1280 (12 taps x 2560 = 30720 = 24 x 1280), gains `1 + s*(profile - 1)` with the
  donor's profile `(1,1,1,1,1,1,1,2.5,5.0,1.1,4.0,1)` tiled twice — taps 8, 9 and 11 of each half
  get 2.5x, 5x and 4x at full strength, everything else stays put;
* a global `1 + s*(15 - 1)` on top, which is what makes s=1 a x15 lift of the whole conditioning;
* `ref = txtfusion(x)`, `cand = txtfusion(gain_scaled)`, and out = `ref + (cand - ref)*scale`
  where `scale = clamp(0.75/rms(cand-ref)/rms(ref), max=1)` — per token, so the residual is
  admitted in full when it is small and clipped to 75% of the reference token's own rms when it
  is large. Without the cap a strong s can push a token anywhere, and the renders go to mush.

Fizgig computes this ONCE PER PREVIEW, not once per step, because the Krea 2 text path is
loop-invariant: `txtfusion`/`txtmlp` see only the conditioning, its mask and the attention mode
(`SingleStreamDiT.forward`), and the timestep feeds only the image modulation. The donor derives
it inside every model call, which costs a second text-fusion pass per step for a tensor that never
changes. `scripts/check_text_boost.py` measures the two against each other.

Measured 2026-11-21 on an NVIDIA RTX PRO 6000 Blackwell against a fusion at the shipped config's
shape (`python -m fizgig.scripts.check_text_boost --fusion --device cuda --log <file>`): one
text-fusion pass 0.0182 s, the enhancement 0.0369 s, so +0.0187 s per preview computed once, and
+0.375 s for a 20-step preview were it re-derived every step - that last figure is the same
increment times the step count, not a separate run. Best of 15 with the card synchronised after
every call, worst within 3% of the mean. The split is the point: two passes would be 0.0364 s, so
the gain scaling and the capped residual cost about half a millisecond between them - the feature
is one extra text-fusion pass and nothing else. The parity claim is not timing-dependent and is
exact: 20 recomputations bit-identical to computing once, and the residual never past 0.75 of the
reference token's rms across seven input scales.
"""
from __future__ import annotations

import torch

#: The donor's per-tap emphasis, in encoder-tap order. Taps 0-6 are the early layers (little
#: signal), tap 8 is the strongest, tap 10 lands between them.
PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)

#: Tiled twice: the conditioning is flattened to 24 chunks of 1280, so each tap appears in two
#: chunks and both halves get the same shape of emphasis.
CHUNK_PROFILE = PROFILE_12 + PROFILE_12

CHUNKS, CHUNK_DIM = 24, 1280            # 24 x 1280 == 12 x 2560
TAPS, TAP_DIM = 12, 2560                 # the Krea 2 text conditioning's shape
MAX_STRENGTH = 2.0                       # the donor's slider top; above it, clamp
GLOBAL_MULTIPLIER = 15.0                 # the lift at s=1, applied to every chunk
TOKEN_REL_CAP = 0.75                     # the residual may not exceed this fraction of the token
SCALE_MIN, SCALE_MAX = 0.25, 4.0         # the Advanced node's text_scale range (neutral 1)


def clamped(strength) -> float:
    """Strength in donor terms: 0 (or anything unusable) means off, 2 is the top.

    A negative or NaN value is off rather than an error — the box is a user typing, and the
    preview should fall back to the plain render rather than to an exception mid-run."""
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return 0.0
    if not s > 0.0 or s != s:       # 0, negative, -0, NaN
        return 0.0
    return min(s, MAX_STRENGTH)


def text_scale(value) -> float:
    """The Advanced node's `text_scale`: a flat multiplier on the txtmlp output, 0.25..4, neutral 1.
    Missing, blank or garbage is the neutral 1 — this one is independent of the strength."""
    try:
        s = float(value)
    except (TypeError, ValueError):
        return 1.0
    if s != s:
        return 1.0
    return min(max(s, SCALE_MIN), SCALE_MAX)


def applies(context) -> bool:
    """True only for the Krea 2 text conditioning: 12 encoder taps of 2560. Anything else is a
    no-op, which is also the guard against a future family borrowing this file by accident."""
    return context.dim() == 4 and tuple(context.shape[-2:]) == (TAPS, TAP_DIM)


def gains(strength, device, dtype) -> torch.Tensor:
    """The 24 per-chunk multipliers, built in float32 exactly as the donor builds them."""
    base = torch.tensor(CHUNK_PROFILE, device=device, dtype=torch.float32)
    return (1.0 + clamped(strength) * (base - 1.0)).to(dtype)


def scaled_context(context, strength) -> torch.Tensor:
    """The conditioning the second text-fusion pass runs on: every chunk times its gain times the
    global lift. Returns `context` itself when off or when the shape is not Krea 2's."""
    s = clamped(strength)
    if s <= 0.0 or not applies(context):
        return context
    b, seq = context.shape[0], context.shape[1]
    g = gains(s, context.device, context.dtype).view(1, 1, CHUNKS, 1)
    return (context.reshape(b, seq, CHUNKS, CHUNK_DIM) * g
            * (1.0 + s * (GLOBAL_MULTIPLIER - 1.0))).reshape(b, seq, TAPS, TAP_DIM)


def capped_residual(ref, cand) -> torch.Tensor:
    """`ref + (cand - ref)*scale`, with the cap maths in float32 and the result back in `cand`'s
    dtype. Per token: a residual already under the cap passes through, a bigger one is clipped."""
    ref32, cand32 = ref.float(), cand.float()
    delta = cand32 - ref32
    base = ref32.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    rel = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8) / base
    return (ref32 + delta * (TOKEN_REL_CAP / rel).clamp(max=1.0)).to(cand.dtype)


def enhance(run, context, strength) -> torch.Tensor:
    """Run the text fusion twice and return the bounded enhancement.

    `run(context)` must be the DiT's `txtfusion` with the same attention params it uses for the
    plain path — the two calls have to be numerical twins or the "residual" is meaningless.
    `SingleStreamDiT.enhanced_text` is what builds that closure; nothing outside Krea 2 should."""
    s = clamped(strength)
    ref = run(context)
    if s <= 0.0 or not applies(context):
        return ref
    return capped_residual(ref, run(scaled_context(context, s)))
