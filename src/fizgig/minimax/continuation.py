"""MiniMax H3: a clip longer than one render, built in pieces (Studio's In pieces tick).

Each piece is an ordinary clip of at most 124 frames, so memory stays at one clip's worth however long the result.
A piece after the first is shown the 39 frames (12 latents) before it:

* all 12 ride as clean guide rows (the model's own first / last frame mechanism, one row per latent), and
* the first 7 are also put back into the noisy clip at every step at that step's noise level (a pin), so they come
  out as they went in. The last 5 are made again, and
* those re-made 17 frames fade in over the same frames of the piece before (an S-curve, per pixel).

Then, on the decoded frames: a 5-frame clip made between a real frame either side of each join (first / last frame
guides, a neutral prompt) is laid over the join, and the lighting around each join is evened out area by area (each
area against the same area in the neighbouring frames; nothing is averaged over the whole frame). No latent is
corrected and nothing is re-encoded except the two bridge frames per join.

A prompt can change through the clip: a line that starts with a time ("8s: ...") is the prompt from that time on;
a change arrives at the nearest start of a piece (with 5.2 s pieces: 5.2 s, then every 3.5 s).
"""
from __future__ import annotations

import math
import re

import torch

PIECE = 124                 # frames in a full piece, unless the caller names another length
CTX = 12                    # latents of the piece before that the next one is shown (39 frames)
PIN = 7                     # of those, the latents kept as they are (22 frames); the other 5 (17 frames) are re-made
BRIDGE = 5                  # frames of the clip laid over each join
BRIDGE_PROMPT = "a video"
BRIDGE_STEPS = 4
GUIDE_FRAMES = (1, 4, 4, 4, 4)      # pixel frames each latent covers, repeating


def _frames_of(latents: int) -> int:
    from fizgig.minimax.model import pixel_frames_for_latent
    return pixel_frames_for_latent(latents)


def piece_frames(piece=None) -> int:
    """A piece's length in frames: the caller's, moved onto the grid and no shorter than the 39 shown frames plus 17
    new ones; 124 when none is named."""
    n = int(piece or PIECE)
    return max(_frames_of(CTX) + 17, 17 * round((n - 5) / 17) + 5)


def piece_lengths(frames: int, piece=None) -> list:
    """A clip of `frames` (on the model's 17n + 5 grid) as pieces: the first up to `piece` frames (124 unless
    named), each later one its 39 shown frames plus up to piece - 39 new ones (so every piece is itself on the
    grid)."""
    frames, piece = int(frames), piece_frames(piece)
    if frames <= piece:
        return [frames]
    ctx, new = _frames_of(CTX), piece - _frames_of(CTX)
    out, left = [piece], frames - piece
    while left > 0:
        n = min(new, left)
        out.append(n + ctx)
        left -= n
    return out


def joins(lengths) -> list:
    """The first new frame of each piece after the first."""
    ctx, at, out = _frames_of(CTX), 0, []
    for i, n in enumerate(lengths):
        if i:
            out.append(at)
        at += n if not i else n - ctx
    return out


_TIMED = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\s*:\s*(.*)$", re.I)


def timed_prompts(text: str) -> list:
    """[(seconds, prompt)]: the text up to the first timed line is the prompt from 0; a line starting "8s:" opens
    the prompt from 8 seconds (its own line and those after it, up to the next timed line)."""
    parts = [(0.0, [])]
    for line in (text or "").splitlines():
        m = _TIMED.match(line)
        if m:
            parts.append((float(m.group(1)), [m.group(2)]))
        else:
            parts[-1][1].append(line)
    out = [(t, "\n".join(lines).strip()) for t, lines in parts]
    out = sorted((p for p in out if p[1]), key=lambda p: p[0])
    return out or [(0.0, (text or "").strip())]


def piece_prompts(text: str, lengths, fps: float = 24.0) -> list:
    """Each piece's prompt: the one in force at the middle of the piece's new frames, so a timed change lands on the
    nearest start of a piece."""
    timed = timed_prompts(text)
    ctx, starts = _frames_of(CTX), [0] + joins(lengths)
    out = []
    for i, (s, n) in enumerate(zip(starts, lengths)):
        mid = 0.0 if not i else (s + (n - ctx) / 2.0) / fps
        out.append(next((p for t, p in reversed(timed) if t <= mid), timed[0][1]))
    return out


class Continue:
    """The model as a later piece calls it: shown the piece before. It reads the engine's model at each call (a
    render may reload or re-plan it), adds the guide rows, and puts the pinned latents and their sound back at the
    call's noise level."""

    def __init__(self, engine, lat, aud, seed):
        g = torch.Generator("cpu").manual_seed(int(seed) + 991)
        self.__dict__.update(engine=engine, lat=lat, aud=aud, n_v=torch.randn(lat.shape, generator=g),
                             n_a=torch.randn(aud.shape, generator=g))

    def __getattr__(self, k):
        return getattr(self.engine.dit, k)

    def __setattr__(self, k, v):
        setattr(self.engine.dit, k, v)

    def __call__(self, x, t, text, audio_rows=None, return_audio=False, **kw):
        from fizgig.minimax.model import AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT, remap_sigma
        sv = float(1.0 - t.reshape(-1)[0])
        x = x.clone()
        x[:, :, :PIN] = ((1 - sv) * self.lat[:, :, :PIN] + sv * self.n_v[:, :, :PIN]).to(x)
        f, kf = 0, []
        for k in range(CTX):                             # the shown frames as clean guide rows, each on its moment
            kf.append((f, self.lat[:, :, k:k + 1].to(device=x.device, dtype=torch.float32)))
            f += GUIDE_FRAMES[k % 5]
        kw = dict(kw, keyframes=kf + list(kw.get("keyframes") or ()))
        if audio_rows is not None and self.aud is not None:
            sa = float(remap_sigma(torch.tensor(max(sv, 1e-6)), VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT))
            d, n = audio_rows.shape[-1], self.aud.shape[1]
            a = audio_rows.clone().reshape(2, -1, d)
            a[:, :n] = ((1 - sa) * self.aud + sa * self.n_a).to(a)
            audio_rows = a.reshape(-1, d)
        return self.engine.dit(x, t, text, audio_rows=audio_rows, return_audio=return_audio, **kw)


def deflicker(frames, join_frames, sigma_t: float = 4.0, reach: int = 24):
    """Even out the lighting around each join, in place, on a list of float32 [H, W, 3] arrays (0-255).

    A frame's lighting field is its heavily blurred picture per colour channel (a blur of 1/6 of the frame: lighting
    and colour, no detail), worked out on a small copy. Near a join the field is smoothed over time and each pixel
    is scaled by smoothed / own, within 0.85-1.18; the correction fades to nothing `reach` frames from a join."""
    import cv2
    import numpy as np
    n = len(frames)
    if not join_frames or n < 3:
        return frames
    rad = int(3 * sigma_t)
    near = [t for t in range(n) if min(abs(t - j) for j in join_frames) < reach]
    need = sorted({u for t in near for u in range(max(0, t - rad), min(n, t + rad + 1))})
    h, w = frames[0].shape[:2]
    s = 96.0 / max(h, w)
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    low = {u: np.log(cv2.GaussianBlur(cv2.resize(frames[u], (sw, sh), interpolation=cv2.INTER_AREA), (0, 0),
                                      max(sw, sh) / 6.0) + 1.0) for u in need}
    ker = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_t) ** 2)
    out = {}
    for t in near:
        dist = min(abs(t - j) for j in join_frames)
        half = reach // 2
        wt = 1.0 if dist <= half else 0.5 * (1 + math.cos(math.pi * (dist - half) / (reach - half)))
        a, b = max(0, t - rad), min(n, t + rad + 1)
        k = ker[a - t + rad:b - t + rad]
        sm = np.tensordot(k / k.sum(), np.stack([low[u] for u in range(a, b)]), axes=(0, 0))
        gain = np.clip(np.exp(sm - low[t]), 0.85, 1.18)
        gain = cv2.resize(gain.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        out[t] = frames[t] * (1.0 + wt * (gain - 1.0))
    for t, f in out.items():
        frames[t] = f
    return frames


def render_long(eng, state, *, frames, steps, turbo_strength, with_audio=True, no_lora=False, piece=None):
    """The clip in pieces of `piece` frames (124 unless named) -> (latent of the whole clip, audio rows, [PIL
    frames]). `eng` is the H3 Studio engine."""
    import numpy as np
    from PIL import Image
    from fizgig.minimax.model import audio_latents_for_frames
    lengths = piece_lengths(frames, piece)
    prompts = piece_prompts(state.prompt, lengths)
    ctx_f, pin_f = _frames_of(CTX), _frames_of(PIN)
    ctx_a = audio_latents_for_frames(ctx_f)
    user_kf = list(getattr(state, "keyframes", None) or ())
    starts = [0] + [j - ctx_f for j in joins(lengths)]            # each piece's first frame in the whole clip
    # End on the first frame (Studio's tick) with no last-frame picture of the user's: the last piece is guided to
    # the clip's own opening frame - the first piece's first latent, as it is (no decode, no encode)
    end = (bool(getattr(eng, "clip_end_on_first", False)) and len(lengths) > 1
           and not any(g == int(frames) - 1 for g, _z in user_kf))
    boost = getattr(eng, "speed_boost", False)
    eng.speed_boost = False             # a piece is pinned at full size from the first step
    lats, auds, out = [], [], []
    try:
        for i, (n, prompt) in enumerate(zip(lengths, prompts)):
            eng._status(f"Piece {i + 1} of {len(lengths)}…")
            st = state.copy()
            st.prompt, st.preview_frames = prompt, n
            st.references = getattr(state, "references", None)
            # a first / last frame picture belongs to the piece its frame falls in (a later piece: its new frames)
            st.keyframes = [(g - starts[i], z) for g, z in user_kf
                            if (starts[i] + (ctx_f if i else 0)) <= g < starts[i] + n] or None
            if end and i == len(lengths) - 1:
                st.keyframes = list(st.keyframes or ()) + [(n - 1, lats[0][:, :, :1].clone())]
            model = None
            if i:
                kl = lats[-1][:, :, -CTX:].float()
                ka = (auds[-1].reshape(2, -1, auds[-1].shape[-1])[:, -ctx_a:].float()
                      if auds[-1] is not None else None)
                model = Continue(eng, kl, ka, state.seed)
            lat, aud = eng.render_latent(st, frames=n, steps=steps, turbo_strength=turbo_strength, no_lora=no_lora,
                                         model=model, guide_rows=CTX if i else 0)
            if i:
                lat[:, :, :PIN] = kl[:, :, :PIN]
                if aud is not None and ka is not None:
                    a = aud.reshape(2, -1, aud.shape[-1])
                    a[:, :ctx_a] = ka
                    aud = a.reshape(-1, aud.shape[-1])
            lats.append(lat)
            auds.append(aud)
            fr = [np.asarray(im.convert("RGB")).astype(np.float32) for im in eng.decode_clip_frames(lat)]
            if not i:
                out = fr
            else:
                k = ctx_f - pin_f                        # the re-made frames fade in over the piece before
                for j in range(k):
                    w = (j + 1) / (k + 1)
                    w = w * w * (3 - 2 * w)
                    out[len(out) - k + j] = (1 - w) * out[len(out) - k + j] + w * fr[pin_f + j]
                out.extend(fr[ctx_f:])
        js = joins(lengths)
        width, height = int(state.preview_width), int(state.preview_height)
        for k, j in enumerate(js):
            eng._status(f"Join {k + 1} of {len(js)}…")
            a = j - BRIDGE // 2
            pic = lambda arr: Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            st = state.copy()
            st.prompt, st.preview_frames, st.references = BRIDGE_PROMPT, BRIDGE, None
            st.keyframes = [(0, eng.encode_keyframe(pic(out[a]), width, height)),
                            (BRIDGE - 1, eng.encode_keyframe(pic(out[a + BRIDGE - 1]), width, height))]
            bl, _ = eng.render_latent(st, frames=BRIDGE, steps=BRIDGE_STEPS, turbo_strength=turbo_strength,
                                      no_lora=no_lora)
            for q, im in enumerate(eng.decode_clip_frames(bl)[:BRIDGE]):
                w = math.sin(math.pi * q / (BRIDGE - 1)) ** 2
                out[a + q] = (1 - w) * out[a + q] + w * np.asarray(im.convert("RGB")).astype(np.float32)
        deflicker(out, js)
    finally:
        eng.speed_boost = boost
    full = torch.cat([lats[0]] + [z[:, :, CTX:] for z in lats[1:]], dim=2)
    aud = None
    if with_audio and all(a is not None for a in auds):
        d = auds[0].shape[-1]
        aud = torch.cat([auds[0].reshape(2, -1, d)] + [a.reshape(2, -1, d)[:, ctx_a:] for a in auds[1:]],
                        dim=1).reshape(-1, d)
    return full, aud, [Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)) for f in out]
