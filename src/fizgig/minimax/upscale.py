"""Studio's Upscale for an H3 clip: a finished clip is stretched to a larger size and finished again there at a low
denoise, a section at a time, so the larger size never has to hold the whole clip.

Each section is an ordinary first / last frame clip. Its two end frames are upscaled on their own first (stretched,
then finished as single frames) and shown to the model as the section's first and last frame; the frames between
start from the stretched clip at the chosen noise level. The next section starts on the finished last frame of the
one before, so two sections share that frame and nothing is blended at a join.

The stretched frames go back to a latent through the video encoder in 17-frame groups (the model's clock) and in
tiles across the picture: a whole 17-frame group costs about 0.9 + 52 x megapixels GiB, a 384-px tile 8.2 GiB.
"""
from __future__ import annotations

import torch

SHIFT = 12.0                 # the sampler's schedule shift (sampling.sample_schedule's)
SECTION = 124                # frames in a section unless the page sets another length
MIN_SECTION = 22
TAIL = 56                    # the last section is at least this long: a leftover of a few frames is not a clip
DENOISE, STEPS = 0.4, 3


def grid_floor(n: int) -> int:
    """The longest clip length on the model's 17n+5 grid that is at most n frames (never under 22)."""
    return max(MIN_SECTION, 17 * ((int(n) - 5) // 17) + 5)


def grid_ceil(n: int) -> int:
    """The shortest clip length on the grid that is at least n frames."""
    return max(MIN_SECTION, 17 * -((5 - int(n)) // 17) + 5)


def section_frames(frames: int = 0) -> int:
    """A section length on the grid: the page's box (0 or nothing = 124)."""
    return grid_floor(int(frames)) if int(frames or 0) > 0 else SECTION


def sections(total: int, section: int = SECTION) -> "list[tuple[int, int]]":
    """(start, length) of each section over a clip of `total` frames. Every length is on the frame grid and a section
    starts on the last frame of the one before. A clip whose length the sections cannot land on exactly, or that
    would leave a last section of under 56 frames, gets a last section that ends on the clip's last frame and starts
    earlier, on a frame already finished."""
    total, section = int(total), section_frames(section)
    if total < MIN_SECTION:
        raise ValueError(f"A clip needs at least {MIN_SECTION} frames to be upscaled (this one has {total}).")
    out, a = [], 0
    while total - a > 1:
        r = total - a
        if r > section:
            out.append((a, section))
            a += section - 1
            continue
        if not out:
            g = grid_floor(r)
            out.append((0, g))
            if g != r:
                out.append((total - g, g))
        else:
            g = r if (r == grid_floor(r) and r >= TAIL) else min(section, grid_ceil(max(r, TAIL)))
            out.append((total - g, g))
        break
    return out


def low_sigmas(denoise: float, steps: int, bias: float = 0.0) -> "list[float]":
    """`steps` noise levels from `denoise` down to 0, spaced as the model's own schedule spaces them (shift 12).
    bias (-1 to 1) moves the levels between the two ends: above 0 they crowd toward 0 (more of the steps on fine
    detail), below 0 toward `denoise` (more of them on the larger changes). 0 is the model's own spacing."""
    s0, k = min(1.0, max(0.02, float(denoise))), max(1, int(steps))
    p = 3.0 ** min(1.0, max(-1.0, float(bias or 0.0)))
    t0 = s0 / (SHIFT - (SHIFT - 1.0) * s0)
    return [SHIFT * t / (1.0 + (SHIFT - 1.0) * t) for t in (t0 * (1.0 - i / k) ** p for i in range(k + 1))]


def tile_for(free_gib: float) -> int:
    """The encoder's tile, in pixels, for the memory free on the card (a 17-frame tile: 384 px 8.2 GiB, 256 px 4.0,
    192 px about 2.8, 128 px about 1.8)."""
    return 384 if free_gib >= 10.0 else 256 if free_gib >= 5.5 else 192 if free_gib >= 3.5 else 128


def _positions(size: int, tile: int, overlap: int) -> "list[int]":
    if size <= tile:
        return [0]
    return sorted(set(list(range(0, size - tile, tile - overlap)) + [size - tile]))


@torch.no_grad()
def encode_tiled(encoder, frames, device, tile: int = 384, overlap: int = 64):
    """[PIL frames of one size, a count on the 17n+5 grid] -> latent [1, 24, T, H/16, W/16] on the CPU. 17-frame
    groups then the closing 5, each group encoded in tiles of `tile` pixels blended over `overlap`."""
    import numpy as np
    arr = np.stack([np.asarray(im.convert("RGB")) for im in frames])
    f, h, w = arr.shape[:3]
    if f < 5 or (f - 5) % 17:
        raise ValueError(f"{f} frames is not on the model's 17n+5 grid")
    th, tw = min(tile, h), min(tile, w)
    overlap = min(overlap, (tile // 64) * 16)          # a small tile blends over a quarter of itself
    k = overlap // 16
    edge = torch.linspace(0, 1, k + 2)[1:-1]
    lat, wsum, i, ti = None, torch.zeros(1, 1, 1, h // 16, w // 16), 0, 0
    for size in [17] * ((f - 5) // 17) + [5]:
        vid = torch.from_numpy(arr[i:i + size]).permute(3, 0, 1, 2).float().div(127.5).sub(1.0).unsqueeze(0)
        n_t = 0
        for y0 in _positions(h, th, overlap):
            for x0 in _positions(w, tw, overlap):
                z = encoder.encode(vid[:, :, :, y0:y0 + th, x0:x0 + tw].to(device)).float().cpu()
                if lat is None:
                    lat = torch.zeros(1, z.shape[1], 5 * ((f - 5) // 17) + 2, h // 16, w // 16)
                n_t = z.shape[2]
                wy, wx = torch.ones(th // 16), torch.ones(tw // 16)
                if y0 > 0:
                    wy[:k] = edge
                if y0 + th < h:
                    wy[-k:] = torch.minimum(wy[-k:], edge.flip(0))
                if x0 > 0:
                    wx[:k] = edge
                if x0 + tw < w:
                    wx[-k:] = torch.minimum(wx[-k:], edge.flip(0))
                m = (wy.view(-1, 1) * wx.view(1, -1)).view(1, 1, 1, th // 16, tw // 16)
                ys, xs = slice(y0 // 16, (y0 + th) // 16), slice(x0 // 16, (x0 + tw) // 16)
                lat[:, :, ti:ti + n_t, ys, xs] += z * m
                if ti == 0:
                    wsum[:, :, :, ys, xs] += m
        i += size
        ti += n_t
    return lat / wsum


@torch.no_grad()
def finish(model, x, emb, sigmas, *, audio=None, audio_noise=None, keyframes=None, dtype=torch.float32,
           on_step=None):
    """The sampler's own steps (res_multistep) over `sigmas`, from a latent already at sigmas[0]. audio: finished
    sound rows, shown at each step's noise level (mixed with audio_noise) and left as they are; without them
    audio_noise rides along as the sound. on_step(i, n) may raise to stop. Returns the finished latent."""
    from fizgig.minimax import sampling
    from fizgig.minimax.model import AUDIO_SIGMA_SHIFT, remap_sigma
    kw = {"keyframes": keyframes} if keyframes else {}
    prev, n = None, len(sigmas) - 1
    for i in range(n):
        if on_step is not None:
            on_step(i, n)
        s, sn = float(sigmas[i]), float(sigmas[i + 1])
        t = torch.tensor([1.0 - s], device=x.device)
        a_in = None
        if audio_noise is not None:
            sv = max(s, 1e-6)
            sa = float(remap_sigma(torch.tensor(sv), SHIFT, AUDIO_SIGMA_SHIFT))
            a_in = (((1.0 - sa) * audio + sa * audio_noise) if audio is not None else audio_noise * (sa / sv)).to(dtype)
        out = model(x.to(dtype), t, emb, audio_rows=a_in, return_audio=True, **kw)
        out = (out[0] if isinstance(out, (tuple, list)) else out).float()
        den = x + s * out
        if prev is not None and sn > 0:
            ax, hb1, hb2 = sampling._res_multistep_coeffs(s, sn, float(sigmas[i - 1]))
            x = ax * x + hb1 * den + hb2 * prev
        else:
            x = x + (s - sn) * out
        prev = den
    return x
