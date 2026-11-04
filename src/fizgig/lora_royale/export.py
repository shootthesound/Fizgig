"""Export the LoRA Royale crossfade as a shareable GIF or MP4.

The morph is the product: a face resolving epoch-by-epoch, ping-pong looped,
with an epoch ticker and a 'Fizgig · LoRA Royale' tag burned in. Every clip is
an advert for the tool, and the tool makes the clip for you.

Frames are built from the same Image.blend the crossfade slider uses, so the
exported sweep is exactly what you saw. PIL writes the GIF; the MP4 is H.264 +
yuv420p + faststart via ffmpeg (bundled imageio-ffmpeg or a system ffmpeg) so it
plays on Reddit / X / Instagram / Discord — falling back to OpenCV mp4v only if no
ffmpeg is present.
"""

from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# Speed preset -> (frames_per_transition, hold_frames, fps). fps is what sets
# playback speed for the fixed-length travel clips; the epoch morph also reacts
# to frames_per_transition/hold. All three vary so the control is felt.
SPEED_PRESETS = {
    "Slow":   (24, 12, 14),
    "Normal": (16, 8, 22),
    "Fast":   (10, 4, 32),
}


def _load_font(size: int):
    for name in ("seguisb.ttf", "segoeui.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    """Greedy word-wrap `text` to fit `max_width` at `font`. A single word wider
    than max_width is left on its own line (font-shrinking handles it)."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def _pill_lines(draw, lines, font, x_left, bottom, fg, bg, pad):
    """Draw a rounded translucent multi-line pill, bottom edge at `bottom`,
    left edge at `x_left`, growing upward."""
    px, py = pad
    asc, desc = font.getmetrics()
    lh = asc + desc
    gap = max(1, lh // 6)
    tw = max((draw.textlength(ln, font=font) for ln in lines), default=0)
    th = lh * len(lines) + gap * (len(lines) - 1)
    w, h = int(tw + px * 2), int(th + py * 2)
    y1 = bottom
    y0 = y1 - h
    rad = max(6, min(h // 2, lh // 2 + py))
    draw.rounded_rectangle([x_left, y0, x_left + w, y1], radius=rad, fill=bg)
    cy = y0 + py
    for ln in lines:
        draw.text((x_left + px, cy), ln, font=font, fill=fg)
        cy += lh + gap
    return w, h


def _decorate(img: Image.Image, epoch_label=None, brand=True, badge=None) -> Image.Image:
    """Burn a corner badge (bottom-left, gold) + brand pill (bottom-right).

    `badge` is raw text; `epoch_label` is a convenience that renders as
    "EPOCH <n>". Pass at most one. The badge keeps the default font size for
    short text; only when it's long enough to reach the Fizgig tag does it
    shrink the font and wrap to multiple lines to avoid colliding."""
    badge_text = badge if badge is not None else (f"EPOCH {epoch_label}" if epoch_label is not None else None)
    if badge_text is None and not brand:
        return img.convert("RGB")
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    W, H = base.size
    # Size text/margins off the SHORTER side so portrait (narrow) frames don't
    # get oversized pills that collide in the middle.
    short = min(W, H)
    margin = max(8, short // 40)
    fs = max(13, short // 26)
    pad = (max(6, fs // 2), max(4, fs // 3))

    # Brand pill (bottom-right) drawn first so we know how much room it claims.
    brand_w = 0
    if brand:
        bfont = _load_font(fs)
        btext = "Fizgig · LoRA Royale"
        bw = int(draw_len(d, btext, bfont) + pad[0] * 2)
        _pill_lines(d, [btext], bfont, W - margin - bw, H - margin,
                    (255, 255, 255, 235), (0, 0, 0, 140), pad)
        brand_w = bw

    if badge_text is not None:
        gap = margin
        avail = (W - margin - brand_w - gap) - margin if brand else (W - 2 * margin)
        avail = max(60, avail)
        min_fs = max(10, short // 60)
        # Keep the default size for short text; shrink only until each line fits.
        bfs = fs
        while bfs > min_fs:
            f = _load_font(bfs)
            lines = _wrap_text(d, badge_text, f, avail)
            if max(draw_len(d, ln, f) for ln in lines) <= avail:
                break
            bfs -= 2
        bfont = _load_font(bfs)
        blines = _wrap_text(d, badge_text, bfont, avail)
        bpad = (max(6, bfs // 2), max(4, bfs // 3))
        _pill_lines(d, blines, bfont, margin, H - margin, (255, 210, 74, 245), (0, 0, 0, 140), bpad)

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_len(draw, text, font):
    return draw.textlength(text, font=font)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def deflicker_frames(images: List[Image.Image], strength: float = 1.0,
                     sigma: Optional[float] = None) -> List[Image.Image]:
    """Timelapse-style luminance deflicker for a rendered frame sequence.

    The same idea as DaVinci Resolve's Timelapse Deflicker: measure each frame's
    brightness, fit a smooth low-frequency baseline across the sweep, and pull each
    frame's exposure toward that baseline. High-frequency frame-to-frame wobble
    (flicker) is removed; the intended slow brightness arc (a sunset darkening, a
    Time-of-day morph) lives in the baseline and is preserved — so it's safe to run
    even on lighting dimensions.

    Multiplicative gain, because exposure flicker is multiplicative. The gain is a
    single per-frame scalar applied uniformly to R/G/B, so colour balance is kept.

    `strength` 0..1 blends the correction (1 = full pull to baseline). `sigma` is the
    Gaussian width in frames; auto from frame count when None (smaller window = treats
    slower drift as flicker; the auto value keeps the global arc and removes jitter).
    """
    if not images or len(images) < 3:
        return images
    import numpy as np
    arrs, means = [], []
    for im in images:
        a = np.asarray(im.convert("RGB"), dtype=np.float32)
        arrs.append(a)
        luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        means.append(float(luma.mean()))
    means = np.asarray(means, dtype=np.float32)
    n = len(means)
    if sigma is None:
        sigma = max(1.0, n / 10.0)
    radius = max(1, int(round(sigma * 3)))
    ker = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    ker /= ker.sum()
    baseline = np.convolve(np.pad(means, radius, mode="reflect"), ker, mode="valid")

    strength = max(0.0, min(1.0, float(strength)))
    out: List[Image.Image] = []
    for i, a in enumerate(arrs):
        m = means[i]
        if m <= 1e-3:
            out.append(images[i])
            continue
        g = baseline[i] / m
        g = 1.0 + strength * (g - 1.0)
        g = max(0.5, min(2.0, g))             # guard near-black/blown frames
        out.append(Image.fromarray(np.clip(a * g, 0, 255).astype(np.uint8)))
    return out


def build_frames(images: List[Tuple], speed: str = "Normal", pingpong: bool = True,
                 brand: bool = True, show_epoch: bool = True,
                 max_size: Optional[int] = 768) -> List[Image.Image]:
    """images: [(label, PIL)] in epoch order. Returns decorated RGB frames.

    Holds on each epoch, then blends to the next (the slider's morph). With
    pingpong, the return sweep replays the transitions in reverse for a
    seamless loop. Epoch ticker shows the dominant epoch in each frame.
    """
    if not images:
        return []
    fpt, hold, _fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])

    # Normalise size: all frames share the first image's box (even dims for MP4),
    # optionally downscaled so clips stay shareable.
    base_w, base_h = images[0][1].size
    if max_size and max(base_w, base_h) > max_size:
        scale = max_size / float(max(base_w, base_h))
        base_w, base_h = int(base_w * scale), int(base_h * scale)
    base_w, base_h = max(2, _even(base_w)), max(2, _even(base_h))

    def prep(im):
        return im if im.size == (base_w, base_h) else im.resize((base_w, base_h), Image.LANCZOS)

    norm = [(label, prep(im)) for label, im in images]

    def decorate(im, label):
        return _decorate(im, epoch_label=(label if show_epoch else None), brand=brand)

    frames: List[Image.Image] = []
    if len(norm) == 1:
        label, im = norm[0]
        frames.extend([decorate(im, label)] * max(hold, fpt))
        return frames

    for i in range(len(norm) - 1):
        (la, a), (lb, b) = norm[i], norm[i + 1]
        frames.extend([decorate(a, la)] * hold)              # hold on epoch i
        for k in range(1, fpt + 1):
            alpha = k / float(fpt + 1)
            label = la if alpha < 0.5 else lb                # dominant epoch
            frames.append(decorate(Image.blend(a, b, alpha), label))
    # hold on the final epoch
    lz, z = norm[-1]
    frames.extend([decorate(z, lz)] * hold)

    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    return frames


def frames_from_sequence(images: List[Image.Image], pingpong: bool = True,
                         brand: bool = True, label: Optional[str] = None,
                         labels: Optional[List[str]] = None,
                         max_size: Optional[int] = 768) -> List[Image.Image]:
    """Decorate an already-smooth sequence (e.g. a seed- or prompt-travel sweep)
    1:1 — no blending, since the frames are already continuous. Ping-pong for a
    seamless loop.

    Badge: pass `labels` (one per frame, e.g. the dominant prompt-travel word) for
    a ticking badge, or a single static `label` (e.g. "EPOCH 10"). `labels` wins.

    `images`: list of PIL frames in order.
    """
    if not images:
        return []
    base_w, base_h = images[0].size
    if max_size and max(base_w, base_h) > max_size:
        scale = max_size / float(max(base_w, base_h))
        base_w, base_h = int(base_w * scale), int(base_h * scale)
    base_w, base_h = max(2, _even(base_w)), max(2, _even(base_h))

    out = []
    for i, im in enumerate(images):
        if im.size != (base_w, base_h):
            im = im.resize((base_w, base_h), Image.LANCZOS)
        badge = labels[i] if (labels and i < len(labels)) else label
        out.append(_decorate(im, badge=badge, brand=brand))
    if pingpong and len(out) > 2:
        out = out + out[-2:0:-1]
    return out


def _fit_height(im: Image.Image, h: int) -> Image.Image:
    """Resize to height `h` preserving aspect (even width for MP4). No crop, no
    distortion — the reference photo keeps its shape whatever its aspect, and just
    sits in a wider/narrower panel beside the epoch render."""
    im = im.convert("RGB")
    w0, h0 = im.size
    w = max(2, _even(round(w0 * h / max(1, h0))))
    return im.resize((w, h), Image.LANCZOS)


def _score_badge(score) -> Tuple[str, tuple]:
    """(text, pill_rgba) for a cosine likeness score, colour-coded: green (strong,
    >=0.5), amber (0.35-0.5), red (weak), grey (no face / NaN)."""
    import math
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "NO FACE", (110, 110, 116, 235)
    s = float(score)
    if s >= 0.5:
        col = (60, 180, 110, 245)
    elif s >= 0.35:
        col = (235, 185, 70, 245)
    else:
        col = (210, 90, 80, 245)
    return f"LIKENESS {s:.2f}", col


def build_likeness_frames(ref_image: Image.Image, images: List[Tuple], scores: dict,
                          speed: str = "Normal", pingpong: bool = True, brand: bool = True,
                          max_size: Optional[int] = 768) -> List[Image.Image]:
    """Side-by-side likeness clip: [ reference | epoch ] with the ArcFace likeness
    score burned in, morphing epoch-by-epoch like the crossfade — the reference panel
    stays fixed while the epoch panel blends and the score ticks. `scores` maps the
    epoch label to its cosine (NaN / missing → 'NO FACE'). Mirrors build_frames'
    hold + blend + ping-pong timing."""
    if ref_image is None or not images:
        return []
    fpt, hold, _fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])

    # Match both panels to a common HEIGHT, preserving each image's aspect (no crop,
    # no distortion) — the source can be any shape. The composite is wide (side by side):
    # W = ref_w + gap + epoch_w. Height capped by max_size for shareability.
    H = images[0][1].size[1]
    if max_size:
        H = min(H, max_size // 2)
    H = max(2, _even(H))

    ref_p = _fit_height(ref_image, H)
    epochs = [(label, _fit_height(im, H)) for label, im in images]
    ref_w = ref_p.size[0]
    ep_w = epochs[0][1].size[0]
    gap = max(2, _even(H // 64))
    W = max(2, _even(ref_w + gap + ep_w))
    margin = max(8, H // 40)
    fs = max(13, H // 26)
    pad = (max(6, fs // 2), max(4, fs // 3))
    sfs = max(16, H // 15)
    spad = (max(9, sfs // 2), max(6, sfs // 3))

    def compose(epoch_img: Image.Image, label) -> Image.Image:
        canvas = Image.new("RGB", (W, H), (16, 16, 18))
        canvas.paste(ref_p, (0, 0))
        canvas.paste(epoch_img, (ref_w + gap, 0))
        base = canvas.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        f = _load_font(fs)
        # Panel labels (bottom-left of each panel).
        _pill_lines(d, ["SOURCE"], f, margin, H - margin, (255, 255, 255, 235), (0, 0, 0, 150), pad)
        ep_lbl = f"EPOCH {label}" if str(label).isdigit() else str(label)
        _pill_lines(d, [ep_lbl], f, ref_w + gap + margin, H - margin, (255, 210, 74, 245), (0, 0, 0, 150), pad)
        # Likeness score headline, centered at the top.
        stext, scol = _score_badge(scores.get(label) if scores else None)
        sf = _load_font(sfs)
        asc, desc = sf.getmetrics()
        s_h = (asc + desc) + 2 * spad[1]
        sw = int(draw_len(d, stext, sf) + spad[0] * 2)
        _pill_lines(d, [stext], sf, (W - sw) // 2, margin + s_h, (20, 20, 22, 255), scol, spad)
        # Brand pill (bottom-right of the whole composite).
        if brand:
            bf = _load_font(fs)
            bt = "Fizgig · LoRA Royale"
            bw = int(draw_len(d, bt, bf) + pad[0] * 2)
            _pill_lines(d, [bt], bf, W - margin - bw, H - margin, (255, 255, 255, 235), (0, 0, 0, 140), pad)
        return Image.alpha_composite(base, overlay).convert("RGB")

    frames: List[Image.Image] = []
    if len(epochs) == 1:
        l, im = epochs[0]
        frames.extend([compose(im, l)] * max(hold, fpt))
        return frames
    for i in range(len(epochs) - 1):
        (la, a), (lb, b) = epochs[i], epochs[i + 1]
        frames.extend([compose(a, la)] * hold)               # hold on epoch i
        for k in range(1, fpt + 1):
            alpha = k / float(fpt + 1)
            label = la if alpha < 0.5 else lb                # dominant epoch -> its score
            frames.append(compose(Image.blend(a, b, alpha), label))
    lz, z = epochs[-1]
    frames.extend([compose(z, lz)] * hold)                   # hold on the final epoch
    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    return frames


def write_gif(frames: List[Image.Image], path: str, speed: str = "Normal", loop: int = 0):
    if not frames:
        raise ValueError("No frames to export.")
    _, _, fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])
    dur = int(round(1000.0 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=dur, loop=loop, optimize=True, disposal=2)


def _find_ffmpeg():
    """Locate an ffmpeg binary: prefer imageio-ffmpeg's bundled, cross-platform one
    (pip-installed, no system setup), then a system ffmpeg on PATH. Returns a path or
    None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


def write_mp4(frames: List[Image.Image], path: str, speed: str = "Normal"):
    """Write a web-/social-compatible MP4 — H.264 (libx264) + yuv420p + faststart —
    by piping raw frames to ffmpeg. This is what Reddit / X / Instagram / Discord
    require; the old OpenCV `mp4v` output is MPEG-4 Part 2 (plays in desktop players
    but is rejected or fails to transcode on most web platforms).

    Falls back to OpenCV `mp4v` only if no ffmpeg is found, so export still works —
    just not web-standard (a warning is printed)."""
    if not frames:
        raise ValueError("No frames to export.")
    import numpy as np
    _, _, fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])
    # H.264 + yuv420p require even dimensions.
    w, h = frames[0].size
    w, h = max(2, _even(w)), max(2, _even(h))

    def _prep(fr):
        im = fr.convert("RGB")
        return im if im.size == (w, h) else im.resize((w, h), Image.LANCZOS)

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        import subprocess
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "medium", "-movflags", "+faststart", path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fr in frames:
                proc.stdin.write(np.asarray(_prep(fr), dtype=np.uint8).tobytes())
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (exit {rc}) writing {path}. Try GIF instead.")
        return

    # No ffmpeg — fall back to OpenCV mp4v (not web-standard).
    import cv2
    print("[export] WARNING: ffmpeg not found — writing mp4v (MPEG-4 Part 2). This plays "
          "locally but may be rejected by Reddit / web. Install ffmpeg or "
          "`pip install imageio-ffmpeg` for H.264 output.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer (codec unavailable). Try GIF instead.")
    try:
        for fr in frames:
            arr = np.array(_prep(fr))[:, :, ::-1]    # RGB -> BGR
            writer.write(np.ascontiguousarray(arr))
    finally:
        writer.release()


def build_comparison_grid(rows, col_labels, *, row_labels=None, cell=None,
                          bg=(255, 255, 255), fg=(20, 20, 20), gap=0, brand=False):
    """Compose a labelled comparison sheet: one row per prompt, one column per condition.

    The share format people actually post for a new LoRA — "no lora | with lora" side by side
    on the same seed, or a run's epochs across the columns.

    rows:       list of rows, each a list of PIL images (one per column, same order as col_labels)
    col_labels: header text per column (drawn once, above the grid)
    row_labels: optional short caption drawn down the left gutter (rotated), e.g. the prompt
    cell:       (w, h) to resize every image to; defaults to the first image's size
    gap:        pixels between cells (bg-coloured)
    brand:      draw a small "Fizgig" mark bottom-right

    Returns a single RGB PIL image.
    """
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("No rows to compose.")
    n_cols = max(len(r) for r in rows)
    if cell is None:
        cell = rows[0][0].size
    cw, ch = int(cell[0]), int(cell[1])

    probe = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(probe)

    # Header font scales with cell width; shrink until the widest label fits its column.
    fs = max(14, int(cw * 0.055))
    font = _load_font(fs)
    while fs > 9 and any(d0.textlength(str(t), font=font) > cw * 0.94 for t in col_labels):
        fs -= 1
        font = _load_font(fs)
    asc, desc = font.getmetrics()
    header_h = asc + desc + max(8, int(fs * 0.7)) if col_labels else 0

    gutter = 0
    rlfont = None
    if row_labels:
        rfs = max(11, int(cw * 0.035))
        rlfont = _load_font(rfs)
        gutter = rfs + max(8, int(rfs * 0.8))

    W = gutter + n_cols * cw + (n_cols - 1) * gap
    H = header_h + len(rows) * ch + (len(rows) - 1) * gap
    sheet = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(sheet)

    for ci, label in enumerate(col_labels or []):
        cx = gutter + ci * (cw + gap) + cw // 2
        tw = draw.textlength(str(label), font=font)
        draw.text((cx - tw / 2, max(2, (header_h - asc - desc) // 2)), str(label), font=font, fill=fg)

    for ri, row in enumerate(rows):
        y = header_h + ri * (ch + gap)
        for ci in range(n_cols):
            if ci >= len(row) or row[ci] is None:
                continue
            im = row[ci]
            if im.mode != "RGB":
                from fizgig.utils.image import flatten
                im = flatten(im)
            if im.size != (cw, ch):
                im = im.resize((cw, ch), Image.LANCZOS)
            sheet.paste(im, (gutter + ci * (cw + gap), y))
        if row_labels and ri < len(row_labels) and row_labels[ri]:
            # Rotated caption in the left gutter, clipped to the row height.
            txt = str(row_labels[ri])
            strip = Image.new("RGB", (ch, gutter), bg)
            sd = ImageDraw.Draw(strip)
            while rlfont.size > 9 and sd.textlength(txt, font=rlfont) > ch - 8:
                rlfont = _load_font(rlfont.size - 1)
            sd.text((4, max(0, (gutter - rlfont.size) // 2 - 1)), txt, font=rlfont, fill=fg)
            sheet.paste(strip.rotate(90, expand=True), (0, y))

    if brand:
        bfs = max(10, int(cw * 0.03))
        bfont = _load_font(bfs)
        bt = "Fizgig"
        bw = draw.textlength(bt, font=bfont)
        draw.text((W - bw - max(6, bfs // 2), H - bfs - max(6, bfs // 2)), bt, font=bfont,
                  fill=(150, 150, 150))
    return sheet
