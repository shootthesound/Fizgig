"""Gizmo — the clip prep tool for Fizgig.

Video models train on clips, and clips have to be exactly right: for MiniMax H3, 24 fps, a frame
count on H3's 17n+5 grid, dimensions on a multiple of 32, audio at 32 kHz stereo; for Vimoo 1.0
Lite, 24 fps, 4n+1 frames, multiples of 32, no sound. Gizmo asks which model when it opens
(MODELS below) and the lengths, sizes, sound and tabs follow it. Fizgig refuses anything
off-spec rather than quietly fixing it, because a training app that silently transcodes footage
makes two identical-looking datasets train differently and nobody can tell why.

That leaves someone with an hour of source video and a spec sheet. Gizmo is the answer: open a
video, scrub to a moment, pick a length, save. Out comes a clip Fizgig accepts. Six sections from
one source is six marks and six clicks, and there is no project file to manage — mark, save,
repeat, close.

It is deliberately a separate app:

  * It keeps ffmpeg transcoding out of the training path, where it does not belong.
  * It opens in under a second, because it imports no torch and never touches Fizgig's GUI.

Two things it will NOT do, both on purpose:

  * It does not write a still beside each clip. That was the original design and it is wrong:
    Fizgig's latent cache keys on the filename stem with the extension stripped, so walk_03.mp4
    and walk_03.png collide on the same cache file and one silently overwrites the other.
  * It does not caption. Captioning happens in Fizgig afterwards, on the whole folder at once,
    which is also why muting is a filename suffix — there are no .txt files yet to keep in sync.

Run: Launch Gizmo.bat (or launch_gizmo.sh, or Open Gizmo on Fizgig's Image Prep tab)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# --- the clip spec ---------------------------------------------------------------------------
# Mirrored from src/fizgig/minimax/clip.py rather than imported: that module pulls in
# fizgig.minimax.model, which pulls in torch, and a prep tool that takes ten seconds to open
# because it loaded CUDA would be a bad tool. tests/test_gizmo_spec.py asserts the two agree, so
# the copy cannot drift without something failing loudly.
FPS = 24
GRID_FRAMES = (5, 22, 39, 56, 73, 90, 107, 124)          # 17n+5, every length the VAE encodes
LATENT_FRAMES = {f: 5 * n + 2 for n, f in enumerate(GRID_FRAMES)}   # what each costs in the DiT

# What each length can be TRAINED at, by card: frames -> {free VRAM in GB: largest training
# megapixels}. None means that card cannot do it at all.
#
# Two ceilings, and the lower one wins.
#
# The TRAINING step comes from Fizgig's own swap planner (trainer.plan_vram) rather than from
# guesswork, by feeding a clip's token load in as the equivalent megapixel value — which is
# arithmetically what it is. Its activation term is linear in tokens while attention is
# quadratic, and it was anchored on stills of 256-1024 tokens, so at the long lengths it is an
# extrapolation and where it is wrong it will be optimistic. That is what sets the LENGTH limits
# here, and the long ones still want a real run behind them.
#
# The CACHING pass is measured, on a 5090 with ~30 GiB free, encoding in 17-frame groups:
#
#     0.25 MP  ~14.2 GiB      0.5 MP  ~23.2 GiB      1.0 MP  out of memory
#
# Flat in clip length — 124 frames costs the same as 22 — because the groups bound it. That is
# what sets the MEGAPIXEL limits: 1 MP is unreachable for clips of any length on a 32 GB card,
# and 0.25 MP at ~14 GiB is already marginal on a 16 GB one. Caching is fp32 today; the file is
# natively fp16, which measures 8.05 GiB and 0.09% different, so these could move.
#
# Frozen into a table because Gizmo must not import the trainer — that would pull torch in
# behind it, and a prep tool that takes ten seconds to open because it loaded CUDA is a bad tool.
CLIP_VRAM = {
    5: {16: 0.25, 24: 0.5, 32: 0.5},
    22: {16: 0.25, 24: 0.5, 32: 0.5},
    39: {16: 0.25, 24: 0.5, 32: 0.5},
    56: {16: 0.25, 24: 0.5, 32: 0.5},
    73: {16: None, 24: 0.25, 32: 0.5},
    90: {16: None, 24: 0.25, 32: 0.5},
    107: {16: None, 24: 0.25, 32: 0.25},
    124: {16: None, 24: 0.25, 32: 0.25},
}
SIZE_STEP = 32
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2
MUTE_SUFFIX = "_mute"

# --- audio / voice mode ------------------------------------------------------------------------
# A voice training item's duration is quantized by the same frame grid as a clip's: H3 packs
# round(frames/24*40) audio latents per grid slot, at 800 samples each. Gizmo exports segments at
# EXACTLY latents x 800 samples, so the trainer's strict duration check always passes with no
# tolerance spent. The 5-frame slot (0.2 s) is deliberately absent — fifty milliseconds of voice
# trains nothing worth a file.
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
AUDIO_HOP = 800                                # the audio VAE's samples-per-latent
AUDIO_GRID_FRAMES = (22, 39, 56, 73, 90, 107, 124)

# --- the model the clips are for ----------------------------------------------------------------
# Gizmo asks on open which model it is cutting for, and everything model-shaped follows: the frame
# grid, what a length costs, whether a clip's sound is training data, which tabs exist. The
# numbers are mirrored from each family's description (src/fizgig/families/<key>.py: clip_spec,
# train_memory), not imported, for the same no-torch reason as above.
H3_CLIP_VRAM = CLIP_VRAM
VIMOO_TOKEN_MP = 5.2      # measured: latent frames x megapixels up to this trains on a 16 GB card


def _vimoo_note(frames):
    """Vimoo's per-length line, from its measured memory (flat to 5.2 MP of tokens on a 16 GB RTX 4090 Laptop GPU -
    INT8 13.1-14.3 GB, NF4 9.6-10.8 GB - and on a 5090); nothing past it claimed. -> (text, warn)."""
    lat = (frames - 1) // 4 + 1
    mp = VIMOO_TOKEN_MP / lat
    return (f"Measured to train on a 16 GB card up to {mp:.2f} MP at this length ({lat} latent frames; a clip's "
            f"latent frames x its megapixels was measured up to {VIMOO_TOKEN_MP:g}). Bigger is not refused, just "
            f"not measured yet.", mp < 0.25)


MODELS = {
    "h3": {"name": "MiniMax H3", "short": "H3", "tabs": ("video", "voice"), "sound": True,
           "grid": (5, 22, 39, 56, 73, 90, 107, 124), "latent": lambda f, g: 5 * g.index(f) + 2,
           "default": 22, "vram": H3_CLIP_VRAM, "note": None,
           "what": "Video clips with their sound, and voice segments.",
           "grid_tip": "Not a free choice: H3's VAE encodes video in groups of 17, so the only lengths that exist "
                       "are 5, 22, 39, 56, 73, 90, 107 and 124.\n\n22 is the shortest that shows real movement."},
    "vimoo": {"name": "Vimoo 1.0 Lite", "short": "Vimoo", "tabs": ("video",), "sound": False,
              "grid": (17, 25, 33, 49, 73, 97, 121, 241), "latent": lambda f, g: (f - 1) // 4 + 1,
              "default": 49, "vram": None, "note": _vimoo_note,
              "what": "Video clips - Vimoo trains picture only, no sound.",
              "grid_tip": "Not a free choice: Vimoo's VAE encodes the first frame alone and the rest in groups of "
                          "4, so a length is 4n+1 - these are the useful ones up to its 10-second maximum.\n\n"
                          "49 frames (2 s) is a good default."},
    "acestep": {"name": "ACE-Step", "short": "ACE-Step", "tabs": ("music",), "sound": True,
                "grid": (5, 22, 39, 56, 73, 90, 107, 124), "latent": lambda f, g: 5 * g.index(f) + 2,
                "default": 22, "vram": H3_CLIP_VRAM, "note": None, "what": "Song sections, captioned.",
                "grid_tip": ""},
}
MODEL = MODELS["h3"]


def use_model(key):
    """Point the module's clip constants at one model - before the window builds."""
    global MODEL, GRID_FRAMES, LATENT_FRAMES, CLIP_VRAM
    MODEL = MODELS.get(key) or MODELS["h3"]
    GRID_FRAMES = MODEL["grid"]
    LATENT_FRAMES = {f: MODEL["latent"](f, GRID_FRAMES) for f in GRID_FRAMES}
    CLIP_VRAM = MODEL["vram"] or {}
    return MODEL


def choose_model(root, last):
    """The model picker shown when Gizmo opens: a button per model, the last choice focused (Enter or a click takes
    one). None when the picker is closed."""
    win = tk.Toplevel(root)
    win.title("Gizmo - which model are these for?")
    win.configure(bg=COLORS["bg_deep"])
    win.resizable(False, False)
    pick = {"key": None}
    tk.Label(win, text="Which model are you preparing for?", font=(FONT_FAMILY, 14, "bold"),
             bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(18, 4))
    tk.Label(win, text="Lengths, sizes and sound follow the model. Change it later from the top of the window.",
             font=(FONT_FAMILY, 10), bg=COLORS["bg_deep"], fg=COLORS["text_secondary"]).pack(anchor="w", padx=20)
    focus = None
    for key, m in MODELS.items():
        def take(k=key):
            pick["key"] = k
            win.destroy()
        row = tk.Frame(win, bg=COLORS["bg_deep"])
        row.pack(fill=tk.X, padx=20, pady=(10, 0))
        b = tk.Button(row, text=m["name"], width=18, command=take, font=(FONT_FAMILY, 11, "bold"),
                      bg=COLORS["accent"] if key == last else COLORS["bg_surface"], fg=COLORS["text_primary"],
                      activebackground=COLORS["accent"], activeforeground=COLORS["text_primary"], relief=tk.FLAT,
                      cursor="hand2", pady=6)
        b.pack(side=tk.LEFT)
        b.bind("<Return>", lambda _e, k=key: take(k))
        tk.Label(row, text=m["what"], font=(FONT_FAMILY, 10), bg=COLORS["bg_deep"],
                 fg=COLORS["text_secondary"]).pack(side=tk.LEFT, padx=(12, 0))
        if focus is None or key == last:
            focus = b
    tk.Frame(win, bg=COLORS["bg_deep"], height=18).pack()
    win.update_idletasks()
    win.geometry(f"+{root.winfo_screenwidth() // 2 - win.winfo_width() // 2}"
                 f"+{root.winfo_screenheight() // 3 - win.winfo_height() // 2}")
    win.bind("<Escape>", lambda _e: win.destroy())
    win.lift()
    win.focus_force()
    focus.focus_set()
    root.wait_window(win)
    return pick["key"]


# --- music mode (ACE-Step) ---------------------------------------------------------------------
# ACE-Step has no length grid: a song section of any length trains, and the trainer cuts anything
# over four minutes to its first four. Sections export at ACE-Step's own 48 kHz stereo.
MUSIC_SAMPLE_RATE = 48000
MUSIC_MAX_S = 240.0
MUSIC_MIN_S = 10.0
MUSIC_WORDS_EXAMPLE = "chiptune, 8-bit, instrumental, square wave lead, arpeggios"


def audio_latents_for(frames):
    return int(round(frames / FPS * 40))


def hop_exact_samples(frames):
    """The exact sample count a segment of `frames` exports at — and the exact count Fizgig's
    cache validation expects. 22 frames -> 29600 (0.925 s) ... 124 -> 165600 (5.175 s)."""
    return audio_latents_for(frames) * AUDIO_HOP


def is_audio_file(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


# --- the voice recorder ------------------------------------------------------------------------
# Capture runs CONTINUOUSLY while the recorder window is open, writing raw s16le PCM at the
# training rate; push-to-record only marks timestamps, and the take is sliced out by byte
# offset afterwards. Starting ffmpeg on press would clip the first word every time (opening a
# capture device costs ~300-500 ms); with the mic already rolling, a pre-roll cushion even
# catches a word that starts before the thumb lands. Raw PCM is the load-bearing format
# choice: headerless, so byte offset = time exactly, slicing a growing file is always valid,
# and terminate() is harmless — there is no header to finalise.

CAPTURE_RATE = 32000                 # the training rate — takes need no resample
CAPTURE_CHANNELS = 2
CAPTURE_BPS = CAPTURE_RATE * CAPTURE_CHANNELS * 2      # bytes per second of s16le
TAKE_PRE_ROLL = 0.25                 # seconds kept BEFORE the press
TAKE_POST_ROLL = 0.15                # ...and after the release


def parse_dshow_devices(stderr_text):
    """Audio device names out of `ffmpeg -list_devices true -f dshow -i dummy` stderr.

    The banner lists devices as `[dshow @ ...] "Name" (audio)` (older builds print a
    `DirectShow audio devices` section header instead of the per-line tag). Both forms are
    handled; anything unparseable yields [] and the caller falls back to "default"."""
    devices, in_audio_section = [], False
    for line in (stderr_text or "").splitlines():
        if "DirectShow audio devices" in line:
            in_audio_section = True
            continue
        if "DirectShow video devices" in line:
            in_audio_section = False
            continue
        m = re.search(r'"([^"]+)"', line)
        if not m or "Alternative name" in line:
            continue
        if "(audio)" in line or (in_audio_section and "(video)" not in line):
            devices.append(m.group(1))
    return devices


def parse_pactl_sources(stdout_text):
    """Source names out of `pactl list sources short`. Monitor sources (the loopback of an
    OUTPUT) are skipped — recording your own speakers is never what a voice dataset wants."""
    out = []
    for line in (stdout_text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and ".monitor" not in parts[1]:
            out.append(parts[1])
    return out


def list_capture_devices(ffmpeg):
    """Microphones this machine offers, best-effort. Always safe to call; [] means none
    found (the recorder then offers just 'default' on Linux, or disables on Windows —
    dshow cannot record without a concrete device name)."""
    try:
        if os.name == "nt":
            p = _run([ffmpeg, "-hide_banner", "-list_devices", "true",
                      "-f", "dshow", "-i", "dummy"])
            return parse_dshow_devices((p.stderr or b"").decode("utf-8", "replace"))
        import shutil
        if shutil.which("pactl"):
            p = _run(["pactl", "list", "sources", "short"])
            found = parse_pactl_sources((p.stdout or b"").decode("utf-8", "replace"))
            if found:
                return found
        return ["default"] if shutil.which("arecord") or shutil.which("pactl") else []
    except Exception:
        return []


def build_capture_command(ffmpeg, device, out_raw):
    """The continuous-capture command, or None when this machine cannot record (a pod).
    Output is raw s16le at the training rate — see the section comment for why raw."""
    if not ffmpeg:
        return None
    tail = ["-ac", str(CAPTURE_CHANNELS), "-ar", str(CAPTURE_RATE),
            "-f", "s16le", "-y", out_raw]
    if os.name == "nt":
        if not device:
            return None                       # dshow needs a real device name
        # audio_buffer_size 50 ms: dshow's default buffers ~half a second before delivering,
        # which is half a second of write lag between the microphone and the raw file — the
        # lag every timestamp-to-byte mapping then has to survive. Small buffer, small lag.
        return [ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "dshow", "-audio_buffer_size", "50", "-i", f"audio={device}"] + tail
    import shutil
    if shutil.which("pactl") or (device and device != "alsa-default"):
        return [ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "pulse", "-i", device or "default"] + tail
    if shutil.which("arecord"):
        return [ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "alsa", "-i", "default"] + tail
    return None


def take_envelope(chunk, window_s=0.01):
    """Peak amplitude per `window_s` window of a raw s16le stereo chunk — the take's shape,
    cheap enough to run on every release (array module, no numpy)."""
    import array
    a = array.array("h", chunk[: len(chunk) // 2 * 2])
    win = max(1, int(window_s * CAPTURE_RATE)) * CAPTURE_CHANNELS
    peaks = []
    for i in range(0, len(a), win):
        seg = a[i:i + win]
        peaks.append(max(max(seg), -min(seg)) if len(seg) else 0)
    return peaks


def find_speech_bounds(peaks, window_s=0.01, min_run_s=0.12, attach_gap_s=0.12):
    """The first and last speech in an envelope, as (start_window, end_window).

    Two-phase, because real speech is not a steady tone. ANCHORS first: runs that stay above
    the threshold for min_run_s — the R key's click is loud but short, so it can never anchor.
    Then ATTACH: quieter activity within attach_gap_s of the speech is pulled in, because soft
    onsets and tails (an s, an f, a trailing breath of a word) cluster right against the loud
    part and belong to it — cutting at the anchor alone clipped word edges and read real takes
    short (Peter). A click stays excluded because it sits ALONE, behind a gap bigger than
    anything inside speech. The threshold rides the take's own noise floor (20th percentile)
    with an absolute floor, so a hot mic and a quiet one both work. None when the take holds
    no sustained sound at all."""
    if not peaks:
        return None
    floor = sorted(peaks)[len(peaks) // 5]
    threshold = max(floor * 2.5, 400)            # 400/32767 ~ -38 dBFS
    need = max(1, int(round(min_run_s / window_s)))
    active = [p >= threshold for p in peaks]
    runs, start = [], None
    for i, on in enumerate(active + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    anchors = [r for r in runs if r[1] - r[0] >= need]
    if not anchors:
        return None
    a, b = anchors[0][0], anchors[-1][1]
    gap = max(1, int(round(attach_gap_s / window_s)))
    changed = True
    while changed:                               # chains of soft clusters attach one by one
        changed = False
        for s, e in runs:
            if e <= a and a - e <= gap and s < a:
                a, changed = s, True
            if s >= b and s - b <= gap and e > b:
                b, changed = e, True
    return a, b


def trim_take_silence(chunk, margin_lead_s=0.15, margin_tail_s=0.2):
    """Trim a raw take to its speech, with a comfort margin either side.

    Cuts the leading and trailing silence AND the key-down/key-up click that push-to-record
    inevitably captures (Peter). Conservative on purpose: a take with no sustained sound, or
    one the trim would reduce to almost nothing, comes back unchanged — a bad trim loses a
    take, no trim only costs a moment of silence. Returns (chunk, cut_lead_s, cut_tail_s)."""
    window_s = 0.01
    total_s = len(chunk) / CAPTURE_BPS
    bounds = find_speech_bounds(take_envelope(chunk, window_s), window_s)
    if bounds is None:
        return chunk, 0.0, 0.0
    lead = max(0.0, bounds[0] * window_s - margin_lead_s)
    end = min(total_s, bounds[1] * window_s + margin_tail_s)
    frame = CAPTURE_CHANNELS * 2
    a = int(lead * CAPTURE_BPS) // frame * frame
    b = int(end * CAPTURE_BPS) // frame * frame
    if b - a < int(0.2 * CAPTURE_BPS):
        return chunk, 0.0, 0.0
    return chunk[a:b], lead, total_s - end


TAKE_CONTEXT = 0.75                  # seconds of extra session audio read around a take


def cut_take_to_slot(raw_path, press_s, release_s, out_wav, ffmpeg):
    """Cut a take that is EXACTLY one grid slot long, fitted around its speech.

    Trimming down to the speech left takes just under a slot, and the editor's auto-pick
    then snapped DOWN and cut words (Peter). This goes the other way: find the speech, pick
    the smallest slot that HOLDS it (with margins), and widen the cut to exactly that length
    using the real room tone around the words — the mic rolls continuously, so that context
    exists on both sides of the hold. Only when the session itself runs out is the tail
    zero-padded. The take that lands in the editor is slot-exact: the auto-picked length
    always fits the words, and the export re-fits nothing. Returns the duration in seconds.

    Speech longer than the biggest slot is a LONG take — someone talking for a couple of
    minutes to carve up in the editor — and is kept IN FULL, trimmed to its speech. (It
    used to be centred into one 5.2 s slot, which destroyed it; there is no length limit
    on a recording, only on what the editor exports per segment.)
    """
    # A hold much longer than the biggest slot is a long take by INTENT, whatever the speech
    # detector would say — its noise floor rides the take's 20th percentile, and once speech
    # dominates a minutes-long hold that percentile IS speech, so the detected bounds collapse
    # to the loudest stretch and the slot fit truncates the take (Peter's 1-minute recording
    # came back as 5.2 s). Hold length is the honest signal: keep the whole thing, verbatim.
    if (release_s - press_s) > hop_exact_samples(AUDIO_GRID_FRAMES[-1]) / CAPTURE_RATE + 2.0:
        return slice_take(raw_path, press_s, release_s, out_wav, ffmpeg)

    size = os.path.getsize(raw_path)
    frame = CAPTURE_CHANNELS * 2
    a0 = max(0, int((press_s - TAKE_PRE_ROLL - TAKE_CONTEXT) * CAPTURE_BPS)) // frame * frame
    b0 = min(size, int((release_s + TAKE_POST_ROLL + TAKE_CONTEXT)
                       * CAPTURE_BPS)) // frame * frame
    if b0 <= a0:
        return 0.0
    with open(raw_path, "rb") as f:
        f.seek(a0)
        chunk = f.read(b0 - a0)
    chunk_s = len(chunk) / CAPTURE_BPS

    window_s = 0.01
    bounds = find_speech_bounds(take_envelope(chunk, window_s), window_s)
    base_s = a0 / CAPTURE_BPS                  # where the chunk sits in session time
    if bounds is None:
        # nothing sustained — fall back to the plain press/release cut, untrimmed
        sa = max(0.0, press_s - TAKE_PRE_ROLL - base_s)
        sb = min(chunk_s, release_s + TAKE_POST_ROLL - base_s)
    else:
        sa = max(0.0, bounds[0] * window_s - 0.15)
        sb = min(chunk_s, bounds[1] * window_s + 0.2)
    speech_s = max(0.05, sb - sa)

    slot = next((f for f in AUDIO_GRID_FRAMES
                 if hop_exact_samples(f) / CAPTURE_RATE >= speech_s - 0.01), None)
    if slot is None:
        # LONG take: keep the whole speech span (margins included), no slot fit. The editor
        # is the tool for carving it into segments; segments snap to the grid, sources don't.
        a = max(0, int(sa * CAPTURE_BPS)) // frame * frame
        b = min(len(chunk), (int(sb * CAPTURE_BPS) // frame) * frame)
        piece = chunk[a:b]
        need_s = len(piece) / CAPTURE_BPS
    else:
        need_bytes = hop_exact_samples(slot) * frame
        need_s = need_bytes / CAPTURE_BPS
        # widen around the speech with real context, favouring whichever side has room
        extra = need_s - speech_s
        left = min(extra / 2, sa)
        right = min(extra - left, chunk_s - sb)
        left = min(extra - right, sa)
        start_s = sa - left
        a = max(0, int(start_s * CAPTURE_BPS)) // frame * frame
        piece = chunk[a: a + need_bytes]
        if len(piece) < need_bytes:
            piece += b"\x00" * (need_bytes - len(piece))

    tmp = out_wav + ".raw"
    with open(tmp, "wb") as f:
        f.write(piece)
    p = _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
              "-f", "s16le", "-ar", str(CAPTURE_RATE), "-ac", str(CAPTURE_CHANNELS),
              "-i", tmp, "-c:a", "pcm_s16le", out_wav])
    try:
        os.remove(tmp)
    except OSError:
        pass
    if p.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-200:]
                           or "could not write the take")
    return need_s


def wait_for_capture(raw_path, until_s, timeout=6.0, poll=0.05):
    """Block until the session raw file actually CONTAINS audio up to `until_s` seconds.

    ffmpeg writes a few hundred milliseconds behind the microphone, so a take sliced at the
    instant of release is clamped at the current file size and loses its tail — which is what
    made short sentences read as "too short" and the auto-picked length cut endings off
    (Peter). The timeout is generous because it only ever costs a slow machine a wait, while
    timing out early costs the take its tail. Returns True when the bytes arrived, False on
    timeout (slice what exists)."""
    import time as _time
    need = int(until_s * CAPTURE_BPS)
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            if os.path.getsize(raw_path) >= need:
                return True
        except OSError:
            pass
        _time.sleep(poll)
    return False


def slice_take(raw_path, press_s, release_s, out_wav, ffmpeg, trim_silence=False):
    """Cut [press - pre-roll, release + post-roll] out of the session raw and wrap it as a
    training-rate wav. Byte math on s16le: offset = secs x 32000 x 2ch x 2B, snapped to a
    whole frame so channels never swap. trim_silence=True then shrinks the cut to its speech
    (see trim_take_silence) — the recorder's default. Returns the duration in seconds."""
    size = os.path.getsize(raw_path)
    frame = CAPTURE_CHANNELS * 2
    a = max(0, int((press_s - TAKE_PRE_ROLL) * CAPTURE_BPS)) // frame * frame
    b = min(size, int((release_s + TAKE_POST_ROLL) * CAPTURE_BPS)) // frame * frame
    if b <= a:
        return 0.0
    with open(raw_path, "rb") as f:
        f.seek(a)
        chunk = f.read(b - a)
    if trim_silence:
        chunk, _lead, _tail = trim_take_silence(chunk)
    tmp = out_wav + ".raw"
    with open(tmp, "wb") as f:
        f.write(chunk)
    p = _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
              "-f", "s16le", "-ar", str(CAPTURE_RATE), "-ac", str(CAPTURE_CHANNELS),
              "-i", tmp, "-c:a", "pcm_s16le", out_wav])
    try:
        os.remove(tmp)
    except OSError:
        pass
    if p.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-200:]
                           or "could not write the take")
    return len(chunk) / CAPTURE_BPS


# Prompted sentences, in CATEGORIES — each spanning short interjections to lines that fill
# the 5.2 s slot, because a voice dataset wants range on two axes at once: length (phrasing)
# and tone (delivery). The picker rotates categories so consecutive takes change flavour.
#
# plain    — Harvard/IEEE sentences (public domain, phonetically balanced) + everyday lines
# pangrams — every letter in one breath; the densest phonetic coverage per second
# literary — public-domain lines with drama in them; a LINE invites a performance
# silly    — deliberately ridiculous: a genuine smile raises the formants, and you cannot
#            fake that on request the way you can fake "cheerfully"
# dark     — a low, tight tone the smiley ones never reach (gothic classics + originals)
SENTENCE_CATEGORIES = {
    "plain": [
        "Here we go.",
        "Not today.",
        "No, thank you.",
        "Right, let's begin.",
        "That's enough.",
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "It's easy to tell the depth of a well.",
        "These days a chicken leg is a rare dish.",
        "Rice is often served in round bowls.",
        "The juice of lemons makes fine punch.",
        "The box was thrown beside the parked truck.",
        "Four hours of steady work faced us.",
        "A large size in stockings is hard to sell.",
        "The boy was there when the sun rose.",
        "A rod is used to catch pink salmon.",
        "The source of the huge river is the clear spring.",
        "Kick the ball straight and follow through.",
        "Help the woman get back to her feet.",
        "A pot of tea helps to pass the evening.",
        "Smoky fires lack flame and heat.",
        "The soft cushion broke the man's fall.",
        "The salt breeze came across from the sea.",
        "The girl at the booth sold fifty bonds.",
        "The small pup gnawed a hole in the sock.",
        "The fish twisted and turned on the bent hook.",
        "Press the pants and sew a button on the vest.",
        "The swan dive was far short of perfect.",
        "The beauty of the view stunned the young boy.",
        "Two blue fish swam in the tank.",
        "Her purse was full of useless trash.",
        "The colt reared and threw the tall rider.",
        "It snowed, rained, and hailed the same morning.",
        "Read verse out loud for pleasure.",
        "Hoist the load to your left shoulder.",
        "Take the winding path to reach the lake.",
        "Note closely the size of the gas tank.",
        "Wipe the grease off his dirty face.",
        "Mend the coat before you go out.",
        "The wrist was badly strained and hung limp.",
        "The stray cat gave birth to kittens.",
        "The young girl gave no clear response.",
        "The meal was cooked before the bell rang.",
        "What joy there is in living.",
        "The old clock in the hallway has kept perfect time for ninety years.",
        "Every window in the street went dark at exactly the same moment.",
        "Fold the map along the river and the two towns touch.",
        "Nobody remembers who planted the orchard, but everyone eats the apples.",
        "The kettle sang, the rain kept its steady rhythm, and nobody felt like moving at all.",
        "By the time the bus finally arrived, all three of us had completely forgotten where we were going.",
        "The lighthouse keeper writes one letter every winter and never says who it is for.",
        "The last train of the night carries more stories than passengers, and it keeps them all.",
        "When the tide goes out past the third rock, the whole bay smells of salt and iron.",
        "The valley keeps its snow until April, then lets the river have all of it at once.",
        "String the lights from the pier to the boathouse and the whole harbour becomes a stage.",
        "Read the letter slowly, because the second page changes what the first page means.",
        "The tide rolled the little boat gently against the jetty while the gulls argued over the nets.",
        "She counted the coins twice, smiled at the total, and slid them slowly back across the counter.",
        "The wide road shimmered in the summer heat, and the dusty travellers walked on without a word.",
        "It rained through the festival, and somehow that is the year everyone remembers best.",
    ],
    "pangrams": [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "Jackdaws love my big sphinx of quartz.",
        "Bright vixens jump; dozy fowl quack.",
        "Amazingly few discotheques provide jukeboxes.",
        "The quick brown fox jumps over the lazy dog while the five boxing wizards watch.",
    ],
    "literary": [
        "Call me Ishmael.",
        "What light through yonder window breaks.",
        "Now is the winter of our discontent.",
        "Once more unto the breach, dear friends, once more.",
        "All the world's a stage, and all the men and women merely players.",
        "The clock struck thirteen as he walked out into the cold.",
        "It was the best of times, it was the worst of times, it was the age of wisdom.",
        "Once upon a midnight dreary, while I pondered, weak and weary, over many a curious volume.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife.",
    ],
    "silly": [
        "Not the llama again.",
        "More soup, vicar?",
        "The parrot disapproves.",
        "Oh, really?",
        "Come on, then.",
        "Well, look at that.",
        "The committee has voted, and the office llama stays.",
        "My grandmother arm-wrestled the postman for the last biscuit.",
        "Please stop teaching the parrot to impersonate my dentist.",
        "The wizard filed a noise complaint against the volcano.",
        "Somebody has buttered the entire staircase again.",
        "Our new intern is a heron, and honestly, he's doing fine.",
        "The moon rang twice and hung up before I could answer.",
        "I have strong opinions about the municipal soup fountain.",
        "The baker swears the missing bread was taken by a very polite ghost.",
        "The cat has decided that the piano belongs to her, and negotiations have failed.",
        "After a long and frankly emotional meeting, the committee has voted, and the office llama stays.",
        "The magician retired quietly to the seaside, but the rabbits kept arriving for years.",
        "Our neighbour has taught his pigeons to queue, and frankly the town is better for it.",
        "The orchestra tuned for so long that the audience began to applaud the tuning itself.",
        "A good soup takes patience, a heavy pot, and at least one opinionated grandmother.",
        "Paint dries, concrete sets, and yet the committee meeting simply goes on and on.",
        "The recipe has been in the family so long that nobody dares to actually read it.",
    ],
    "dark": [
        "Don't turn around.",
        "Not yet.",
        "It sees you.",
        "Stay very still.",
        "It was the beating of his hideous heart.",
        "The children of the night, what music they make.",
        "You are my creator, but I am your master.",
        "The house counts its guests at midnight, and the number is never right.",
        "Do not answer the knocking, whatever voice it uses.",
        "Deep into that darkness peering, long I stood there, wondering, fearing.",
        "The lights went out one floor at a time, and then the footsteps started.",
        "Something has been wearing his face all week, and it is getting better at it.",
        "Bring the lantern closer, because something moved at the edge of the field.",
        "Under the floorboards of the old schoolhouse, someone had hidden a jar of perfectly ordinary buttons.",
        "He whistled the same four notes at every doorway, and every door opened but one.",
        "The museum's quietest room holds a single chair, and the chair is always slightly warm.",
        "Every photograph in the album is smiling, except the one that was taken last.",
        "If you hear my voice on the stairs tonight, do not open the door, and do not answer it.",
        "Somewhere between the second lighthouse and the harbour wall, the fog swallowed the ferry whole.",
        "Nothing in the attic has moved for thirty years, which is strange, because something breathes up there.",
    ],
}

# The flat pool, for anything that just wants sentences.
HARVARD_SENTENCES = [s for _cat in SENTENCE_CATEGORIES.values() for s in _cat]

# Delivery prompts — an emotional-range axis for the dataset, and an honest caption clause.
DELIVERY_PROMPTS = {
    "plain": "",
    "cheerfully": "cheerfully",
    "wearily": "wearily",
    "as a question": "questioningly",
    "quietly": "quietly",
    "excitedly": "excitedly",
    "as if annoyed": "irritably",
}

WORDS_PER_SECOND = 2.6


def _sentences_in_band(sentences, frames):
    """The category's sentences whose word count fits the slot: within +-40% of what the
    slot's seconds hold at a natural reading pace (floor of 2 words either way)."""
    secs = hop_exact_samples(frames) / AUDIO_SAMPLE_RATE
    want = secs * WORDS_PER_SECOND
    tol = max(2, want * 0.4)
    return [s for s in sentences if abs(len(s.split()) - want) <= tol]


class SentencePicker:
    """Rotates CATEGORIES take-to-take (Peter): a plain line, then a pangram, then something
    dark — tone variety arrives on its own instead of by chance. A category with nothing in
    the slot's length band (pangrams at the 0.9 s slot) is skipped, not forced; within a
    category the pick is random and avoids the immediately previous sentence."""

    def __init__(self, rng=None):
        import random as _random
        self.rng = rng or _random
        self._order = list(SENTENCE_CATEGORIES)
        self._i = 0
        self._last = None

    def pick(self, frames):
        for _ in range(len(self._order)):
            cat = self._order[self._i % len(self._order)]
            self._i += 1
            band = _sentences_in_band(SENTENCE_CATEGORIES[cat], frames)
            if band:
                choices = [s for s in band if s != self._last] or band
                self._last = self.rng.choice(choices)
                return self._last
        # no category fits (cannot happen with the shipped pool; belt for a pruned one):
        # nearest length overall
        secs = hop_exact_samples(frames) / AUDIO_SAMPLE_RATE
        want = secs * WORDS_PER_SECOND
        self._last = min(HARVARD_SENTENCES, key=lambda s: abs(len(s.split()) - want))
        return self._last


_DEFAULT_PICKER = None


def pick_sentence(frames, rng=None):
    """Module-level convenience over one shared rotating picker (a private rng gets its own,
    so tests stay deterministic)."""
    global _DEFAULT_PICKER
    if rng is not None:
        return SentencePicker(rng).pick(frames)
    if _DEFAULT_PICKER is None:
        _DEFAULT_PICKER = SentencePicker()
    return _DEFAULT_PICKER.pick(frames)


# --- persisted settings (mic + default whisper language) ----------------------------------------
# Fizgig's folder (this file lives in src/gizmo); the settings live in its user_data folder
FIZGIG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_FILE = os.path.join(FIZGIG_DIR, "user_data", "gizmo_settings.json")
SETTINGS_DEFAULTS = {"microphone": "", "whisper_language": "Auto detect",
                     "random_delivery": True, "trim_takes": True, "free_speech": False,
                     "trigger_word": "", "clip_out_dir": "", "voice_out_dir": "",
                     "music_trigger_word": "", "music_words": "", "music_out_dir": "",
                     "music_fade_in": "0.1", "music_fade_out": "1.5",
                     "last_model": "h3"}


def load_settings():
    """user_data/gizmo_settings.json, or defaults. Corrupt/absent -> defaults, never
    a crash — losing a mic choice is an inconvenience, losing the app is not."""
    import json
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {**SETTINGS_DEFAULTS, **{k: v for k, v in data.items()
                                        if k in SETTINGS_DEFAULTS}}
    except Exception:
        return dict(SETTINGS_DEFAULTS)


def save_settings(settings):
    import json
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({k: settings.get(k, v) for k, v in SETTINGS_DEFAULTS.items()},
                      f, indent=2)
    except OSError:
        pass                                   # a read-only install keeps working unsaved


WHISPER_LANGUAGES = ["Auto detect", "English", "French", "German", "Spanish", "Italian",
                     "Portuguese", "Dutch", "Polish", "Welsh", "Russian", "Ukrainian",
                     "Japanese", "Chinese", "Korean", "Arabic", "Hindi"]

# Preferences' "Download models for me" warms the HuggingFace cache with exactly these
# patterns (fizgig.scripts.fetch_models, the hf-model: branch) — the two lists MUST stay
# identical, or the local check below misses a file the loader wants and goes online anyway.
_WHISPER_REPO = "openai/whisper-base"
_WHISPER_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def local_whisper_dir():
    """The cached Whisper snapshot, or None. Never touches the network.

    This is how Transcribe honours the Preferences downloader: once the fetcher (or a past
    online first-use) has the files, the pipeline is built from this local path and Gizmo
    transcribes fully offline — no version check, no reach for the hub at all."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(_WHISPER_REPO, local_files_only=True,
                                 allow_patterns=_WHISPER_PATTERNS)
    except Exception:
        return None


def voice_output_name(src_path, out_dir, claimed=()):
    """<source stem>_01.wav, skipping past files on disk and names the queue has spoken for.
    No mute variant: a muted voice file would be a training item with nothing in it."""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_") or "voice"
    for i in range(1, 1000):
        name = f"{stem}_{i:02d}.wav"
        if not os.path.exists(os.path.join(out_dir, name)) and name not in claimed:
            return os.path.join(out_dir, name)
    raise RuntimeError("1000 segments from one recording — give the output folder a clean start.")

# There is deliberately no size choice. Fizgig resizes clips down to the Target Megapixels on its
# Training tab — measured: a 1280x704 clip reaches the VAE as 672x384 at 0.25 MP and untouched at
# 1.0 — so cutting at native resolution keeps that decision open for every run afterwards, while
# cutting small forecloses it and can only be undone by re-cutting the set. Nothing is upscaled,
# so native is only ever the source's own detail.

# Crop shapes. H3 puts no constraint on aspect at all — its rotary position grid is normalised by
# sqrt(h*w), so it is aspect-agnostic by construction, and the only hard rule is that both sides
# are multiples of 32. These are here for CONSISTENCY, not legality: a dataset of thirty
# hand-drawn near-rectangles trains a model on thirty slightly different framings, and locking the
# drag to one shape costs nothing. Free stays the default.
CROP_SHAPES = (
    ("Free — drag any shape", None),
    ("Match the source", "source"),
    ("1:1 square", 1 / 1),
    ("16:9 wide", 16 / 9),
    ("9:16 vertical", 9 / 16),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("4:5 portrait", 4 / 5),
    ("21:9 ultrawide", 21 / 9),
)

# --- palette --------------------------------------------------------------------------------
# The Fizgig app's colours (its web look, src/fizgig/web/static/style.css), copied rather than imported: gizmo.py
# must never load the app.
COLORS = {
    "bg_deep": "#12161D", "bg_surface": "#1A2029", "bg_hover": "#252D3A", "bg_raised": "#212834",
    "bg_inset": "#10141A",
    "text_primary": "#F0F4F8", "text_secondary": "#93A3B5", "text_explain": "#C3CDD9",
    "text_muted": "#66778A",
    "accent": "#3B82F6", "accent_hover": "#60A5FA",
    "border": "#2A3341",
    "success": "#10B981", "warning": "#F59E0B", "error": "#EF4444",
}
FONT_FAMILY = "Segoe UI"

PREVIEW_W, PREVIEW_H = 512, 288
END_W, END_H = 320, 180

# Windows: every ffmpeg call would flash a console window, and under pythonw there is no console
# to inherit in the first place.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW, **kw)


def find_ffmpeg():
    """imageio-ffmpeg's bundled binary first — it is already pinned in requirements.txt, so
    Gizmo adds no dependency at all — then whatever is on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


# --- reading a source video -------------------------------------------------------------------

def probe_source(ffmpeg, path):
    """Facts about the file the user opened. ffmpeg with no output prints its stream banner to
    stderr and exits non-zero; that is the documented way to read this without ffprobe, which
    imageio-ffmpeg does not bundle."""
    out = _run([ffmpeg, "-hide_banner", "-i", path], text=True).stderr
    info = {"fps": None, "width": None, "height": None, "duration": None,
            "has_audio": False, "sample_rate": None, "channels": None, "vcodec": None,
            "sar": 1.0, "rotation": 0, "display_width": None, "display_height": None}

    m = re.search(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})", out, re.S)
    if m:
        info["vcodec"] = m.group(1)
        info["width"], info["height"] = int(m.group(2)), int(m.group(3))
    # Anamorphic sources store non-square pixels and carry the correction as a sample aspect
    # ratio — 1440x1080 with SAR 4:3 is a 1920x1080 picture. Ignore it and everything from such
    # a camera trains squashed, which looks like a bug in Gizmo rather than in the footage.
    m = re.search(r"SAR (\d+):(\d+)", out)
    if m and int(m.group(2)):
        info["sar"] = int(m.group(1)) / int(m.group(2))
    # Phones shot portrait for years by recording landscape frames and tagging them "rotate 90".
    # ffmpeg turns them upright on decode, so the frame that arrives is portrait while the banner
    # still reports the landscape dimensions it was STORED at. Take the banner at its word and
    # every such clip is scaled into a landscape box — squashed. Newer files store the pixels the
    # right way up, which is why only old ones misbehave.
    m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", out)
    if m:
        info["rotation"] = int(round(float(m.group(1)))) % 360
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", out)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\d+:\d+.*?: Audio: [^,]+, (\d+) Hz, (\w+)", out)
    if m:
        info["has_audio"] = True
        info["sample_rate"] = int(m.group(1))
        info["channels"] = {"mono": 1, "stereo": 2}.get(m.group(2))

    if not info["width"] or not info["fps"]:
        raise ValueError(f"{os.path.basename(path)} has no video stream Gizmo can read.")
    # The size the picture is MEANT to be seen at, after the sample aspect ratio and after any
    # rotation. Everything downstream — the preview, the target size, the crop, the export —
    # works from these, so ordinary upright square-pixel footage is unaffected and the awkward
    # cases come out the right shape instead of squashed.
    dw = max(2, int(round(info["width"] * info["sar"])))
    dh = info["height"]
    if info["rotation"] in (90, 270):
        dw, dh = dh, dw
    info["display_width"], info["display_height"] = dw, dh
    return info


def fit_within(src_w, src_h, box_w, box_h):
    """Largest size with the source's shape that fits inside the box. At least 2x2."""
    scale = min(box_w / src_w, box_h / src_h)
    return max(2, int(round(src_w * scale))), max(2, int(round(src_h * scale)))


def grab_frame(ffmpeg, path, seconds, box=(PREVIEW_W, PREVIEW_H), src_size=None):
    """One frame at `seconds`, scaled to fit `box`, as a PIL image.

    -ss BEFORE -i is the fast form, and has been frame-accurate since ffmpeg 2.1 (it seeks to the
    preceding keyframe and decodes forward). Scrubbing a long source with output-seek instead
    would decode everything up to the mark and take seconds per frame.

    The output size is COMPUTED and passed to ffmpeg, never inferred from how many bytes came
    back. Inferring it worked for 16:9 and quietly sheared everything else: raw RGB carries no
    dimensions, and picking the first width that divides the byte count lands on 512x216 for a
    4:3 source that is really 384x288 — same pixel count, wrong shape, a badly skewed picture.
    """
    if src_size is None:
        info = probe_source(ffmpeg, path)
        src_size = (info["display_width"], info["display_height"])
    w, h = fit_within(src_size[0], src_size[1], *box)
    p = _run([ffmpeg, "-hide_banner", "-loglevel", "error",
              "-ss", f"{max(0.0, seconds):.3f}", "-i", path,
              "-frames:v", "1", "-vf", f"scale={w}:{h},setsar=1",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    # Checked, not trusted: a mismatch means the filter did something other than what was asked,
    # and building an image from the wrong shape is precisely the bug this replaced.
    if not p.stdout or len(p.stdout) != w * h * 3:
        return None
    return Image.frombytes("RGB", (w, h), p.stdout)


# --- writing a clip ----------------------------------------------------------------------------

def snap(value):
    """Down to a multiple of 32, never to zero."""
    return max(SIZE_STEP, int(value) // SIZE_STEP * SIZE_STEP)


def target_size(src_w, src_h, megapixels):
    """Best /32 size at roughly `megapixels` that keeps the source's shape.

    Both sides have to be multiples of 32 for H3, and flooring each one independently is the
    obvious way to get there — but it BENDS THE PICTURE. 1920x1080 floored at 0.26 MP gives
    672x352, which is 1.909 against a true 1.778: a 7.4% horizontal stretch, and a face stretched
    7% trains as a face that is 7% too wide.

    So the pair is searched for instead of computed: every /32 height near the target, each with
    the /32 widths either side of the ideal, scored on aspect error first and area second. That
    gets most sizes under 2%, and build_export_command turns what is left into a crop rather than
    a stretch — losing a sliver of edge, which is honest, instead of reshaping everything in
    frame, which is not.

    Never larger than the source, on either axis. Upscaling would invent detail and then charge
    tokens for it, so a small source simply comes out at its own size.
    """
    aspect = src_w / src_h
    max_w, max_h = snap(src_w), snap(src_h)
    # Clamped to what the source actually has. Without this, asking a 480x360 clip for 1 MP
    # leaves every candidate hopelessly far from the target, the area term stops telling them
    # apart, and the search hands back 384x288 — throwing away a third of the pixels that were
    # there, to buy 2% of aspect. Asking for more than the source holds should mean "all of it".
    want_area = min(megapixels * 1_000_000, max_w * max_h)

    best = None
    for h in range(SIZE_STEP, max_h + SIZE_STEP, SIZE_STEP):
        ideal_w = h * aspect
        for w in {snap(ideal_w), snap(ideal_w) + SIZE_STEP}:
            if w < SIZE_STEP or w > max_w:
                continue
            err = abs((w / h) - aspect) / aspect
            # Both matter, so both are in one cost rather than one outranking the other. Aspect
            # alone would be a trap: 2560x1080 has exactly one /32 pair at its true 2.370, and
            # it is 2048x864 — so asking for a small clip would hand back a 1.8 MP one. Weighted
            # at 3x because the leftover aspect error is only ever a crop of that size, while
            # missing the requested megapixels costs detail that cannot come back.
            cost = abs(w * h - want_area) / want_area + 3.0 * err
            if best is None or cost < best[0]:
                best = (cost, (w, h))
    return best[1] if best else (max_w, max_h)


def tokens_for(width, height, frames):
    """What this clip costs the DiT: 32x32 pixel patches, times latent frames. Attention is
    quadratic in this number, which is the whole reason clips stop at 39 frames."""
    return (width // SIZE_STEP) * (height // SIZE_STEP) * LATENT_FRAMES[frames]


def build_export_command(ffmpeg, src, dst, start_s, frames, width, height,
                         keep_every=None, with_audio=True, crop=None, sar=1.0):
    """One pass: trim, retime, resize, and normalise the audio.

    Frame RATE is where a dataset goes quietly wrong, so it is worth being explicit about the two
    filters here:

      fps=24 preserves wall-clock time — it drops or duplicates so one real second stays one
      second. That keeps motion at true speed from any source (60 fps drops roughly every 2.5th
      frame, 25 fps drops 1 in 25) and it normalises the variable frame rate that most phone
      video has. It is the default and needs no decision from the user.

      select+setpts KEEPS every k-th source frame and re-times them to 24 fps, which plays back
      slower than life: 60 fps at every 2nd frame is 1.25x slow, 120 fps at every 2nd is 2.5x.
      This is the opposite of the intuition — it CREATES slow motion rather than avoiding it — so
      it is only ever offered as an explicit choice against the detected source rate.

    Audio is trimmed by atrim rather than left to -shortest, which decides on stream ends and can
    hand back a track a few samples long or short. Fizgig pads and trims the waveform to the frame
    count anyway, so this is belt and braces — but the belt is cheap.
    """
    if keep_every:
        vf = f"select='not(mod(n\\,{keep_every}))',setpts=N/({FPS}*TB)"
    else:
        vf = f"fps={FPS}"

    if crop:
        cx, cy, cw, ch = crop
        # A crop is expressed in DISPLAY pixels, because that is what the user drew on. For the
        # rare anamorphic source those are not the stored pixels, so the frame is normalised to
        # square pixels first and the crop lands where it was drawn.
        if abs(sar - 1.0) > 1e-3:
            vf += ",scale=iw*sar:ih:flags=lanczos,setsar=1"
        vf += f",crop={cw}:{ch}:{cx}:{cy}"
    # Cover, then crop — never stretch. force_original_aspect_ratio=increase scales the frame
    # until it covers the target box with its shape intact, and the crop takes the middle. The
    # target is chosen to sit within ~2% of the source aspect, so what the crop removes is a
    # sliver of edge. Plain scale=W:H instead would reshape everything in frame to make the
    # numbers fit, which is the one thing a training clip must not do.
    vf += (f",scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos"
           f",crop={width}:{height},setsar=1")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, start_s):.3f}", "-i", src,
           "-vf", vf, "-frames:v", str(frames),
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-r", str(FPS)]
    if with_audio:
        cmd += ["-af", f"aresample={AUDIO_SAMPLE_RATE},atrim=0:{frames / FPS:.6f},"
                       f"asetpts=PTS-STARTPTS",
                "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    return cmd + ["-movflags", "+faststart", dst]


def output_name(src_path, out_dir, muted, claimed=()):
    """<source stem>_01.mp4, or _01_mute.mp4, skipping past anything already there.

    `claimed` is names the queue has spoken for but not yet written — without it, two clips marked
    before either is exported would both be handed _01 and the second would overwrite the first.

    A source whose own name ends in _mute has that stripped first — otherwise every clip cut from
    it would read as muted to Fizgig regardless of what was chosen here.
    """
    stem = os.path.splitext(os.path.basename(src_path))[0]
    while stem.lower().endswith(MUTE_SUFFIX):
        stem = stem[:-len(MUTE_SUFFIX)]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_") or "clip"
    for i in range(1, 1000):
        # An index is free only if NEITHER spelling is taken. Checking just the muted one would
        # hand out _01 and _01_mute as two different clips from the same source, which reads as
        # one clip saved twice.
        taken = any(os.path.exists(os.path.join(out_dir, f"{stem}_{i:02d}{suffix}.mp4"))
                    or f"{stem}_{i:02d}{suffix}.mp4" in claimed
                    for suffix in ("", MUTE_SUFFIX))
        if not taken:
            return os.path.join(out_dir, f"{stem}_{i:02d}{MUTE_SUFFIX if muted else ''}.mp4")
    raise RuntimeError("1000 clips from one source — give the output folder a clean start.")


def enable_file_drop(root, on_drop):
    """Let a video be dragged onto the window. Returns True if it took.

    Through the Windows shell API rather than by adding tkinterdnd2: a new dependency would mean
    everyone re-running the installer for a convenience, and Gizmo's install story is that it
    needs nothing Fizgig does not already have.

    Tk has no drag-and-drop, so the window procedure is subclassed to catch WM_DROPFILES. THE
    RULE THAT MAKES THIS SAFE: the procedure must not touch Tk. Not one call.

    That rule was learned the hard way. The first version marshalled the path with root.after,
    which is the ordinary way to get work back onto the GUI thread — and it killed the
    interpreter outright on the first real drag:

        Fatal Python error: PyEval_RestoreThread: the function must be called with the GIL held
        ... the GIL is released (the current Python thread state is NULL)

    Tk's mainloop sits inside Tcl with the GIL released. A drop arrives, Windows dispatches into
    the window procedure, and root.after re-enters Tcl from underneath the loop already running
    it. The ctypes callback was never the problem — one that only touches plain Python is fine,
    which is what this does: append the path to a list, and let an ordinary Tk timer, running
    where Tk expects to be running, pick it up.

    None of this reproduces with a synthesised drop sent from inside root.update(), because then
    Python is already executing and Tcl is not mid-loop. It has to be POSTED and left to arrive
    on an idle mainloop, exactly as Explorer's does.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        root.update_idletasks()
        # wm_frame is the REAL top-level window. winfo_id is Tk's child frame, which never
        # receives WM_DROPFILES however willing it looks.
        hwnd = int(root.wm_frame(), 16)
        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        WM_DROPFILES, GWLP_WNDPROC = 0x0233, -4

        # Pointer-sized by hand: WPARAM and the handle SetWindowLongPtrW returns do not fit
        # ctypes' default c_int, and the resulting OverflowError is raised inside the window
        # procedure where nothing can catch it.
        _64 = ctypes.sizeof(ctypes.c_void_p) == 8
        LONG_PTR = ctypes.c_longlong if _64 else ctypes.c_long
        UINT_PTR = ctypes.c_ulonglong if _64 else ctypes.c_ulong
        WNDPROC = ctypes.WINFUNCTYPE(LONG_PTR, ctypes.c_void_p, ctypes.c_uint,
                                     UINT_PTR, LONG_PTR)
        for fn, args, res in (
                (user32.CallWindowProcW,
                 [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (user32.SetWindowLongPtrW,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_void_p),
                (user32.DefWindowProcW,
                 [ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (shell32.DragQueryFileW,
                 [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint], ctypes.c_uint),
                (shell32.DragFinish, [ctypes.c_void_p], None)):
            fn.argtypes, fn.restype = args, res

        shell32.DragAcceptFiles(ctypes.c_void_p(hwnd), True)
        inbox, old = [], [None]

        def proc(h, msg, wparam, lparam):
            # Plain Python only. No Tk, no Tcl, nothing that can raise if it can be helped — this
            # runs on every message the window receives.
            try:
                if msg == WM_DROPFILES:
                    try:
                        buf = ctypes.create_unicode_buffer(1024)
                        if shell32.DragQueryFileW(ctypes.c_void_p(wparam), 0, buf, 1024):
                            inbox.append(buf.value)
                    finally:
                        shell32.DragFinish(ctypes.c_void_p(wparam))
                    return 0
                return user32.CallWindowProcW(old[0], h, msg, wparam, lparam)
            except Exception:
                try:
                    return user32.DefWindowProcW(h, msg, wparam, lparam)
                except Exception:
                    return 0

        cb = WNDPROC(proc)
        old[0] = user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), GWLP_WNDPROC,
                                          ctypes.cast(cb, ctypes.c_void_p))
        if not old[0]:
            return False
        root._drop_callback = cb          # held, or it is collected and the window proc dangles

        def drain():
            while inbox:
                on_drop(inbox.pop(0))
            root.after(120, drain)

        root.after(120, drain)
        return True
    except Exception:
        return False


def audio_playback_backend():
    """How this machine can play a WAV, or None if it cannot.

    Windows has winsound in the standard library, which is the whole reason listening costs no
    new dependency. Linux — which is what a RunPod pod is — usually has neither a player nor an
    audio device in the container, so the feature disables itself with a reason rather than
    failing at the moment someone presses the button.
    """
    if os.name == "nt":
        try:
            import winsound  # noqa: F401
            return "winsound"
        except Exception:
            return None
    import shutil
    for exe in ("paplay", "aplay", "afplay"):
        if shutil.which(exe):
            return exe
    return None


_PLAYBACK = None            # resolved once, on first use


def _play_wav(path, loop=False):
    """loop: repeat seamlessly until stopped (winsound only; returns whether it loops)."""
    global _PLAYBACK
    if _PLAYBACK is None:
        _PLAYBACK = audio_playback_backend() or ""
    if not _PLAYBACK:
        raise RuntimeError("no way to play audio on this machine")
    if _PLAYBACK == "winsound":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | (winsound.SND_LOOP if loop else 0))
        return bool(loop)
    subprocess.Popen([_PLAYBACK, path], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return False


def _stop_wav():
    if _PLAYBACK == "winsound":
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)


def set_live_volume(pct):
    """Listening volume that takes effect MID-PLAY, no restart.

    winsound has no volume of its own, but it plays through the process's wave-out session, and
    winmm's waveOutSetVolume moves that session live — so the slider works while a preview is
    playing instead of demanding a stop-and-start. Per-process: the system volume and every
    other app are untouched, and nothing here ever reaches a saved file.

    Returns True when the live path took effect; False means the caller should fall back to
    baking the gain into the preview file (the non-Windows external-player path)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        v = max(0, min(0xFFFF, int(0xFFFF * float(pct) / 100.0)))
        return ctypes.windll.winmm.waveOutSetVolume(0, v | (v << 16)) == 0
    except Exception:
        return False


def count_frames(ffmpeg, path):
    """Decoded frame count. The container banner cannot be trusted for this (duration x fps
    rounds), and it is the one value Fizgig refuses a clip over, so it is checked for real."""
    out = _run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0",
                "-c", "copy", "-f", "null", "-"], text=True).stderr
    hits = re.findall(r"frame=\s*(\d+)", out)
    return int(hits[-1]) if hits else None


# --- auto-chop (scene detection) ----------------------------------------------------------------
# ffmpeg's scene filter scores every frame 0-1 on how different it looks from the one before;
# a score over the threshold is a cut. Segments are then planned BETWEEN the cuts, so no clip
# ever straddles one — a clip that crosses a cut trains the cut as if it were motion.

SCENE_SENSITIVITIES = (
    ("Normal — most footage", 0.30),
    ("More cuts — fast-cut or handheld footage", 0.22),
    ("Fewer cuts — slow or static footage", 0.40),
)
AUTOCHOP_MAX_SEGMENTS = 400          # a review grid past this stops being a review


def build_scene_scan_command(ffmpeg, src, threshold):
    """The detection pass. Frames are shrunk before scoring because the score only needs the
    broad strokes of the picture — scoring 4K frames buys nothing and slows the scan. showinfo
    prints each selected frame's pts_time to stderr, which is how the cut list comes back
    without ffprobe (imageio-ffmpeg does not bundle ffprobe). The comma in gt() is escaped
    because ffmpeg's filtergraph parser splits on commas without respecting parentheses."""
    vf = f"scale=320:-2:flags=bilinear,select=gt(scene\\,{threshold:g}),showinfo"
    return [ffmpeg, "-hide_banner", "-i", src, "-vf", vf, "-an", "-f", "null", "-"]


_SHOWINFO_TIME = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
_PROGRESS_TIME = re.compile(r"\btime=(\d+):(\d+):(\d+(?:\.\d+)?)")


def parse_scene_time(stderr_line):
    """The cut timestamp out of a showinfo line, or None for any other stderr line."""
    if "pts_time" not in stderr_line:
        return None
    m = _SHOWINFO_TIME.search(stderr_line)
    return float(m.group(1)) if m else None


def parse_progress_time(stderr_line):
    """How far through the file ffmpeg's periodic stats line says the scan is, or None."""
    m = _PROGRESS_TIME.search(stderr_line)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def plan_autochop(cut_times, duration, span_s, fill=True):
    """Clip start times from a cut list. Scenes are the spans between consecutive cuts (plus
    the start and end of the file); each scene yields clips of span_s seconds that stay inside
    it. fill=True packs a long scene with back-to-back clips; False takes one from the top of
    each scene. Scenes shorter than one clip are dropped rather than padded — micro-scenes
    from a fast dissolve disappear here on their own."""
    bounds = [0.0] + sorted(t for t in cut_times if 0.0 < t < duration) + [duration]
    starts = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a + 1e-6 < span_s:
            continue
        if fill:
            t = a
            while t + span_s <= b + 1e-6:
                starts.append(t)
                t += span_s
        else:
            starts.append(a)
    return starts


# --- the app -----------------------------------------------------------------------------------

class ToolTip:
    """Hover help. Same behaviour and look as Fizgig's, reimplemented rather than imported —
    gizmo.py must never load lora_trainer_gui.py.

    Delayed by half a second so sweeping the mouse across the frame-step buttons does not leave a
    trail of popups, and bound to <Button> as well as <Leave> because a tooltip left hanging over
    a button you just clicked hides the thing you clicked.
    """

    def __init__(self, widget, text, delay=500):
        self.widget, self.text, self.delay = widget, text, delay
        self.window = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self.window or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify=tk.LEFT, wraplength=380,
                 background=COLORS["bg_surface"], foreground=COLORS["text_primary"],
                 relief=tk.SOLID, borderwidth=1, font=(FONT_FAMILY, 9),
                 padx=8, pady=6).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.window:
            self.window.destroy()
            self.window = None


class Gizmo:
    def __init__(self, root):
        self.root = root
        root.title(f"Gizmo — prep for {MODEL['name']}")
        root.configure(bg=COLORS["bg_deep"])
        # Wide enough that the vertical scrollbar never eats into the content (which measures
        # ~985 px), tall enough for two cards at a time. The rest scrolls.
        root.geometry("%dx%d" % _scaled(root, 1060, 900))
        root.minsize(*_scaled(root, 1010, 620))

        self.ffmpeg = find_ffmpeg()
        self.settings = load_settings()
        self._recorder = None
        self.src = None
        self.info = None
        self.frame_pos = 0            # in SOURCE frames
        self.photo = None             # live references, or Tk garbage-collects the images
        self.end_photo = None
        self._scrub_job = None
        self._scrub_inflight = False  # one live grab at a time while the slider drags
        self._scrub_shown = -1        # the frame the live preview last painted
        self._busy = False
        self._motion_keep = [None]    # parallel to the Motion dropdown; see _keep_every
        self.queue = []               # marked clips, exported in one go — see add_to_queue
        self._chop_win = None         # the auto-chop review window, or None
        self._chop_proc = None        # the running scene-scan ffmpeg, for Cancel
        self._chop_alive = False      # workers check this and stop when the window closes
        self._chop_cands = []         # [{"start", "keep", "photo", ...}] in the review grid
        self.crop = None              # (x, y, w, h) in the source's DISPLAY pixels, or None
        self._crop_anchor = None
        self._radios = []             # every radio button, armed together once a video is open
        self._playing = False
        self._play_job = None
        self._video_sound = None      # (span_start_s, span_len_s, consumed_s) while paused
        self._video_offset = 0.0      # how far into the current span playback started
        self._video_t0 = 0.0
        # Video playback (the ▶ button): its own small state machine, parallel to the sound
        # button's — see video_toggle_play. The generation counter is how stale readers,
        # ticks and sound workers die: anything holding an old gen returns silently.
        self._vplay_gen = 0
        self._vplay_playing = False
        self._vplay_resume = None     # clip-time seconds to resume from after a pause
        self._vplay_frames = []       # decoded PIL frames, appended by the reader thread
        self._vplay_total = 0
        self._vplay_eof = False
        self._vplay_proc = None
        self._vplay_job = None
        self._vplay_t0 = None         # wall clock at first painted frame; None = not started
        self._vplay_offset = 0.0
        self._vplay_last_idx = -1
        self._vplay_sound_ready = True

        # Voice / audio mode state — parallel world, deliberately separate variables: a video
        # open on one tab and a recording on the other must never share a playhead.
        self.audio_src = None
        self.audio_dur = 0.0          # seconds
        self.audio_from_video = False
        self.audio_pos = 0.0          # segment start, seconds
        self.audio_peaks = None       # [(lo, hi) 0..1] per column of the CURRENT view, or None
        self._audio_samples = None    # 8 kHz mono f32 array — peaks re-slice from this on zoom
        self._audio_peak_scale = 1.0  # one normalisation for the whole file, so zoom keeps scale
        self.audio_view = None        # (start_s, dur_s) window, None = the whole file
        self._audio_peaks_key = None  # cache key so refresh doesn't re-slice unchanged views
        # The Voice and Music tabs share one editor (waveform, player, queue card); what
        # differs is the kind of item it cuts. Each kind keeps its own queue.
        self._audio_kind = "voice"
        self._audio_queues = {"voice": [], "music": []}
        self.music_len = MUSIC_MAX_S  # music section length, seconds (no grid)
        self._audio_playing = False
        self._audio_play_mode = None  # "file" (listen through) or "segment" (the export span)
        self._audio_play_from = 0.0   # where the current playback started, for the cursor
        self._audio_play_t0 = 0.0
        self._audio_play_rate = 1.0   # J/K/L shuttle speed
        self._audio_play_dir = 1      # 1 forward, -1 backward (J)
        self._audio_resume = None     # (mode, position) left by pause, for space to resume
        self._audio_autopan_job = None  # the move-drag edge-scroll timer, when armed
        self._audio_play_job = None
        self._audio_cursor_job = None
        self._whisper_busy = False

        self._style()
        self._build()
        # Three ways in besides the Open button, none of them a dependency: drop a video on the
        # window, hand Gizmo a path on the command line — which is what dropping one onto the
        # launcher does, and what Explorer's "Open with" does — or paste a path with Ctrl+V.
        self.can_drop = enable_file_drop(self.root, self.open_path)
        self.root.bind("<Control-v>", lambda _e: self.paste_path())
        # The recorder is an in-window card now, so closing the APP can be the only exit while
        # the mic is rolling — its ffmpeg must die with the window, or an orphan keeps
        # recording to temp forever.

        def _quit():
            rec = self._recorder
            if rec is not None and rec.proc is not None and rec.proc.poll() is None:
                try:
                    rec.proc.terminate()
                except Exception:
                    pass
            self.root.destroy()
        self.root.protocol("WM_DELETE_WINDOW", _quit)
        self.root.bind("<<GizmoQuit>>", lambda _e: _quit())
        if not self.ffmpeg:
            messagebox.showerror(
                "Gizmo — no ffmpeg",
                "Gizmo needs ffmpeg, which normally arrives with Fizgig's install.\n\n"
                "Run install_fizgig.bat to restore it, or put ffmpeg on your PATH.")

    def change_model(self):
        """Reopen Gizmo on the picker: the grid, tabs and sound rows are built for one model."""
        if (self.queue or any(self._audio_queues.values())) and not messagebox.askyesno(
                "Gizmo", "Changing the model reopens Gizmo, and the clips waiting in the queue are not saved. "
                         "Carry on?"):
            return
        self.settings["last_model"] = next(k for k, m in MODELS.items() if m is MODEL)
        save_settings(self.settings)
        subprocess.Popen([sys.executable, os.path.abspath(__file__)])
        self.root.event_generate("<<GizmoQuit>>")

    @property
    def audio_queue(self):
        return self._audio_queues[self._audio_kind]

    @audio_queue.setter
    def audio_queue(self, jobs):
        self._audio_queues[self._audio_kind] = jobs

    def _music_mode(self):
        return self._audio_kind == "music"

    # -- chrome ---------------------------------------------------------------------------------
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        # clam draws every widget with a light bevel (lightcolor / darkcolor) and a pale border: white outlines round
        # each tab, dropdown, slider and scrollbar on the dark cards. One set of quiet edges for everything.
        s.configure(".", bordercolor=COLORS["border"], lightcolor=COLORS["bg_hover"], darkcolor=COLORS["bg_hover"],
                    troughcolor=COLORS["bg_inset"], focuscolor=COLORS["accent"])
        s.configure("G.TCombobox", lightcolor=COLORS["bg_hover"], darkcolor=COLORS["bg_hover"])
        s.configure("G.TCombobox", fieldbackground=COLORS["bg_hover"],
                    background=COLORS["bg_hover"], foreground=COLORS["text_primary"],
                    arrowcolor=COLORS["text_secondary"], bordercolor=COLORS["border"],
                    font=(FONT_FAMILY, 10))
        # configure() alone is not enough and the failure is invisible until you look: a READONLY
        # combobox — which all of these are — draws its text as a selection, so it takes
        # selectforeground on selectbackground, and clam's stock values for those are a light
        # system pair. The result is white text on a white field.
        s.map("G.TCombobox",
              fieldbackground=[("disabled", COLORS["bg_surface"]),
                               ("readonly", COLORS["bg_hover"]),
                               ("!disabled", COLORS["bg_hover"])],
              foreground=[("disabled", COLORS["text_muted"]),
                          ("readonly", COLORS["text_primary"]),
                          ("!disabled", COLORS["text_primary"])],
              selectbackground=[("disabled", COLORS["bg_surface"]),
                                ("readonly", COLORS["bg_hover"]),
                                ("!disabled", COLORS["bg_hover"])],
              selectforeground=[("disabled", COLORS["text_muted"]),
                                ("readonly", COLORS["text_primary"]),
                                ("!disabled", COLORS["text_primary"])],
              arrowcolor=[("disabled", COLORS["text_muted"])],
              bordercolor=[("focus", COLORS["accent"])])

        # The list that drops DOWN is a plain Tk listbox built by Tcl, which ttk styling never
        # reaches. Without this it opens white-on-white too, and only when clicked.
        for option, value in (("background", COLORS["bg_surface"]),
                              ("foreground", COLORS["text_primary"]),
                              ("selectBackground", COLORS["accent"]),
                              ("selectForeground", COLORS["text_primary"])):
            self.root.option_add(f"*TCombobox*Listbox.{option}", value)

        s.configure("G.Horizontal.TScale", background=COLORS["bg_hover"], troughcolor=COLORS["bg_inset"],
                    bordercolor=COLORS["border"], lightcolor=COLORS["bg_hover"], darkcolor=COLORS["bg_hover"])
        # The window's own scrollbar, which clam otherwise draws in stock light grey against
        # every dark card.
        s.configure("Vertical.TScrollbar", background=COLORS["bg_hover"],
                    troughcolor=COLORS["bg_deep"], bordercolor=COLORS["bg_deep"],
                    arrowcolor=COLORS["text_secondary"])
        s.map("Vertical.TScrollbar", background=[("active", COLORS["accent"])])
        s.configure("Horizontal.TScrollbar", background=COLORS["bg_hover"],
                    troughcolor=COLORS["bg_deep"], bordercolor=COLORS["bg_deep"],
                    arrowcolor=COLORS["text_secondary"])
        s.map("Horizontal.TScrollbar", background=[("active", COLORS["accent"])])

        # The mode tabs. Clam's stock notebook is a light strip on our dark ground; restyle it
        # to read as part of the app rather than a window inside a window.
        s.configure("G.TNotebook", background=COLORS["bg_deep"], borderwidth=0, bordercolor=COLORS["bg_deep"],
                    lightcolor=COLORS["bg_deep"], darkcolor=COLORS["bg_deep"], tabmargins=(18, 0, 18, 0))
        s.configure("G.TNotebook.Tab", background=COLORS["bg_surface"],
                    foreground=COLORS["text_secondary"], font=(FONT_FAMILY, 11, "bold"),
                    padding=(18, 8), borderwidth=0, bordercolor=COLORS["border"],
                    lightcolor=COLORS["bg_surface"], darkcolor=COLORS["bg_surface"])
        s.map("G.TNotebook.Tab",
              background=[("selected", COLORS["accent"]), ("active", COLORS["bg_hover"])],
              foreground=[("selected", COLORS["text_primary"])],
              lightcolor=[("selected", COLORS["accent"])], darkcolor=[("selected", COLORS["accent"])],
              bordercolor=[("selected", COLORS["accent"])],
              expand=[("selected", (0, 0, 0, 0))],         # every tab the same size, the chosen one just lit
              padding=[("selected", (18, 8)), ("!selected", (18, 8))])

    def open_fizgig(self):
        """The Fizgig app's window (fizgig.py: the web UI in its own window, joining one already
        running). Its own process, so closing Gizmo leaves it open."""
        here = FIZGIG_DIR
        exe = sys.executable
        if os.name == "nt" and os.path.basename(exe).lower() == "python.exe":
            exe = os.path.join(os.path.dirname(exe), "pythonw.exe")   # no console window beside the app
        try:
            subprocess.Popen([exe, os.path.join(here, "fizgig.py")], cwd=here,
                             creationflags=_NO_WINDOW)
        except OSError as e:
            messagebox.showerror("Gizmo", f"Couldn't open Fizgig: {e}")

    def _card(self, parent, title, description=None):
        outer = tk.Frame(parent, bg=COLORS["bg_surface"], highlightthickness=1,
                         highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        outer.pack(fill=tk.X, padx=18, pady=(0, 14))
        inner = tk.Frame(outer, bg=COLORS["bg_surface"])
        inner.pack(fill=tk.BOTH, expand=True, padx=22, pady=18)
        tk.Label(inner, text=title, font=(FONT_FAMILY, 12, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(anchor="w")
        if description:
            tk.Label(inner, text=description, font=(FONT_FAMILY, 10), justify=tk.LEFT,
                     wraplength=900, bg=COLORS["bg_surface"],
                     fg=COLORS["text_secondary"]).pack(anchor="w", pady=(3, 8))
        body = tk.Frame(inner, bg=COLORS["bg_surface"])
        body.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        return body

    def _button(self, parent, text, command, kind="normal", tip=None, pad=14):
        # takefocus=0 plus a refocus after every click: a button that keeps keyboard focus
        # eats the space bar (Tk's Button class binding fires before any root handler can),
        # and the space bar's job in this tool is the transport, not re-pressing Open.
        # The refocus goes to the button's OWN toplevel, never self.root: focusing the root
        # from a button that lives on a popup ACTIVATES the main window, which puts the popup
        # behind it — the recorder's New sentence button "closing" the recorder was exactly
        # this.
        def _run_and_refocus():
            command()
            try:
                b.winfo_toplevel().focus_set()
            except tk.TclError:
                pass
        b = tk.Button(parent, text=text, command=_run_and_refocus,
                      bg={"normal": COLORS["bg_hover"], "accent": COLORS["accent"]}[kind],
                      fg=COLORS["text_primary"], takefocus=0,
                      activebackground=COLORS["accent_hover"],
                      activeforeground=COLORS["text_primary"],
                      font=(FONT_FAMILY, 10, "bold" if kind == "accent" else "normal"),
                      relief=tk.FLAT, bd=0, padx=pad, pady=7, cursor="hand2")
        if tip:
            ToolTip(b, tip)
        return b

    def _build(self):
        # Scrollable, because the whole app is four stacked cards and a 1080p laptop with a
        # taskbar has about 950 usable pixels. Without this the Save button is the thing that
        # falls off the bottom.
        shell = tk.Frame(self.root, bg=COLORS["bg_deep"])
        shell.pack(fill=tk.BOTH, expand=True)
        self._scroll_canvas = canvas = tk.Canvas(shell, bg=COLORS["bg_deep"],
                                                 highlightthickness=0)
        bar = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=canvas.yview)
        body = tk.Frame(canvas, bg=COLORS["bg_deep"])
        self._scroll_window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        body.bind("<Configure>",
                  lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(self._scroll_window, width=e.width))
        self.root.bind_all("<MouseWheel>",
                           lambda e: (canvas.yview_scroll(int(-e.delta / 120), "units")
                                      if e.widget.winfo_toplevel() is self.root else None))

        head = tk.Frame(body, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=18, pady=(16, 12))
        self._button(head, "⚙", self.open_settings, pad=10,
                     tip="Settings — microphone for the recorder, default Whisper language. "
                         "Saved beside gizmo.py, kept between sessions.").pack(side=tk.RIGHT)
        self._button(head, f"{MODEL['name']} ▾", self.change_model, pad=10,
                     tip="The model these clips are for: lengths, sizes and sound follow it. Click to "
                         "pick another (Gizmo reopens with it).").pack(side=tk.RIGHT, padx=(0, 8))
        self._button(head, "Open Fizgig", self.open_fizgig, kind="accent",
                     tip="Open the Fizgig app (it uses the one already running if there is one), to caption "
                         "and train on what you cut here.").pack(side=tk.RIGHT, padx=(0, 8))
        tk.Label(head, text="Gizmo", font=(FONT_FAMILY, 22, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor="w")
        tk.Label(head, text="Cut training clips, voice segments and song sections Fizgig will "
                            "accept — sized, with the sound sorted out, and voice and music captioned.",
                 font=(FONT_FAMILY, 11), bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                 wraplength=760, justify=tk.LEFT).pack(anchor="w")

        # Two modes, one window: video clips and voice segments share the queue-then-export
        # rhythm but nothing else, so each gets its own tab rather than a mode toggle whose
        # state has to be remembered.
        self.notebook = ttk.Notebook(body, style="G.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        video_tab = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        audio_tab = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        music_tab = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        self.notebook.add(video_tab, text="Video clips")
        self.notebook.add(audio_tab, text="Voice / audio")
        self.notebook.add(music_tab, text="Music (ACE-Step)")
        self._video_tab, self._audio_tab, self._music_tab = video_tab, audio_tab, music_tab
        self._model_tabs = {"video": video_tab, "voice": audio_tab, "music": music_tab}

        # Settings come BEFORE finding the moment, because the last-frame preview is only
        # meaningful once the length and the motion are settled — scrub first and it shows the
        # end of whatever length happened to be left over from the previous clip.
        tk.Frame(video_tab, bg=COLORS["bg_deep"], height=10).pack()
        self._build_source_card(video_tab)
        self._build_settings_card(video_tab)
        self._build_scrub_card(video_tab)
        self._build_queue_card(video_tab)

        # The voice and music tabs share ONE editor, a child of the notebook packed into
        # whichever of the two is showing (pack's in_ allows any descendant of its parent).
        for t in (audio_tab, music_tab):
            tk.Frame(t, bg=COLORS["bg_deep"], height=10).pack()
        self._audio_editor = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        self._audio_editor.pack(in_=audio_tab, fill=tk.BOTH, expand=True)
        self._build_audio_tab(self._audio_editor)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        # the model's own tabs only; selecting the first one shown moves the shared audio editor there
        for kind, tab in self._model_tabs.items():
            if kind not in MODEL["tabs"]:
                self.notebook.hide(tab)
        self.notebook.select(self._model_tabs[MODEL["tabs"][0]])
        self._on_tab_changed()

        self._bind_keys()
        self._set_enabled(False)
        self._refresh_cost()
        self._refresh_queue_box()

        # Nothing HOLDS keyboard focus in this tool. A combobox, listbox or slider that kept
        # it after a click would eat the transport keys (space/J/K/L, arrows) exactly the way
        # a focused button ate space — so each hands focus back to the window once used.
        # Class-wide: covers both tabs and anything added later.
        for cls, seq in (("TCombobox", "<<ComboboxSelected>>"),
                         ("Listbox", "<ButtonRelease-1>"),
                         ("TScale", "<ButtonRelease-1>")):
            self.root.bind_class(cls, seq, lambda _e: self.root.focus_set(), add="+")

        # The other half: a text field RELEASES focus when you click away from it. A canvas or
        # label never takes focus, so without this, typing a trigger word then clicking the
        # waveform left the entry focused — and space typed into it (well, was shielded into
        # doing nothing) instead of driving the transport.
        def _click_anywhere(e):
            try:
                # Main window only: bind_all reaches every toplevel, and yanking focus out of
                # the recorder popup would drop a push-to-record hold mid-take.
                if e.widget.winfo_toplevel() is not self.root:
                    return
                cls = e.widget.winfo_class()
            except Exception:
                return
            if cls not in ("Entry", "TEntry", "Text", "Listbox", "TCombobox"):
                self.root.focus_set()
        self.root.bind_all("<Button-1>", _click_anywhere, add="+")
        self.root.focus_set()

    def _build_source_card(self, body):
        c = self._card(body, "1. Source video", "Any format, any frame rate, any size. Opening "
                       "another video keeps your queue, so one export can hold clips from many.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self._button(row, "Open video…", self.open_video, "accent",
                     tip="Open the video you want to cut clips out of.\n\n"
                         "Anything ffmpeg can read: phone footage, a camera file, a download. "
                         "Gizmo never modifies it — it only reads.").pack(side=tk.LEFT)
        self.src_label = tk.Label(row, text="No video open", font=(FONT_FAMILY, 10),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.src_label.pack(side=tk.LEFT, padx=12)
        self.drop_hint = tk.Label(row, text="…or just drag a video onto this window",
                                  font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                                  fg=COLORS["text_muted"])
        self.drop_hint.pack(side=tk.RIGHT)
        ToolTip(self.drop_hint,
                "Drag a video from Explorer straight onto this window.\n\nGizmo also opens one "
                "dragged onto the Launch Gizmo shortcut, one sent with right-click and Open "
                "with, or a path pasted with Ctrl+V.")
        self.src_info = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.src_info.pack(anchor="w", pady=(8, 0))
        ToolTip(self.src_info,
                "What Gizmo read from the file. None of it has to be right — the whole point is "
                "that Gizmo converts it. It is here so you can see what you are starting from.")

    def _build_scrub_card(self, body):
        c = self._card(body, "3. Find the moment",
                       "The playhead is where the clip STARTS, and the last frame follows from "
                       "the length you set above. Drag it, step with the arrow keys (Shift for "
                       "ten at a time, Home for the beginning), or type an exact frame below.")
        shots = tk.Frame(c, bg=COLORS["bg_surface"])
        shots.pack()
        # Two frames, not one: choosing a clip means knowing what is in it, and the end frame is
        # the half you cannot see from the playhead. It costs one more ffmpeg seek per scrub,
        # which is why it is small and shares the same debounce.
        first = tk.Frame(shots, bg=COLORS["bg_surface"])
        first.pack(side=tk.LEFT)
        tk.Label(first, text="FIRST FRAME", font=(FONT_FAMILY, 8, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(anchor="w")
        self.canvas = tk.Canvas(first, width=PREVIEW_W, height=PREVIEW_H, bg="#000000",
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        ToolTip(self.canvas,
                "The first frame of the clip you are about to save.\n\n"
                "With Choose an area set above, drag here to pick the part of the frame to train "
                "on. Drag again to redraw it.")
        self.canvas.bind("<Button-1>", self._crop_press)
        self.canvas.bind("<B1-Motion>", self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_release)

        last = tk.Frame(shots, bg=COLORS["bg_surface"])
        last.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(last, text="LAST FRAME", font=(FONT_FAMILY, 8, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(anchor="w")
        self.end_canvas = tk.Canvas(last, width=END_W, height=END_H, bg="#000000",
                                    highlightthickness=1, highlightbackground=COLORS["border"])
        self.end_canvas.pack()
        ToolTip(self.end_canvas,
                "Where the clip ENDS, at the length you have chosen.\n\n"
                "Worth a glance before saving: it is how you catch a clip that runs past the "
                "moment you wanted, or into a cut.")

        self.scale = ttk.Scale(c, from_=0, to=100, orient=tk.HORIZONTAL,
                               command=self._on_scrub, style="G.Horizontal.TScale")
        self.scale.pack(fill=tk.X, pady=(10, 6))
        ToolTip(self.scale, "Drag to move through the source video. The clip starts here.")

        nav = tk.Frame(c, bg=COLORS["bg_surface"])
        nav.pack()
        self.vplay_btn = self._button(
            nav, "▶ Play", self.video_toggle_play, pad=10,
            tip="Watch the marked clip — exactly the frames and pace Save would export, slow "
                "motion included, with its sound when the source has a track.\n\nClick again "
                "to pause; a paused clip resumes where it stopped.")
        self.vplay_btn.pack(side=tk.LEFT, padx=(0, 10))
        self._button(nav, "⏮ Start", self.go_to_start, pad=10,
                     tip="Jump to the beginning of the source video.  (Home)").pack(
            side=tk.LEFT, padx=3)
        for text, delta, tip in (
                ("◀◀ 10", -10, "Back ten frames.  (Shift + ←)"),
                ("◀ 1", -1, "Back one frame.  (←)"),
                ("1 ▶", 1, "Forward one frame.  (→)"),
                ("10 ▶▶", 10, "Forward ten frames.  (Shift + →)")):
            self._button(nav, text, lambda d=delta: self.step(d), pad=10,
                         tip=tip).pack(side=tk.LEFT, padx=3)
        # Typed as well as dragged. A slider cannot land on frame 1483 of 9000, and "start it
        # exactly where the last clip ended" is an ordinary thing to want.
        posrow = tk.Frame(c, bg=COLORS["bg_surface"])
        posrow.pack(pady=(10, 0))
        tk.Label(posrow, text="Start at frame", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_secondary"]).pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = tk.Entry(posrow, textvariable=self.pos_var, width=8,
                                  justify="right", bg=COLORS["bg_hover"],
                                  fg=COLORS["text_primary"],
                                  insertbackground=COLORS["text_primary"], relief=tk.FLAT,
                                  font=("Consolas", 10))
        self.pos_entry.pack(side=tk.LEFT, padx=6, ipady=3)
        self._shield_from_hotkeys(self.pos_entry)
        self.pos_entry.bind("<Return>", lambda _e: self._go_to_typed())
        self.pos_entry.bind("<KP_Enter>", lambda _e: self._go_to_typed())
        self.pos_entry.bind("<FocusOut>", lambda _e: self._go_to_typed())
        POS_TIP = ("Type an exact frame and press Enter.\n\n"
                   "Add s for seconds instead — 12.5s. Out-of-range values are pulled back to "
                   "the nearest frame that exists rather than refused.")
        ToolTip(self.pos_entry, POS_TIP)
        self.pos_label = tk.Label(posrow, text="", font=(FONT_FAMILY, 10, "bold"),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_primary"])
        self.pos_label.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(self.pos_label, POS_TIP)
        self.span_label = tk.Label(c, text="", font=(FONT_FAMILY, 9),
                                   bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.span_label.pack()
        ToolTip(self.span_label,
                "How much of the SOURCE this clip uses. Slow motion uses more of it than real "
                "time does for the same number of output frames.")

    def _build_settings_card(self, body):
        c = self._card(body, "2. Clip settings",
                       "Set these first — the length decides where the clip ends, so the preview "
                       "on the next card depends on them.")
        grid = tk.Frame(c, bg=COLORS["bg_surface"])
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)

        def label(r, text, tip):
            lbl = tk.Label(grid, text=text, font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                           fg=COLORS["text_secondary"])
            lbl.grid(row=r, column=0, sticky="w", pady=5, padx=(0, 12))
            ToolTip(lbl, tip)
            return lbl

        LEN_TIP = ("How many frames the clip is.\n\n" + MODEL["grid_tip"] + " Beyond that the cost climbs "
                   "hard — attention scales with the square of the token count — so the line "
                   "below says what each length can be trained at on 16, 24 and 32 GB. Longer is "
                   "offered rather than withheld: whether it is affordable depends on the card "
                   "you have, and you know which that is.")
        label(0, "Length:", LEN_TIP)
        self.len_var = tk.StringVar()
        self.len_box = ttk.Combobox(grid, textvariable=self.len_var, state="readonly",
                                    style="G.TCombobox", width=46,
                                    values=[self._length_label(f) for f in GRID_FRAMES])
        self.len_box.current(GRID_FRAMES.index(MODEL["default"]))
        self.len_box.grid(row=0, column=1, sticky="w", pady=5)
        self.len_box.bind("<<ComboboxSelected>>", lambda _e: self._on_length_changed())
        ToolTip(self.len_box, LEN_TIP)
        self.len_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 wraplength=820, bg=COLORS["bg_surface"],
                                 fg=COLORS["text_explain"])
        self.len_note.grid(row=1, column=1, sticky="w")
        ToolTip(self.len_note, LEN_TIP)

        SIZE_TIP = ("The size the clip is saved at — whatever you cut, at its own resolution.\n\n"
                    "There is nothing to choose because there is no good reason to throw pixels "
                    "away here: Fizgig resizes clips down to the Target Megapixels on its "
                    "Training tab, so cutting large keeps that decision open, and cutting small "
                    "cannot be undone without re-cutting the set.\n\n"
                    f"Both sides land on a multiple of 32, which is what {MODEL['short']} needs, and the shape "
                    "is kept — nothing is ever scaled up.")
        label(2, "Size:", SIZE_TIP)
        self.size_label = tk.Label(grid, text="", font=(FONT_FAMILY, 10),
                                   bg=COLORS["bg_surface"], fg=COLORS["text_primary"])
        self.size_label.grid(row=2, column=1, sticky="w", pady=5)
        ToolTip(self.size_label, SIZE_TIP)

        MOTION_TIP = ("Real time is what you want almost always: whatever the source runs at, one "
                      "real second stays one second.\n\n"
                      "The slow-motion options keep MORE of the original frames instead of "
                      "resampling, so the action plays back slower than life. Only worth it when "
                      "the slow-motion look is the thing you are training.")
        label(4, "Motion:", MOTION_TIP)
        self.motion_var = tk.StringVar()
        self.motion_box = ttk.Combobox(grid, textvariable=self.motion_var, state="readonly",
                                       style="G.TCombobox", width=46, values=[])
        self.motion_box.grid(row=4, column=1, sticky="w", pady=5)
        self.motion_box.bind("<<ComboboxSelected>>", lambda _e: self._on_motion_changed())
        ToolTip(self.motion_box, MOTION_TIP)

        CROP_TIP = ("Train on part of the frame instead of all of it.\n\n"
                    "A clip's cost is its pixels, so a wide shot spends most of it on background. "
                    "Crop to the subject and every token goes on what you actually want learned — "
                    "and one source can give several clips, one per subject.\n\n"
                    "The crop holds still for the whole clip, so check the LAST frame preview: "
                    "the rectangle is drawn on both, and a subject that walks out of it is "
                    "obvious there and nowhere else.")
        label(5, "Crop:", CROP_TIP)
        croprow = tk.Frame(grid, bg=COLORS["bg_surface"])
        croprow.grid(row=5, column=1, sticky="w", pady=5)
        self.crop_var = tk.BooleanVar(value=False)
        for text, val, tip in (
                ("Whole frame", False, "Use everything in shot."),
                ("Choose an area", True,
                 "Then drag a rectangle on the FIRST FRAME preview below. It snaps to a "
                 "multiple of 32, and drag again to redraw it.")):
            rb = tk.Radiobutton(croprow, text=text, variable=self.crop_var, value=val,
                                command=self._on_crop_mode_changed,
                                bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                                selectcolor=COLORS["bg_deep"],
                                activebackground=COLORS["bg_surface"],
                                activeforeground=COLORS["text_primary"],
                                font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                cursor="hand2")
            rb.pack(side=tk.LEFT, padx=(0, 16))
            ToolTip(rb, tip)
            self._radios.append(rb)
        self.crop_clear_btn = self._button(croprow, "Reset", self.clear_crop, pad=10,
                                           tip="Go back to the whole frame.")
        self.crop_clear_btn.pack(side=tk.LEFT)

        SHAPE_TIP = ("Lock the crop to a shape while you drag it.\n\n"
                     f"{MODEL['short']} does not care — any shape trains, as long as both sides land on a "
                     "multiple of 32, which Gizmo handles. This is for consistency: thirty "
                     "hand-drawn rectangles are thirty slightly different framings, and one "
                     "shape across the set is usually what you meant.\n\n"
                     "The grid is coarse, so the lock gets as close as multiples of 32 allow "
                     "and tells you what it actually managed.")
        tk.Label(croprow, text="Shape", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.shape_var = tk.StringVar()
        self.shape_box = ttk.Combobox(croprow, textvariable=self.shape_var, state="readonly",
                                      style="G.TCombobox", width=22,
                                      values=[name for name, _ in CROP_SHAPES])
        self.shape_box.current(0)
        self.shape_box.pack(side=tk.LEFT)
        self.shape_box.bind("<<ComboboxSelected>>", lambda _e: self._on_shape_changed())
        ToolTip(self.shape_box, SHAPE_TIP)
        self.crop_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                  wraplength=820, bg=COLORS["bg_surface"],
                                  fg=COLORS["text_explain"])
        self.crop_note.grid(row=6, column=1, sticky="w")

        SOUND_TIP = ("H3 generates sound as well as video, so a clip's audio can be training data "
                     "too.\n\n"
                     "Mute the ones where it should not be — a cough, the wrong speaker, music "
                     "over the top. A muted clip trains its video exactly the same; only the "
                     "sound is ignored.")
        label(7, "Sound:", SOUND_TIP)
        snd = tk.Frame(grid, bg=COLORS["bg_surface"])
        snd.grid(row=7, column=1, sticky="w", pady=5)
        # Sticky between saves on purpose: one source video routinely has sections worth training
        # on and sections with a cough, the wrong speaker, or music over the top, so this gets
        # toggled far more often than it gets reset.
        self.mute_var = tk.BooleanVar(value=False)
        for text, val, tip in (
                ("Train on this clip's sound", False,
                 "The clip's audio becomes a training target — this is how H3 learns a voice."),
                ("Mute — video only", True,
                 "Saves with _mute in the filename. Fizgig trains the video and ignores the "
                 "sound.\n\nThe audio stays in the file, so you can still play it back — and "
                 "change your mind later by renaming, without re-exporting.")):
            rb = tk.Radiobutton(snd, text=text, variable=self.mute_var, value=val,
                                command=self._refresh_planned_name,
                                bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                                selectcolor=COLORS["bg_deep"],
                                activebackground=COLORS["bg_surface"],
                                activeforeground=COLORS["text_primary"], font=(FONT_FAMILY, 10),
                                highlightthickness=0, bd=0, cursor="hand2")
            rb.pack(side=tk.LEFT, padx=(0, 16))
            ToolTip(rb, tip)
            self._radios.append(rb)

        # Listen before deciding. Muting is a judgement about what the clip actually sounds like,
        # and there is no way to make that judgement from a picture of it.
        label(8, "Listen:", "Play the marked section so you can hear what you would be training "
                            "on before you choose.")
        listen = tk.Frame(grid, bg=COLORS["bg_surface"])
        listen.grid(row=8, column=1, sticky="w", pady=5)
        self.play_btn = self._button(
            listen, "▶  Play sound", self.toggle_play, pad=10,
            tip="Play the audio under the clip you have marked — exactly the section that would "
                "be saved.\n\nTakes a moment to extract the first time you press it.")
        self.play_btn.pack(side=tk.LEFT)
        tk.Label(listen, text="Volume", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(listen, from_=0, to=100, orient=tk.HORIZONTAL, length=150,
                                      variable=self.volume_var, style="G.Horizontal.TScale",
                                      command=lambda v: set_live_volume(v))
        self.volume_scale.pack(side=tk.LEFT)
        ToolTip(self.volume_scale,
                "Listening volume only — moves live while the sound plays. The saved clip's "
                "audio is never touched — a quiet source stays quiet, and H3 sees exactly what "
                "your camera recorded.")

        self.sound_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=820, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.sound_note.grid(row=9, column=1, sticky="w")
        if not MODEL["sound"]:
            for r in (7, 8, 9):
                for w in grid.grid_slaves(row=r):
                    w.grid_remove()

        OUT_TIP = ("Where the clips land.\n\n"
                   "Point it at your Fizgig training folder and the clips are ready to caption "
                   "the moment you are done here. Defaults to a fizgig_clips folder beside the "
                   "source video.")
        label(10, "Save to:", OUT_TIP)
        outrow = tk.Frame(grid, bg=COLORS["bg_surface"])
        outrow.grid(row=10, column=1, sticky="ew", pady=5)
        # Remembered across sessions; the beside-the-source default only fills it when
        # nothing was remembered.
        self.out_var = tk.StringVar(value=self.settings.get("clip_out_dir", ""))
        self.out_var.trace_add("write", lambda *_a: self._refresh_planned_name())
        self._persist_var(self.out_var, "clip_out_dir")
        out_entry = tk.Entry(outrow, textvariable=self.out_var, bg=COLORS["bg_hover"],
                             fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                             relief=tk.FLAT, font=(FONT_FAMILY, 10))
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self._shield_from_hotkeys(out_entry)
        ToolTip(out_entry, OUT_TIP)
        self._button(outrow, "Browse…", self.pick_output,
                     tip="Choose the folder to save clips into.").pack(side=tk.LEFT, padx=(8, 0))

        self.cost_label = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=900, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.cost_label.pack(anchor="w", pady=(10, 0))
        ToolTip(self.cost_label,
                "Training cost, so a length is never an accidental decision.\n\n"
                "Attention scales with the square of this number, so a clip is dramatically more "
                "expensive to train than a still — that is the whole reason clips are short.")

    def _build_queue_card(self, body):
        c = self._card(body, "4. Mark it, then move on",
                       "Marking is fast and encoding is slow, so they are separate. Add every "
                       "section you want, then open the next video and keep adding: the queue "
                       "keeps them all, and Export writes the lot in one go — keep scrubbing "
                       "while it runs. Gizmo doesn't caption video clips: export them into "
                       "your training folder, then caption them on Fizgig's Captions tab.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self.add_btn = self._button(
            row, "➕  Add to queue", self.add_to_queue, "accent",
            tip="Remember this clip — where it starts, how long, its size, its sound.  (Ctrl+S)"
                "\n\nNothing is written yet, so adding is instant and you can carry on marking.")
        self.add_btn.pack(side=tk.LEFT)
        self.chop_btn = self._button(
            row, "✂ Auto-chop…", self.open_autochop,
            tip="Find the cuts in the whole source and offer every segment between them as a "
                "thumbnail — untick the ones you don't want, and the rest join the queue.\n\n"
                "Each kept segment is queued exactly as if you had pressed Add there, with the "
                "settings as they are now.")
        self.chop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.export_btn = self._button(
            row, "Export queue", self.export_queue,
            tip="Encode everything in the queue, one after another.\n\nA second or two per clip. "
                "Anything that fails says why and the rest still finish.")
        self.export_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.Label(row, text="", font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                               fg=COLORS["text_secondary"])
        self.status.pack(side=tk.LEFT, padx=12)
        self.open_btn = self._button(row, "📂 Open folder", self.open_output_folder,
                                     tip="Open the save folder in your file browser.")
        self.open_btn.pack(side=tk.RIGHT)

        # The filename BEFORE anything is written, so the mute decision is visible as a
        # consequence rather than as a radio button whose meaning has to be remembered.
        self.name_label = tk.Label(c, text="", font=("Consolas", 9), justify=tk.LEFT,
                                   bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.name_label.pack(anchor="w", pady=(8, 0))
        ToolTip(self.name_label,
                "What Add would queue right now. The number steps past both the files already in "
                "the folder and anything else waiting in the queue.")

        self.queue_box = tk.Listbox(c, height=7, bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                    font=("Consolas", 9), relief=tk.FLAT, highlightthickness=1,
                                    highlightbackground=COLORS["border"],
                                    selectbackground=COLORS["accent"])
        self.queue_box.pack(fill=tk.X, pady=(10, 0))
        self.queue_box.bind("<Double-Button-1>", lambda _e: self.recall_selected())
        ToolTip(self.queue_box,
                "The queue. ✓ has been written, · is still waiting.\n\n"
                "Double-click a waiting row to put the playhead and settings back where they "
                "were when you added it. Gizmo keeps no project file — when you have what you "
                "need, close it.")

        qrow = tk.Frame(c, bg=COLORS["bg_surface"])
        qrow.pack(fill=tk.X, pady=(6, 0))
        self._button(qrow, "Remove selected", self.remove_selected, pad=10,
                     tip="Drop the selected row from the queue. Already-exported clips stay on "
                         "disk — delete those in your file browser.").pack(side=tk.LEFT)
        self._button(qrow, "Clear queue", self.clear_queue, pad=10,
                     tip="Empty the waiting list. Nothing already exported is "
                         "touched.").pack(side=tk.LEFT, padx=(8, 0))

    # -- state ----------------------------------------------------------------------------------
    def _length_label(self, frames):
        return f"{frames} frames — {frames / FPS:.2f} s"

    def _frames(self):
        return GRID_FRAMES[self.len_box.current()] if self.len_box.current() >= 0 else MODEL["default"]

    def _size(self):
        """The export size: the crop at its own resolution, or the whole frame at its own.

        There is no size CHOICE any more. Fizgig rescales clips down to its Target Megapixels at
        training time, so cutting at native keeps that decision open for every run afterwards,
        while cutting small forecloses it and can only be undone by re-cutting the whole set.
        Nothing is ever scaled up, so this is only ever the source's own detail.
        """
        if not self.info:
            return (None, None)
        if self.crop:
            return (self.crop[2], self.crop[3])
        # The /32 pair nearest the source's own shape and area — the aspect search with the
        # target clamped to what the source holds, which is exactly "as big as it goes".
        return target_size(self.info["display_width"], self.info["display_height"], 99.0)

    def _keep_every(self):
        """None for real time, else keep every k-th source frame. Read from a list held beside
        the dropdown rather than parsed back out of its label — the label is prose and 'keep
        every frame' has no number in it to parse."""
        i = self.motion_box.current()
        return self._motion_keep[i] if 0 <= i < len(self._motion_keep) else None

    def _set_enabled(self, on):
        state = tk.NORMAL if on else tk.DISABLED
        for w in (self.add_btn, self.chop_btn, self.export_btn, self.scale, self.open_btn,
                  self.pos_entry, self.crop_clear_btn, self.vplay_btn, *self._radios):
            w.configure(state=state)
        for box in (self.len_box, self.motion_box, self.shape_box):
            box.configure(state="readonly" if on else tk.DISABLED)
        # Listening needs a track to listen to AND a way to play it. Both reasons are spelled out
        # in _fill_sound_note rather than left as a greyed button with no explanation.
        can_play = bool(on and self.info and self.info["has_audio"]
                        and (audio_playback_backend() or ""))
        self.play_btn.configure(state=tk.NORMAL if can_play else tk.DISABLED)
        self.volume_scale.configure(state=tk.NORMAL if can_play else tk.DISABLED)

    # -- keyboard ---------------------------------------------------------------------------------
    def _bind_keys(self):
        """Frame-accurate marking with a mouse is miserable, and this is a tool you use thirty
        times in a row. Bound on the root so they work wherever focus happens to be — except
        inside the output-folder Entry, where a left arrow has to mean 'move the cursor'."""
        # Each key routes by the active tab: on Voice/audio the arrows nudge the playhead and
        # Ctrl+S marks a segment, on Video clips they keep their original meanings.
        for seq, fn, afn in (("<Left>", lambda: self.step(-1), lambda: self._audio_nudge(-0.05)),
                             ("<Right>", lambda: self.step(1), lambda: self._audio_nudge(0.05)),
                             ("<Shift-Left>", lambda: self.step(-10),
                              lambda: self._audio_nudge(-0.5)),
                             ("<Shift-Right>", lambda: self.step(10),
                              lambda: self._audio_nudge(0.5)),
                             ("<Home>", self.go_to_start, lambda: self._audio_seek(0.0)),
                             ("<Control-s>", self.add_to_queue, None)):
            def _route(_e, f=fn, af=afn):
                if self._audio_tab_active():
                    (af or self.audio_add_to_queue)()
                else:
                    f()
                return "break"
            self.root.bind(seq, _route)

        # Transport keys on both tabs: space = play/pause/resume, and the editing-suite J-K-L
        # shuttle. On Voice, J/L shuttle backward/forward (tap again for 1.5/2/3x); on Video —
        # which has no moving playhead to shuttle — J restarts the section's sound, L plays or
        # resumes it. The focus guard leaves a focused Button or Listbox its own space/keys;
        # text fields are already shielded by bindtags.
        def _video_l():
            if not self._playing:
                self._video_space()

        def _video_j():
            self._stop_audio()
            self._video_sound = None
            self._video_play_sound(0.0)

        for seq, afn, vfn in (("<space>", self._audio_space, self._video_space),
                              ("<j>", lambda: self._audio_shuttle(-1), _video_j),
                              ("<k>", self._audio_pause, self._video_pause),
                              ("<l>", lambda: self._audio_shuttle(1), _video_l)):
            def _troute(_e, af=afn, vf=vfn):
                # Text fields are shielded by bindtags and never reach here; this guard is the
                # belt-and-braces for a future unshielded one. Buttons no longer take focus at
                # all (see _button), so the transport keys always work.
                focus = self.root.focus_get()
                if focus is not None and focus.winfo_class() in ("Entry", "TEntry", "Text"):
                    return None
                (af if self._audio_tab_active() else vf)()
                return "break"
            self.root.bind(seq, _troute)

    def _audio_tab_active(self):
        try:
            return self.notebook.select() in (str(self._audio_tab), str(self._music_tab))
        except Exception:
            return False

    def _audio_host_tab(self):
        """The tab an opened recording belongs on: Music when it is showing, else Voice."""
        return self._music_tab if self._music_mode() else self._audio_tab

    def _on_tab_changed(self, _e=None):
        """Move the shared audio editor to the Voice or Music tab and switch what it cuts."""
        sel = self.notebook.select()
        if sel == str(self._music_tab):
            kind, host = "music", self._music_tab
        elif sel == str(self._audio_tab):
            kind, host = "voice", self._audio_tab
        else:
            return
        if kind == self._audio_kind:
            return
        self._audio_stop()
        self._audio_resume = None
        rec = getattr(self, "_recorder", None)
        if kind == "music" and rec is not None and rec.active:
            rec.close()
        self._audio_editor.pack_forget()
        self._audio_editor.pack(in_=host, fill=tk.BOTH, expand=True)
        self._audio_kind = kind
        music = kind == "music"
        # The parts that differ: the record button, the caption card, the save folder.
        if music:
            self._audio_record_btn.pack_forget()
            self._voice_caption_outer.pack_forget()
            self._music_caption_outer.pack(fill=tk.X, padx=16, pady=(0, 12),
                                           before=self._audio_wave_outer)
        else:
            self._audio_record_btn.pack(side=tk.LEFT, padx=(8, 0), after=self._audio_open_btn)
            self._music_caption_outer.pack_forget()
            self._voice_caption_outer.pack(fill=tk.X, padx=16, pady=(0, 12),
                                           before=self._audio_wave_outer)
        self._audio_source_desc.configure(text=self._MUSIC_SOURCE_DESC if music
                                          else self._VOICE_SOURCE_DESC)
        self.audio_out_var = self._music_out_var if music else self._voice_out_var
        self._audio_out_entry.configure(textvariable=self.audio_out_var)
        if music and self.audio_src:
            if not self.music_lyrics.get("1.0", tk.END).strip():
                lyrics = self._music_song_lyrics(self.audio_src)
                self.music_lyrics.insert("1.0", lyrics)
                self._music_lyrics_from_song = bool(lyrics)
            self._music_set_len(self.music_len)
            if not self.audio_out_var.get():
                self.audio_out_var.set(os.path.join(os.path.dirname(self.audio_src),
                                                    "fizgig_music"))
        self.audio_status.configure(text="")
        self._audio_refresh_queue_box()
        self._audio_refresh()

    @staticmethod
    def _shield_from_hotkeys(widget):
        """Stop root-level hotkeys reaching a text field.

        Tk delivers a key along the widget's bindtags — widget, class, toplevel, all — so a hotkey
        bound on the root fires even while someone is typing, and ← would scrub the video instead
        of moving the cursor. Dropping the toplevel tag is the fix; the class bindings that make
        an Entry an Entry are untouched.

        This is done by bindtags rather than by inspecting the event, because event.widget reports
        the TOPLEVEL for a key that arrived at a child — so the obvious check silently never
        matches, which is exactly the kind of guard that looks right and does nothing.
        """
        widget.bindtags((str(widget), widget.winfo_class(), "all"))

    def _persist_var(self, var, key):
        """Remember a field in gizmo_settings.json as it changes, debounced 400 ms so typing
        doesn't hammer the file. The save directories and the trigger word ride this — the
        things you set once and resent retyping every session (Peter)."""
        job = [None]

        def write(*_a):
            if job[0] is not None:
                self.root.after_cancel(job[0])

            def save():
                job[0] = None
                self.settings[key] = var.get().strip()
                save_settings(self.settings)
            job[0] = self.root.after(400, save)
        var.trace_add("write", write)

    def go_to_start(self):
        if self.src:
            self.scale.set(0)

    # Length and motion both move the END of the clip, so both have to redraw the last-frame
    # preview as well as the cost line.
    def _on_length_changed(self):
        self._vplay_stop()                     # the marked clip just changed shape
        self._refresh_cost()
        self.show_frame()

    def _on_motion_changed(self):
        self._vplay_stop()
        self._update_pos_labels()
        self.show_frame()

    # -- opening ---------------------------------------------------------------------------------
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open a source video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.wmv"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_video(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that file", str(exc))

    def open_path(self, path):
        """Open a file that arrived from somewhere other than the Open button — a drop, the
        command line, or the clipboard. Refuses politely rather than by traceback."""
        path = str(path).strip().strip('"')
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Gizmo", f"Not a file:\n{path}")
            return
        if os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            messagebox.showinfo(
                "Gizmo — that is a still",
                "Gizmo cuts clips out of video. Still images go straight into your training "
                "folder and are prepared on Fizgig's Image Prep tab.")
            return
        # An audio file goes to the Voice tab wherever you are. A VIDEO follows the tab you are
        # on: dropped while the Voice tab is up, it is an audio source (you asked for its sound
        # by being here) — anywhere else it opens as footage, as before.
        if is_audio_file(path) and not set(MODEL["tabs"]) & {"voice", "music"}:
            messagebox.showinfo("Gizmo", f"{MODEL['name']} trains on video, not sound alone - open a video, "
                                         f"or pick another model at the top of the window.")
            return
        if is_audio_file(path) or self._audio_tab_active() or "video" not in MODEL["tabs"]:
            self.notebook.select(self._audio_host_tab())
            try:
                self.load_audio(path)
            except Exception as exc:
                messagebox.showerror("Gizmo — cannot read that audio", str(exc))
            return
        try:
            self.load_video(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that file", str(exc))

    def paste_path(self):
        try:
            self.open_path(self.root.clipboard_get())
        except tk.TclError:
            pass                              # empty or non-text clipboard

    def load_video(self, path):
        """Everything opening a video does, with no dialog in the way — so it can be tested."""
        info = probe_source(self.ffmpeg, path)
        self._stop_audio()
        self._vplay_stop()
        # An open auto-chop review belongs to the video it scanned, and this is a new one.
        self._autochop_close()
        # A crop belongs to the frame it was drawn on, and a new video is a new frame.
        self.crop = None
        self.crop_var.set(False)
        self.src, self.info = path, info
        self.root.title(f"Gizmo — {os.path.basename(path)}")
        self.frame_pos = 0
        self._scrub_shown = -1        # the live preview's memory belongs to the old video
        total = int((info["duration"] or 0) * info["fps"])
        self.scale.configure(from_=0, to=max(1, total - 1))
        self.scale.set(0)
        self.src_label.configure(text=os.path.basename(path), fg=COLORS["text_primary"])

        if info["has_audio"]:
            layout = {1: "mono", 2: "stereo"}.get(info["channels"], f"{info['channels']} ch")
            audio = f"{info['sample_rate']} Hz {layout}"
        else:
            audio = "no audio track"
        # The size as it will be SEEN. A rotated or anamorphic file is stored at something else,
        # and reporting the stored numbers here would contradict the picture right beside them.
        facts = (f"{info['display_width']}x{info['display_height']}  ·  {info['fps']:g} fps  ·  "
                 f"{info['duration'] or 0:.1f} s  ·  {audio}")
        if info["rotation"]:
            facts += f"  ·  rotated {info['rotation']}° (stored {info['width']}x{info['height']})"
        # A source slower than 24 fps cannot be resampled up honestly: ffmpeg duplicates frames
        # to reach 24, and the model learns that nothing moves between them. Worth saying out
        # loud rather than silently producing it — animation at 12 fps is the usual case, and it
        # is a legitimate thing to train on as long as the person knows what they are getting.
        if info["fps"] < FPS - 0.1:
            self.src_info.configure(
                text=facts + f"\n⚠ Under {FPS} fps: reaching {FPS} means duplicating frames, so "
                             f"some of what {MODEL['short']} learns about motion will be 'nothing happened'. "
                             f"Fine for animation you want that way; not what you want from "
                             f"under-cranked live footage.",
                fg=COLORS["warning"])
        else:
            self.src_info.configure(text=facts, fg=COLORS["text_explain"])

        if not self.out_var.get():
            self.out_var.set(os.path.join(os.path.dirname(path), "fizgig_clips"))
        self._fill_motion()
        self._fill_sound_note()
        self._set_enabled(True)
        self.show_frame()
        self._refresh_cost()
        self._refresh_planned_name()

    def _fill_motion(self):
        fps = self.info["fps"]
        opts = [f"Real time — resample {fps:g} fps to {FPS}"]
        self._motion_keep = [None]
        # Only offered where it would actually be visible: 60 fps gives 1.25x and 2.5x, 50 fps
        # gives 2.08x, and 30 fps gives nothing worth a dropdown entry.
        for k in (2, 1):
            factor = fps / k / FPS
            if factor > 1.05:
                every = "every frame" if k == 1 else f"every {k}nd frame" if k == 2 \
                    else f"every {k}th frame"
                opts.append(f"{factor:.2f}x slow motion — keep {every}")
                self._motion_keep.append(k)
        self.motion_box.configure(values=opts)
        self.motion_box.current(0)

    def _fill_sound_note(self):
        info = self.info
        if not info["has_audio"]:
            self.mute_var.set(True)
            self.sound_note.configure(
                text="This source has no audio track, so there is nothing to train on either way.",
                fg=COLORS["text_muted"])
            return
        note = ("Muting adds _mute to the filename and Fizgig trains that clip's video only. The "
                "audio stays in the file, so you can still play it back and change your mind by "
                "renaming.")
        if not (audio_playback_backend() or ""):
            # Almost always a pod: the container has no audio device and no player. Worth naming,
            # because a greyed Play button with no reason reads as a bug.
            note += ("\nPlayback is not available on this machine, so the Play button is off — "
                     "everything else works normally.")
        self.sound_note.configure(text=note, fg=COLORS["text_explain"])

    # -- scrubbing --------------------------------------------------------------------------------
    def _on_scrub(self, value):
        if not self.src:
            return
        if self._vplay_playing or self._vplay_resume is not None:
            self._vplay_stop()                 # a moved playhead outdates play AND its resume
        self.frame_pos = int(float(value))
        self._update_pos_labels()
        # Live while dragging, not on release: a debounce alone kept cancelling itself for the
        # whole drag, so the picture froze until the thumb stopped (Peter). Instead, at most one
        # grab is ever in flight, and the moment it lands the preview re-grabs if the playhead
        # has moved on — so the picture follows the drag at whatever rate this machine decodes,
        # never queueing a backlog of ffmpeg calls. Only the FIRST frame is chased (half the
        # cost per update); the trailing debounce below still renders the resting position in
        # full, first and last frame both.
        self._scrub_chase()
        if self._scrub_job:
            self.root.after_cancel(self._scrub_job)
        self._scrub_job = self.root.after(120, self.show_frame)

    def _scrub_chase(self):
        if self._scrub_inflight or self._scrub_shown == self.frame_pos:
            return
        self._scrub_inflight = True
        self._scrub_shown = self.frame_pos
        threading.Thread(target=self._live_grab_worker,
                         args=(self.frame_pos / self.info["fps"],), daemon=True).start()

    def _live_grab_worker(self, start_s):
        size = (self.info["display_width"], self.info["display_height"])
        try:
            img = grab_frame(self.ffmpeg, self.src, start_s, src_size=size)
        except Exception:
            img = None
        self.root.after(0, self._live_grab_done, img)

    def _live_grab_done(self, img):
        self._scrub_inflight = False
        if not self.src:
            return
        if img is not None:
            self.photo = self._paint(self.canvas, img, PREVIEW_W, PREVIEW_H)
            self._draw_crop_overlay()
        self._scrub_chase()               # the drag moved on while that frame decoded

    def step(self, delta):
        if not self.src:
            return
        self.scale.set(max(0, self.frame_pos + delta))

    def _last_frame(self):
        """The last frame that exists in the source, as an index."""
        return max(0, int((self.info["duration"] or 0) * self.info["fps"]) - 1)

    def _go_to_typed(self):
        """Jump to a typed frame — or a typed time, with an s on the end.

        Clamped rather than refused: someone typing 9999 into a 600-frame video means 'the end',
        and an error dialog for that is a worse answer than going there. A value that is not a
        number at all just snaps back to where the playhead already is.
        """
        if not self.src:
            return
        raw = self.pos_var.get().strip().lower().replace(",", "")
        try:
            if raw.endswith("s"):
                frame = int(round(float(raw[:-1]) * self.info["fps"]))
            else:
                frame = int(round(float(raw)))
        except ValueError:
            self._sync_pos_entry()
            return
        frame = max(0, min(frame, self._last_frame()))
        if frame != self.frame_pos:
            self.scale.set(frame)
        self._sync_pos_entry()

    def _sync_pos_entry(self):
        """Keep the box showing the truth — it is an input AND a readout, and after a drag the
        two disagree unless this runs."""
        if self.pos_var.get() != str(self.frame_pos):
            self.pos_var.set(str(self.frame_pos))

    def _update_pos_labels(self):
        fps = self.info["fps"]
        t = self.frame_pos / fps
        self._sync_pos_entry()
        self.pos_label.configure(
            text=f"of {self._last_frame()}   ·   {t:6.2f} s")
        frames = self._frames()
        k = self._keep_every()
        # How much SOURCE this clip eats depends on the motion choice: real time takes the same
        # wall-clock span, slow motion takes k source frames per output frame.
        span = frames * k / fps if k else frames / FPS
        end, duration = t + span, self.info["duration"] or 0
        text = f"clip covers {t:.2f} s → {end:.2f} s of the source"
        # Caught here as well as after the export, because a run-out is easy to walk into by
        # scrubbing near the end and the export only discovers it after encoding.
        if duration and end > duration:
            self.span_label.configure(
                text=text + f" — but the source ends at {duration:.2f} s. Move earlier, or "
                            f"choose a shorter length.",
                fg=COLORS["warning"])
        else:
            self.span_label.configure(text=text, fg=COLORS["text_explain"])

    def show_frame(self):
        self._scrub_job = None
        if not self.src:
            return
        fps = self.info["fps"]
        start = self.frame_pos / fps
        k = self._keep_every()
        span = self._frames() * k / fps if k else self._frames() / FPS
        # The last frame of the clip, not the frame after it — hence the one-frame step back.
        threading.Thread(target=self._grab_worker,
                         args=(start, max(start, start + span - 1 / fps)),
                         daemon=True).start()

    def _grab_worker(self, start_s, end_s):
        size = (self.info["display_width"], self.info["display_height"])
        try:
            first = grab_frame(self.ffmpeg, self.src, start_s, src_size=size)
        except Exception:
            first = None
        try:
            last = grab_frame(self.ffmpeg, self.src, end_s, box=(END_W, END_H), src_size=size)
        except Exception:
            last = None
        self.root.after(0, self._show_images, first, last)

    def _show_images(self, first, last):
        self.photo = self._paint(self.canvas, first, PREVIEW_W, PREVIEW_H)
        self.end_photo = self._paint(self.end_canvas, last, END_W, END_H)
        self._draw_crop_overlay()

    def _paint(self, canvas, img, w, h):
        """Draw into a canvas and hand back the PhotoImage — which the caller MUST keep, or Tk
        garbage-collects it and the canvas goes blank."""
        canvas.delete("all")
        if img is None:
            canvas.create_text(w // 2, h // 2, text="(no frame here)",
                               fill=COLORS["text_muted"], font=(FONT_FAMILY, 10))
            canvas._shown = None
            return None
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(w // 2, h // 2, image=photo)
        # Remembered so canvas coordinates can be turned back into source ones: the image is
        # centred, so it sits at an offset that depends on its shape.
        canvas._shown = (img.size[0], img.size[1],
                         (w - img.size[0]) // 2, (h - img.size[1]) // 2)
        return photo

    # -- cropping ---------------------------------------------------------------------------------
    def _canvas_to_source(self, canvas, cx, cy):
        """A point on a preview canvas -> a point in the source's display pixels."""
        shown = getattr(canvas, "_shown", None)
        if not shown or not self.info:
            return None
        iw, ih, ox, oy = shown
        sx = (cx - ox) / iw * self.info["display_width"]
        sy = (cy - oy) / ih * self.info["display_height"]
        return sx, sy

    def _source_to_canvas(self, canvas, sx, sy):
        shown = getattr(canvas, "_shown", None)
        if not shown or not self.info:
            return None
        iw, ih, ox, oy = shown
        return (ox + sx / self.info["display_width"] * iw,
                oy + sy / self.info["display_height"] * ih)

    def _crop_press(self, event):
        if not self.src or not self.crop_var.get():
            return
        pt = self._canvas_to_source(self.canvas, event.x, event.y)
        if pt:
            self._crop_anchor = pt

    def _crop_drag(self, event):
        if not getattr(self, "_crop_anchor", None):
            return
        pt = self._canvas_to_source(self.canvas, event.x, event.y)
        if not pt:
            return
        self.crop = self._snap_crop(self._crop_anchor, pt)
        self._draw_crop_overlay()

    def _crop_release(self, _event):
        if not getattr(self, "_crop_anchor", None):
            return
        self._crop_anchor = None
        if self.crop:
            # The crop decides the shape AND how many pixels there are to work with, so the size
            # menu is rebuilt from it — keeping whichever tier was chosen rather than the pixel
            # dimensions, which have just changed under it.
            self._refresh_cost()
            self.show_frame()
        self._describe_crop()

    def _crop_aspect(self):
        """The locked aspect, or None for free-form."""
        i = self.shape_box.current()
        value = CROP_SHAPES[i][1] if 0 <= i < len(CROP_SHAPES) else None
        if value == "source" and self.info:
            return self.info["display_width"] / self.info["display_height"]
        return value if isinstance(value, float) else None

    def _snap_crop(self, a, b):
        """Two dragged corners -> an (x, y, w, h) box on the /32 grid, inside the frame.

        Snapped because H3 needs multiples of 32 and because a crop that is 31 pixels off the
        grid would otherwise be quietly rounded later, moving the framing away from what was
        drawn. Anything smaller than 32 in either direction is not a crop, it is a slip.

        With a shape locked, the /32 pair nearest that ratio is searched for rather than computed:
        the grid is coarse enough that dividing and rounding lands surprisingly far off — 16:9 at
        a height of 352 rounds to 640x352, which is 1.82. Both sides move, so the result is the
        closest the grid can actually get.
        """
        dw, dh = self.info["display_width"], self.info["display_height"]
        max_w = int(dw // SIZE_STEP) * SIZE_STEP
        max_h = int(dh // SIZE_STEP) * SIZE_STEP
        x0, x1 = sorted((max(0.0, min(a[0], dw)), max(0.0, min(b[0], dw))))
        y0, y1 = sorted((max(0.0, min(a[1], dh)), max(0.0, min(b[1], dh))))
        x = int(x0 // SIZE_STEP) * SIZE_STEP
        y = int(y0 // SIZE_STEP) * SIZE_STEP
        w = max(SIZE_STEP, int(round((x1 - x) / SIZE_STEP)) * SIZE_STEP)
        h = max(SIZE_STEP, int(round((y1 - y) / SIZE_STEP)) * SIZE_STEP)

        aspect = self._crop_aspect()
        if aspect:
            # Fitted INSIDE what was dragged, so the rectangle never grows past the gesture, then
            # scored on how close the grid can get. Ties go to the larger box.
            span_w = min(w, max_w - x)
            span_h = min(h, max_h - y)
            best = None
            for cw in range(SIZE_STEP, span_w + SIZE_STEP, SIZE_STEP):
                for ch in {int(round(cw / aspect / SIZE_STEP)) * SIZE_STEP,
                           int(round(cw / aspect / SIZE_STEP)) * SIZE_STEP + SIZE_STEP}:
                    if ch < SIZE_STEP or ch > span_h:
                        continue
                    score = (round(abs((cw / ch) - aspect) / aspect, 4), -(cw * ch))
                    if best is None or score < best[0]:
                        best = (score, (cw, ch))
            if best is None:
                return None
            w, h = best[1]

        w = min(w, max_w - x)
        h = min(h, max_h - y)
        if w < SIZE_STEP or h < SIZE_STEP:
            return None
        return (x, y, w, h)

    def _draw_crop_overlay(self):
        """Outline the crop on BOTH previews. The last frame is the point: a crop holds still and
        a subject does not, so whether they are still inside at the end is only visible there."""
        for canvas in (self.canvas, self.end_canvas):
            canvas.delete("crop")
            if not self.crop or not getattr(canvas, "_shown", None):
                continue
            x, y, w, h = self.crop
            p0 = self._source_to_canvas(canvas, x, y)
            p1 = self._source_to_canvas(canvas, x + w, y + h)
            if not p0 or not p1:
                continue
            iw, ih, ox, oy = canvas._shown
            # Dim what is being thrown away rather than only ringing what is kept — the discarded
            # part is the thing worth seeing.
            for box in ((ox, oy, ox + iw, p0[1]), (ox, p1[1], ox + iw, oy + ih),
                        (ox, p0[1], p0[0], p1[1]), (p1[0], p0[1], ox + iw, p1[1])):
                canvas.create_rectangle(*box, fill="#000000", outline="", stipple="gray50",
                                        tags="crop")
            canvas.create_rectangle(*p0, *p1, outline=COLORS["accent"], width=2, tags="crop")

    def _on_shape_changed(self):
        """Re-fit an existing crop to the new shape, around the same centre — otherwise picking a
        shape does nothing visible until the next drag, which reads as the control being broken."""
        if not self.src:
            return
        if self.crop:
            x, y, w, h = self.crop
            cx, cy = x + w / 2, y + h / 2
            refit = self._snap_crop((cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2))
            if refit:
                # Re-centre: the snap anchors at the top-left corner, so a shape change would
                # otherwise slide the framing down and right.
                nx, ny, nw, nh = refit
                nx = max(0, min(int((cx - nw / 2) // SIZE_STEP) * SIZE_STEP,
                                int(self.info["display_width"] // SIZE_STEP) * SIZE_STEP - nw))
                ny = max(0, min(int((cy - nh / 2) // SIZE_STEP) * SIZE_STEP,
                                int(self.info["display_height"] // SIZE_STEP) * SIZE_STEP - nh))
                self.crop = (nx, ny, nw, nh)
            self._refresh_cost()
            self._draw_crop_overlay()
            self.show_frame()
        self._describe_crop()

    def _on_crop_mode_changed(self):
        if not self.crop_var.get():
            self.clear_crop()
            return
        self._describe_crop()

    def clear_crop(self):
        self.crop = None
        self.crop_var.set(False)
        self._crop_anchor = None
        if self.info:
            self._refresh_cost()
        self._draw_crop_overlay()
        self._describe_crop()
        self._refresh_planned_name()

    def _describe_crop(self):
        if not self.crop_var.get():
            self.crop_note.configure(text="", fg=COLORS["text_explain"])
            return
        if not self.crop:
            self.crop_note.configure(
                text="Drag a rectangle on the FIRST FRAME preview below.",
                fg=COLORS["accent"])
            return
        x, y, w, h = self.crop
        share = (w * h) / (self.info["display_width"] * self.info["display_height"]) * 100
        text = f"Cropping to {w} x {h} at ({x}, {y}) — {share:.0f}% of the frame."
        aspect = self._crop_aspect()
        if aspect:
            # What the lock actually managed, not what was asked for. The /32 grid cannot hit
            # every ratio, and quietly reporting the requested one would be a small lie.
            got = w / h
            text += (f" Shape {got:.3f}"
                     + ("" if abs(got - aspect) / aspect < 0.005
                        else f" — the closest to {aspect:.3f} the 32-pixel grid allows"))
        tw, th = self._size()
        # Cropped below the output size means the clip is upscaled-in-effect: there is simply
        # less real detail behind the same number of pixels. Worth saying, not worth refusing.
        if tw and (w < tw or h < th):
            self.crop_note.configure(
                text=text + f" That is smaller than the {tw} x {th} output, so the clip will be "
                            f"softer than the source — crop wider, or pick a smaller size.",
                fg=COLORS["warning"])
        else:
            self.crop_note.configure(text=text, fg=COLORS["text_explain"])

    # -- listening --------------------------------------------------------------------------------
    def toggle_play(self):
        """Play the marked section's audio, or stop it if it is already playing.

        The volume slider is applied by ffmpeg while extracting, because the only playback that
        needs no new dependency is winsound, and winsound has no volume of its own. That also
        keeps the promise the tooltip makes: the gain exists in a temp file for listening and
        never reaches the saved clip.
        """
        self._vplay_stop()                     # full playback yields to the sound-only button
        if self._playing:
            self._stop_audio()
            self._video_sound = None
            return
        self._video_sound = None
        self._video_play_sound(0.0)

    def _video_play_sound(self, offset):
        """Play the marked section's audio from `offset` seconds into it — 0 for the button,
        a stored consumed-time for space's resume."""
        if not self.src or not self.info or not self.info["has_audio"]:
            return
        fps = self.info["fps"]
        start = self.frame_pos / fps
        k = self._keep_every()
        # Real time: the audio you would get. Slow motion: the source audio under the section,
        # which is what you are judging — the saved clip's audio comes from the same span.
        span = self._frames() * k / fps if k else self._frames() / FPS
        if offset >= span - 0.05:
            offset = 0.0                       # a stale resume past the end starts over
        # Live volume when the backend allows it — the slider then works MID-play; otherwise
        # the gain is baked into the preview file as before.
        gain = 1.0 if set_live_volume(self.volume_var.get()) \
            else max(0.0, float(self.volume_var.get())) / 100.0
        self._video_offset = offset
        self.play_btn.configure(text="■  Stop")
        self._playing = True
        threading.Thread(target=self._play_worker,
                         args=(start + offset, span - offset, gain), daemon=True).start()

    def _video_pause(self):
        """Freeze the marked section's sound where it is; space resumes from there."""
        if not self._playing:
            return
        import time as _time
        consumed = self._video_offset + (_time.monotonic() - self._video_t0)
        self._stop_audio()
        self._video_sound = consumed
        self.status.configure(text=f"paused — space resumes", fg=COLORS["text_secondary"])

    def _video_space(self):
        """Space on the video tab: play the marked section's sound / pause / resume. Once the
        ▶ button has full playback going (or paused), space toggles THAT instead — two
        transports fighting over one space bar would play sound under the moving picture."""
        if self._vplay_playing or self._vplay_resume is not None:
            self.video_toggle_play()
            return
        if self._playing:
            self._video_pause()
        elif self._video_sound is not None:
            offset, self._video_sound = self._video_sound, None
            self._video_play_sound(offset)
        else:
            self._video_play_sound(0.0)

    def _play_worker(self, start, span, gain):
        err = None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_preview.wav")
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", self.src,
                      "-vn", "-af", f"volume={gain:.3f}", "-ac", "2", "-ar", "44100",
                      "-c:a", "pcm_s16le", wav])
            if p.returncode != 0 or not os.path.exists(wav):
                raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:]
                                   or "could not extract the audio")
            _play_wav(wav)
        except Exception as exc:
            err = str(exc)
        self.root.after(0, self._play_done, span, err)

    def _play_done(self, span, err):
        if err:
            self._playing = False
            self.play_btn.configure(text="▶  Play sound")
            self.status.configure(text="Could not play the sound", fg=COLORS["error"])
            messagebox.showerror("Gizmo — cannot play that audio", err)
            return
        # winsound plays asynchronously and never says when it finished, so the button is reset
        # on a timer the length of the clip. Under a second either way — nobody is counting.
        # The clock for pause's consumed-time starts here, when the sound actually starts.
        import time as _time
        self._video_t0 = _time.monotonic()
        self._play_job = self.root.after(int(span * 1000) + 200, self._stop_audio)

    def _stop_audio(self):
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        _stop_wav()
        self._playing = False
        if self.play_btn.winfo_exists():
            self.play_btn.configure(text="▶  Play sound")

    # -- video playback (the ▶ button on the scrub card) --------------------------------------
    # Plays the MARKED CLIP — the same span, retime filters and pace that Save would export,
    # slow motion included — as moving pictures on the first-frame canvas, with the clip's
    # sound alongside. One ffmpeg decodes the span to raw frames at preview size on a reader
    # thread; a wall-clock tick paints the frame the elapsed time calls for, dropping frames
    # if decode falls behind rather than drifting. The clock starts only when the first frame
    # AND the sound are both ready, so picture and audio leave the gate together.

    def video_toggle_play(self):
        """The combined play/pause: play from the playhead, pause where it is, resume."""
        if self._vplay_playing:
            self._video_clip_pause()
        elif self._vplay_resume is not None:
            offset, self._vplay_resume = self._vplay_resume, None
            self._video_clip_play(offset)
        else:
            self._video_clip_play(0.0)

    def _vplay_clip_len(self):
        return self._frames() / FPS            # the clip's OWN timeline, both motion modes

    def _video_clip_play(self, offset):
        if not self.src or not self.info:
            return
        self._stop_audio()                     # the sound-only button yields to full playback
        clip_len = self._vplay_clip_len()
        if offset >= clip_len - 0.04:
            offset = 0.0
        fps = self.info["fps"]
        k = self._keep_every()
        start = self.frame_pos / fps
        # `offset` is clip time. In real time that IS source time; in slow motion each clip
        # second consumes k*FPS/fps source seconds. The export's audio track is the source's
        # real-time audio from `start` either way, so sound maps 1:1 to clip time.
        src_off = offset * k * FPS / fps if k else offset
        n_frames = max(1, int(round((clip_len - offset) * FPS)))
        w, h = fit_within(self.info["display_width"], self.info["display_height"],
                          PREVIEW_W, PREVIEW_H)
        if k:
            vf = f"select='not(mod(n\\,{k}))',setpts=N/({FPS}*TB)"
        else:
            vf = f"fps={FPS}"
        vf += f",scale={w}:{h}"
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error",
               "-ss", f"{max(0.0, start + src_off):.3f}", "-i", self.src,
               "-vf", vf, "-frames:v", str(n_frames), "-an",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        self._vplay_gen += 1
        gen = self._vplay_gen
        self._vplay_frames = []
        self._vplay_total = n_frames
        self._vplay_eof = False
        self._vplay_offset = offset
        self._vplay_t0 = None
        self._vplay_last_idx = -1
        self._vplay_playing = True
        self.vplay_btn.configure(text="⏸ Pause")

        has_sound = bool(self.info["has_audio"] and (audio_playback_backend() or ""))
        self._vplay_sound_ready = not has_sound
        if has_sound:
            gain = 1.0 if set_live_volume(self.volume_var.get()) \
                else max(0.0, float(self.volume_var.get())) / 100.0
            threading.Thread(target=self._vplay_sound_worker,
                             args=(start + offset, clip_len - offset, gain, gen),
                             daemon=True).start()
        threading.Thread(target=self._vplay_reader, args=(cmd, w, h, gen),
                         daemon=True).start()
        self._vplay_job = self.root.after(20, self._vplay_tick, gen)

    def _vplay_reader(self, cmd, w, h, gen):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    creationflags=_NO_WINDOW)
        except Exception:
            self._vplay_eof = True
            return
        self._vplay_proc = proc
        need = w * h * 3
        try:
            while gen == self._vplay_gen and len(self._vplay_frames) < self._vplay_total:
                buf = b""
                while len(buf) < need:
                    chunk = proc.stdout.read(need - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < need:
                    break                       # source ran out under the clip — play what is
                self._vplay_frames.append(Image.frombytes("RGB", (w, h), buf))
        finally:
            self._vplay_eof = True
            try:
                proc.kill()
            except Exception:
                pass

    def _vplay_sound_worker(self, start_s, span, gain, gen):
        """The clip's audio, extracted exactly as the sound button does it, into its own temp
        file so the two players never fight over one. ready is set even on failure — a broken
        track must not hold the picture at the gate forever."""
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_vpreview.wav")
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{start_s:.3f}", "-t", f"{span:.3f}", "-i", self.src,
                      "-vn", "-af", f"volume={gain:.3f}", "-ac", "2", "-ar", "44100",
                      "-c:a", "pcm_s16le", wav])
            if gen != self._vplay_gen:
                return
            if p.returncode == 0 and os.path.exists(wav):
                _play_wav(wav)
        except Exception:
            pass
        finally:
            if gen == self._vplay_gen:
                self._vplay_sound_ready = True

    def _vplay_tick(self, gen):
        if gen != self._vplay_gen or not self._vplay_playing:
            return
        import time as _time
        if self._vplay_t0 is None:
            # Hold the gate until the first frame and the sound are both ready.
            if self._vplay_frames and self._vplay_sound_ready:
                self._vplay_t0 = _time.monotonic()
            self._vplay_job = self.root.after(20, self._vplay_tick, gen)
            return
        idx = int((_time.monotonic() - self._vplay_t0) * FPS)
        frames = self._vplay_frames
        if idx >= self._vplay_total or (self._vplay_eof and idx >= len(frames)):
            self._vplay_stop(repaint=True)     # ran its course — back to the still preview
            return
        paint = min(idx, len(frames) - 1)
        if paint >= 0 and paint != self._vplay_last_idx:
            self._vplay_last_idx = paint
            self.photo = self._paint(self.canvas, frames[paint], PREVIEW_W, PREVIEW_H)
            # The crop rectangle rides on top of the moving picture — watching whether the
            # subject stays inside it is half of what playback is for.
            self._draw_crop_overlay()
        self._vplay_job = self.root.after(20, self._vplay_tick, gen)

    def _video_clip_pause(self):
        import time as _time
        consumed = self._vplay_offset + ((_time.monotonic() - self._vplay_t0)
                                         if self._vplay_t0 is not None else 0.0)
        self._vplay_stop()                     # keeps the paused frame on the canvas
        self._vplay_resume = min(consumed, self._vplay_clip_len())

    def _vplay_stop(self, repaint=False):
        """Tear playback down. The gen bump is the actual kill switch — reader, tick and
        sound worker all check it. Resume memory is cleared here; pause re-sets it after."""
        self._vplay_gen += 1
        self._vplay_playing = False
        self._vplay_resume = None
        if self._vplay_job is not None:
            try:
                self.root.after_cancel(self._vplay_job)
            except Exception:
                pass
            self._vplay_job = None
        if self._vplay_proc is not None:
            try:
                self._vplay_proc.kill()
            except Exception:
                pass
            self._vplay_proc = None
        _stop_wav()
        self._vplay_frames = []
        if self.vplay_btn.winfo_exists():
            self.vplay_btn.configure(text="▶ Play")
        if repaint and self.src:
            self.show_frame()                  # restores first+last frames and crop overlay

    # -- voice / audio mode -----------------------------------------------------------------------
    #
    # The same rhythm as the video side — open one file, mark the pieces you want, export the lot
    # — with the axes swapped: there is nothing to crop or retime, only WHERE a segment starts
    # and WHICH grid length it runs. Exports are wav, 32 kHz stereo, at exactly latents x 800
    # samples, each with its caption .txt beside it — so a segment dropped into a training folder
    # passes Fizgig's strict checks with nothing to configure.

    AUDIO_WAVE_W, AUDIO_WAVE_H = 940, 150

    def _build_audio_tab(self, body):
        # The record card pack-swaps with the source card, so both need handles: the tab body
        # to build into, and the source card's OUTER frame to hide and restore in place.
        self._audio_tab_body = body
        # 1. source
        c = self._card(body, "1. Source recording", self._VOICE_SOURCE_DESC)
        self._audio_source_outer = c.master.master      # body -> inner -> outer (see _card)
        self._audio_source_desc = c.master.winfo_children()[1]
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self._audio_open_btn = self._button(
            row, "Open audio…", self.open_audio, "accent",
            tip="Open a recording — or a video whose SOUND you want.\n\nAny format or "
                "sample rate; Gizmo converts on export. The original is never modified.")
        self._audio_open_btn.pack(side=tk.LEFT)
        self._audio_record_btn = self._button(
            row, "🎙 Record…", self.open_recorder,
            tip="Record a voice dataset right here — no files needed. Gizmo prompts "
                "sentences to read (hold to record, release when done), writes the "
                "caption for you, and each take drops straight into the editor "
                "ready to queue.")
        self._audio_record_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.audio_src_label = tk.Label(row, text="No recording open", font=(FONT_FAMILY, 10),
                                        bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.audio_src_label.pack(side=tk.LEFT, padx=12)
        tk.Label(row, text="…or drag a wav/mp3 onto this window",
                 font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.RIGHT)

        # 2. caption + length — before the waveform, same doctrine as the video tab: decide what
        # a segment IS before hunting for where it starts.
        c = self._card(body, "2. Caption and length",
                       "The caption teaches the voice, so describe only the sound — tone, pace, "
                       "accent, mood. No visual claims: there is no picture for them to be true "
                       "of. Transcribe appends the spoken words, which helps the voice bind.")
        self._voice_caption_outer = c.master.master
        trow = tk.Frame(c, bg=COLORS["bg_surface"])
        trow.pack(fill=tk.X)
        tk.Label(trow, text="Trigger word:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        # Remembered across sessions (Peter): a voice dataset is usually built over several
        # sittings, and retyping the same trigger word every open is pure friction.
        self.audio_trigger_var = tk.StringVar(value=self.settings.get("trigger_word", ""))
        self._persist_var(self.audio_trigger_var, "trigger_word")
        trig = tk.Entry(trow, textvariable=self.audio_trigger_var, width=24,
                        bg=COLORS["bg_hover"], fg=COLORS["text_primary"],
                        insertbackground=COLORS["text_primary"], relief=tk.FLAT)
        trig.pack(side=tk.LEFT, padx=(8, 0), ipady=4)
        self._shield_from_hotkeys(trig)
        ToolTip(trig, "Leads every caption, exactly like the Captions tab's Qwen tool: "
                      "\"<trigger>, <description> saying …\". Remembered between sessions.")
        self.audio_whisper_btn = self._button(
            trow, "🎤 Transcribe", self.audio_transcribe,
            tip="Listen to the marked segment with Whisper (a small speech-recognition model, "
                "~300 MB, downloaded once — Preferences' ⬇ Download models for me fetches it "
                "up front so this works offline) and append the words it hears to the caption "
                "as saying \"…\". The description half stays yours — no model can hear that a "
                "voice is warm.")
        self.audio_whisper_btn.pack(side=tk.RIGHT)
        _default_lang = self.settings.get("whisper_language", "Auto detect")
        self.audio_lang_var = tk.StringVar(
            value=_default_lang if _default_lang in WHISPER_LANGUAGES else "Auto detect")
        _lang = ttk.Combobox(trow, textvariable=self.audio_lang_var, width=11,
                             state="readonly", style="G.TCombobox",
                             values=WHISPER_LANGUAGES)
        _lang.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(_lang, "The language Whisper listens FOR. Auto detect usually works, but a "
                       "short segment can fool it — it once heard Welsh in English and looped. "
                       "Pick the language and the guess is skipped entirely.")
        self.audio_caption = tk.Text(c, height=3, wrap=tk.WORD, bg=COLORS["bg_hover"],
                                     fg=COLORS["text_primary"], relief=tk.FLAT,
                                     insertbackground=COLORS["text_primary"],
                                     font=(FONT_FAMILY, 10))
        self.audio_caption.pack(fill=tk.X, pady=(8, 0))
        self._shield_from_hotkeys(self.audio_caption)
        ToolTip(self.audio_caption,
                "This segment's caption — saved as a .txt beside the exported wav. It stays put "
                "between Adds, so describe the voice once and only touch it when something "
                "changes (a different emotion, a different speaker).")

        lrow = tk.Frame(c, bg=COLORS["bg_surface"])
        lrow.pack(fill=tk.X, pady=(10, 0))
        tk.Label(lrow, text="Segment length:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        self.audio_frames_var = tk.IntVar(value=124)
        for f in AUDIO_GRID_FRAMES:
            secs = hop_exact_samples(f) / AUDIO_SAMPLE_RATE
            rb = tk.Radiobutton(
                lrow, text=f"{secs:.1f} s", variable=self.audio_frames_var, value=f,
                command=self._audio_refresh, bg=COLORS["bg_surface"],
                fg=COLORS["text_primary"], selectcolor=COLORS["bg_deep"],
                activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                font=(FONT_FAMILY, 10))
            rb.pack(side=tk.LEFT, padx=(10, 0))
            ToolTip(rb, f"Exactly {hop_exact_samples(f) / AUDIO_SAMPLE_RATE:.3f} s "
                        f"({audio_latents_for(f)} audio latents) — one of the grid lengths H3 "
                        f"trains on, the same clock as video clips. Longer segments carry more "
                        f"of the voice per item; 5.2 s is the ceiling the model packs.")

        self._build_music_caption_card(body)

        # 3. the waveform
        c = self._card(body, "3. Find the moment",
                       "Play from here and just listen — click anywhere on the waveform and "
                       "playback jumps there, which is how you scrub sound. The shaded span is "
                       "what would export: grab it to slide it, drag either END to snap it to "
                       "the next allowed length, click outside it to place it, Play segment to "
                       "check it. Space is play / pause / resume, J-K-L shuttles like an edit "
                       "suite (tap J or L again for 1.5/2/3x), ←/→ nudge 50 ms, Shift 500 ms, "
                       "Home for the start.")
        self._audio_wave_outer = c.master.master
        self.audio_canvas = tk.Canvas(c, width=self.AUDIO_WAVE_W, height=self.AUDIO_WAVE_H,
                                      bg=COLORS["bg_deep"], highlightthickness=1,
                                      highlightbackground=COLORS["border"], cursor="crosshair")
        self.audio_canvas.pack()
        self.audio_canvas.bind("<Button-1>", self._audio_press)
        self.audio_canvas.bind("<B1-Motion>", self._audio_drag)
        self.audio_canvas.bind("<ButtonRelease-1>", self._audio_release)
        self.audio_canvas.bind("<Motion>", self._audio_hover)
        # Shift+wheel pans a zoomed view sideways (plain wheel zooms).
        self.audio_canvas.bind(
            "<Shift-MouseWheel>",
            lambda e: (self._audio_pan(-0.15 if e.delta > 0 else 0.15), "break")[1])
        # The nav bar under the waveform: where the zoomed window sits in the whole recording,
        # and the handle to slide it. Full-width (and inert) when not zoomed.
        self.audio_scroll = ttk.Scrollbar(c, orient=tk.HORIZONTAL, command=self._audio_xview)
        self.audio_scroll.pack(fill=tk.X, pady=(2, 0))
        ToolTip(self.audio_scroll,
                "Where the zoomed view sits in the whole recording — drag to scroll along it. "
                "Shift+wheel over the waveform pans too; plain wheel zooms.")
        # Wheel over the waveform zooms it, anchored on the mouse — the sound you are pointing
        # at stays under the pointer, map-style. Returning "break" keeps the page from
        # scrolling; everywhere else the wheel still scrolls the window.
        def _wheel_zoom(e):
            v0, vd = self._audio_view_bounds()
            self._audio_zoom(4.0 if e.delta > 0 else 0.25,
                             focus_t=v0 + e.x / self.AUDIO_WAVE_W * vd)
            return "break"
        self.audio_canvas.bind("<MouseWheel>", _wheel_zoom)
        prow = tk.Frame(c, bg=COLORS["bg_surface"])
        prow.pack(fill=tk.X, pady=(8, 0))
        self.audio_listen_btn = self._button(
            prow, "▶  Play from here", lambda: self.audio_toggle_play("file"),
            tip="Listen through the recording from the playhead — click the waveform while it "
                "plays to jump around. This is how you FIND the moment; Play segment is how "
                "you check it.")
        self.audio_listen_btn.pack(side=tk.LEFT)
        self.audio_play_btn = self._button(
            prow, "▶  Play segment", lambda: self.audio_toggle_play("segment"),
            tip="Hear exactly what would export — the marked span, nothing more.  (volume is "
                "listening only, never saved)")
        self.audio_play_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.audio_volume_var = tk.DoubleVar(value=80)
        _vol = ttk.Scale(prow, from_=0, to=100, orient=tk.HORIZONTAL, length=140,
                         style="G.Horizontal.TScale", variable=self.audio_volume_var,
                         command=lambda v: set_live_volume(v))
        _vol.pack(side=tk.LEFT, padx=(12, 0))
        ToolTip(_vol, "Listening volume only — moves live while a preview plays. The exported "
                      "segment's audio is never touched.")
        # Fine placement: zoom to see the detail, nudge to land on it. Both live beside the
        # play buttons because they are all one activity.
        for txt, fn, tip in (
                ("🔍−", lambda: self._audio_zoom(0.25), "Zoom out 4x. Fit shows the whole file."),
                ("🔍+", lambda: self._audio_zoom(4.0),
                 "Zoom in 4x around the playhead — one pixel gets 16x finer per press, which "
                 "is what makes trimming a breath off a segment start possible."),
                ("Fit", lambda: self._audio_zoom(None), "Show the whole recording again."),
                ("−0.5s", lambda: self._audio_nudge(-0.5), "Nudge the segment half a second "
                                                           "earlier.  (Shift+←)"),
                ("−50ms", lambda: self._audio_nudge(-0.05), "Nudge the segment 50 ms earlier — "
                                                            "restarts playback if something is "
                                                            "playing.  (←)"),
                ("+50ms", lambda: self._audio_nudge(0.05), "Nudge the segment 50 ms later.  (→)"),
                ("+0.5s", lambda: self._audio_nudge(0.5), "Nudge the segment half a second "
                                                          "later.  (Shift+→)")):
            self._button(prow, txt, fn, pad=8, tip=tip).pack(side=tk.LEFT, padx=(6, 0))
        # Its own line UNDER the buttons — in the button row it fought a dozen controls for
        # width and lost, showing up truncated or not at all.
        self.audio_pos_label = tk.Label(c, text="", font=("Consolas", 10),
                                        bg=COLORS["bg_surface"], fg=COLORS["text_secondary"])
        self.audio_pos_label.pack(anchor=tk.W, pady=(6, 0))

        # 4. queue + export
        c = self._card(body, "4. Mark it, then move on",
                       "Add every section you want, then open the next file and keep adding: the "
                       "queue keeps them all, and Export writes the lot in one go — each wav "
                       "lands with its caption .txt beside it, ready for the training folder.")
        orow = tk.Frame(c, bg=COLORS["bg_surface"])
        orow.pack(fill=tk.X)
        tk.Label(orow, text="Save to:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        self._voice_out_var = tk.StringVar(value=self.settings.get("voice_out_dir", ""))
        self._persist_var(self._voice_out_var, "voice_out_dir")
        self._music_out_var = tk.StringVar(value=self.settings.get("music_out_dir", ""))
        self._persist_var(self._music_out_var, "music_out_dir")
        self.audio_out_var = self._voice_out_var
        oe = self._audio_out_entry = tk.Entry(
            orow, textvariable=self.audio_out_var, bg=COLORS["bg_hover"],
            fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"], relief=tk.FLAT)
        oe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8), ipady=4)
        self._shield_from_hotkeys(oe)
        ToolTip(oe, "Where the segments and their caption .txt files land — point it at your "
                    "training folder to skip a copy step.")
        self._button(orow, "Browse…", self._audio_pick_output, pad=10,
                     tip="Pick the save folder.").pack(side=tk.LEFT)

        qrow = tk.Frame(c, bg=COLORS["bg_surface"])
        qrow.pack(fill=tk.X, pady=(10, 0))
        self.audio_add_btn = self._button(
            qrow, "➕  Add to queue", self.audio_add_to_queue, "accent",
            tip="Remember this segment — where it starts, its length, its caption.  (Ctrl+S)\n\n"
                "Nothing is written yet; keep marking.")
        self.audio_add_btn.pack(side=tk.LEFT)
        self.audio_export_btn = self._button(
            qrow, "Export queue", self.audio_export_queue,
            tip="Write every queued segment: 32 kHz stereo wav at the exact grid length, caption "
                ".txt beside it. Fizgig accepts them as-is.")
        self.audio_export_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.audio_status = tk.Label(qrow, text="", font=(FONT_FAMILY, 10),
                                     bg=COLORS["bg_surface"], fg=COLORS["text_secondary"])
        self.audio_status.pack(side=tk.LEFT, padx=12)
        self._button(qrow, "📂 Open folder", self._audio_open_folder,
                     tip="Open the save folder in your file browser.").pack(side=tk.RIGHT)

        self.audio_queue_box = tk.Listbox(
            c, height=6, bg=COLORS["bg_deep"], fg=COLORS["text_primary"], font=("Consolas", 9),
            relief=tk.FLAT, highlightthickness=1, highlightbackground=COLORS["border"],
            selectbackground=COLORS["accent"])
        self.audio_queue_box.pack(fill=tk.X, pady=(10, 0))
        self.audio_queue_box.bind("<Double-Button-1>", lambda _e: self._audio_recall())
        ToolTip(self.audio_queue_box,
                "The queue. ✓ has been written, · is still waiting.\n\nDouble-click a waiting "
                "row to put the playhead, length and caption back where they were.")
        brow = tk.Frame(c, bg=COLORS["bg_surface"])
        brow.pack(fill=tk.X, pady=(6, 0))
        self._button(brow, "Remove selected", self._audio_remove_selected, pad=10,
                     tip="Drop the selected row from the queue. Already-exported segments stay "
                         "on disk — delete those in your file browser.").pack(side=tk.LEFT)
        self._button(brow, "Clear queue", self._audio_clear_queue, pad=10,
                     tip="Empty the waiting list. Nothing already exported is "
                         "touched.").pack(side=tk.LEFT, padx=(8, 0))

        self._audio_set_enabled(False)

    _VOICE_SOURCE_DESC = ("A voice memo, a podcast, an interview — any wav, mp3, flac or m4a. Or "
                          "a VIDEO file, to use just its audio track: nothing is shown, you "
                          "scrub the sound itself. Opening another file keeps your queue.")
    _MUSIC_SOURCE_DESC = ("A song — any wav, mp3, flac, ogg or m4a. Cut it into sections for "
                          "ACE-Step training, or keep it whole: songs of any length mix fine "
                          "in one dataset, and longer sections (2 to 4 minutes) carry more of "
                          "the song. Opening another song keeps your queue.")

    def _build_music_caption_card(self, body):
        """Music mode's card 2: caption, lyrics, section length and the fades at the cuts.
        Built hidden; the tab switch swaps it in for the voice caption card."""
        c = self._card(body, "2. Caption, lyrics and length",
                       "Every section's caption is the trigger word, then words describing the "
                       "music — the same words on every song of one sound. Lyrics are the "
                       "lines sung in THIS section; leave them empty for an instrumental.")
        self._music_caption_outer = c.master.master
        self._music_caption_outer.pack_forget()

        def entry(parent, var, width=None):
            e = tk.Entry(parent, textvariable=var, bg=COLORS["bg_hover"],
                         fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                         relief=tk.FLAT, **({"width": width} if width else {}))
            self._shield_from_hotkeys(e)
            return e

        def label(parent, text, **kw):
            return tk.Label(parent, text=text, font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                            fg=COLORS["text_primary"], **kw)

        trow = tk.Frame(c, bg=COLORS["bg_surface"])
        trow.pack(fill=tk.X)
        label(trow, "Trigger word:").pack(side=tk.LEFT)
        self.music_trigger_var = tk.StringVar(value=self.settings.get("music_trigger_word", ""))
        self._persist_var(self.music_trigger_var, "music_trigger_word")
        te = entry(trow, self.music_trigger_var, 18)
        te.pack(side=tk.LEFT, padx=(8, 16), ipady=4)
        ToolTip(te, "Leads every caption, e.g. fzgchip. Use the same word in your sample "
                    "prompts. Remembered between sessions.")
        label(trow, "Describe the music:").pack(side=tk.LEFT)
        self.music_words_var = tk.StringVar(value=self.settings.get("music_words", ""))
        self._persist_var(self.music_words_var, "music_words")
        we = entry(trow, self.music_words_var)
        we.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), ipady=4)
        ToolTip(we, "Genre, instruments, mood, vocals — the words you will prompt with. "
                    f"For example: {MUSIC_WORDS_EXAMPLE}")
        # A greyed example while the box is empty — never saved, gone on focus.
        hint = tk.Label(we, text=MUSIC_WORDS_EXAMPLE, font=(FONT_FAMILY, 10),
                        bg=COLORS["bg_hover"], fg=COLORS["text_muted"], cursor="xterm")

        def _hint(*_a):
            if self.music_words_var.get() or self.root.focus_get() is we:
                hint.place_forget()
            else:
                hint.place(x=2, rely=0.5, anchor="w")
        hint.bind("<Button-1>", lambda _e: we.focus_set())
        we.bind("<FocusIn>", _hint, add="+")
        we.bind("<FocusOut>", _hint, add="+")
        self.music_words_var.trace_add("write", _hint)
        self.root.after(200, _hint)

        lrow = tk.Frame(c, bg=COLORS["bg_surface"])
        lrow.pack(fill=tk.X, pady=(10, 0))
        label(lrow, "Lyrics in this section (optional):").pack(side=tk.LEFT)
        self.music_lyrics_note = tk.Label(lrow, text="", font=(FONT_FAMILY, 9),
                                          bg=COLORS["bg_surface"], fg=COLORS["warning"])
        self.music_lyrics_note.pack(side=tk.LEFT, padx=(10, 0))
        self.music_lyrics = tk.Text(c, height=4, wrap=tk.WORD, bg=COLORS["bg_hover"],
                                    fg=COLORS["text_primary"], relief=tk.FLAT,
                                    insertbackground=COLORS["text_primary"],
                                    font=(FONT_FAMILY, 10))
        self.music_lyrics.pack(fill=tk.X, pady=(4, 0))
        self._shield_from_hotkeys(self.music_lyrics)

        def _edited(_e):
            # Edited by hand: the warning about the song's own lyrics has been heard.
            if getattr(self, "_music_lyrics_from_song", False):
                self._music_lyrics_from_song = False
                self._music_refresh_lyrics_note()
        self.music_lyrics.bind("<KeyRelease>", _edited, add="+")
        ToolTip(self.music_lyrics,
                "Saved as a .lyrics.txt beside the section. When the song has a .lyrics.txt "
                "of its own it is filled in here — trim it to the lines sung in the section, "
                "or the model learns words that aren't there.")

        srow = tk.Frame(c, bg=COLORS["bg_surface"])
        srow.pack(fill=tk.X, pady=(10, 0))
        label(srow, "Section length:").pack(side=tk.LEFT)
        self.music_len_var = tk.StringVar(value=f"{self.music_len:.1f}")
        le = entry(srow, self.music_len_var, 7)
        le.pack(side=tk.LEFT, padx=(8, 4), ipady=4)
        le.bind("<Return>", lambda _e: self._music_len_typed())
        le.bind("<FocusOut>", lambda _e: self._music_len_typed(), add="+")
        ToolTip(le, f"Seconds, {MUSIC_MIN_S:.0f} to {MUSIC_MAX_S:.0f}. Or drag either end of "
                    "the shaded span on the waveform.")
        label(srow, "s").pack(side=tk.LEFT)
        self._button(srow, "Whole song", self._music_whole, pad=10,
                     tip="The whole song in one section — or its first four minutes, "
                         "the most ACE-Step trains on.").pack(side=tk.LEFT, padx=(10, 0))

        frow = tk.Frame(c, bg=COLORS["bg_surface"])
        frow.pack(fill=tk.X, pady=(10, 0))
        label(frow, "Fade at a cut — in:").pack(side=tk.LEFT)
        self.music_fade_in_var = tk.StringVar(value=self.settings.get("music_fade_in", "0.1"))
        self._persist_var(self.music_fade_in_var, "music_fade_in")
        fi = entry(frow, self.music_fade_in_var, 5)
        fi.pack(side=tk.LEFT, padx=(8, 4), ipady=4)
        label(frow, "s    out:").pack(side=tk.LEFT)
        self.music_fade_out_var = tk.StringVar(value=self.settings.get("music_fade_out", "1.5"))
        self._persist_var(self.music_fade_out_var, "music_fade_out")
        fo = entry(frow, self.music_fade_out_var, 5)
        fo.pack(side=tk.LEFT, padx=(8, 4), ipady=4)
        label(frow, "s").pack(side=tk.LEFT)
        tk.Label(frow, text="Only where a section cuts into the song — its own start and end "
                            "stay as they are. 0 turns a fade off.",
                 font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_explain"]).pack(side=tk.LEFT, padx=(12, 0))
        for w in (fi, fo):
            ToolTip(w, "A short fade-in stops a click where the section starts mid-song; a "
                       "fade-out of a second or two ends it like a song instead of a cut-off.")

    # -- music: section length ------------------------------------------------------------------
    def _music_set_len(self, seconds):
        """Clamp a section length to what ACE-Step trains on and what is left of the song."""
        room = max(0.0, self.audio_dur - self.audio_pos) if self.audio_src else MUSIC_MAX_S
        top = min(MUSIC_MAX_S, room) if self.audio_src else MUSIC_MAX_S
        # Tenths of a second: a drag lands on 99.9 s rather than 99.8825 s.
        self.music_len = int(max(min(MUSIC_MIN_S, top), min(float(seconds), top)) * 10) / 10
        self.music_len_var.set(f"{self.music_len:.1f}")
        self._music_refresh_lyrics_note()

    def _music_len_typed(self):
        try:
            secs = float(self.music_len_var.get())
        except ValueError:
            self.music_len_var.set(f"{self.music_len:.1f}")
            return
        if abs(secs - self.music_len) > 1e-3:
            self._music_set_len(secs)
            self._audio_refresh()
            self._audio_arm_rescrub()

    def _music_whole(self):
        if not self.audio_src:
            return
        self.audio_pos = 0.0
        self._music_set_len(MUSIC_MAX_S)
        self._audio_seek(0.0)

    def _music_song_lyrics(self, path):
        """The song's own lyrics file (<stem>.lyrics.txt), or ""."""
        side = os.path.splitext(path)[0] + ".lyrics.txt"
        try:
            with open(side, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _music_refresh_lyrics_note(self):
        """Warn when the song's own lyrics sit in the box but the section is only part of it."""
        if not hasattr(self, "music_lyrics_note"):
            return
        whole = (self.audio_src and self.audio_pos < 0.05
                 and self.audio_pos + self.music_len > self.audio_dur - 0.05)
        from_song = getattr(self, "_music_lyrics_from_song", False)
        self.music_lyrics_note.configure(
            text="" if whole or not from_song else
            "⚠ from the song's .lyrics.txt — trim to the lines sung in this section")

    def _music_caption_text(self):
        words = " ".join(self.music_words_var.get().split()).strip(", ")
        trig = self.music_trigger_var.get().strip().rstrip(",")
        return ", ".join(p for p in (trig, words) if p)

    # -- audio: opening -----------------------------------------------------------------------------
    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Open a recording — or a video, for its audio track",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                       ("Video files (audio track)", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        self.notebook.select(self._audio_host_tab())
        try:
            self.load_audio(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that audio", str(exc))

    def load_audio(self, path):
        """Open a recording (or a video's audio track): probe duration, decode peaks in the
        background, arm the tab."""
        self._audio_stop()
        # Duration by decoding to null — the one number the container cannot lie about, and it
        # covers every format the same way. A few hundred ms even for long files.
        p = _run([self.ffmpeg, "-hide_banner", "-i", path, "-vn",
                  "-f", "null", os.devnull])
        m = None
        for m in re.finditer(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)",
                             (p.stderr or b"").decode("utf-8", "replace")):
            pass
        if m is None:
            raise RuntimeError(f"{os.path.basename(path)}: no decodable audio in this file"
                               + (" — it has a picture but no sound"
                                  if not is_audio_file(path) else ""))
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        music = self._music_mode()
        min_dur = MUSIC_MIN_S if music else             hop_exact_samples(AUDIO_GRID_FRAMES[0]) / AUDIO_SAMPLE_RATE
        if dur < min_dur:
            raise RuntimeError(f"{os.path.basename(path)}: {dur:.2f} s of audio — the shortest "
                               f"{'section ACE-Step' if music else 'segment H3'} trains on "
                               f"here is {min_dur:.1f} s.")
        self.audio_src = path
        self.audio_dur = dur
        self.audio_from_video = not is_audio_file(path)
        self.audio_pos = 0.0
        self.audio_peaks = None
        self._audio_samples = None
        self._audio_peaks_key = None
        self.audio_view = None
        self.audio_src_label.configure(
            text=f"{os.path.basename(path)} — {dur:.1f} s"
                 + ("  (a video's audio track)" if self.audio_from_video else ""),
            fg=COLORS["text_primary"])
        if not self.audio_out_var.get():
            self.audio_out_var.set(os.path.join(os.path.dirname(path),
                                                "fizgig_music" if music else "fizgig_voice"))
        if music:
            lyrics = self._music_song_lyrics(path)
            self.music_lyrics.delete("1.0", tk.END)
            self.music_lyrics.insert("1.0", lyrics)
            self._music_lyrics_from_song = bool(lyrics)
            self._music_set_len(MUSIC_MAX_S)
        self._audio_set_enabled(True)
        self._audio_refresh()
        self.audio_status.configure(text="reading the waveform…", fg=COLORS["text_secondary"])
        threading.Thread(target=self._audio_peaks_worker, args=(path,), daemon=True).start()

    _AUDIO_PEAK_RATE = 8000

    def _audio_peaks_worker(self, path):
        """Decode to 8 kHz mono f32 and KEEP the samples — zoom re-slices them into per-column
        (lo, hi) pairs on demand. array-module slices keep every reduction at C speed with no
        numpy dependency. The normalisation is computed once over the whole file, so zooming
        into a quiet stretch shows it quiet instead of rescaling it to fill the canvas."""
        import array
        try:
            p = _run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn",
                      "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                      "-ar", str(self._AUDIO_PEAK_RATE), "-"])
            samples = array.array("f")
            samples.frombytes(p.stdout or b"")
            if not samples:
                raise RuntimeError("decoded no samples")
            scale = max(1e-6, max(samples), -min(samples))
        except Exception as exc:
            err = str(exc)
            self.root.after(0, lambda: (self.audio_status.configure(
                text=f"waveform failed: {err[:60]}", fg=COLORS["error"])))
            return
        def done():
            if self.audio_src == path:          # not superseded by another open
                self._audio_samples = samples
                self._audio_peak_scale = scale
                self._audio_peaks_key = None
                # The decoded samples are the ground truth for duration. The load-time probe
                # parses ffmpeg's progress line, which can under-report — and then the
                # waveform (drawn from these samples) visibly extends past where the segment
                # clamp thinks the file ends, so the box refuses to move into sound that is
                # plainly on screen (Peter). Trust the samples.
                real = len(samples) / self._AUDIO_PEAK_RATE
                if abs(real - self.audio_dur) > 0.02:
                    self.audio_dur = real
                self.audio_status.configure(text="", fg=COLORS["text_secondary"])
                self._audio_refresh()
        self.root.after(0, done)

    def _audio_world(self):
        """(start_s, length_s) of the scrollable WORLD: the file plus a margin either side.

        The view is always allowed to be wider than the audio, because a segment that fills
        the whole file otherwise has its edge grips pinned to the canvas edges where they
        cannot be grabbed (Peter) — the margin is dead air on screen, and grabbing room in
        the hand."""
        m = max(0.4, self.audio_dur * 0.06)
        return -m, max(self.audio_dur, 1e-6) + 2 * m

    def _audio_view_bounds(self):
        """(start_s, dur_s) of what the canvas shows — the whole world unless zoomed."""
        if self.audio_view is None:
            return self._audio_world()
        return self.audio_view

    def _audio_slice_peaks(self):
        """Per-column (lo, hi) for the current view, cached until the view or samples change."""
        if self._audio_samples is None:
            self.audio_peaks = None
            return
        v0, vd = self._audio_view_bounds()
        key = (round(v0, 4), round(vd, 4), len(self._audio_samples))
        if key == self._audio_peaks_key:
            return
        s = self._audio_samples
        r = self._AUDIO_PEAK_RATE
        # lo_i may be NEGATIVE now the view can start before the file (the world margin);
        # columns outside the audio draw flat, and a negative python slice must never wrap.
        lo_i = int(v0 * r)
        n = max(1, int(vd * r))
        n_cols = self.AUDIO_WAVE_W
        step = max(1, n // n_cols)
        peaks = []
        for i in range(n_cols):
            a = lo_i + i * step
            b = a + step
            chunk = s[max(0, a): max(0, b)] if b > 0 else []
            peaks.append((min(chunk) / self._audio_peak_scale,
                          max(chunk) / self._audio_peak_scale) if len(chunk) else (0.0, 0.0))
        self.audio_peaks = peaks
        self._audio_peaks_key = key

    # -- audio: the waveform ------------------------------------------------------------------------
    def _audio_seek(self, seconds):
        if not self.audio_src:
            return
        span = self._audio_span()
        self.audio_pos = max(0.0, min(float(seconds), self.audio_dur - span))
        self._audio_resume = None     # a moved playhead invalidates a paused position
        # Zoomed in and the playhead left the window: follow it, keeping a fifth of margin.
        # NEVER while a drag is in progress — the pan would shift the view under the cursor,
        # the same mouse position would map to a new time, and that feedback loop flings the
        # segment across the whole recording on the slightest move.
        if self.audio_view is not None and not getattr(self, "_audio_dragging", False):
            v0, vd = self.audio_view
            if not (v0 <= self.audio_pos <= v0 + vd * 0.95):
                w0, wl = self._audio_world()
                v0 = max(w0, min(self.audio_pos - vd * 0.2, w0 + wl - vd))
                self.audio_view = (v0, vd)
        self._audio_refresh()
        self._audio_arm_rescrub()

    def _audio_arm_rescrub(self):
        """Adjusting anything WHILE listening restarts playback from the new state, in
        whichever mode is running — a nudge or an edge-resize during Play segment should not
        demand another button press. Debounced so a drag or a held arrow relaunches once."""
        if self._audio_playing:
            if self._audio_cursor_job is not None:
                self.root.after_cancel(self._audio_cursor_job)
                self._audio_cursor_job = None
            if getattr(self, "_audio_rescrub_job", None) is not None:
                self.root.after_cancel(self._audio_rescrub_job)
            self._audio_rescrub_job = self.root.after(180, self._audio_rescrub)

    def _audio_rescrub(self):
        self._audio_rescrub_job = None
        if self._audio_playing and self._audio_play_mode:
            mode = self._audio_play_mode
            self._audio_stop()
            self.audio_toggle_play(mode)

    def _audio_nudge(self, delta):
        self._audio_seek(self.audio_pos + delta)

    _AUDIO_EDGE_PX = 7                        # how close to a span edge counts as grabbing it

    def _audio_edge_at(self, x):
        """"left"/"right" when x is on a span edge, else None. Right wins a tie on a tiny
        span — the more common resize."""
        if not self.audio_src:
            return None
        v0, vd = self._audio_view_bounds()
        x0 = (self.audio_pos - v0) / vd * self.AUDIO_WAVE_W
        x1 = (self.audio_pos + self._audio_span() - v0) / vd * self.AUDIO_WAVE_W
        if abs(x - x1) <= self._AUDIO_EDGE_PX:
            return "right"
        if abs(x - x0) <= self._AUDIO_EDGE_PX:
            return "left"
        return None

    def _nearest_grid_frames(self, len_s, max_len_s):
        """The allowed length closest to len_s that still fits in max_len_s."""
        fits = [f for f in AUDIO_GRID_FRAMES
                if hop_exact_samples(f) / AUDIO_SAMPLE_RATE <= max_len_s + 0.01]
        if not fits:
            fits = [AUDIO_GRID_FRAMES[0]]
        return min(fits, key=lambda f: abs(hop_exact_samples(f) / AUDIO_SAMPLE_RATE - len_s))

    def _audio_press(self, event):
        """Button down, three grips: an EDGE of the span resizes it (snapping to the allowed
        lengths), inside it DRAGS it (no jump — it stays under your hand), outside it places
        it at the click."""
        if not self.audio_src:
            return
        self._audio_dragging = True
        self._audio_limit_hinted = False
        self._audio_edge = self._audio_edge_at(event.x)
        v0, vd = self._audio_view_bounds()
        t = v0 + event.x / self.AUDIO_WAVE_W * vd
        if self._audio_edge == "left":
            self._audio_edge_end = self.audio_pos + self._audio_span()   # the anchor
        elif self._audio_edge is None:
            if self.audio_pos <= t <= self.audio_pos + self._audio_span():
                self._audio_grab = t - self.audio_pos
            else:
                self._audio_grab = 0.0
                self._audio_seek(t)

    def _audio_drag(self, event):
        if not self.audio_src:
            return
        v0, vd = self._audio_view_bounds()
        # The cursor clamped to the canvas, THEN mapped: dragging past the edge parks the
        # segment at the view's boundary instead of extrapolating off into the rest of the
        # file. Panning further is a release away.
        x = max(0, min(event.x, self.AUDIO_WAVE_W))
        t = v0 + x / self.AUDIO_WAVE_W * vd
        edge = getattr(self, "_audio_edge", None)
        if edge and self._music_mode():
            # Music has no grid: an edge goes where the hand puts it, within ACE-Step's
            # 10 s to 4 min and the song's own ends.
            if edge == "right":
                self._music_set_len(t - self.audio_pos)
                self._audio_refresh()
                self._audio_arm_rescrub()
            else:
                end = self._audio_edge_end
                start = max(0.0, end - MUSIC_MAX_S, min(t, end - MUSIC_MIN_S))
                self.music_len = end - start
                self._audio_seek(start)
                self._music_set_len(end - start)
            return
        if edge == "right":
            frames = self._nearest_grid_frames(t - self.audio_pos,
                                               self.audio_dur - self.audio_pos)
            if frames != self.audio_frames_var.get():
                self.audio_frames_var.set(frames)
                self._audio_refresh()
                self._audio_arm_rescrub()
            elif (t - self.audio_pos) > hop_exact_samples(frames) / AUDIO_SAMPLE_RATE + 0.25 \
                    and not getattr(self, "_audio_limit_hinted", False):
                # The drag is demanding more than the largest length that fits from this
                # position. Doing silently nothing here read as "the box is stuck" (Peter) —
                # say what the wall is and what would move past it.
                self._audio_limit_hinted = True
                self.audio_status.configure(
                    text="no longer length fits from here — drag the box's middle to move "
                         "it, or start it earlier", fg=COLORS["warning"])
        elif edge == "left":
            end = self._audio_edge_end
            frames = self._nearest_grid_frames(end - t, end)
            span = hop_exact_samples(frames) / AUDIO_SAMPLE_RATE
            if frames != self.audio_frames_var.get() or abs(end - span - self.audio_pos) > 1e-6:
                self.audio_frames_var.set(frames)
                self._audio_seek(end - span)     # seek arms the rescrub itself
        else:
            self._audio_seek(t - getattr(self, "_audio_grab", 0.0))
            # Edge-autoscroll for a MOVE drag: parking the cursor at the canvas edge pans the
            # view on a timer while the box stays under the hand — without it the box stopped
            # at the view's boundary with audio sitting just off-screen, and moving further
            # meant release-pan-regrab (Peter). Timer-driven on purpose: the old fling bug
            # came from panning inside the seek itself, where a stationary cursor remapped to
            # new time every refresh; a clock advances one measured step per tick instead.
            self._audio_last_drag_x = x
            if (self.audio_view is not None
                    and (x >= self.AUDIO_WAVE_W - self._AUTO_PAN_ZONE
                         or x <= self._AUTO_PAN_ZONE)
                    and self._audio_autopan_job is None):
                self._audio_autopan_job = self.root.after(50, self._audio_autopan_tick)

    _AUTO_PAN_ZONE = 14

    def _audio_autopan_tick(self):
        self._audio_autopan_job = None
        if (not getattr(self, "_audio_dragging", False)
                or getattr(self, "_audio_edge", None) is not None
                or self.audio_view is None):
            return
        x = getattr(self, "_audio_last_drag_x", None)
        if x is None:
            return
        v0, vd = self.audio_view
        w0, wl = self._audio_world()
        step = vd * 0.04
        if x >= self.AUDIO_WAVE_W - self._AUTO_PAN_ZONE and v0 + vd < w0 + wl - 1e-9:
            nv0 = min(v0 + step, w0 + wl - vd)
        elif x <= self._AUTO_PAN_ZONE and v0 > w0 + 1e-9:
            nv0 = max(v0 - step, w0)
        else:
            return                             # cursor left the zone — the next motion re-arms
        self.audio_view = (nv0, vd)
        t = nv0 + x / self.AUDIO_WAVE_W * vd   # the box stays under the hand in the new view
        self._audio_seek(t - getattr(self, "_audio_grab", 0.0))
        self._audio_autopan_job = self.root.after(50, self._audio_autopan_tick)

    def _audio_release(self, _event):
        self._audio_dragging = False
        self._audio_edge = None
        if getattr(self, "_audio_autopan_job", None) is not None:
            self.root.after_cancel(self._audio_autopan_job)
            self._audio_autopan_job = None
        self._audio_seek(self.audio_pos)      # one follow-check now the drag is over

    def _audio_pan(self, frac):
        """Slide a zoomed view sideways by `frac` of its own width. No-op unzoomed."""
        if self.audio_view is None:
            return
        v0, vd = self.audio_view
        w0, wl = self._audio_world()
        self.audio_view = (max(w0, min(v0 + frac * vd, w0 + wl - vd)), vd)
        self._audio_refresh()

    def _audio_xview(self, *args):
        """The nav bar's command: moveto from a drag, scroll from its arrows/trough."""
        if self.audio_view is None or not self.audio_dur:
            return
        v0, vd = self.audio_view
        w0, wl = self._audio_world()
        if args and args[0] == "moveto":
            n0 = w0 + float(args[1]) * wl
        elif args and args[0] == "scroll":
            n0 = v0 + int(args[1]) * vd * (0.1 if args[2] == "units" else 0.9)
        else:
            return
        self.audio_view = (max(w0, min(n0, w0 + wl - vd)), vd)
        self._audio_refresh()

    def _audio_hover(self, event):
        """A resize cursor over the span's edges, so the grips announce themselves."""
        if getattr(self, "_audio_dragging", False):
            return
        want = "sb_h_double_arrow" if self._audio_edge_at(event.x) else "crosshair"
        if self.audio_canvas.cget("cursor") != want:
            self.audio_canvas.configure(cursor=want)

    def _audio_zoom(self, factor, focus_t=None):
        """Zoom the waveform. factor >1 in, <1 out, None = fit all. The anchor — what stays
        put on screen — is `focus_t` when given (the wheel passes the time under the mouse, so
        the sound you are pointing at stays under the pointer), else the playhead (buttons)."""
        if not self.audio_src:
            return
        if factor is None:
            self.audio_view = None
        else:
            v0, vd = self._audio_view_bounds()
            w0, wl = self._audio_world()
            anchor = self.audio_pos if focus_t is None else max(0.0, min(focus_t, self.audio_dur))
            nd = max(0.5, min(wl, vd / factor))
            if nd >= wl:
                self.audio_view = None
            else:
                # The anchor keeps its screen position: same fraction of the window.
                frac = (anchor - v0) / vd if vd else 0.5
                frac = max(0.0, min(1.0, frac))
                n0 = max(w0, min(anchor - frac * nd, w0 + wl - nd))
                self.audio_view = (n0, nd)
        self._audio_refresh()

    def _audio_span(self):
        if self._music_mode():
            return self.music_len
        return hop_exact_samples(self.audio_frames_var.get()) / AUDIO_SAMPLE_RATE

    def _audio_refresh(self):
        """Redraw the waveform (the current VIEW of it), the segment span and the readout."""
        c = self.audio_canvas
        c.delete("all")
        W, H = self.AUDIO_WAVE_W, self.AUDIO_WAVE_H
        mid = H / 2
        v0, vd = self._audio_view_bounds()

        def to_x(t):
            return (t - v0) / vd * W

        if self.audio_src and self.audio_dur:
            # The world margins, visibly NOT audio: a segment stops at the file's edge, and
            # without the colour change you wonder why it won't stretch further (Peter).
            xz, xe = to_x(0.0), to_x(self.audio_dur)
            if xz > 0:
                c.create_rectangle(0, 0, xz, H, fill="#12161D", outline="")
            if xe < W:
                c.create_rectangle(xe, 0, W, H, fill="#12161D", outline="")
            span = self._audio_span()
            self.audio_pos = max(0.0, min(self.audio_pos, max(0.0, self.audio_dur - span)))
            x0 = max(0, to_x(self.audio_pos))
            x1 = min(W, to_x(self.audio_pos + span))
            if x1 > 0 and x0 < W:
                c.create_rectangle(x0, 0, x1, H, fill=COLORS["bg_hover"], outline="")
        self._audio_slice_peaks()
        if self.audio_peaks:
            for x, (lo, hi) in enumerate(self.audio_peaks):
                c.create_line(x, mid - hi * (mid - 4), x, mid - lo * (mid - 4),
                              fill=COLORS["accent"])
        else:
            c.create_line(0, mid, W, mid, fill=COLORS["border"])
            if self.audio_src:
                c.create_text(W / 2, mid - 14, text="reading the waveform…",
                              fill=COLORS["text_muted"], font=(FONT_FAMILY, 10))
        if self.audio_src and self.audio_dur:
            # The file's own ends, drawn ON TOP of the waveform: the wall a segment edge
            # stops at, made visible.
            for xb in (to_x(0.0), to_x(self.audio_dur)):
                if 0 <= xb <= W:
                    c.create_line(xb, 0, xb, H, fill=COLORS["border"])
            span = self._audio_span()
            x0, x1 = to_x(self.audio_pos), to_x(self.audio_pos + span)
            if x1 > 0 and x0 < W:
                c.create_rectangle(max(0, x0), 0, min(W, x1), H,
                                   outline=COLORS["accent"], width=2)
            # The view's own clock, so a zoomed window says where in the file it is.
            c.create_text(4, H - 4, anchor="sw", text=f"{v0:.2f} s",
                          fill=COLORS["text_muted"], font=(FONT_FAMILY, 8))
            c.create_text(W - 4, H - 4, anchor="se", text=f"{v0 + vd:.2f} s",
                          fill=COLORS["text_muted"], font=(FONT_FAMILY, 8))
            zoomed = f"   ·   view {vd:.1f} s of {self.audio_dur:.1f} s" \
                if self.audio_view is not None else ""
            self.audio_pos_label.configure(
                text=f"{self.audio_pos:6.2f} s  →  {self.audio_pos + span:6.2f} s   "
                     f"of {self.audio_dur:.1f} s{zoomed}")
        else:
            self.audio_pos_label.configure(text="")
        # The nav bar mirrors the view: thumb position and size are the window's place in the
        # WORLD (file + margins). Unzoomed it fills the trough and there is nothing to drag.
        if hasattr(self, "audio_scroll"):
            if self.audio_dur:
                w0, wl = self._audio_world()
                self.audio_scroll.set((v0 - w0) / wl, (v0 + vd - w0) / wl)
            else:
                self.audio_scroll.set(0.0, 1.0)
        self._audio_refresh_planned_name()

    # -- audio: playback ----------------------------------------------------------------------------
    def audio_toggle_play(self, mode="segment"):
        """The two play BUTTONS: "file" from the playhead onward (finding the moment),
        "segment" exactly the export span (checking it). While playing they read Pause and DO
        pause — space resumes from there; pressing a play button starts its mode fresh."""
        if self._audio_playing:
            self._audio_pause()
            return
        self._audio_start(mode)

    def _audio_start(self, mode, start=None, rate=1.0, direction=1):
        """Start playback: mode x position x speed x direction — the one engine every play
        control drives. `start` None means the playhead (which for "segment" IS the segment
        start); J/K/L and pause/resume pass explicit positions and rates."""
        if not self.audio_src:
            return
        if self._audio_playing:
            self._audio_stop()
        if start is None:
            start = self.audio_pos
        start = max(0.0, min(float(start), self.audio_dur))
        if mode == "segment":
            seg_end = self.audio_pos + self._audio_span()
            # Resuming inside the span plays the REMAINDER; a stale resume point past the end
            # starts the segment over — the next tap should always produce sound.
            if not (self.audio_pos <= start < seg_end - 0.05):
                start = self.audio_pos
            span = seg_end - start
            self.audio_play_btn.configure(text="⏸  Pause")
        elif direction < 0:
            span = min(start, 600.0)            # backward: what lies behind the point
            self.audio_listen_btn.configure(text="⏸  Pause")
        else:
            # To the end of the recording, capped at ten minutes of extraction — nobody scrubs
            # longer than that in one sitting, and the temp wav stays sane.
            span = min(max(0.1, self.audio_dur - start), 600.0)
            self.audio_listen_btn.configure(text="⏸  Pause")
        if span <= 0.02:
            return
        # Live volume when the backend allows it — the slider then works MID-play. Only when it
        # doesn't (non-Windows external player) is the gain baked into the preview file.
        gain = 1.0 if set_live_volume(self.audio_volume_var.get()) \
            else max(0.0, float(self.audio_volume_var.get())) / 100.0
        self._audio_playing = True
        self._audio_play_mode = mode
        self._audio_play_rate = float(rate)
        self._audio_play_dir = 1 if direction >= 0 else -1
        self._audio_play_from = start
        self._audio_resume = None
        if rate != 1.0 or direction < 0:
            self.audio_status.configure(
                text=f"{'◀' if direction < 0 else '▶'} {rate:g}x", fg=COLORS["text_secondary"])
        threading.Thread(target=self._audio_play_worker,
                         args=(start, span, gain, float(rate), self._audio_play_dir),
                         daemon=True).start()

    @staticmethod
    def _atempo_chain(rate):
        """ffmpeg's atempo tops out at 2.0 per stage, so 3x is a chain. Returns filter parts."""
        parts = []
        while rate > 2.0:
            parts.append("atempo=2.0")
            rate /= 2.0
        if abs(rate - 1.0) > 1e-6:
            parts.append(f"atempo={rate:g}")
        return parts

    def _audio_play_worker(self, start, span, gain, rate=1.0, direction=1):
        err = None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_voice_preview.wav")
            # Backward (J) extracts what lies BEHIND the point and reverses it; speed is an
            # atempo chain either way. All of it happens in the extraction, because winsound
            # can only ever play a file forward at 1x.
            ss = start - span if direction < 0 else start
            filters = [f"volume={gain:.3f}"]
            if direction < 0:
                filters.append("areverse")
            filters += self._atempo_chain(rate)
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{ss:.3f}", "-t", f"{span:.3f}", "-i", self.audio_src,
                      "-vn", "-af", ",".join(filters), "-ac", "2", "-ar", "44100",
                      "-c:a", "pcm_s16le", wav])
            if p.returncode != 0 or not os.path.exists(wav):
                raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:]
                                   or "could not extract the audio")
            _play_wav(wav)
        except Exception as exc:
            err = str(exc)
        self.root.after(0, self._audio_play_done, span / max(rate, 1e-6), err)

    def _audio_play_done(self, span, err):
        if err:
            self._audio_stop()
            messagebox.showerror("Gizmo — cannot play that audio", err)
            return
        # winsound only starts once the extraction lands, so the cursor clock starts HERE, not
        # when the button was pressed — an early t0 would draw the cursor ahead of the sound.
        import time as _time
        self._audio_play_t0 = _time.monotonic()
        self._audio_play_job = self.root.after(int(span * 1000) + 200, self._audio_stop)
        self._audio_cursor_tick()

    def _audio_cursor_tick(self):
        """The moving playback line — wall-clock driven, since winsound reports nothing. Close
        enough for scrubbing by ear, which is what it exists for."""
        if not self._audio_playing:
            return
        cur = self._audio_playback_pos()
        c = self.audio_canvas
        c.delete("playcursor")
        if self.audio_dur:
            v0, vd = self._audio_view_bounds()
            # Listening past the edge of a zoomed view: pan the window along with the sound
            # (either direction), so scrub-by-ear works zoomed in too.
            if self.audio_view is not None and 0 < cur < self.audio_dur:
                if cur > v0 + vd:
                    self.audio_view = (min(cur - vd * 0.1, self.audio_dur - vd), vd)
                    self._audio_refresh()
                elif cur < v0:
                    self.audio_view = (max(0.0, cur - vd * 0.9), vd)
                    self._audio_refresh()
                v0, vd = self._audio_view_bounds()
            x = (cur - v0) / vd * self.AUDIO_WAVE_W
            if 0 <= x <= self.AUDIO_WAVE_W:
                c.create_line(x, 0, x, self.AUDIO_WAVE_H, fill=COLORS["success"], width=2,
                              tags="playcursor")
        self._audio_cursor_job = self.root.after(50, self._audio_cursor_tick)

    def _audio_playback_pos(self):
        """Where the sound IS right now — direction- and speed-aware wall-clock estimate."""
        import time as _time
        elapsed = _time.monotonic() - self._audio_play_t0
        cur = self._audio_play_from + self._audio_play_dir * self._audio_play_rate * elapsed
        return max(0.0, min(cur, self.audio_dur))

    def _audio_stop(self):
        for attr in ("_audio_play_job", "_audio_cursor_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)
        _stop_wav()
        self._audio_playing = False
        self._audio_play_mode = None
        self._audio_play_rate = 1.0
        self._audio_play_dir = 1
        try:
            if self.audio_play_btn.winfo_exists():
                self.audio_play_btn.configure(text="▶  Play segment")
                self.audio_listen_btn.configure(text="▶  Play from here")
                self.audio_canvas.delete("playcursor")
        except (AttributeError, tk.TclError):
            pass                              # closing before the tab was built

    # -- audio: pause / resume and the J-K-L shuttle -----------------------------------------------
    def _audio_pause(self):
        """Freeze where the sound is; space (or K) again resumes from exactly there."""
        if not self._audio_playing:
            return
        mode, cur = self._audio_play_mode, self._audio_playback_pos()
        self._audio_stop()
        self._audio_resume = (mode, cur)
        self.audio_status.configure(text=f"paused at {cur:.2f} s — space resumes",
                                    fg=COLORS["text_secondary"])

    def _audio_space(self):
        """Space: play / pause / resume — the one key that always does the obvious thing.
        Idle it plays the segment from its start; playing it pauses; paused it resumes."""
        if self._audio_playing:
            self._audio_pause()
        elif self._audio_resume is not None:
            mode, cur = self._audio_resume
            self._audio_start(mode, start=cur)
        else:
            self._audio_start("segment")

    _SHUTTLE_RATES = (1.0, 1.5, 2.0, 3.0)

    def _audio_shuttle(self, direction):
        """The editing-suite J/L shuttle: tap to play that way, tap again to go faster
        (1 -> 1.5 -> 2 -> 3x), from wherever the sound currently is."""
        if self._audio_playing and self._audio_play_dir == direction \
                and self._audio_play_mode == "file":
            rates = self._SHUTTLE_RATES
            i = min(len(rates) - 1,
                    next((n for n, r in enumerate(rates)
                          if r >= self._audio_play_rate - 1e-6), 0) + 1)
            self._audio_start("file", start=self._audio_playback_pos(),
                              rate=rates[i], direction=direction)
            return
        start = (self._audio_playback_pos() if self._audio_playing
                 else (self._audio_resume[1] if self._audio_resume else self.audio_pos))
        self._audio_start("file", start=start, rate=1.0, direction=direction)

    # -- audio: whisper -----------------------------------------------------------------------------
    def audio_transcribe(self):
        """Whisper the marked segment and append `saying "…"` to the caption.

        transformers is already in Fizgig's venv (it runs the EN→ZH translator the same way), so
        no new dependency — just a one-time ~300 MB model download on first use, or none at all
        when Preferences' downloader already fetched it (see local_whisper_dir)."""
        if self._whisper_busy or not self.audio_src:
            return
        self._whisper_busy = True
        # Decided once per click, not per 400 ms tick — the check walks the HF cache dir.
        self._whisper_ready = hasattr(self, "_whisper_pipe") or local_whisper_dir() is not None
        self.audio_whisper_btn.configure(state=tk.DISABLED, text="⏳ Transcribing…")
        self._whisper_dots = 0
        self._whisper_progress_tick()
        lang = self.audio_lang_var.get()
        threading.Thread(target=self._whisper_worker,
                         args=(self.audio_pos, self._audio_span(),
                               None if lang == "Auto detect" else lang.lower()),
                         daemon=True).start()

    def _whisper_progress_tick(self):
        """A visibly alive status line while Whisper works — the first run downloads ~300 MB
        and a silent, disabled button for that long reads as a hang."""
        if not self._whisper_busy:
            return
        self._whisper_dots = (self._whisper_dots + 1) % 4
        base = ("listening" if getattr(self, "_whisper_ready", False)
                else "listening — first use downloads ~300 MB, please wait")
        self.audio_status.configure(text=base + "." * self._whisper_dots,
                                    fg=COLORS["text_secondary"])
        self.root.after(400, self._whisper_progress_tick)

    @staticmethod
    def _whisper_degenerate(text, span):
        """Whisper's two classic failures, caught by arithmetic rather than trust.

        A wrong language guess feeds its repetition loop, and the result is paragraphs of the
        same phrase — from 4.5 seconds of audio, which cannot physically hold more than ~30
        characters a second of speech. A loop also has almost no unique words. Either signal
        means the transcript is hallucinated, not heard."""
        words = text.split()
        if len(text) > max(80, span * 30):
            return True
        if len(words) >= 10 and len(set(w.lower() for w in words)) / len(words) < 0.4:
            return True
        return False

    def _whisper_worker(self, start, span, lang=None):
        """lang None = auto-detect, with an English-forced retry if the output degenerates
        (a wrong guess is the usual cause of the loop). A chosen language skips the guess
        entirely — and the retry, since there is no better guess left to make."""
        text, err = None, None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_whisper.wav")
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", self.audio_src,
                      "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
            if p.returncode != 0:
                raise RuntimeError("could not extract the segment")
            from transformers import pipeline
            if not hasattr(self, "_whisper_pipe"):
                # Local snapshot first: a path never goes online. The hub name is the
                # fallback for the one-time first-use download.
                self._whisper_pipe = pipeline("automatic-speech-recognition",
                                              model=local_whisper_dir() or _WHISPER_REPO,
                                              device=-1)

            # Raw samples, not the file path: handed a path, transformers shells out to its
            # own ffmpeg to decode it — without the no-window flag, so a black console
            # flashes over the app on every transcription. We just extracted this wav with
            # our flashless ffmpeg; reading it back is a wave-module one-liner.
            import wave as _wave

            import numpy as _np
            with _wave.open(wav) as _w:
                _raw = _np.frombuffer(_w.readframes(_w.getnframes()),
                                      dtype=_np.int16).astype(_np.float32) / 32768.0
            _sample = {"raw": _raw, "sampling_rate": 16000}

            def hear(language):
                kw = ({"generate_kwargs": {"language": language, "task": "transcribe"}}
                      if language else {})
                return (self._whisper_pipe(dict(_sample), **kw).get("text") or "").strip()

            text = hear(lang)
            if lang is None and self._whisper_degenerate(text, span):
                text = hear("english")
            if self._whisper_degenerate(text, span):
                text = None
                err = ("Whisper looped on this segment — it hallucinated repeating text "
                       "instead of hearing it, which it does now and then on short or "
                       "ambiguous audio.\n\nNudge the start a little (even 100 ms changes the "
                       "outcome) and try again, pick the language from the dropdown, or just "
                       "type the words.")
        except Exception as exc:
            err = (f"Whisper could not run: {exc}\n\nIt needs one ~300 MB download the first "
                   f"time (internet required once — or fetch it in advance with Preferences → "
                   f"⬇ Download models for me, and it works offline forever after). The "
                   f"caption still works without it — describe the voice and skip the "
                   f"transcript.")
        self.root.after(0, self._whisper_done, text, err)

    def _whisper_done(self, text, err):
        self._whisper_busy = False
        self.audio_whisper_btn.configure(state=tk.NORMAL, text="🎤 Transcribe")
        if err:
            self.audio_status.configure(text="transcription failed", fg=COLORS["error"])
            messagebox.showerror("Gizmo — Whisper", err)
            return
        self.audio_status.configure(text="", fg=COLORS["text_secondary"])
        if not text:
            self.audio_status.configure(text="Whisper heard no words in that segment",
                                        fg=COLORS["text_secondary"])
            return
        # The box is CLEARED first, always — the transcript describes this segment and this
        # segment only, and anything left over from the previous one (its description, its
        # words) would silently ride along into the caption. Type the voice description in
        # front of the transcript after it lands.
        self.audio_caption.delete("1.0", tk.END)
        self.audio_caption.insert("1.0", f'saying "{text}"')

    # -- audio: queue + export ----------------------------------------------------------------------
    def _audio_caption_text(self):
        cap = " ".join(self.audio_caption.get("1.0", tk.END).split())
        trig = self.audio_trigger_var.get().strip().rstrip(",")
        if trig and cap:
            return f"{trig}, {cap}"
        return trig or cap

    def _audio_claimed(self):
        return {os.path.basename(j["dst"]) for j in self.audio_queue}

    def _audio_refresh_planned_name(self):
        # (the queue card shows names on Add) — the music lyrics warning follows the span
        if self._music_mode():
            self._music_refresh_lyrics_note()

    def audio_add_to_queue(self):
        if not self.audio_src:
            return
        out_dir = self.audio_out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Gizmo", "Pick a save folder first.")
            return
        if self._music_mode():
            self._music_add_to_queue(out_dir)
            return
        if not self._audio_caption_text():
            # A voice item with no caption is dropped by Fizgig's caption filter — refusing at
            # the mark is kinder than a warning at training time.
            messagebox.showwarning("Gizmo — caption first",
                                   "Write the caption (or at least a trigger word) before "
                                   "marking — every exported segment needs its .txt to train.")
            return
        frames = self.audio_frames_var.get()
        span = self._audio_span()
        if self.audio_pos + span > self.audio_dur + 0.01:
            messagebox.showwarning("Gizmo", "That segment runs past the end of the recording.")
            return
        dst = voice_output_name(self.audio_src, out_dir, self._audio_claimed())
        self.audio_queue.append({
            "src": self.audio_src, "dst": dst, "start": self.audio_pos, "frames": frames,
            "caption": self._audio_caption_text(), "done": False,
        })
        self._audio_refresh_queue_box()
        self.audio_status.configure(text=f"marked {os.path.basename(dst)}",
                                    fg=COLORS["text_secondary"])

    def _audio_refresh_queue_box(self):
        self.audio_queue_box.delete(0, tk.END)
        for j in self.audio_queue:
            secs = j["len"] if "len" in j else hop_exact_samples(j["frames"]) / AUDIO_SAMPLE_RATE
            mark = "✓" if j["done"] else "·"
            cap = j["caption"][:46] + ("…" if len(j["caption"]) > 46 else "")
            self.audio_queue_box.insert(
                tk.END, f'{mark} {os.path.basename(j["dst"]):<22} {j["start"]:7.2f}s '
                        f'{secs:4.1f}s  {cap}')

    def _audio_recall(self):
        sel = self.audio_queue_box.curselection()
        if not sel:
            return
        j = self.audio_queue[sel[0]]
        if j["done"]:
            return
        if "len" in j:                          # a music section
            self.music_lyrics.delete("1.0", tk.END)
            self.music_lyrics.insert("1.0", j["lyrics"])
            self._music_lyrics_from_song = False
            self.audio_pos = j["start"]
            self._music_set_len(j["len"])
            self._audio_seek(j["start"])
            return
        self.audio_frames_var.set(j["frames"])
        self.audio_caption.delete("1.0", tk.END)
        cap = j["caption"]
        trig = self.audio_trigger_var.get().strip().rstrip(",")
        if trig and cap.startswith(trig + ", "):
            cap = cap[len(trig) + 2:]
        self.audio_caption.insert("1.0", cap)
        self._audio_seek(j["start"])

    def _audio_remove_selected(self):
        sel = self.audio_queue_box.curselection()
        if sel:
            del self.audio_queue[sel[0]]
            self._audio_refresh_queue_box()

    def _audio_clear_queue(self):
        self.audio_queue = [j for j in self.audio_queue if j["done"]]
        self._audio_refresh_queue_box()

    def audio_export_queue(self):
        jobs = [j for j in self.audio_queue if not j["done"]]
        if not jobs:
            self.audio_status.configure(text="nothing waiting", fg=COLORS["text_secondary"])
            return
        self.audio_export_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._audio_export_worker, args=(jobs,), daemon=True).start()

    def _music_add_to_queue(self, out_dir):
        cap = self._music_caption_text()
        if not cap:
            messagebox.showwarning("Gizmo — caption first",
                                   "Write a trigger word or describe the music before marking "
                                   "— every section needs its caption .txt to train.")
            return
        span = self.music_len
        if self.audio_pos + span > self.audio_dur + 0.01:
            messagebox.showwarning("Gizmo", "That section runs past the end of the song.")
            return

        def secs(var):
            try:
                return max(0.0, float(var.get()))
            except ValueError:
                return 0.0
        dst = voice_output_name(self.audio_src, out_dir, self._audio_claimed())
        lyrics = self.music_lyrics.get("1.0", tk.END).strip()
        self.audio_queue.append({
            "src": self.audio_src, "dst": dst, "start": self.audio_pos, "len": span,
            "caption": cap, "lyrics": lyrics, "done": False,
            # Fades only where the section cuts into the song.
            "fade_in": secs(self.music_fade_in_var) if self.audio_pos > 0.05 else 0.0,
            "fade_out": (secs(self.music_fade_out_var)
                         if self.audio_pos + span < self.audio_dur - 0.05 else 0.0),
        })
        self._audio_refresh_queue_box()
        self.audio_status.configure(text=f"marked {os.path.basename(dst)}",
                                    fg=COLORS["text_secondary"])

    def _music_export_one(self, j):
        """A section as 48 kHz stereo wav, its caption .txt, its lyrics, and the song's
        bpm/key .json when it has one."""
        span = j["len"]
        fades = []
        if j["fade_in"] > 0:
            fades.append(f"afade=t=in:st=0:d={min(j['fade_in'], span / 4):.3f}")
        if j["fade_out"] > 0:
            d = min(j["fade_out"], span / 4)
            fades.append(f"afade=t=out:st={span - d:.3f}:d={d:.3f}")
        p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                  "-ss", f'{j["start"]:.3f}', "-t", f"{span:.3f}", "-i", j["src"], "-vn",
                  *(["-af", ",".join(fades)] if fades else []),
                  "-ac", "2", "-ar", str(MUSIC_SAMPLE_RATE), "-c:a", "pcm_s16le", j["dst"]])
        if p.returncode != 0 or not os.path.exists(j["dst"]):
            raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-200:])
        stem = os.path.splitext(j["dst"])[0]
        with open(stem + ".txt", "w", encoding="utf-8") as f:
            f.write(j["caption"])
        if j["lyrics"]:
            with open(stem + ".lyrics.txt", "w", encoding="utf-8") as f:
                f.write(j["lyrics"] + "\n")
        meta = os.path.splitext(j["src"])[0] + ".json"
        if os.path.isfile(meta):
            shutil.copyfile(meta, stem + ".json")

    def _audio_export_worker(self, jobs):
        wrote, failed = 0, []
        for j in jobs:
            try:
                os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
                if "len" in j:
                    self._music_export_one(j)
                    j["done"] = True
                    wrote += 1
                    self.root.after(0, self._audio_refresh_queue_box)
                    continue
                n = hop_exact_samples(j["frames"])
                # aresample first, then trim BY SAMPLE COUNT at the target rate — the exact
                # length Fizgig's cache validation demands, no tolerance spent.
                p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                          "-ss", f'{j["start"]:.3f}', "-i", j["src"], "-vn",
                          "-af", f"aresample={AUDIO_SAMPLE_RATE},atrim=end_sample={n},"
                                 f"apad=whole_len={n}",
                          "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
                          "-c:a", "pcm_s16le", j["dst"]])
                if p.returncode != 0 or not os.path.exists(j["dst"]):
                    raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-200:])
                with open(os.path.splitext(j["dst"])[0] + ".txt", "w", encoding="utf-8") as f:
                    f.write(j["caption"])
                j["done"] = True
                wrote += 1
            except Exception as exc:
                failed.append(f'{os.path.basename(j["dst"])}: {exc}')
            self.root.after(0, self._audio_refresh_queue_box)
        def done():
            self.audio_export_btn.configure(state=tk.NORMAL)
            if failed:
                self.audio_status.configure(text=f"wrote {wrote}, {len(failed)} failed",
                                            fg=COLORS["error"])
                messagebox.showerror("Gizmo — some segments failed", "\n".join(failed[:6]))
            else:
                self.audio_status.configure(
                    text=f"wrote {wrote} {'section' if 'len' in jobs[0] else 'segment'}(s) "
                         f"+ captions", fg=COLORS["success"])
        self.root.after(0, done)

    def _audio_pick_output(self):
        d = filedialog.askdirectory(title="Where should the song sections go?"
                                    if self._music_mode() else
                                    "Where should the voice segments go?",
                                    initialdir=self.audio_out_var.get() or os.path.expanduser("~"))
        if d:
            self.audio_out_var.set(d)

    def _audio_open_folder(self):
        d = self.audio_out_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Gizmo", "Nothing there yet — export a segment first.")
            return
        try:
            if os.name == "nt":
                os.startfile(d)                                     # noqa: S606
            else:
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot open {d}\n\n{exc}")

    def _audio_set_enabled(self, on):
        state = tk.NORMAL if on else tk.DISABLED
        for w in (self.audio_listen_btn, self.audio_play_btn, self.audio_add_btn,
                  self.audio_export_btn, self.audio_whisper_btn):
            w.configure(state=state)

    # -- the voice recorder card --------------------------------------------------------------------
    def open_recorder(self):
        """Swap the Voice tab's source card for the record card. Built once per session and
        then shown again with its takes intact — the whole point of it not being a popup."""
        self.notebook.select(self._audio_tab)
        if getattr(self, "_recorder", None) is None:
            self._recorder = GizmoRecorder(self)
        self._recorder.show()

    def open_settings(self):
        """The ⚙ popup: microphone + default Whisper language, persisted beside gizmo.py."""
        win = tk.Toplevel(self.root)
        win.title("Gizmo — settings")
        win.configure(bg=COLORS["bg_deep"], padx=18, pady=16)
        win.transient(self.root)
        win.resizable(False, False)

        tk.Label(win, text="Microphone", font=(FONT_FAMILY, 10, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 2))
        mic_var = tk.StringVar(value=self.settings.get("microphone", ""))
        mic_combo = ttk.Combobox(win, textvariable=mic_var, width=42, style="G.TCombobox")
        mic_combo.grid(row=1, column=0, sticky=tk.W)
        ToolTip(mic_combo, "The input device the recorder captures from. Scan lists what "
                           "this machine offers; on Linux, 'default' follows the system mic.")

        def scan():
            scan_btn.configure(state=tk.DISABLED, text="scanning…")

            def worker():
                devices = list_capture_devices(self.ffmpeg)

                def done():
                    scan_btn.configure(state=tk.NORMAL, text="Scan")
                    mic_combo.configure(values=devices or ["default"])
                    if devices and not mic_var.get():
                        mic_var.set(devices[0])
                    if not devices:
                        tk.Label(win, text="No microphone found on this machine.",
                                 font=(FONT_FAMILY, 9), bg=COLORS["bg_deep"],
                                 fg=COLORS["warning"]).grid(row=2, column=0, sticky=tk.W)
                self.root.after(0, done)
            threading.Thread(target=worker, daemon=True).start()

        scan_btn = self._button(win, "Scan", scan, pad=10, tip="Look for microphones again.")
        scan_btn.grid(row=1, column=1, padx=(8, 0))

        tk.Label(win, text="Whisper language", font=(FONT_FAMILY, 10, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).grid(
            row=3, column=0, sticky=tk.W, pady=(12, 2))
        lang_var = tk.StringVar(value=self.settings.get("whisper_language", "Auto detect"))
        lang_combo = ttk.Combobox(win, textvariable=lang_var, width=42, state="readonly",
                                  style="G.TCombobox", values=WHISPER_LANGUAGES)
        lang_combo.grid(row=4, column=0, sticky=tk.W)
        ToolTip(lang_combo, "The default for the Transcribe dropdown. Pick your language "
                            "once and Whisper stops guessing (a wrong guess on a short "
                            "segment is what makes it hallucinate).")

        rand_var = tk.BooleanVar(value=bool(self.settings.get("random_delivery", True)))
        rand_cb = tk.Checkbutton(win, text="Vary the delivery style between takes",
                                 variable=rand_var, bg=COLORS["bg_deep"],
                                 fg=COLORS["text_primary"], selectcolor=COLORS["bg_surface"],
                                 activebackground=COLORS["bg_deep"],
                                 activeforeground=COLORS["text_primary"],
                                 font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                 cursor="hand2")
        rand_cb.grid(row=5, column=0, sticky=tk.W, pady=(12, 0))
        ToolTip(rand_cb, "After every take the recorder rolls a random delivery prompt "
                         "(cheerfully, wearily, quietly…) so the dataset's emotional range "
                         "fills itself. Off keeps whatever the delivery dropdown says — and "
                         "either way the dropdown can be changed by hand for the next take.")

        trim_var = tk.BooleanVar(value=bool(self.settings.get("trim_takes", True)))
        trim_cb = tk.Checkbutton(win, text="Trim silence and key clicks from takes",
                                 variable=trim_var, bg=COLORS["bg_deep"],
                                 fg=COLORS["text_primary"], selectcolor=COLORS["bg_surface"],
                                 activebackground=COLORS["bg_deep"],
                                 activeforeground=COLORS["text_primary"],
                                 font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                 cursor="hand2")
        trim_cb.grid(row=6, column=0, sticky=tk.W, pady=(4, 0))
        ToolTip(trim_cb, "Each take is trimmed to its speech: the silence either side and "
                         "the hold key's click are cut off automatically. Turn off if takes "
                         "come out clipped on your microphone — the untrimmed take keeps "
                         "everything between a quarter-second before the press and after the "
                         "release.")

        def save():
            self.settings["microphone"] = mic_var.get().strip()
            self.settings["whisper_language"] = lang_var.get()
            self.settings["random_delivery"] = bool(rand_var.get())
            self.settings["trim_takes"] = bool(trim_var.get())
            save_settings(self.settings)
            if hasattr(self, "audio_lang_var"):
                self.audio_lang_var.set(lang_var.get())
            win.destroy()

        row = tk.Frame(win, bg=COLORS["bg_deep"])
        row.grid(row=7, column=0, columnspan=2, sticky=tk.E, pady=(16, 0))
        self._button(row, "Save", save, "accent", pad=12,
                     tip="Keep these — they persist in gizmo_settings.json.").pack(side=tk.LEFT)
        self._button(row, "Cancel", win.destroy, pad=12,
                     tip="Close without changing anything.").pack(side=tk.LEFT, padx=(8, 0))
        scan()

    # -- cost -------------------------------------------------------------------------------------
    def _refresh_length_note(self):
        """What this length can be trained at, per card. The number people actually need is not
        the token count — it is whether their own card can run it."""
        frames = self._frames()
        if MODEL.get("note"):
            text, warn = MODEL["note"](frames)
            self.len_note.configure(text=text, fg=COLORS["warning"] if warn else COLORS["text_explain"])
            return
        by_card = CLIP_VRAM.get(frames, {})
        parts = []
        for gb in (16, 24, 32):
            mp = by_card.get(gb)
            parts.append(f"{gb} GB: {'—' if mp is None else f'up to {mp:g} MP'}")
        note = "Trainable at  " + "   ·   ".join(parts)
        if by_card.get(16) is None:
            note += "   (a 16 GB card cannot place a clip this long at all)"
        self.len_note.configure(
            text=note + "\nEstimated from Fizgig's own swap planner and not yet confirmed by a "
                        "real run at these lengths — where it is wrong it will be optimistic.",
            fg=COLORS["warning"] if by_card.get(24) is None else COLORS["text_explain"])

    def _refresh_cost(self):
        self._refresh_length_note()
        frames = self._frames()
        w, h = self._size()
        if not w:
            self.size_label.configure(text="—")
            self.cost_label.configure(text="")
            return
        self.size_label.configure(
            text=f"{w} x {h}   ({w * h / 1e6:.2f} MP)"
                 + ("   — the crop, at its own resolution" if self.crop
                    else "   — the whole frame, at its own resolution"))
        # Costed at the TRAINING resolution, not at the size on disk. The stored size is native
        # and training rescales it, so counting tokens at 1888x1056 would report a number no run
        # ever pays — the useful figure is what a step costs at the megapixels you train at.
        lat = LATENT_FRAMES[frames]
        tok = int(0.25e6 / (SIZE_STEP * SIZE_STEP)) * lat
        self.cost_label.configure(
            text=f"{frames} frames is {lat} latent frames — about {tok:,} tokens a step at "
                 f"0.25 MP training, {lat}x a still, and four times that at 1 MP. Attention "
                 f"scales with the square of it, which is what the line above is really saying.",
            fg=COLORS["warning"] if lat > 12 else COLORS["text_explain"])
        if self.src:
            self._update_pos_labels()

    # -- saving -----------------------------------------------------------------------------------
    def pick_output(self):
        d = filedialog.askdirectory(title="Where should the clips go?",
                                    initialdir=self.out_var.get() or os.path.expanduser("~"))
        if d:
            self.out_var.set(d)

    def open_output_folder(self):
        d = self.out_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Gizmo", "Nothing there yet — export a clip first.")
            return
        try:
            if os.name == "nt":
                os.startfile(d)                                     # noqa: S606
            else:
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot open {d}\n\n{exc}")

    # -- the queue ---------------------------------------------------------------------------------
    def _claimed_names(self):
        """Names already spoken for by the queue. output_name only checks the disk, so without
        this two clips added before either is exported would both be handed _01."""
        return {os.path.basename(j["dst"]) for j in self.queue}

    def _planned_job(self, start_frame=None):
        """The job Add would create right now, or None if there isn't one. start_frame overrides
        the playhead — auto-chop queues jobs at its own positions with everything else as set."""
        out_dir = self.out_var.get().strip()
        if not self.src or not out_dir:
            return None
        w, h = self._size()
        if not w:
            return None
        muted = bool(self.mute_var.get())
        sf = self.frame_pos if start_frame is None else start_frame
        return {
            "src": self.src, "dst": output_name(self.src, out_dir, muted, self._claimed_names()),
            "start_frame": sf, "start": sf / self.info["fps"],
            "frames": self._frames(), "w": w, "h": h,
            "keep_every": self._keep_every(), "muted": muted,
            "crop": self.crop, "sar": self.info["sar"],
            # A muted clip still carries its audio: the _mute suffix is the instruction, and
            # keeping the track means the decision is reversible by rename, not by re-export.
            "with_audio": bool(self.info["has_audio"]),
            "motion_index": self.motion_box.current(),
            "done": False, "error": None,
        }

    def _refresh_planned_name(self, *_a):
        job = self._planned_job()
        self.name_label.configure(
            text=(f"next:  {os.path.basename(job['dst'])}" if job else ""))

    def add_to_queue(self):
        if not self.src:
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Gizmo", "Choose a folder to save the clips into first.")
            return
        job = self._planned_job()
        if job is None:
            return
        # Refused at ADD time rather than at export, so a queue of thirty does not stop halfway
        # for something that was knowable the moment it was marked.
        duration = self.info["duration"] or 0
        k = job["keep_every"]
        span = (job["frames"] * k / self.info["fps"]) if k else job["frames"] / FPS
        if duration and job["start"] + span > duration + 1e-3:
            messagebox.showwarning(
                "Gizmo — that clip runs off the end",
                f"The source ends at {duration:.2f} s, and this clip would need up to "
                f"{job['start'] + span:.2f} s.\n\nMove the playhead earlier, or choose a shorter "
                f"length.")
            return
        self.queue.append(job)
        self._refresh_queue_box()
        self._refresh_planned_name()
        self.status.configure(text=f"Queued {os.path.basename(job['dst'])}",
                              fg=COLORS["text_secondary"])

    def _queue_row(self, i, job):
        mark = "✗" if job["error"] else ("✓" if job["done"] else "·")
        sound = "muted" if job["muted"] else ("sound" if job["with_audio"] else "silent")
        slow = f" {self.info['fps'] / job['keep_every'] / FPS:.2f}x slow" \
            if job["keep_every"] and self.info else ""
        crop = "  cropped" if job.get("crop") else ""
        return (f"{mark} {i + 1:2d}  {os.path.basename(job['dst']):<28} "
                f"{job['frames']:>2}f  {job['w']}x{job['h']}  @{job['start']:6.2f}s  "
                f"{sound}{slow}{crop}")

    def _refresh_queue_box(self):
        self.queue_box.delete(0, tk.END)
        for i, job in enumerate(self.queue):
            self.queue_box.insert(tk.END, self._queue_row(i, job))
            if job["error"]:
                self.queue_box.itemconfigure(i, foreground=COLORS["error"])
            elif job["done"]:
                self.queue_box.itemconfigure(i, foreground=COLORS["success"])
        pending = sum(1 for j in self.queue if not j["done"])
        self.export_btn.configure(
            text=f"Export queue ({pending})" if pending else "Export queue")
        self.queue_box.see(tk.END)

    def recall_selected(self):
        """Put the playhead and settings back where they were when a row was added — the undo
        for 'I queued that one two frames late'."""
        sel = self.queue_box.curselection()
        if not sel or self._busy:
            return
        job = self.queue[sel[0]]
        if job["done"]:
            return
        if job["src"] != self.src:
            messagebox.showinfo("Gizmo", "That clip came from a different source video.")
            return
        self.len_box.current(GRID_FRAMES.index(job["frames"]))
        # Crop first: it rebuilds the size menu, so restoring the size before it would be undone.
        self.crop = job.get("crop")
        self.crop_var.set(bool(self.crop))
        self._describe_crop()
        if 0 <= job["motion_index"] < len(self.motion_box["values"]):
            self.motion_box.current(job["motion_index"])
        self.mute_var.set(job["muted"])
        self.scale.set(job["start_frame"])
        self._refresh_cost()
        self._refresh_planned_name()

    def remove_selected(self):
        sel = self.queue_box.curselection()
        if not sel or self._busy:
            return
        if self.queue[sel[0]]["done"]:
            messagebox.showinfo("Gizmo", "That one is already written. Delete the file itself if "
                                         "you don't want it.")
            return
        del self.queue[sel[0]]
        self._refresh_queue_box()
        self._refresh_planned_name()

    def clear_queue(self):
        if self._busy:
            return
        self.queue = [j for j in self.queue if j["done"]]
        self._refresh_queue_box()
        self._refresh_planned_name()

    # -- auto-chop ---------------------------------------------------------------------------------
    # The alternative to scrubbing an hour of footage by hand: detect the cuts once, offer every
    # segment as a thumbnail, and let keep-or-skip be a click instead of a scrub-mark-repeat
    # loop. Everything kept lands in the ordinary queue as an ordinary job — auto-chop has no
    # export path of its own, so whatever the queue learns, this inherits.

    _CHOP_THUMB = (200, 112)

    def open_autochop(self):
        if not self.src or not self.ffmpeg:
            return
        if self._chop_win is not None and self._chop_win.winfo_exists():
            self._chop_win.lift()
            return
        if not self.out_var.get().strip():
            messagebox.showwarning("Gizmo", "Choose a folder to save the clips into first.")
            return

        win = tk.Toplevel(self.root)
        self._chop_win = win
        self._chop_alive = True
        self._chop_cancelled = False
        self._chop_cands = []
        self._chop_blank = tk.PhotoImage(width=self._CHOP_THUMB[0],
                                         height=self._CHOP_THUMB[1])
        win.title(f"Auto-chop — {os.path.basename(self.src)}")
        win.configure(bg=COLORS["bg_deep"])
        win.geometry("%dx%d" % _scaled(win, 920, 700))
        win.minsize(*_scaled(win, 760, 480))
        win.protocol("WM_DELETE_WINDOW", self._autochop_close)

        top = tk.Frame(win, bg=COLORS["bg_deep"])
        top.pack(fill=tk.X, padx=14, pady=(12, 4))
        tk.Label(top, text="Auto-chop", font=(FONT_FAMILY, 13, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor="w")
        tk.Label(top, wraplength=860, justify=tk.LEFT, font=(FONT_FAMILY, 9),
                 bg=COLORS["bg_deep"], fg=COLORS["text_explain"],
                 text="Scan finds the cuts, then every segment that fits a clip appears below. "
                      "Click a segment to keep or skip it; double-click to jump the main window "
                      "there for a closer look. Kept segments are queued exactly as if you had "
                      "pressed Add at each one — same Length, Motion, Crop and Sound as the "
                      "settings card, so set those first.").pack(anchor="w", pady=(2, 0))

        ctl = tk.Frame(win, bg=COLORS["bg_deep"])
        ctl.pack(fill=tk.X, padx=14, pady=(6, 4))
        tk.Label(ctl, text="Cut detection", font=(FONT_FAMILY, 9), bg=COLORS["bg_deep"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT)
        self._chop_sense = ttk.Combobox(ctl, state="readonly", style="G.TCombobox", width=36,
                                        values=[name for name, _ in SCENE_SENSITIVITIES])
        self._chop_sense.current(0)
        self._chop_sense.pack(side=tk.LEFT, padx=(6, 16))
        ToolTip(self._chop_sense,
                "How different two frames have to look before it counts as a cut.\n\n"
                "Normal suits most footage. If the grid comes back with scenes visibly glued "
                "together, scan again with More cuts; if one real scene arrives as confetti, "
                "Fewer cuts.")
        self._chop_fill_var = tk.BooleanVar(value=True)
        fill_cb = tk.Checkbutton(ctl, text="Several clips from long scenes",
                                 variable=self._chop_fill_var, bg=COLORS["bg_deep"],
                                 fg=COLORS["text_primary"], selectcolor=COLORS["bg_surface"],
                                 activebackground=COLORS["bg_deep"],
                                 activeforeground=COLORS["text_primary"],
                                 font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                 cursor="hand2")
        fill_cb.pack(side=tk.LEFT, padx=(0, 16))
        ToolTip(fill_cb,
                "On: a scene longer than one clip yields back-to-back clips through its whole "
                "length — the entire video gets offered, and you cull.\n\n"
                "Off: one clip from the top of each scene — a quick sampler of the footage.")
        self._chop_scan_btn = self._button(ctl, "Scan", self._autochop_scan, "accent", pad=16,
                                           tip="Decode the source once and find its cuts. A few "
                                               "seconds per minute of footage.")
        self._chop_scan_btn.pack(side=tk.LEFT)
        self._chop_cancel_btn = self._button(ctl, "Cancel scan", self._autochop_cancel_scan,
                                             pad=10, tip="Stop the scan.")

        self._chop_status = tk.Label(win, text="", font=(FONT_FAMILY, 10), anchor="w",
                                     bg=COLORS["bg_deep"], fg=COLORS["text_secondary"])
        self._chop_status.pack(fill=tk.X, padx=14)

        body = tk.Frame(win, bg=COLORS["bg_deep"])
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 0))
        self._chop_canvas = tk.Canvas(body, bg=COLORS["bg_deep"], highlightthickness=0)
        sb = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self._chop_canvas.yview)
        self._chop_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._chop_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._chop_grid = tk.Frame(self._chop_canvas, bg=COLORS["bg_deep"])
        self._chop_canvas.create_window((0, 0), window=self._chop_grid, anchor="nw")
        self._chop_grid.bind("<Configure>", lambda _e: self._chop_canvas.configure(
            scrollregion=self._chop_canvas.bbox("all")))
        # Bound on the Toplevel, not bind_all: events from the cards bubble up to their own
        # toplevel, and the main window's wheel stays the main window's.
        win.bind("<MouseWheel>", lambda e: self._chop_canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units"))
        win.bind("<Button-4>", lambda _e: self._chop_canvas.yview_scroll(-1, "units"))
        win.bind("<Button-5>", lambda _e: self._chop_canvas.yview_scroll(1, "units"))

        foot = tk.Frame(win, bg=COLORS["bg_deep"])
        foot.pack(fill=tk.X, padx=14, pady=10)
        self._chop_keep_all = self._button(foot, "Keep all", lambda: self._autochop_set_all(True),
                                           pad=10, tip="Mark every segment as a keeper.")
        self._chop_keep_all.pack(side=tk.LEFT)
        self._chop_skip_all = self._button(foot, "Skip all", lambda: self._autochop_set_all(False),
                                           pad=10, tip="Unmark everything — then click just the "
                                                       "ones you want.")
        self._chop_skip_all.pack(side=tk.LEFT, padx=(8, 0))
        self._chop_count = tk.Label(foot, text="", font=(FONT_FAMILY, 10),
                                    bg=COLORS["bg_deep"], fg=COLORS["text_secondary"])
        self._chop_count.pack(side=tk.LEFT, padx=14)
        self._button(foot, "Close", self._autochop_close, pad=12,
                     tip="Close without queueing. Nothing has been written either "
                         "way.").pack(side=tk.RIGHT)
        self._chop_add_btn = self._button(foot, "➕  Queue the kept segments",
                                          self._autochop_add_kept, "accent",
                                          tip="Add every kept segment to the queue, then export "
                                              "from the main window as usual.")
        self._chop_add_btn.pack(side=tk.RIGHT, padx=(0, 8))
        for w in (self._chop_keep_all, self._chop_skip_all, self._chop_add_btn):
            w.configure(state=tk.DISABLED)
        self._chop_status.configure(text="Press Scan to find the cuts.")

    def _autochop_close(self):
        self._chop_alive = False
        if self._chop_proc is not None:
            try:
                self._chop_proc.terminate()
            except Exception:
                pass
            self._chop_proc = None
        if self._chop_win is not None:
            try:
                self._chop_win.destroy()
            except Exception:
                pass
            self._chop_win = None
        self._chop_cands = []

    def _autochop_cancel_scan(self):
        self._chop_cancelled = True
        if self._chop_proc is not None:
            try:
                self._chop_proc.terminate()
            except Exception:
                pass

    def _chop_after(self, fn, *args):
        """Marshal a worker result to the GUI thread, dropped silently if the window has gone —
        a scan finishing after Close must not raise into a dead Toplevel."""
        def run():
            if self._chop_alive and self._chop_win is not None and self._chop_win.winfo_exists():
                fn(*args)
        self.root.after(0, run)

    def _chop_span(self):
        """One clip's span in SOURCE seconds — the same arithmetic add_to_queue guards with:
        slow motion consumes keep_every source frames per output frame."""
        k = self._keep_every()
        return (self._frames() * k / self.info["fps"]) if k else self._frames() / FPS

    def _autochop_scan(self):
        if self._chop_proc is not None:
            return
        thr = SCENE_SENSITIVITIES[self._chop_sense.current()][1]
        self._chop_cancelled = False
        for w in self._chop_grid.winfo_children():
            w.destroy()
        self._chop_cands = []
        self._autochop_refresh_count()
        for w in (self._chop_keep_all, self._chop_skip_all, self._chop_add_btn):
            w.configure(state=tk.DISABLED)
        self._chop_scan_btn.configure(state=tk.DISABLED)
        self._chop_cancel_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._chop_status.configure(text="Scanning…", fg=COLORS["text_secondary"])
        threading.Thread(target=self._autochop_scan_worker,
                         args=(self.src, thr, self.info["duration"] or 0.0), daemon=True).start()

    def _autochop_scan_worker(self, src, threshold, duration):
        cuts, err = [], None
        try:
            cmd = build_scene_scan_command(self.ffmpeg, src, threshold)
            self._chop_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.PIPE, text=True,
                                               errors="replace", creationflags=_NO_WINDOW)
            last_shown = -5.0
            for line in self._chop_proc.stderr:
                t = parse_scene_time(line)
                if t is not None:
                    cuts.append(t)
                p = t if t is not None else parse_progress_time(line)
                if p is not None and p - last_shown >= 2.0 and duration:
                    last_shown = p
                    self._chop_after(self._chop_status.configure,
                                     {"text": f"Scanning…  {p / duration:4.0%} — "
                                              f"{len(cuts)} cuts so far"})
            rc = self._chop_proc.wait()
            # A terminated scan comes back non-zero with a partial cut list — that is a
            # cancellation, not a failure. -f null runs to EOF and exits 0 on success.
            if rc != 0 and not self._chop_cancelled:
                err = "ffmpeg could not scan this file"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        finally:
            self._chop_proc = None
        self._chop_after(self._autochop_scanned, cuts, err, self._chop_cancelled)

    def _autochop_scanned(self, cuts, err, cancelled=False):
        self._chop_cancel_btn.pack_forget()
        self._chop_scan_btn.configure(state=tk.NORMAL)
        if cancelled:
            self._chop_status.configure(text="Scan stopped.", fg=COLORS["text_secondary"])
            return
        if err:
            self._chop_status.configure(text=f"Scan failed — {err}", fg=COLORS["error"])
            return
        duration = self.info["duration"] or 0.0
        span = self._chop_span()
        starts = plan_autochop(cuts, duration, span, fill=bool(self._chop_fill_var.get()))
        note = ""
        if len(starts) > AUTOCHOP_MAX_SEGMENTS:
            note = (f"  (showing the first {AUTOCHOP_MAX_SEGMENTS} of {len(starts)} — a longer "
                    f"Length or Fewer cuts would bring the rest into reach)")
            starts = starts[:AUTOCHOP_MAX_SEGMENTS]
        if not starts:
            self._chop_status.configure(
                text=f"{len(cuts)} cuts found, but every scene is shorter than one "
                     f"{self._frames()}-frame clip ({span:.1f} s). A shorter Length on the "
                     f"settings card would fit more of them.", fg=COLORS["warning"])
            return
        self._chop_status.configure(
            text=f"{len(cuts)} cuts — {len(starts)} segments fit a {self._frames()}-frame "
                 f"clip{note}", fg=COLORS["text_secondary"])

        fps = self.info["fps"]
        cols = 4
        for i, s in enumerate(starts):
            cand = {"start": s, "keep": True, "photo": None}
            card = tk.Frame(self._chop_grid, bg=COLORS["bg_surface"], highlightthickness=2,
                            highlightbackground=COLORS["success"])
            card.grid(row=i // cols, column=i % cols, padx=5, pady=5)
            # A Label sizes in TEXT units until it holds an image, so every card starts with a
            # shared blank at thumbnail size — the grid lays out correctly before any real
            # frame arrives, and the reference on self keeps Tk from collecting it.
            thumb = tk.Label(card, bg="#000000", image=self._chop_blank)
            thumb.pack()
            mins, secs = divmod(s, 60.0)
            cap = tk.Label(card, text=f"at {int(mins)}:{secs:04.1f}  ·  {span:.1f} s",
                           font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                           fg=COLORS["text_secondary"])
            cap.pack(fill=tk.X)
            state = tk.Label(card, text="✓ keep", font=(FONT_FAMILY, 9, "bold"),
                             bg=COLORS["bg_surface"], fg=COLORS["success"])
            state.pack(fill=tk.X, pady=(0, 3))
            cand.update(card=card, thumb=thumb, state=state)
            for w in (card, thumb, cap, state):
                w.bind("<Button-1>", lambda _e, c=cand: self._autochop_toggle(c))
                w.bind("<Double-Button-1>",
                       lambda _e, sf=int(round(s * fps)): self._autochop_seek(sf))
            self._chop_cands.append(cand)
        for w in (self._chop_keep_all, self._chop_skip_all, self._chop_add_btn):
            w.configure(state=tk.NORMAL)
        self._autochop_refresh_count()
        threading.Thread(target=self._autochop_thumb_worker,
                         args=(list(self._chop_cands), self.src,
                               (self.info["display_width"], self.info["display_height"]), span),
                         daemon=True).start()

    def _autochop_thumb_worker(self, cands, src, src_size, span):
        """One frame from the MIDDLE of each segment — the first frame of a scene is where the
        transition lives, and the middle is what the segment is actually of."""
        for cand in cands:
            if not self._chop_alive:
                return
            img = grab_frame(self.ffmpeg, src, cand["start"] + span / 2.0,
                             box=self._CHOP_THUMB, src_size=src_size)
            if img is not None:
                self._chop_after(self._autochop_set_thumb, cand, img)

    def _autochop_set_thumb(self, cand, img):
        if not cand["thumb"].winfo_exists():
            return
        cand["photo"] = ImageTk.PhotoImage(img)     # held on the cand, or Tk collects it
        cand["thumb"].configure(image=cand["photo"], width=img.width, height=img.height)

    def _autochop_toggle(self, cand):
        cand["keep"] = not cand["keep"]
        keep = cand["keep"]
        cand["card"].configure(
            highlightbackground=COLORS["success"] if keep else COLORS["border"])
        cand["state"].configure(text="✓ keep" if keep else "— skipped",
                                fg=COLORS["success"] if keep else COLORS["text_muted"])
        self._autochop_refresh_count()

    def _autochop_set_all(self, keep):
        for cand in self._chop_cands:
            if cand["keep"] != keep:
                self._autochop_toggle(cand)

    def _autochop_refresh_count(self):
        kept = sum(1 for c in self._chop_cands if c["keep"])
        self._chop_count.configure(
            text=f"{kept} of {len(self._chop_cands)} kept" if self._chop_cands else "")

    def _autochop_seek(self, start_frame):
        """Double-click: put the main playhead at the segment so it can be scrubbed, listened
        to, or fine-tuned by hand. The review window stays open."""
        self.scale.set(start_frame)

    def _autochop_add_kept(self):
        kept = [c for c in self._chop_cands if c["keep"]]
        if not kept:
            messagebox.showinfo("Gizmo", "Nothing is marked keep.")
            return
        fps = self.info["fps"]
        added = 0
        for cand in kept:
            job = self._planned_job(start_frame=int(round(cand["start"] * fps)))
            if job is None:
                break
            self.queue.append(job)
            added += 1
        self._refresh_queue_box()
        self._refresh_planned_name()
        self.status.configure(text=f"Queued {added} auto-chopped clip"
                                   f"{'' if added == 1 else 's'}",
                              fg=COLORS["text_secondary"])
        self._autochop_close()

    # -- exporting ---------------------------------------------------------------------------------
    def export_queue(self):
        if self._busy:
            return
        pending = [j for j in self.queue if not j["done"]]
        if not pending:
            messagebox.showinfo("Gizmo", "Nothing queued. Mark a clip and press Add to queue.")
            return
        out_dir = self.out_var.get().strip()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot create {out_dir}\n\n{exc}")
            return

        self._busy = True
        self.export_btn.configure(state=tk.DISABLED)
        self.add_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._export_worker, args=(pending,), daemon=True).start()

    def _export_worker(self, jobs):
        for n, job in enumerate(jobs, 1):
            self.root.after(0, lambda n=n, job=job: self.status.configure(
                text=f"Exporting {n} of {len(jobs)} — {os.path.basename(job['dst'])}",
                fg=COLORS["text_secondary"]))
            job["error"] = self._export_one(job)
            job["done"] = job["error"] is None
            self.root.after(0, self._refresh_queue_box)
        self.root.after(0, self._export_done, jobs)

    def _export_one(self, job):
        """Write one clip. Returns an error string, or None on success."""
        try:
            cmd = build_export_command(self.ffmpeg, job["src"], job["dst"], job["start"],
                                       job["frames"], job["w"], job["h"], job["keep_every"],
                                       job["with_audio"], job.get("crop"), job.get("sar", 1.0))
            p = _run(cmd, text=True)
            if p.returncode != 0 or not os.path.exists(job["dst"]):
                return (p.stderr or "").strip()[-400:] or "ffmpeg failed"
            # Verified, not assumed: frame count is the one thing Fizgig refuses a clip over, and
            # a source that ends mid-clip silently yields a short one.
            got = count_frames(self.ffmpeg, job["dst"])
            if got is not None and got != job["frames"]:
                os.remove(job["dst"])
                return (f"only {got} frames, not {job['frames']} — the source runs out before "
                        f"the clip does")
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _export_done(self, jobs):
        self._busy = False
        self.export_btn.configure(state=tk.NORMAL)
        self.add_btn.configure(state=tk.NORMAL)
        failed = [j for j in jobs if j["error"]]
        ok = len(jobs) - len(failed)
        if failed:
            self.status.configure(text=f"{ok} written, {len(failed)} failed",
                                  fg=COLORS["warning"])
            detail = "\n\n".join(f"{os.path.basename(j['dst'])}: {j['error']}" for j in failed[:6])
            messagebox.showwarning(
                "Gizmo — some clips did not export",
                f"{ok} written, {len(failed)} failed. The failures are marked ✗ in the queue and "
                f"can be adjusted and exported again.\n\n{detail}")
        else:
            self.status.configure(text=f"{ok} clip{'' if ok == 1 else 's'} written",
                                  fg=COLORS["success"])
        self._refresh_planned_name()


class GizmoRecorder:
    """The push-to-record window.

    The mic rolls CONTINUOUSLY while this window is open (one ffmpeg writing raw PCM);
    holding the button — or the R key — only marks timestamps, and the take is sliced out
    afterwards with cushions either side. That is what makes push-to-record clip nothing:
    the capture was already running when the thumb landed.

    A CARD, not a popup (Peter): this swaps in for the Voice tab's Source recording card and
    Done swaps it back out. The popup lost its takes when it closed; a hidden card keeps
    them — leave record mode, come back, and the takes list is exactly as you left it. It
    also dissolves the popup's focus problem outright: everything lives in the main window,
    where the R binding already routes.

    R, not space and not Shift (Peter): space already means too much — it is the transport on
    both main tabs and Tk's activate key for any focused widget — and Shift walks into
    Windows' accessibility traps (five presses pops the Sticky Keys prompt, an 8 s hold pops
    Filter Keys). R is easy to find without looking, means record, and does nothing else
    here. Bound by both keysyms so Caps Lock cannot un-bind it.
    """

    LEVEL_POLL_MS = 100

    def __init__(self, app):
        self.app = app
        self.proc = None
        self.raw = os.path.join(tempfile.gettempdir(), "gizmo_capture.raw")
        self.takes_dir = os.path.join(tempfile.gettempdir(), "gizmo_takes")
        os.makedirs(self.takes_dir, exist_ok=True)
        self.t0 = None                # monotonic when the capture actually started writing
        self._capture_launched = 0.0  # monotonic at Popen — the clock's hard floor
        self._pending_press = None    # a press buffered while the device was still opening
        self.press_t = None
        self.takes = []               # {path, dur, sentence, delivery}
        self._hold_release_job = None
        self.active = False           # showing + mic rolling; the card itself is permanent
        self._root_binds = []

        # The card. Built ONCE, in the Voice tab's card stack, and never destroyed — show()
        # and close() pack-swap it with the source card. Accent border: record mode should
        # look like a mode.
        S = COLORS["bg_surface"]
        outer = self.frame = tk.Frame(app._audio_tab_body, bg=S, highlightthickness=1,
                                      highlightbackground=COLORS["accent"])
        inner = tk.Frame(outer, bg=S)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        head = tk.Frame(inner, bg=S)
        head.pack(fill=tk.X)
        tk.Label(head, text="1. Record a voice", font=(FONT_FAMILY, 12, "bold"),
                 bg=S, fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        app._button(head, "✓ Done — back to files", self.close, "accent", pad=12,
                    tip="Stop the microphone and put the Source recording card back. Your "
                        "takes are kept — press Record again and they are still "
                        "here.").pack(side=tk.RIGHT)
        tk.Label(inner, text="Hold the button (or the R key), read the sentence, let go. Each "
                             "take opens in the editor below with its caption already written "
                             "— listen if you like, press Add to queue when it's a keeper, "
                             "then record the next one.",
                 font=(FONT_FAMILY, 9), justify=tk.LEFT, wraplength=900,
                 bg=S, fg=COLORS["text_explain"]).pack(anchor="w", pady=(2, 8))

        # 1. the sentence to read
        tk.Label(inner, text="Read this:", font=(FONT_FAMILY, 10),
                 bg=S, fg=COLORS["text_secondary"]).pack(anchor=tk.W)
        self.sentence_var = tk.StringVar()
        tk.Label(inner, textvariable=self.sentence_var, font=(FONT_FAMILY, 15, "bold"),
                 bg=S, fg=COLORS["text_primary"], wraplength=880,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 8))
        srow = tk.Frame(inner, bg=S)
        srow.pack(fill=tk.X)
        app._button(srow, "🎲 New sentence", self.new_sentence, pad=10,
                    tip="Another prompt. Lengths rotate from a quick interjection to a line "
                        "that fills the longest slot — a voice dataset wants range in "
                        "phrasing as much as in tone — and the sentences are pangrams, "
                        "Harvard lines and character pieces: phonetically dense on "
                        "purpose.").pack(side=tk.LEFT)
        tk.Label(srow, text="delivery:", font=(FONT_FAMILY, 9),
                 bg=S, fg=COLORS["text_secondary"]).pack(side=tk.LEFT, padx=(14, 4))
        self.delivery_var = tk.StringVar(value="plain")
        dcombo = ttk.Combobox(srow, textvariable=self.delivery_var, width=13,
                              state="readonly", style="G.TCombobox",
                              values=list(DELIVERY_PROMPTS))
        dcombo.pack(side=tk.LEFT)
        ToolTip(dcombo, "Say it that way, and the caption records it honestly — emotional "
                        "range across takes is dataset variety a voice LoRA wants.\n\nRolled "
                        "at random after every take so the range arrives on its own; pick one "
                        "here to override the next take, or turn the rolling off in ⚙ "
                        "settings.")
        self.free_var = tk.BooleanVar(value=bool(app.settings.get("free_speech", False)))

        def _save_free():
            app.settings["free_speech"] = bool(self.free_var.get())
            save_settings(app.settings)

        free_cb = tk.Checkbutton(srow, text="Free speech", variable=self.free_var,
                                 command=_save_free, bg=S, fg=COLORS["text_primary"],
                                 selectcolor=COLORS["bg_deep"], activebackground=S,
                                 activeforeground=COLORS["text_primary"],
                                 font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                 takefocus=0)
        free_cb.pack(side=tk.RIGHT)
        ToolTip(free_cb, "Ignore the prompts and just talk — hold as long as you like "
                         "(minutes are fine) and the whole recording drops into the editor "
                         "untouched: no auto-fit, no trimming. Mark each bit you want on the "
                         "waveform, caption it (Transcribe helps), Add to queue, repeat.")

        # 2. the hold button. Press/release bindings, NOT command — and NOT app._button,
        # whose focus-return wrapper would fight the hold.
        self.rec_btn = tk.Button(inner, text="🎙  HOLD to record  (or hold R)",
                                 font=(FONT_FAMILY, 13, "bold"), bg=COLORS["bg_hover"],
                                 fg=COLORS["text_primary"], activebackground="#B91C1C",
                                 activeforeground=COLORS["text_primary"], relief=tk.FLAT,
                                 bd=0, padx=24, pady=16, cursor="hand2", takefocus=0)
        self.rec_btn.pack(fill=tk.X, pady=(12, 6))
        ToolTip(self.rec_btn,
                "Hold while you speak, release when the sentence ends — no need to rush "
                "either: the silence at both ends, and the key click itself, are trimmed off "
                "the take automatically. R works as the hold key too, unless you are typing "
                "in a text field.")
        self.rec_btn.bind("<ButtonPress-1>", self._press)
        self.rec_btn.bind("<ButtonRelease-1>", self._release)

        # 3. level + elapsed. The meter is built but NOT packed — HIDDEN for now (Peter): its
        # data source is the raw file's size polled at 100 ms, which trails the microphone by
        # the writer's flush lag, and a meter that answers half a second late is worse than no
        # meter. To bring it back, pack lrow and feed it from the capture stream directly
        # rather than from the file.
        lrow = tk.Frame(inner, bg=S)
        tk.Label(lrow, text="Mic level", font=(FONT_FAMILY, 9), bg=S,
                 fg=COLORS["text_secondary"]).pack(side=tk.LEFT, padx=(0, 8))
        self.level = tk.Canvas(lrow, height=22, bg=COLORS["bg_deep"],
                               highlightthickness=1, highlightbackground=COLORS["border"])
        self.level.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(self.level, "Live input level — talk and it should move. Flat while you "
                            "speak means the wrong microphone: pick another in ⚙ settings.")
        self.status = tk.Label(inner, text="", font=(FONT_FAMILY, 10),
                               bg=S, fg=COLORS["text_secondary"])
        self.status.pack(anchor=tk.W, pady=(4, 0))

        # 4. takes
        tk.Label(inner, text="Takes (double-click to load one into the editor):",
                 font=(FONT_FAMILY, 9), bg=S,
                 fg=COLORS["text_secondary"]).pack(anchor=tk.W, pady=(10, 2))
        self.takes_box = tk.Listbox(inner, height=5, width=64, bg=COLORS["bg_deep"],
                                    fg=COLORS["text_primary"], font=("Consolas", 9),
                                    relief=tk.FLAT, highlightthickness=1,
                                    highlightbackground=COLORS["border"],
                                    selectbackground=COLORS["accent"])
        self.takes_box.pack(fill=tk.X)
        self.takes_box.bind("<Double-Button-1>", lambda _e: self.use_selected())
        ToolTip(self.takes_box, "Every release lands here, and the newest take loads into the "
                                "editor automatically with its caption written for you. "
                                "Listen, press Add to queue (Ctrl+S) to keep it, and hold for "
                                "the next one. The list survives Done: takes are only ever "
                                "added.")

        self.new_sentence()

    def alive(self):
        """Record mode is on: the card is showing and the mic is (or is starting) live."""
        try:
            return self.active and self.frame.winfo_exists()
        except tk.TclError:
            return False

    def show(self):
        """Enter record mode: swap this card in for the source card, arm R, start the mic."""
        if self.active:
            return
        self.active = True
        self.frame.pack(fill=tk.X, padx=16, pady=(0, 12),
                        before=self.app._audio_source_outer)
        self.app._audio_source_outer.pack_forget()
        # R on the ROOT — where keyboard focus actually lives — released again on close.
        # Guarded like the transport keys: a focused text field keeps its letters. Both
        # keysyms, so Caps Lock cannot un-bind it.
        for sym in ("r", "R"):
            self._root_binds.append((f"<KeyPress-{sym}>",
                                     self.app.root.bind(f"<KeyPress-{sym}>",
                                                        self._root_press, add="+")))
            self._root_binds.append((f"<KeyRelease-{sym}>",
                                     self.app.root.bind(f"<KeyRelease-{sym}>",
                                                        self._root_release, add="+")))
        # DISABLED until the mic is actually live: the device takes a moment to open, and an
        # armed-looking button invites speaking into a mic that is not capturing yet — which
        # is how the first words of a session's first take went missing (Peter). The press
        # buffer below stays as the belt for the R key.
        self.rec_btn.configure(state=tk.DISABLED, text="⏳  the mic is starting…")
        self.status.configure(text="starting the microphone…", fg=COLORS["text_secondary"])
        self.maybe_roll_delivery()
        self._start_capture()

    def maybe_roll_delivery(self):
        """A random delivery per take (Peter): the emotional-range axis fills itself instead
        of waiting for someone to remember the dropdown exists. The roll is VISIBLE — it sets
        the dropdown, so the instruction is explicit — and the dropdown stays live to
        override the next take by hand. The ⚙ settings checkbox turns the rolling off."""
        if not self.app.settings.get("random_delivery", True):
            return
        import random
        others = [d for d in DELIVERY_PROMPTS if d != self.delivery_var.get()]
        self.delivery_var.set(random.choice(others))

    # -- capture lifecycle -------------------------------------------------------------------
    def _start_capture(self):
        device = self.app.settings.get("microphone", "")
        if os.name == "nt" and not device:
            # First ever use, no saved mic. The scan is an ffmpeg subprocess that takes a
            # second or two, and it used to run ON the GUI thread — record mode appeared
            # frozen and the mic went live late, which is most of the cold-start delay
            # (Peter). Scan in the background, save the pick, and come back here.
            self.status.configure(text="finding your microphone…",
                                  fg=COLORS["text_secondary"])

            def scan():
                devices = list_capture_devices(self.app.ffmpeg)

                def done():
                    if not self.alive():
                        return
                    if devices:
                        self.app.settings["microphone"] = devices[0]
                        save_settings(self.app.settings)
                        self._start_capture()
                    else:
                        self.rec_btn.configure(state=tk.DISABLED)
                        self.status.configure(
                            text="no microphone available on this machine",
                            fg=COLORS["warning"])
                self.app.root.after(0, done)
            threading.Thread(target=scan, daemon=True).start()
            return
        # A FRESH, uniquely named raw per record session. Reusing one path was the bug that
        # collapsed 4-second takes to under a second (Peter): close()'s delete can fail while
        # the old ffmpeg still holds the handle, the next session's first size-poll then read
        # the STALE file before the new ffmpeg truncated it, and the clock backed itself off
        # by minutes of old audio — a mistake the running minimum could never recover from.
        # A new name every time means the first byte seen is always this session's.
        import time as _time
        self.raw = os.path.join(tempfile.gettempdir(),
                                f"gizmo_capture_{os.getpid()}_{int(_time.time() * 1000)}.raw")
        try:                                   # yesterday's leftovers, best effort
            import glob
            for old in glob.glob(os.path.join(tempfile.gettempdir(), "gizmo_capture*.raw")):
                if old != self.raw:
                    os.remove(old)
        except OSError:
            pass
        cmd = build_capture_command(self.app.ffmpeg, device, self.raw)
        if cmd is None:
            self.rec_btn.configure(state=tk.DISABLED)
            self.status.configure(text="no microphone available on this machine",
                                  fg=COLORS["warning"])
            return
        self._device = device or "default"     # for the live status — which mic is this?
        # The hard floor for the clock: audio cannot have been captured before the process
        # that captures it existed. Every t0 estimate below is clamped to this.
        self._capture_launched = _time.monotonic()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        except Exception as exc:
            self.status.configure(text=f"could not start capture: {exc}", fg=COLORS["error"])
            return
        self._await_first_bytes()

    def _await_first_bytes(self):
        """The clock starts when ffmpeg actually WRITES, not when it was launched — the
        device takes a few hundred ms to open, and stamping early would shift every take."""
        if not self.alive():
            return
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-200:]
            self.status.configure(text=f"microphone failed: {err or 'unknown'}",
                                  fg=COLORS["error"])
            self.rec_btn.configure(state=tk.DISABLED)
            return
        try:
            size = os.path.getsize(self.raw)
            if size > 0:
                import time as _time
                # NOT "now": by the time the file first shows bytes, ffmpeg has flushed a
                # buffered burst, so the file already holds audio from BEFORE this moment.
                # Backing now off by the audio the file contains lands t0 on the capture's
                # true start — stamping now instead shifted every press/release mapping
                # early by the burst, which is what put takes out of sync with the thumb
                # (short measured durations, clipped endings). Clamped to the launch time:
                # no estimate may claim audio from before the capture process existed.
                # Refined further by the running minimum in _level_tick.
                self.t0 = max(_time.monotonic() - size / CAPTURE_BPS,
                              self._capture_launched)
                # Which mic, by name — the answer to "is it even using the right one?"
                # without a trip to settings (Peter).
                self.status.configure(
                    text=f"live: {getattr(self, '_device', 'microphone')} — hold and speak",
                    fg=COLORS["success"])
                self.rec_btn.configure(state=tk.NORMAL,
                                       text="🎙  HOLD to record  (or hold R)")
                if self._pending_press is not None:
                    # A press arrived while the device was opening. It was buffered, not
                    # dropped — convert it now, clamped to the start of what was captured.
                    self.press_t = max(0.0, self._pending_press - self.t0)
                    self._pending_press = None
                    self.rec_btn.configure(bg="#B91C1C",
                                           text="●  recording — release when done")
                self._level_tick()
                return
        except OSError:
            pass
        self.app.root.after(50, self._await_first_bytes)

    def _now(self):
        import time as _time
        return _time.monotonic() - self.t0

    def _level_tick(self):
        if not self.alive() or self.t0 is None:
            return
        try:
            size = os.path.getsize(self.raw)
            # The clock rides the bytes: (now - size/BPS) equals the capture's true start
            # plus however far the writer is lagging right now, so its running MINIMUM
            # converges on the true start — clamped to the launch time, so nothing the file
            # holds can ever drag the clock before the capture process existed. That keeps
            # press/release timestamps mapped to the right bytes for the whole session,
            # whatever the buffering does.
            import time as _time
            self.t0 = max(min(self.t0, _time.monotonic() - size / CAPTURE_BPS),
                          self._capture_launched)
            window = int(0.1 * CAPTURE_BPS) // 4 * 4
            with open(self.raw, "rb") as f:
                f.seek(max(0, size - window))
                chunk = f.read(window)
            peak = 0
            for i in range(0, len(chunk) - 1, 2):
                v = int.from_bytes(chunk[i:i + 2], "little", signed=True)
                peak = max(peak, abs(v))
        except OSError:
            peak = 0
        self.level.delete("all")
        frac = min(1.0, peak / 28000)
        colour = COLORS["success"] if frac < 0.85 else COLORS["warning"]
        lw = max(1, self.level.winfo_width())          # the meter stretches with the card
        lh = max(1, self.level.winfo_height())
        self.level.create_rectangle(0, 0, lw * frac, lh, fill=colour, outline="")
        if self.press_t is not None:
            self.status.configure(text=f"recording…  {self._now() - self.press_t:.1f} s",
                                  fg="#EF4444")
        self.app.root.after(self.LEVEL_POLL_MS, self._level_tick)

    # -- push-to-record ----------------------------------------------------------------------
    def _press(self, _e=None):
        if self.press_t is not None or self._pending_press is not None:
            return
        if self.t0 is None:
            # The device is still opening. Dropping the press silently made the first record
            # of a session feel dead (Peter) — remember it instead, arm the button, and
            # _await_first_bytes converts it the moment the capture is live.
            import time as _time
            self._pending_press = _time.monotonic()
            self.rec_btn.configure(bg="#B91C1C", text="●  mic is waking — keep holding…")
            return
        self.press_t = self._now()
        self.rec_btn.configure(bg="#B91C1C", text="●  recording — release when done")

    def _release(self, _e=None):
        if self._pending_press is not None:
            # Released before the mic came alive — nothing was captured to slice.
            self._pending_press = None
            self.rec_btn.configure(bg=COLORS["bg_hover"],
                                   text="🎙  HOLD to record  (or hold R)")
            self.status.configure(text="the mic was still waking — hold again once it says "
                                       "live", fg=COLORS["warning"])
            return
        if self.press_t is None:
            return
        press, self.press_t = self.press_t, None
        release = self._now()
        self.rec_btn.configure(bg=COLORS["bg_hover"],
                               text="🎙  HOLD to record  (or hold R)")
        sentence, delivery = self.sentence_var.get(), self.delivery_var.get()

        free = bool(self.free_var.get()) if hasattr(self, "free_var") else False

        def worker():
            path = os.path.join(self.takes_dir, f"take_{len(self.takes) + 1:03d}.wav")
            try:
                # The writer runs behind the microphone; slice only once the file actually
                # holds the tail context, or the take loses its end to the flush lag.
                if free:
                    # Free speech: the whole hold, verbatim — no slot fit, no trimming, no
                    # heuristics. The editor is where it gets carved into segments.
                    wait_for_capture(self.raw, release + TAKE_POST_ROLL)
                    dur = slice_take(self.raw, press, release, path, self.app.ffmpeg)
                elif bool(self.app.settings.get("trim_takes", True)):
                    wait_for_capture(self.raw, release + TAKE_POST_ROLL + TAKE_CONTEXT)
                    dur = cut_take_to_slot(self.raw, press, release, path, self.app.ffmpeg)
                else:
                    wait_for_capture(self.raw, release + TAKE_POST_ROLL)
                    dur = slice_take(self.raw, press, release, path, self.app.ffmpeg)
            except Exception as exc:
                self.app.root.after(0, lambda: self.status.configure(
                    text=f"take failed: {exc}", fg=COLORS["error"]))
                return
            self.app.root.after(0, self._take_done, path, dur, sentence, delivery, free)
        threading.Thread(target=worker, daemon=True).start()

    def _take_done(self, path, dur, sentence, delivery, free=False):
        if not self.alive():
            return
        min_dur = hop_exact_samples(AUDIO_GRID_FRAMES[0]) / AUDIO_SAMPLE_RATE
        if dur < min_dur and not free:
            self.status.configure(
                text=f"too short ({dur:.2f} s) — the smallest slot is {min_dur:.1f} s",
                fg=COLORS["warning"])
            return
        max_slot = hop_exact_samples(AUDIO_GRID_FRAMES[-1]) / AUDIO_SAMPLE_RATE
        # Free-speech takes are long-style whatever their length: no prompt was read, so the
        # prompt must never become their caption.
        is_long = free or dur > max_slot + 0.05
        self.takes.append({"path": path, "dur": dur, "sentence": sentence,
                           "delivery": delivery, "long": is_long})
        label = (("(free speech — carve it up in the editor)" if free
                  else "(long take — carve it up in the editor)") if is_long
                 else (sentence if len(sentence) <= 40 else sentence[:39] + "…"))
        self.takes_box.insert(tk.END, f"{dur:4.1f}s  {label}")
        self.takes_box.see(tk.END)
        if is_long:
            self.status.configure(
                text=f"long take ({dur:.0f} s) — mark each bit you want on the waveform, "
                     f"caption it (Transcribe helps), queue, repeat", fg=COLORS["success"])
        else:
            self.status.configure(text=f"take {len(self.takes)} — loaded into the editor",
                                  fg=COLORS["success"])
        self.use_take(len(self.takes) - 1)
        self.new_sentence()
        self.maybe_roll_delivery()

    # -- R as the hold key (with auto-repeat filtering) --------------------------------------
    def _root_key_ok(self):
        """Whether an R arriving via the MAIN window should record: yes unless someone is
        typing in a text field (belt — shielded fields never reach the root binding anyway)."""
        if not self.alive():
            return False
        try:
            focus = self.app.root.focus_get()
        except (KeyError, tk.TclError):
            return True                # focus in a window Tk can't name; not a text field
        return not (focus is not None
                    and focus.winfo_class() in ("Entry", "TEntry", "Text"))

    def _root_press(self, e):
        return self._hold_press(e) if self._root_key_ok() else None

    def _root_release(self, e):
        return self._hold_release(e) if self._root_key_ok() else None

    def _hold_press(self, _e):
        # Auto-repeat delivers release+press pairs back to back; a REAL press has no pending
        # ghost release to cancel.
        if self._hold_release_job is not None:
            self.app.root.after_cancel(self._hold_release_job)
            self._hold_release_job = None
            return "break"
        self._press()
        return "break"

    def _hold_release(self, _e):
        self._hold_release_job = self.app.root.after(60, self._hold_release_real)
        return "break"

    def _hold_release_real(self):
        self._hold_release_job = None
        self._release()

    # -- takes -> the editor -----------------------------------------------------------------
    def use_selected(self):
        sel = self.takes_box.curselection()
        if sel:
            self.use_take(sel[0])

    def use_take(self, idx):
        t = self.takes[idx]
        app = self.app
        try:
            app.load_audio(t["path"])
        except Exception as exc:
            self.status.configure(text=str(exc), fg=COLORS["error"])
            return
        fit = max((f for f in AUDIO_GRID_FRAMES
                   if hop_exact_samples(f) / AUDIO_SAMPLE_RATE <= t["dur"]), default=None)
        if fit:
            app.audio_frames_var.set(fit)
        app.audio_caption.delete("1.0", tk.END)
        if t.get("long"):
            # A long take is free talk, not the prompted sentence — prefilling the prompt
            # would caption every segment with words that were never said. Leave it empty:
            # caption per segment (Transcribe reads the marked span).
            pass
        else:
            adverb = DELIVERY_PROMPTS.get(t["delivery"], "")
            caption = (f'saying {adverb} "{t["sentence"]}"' if adverb
                       else f'saying "{t["sentence"]}"')
            app.audio_caption.insert("1.0", caption)
        app._audio_seek(0.0)

    def new_sentence(self):
        """A fresh prompt, sized by the recorder's OWN rotating length — a shuffled bag of
        every grid slot, so seven takes cover the full 0.9-5.2 s range.

        Deliberately NOT the editor's segment-length dropdown: every finished take sets that
        dropdown to its own best-fit slot, so sizing prompts from it was a feedback loop —
        one short take locked it to the 0.9 s slot, which only fits three-word sentences,
        which produce short takes (Peter). Length variety is a dataset axis, same as
        delivery; it has to be driven, not inherited."""
        if not getattr(self, "_len_bag", None):
            import random
            self._len_bag = list(AUDIO_GRID_FRAMES)
            random.shuffle(self._len_bag)
        self._prompt_frames = self._len_bag.pop()
        self.sentence_var.set(pick_sentence(self._prompt_frames))

    def close(self):
        """Done — back to file mode. The mic stops and R is released; the card and its takes
        survive hidden, ready for the next show()."""
        if not self.active:
            return
        self.active = False
        # The root bindings must not outlive record mode — R has to type an r again.
        # (tkinter's unbind clears the whole sequence regardless of funcid, which is fine:
        # nothing else on the root binds R.)
        for seq, _funcid in self._root_binds:
            try:
                self.app.root.unbind(seq)
            except tk.TclError:
                pass
        self._root_binds = []
        self.press_t = None
        self._pending_press = None
        self.t0 = None                # stops the level ticker at its next alive() check
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()          # raw output: nothing to finalise, safe to kill
            except Exception:
                pass
        self.proc = None
        try:
            os.path.exists(self.raw) and os.remove(self.raw)
        except OSError:
            pass                               # ffmpeg may still hold it for an instant
        try:
            self.app._audio_source_outer.pack(fill=tk.X, padx=16, pady=(0, 12),
                                              before=self.frame)
            self.frame.pack_forget()
        except tk.TclError:
            pass                               # app teardown mid-close


def _dpi_aware():
    """Windows: draw at the screen's real scaling. Unaware, a 125/150 % laptop screen gets the window drawn at
    100 % and stretched as a picture, which blurs every letter. Aware, Tk sizes its fonts for the screen."""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)      # system-aware: sharp at the main screen's scaling
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()          # Windows 7 / 8.0
    except Exception:
        pass                                                    # never stop Gizmo starting over this


def _scaled(win, w, h):
    """A size laid out at 100 %, for this screen's scaling (fonts grow with it, so the room must too), kept on the
    screen."""
    try:
        k = max(1.0, float(win.winfo_fpixels("1i")) / 96.0)
    except tk.TclError:
        k = 1.0
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    return min(int(w * k), sw - 40), min(int(h * k), sh - 80)


def main():
    _dpi_aware()
    root = tk.Tk()
    root.withdraw()
    settings = load_settings()
    key = choose_model(root, settings.get("last_model") or "h3")
    if key is None:
        root.destroy()
        return
    settings["last_model"] = key
    save_settings(settings)
    use_model(key)
    root.deiconify()
    app = Gizmo(root)
    # A path on the command line, which is what Windows hands over when a video is dropped onto
    # the launcher or opened with Gizmo from Explorer.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(50, lambda: app.open_path(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
