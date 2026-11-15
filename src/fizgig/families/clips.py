"""Clips against a family's ClipSpec (description.clip_spec): probe, check, decode.

Fizgig does not PREPARE clips - Gizmo does. This module only decides whether a file is already on the family's
spec and, if it is, decodes it. Nothing here transcodes or trims the picture: the moment a training app starts
silently fixing footage, two datasets that look identical train differently and nobody can tell why. Off-spec files
are REFUSED with the value found and what to do, because the failure they would otherwise cause is invisible - a
30 fps clip trains motion at the wrong speed and looks perfectly fine doing it. Audio is the exception: its sample
rate and channel count are decode details (read_audio converts them, as the voice path's audio files always have),
so any rate trains.

Shared by every family whose description's `media` includes "clip"; the dataset layer and the launch checks reach it
through the description, never through a model package.
"""
import math
import os
import re
import subprocess

import numpy as np

VIDEO_EXTENSIONS = (".mp4",)


class ClipRejected(ValueError):
    """A clip that does not meet its family's spec. The message is user-facing: it names the value found and what
    to do, because a silent trim or resample is how a dataset goes quietly wrong."""


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def grid_frames(spec) -> tuple:
    """Every frame count the family's video VAE encodes: frame_offset + n * frame_step, up to max_frames."""
    return tuple(range(spec.frame_offset, spec.max_frames + 1, spec.frame_step))


def is_muted(path: str, spec) -> bool:
    """`walk_03_mute.mp4` -> True. The suffix wins over the file's contents: it is a statement of intent, and the
    audio stays in the file so the decision is reversible by rename."""
    return bool(spec.mute_suffix) and os.path.splitext(os.path.basename(path))[0].lower().endswith(spec.mute_suffix)


def ffmpeg() -> str:
    from fizgig.lora_royale.export import _find_ffmpeg
    exe = _find_ffmpeg()
    if not exe:
        raise ClipRejected("no ffmpeg available - reinstall to restore the bundled imageio-ffmpeg")
    return exe


def probe(path: str) -> dict:
    """Container facts, read from ffmpeg's own stream banner. No decode.

    ffmpeg with no output exits non-zero and prints the streams to stderr; that is the documented way to get this
    without ffprobe, which imageio-ffmpeg does not bundle."""
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    info = {"fps": None, "width": None, "height": None, "has_audio": False, "sample_rate": None, "channels": None}
    m = re.search(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})", out, re.S)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", out)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Stream #\d+:\d+.*?: Audio: [^,]+, (\d+) Hz, (\w+)", out)
    if m:
        info["has_audio"] = True
        info["sample_rate"] = int(m.group(1))
        info["channels"] = {"mono": 1, "stereo": 2}.get(m.group(2), None)
    if info["width"] is None or info["fps"] is None:
        raise ClipRejected(f"{os.path.basename(path)}: no readable video stream")
    return info


def validate(path: str, spec, family: str, info: dict = None, frames: int = None) -> dict:
    """Raise ClipRejected unless the file is on `spec` (`family` names the model in the message). Returns the probe
    dict. frames, when known (the caller has just decoded), is checked against the grid: frame COUNT is the one
    thing the banner cannot be trusted for - duration x fps rounds - so it is verified after decoding."""
    info = info or probe(path)
    name = os.path.basename(path)
    where = "Prepare it with Gizmo, or see the clip spec in the README."
    if abs(info["fps"] - spec.fps) > 0.01:
        raise ClipRejected(
            f"{name}: {info['fps']:g} fps, but {family} runs at {spec.fps}. Training it as-is would learn motion at "
            f"the wrong speed and look fine doing it. {where}")
    for dim, label in ((info["width"], "width"), (info["height"], "height")):
        if dim % spec.edge_multiple:
            raise ClipRejected(f"{name}: {info['width']}x{info['height']} - {label} must be a multiple of "
                               f"{spec.edge_multiple}. {where}")
    # Audio of any sample rate or channel count trains as it is: read_audio decodes it to the spec's rate and channels
    # when it caches, and the file is never rewritten. Unlike the frame rate, that changes nothing about what trains.
    grid = grid_frames(spec)
    if frames is not None and frames not in grid:
        raise ClipRejected(
            f"{name}: {frames} frames. {family} encodes on a {spec.frame_step}n+{spec.frame_offset} grid, so a clip "
            f"has to be one of {', '.join(str(f) for f in grid)} frames. {where}")
    return info


def problem(path: str, spec, family: str) -> str:
    """Why this clip cannot train as it is ("" = fine): the launch's check, before anything is cached."""
    if not is_video(path):
        return ""
    try:
        validate(path, spec, family)
    except ClipRejected as e:
        return str(e)
    return ""


def read_frames(path: str) -> np.ndarray:
    """Decode to uint8 (T, H, W, 3) RGB, every frame."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    out = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    if not out:
        raise ClipRejected(f"{os.path.basename(path)}: decoded no frames")
    return np.stack(out)


def hold_to_grid(frames, spec):
    """A clip (e.g. a slider pair's edited pole, which may have dropped frames) held on its last frame up to the
    next length the family encodes - or cut to the longest. Returns (frames, length before)."""
    grid = grid_frames(spec)
    n = len(frames)
    if n in grid:
        return frames, n
    want = next((g for g in grid if g >= n), grid[-1])
    return (list(frames) + [frames[-1]] * want)[:want], n


def read_audio(path: str, spec) -> np.ndarray:
    """Decode to float32 (channels, L) at the spec's rate in [-1, 1], or None when there is no track.

    Muted clips return None too: the mute suffix is checked FIRST, so the audio is never read and the item simply
    has no audio target. That is deliberately not the same as a target of silence, which would teach the model that
    this footage sounds like nothing."""
    if is_muted(path, spec) or not probe(path)["has_audio"]:
        return None
    raw = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(spec.audio_channels), "-ar", str(spec.audio_rate), "-"], capture_output=True).stdout
    if not raw:
        return None
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, spec.audio_channels).T.copy()
    # a track that decodes to digital silence counts as muted: a silence target is worse than no target at all
    if not np.any(wav):
        return None
    return wav


def expected_audio_samples(frames: int, spec) -> int:
    """Samples covering `frames` pixel frames - what a clip's audio should decode to."""
    return int(math.ceil(frames / spec.fps * spec.audio_rate))
