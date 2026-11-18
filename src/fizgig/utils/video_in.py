"""A video file read for Studio's Upscale: its facts, one frame as a picture, and a stretch of it as frames + sound.

Everything goes through the bundled ffmpeg, so any file it reads works (a phone clip's rotation included: ffmpeg
turns the frames upright, and the sizes here are the upright ones)."""
import io
import os
import re
import subprocess

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
# A file whose pixels are not square (a portrait clip stored 1920x1080 and shown 9:16) is first made the shape it
# plays in, at the same number of pixels (that one: 1080x1920). Every size and box here is in those square pixels
SQUARE = ("scale='if(eq(sar,1)+eq(sar,0),iw,round(sqrt(iw*ih*dar)/2)*2)':"
          "'if(eq(sar,1)+eq(sar,0),ih,round(sqrt(iw*ih/dar)/2)*2)',setsar=1")
_INFO = {}


def _ffmpeg():
    from fizgig.lora_royale.export import _find_ffmpeg      # light: the server asks too, without loading torch
    exe = _find_ffmpeg()
    if not exe:
        raise ValueError("No ffmpeg here to read the video with: reinstall Fizgig to restore the bundled one.")
    return exe


def _run(args):
    return subprocess.run([_ffmpeg(), "-hide_banner", *args], capture_output=True)


def info(path: str) -> dict:
    """{width, height, seconds, fps, has_audio}: the frames' size as they decode (upright), the length and rate
    from the file's banner. Kept per file and modified time."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key in _INFO:
        return _INFO[key]
    banner = _run(["-i", path]).stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", banner)
    seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", banner)
    fps = float(m.group(1)) if m else 0.0
    from PIL import Image
    shot = _run(["-v", "error", "-i", path, "-frames:v", "1", "-vf", SQUARE, "-f", "image2pipe", "-vcodec", "bmp",
                 "-"]).stdout
    if not shot:
        raise ValueError(f"{os.path.basename(path)}: no video in this file that Fizgig can read.")
    with Image.open(io.BytesIO(shot)) as im:
        width, height = im.size
    out = {"width": width, "height": height, "seconds": round(seconds, 3), "fps": fps,
           "has_audio": bool(re.search(r"Stream #\d+:\d+.*?: Audio:", banner))}
    if len(_INFO) > 200:
        _INFO.clear()
    _INFO[key] = out
    return out


def frame_jpeg(path: str, at: float = 0.0, side: int = 1280) -> bytes:
    """The frame at `at` seconds as a JPEG, its longer side at most `side` pixels."""
    side = max(64, int(side))
    vf = (f"{SQUARE},scale='min({side},iw)':'min({side},ih)':force_original_aspect_ratio=decrease:"
          f"force_divisible_by=2")
    got = _run(["-v", "error", "-ss", f"{max(0.0, float(at)):.3f}", "-i", path, "-frames:v", "1", "-vf", vf,
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"]).stdout
    if not got and at > 0:              # asked for the very end: the last frame there is
        got = _run(["-v", "error", "-sseof", "-0.5", "-i", path, "-update", "1", "-vf", vf, "-f", "image2pipe",
                    "-vcodec", "mjpeg", "-q:v", "3", "-"]).stdout
    if not got:
        raise ValueError("Couldn't read a frame from that video.")
    return got


def read(path: str, start: float, frames: int, rect, width: int, height: int, fps: float, rate: int = 0):
    """`frames` frames from `start` seconds at `fps`: the box rect (x0, y0, x1, y1 in the video's pixels) cut out and
    scaled to width x height. -> (PIL frames, the sound as a [2, samples] tensor at `rate` or None). Fewer frames
    come back when the video ends first."""
    from PIL import Image
    x0, y0, x1, y1 = (int(v) for v in rect)
    width, height, frames = int(width), int(height), int(frames)
    vf = f"{SQUARE},crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={width}:{height}:flags=lanczos,fps={fps:g}"
    proc = subprocess.Popen([_ffmpeg(), "-hide_banner", "-v", "error", "-ss", f"{max(0.0, float(start)):.3f}",
                             "-i", path, "-an", "-vf", vf, "-frames:v", str(frames), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, out = width * height * 3, []
    try:
        while len(out) < frames:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            out.append(Image.frombytes("RGB", (width, height), buf))
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    wav = None
    if rate and out:
        import numpy as np
        import torch
        raw = _run(["-v", "error", "-ss", f"{max(0.0, float(start)):.3f}", "-t", f"{len(out) / fps:.4f}", "-i", path,
                    "-vn", "-ac", "2", "-ar", str(int(rate)), "-f", "f32le", "-"]).stdout
        if raw:
            a = torch.from_numpy(np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy())
            n = int(round(rate * len(out) / fps))
            wav = a[:, :n] if a.shape[1] >= n else torch.nn.functional.pad(a, (0, n - a.shape[1]))
    return out, wav
