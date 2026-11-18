"""A preview MP4 with sound keeps every frame: -shortest used to cut the last one when the wav ran a hair short (a
56-frame H3 clip came out 55, which H3 then refuses as a training clip). Needs ffmpeg; skipped without it."""
import re
import subprocess

import pytest
import torch

from fizgig.minimax.common import _find_ffmpeg, write_preview_mp4, write_wav


@pytest.mark.parametrize("secs", [2.30, 2.50])
def test_every_frame_kept(tmp_path, secs):
    ff = _find_ffmpeg()
    if not ff:
        pytest.skip("no ffmpeg")
    wav = str(tmp_path / "s.wav")
    write_wav(wav, torch.zeros(2, int(44100 * secs)), sample_rate=44100)
    out = str(tmp_path / "c.mp4")
    write_preview_mp4(out, torch.rand(3, 56, 32, 32), wav, fps=24)
    r = subprocess.run([ff, "-i", out, "-f", "null", "-"], capture_output=True, text=True)
    assert re.findall(r"frame=\s*(\d+)", r.stderr)[-1] == "56"
