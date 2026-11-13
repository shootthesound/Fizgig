"""Studio's Upscale reads a video file: its size, length and sound, one frame as a picture, and a stretch of it cut to
a box and scaled, at the model's frame rate."""
import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.utils import video_in  # noqa: E402


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    from fizgig.minimax.common import _find_ffmpeg
    ff = _find_ffmpeg()
    if not ff:
        pytest.skip("no ffmpeg here")
    path = str(tmp_path_factory.mktemp("v") / "in.mp4")
    made = subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                           "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-pix_fmt", "yuv420p", "-shortest",
                           path], capture_output=True)
    if made.returncode or not os.path.isfile(path):
        pytest.skip("this ffmpeg can't make the test clip")
    return path


def test_info(clip):
    i = video_in.info(clip)
    assert (i["width"], i["height"], i["has_audio"]) == (320, 180, True)
    assert abs(i["seconds"] - 3) < 0.2 and abs(i["fps"] - 30) < 0.01


def test_one_frame(clip):
    from PIL import Image
    with Image.open(io.BytesIO(video_in.frame_jpeg(clip, 1.0, side=160))) as im:
        assert im.size == (160, 90)
    assert video_in.frame_jpeg(clip, 99.0)                  # past the end: the last frame, not an error


def test_read_cuts_scales_and_keeps_the_sound(clip):
    frames, wav = video_in.read(clip, 0.5, 39, (40, 10, 280, 170), 96, 64, 24, rate=32000)
    assert len(frames) == 39 and frames[0].size == (96, 64)
    assert wav.shape == (2, round(32000 * 39 / 24)) and float(wav.abs().max()) > 0.01
    assert frames[0].tobytes() != frames[-1].tobytes()      # the picture moves


def test_read_stops_where_the_video_ends(clip):
    frames, _wav = video_in.read(clip, 2.5, 39, (0, 0, 320, 180), 64, 32, 24)
    assert 5 <= len(frames) < 39
