"""Captions for clips from several frames (#351, @CognitiveDiffusion): a captioner that can look at several pictures
(Qwen3-VL) is handed that many frames spread through each clip, in order, with the instruction told they are one clip;
pictures, and captioners that take one picture, are captioned as before."""
import types

import numpy as np
import pytest

from fizgig.captioners import CaptionRequest, CaptionResult, get
from fizgig.scripts import batch_caption as bc


def _clip(path, n=24):
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 10 % 256, np.uint8))     # each frame its own grey, to tell them apart
    w.release()
    return str(path)


def _state(frames):
    seen = []

    class Cap:
        def caption(self, image, request):
            seen.append((image, request.instruction))
            return CaptionResult(text="a caption")
    return types.SimpleNamespace(desc=types.SimpleNamespace(frames=frames), captioner=Cap()), seen


def test_a_clip_is_seen_as_frames_in_order(tmp_path):
    clip = _clip(tmp_path / "c.mp4")
    st, seen = _state(8)
    out = bc.caption_file(st, clip, CaptionRequest(instruction="Caption it."), {"clip_frames": 4, "trigger": "zwx"})
    assert out == "zwx, a caption"
    image, instr = seen[0]
    assert isinstance(image, list) and len(image) == 4
    greys = [int(np.asarray(f)[0, 0, 0]) for f in image]
    assert greys == sorted(greys) and len(set(greys)) == 4            # spread through the clip, start to end
    assert instr.startswith("These 4 pictures are frames from one video clip") and instr.endswith("Caption it.")


def test_one_frame_pictures_and_single_picture_captioners_as_before(tmp_path):
    clip = _clip(tmp_path / "c.mp4")
    st, seen = _state(8)
    bc.caption_file(st, clip, CaptionRequest(instruction="Caption it."), {"clip_frames": 1})
    st1, seen1 = _state(1)                                               # e.g. Florence: one picture only
    bc.caption_file(st1, clip, CaptionRequest(instruction="Caption it."), {"clip_frames": 8})
    from PIL import Image
    pic = tmp_path / "p.png"
    Image.new("RGB", (32, 32)).save(pic)
    bc.caption_file(st, str(pic), CaptionRequest(instruction="Caption it."), {"clip_frames": 8})
    for image, instr in seen + seen1:
        assert not isinstance(image, list) and instr == "Caption it."


def test_a_short_clip_gives_what_it_has(tmp_path):
    frames = bc.clip_frames(_clip(tmp_path / "c.mp4", n=3), 8)
    assert len(frames) == 3


def test_only_qwen_takes_frames_and_the_page_is_told():
    assert get("qwen3vl").frames == 8
    from fizgig.captioners.registry import CAPTIONERS
    assert {d.key for d in CAPTIONERS if d.frames > 1} == {"qwen3vl"}
