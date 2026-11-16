"""Gizmo's crop box (#350): dragging from the pressed corner in any direction, reaching the frame's
edges, a locked shape growing smoothly, placing, typed sizes and saved boxes on another size."""
import os
import sys

import pytest

pytest.importorskip("tkinter")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gizmo"))
import gizmo as G  # noqa: E402

FW, FH = 1920, 1080


def inside(box):
    x, y, w, h = box
    return x >= 0 and y >= 0 and x + w <= FW and y + h <= FH and w % 32 == 0 and h % 32 == 0


def test_free_drag_any_direction_keeps_the_pressed_corner():
    assert G.crop_from_drag((100, 100), (420, 300), FW, FH) == (100, 100, 320, 192)
    x, y, w, h = G.crop_from_drag((900, 800), (500, 500), FW, FH)     # up and to the left
    assert (x + w, y + h) == (900, 800) and (w, h) == (384, 288)


def test_drag_past_an_edge_reaches_it():
    # 1080 is not a multiple of 32: the old box stopped at 1056 ("refuses to cover the lower edge")
    for aspect in (None, 1.0, 16 / 9, 9 / 16):
        box = G.crop_from_drag((300, 400), (FW + 50, FH + 50), FW, FH, aspect)
        assert inside(box)
        x, y, w, h = box
        assert x + w == FW or y + h == FH
    x, y, w, h = G.crop_from_drag((300, 400), (900, FH + 10), FW, FH)
    assert y + h == FH


def test_locked_shape_grows_smoothly_and_stays_near_the_shape():
    sizes = []
    for px in range(512, 1900, 8):
        box = G.crop_from_drag((0, 0), (px, px * 9 / 16), FW, FH, 16 / 9)
        assert inside(box)
        w, h = box[2], box[3]
        assert abs(w / h - 16 / 9) / (16 / 9) <= G.CROP_SHAPE_TOLERANCE
        sizes.append(w)
    # no jumping back and forth, and no big leaps (the old exact-ratio fit hopped 512 -> 1024)
    assert sizes == sorted(sizes)
    assert max(b - a for a, b in zip(sizes, sizes[1:])) <= 96


def test_locked_shape_follows_the_side_pulled_further():
    x, y, w, h = G.crop_from_drag((0, 0), (200, 900), FW, FH, 1.0)    # pulled down far, little across
    assert h >= 864 and w == h


def test_move_place_largest():
    box = (100, 100, 640, 640)
    assert G.crop_moved(box, 5000, 5000, FW, FH) == (FW - 640, FH - 640, 640, 640)
    assert G.crop_placed(box, 0, None, FW, FH) == (0, 100, 640, 640)
    assert G.crop_placed(box, 2, 2, FW, FH) == (FW - 640, FH - 640, 640, 640)
    assert G.crop_placed(box, 1, 1, FW, FH) == ((FW - 640) // 2, (FH - 640) // 2, 640, 640)
    assert G.crop_largest(FW, FH, 9 / 16) == ((FW - 576) // 2, (FH - 1024) // 2, 576, 1024)
    assert G.crop_largest(FW, FH)[2:] == (1920, 1056)


def test_typed_sizes():
    box = (1500, 0, 320, 320)
    x, y, w, h = G.crop_resized(box, w=700, frame_w=FW, frame_h=FH)
    assert (w, h) == (704, 320) and x + w <= FW                         # slid in, not past the edge
    x, y, w, h = G.crop_resized(box, h=576, frame_w=FW, frame_h=FH, aspect=1.0)
    assert (w, h) == (576, 576)


def test_saved_box_on_another_size():
    assert G.crop_rescaled((1240, 64, 640, 896), 1920, 1080, FW, FH) == (1240, 64, 640, 896)
    box = G.crop_rescaled((1240, 64, 640, 896), 1920, 1080, 1280, 720)
    x, y, w, h = box
    assert x + w <= 1280 and y + h <= 720 and (w, h) == (416, 608)


def test_export_keeps_the_crop_inside_the_frame():
    cmd = G.build_export_command("ffmpeg", "a.mp4", "b.mp4", 0.0, 25, 608, 1056, crop=(1312, 24, 608, 1056))
    vf = cmd[cmd.index("-vf") + 1]
    assert "crop=608:1056:'min(1312\\,iw-ow)':'min(24\\,ih-oh)'" in vf
