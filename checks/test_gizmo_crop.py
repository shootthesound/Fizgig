"""Gizmo's crop box (#350, #367): drawn exactly as dragged, in any direction, the pressed corner staying
put and an edge the pointer passes holding flush; moved by dragging inside it; resized from its corners and
sides; Place arrows, Largest, typed sizes and saved boxes. Any size: the export fits it to the 32 grid as it
fits the whole frame. (#367: the box snapped to 32 while dragged and slid to an edge dragged past, so dragging
up from the bottom made it jump and it couldn't be laid flush with the bottom.)"""
import os
import sys

import pytest

pytest.importorskip("tkinter")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "gizmo"))
import gizmo as G  # noqa: E402

FW, FH = 1920, 1080


def inside(box, fw=FW, fh=FH):
    x, y, w, h = box
    return x >= 0 and y >= 0 and x + w <= fw and y + h <= fh


def test_a_drag_is_the_box_drawn_from_the_pressed_corner_any_direction():
    assert G.crop_from_drag((100, 100), (420, 300), FW, FH) == (100, 100, 320, 200)
    assert G.crop_from_drag((900, 800), (500, 500), FW, FH) == (500, 500, 400, 300)     # up and to the left
    sizes = [G.crop_from_drag((900, 800), (900 - d, 800 - d), FW, FH)[2] for d in range(40, 600, 7)]
    assert sizes == list(range(40, 600, 7))                     # follows the pointer, no 32-pixel steps


def test_dragged_past_an_edge_it_holds_flush_and_the_pressed_corner_stays():
    assert G.crop_from_drag((300, 400), (FW + 50, FH + 50), FW, FH) == (300, 400, FW - 300, FH - 400)
    # #367: up from the bottom of a 640x476 frame, past the top - flush top AND bottom, never jumping
    assert G.crop_from_drag((300, 476), (100, -40), 640, 476) == (100, 0, 200, 476)
    assert G.crop_from_drag((300, 500), (100, 100), 640, 476) == (100, 100, 200, 376)   # pressed below the frame
    for aspect in (1.0, 16 / 9, 9 / 16):
        box = G.crop_from_drag((300, 400), (FW + 50, FH + 50), FW, FH, aspect)
        assert inside(box) and box[:2] == (300, 400)


def test_locked_shape_grows_smoothly_from_the_side_pulled_further():
    widths = []
    for px in range(200, 1900, 8):
        x, y, w, h = G.crop_from_drag((0, 0), (px, px * 9 / 16), FW, FH, 16 / 9)
        assert abs(w / h - 16 / 9) < 0.01 and (x, y) == (0, 0)
        widths.append(w)
    assert widths == sorted(widths) and max(b - a for a, b in zip(widths, widths[1:])) <= 9
    assert G.crop_from_drag((0, 0), (200, 900), FW, FH, 1.0) == (0, 0, 900, 900)
    assert G.crop_from_drag((1800, 100), (1900, 900), FW, FH, 1.0) == (1800, 100, 120, 120)   # as far as the frame allows


def test_a_click_makes_no_box():
    assert G.crop_from_drag((100, 100), (110, 120), FW, FH) is None


def test_resize_from_a_corner_or_side():
    box = (100, 100, 400, 300)
    assert G.crop_resize_drag(box, "se", (700, 600), FW, FH) == (100, 100, 600, 500)
    assert G.crop_resize_drag(box, "nw", (600, 500), FW, FH) == (500, 400, 100, 100)   # the far corner stays
    assert G.crop_resize_drag(box, "n", (0, 50), FW, FH) == (100, 50, 400, 350)
    assert G.crop_resize_drag(box, "w", (-80, 0), FW, FH) == (0, 100, 500, 300)       # flush with the left
    assert G.crop_resize_drag(box, "s", (0, FH + 40), FW, FH) == (100, 100, 400, FH - 100)
    x, y, w, h = G.crop_resize_drag((600, 100, 400, 300), "s", (0, 700), FW, FH, 4 / 3)   # a side, shape locked
    assert (x, y, w, h) == (400, 100, 800, 600)                                       # widened about its centre
    assert G.crop_resize_drag(box, "s", (0, 700), FW, FH, 4 / 3)[0] == 0              # ... kept inside the frame


def test_move_place_largest():
    box = (100, 100, 640, 640)
    assert G.crop_moved(box, 5000, 5000, FW, FH) == (FW - 640, FH - 640, 640, 640)
    assert G.crop_moved((100, 0, 320, 224), 0, 5000, 640, 476) == (100, 252, 320, 224)   # flush bottom (#367)
    assert G.crop_placed(box, 0, None, FW, FH) == (0, 100, 640, 640)
    assert G.crop_placed(box, 2, 2, FW, FH) == (FW - 640, FH - 640, 640, 640)
    assert G.crop_placed(box, 1, 1, FW, FH) == ((FW - 640) // 2, (FH - 640) // 2, 640, 640)
    assert G.crop_largest(FW, FH, 9 / 16) == (656, 0, 608, 1080)
    assert G.crop_largest(FW, FH) == (0, 0, FW, FH)


def test_typed_sizes():
    box = (1500, 0, 320, 320)
    assert G.crop_resized(box, w=700, frame_w=FW, frame_h=FH) == (1220, 0, 700, 320)   # slid in, not past the edge
    assert G.crop_resized(box, h=576, frame_w=FW, frame_h=FH, aspect=1.0) == (1344, 0, 576, 576)


def test_saved_box_on_another_size():
    assert G.crop_rescaled((1240, 64, 640, 896), 1920, 1080, FW, FH) == (1240, 64, 640, 896)
    box = G.crop_rescaled((1240, 64, 640, 896), 1920, 1080, 1280, 720)
    assert inside(box, 1280, 720) and box == (827, 43, 427, 597)


def test_any_box_exports_on_the_32_grid_at_its_shape():
    for w, h in ((615, 453), (640, 476), (333, 333), (1001, 563)):
        tw, th = G.target_size(w, h, 99.0)
        assert tw % 32 == 0 and th % 32 == 0 and tw <= w and th <= h
        assert abs((tw / th) / (w / h) - 1) < 0.03


def test_whole_frame_export_of_non_square_pixels_keeps_its_edges(tmp_path):
    """#367: a 720x476 clip shown 640 wide (8:9 pixels), exported at Whole frame to 608x448. Only a drawn crop
    corrected the pixel shape, so the whole frame was stretched sideways and ~30 px cut off each side. A coloured
    band at each edge must still be there after the export."""
    import subprocess
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    from PIL import Image
    src, dst, png = tmp_path / "anamorphic.mp4", tmp_path / "out.mp4", tmp_path / "frame.png"
    bands = ("color=c=gray:s=720x476:r=24:d=1[bg];color=c=red:s=24x476:r=24:d=1[l];color=c=blue:s=24x476:r=24:d=1[r];"
             "[bg][l]overlay=0:0[a];[a][r]overlay=696:0,setsar=8/9")
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc=s=16x16:d=1",
                    "-filter_complex", bands, "-frames:v", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                   check=True)
    info = G.probe_source(ffmpeg, str(src))
    assert (info["display_width"], info["display_height"]) == (640, 476)
    cmd = G.build_export_command(ffmpeg, str(src), str(dst), 0.0, 8, 608, 448, with_audio=False, sar=info["sar"])
    assert "scale=iw*sar:ih" in cmd[cmd.index("-vf") + 1]                 # with no crop box too
    subprocess.run(cmd, check=True)
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(dst), "-frames:v", "1", str(png)],
                   check=True)
    im = Image.open(png).convert("RGB")
    assert im.size == (608, 448)
    left, right = im.getpixel((2, 224)), im.getpixel((605, 224))
    assert left[0] > 150 and left[2] < 100                                # red band still at the left edge
    assert right[2] > 150 and right[0] < 100                              # blue band still at the right edge


def test_export_keeps_the_crop_inside_the_frame():
    cmd = G.build_export_command("ffmpeg", "a.mp4", "b.mp4", 0.0, 25, 608, 1056, crop=(1312, 24, 608, 1056))
    vf = cmd[cmd.index("-vf") + 1]
    assert "crop=608:1056:'min(1312\\,iw-ow)':'min(24\\,ih-oh)'" in vf
