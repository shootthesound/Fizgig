"""Asking a PNG how big it is must not inflate the picture to look for a rotation flag it does not have.
Pillow's PngImageFile.getexif() calls load() when open() never saw an eXIf chunk, so every plain PNG paid a
full decode for the answer: 17.8 ms a file, 8.4 s for the Captions tab's first load of one 518-picture folder.
The chunk walk answers in 137 us instead, and a picture that really does carry an orientation is still read the
certain way, wherever in the file its eXIf chunk was written."""
import struct
import zlib

import pytest
from PIL import Image

from fizgig.utils import upright
from fizgig.utils.upright import png_carries_orientation, upright_size

W, H = 4, 8                       # not square: a quarter turn swaps the sides, so a wrong answer shows
ORIENTATION_6 = 6                 # rotate 90 degrees clockwise

# the shortest EXIF that says orientation 6: little-endian TIFF, one IFD entry, no next IFD
TIFF_ORIENTATION_6 = (b"II\x2a\x00\x08\x00\x00\x00"
                      b"\x01\x00"
                      b"\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00"
                      b"\x00\x00\x00\x00")


def _chunk(typ, data):
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def _save_png(path, exif=None):
    im = Image.new("RGB", (W, H))
    if exif is None:
        im.save(path, "PNG")
        return
    e = Image.Exif()
    e[0x0112] = exif
    im.save(path, "PNG", exif=e)


def _late_exif_png(path):
    """A PNG whose eXIf chunk was written after its image data, which is where Pillow's open() stops reading."""
    raw = path.read_bytes()
    head, iend = raw[:-12], raw[-12:]           # IEND is always the last chunk and always 12 bytes
    path.write_bytes(head + _chunk(b"eXIf", TIFF_ORIENTATION_6) + iend)


def test_plain_png_answers_without_inflating(tmp_path):
    p = tmp_path / "plain.png"
    _save_png(p)
    with Image.open(p) as im:
        def no_decode(*a, **k):
            raise AssertionError("the picture was inflated to ask it how big it is")
        im.load = no_decode
        assert upright_size(im, str(p)) == (W, H)


def test_png_written_with_an_orientation_is_read_upright(tmp_path):
    p = tmp_path / "turned.png"
    _save_png(p, ORIENTATION_6)
    with Image.open(p) as im:
        assert upright_size(im, str(p)) == (H, W)


def test_jpeg_orientation_is_unchanged(tmp_path):
    p = tmp_path / "photo.jpg"
    e = Image.Exif()
    e[0x0112] = ORIENTATION_6
    Image.new("RGB", (W, H)).save(p, "JPEG", exif=e)
    with Image.open(p) as im:
        assert upright_size(im, str(p)) == (H, W)


def test_orientation_written_after_the_image_data_is_still_found(tmp_path):
    p = tmp_path / "late.png"
    _save_png(p)
    _late_exif_png(p)
    assert png_carries_orientation(str(p)) is True
    with Image.open(p) as im:
        assert "exif" not in im.info                       # open() never reached the chunk
        assert upright_size(im, str(p)) == (H, W)


def test_asking_without_the_path_is_exactly_as_before(tmp_path):
    p = tmp_path / "late2.png"
    _save_png(p)
    _late_exif_png(p)
    with Image.open(p) as im:
        assert upright_size(im) == (H, W)


@pytest.mark.parametrize("prepare", [
    lambda p: p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x01"),      # stops mid-chunk
    lambda p: p.write_bytes(b"not a png at all"),                             # not a PNG
])
def test_a_file_the_walk_cannot_answer_for_is_asked_the_slow_way(tmp_path, prepare):
    p = tmp_path / "broken.png"
    prepare(p)
    assert png_carries_orientation(str(p)) is True


def test_a_walk_that_hits_its_chunk_ceiling_asks_the_slow_way(tmp_path, monkeypatch):
    p = tmp_path / "plain.png"
    _save_png(p)
    monkeypatch.setattr(upright, "_MAX_CHUNKS", 1)     # IHDR is the first chunk it reads and the last one allowed
    assert png_carries_orientation(str(p)) is True
