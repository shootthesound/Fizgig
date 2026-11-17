"""The Captions tab lists, and its captioners caption, every picture format training reads. .avif trains, but the
Captions tab left it out, so a folder of .avif pictures showed nothing to caption while Start asked for captions."""
from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
from fizgig.scripts import batch_caption
from fizgig.web import captions


def test_captions_cover_every_training_format(tmp_path):
    trains = {e.lower() for e in IMAGE_EXTENSIONS} - {".jxl"}     # JPEG XL only with an optional plugin installed
    assert trains <= captions.IMAGE_EXTS
    assert trains <= batch_caption.IMAGE_EXTENSIONS
    assert captions.IMAGE_EXTS == batch_caption.IMAGE_EXTENSIONS
    for n in ("a.avif", "b.JPG", "c.png", "c-masklabel.png", "d.txt"):
        (tmp_path / n).write_bytes(b"x")
    listed = [p.replace("\\", "/").rsplit("/", 1)[-1] for p in captions.media_files(str(tmp_path), clips=False)]
    assert listed == ["a.avif", "b.JPG", "c.png"]
