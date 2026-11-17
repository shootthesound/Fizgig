"""Auto-recaption finds a stuck picture's file as the training set does. It guessed the name with lowercase extensions
(name + ".jpg"), so on a Linux pod, where names are case-sensitive, a camera's IMG_0001.JPG was never found and was
silently never recaptioned; .avif pictures, which train, were never found anywhere. It now matches the extension in
any case and every format the training set reads, and returns the file's real name."""
from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
from fizgig.families.loss_watch import _picture, _picture_exts


def test_every_training_format_is_looked_for():
    assert set(_picture_exts()) == {e.lower() for e in IMAGE_EXTENSIONS}
    assert _picture_exts()[0] == ".png"                  # the old first choice stays first


def test_the_real_file_whatever_the_case(tmp_path):
    for n in ("IMG_0001.JPG", "IMG_0002.Png", "scan.avif", "plain.jpg", "plain-masklabel.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    listings = {}
    folder = str(tmp_path)
    assert _picture(folder, "IMG_0001", listings).endswith("IMG_0001.JPG")      # the name as it is on disk
    assert _picture(folder, "IMG_0002", listings).endswith("IMG_0002.Png")
    assert _picture(folder, "scan", listings).endswith("scan.avif")
    assert _picture(folder, "plain", listings).endswith("plain.jpg")              # not its mask
    assert _picture(folder, "img_0001", listings) == ""                          # the stem's own case counts
    assert _picture(folder, "notes", listings) == "" and _picture(folder, "missing", listings) == ""
    assert _picture(str(tmp_path / "gone"), "x", listings) == ""
    assert list(listings) == [folder, str(tmp_path / "gone")]                      # each folder listed once


def test_png_wins_over_another_format_of_the_same_name(tmp_path):
    for n in ("a.jpg", "a.png"):
        (tmp_path / n).write_bytes(b"x")
    assert _picture(str(tmp_path), "a", {}).endswith("a.png")
