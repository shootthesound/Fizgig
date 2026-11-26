"""Image Prep's Find duplicate pictures (web/prep_jobs.py dupes_*): the training folder's pictures are scanned, a
resized copy groups with its original, Move out puts the marked one (caption and mask too) in excluded_duplicates/
and takes it out of the groups, and a run that trains sub-folders never trains that folder."""
import os
import time

import numpy as np
from PIL import Image

from fizgig.families.launch import is_own_subfolder
from fizgig.web import prep_jobs


def _pic(path, size=(96, 80), solid=None):
    if solid:
        Image.new("RGB", size, solid).save(path)
        return
    y, x = np.indices((size[1], size[0]))
    rgb = np.stack(((x * 7 + y * 2) % 256, (x * 3 + y * 13) % 256, (x * 11 + y * 3) % 256), axis=2).astype("uint8")
    rgb[7:24, 12:45] = (10, 250, 15)
    Image.fromarray(rgb, "RGB").save(path)


def _done(folder):
    for _ in range(300):
        r = prep_jobs.dupes_status(folder, 4)
        if r["stage"] in ("done", "error"):
            return r
        time.sleep(0.02)
    raise AssertionError("the scan didn't finish")


def test_scan_group_and_move_out(tmp_path):
    folder = str(tmp_path / "set")
    os.makedirs(folder)
    _pic(os.path.join(folder, "a.png"))
    with Image.open(os.path.join(folder, "a.png")) as im:
        im.resize((192, 160)).save(os.path.join(folder, "a_big.png"))
    _pic(os.path.join(folder, "other.png"), solid="red")
    for n in ("a", "a_big", "other"):
        open(os.path.join(folder, n + ".txt"), "w").write(n)
    _pic(os.path.join(folder, "a-masklabel.png"), solid="white")           # a mask is never a picture to compare
    prep_jobs.dupes_scan(folder)
    r = _done(folder)
    assert r["stage"] == "done" and r["scanned"] == 3
    names = [{i["name"] for i in g["items"]} for g in r["groups"]]
    assert {"a.png", "a_big.png"} in names and not any("other.png" in n for n in names)
    m = prep_jobs.dupes_move(folder, [os.path.join(folder, "a.png"), os.path.join(folder, "other.png")], strictness=4)
    assert m["moved"] == 1                                                   # other.png wasn't in a group
    dest = os.path.join(folder, "excluded_duplicates")
    assert sorted(os.listdir(dest)) == ["a-masklabel.png", "a.png", "a.txt"]
    assert os.path.exists(os.path.join(folder, "other.png"))
    assert prep_jobs.dupes_status(folder, 4)["group_count"] == 0


def test_the_moved_out_folder_is_fizgigs_own():
    assert is_own_subfolder("excluded_duplicates") and is_own_subfolder("excluded_by_look")
