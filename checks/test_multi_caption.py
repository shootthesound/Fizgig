"""Several captions per picture (#300): a Training tab tick (off by default) reads each line of a caption file as a
caption of its own - line 1 cached where the one caption always was, the others beside it as `_tealt<k>`, one picked
at random each step. Off, a caption file is one caption, lines and all, exactly as before."""
import os
import re

import pytest
from PIL import Image

from fizgig.dataset import image_dataset as ids
from fizgig.families import form, launch
from fizgig.families.registry import get as get_family


def _pic(path, caption):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(caption)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "a.png"), "zwx, 1girl, red dress\nzwx, a woman in a red dress\n\n  zwx, a portrait  \n")
    _pic(str(d / "b.png"), "zwx, one caption only")
    return d


def test_caption_lines_strip_drop_blanks_and_fuse_each_line():
    rep = [(re.compile(r"(?<!\w)zwx(?!\w)", re.I), "zwxwoman")]
    assert ids.caption_lines(" a \n\n b \r\n c") == ["a", "b", "c"]
    assert ids.caption_lines("zwx, x\nzwx, y", rep) == ["zwxwoman, x", "zwxwoman, y"]


def test_alt_paths_sit_outside_the_te_glob():
    p = os.path.join("c", "a_klein9b_te.safetensors")
    assert ids.text_cache_alt_path(p, 2) == os.path.join("c", "a_klein9b_tealt2.safetensors")
    assert not ids.text_cache_alt_path(p, 1).endswith("_te.safetensors")


def test_datasource_off_reads_the_whole_file_on_keeps_lines(folder):
    off = ids.ImageDirectoryDatasource(str(folder), ".txt")
    on = ids.ImageDirectoryDatasource(str(folder), ".txt", multi_caption=True)
    i = off.image_paths.index(next(p for p in off.image_paths if p.endswith("a.png")))
    assert off.get_caption(i)[1].replace(chr(13), "") == "zwx, 1girl, red dress\nzwx, a woman in a red dress\n\n  zwx, a portrait"
    assert on.get_caption(i)[1] == "zwx, 1girl, red dress\nzwx, a woman in a red dress\nzwx, a portrait"


def _ds(folder, tmp_path, multi):
    return ids.ImageDataset(resolution=(256, 256), caption_extension=".txt", batch_size=1, num_repeats=1,
                            enable_bucket=True, bucket_no_upscale=True, image_directory=str(folder),
                            cache_directory=str(tmp_path / "cache"), architecture=get_family("klein").arch_id,
                            multi_caption=multi)


def test_text_cache_items_one_per_line_when_on(folder, tmp_path):
    def items(multi):
        out = []
        for batch in _ds(folder, tmp_path, multi).retrieve_text_encoder_output_cache_batches(1):
            out += [(os.path.basename(it.text_encoder_output_cache_path), it.caption) for it in batch]
        return sorted(out)
    off = items(False)
    assert len(off) == 2 and all(p.endswith("_te.safetensors") for p, _c in off)
    on = items(True)
    assert len(on) == 4
    assert ("a_" + get_family("klein").arch_id + "_te.safetensors", "zwx, 1girl, red dress") in on
    assert ("a_" + get_family("klein").arch_id + "_tealt2.safetensors", "zwx, a portrait") in on


def test_toml_flag_only_when_ticked(folder):
    desc = get_family("krea2")
    st = {"image_folder": str(folder), "megapixels": 0.25, "LORA_OUTPUT_DIR": str(folder / "out")}
    assert "multi_caption" not in launch.dataset_toml(desc, st)
    assert "multi_caption = true" in launch.dataset_toml(desc, {**st, "multi_caption": True})
    assert not launch.multi_caption_on(get_family("minimax"), {"multi_caption": True})      # H3: its own cache


def test_row_shows_only_for_multi_line_captions(folder, tmp_path):
    desc = get_family("klein")
    assert form.DEFAULTS["MULTI_CAPTION"] is False
    st = form.state(desc, {}, str(folder))
    assert "MULTI_CAPTION" not in st["hidden"]
    assert st["notes"]["MULTI_CAPTION"].startswith("1 of your 2 caption files have more than one line")
    st = form.state(desc, {"MULTI_CAPTION": True}, str(folder))
    assert st["notes"]["MULTI_CAPTION"].startswith("2 pictures, 4 captions (1 with more than one, up to 3 a picture)")
    plain = tmp_path / "plain"
    _pic(str(plain / "c.png"), "one line")
    assert "MULTI_CAPTION" in form.state(desc, {}, str(plain))["hidden"]


def test_trigger_checked_per_line(folder):
    desc = get_family("klein")
    with open(folder / "b.txt", "w", encoding="utf-8") as f:
        f.write("zwx, fine\nno trigger here")
    st = {"CAPTION_TRIGGER_WORD": "zwx", "DATASET_CAPTION_EXT": ".txt"}
    off = launch.start_checks(desc, st, str(folder))
    assert not any("trigger" in w for w in off)                  # off: the file as a whole has it
    on = launch.start_checks(desc, {**st, "MULTI_CAPTION": True}, str(folder))
    assert any(w.startswith("1 caption line in 1 picture doesn't contain the trigger word 'zwx'") for w in on)


def test_add_text_every_line_only_when_asked(tmp_path):
    from fizgig.web import captions
    d = tmp_path / "c"
    _pic(str(d / "a.png"), "zwx, tags\nzwx, a sentence")
    off = captions.add_text(str(d), "photo", "", "zwx", False)["changes"][0]["new"].replace(chr(13), "")
    assert off == "zwx, photo, tags\nzwx, a sentence"                     # as before: the file is one caption
    on = captions.add_text(str(d), "photo", "", "zwx", False, each_line=True)["changes"][0]["new"]
    assert on == "zwx, photo, tags\nzwx, photo, a sentence"


def test_captioner_adds_a_line_once(tmp_path):
    from fizgig.scripts.batch_caption import _write_caption
    p = str(tmp_path / "a.png")
    _write_caption(p, "zwx, tags", ".txt", add_line=True)                 # no file yet: its first
    _write_caption(p, "zwx, a sentence", ".txt", add_line=True)
    _write_caption(p, "zwx, a sentence", ".txt", add_line=True)           # already there: not twice
    assert open(str(tmp_path / "a.txt"), encoding="utf-8").read() == "zwx, tags\nzwx, a sentence"
    _write_caption(p, "replaced", ".txt")
    assert open(str(tmp_path / "a.txt"), encoding="utf-8").read() == "replaced"
