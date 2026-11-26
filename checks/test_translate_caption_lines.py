"""Translate captions, a caption file of several lines (Several captions per picture, the captioner's Add as another
line): each line is translated on its own, keeping its trigger. Translated whole, the first line stayed English and
the Chinese of every line landed on the last one."""
import sys

from fizgig.scripts import translate_captions as tc


def _run(monkeypatch, folder, *args):
    seen = []

    def fake(tok, model, device, english):
        seen.append(english)
        return "中" + str(len(seen))

    monkeypatch.setattr(tc, "_load", lambda: (None, None, "cpu"))
    monkeypatch.setattr(tc, "_translate", fake)
    monkeypatch.setattr(sys, "argv", ["translate_captions", str(folder), *args])
    assert tc.main() == 0
    return seen


def test_each_line_is_translated_on_its_own(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("zwxem, a woman, red hair\nzwxem, a woman smiling at a cafe table\n",
                                    encoding="utf-8")
    seen = _run(monkeypatch, tmp_path)
    assert seen == ["a woman, red hair", "a woman smiling at a cafe table"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "zwxem, a woman, red hair - 中1", "zwxem, a woman smiling at a cafe table - 中2"]


def test_a_one_line_caption_is_written_as_before(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("zwxem, a woman", encoding="utf-8")
    (tmp_path / "b.txt").write_text("a red door", encoding="utf-8")
    _run(monkeypatch, tmp_path)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "zwxem, a woman - 中1"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "a red door - 中2"


def test_skip_existing_translates_only_the_line_added_since(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("zwxem, a woman - 女人\nzwxem, a woman at a cafe", encoding="utf-8")
    (tmp_path / "b.txt").write_text("zwxem, a man - 男人", encoding="utf-8")
    seen = _run(monkeypatch, tmp_path, "--skip_existing")
    assert seen == ["a woman at a cafe"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "zwxem, a woman - 女人", "zwxem, a woman at a cafe - 中1"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "zwxem, a man - 男人"
