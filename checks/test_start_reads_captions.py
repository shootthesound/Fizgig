"""Start reads captions as training does. Training reads a caption saved as UTF-16 (Windows PowerShell's `>`, Notepad's
"Unicode") or Windows-1252 (dataset/caption_file.py, #225); Start's warnings, the regularisation check and Isolation's
subject captions read UTF-8 only, so a folder of UTF-16 captions trained fine but Start warned that none of them had
the trigger word. One reader now, shared, quiet for the checks."""
import sys

from PIL import Image

from fizgig.dataset import caption_file, image_dataset
from fizgig.families import form, launch
from fizgig.families.registry import get

TEXT = "gwelt, a man in a café"
ENCODINGS = {"utf8": TEXT.encode("utf-8"), "utf8bom": TEXT.encode("utf-8-sig"), "utf16": TEXT.encode("utf-16"),
             "cp1252": TEXT.encode("cp1252")}


def test_one_reader_for_training_and_start(tmp_path, caplog):
    assert image_dataset.decode_caption is caption_file.decode_caption        # the training set's own
    for name, raw in ENCODINGS.items():
        p = tmp_path / f"{name}.txt"
        p.write_bytes(raw)
        assert caption_file.read_caption(str(p)) == TEXT, name
    assert "UTF-8" not in caplog.text and "UTF-16" not in caplog.text          # the checks' reads say nothing


def test_start_finds_the_trigger_in_a_utf16_caption(tmp_path):
    folder = tmp_path / "pics"
    folder.mkdir()
    for i, (name, raw) in enumerate(ENCODINGS.items()):
        Image.new("RGB", (512, 512), (90, 60, 30)).save(folder / f"{name}.png")
        (folder / f"{name}.txt").write_bytes(raw)
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "CAPTION_TRIGGER_WORD": "gwelt",
         "DATASET_CONFIG": str(tmp_path / "d.toml")}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(folder),
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    warnings = launch.start_checks(desc, inp, str(folder))
    assert not [w for w in warnings if "trigger word" in w or "no caption" in w], warnings


def test_an_edit_runs_preview_instruction_from_any_caption(tmp_path):
    """An Edit LoRA with no instruction typed previews with its first photo's caption; a Windows-1252 one raised
    UnicodeDecodeError out of the plan (not the OSError it caught), so Start failed."""
    for name in ("cp1252", "utf16"):
        folder = tmp_path / name
        folder.mkdir()
        Image.new("RGB", (64, 64)).save(folder / "a.png")
        (folder / "a.txt").write_bytes(ENCODINGS[name])
        assert launch.edit_instructions({"image_folder": str(folder), "caption_ext": ".txt"}) == [TEXT], name
