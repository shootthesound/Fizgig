"""Two pictures with one name (a.jpg and a.png) share one caption file and one cache file name (the cache is named by
the stem: a_0512x0512_<arch>.safetensors), so at the same size the second overwrites the first and only one trains,
with nothing said. Start now warns, naming the pairs."""
import sys

from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get


def _warnings(tmp_path, names):
    folder = tmp_path / "pics"
    folder.mkdir()
    for n in names:
        Image.new("RGB", (512, 512), (90, 60, 30)).save(folder / n)
        (folder / n).with_suffix(".txt").write_text("gwelt, a man", encoding="utf-8")
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "CAPTION_TRIGGER_WORD": "gwelt",
         "DATASET_CONFIG": str(tmp_path / "d.toml")}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(folder),
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    return [w for w in launch.start_checks(desc, inp, str(folder)) if "used by two pictures" in w]


def test_twins_are_named(tmp_path):
    w = _warnings(tmp_path, ["a.jpg", "a.png", "b.png", "c.webp", "c.bmp"])
    assert len(w) == 1 and w[0].startswith("2 names are used by two pictures (a.jpg and a.png, c.bmp and c.webp)")


def test_no_twins_no_warning(tmp_path):
    assert _warnings(tmp_path, ["a.jpg", "b.png", "a-masklabel.png"]) == []      # a mask is not a twin
