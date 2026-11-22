"""A training folder of clips only (#402): the Training tab hides Masked training and Element boost for it, so the run
leaves them off too - Start doesn't refuse for missing masks, and the train command gets neither flag - even when
the ticks were saved on with a photo folder. A folder with photos keeps them; photos in a training sub-folder count."""
import sys

from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get

MASK_MSG = "Masked training is on"


def _run(tmp_path, folder, **values):
    desc = get("minimax")
    models = {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "MASKED_TRAINING": True, "FACE_BOOST": True,
         "DATASET_MEGAPIXELS": "1.0", **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(folder),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    hidden = form.state(desc, v, str(folder))["hidden"]
    return desc, v, inp, hidden


def _clips(folder, n=2):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"c{i}.mp4").write_bytes(b"0")
        (folder / f"c{i}.txt").write_text("zwxem a clip")


def test_a_clips_only_folder_trains_without_masks(tmp_path):
    d = tmp_path / "clips"
    _clips(d)
    desc, v, inp, hidden = _run(tmp_path, d)
    assert "masks" in hidden and "FACE_BOOST" in hidden
    assert not launch.masked_on(desc, inp) and launch.face_weight(desc, inp) is None
    assert not any(MASK_MSG in p for p in launch.start_problems(desc, inp))


def test_a_photo_folder_keeps_the_mask_check(tmp_path):
    d = tmp_path / "mixed"
    _clips(d)
    Image.new("RGB", (64, 64)).save(d / "p.png")
    (d / "p.txt").write_text("zwxem a photo")
    desc, v, inp, hidden = _run(tmp_path, d)
    assert "masks" not in hidden and launch.masked_on(desc, inp)
    assert any(MASK_MSG in p for p in launch.start_problems(desc, inp))     # no mask beside the photo: refused


def test_photos_in_a_training_sub_folder_count(tmp_path):
    d = tmp_path / "top"
    _clips(d)
    sub = d / "stills"
    sub.mkdir()
    Image.new("RGB", (64, 64)).save(sub / "p.png")
    Image.new("L", (64, 64), 255).save(sub / "p-masklabel.png")
    (sub / "p.txt").write_text("zwxem a photo")
    desc, v, inp, hidden = _run(tmp_path, d, TRAIN_SUBFOLDERS=True)
    assert "masks" not in hidden and launch.masked_on(desc, inp)
    assert not any(MASK_MSG in p for p in launch.start_problems(desc, inp))
    desc, v, inp, hidden = _run(tmp_path, d)                                  # sub-folders not trained: clips only
    assert "masks" in hidden and not launch.masked_on(desc, inp)
