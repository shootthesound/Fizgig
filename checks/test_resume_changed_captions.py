"""#409: a caption edited between runs (the Problem Images window, the Captions tab) is encoded again when the run is
continued. A resume skips caching, so it trained on the old caption; now a caption saved after its text cache runs
the text stage (alone: the pictures' latents are still good)."""
import os
import sys
import time

from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family


def _setup(tmp_path):
    desc = get_family("moss_image")
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(d / f"p{i}.png")
        (d / f"p{i}.txt").write_text("zwxem a photo")
    models = {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)

    def plan(**values):
        v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **values}
        inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(d),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        return launch.plan(desc, inp)

    cdir = launch.cache_dir_for(str(tmp_path / "cache"), str(d))
    os.makedirs(cdir)
    old = time.time() - 600
    for i in range(2):                  # both pictures cached after their captions were written
        for f in (f"p{i}_0064x0064_{desc.arch_id}.safetensors", f"p{i}_{desc.arch_id}_te.safetensors"):
            open(os.path.join(cdir, f), "wb").close()
        os.utime(d / f"p{i}.txt", (old - 60, old - 60))
        os.utime(os.path.join(cdir, f"p{i}_{desc.arch_id}_te.safetensors"), (old, old))
    return d, plan


def _names(p):
    assert not p.problems, p.problems
    return [s.name for s in p.stages]


def _state(tmp_path):
    s = tmp_path / "out" / "t-000010-state"
    s.mkdir(parents=True)
    for f in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        (s / f).write_bytes(b"x")
    return str(s)


def test_a_resume_with_nothing_changed_skips_caching(tmp_path):
    d, plan = _setup(tmp_path)
    p = plan(RESUME_TRAINING=_state(tmp_path))
    assert "Text Encoder Caching" not in _names(p) and "Cache Preparation" not in _names(p)


def test_a_caption_edited_after_the_run_is_encoded_again_on_resume(tmp_path):
    d, plan = _setup(tmp_path)
    (d / "p1.txt").write_text("zwxem a photo of a red door")          # saved now: after its text cache
    p = plan(RESUME_TRAINING=_state(tmp_path))
    assert "Text Encoder Caching" in _names(p) and "Cache Preparation" not in _names(p)
    assert any("1 caption(s) changed" in line for line in p.console)
    assert _names(p).index("Text Encoder Caching") < _names(p).index("Training")
    assert launch.changed_captions(p.dataset_toml) == 1


def test_a_fresh_start_caches_as_before(tmp_path):
    d, plan = _setup(tmp_path)
    (d / "p1.txt").write_text("zwxem a photo of a red door")
    p = plan()
    assert "Cache Preparation" in _names(p) and _names(p).count("Text Encoder Caching") == 1
