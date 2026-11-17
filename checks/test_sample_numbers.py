"""The Samples tab's number boxes at Start, as the Training tab's are. A CFG typed with a decimal comma ("3,5") fell back to
the model's default CFG with nothing said, a seed that wasn't a whole number was dropped, and Steps reached the
trainer, whose argparse stopped the run after the cache stages. Start now names the box; empty boxes mean the model's
own values, and with previews off nothing is checked."""
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get


def _problems(tmp_path, **values):
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    return [p for p in launch.start_problems(desc, inp) if p.startswith("Samples:")]


def test_samples_boxes_that_are_not_numbers(tmp_path):
    assert _problems(tmp_path, SAMPLE_CFG_SCALE="3,5") == ["Samples: CFG Scale must be a number (got '3,5')"]
    assert _problems(tmp_path, SAMPLE_STEPS="20,5") == ["Samples: Steps must be a number (got '20,5')"]
    assert _problems(tmp_path, SAMPLE_SEED="12 34") == ["Samples: Seed must be a number (got '12 34')"]
    assert _problems(tmp_path, SAMPLE_STEPS="0") == ["Samples: Steps must be at least 1 (got 0)"]


def test_good_empty_or_previews_off(tmp_path):
    assert _problems(tmp_path, SAMPLE_CFG_SCALE="3.5", SAMPLE_STEPS="20", SAMPLE_SEED="-1") == []
    assert _problems(tmp_path, SAMPLE_CFG_SCALE="", SAMPLE_STEPS="", SAMPLE_SEED="") == []
    assert _problems(tmp_path, SAMPLE_ENABLED=False, SAMPLE_CFG_SCALE="3,5", SAMPLE_STEPS="x") == []


def test_one_message_per_bad_box(tmp_path):
    """The family Start named a bad rank / alpha / epochs / save cadence twice (network() and numbers(), then
    checks.run again in other words); once now. RefMod's Start, which has only checks.run, still checks them."""
    from fizgig.families import checks
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "NETWORK_DIM": "4,0", "NETWORK_ALPHA": "x",
         "MAX_TRAIN_EPOCHS": "0", "SAVE_EVERY_N_EPOCHS": "a"}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    out = launch.start_problems(desc, inp)
    for word in ("dim", "alpha", "max train", "save every"):
        assert len([p for p in out if word in p.lower()]) == 1, (word, out)
    rm = {"NETWORK_DIM": "x", "NETWORK_ALPHA": "1", "MAX_TRAIN_EPOCHS": "1", "SAVE_EVERY_N_EPOCHS": "1",
          "LORA_OUTPUT_DIR": "o"}
    assert checks.run(rm, blocks_swap=0, swap_max=4, arch_label="A", name_error="") == \
        ["Network dim must be a valid integer"]
