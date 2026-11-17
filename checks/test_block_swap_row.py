"""Blocks Swap only where the model can swap blocks. SDXL, Anima and ACE-Step can't (their drivers' max_blocks_to_swap is
0, so training loads without block swap whatever number it gets), yet the Training tab offered 1-16, and on SDXL
Start refused the listed 12 and 16 with "exceeds maximum (9)" while 1-9 did nothing. Now they have no Blocks Swap row
and Start never refuses a block swap there; every other model's row and Start check are unchanged."""
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get, training_families

NO_SWAP = {"sdxl", "anima", "acestep15xl"}


def _problems(tmp_path, key, swap):
    desc, models = get(key), {}
    for mf in desc.model_files:
        f = tmp_path / f"{key}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "2",
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "BLOCKS_SWAP": swap}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    return [p for p in launch.start_problems(desc, inp) if "Blocks swap" in p]


def test_the_row_follows_the_driver():
    for desc in training_families():
        swaps = desc.load_driver().max_blocks_to_swap() > 0
        assert launch.swaps_blocks(desc) == swaps, desc.key
        assert swaps == (desc.key not in NO_SWAP), desc.key
        keys = {r["key"] for s in form.form(desc)["sections"] for r in s["rows"]}
        assert ("BLOCKS_SWAP" in keys) == swaps, desc.key


def test_start_never_refuses_a_block_swap_the_model_ignores(tmp_path):
    for key in NO_SWAP:
        for swap in ("4  (Light — 14GB fp8 / 20GB bf16)", "12 (Aggressive — 10GB fp8 / 12GB bf16)",
                     "16 (Max — 8GB fp8 / 10GB bf16)", "40"):
            assert _problems(tmp_path, key, swap) == [], (key, swap)


def test_the_others_are_checked_as_before(tmp_path):
    assert _problems(tmp_path, "klein", "16 (Max — 8GB fp8 / 10GB bf16)") == []
    assert _problems(tmp_path, "klein", "Auto (detect from GPU)") == []
    out = _problems(tmp_path, "klein", "31")
    assert len(out) == 1 and "exceeds maximum" in out[0]
