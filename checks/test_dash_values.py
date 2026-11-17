"""A box's value that starts with "-" (a negative "-blurry", a title "-test", a trigger word "-gwelt") reaches the
trainer as typed. argparse reads a lone "-word" after a flag as another option, so such a value stopped the run at
launch ("expected one argument"); the launch now sends it as --flag=value. Numbers ("-1") and every other value go
as before, so a command without such a value is unchanged."""
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get
from fizgig.families.train import setup_parser

DASHED = {"METADATA_TITLE": "-test", "METADATA_AUTHOR": "-me", "METADATA_DESCRIPTION": "-a LoRA",
          "METADATA_LICENSE": "-mit", "METADATA_TAGS": "-tag", "METADATA_TRIGGER_PHRASE": "-gwelt",
          "SAMPLE_NEGATIVE": "-blurry", "SAMPLE_CFG_SCALE": "3.5"}


def _cmd(tmp_path, **values):
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    return next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")


def test_values_that_start_with_a_dash_reach_the_trainer(tmp_path):
    args = setup_parser().parse_args(_cmd(tmp_path, **DASHED)[2:])
    assert (args.metadata_title, args.metadata_author, args.metadata_description, args.metadata_license,
            args.metadata_tags, args.metadata_trigger_phrase, args.sample_negative) == \
        ("-test", "-me", "-a LoRA", "-mit", "-tag", "-gwelt", "-blurry")


def test_other_values_and_numbers_go_as_before(tmp_path):
    plain = {k: v.lstrip("-") for k, v in DASHED.items()}
    cmd = _cmd(tmp_path, **plain)
    assert not [t for t in cmd if t.startswith("--") and "=" in t]
    assert cmd[cmd.index("--metadata_title") + 1] == "test"
    assert launch._dash_values(["--blocks_to_swap", "-1", "--seed", "-5", "--x", "-1e-4"]) == \
        ["--blocks_to_swap", "-1", "--seed", "-5", "--x", "-1e-4"]
    assert launch._dash_values(["--masked_training", "--mask_floor", "0.1"]) == \
        ["--masked_training", "--mask_floor", "0.1"]
    assert launch._dash_values(["--slider_prompts", " -a", " -b"]) == ["--slider_prompts", " -a", " -b"]
