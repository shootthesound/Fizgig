"""A prompt slider's prompts and anchors that start with "-" ("-a photo of a man,"): the launch gives each a space
before it, so the trainer's argparse takes it as a value and not an option, and the trainer strips it - it trains as
typed. Before, such a prompt stopped the trainer at launch."""
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import _anchor_lines, _prompt_texts, setup_parser

ARGS = ["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n"]


def _train_cmd(tmp_path, fam, **values):
    desc, models = get_family(fam), {}
    for mf in desc.model_files:
        f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "4",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning",
         "FAMILY_SLIDER_GUIDANCE": "3", **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    return next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")


def test_a_prompt_or_anchor_that_starts_with_a_dash_reaches_the_trainer(tmp_path):
    cmd = _train_cmd(tmp_path, "krea2", FAMILY_SLIDER_BASE="-a photo of a man,",
                     FAMILY_SLIDER_ANCHORS="-a photo of a dog")
    i, j = cmd.index("--slider_prompts"), cmd.index("--slider_anchors")
    assert cmd[i + 1:i + 4] == [" -a photo of a man,", " -a photo of a man, smiling", " -a photo of a man, frowning"]
    assert cmd[j + 1] == " -a photo of a dog"
    args = setup_parser().parse_args([*ARGS, "--slider_prompts", *cmd[i + 1:i + 4], "--slider_anchors", cmd[j + 1]])
    assert _prompt_texts(args.slider_prompts) == ["-a photo of a man,", "-a photo of a man, smiling",
                                                  "-a photo of a man, frowning"]          # the trainer's: as typed
    assert _anchor_lines(args.slider_anchors) == ["-a photo of a dog"]


def test_any_other_prompt_goes_as_it_did(tmp_path):
    cmd = _train_cmd(tmp_path, "krea2", FAMILY_SLIDER_ANCHORS="a photo of a dog")
    i, j = cmd.index("--slider_prompts"), cmd.index("--slider_anchors")
    assert cmd[i + 1:i + 4] == ["a photo of a man,", "a photo of a man, smiling", "a photo of a man, frowning"]
    assert cmd[j + 1] == "a photo of a dog"
    assert _prompt_texts(None) is None
