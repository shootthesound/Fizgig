"""A prompt slider's boxes of lines with the by-epoch switch on (app.js promptLines), as a photographer meets
them: ▲ ▼ big enough to hit; and the note under an end whose lines are on together (form._slider_plan)."""
import os
import re
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family

CSS = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "style.css")


def _inputs(tmp_path, desc, **values):
    """The form's values and Start's inputs."""
    models = {}
    for mf in desc.model_files:
        f = tmp_path / f"{desc.key}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "8",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
         "FAMILY_SLIDER_NEG": "frowning @1-4\ncrying @5-", "FAMILY_SLIDER_GUIDANCE": "3",
         "FAMILY_SLIDER_BY_EPOCH": True, **values}
    return v, form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))


def test_the_move_buttons_are_big_enough_to_hit():
    rule = re.search(r"\.prompt-move button \{(.*?)\}", open(CSS, encoding="utf-8").read(), re.S).group(1)
    size = dict(re.findall(r"(?<![\w-])((?:min-)?(?:width|height)):\s*(\d+)px", rule))
    assert max(int(size.get("width", 0)), int(size.get("min-width", 0))) >= 20, rule
    assert max(int(size.get("height", 0)), int(size.get("min-height", 0))) >= 20, rule


def test_an_ends_lines_on_together_say_what_they_train_as(tmp_path):
    # as the picture's note: the lines on together, the first time, and the one prompt they make - and when
    desc = get_family("krea2")
    v, _inp = _inputs(tmp_path, desc, MAX_TRAIN_EPOCHS="10", FAMILY_SLIDER_POS="smiling\ngrinning",
                      FAMILY_SLIDER_NEG="frowning @1-4\ncrying @3-\nsulking @9-9")
    notes = form.state(desc, v, "")["notes"]
    assert notes["FAMILY_SLIDER_POS"] == ('"smiling" and "grinning" are on together at epochs 1–10 and train as one '
                                          'prompt: "smiling, grinning". For a change, give them epochs of their own.')
    assert notes["FAMILY_SLIDER_NEG"] == ('"frowning" and "crying" are on together at epochs 3–4 and train as one '
                                          'prompt: "frowning, crying". For a change, give them epochs of their own.')
    one = form.state(desc, {**v, "FAMILY_SLIDER_NEG": "frowning @1-5\ncrying @5-"}, "")["notes"]
    assert one["FAMILY_SLIDER_NEG"].startswith('"frowning" and "crying" are on together at epoch 5 and train as one ')
    alone = form.state(desc, {**v, "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-"}, "")["notes"]
    assert "FAMILY_SLIDER_POS" not in alone                          # none on together: no note
    off = form.state(desc, {**v, "FAMILY_SLIDER_BY_EPOCH": False}, "")["notes"]
    assert off["FAMILY_SLIDER_POS"].startswith(f'Holds 2 lines from "{launch.BY_EPOCH}"')     # the tick off: as before
