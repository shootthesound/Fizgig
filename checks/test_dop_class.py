"""#365: with "Leave the rest of the model alone" (DOP without Fizgig LoRA Isolation) the Subjects card shows "what they
are", and DOP puts it in the trigger word's place. Before, the box only showed under Isolation and DOP dropped the
trigger, so a caption that doesn't say what the subject is ("slcp walking") kept "walking" as the model made it,
not "a man walking". A caption that already says it after the trigger is never doubled ("slcp, a man walking" ->
"a man walking"), so an empty box, or one matching the captions, changes nothing."""
import sys

from fizgig.families import form, launch
from fizgig.families.multi_person import class_phrase, dop_caption
from fizgig.families.registry import get


def test_the_class_takes_the_triggers_place_once():
    man = [("slcp", "a man")]
    assert dop_caption("slcp walking in the rain", man) == "a man walking in the rain"
    assert dop_caption("a photo of slcp at a party", man) == "a photo of a man at a party"
    assert dop_caption("slcp, a man walking", man) == "a man walking"            # said already: not doubled
    assert dop_caption("slcp man walking", man) == "man walking"
    assert dop_caption("slcp, a young man walking", man) == "a young man walking"
    assert dop_caption("slcp walking with a woman", man) == "a man walking with a woman"
    assert dop_caption("slcp walking", [("slcp", "")]) == "walking"                # empty: left out, as before
    assert dop_caption("zwxemwoman smiling", [("zwxemwoman", "a woman")]) == "a woman smiling"   # Isolation's
    assert dop_caption("a man walking", man) is None
    assert [class_phrase(w) for w in ("man", "old man", "a dog", "the car", "", " woman ")] == \
        ["a man", "an old man", "a dog", "the car", "", "a woman"]


def _setup(tmp_path, **values):
    desc, models = get("klein"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    pics = tmp_path / "pics"
    pics.mkdir(exist_ok=True)
    from PIL import Image
    for i in range(2):
        Image.new("RGB", (512, 512)).save(pics / f"p{i}.png")
        (pics / f"p{i}.txt").write_text("slcp walking in the rain", encoding="utf-8")
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "DOP_FOLDER_TRIGGER": "slcp", **values}   # as the server reads it
    return desc, models, pics, v


def test_the_box_shows_with_dop_and_reaches_the_trainer(tmp_path):
    desc, models, pics, v = _setup(tmp_path, FAMILY_DOP=True, FAMILY_DOP_CLASS="man")
    assert "FAMILY_DOP_CLASS" not in form.state(desc, v, str(pics))["hidden"]
    iso = {"FAMILY_MULTI_PERSON": True, "FAMILY_MULTICONCEPT": True, "FAMILY_CONCEPT_COUNT": "2",
           "FAMILY_CONCEPT_DIR": str(pics)}                         # Isolation needs two subjects
    for off in ({"FAMILY_DOP": False}, iso):                         # DOP off, and Isolation without Correct
        st = form.state(desc, {**v, **off}, str(pics))
        assert "FAMILY_DOP_CLASS" in st["hidden"] and st["force"].get("FAMILY_MULTI_PERSON") is not False
    assert "FAMILY_DOP_CLASS" not in form.state(desc, {**v, **iso, "FAMILY_SUBJECT_FIX": True}, str(pics))["hidden"]
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(pics),
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    cmd = next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")
    assert cmd[cmd.index("--dop_trigger") + 1:cmd.index("--dop_class") + 2] == ["slcp", "--dop_class", "a man"]
