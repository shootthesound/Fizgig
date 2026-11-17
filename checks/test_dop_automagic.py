"""#364: Automagic (v3 or v4) can't train with Leave the rest of the model alone or Fizgig LoRA Isolation. DOP's pass
on every second step makes each weight's update flip direction step to step, which Automagic reads as overshooting:
on a pod, H3's Fast preset with two-subject Isolation fell to 5e-6 after epoch 1 and to its 1e-8 floor from epoch 2,
and nothing was learned, while the same preset on one subject without DOP held about 4e-4 (v4 fell the same way). Such
a run now trains with full-precision AdamW at 2e-4 (H3's Slider recipe), and Start and the log say so. Without DOP,
Automagic is left as picked."""
import sys

from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get
from fizgig.families.train import setup_parser


def _plan(tmp_path, fam="minimax", iso=True, dop=False, **values):
    desc, models = get(fam), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    dirs = {}
    for name, trig in (("a", "dlor"), ("b", "zwxem")):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        for i in range(2):
            Image.new("RGB", (512, 512)).save(d / f"{name}{i}.png")
            (d / f"{name}{i}.txt").write_text(f"{trig}, a woman smiling", encoding="utf-8")
        dirs[name] = str(d)
    v = {**form.DEFAULTS, **desc.presets[0][1], "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "DOP_FOLDER_TRIGGER": "dlor", **values}
    if iso:
        v.update(FAMILY_DOP=True, FAMILY_MULTI_PERSON=True, FAMILY_MULTICONCEPT=True, FAMILY_CONCEPT_COUNT="2",
                 FAMILY_CONCEPT_DIR=dirs["b"], DOP_FOLDER_TRIGGER2="zwxem")
    elif dop:
        v.update(FAMILY_DOP=True)
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=dirs["a"],
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    p = launch.plan(desc, inp)
    assert not p.problems, p.problems
    cmd = next(s.cmd for s in p.stages if s.name == "Training")
    return desc, inp, p, setup_parser().parse_args(cmd[2:])


def test_isolation_or_dop_with_automagic_trains_with_adamw(tmp_path):
    for fam, opt, kw in (("minimax", "automagic3", {"iso": True}), ("minimax", "automagic4", {"iso": True}),
                         ("minimax", "automagic3", {"iso": False, "dop": True}), ("krea2", "automagic3", {"iso": True})):
        desc, inp, p, a = _plan(tmp_path, fam=fam, OPTIMIZER_TYPE=opt, **kw)      # any model, not only H3
        assert (a.optimizer_type, a.learning_rate) == ("adamw", 2e-4)
        assert any("AdamW at 2e-4 instead of Automagic" in c for c in p.console)
        assert any("instead of Automagic" in w for w in launch.start_checks(desc, inp, ""))


def test_automagic_without_dop_is_left_as_picked(tmp_path):
    desc, inp, p, a = _plan(tmp_path, iso=False)
    assert (a.optimizer_type, a.learning_rate) == ("automagic3", 1e-6)
    assert not any("instead of Automagic" in c for c in p.console + launch.start_checks(desc, inp, ""))
    desc, inp, p, a = _plan(tmp_path, OPTIMIZER_TYPE="adamw8bit", LEARNING_RATE="1e-4")      # not Automagic: as picked
    assert (a.optimizer_type, a.learning_rate) == ("adamw8bit", 1e-4)
