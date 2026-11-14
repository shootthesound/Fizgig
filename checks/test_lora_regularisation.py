"""A LoRA's regularisation dataset (the Subjects card's last fold, #332): a folder of ordinary pictures of the same kind
of thing, trained beside the subject at its own LR multiplier and in place of DOP / Fizgig LoRA Isolation. The fold,
the greying, the launch giving way, Start's problems and warnings, the weight, the per-epoch share, and the
fine-tune's own regularisation left as it was."""
import os

import pytest
import torch
from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family

STILL = "anima"                  # the reporter's model: a still-picture family with the ability


def _pic(path, caption="a photo"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)
    if caption is not None:
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(caption)


@pytest.fixture
def dirs(tmp_path):
    train, reg = tmp_path / "train", tmp_path / "reg"
    for i in range(30):
        _pic(str(train / f"t{i:02d}.png"), "zwxem, a woman")
    os.makedirs(reg)
    for i in range(500):                        # an empty file is enough to count; captions are what Start reads
        open(reg / f"r{i:03d}.png", "wb").close()
        (reg / f"r{i:03d}.txt").write_text("a woman in a park", encoding="utf-8")
    return train, reg, tmp_path


def _values(train, reg, **kw):
    return {**form.DEFAULTS, "LORA_NAME": "mine", "FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(reg),
            "CAPTION_TRIGGER_WORD": "zwxem", **kw}


def _inputs(desc, values, train, tmp):
    st = form.inputs(desc, values, models={}, python="python", repo_dir=".", image_folder=str(train),
                     cache_root=str(tmp / "cache"), samples_dir=str(tmp / "out" / "sample"))
    st.update(DATASET_CONFIG=str(tmp / "ds.toml"), LORA_OUTPUT_DIR=str(tmp / "out"))
    return st


class _Plan:
    files, console, dirs = [], [], []


def _cmd(desc, st):
    return launch.train_command(desc, st, _Plan())


# 1. the fold: the card's last rows on a still-picture family; none on a clip or song family, and hidden for a slider
def test_the_fold_is_the_subjects_cards_last_part():
    for key in (STILL, "klein", "minimax_image"):
        rows = next(s for s in form.form(get_family(key))["sections"] if s["id"] == "subjects")["rows"]
        assert [r["key"] for r in rows[-4:]] == list(form.REG_KEYS)
        assert all(r.get("fold") == "lorareg" for r in rows[-4:]) and rows[-4]["fold_on"] == "FAMILY_REG_ON"
    for key in ("minimax", "acestep15xl", "vimoo"):
        desc = get_family(key)
        rows = [r["key"] for s in form.form(desc)["sections"] for r in s["rows"]]
        assert not set(form.REG_KEYS) & set(rows)
        assert set(form.REG_KEYS) <= set(form.state(desc, dict(form.DEFAULTS))["hidden"])
    desc = get_family(STILL)
    slider = dict(form.kinds(desc)).get("slider")
    if slider:
        assert set(form.REG_KEYS) <= set(form.state(desc, {**form.DEFAULTS, "FAMILY_KIND": slider})["hidden"])


# 2. greying: the rest of the card while it is on (its values kept), the fold's boxes while it is off
def test_on_greys_the_rest_of_the_card_and_keeps_its_values(dirs):
    train, reg, _tmp = dirs
    desc = get_family(STILL)
    before = _values(train, reg, FAMILY_SUBJECTS="2 subjects", FAMILY_MULTICONCEPT=True, FAMILY_DOP=True,
                     FAMILY_MULTI_PERSON=True, KREA2_LOSS_WATCH=True)
    on = form.state(desc, before, str(train))
    assert set(form._subject_keys(desc)) <= set(on["disabled"]) and "FAMILY_PEOPLE_MODE" in on["disabled"]
    assert on["notes"]["FAMILY_SUBJECTS"] == form.REG_GREYED
    for k in ("FAMILY_MULTICONCEPT", "FAMILY_DOP", "FAMILY_MULTI_PERSON", "FAMILY_SUBJECTS", "KREA2_LOSS_WATCH"):
        assert k not in on["force"] or on["force"][k] == before[k]     # nothing of the card is overwritten
    off = form.state(desc, {**before, "FAMILY_REG_ON": False}, str(train))
    assert {"FAMILY_REG_DIR", "FAMILY_REG_MULT", "FAMILY_REG_USE"} <= set(off["disabled"])
    assert not set(form._subject_keys(desc)) & {"FAMILY_REG_ON"}


# 3. the launch gives way: one subject, no DOP, no Isolation, and the folder as its own block
def test_the_rest_of_the_card_gives_way_at_launch(dirs):
    train, reg, tmp = dirs
    desc = get_family(STILL)
    st = _inputs(desc, _values(train, reg, FAMILY_SUBJECTS="2 subjects", FAMILY_MULTICONCEPT=True,
                               FAMILY_CONCEPT_DIR=str(tmp), FAMILY_DOP=True, FAMILY_MULTI_PERSON=True), train, tmp)
    assert not launch.dop_on(desc, st) and not launch.multi_person_on(desc, st)
    assert st["multiconcept"] is False and st["extra_folders"] == []
    toml = launch.dataset_toml(desc, st)
    assert toml.count("[[datasets]]") == 2
    second = toml.split("[[datasets]]")[2]
    assert 'item_prefix = "reg:"' in second and "is_reg = true" in second and str(reg).replace("\\", "/") in second


# 4. the command line: the multiplier (1 by default) and the per-epoch share, only while it is on
def test_the_command_line(dirs):
    train, reg, tmp = dirs
    desc = get_family(STILL)
    c = _cmd(desc, _inputs(desc, _values(train, reg), train, tmp))
    assert c[c.index("--reg_lr_multiplier") + 1] == "1" and c[c.index("--reg_per_epoch") + 1] == "30"
    c = _cmd(desc, _inputs(desc, _values(train, reg, FAMILY_REG_MULT="0.2"), train, tmp))
    assert c[c.index("--reg_lr_multiplier") + 1] == "0.2"
    off = _cmd(desc, _inputs(desc, _values(train, reg, FAMILY_REG_ON=False), train, tmp))
    assert "--reg_lr_multiplier" not in off and "--reg_per_epoch" not in off


# 5. Start: what it refuses and what it warns about
def test_start_problems_and_warnings(dirs):
    train, reg, tmp = dirs
    desc = get_family(STILL)
    probs = lambda **kw: launch.lora_reg_problems(desc, _inputs(desc, _values(train, reg, **kw), train, tmp))
    assert probs() == []
    assert "does not exist" in probs(FAMILY_REG_DIR=str(tmp / "nowhere"))[0]
    assert "training folder itself" in probs(FAMILY_REG_DIR=str(train))[0]
    assert "pick its folder" in probs(FAMILY_REG_DIR="")[0]
    for i in range(12):
        os.remove(reg / f"r{i:03d}.txt")
    assert "12 of the 500 regularisation pictures have no caption" in probs()[0]
    (reg / "r100.txt").write_text("zwxem, a woman", encoding="utf-8")
    (reg / "r101.txt").write_text("a zwxem woman", encoding="utf-8")
    w = launch.lora_reg_warnings(desc, _inputs(desc, _values(train, reg), train, tmp))
    assert len(w) == 1 and w[0].startswith("2 regularisation captions use your trigger word 'zwxem'")
    assert launch.lora_reg_problems(desc, _inputs(desc, _values(train, reg, FAMILY_REG_ON=False), train, tmp)) == []


# 6. the weight: a regularisation picture at its multiplier, a training picture at 1
def test_a_regularisation_picture_weighs_at_its_multiplier():
    from fizgig.families.loss_watch import Watch
    w = Watch.__new__(Watch)
    w.watch = None
    per = torch.tensor([0.4, 0.8])
    scaled, _rec, kept = w.weigh({"per": per, "per_raw": [0.4, 0.8]}, ["t1", "reg:r1"], {"reg:r1"}, 0.25)
    assert kept == [0, 1] and abs(float(scaled) - (0.4 + 0.25 * 0.8) / 2) < 1e-6


# 6b. the loss watch leaves regularisation pictures alone: never recorded, excluded or throttled
def test_the_loss_watch_leaves_regularisation_pictures_alone():
    from fizgig.families.loss_watch import Watch

    class Fake:
        def __init__(self):
            self.seen = []

        def is_excluded(self, keys):
            return True

        def multiplier(self, keys):
            return 0.5

        def observe(self, **kw):
            self.seen.append(kw["item_keys"])
    w = Watch.__new__(Watch)
    w.watch, w.reg = Fake(), {"reg:r1"}
    assert not w.excluded({"item_keys": ["reg:r1"]}) and w.multiplier({"item_keys": ["reg:r1"]}) == 1.0
    w.observe(1, 1, {"item_keys": ["reg:r1"]}, 0.5, 0.3)
    w.observe_each(1, 2, ["t1", "reg:r1"], [0.5, 0.5], [0.3, 0.3], [0, 1])
    assert w.watch.seen == [["t1"]]


# 7. the per-epoch share: the figure, the command line and the trainer's batches agree
def test_the_per_epoch_share(dirs):
    train, reg, tmp = dirs
    desc = get_family(STILL)
    for use, want in ((form.REG_MATCH, 30), ("10%", 50), ("25%", 125), ("100%", 500)):
        st = _inputs(desc, _values(train, reg, FAMILY_REG_USE=use), train, tmp)
        assert launch.lora_reg_budget(desc, st) == (want, 500, 30)
        c = _cmd(desc, st)
        assert c[c.index("--reg_per_epoch") + 1] == str(want)
    note = form.state(desc, _values(train, reg), str(train))["notes"]["FAMILY_REG_USE"]
    assert note.startswith("30 of 500 pictures, a different 30 each epoch") and "from 30 pictures to 60" in note
    few = tmp / "few"
    for i in range(10):
        _pic(str(few / f"f{i}.png"), "a woman")
    assert launch.lora_reg_budget(desc, _inputs(desc, _values(train, few), train, tmp)) == (10, 10, 30)


def test_the_trainer_takes_a_different_share_each_epoch():
    import random
    from fizgig.dataset.image_dataset import BucketBatchManager

    class It:
        def __init__(self, k):
            self.item_key = k
    for bs in (1, 4):
        bm = BucketBatchManager({(512, 512): [It(f"a{i}") for i in range(300)],
                                 (448, 576): [It(f"b{i}") for i in range(200)]}, bs)
        assert bm.limit(30) == 500
        epochs = []
        for e in range(3):
            random.seed(e)
            bm.shuffle()
            keys = []
            for reso, i in bm.bucket_batch_indices:
                start = i * bm.bucket_batch(reso)
                keys += [it.item_key for it in bm.buckets[reso][start:min(start + bs, bm.quota[reso])]]
            assert len(keys) == 30 == len(set(keys))
            epochs.append(set(keys))
        assert len(bm) == (30 if bs == 1 else 8) and epochs[0] != epochs[1]
    small = BucketBatchManager({(512, 512): [It(f"x{i}") for i in range(10)]}, 1)
    assert small.limit(30) == 10 and small.quota is None and len(small) == 10


# 8. the fine-tune's own regularisation is as it was: its block, its multiplier, no per-epoch share
def test_the_finetunes_regularisation_is_unchanged(dirs):
    train, reg, tmp = dirs
    desc = get_family("krea2")
    v = {**form.DEFAULTS, "LORA_NAME": "ft", "FAMILY_KIND": dict(form.kinds(desc))["finetune"],
         "FAMILY_FT_REG_DIR": str(reg), "FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(tmp)}
    st = _inputs(desc, v, train, tmp)
    assert not launch.lora_reg_on(desc, st) and launch.reg_folder(desc, st) == str(reg)
    assert 'item_prefix = "reg:"' in launch.dataset_toml(desc, st)
    c = _cmd(desc, st)
    assert c[c.index("--reg_lr_multiplier") + 1] == "0.2" and "--reg_per_epoch" not in c
    assert set(form.REG_KEYS) <= set(form.state(desc, v, str(train))["hidden"])


# Move to...: the folder goes with the run only while the tick is on
def test_a_move_takes_the_folder_only_while_it_is_on(dirs):
    from fizgig.web import helpers
    train, reg, _tmp = dirs
    assert "FAMILY_REG_DIR" in helpers._move_extras({"FAMILY_REG_ON": True, "FAMILY_REG_DIR": str(reg)})[0]
    assert not helpers._move_extras({"FAMILY_REG_ON": False, "FAMILY_REG_DIR": str(reg)})[0]
    assert not helpers._move_names({"FAMILY_REG_ON": False, "FAMILY_REG_DIR": "/gone"}, "FAMILY_REG_DIR")


# a regularisation folder inside the training folder is never one of its sub-folders
def test_a_folder_inside_the_training_folder_is_never_a_subfolder(tmp_path):
    train = tmp_path / "train"
    _pic(str(train / "a.png"))
    _pic(str(train / "reg" / "r.png"), "a woman")
    _pic(str(train / "more" / "m.png"))
    desc = get_family(STILL)
    v = _values(train, train / "reg", TRAIN_SUBFOLDERS=True)
    st = _inputs(desc, v, train, tmp_path)
    toml = launch.dataset_toml(desc, st)
    assert 'item_prefix = "reg/"' not in toml and 'item_prefix = "more/"' in toml and 'item_prefix = "reg:"' in toml
    assert form.subfolders_note(desc, v, str(train))[0].startswith("Training on 2 folders")
