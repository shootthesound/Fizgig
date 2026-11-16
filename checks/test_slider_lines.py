"""A prompt slider's lines with their own epochs (picture models, Change the prompts during the run (advanced)
ticked): every box - what
the picture is, each end, leave these alone - holds lines, each on for its own epochs ("smiling @5-8", none = the whole
run), and at each epoch the lines on then make the prompts, a box's lines joined into one. One line in each box
launches exactly as it always did, and a model that doesn't practise on pictures (ACE-Step's songs) keeps its boxes as
typed. Unticked: test_slider_by_epoch.py."""
import json
import sys

import pytest

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families
from fizgig.families.train import (_anchor_lines, _first_lines, _line_boxes, _read_slider_live, _request_boxes,
                                   _state_epoch, setup_parser)

BOXES = {"base": "a photo of a woman, @-2\na photo of a woman in a red coat, @3-",
         "pos": "smiling @1-5\nlaughing @3-",
         "neg": "frowning",
         "anchors": "a photo of a man @-4\na photo of a dog @5-"}


def _rows(desc):
    """The form's rows as the page draws them with Change the prompts during the run (advanced) ticked (app.js
    rowLook)."""
    return {r["key"]: {**r, **r.get("by_epoch", {})} for s in form.form(desc)["sections"] for r in s["rows"]}


def test_a_line_and_its_epochs():
    assert launch.slider_line_parts("smiling @5-8") == ("smiling", 5, 8, None)
    assert launch.slider_line_parts("smiling @5-") == ("smiling", 5, None, None)
    assert launch.slider_line_parts("smiling @-8") == ("smiling", None, 8, None)
    assert launch.slider_line_parts("smiling @5") == ("smiling", 5, None, None)
    assert launch.slider_line_parts("smiling @1-30/3") == ("smiling", 1, 30, 3)        # every 3rd epoch from 1
    assert launch.slider_line_parts(" a photo of a woman, ") == ("a photo of a woman,", None, None, None)
    assert launch.slider_line_parts("mail@home") == ("mail@home", None, None, None)   # no epochs: text
    assert launch.slider_lines_on("smiling @1-30/3", 4) == ["smiling"]
    assert launch.slider_lines_on("smiling @1-30/3", 5) == []


def test_the_lines_on_each_epoch_join_into_one():
    assert launch.slider_at(BOXES, 1) == ("a photo of a woman,", "smiling", "frowning", ["a photo of a man"])
    assert launch.slider_at(BOXES, 3) == ("a photo of a woman in a red coat,", "smiling, laughing", "frowning",
                                          ["a photo of a man"])
    assert launch.slider_at(BOXES, 6) == ("a photo of a woman in a red coat,", "laughing", "frowning",
                                          ["a photo of a dog"])
    assert launch.slider_join(["smiling", "laughing"]) == "smiling, laughing"
    assert launch.slider_join(["a photo of a woman,", "in a red coat"]) == "a photo of a woman, in a red coat"
    plan = launch.slider_plan(BOXES, 8)
    assert [(a, b) for a, b, *_ in plan] == [(1, 2), (3, 4), (5, 5), (6, 8)]
    assert launch.slider_gap("smiling @1-5", 8) == 6 and launch.slider_gap("smiling", 8) is None
    # each line's epochs as the boxes and the plan colour them (app.js epochColour keys on the same text)
    assert launch.slider_lines_on(BOXES["pos"], 3, keys=True) == [("smiling", "1-5"), ("laughing", "3-")]
    assert launch.slider_range_key(None, None) == "" and launch.slider_range_key(0, 8) == "1-8"


def _plan(tmp_path, fam, **values):
    desc = get_family(fam)
    models = {}
    for mf in desc.model_files:
        f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "8",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning",
         "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BY_EPOCH": True, **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    return launch.plan(desc, inp), desc, v


def _train(p):
    assert not p.problems, p.problems
    return next(s.cmd for s in p.stages if s.name == "Training")


def test_one_line_each_launches_as_it_always_did(tmp_path):
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_ANCHORS="a photo of a dog\na photo of a cat")
    cmd = _train(p)
    i = cmd.index("--slider_prompts")
    assert cmd[i:i + 6] == ["--slider_prompts", "a photo of a man,", "a photo of a man, smiling",
                            "a photo of a man, frowning", "--slider_guidance", "3"]
    assert cmd[cmd.index("--slider_anchors"):cmd.index("--slider_anchors") + 5] == [
        "--slider_anchors", "a photo of a dog", "a photo of a cat", "--slider_anchor_weight", "1"]
    assert "--slider_line" not in cmd
    plan = form.state(desc, v, "")["slider_plan"]                 # one section, the whole run
    assert [(s["from"], s["to"]) for s in plan] == [(1, 8)]
    assert plan[0]["lines"]["pos"] == [{"text": "smiling", "key": ""}]


def test_lines_with_epochs_launch_as_lines(tmp_path):
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE=BOXES["base"], FAMILY_SLIDER_POS=BOXES["pos"],
                       FAMILY_SLIDER_ANCHORS=BOXES["anchors"])
    cmd = _train(p)
    lines = [(cmd[k + 1], cmd[k + 2]) for k, t in enumerate(cmd) if t == "--slider_line"]
    assert lines == [("base", "a photo of a woman, @-2"), ("base", "a photo of a woman in a red coat, @3-"),
                     ("pos", "smiling @1-5"),
                     ("pos", "laughing @3-"), ("neg", "frowning"), ("anchor", "a photo of a man @-4"),
                     ("anchor", "a photo of a dog @5-")]
    assert "--slider_prompts" not in cmd and "--slider_anchors" not in cmd
    assert next(t for f, t in p.files if f.endswith("_slider_prompt.txt")) == "a photo of a woman,\n"   # previews
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"]) for s in plan] == [(1, 2), (3, 4), (5, 5), (6, 8)]
    assert plan[1]["lines"]["pos"] == [{"text": "smiling", "key": "1-5"}, {"text": "laughing", "key": "3-"}]
    assert plan[1]["lines"]["base"] == [{"text": "a photo of a woman in a red coat,", "key": "3-"}]
    assert [s["new_pictures"] for s in plan] == [False, True, False, False]     # the picture changes at epoch 3
    assert [s["changed"] for s in plan] == [[], ["pos"], ["anchors"], ["pos"]]  # and what else, section by section


def test_an_epoch_with_no_line_stops_start(tmp_path):
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @1-5")
    assert any('"The +1 end adds" is on at epoch 6' in x for x in p.problems)
    assert form.state(desc, v, "")["notes"]["FAMILY_SLIDER_POS"].startswith("No line is on at epochs 6–8: ")


def test_a_line_every_other_epoch_says_every_epoch_it_leaves_without_one(tmp_path):
    # the picture's one line on epochs 1, 3, 5 ... and an empty second line: the note names the epochs left over
    p, desc, v = _plan(tmp_path, "krea2", MAX_TRAIN_EPOCHS="20", FAMILY_SLIDER_BASE="a photo of a woman, @1-20/2\n")
    assert any('no line of "What the picture is" is on at epoch 2' in x for x in p.problems)     # Start won't run
    assert form.state(desc, v, "")["notes"]["FAMILY_SLIDER_BASE"] == (
        "No line is on at epochs 2, 4, 6, 8, 10 and 5 more: every epoch needs one here. Let a line cover them (a line "
        "with no epochs is on for the whole run), or give them a line of their own - an empty line isn't on. Start "
        "won't run until each has one.")
    assert launch.slider_gaps("a @1-4/2\nb @2-2", 4) == [4]


def test_a_line_outside_the_run_says_so(tmp_path):
    _p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @1-8\nlaughing @9-12",
                        FAMILY_SLIDER_ANCHORS="a photo of a man @3-10")
    notes = form.state(desc, v, "")["notes"]                  # each line named by its words, not its place
    assert notes["FAMILY_SLIDER_POS"] == '"laughing" starts at epoch 9, after the run\'s last (8): it never trains.'
    assert notes["FAMILY_SLIDER_ANCHORS"] == ('"a photo of a man" runs to epoch 10, past the run\'s last (8): it stops '
                                              'at 8.')


def test_picture_lines_on_together_join_into_one_description(tmp_path):
    # as the ends' lines: the picture's lines on at the same epoch join, with a comma, and Start runs
    assert launch.slider_overlap("a man, @1-5\na woman, @5-8", 8) == (5, ["a man,", "a woman,"])
    assert launch.slider_overlap("a man, @1-4\na woman, @5-8", 8) is None
    assert launch.slider_overlap("a man,\na woman,", 8) == (1, ["a man,", "a woman,"])   # no epochs: both all run
    assert launch.slider_numbers([1, 2, 3]) == "1, 2 and 3"
    boxes = {"base": "a close-up photo of a woman,\nwearing a red coat, @5-", "pos": "smiling", "neg": "frowning",
             "anchors": ""}
    assert launch.slider_at(boxes, 4)[:3] == ("a close-up photo of a woman,", "smiling", "frowning")
    assert launch.slider_at(boxes, 5)[0] == "a close-up photo of a woman, wearing a red coat,"
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="a man, @0-2\na woman, @2-8")
    cmd = _train(p)                                            # no problem: Start runs, the lines go to the trainer
    assert [cmd[k + 2] for k, t in enumerate(cmd) if t == "--slider_line" and cmd[k + 1] == "base"] == [
        "a man, @0-2", "a woman, @2-8"]
    a = setup_parser().parse_args(["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                   "--slider_line", "base", "a man, @0-2", "--slider_line", "base", "a woman, @2-8"])
    assert a.slider_line == [["base", "a man, @0-2"], ["base", "a woman, @2-8"]]
    st = form.state(desc, v, "")
    assert st["notes"]["FAMILY_SLIDER_BASE"] == ('"a man," and "a woman," are on together at epoch 2 and train as one '
                                                 'description: "a man, a woman,". For different pictures, give them '
                                                 'epochs of their own.')
    assert [(s["from"], s["to"]) for s in st["slider_plan"]] == [(1, 1), (2, 2), (3, 8)]
    assert [x["text"] for x in st["slider_plan"][1]["lines"]["base"]] == ["a man,", "a woman,"]
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @1-5\nlaughing @5-8")   # the ends' too
    assert not p.problems


def test_every_picture_slider_has_lines_with_epochs_and_songs_keep_theirs():
    keys = ("FAMILY_SLIDER_BASE", "FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG", "FAMILY_SLIDER_ANCHORS")
    for d in training_families():
        if d.slider_training and "prompts" in form.slider_sources(d):
            rows = _rows(d)
            pictures = "photo" in d.media
            for key in keys:
                assert bool(rows[key].get("lines") and rows[key].get("epochs")) == pictures, (d.key, key)
            assert ("_slider_plan" in rows) == pictures == ("FAMILY_SLIDER_BY_EPOCH" in rows), d.key
    songs = next(d for d in training_families() if d.slider_training and "photo" not in d.media)
    assert not launch.slider_scheduled(songs, {**BOXES}, True)
    assert launch.slider_at(BOXES, 9, pictures=False)[:2] == (BOXES["base"], BOXES["pos"])     # as typed


def test_the_trainer_takes_lines_and_starts_where_it_paused(tmp_path):
    a = setup_parser().parse_args(["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                   "--slider_line", "pos", "smiling @1-5", "--slider_line", "base", "a woman,"])
    assert a.slider_line == [["pos", "smiling @1-5"], ["base", "a woman,"]]
    assert _state_epoch(str(tmp_path / "n-000012-state")) == 12 and _state_epoch(None) == 0
    picked = tmp_path / "my pause"     # a hand-picked Resume folder: its epoch is the state's, as the resume reads it
    picked.mkdir()
    (picked / "training_state.json").write_text(json.dumps({"epoch": 7, "global_step": 70}), encoding="utf-8")
    assert _state_epoch(str(picked)) == 7
    assert _first_lines(BOXES) == ["a photo of a woman,", "a photo of a man"]      # the previews, all run
    (tmp_path / ".slider_prompts.json").write_text(json.dumps(
        {"base": BOXES["base"], "pos": BOXES["pos"], "neg": "frowning", "guidance": 2,
         "anchors": BOXES["anchors"]}), encoding="utf-8")
    got = _read_slider_live(str(tmp_path), by_epoch=True)                     # the Run card's boxes, lines and all
    assert got["pos"] == BOXES["pos"] and got["anchors"] == BOXES["anchors"].splitlines()


def test_a_line_whose_epochs_mean_the_whole_run_trains_on_its_words(tmp_path):
    # "every 1" alone was saved as "@-/1": that is the whole run, and it must not stay in the prompt
    boxes = {"base": "a photo of a man,", "pos": "smiling @-/1", "neg": "frowning", "anchors": ""}
    assert launch.slider_scheduled(get_family("krea2"), boxes, True)
    assert launch.slider_at(boxes, 3)[1] == "smiling"
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @-/1")
    assert "--slider_prompts" not in _train(p)                 # the lines go as lines: the trainer takes the words


def test_the_notes_name_lines_by_their_words():
    # with the by-epoch switch on a line's place in its box changes as the lines take their places by epoch: a note
    # names a line by its words, quoted and cut short when long; a blank row or a line with no words (never on) isn't
    # named
    assert launch.slider_overlap("a man, @1-5\n\na woman, @5-8", 8) == (5, ["a man,", "a woman,"])
    assert launch.slider_epoch_notes("smiling @1-8\n\nlaughing @9-12", 8) == [
        '"laughing" starts at epoch 9, after the run\'s last (8): it never trains']
    assert launch.slider_epoch_notes(" @9-12\nsmiling @5-4\nlaughing @3-10", 8) == [
        'the epochs of "smiling" run backwards (5-4): it never trains',
        '"laughing" runs to epoch 10, past the run\'s last (8): it stops at 8']
    assert launch.slider_line_name("a close-up photo of a young woman with short dark hair,") == (
        '"a close-up photo of a young woman..."')
    assert launch.slider_line_name("a man,") == '"a man,"'


def test_the_run_card_shows_the_lines_the_trainer_took(tmp_path):
    from fizgig.web import server
    v = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
         "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
         "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True}
    run = {"family": "krea2", "values": v, "output_dir": str(tmp_path), "progress": {"epoch": 6}}
    st = server._slider_live_status(run)
    assert not st["waiting"] and st["now"]["pos"] == "a man, laughing" and st["now"]["next"] is None
    # Apply at the next epoch: the run's values take the change at once, the trainer only at the end of the epoch
    took = {"base": "a man,", "pos": "smiling @1-4\nlaughing @5-", "neg": "frowning", "guidance": 3.0, "anchors": []}
    asked = {**took, "pos": "smiling @1-4\ngrinning @5-"}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps(asked), encoding="utf-8")
    run["values"] = {**v, "FAMILY_SLIDER_POS": asked["pos"]}
    st = server._slider_live_status(run)
    assert st["waiting"] and st["now"] is None      # none taken yet, and no record of what it started on: not shown
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**took, "epoch": 3}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["waiting"] and st["now"]["pos"] == "a man, laughing"     # the last change it took, not the new one
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**asked, "epoch": 6}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert not st["waiting"] and st["now"]["pos"] == "a man, grinning"

def test_practice_pictures_are_kept_only_for_the_very_same_render():
    from fizgig.families.train import _BankCache, _bank_key
    # what the description is - its text and the negative's - not their encodings, which another batch moves a little
    a, neg, speed, on = "a photo of a woman,", "blurry", ("speed", 8, 2.0, None, ()), (0.6, True, (), ())
    k = _bank_key(a, neg, speed, on, 768, 16, 42)
    assert k == _bank_key(str(a), str(neg), speed, on, 768, 16, 42)
    for changed in (_bank_key("a photo of a man,", neg, speed, on, 768, 16, 42),        # the description
                    _bank_key(a, None, speed, on, 768, 16, 42),                         # the negative
                    _bank_key(a, "ugly", speed, on, 768, 16, 42),
                    _bank_key(a, neg, ("plain", 30, 4.0), None, 768, 16, 42),           # no speed LoRA
                    _bank_key(a, neg, speed, (0.8, True, (), ()), 768, 16, 42),         # its strength
                    _bank_key(a, neg, speed, on, 512, 16, 42),                          # the size
                    _bank_key(a, neg, speed, on, 768, 8, 42),                           # how many
                    _bank_key(a, neg, speed, on, 768, 16, 7)):                          # the seed
        assert changed != k
    cache = _BankCache(keep=2)
    cache.put("a", ["A"])
    cache.put("b", ["B"])
    assert cache.get("a") == ["A"]                  # used: the most recent now
    cache.put("c", ["C"])                           # over 2: the least recently used (b) goes
    assert cache.get("b") is None and cache.get("a") == ["A"] and cache.get("c") == ["C"]
    assert cache.get("nope") is None


def test_the_plan_says_where_practice_pictures_come_again(tmp_path):
    _p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="A, @1-8/2\nB, @2-8/2")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["new_pictures"], s["again"]) for s in plan][:4] == [
        (1, False, None), (2, True, None), (3, True, 1), (4, True, 2)]

def test_the_keep_tick_is_on_by_default_and_unticked_renders_every_time(tmp_path):
    from fizgig.families.train import _BankCache
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="A, @1-8/2\nB, @2-8/2")
    cmd = _train(p)                                                   # on: the trainer keeps its banks
    assert "--slider_by_epoch" in cmd and cmd[cmd.index("--slider_bank_keep") + 1] == str(launch.SLIDER_BANKS_KEPT)
    assert [s["again"] for s in form.state(desc, v, "")["slider_plan"]][2:4] == [1, 2]
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="A, @1-8/2\nB, @2-8/2", FAMILY_SLIDER_KEEP=False)
    cmd = _train(p)
    assert cmd[cmd.index("--slider_bank_keep") + 1] == "0"
    assert all(s["again"] is None for s in form.state(desc, v, "")["slider_plan"])
    assert setup_parser().parse_args(["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                      "--slider_bank_keep", "0"]).slider_bank_keep == 0
    never = _BankCache(keep=0)
    never.put("a", ["A"])
    assert never.get("a") is None                                     # keeps nothing: every bank renders anew
    rows = _rows(desc)
    assert rows["FAMILY_SLIDER_KEEP"]["kind"] == "check" and rows["FAMILY_SLIDER_KEEP"]["default"] is True
    pairs = form.state(desc, {**v, "FAMILY_SLIDER_SOURCE": "Photo pairs"}, "")
    assert "FAMILY_SLIDER_KEEP" in pairs["hidden"]


def test_the_caches_hold_at_most_their_bytes_the_newest_always():
    import torch
    from fizgig.families.train import SLIDER_CACHE_BYTES, _BankCache, _nbytes
    forty = lambda: [torch.zeros(10, dtype=torch.float32)]           # 40 bytes
    assert _nbytes([{"a": torch.zeros(2, 3, dtype=torch.bfloat16), "b": torch.zeros(4, dtype=torch.bool)}]) == 16
    assert _BankCache().max_bytes == SLIDER_CACHE_BYTES == 1 << 30
    cache = _BankCache(keep=10, max_bytes=100)
    for k in "abc":
        cache.put(k, forty())
    assert cache.get("a") is None and cache.get("b") and cache.get("c")     # 120 bytes: the least recently used goes
    cache.put("d", [torch.zeros(50, dtype=torch.float32)])           # 200 bytes alone: kept, the rest go
    assert [k for k in "bcd" if cache.get(k)] == ["d"] and set(cache.sizes) == {"d"}

def test_the_plan_reads_the_clean_end_share_as_the_driver_does():
    from fizgig.moss_image.driver import MOSSImageDriver
    moss = get_family("moss_image")
    key = next(o.key for o in moss.options if str(o.tokens).startswith("lownoise_pct="))
    for val in ("11:7:40:8:3", "75:6:40:10:3", "30:5:60", "75:6:40:10", "40", "60%", "30:7:60:8:3"):
        m = MOSSImageDriver.__new__(MOSSImageDriver)
        m.options = {"lownoise_pct": val}
        for e in range(1, 25):
            assert form.clean_share_at(moss, {key: val}, e) == m._share(e), (val, e)
    assert form.clean_share_at(moss, {key: ""}, 3) is None                 # empty: the model's own schedule
    assert form.clean_share_at(get_family("fibo2"), {}, 3) is None          # no Clean-end share box on its form


def test_the_plan_shows_how_each_run_of_epochs_trains(tmp_path):
    _p, desc, v = _plan(tmp_path, "moss_image", MAX_TRAIN_EPOCHS="20", MOSS_LOWNOISE_PCT="11:7:40:8:3")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"], s["detail"], s["half"]) for s in plan] == [
        (1, 7, "11", False), (8, 10, "40", False), (11, 11, "11", True), (12, 13, "40", False), (14, 14, "11", True),
        (15, 16, "40", False), (17, 17, "11", True), (18, 19, "40", False), (20, 20, "11", True)]
    assert not any(s["new_pictures"] for s in plan)               # the same prompts all run: one render
    _p, desc, v = _plan(tmp_path, "krea2")                         # no Clean-end share: one section, nothing on it
    assert [(s["from"], s["to"], s["detail"], s["half"]) for s in form.state(desc, v, "")["slider_plan"]] == [
        (1, 8, None, False)]


def test_the_push_strength_by_epoch():
    # one number all run; a Split, A for the first N epochs then B; a Slide, A until epoch H, then in a straight line
    # to B by epoch E, B after - as MOSS-Image's Clean-end share box, without its fifth part
    assert [launch.slider_push_at("3", e) for e in (1, 40)] == [3, 3] and launch.slider_push_at(" 2.5 ", 7) == 2.5
    assert [launch.slider_push_at("3:10:2", e) for e in (1, 10, 11, 30)] == [3, 3, 2, 2]
    assert [launch.slider_push_at("3:4:2:8", e) for e in (1, 4, 5, 6, 7, 8, 20)] == [3, 3, 2.75, 2.5, 2.25, 2, 2]
    assert launch.slider_push_at("2:10:3:10", 11) == 3              # a Slide that ends where it starts: a Split
    for bad in ("", None, "abc", "3:10", "3:10:2:20:3", "3::2", "nan", "3:10:inf"):
        assert launch.slider_push_at(bad, 1) is None, bad
    assert launch.slider_push_problems("3", 20) == launch.slider_push_problems("3:10:2:20", 20) == []
    assert launch.slider_push_problems("0", 20) == launch.slider_push_problems("", 20) == [
        "Slider push strength must be a number above 0"]               # one number: refused as it always was
    assert launch.slider_push_problems("3:10", 20) == [
        "Slider push strength must be a number above 0, a Split such as 3:10:2 (3 for the first 10 epochs, then 2) or "
        "a Slide such as 3:10:2:20 (3 until epoch 10, then sliding to 2 by epoch 20)"]
    assert launch.slider_push_problems("3:10:0", 20) == [
        "Slider push strength must be above 0 at every epoch - it is 0 at epoch 11"]
    assert launch.slider_push_problems("3:10:0", 10) == []             # 0 only after the run's last epoch
    assert launch.slider_push_problems("3:10:-1:20", 20) == [
        "Slider push strength must be above 0 at every epoch - it is -0.2 at epoch 18"]
    assert launch.slider_push_problems("3:10:2:5", 20) == [
        "Slider push strength holds until epoch 10 but slides by epoch 5, before that - give a later one"]
    assert launch.slider_push_text(3.0) == launch.slider_push_text(" 3 ") == "3"      # an older file's number too
    assert launch.slider_push_text(" 3:10:2 ") == "3:10:2" and launch.slider_push_text(None) == ""


def test_the_push_strength_goes_to_the_trainer_as_typed(tmp_path):
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_GUIDANCE=" 3:6:2 ")       # one line each
    cmd = _train(p)
    assert "--slider_prompts" in cmd and cmd[cmd.index("--slider_guidance") + 1] == "3:6:2"
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_GUIDANCE="3:4:2:6",
                         FAMILY_SLIDER_POS="smiling @1-4\nlaughing @5-")
    cmd = _train(p)                                                                   # lines with their own epochs
    assert "--slider_line" in cmd and cmd[cmd.index("--slider_guidance") + 1] == "3:4:2:6"
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_GUIDANCE="3:4:0")          # 0 from epoch 5: Start says so
    assert "Slider push strength must be above 0 at every epoch - it is 0 at epoch 5" in p.problems
    args = ["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n"]
    assert setup_parser().parse_args(args + ["--slider_guidance", "3:10:2"]).slider_guidance == "3:10:2"
    assert setup_parser().parse_args(args).slider_guidance == "3"                   # the default, as before


def test_the_plan_cuts_where_the_push_strength_changes(tmp_path):
    _p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_GUIDANCE="3:4:2:6",
                        FAMILY_SLIDER_POS="smiling @1-2\nlaughing @3-")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"], s["push"]) for s in plan] == [(1, 2, "3"), (3, 4, "3"), (5, 5, "2.5"), (6, 8, "2")]
    _p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_GUIDANCE="2.5")              # one number: the whole run
    assert [(s["from"], s["to"], s["push"]) for s in form.state(desc, v, "")["slider_plan"]] == [(1, 8, "2.5")]
    push = _rows(desc)["FAMILY_SLIDER_GUIDANCE"]          # Single, Split and Slide, as the Clean-end share's box
    assert push["kind"] == "split" and push["split"] == ["3", "10", "2"] and push["slide"] == ["3", "10", "2", "15"]
    songs = next(d for d in training_families() if d.slider_training and "photo" not in d.media)
    assert _rows(songs)["FAMILY_SLIDER_GUIDANCE"]["kind"] == "entry"                   # songs keep the plain box


def test_the_run_card_brings_a_push_strength_as_text_or_a_number(tmp_path):
    live, boxes = tmp_path / ".slider_prompts.json", {"base": "a man,", "pos": "smiling", "neg": "frowning"}
    for guidance, read in (("3:10:2", "3:10:2"), (" 3:4:2:6 ", "3:4:2:6"), (2.5, "2.5"), (3, "3"), ("3.0", "3")):
        live.write_text(json.dumps({**boxes, "guidance": guidance}), encoding="utf-8")
        assert _read_slider_live(str(tmp_path), by_epoch=True)["guidance"] == read, guidance
    for bad in ("3:10", 0, "", "3:10:2:5"):
        live.write_text(json.dumps({**boxes, "guidance": bad}), encoding="utf-8")
        assert _read_slider_live(str(tmp_path), by_epoch=True) is None, bad
    live.write_text(json.dumps({**boxes, "guidance": "3:10:0"}), encoding="utf-8")
    assert _read_slider_live(str(tmp_path), by_epoch=True)["guidance"] == "3:10:0"     # its form is fine ...
    assert _read_slider_live(str(tmp_path), 20, by_epoch=True) is None                  # ... but 0 from epoch 11 of 20


def test_lines_join_in_the_order_the_boxes_keep_them():
    # app.js promptLines sortLines: the lines with epochs of their own by start, then the later end first, then every
    # how many, then as they stood, in the rows such lines hold; a line on all run and a blank row keep their rows
    for typed, joined in (
            ("b @3-5\na @1-2", "a @1-2\nb @3-5"),                                          # by start
            ("woman,\nred coat, @5-\nhat, @1-4", "woman,\nhat, @1-4\nred coat, @5-"),       # row 1 (all run) stays
            ("red coat @1-5\nwoman @1-10\nhat @1-", "hat @1-\nwoman @1-10\nred coat @1-5"),  # a start alike: later end
            ("a @1-10/3\nb @1-10\nc @1-10/2", "b @1-10\nc @1-10/2\na @1-10/3"),            # then every how many
            ("y @2-4\nx @2-4", "y @2-4\nx @2-4"),                                          # alike: as they stood
            ("x @-8\ny @1-8\nq @0-3", "x @-8\ny @1-8\nq @0-3"),                            # no start, or 0: epoch 1
            ("b @2-\n\na @1-\nz @-/1", "a @1-\n\nb @2-\nz @-/1"),                          # blank row, "@-/1" (all run)
            (" @3-4\nsmiling @1-2\nmail@home", "smiling @1-2\n @3-4\nmail@home"),          # no words yet: by its epochs
            ("c @5\nb @5-5\na @5-9", "c @5\na @5-9\nb @5-5")):                             # "@5": from 5 to the end
        assert launch.slider_sort(typed) == joined, typed
        assert launch.slider_sort(joined) == joined, joined


def test_start_the_plan_and_the_run_card_join_lines_in_that_order(tmp_path):
    # typed out of order (the boxes sort a line as you leave it): the command, the plan and its note join them in that
    # order, and the note names the lines by their words
    p, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="c, @5-8\nb, @3-4\na, @1-3",
                       FAMILY_SLIDER_POS="laughing @3-\nsmiling @1-5")
    cmd = _train(p)
    assert [(cmd[k + 1], cmd[k + 2]) for k, t in enumerate(cmd) if t == "--slider_line"][:5] == [
        ("base", "a, @1-3"), ("base", "b, @3-4"), ("base", "c, @5-8"), ("pos", "smiling @1-5"), ("pos", "laughing @3-")]
    st = form.state(desc, v, "")
    assert st["notes"]["FAMILY_SLIDER_BASE"] == ('"a," and "b," are on together at epoch 3 and train as one '
                                                 'description: "a, b,". For different pictures, give them epochs of '
                                                 'their own.')
    plan = st["slider_plan"]
    assert [(s["from"], s["to"]) for s in plan] == [(1, 2), (3, 3), (4, 4), (5, 5), (6, 8)]
    assert [x["text"] for x in plan[1]["lines"]["base"]] == ["a,", "b,"]
    assert [x["text"] for x in plan[2]["lines"]["pos"]] == ["smiling", "laughing"]
    from fizgig.web import server                  # the Run card's "trains on now", from the run's values as typed
    now = server._slider_live_status({"family": "krea2", "values": v, "output_dir": str(tmp_path / "out"),
                                      "progress": {"epoch": 3}})["now"]
    assert now["pos"] == "a, b, smiling, laughing" and now["next"] == 4
    # the trainer sorts a Run card request too (an older page's, a file written by hand); a song model's as it came
    (tmp_path / ".slider_prompts.json").write_text(json.dumps(
        {"base": "a man,", "pos": "laughing @3-\nsmiling @1-5", "neg": "frowning", "guidance": "3",
         "anchors": ["a dog @4-", "a cat @1-3"]}), encoding="utf-8")
    new = _read_slider_live(str(tmp_path))
    assert _request_boxes(new) == {"base": "a man,", "pos": "smiling @1-5\nlaughing @3-", "neg": "frowning",
                                   "anchors": "a cat @1-3\na dog @4-"}
    assert _request_boxes(new, pictures=False)["pos"] == "laughing @3-\nsmiling @1-5"


def test_the_previews_show_the_first_line_with_words(tmp_path):
    # a line whose epochs came before its words is no preview: the first line with words is, in the order lines join
    assert _first_lines({"base": " @1-5\na photo of a woman,", "anchors": " @1-2\na photo of a man @3-"}) == [
        "a photo of a woman,", "a photo of a man"]
    assert _first_lines({"base": "a woman, @3-\na man, @1-2", "anchors": ""}) == ["a man,"]
    for base, shown in ((" @1-5\na photo of a woman,\na photo of a man, @6-", "a photo of a woman,\n"),
                        ("a woman, @3-\na man, @1-2", "a man,\n")):
        p, desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE=base)
        _train(p)
        assert next(t for f, t in p.files if f.endswith("_slider_prompt.txt")) == shown, base
    rows = _rows(desc)
    assert "the dial on the first line with words" in rows["FAMILY_SLIDER_BASE"]["hint"]
    assert "a Pause and Resume renders them again once" in rows["FAMILY_SLIDER_KEEP"]["hint"]


def test_a_line_that_starts_with_a_dash_reaches_the_trainer(tmp_path):
    # as typed, most Pythons' argparse takes "-smiling" for an option (newer 3.12 ones take it as a value): the launch
    # gives it a space before it, which every one takes as a value
    args = ["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n"]
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="-smiling @1-4\nlaughing @5-",
                         FAMILY_SLIDER_ANCHORS="-a dog")
    cmd = _train(p)
    given = [x for k, t in enumerate(cmd) if t == "--slider_line" for x in cmd[k:k + 3]]
    assert given[given.index("pos") + 1] == " -smiling @1-4"                # a space before it: a value
    boxes = _line_boxes(setup_parser().parse_args(args + given).slider_line)
    assert launch.slider_at(boxes, 1)[1] == "-smiling" and boxes["anchors"] == "-a dog"      # the trainer strips it
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_ANCHORS="-smiling\na dog")    # one line each: the anchors
    cmd = _train(p)
    i = cmd.index("--slider_anchors")
    assert cmd[i + 1:i + 3] == [" -smiling", "a dog"]
    assert _anchor_lines(setup_parser().parse_args(args + cmd[i:i + 3]).slider_anchors) == ["-smiling", "a dog"]


def test_no_half_rate_epochs_where_the_optimizer_owns_the_rate(tmp_path):
    moss, v = get_family("moss_image"), {"MOSS_LOWNOISE_PCT": "11:7:40:8:3"}
    assert form.clean_share_at(moss, v, 11) == ("11", True)                  # every 3rd epoch after 8: half the LR
    for opt in ("automagic3", "automagic4", " Automagic4 "):                  # they own the rate: the trainer leaves
        assert form.clean_share_at(moss, {**v, "OPTIMIZER_TYPE": opt}, 11) == ("11", False), opt     # it out
    assert form.clean_share_at(moss, {**v, "OPTIMIZER_TYPE": "prodigy"}, 11) == ("11", True)   # a multiplier: kept
    ft = next(label for k, label in form.kinds(moss) if k == "finetune")     # a fine-tune trains it as Adafactor
    assert form.clean_share_at(moss, {**v, "OPTIMIZER_TYPE": "automagic4", "FAMILY_KIND": ft}, 11) == ("11", True)
    _p, desc, vals = _plan(tmp_path, "moss_image", MAX_TRAIN_EPOCHS="20", MOSS_LOWNOISE_PCT="11:7:40:8:3",
                           OPTIMIZER_TYPE="automagic4")
    plan = form.state(desc, vals, "")["slider_plan"]
    assert not any(s["half"] for s in plan) and [s["detail"] for s in plan][:3] == ["11", "40", "11"]


def test_a_set_of_prompts_that_comes_back_takes_its_encodings_again():
    import torch
    from fizgig.families.train import SLIDER_ENCODINGS_KEPT, _BankCache, _cached_encode
    calls = []

    def encode(texts):                 # as a text encoder that pads to the longest text it encodes with
        calls.append(list(texts))
        n = max(map(len, texts))
        return [{"hidden_states": torch.arange(n, dtype=torch.float32) * len(t)} for t in texts]

    cache = _BankCache(keep=SLIDER_ENCODINGS_KEPT)
    a, b = ["a woman,", "a woman, smiling", "a woman, frowning"], ["a man,", "a man, smiling", "a man, frowning"]
    first = _cached_encode(cache, a, encode)
    _cached_encode(cache, b, encode)
    again = _cached_encode(cache, a, encode)          # lines taking turns: a, b, a - the text encoder twice, not 3
    assert again is first and calls == [a, b]
    _cached_encode(cache, a + ["a photo of a dog in the snow"], encode)    # other texts with them: encoded anew
    _cached_encode(cache, a[::-1], encode)                                 # and in another order
    assert len(calls) == 4

def test_a_one_line_prompt_that_starts_with_a_dash_reaches_the_trainer(tmp_path):
    from fizgig.families.train import _prompt_texts
    p, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="-a photo of a man,")
    cmd = _train(p)
    i = cmd.index("--slider_prompts")
    assert cmd[i + 1:i + 4] == [" -a photo of a man,", " -a photo of a man, smiling", " -a photo of a man, frowning"]
    args = setup_parser().parse_args(["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                      "--slider_prompts", *cmd[i + 1:i + 4]])        # argparse takes them as values
    assert _prompt_texts(args.slider_prompts) == ["-a photo of a man,", "-a photo of a man, smiling",
                                                  "-a photo of a man, frowning"]   # the trainer's: as typed
    assert _prompt_texts(["a photo of a man,"]) == ["a photo of a man,"] and _prompt_texts(None) is None