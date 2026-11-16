"""Adjust each section: with Change the prompts during the run (advanced) on, the plan's tick (FAMILY_SLIDER_TWEAKS_ON,
off by default) gives its sections their own rate - a percent of the learning rate, in place of the schedule's -, fine
detail - the epoch's Clean-end share (MOSS-Image's) -, push strength - in place of the Push strength box's - and how
firmly - the anchors' weight, with Leave these alone lines. They are kept as text by epochs (FAMILY_SLIDER_TWEAKS,
"5-8:rate=25,detail=40,push=2,firm=1.5;12:rate=50", a later entry winning) and sent only with both ticks on
(--slider_tweaks)."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import _first_lines, _line_boxes, _slider_next, _tweak_firm, _tweak_lr, setup_parser

TEXT = "5-8:rate=25,detail=40;6:rate=50"
ARGS = ["--family", "moss_image", "--dit", "d", "--output_dir", "o", "--output_name", "n"]
APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _plan(tmp_path, fam, **values):
    desc, models = get_family(fam), {}
    for mf in desc.model_files:
        f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "12",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning",
         "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BY_EPOCH": True, "FAMILY_SLIDER_TWEAKS_ON": True,
         "FAMILY_SLIDER_TWEAKS": TEXT, **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    p = launch.plan(desc, inp)
    return p, next((s.cmd for s in p.stages if s.name == "Training"), None), desc, v


def _sent(cmd):
    return cmd[cmd.index("--slider_tweaks") + 1] if "--slider_tweaks" in cmd else None


def test_the_text_and_what_each_epoch_takes():
    assert launch.slider_tweaks(TEXT) == [(5, 8, {"rate": 25.0, "detail": 40.0}), (6, 6, {"rate": 50.0})]
    assert launch.slider_tweaks(" 5 - 8 : rate = 25 ;; 12:DETAIL=40.5;") == [(5, 8, {"rate": 25.0}),
                                                                           (12, 12, {"detail": 40.5})]
    for bad in ("x", "5-8", "5-8:", "5-8:rate", "5-8:rate=", "5-8:rate=abc", "5-8:speed=2", "5-8:rate=1,rate=2",
                "-5:rate=2", "a-8:rate=2", "5-8:rate=nan", "5-8:rate=inf", "5-8:rate=1_0", "5-8:rate=2=3"):
        assert launch.slider_tweaks(bad) == [], bad
    # a later entry wins, setting by setting
    assert launch.slider_tweak_at(TEXT, 5) == launch.slider_tweak_at(TEXT, 8) == {"rate": 25.0, "detail": 40.0}
    assert launch.slider_tweak_at(TEXT, 6) == {"rate": 50.0, "detail": 40.0}
    assert launch.slider_tweak_at(TEXT, 9) == launch.slider_tweak_at("", 1) == launch.slider_tweak_at(None, 1) == {}
    assert launch.slider_tweaks_text(launch.slider_tweaks(TEXT)) == TEXT
    assert launch.slider_tweaks_text([(3, 3, {"detail": 12.5}), (4, 9, {})]) == "3:detail=12.5"     # nothing set: out
    assert launch.slider_tweaks_in_run("5-12:rate=25;10:detail=3", 9) == "5-9:rate=25"


def test_start_refuses_what_cant_train(tmp_path):
    problems = launch.slider_tweak_problems
    assert problems(TEXT, 12) == [] and problems("5-8:rate=1;9:detail=0;10-12:rate=100,detail=100", 12) == []
    assert problems("5-8:rate=abc", 12) == [
        "Slider: \"5-8:rate=abc\" in the sections' own settings can't be read - each one is epochs and what they set, "
        "e.g. 5-8:rate=25,push=2"]
    assert problems("5-8:rate=0;9:rate=150;10:detail=-1;11:detail=101", 12) == [
        "Slider: a section's rate must be 1 to 100% of the learning rate - it is 0 for epochs 5-8",
        "Slider: a section's rate must be 1 to 100% of the learning rate - it is 150 for epoch 9",
        "Slider: a section's fine detail must be 0 to 100% - it is -1 for epoch 10",
        "Slider: a section's fine detail must be 0 to 100% - it is 101 for epoch 11"]
    assert problems("10-14:rate=50", 12) == [
        "Slider: a section is adjusted for epochs 10-14, past the run's last (12) - drop it (under the plan) or set "
        "Max Train Epochs to 14"]
    assert problems("8-5:rate=50", 12) == [
        "Slider: a section's epochs run backwards (8-5) - give its first epoch first"]
    assert problems("0:rate=50", 12) == [
        "Slider: a section is adjusted for epoch 0 - epochs start at 1"]
    assert problems("5-8:rate=150;14:rate=50", 12, ("detail",)) == []       # only what the run can change is checked
    p, cmd, _desc, _v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS="5-8:rate=0")
    assert cmd is None and p.problems == [
        "Slider: a section's rate must be 1 to 100% of the learning rate - it is 0 for epochs 5-8"]
    for off in ({"FAMILY_SLIDER_TWEAKS_ON": False}, {"FAMILY_SLIDER_BY_EPOCH": False}):     # kept, not checked
        p, cmd, _desc, v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS="5-8:rate=0", **off)
        assert not p.problems and _sent(cmd) is None and v["FAMILY_SLIDER_TWEAKS"] == "5-8:rate=0", off


def test_the_command_carries_them_only_with_both_ticks_on(tmp_path):
    _p, cmd, _desc, _v = _plan(tmp_path, "moss_image")
    assert _sent(cmd) == TEXT and "--slider_by_epoch" in cmd
    for off in ({"FAMILY_SLIDER_TWEAKS_ON": False}, {"FAMILY_SLIDER_BY_EPOCH": False}):
        _p, cmd, _desc, _v = _plan(tmp_path, "moss_image", **off)
        assert cmd is not None and _sent(cmd) is None, off
    _p, cmd, _desc, _v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS="")
    assert _sent(cmd) is None                                          # nothing set: no flag
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2")                       # no fine detail to set on Krea 2: the rate only
    assert _sent(cmd) == "5-8:rate=25;6:rate=50"
    a = setup_parser().parse_args(ARGS + ["--slider_by_epoch", "--slider_tweaks", TEXT])
    assert a.slider_tweaks == TEXT and setup_parser().parse_args(ARGS).slider_tweaks == ""


def test_automagic_drops_the_rate(tmp_path):
    p, cmd, desc, v = _plan(tmp_path, "moss_image", OPTIMIZER_TYPE="automagic4",
                            FAMILY_SLIDER_TWEAKS="5-8:rate=150,detail=40;10:rate=50")
    assert not p.problems and _sent(cmd) == "5-8:detail=40"            # left out, not refused (150 too)
    st = form.state(desc, v, "")
    assert {k: x for k, x in st["slider_tweaks"].items() if k != "hint"} == {
        "on": True, "automagic": True, "rate_off": form.TWEAK_RATE_OFF, "detail": True}
    assert "learning rate" not in st["slider_tweaks"]["hint"]         # the tick's hint: what they can set
    assert "_slider_tweak_rate" in st["disabled"]                     # the sections' rate controls, greyed
    assert all(s["rate"] == 100 and not s["rate_set"] and not s["half"] for s in st["slider_plan"])
    # its fine detail is kept: epochs 5-8, cut where the schedule's own share changes (11 to 40 at epoch 8)
    assert [(s["from"], s["to"], s["detail"], s["detail_own"]) for s in st["slider_plan"] if s["detail_set"]] == [
        (5, 7, "40", "11"), (8, 8, "40", "40")]
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2", OPTIMIZER_TYPE="automagic3")
    assert _sent(cmd) is None                                          # nothing left to send
    st = form.state(get_family("krea2"), {**v, "OPTIMIZER_TYPE": "adamw"}, "")
    assert st["slider_tweaks"]["rate_off"] == "" and not st["slider_tweaks"]["detail"]
    assert "_slider_tweak_rate" not in st["disabled"]


def test_the_plan_cuts_where_a_section_sets_its_own(tmp_path):
    _p, _cmd, desc, v = _plan(tmp_path, "krea2", MAX_TRAIN_EPOCHS="8", FAMILY_SLIDER_TWEAKS="3-4:rate=25;4:rate=25")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"], s["rate"], s["rate_set"], s["rate_own"]) for s in plan] == [
        (1, 2, 100, False, 100), (3, 4, 25, True, 100), (5, 8, 100, False, 100)]
    assert all(s["detail"] is None and not s["detail_set"] for s in plan)
    st = form.state(desc, {**v, "FAMILY_SLIDER_TWEAKS_ON": False}, "")    # off: the schedule's, kept aside
    assert [(s["from"], s["to"], s["rate_set"]) for s in st["slider_plan"]] == [(1, 8, False)]
    assert st["slider_tweaks"]["on"] is False
    # MOSS-Image's Clean-end share (11:7:40:8:3): a section cut where the schedule or what it sets changes
    _p, _cmd, desc, v = _plan(tmp_path, "moss_image")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"], s["detail"], s["detail_set"], s["rate"], s["half"]) for s in plan] == [
        (1, 4, "11", False, 100, False), (5, 5, "40", True, 25, False), (6, 6, "40", True, 50, False),
        (7, 7, "40", True, 25, False), (8, 8, "40", True, 25, False), (9, 10, "40", False, 100, False),
        (11, 11, "11", False, 50, True), (12, 12, "40", False, 100, False)]
    assert [(s["detail_own"], s["rate_own"]) for s in plan[3:5]] == [("11", 100), ("40", 100)]   # for the tooltip
    _p, _cmd, desc, v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS="11:rate=100")   # its ½ rate, set back
    eleven = next(s for s in form.state(desc, v, "")["slider_plan"] if s["from"] == 11)
    assert (eleven["rate"], eleven["rate_set"], eleven["rate_own"], eleven["half"]) == (100, True, 50, False)


def test_the_driver_trains_the_detail_and_the_rate_the_plan_shows(tmp_path):
    from fizgig.moss_image.driver import MOSSImageDriver
    _p, cmd, desc, v = _plan(tmp_path, "moss_image", MAX_TRAIN_EPOCHS="20", MOSS_LOWNOISE_PCT="11:7:40:8:3",
                             FAMILY_SLIDER_TWEAKS="5-8:detail=25;11:detail=33.333;14:rate=100;15-16:rate=10")
    text, plan = _sent(cmd), form.state(desc, v, "")["slider_plan"]
    m = MOSSImageDriver.__new__(MOSSImageDriver)
    m.options = {"lownoise_pct": "11:7:40:8:3"}
    plain = {e: m._share(e) for e in range(1, 21)}
    at = {e: launch.slider_tweak_at(text, e) for e in range(1, 21)}        # as train.py sets them up
    m.share_overrides = {e: t["detail"] for e, t in at.items() if "detail" in t}
    for seg in plan:
        for e in range(seg["from"], seg["to"] + 1):
            assert m._share(e) == (seg["detail"], plain[e][1]), e         # its share, its half rate the schedule's
            assert m.scheduled_lr(e) == m.step_policy({}, e)[1], e           # its multiplier is its schedule's
            assert _tweak_lr(m.step_policy({}, e)[1], at[e], m.scheduled_lr(e)) == seg["rate"] / 100, e   # trains at
    assert m._share(11) == ("33.33", True) and m._share(6) == ("25", False) and m._share(14) == ("11", True)
    assert _tweak_lr(m.step_policy({}, 14)[1], at[14], m.scheduled_lr(14)) == 1.0
    assert _tweak_lr(m.step_policy({}, 17)[1], at[17], m.scheduled_lr(17)) == 0.5
    del m.share_overrides
    assert all(m._share(e) == plain[e] for e in plain)                  # none set: as before


def test_a_sections_rate_takes_the_schedules_place_and_keeps_the_drivers_other_factors():
    import types
    import torch
    from fizgig.families.driver import FamilyDriver
    from fizgig.minimax.driver import MiniMaxDriver
    assert _tweak_lr(0.5, {}) == 0.5 and _tweak_lr(0.5, None) == 0.5 and _tweak_lr(0.1, {"detail": 40.0}) == 0.1
    assert _tweak_lr(0.5, {"rate": 100.0}, 0.5) == 1.0          # a half-rate epoch (MOSS-Image) set back to the full
    assert _tweak_lr(1.0, {"rate": 25.0, "detail": 3.0}) == 0.25
    assert FamilyDriver.scheduled_lr(None, 3) == 1.0 and MiniMaxDriver.scheduled_lr is FamilyDriver.scheduled_lr
    # MiniMax H3's Adapter-relative LR (adapter_ramp): its ramp stays under a section's rate, which scales it
    h3 = MiniMaxDriver.__new__(MiniMaxDriver)
    h3.options, h3._ramp = {"adapter_ramp": "0.005"}, types.SimpleNamespace(mult=0.125)
    skip, mult = h3.step_policy({"latents": torch.zeros(1, 24, 8, 8)}, 3)      # a prompt slider's practice picture
    assert (skip, mult) == (False, 0.125)
    assert _tweak_lr(mult, {"rate": 50.0}, h3.scheduled_lr(3)) == 0.0625 and _tweak_lr(mult, {}, 1.0) == 0.125


def test_the_tick_sits_on_the_plan_and_is_off_by_default():
    rows = lambda fam: {r["key"]: r for s in form.form(get_family(fam))["sections"] for r in s["rows"] if r}
    tick = rows("moss_image")["_slider_plan"]["tick"]
    assert (tick["key"], tick["kind"], tick["default"]) == ("FAMILY_SLIDER_TWEAKS_ON", "check", False)
    assert tick["label"] == launch.TWEAKS == "Adjust each section"
    # its first sentence shows under it (app.js planRow), the rest behind More: what this run's sections can set
    assert tick["hint"].startswith("Gives each section below its own settings, in place of the run's. A section can "
                                   "set its learning rate (Normal or a share of it, such as 50%), fine detail (the "
                                   "share of its steps on lightly noised pictures) and push strength. ")
    assert rows("krea2")["_slider_plan"]["tick"]["hint"].startswith(               # Krea 2: no fine detail
        "Gives each section below its own settings, in place of the run's. A section can set its learning rate (Normal "
        "or a share of it, such as 50%) and push strength. ")
    assert form.DEFAULTS["FAMILY_SLIDER_TWEAKS_ON"] is False and form.DEFAULTS["FAMILY_SLIDER_TWEAKS"] == ""
    moss, v = get_family("moss_image"), {"FAMILY_SLIDER_BY_EPOCH": True, "FAMILY_SLIDER_TWEAKS_ON": True}
    assert launch.slider_tweaks_on(moss, v)
    assert not launch.slider_tweaks_on(moss, {**v, "FAMILY_SLIDER_BY_EPOCH": False})
    assert not launch.slider_tweaks_on(moss, {"FAMILY_SLIDER_BY_EPOCH": True})        # missing: off
    assert not launch.slider_tweaks_on(get_family("acestep15xl"), v)                   # songs: no plan
    off = {**form.DEFAULTS, "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts"}
    assert form.state(moss, off, "")["slider_tweaks"] is None and "_slider_plan" in form.state(moss, off, "")["hidden"]


def test_what_is_set_past_the_run_can_be_dropped(tmp_path):
    p, _cmd, desc, v = _plan(tmp_path, "moss_image", MAX_TRAIN_EPOCHS="8",
                             FAMILY_SLIDER_TWEAKS="5-12:rate=25;10:detail=3")
    assert p.problems == [
        "Slider: a section is adjusted for epochs 5-12, past the run's last (8) - drop it (under the plan) or set Max "
        "Train Epochs to 12",
        "Slider: a section is adjusted for epoch 10, past the run's last (8) - drop it (under the plan) or set Max "
        "Train Epochs to 10"]
    st = form.state(desc, v, "")
    assert st["notes"]["_slider_plan"] == ("A section is adjusted past the run's last epoch (8), so Start won't run. "
                                           "Drop what's past it, or set Max Train Epochs to 12.")
    assert st["actions"]["_slider_plan"] == {"label": "Drop it", "values": {"FAMILY_SLIDER_TWEAKS": "5-8:rate=25"}}
    p, _cmd, _desc, v = _plan(tmp_path, "moss_image", MAX_TRAIN_EPOCHS="8", FAMILY_SLIDER_TWEAKS="5-8:rate=25")
    assert not p.problems and "_slider_plan" not in form.state(desc, v, "")["notes"]


def test_resume_takes_the_plans_tweaks():
    from fizgig.web import server
    v = {"FAMILY_SLIDER_BY_EPOCH": True, "FAMILY_SLIDER_TWEAKS": "11:rate=50", "FAMILY_SLIDER_TWEAKS_ON": True}
    got = server._resume_slider_values(v, {"FAMILY_SLIDER_TWEAKS": "11:rate=25", "SAMPLE_SEED": "1"})
    assert got["FAMILY_SLIDER_TWEAKS"] == "11:rate=25" and got["FAMILY_SLIDER_TWEAKS_ON"] is True   # Pause, change, Resume
    assert server._resume_slider_values(v, {"FAMILY_SLIDER_TWEAKS": ""})["FAMILY_SLIDER_TWEAKS"] == ""


# ---- a section's own push strength and how firmly --------------------------------------------------------------------
ALL = "5-8:rate=25,detail=40,push=2,firm=1.5;12:rate=50"


def test_push_and_firm_in_the_text_above_0_as_start_takes_the_boxes(tmp_path):
    assert launch.slider_tweaks(ALL) == [(5, 8, {"rate": 25.0, "detail": 40.0, "push": 2.0, "firm": 1.5}),
                                         (12, 12, {"rate": 50.0})]
    assert launch.slider_tweaks_text(launch.slider_tweaks(ALL)) == ALL
    assert launch.slider_tweak_at("5-8:push=2;6:firm=0.5,PUSH=3", 6) == {"push": 3.0, "firm": 0.5}   # a later one wins
    for bad in ("5-8:push", "5-8:push=x", "5-8:firm=1,firm=2", "5-8:push=inf", "5-8:strength=2"):
        assert launch.slider_tweaks(bad) == [], bad
    problems = launch.slider_tweak_problems
    assert problems("5-8:push=0.01,firm=0.01;9:push=50,firm=20", 12) == []     # above 0, no highest
    assert problems("5-8:push=0;9:firm=-1;10:push=-2,firm=0;14:push=2", 12) == [
        "Slider: a section's push strength must be above 0 - it is 0 for epochs 5-8",
        "Slider: a section's \"how firmly\" must be above 0 - it is -1 for epoch 9",
        "Slider: a section's push strength must be above 0 - it is -2 for epoch 10",
        "Slider: a section's \"how firmly\" must be above 0 - it is 0 for epoch 10",
        "Slider: a section is adjusted for epoch 14, past the run's last (12) - drop it (under the plan) or set Max "
        "Train Epochs to 14"]
    p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_TWEAKS="5-8:push=0;9:firm=0")     # no anchors: no firm
    assert cmd is None and p.problems == ["Slider: a section's push strength must be above 0 - it is 0 for epochs 5-8"]
    p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_TWEAKS="9:firm=0", FAMILY_SLIDER_ANCHORS="a dog @3-")
    assert cmd is None and p.problems == ["Slider: a section's \"how firmly\" must be above 0 - it is 0 for epoch 9"]


def test_push_goes_with_both_ticks_on_and_firm_only_with_anchors(tmp_path):
    assert launch.slider_has_anchors({"FAMILY_SLIDER_ANCHORS": "\na photo of a dog @3-"})
    assert not launch.slider_has_anchors({"FAMILY_SLIDER_ANCHORS": " @3-\n\n"}) and not launch.slider_has_anchors({})
    _p, cmd, _desc, _v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS=ALL)
    assert _sent(cmd) == "5-8:rate=25,detail=40,push=2;12:rate=50"
    _p, cmd, _desc, _v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS=ALL, FAMILY_SLIDER_ANCHORS="a dog @3-")
    assert _sent(cmd) == ALL
    for off in ({"FAMILY_SLIDER_TWEAKS_ON": False}, {"FAMILY_SLIDER_BY_EPOCH": False}):
        _p, cmd, _desc, _v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_TWEAKS=ALL, FAMILY_SLIDER_ANCHORS="a dog",
                                   **off)
        assert cmd is not None and _sent(cmd) is None, off
    # Automagic drops the rate only, fine detail is MOSS-Image's: push and firm go for any model with the tick
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2", OPTIMIZER_TYPE="automagic3", FAMILY_SLIDER_TWEAKS=ALL,
                               FAMILY_SLIDER_ANCHORS="a photo of a dog")
    assert _sent(cmd) == "5-8:push=2,firm=1.5"
    a = setup_parser().parse_args(ARGS + ["--slider_by_epoch", "--slider_tweaks", ALL])
    assert a.slider_tweaks == ALL


def test_the_plan_cuts_where_a_section_sets_its_push_or_how_firmly(tmp_path):
    _p, _cmd, desc, v = _plan(tmp_path, "krea2", MAX_TRAIN_EPOCHS="8", FAMILY_SLIDER_ANCHORS="a photo of a dog @4-",
                              FAMILY_SLIDER_TWEAKS="2-3:push=2.5;5-6:firm=2;1-3:firm=9")
    plan = form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["to"], s["push"], s["push_set"], s["push_own"], s["firm"], s["firm_set"]) for s in plan] == [
        (1, 1, "3", False, "3", None, False), (2, 3, "2.5", True, "3", None, False),    # no anchors on: no firm, no cut
        (4, 4, "3", False, "3", "1", False), (5, 6, "3", False, "3", "2", True), (7, 8, "3", False, "3", "1", False)]
    assert all(s["firm_own"] == "1" for s in plan)                     # How firmly, for the tooltip and ↺
    st = form.state(desc, {**v, "FAMILY_SLIDER_TWEAKS_ON": False}, "")      # off: the boxes' own, kept aside
    assert [(s["from"], s["to"], s["push"], s["firm"]) for s in st["slider_plan"]] == [(1, 3, "3", None),
                                                                                      (4, 8, "3", "1")]
    assert not any(s["push_set"] or s["firm_set"] for s in st["slider_plan"])
    # the tick's hint follows the form: no rate under Automagic, how firmly with anchors
    assert form.state(desc, {**v, "OPTIMIZER_TYPE": "automagic4"}, "")["slider_tweaks"]["hint"].startswith(
        "Gives each section below its own settings, in place of the run's. A section can set its push strength and how "
        "firmly it keeps the Leave these alone lines. ")


def test_the_trainer_takes_the_push_and_how_firmly_the_plan_shows(tmp_path):
    _p, cmd, desc, v = _plan(tmp_path, "krea2", MAX_TRAIN_EPOCHS="10", FAMILY_SLIDER_GUIDANCE="3:4:2:8",
                             FAMILY_SLIDER_ANCHOR_WEIGHT="1.25", FAMILY_SLIDER_POS="smiling @1-5\nlaughing @6-",
                             FAMILY_SLIDER_ANCHORS="a photo of a dog @3-",
                             FAMILY_SLIDER_TWEAKS="2-3:push=4;4:firm=0.5;6-7:push=1.5,firm=2;9:firm=3")
    text, plan = _sent(cmd), form.state(desc, v, "")["slider_plan"]
    assert [(s["from"], s["push"], s["push_set"], s["firm"], s["firm_set"]) for s in plan] == [
        (1, "3", False, None, False), (2, "4", True, None, False), (3, "4", True, "1.25", False),
        (4, "3", False, "0.5", True), (5, "2.75", False, "1.25", False), (6, "1.5", True, "2", True),
        (7, "1.5", True, "2", True), (8, "2", False, "1.25", False), (9, "2", False, "3", True),
        (10, "2", False, "1.25", False)]
    # the trainer as the command starts it: its plan, its anchors' weight and its lines
    guidance = cmd[cmd.index("--slider_guidance") + 1]
    weight = setup_parser().parse_args(ARGS + ["--slider_anchor_weight", cmd[cmd.index("--slider_anchor_weight") + 1]])
    boxes = _line_boxes([(cmd[k + 1], cmd[k + 2]) for k, t in enumerate(cmd) if t == "--slider_line"])
    base, pos, neg, anchors = launch.slider_at(boxes, 1)
    cur = {"boxes": boxes, "plan": guidance, "prompts": [base, f"{base} {pos}", f"{base} {neg}"], "anchors": anchors,
           "texts": [base, pos, neg], "preview": _first_lines(boxes)}
    for seg in plan:
        for e in range(seg["from"], seg["to"] + 1):
            # the push it trains at: the start's (train_family), then each epoch boundary's (slider_live)
            push = (launch.slider_push_now(guidance, text, e) if e == 1 else
                    _slider_next(None, cur, e - 1, True, desc, text)["guidance"])
            assert f"{round(push, 2):g}" == seg["push"], e
            # the anchors' weight each step takes (_prompt_slider_step), where the section has anchors on
            firm = _tweak_firm(weight.slider_anchor_weight, launch.slider_tweak_at(text, e))
            assert seg["firm"] is None or float(seg["firm"]) == firm, e
    assert _slider_next(None, cur, 5, True, desc)["guidance"] == 2.5        # no tweaks given: the plan's
    assert _tweak_firm(1.25, {}) == _tweak_firm(1.25, {"rate": 50.0}) == 1.25 and _tweak_firm(1, {"firm": 3.0}) == 3.0


def test_the_run_card_shows_a_sections_own_push(tmp_path):
    from fizgig.web import server
    v = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "12",
         "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling",
         "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True, "FAMILY_SLIDER_TWEAKS_ON": True,
         "FAMILY_SLIDER_TWEAKS": "5-8:push=2;12:rate=50,push=1.5"}
    run = {"family": "krea2", "values": v, "output_dir": str(tmp_path), "progress": {"epoch": 6}}
    st = server._slider_live_status(run)               # one line each, one number: shown for the sections' own push
    assert st["guidance"] == "3" and st["now"]["push"] == "2" and st["now"]["next"] is None
    assert st["push_note"] == ("Epochs 5–8 and 12 keep their own push strength, set in the Slider card's plan. Apply "
                               "changes the rest.")
    st = server._slider_live_status({**run, "progress": {"epoch": 9}})
    assert st["now"]["push"] == "3" and st["push_note"] == ("Epoch 12 keeps its own push strength, set in the Slider "
                                                            "card's plan. Apply changes the rest.")
    st = server._slider_live_status({**run, "progress": {"epoch": 12}})
    assert st["now"]["push"] == "1.5" and st["push_note"] is None          # none left for Apply to reach
    off = server._slider_live_status({**run, "values": {**v, "FAMILY_SLIDER_TWEAKS_ON": False}})
    assert off["now"] is None and off["push_note"] is None                # the tick off: the box's one number
    # Apply took 2.5 at the end of epoch 6: the sections keep their own, the rest trains at the new one
    took = {"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "2.5", "anchors": []}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps(took), encoding="utf-8")
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**took, "epoch": 6}), encoding="utf-8")
    run["values"] = {**v, "FAMILY_SLIDER_GUIDANCE": "2.5"}
    assert [server._slider_live_status({**run, "progress": {"epoch": e}})["now"]["push"] for e in (7, 9, 12)] == [
        "2", "2.5", "1.5"]


def test_the_page_writes_the_text_as_launch_reads_it_and_says_the_run_cards_note():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    grab = lambda pat: re.search(pat, src, re.S).group(0)
    typed = "5-8:push=2,firm=1.5,rate=25,detail=40;6:push=x;12:rate=50"      # an entry it can't read goes
    js = (grab(r"const TWEAK_NUM = [^\n]*\n") + grab(r"const TWEAK_KEYS = [^\n]*\n")
          + grab(r"function tweakEpochs\(text\) \{.*?\n\}\n") + grab(r"function tweakText\(byEpoch\) \{.*?\n\}\n")
          + f"console.log(JSON.stringify(tweakText(tweakEpochs({json.dumps(typed)}))));")
    out = json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8"))
    assert out == ALL                                                  # in launch.py's order
    assert all(launch.slider_tweak_at(out, e) == launch.slider_tweak_at(typed, e) for e in range(1, 13))
    note = "Epoch 12 keeps its own push strength, set in the Slider card's plan. Apply changes the rest."
    sl = {"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "3", "anchors": "", "now": None,
          "by_epoch": True, "epoch": None, "waiting": False, "failed": None}
    js = ("const nodes = {}; const $ = (id) => nodes[id] || (nodes[id] = {style: {}, value: '', textContent: ''});"
          "const S = {sl: {shown: null, edited: false}}; const SL_FIELDS = {base: 'slBase'};"
          "const sliderLiveLook = () => false;\n" + grab(r"function sliderLiveDraw\(sl\) \{.*?\n\}\n")
          + f"const out = [{json.dumps(note)}, null].map((n) => (sliderLiveDraw({{...{json.dumps(sl)}, push_note: n}}),"
          "[$('slPushNote').textContent, $('slPushNote').style.display]));"
          "console.log(JSON.stringify(out));")
    assert json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8")) == [[note, ""], ["", "none"]]

def test_a_number_that_reads_as_infinity_is_refused():
    assert launch.slider_tweaks("5:push=1e400") == [] and launch.slider_tweaks("5:firm=2,push=3") == [
        (5, 5, {"firm": 2.0, "push": 3.0})]
    said = launch.slider_tweak_problems("5:push=1e400", 10)
    assert len(said) == 1 and "can't be read" in said[0]


def test_push_strength_and_how_firmly_say_which_epochs_set_their_own(tmp_path):
    # Adjust each section on: the boxes stay the run's own, for the rest, and say which epochs set theirs
    _p, _cmd, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_ANCHORS="a photo of a dog @5-12",
                              FAMILY_SLIDER_TWEAKS="5-8:push=2,firm=1.5;12:push=4")
    st = form.state(desc, v, "")
    assert st["notes"]["FAMILY_SLIDER_GUIDANCE"] == ("Epochs 5–8 and 12 use their own push strength from the plan "
                                                     "below; this box sets the rest.")
    assert st["notes"]["FAMILY_SLIDER_ANCHOR_WEIGHT"] == ('Epochs 5–8 use their own "how firmly" from the plan below; '
                                                          "this box sets the rest.")
    assert not {"FAMILY_SLIDER_GUIDANCE", "FAMILY_SLIDER_ANCHOR_WEIGHT"} & set(st["disabled"])     # still editable
    one = form.state(desc, {**v, "FAMILY_SLIDER_TWEAKS": "12:push=4"}, "")["notes"]
    assert one["FAMILY_SLIDER_GUIDANCE"].startswith("Epoch 12 uses its own push strength")
    assert "FAMILY_SLIDER_ANCHOR_WEIGHT" not in one
    off = form.state(desc, {**v, "FAMILY_SLIDER_TWEAKS_ON": False}, "")["notes"]       # the tick off: none
    assert "FAMILY_SLIDER_GUIDANCE" not in off and "FAMILY_SLIDER_ANCHOR_WEIGHT" not in off
    assert launch.slider_epochs_said([]) == ("", False)
    assert launch.slider_epochs_said([3, 1, 2, 7]) == ("Epochs 1–3 and 7", True)