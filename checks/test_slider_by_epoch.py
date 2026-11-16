"""Change the prompts during the run (advanced): a picture model's tick on the Slider card (FAMILY_SLIDER_BY_EPOCH, off
when missing) for a prompt slider's lines with their own epochs, its push strength by epoch, its plan and its kept
practice pictures.
Off, the prompt slider is the plain slider's - its form, Start's checks, its command and the trainer - and what only the tick
trains (several lines, a Split or a Slide) is refused, saying to turn it on. The tick on: test_slider_lines.py."""
import importlib.util
import io
import json
import os
import sys

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import _read_slider_live, _request_boxes, setup_parser

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LINES = ('Slider: "The +1 end adds" has 2 lines - turn on "Change the prompts during the run (advanced)" for lines '
         'with their own epochs, or keep one line')
SPLIT = ('Slider push strength 3:10:2 changes by epoch - turn on "Change the prompts during the run (advanced)" for a '
         'Split or a Slide, or give one number')
ARGS = ["--family", "krea2", "--dit", "d", "--output_dir", "o", "--output_name", "n"]


def _plan(tmp_path, fam, **values):
    desc, models = get_family(fam), {}
    for mf in desc.model_files:
        f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "8",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning",
         "FAMILY_SLIDER_GUIDANCE": "3", **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    p = launch.plan(desc, inp)
    return p, next((s.cmd for s in p.stages if s.name == "Training"), None), desc, v


# the golden file made again, when upstream changed the plain slider on purpose
REMAKE = ("With the tick off, the slider must be as upstream had it (ea4f47d4). If upstream changed it on purpose, "
          "make the file again from the repo, with the upstream commit the branch starts from: git archive "
          "--format=tar -o base.tar ea4f47d4 ; tar -x -f base.tar -C <an empty folder> ; python -I "
          "checks/data/make_slider_off_upstream.py <that folder>/src checks/data/slider_off_upstream.json")


def test_off_is_the_plain_slider(tmp_path):
    # the golden file: what the plain slider decides - its part of the command, Start's slider problems, the slider
    # previews' prompt and the Slider card's rows on show (their keys, kinds, labels and values, not their hints) -
    # made once from the upstream base (how: data/make_slider_off_upstream.py) for every model with slider training
    # then; the same here, the tick off
    spec = importlib.util.spec_from_file_location("make_slider_off_upstream",
                                                  os.path.join(DATA, "make_slider_off_upstream.py"))
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    with open(os.path.join(DATA, "slider_off_upstream.json"), encoding="utf-8") as f:
        golden = json.load(f)["models"]
    got = maker.observe(str(tmp_path))
    assert golden and set(golden) <= set(got), REMAKE        # a model upstream added since has nothing to compare
    for fam in golden:
        assert got[fam]["rows"] == golden[fam]["rows"], (fam, REMAKE)
        assert set(got[fam]["cases"]) == set(golden[fam]["cases"]) == set(maker.CASES), (fam, REMAKE)
        for name, case in golden[fam]["cases"].items():
            assert got[fam]["cases"][name] == case, (fam, name, REMAKE)


def test_the_tick_sends_its_switch_and_only_then(tmp_path):
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_KEEP=False)
    assert "--slider_by_epoch" not in cmd and "--slider_bank_keep" not in cmd
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BY_EPOCH=True, FAMILY_SLIDER_KEEP=False)
    assert "--slider_by_epoch" in cmd and cmd[cmd.index("--slider_bank_keep") + 1] == "0"
    _p, cmd, _desc, _v = _plan(tmp_path, "acestep15xl", FAMILY_SLIDER_BY_EPOCH=True)     # songs have no tick
    assert "--slider_by_epoch" not in cmd and "--slider_bank_keep" not in cmd


def test_off_a_prompt_that_starts_with_a_dash_still_reaches_the_trainer(tmp_path):
    from fizgig.families.train import _anchor_lines, _prompt_texts
    _p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_BASE="-a photo of a man,",
                               FAMILY_SLIDER_ANCHORS="-a dog")
    i, j = cmd.index("--slider_prompts"), cmd.index("--slider_anchors")
    a = setup_parser().parse_args(ARGS + cmd[i:i + 4] + cmd[j:j + 2])       # a space before each: values, not options
    assert _prompt_texts(a.slider_prompts) == ["-a photo of a man,", "-a photo of a man, smiling",
                                               "-a photo of a man, frowning"]
    assert _anchor_lines(a.slider_anchors) == ["-a dog"]


def test_off_start_refuses_what_only_the_tick_trains(tmp_path):
    p, cmd, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling\n\nlaughing", FAMILY_SLIDER_GUIDANCE="3:10:2")
    assert cmd is None and p.problems == [LINES, SPLIT]           # in place of the plain slider's "must be a number above 0"
    assert v["FAMILY_SLIDER_POS"] == "smiling\n\nlaughing"        # kept as typed: the tick on trains them
    p, cmd, _desc, _v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @1-4\nlaughing @5-",
                              FAMILY_SLIDER_GUIDANCE="3:10:2", FAMILY_SLIDER_BY_EPOCH=True)
    assert not p.problems and "--slider_line" in cmd and cmd[cmd.index("--slider_guidance") + 1] == "3:10:2"
    assert launch.slider_off_problems(desc, {"base": "a,\nb,"}, "3") == [
        'Slider: "What the picture is" has 2 lines - turn on "Change the prompts during the run (advanced)" for '
        'lines with their own epochs, or keep one line']
    # songs have no tick: the plain slider's check, and several lines go as typed, as before
    p, _cmd, _desc, _v = _plan(tmp_path, "acestep15xl", FAMILY_SLIDER_GUIDANCE="3:10:2")
    assert p.problems == ["Slider push strength must be a number above 0"]
    p, _cmd, _desc, _v = _plan(tmp_path, "acestep15xl", FAMILY_SLIDER_POS="loud\nfast")
    assert not p.problems


def test_off_a_line_that_ends_in_epochs_trains_as_typed_with_a_note(tmp_path):
    p, cmd, desc, v = _plan(tmp_path, "krea2", FAMILY_SLIDER_POS="smiling @5-8",
                            FAMILY_SLIDER_ANCHORS="a photo of a man @-4\n\na dog @2\na cat")
    i = cmd.index("--slider_prompts")
    assert cmd[i + 2] == "a photo of a man, smiling @5-8"                       # as before
    assert cmd[cmd.index("--slider_anchors") + 1:cmd.index("--slider_anchor_weight")] == [
        "a photo of a man @-4", "a dog @2", "a cat"]
    st = form.state(desc, v, "")
    tail = ('an @ and the epochs after it at the end of a line are part of its words, and the line trains on every '
            'epoch ({}). Tick it to give lines epochs of their own.')
    assert st["notes"]["FAMILY_SLIDER_POS"] == ('With "Change the prompts during the run (advanced)" off, '
                                                + tail.format('"@5-8"'))
    assert st["notes"]["FAMILY_SLIDER_ANCHORS"] == ('With "Change the prompts during the run (advanced)" off, '
                                                    + tail.format('line 1: "@-4"; line 3: "@2"'))
    assert "FAMILY_SLIDER_BASE" not in st["notes"]
    assert not st["slider_by_epoch"] and st["slider_plan"] is None
    assert {"_slider_plan", "FAMILY_SLIDER_KEEP"} <= set(st["hidden"])
    st = form.state(desc, {**v, "FAMILY_SLIDER_BY_EPOCH": True}, "")             # on: its plan and its own notes
    assert st["slider_by_epoch"] and st["slider_plan"]
    assert st["notes"]["FAMILY_SLIDER_POS"].startswith("No line is on at epochs 1–4")
    assert not {"_slider_plan", "FAMILY_SLIDER_KEEP"} & set(st["hidden"])
    # several lines of an end: their box waits read-only and says why (Start refuses them); songs: no note at all
    assert form.state(desc, {**v, "FAMILY_SLIDER_POS": "a @1-2\nb @3-"}, "")["notes"]["FAMILY_SLIDER_POS"].startswith(
        'Holds 2 lines from "Change the prompts during the run (advanced)"')
    songs = get_family("acestep15xl")
    assert "FAMILY_SLIDER_POS" not in form.state(songs, {**v, "FAMILY_SLIDER_SOURCE": "Prompts"}, "")["notes"]


def test_the_tick_is_the_prompt_parts_first_row_and_off_when_missing():
    desc = get_family("krea2")
    rows = next(s for s in form.form(desc)["sections"] if s["id"] == "slider")["rows"]
    keys = [r["key"] for r in rows]
    tick = rows[keys.index("FAMILY_SLIDER_BY_EPOCH")]
    assert keys.index("FAMILY_SLIDER_BY_EPOCH") + 1 == keys.index("FAMILY_SLIDER_BASE")
    assert tick["kind"] == "check" and tick["default"] is False
    assert tick["label"] == launch.BY_EPOCH == "Change the prompts during the run (advanced)"
    base = rows[keys.index("FAMILY_SLIDER_BASE")]                # the plain slider's box, its lines look laid over it when on
    assert base["kind"] == "entry" and base["by_epoch"]["lines"] and base["by_epoch"]["autofill"]
    pairs = form.state(desc, {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Photo pairs"}, "")
    assert "FAMILY_SLIDER_BY_EPOCH" in pairs["hidden"]
    songs = get_family("acestep15xl")
    assert all("FAMILY_SLIDER_BY_EPOCH" != r["key"] and "by_epoch" not in r
               for s in form.form(songs)["sections"] for r in s["rows"])
    v = {**form.DEFAULTS}
    v.pop("FAMILY_SLIDER_BY_EPOCH")                             # an older preset or run: off
    assert not launch.slider_by_epoch(desc, v) and form.DEFAULTS["FAMILY_SLIDER_BY_EPOCH"] is False
    assert launch.slider_by_epoch(desc, {"FAMILY_SLIDER_BY_EPOCH": True})
    assert not launch.slider_by_epoch(songs, {"FAMILY_SLIDER_BY_EPOCH": True})


RUN = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
       "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling",
       "FAMILY_SLIDER_NEG": "frowning"}


def test_off_apply_refuses_as_start_does_and_keeps_the_boxes_as_typed(tmp_path, monkeypatch):
    from fizgig.web import server

    class Running:
        poll = staticmethod(lambda: None)
    for k, x in {"proc": Running(), "family": "krea2", "values": dict(RUN), "output_dir": str(tmp_path),
                 "history_id": None, "progress": {"epoch": 2}}.items():
        monkeypatch.setitem(server._run, k, x)

    def apply(**body):                       # the Run card's Apply at the next epoch, without a socket
        data = json.dumps({"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "3", **body}).encode()
        h = server.Handler.__new__(server.Handler)
        h.path, h.rfile = "/api/run/slider_prompts", io.BytesIO(data)
        h.headers = {"Host": "127.0.0.1", "X-Fizgig-Token": server.TOKEN, "Content-Length": str(len(data))}
        said = []
        h.json = lambda d, code=200, extra=(): said.append((code, d))
        h._post()
        return said[0]

    assert apply(pos="smiling\nlaughing") == (400, {"error": LINES})
    assert apply(guidance="3:10:2") == (400, {"error": SPLIT})
    assert apply(guidance="abc") == (400, {"error": "Push strength must be a number above 0."})    # the plain slider's words
    assert not (tmp_path / server.SLIDER_LIVE).exists()
    assert apply(guidance=" 2.5 ", anchors="a dog @4-\na cat @1-3") == (200, {"ok": True})
    live = json.loads((tmp_path / server.SLIDER_LIVE).read_text(encoding="utf-8"))
    assert live["guidance"] == "2.5" and live["anchors"] == ["a dog @4-", "a cat @1-3"]     # as typed, not sorted
    assert server._run["values"]["FAMILY_SLIDER_ANCHORS"] == "a dog @4-\na cat @1-3"
    out = tmp_path / "another run"                     # nothing waiting there: what the epoch trains on, if shown
    out.mkdir()
    run = {"family": "krea2", "values": {**RUN, "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-"},
           "output_dir": str(out), "progress": {"epoch": 3}}
    st = server._slider_live_status(run)
    assert st["by_epoch"] is False and st["now"] is None and not st["waiting"]   # plain boxes, no "trains on now"
    st = server._slider_live_status({**run, "values": {**run["values"], "FAMILY_SLIDER_BY_EPOCH": True}})
    assert st["by_epoch"] is True and st["now"]["pos"] == "a man, smiling"


def test_the_trainer_takes_lines_and_a_push_by_epoch_only_with_its_switch(tmp_path):
    a = setup_parser().parse_args(ARGS)
    assert a.slider_by_epoch is False and a.slider_bank_keep == 0 and a.slider_guidance == "3"    # nothing kept
    a = setup_parser().parse_args(ARGS + ["--slider_by_epoch", "--slider_bank_keep", "32",
                                          "--slider_line", "pos", "smiling @1-4"])
    assert a.slider_by_epoch is True and a.slider_bank_keep == 32 and a.slider_line == [["pos", "smiling @1-4"]]
    live, boxes = tmp_path / ".slider_prompts.json", {"base": "a man,", "pos": "smiling", "neg": "frowning"}
    for guidance, off, on in (("3:10:2", None, "3:10:2"), (" 2.5 ", "2.5", "2.5"), (3, "3", "3"), (0, None, None)):
        live.write_text(json.dumps({**boxes, "guidance": guidance}), encoding="utf-8")
        got_off, got_on = _read_slider_live(str(tmp_path)), _read_slider_live(str(tmp_path), by_epoch=True)
        assert (got_off and got_off["guidance"]) == off and (got_on and got_on["guidance"]) == on, guidance
    live.write_text(json.dumps({**boxes, "guidance": 3, "anchors": ["a dog @4-", "a cat @1-3"]}), encoding="utf-8")
    new = _read_slider_live(str(tmp_path))
    assert _request_boxes(new, False)["anchors"] == "a dog @4-\na cat @1-3"      # without it: as it came
    assert launch.slider_at(_request_boxes(new, False), 1, False)[3] == ["a dog @4-", "a cat @1-3"]


def test_off_a_box_with_lines_stays_editable_and_says_why(tmp_path):
    # lines typed with the tick on, then off: the box stays editable (the page shows them as lines, app.js renderRow),
    # and says Start trains one line in each box, as the original slider does
    p, _cmd, desc, v = _plan(tmp_path, "moss_image", FAMILY_SLIDER_POS="smiling @1-5\nlaughing @6-",
                             FAMILY_SLIDER_GUIDANCE="3:10:2")
    st = form.state(desc, v, "")
    assert not {"FAMILY_SLIDER_BASE", "FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG"} & set(st["disabled"])
    assert st["notes"]["FAMILY_SLIDER_POS"] == (
        'Holds 2 lines from "Change the prompts during the run (advanced)". With it off, each box trains one line, as '
        'the original slider does: keep one line here, or tick "Change the prompts during the run (advanced)" to use '
        'them all.')
    assert "needs \"Change the prompts during the run (advanced)\"" in st["notes"]["FAMILY_SLIDER_GUIDANCE"]
    assert LINES in p.problems                                          # Start: one line, as the original
    st = form.state(desc, {**v, "FAMILY_SLIDER_BY_EPOCH": True}, "")     # on: its lines
    assert "FAMILY_SLIDER_POS" not in st["disabled"]