"""What a prompt slider trains on from the next epoch, decided at each epoch boundary with nothing encoded or rendered
(train._slider_next; the trainer's slider_live does the rest): with the by-epoch switch off, the Run card's
request taken as before; on, lines with their own epochs, a push strength by epoch and the previews on the first
lines. And a request the trainer couldn't take, or the run's own lines it couldn't switch to, as the Run card shows
them."""
import json
import os
import re
import shutil
import subprocess

import pytest

from fizgig.families import launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import (SLIDER_TRIES, _first_lines, _line_boxes, _lines_failure, _lines_key,
                                   _read_slider_live, _slider_failure, _slider_next, _slider_tries)

KREA = get_family("krea2")
MAN = ["a photo of a man,", "a photo of a man, smiling", "a photo of a man, frowning"]
LINES = {"base": "a photo of a woman, @-2\na photo of a woman in a red coat, @3-", "pos": "smiling @1-5\nlaughing @3-",
         "neg": "frowning", "anchors": "a photo of a man @-4\na photo of a dog @5-"}
COAT = "a photo of a woman in a red coat,"
APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _one_line(plan="3"):
    """A prompt slider started on one line each (--slider_prompts), as the trainer holds it."""
    return {"boxes": None, "plan": plan, "prompts": list(MAN), "anchors": [], "texts": None, "preview": MAN[:1]}


def _lines(boxes, plan="3"):
    """A prompt slider started on lines with their own epochs (--slider_line), as the trainer holds it at epoch 1."""
    boxes = _line_boxes([("anchor" if k == "anchors" else k, x) for k, t in boxes.items() for x in t.splitlines()])
    base, pos, neg, anchors = launch.slider_at(boxes, 1)
    return {"boxes": boxes, "plan": plan, "prompts": [base, f"{base} {pos}", f"{base} {neg}"], "anchors": anchors,
            "texts": [base, pos, neg], "preview": _first_lines(boxes)}


def _moved(cur, nxt):
    """The slider once slider_live trained on `nxt` (all encoded and rendered)."""
    keep = ("boxes", "plan") if nxt["push_only"] else ("boxes", "plan", "prompts", "anchors", "texts", "preview")
    return {**cur, **{k: nxt[k] for k in keep}}


def _ask(tmp_path, by_epoch, **request):
    """The Run card's request as the trainer reads it (8 epochs)."""
    (tmp_path / ".slider_prompts.json").write_text(json.dumps({"guidance": "3", **request}), encoding="utf-8")
    return _read_slider_live(str(tmp_path), 8, by_epoch)


def test_off_one_line_each_as_before(tmp_path):
    cur = _one_line()
    nxt = _slider_next(None, cur, 2, False, KREA)          # no request: the strength alone, one number all run
    assert nxt["push_only"] and nxt["took"] is None and nxt["prompts"] == MAN and nxt["guidance"] == 3.0
    new = _ask(tmp_path, False, base="a photo of a woman,", pos="laughing @5-8", neg="crying", guidance="2.5",
               anchors=["a photo of a dog", "a photo of a cat"])
    nxt = _slider_next(new, cur, 2, False, KREA)           # a request: its prompts as typed, as before
    assert nxt["took"] == new and not nxt["push_only"] and nxt["base_changed"] and not nxt["scheduled"]
    assert nxt["prompts"] == ["a photo of a woman,", "a photo of a woman, laughing @5-8", "a photo of a woman, crying"]
    assert nxt["anchors"] == ["a photo of a dog", "a photo of a cat"] and nxt["guidance"] == 2.5
    assert nxt["preview"] == ["a photo of a woman,", "a photo of a dog"]       # the plain slider's: the base and first anchor
    cur = _moved(cur, nxt)
    assert _slider_next(new, cur, 3, False, KREA)["took"] is None      # taken: the same file asks nothing new
    nxt = _slider_next({**new, "guidance": "2"}, cur, 3, False, KREA)   # a new number alone: nothing encoded
    assert nxt["took"] and nxt["push_only"] and nxt["guidance"] == 2.0 and nxt["prompts"] == cur["prompts"]


def test_off_a_request_the_same_as_now_asks_nothing(tmp_path):
    # as before: the same three prompts, push strength and anchors as it trains on now - nothing taken, so slider_live
    # writes nothing for the Run card (with the tick on it is taken: test_on_a_request_the_same_as_now_is_taken_...)
    cur = {**_one_line(), "anchors": ["a photo of a dog"], "preview": [MAN[0], "a photo of a dog"]}
    same = {"base": " a photo of a man, ", "pos": "smiling", "neg": "frowning", "anchors": ["a photo of a dog", " "]}
    for guidance in ("3", " 3.0 ", 3):
        nxt = _slider_next(_ask(tmp_path, False, **same, guidance=guidance), cur, 2, False, KREA)
        assert nxt["took"] is None and nxt["push_only"] and nxt["prompts"] == MAN and nxt["guidance"] == 3.0, guidance
        assert nxt["anchors"] == ["a photo of a dog"] and nxt["boxes"] is None
    for change in ({"guidance": "2.5"}, {"pos": "grinning"}, {"anchors": []}, {"anchors": ["a photo of a cat"]}):
        new = _ask(tmp_path, False, **{**same, **change})          # anything else asked: taken, as before
        assert _slider_next(new, cur, 2, False, KREA)["took"] == new, change


def test_on_the_lines_move_on_by_themselves():
    cur = _lines(LINES)
    assert cur["prompts"][1] == "a photo of a woman, smiling" and cur["anchors"] == ["a photo of a man"]
    nxt = _slider_next(None, cur, 1, True, KREA)            # epoch 2: the same lines
    assert nxt["push_only"] and nxt["took"] is None
    nxt = _slider_next(None, cur, 2, True, KREA)            # epoch 3: a new picture and a second +1 line
    assert not nxt["push_only"] and nxt["took"] is None and nxt["scheduled"] and nxt["base_changed"]
    assert nxt["prompts"] == [COAT, f"{COAT} smiling, laughing", f"{COAT} frowning"]
    cur = _moved(cur, nxt)
    assert _slider_next(None, cur, 3, True, KREA)["push_only"]       # epoch 4: as epoch 3
    nxt = _slider_next(None, cur, 4, True, KREA)            # epoch 5: another anchor, the same picture
    assert not nxt["push_only"] and not nxt["base_changed"] and nxt["anchors"] == ["a photo of a dog"]
    cur = _moved(cur, nxt)
    nxt = _slider_next(None, cur, 5, True, KREA)            # epoch 6: smiling is over
    assert nxt["prompts"][1] == f"{COAT} laughing" and not nxt["base_changed"]


def test_on_a_request_with_new_lines(tmp_path):
    cur = _moved(_lines(LINES), _slider_next(None, _lines(LINES), 2, True, KREA))     # training epoch 3's lines
    new = _ask(tmp_path, True, base=LINES["base"], pos="grinning @4-\nsmiling @1-5", neg="frowning",
               anchors=LINES["anchors"].splitlines())
    nxt = _slider_next(new, cur, 3, True, KREA)
    assert nxt["took"] == new and nxt["boxes"]["pos"] == "smiling @1-5\ngrinning @4-"   # in the order they join in
    assert not nxt["push_only"] and not nxt["base_changed"] and nxt["prompts"][1] == f"{COAT} smiling, grinning"
    assert _slider_next(new, _moved(cur, nxt), 4, True, KREA)["took"] is None


def test_on_a_request_the_same_as_now_is_taken_with_nothing_encoded(tmp_path):
    new = _ask(tmp_path, True, base="a photo of a man,", pos="smiling", neg="frowning")
    nxt = _slider_next(new, _one_line(), 2, True, KREA)     # the Run card's own values: taken, the card stops waiting
    assert nxt["took"] == new and nxt["push_only"] and nxt["prompts"] == MAN
    # lines that come in later only: the same prompts now, and the request taken all the same
    later = _ask(tmp_path, True, base=LINES["base"], pos="smiling @1-5\nlaughing @3-6\ngrinning @7-", neg="frowning",
                 anchors=LINES["anchors"].splitlines())
    nxt = _slider_next(later, _lines(LINES), 1, True, KREA)
    assert nxt["took"] == later and nxt["push_only"] and nxt["boxes"]["pos"].endswith("grinning @7-")


def test_on_a_push_by_epoch_changes_alone(tmp_path):
    cur = _one_line("3:4:2")                                # a Split: 3 for the first 4 epochs, then 2
    assert [_slider_next(None, cur, d, True, KREA)["guidance"] for d in (3, 4)] == [3.0, 2.0]
    assert all(_slider_next(None, cur, d, True, KREA)["push_only"] for d in (3, 4))
    lines = _lines(LINES, "3:3:2")                          # with lines too: epoch 4 trains epoch 3's lines, at 2
    lines = _moved(lines, _slider_next(None, lines, 2, True, KREA))
    nxt = _slider_next(None, lines, 3, True, KREA)
    assert nxt["push_only"] and nxt["took"] is None and nxt["guidance"] == 2.0
    new = _ask(tmp_path, True, base="a photo of a man,", pos="smiling", neg="frowning", guidance="3:6:1:8")
    nxt = _slider_next(new, _one_line(), 6, True, KREA)     # a Slide from the Run card: 2 at epoch 7
    assert nxt["took"] == new and nxt["push_only"] and nxt["plan"] == "3:6:1:8" and nxt["guidance"] == 2.0


def test_on_a_box_with_no_line_on_keeps_the_one_it_had():
    cur = _lines({"base": "a man, @1-2\na woman, @3-", "pos": "smiling @1-2", "neg": "frowning", "anchors": ""})
    nxt = _slider_next(None, cur, 2, True, KREA)            # epoch 3: a new picture, no +1 line on
    assert nxt["texts"] == ["a woman,", "smiling", "frowning"] and nxt["base_changed"]
    assert nxt["prompts"] == ["a woman,", "a woman, smiling", "a woman, frowning"]


def test_on_the_previews_stay_on_the_first_lines(tmp_path):
    cur = _lines(LINES)
    for done in range(1, 7):                                # the lines move on, the previews stay all run
        nxt = _slider_next(None, cur, done, True, KREA)
        assert nxt["preview"] == ["a photo of a woman,", "a photo of a man"], done
        cur = _moved(cur, nxt)
    # a request with another first line: the previews follow it, though the epoch's prompts are the same
    new = _ask(tmp_path, True, base="a photo of a girl, @-2\n" + COAT + " @3-", pos=LINES["pos"], neg="frowning",
               anchors=LINES["anchors"].splitlines())
    nxt = _slider_next(new, cur, 6, True, KREA)
    assert nxt["preview"] == ["a photo of a girl,", "a photo of a man"] and nxt["prompts"] == cur["prompts"]
    assert not nxt["push_only"] and not nxt["base_changed"]
    one = _slider_next(_ask(tmp_path, True, base="a photo of a girl,", pos="smiling", neg="frowning",
                            anchors=["a photo of a cat"]), _one_line(), 2, True, KREA)
    assert one["preview"] == ["a photo of a girl,", "a photo of a cat"]     # one line each: the picture and an anchor


def test_a_failure_said_short_on_one_line():
    assert _slider_failure(RuntimeError("CUDA out of memory.\n  Tried to allocate 2.00 GiB")) == (
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert _slider_failure(ValueError("x" * 300)) == "ValueError: " + "x" * 160 + "…"
    assert _slider_failure(MemoryError()) == "MemoryError"


RUN = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
       "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
       "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True}


def test_the_run_card_says_a_change_it_could_not_take(tmp_path):
    from fizgig.web import server
    asked = {"base": "a man,", "pos": "smiling @1-4\ngrinning @5-", "neg": "frowning", "guidance": "3", "anchors": []}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps(asked), encoding="utf-8")
    run = {"family": "krea2", "values": {**RUN, "FAMILY_SLIDER_POS": asked["pos"]}, "output_dir": str(tmp_path),
           "progress": {"epoch": 6}, "slider_start": {k: RUN.get(k) for k in server._SLIDER_LIVE_KEYS.values()}}
    failed = {"epoch": 5, "failed": "OutOfMemoryError: CUDA out of memory.", "at": 2.0, "asked": asked, "took": None}
    applied = tmp_path / server.SLIDER_LIVE_APPLIED
    applied.write_text(json.dumps(failed), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["failed"] == "OutOfMemoryError: CUDA out of memory." and st["epoch"] == 5 and not st["waiting"]
    assert st["now"]["pos"] == "a man, laughing"            # its own lines meanwhile: none taken yet, as it started
    took = {**asked, "pos": "smiling @1-4\nbeaming @5-", "epoch": 2, "practice_rerendered": False, "at": 1.0}
    applied.write_text(json.dumps({**failed, "took": took}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["failed"] and st["epoch"] == 5 and st["now"]["pos"] == "a man, beaming"     # the last change it took
    # another Apply waits as any does (the failure was the earlier one's), and a change taken writes over it
    again = {**asked, "pos": "smiling @1-4\nlaughing @5-"}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps(again), encoding="utf-8")
    run["values"]["FAMILY_SLIDER_POS"] = again["pos"]
    st = server._slider_live_status(run)
    assert st["failed"] is None and st["waiting"] and st["now"]["pos"] == "a man, beaming"
    applied.write_text(json.dumps({**again, "epoch": 6, "practice_rerendered": False, "at": 3.0}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["failed"] is None and not st["waiting"] and st["epoch"] == 6 and st["now"]["pos"] == "a man, laughing"


def test_the_runs_own_lines_it_cant_switch_to_are_tried_twice_then_left(tmp_path):
    # no Run card request: a switch to the run's own lines that fails is counted for the lines it switches to - tried
    # again at the next boundary, then not again (the current lines train on); other lines, later, are tried anew
    cur = _lines(LINES)
    coat = _lines_key(_slider_next(None, cur, 2, True, KREA))       # epoch 3: a new picture and a second +1 line
    assert SLIDER_TRIES == 2 and _slider_tries(None, coat) == 0
    assert _lines_key(_slider_next(None, cur, 3, True, KREA)) == coat      # it failed: epoch 4 asks the same lines
    assert _slider_tries({"at": coat, "n": 1}, coat) == 1 < SLIDER_TRIES    # ... tried once more
    assert _slider_tries({"at": coat, "n": 2}, coat) == SLIDER_TRIES        # ... then not again
    dog = _lines_key(_slider_next(None, cur, 4, True, KREA))        # epoch 5: other lines (another anchor)
    assert dog != coat and _slider_tries({"at": coat, "n": 2}, dog) == 0
    said = _lines_failure(5, "OutOfMemoryError: CUDA out of memory.")
    assert said == ("Couldn't switch to the lines for epoch 5 (OutOfMemoryError: CUDA out of memory.) - training goes "
                    "on with the current lines.")
    # the Run card says so, beside what it says of a request, and the epoch shows the lines it trains on instead
    from fizgig.web import server
    run = {"family": "krea2", "values": dict(RUN), "output_dir": str(tmp_path), "progress": {"epoch": 6}}
    st = server._slider_live_status(run)
    assert st["lines_failed"] is None and st["now"]["pos"] == "a man, laughing"
    kept = {"epoch": 5, "said": said, "pos": "a man, smiling", "neg": "a man, frowning", "anchors": []}
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({"lines_failed": kept}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["lines_failed"] == said and st["now"]["pos"] == "a man, smiling" and not st["waiting"]
    assert st["epoch"] is None and st["failed"] is None              # no request taken or failed: nothing else said


def test_the_card_says_it_in_plain_words():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    fn = re.search(r"function sliderLiveDraw\(sl\) \{.*?\n\}\n", open(APP, encoding="utf-8").read(), re.S).group(0)
    status = {"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "3", "anchors": "", "now": None,
              "by_epoch": True}
    cases = [{**status, "epoch": 5, "waiting": False, "failed": "OutOfMemoryError: CUDA out of memory."},
             {**status, "epoch": 2, "waiting": True, "failed": None},
             {**status, "epoch": 6, "waiting": False, "failed": None}]
    js = ("const nodes = {}; const $ = (id) => nodes[id] || (nodes[id] = {style: {}, value: '', textContent: ''});"
          "const S = {sl: {shown: null, edited: false}}; const SL_FIELDS = {base: 'slBase'};"
          "const sliderLiveLook = () => false;\n" + fn + f"const cases = {json.dumps(cases)};"
          "console.log(JSON.stringify(cases.map((sl) => (sliderLiveDraw(sl), $('slSay').textContent))));")
    said = json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8"))
    assert said == ["Could not take them at the end of epoch 5 (OutOfMemoryError: CUDA out of memory.) - training goes "
                    "on with its own lines; they're tried again at the next epoch.",
                    "Waiting for the end of this epoch…", "Training on these since epoch 7."]


def test_off_the_run_on_show_keeps_the_boxes_it_started_with():
    # the run on show (app.js sliderLiveFollow): with the tick on, its Slider boxes follow an Apply taken; off, as
    # before, they stay as the run started
    if not shutil.which("node"):
        pytest.skip("node not installed")
    fn = re.search(r"const SL_KEYS = .*?\nfunction sliderLiveFollow\(\) \{.*?\n\}\n",
                   open(APP, encoding="utf-8").read(), re.S).group(0)
    js = ("const S = {}; const ctxKind = () => 'run', ctxIsLiveRun = () => true, syncInputs = () => {}, "
          "refreshStateSoon = () => {};\n" + fn + "console.log(JSON.stringify([false, true].map((by_epoch) => {"
          "Object.assign(S, {ctxBusy: false, ctx: {}, values: {FAMILY_SLIDER_POS: 'smiling'}, "
          "run: {values: {FAMILY_SLIDER_POS: 'grinning'}, slider_live: {by_epoch}}});"
          "sliderLiveFollow(); return S.values.FAMILY_SLIDER_POS; })));")
    assert json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8")) == ["smiling", "grinning"]
