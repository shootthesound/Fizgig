"""The Run card's Apply at the next epoch (POST /api/run/slider_prompts) refuses a prompt slider's lines as Start
does, with Start's words (launch.slider_line_problems): an epoch at which a box of the picture or an end has no line
on. Lines on together, the picture's too, join into one. Only for a model that practises on pictures, with Train by
epoch (advanced) ticked; one line each trains as it always did."""
from fizgig.families import launch
from fizgig.families.registry import get as get_family, training_families

PICTURES = get_family("krea2")
SONGS = next(d for d in training_families() if d.slider_training and "photo" not in d.media)
NO_LINE = ('Slider: no line of "The +1 end adds" is on at epoch 6 - every epoch needs one (a line with no epochs is on '
           'for the whole run)')


def _boxes(base="a photo of a woman,", pos="smiling", neg="frowning", anchors=""):
    return {"base": base, "pos": pos, "neg": neg, "anchors": anchors}


def test_one_line_each_is_fine():
    assert launch.slider_line_problems(PICTURES, _boxes(), 10) == []
    assert launch.slider_line_problems(PICTURES, _boxes(pos="smiling @1-5\nlaughing @6-"), 10) == []


def test_an_epoch_with_no_line_is_refused_in_starts_words():
    assert launch.slider_line_problems(PICTURES, _boxes(pos="smiling @1-5"), 10) == [NO_LINE]
    # every 3rd epoch from 1 leaves epoch 2 without a line
    assert launch.slider_line_problems(PICTURES, _boxes(neg="frowning @1-10/3"), 10) == [
        'Slider: no line of "The -1 end adds" is on at epoch 2 - every epoch needs one (a line with no epochs is on '
        'for the whole run)']


def test_picture_lines_on_together_join_into_one():
    boxes = _boxes(base="a man, @1-5\na woman, @5-")
    assert launch.slider_line_problems(PICTURES, boxes, 10) == []
    assert launch.slider_at(boxes, 5)[0] == "a man, a woman,"
    assert launch.slider_line_problems(PICTURES, _boxes(base="a man,\nin a red coat, @5-"), 10) == []


def test_lines_every_other_epoch_take_turns():
    assert launch.slider_line_problems(PICTURES, _boxes(base="a man, @1-10/2\na woman, @2-10/2"), 10) == []
    boxes = _boxes(base="a man, @1-10/2\na woman, @2-10/2\na child, @4-4")      # a third joins the one on at epoch 4
    assert launch.slider_line_problems(PICTURES, boxes, 10) == []
    assert [launch.slider_at(boxes, e)[0] for e in (3, 4)] == ["a man,", "a woman, a child,"]


def test_a_song_model_takes_its_boxes_as_typed():
    assert launch.slider_line_problems(SONGS, _boxes(pos="smiling @1-5", base="a @1-5\nb @5-"), 10) == []


def test_apply_takes_a_song_models_boxes_as_before_and_a_picture_models_by_line():
    from fizgig.web import server
    b = {"base": "  a song,\n\n  sung softly \n", "pos": " happy ", "neg": "sad"}
    assert server._slider_live_boxes(SONGS, b) == {"base": "a song,\n\n  sung softly", "pos": "happy", "neg": "sad"}
    assert server._slider_live_boxes(PICTURES, b) == {"base": "a song,\nsung softly", "pos": "happy", "neg": "sad"}


def test_start_refuses_the_same_in_the_same_words():
    v = {"FAMILY_SLIDER": True, "FAMILY_SLIDER_SOURCE": "prompts", "FAMILY_SLIDER_GUIDANCE": "3",
         "MAX_TRAIN_EPOCHS": "10", "FAMILY_SLIDER_BASE": "a man, @1-5\na woman, @5-",
         "FAMILY_SLIDER_POS": "smiling @1-5", "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True}
    said = launch.slider_line_problems(PICTURES, launch.slider_boxes(v), 10)
    assert said == [NO_LINE]                                   # the picture lines on together at epoch 5 are fine
    assert set(said) <= set(launch.problems(PICTURES, v))
    assert not any("together" in x for x in launch.problems(PICTURES, v))


def test_the_epochs_still_to_come_can_be_checked_alone():
    # the Run card says so when only epochs already trained have no line (Apply checks the whole run, as a Pause and
    # Resume starts from these values)
    later = _boxes(pos="grinning @7-")
    assert launch.slider_line_problems(PICTURES, later, 10)                     # epochs 1 to 6: no +1 line
    assert launch.slider_line_problems(PICTURES, later, 10, first=7) == []
    assert launch.slider_overlap("a man, @1-5\na woman, @5-8", 8, first=6) is None


def test_start_and_apply_refuse_a_push_strength_in_the_same_words():
    v = {"FAMILY_SLIDER": True, "FAMILY_SLIDER_SOURCE": "prompts", "FAMILY_SLIDER_GUIDANCE": "3:4:2:2",
         "MAX_TRAIN_EPOCHS": "10", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling",
         "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True}
    said = launch.slider_push_problems("3:4:2:2", 10)
    assert said == ["Slider push strength holds until epoch 4 but slides by epoch 2, before that - give a later one"]
    assert set(said) <= set(launch.problems(PICTURES, v))
    assert not any("push" in x for x in launch.problems(PICTURES, {**v, "FAMILY_SLIDER_GUIDANCE": "3:4:2:6"}))


def test_the_run_card_shows_the_push_strength_the_epoch_trains_at(tmp_path):
    import json
    from fizgig.web import server
    v = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "20",
         "FAMILY_SLIDER_GUIDANCE": "3:10:2", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling",
         "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_BY_EPOCH": True}
    run = {"family": "krea2", "values": v, "output_dir": str(tmp_path), "progress": {"epoch": 11}}
    st = server._slider_live_status(run)               # one line each: shown for its push strength by epoch
    assert st["guidance"] == "3:10:2" and not st["waiting"]
    assert st["now"] == {"epoch": 11, "pos": "a man, smiling", "neg": "a man, frowning", "anchors": [], "push": "2",
                         "next": None}
    assert server._slider_live_status({**run, "values": {**v, "FAMILY_SLIDER_GUIDANCE": "3"}})["now"] is None
    # Apply at the next epoch, compared as text (a number's float() failed on 3:10:2): what waits trains only later
    took = {"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "3:10:2", "anchors": []}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps({**took, "guidance": "3:10:2.5"}), encoding="utf-8")
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**took, "epoch": 4}), encoding="utf-8")
    run["values"] = {**v, "FAMILY_SLIDER_GUIDANCE": "3:10:2.5"}
    st = server._slider_live_status(run)
    assert st["waiting"] and st["now"]["push"] == "2"                  # the plan it took, not the one that waits
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**took, "guidance": "3:10:2.5", "epoch": 11}),
                                                       encoding="utf-8")
    st = server._slider_live_status(run)
    assert not st["waiting"] and st["now"]["push"] == "2.5"
    # an older trainer wrote the number it took (3.0): the same as the Run card's 3
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps({**took, "guidance": "3"}), encoding="utf-8")
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**took, "guidance": 3.0, "epoch": 11}),
                                                       encoding="utf-8")
    assert not server._slider_live_status({**run, "values": {**v, "FAMILY_SLIDER_GUIDANCE": "3"}})["waiting"]


SLIDER_RUN = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
              "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,",
              "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-", "FAMILY_SLIDER_NEG": "frowning",
              "FAMILY_SLIDER_BY_EPOCH": True}


def test_apply_keeps_each_box_in_the_order_its_lines_join_in(tmp_path, monkeypatch):
    import io
    import json
    from fizgig.web import server

    class Running:
        poll = staticmethod(lambda: None)
    for k, x in {"proc": Running(), "family": "krea2", "values": dict(SLIDER_RUN), "output_dir": str(tmp_path),
                 "history_id": None, "progress": {"epoch": 2}}.items():
        monkeypatch.setitem(server._run, k, x)
    body = json.dumps({"base": "a man,", "pos": "laughing @3-\nsmiling @1-5", "neg": "frowning", "guidance": "3",
                       "anchors": "a dog @4-\na cat @1-3"}).encode()
    h = server.Handler.__new__(server.Handler)               # the Run card's Apply at the next epoch, without a socket
    h.path, h.rfile = "/api/run/slider_prompts", io.BytesIO(body)
    h.headers = {"Host": "127.0.0.1", "X-Fizgig-Token": server.TOKEN, "Content-Length": str(len(body))}
    said = []
    h.json = lambda data, code=200, extra=(): said.append((code, data))
    h._post()
    assert said == [(200, {"ok": True})]
    live = json.loads((tmp_path / server.SLIDER_LIVE).read_text(encoding="utf-8"))
    assert live["pos"] == "smiling @1-5\nlaughing @3-" and live["anchors"] == ["a cat @1-3", "a dog @4-"]
    assert server._run["values"]["FAMILY_SLIDER_POS"] == "smiling @1-5\nlaughing @3-"       # a Pause and Resume's too
    assert server._run["values"]["FAMILY_SLIDER_ANCHORS"] == "a cat @1-3\na dog @4-"


def test_while_its_first_change_waits_the_run_card_shows_what_it_started_on(tmp_path):
    import json
    from fizgig.web import server
    asked = {"base": "a man,", "pos": "smiling @1-4\ngrinning @5-", "neg": "frowning", "guidance": "3", "anchors": []}
    (tmp_path / server.SLIDER_LIVE).write_text(json.dumps(asked), encoding="utf-8")
    run = {"family": "krea2", "values": {**SLIDER_RUN, "FAMILY_SLIDER_POS": asked["pos"]}, "output_dir": str(tmp_path),
           "progress": {"epoch": 6},                     # as Start keeps them (server._start_one), before any Apply
           "slider_start": {key: SLIDER_RUN.get(key) for key in server._SLIDER_LIVE_KEYS.values()}}
    st = server._slider_live_status(run)
    assert st["waiting"] and st["pos"] == asked["pos"]          # the boxes hold the change that waits ...
    assert st["now"]["pos"] == "a man, laughing" and st["now"]["next"] is None     # ... the epoch trains on the start
    assert server._slider_live_status({**run, "slider_start": None})["now"] is None      # no record of it: not shown
    (tmp_path / server.SLIDER_LIVE_APPLIED).write_text(json.dumps({**asked, "epoch": 6}), encoding="utf-8")
    st = server._slider_live_status(run)
    assert not st["waiting"] and st["now"]["pos"] == "a man, grinning"      # taken: what it trains on from now