"""MiniMax H3 Image's Likeness optimization: its default schedule (run six, 12 Nov 2026 - the Before share from 16.67
climbing to 50, alternating with 75, hinge at epoch 6, 75 held after) and every box of its Advanced section."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.minimax_image.driver import MiniMaxH3ImageDriver  # noqa: E402


def _share(opts, epoch):
    d = MiniMaxH3ImageDriver.__new__(MiniMaxH3ImageDriver)
    d.options = opts
    d.epoch = epoch
    return d._clean_share()


def test_default_schedule_is_run_six():
    on = {"likeness_opt": "1"}                                       # the CLI's plain tick: the whole default
    assert [_share(on, e) for e in range(1, 11)] == [16.67, 75, 33.34, 75, 50, 75, 75, 75, 75, 75]


def test_the_form_sends_run_six():
    from fizgig.families import minimax_image as M
    preset = dict(M.MINIMAX_IMAGE.presets)["✨ MiniMax H3 Image Standard (rank 32, full model)"]
    toks = {"likeness_opt": "1", "likeness_before": preset["MINIMAX_IMAGE_LIKENESS_BEFORE"],
            "likeness_before_end": preset["MINIMAX_IMAGE_LIKENESS_BEFORE_END"],
            "likeness_hinge": preset["MINIMAX_IMAGE_LIKENESS_HINGE"], "likeness_after": preset["MINIMAX_IMAGE_LIKENESS_AFTER"],
            "likeness_alternate": "before"}
    assert [_share(toks, e) for e in range(1, 9)] == [_share({"likeness_opt": "1"}, e) for e in range(1, 9)]
    keys = {o.key: o for o in M.MINIMAX_IMAGE.options}
    assert keys["MINIMAX_IMAGE_LIKENESS_ALTERNATE"].choice_labels()[0] == M.LIKENESS_DEFAULTS["ALTERNATE"]
    for k, v in M.LIKENESS_DEFAULTS.items():
        if k != "ALTERNATE":
            assert keys["MINIMAX_IMAGE_LIKENESS_" + k].default == v


def test_boxes():
    o = {"likeness_opt": "1", "likeness_before": "80", "likeness_after": "40", "likeness_hinge": "5",
         "likeness_alternate": "after"}
    assert [_share(o, e) for e in range(1, 10)] == [80] * 5 + [40, 80, 40, 80]
    blank = {"likeness_opt": "1", "likeness_before": "", "likeness_after": "x", "likeness_hinge": ""}
    assert [_share(blank, e) for e in (1, 2, 6, 7)] == [16.67, 75, 75, 75]       # blanks fall back to the defaults
    o = {"likeness_opt": "1", "likeness_before": "40", "likeness_alternate": "off"}
    assert _share(o, 3) == 40                                        # a schedule given: no default climb


def test_alternate_before_the_hinge():
    o = {"likeness_opt": "1", "likeness_before": "50", "likeness_after": "75", "likeness_hinge": "6",
         "likeness_alternate": "before"}
    assert [_share(o, e) for e in range(1, 10)] == [50, 75, 50, 75, 50, 75, 75, 75, 75]


def test_alternating_off():
    o = {"likeness_opt": "1", "likeness_before": "50", "likeness_after": "75", "likeness_hinge": "3",
         "likeness_alternate": "off"}
    assert [_share(o, e) for e in range(1, 7)] == [50, 50, 50, 75, 75, 75]


def test_slide():
    o = {"likeness_opt": "1", "likeness_before": "50", "likeness_after": "75", "likeness_hinge": "8",
         "likeness_slide": "5", "likeness_alternate": "off"}
    assert [_share(o, e) for e in range(7, 16)] == [50, 50, 55, 60, 65, 70, 75, 75, 75]
    o["likeness_alternate"] = "after"          # the After side's alternating starts once the slide ends
    assert [_share(o, e) for e in range(13, 17)] == [75, 75, 50, 75]


def test_before_rises_to():
    o = {"likeness_opt": "1", "likeness_before": "16.67", "likeness_before_end": "50", "likeness_after": "75",
         "likeness_hinge": "8", "likeness_alternate": "before"}
    got = [_share(o, e) for e in range(1, 11)]
    assert got[1::2][:4] == [75, 75, 75, 75] and got[8:] == [75, 75]          # the 75 epochs and the hold untouched
    assert [round(x, 1) for x in got[0:8:2]] == [16.7, 27.8, 38.9, 50]         # its own epochs climb, the last at 50
    o.update(likeness_alternate="off", likeness_before="20", likeness_before_end="40", likeness_hinge="5")
    assert [_share(o, e) for e in range(1, 7)] == [20, 25, 30, 35, 40, 75]     # not alternating: every epoch climbs
    o.update(likeness_before_end="")
    assert _share(o, 3) == 20                                                  # blank: no climb
    o.update(likeness_before="50", likeness_after="50", likeness_before_end="70", likeness_alternate="before")
    assert _share(o, 2) == 50                                                  # an After epoch never climbs


def test_boxes_only_sent_while_ticked_and_behind_advanced():
    from fizgig.families.form import _option_row
    from fizgig.families.minimax_image import MINIMAX_IMAGE as d
    keys = {o.key: o for o in d.options}
    assert "advanced_of" not in _option_row(keys["MINIMAX_IMAGE_LIKENESS_OPT"])
    for k in ("MINIMAX_IMAGE_LIKENESS_BEFORE", "MINIMAX_IMAGE_LIKENESS_HINGE", "MINIMAX_IMAGE_LIKENESS_AFTER",
              "MINIMAX_IMAGE_LIKENESS_ALTERNATE", "MINIMAX_IMAGE_LIKENESS_SLIDE", "MINIMAX_IMAGE_LIKENESS_BEFORE_END"):
        assert keys[k].requires == "MINIMAX_IMAGE_LIKENESS_OPT"
        assert _option_row(keys[k])["advanced_of"] == "MINIMAX_IMAGE_LIKENESS_OPT"   # inside the tick's row


def test_off_is_the_model_default_and_cli_share_still_works():
    assert _share({}, 3) is None
    assert _share({"lownoise_pct": "40"}, 3) == 40.0
