"""MiniMax H3 Image's Likeness optimization: 75% of steps at the clean end for epochs 1-8, then 50 and 75 in turn."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.minimax_image.driver import MiniMaxH3ImageDriver  # noqa: E402


def _share(opts, epoch):
    d = MiniMaxH3ImageDriver.__new__(MiniMaxH3ImageDriver)
    d.options = opts
    d.epoch = epoch
    return d._clean_share()


def test_schedule():
    on = {"likeness_opt": "1"}
    assert [_share(on, e) for e in range(1, 9)] == [75] * 8
    assert [_share(on, e) for e in range(9, 15)] == [50, 75, 50, 75, 50, 75]


def test_boxes():
    o = {"likeness_opt": "1", "likeness_before": "80", "likeness_after": "40", "likeness_hinge": "5"}
    assert [_share(o, e) for e in range(1, 10)] == [80] * 5 + [40, 80, 40, 80]
    blank = {"likeness_opt": "1", "likeness_before": "", "likeness_after": "x", "likeness_hinge": ""}
    assert [_share(blank, e) for e in (8, 9, 10)] == [75, 50, 75]


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


def test_boxes_only_sent_while_ticked():
    from fizgig.families.minimax_image import MINIMAX_IMAGE as d
    keys = {o.key: o for o in d.options}
    for k in ("MINIMAX_IMAGE_LIKENESS_BEFORE", "MINIMAX_IMAGE_LIKENESS_HINGE", "MINIMAX_IMAGE_LIKENESS_AFTER",
              "MINIMAX_IMAGE_LIKENESS_ALTERNATE", "MINIMAX_IMAGE_LIKENESS_SLIDE", "MINIMAX_IMAGE_LIKENESS_BEFORE_END"):
        assert keys[k].requires == "MINIMAX_IMAGE_LIKENESS_OPT"


def test_off_is_the_model_default_and_cli_share_still_works():
    assert _share({}, 3) is None
    assert _share({"lownoise_pct": "40"}, 3) == 40.0
