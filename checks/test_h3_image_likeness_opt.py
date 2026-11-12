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


def test_three_boxes():
    o = {"likeness_opt": "1", "likeness_first": "80", "likeness_alt": "40", "likeness_hinge": "5"}
    assert [_share(o, e) for e in range(1, 10)] == [80] * 5 + [40, 80, 40, 80]
    blank = {"likeness_opt": "1", "likeness_first": "", "likeness_alt": "x", "likeness_hinge": ""}
    assert [_share(blank, e) for e in (8, 9, 10)] == [75, 50, 75]


def test_alternate_before_the_hinge():
    o = {"likeness_opt": "1", "likeness_hinge": "6", "likeness_when": "before"}
    assert [_share(o, e) for e in range(1, 10)] == [75, 50, 75, 50, 75, 50, 75, 75, 75]


def test_boxes_only_sent_while_ticked():
    from fizgig.families.minimax_image import MINIMAX_IMAGE as d
    keys = {o.key: o for o in d.options}
    for k in ("MINIMAX_IMAGE_LIKENESS_FIRST", "MINIMAX_IMAGE_LIKENESS_ALT", "MINIMAX_IMAGE_LIKENESS_HINGE",
              "MINIMAX_IMAGE_LIKENESS_WHEN"):
        assert keys[k].requires == "MINIMAX_IMAGE_LIKENESS_OPT"


def test_off_is_the_model_default_and_cli_share_still_works():
    assert _share({}, 3) is None
    assert _share({"lownoise_pct": "40"}, 3) == 40.0
