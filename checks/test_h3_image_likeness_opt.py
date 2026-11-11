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


def test_off_is_the_model_default_and_cli_share_still_works():
    assert _share({}, 3) is None
    assert _share({"lownoise_pct": "40"}, 3) == 40.0
