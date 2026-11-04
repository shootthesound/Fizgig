"""The loss graph's LR line: logged with 6 digits, old 4-digit logs still read, and the JS scale runs 0..start."""
import json, os, re, shutil, subprocess, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.training.step_log import StepLog, read, loss_path  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def test_lr_logged_with_six_digits(tmp_path):
    lg = StepLog(str(tmp_path), "x")
    lg.add(1, 1, 0.1, lr=9.999876e-05)
    lg.flush()
    rec = json.loads(open(loss_path(str(tmp_path), "x")).readline())
    assert rec["lr"] == 9.99988e-05


def test_old_four_digit_log_reads(tmp_path):
    p = loss_path(str(tmp_path), "old")
    with open(p, "w") as f:
        f.write('{"s": 1, "e": 1, "l": 0.08, "lr": 0.0001}\n{"s": 2, "e": 1, "l": 0.07, "lr": 9.999e-05}\n'
                '{"s": 3, "e": 1, "l": 0.07}\n')
    d = read(p)
    assert len(d["points"]) == 3


def _lr_axis(values):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"function lrAxis\(lv\) \{.*?\n\}\n", src, re.S).group(0)
    js = fn + f"const a = lrAxis({json.dumps(values)}); console.log(JSON.stringify({{top: a.top, climbs: a.climbs, pct: a.pct({values[-1]})}}));"
    return json.loads(subprocess.check_output(["node", "-e", js], text=True))


def test_cosine_starts_at_top():
    a = _lr_axis([1e-4, 9.9999e-5, 9.99e-5, 5e-5])
    assert a["top"] == 1e-4 and not a["climbs"] and a["pct"] == "50.0"


def test_warmup_peak_is_top():
    a = _lr_axis([1e-5, 5e-5, 1e-4, 9.9e-5])
    assert 1e-4 <= a["top"] <= 1.1e-4 + 1e-12   # a climb (warmup) gets the same small headroom


def test_prodigy_growing_gets_headroom():
    a = _lr_axis([1e-6, 1e-5, 1e-4])
    assert a["climbs"] and abs(a["top"] - 1.1e-4) < 1e-12


def test_tiny_decay_reads_near_100():
    assert _lr_axis([1e-4, 9.999e-5])["pct"] == "99.9"


def test_prodigy_rise_then_cosine_not_pinned_to_one():
    import math
    rise = [1e-6 * (1.5 ** i) for i in range(12)]          # d*lr grows ~1e-6 -> ~8.6e-5
    peak = rise[-1]
    decay = [peak * 0.5 * (1 + math.cos(math.pi * i / 40)) for i in range(1, 30)]
    a = _lr_axis(rise + decay)
    assert a["climbs"] and peak * 1.09 < a["top"] < peak * 1.16 and a["top"] < 1e-3


def _now_label(values):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"function lrNowLabel\(x\) \{.*?\n\}\n", src, re.S).group(0)
    js = fn + f"console.log(JSON.stringify(lrNowLabel({json.dumps(values)}.at(-1))));"
    return json.loads(subprocess.check_output(["node", "-e", js], text=True))


def test_right_axis_top_label_is_the_rate_now():
    assert _now_label([1e-4, 9.0e-5, 6.55e-5]) == "6.55e-5"          # a decaying (cosine) run: the latest value


def test_right_axis_top_label_follows_a_rising_rate():                  # Prodigy's d*lr climbing
    assert _now_label([1e-6, 1e-5, 2.31e-4]) == "2.31e-4"


def test_no_lr_readout_strip_and_the_top_label_is_live():
    src = open(APP, encoding="utf-8").read()
    css = open(os.path.join(os.path.dirname(APP), "style.css"), encoding="utf-8").read()
    for gone in ("lg-lr-tag", "lg-lr-readout", "lrReadout", "Learning rate now", "% of start"):
        assert gone not in src and gone not in css
    draw = src[src.index("function lossGraph("):]
    assert "lrNowLabel(lp[3]), T + 14" in draw                     # the top of the scale shows the latest point
    assert "ax.top.toExponential" not in draw and "v.toExponential(1)" not in draw
    assert re.search(r"const L = 54, R = [^,]+, T = 10, B = 26;", draw)   # no extra strip on top
