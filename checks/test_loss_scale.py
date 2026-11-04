"""The loss graph's left scale in the Epoch average / Running avg views: fitted to the averages and validation
(10% padding, a minimum height for a near-flat series); the default view's scale is untouched."""
import json, math, os, re, shutil, subprocess
import pytest

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _range(vals, log=False):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"function trendRange\(vals, log, minFrac = 0\.05\) \{.*?\n\}\n", src, re.S).group(0)
    js = fn + f"console.log(JSON.stringify(trendRange({json.dumps(vals)}, {json.dumps(log)})));"
    return json.loads(subprocess.check_output(["node", "-e", js], text=True))


EM = [0.1063, 0.104, 0.101, 0.1035, 0.098]
VAL = [0.1232, 0.1228, 0.1225, 0.1226, 0.1221]


def test_fits_averages_and_validation_with_ten_percent_padding():
    lo, hi = _range(EM + VAL)
    span = 0.1232 - 0.098
    assert math.isclose(lo, 0.098 - 0.1 * span) and math.isclose(hi, 0.1232 + 0.1 * span)
    assert hi - lo < 0.04                     # not the raw steps' 0.036..0.276


def test_averages_trend_fills_the_plot():
    lo, hi = _range(EM)
    assert (0.1063 - 0.098) / (hi - lo) > 0.8


def test_near_constant_series_gets_a_minimum_height():
    lo, hi = _range([0.1225, 0.12251, 0.12249])
    assert hi - lo >= 0.05 * 0.1225 and lo < 0.1225 < hi
    assert math.isclose((lo + hi) / 2, 0.1225, rel_tol=1e-6)


def test_single_value_and_junk():
    lo, hi = _range([0.1, None, "x"])
    assert lo < 0.1 < hi and hi - lo > 0
    assert _range([]) is None


def test_log_scale_works_in_decades():
    lo, hi = _range([0.1, 0.01], log=True)
    assert math.isclose(lo, -2.1) and math.isclose(hi, -0.9)
    lo, hi = _range([0.1, 0.1], log=True)
    assert math.isclose(hi - lo, 0.024)


def test_draw_uses_the_fit_only_in_the_trend_views_and_clips_the_steps():
    draw = open(APP, encoding="utf-8").read()
    draw = draw[draw.index("function lossGraph("):]
    assert "(set.emean && em.length) || (set.ravg && ra.length) ? trendRange(" in draw
    assert 'mk("clipPath"' in draw and '"clip-path": `url(#${id})`' in draw
    assert "const YS = fit ?" in draw and ": Y;" in draw          # the default view draws exactly as before
    css = open(os.path.join(os.path.dirname(APP), "style.css"), encoding="utf-8").read()
    assert 'class: "lg-steps-fit"' in draw and ".lg-steps-fit .lg-raw" in css and ".lg-steps-fit .lg-smooth" in css
