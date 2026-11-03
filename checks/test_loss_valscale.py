"""The loss graph's validation scale: in the Epoch average / Running avg views, with validation shown and 2+ checks,
validation gets its own scale (its numbers in teal, a second column beside the loss numbers, titled "val"), so the
averages and validation each fill the plot height instead of sharing one scale with a gap between them; hover text
keeps the real values; one check, validation hidden or the default (raw) view draw as before; the log scale fits
each in decades; the running avg's first-epoch start (a mean of a few steps) no longer stretches the scale. Runs the
page's own lossGraph in jsdom (skipped without node / jsdom)."""
import json, math, os, random, re, shutil, subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
JSDOM_DIRS = [p for p in (os.environ.get("FIZGIG_JSDOM"), "/tmp/jsd/node_modules",
                          os.path.join(HERE, "node_modules")) if p]

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _node(js):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    return json.loads(subprocess.check_output(["node", "-e", js], text=True))


def _fn(name):
    src = open(APP, encoding="utf-8").read()
    return re.search(r"function " + name + r"\(.*?\n\}\n", src, re.S).group(0)


def test_val_scale_only_in_trend_views_with_two_checks():
    js = _fn("valScale") + "console.log(JSON.stringify([valScale(true, true, 2), valScale(true, true, 33), " \
         "valScale(true, true, 1), valScale(true, false, 33), valScale(false, true, 33), valScale(0, true, 5)]));"
    assert _node(js) == [True, True, False, False, False, False]


def test_running_avg_fit_skips_the_first_epoch():
    ra = [[1, 0.05], [5, 0.2], [10, 0.104], [20, 0.1]]
    js = _fn("raFit") + f"""console.log(JSON.stringify([raFit({ra}, [[1, 1], [2, 10]]), raFit({ra}, [[1, 1]]),
        raFit({ra}, []), raFit([[1, 0.05]], [[1, 1], [2, 10]])]));"""
    a, b, c, d = _node(js)
    assert a == [[10, 0.104], [20, 0.1]] and b == ra and c == ra and d == [[1, 0.05]]


def _data(n_val=33, epochs=33, spe=60):
    rnd = random.Random(4)
    pts, em, ra, val, eps, win, step = [], [], [], [], [], [], 0
    for e in range(1, epochs + 1):
        eps.append([e, step + 1])
        base = 0.1035 - 0.0075 * (e - 1) / max(1, epochs - 1)
        ls = []
        for i in range(spe):
            step += 1
            x = max(0.02, rnd.lognormvariate(math.log(base * 0.8), 0.65))
            pts.append([step, e, round(x, 5), 3e-4 * (1 - step / (epochs * spe)) + 1e-5])
            ls.append(x)
            win = (win + [x])[-spe:]
            if i % 6 == 0:
                ra.append([step, sum(win) / len(win)])
        em.append([step, e, sum(ls) / len(ls)])
        if e <= n_val:
            val.append([step, e, round(0.1415 - 0.0025 * (e - 1) / max(1, epochs - 1) + rnd.gauss(0, 0.0007), 5),
                        None, 0])
    return {"points": pts, "epoch_mean": em, "running_avg": ra, "val": val, "epochs": eps, "steps": step,
            "stride": 1}


HARNESS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body><div id='host'></div></body>", {url: "http://localhost/"});
global.window = dom.window; global.document = dom.window.document; global.localStorage = dom.window.localStorage;
global.requestAnimationFrame = (f) => f(); global.ResizeObserver = class { observe() {} };
const src = fs.readFileSync(process.env.APP, "utf8");
const head = src.slice(src.indexOf("const $ = "), src.indexOf("const S = {"));
const lg = src.slice(src.indexOf("// ---- the loss graph: a run"), src.indexOf("// the run card's graph"));
eval(head + lg + "; global.lossGraph = lossGraph;");
const data = JSON.parse(process.env.DATA), out = {};
for (const [name, st] of Object.entries(JSON.parse(process.env.VIEWS))) {
  localStorage.setItem("fizgig_lossgraph", JSON.stringify(st));
  const host = document.getElementById("host"); host.replaceChildren(lossGraph(data));
  const svg = host.querySelector("svg");
  const cy = (cls) => [...svg.querySelectorAll("circle." + cls)].map((c) => Number(c.getAttribute("cy")));
  const txt = (sel) => [...svg.querySelectorAll(sel)].map((t) => t.textContent);
  out[name] = {val: cy("lg-val-dot"), em: cy("lg-emean-dot"), valAxis: txt("text.lg-val-axis"),
    axis: txt("text.lg-axis:not(.lg-val-axis):not(.lg-lr-axis):not(.lg-axis-title)").filter((t) => !t.startsWith("e")),
    valTips: [...svg.querySelectorAll("circle.lg-val-dot title")].map((t) => t.textContent),
    legend: host.querySelector(".lg-legend").textContent, vb: svg.getAttribute("viewBox")};
}
console.log(JSON.stringify(out));
"""

T, PLOT = 10, 240 - 10 - 26        # the plot's top and height at the harness's width (800 -> H 240)


def _render(data, views):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    env = {**os.environ, "NODE_PATH": d, "APP": os.path.abspath(APP), "DATA": json.dumps(data),
           "VIEWS": json.dumps(views)}
    return json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))


def _fill(ys):
    return (max(ys) - min(ys)) / PLOT


ON = dict(smooth=0.9, log=False, clip=True, lr=True)


def test_each_fills_the_plot_with_its_own_teal_scale():
    data = _data()
    vals = [p[2] for p in data["val"]]
    out = _render(data, {"epoch": {**ON, "emean": True, "ravg": False, "val": True},
                         "both": {**ON, "emean": True, "ravg": True, "val": True},
                         "ravg": {**ON, "emean": False, "ravg": True, "val": True}})
    for name in ("epoch", "both", "ravg"):
        o = out[name]
        assert _fill(o["val"]) > 0.75                                   # validation's trend fills the height
        assert all(T - 0.5 <= y <= T + PLOT + 0.5 for y in o["val"])
        assert o["valAxis"][-1] == "val" and len(o["valAxis"]) == 4      # three teal numbers + its title
        nums = [float(x) for x in o["valAxis"][:3]]
        assert nums[0] < min(vals) < max(vals) < nums[2]                # bottom..top around the real values
        assert "its own scale, the teal numbers" in o["legend"]
        tips = [float(t.rsplit(": ", 1)[1]) for t in o["valTips"]]
        assert tips == pytest.approx(vals, abs=5e-5)                    # hover: the real values
    assert _fill(out["epoch"]["em"]) > 0.75                             # and the epoch averages fill it too
    assert max(float(x) for x in out["epoch"]["axis"]) < min(vals)      # the loss scale no longer reaches validation


def test_log_scale_fits_each_in_decades():
    out = _render(_data(), {"log": {**ON, "log": True, "emean": True, "ravg": False, "val": True}})["log"]
    assert _fill(out["val"]) > 0.75 and _fill(out["em"]) > 0.75 and len(out["valAxis"]) == 4


def test_falls_back_as_before():
    data = _data()
    out = _render(data, {"hidden": {**ON, "emean": True, "ravg": True, "val": False},
                         "raw": {**ON, "emean": False, "ravg": False, "val": True}})
    for name in ("hidden", "raw"):
        assert out[name]["valAxis"] == [] and out[name]["vb"] == "0 0 800 240"
        assert "its own scale" not in out[name]["legend"]
    assert out["hidden"]["val"] == []
    one = _render(_data(n_val=1), {"one": {**ON, "emean": True, "ravg": False, "val": True}})["one"]
    assert one["valAxis"] == [] and len(one["val"]) == 1               # one check: on the shared scale, as before
    assert max(float(x) for x in one["axis"]) > 0.14                   # (the shared scale reaches it)
