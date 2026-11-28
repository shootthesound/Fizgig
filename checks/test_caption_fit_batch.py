"""The Captions tab fits every caption box in one pass, not one pass a box (fitBatch in app.js).

Each caption box used to be sized twice - once by the fit its setter queued, once by a sweep afterwards -
and each sizing measures the box, writes its height, then measures again, so the second measurement is
taken against a page the write has just dirtied. A page of 200 pictures with two-line captions is 400
boxes and 800 of those round trips: the reported 15-20 s to open the tab. `fitBatch` queues one fit a
frame that does every frame, every drop, every read and every write as its own pass, so the page is
measured twice whatever it holds. This runs the real `fitFlush` under node against a stand-in DOM that
counts a read taken while a write is un-flushed, and asserts the count stays flat as the boxes grow.
Skipped when node is missing.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js"
START = "const fitFrame = (ta) =>"


def _fit_block():
    """the sizing source: fitFrame through the fitBatch that queues it"""
    s = APP.read_text(encoding="utf-8")
    a = s.index(START)
    e = s.index("const fitBatch", a)
    return s[a:s.index("\n", e) + 1]


HARNESS = r"""
// A stand-in DOM where every layout number is a counted read and every height is a counted write. A read
// taken while a write is un-flushed is one forced layout - the cost that made the tab take 15-20 s.
let reads = 0, writes = 0, forced = 0, dirty = false;
const read = () => { reads++; if (dirty) { forced++; dirty = false; } };
const write = () => { writes++; dirty = true; };
class Box {
  constructor(scroll) {
    this.isConnected = true; this.offsetParent = {}; this._capHl = null; this._scroll = scroll || 40;
    this.style = { set height(v) { this._h = v; write(); } };
  }
  get offsetHeight() { read(); return 20; }
  get clientHeight() { read(); return 18; }
  get scrollHeight() { read(); return this._scroll; }
  get offsetLeft() { read(); return 0; }
  closest() { return this._inCaption === false ? null : {}; }      // inside a .caption-lines by default
}
let boxes = [], raf = null, rafCount = 0;
const document = { querySelectorAll: () => boxes };
const innerHeight = 900;
const requestAnimationFrame = (f) => { rafCount++; raf = f; };
const capHlPaint = () => {};
const capHlPlace = () => {};
%s

// a render asks for a fit once per caption box, the way the captionLines setter does
const run = (n, scroll) => {
  reads = writes = forced = rafCount = 0; dirty = false; raf = null;
  boxes = Array.from({length: n}, () => new Box(scroll));
  for (let i = 0; i < n; i++) fitBatch();
  if (raf) raf();
  return {reads, writes, forced, rafCount};
};
const r = {};
r.two = run(2);
r.four_hundred = run(400);
r.written = (() => { run(1); return boxes[0].style._h; })();
const uncapped = (() => { run(1, 5000); return boxes[0].style._h; })();     // a caption line: no ceiling
const capped = (() => {                                                    // the big view's box: 40%% of the window
  run(1, 5000); boxes[0]._inCaption = false; raf(); return boxes[0].style._h;
})();
console.log(JSON.stringify([r, uncapped, capped]));
"""


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_one_fit_a_page_not_one_a_box():
    js = HARNESS % _fit_block()
    [r, uncapped, capped] = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True,
                                                       check=True).stdout)
    # 400 requests still fit once: one queued flush, not a rAF apiece
    assert r["two"]["rafCount"] == 1
    assert r["four_hundred"]["rafCount"] == 1
    # each box is read four times (frame twice, size twice) and written twice (dropped, fitted)
    assert r["two"]["reads"] == 8 and r["two"]["writes"] == 4
    assert r["four_hundred"]["reads"] == 1600 and r["four_hundred"]["writes"] == 800
    # the whole point: forced layouts do not grow with the boxes
    assert r["two"]["forced"] == r["four_hundred"]["forced"]
    assert r["four_hundred"]["forced"] <= 2
    # and a box still gets the height its text asks for, capped only outside a caption
    assert r["written"] == "42px"                      # scrollHeight 40 + the 2px border
    assert uncapped == "5002px"
    assert capped == "360px"                            # round(900 * 0.4)


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_fit_text_still_sizes_one_box_on_its_own():
    """the keystroke path: `fitText` runs the same four steps for a single box"""
    js = (HARNESS % _fit_block()).replace(
        "console.log(JSON.stringify([r, uncapped, capped]));",
        "reads = writes = forced = 0; dirty = false;\n"
        "boxes = [new Box(40)]; fitText(boxes[0], true);\n"
        "console.log(JSON.stringify([boxes[0].style._h, forced]));")
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["42px", 1]
