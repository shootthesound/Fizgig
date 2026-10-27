"""The web console keeps a text selection while the log updates (followLog in app.js).

Runs followLog under node with a tiny stand-in DOM; skipped when node is missing.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js"


def _source(name):
    s = APP.read_text(encoding="utf-8")
    a = s.index(f"function {name}(")
    depth, i = 0, s.index("{", a)
    while True:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
        if depth == 0:
            return s[a:i]


HARNESS = r"""
class Text { constructor(v) { this.nodeType = 3; this.nodeValue = v; this.parent = null; }
  remove() { const k = this.parent.kids; k.splice(k.indexOf(this), 1); this.parent = null; } }
class Pre { constructor() { this.kids = []; this.scrollTop = 0; this.clientHeight = 100; this.scrollHeight = 100; }
  get firstChild() { return this.kids[0] || null; } get lastChild() { return this.kids[this.kids.length - 1] || null; }
  get textContent() { return this.kids.map((n) => n.nodeValue).join(""); }
  set textContent(v) { this.kids.forEach((n) => { n.parent = null; }); this.kids = v ? [this.adopt(new Text(v))] : []; }
  adopt(n) { n.parent = this; return n; }
  append(n) { this.kids.push(this.adopt(n)); }
  insertBefore(n, ref) { this.kids.splice(this.kids.indexOf(ref), 0, this.adopt(n)); } }
const out = new Pre(); let picked = null; const listeners = {};
const document = { createTextNode: (v) => new Text(v), addEventListener: (t, f) => { listeners[t] = f; } };
const window = { getSelection: () => picked && { isCollapsed: false, rangeCount: 1,
  containsNode: (n) => n === out ? picked.parent === out : n === picked } };
const $ = () => out; const S = { logSince: 0 }; const TOKEN = "";
let src; class EventSource { constructor() { src = this; } close() {} }
const send = (l) => src.onmessage({ data: JSON.stringify(l), lastEventId: "" });
let timers = []; const setTimeout = (f) => { timers.push(f); }; const tick = () => { while (timers.length) timers.shift()(); };
%s
%s
followLog();
const r = {};
send("\u0000clear\n"); send("a\n"); send("b\n"); send("\rsteps 1"); send("\rsteps 2"); tick();
r.basic = out.textContent;
const firstA = out.kids[0]; picked = firstA;               // select an old line: updates carry on
send("\rsteps 3"); send("c\n"); tick();
r.oldSel = [out.textContent, out.kids[0] === firstA];
picked = out.lastChild;                                     // select the live line: updates wait
send("\rsteps 4"); send("\rsteps 5"); send("d\n"); tick();
r.held = out.textContent;
picked = null; listeners.selectionchange(); tick();                // selection gone: catch up in one go
r.caught = out.textContent;
send("\u0000clear\n"); for (let i = 0; i < 30; i++) send("x".repeat(9999) + "\n"); tick();
r.trimmed = out.textContent.length;
send("\u0000clear\n"); for (let i = 0; i < 5000; i++) send(i %% 2 ? "line " + i + "\n" : "\rbar " + i);
r.queued = timers.length; tick();                          // a new tab's backlog: one draw, not one per line
r.backlog = [out.kids.length, out.textContent.endsWith("line 4999\n")];
console.log(JSON.stringify(r));
"""


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_console_keeps_selection():
    js = HARNESS % (_source("selTouches"), _source("followLog"))
    r = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert r["basic"] == "a\nb\nsteps 2"
    assert r["oldSel"] == ["a\nb\nsteps 3c\n", True]
    assert r["held"] == "a\nb\nsteps 3c\n"
    assert r["caught"] == "a\nb\nsteps 3c\nsteps 5d\n"
    assert 199000 <= r["trimmed"] <= 200000
    assert r["queued"] == 1
    assert r["backlog"] == [2, True]
