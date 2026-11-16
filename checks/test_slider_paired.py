"""Pair +1 and -1 lines: with the by-epoch switch on, a switch on the right of the -1 box's heading
(FAMILY_SLIDER_PAIRED, off when missing) pairs each -1 line with a +1 line, both ways: the two share their epochs and
their place (the +1 box's order), and a line added, removed, moved or split in either box is in the other too, each
line's words kept with it by a pair id (app.js pairedLines, promptLines). It only shapes the page: the form's rows and
state, Start and the trainer's two boxes of lines are the same either way."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")
# a small page in node for app.js promptLines: the parts of the DOM its boxes of lines use (moving a node that holds
# the focus drops it, as a browser does)
DOM = r"""
class T { constructor(t) { this.nodeType = 3; this.textContent = String(t); this.parentNode = null; } }
const out = (n) => { const p = n.parentNode; if (!p) return;
  if (n.contains && n.contains(document.activeElement)) document.activeElement = document.body;
  p.childNodes.splice(p.childNodes.indexOf(n), 1); n.parentNode = null; };
const one = (e, s) => { const m = /^([a-z]*)((?:\.[\w-]+)*)$/i.exec(s);
  return (!m[1] || e.tagName === m[1].toUpperCase()) &&
    m[2].split(".").filter(Boolean).every((c) => e.classList.contains(c)); };
const fits = (e, sel) => { const ps = sel.split(" "); if (!one(e, ps.pop())) return false;
  for (let x = e.parentNode; ps.length && x && x.nodeType === 1; x = x.parentNode) {
    if (one(x, ps[ps.length - 1])) ps.pop(); }
  return !ps.length; };
const all = (root, sel) => root.children.flatMap((c) => [...(fits(c, sel) ? [c] : []), ...all(c, sel)]);
class E {
  constructor(tag) { Object.assign(this, {nodeType: 1, tagName: tag.toUpperCase(), childNodes: [], parentNode: null,
    className: "", dataset: {}, on: {}, value: "", disabled: false, checked: false, selectionStart: 0, selectionEnd: 0,
    _t: "", style: {setProperty() {}}}); }
  get textContent() { return this.childNodes.length ? this.childNodes.map((c) => c.textContent).join("") : this._t; }
  set textContent(t) { [...this.childNodes].forEach(out); this._t = String(t); }
  get classList() { const n = this, has = () => n.className.split(" ").filter(Boolean), api = {
    add: (c) => { if (!api.contains(c)) n.className = [...has(), c].join(" "); },
    remove: (c) => { n.className = has().filter((x) => x !== c).join(" "); }, contains: (c) => has().includes(c),
    toggle: (c, on = !api.contains(c)) => { if (on) api.add(c); else api.remove(c); return on; }}; return api; }
  setAttribute(k, v) { if (k === "checked") this.checked = true;
    else if (k === "style") this.style.display = /display:\s*none/.test(v) ? "none" : "";
    else if (k.startsWith("data-")) this.dataset[k.slice(5)] = v; else this[k] = v; }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  put(n, ref) { n = typeof n === "string" ? new T(n) : n; out(n);
    this.childNodes.splice(ref ? this.childNodes.indexOf(ref) : this.childNodes.length, 0, n); n.parentNode = this; }
  append(...ns) { ns.forEach((n) => this.put(n, null)); }
  prepend(n) { this.put(n, this.childNodes[0] || null); }
  insertBefore(n, ref) { if (n !== ref) this.put(n, ref || null); return n; }
  at(n, d) { const p = this.parentNode; out(n); p.childNodes.splice(p.childNodes.indexOf(this) + d, 0, n);
    n.parentNode = p; }
  after(n) { this.at(n, 1); }
  before(n) { this.at(n, 0); }
  remove() { out(this); }
  replaceChildren(...ns) { [...this.childNodes].forEach(out); this._t = ""; this.append(...ns); }
  sib(d) { const k = this.parentNode ? this.parentNode.children : []; return k[k.indexOf(this) + d] || null; }
  get nextElementSibling() { return this.sib(1); }
  get previousElementSibling() { return this.sib(-1); }
  get lastElementChild() { return this.children[this.children.length - 1] || null; }
  contains(n) { for (; n; n = n.parentNode) if (n === this) return true; return false; }
  closest(sel) { for (let x = this; x && x.nodeType === 1; x = x.parentNode) if (fits(x, sel)) return x; return null; }
  get isConnected() { let x = this; while (x.parentNode) x = x.parentNode; return x === document.documentElement; }
  querySelectorAll(sel) { return all(this, sel); }
  querySelector(sel) { return all(this, sel)[0] || null; }
  addEventListener(t, f) { (this.on[t] = this.on[t] || []).push(f); }
  fire(t, ev = {}) { const e = {target: this, isComposing: false, defaultPrevented: false, ...ev};
    e.preventDefault = () => { e.defaultPrevented = true; }; (this.on[t] || []).forEach((f) => f(e)); return e; }
  focus() { document.activeElement = this; }
  setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; }
  get offsetParent() { return null; }
  get offsetWidth() { return 0; }
}
const document = {createElement: (t) => new E(t), documentElement: new E("html"), body: new E("body")};
document.documentElement.append(document.body); document.activeElement = document.body;
document.querySelectorAll = (s) => all(document.documentElement, s);
const window = {}, requestAnimationFrame = () => 0, svgEl = () => document.createElement("i");
class ResizeObserver { observe() {} }
const S = {values: {}, state: {}, fam: null};
const onValue = (r, v) => { S.values[r.key] = v; };
"""
# krea2's +1 and -1 boxes in their lines look, paired ("smiling @1-4 / laughing @5-" and "frowning @1-4 / crying @5-"),
# built apart and then put on the page, as a form draws; the steps run once they have paired up, each note() taking
# what both boxes save
PAGE = r"""
S.fam = {form: {sections: [{id: "slider", rows: ROWS}]}};
const row = (k) => { const r = ROWS.find((x) => x.key === k); return {...r, ...r.by_epoch}; };
Object.assign(S.values, {FAMILY_SLIDER_BY_EPOCH: true, FAMILY_SLIDER_PAIRED: true, MAX_TRAIN_EPOCHS: "10",
  FAMILY_SLIDER_POS: "smiling @1-4\nlaughing @5-", FAMILY_SLIDER_NEG: "frowning @1-4\ncrying @5-"}, VALUES);
const POS = promptLines(row("FAMILY_SLIDER_POS"), el("div", {}));
const NEG = promptLines(row("FAMILY_SLIDER_NEG"), el("div", {}));
document.body.append(POS, NEG);
const P = () => pairBoxes.FAMILY_SLIDER_POS.items(), N = () => pairBoxes.FAMILY_SLIDER_NEG.items();
const box = (it) => it.querySelector("textarea"), ep = (it, k) => it.querySelectorAll(".prompt-eps input")[k];
const log = [], note = (what, ...more) => log.push([what, S.values.FAMILY_SLIDER_POS, S.values.FAMILY_SLIDER_NEG,
  ...more]);
// what the number column shows; the switch; a box drawn again (a reload, the Run card's Apply); a line's chain button
const nums = (xs) => xs.map((it) => it.querySelector(".prompt-num").textContent).join(" ");
const tick = () => new Promise((res) => setTimeout(res, 0));
const flip = (on) => { const t = NEG.querySelector(".prompt-head input"); t.checked = on; t.fire("change"); };
const redraw = async (...keys) => { for (const k of keys) {
  [POS, NEG].forEach((w) => w.querySelectorAll("input").filter((e) => e.id === "f_" + k).forEach((e) => e._sync())); }
  await tick(); };
const chain = (it) => it.querySelector(".prompt-chain").fire("click");
const typed = (it, words, a, b) => { const t = box(it); t.value = words; t.fire("input");
  [a, b].forEach((x, k) => { if (x !== undefined) { ep(it, k).value = x; ep(it, k).fire("input"); } }); };
setTimeout(async () => {
"""


def _paired(cases):
    """app.js pairedLines in node, case by case: [the -1 lines as saved, the +1 lines as the +1 box saves them]."""
    src = open(APP, encoding="utf-8").read()
    grab = lambda name: re.search(r"function " + name + r"\(.*?\n\}\n", src, re.S).group(0)
    js = (grab("epochsText") + grab("pairedLines") + f"const cases = {json.dumps(cases)};"
          "console.log(JSON.stringify(cases.map(([plus, words]) => [pairedLines(plus, words),"
          " pairedLines(plus, Object.fromEntries(plus.map((p) => [p.id, p.text])))])));")
    return json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8"))


def test_each_minus_line_follows_its_plus_line():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    line = lambda i, text, a="", b="", n="": {"id": i, "text": text, "epochs": [a, b, n]}
    cases = [
        # +1 line 2 moved (or sorted) to the top, a new +1 line 9 and line 4 gone: each -1 line's words go with its
        # own +1 line, the new one has none yet, and line 4's -1 line goes with it
        ([line("2", "laughing", "1", "4"), line("9", "grinning", "5", "8", "2"), line("1", "smiling")],
         {"1": "frowning", "2": "crying", "4": "pouting"}),
        # the epochs as the +1 box saves them: numbers only, and "every" only above 1
        ([line("1", "a", "5", "", "1"), line("2", "b", "", "8"), line("3", "c", "3", "9", "3")],
         {"1": "sad", "2": "glum", "3": "sulking"}),
        # switched on, or drawn: paired by place - the -1 box shorter gets an empty line on the +1 line's epochs ...
        ([line("0", "smiling", "1", "4"), line("1", "laughing", "5"), line("2", "grinning", "9")],
         {"0": "sad", "1": "angry"}),
        # ... and the +1 box shorter got an empty line, whose -1 line keeps its words with no epochs
        ([line("0", "smiling", "1", "4"), line("1", "")], {"0": "sad", "1": "angry"}),
    ]
    got = _paired(cases)
    assert [minus for minus, _plus in got] == [["crying @1-4", " @5-8/2", "frowning"],
                                               ["sad @5-", "glum @-8", "sulking @3-9/3"],
                                               ["sad @1-4", "angry @5-", " @9-"],
                                               ["sad @1-4", "angry"]]
    # as the trainer reads them (launch.slider_line_parts): each -1 line on its +1 line's epochs, with its own words
    for (plus, words), (minus, plus_saved) in zip(cases, got):
        assert len(minus) == len(plus_saved) == len(plus)
        for p, m, ps in zip(plus, minus, plus_saved):
            assert launch.slider_line_parts(m)[1:] == launch.slider_line_parts(ps)[1:], (m, ps)
            assert launch.slider_line_parts(m)[0] == words.get(p["id"], ""), m


def _page(steps, **values):
    """PAGE's boxes (app.js promptLines, krea2's rows; `values` over PAGE's) after `steps`: [[what, +1 box saved, -1
    box saved, ...], ...]."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    rows = {r["key"]: r for s in form.form(get_family("krea2"))["sections"] for r in s["rows"]}
    js = ('"use strict";\n' + DOM + re.search(r"const el = \(tag.*?\n\};\n", src, re.S).group(0)
          + src[src.index("// a line's epochs as a colour"):src.index("// the Samples tab's prompt sets")]
          + f"const ROWS = {json.dumps([rows['FAMILY_SLIDER_POS'], rows['FAMILY_SLIDER_NEG']])};\n"
          + f"const VALUES = {json.dumps(values)};\n" + PAGE + steps
          + "\nconsole.log(JSON.stringify(log)); }, 0);\n")
    run = subprocess.run(["node", "-"], input=js, capture_output=True, text=True, encoding="utf-8")   # too long for -e
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_the_minus_box_changes_its_pairs_too():
    got = _page(r"""
      note("drawn");
      const t2 = ep(N()[1], 1); t2.value = "8"; t2.fire("input"); note("-1 epoch");      // reaches its +1 line at once
      const t1 = ep(P()[0], 1); t1.value = "3"; t1.fire("input"); note("+1 epoch");      // and the other way round
      NEG.querySelector(".prompt-actions button").fire("mousedown", {button: 0});         // + Add a line
      note("-1 add", document.activeElement === box(N()[2]));
      box(N()[2]).value = "pouting"; box(N()[2]).fire("input");
      N()[2].querySelector(".prompt-up").fire("click"); note("-1 up");
      N()[1].querySelector(".prompt-del").fire("click"); note("-1 x");
      const b1 = box(N()[0]); b1.value = "frowning hard"; b1.setSelectionRange(8, 8);       // Enter in "frowning| hard"
      b1.fire("keydown", {key: "Enter"}); note("-1 split", document.activeElement === box(N()[1]));
      const b2 = box(N()[1]); b2.setSelectionRange(0, 0);                                   // its +1 line: empty
      b2.fire("keydown", {key: "Backspace"}); note("-1 join");
      const b3 = box(N()[1]); b3.setSelectionRange(0, 0);                                   // its +1 line: "laughing"
      note("-1 no join", b3.fire("keydown", {key: "Backspace"}).defaultPrevented);
      note("look", nums(P()), nums(N()), N().map((it) => it.querySelector(".prompt-pair").textContent),
        N().every((it) => [...it.querySelectorAll(".prompt-eps input")].every((i) => !i.disabled)),
        NEG.querySelector(".prompt-actions button").style.display !== "none");
    """)
    assert got == [
        ["drawn", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-"],
        ["-1 epoch", "smiling @1-4\nlaughing @5-8", "frowning @1-4\ncrying @5-8"],
        ["+1 epoch", "smiling @1-3\nlaughing @5-8", "frowning @1-3\ncrying @5-8"],
        # a pair: an empty +1 line and the new -1 line, focused
        ["-1 add", "smiling @1-3\nlaughing @5-8\n", "frowning @1-3\ncrying @5-8\n", True],
        ["-1 up", "smiling @1-3\n\nlaughing @5-8", "frowning @1-3\npouting\ncrying @5-8"],
        ["-1 x", "smiling @1-3\nlaughing @5-8", "frowning @1-3\ncrying @5-8"],
        # a new pair after it: the rest of the words on the -1 line, its +1 line empty, both on the same epochs
        ["-1 split", "smiling @1-3\n @1-3\nlaughing @5-8", "frowning @1-3\n hard @1-3\ncrying @5-8", True],
        ["-1 join", "smiling @1-3\nlaughing @5-8", "frowning hard @1-3\ncrying @5-8"],
        ["-1 no join", "smiling @1-3\nlaughing @5-8", "frowning hard @1-3\ncrying @5-8", False],   # "laughing" stays
        ["look", "smiling @1-3\nlaughing @5-8", "frowning hard @1-3\ncrying @5-8", "1 2", "1 2", ["↔", "↔"], True,
         True]]


def test_pairs_take_their_places_together():
    got = _page(r"""
      const f2 = ep(N()[1], 0); f2.value = "1"; f2.fire("input");          // -1 line 2 on from epoch 1, to 8 (its +1
      ep(N()[1], 1).value = "8"; ep(N()[1], 1).fire("input");               // line too), then left: the pair goes first
      N()[1].fire("focusout", {relatedTarget: null}); note("-1 left", N()[0].classList.contains("prompt-landed"));
      const b = box(N()[0]); b.setSelectionRange(b.value.length, b.value.length);   // a paste of several lines: pairs
      b.fire("paste", {clipboardData: {getData: () => "\nsulking @5-6\nweeping"}}); note("-1 paste");
      autoFill = true;                                  // Auto-fill in the -1 box: its last epoch follows its first ...
      const f4 = ep(N()[3], 0); f4.value = "9"; f4.fire("input"); note("-1 auto-fill");
      const g4 = ep(P()[3], 0); g4.value = "7"; g4.fire("input"); note("+1 auto-fill");    // ... filled in, both sides
    """)
    assert got == [
        ["-1 left", "laughing @1-8\nsmiling @1-4", "crying @1-8\nfrowning @1-4", True],
        # the pasted lines after it, as pairs (their +1 lines empty, on the epochs they bring), sorted as pairs
        ["-1 paste", "laughing @1-8\nsmiling @1-4\n\n @5-6", "crying @1-8\nfrowning @1-4\nweeping\nsulking @5-6"],
        ["-1 auto-fill", "laughing @1-8\nsmiling @1-4\n\n @9-9", "crying @1-8\nfrowning @1-4\nweeping\nsulking @9-9"],
        ["+1 auto-fill", "laughing @1-8\nsmiling @1-4\n\n @7-7", "crying @1-8\nfrowning @1-4\nweeping\nsulking @7-7"]]


def test_the_pairs_are_found_again_by_their_lines():
    # FAMILY_SLIDER_PAIRS (app.js pairsText): each pair's +1 line and -1 line, by place and text, in the +1 box's order;
    # pairsFrom finds them again in the boxes' lines as they are now
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(APP, encoding="utf-8").read()
    grab = lambda name: re.search(r"function " + name + r"\(.*?\n\}\n", src, re.S).group(0)
    plus, minus, both = ["smiling @1-4", "laughing @5-"], ["frowning @1-4", "crying @5-"], [[0, 0], [1, 1]]
    cases = [
        [[plus, minus, both], plus, minus],                                   # drawn again as they were
        # a +1 line put in between (the switch off, a preset): each pair found by its text, the new line in none
        [[plus, minus, both], ["smiling @1-4", "grinning @2-3", "laughing @5-"], minus],
        # the Run card's Apply: one pair's +1 line in other words - that pair is gone, the other kept
        [[plus, minus, both], ["smiling @1-4", "laughing hard @5-"], minus],
        [[["a", "b"], ["", ""], both], ["a", "b"], ["", ""]],                # lines alike: each where it was
        ["", plus, minus], ["nonsense", plus, minus], ['[[0, "smiling @1-4", 0]]', plus, minus],   # none kept
    ]
    js = (grab("pairsText") + grab("pairsFrom") + f"const cases = {json.dumps(cases)};"
          "console.log(JSON.stringify([pairsText(...cases[0][0]), cases.map(([k, p, m]) =>"
          " pairsFrom(typeof k === 'string' ? k : pairsText(...k), p, m))]));")
    text, got = json.loads(subprocess.check_output(["node", "-e", js], encoding="utf-8"))
    assert json.loads(text) == [[0, "smiling @1-4", 0, "frowning @1-4"], [1, "laughing @5-", 1, "crying @5-"]]
    assert got == [both, [[0, 0], [2, 1]], [[0, 0]], both, [], [], []]


def test_pairs_keep_their_numbers_through_a_redraw_and_the_switch():
    got = _page(r"""
      note("drawn", nums(P()), nums(N()), JSON.parse(S.values.FAMILY_SLIDER_PAIRS).length);
      await redraw("FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG"); note("redrawn", nums(P()), nums(N()));
      // off, a +1 line put in between the pairs (Enter in "smiling| wide"), then on again: the new line on its own
      flip(false); note("off", nums(P()), nums(N()));
      const b = box(P()[0]); b.value = "smiling wide"; b.setSelectionRange(7, 7); b.fire("keydown", {key: "Enter"});
      typed(P()[1], "grinning", "2", "3");
      flip(true); note("on", nums(P()), nums(N()));
      // off, a -1 line added and given its epochs, the boxes drawn again (a reload), then on: the same
      flip(false);
      NEG.querySelector(".prompt-actions button").fire("mousedown", {button: 0});
      typed(N()[2], "pouting", "2", "3"); N()[2].fire("focusout", {relatedTarget: null});
      await redraw("FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG");
      flip(true); note("on again", nums(P()), nums(N()));
      ep(P()[2], 1).value = "8"; ep(P()[2], 1).fire("input"); note("pair 2 on 5-8");   // two-way: pairs only
      ep(N()[1], 0).value = "1"; ep(N()[1], 0).fire("input"); note("-1 line B from 1");
      // off, pair 2's -1 line in other words, a reload, then on: still a pair (kept as the boxes change, off too)
      flip(false); typed(N()[2], "crying hard"); await redraw("FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG");
      flip(true); note("words changed while off", nums(P()), nums(N()));
    """)
    assert got == [
        ["drawn", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 2", "1 2", 2],
        ["redrawn", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 2", "1 2"],
        ["off", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 2", "1 2"],
        # no shift: "crying" keeps "laughing"'s epochs, and the new +1 line is on its own (A)
        ["on", "smiling @1-4\ngrinning @2-3\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 A 2", "1 2"],
        # the -1 box's line on its own takes the next letter after the +1 box's (B): never a pair with A
        ["on again", "smiling @1-4\ngrinning @2-3\nlaughing @5-", "frowning @1-4\npouting @2-3\ncrying @5-",
         "1 A 2", "1 B 2"],
        ["pair 2 on 5-8", "smiling @1-4\ngrinning @2-3\nlaughing @5-8", "frowning @1-4\npouting @2-3\ncrying @5-8"],
        ["-1 line B from 1", "smiling @1-4\ngrinning @2-3\nlaughing @5-8", "frowning @1-4\npouting @1-3\ncrying @5-8"],
        ["words changed while off", "smiling @1-4\ngrinning @2-3\nlaughing @5-8",
         "frowning @1-4\npouting @1-3\ncrying hard @5-8", "1 A 2", "1 B 2"]]


def test_a_run_card_change_keeps_the_pairs_that_still_match():
    got = _page(r"""
      // the run's +1 box changed by the Run card's Apply: a new line, and pair 2's +1 line in other words
      S.values.FAMILY_SLIDER_POS = "smiling @1-4\ngrinning @2-3\nlaughing hard @5-";
      await redraw("FAMILY_SLIDER_POS"); note("+1 changed", nums(P()), nums(N()), S.values.FAMILY_SLIDER_PAIRS);
      // ... and its -1 box, its lines in another order: pair 1 found where its line went, the lines in their places
      S.values.FAMILY_SLIDER_NEG = "crying @5-\nfrowning @1-4";
      await redraw("FAMILY_SLIDER_NEG"); note("-1 changed", nums(P()), nums(N()));
    """)
    assert got == [
        ["+1 changed", "smiling @1-4\ngrinning @2-3\nlaughing hard @5-", "frowning @1-4\ncrying @5-", "1 A B", "1 C",
         '[[0,"smiling @1-4",0,"frowning @1-4"]]'],
        ["-1 changed", "smiling @1-4\ngrinning @2-3\nlaughing hard @5-", "frowning @1-4\ncrying @5-", "1 A B", "1 C"]]


def test_unpair_and_pair_it():
    got = _page(r"""
      chain(P()[1]); note("unpair 2", nums(P()), nums(N()));          // both have words: both on their own
      chain(N()[1]); note("pair crying", nums(P()), nums(N()));       // a +1 partner with no words, on its epochs
      chain(N()[1]); note("unpair it", nums(P()), nums(N()));         // its +1 line has no words: it goes
      POS.querySelector(".prompt-actions button").fire("mousedown", {button: 0});    // + Add a line: a pair
      typed(P()[2], "grinning"); note("add", nums(P()), nums(N()));
      chain(P()[2]); note("unpair grinning", nums(P()), nums(N()));   // its -1 line has no words: it goes
    """)
    assert got == [
        ["unpair 2", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 A", "1 B"],
        ["pair crying", "smiling @1-4\nlaughing @5-\n @5-", "frowning @1-4\ncrying @5-", "1 A 2", "1 2"],
        ["unpair it", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 A", "1 B"],
        ["add", "smiling @1-4\nlaughing @5-\ngrinning", "frowning @1-4\ncrying @5-\n", "1 A 2", "1 B 2"],
        ["unpair grinning", "smiling @1-4\nlaughing @5-\ngrinning", "frowning @1-4\ncrying @5-", "1 A B", "1 C"]]


def test_lines_on_their_own_take_their_places_among_the_pairs_and_act_alone():
    got = _page(r"""
      chain(P()[1]);                                                  // laughing (A) and crying (B) on their own
      typed(N()[1], "crying", "1", "2"); note("-1 B on 1-2");         // its own epochs: the +1 box as it was
      ep(P()[0], 0).value = "3"; ep(P()[0], 0).fire("input");         // pair 1 from epoch 3 (both boxes), then left:
      P()[0].fire("focusout", {relatedTarget: null}); note("pair 1 on 3-4", nums(P()), nums(N()));   // B goes first
      NEG.querySelector(".prompt-actions button").fire("mousedown", {button: 0});    // a new pair, on 6-7, left
      typed(N()[2], "sulking", "6", "7"); N()[2].fire("focusout", {relatedTarget: null});
      note("pair 2 on 6-7", nums(P()), nums(N()));
      const b = box(N()[0]); b.setSelectionRange(6, 6); b.fire("keydown", {key: "Enter"});   // Enter on B: on its own
      note("-1 B split", nums(P()), nums(N()));
      N()[1].querySelector(".prompt-del").fire("click"); note("-1 C gone", nums(P()), nums(N()));   // x: it alone
    """)
    assert got == [
        ["-1 B on 1-2", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @1-2"],
        ["pair 1 on 3-4", "smiling @3-4\nlaughing @5-", "crying @1-2\nfrowning @3-4", "1 A", "B 1"],
        ["pair 2 on 6-7", "smiling @3-4\nlaughing @5-\n @6-7", "crying @1-2\nfrowning @3-4\nsulking @6-7", "1 A 2",
         "B 1 2"],
        ["-1 B split", "smiling @3-4\nlaughing @5-\n @6-7", "crying @1-2\n @1-2\nfrowning @3-4\nsulking @6-7",
         "1 A 2", "B C 1 2"],
        ["-1 C gone", "smiling @3-4\nlaughing @5-\n @6-7", "crying @1-2\nfrowning @3-4\nsulking @6-7", "1 A 2",
         "B 1 2"]]


def test_the_switch_sits_on_the_minus_box_and_is_off_by_default():
    assert form.DEFAULTS["FAMILY_SLIDER_PAIRED"] is False and form.DEFAULTS["FAMILY_SLIDER_PAIRS"] == ""
    for d in training_families():
        if not d.slider_training:
            continue
        rows = [r for s in form.form(d)["sections"] for r in s["rows"]]
        assert not {"FAMILY_SLIDER_PAIRED", "FAMILY_SLIDER_PAIRS"} & {r["key"] for r in rows}, d.key   # no rows
        pairs = {r["key"]: r["by_epoch"]["pair"] for r in rows if "pair" in r.get("by_epoch", {})}
        if launch.slider_picture_lines(d):
            assert list(pairs) == ["FAMILY_SLIDER_NEG"], d.key           # in the lines look of the -1 box only
            p = pairs["FAMILY_SLIDER_NEG"]
            assert (p["key"], p["kind"], p["default"], p["follows"], p["label"], p["keep"]) == (
                "FAMILY_SLIDER_PAIRED", "check", False, "FAMILY_SLIDER_POS", "Pair +1 and -1 lines",
                "FAMILY_SLIDER_PAIRS"), d.key
        else:
            assert not pairs, d.key                                       # songs: no lines, no switch


def _inputs(tmp_path, desc, **values):
    """The form's values (a value of None: the key missing, as in an older preset or run) and Start's inputs."""
    models = {}
    for mf in desc.model_files:
        f = tmp_path / f"{desc.key}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "8",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts",
         "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
         "FAMILY_SLIDER_NEG": "frowning @1-4\ncrying @5-", "FAMILY_SLIDER_GUIDANCE": "3",
         "FAMILY_SLIDER_BY_EPOCH": True, **values}
    v = {k: x for k, x in v.items() if x is not None}
    return v, form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))


def test_it_only_shapes_the_page(tmp_path):
    desc = get_family("krea2")
    kept = '[[0,"smiling @1-4",0,"frowning @1-4"],[1,"laughing @5-",1,"crying @5-"]]'
    (off, i_off), (on, i_on), (old, i_old) = (_inputs(tmp_path, desc, FAMILY_SLIDER_PAIRED=x, FAMILY_SLIDER_PAIRS=k)
                                              for x, k in ((False, ""), (True, kept), (None, None)))
    assert "FAMILY_SLIDER_PAIRED" not in old and "FAMILY_SLIDER_PAIRS" not in old
    assert (i_off["FAMILY_SLIDER_PAIRED"], i_on["FAMILY_SLIDER_PAIRED"], i_old["FAMILY_SLIDER_PAIRED"]) == (
        False, True, False)                                              # a bool in the inputs, off when missing
    assert (i_off["FAMILY_SLIDER_PAIRS"], i_on["FAMILY_SLIDER_PAIRS"], i_old["FAMILY_SLIDER_PAIRS"]) == (
        "", kept, "")                                                    # the pairs: text, "" when missing
    assert _inputs(tmp_path, desc, FAMILY_SLIDER_PAIRED="")[1]["FAMILY_SLIDER_PAIRED"] is False
    assert form.state(desc, on, "") == form.state(desc, off, "") == form.state(desc, old, "")    # the rows on show
    plans = [launch.plan(desc, i) for i in (i_off, i_on, i_old)]
    assert plans[0] == plans[1] == plans[2] and not plans[0].problems                            # Start
    cmd = next(s.cmd for s in plans[1].stages if s.name == "Training")      # the trainer: two boxes of lines
    ends = [(cmd[k + 1], cmd[k + 2]) for k, t in enumerate(cmd) if t == "--slider_line" and cmd[k + 1] != "base"]
    assert ends == [("pos", "smiling @1-4"), ("pos", "laughing @5-"), ("neg", "frowning @1-4"), ("neg", "crying @5-")]


def test_the_notes_name_lines_by_their_words_where_the_boxes_show_pairs(tmp_path):
    # paired, a box shows a pair's number or a letter, not a line's place: "laughing" is pair 2 in the +1 box's third
    # row, after "grinning" on its own (A), and "crying" its partner in the -1 box's second - the notes under the boxes
    # name lines by their words
    values = {"MAX_TRAIN_EPOCHS": "10", "FAMILY_SLIDER_PAIRED": True,
              "FAMILY_SLIDER_POS": "smiling @1-4\ngrinning @2-3\nlaughing @5-12",
              "FAMILY_SLIDER_NEG": "frowning @1-4\ncrying @5-12",
              "FAMILY_SLIDER_PAIRS": '[[0,"smiling @1-4",0,"frowning @1-4"],[2,"laughing @5-12",1,"crying @5-12"]]'}
    assert _page('note("shown", nums(P()), nums(N()));', **values)[0][3:] == ["1 A 2", "1 2"]
    desc = get_family("krea2")
    v, _inp = _inputs(tmp_path, desc, FAMILY_SLIDER_BASE="a man, @1-6\na woman, @5-", **values)
    notes = form.state(desc, v, "")["notes"]
    assert notes["FAMILY_SLIDER_POS"] == ('"laughing" runs to epoch 12, past the run\'s last (10): it stops at 10. '
                                          '"smiling" and "grinning" are on together at epochs 2–3 and train as one '
                                          'prompt: "smiling, grinning". For a change, give them epochs of their own.')
    assert notes["FAMILY_SLIDER_NEG"] == '"crying" runs to epoch 12, past the run\'s last (10): it stops at 10.'
    assert notes["FAMILY_SLIDER_BASE"] == ('"a man," and "a woman," are on together at epoch 5 and train as one '
                                           'description: "a man, a woman,". For different pictures, give them epochs '
                                           'of their own.')
    assert not any(re.search(r"\b[Ll]ines? \d", n) for n in notes.values())    # no line named by its place
