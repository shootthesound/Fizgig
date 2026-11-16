"""A prompt slider's Change the prompts during the run (advanced), in words read at a glance: the switch's name and
what it's for, each long hint of its rows as its first sentence and a small More (and Less) - its rows only -, Split and
Slide that start from the number in Single, the plan's how firmly in every section (greyed where no Leave these alone
line is on) and its rate said one way, Reuse practice pictures under the plan, and the boxes' headings in one style.
With the switch off the card is as before: test_slider_by_epoch.py."""
import json
import os
import re
import shutil
import subprocess

import pytest

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family, training_families

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")
MODE = {"FAMILY_SLIDER_BY_EPOCH": "on", "_slider_plan": True, "FAMILY_SLIDER_KEEP": True}   # its rows with More
# a small page in node for app.js's own parts (the DOM they use); shown(n): what a node shows, its hidden parts left out
DOM = r"""
class T { constructor(t) { this.nodeType = 3; this.textContent = String(t); this.parentNode = null; } }
const out = (n) => { const p = n.parentNode;
  if (p) { p.childNodes.splice(p.childNodes.indexOf(n), 1); n.parentNode = null; } };
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
    className: "", dataset: {}, attrs: {}, on: {}, value: "", disabled: false, checked: false, hidden: false,
    selectionStart: 0, selectionEnd: 0, _t: "", style: {setProperty() {}}}); }
  get textContent() { return this.childNodes.length ? this.childNodes.map((c) => c.textContent).join("") : this._t; }
  set textContent(t) { [...this.childNodes].forEach(out); this._t = String(t); }
  get classList() { const n = this, has = () => n.className.split(" ").filter(Boolean), api = {
    add: (c) => { if (!api.contains(c)) n.className = [...has(), c].join(" "); },
    remove: (c) => { n.className = has().filter((x) => x !== c).join(" "); }, contains: (c) => has().includes(c),
    toggle: (c, on = !api.contains(c)) => { if (on) api.add(c); else api.remove(c); return on; }}; return api; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "checked") this.checked = true;
    else if (k === "style") this.style.display = /display:\s*none/.test(v) ? "none" : "";
    else if (k.startsWith("data-")) this.dataset[k.slice(5)] = v; else this[k] = v; }
  hasAttribute(k) { return k in this.attrs; }
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
  get isConnected() { return false; }
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
const document = {createElement: (t) => new E(t), activeElement: null};
const window = {}, requestAnimationFrame = () => 0, svgEl = () => document.createElement("i");
class ResizeObserver { observe() {} }
const S = {values: {}, state: {}, fam: null};
const onValue = (r, v) => { S.values[r.key] = v; };
const shown = (n) => (n.nodeType === 3 ? n.textContent : n.hidden ? "" :
  n.childNodes.length ? n.childNodes.map(shown).join("") : n._t);
"""


def _node(steps, **data):
    """app.js's own parts in DOM's page - el, splitRow, the plan's badges (planCond, planTweaks) and a prompt slider's
    boxes (promptLines, moreHint, switchSub) -, `data` as consts, then `steps`, which log() what they find: the log."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    src = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    grab = lambda pat: re.search(pat, src, re.S).group(0)
    js = ('"use strict";\n' + DOM + grab(r"const el = \(tag.*?\n\};\n") + grab(r"const TWEAK_NUM = [^\n]*\n")
          + grab(r"function splitRow\(r, wrap\) \{.*?\n\}\n")
          + src[src.index("// how a section trains as badges"):src.index("// the Samples tab's prompt sets")]
          + "".join(f"const {k} = {json.dumps(x)};\n" for k, x in data.items())
          + "const LOG = [], log = (...x) => LOG.push(x.length > 1 ? x : x[0]);\n" + steps
          + "\nconsole.log(JSON.stringify(LOG));\n")
    run = subprocess.run(["node", "-"], input=js, capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _card(fam):
    """The Slider card's rows as form() builds them, by key, in order."""
    sec = next(s for s in form.form(get_family(fam))["sections"] if s["id"] == "slider")
    return {r["key"]: r for r in sec["rows"] if r}


def _look(r):
    """A row as the page draws it with the switch on (app.js rowLook)."""
    return {**r, **r.get("by_epoch", {})}


def test_the_switch_says_what_it_is_for():
    from fizgig.families.train import setup_parser
    tick = _card("krea2")["FAMILY_SLIDER_BY_EPOCH"]
    assert tick["label"] == launch.BY_EPOCH == "Change the prompts during the run (advanced)"
    assert tick["hint"].startswith("Use different words at different stages, so the dial learns only the change. ")
    # the old name nowhere the switch is named: the form, its notes with it off (lines, a line ending in epochs, a
    # push strength by epoch), Start and the trainer's --help
    desc = get_family("krea2")
    v = {**form.DEFAULTS, "FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
         "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
         "FAMILY_SLIDER_NEG": "frowning @2-3", "FAMILY_SLIDER_GUIDANCE": "3:4:2"}
    said = " ".join([json.dumps([form.form(desc), form.state(desc, v, "")["notes"]]),
                     *launch.slider_off_problems(desc, launch.slider_boxes(v), v["FAMILY_SLIDER_GUIDANCE"]),
                     " ".join(setup_parser().format_help().split())])
    assert "Train by epoch" not in said and said.count(launch.BY_EPOCH) >= 8


def test_more_is_on_the_modes_rows_only():
    # its switch (More only while it's on), the plan, Reuse practice pictures and every box's lines look; any other
    # row of any model's form, and the boxes' plain look, as before
    for d in training_families():
        for s in form.form(d)["sections"]:
            for r in filter(None, s["rows"]):
                mode = s["id"] == "slider" and launch.slider_picture_lines(d)
                assert r.get("more") == (MODE.get(r["key"]) if mode else None), (d.key, r["key"])
                assert r.get("by_epoch", {"more": True})["more"] is True, (d.key, r["key"])


def test_a_long_hint_shows_its_first_sentence_then_more():
    got = _node(r"""
      const h = moreHint("Prompts the dial must not change, e.g. a dog. Each makes a step longer. Empty = none.");
      const link = h.querySelector("a.hint-more-link");
      log(shown(h), link.attrs["aria-expanded"]);
      link.fire("click"); log(shown(h), link.attrs["aria-expanded"]);
      link.fire("click"); log(shown(h));
      log(shown(moreHint(" One sentence only. ")), !!moreHint("One sentence only.").querySelector("a"));
      log(shown(moreHint("First. Second.", true)));
    """)
    assert got == [
        ["Prompts the dial must not change, e.g. a dog. More", "false"],             # not cut at "e.g."
        ["Prompts the dial must not change, e.g. a dog. Each makes a step longer. Empty = none. Less", "true"],
        "Prompts the dial must not change, e.g. a dog. More",
        ["One sentence only.", False],
        "First. Second. Less"]


def test_the_modes_long_hints_open_on_a_short_first_sentence():
    hints = {}
    for fam in ("krea2", "moss_image"):
        card = _card(fam)
        for k, r in card.items():
            look = _look(r)
            for h in ("hint", "split_hint", "slide_hint") if look.get("more") else ():
                if look.get(h):
                    hints[f"{fam} {k} {h}"] = look[h]
        hints[f"{fam} tweaks"] = card["_slider_plan"]["tick"]["hint"]
    assert len(hints) == 2 * 11
    got = _node("for (const [k, t] of Object.entries(HINTS)) { const h = moreHint(t); "
                "log(k, shown(h), !!h.querySelector('a')); }", HINTS=hints)
    first = {}
    for k, text, more in got:
        assert more and text.endswith(". More") and len(text) <= 140, (k, text)
        first[k] = text[:-len(" More")]
    assert first["krea2 FAMILY_SLIDER_BY_EPOCH hint"] == ("Use different words at different stages, so the dial "
                                                          "learns only the change.")
    assert first["moss_image tweaks"] == first["krea2 tweaks"] == ("Gives each section below its own settings, in "
                                                                   "place of the run's.")
    assert first["krea2 FAMILY_SLIDER_KEEP hint"] == ("A picture description that comes back at a later epoch uses its "
                                                      "practice pictures again, instead of rendering them anew.")
    assert first["krea2 FAMILY_SLIDER_GUIDANCE split_hint"] == ("One push strength for the first epochs, then another "
                                                                "to the end.")


def test_the_push_strengths_hints_name_the_models_own_default():
    for fam, own in (("krea2", "3"), ("moss_image", "2")):
        push = _look(_card(fam)["FAMILY_SLIDER_GUIDANCE"])
        assert push["from_single"] and push["split"][0] == push["slide"][0] == own
        assert push["hint"].startswith("How hard the +1 and -1 ends are pushed apart, one number for the whole run - "
                                       f"this model's own default is {own}. ")
        # Split and Slide start from the number in Single: their hints name no number a box could contradict
        for k in ("split_hint", "slide_hint"):
            assert "start from the number in Single" in push[k] and not re.search(r"\d", push[k]), (fam, k)


def test_split_and_slide_start_from_the_number_in_single():
    push = _look(_card("krea2")["FAMILY_SLIDER_GUIDANCE"])               # its own: 3, easing to 2
    clean = {"key": "CLEAN", "label": "Clean-end share", "kind": "split", "default": "30", "suffix": "%",
             "hint": "Share.", "split": ["75", "5", "40"], "slide": ["60", "7", "30", "15", "2"]}    # MOSS-Image's
    got = _node(r"""
      const box = (r, v) => { S.values[r.key] = v; const w = splitRow(r, el("div", {}));
        const pick = (c) => { const b = w.querySelectorAll("input").find((i) => i.type === "radio" && i.value === c);
          b.checked = true; b.fire("change"); return S.values[r.key]; };
        const type = (k, x) => { const b = w.querySelectorAll("input.cad-num")[k]; b.value = x; b.fire("input");
          return S.values[r.key]; };
        return {pick, type}; };
      const p = box(PUSH, "4");
      log(p.pick("Split"), p.pick("Slide"), p.pick("Single"));
      log(p.type(0, "6"), p.pick("Split"), p.type(1, "5"), p.pick("Single"), p.type(0, "2"), p.pick("Split"));
      log(p.pick("Slide"), p.pick("Single"), p.type(0, ""), p.pick("Slide"));
      const c = box(CLEAN, "50");
      log(c.pick("Split"), c.pick("Slide"));
    """, PUSH=push, CLEAN=clean)
    assert got == [
        ["4:10:2.5", "4:10:2.5:15", "4"],                     # from 4: 4, then about two thirds of it (to a half)
        # Single 6: Split from it; numbers typed in Split stay theirs when Single changes again
        ["6", "6:10:4", "5:10:4", "6", "2", "5:10:4"],
        ["2:10:1.5:15", "2", "", "2:10:1.5:15"],             # Slide's untouched: from 2; no number in Single: as it was
        ["75:5:40", "60:7:30:15:2"]]                          # the Clean-end share's own, as before


def test_the_plan_says_the_rate_and_how_firmly_one_way():
    seg = {"from": 11, "to": 11, "push": "3", "push_set": False, "push_own": "3", "detail": "11", "detail_set": False,
           "detail_own": "11", "rate": 50, "rate_set": False, "rate_own": 50, "half": True, "firm": None,
           "firm_set": False, "firm_own": "1"}
    got = _node(r"""
      for (const s of [SEG, {...SEG, firm: "1.5"}]) log(planCond(s).filter(Boolean).map((b) => b.textContent));
      for (const s of [SEG, {...SEG, firm: "1.5"}]) {        // Adjust each section on
        const out = planTweaks(s, {on: true, detail: true, rate_off: ""}, () => {});
        const sel = out.map((b) => b.querySelector("select")).find(Boolean);
        log(out.map((b) => [b.childNodes[0].textContent, b.classList.contains("plan-tweak-off"), b.title || "",
          b.querySelectorAll("input").map((i) => [i.value, i.hasAttribute("readonly")])]),
          sel.children.find((o) => o.value === sel.value).textContent);
      }
      log(scheduleOwn(SEG, "rate"), scheduleOwn({...SEG, rate_own: 100}, "rate"));
    """, SEG=seg)
    assert got[:2] == [["push 3", "fine detail 11%", "rate 50%"],
                       ["push 3", "fine detail 11%", "rate 50%", "how firmly (leave alone) 1.5"]]
    off, on = got[2:4]
    assert off == [[["push ", False, "", [["3", False]]], ["fine detail ", False, "", [["11", False]]],
                    ["rate ", False, "", []],
                    # in every section: greyed and read-only where no Leave these alone line is on
                    ["how firmly (leave alone) ", True, "No Leave these alone line on these epochs", [["", True]]]],
                   "50%"]                                   # the rate control says 50%, as the badge does
    assert on[0][3] == ["how firmly (leave alone) ", False, "", [["1.5", False]]]
    assert got[4] == ["50%", "Normal"]                     # where ↺ and the tooltips say it comes from


def test_reuse_practice_pictures_sits_under_the_plan():
    card = _card("krea2")
    keys = list(card)
    assert keys.index("FAMILY_SLIDER_KEEP") == keys.index("_slider_plan") + 1
    assert card["FAMILY_SLIDER_KEEP"]["label"] == "Reuse practice pictures when a description comes back"
    assert card["FAMILY_SLIDER_KEEP"]["kind"] == "check" and card["FAMILY_SLIDER_KEEP"]["default"] is True


def test_the_boxes_headings_share_one_style_and_the_switch_shows_more_only_while_on():
    css = open(os.path.join(STATIC, "style.css"), encoding="utf-8").read()
    rule = lambda sel: re.search(re.escape(sel) + r"\s*\{([^}]*)\}", css).group(1)
    prop = lambda body, name: re.search(r"(?:^|[;\s])" + name + r"\s*:\s*([^;]+);", body).group(1).strip()
    label = rule('.section-card[id^="sec-"] .form-row .field-label')                          # boxes 1, 2, 3 and 5
    title = rule('.section-card[id^="sec-"] .form-row.kind-cadence[data-look="by_epoch"] .cad-title')    # box 4
    assert [prop(title, p) for p in ("color", "font-size")] == [prop(label, p) for p in ("color", "font-size")]
    assert prop(rule(".kind-cadence .cad-title"), "font-weight") == prop(label, "font-weight") == "600"
    # the switch's hint: as every row's (two lines, the rest on a click) while it's off, the first sentence and More
    # while it's on
    assert re.search(r"\.form-row\.hint-more-on:has\(input:checked\) > \.row-hint, \.form-row\.hint-more-on:not\("
                     r":has\(input:checked\)\) > \.hint-more \{\s*display: none; \}", css)


def test_the_slider_heads_style_reaches_no_other_prompt_head():
    # the slider's heads with a switch are divs (app.js planRow, promptLines): their rules say div.prompt-head, so the
    # Samples previews grid's th.prompt-head keeps master's rules only - with the switch off, that table as before
    css = re.sub(r"/\*.*?\*/", "", open(os.path.join(STATIC, "style.css"), encoding="utf-8").read(), flags=re.S)
    split = lambda s: [x.strip() for x in re.split(r",(?![^()]*\))", s)]      # not at the commas inside :is(...)
    heads = [x for head in re.findall(r"([^{}]+)\{", css) for x in split(head) if "prompt-head" in x]
    master = {".prompt-head .preview-prompt", ".prompt-grid th.prompt-head", ".prompt-grid th.prompt-head b"}
    assert master <= set(heads), heads
    assert [x for x in heads if x not in master and not re.search(r"(?<![\w-])div\.prompt-head\b", x)] == [], heads
