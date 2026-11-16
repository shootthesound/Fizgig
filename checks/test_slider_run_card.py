"""The by-epoch switch: practice pictures kept by what the description is (its text), and the plan's "again" as the
trainer's cache normally keeps them; MOSS-Image's Clean-end share read by one function (its driver and the plan);
Resume takes the plan's own settings from the form only while it shows the paused run's setup; a Run card request the
trainer can't take tried once more, then waiting for a new Apply; a push strength alone by epoch kept short in the
metadata; lines with no picture line on the first epoch refused before anything loads; and the Run card's boxes as
numbered lines with their own epochs and its push strength as Single, Split or Slide (app.js slLines, slPush) - with
the tick off, the panel as before."""
import json
import os
import re
import shutil
import subprocess

import pytest

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")
RUN = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8",
       "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_SLIDER_BASE": "a man,", "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-",
       "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_ANCHORS": "", "FAMILY_SLIDER_BY_EPOCH": True}


def _values(**values):
    return {**form.DEFAULTS, "LORA_NAME": "t", "MAX_TRAIN_EPOCHS": "8", "FAMILY_KIND": "slider",
            "FAMILY_SLIDER_SOURCE": "Prompts", "FAMILY_SLIDER_BASE": "a photo of a man,",
            "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning", "FAMILY_SLIDER_GUIDANCE": "3",
            "FAMILY_SLIDER_BY_EPOCH": True, **values}


def _grab(*patterns):
    src = open(APP, encoding="utf-8").read()
    return "".join(re.search(p, src, re.S).group(0) for p in patterns)


def _node(js):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    run = subprocess.run(["node", "-"], input=js, capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


# ---- practice pictures, kept by what the description is -------------------------------------------------------------
def test_the_plans_again_is_what_the_trainers_cache_normally_keeps(monkeypatch):
    # by how many descriptions it keeps - not by its 1 GB, nor a Resume, which starts with none: "normally", it says
    from fizgig.families.train import _BankCache, _bank_key
    desc = get_family("krea2")
    base = "A, @1-2\nB, @3-3\nC, @4-4\nA, @5-5\nB, @6-6\nD, @7-7\nE, @8-8\nA, @9-9\nC, @10-10"
    said = {}
    for keep in (0, 2, 3, 32):
        monkeypatch.setattr(launch, "SLIDER_BANKS_KEPT", keep or 32)
        plan = form.state(desc, _values(MAX_TRAIN_EPOCHS="10", FAMILY_SLIDER_BASE=base,
                                        FAMILY_SLIDER_KEEP=bool(keep)), "")["slider_plan"]
        cache, took = _BankCache(keep=keep), []
        for n, seg in enumerate(plan):          # as the trainer goes: a new description looks for its pictures by text
            if n and not seg["new_pictures"]:
                continue
            key = _bank_key(launch.slider_join([x["text"] for x in seg["lines"]["base"]]), None, ("plain", 28, 4.0),
                            None, 768, 16, 42)
            kept = cache.get(key)
            if kept is None:
                cache.put(key, [seg["from"]])   # rendered at this epoch
            took.append((seg["from"], kept[0] if kept else None))
        assert [(s["from"], s["again"]) for n, s in enumerate(plan) if not n or s["new_pictures"]] == took, keep
        said[keep] = [again for _e, again in took]
    assert said[0] == [None] * 9                                        # unticked: every change renders anew
    assert said[32] == [None, None, None, 1, 3, None, None, 1, 4]
    # 3 kept: A at 9 had gone (the least recently used of A, B, D, E), so it renders again
    assert said[3] == [None, None, None, 1, 3, None, None, None, None]


# ---- MOSS-Image's Clean-end share: one function ----------------------------------------------------------------------
def test_moss_images_clean_end_share_is_read_by_one_function():
    from fizgig.moss_image import driver as moss_driver
    from fizgig.moss_image.clean_share import clean_share
    assert moss_driver.clean_share is form.clean_share is clean_share
    moss = get_family("moss_image")
    key = next(o.key for o in moss.options if str(o.tokens).startswith("lownoise_pct="))
    m = moss_driver.MOSSImageDriver.__new__(moss_driver.MOSSImageDriver)
    for text in ("11:7:40:8:3", "75:6:40:10:3", "30:5:60", "75:6:40:10", "40", "60%", ""):
        m.options = {"lownoise_pct": text}
        for cycle in (0, 3):                     # a LoRA's epochs, and a fine-tune's rotations of 3 epochs
            m._ft_cycle = cycle
            assert all(m._share(e) == clean_share(text, e, cycle) for e in range(1, 31)), (text, cycle)
        assert all(form.clean_share_at(moss, {key: text}, e) == (clean_share(text, e) if text else None)
                   for e in range(1, 31)), text
    assert [clean_share("75:2:40", e, 3)[0] for e in range(1, 10)] == ["75"] * 6 + ["40"] * 3    # by rotation
    m.options, m._ft_cycle = {"lownoise_pct": "75:5"}, 0
    with pytest.raises(RuntimeError, match="Clean-end share '75:5': give share:epochs:share"):
        m._share(3)
    assert form.clean_share_at(moss, {key: "75:5"}, 3) is None and form.clean_share_at(moss, {key: "x"}, 3) is None
    assert form.clean_share_at(moss, {key: "11:7:40:8:x"}, 3) is None


# ---- Resume: the plan's own settings from the form only for the run's setup ------------------------------------------
def test_resume_takes_the_plans_own_settings_only_while_the_form_shows_the_run():
    from fizgig.web import server
    run = {"family": "krea2", "values": {**RUN, "FAMILY_SLIDER_TWEAKS": "5-8:push=2", "FAMILY_SLIDER_TWEAKS_ON": True},
           "state": "paused"}
    same = {**RUN, "FAMILY_SLIDER_POS": " smiling @1-4\n\nlaughing @5- \n", "SAMPLE_SEED": "7",
            "FAMILY_SLIDER_TWEAKS": "", "FAMILY_SLIDER_TWEAKS_ON": False}     # its boxes, blank lines aside
    other = {**same, "FAMILY_SLIDER_POS": "grinning"}                        # another setup of the same model
    cases = [["run", "krea2", other], ["draft", "krea2", same], ["draft", "krea2", other],
             ["draft", "moss_image", same]]
    js = (_grab(r"const SIZE_KEYS = [^\n]*\n", r"const RESUME_KEYS = \[.*?\];[^\n]*\n",
                r"const RESUME_TWEAK_KEYS = [^\n]*\n",
                r"const SLIDER_BOX_KEYS = [^\n]*\n", r"function resumeTweaksGo\(.*?\n\}\n",
                r"async function resumeClick\(\) \{.*?\n\}\n")
          + "const $ = () => ({style: {}}), start = (path, body) => body.samples; let KIND = '';\n"
          + "const ctxKind = () => KIND; const S = {};\n"
          + f"(async () => {{ const out = []; for (const [kind, family, values] of {json.dumps(cases)}) {{\n"
          + f"  KIND = kind; Object.assign(S, {{run: {json.dumps(run)}, ctx: {{kind}}, family, values}});\n"
          + "  const s = await resumeClick(); out.push([s.FAMILY_SLIDER_TWEAKS ?? null, s.SAMPLE_SEED ?? null]); }\n"
          + "console.log(JSON.stringify(out)); })();\n")
    assert _node(js) == [
        ["", "7"],              # the run's own view: its plan as the form has it now - emptied too
        ["", "7"],              # your setup, the same model and the same boxes: the same
        [None, "7"],            # another setup: the run keeps its own (the previews still go)
        [None, None]]           # another model: nothing from the form, as before
    # the server takes what the page sends: not sent, the run keeps its own, even when the form's were empty
    v = {"FAMILY_SLIDER_BY_EPOCH": True, "FAMILY_SLIDER_TWEAKS": "5-8:push=2", "FAMILY_SLIDER_TWEAKS_ON": True}
    assert server._resume_slider_values(v, {"SAMPLE_SEED": "7"}) == v
    assert server._resume_slider_values(v, {"FAMILY_SLIDER_TWEAKS": "", "FAMILY_SLIDER_TWEAKS_ON": False}) == {
        **v, "FAMILY_SLIDER_TWEAKS": "", "FAMILY_SLIDER_TWEAKS_ON": False}


# ---- a Run card request the trainer can't take ----------------------------------------------------------------------
def test_a_request_it_cant_take_is_tried_once_more_then_waits_for_apply(tmp_path):
    from fizgig.families.train import SLIDER_TRIES, _slider_tries
    from fizgig.web import server
    assert SLIDER_TRIES == 2                     # at the boundary it came to, and once more at the next
    assert _slider_tries(None, 5.0) == 0 and _slider_tries({"at": 5.0, "n": 1}, 5.0) == 1
    assert _slider_tries({"at": 5.0, "n": 2}, 6.0) == 0           # a new Apply writes the file anew: tried anew
    asked = {"base": "a man,", "pos": "smiling @1-4\ngrinning @5-", "neg": "frowning", "guidance": "3", "anchors": []}
    live, applied = tmp_path / server.SLIDER_LIVE, tmp_path / server.SLIDER_LIVE_APPLIED
    live.write_text(json.dumps(asked), encoding="utf-8")
    at = os.path.getmtime(live)
    run = {"family": "krea2", "values": {**RUN, "FAMILY_SLIDER_POS": asked["pos"]}, "output_dir": str(tmp_path),
           "progress": {"epoch": 6}, "slider_start": {k: RUN.get(k) for k in server._SLIDER_LIVE_KEYS.values()}}
    failed = {"epoch": 5, "failed": "OutOfMemoryError: CUDA out of memory.", "at": 2.0, "asked": asked, "took": None,
              "retry": False, "request_at": at}
    applied.write_text(json.dumps(failed), encoding="utf-8")
    st = server._slider_live_status(run)
    assert st["failed"] and st["retry"] is False and not st["waiting"] and st["epoch"] == 5
    applied.write_text(json.dumps({**failed, "retry": True}), encoding="utf-8")
    assert server._slider_live_status(run)["retry"] is True
    older = {k: x for k, x in failed.items() if k not in ("retry", "request_at")}     # an older trainer: every epoch
    applied.write_text(json.dumps(older), encoding="utf-8")
    assert server._slider_live_status(run)["retry"] is True
    applied.write_text(json.dumps(failed), encoding="utf-8")
    os.utime(live, (at + 5, at + 5))                      # Apply again, the same values: tried again, so it waits
    st = server._slider_live_status(run)
    assert st["failed"] is None and st["retry"] is None and st["waiting"]
    # the card says which, in plain words (app.js sliderLiveDraw)
    status = {"base": "a man,", "pos": "smiling", "neg": "frowning", "guidance": "3", "anchors": "", "now": None,
              "by_epoch": True, "epoch": 5, "waiting": False, "failed": "OutOfMemoryError: CUDA out of memory."}
    js = ("const nodes = {}; const $ = (id) => nodes[id] || (nodes[id] = {style: {}, value: '', textContent: ''});"
          "const S = {sl: {shown: null, edited: false}}; const SL_FIELDS = {base: 'slBase'};"
          "const sliderLiveLook = () => false;\n" + _grab(r"function sliderLiveDraw\(sl\) \{.*?\n\}\n")
          + f"console.log(JSON.stringify([true, false].map((retry) => (sliderLiveDraw({{...{json.dumps(status)}, "
          "retry}), $('slSay').textContent))));")
    assert _node(js) == [
        "Could not take them at the end of epoch 5 (OutOfMemoryError: CUDA out of memory.) - training goes on with its "
        "own lines; they're tried again at the next epoch.",
        "Could not take them at the end of epoch 5 (OutOfMemoryError: CUDA out of memory.) - training goes on with its "
        "own lines; they failed twice, so they wait: press Apply to try them again."]


def test_a_push_strength_alone_by_epoch_keeps_the_metadata_short():
    from fizgig.families.train import _history_entry
    assert _history_entry(4, 2.5) == {"epoch": 4, "guidance": 2.5}         # a Split, a Slide or a section's own
    assert _history_entry(4, 2.5, ["a,", "a, b", "a, c"], ["a dog"]) == {    # a Run card request: its prompts too
        "epoch": 4, "prompts": ["a,", "a, b", "a, c"], "guidance": 2.5, "anchors": ["a dog"]}


def test_lines_with_no_picture_line_on_the_first_epoch_are_refused():
    # train_family checks --slider_line before any model loads (_slider_line_start): here with nothing loaded at all
    from fizgig.families.train import _slider_line_start
    krea = get_family("krea2")
    start = lambda lines, first=1: _slider_line_start(krea, lines, True, 8, first)
    with pytest.raises(RuntimeError, match=r"no picture line is on at epoch 1 - every epoch needs a picture, a \+1 "):
        start([("base", "a man, @3-"), ("pos", "smiling"), ("neg", "frowning")])
    with pytest.raises(RuntimeError, match=r"no picture / -1 line is on at epoch 1"):
        start([("base", "a man, @3-"), ("pos", "smiling"), ("neg", "frowning @2-")])
    boxes, *at = start([("base", "a man, @3-"), ("pos", "smiling"), ("neg", "frowning")], first=4)   # a Resume at 4
    assert at == ["a man,", "smiling", "frowning", []] and boxes["base"] == "a man, @3-"
    with pytest.raises(RuntimeError, match=r"--slider_line takes base, pos, neg or anchor \(not bse\)"):
        start([("bse", "a man,")])
    with pytest.raises(RuntimeError, match="need --slider_by_epoch"):
        _slider_line_start(krea, [("base", "a man,")], False, 8, 1)
    with pytest.raises(RuntimeError, match="for models that train on pictures"):
        _slider_line_start(get_family("acestep15xl"), [("base", "a song,")], True, 8, 1)


# ---- the Run card's boxes --------------------------------------------------------------------------------------------
# a small page in node for app.js's Run card: the parts of the DOM its boxes use
DOM = r"""
class T { constructor(t) { this.nodeType = 3; this.textContent = String(t); this.parentNode = null; } }
const one = (e, s) => (s[0] === "." ? e.className.split(" ").includes(s.slice(1)) : e.tagName === s.toUpperCase());
class E {
  constructor(tag) { Object.assign(this, {nodeType: 1, tagName: tag.toUpperCase(), childNodes: [], parentNode: null,
    className: "", dataset: {}, on: {}, value: "", checked: false, title: "", _t: "",
    style: {setProperty(k, v) { this[k] = v; }}}); }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get classList() { const e = this, has = () => e.className.split(" ").filter(Boolean);
    return {contains: (c) => has().includes(c), toggle: (c, on = !has().includes(c)) => {
      e.className = [...has().filter((x) => x !== c), ...(on ? [c] : [])].join(" "); return on; }}; }
  get firstChild() { return this.childNodes[0] || null; }
  get textContent() { return this.childNodes.length ? this.childNodes.map((c) => c.textContent).join("") : this._t; }
  set textContent(t) { this.replaceChildren(); this._t = String(t); }
  setAttribute(k, v) { if (k === "checked") this.checked = true; else this[k] = v; }
  removeAttribute(k) { this[k] = ""; }
  put(n, at) { n = typeof n === "string" ? new T(n) : n; n.parentNode = this; this.childNodes.splice(at, 0, n); }
  append(...ns) { ns.forEach((n) => { if (n.parentNode) n.remove(); this.put(n, this.childNodes.length); }); }
  after(n) { if (n.parentNode) n.remove(); this.parentNode.put(n, this.parentNode.childNodes.indexOf(this) + 1); }
  replaceWith(n) { const p = this.parentNode, i = p.childNodes.indexOf(this); this.remove(); p.put(n, i); }
  remove() { const p = this.parentNode; if (p) { p.childNodes.splice(p.childNodes.indexOf(this), 1); }
    this.parentNode = null; }
  replaceChildren(...ns) { [...this.childNodes].forEach((c) => { c.parentNode = null; }); this.childNodes = [];
    this._t = ""; this.append(...ns); }
  querySelectorAll(sel) {
    if (sel.startsWith(":scope > ")) return this.children.filter((c) => one(c, sel.slice(9)));
    return this.children.flatMap((c) => [...(one(c, sel) ? [c] : []), ...c.querySelectorAll(sel)]); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  addEventListener(t, f) { (this.on[t] = this.on[t] || []).push(f); }
  fire(t) { (this.on[t] || []).forEach((f) => f({target: this})); }
  focus() { document.activeElement = this; }
}
const document = {createElement: (t) => new E(t)};
// the panel as index.html has it
const panel = new E("div"), find = (n, id) => (n.id === id ? n : n.children.map((c) => find(c, id)).find(Boolean));
panel.id = "sliderLive";
const $ = (id) => find(panel, id);
const box = (id, tag, title) => { const b = new E(tag); Object.assign(b, {id, title: title || ""});
  if (tag === "input") b.type = "text"; return b; };
const row = (tag, label, ...kids) => { const r = new E(tag), s = new E("span"); r.className = "slider-live-row";
  s.textContent = label; r.append(s, ...kids); panel.append(r); };
row("label", "What it is", box("slBase", "input", "The base: each end's words are added after it with a space, as at " +
  "the start"));
row("label", "+1 end adds", box("slPos", "input"));
row("label", "−1 end adds", box("slNeg", "input"));
row("label", "Leave alone", box("slAnchors", "textarea", "Prompts the dial must not change, one a line (empty = " +
  "none)"));
const actions = new E("div"), push = box("slGuidance", "input");
actions.className = "slider-live-actions"; push.className = "w-port";
actions.append(push, box("slSay", "span"));
row("div", "Push strength", actions, box("slPushNote", "div"));
panel.append(box("slNow", "div"));
const S = {sl: {shown: null, edited: false}};
// what the panel shows of each box: its element, words and look, and the boxes of lines beside it
const shape = () => Object.values(SL_FIELDS).map((id) => { const b = $(id);
  return [id, b.tagName, b.type || "", b.title || "", b.style.display || "", b.className,
    b.parentNode.querySelectorAll(":scope > .sl-edit").length]; });
const lines = (id) => $(id).parentNode.querySelector(".sl-edit").querySelectorAll(".prompt-line").map((it) =>
  [it.firstChild.textContent, ...it.querySelectorAll("input").map((i) => i.value)]);
const typed = (input, text) => { input.value = text; input.fire("input"); };
"""


def test_the_run_card_shows_lines_with_their_epochs_and_the_push_as_single_split_or_slide():
    js = ('"use strict";\n' + DOM
          + _grab(r"const el = \(tag.*?\n\};\n", r"const SLIDER_AT = [^\n]*\n",
                  r"function epochsText\(eps\) \{.*?\n\}\n", r"function epochColour\(key\) \{.*?\n\}\n",
                  r"function lineParts\(text\) \{.*?\n\}\n", r"function lineSpan\(\[a, b, n\]\) \{.*?\n\}\n",
                  r"function lineKey\(\[a, b, n\]\) \{.*?\n\}\n",
                  r"function lineBars\(lines, epochs, joins, name\) \{.*?\n\}\n",
                  r"function showBar\(it, b, run, at = 0\) \{.*?\n\}\n", r"const SL_FIELDS = [^\n]*\n",
                  r"const SL_BASE_TITLE = [^\n]*\n", r"function endTag\(text, k\) \{.*?\n\}\n",
                  r"function sliderLiveLook\(on\) \{.*?\n\}\n", r"function slLines\(k, box\) \{.*?\n\}\n",
                  r"function slPush\(box\) \{.*?\n\}\n", r"function sliderLiveDraw\(sl\) \{.*?\n\}\n")
          + r"""
const out = {}, before = shape();
const sl = {base: "a photo of a woman, @-2\nwearing a red coat, @3-35/2", pos: "smiling @1-35\nlaughing @36-",
  neg: "frowning", guidance: "2:10:1.5:15", anchors: "a photo of a man @-4", now: null, by_epoch: true, epoch: null,
  waiting: false, failed: null, push_note: null};
// off: one line each, as before
sliderLiveDraw({...sl, by_epoch: false, base: "a man,", pos: "smiling", neg: "frowning", guidance: "3", anchors: ""});
out.off = [JSON.stringify(shape()) === JSON.stringify(before), $("slPos").value];
sliderLiveDraw(sl);                               // on: numbered lines with their epochs, Single / Split / Slide
out.on = [shape().map((s) => [s[1], s[4], s[6]]), lines("slBase"), lines("slPos"), lines("slAnchors")];
// the +1 and -1 ends told apart: "+1" / "−1" in the label as a tag in its end's colour, its box of lines edged in it
const cap = (id) => $(id).parentNode.querySelector(":scope > span");
out.tags = [cap("slPos").children.map((c) => [c.className, c.textContent]), cap("slNeg").textContent,
  $("slNeg").parentNode.querySelector(".prompt-lines").className];
const ed = $("slGuidance").parentNode.querySelector(".sl-edit"), radio = (m) => ed.querySelectorAll("input")
  .find((i) => i.type === "radio" && i.value === m);
const shown = () => ed.querySelectorAll(".sl-push-line").find((l) => l.style.display !== "none")
  .querySelectorAll("input").map((i) => i.value);
out.push = [["Single", "Split", "Slide"].find((m) => radio(m).checked), shown()];
// edits: a line's last epoch, a line added and one removed, the push strength's mode and numbers
const second = $("slBase").parentNode.querySelectorAll(".prompt-line")[1];
typed(second.querySelectorAll("input")[2], "30x");
out.edit = [$("slBase").value, S.sl.edited];
const pos = $("slPos").parentNode.querySelector(".sl-edit");
S.sl.edited = false;
pos.querySelector(".sl-add").fire("click");
out.added = S.sl.edited;                          // an edit too: the run's values coming in don't draw over the line
const third = pos.querySelectorAll(".prompt-line")[2];
typed(third.querySelectorAll("input")[0], "grinning"); typed(third.querySelectorAll("input")[1], "5");
typed(third.querySelectorAll("input")[3], "3");
pos.querySelectorAll(".prompt-del")[0].fire("click");
out.lines = [$("slPos").value, lines("slPos").map((l) => l[0])];
radio("Split").checked = true; radio("Split").fire("change");
const split = [$("slGuidance").value, shown()];
typed(ed.querySelectorAll(".sl-push-line")[1].querySelectorAll("input")[0], "3");
split.push($("slGuidance").value);
radio("Single").checked = true; radio("Single").fire("change"); split.push($("slGuidance").value);
radio("Split").checked = true; radio("Split").fire("change"); split.push($("slGuidance").value);
out.split = split;
// the run's text again (Undo edits), and a push strength in no mode: as typed, in Single
S.sl = {shown: null, edited: false};
sliderLiveDraw({...sl, guidance: "3:10"});
out.undo = [$("slBase").value, lines("slBase").length, $("slGuidance").value, shown(), radio("Single").checked];
sliderLiveDraw({...sl, by_epoch: false, base: "a man,", pos: "smiling", neg: "frowning", guidance: "3", anchors: ""});
out.back = [JSON.stringify(shape()) === JSON.stringify(before), $("slBase").title, typeof $("slAnchors")._fill,
  cap("slPos").textContent, cap("slPos").children.length];
out.barsOff = typeof $("slAnchors")._bars;
// on again: each line's bar, as the Slider card's (lineBars, showBar) - across the run's own epochs (8), solid on its
// epochs, dashes on those it shares with a line above it, faint between them every few epochs, in its colour - and a
// thin mark at the epoch in progress: [pieces [kind, left, width], marks, title, --c, its colour down its edge]
const bars = (id) => $(id).parentNode.querySelector(".sl-edit").querySelectorAll(".prompt-line").map((it) => {
  const bar = it.querySelector(".prompt-bar");
  return [bar.querySelectorAll("i").map((p) => [p.className, p.style.left, p.style.width]),
    bar.querySelectorAll(".prompt-at").map((m) => m.style.left), bar.title, it.style["--c"],
    it.className.split(" ").includes("prompt-timed")]; });
S.run = {progress: {total_epochs: 8, epoch: 3}};
const at5 = {...sl, pos: "smiling @1-4\nlaughing @3-6\ngrinning @2-/3", anchors: "a dog @2-3\na cat @3-4",
  now: {epoch: 5, pos: "a man, laughing", neg: "a man, frowning", anchors: [], push: null, next: null}};
sliderLiveDraw(at5);                                     // the mark: the epoch it trains on now, 5
out.bars = [bars("slPos"), bars("slAnchors").map((b) => [b[0], b[2]]), epochColour("1-4"), epochColour("3-6"),
  epochColour("2-/3")];
sliderLiveDraw({...at5, now: null, epoch: 6});           // no now: the run's own epoch in progress, 3
out.mark = [bars("slPos")[0][1]];
S.run = {values: {MAX_TRAIN_EPOCHS: "8"}};                 // no progress yet (its Max Train Epochs): no mark - never
sliderLiveDraw({...at5, now: null, epoch: 2});           // the epoch after the last change it took
out.mark.push(bars("slPos")[0][1]);
// drawn again as you edit: smiling on epochs 1-2 (laughing shares none with it now), and a new line, grey
const smiling = $("slPos").parentNode.querySelectorAll(".prompt-line")[0];
typed(smiling.querySelectorAll("input")[2], "2");
$("slPos").parentNode.querySelector(".sl-add").fire("click");
const drawn = bars("slPos");
out.redraw = [drawn[0][0], drawn[1][0], drawn[1][2], drawn[3]];
console.log(JSON.stringify(out));
""")
    out = _node(js)
    assert out["off"] == [True, "smiling"]                            # off: the plain boxes, untouched
    # on: each box's text hidden (base, +1, -1, push, anchors), edited beside it
    assert out["on"][0] == [["TEXTAREA", "none", 1], ["TEXTAREA", "none", 1], ["TEXTAREA", "none", 1],
                            ["INPUT", "none", 1], ["TEXTAREA", "none", 1]]
    assert out["on"][1:] == [[["1", "a photo of a woman,", "", "2", ""], ["2", "wearing a red coat,", "3", "35", "2"]],
                             [["1", "smiling", "1", "35", ""], ["2", "laughing", "36", "", ""]],
                             [["1", "a photo of a man", "", "4", ""]]]
    assert out["push"] == ["Slide", ["2", "10", "1.5", "15"]]
    assert out["tags"] == [[["end-tag end-pos", "+1"]], "−1 end adds", "prompt-lines end-zone end-neg"]
    assert out["edit"] == ["a photo of a woman, @-2\nwearing a red coat, @3-30/2", True]       # numbers only
    assert out["added"] is True                                         # + Add a line: an edit, kept as the run moves
    assert out["lines"] == ["laughing @36-\ngrinning @5-/3", ["1", "2"]]
    assert out["split"] == ["2:10:1.5", ["2", "10", "1.5"], "3:10:1.5", "2", "3:10:1.5"]     # each mode its own
    assert out["undo"] == ["a photo of a woman, @-2\nwearing a red coat, @3-35/2", 2, "3:10", ["3:10"], True]
    assert out["back"] == [True, "The base: each end's words are added after it with a space, as at the start",
                           "undefined", "+1 end adds", 0]                      # off again: the label as before
    assert out["barsOff"] == "undefined"                                # ... and no bars to draw
    # each line's bar, as the Slider card's: [pieces [kind, left, width], marks, title, --c, its colour down its edge]
    pos, anchors, c14, c36, c2n = out["bars"]
    assert [b[0] for b in pos] == [[["", "0%", "50%"]], [["dash", "25%", "25%"], ["", "50%", "25%"]],   # 3, 4: shared
                                   [["dash", "12.5%", "12.5%"], ["off", "25%", "25%"], ["dash", "50%", "12.5%"],
                                    ["off", "62.5%", "25%"], ["", "87.5%", "12.5%"]]]        # every 3: 2, 5 and 8
    assert [b[1] for b in pos] == [["56.25%"]] * 3                      # the mark: now's epoch 5 of 8, not progress's
    assert [b[2] for b in pos] == ["On for epochs 1–4 of the run's 8. Shares epochs with line 2, 3",
                                   "On for epochs 3–6 of the run's 8. Shares epochs with line 1, 3",
                                   "On for epochs 2–8 of the run's 8, every 3: 2, 5, 8. Shares epochs with line 1, 2"]
    assert [b[3:] for b in pos] == [[c14, True], [c36, True], [c2n, True]] and c14.startswith("hsl(")
    assert anchors == [[[["", "12.5%", "25%"]], "On for epochs 2–3 of the run's 8"],       # never joined: no dashes
                       [[["", "25%", "25%"]], "On for epochs 3–4 of the run's 8"]]
    assert out["mark"] == [["31.25%"], []]          # no now: the run's epoch 3 in progress; neither: none (not 2 + 1)
    assert out["redraw"] == [[["", "0%", "25%"]], [["", "25%", "50%"]],
                             "On for epochs 3–6 of the run's 8. Shares epochs with line 3",
                             [[["", "0%", "100%"]], [], "No words yet, so training skips this line. On for the whole "
                              "run: epochs 1–8 (empty epoch boxes follow Max Train Epochs)", "var(--text-muted)",
                              False]]                                  # + Add a line: no words yet, grey
    # Apply sends the boxes' text, read as the trainer reads it
    assert launch.slider_line_parts("wearing a red coat, @3-30/2") == ("wearing a red coat,", 3, 30, 2)
    assert launch.slider_line_parts("grinning @5-/3") == ("grinning", 5, None, 3)
    assert launch.slider_push_problems("3:10:1.5", 20) == [] and launch.slider_push_at("3:10:1.5", 11) == 1.5
