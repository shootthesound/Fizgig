"""Studio in two tabs (#395): Studio holds one model at a time, so a load or a primary LoRA swap in one tab is a new
load to every other tab. The other tab says so (its model picker still shows its own setup) and its Render loads its
own setup again instead of rendering on the other tab's model or LoRA."""
import json
import os
import shutil
import subprocess

import pytest

from fizgig.web import studio

APP = os.path.join(os.path.dirname(studio.__file__), "static", "app.js")


def test_a_primary_swap_is_a_new_load_to_other_tabs(monkeypatch):
    monkeypatch.setattr(studio, "_S", dict(studio._S, load_id=4, load_by="queue"))
    monkeypatch.setattr(studio, "_send", lambda cmd: None)
    assert studio.primary("") == 5
    st = studio.status()
    assert st["load_id"] == 5 and st["load_by"] == "page"


def test_who_loaded_is_known(monkeypatch):
    monkeypatch.setattr(studio, "_S", dict(studio._S))
    studio.set_load_key('{"a": 1}', page=True)
    assert studio.status()["load_by"] == "page"
    studio.set_load_key('{"a": 2}')                       # a Studio queue job's load
    assert studio.status()["load_by"] == "queue"


def test_the_other_tab_says_so_and_never_swaps_on_a_model_it_didnt_load():
    js = open(APP, encoding="utf-8").read()
    poll = js[js.index("async function studioPollOnce"):]
    poll = poll[:poll.index("\n}\n")]
    assert 'S.st.foreign && st.load_by === "page"' in poll and "Another Studio tab" in poll
    # the tab's own swap (Render with another primary, Unload primary) takes the new load id as its own
    swap = js[js.index("async function studioSwapPrimary"):]
    swap = swap[:swap.index("\n}\n")]
    assert "S.st.myLoad = r.load_id" in swap
    unload = js[js.index('$("stPrimaryUnload").addEventListener'):]
    unload = unload[:unload.index("\n  });\n")]
    assert "if (S.st.foreign)" in unload and "S.st.myLoad = r.load_id" in unload
    assert unload.index("if (S.st.foreign)") < unload.index('api("/api/studio/primary"')


# studioPollOnce run in node on stub page elements: each status is one poll's reply; returns what the tab drew
_POLL = r"""
const src = %s, polls = %s;
const els = {};
const el = () => ({style: {}, dataset: {}, value: "", checked: false, textContent: "", disabled: false, src: "",
  classList: {toggle() {}, contains() { return false; }, add() {}, remove() {}},
  parentElement: {classList: {toggle() {}}}, removeAttribute() {}, load() {}});
const keep = new Set(["Promise", "JSON", "Math", "Object", "String", "Number", "Array", "Set", "Map", "Boolean",
  "encodeURIComponent", "setTimeout", "clearTimeout", "console", "undefined", "RegExp", "Date", "isNaN"]);
const scope = new Proxy({
  S: {st: {shown: 3, blocks: {}, primaryBlocks: new Set(), donorBlocks: new Set()}, stq: null},
  $: (id) => els[id] || (els[id] = el()), TOKEN: "t", document: {querySelector: () => null},
  api: async (u) => u === "/api/studio/status" ? polls.shift() : {},
}, {has: (t, k) => typeof k === "string" && !keep.has(k), get: (t, k) => k in t ? t[k] : (() => false)});
const run = new Function("scope", "with (scope) { " + src + "; return studioPollOnce; }")(scope);
(async () => {
  const out = [];
  while (polls.length) {
    await run();
    out.push({status: els.stStatus.textContent, base: els.stBase ? els.stBase.src : "", shown: scope.S.st.shown,
              cancel: els.stCancel.disabled, foreign: !!scope.S.st.foreign});
  }
  console.log(JSON.stringify(out));
})();
"""


def _polls(*replies):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    js = open(APP, encoding="utf-8").read()
    fn = js[js.index("async function studioPollOnce"):]
    fn = fn[:fn.index("\n}\n") + 2]
    base = {"state": "ready", "rendering": False, "progress": None, "status": "", "media": {}, "act": {},
            "pending": False, "primary_blocks": [], "donor_blocks": []}
    replies = [{**base, **r} for r in replies]
    return json.loads(subprocess.check_output(["node", "-e", _POLL % (json.dumps(fn), json.dumps(replies))],
                                              text=True))


def test_a_tab_that_never_noted_its_load_still_sees_another_tabs():
    # a tab open (or reloaded) with load 7 on: its own. Another tab loads over it (8) and renders gen 9: this tab says
    # so, keeps its own picture (gen 3) and its Cancel off the other tab's render
    first, second = _polls({"load_id": 7, "load_by": "page", "shown": 3},
                           {"load_id": 8, "load_by": "page", "shown": 9, "rendering": True,
                            "status": "Baseline ready - rendering Tweaked..."})
    assert not first["foreign"]
    assert second["foreign"] and "Another Studio tab" in second["status"] and "rendering Tweaked" in second["status"]
    assert second["base"] == "" and second["cancel"] is True
    assert second["shown"] == 9            # seen, not drawn: the tab's next picture is its own render


def test_a_tabs_own_render_still_draws():
    first, second = _polls({"load_id": 7, "load_by": "page", "shown": 3}, {"load_id": 7, "load_by": "page", "shown": 4})
    assert not second["foreign"] and "gen=4" in second["base"] and "Another Studio tab" not in second["status"]
