"""Studio in two tabs (#395): Studio holds one model at a time, so a load or a primary LoRA swap in one tab is a new
load to every other tab. The other tab says so (its model picker still shows its own setup) and its Render loads its
own setup again instead of rendering on the other tab's model or LoRA."""
import os

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
