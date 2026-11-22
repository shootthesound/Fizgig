"""The Studio queue swaps a job's primary LoRA on the loaded model when nothing else differs, as the page does (#394),
instead of loading the whole model again; the load's settings follow the swap, so the page and later jobs know
what is loaded."""
import json

from fizgig.web import studio, studio_queue as Q

A, B = "a.safetensors", "b.safetensors"


def _load(primary, **kw):
    return dict({"family": "zimage", "primary": primary, "donor": "", "fast": True, "turbo": True, "negative": "",
                 "h3_model": "", "base": "", "canvas": None, "frankenstein": False}, **kw)


def _setup(monkeypatch, tmp_path):
    for p in (A, B):
        (tmp_path / p).write_bytes(b"x")
    a, b = str(tmp_path / A), str(tmp_path / B)
    key = json.dumps(_load(a), sort_keys=True)
    monkeypatch.setattr(studio, "_S", dict(studio._S, state="ready", load_key=key, page_key=key, primary=a,
                                           family="zimage", rendering=False, load_id=3))
    sent, loads, renders = [], [], []
    monkeypatch.setattr(studio, "_send", sent.append)
    monkeypatch.setattr(studio, "render_job", lambda state, extra, job: renders.append(job) or True)
    monkeypatch.setattr(studio, "generating", lambda: False)
    monkeypatch.setattr(Q, "_save", lambda: None)
    monkeypatch.setitem(Q._hooks, "load", loads.append)
    monkeypatch.setitem(Q._hooks, "busy", None)
    return a, b, key, sent, loads, renders


def _job(monkeypatch, load):
    it = {"id": "j1", "status": "waiting", "load": load, "render": {"state": {}, "extra": {}}}
    monkeypatch.setattr(Q, "_q", dict(Q._q, items=[it], job="", paused=False, waiting=""))
    return it


def test_a_job_with_another_primary_swaps_it_in_place(monkeypatch, tmp_path):
    a, b, key, sent, loads, renders = _setup(monkeypatch, tmp_path)
    it = _job(monkeypatch, _load(b))
    Q._tick()
    assert sent == [{"cmd": "primary", "path": b}] and not loads and it["status"] == "swapping"
    studio._event({"ev": "primary", "primary": b})
    Q._tick()
    assert renders == ["j1"] and not loads
    # what's loaded is the job's load now; the page's own settings stay (its next render loads its LoRA again)
    assert studio.load_key() == json.dumps(_load(b), sort_keys=True)
    assert studio._S["page_key"] == key and studio.status()["load_by"] == "queue"


def test_a_lora_that_cant_load_fails_the_job(monkeypatch, tmp_path):
    a, b, key, sent, loads, renders = _setup(monkeypatch, tmp_path)
    it = _job(monkeypatch, _load(b))
    Q._tick()
    studio._event({"ev": "primary", "primary": a, "error": "adapts nothing"})
    Q._tick()
    assert it["status"] == "failed" and not renders and studio.load_key() == key


def test_anything_else_different_still_loads(monkeypatch, tmp_path):
    a, b, key, sent, loads, renders = _setup(monkeypatch, tmp_path)
    _job(monkeypatch, _load(b, donor="d.safetensors"))
    Q._tick()
    assert not sent and len(loads) == 1


def test_a_page_swap_keeps_the_page_rendering(monkeypatch, tmp_path):
    a, b, key, sent, loads, renders = _setup(monkeypatch, tmp_path)
    studio.primary(b)
    studio._event({"ev": "primary", "primary": b})
    assert studio.load_key() == studio._S["page_key"] == json.dumps(_load(b), sort_keys=True)


def test_the_page_swaps_its_own_lora_back_after_a_job(monkeypatch, tmp_path):
    # a queue job swapped its LoRA on: the page's next Render swaps the page's own back (seconds), never a reload
    a, b, key, sent, loads, renders = _setup(monkeypatch, tmp_path)
    _job(monkeypatch, _load(b))
    Q._tick()
    studio._event({"ev": "primary", "primary": b})
    assert studio.adopt(_load(a)) == {"swap": True, "primary": b}
    studio.primary(a)
    studio._event({"ev": "primary", "primary": a})
    assert studio.load_key() == studio._S["page_key"] == key
    # the same primary as the job's: the page renders on it as it is; anything else different: it loads again
    studio._S.update(load_key=json.dumps(_load(b), sort_keys=True), page_key=key)
    assert studio.adopt(_load(b)) == {"same": True} and studio._S["page_key"] == studio.load_key()
    assert studio.adopt(_load(a, donor="d.safetensors")) == {}


def test_the_page_asks_before_it_reloads():
    import os
    js = open(os.path.join(os.path.dirname(studio.__file__), "static", "app.js"), encoding="utf-8").read()
    render = js[js.index("async function studioRender"):js.index("function studioRollSeed")]
    assert render.count("await studioAdopt()") == 2 and 'S.st.loadedKey = ""' not in render.split("async function studioAdopt")[0]
    poll = js[js.index("async function studioPollOnce"):]
    assert "studioAdopt().then(studioStart)" in poll[:poll.index("\n}\n")]
    adopt = js[js.index("async function studioAdopt"):js.index("function studioRollSeed")]
    assert 'api("/api/studio/adopt", studioLoadBody())' in adopt
