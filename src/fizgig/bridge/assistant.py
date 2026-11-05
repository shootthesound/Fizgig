"""The Assistant tab: one conversation with a local model driving Fizgig's tools (and Affinity through them), run a
turn at a time in the background; the page polls the steps."""
from __future__ import annotations

import threading
import time

from fizgig.bridge.agent import DEFAULT_LLM, LLM, Agent, LocalTools, SplitTools

_lock = threading.Lock()
_S = {"agent": None, "busy": False, "events": [], "n": 0, "started": 0.0, "key": None}


def settings(prefs):
    return {"url": str(prefs.get("assistant_llm_url") or DEFAULT_LLM),
            "model": str(prefs.get("assistant_llm_model") or ""),
            "free_gpu": str(prefs.get("assistant_free_gpu") or "1") == "1",
            "vision": str(prefs.get("assistant_vision") or "1") == "1",
            "think": str(prefs.get("assistant_think") or "0") == "1",
            # a GPU machine for Fizgig's side (a pod): its address and login; Affinity stays on this computer
            "pod": str(prefs.get("assistant_fizgig_url") or "").strip(),
            "pod_password": str(prefs.get("assistant_fizgig_password") or ""),
            "key": str(prefs.get("assistant_llm_key") or "") or "local"}


def _event(kind, data):
    with _lock:
        _S["n"] += 1
        ev = {"id": _S["n"], "kind": kind, "t": round(time.time() - _S["started"], 1), **data}
        pics = ev.pop("pictures", None) or []   # pictures as data URIs the page shows as they are
        if pics:
            ev["srcs"] = [f"data:{im.get('mimeType', 'image/jpeg')};base64,{im.get('data', '')}" for im in pics]
        _S["events"].append(ev)
        del _S["events"][:-400]


def send(text, server, prefs):
    st = settings(prefs)
    key = (st["url"], st["model"], st["free_gpu"], st["vision"], st["pod"], st["key"], st["think"])
    with _lock:
        if _S["busy"]:
            raise ValueError("The assistant is still working on the last message.")
        if _S["agent"] is None or _S["key"] != key:
            tools = SplitTools(LocalTools(server), st["pod"], st["pod_password"]) if st["pod"] else LocalTools(server)
            _S["agent"] = Agent(LLM(st["url"], st["model"], st["key"], think=st["think"]), tools, vision=st["vision"],
                                free_gpu=st["free_gpu"] and not st["pod"], on_event=_event)
            _S["key"] = key
        _S["busy"] = True
        if not _S["started"]:
            _S["started"] = time.time()
        agent = _S["agent"]
    _event("user", {"text": text})

    def work():
        try:
            agent.ask(text)
        except Exception as e:
            _event("error", {"text": f"{type(e).__name__}: {e}"})
        finally:
            with _lock:
                _S["busy"] = False

    threading.Thread(target=work, daemon=True, name="assistant").start()
    return {"ok": True}


def status(since=0):
    with _lock:
        a = _S["agent"]
        return {"busy": _S["busy"], "events": [e for e in _S["events"] if e["id"] > since],
                "stats": dict(a.stats) if a else {}, "model": a.llm.model if a else ""}


def reset():
    with _lock:
        if _S["busy"]:
            raise ValueError("Wait for the assistant to finish first.")
        _S.update(agent=None, events=[], n=0, started=0.0, key=None)
    return {"ok": True}


def models(url):
    return LLM(url or DEFAULT_LLM).models()
