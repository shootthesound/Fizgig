"""A local LLM driving Fizgig (and through it, Affinity): any OpenAI-compatible chat server with tool calling -
llama.cpp's llama-server (--jinja), LM Studio, Ollama, vLLM. The model sees Fizgig's MCP tools as functions; pictures
a tool returns go back to it as images when it can see (a vision model).

On a card the model and Fizgig can't share, the agent frees the GPU before a tool that needs it (a llama.cpp router
unloads the model; the next message loads it again) - on a machine with memory for both, it never has to.

    python -m fizgig.bridge.agent "make a training folder from my Affinity layers" [--llm http://127.0.0.1:38088/v1]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

GPU_TOOLS = {"render", "start_training", "caption_folder", "make_masks"}
DEFAULT_LLM = "http://127.0.0.1:38088/v1"
SYSTEM = (
    "You are Fizgig's assistant, on the user's own computer. You drive Fizgig (it trains LoRAs for image models and "
    "renders with them) and Affinity by Canva (the user's design app) through tools. Work in small steps: look before "
    "you act (fizgig_status, affinity_documents, list_pictures), then do what was asked, then check the result (look "
    "at the picture, affinity_view). Say what you did in a few plain sentences: folders, files, what to look at. Never "
    "stop, delete or overwrite anything the user didn't ask about. A training run takes minutes to hours: start it, "
    "say where to watch it, and stop there. Pictures go into Affinity as new layers; nothing in the user's documents "
    "is changed otherwise.")


class LLM:
    """An OpenAI-compatible chat endpoint."""

    def __init__(self, base=DEFAULT_LLM, model="", key="local"):
        self.base, self.model, self.key = base.rstrip("/"), model, key

    def _req(self, path, body=None, timeout=900, method=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method or ("POST" if body is not None else "GET"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def models(self):
        try:
            return [m.get("id") for m in self._req("/models", timeout=5).get("data", [])]
        except Exception:
            return []

    def pick_model(self):
        if not self.model:
            ms = self.models()
            if not ms:
                raise ConnectionError(f"No model server answers at {self.base}.")
            self.model = ms[0]
        return self.model

    def chat(self, messages, tools):
        body = {"model": self.pick_model(), "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": 0.6}
        r = self._req("/chat/completions", body)
        return r["choices"][0]["message"], r.get("usage") or {}

    def free_gpu(self):
        """Unload the model from the GPU, where the server can (a llama.cpp router: POST /models/unload). True if it
        did; the next chat loads it again."""
        root = self.base[:-3] if self.base.endswith("/v1") else self.base
        try:
            req = urllib.request.Request(root + "/models/unload", data=json.dumps({"model": self.pick_model()}).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
            return True
        except Exception:
            return False


class RemoteTools:
    """Fizgig's tools over HTTP (its /mcp), for the agent outside the web UI."""

    def __init__(self, url):
        self.url, self._n = url, 0
        self.tools = {t["name"]: t for t in self._rpc("tools/list").get("tools", [])}

    def _rpc(self, method, params=None):
        self._n += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._n, "method": method, "params": params or {}}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            msg = json.loads(r.read())
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message"))
        return msg.get("result") or {}

    def schemas(self):
        return list(self.tools.values())

    def call(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        return r.get("content") or [], bool(r.get("isError"))


class LocalTools:
    """Fizgig's tools in the web UI's own process (bridge.mcp.Server)."""

    def __init__(self, server):
        self.server = server

    def schemas(self):
        return [t.schema() for t in self.server.tools.values()]

    def call(self, name, args):
        return self.server.call(name, args)


class Agent:
    def __init__(self, llm: LLM, tools, vision=True, free_gpu=True, on_event=None, max_steps=24):
        self.llm, self.tools, self.vision, self.free_gpu = llm, tools, vision, free_gpu
        self.on_event = on_event or (lambda kind, data: None)
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.stats = {"llm_seconds": 0.0, "tool_seconds": 0.0, "gpu_handovers": 0, "handover_seconds": 0.0}

    def _functions(self):
        return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                                  "parameters": s.get("inputSchema") or {"type": "object"}}}
                for s in self.tools.schemas()]

    def ask(self, text):
        """One user turn: the model calls tools until it answers. Its answer."""
        self.messages.append({"role": "user", "content": text})
        functions = self._functions()
        for _step in range(self.max_steps):
            t0 = time.time()
            self.on_event("thinking", {})
            msg, usage = self.llm.chat(self.messages, functions)
            self.stats["llm_seconds"] += time.time() - t0
            calls = msg.get("tool_calls") or []
            self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")
                                  and v is not None} | {"role": "assistant"})
            if msg.get("content") and calls:
                self.on_event("note", {"text": msg["content"]})
            if not calls:
                answer = (msg.get("content") or "").strip()
                self.on_event("answer", {"text": answer, "stats": self.stats})
                return answer
            pictures = []
            for c in calls:
                fn = c.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                if name in GPU_TOOLS and self.free_gpu:
                    h0 = time.time()
                    if self.llm.free_gpu():
                        self.stats["gpu_handovers"] += 1
                        self.stats["handover_seconds"] += time.time() - h0
                        self.on_event("gpu", {"text": "The model stepped off the GPU for Fizgig."})
                self.on_event("tool", {"name": name, "args": args})
                t1 = time.time()
                content, bad = self.tools.call(name, args)
                self.stats["tool_seconds"] += time.time() - t1
                texts = "\n".join(x.get("text", "") for x in content if x.get("type") == "text")
                imgs = [x for x in content if x.get("type") == "image"]
                self.on_event("result", {"name": name, "error": bad, "text": texts[:1500], "images": len(imgs),
                                         "image": imgs[0] if imgs else None})
                self.messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                                      "content": (texts or "(done)")[:12000] + (f"\n[{len(imgs)} picture(s) attached]"
                                                                                if imgs and self.vision else "")})
                pictures += [(name, i) for i in imgs]
            if pictures and self.vision:      # a tool's pictures, for a model that can see
                self.messages.append({"role": "user", "content": [{"type": "text", "text": "The pictures the tools "
                                                                   "returned:"}] + [
                    {"type": "image_url", "image_url": {"url": f"data:{i['mimeType']};base64,{i['data']}"}}
                    for _n, i in pictures[:4]]})
        answer = "I stopped after many steps without finishing; tell me how to carry on."
        self.on_event("answer", {"text": answer, "stats": self.stats})
        return answer


def main():
    ap = argparse.ArgumentParser(description="Fizgig's assistant: a local LLM using Fizgig's tools")
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--llm", default=os.environ.get("FIZGIG_LLM_URL", DEFAULT_LLM))
    ap.add_argument("--model", default=os.environ.get("FIZGIG_LLM_MODEL", ""))
    ap.add_argument("--fizgig", default=os.environ.get("FIZGIG_MCP_URL", "http://127.0.0.1:8081/mcp"))
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--keep-gpu", action="store_true", help="never unload the model (a machine with room for both)")
    a = ap.parse_args()

    def show(kind, d):
        if kind == "tool":
            print(f"  -> {d['name']} {json.dumps(d['args'])[:300]}", flush=True)
        elif kind == "result":
            print(f"  <- {'ERROR ' if d['error'] else ''}{d['text'][:300]}" + (f" [+{d['images']} picture]"
                                                                             if d["images"] else ""), flush=True)
        elif kind in ("note", "gpu"):
            print(f"  .. {d['text'][:400]}", flush=True)

    agent = Agent(LLM(a.llm, a.model), RemoteTools(a.fizgig), vision=not a.no_vision, free_gpu=not a.keep_gpu,
                  on_event=show)
    print(agent.ask(" ".join(a.prompt)))
    print(json.dumps(agent.stats))


if __name__ == "__main__":
    main()
