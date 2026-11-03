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
    "is changed otherwise. With a LoRA of a person, say who they are in every prompt after the trigger word ('gwelt, "
    "a man ...'): a quickly trained LoRA needs it. Look at every picture you make: if one is wrong (not the person, "
    "broken), make it again with a new seed and a clearer prompt before you finish - fix it, don't just report it. To "
    "redo a picture already in Affinity, place the new one with into_layer and replace. To train in the cloud (the "
    "user asks for a rented GPU or gives a price), use cloud_train with their price as max_price, then cloud_status "
    "with wait=true until it's done, and carry on with the LoRA it brought home; if it failed, say why in one line "
    "and train here with start_training (the same model, preset and settings). For a picture of a person that goes into "
    "a design, render 4 variations in one call and run score_pictures on them with the LoRA's training folder; place "
    "its pick, and say in one line why it won (likeness, eyes, facing). To design a layout from nothing (a poster, a "
    "flyer, a card), make the document with affinity_new_document, then build it with affinity_add_shape (background, "
    "panels, an empty photo frame) and affinity_add_text (one call per line), name every layer, look at it with "
    "affinity_view, fix what's off, and affinity_save it.")


class LLM:
    """An OpenAI-compatible chat endpoint."""

    def __init__(self, base=DEFAULT_LLM, model="", key="local"):
        self.base, self.model, self.key = base.rstrip("/"), model, key

    def _req(self, path, body=None, timeout=900, method=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method or ("POST" if body is not None else "GET"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                                              "User-Agent": "Fizgig-assistant/1"})   # a proxy refuses Python's own
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

    def load(self, timeout=300):
        """Load the model back after free_gpu (a llama.cpp router: POST /models/load, then wait until it says loaded);
        True once it is. Servers without this load on the next chat anyway."""
        if self._state() is None:               # not a router: the next chat loads it
            return False
        self._until(lambda s: s not in ("unloading", "stopping"), 120)      # a stop still finishing first
        if not self._router("/models/load"):
            return False
        return self._until(lambda s: s == "loaded", timeout)

    def free_gpu(self):
        """Unload the model from the GPU, where the server can (a llama.cpp router: POST /models/unload), and wait
        until it has gone. True if it did; load() (or the next chat) brings it back."""
        if self._state() in (None, "unloaded"):
            return self._state() == "unloaded"
        return self._router("/models/unload") and self._until(lambda s: s == "unloaded", 120)

    def _router(self, path):
        root = self.base[:-3] if self.base.endswith("/v1") else self.base
        try:
            req = urllib.request.Request(root + path, data=json.dumps({"model": self.pick_model()}).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
            return True
        except Exception:
            return False

    def _state(self):
        """A llama.cpp router's word for the model (loaded / loading / unloaded ...); None from other servers."""
        try:
            for m in self._req("/models", timeout=5).get("data", []):
                if m.get("id") == self.pick_model():
                    return (m.get("status") or {}).get("value")
        except Exception:
            pass
        return None

    def _until(self, ok, timeout):
        end = time.time() + timeout
        while time.time() < end:
            s = self._state()
            if s is None or ok(s):
                return s is not None
            time.sleep(0.5)
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


class SplitTools:
    """Affinity's tools here, Fizgig's on a pod (its /mcp, logged in): the language model, training and renders run
    there, the user's documents stay on this computer. Pictures cross where the two meet: a training folder made from
    Affinity's layers goes up to the pod, and a picture from the pod comes down before it goes into Affinity."""

    def __init__(self, local: LocalTools, pod_url, password):
        from fizgig.web.helpers import Client
        self.local, self.pod = local, Client(pod_url.rstrip("/"))
        self.pod.login(password)
        self._n = 0
        self.remote = {t["name"]: t for t in self._rpc("tools/list").get("tools", [])}

    def _rpc(self, method, params=None):
        self._n += 1
        msg = self.pod.call("/mcp", {"jsonrpc": "2.0", "id": self._n, "method": method, "params": params or {}},
                            timeout=3600)
        if msg.get("error"):
            err = msg["error"]
            raise RuntimeError(err.get("message") if isinstance(err, dict) else err)
        return msg.get("result") or {}

    def schemas(self):
        here = [s for s in self.local.schemas() if s["name"].startswith("affinity_")]
        return here + [s for n, s in self.remote.items() if not n.startswith("affinity_") and n != "release_gpu"]

    def call(self, name, args):
        if not name.startswith("affinity_"):
            r = self._rpc("tools/call", {"name": name, "arguments": args})
            return r.get("content") or [], bool(r.get("isError"))
        if name == "affinity_to_training_folder":        # exported here, then sent to the pod's training data
            return self._to_pod_folder(args)
        if name == "affinity_place" and args.get("path") and not os.path.isfile(args["path"]):
            args = {**args, "path": self._from_pod(args["path"])}
        return self.local.call(name, args)

    def _to_pod_folder(self, args):
        import re
        import tempfile
        from fizgig.bridge import affinity, cloud
        from fizgig.bridge.mcp import text
        name = re.sub(r'[\\/:*?"<>|]+', "_", os.path.basename(str(args.get("folder") or "").rstrip("/\\"))).strip(" .")
        if not name:
            return [text("Give the training folder a name.")], True
        try:
            paths = affinity.export_layers(args.get("document", ""), args.get("layers"), bool(args.get("whole")),
                                           tag=name, selected=bool(args.get("selected")))
            if not paths:
                return [text("Affinity exported nothing (no layers selected or visible?).")], True
            with tempfile.TemporaryDirectory() as tmp:
                names = affinity.copy_into(paths, tmp)
                folder = cloud._send(self.pod, tmp, name)
        except Exception as e:                                   # noqa: BLE001 - the model reads the reason
            return [text(f"{type(e).__name__}: {e}")], True
        return [text(json.dumps({"folder": folder, "pictures": names, "on": "the GPU machine"}))], False

    def _from_pod(self, path):
        import urllib.parse
        from fizgig.bridge import affinity
        dest = os.path.join(affinity.bridge_dir("from the GPU machine"), os.path.basename(path))
        if not os.path.isfile(dest):
            self.pod.download("/api/media?path=" + urllib.parse.quote(path), dest, lambda *a: None)
        return dest


class Agent:
    def __init__(self, llm: LLM, tools, vision=True, free_gpu=True, on_event=None, max_steps=24):
        self.llm, self.tools, self.vision, self.free_gpu = llm, tools, vision, free_gpu
        self.on_event = on_event or (lambda kind, data: None)
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.stats = {"llm_seconds": 0.0, "tool_seconds": 0.0, "gpu_handovers": 0, "handover_seconds": 0.0,
                      "reload_seconds": 0.0, "release_seconds": 0.0}
        self._off_gpu = False

    def _functions(self):
        return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                                  "parameters": s.get("inputSchema") or {"type": "object"}}}
                for s in self.tools.schemas()]

    def ask(self, text):
        """One user turn: the model calls tools until it answers. Its answer."""
        self.messages.append({"role": "user", "content": text})
        functions = self._functions()
        for _step in range(self.max_steps):
            if self._off_gpu:                 # back on the card after a handover: the time that costs, measured
                r0 = time.time()
                try:                          # Fizgig's render model makes room first (it reloads for the next one)
                    content, _bad = self.tools.call("release_gpu", {})
                    self.stats["release_seconds"] += time.time() - r0
                except Exception:
                    pass
                r0 = time.time()
                self.on_event("gpu", {"text": "Fizgig's models left the GPU; loading the language model back..."})
                if self.llm.load():
                    took = time.time() - r0
                    self.stats["reload_seconds"] += took
                    self.on_event("gpu", {"text": f"The model is back on the GPU ({took:.0f} s)."})
                self._off_gpu = False
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
                        self._off_gpu = True
                        self.on_event("gpu", {"text": "The model stepped off the GPU so Fizgig can use it."})
                self.on_event("tool", {"name": name, "args": args})
                t1 = time.time()
                content, bad = self.tools.call(name, args)
                texts = "\n".join(x.get("text", "") for x in content if x.get("type") == "text")
                if name == "start_training" and not bad and self.free_gpu and '"started": true' in texts:
                    # a training run holds the card: the language model can't come back beside it, so the agent
                    # waits the run out here (a machine with memory for both would keep talking meanwhile)
                    self.on_event("gpu", {"text": "Training has the GPU; the language model waits until it's done."})
                    w0 = time.time()
                    more, _b = self.tools.call("wait_for_training", {"minutes": 240})
                    self.stats["waited_for_training_seconds"] = self.stats.get("waited_for_training_seconds", 0) + \
                        time.time() - w0
                    texts += "\n\nThe run, after waiting for it:\n" + "\n".join(
                        x.get("text", "") for x in more if x.get("type") == "text")
                self.stats["tool_seconds"] += time.time() - t1
                imgs = [x for x in content if x.get("type") == "image"]
                self.on_event("result", {"name": name, "error": bad, "text": texts[:1500], "images": len(imgs),
                                         "pictures": imgs[:4]})
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
