"""The Model Context Protocol's JSON-RPC: Fizgig's own server side (a tool registry answering initialize /
tools/list / tools/call, served over Streamable HTTP at the web UI's /mcp) and a client for an SSE server (Affinity's).
Standard library only, as the web UI is."""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")     # newest first; we answer in the client's


class ToolError(Exception):
    """A tool's plain-English failure: it goes back to the model as the tool's result (isError), not as a protocol
    error, so the model can read it and try something else."""


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, Any]                      # JSON Schema properties
    run: Callable[..., Any]                     # (**arguments) -> str | dict | list | Content list
    required: List[str] = field(default_factory=list)

    def schema(self):
        return {"name": self.name, "description": self.description,
                "inputSchema": {"type": "object", "properties": self.params, "required": self.required}}


def text(t):
    return {"type": "text", "text": t if isinstance(t, str) else json.dumps(t, indent=1, default=str)}


def image(data_b64, mime="image/png"):
    return {"type": "image", "data": data_b64, "mimeType": mime}


def as_content(result):
    """A tool's return value as MCP content: a list of content items passes through, anything else is text."""
    if isinstance(result, list) and result and all(isinstance(c, dict) and "type" in c for c in result):
        return result
    return [text(result)]


class Server:
    """A tool registry speaking MCP's server side. handle(message) -> the reply (None for a notification)."""

    def __init__(self, name, version, instructions=""):
        self.name, self.version, self.instructions = name, version, instructions
        self.tools: Dict[str, Tool] = {}

    def add(self, tool: Tool):
        self.tools[tool.name] = tool
        return tool

    def call(self, name, arguments):
        """Run a tool: (content, is_error)."""
        tool = self.tools.get(name)
        if tool is None:
            return [text(f"No tool named {name}.")], True
        try:
            return as_content(tool.run(**(arguments or {}))), False
        except ToolError as e:
            return [text(str(e))], True
        except TypeError as e:                  # arguments that don't fit the tool
            return [text(f"{name}: {e}")], True
        except Exception as e:
            logger.exception("mcp tool %s", name)
            return [text(f"{name} failed: {type(e).__name__}: {e}")], True

    def handle(self, msg):
        if isinstance(msg, list):               # a batch (2025-03-26)
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict):
            return _err(None, -32600, "Invalid request")
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:                         # a notification (initialized, cancelled ...): nothing to answer
            return None
        if method == "initialize":
            want = params.get("protocolVersion")
            return _ok(mid, {"protocolVersion": want if want in VERSIONS else VERSIONS[0],
                             "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": self.name, "version": self.version},
                             **({"instructions": self.instructions} if self.instructions else {})})
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": [t.schema() for t in self.tools.values()]})
        if method == "tools/call":
            content, bad = self.call(params.get("name"), params.get("arguments"))
            return _ok(mid, {"content": content, "isError": bad})
        return _err(mid, -32601, f"Method not found: {method}")


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


class SSEClient:
    """An MCP client for a server on the HTTP+SSE transport (2024-11-05): GET the SSE url, which names a POST endpoint
    in its first `endpoint` event; requests go by POST, replies come back on the stream. Affinity's built-in server
    speaks this, and handles one session at a time - keep one client and reuse it."""

    def __init__(self, url, client_name="fizgig", timeout=120):
        self.url, self.client_name, self.timeout = url, client_name, timeout
        self._post_url: Optional[str] = None
        self._replies: Dict[Any, "queue.Queue"] = {}
        self._lock = threading.Lock()
        self._next = 0
        self._resp = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self.server_info: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []

    # -- connection
    def connect(self, connect_timeout=5):
        req = urllib.request.Request(self.url, headers={"Accept": "text/event-stream"})
        self._resp = urllib.request.urlopen(req, timeout=connect_timeout)
        try:
            self._resp.fp.raw._sock.settimeout(None)    # the stream stays open; reads wait for events
        except AttributeError:
            pass
        threading.Thread(target=self._read, daemon=True, name="mcp-sse").start()
        if not self._ready.wait(connect_timeout):
            self.close()
            raise ConnectionError(f"{self.url} sent no endpoint event")
        init = self.request("initialize", {"protocolVersion": VERSIONS[0], "capabilities": {},
                                           "clientInfo": {"name": self.client_name, "version": "1"}})
        self.server_info = init
        self.notify("notifications/initialized")
        self.tools = self.request("tools/list").get("tools", [])
        return self

    @property
    def alive(self):
        return self._ready.is_set() and not self._closed.is_set()

    def close(self):
        self._closed.set()
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass

    def _read(self):
        event, data = "message", []
        try:
            for raw in self._resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if data:
                        self._event(event, "\n".join(data))
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
        except Exception as e:
            if not self._closed.is_set():
                logger.info("mcp sse stream ended: %s", e)
        finally:
            self._closed.set()
            for q in list(self._replies.values()):   # anyone waiting hears the stream is gone
                q.put(None)

    def _event(self, event, data):
        if event == "endpoint":
            self._post_url = urljoin(self.url, data.strip())
            self._ready.set()
            return
        try:
            msg = json.loads(data)
        except ValueError:
            return
        for m in (msg if isinstance(msg, list) else [msg]):
            q = self._replies.get(m.get("id")) if isinstance(m, dict) else None
            if q is not None:
                q.put(m)

    # -- messages
    def _post(self, msg):
        body = json.dumps(msg).encode("utf-8")
        req = urllib.request.Request(self._post_url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()

    def notify(self, method, params=None):
        self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params=None, timeout=None):
        if not self.alive:
            raise ConnectionError("The MCP connection is closed.")
        with self._lock:
            self._next += 1
            mid = self._next
        q = self._replies[mid] = queue.Queue()
        try:
            self._post({"jsonrpc": "2.0", "id": mid, "method": method, **({"params": params} if params else {})})
            try:
                reply = q.get(timeout=timeout or self.timeout)
            except queue.Empty:
                raise TimeoutError(f"{method}: no reply in {timeout or self.timeout} s")
        finally:
            self._replies.pop(mid, None)
        if reply is None:
            raise ConnectionError("The MCP connection closed while waiting.")
        if "error" in reply:
            raise RuntimeError(reply["error"].get("message") or str(reply["error"]))
        return reply.get("result") or {}

    def call_tool(self, name, arguments=None, timeout=None):
        """(content list, is_error)."""
        r = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        return r.get("content") or [], bool(r.get("isError"))


def content_text(content):
    """The text items of a tool result, joined."""
    return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
