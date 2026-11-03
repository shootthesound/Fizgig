"""Fizgig's MCP server for an AI app that starts MCP servers itself over stdio (Claude Desktop's config, Codex,
LM Studio ...): it relays each JSON-RPC line to the Fizgig web UI running on this computer (its /mcp) and prints the
reply. Fizgig must be running, with Preferences > Network > Let AI apps use Fizgig (MCP) on.

    python src/fizgig/scripts/fizgig_mcp.py [--port 8081]

The port defaults to the one in Fizgig's Preferences (FIZGIG_MCP_URL overrides the whole address)."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _url(port):
    if os.environ.get("FIZGIG_MCP_URL"):
        return os.environ["FIZGIG_MCP_URL"]
    if not port:
        from fizgig.user_data import PREFS_FILE
        try:
            with open(PREFS_FILE, encoding="utf-8") as f:
                port = int(json.load(f).get("web_port") or 8081)
        except (OSError, ValueError):
            port = 8081
    return f"http://127.0.0.1:{port}/mcp"


def _say(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=None)
    url = _url(ap.parse_args().port)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            mid = json.loads(line).get("id")
        except (ValueError, AttributeError):
            mid = None
        req = urllib.request.Request(url, data=line.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                body = r.read()
            if body and r.status != 202:
                sys.stdout.write(body.decode("utf-8").replace("\n", " ") + "\n")
                sys.stdout.flush()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error") or f"HTTP {e.code}"
            except ValueError:
                msg = f"HTTP {e.code}"
            if mid is not None:
                _say({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": msg}})
        except OSError as e:
            if mid is not None:
                _say({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000,
                                                             "message": f"Fizgig isn't running at {url} ({e})."}})


if __name__ == "__main__":
    main()
