"""Fizgig's MCP server core (fizgig.bridge.mcp): the JSON-RPC an AI app speaks to /mcp."""
from fizgig.bridge.mcp import Server, Tool, ToolError, image, text


def _server():
    srv = Server("fizgig", "test", "hello")

    def add(a, b=1):
        return {"sum": a + b}

    def boom():
        raise ToolError("That folder isn't there.")

    def pic():
        return [text("a picture"), image("QUJD", "image/png")]

    srv.add(Tool("add", "adds", {"a": {"type": "integer"}, "b": {"type": "integer"}}, add, ["a"]))
    srv.add(Tool("boom", "fails", {}, boom))
    srv.add(Tool("pic", "a picture", {}, pic))
    return srv


def test_initialize_answers_in_the_clients_version():
    r = _server().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18"}})
    assert r["result"]["protocolVersion"] == "2025-06-18"
    assert r["result"]["capabilities"]["tools"] is not None
    assert r["result"]["instructions"] == "hello"
    old = _server().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999"}})
    assert old["result"]["protocolVersion"] == "2025-11-25"


def test_tools_list_and_call():
    srv = _server()
    names = [t["name"] for t in srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
    assert names == ["add", "boom", "pic"]
    r = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 2}}})
    assert r["result"]["isError"] is False and '"sum": 3' in r["result"]["content"][0]["text"]


def test_a_tools_failure_is_its_result_not_a_protocol_error():
    srv = _server()
    r = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "boom"}})
    assert r["result"]["isError"] is True and "isn't there" in r["result"]["content"][0]["text"]
    bad = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "add", "arguments": {}}})
    assert bad["result"]["isError"] is True                # a missing argument, said
    none = srv.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope"}})
    assert none["result"]["isError"] is True


def test_pictures_pass_through_and_notifications_get_no_reply():
    srv = _server()
    r = srv.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "pic"}})
    assert [c["type"] for c in r["result"]["content"]] == ["text", "image"]
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.handle({"jsonrpc": "2.0", "id": 8, "method": "nope"})["error"]["code"] == -32601
    batch = srv.handle([{"jsonrpc": "2.0", "id": 9, "method": "ping"},
                        {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert len(batch) == 1 and batch[0]["id"] == 9
