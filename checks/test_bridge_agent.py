"""The Assistant's step loop (fizgig.bridge.agent): a model that keeps asking for the same look is stopped."""
from fizgig.bridge.agent import Agent


def _llm(script):
    class Llm:
        n = 0

        def chat(self, messages, tools):
            self.n += 1
            if self.n <= len(script):
                name, args = script[self.n - 1]
                return {"role": "assistant", "tool_calls": [{"id": str(self.n), "function": {
                    "name": name, "arguments": args}}]}, {}
            return {"role": "assistant", "content": "done"}, {}
    return Llm()


class _Tools:
    def __init__(self):
        self.ran = []

    def schemas(self):
        return [{"name": n, "inputSchema": {"type": "object"}} for n in ("affinity_documents", "render")]

    def call(self, name, args):
        self.ran.append(name)
        return [{"type": "text", "text": "[]"}], False


def test_a_look_asked_a_third_time_with_nothing_done_between_is_not_run():
    tools, seen = _Tools(), []
    Agent(_llm([("affinity_documents", "{}")] * 4), tools, free_gpu=False,
          on_event=lambda k, d: seen.append((k, d))).ask("make them black and white")
    assert tools.ran == ["affinity_documents", "affinity_documents"]
    refused = [d for k, d in seen if k == "result" and d["error"]]
    assert len(refused) == 2 and "Not run" in refused[0]["text"]


def test_a_look_after_something_was_done_runs_again():
    tools = _Tools()
    Agent(_llm([("affinity_documents", "{}"), ("affinity_documents", "{}"), ("render", "{}"),
                ("affinity_documents", "{}"), ("affinity_documents", "{}")]), tools, free_gpu=False).ask("x")
    assert tools.ran.count("affinity_documents") == 4
