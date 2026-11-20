"""#386: the LoRA name follows the training folder, and Add to queue leaves the form on your own setup."""
import os
import re

import pytest

server = pytest.importorskip("fizgig.web.server")
APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def test_name_per_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "LAST_USED_FILE", str(tmp_path / ".last_used.json"))
    a, b = str(tmp_path / "abc"), str(tmp_path / "xyz")
    server.set_folder_trigger(a, "abc,")                 # the trigger word has its own map
    server.set_folder_lora_name(a, "abc_lora")
    server.set_folder_lora_name(b, "")                   # an empty name records nothing
    assert server.folder_lora_name(a) == "abc_lora" and server.folder_lora_name(b) == ""
    assert server.folder_lora_name(a + os.sep) == "abc_lora" or os.name != "nt"   # same folder, however typed
    assert server.folder_trigger(a) == "abc,"
    server.set_folder_lora_name(a, "abc_v2")
    assert server.folder_lora_name(a) == "abc_v2" and server.folder_lora_name("") == ""


def test_add_to_queue_stays_on_your_setup():
    js = open(APP, encoding="utf-8").read()
    add = js[js.index('$("btnQueue").addEventListener("click"'):js.index('$("queueEditSave")')]
    assert 'ctxSwitch({kind: "queue"' not in add          # adding no longer turns the form into the queued run
    at = js.index('$("btnQueueCancel")')
    cancel = js[at:js.index('$("webQueueStart")', at)]
    assert re.search(r'ctxSwitch\(\{kind: "draft"\}\)', cancel)   # Cancel edit leaves the queued run
