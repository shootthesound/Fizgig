"""#392: a file dragged onto any box with an Upload button (the training form's picture rows, the Samples reference
pictures, Studio's slots, Royale, the Profiler, Extract, Metadata) goes up as Upload... sends it; a file the box
doesn't take is refused by its accept list."""
import json
import os
import re
import shutil
import subprocess

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "fizgig", "web", "static",
                   "app.js")


def _src():
    with open(APP, encoding="utf-8") as f:
        return f.read()


def test_every_upload_box_takes_a_drop():
    s = _src()
    body = s[s.index("function uploadButton("):s.index("let wired = false;")]
    assert "dropOn(input)" in body and "dropOn(btn)" in body
    assert 'addEventListener("drop"' in body and "uploadAccepts(f, picker.accept)" in body


def test_accept_list():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    fn = re.search(r"function uploadAccepts\(f, accept\) \{.*?\n\}", _src(), re.S).group(0)
    cases = [[{"name": "a.PNG", "type": "image/png"}, ".safetensors,image/*", True],
             [{"name": "m.onnx", "type": ""}, ".onnx", True],
             [{"name": "p.jpg", "type": "image/jpeg"}, ".onnx", False],
             [{"name": "x.bin", "type": ""}, "", True]]
    out = subprocess.run([node, "-e", fn + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([f, a]) => "
                                            "uploadAccepts(f, a))))"], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [c[2] for c in cases]
