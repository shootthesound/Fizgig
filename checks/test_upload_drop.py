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


def test_single_purpose_boxes_take_only_their_kind():
    """A LoRA box takes .safetensors and a photo box pictures, by Upload... and by a drop alike: a picture dropped on
    Metadata's file box once uploaded and then read as a missing LoRA."""
    s = _src()
    for call in ['"ROYALE_SINGLE", accept: ".safetensors"', '"PROFILER_LORA", accept: ".safetensors"',
                 '"PROFILER_PHOTO", accept: "image/*"', '"EXTRACT_SOURCE", accept: ".safetensors"',
                 '"EXTRACT_" + id, accept: ".safetensors"', '"METADATA_FILE", accept: ".safetensors"',
                 '"METADATA_THUMB", accept: "image/*"']:
        assert "key: " + call + "}" in s, call
    from fizgig.families import form
    from fizgig.families.registry import FAMILIES
    want = {"CONTEXT_LORA_PATH": ".safetensors", "FAMILY_EDIT_REF": "image/*", "METADATA_THUMBNAIL": "image/*",
            "SAMPLE_REFERENCE": "image/*", "SAMPLE_FIRST_FRAME": "image/*", "SAMPLE_LAST_FRAME": "image/*"}
    seen = set()
    for desc in FAMILIES.values():
        for sec in form.form(desc)["sections"]:
            for r in sec["rows"]:
                if r and r["key"] in want:
                    assert r.get("accept") == want[r["key"]], (desc.key, r["key"])
                    seen.add(r["key"])
    assert seen == set(want)
