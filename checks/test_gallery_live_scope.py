"""Live caption updates read mounted rows, without scanning the media gallery."""
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlencode

import pytest

from fizgig.web import server


def test_selected_live_caption_reads_only_requested_sidecar(tmp_path, monkeypatch):
    root = tmp_path / "data"
    red = root / "red"
    red.mkdir(parents=True)
    image = red / "same.png"
    image.touch()
    image.with_suffix(".txt").write_text("new caption", encoding="utf-8")
    monkeypatch.setattr(server, "_tool_folders", lambda *args, **kw: [str(root), str(red)])
    monkeypatch.setattr(server.captions, "with_crops", lambda root, folders: folders)
    monkeypatch.setattr(server.os, "listdir", lambda *args: pytest.fail("Live update scanned media"))
    handler = object.__new__(server.Handler)
    handler.json = lambda body, status=200: (body, status)
    body, status = handler._captions(str(root), ".txt", "krea2", {}, only=[str(image)])
    assert status == 200
    assert body["items"] == [{"name": "same.png", "path": str(image), "caption": "new caption", "relative": "red/same.png"}]
    assert handler._captions(str(root), ".txt", "krea2", {}, only=[])[0] == {"items": []}


@pytest.mark.parametrize("name", ["outside.png", "same-masklabel.png", "not-media.txt"])
def test_live_selection_rejects_outside_and_nonmedia(tmp_path, monkeypatch, name):
    root = tmp_path / "data"
    root.mkdir()
    image = (tmp_path if name == "outside.png" else root) / name
    image.touch()
    monkeypatch.setattr(server, "_tool_folders", lambda *args, **kw: [str(root)])
    monkeypatch.setattr(server.captions, "with_crops", lambda root, folders: folders)
    handler = object.__new__(server.Handler)
    handler.json = lambda body, status=200: (body, status)
    assert handler._captions(str(root), ".txt", "krea2", {}, only=[str(image)])[1] == 400


def test_candidate_scope_checks_ancestors_without_walking(tmp_path, monkeypatch):
    root = tmp_path / "data"
    names = ["red/nested", "output/nested", "cache/nested", "reg/nested", "lora-reg/nested",
             "originals/nested", "originals_3/nested", ".hidden/nested", "red/removed/nested"]
    for name in names:
        (root / name).mkdir(parents=True)
    desc = server._files_family("krea2")
    monkeypatch.setattr(server, "load_prefs", lambda: {})
    monkeypatch.setattr(server.form, "inputs", lambda *a, **kw: {
        "LORA_OUTPUT_DIR": str(root / "output"), "cache_root": str(root / "cache")})
    monkeypatch.setattr(server.launch, "subfolders_on", lambda *a: True)
    monkeypatch.setattr(server.launch, "ft_reg_dir", lambda *a: str(root / "reg"))
    monkeypatch.setattr(server.launch, "lora_reg_dir", lambda *a: str(root / "lora-reg"))
    monkeypatch.setattr(server.launch, "subfolders", lambda *a, **kw: pytest.fail("Selected scope walked media"))
    candidates = [str(root / name) for name in names] + [str(root / ".."), str(root / "red" / ".." / "..")]
    assert server._tool_folders(str(root), desc.key, {"TRAIN_SUBFOLDERS": True}, candidates) == [str(root), str(root / "red/nested")]
    assert server._tool_folders(str(root), desc.key, {"TRAIN_SUBFOLDERS": False}, candidates) == [str(root)]
    monkeypatch.setattr(server.launch, "subfolders_on", lambda *a: False)
    assert server._tool_folders(str(root), desc.key, {"TRAIN_SUBFOLDERS": True}, candidates) == [str(root)]


def test_candidate_scope_refuses_symlink_ancestors(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks unavailable")
    monkeypatch.setattr(server, "load_prefs", lambda: {})
    monkeypatch.setattr(server.form, "inputs", lambda *a, **kw: {})
    monkeypatch.setattr(server.launch, "subfolders_on", lambda *a: True)
    monkeypatch.setattr(server.launch, "ft_reg_dir", lambda *a: "")
    monkeypatch.setattr(server.launch, "lora_reg_dir", lambda *a: "")
    assert server._tool_folders(str(root), "krea2", {"TRAIN_SUBFOLDERS": True}, [str(root / "link/nested")]) == [str(root)]


def test_live_get_and_single_ai_route_never_walk_dataset(tmp_path, monkeypatch):
    nested = tmp_path / "red"
    nested.mkdir()
    image = nested / "a.png"
    image.touch()
    image.with_suffix(".txt").write_text("current", encoding="utf-8")
    monkeypatch.setattr(server, "load_prefs", lambda: {})
    monkeypatch.setattr(server.launch, "subfolders", lambda *a, **kw: pytest.fail("Per-image request walked dataset"))
    handler = object.__new__(server.Handler)
    handler._host_ok = lambda: True
    handler._authed = lambda: True
    handler._api_ok = lambda *args: True
    handler.headers = {}
    handler.json = lambda body, status=200: body
    values = {"TRAIN_SUBFOLDERS": True}
    handler.path = "/api/captions?" + urlencode({"folder": str(tmp_path), "family": "krea2",
        "values": json.dumps(values), "only": json.dumps([str(image)])})
    assert handler._get()["items"][0]["caption"] == "current"
    handler.path = "/api/captions/ai"
    handler._body = lambda: {"family": "krea2", "folder": str(tmp_path), "values": values,
                             "model": "florence2_base", "only": [str(image)]}
    monkeypatch.setattr(server, "_caption_gpu_check", lambda *a: None)
    monkeypatch.setattr(server.captions, "ai_captions", lambda *a, **kw: kw)
    result = handler._post()
    assert result["only"] == [str(image)]
    assert set(result["folders"]) == {str(tmp_path), str(nested)}


def test_browser_live_updates_only_mounted_items():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    source = (Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js").read_text(encoding="utf-8")
    start = source.index("async function captionsLive(")
    function = source[start:source.index("\n}", start) + 2]
    script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const calls = [];
const visible = {path:'C:/photos/red/a.png', _ta:{isConnected:true}};
const removed = {path:'C:/photos/blue/a.png', _ta:{isConnected:false}};
const S = {imageFolder:'C:/photos', cap:{items:[visible, removed, {path:'C:/photos/no-textarea.png'}]}};
const context = vm.createContext({S, encodeURIComponent,
 document:{querySelector:()=>({})}, capFolder:()=>S.imageFolder, capExt:()=>'.txt', datasetToolQuery:()=>'',
 api:async url=>{calls.push(url);return {items:[]};}});
vm.runInContext(FUNCTION,context);
(async()=>{
 await context.captionsLive();
 assert.deepEqual(JSON.parse(new URL(calls[0],'http://test').searchParams.get('only')),[visible.path]);
 visible._ta.isConnected=false;
 await context.captionsLive();
 assert.equal(calls.length,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''.replace("FUNCTION", json.dumps(function))
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
