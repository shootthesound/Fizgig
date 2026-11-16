"""The browser sends the same training-folder scope to preparation and caption tools."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


APP = Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js"


def _function(source, name):
    for prefix in ("async function ", "function "):
        marker = prefix + name + "("
        if marker in source:
            start = source.index(marker)
            return source[start:source.index("\n}", start) + 2]
    raise AssertionError(f"Missing browser function {name}")


@pytest.mark.parametrize("subfolders", [False, True])
def test_prep_and_caption_requests_keep_the_current_dataset_scope(subfolders):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    source = APP.read_text(encoding="utf-8")
    functions = "\n".join(_function(source, name) for name in (
        "datasetToolValues", "datasetToolQuery", "capJobBody", "capRun", "prepRun", "prepSummary",
        "captionsLive", "loadCaptions"))
    script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const scope = {
  TRAIN_SUBFOLDERS: @@BOOL@@, FAMILY_KIND: 'Fine-tune', FAMILY_MULTICONCEPT: false,
  FAMILY_DOP: false, FAMILY_MULTI_PERSON: false, FAMILY_REG_ON: true,
  FAMILY_REG_DIR: 'C:/photos/regularisation & reference', FAMILY_FT_REG_DIR: 'C:/photos/reg ft',
  LORA_OUTPUT_DIR: 'C:/photos/output'
};
const calls = [], nodes = new Map();
function $(id) {
  if (!nodes.has(id)) nodes.set(id, {value: '', checked: false, style: {}, textContent: '',
    replaceChildren() {}});
  return nodes.get(id);
}
const S = {family: 'krea2', imageFolder: 'C:/photos/subject & outfits',
  values: {...scope, DATASET_MEGAPIXELS: 1, UNRELATED_PRIVATE_SETTING: 'do not send'},
  prep: {opts: {face_tools: false}}, cap: {opts: {}, optsFamily: 'krea2', items: [
    {path:'C:/photos/subject & outfits/red/a.png', _ta:{isConnected:true}},
    {path:'C:/photos/subject & outfits/blue/a.png', _ta:{isConnected:false}}
  ]}};
const context = vm.createContext({S, $, TOKEN: 'test', encodeURIComponent,
  document: {querySelectorAll: () => [], querySelector: () => ({}), activeElement: null},
  prepSettings: () => ({mode: 'Resize Only', mp: 1, replace: false, crops_only: false, min_face: 400}),
  capFolder: () => S.imageFolder, capExt: () => '.txt', capInstruction: () => 'describe',
  subfoldersHint: () => '', pollPrep() {}, pollCaptions() {}, renderCaptions() {},
  el: () => ({}),
  api: async (url, body) => {calls.push({url, body}); return {items: [], lines: [], count: 2};}
});
vm.runInContext(FUNCTIONS, context);
const plain = (value) => JSON.parse(JSON.stringify(value));
function checkBody(call, endpoint) {
  assert.equal(call.url, endpoint);
  assert.equal(call.body.folder, S.imageFolder);
  assert.equal(call.body.family, S.family);
  assert.deepEqual(plain(call.body.values), scope);
}
function checkQuery(call, endpoint) {
  const url = new URL(call.url, 'http://test');
  assert.equal(url.pathname, endpoint);
  assert.equal(url.searchParams.get('folder'), S.imageFolder);
  assert.equal(url.searchParams.get('family'), S.family);
  assert.deepEqual(JSON.parse(url.searchParams.get('values')), scope);
  return url;
}
(async () => {
  assert.deepEqual(plain(context.datasetToolValues()), scope);
  await context.prepRun();
  checkBody(calls.pop(), '/api/prep/run');
  await context.prepSummary();
  checkQuery(calls.pop(), '/api/prep/summary');
  // Exercise the actual job body and sender, as used by all AI/static/translation actions.
  await context.capRun('/api/captions/ai', context.capJobBody({only: ['C:/photos/subject & outfits/red/a.png']}));
  const captionJob = calls.pop();
  checkBody(captionJob, '/api/captions/ai');
  assert.equal(captionJob.body.only.length, 1);
  await context.captionsLive();
  const live = checkQuery(calls.pop(), '/api/captions');
  assert.deepEqual(JSON.parse(live.searchParams.get('only')), ['C:/photos/subject & outfits/red/a.png']);
  S.cap.items[0]._ta.isConnected = false;
  await context.captionsLive();
  assert.equal(calls.length, 0);
  await context.loadCaptions();
  const listing = checkQuery(calls.pop(), '/api/captions');
  assert.equal(listing.searchParams.get('values'), live.searchParams.get('values'));
  assert.equal(listing.searchParams.get('ext'), '.txt');
  assert.equal(calls.length, 0);
})().catch((error) => {console.error(error); process.exitCode = 1;});
'''.replace("@@BOOL@@", json.dumps(subfolders)).replace("FUNCTIONS", json.dumps(functions))
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
