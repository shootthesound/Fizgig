"""Affinity by Canva's side of the bridge: a client for its built-in MCP server (Affinity 3.2 and later: Edit >
Settings > Model Context Protocol > Enable Affinity MCP, then restart Affinity) and the round trips Fizgig needs.

Affinity's scripts reach only the Desktop, so pictures cross through a folder there, `Fizgig Bridge`: Affinity exports
layers into it and Fizgig copies them into a training folder; Fizgig copies a render into it and Affinity places it as
a layer. The scripts below use the SDK as the Affinity 3.3 documentation describes it."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import threading
import time

from fizgig.bridge.mcp import SSEClient, ToolError, content_text

URLS = ("http://[::1]:6767/sse", "http://127.0.0.1:6767/sse", "http://localhost:6767/sse")
BRIDGE = "Fizgig Bridge"
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

_lock = threading.Lock()
_conn = {"client": None, "desktop": "", "preamble": False}


def _url_list():
    own = os.environ.get("AFFINITY_MCP_URL")
    return (own,) if own else URLS


def client():
    """The one connection to Affinity (its server handles one session at a time), made on first use and again when
    Affinity was restarted."""
    with _lock:
        c = _conn["client"]
        if c is not None and c.alive:
            return c
        last = None
        for url in _url_list():
            try:
                c = SSEClient(url, client_name="fizgig-bridge").connect()
                break
            except Exception as e:
                last, c = e, None
        if c is None:
            raise ToolError("Affinity isn't answering. Open Affinity (3.2 or later), turn on Edit > Settings > Model "
                            "Context Protocol > Enable Affinity MCP and Access files on your Desktop, then quit and "
                            f"reopen Affinity. ({last})")
        _conn.update(client=c, desktop="", preamble=False)
        return c


def _prepare(c):
    """Once per session: Affinity wants its SDK preamble read before scripts run."""
    if _conn["preamble"]:
        return
    try:
        c.call_tool("read_sdk_documentation_topic", {"filename": "preamble"})
    except Exception:
        pass
    _conn["preamble"] = True


def run_script(js, timeout=180):
    """Run JavaScript in Affinity; its console output. A script error comes back as a ToolError."""
    c = client()
    _prepare(c)
    content, bad = c.call_tool("execute_script", {"script": js}, timeout=timeout)
    out = content_text(content)
    if bad:
        raise ToolError(f"Affinity: {out.strip() or 'the script failed'}")
    return out


def _json_out(js, timeout=180):
    """A script that ends by logging one JSON line marked FIZGIG:; that object."""
    out = run_script(js, timeout)
    for line in reversed(out.splitlines()):
        i = line.find("FIZGIG:")
        if i >= 0:
            return json.loads(line[i + 7:])
    raise ToolError(f"Affinity sent no answer: {out.strip()[:400]}")


def desktop():
    """The Desktop as Affinity sees it (it may be OneDrive's)."""
    if not _conn["desktop"]:
        r = _json_out("const { Application } = require('/application.js');\n"
                      "console.log('FIZGIG:' + JSON.stringify({d: Application.userDesktopPath}));")
        _conn["desktop"] = os.path.normpath(r["d"])
    return _conn["desktop"]


def bridge_dir(*parts):
    d = os.path.join(desktop(), BRIDGE, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def status():
    c = client()
    return {"connected": True, "server": c.server_info.get("serverInfo", {}),
            "tools": [t.get("name") for t in c.tools], "desktop": desktop(),
            "bridge_folder": os.path.join(desktop(), BRIDGE)}


_DOC = """
const { Document } = require('/document.js');
function pick(id) {
  if (!id) return Document.current;
  for (const d of Document.all) if (String(d.sessionUuid) === id) return d;
  const t = String(id).toLowerCase().replace(/\\.af$/, '');                     // or by its title
  for (const d of Document.all) if (String(d.title).toLowerCase().replace(/\\.af$/, '') === t) return d;
  throw new Error('No open document ' + id);
}
function nm(n) { try { return n.userDescription || n.description || ''; } catch (e) { return ''; } }
"""


def documents():
    """The open documents and each one's layers (top first)."""
    js = _DOC + """
const out = [];
for (const d of Document.all) {
  const layers = [];
  let i = 0;
  for (const l of d.layers) {
    let box = null; try { const b = l.spreadBaseBox; box = [b.x, b.y, b.width, b.height].map(v => Math.round(v)); } catch (e) {}
    let vis = true; try { vis = l.isVisible; } catch (e) {}
    layers.push({index: i++, name: nm(l), kind: l.constructor.name, visible: vis, box: box});
  }
  layers.reverse();
  out.push({id: String(d.sessionUuid), title: d.title, path: d.path || '', width: d.widthPixels,
            height: d.heightPixels, current: Document.current && String(Document.current.sessionUuid) === String(d.sessionUuid),
            layers: layers});
}
console.log('FIZGIG:' + JSON.stringify(out));
"""
    return _json_out(js)


def export_layers(doc_id="", layers=None, whole=False, tag="", selected=False):
    """PNGs of a document's layers (by index, bottom = 0; else the selected ones with selected; else every visible
    picture layer), or of the whole document, into the bridge folder's from-affinity/<tag> folder. Their paths."""
    tag = re.sub(r"[^A-Za-z0-9 _-]+", "_", tag or time.strftime("%Y%m%d-%H%M%S")).strip() or "export"
    out = bridge_dir("from Affinity", tag)
    js = _DOC + f"""
const {{ FileExportOptions, FileExportArea }} = require('/document.js');
const {{ Selection }} = require('/selections.js');
const d = pick({json.dumps(doc_id)});
const dir = {json.dumps(out.replace(os.sep, "/"))};
const png = FileExportOptions.createWithPresetName('PNG');
const saved = [];
function safe(s) {{ return String(s).replace(/[^A-Za-z0-9 _-]+/g, '_').trim() || 'layer'; }}
if ({json.dumps(bool(whole))}) {{
  const p = dir + '/' + safe(d.title.replace(/\\.[^.]+$/, '')) + '.png';
  const r = d.export(p, png, FileExportArea.createForWholeDocument());
  for (const rec of r.all) if (rec.isSuccess) saved.push(rec.path);
}} else {{
  const all = []; for (const l of d.layers) all.push(l);
  let want = {json.dumps(list(layers) if layers else None)};
  const vis = (l) => {{ try {{ return l.isVisible; }} catch (e) {{ return true; }} }};
  let nodes = [];
  if (want) nodes = want.map(i => all[i]).filter(Boolean);
  else if ({json.dumps(bool(selected))}) {{ try {{ for (const n of d.selection.nodes) nodes.push(n); }} catch (e) {{}} }}
  else {{
    nodes = all.filter(l => vis(l) && /Raster|Image/.test(l.constructor.name));   // the pictures, not text or shapes
    if (!nodes.length) nodes = all.filter(vis);
  }}
  let k = 0;
  for (const n of nodes) {{
    const sel = Selection.create(d, n);
    d.selection = sel;
    const p = dir + '/' + String(++k).padStart(3, '0') + ' ' + safe(nm(n) || n.constructor.name) + '.png';
    const r = d.export(p, png, FileExportArea.createForSelection(sel));
    for (const rec of r.all) if (rec.isSuccess) saved.push(rec.path);
  }}
}}
console.log('FIZGIG:' + JSON.stringify({{saved: saved}}));
"""
    r = _json_out(js, timeout=600)
    return [os.path.normpath(p) for p in r.get("saved", [])]


def place_image(path, doc_id="", name="", new_document=False, into_layer=None, box=None):
    """A picture on this computer as a new pixel layer: in the given (or current) document, or in a new one of its
    size. into_layer (an index from documents()) or box [x, y, w, h]: scaled to cover that area, centred, and put
    just above that layer. It is copied into the bridge folder first (Affinity reads only the Desktop). The
    document's id."""
    if not os.path.isfile(path):
        raise ToolError(f"{path} isn't there.")
    if os.path.splitext(path)[1].lower() not in PICTURES:
        raise ToolError("Only pictures can be placed in Affinity.")
    dest_dir = bridge_dir("to Affinity")
    dest = os.path.join(dest_dir, os.path.basename(path))
    if os.path.abspath(dest) != os.path.abspath(path):
        stem, ext = os.path.splitext(dest)
        n = 2
        while os.path.exists(dest):
            dest = f"{stem}_{n}{ext}"
            n += 1
        shutil.copy2(path, dest)
    name = name or os.path.splitext(os.path.basename(path))[0]
    if into_layer is not None and not box and not new_document:
        docs = documents()
        d = next((x for x in docs if doc_id in (x["id"], x["title"])) if doc_id else
                 (x for x in docs if x.get("current")), None)
        layer = next((l for l in (d or {}).get("layers", []) if l["index"] == int(into_layer)), None)
        if layer is None or not layer.get("box"):
            raise ToolError(f"No layer {into_layer} with an area in that document (see affinity_documents).")
        box = layer["box"]
    if box:                                     # the picture cut to the area's shape, centred: it fits exactly
        from PIL import Image
        with Image.open(dest) as im:
            im.load()
        want = float(box[2]) / max(1.0, float(box[3]))
        w, h = im.size
        if abs(w / h - want) > 0.005:
            cw, ch = (round(h * want), h) if w / h > want else (w, round(w / want))
            im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
            dest = os.path.splitext(dest)[0] + "_fit.png"
            im.save(dest)
        into_layer = None
    js = _DOC + f"""
const {{ Bitmap, RasterFormat }} = require('/rasterobject.js');
const {{ RasterNodeDefinition }} = require('/nodes.js');
const {{ NewDocumentOptions, UnitType }} = require('/document.js');
const bm = Bitmap.loadFromFile({json.dumps(dest.replace(os.sep, "/"))}, RasterFormat.RGBA8);
let d;
if ({json.dumps(bool(new_document))} || !Document.current) {{
  const o = NewDocumentOptions.createDefault();
  o.units = UnitType.Pixel; o.width = bm.width; o.height = bm.height; o.dpi = 72; o.createArtboard = false; o.isTransparentBackground = true;
  d = Document.create(o);
}} else d = pick({json.dumps(doc_id)});
const {{ Transform }} = require('/geometry.js');
const {{ Selection }} = require('/selections.js');
const def = RasterNodeDefinition.create(RasterFormat.RGBA8);
def.bitmap = bm;
const all = []; for (const l of d.layers) all.push(l);
const target = {json.dumps(into_layer)} === null ? null : all[{json.dumps(into_layer)}];
d.addNode(def);
const node = d.selection.firstNode;
try {{ node.userDescription = {json.dumps(name)}; }} catch (e) {{}}
let area = {json.dumps(box)};
if (!area && target) {{ const b = target.spreadBaseBox; area = [b.x, b.y, b.width, b.height]; }}
let placed = null;
if (area) {{
  const now = node.spreadBaseBox;
  const s = Math.max(area[2] / bm.width, area[3] / bm.height);
  const w = bm.width * s, h = bm.height * s;
  const t = Transform.createTranslate(area[0] + (area[2] - w) / 2 - now.x * s, area[1] + (area[3] - h) / 2 - now.y * s)
    .multiply(Transform.createScale(s, s));
  d.applyTransform(t, Selection.create(d, node));
  const b = node.spreadBaseBox; placed = [b.x, b.y, b.width, b.height].map(v => Math.round(v));
}}
console.log('FIZGIG:' + JSON.stringify({{id: String(d.sessionUuid), width: bm.width, height: bm.height, placed: placed}}));
"""
    return _json_out(js)


def view(doc_id="", spread=0):
    """The document as Affinity draws it (JPEG, longest side 1024): (base64, mime)."""
    c = client()
    if not doc_id:
        doc_id = next((d["id"] for d in documents() if d.get("current")), "")
        if not doc_id:
            raise ToolError("No document is open in Affinity.")
    content, bad = c.call_tool("render_spread", {"document_session_uuid": doc_id, "spread_index": int(spread)})
    if bad:
        raise ToolError(f"Affinity: {content_text(content)}")
    for item in content:
        if item.get("type") == "image":
            return item["data"], item.get("mimeType", "image/jpeg")
    raise ToolError("Affinity sent no picture.")


def copy_into(paths, folder):
    """Copy exported pictures into a training folder (never over a file there). The names written."""
    os.makedirs(folder, exist_ok=True)
    out = []
    for p in paths:
        dest = os.path.join(folder, os.path.basename(p))
        stem, ext = os.path.splitext(dest)
        n = 2
        while os.path.exists(dest):
            dest = f"{stem}_{n}{ext}"
            n += 1
        shutil.copy2(p, dest)
        out.append(os.path.basename(dest))
    return out


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
