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
_conn = {"client": None, "desktop": "", "preamble": False, "used": 0.0, "busy": 0, "reaper": False}
IDLE = 30                   # seconds: then the connection is let go, so another app (Claude Desktop ...) can have it


def _reap():
    """Close the connection once it's idle: Affinity's server takes one session at a time."""
    while True:
        time.sleep(5)
        with _lock:
            c = _conn["client"]
            if c is not None and not _conn["busy"] and time.time() - _conn["used"] > IDLE:
                c.close()
                _conn.update(client=None, preamble=False)


class _Use:
    """A stretch of work on the connection: it is not reaped meanwhile."""

    def __enter__(self):
        with _lock:
            _conn["busy"] += 1
        return client()

    def __exit__(self, *exc):
        with _lock:
            _conn["busy"] -= 1
            _conn["used"] = time.time()


def _url_list():
    own = os.environ.get("AFFINITY_MCP_URL")
    return (own,) if own else URLS


def client():
    """The one connection to Affinity (its server handles one session at a time), made on first use and again when
    Affinity was restarted."""
    with _lock:
        _conn["used"] = time.time()
        if not _conn["reaper"]:
            _conn["reaper"] = True
            threading.Thread(target=_reap, daemon=True, name="affinity-idle").start()
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
        _conn.update(client=c, preamble=False)
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
    with _Use() as c:
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


def documents(only=""):
    """The open documents and each one's layers (top first); only: just the one with this id or title."""
    js = _DOC + """
const out = [];
const only = ONLY, bare = (t) => String(t).toLowerCase().replace(/\\.af$/, '');
for (const d of Document.all) {
  if (only && String(d.sessionUuid) !== only && bare(d.title) !== bare(only)) continue;
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
""".replace("ONLY", json.dumps(only or ""))
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


def place_image(path, doc_id="", name="", new_document=False, into_layer=None, box=None, replace=False):
    """A picture on this computer as a new pixel layer: in the given (or current) document, or in a new one of its
    size. into_layer (an index from documents()) or box [x, y, w, h]: cut to that area's shape and fitted to it;
    replace (with into_layer): that layer is hidden, never deleted. It is copied into the bridge folder first
    (Affinity reads only the Desktop). The document's id."""
    hide = int(into_layer) if replace and into_layer is not None else None
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
        docs = documents(doc_id)
        d = docs[0] if doc_id and docs else next((x for x in docs if x.get("current")), None)
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
const old = {json.dumps(hide)} === null ? null : all[{json.dumps(hide)}];
d.addNode(def);
const node = d.selection.firstNode;
try {{ node.userDescription = {json.dumps(name)}; }} catch (e) {{}}
let hidden = null;
if (old) try {{ d.setVisible(false, old); hidden = nm(old) || ('layer ' + {json.dumps(hide)}); }} catch (e) {{}}   // replaced: hidden, still there
let area = {json.dumps(box)};
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
console.log('FIZGIG:' + JSON.stringify({{id: String(d.sessionUuid), width: bm.width, height: bm.height, placed: placed, hidden: hidden}}));
"""
    return _json_out(js)


def _free_name(name):
    """name, or 'name 2', 'name 3' ... when a document by that name is open or its file is in the bridge folder."""
    safe = re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "Untitled"
    taken = {d["title"].lower().removesuffix(".af") for d in documents()}
    n, base = 1, safe
    while safe.lower() in taken or os.path.exists(os.path.join(bridge_dir(), safe + ".af")):
        n += 1
        safe = f"{base} {n}"
    return safe


def new_document(width, height, name="", background=None):
    """A new Affinity document of this many pixels (transparent, or filled with background [r, g, b]); saved into the
    bridge folder as <name>.af when named. Its id."""
    safe = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    path = ""
    if safe:
        safe = _free_name(safe)
        path = os.path.join(bridge_dir(), safe + ".af").replace(os.sep, "/")
    js = f"""
const {{ Document, NewDocumentOptions, UnitType }} = require('/document.js');
const o = NewDocumentOptions.createDefault();
o.units = UnitType.Pixel; o.width = {int(width)}; o.height = {int(height)}; o.dpi = 72; o.createArtboard = false;
o.isTransparentBackground = {json.dumps(background is None)};
const d = Document.create(o);
const bg = {json.dumps(list(background) if background else None)};
if (bg) {{
  const {{ ShapeRectangle }} = require('/shapes.js');
  const {{ ShapeNodeDefinition }} = require('/nodes.js');
  const {{ Rectangle }} = require('/geometry.js');
  const {{ FillDescriptor, BlendMode }} = require('/fills.js');
  const {{ RGBA8 }} = require('/colours.js');
  d.addNode(ShapeNodeDefinition.create(ShapeRectangle.create(), new Rectangle(0, 0, {int(width)}, {int(height)}),
    FillDescriptor.createSolid(RGBA8(bg[0], bg[1], bg[2]), BlendMode.Normal), null, null, null));
  try {{ d.layers.at(0).userDescription = 'Background'; }} catch (e) {{}}
}}
const p = {json.dumps(path)};
if (p) d.saveAs(p);
console.log('FIZGIG:' + JSON.stringify({{id: String(d.sessionUuid), title: {json.dumps(safe)}, width: d.widthPixels, height: d.heightPixels, saved: p}}));
"""
    return _json_out(js)


def _rgba(c, what):
    c = list(c or [])
    if len(c) not in (3, 4) or not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in c):
        raise ToolError(f"{what}: a colour is [r, g, b] or [r, g, b, a], each 0-255.")
    return [int(v) for v in c] + ([255] if len(c) == 3 else [])


def add_shape(doc_id="", box=None, fill=None, fill_to=None, angle=90, outline=None, outline_width=0, name=""):
    """A rectangle layer at box [x, y, w, h]: a solid fill, or a linear gradient fill -> fill_to (angle 90 = top to
    bottom, 0 = left to right), and/or an outline. Named, so a later step can find it. The layer's area."""
    if not box or len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise ToolError("box is [x, y, width, height] in pixels.")
    if fill is None and outline is None:
        raise ToolError("Give a fill, an outline, or both.")
    fill = _rgba(fill, "fill") if fill is not None else None
    fill_to = _rgba(fill_to, "fill_to") if fill_to is not None else None
    outline = _rgba(outline, "outline") if outline is not None else None
    js = _DOC + f"""
const {{ ShapeRectangle }} = require('/shapes.js');
const {{ ShapeNodeDefinition }} = require('/nodes.js');
const {{ Rectangle, Transform }} = require('/geometry.js');
const {{ FillDescriptor, BlendMode, GradientFill, GradientFillType }} = require('/fills.js');
const {{ RGBA8, Gradient }} = require('/colours.js');
const {{ LineStyleDescriptor }} = require('/linestyle.js');
const d = pick({json.dumps(doc_id)});
const b = {json.dumps([float(v) for v in box])}, f = {json.dumps(fill)}, f2 = {json.dumps(fill_to)}, o = {json.dumps(outline)};
const col = (c) => RGBA8(c[0], c[1], c[2], c[3]);
d.addNode(ShapeNodeDefinition.create(ShapeRectangle.create(), new Rectangle(b[0], b[1], b[2], b[3]),
  f ? FillDescriptor.createSolid(col(f), BlendMode.Normal) : null,
  o ? FillDescriptor.createSolid(col(o), BlendMode.Normal) : null,
  o ? LineStyleDescriptor.createDefault({float(outline_width or 2)}) : null, null));
const n = d.layers.at(d.layers.length - 1);
try {{ n.userDescription = {json.dumps(name)}; }} catch (e) {{}}
if (f && f2) {{
  const g = Gradient.create([{{ colour: col(f), position: 0, midpoint: 0.5 }}, {{ colour: col(f2), position: 1, midpoint: 0.5 }}]);
  const a = {float(angle)} * Math.PI / 180, bx = n.baseBoxInterface.baseBox;
  const xf = Transform.createTranslate(bx.x + bx.width / 2, bx.y + bx.height / 2)
    .multiply(Transform.createScale(bx.width, bx.height)).multiply(Transform.createRotate(a))
    .multiply(Transform.createTranslate(-0.5, 0));
  d.setBrushFillDescriptor(FillDescriptor.create(GradientFill.create(g, GradientFillType.Linear), false, xf, BlendMode.Normal, true), n);
}}
const s = n.spreadBaseBox;
console.log('FIZGIG:' + JSON.stringify({{name: nm(n), box: [s.x, s.y, s.width, s.height].map(v => Math.round(v))}}));
"""
    return _json_out(js)


def add_text(doc_id="", text="", x=0, y=0, size=48, bold=False, colour=None, font="", name=""):
    """A line of artistic text with its baseline's left end at (x, y): size in pixels, bold or regular, a colour,
    a font (its PostScript name; Arial when left out). The layer's area."""
    if not str(text).strip():
        raise ToolError("No text given.")
    colour = _rgba(colour if colour is not None else [255, 255, 255], "colour")
    font = font or ("Arial-BoldMT" if bold else "ArialMT")
    js = _DOC + f"""
const {{ ArtTextNodeDefinition }} = require('/nodes.js');
const {{ FillDescriptor, BlendMode }} = require('/fills.js');
const {{ RGBA8 }} = require('/colours.js');
const {{ StoryBuilder }} = require('/storybuilder.js');
const {{ Font, FontWeight, FontWidth }} = require('/fonts.js');
const d = pick({json.dumps(doc_id)});
const c = {json.dumps(colour)};
const sb = StoryBuilder.create();
sb.setToArtisticTextDefaultStyle(d.dpi, d.format);
const ga = sb.glyphAtts;
ga.font = Font.create({json.dumps(font)}, {"FontWeight.Bold" if bold else "FontWeight.Normal"}, false, FontWidth.Normal);
ga.height = {float(size)};
ga.brushFill = FillDescriptor.createSolid(RGBA8(c[0], c[1], c[2], c[3]), BlendMode.Normal);
sb.setGlyphAtts(ga);
sb.addText({json.dumps(str(text))});
d.addNode(ArtTextNodeDefinition.createFromStoryBuilder({{ x: {float(x)}, y: {float(y)} }}, sb));
const n = d.layers.at(d.layers.length - 1);
try {{ n.userDescription = {json.dumps(name or str(text)[:40])}; }} catch (e) {{}}
const s = n.spreadBaseBox;
const r = {{name: nm(n), box: [s.x, s.y, s.width, s.height].map(v => Math.round(v))}};
const over = Math.round(Math.max(s.x + s.width - d.widthPixels, -s.x, 0));
if (over > 0) r.off_page = 'runs ' + over + ' px off the page: hide this layer and write it smaller or in two lines';
console.log('FIZGIG:' + JSON.stringify(r));
"""
    return _json_out(js)


def hide_layer(doc_id="", layer=None, name=""):
    """Hide a layer (by index from documents(), bottom = 0, or by its name); never deleted. Its name."""
    js = _DOC + f"""
const d = pick({json.dumps(doc_id)});
const all = []; for (const l of d.layers) all.push(l);
const i = {json.dumps(None if layer is None else int(layer))}, want = {json.dumps(name)};
const n = i !== null ? all[i] : all.slice().reverse().find(l => String(nm(l)).trim() === String(want).trim());
if (!n) throw new Error('No such layer: ' + (i !== null ? i : want));
d.setVisible(false, n);
console.log('FIZGIG:' + JSON.stringify({{hidden: nm(n) || ('layer ' + i)}}));
"""
    return _json_out(js)


def save(doc_id="", name=""):
    """Save the document where it is; a new one goes into the bridge folder under name (else its title, else a dated
    name), with a number added when that name is taken."""
    docs = documents(doc_id) if doc_id else [d for d in documents() if d.get("current")]
    if not docs:
        raise ToolError(f"No open document {doc_id}." if doc_id else "No document is open in Affinity.")
    d = docs[0]
    path = ""
    if not d.get("path") or name:
        title = d["title"].removesuffix(".af")
        base = name or (title if title and not title.startswith("<") else time.strftime("Fizgig %Y-%m-%d %H%M%S"))
        path = os.path.join(bridge_dir(), _free_name(base) + ".af").replace(os.sep, "/")
    js = _DOC + f"""
const d = pick({json.dumps(d["id"])});
const p = {json.dumps(path)};
const r = p ? d.saveAs(p) : d.save();
console.log('FIZGIG:' + JSON.stringify({{saved: p || d.path, result: String(r)}}));
"""
    return _json_out(js)


def view(doc_id="", spread=0):
    """The document as Affinity draws it (JPEG, longest side 1024): (base64, mime)."""
    if not doc_id:
        doc_id = next((d["id"] for d in documents() if d.get("current")), "")
        if not doc_id:
            raise ToolError("No document is open in Affinity.")
    else:                                       # a title works too
        t, docs = doc_id.lower().removesuffix(".af"), documents()
        found = next((d["id"] for d in docs if doc_id == d["id"] or t == d["title"].lower().removesuffix(".af")), "")
        if not found:
            raise ToolError(f"No open Affinity document is called or has the id {doc_id!r}. Open: " + (
                "; ".join(f"{d['title']} (id {d['id']})" for d in docs) or "none") + ".")
        doc_id = found
    with _Use() as c:
        content, bad = c.call_tool("render_spread", {"document_session_uuid": doc_id, "spread_index": int(spread)})
    if bad:
        raise ToolError(f"Affinity: {content_text(content)}")
    for item in content:
        if item.get("type") == "image":
            return item["data"], item.get("mimeType", "image/jpeg")
    raise ToolError("Affinity sent no picture.")


def copy_into(paths, folder):
    """Copy exported pictures into a training folder (never over a file there; one already there, the same bytes, is
    left as it is). The names written or found."""
    import filecmp
    os.makedirs(folder, exist_ok=True)
    out = []
    for p in paths:
        dest = os.path.join(folder, os.path.basename(p))
        if os.path.isfile(dest) and filecmp.cmp(p, dest, shallow=False):
            out.append(os.path.basename(dest))      # the same picture again (the export run twice)
            continue
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
