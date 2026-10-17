"""What a LoRA does, measured: the Profiler for the standard-layer (driver) families.

Two instruments, written into one report:

* Weights (no model, instant): every module's real update dW = (alpha / rank) * up @ down, through its singular
  values - taken exactly from two QR factorisations and an r x r SVD, never forming dW (a LoKR's are the products of
  its factors' singular values). From them: each block's true size, and how much of the LoRA lives in its top 1, 2,
  4, 8 ... directions - what a Fast SVD extract to that rank keeps.

* Rendering (the workbench engine, a few minutes): the LoRA on fixed seeds whole, off, with each group of
  neighbouring blocks ALONE and with each group LEFT OUT (and, in the thorough pass, each single block left out).
  Measured per condition as a share of what the whole LoRA does: how far the picture moves, and - given photos of the
  subject - how much likeness the trigger prompt gets (face embedder) and how much a plain class prompt drifts toward the
  subject (bleed). Groups, because likeness is spread over many blocks: switch one off and its neighbours cover for
  it (measured on a Krea 2 character LoRA: no single block's removal cost more than a few % of its likeness).

The report keeps the `<name>.json` sidecar schema the Repair Studio cross-link reads (hash + top_active_blocks).
"""
import base64
import datetime
import html as _html
import io
import json
import os

import numpy as np
import torch

OUTSIDE = "outside"
KS = (1, 2, 4, 8, 16, 32, 64, 128)


# ---- weights ----------------------------------------------------------------------------------------------------
def _lowrank_svals(up, down):
    """Singular values of up @ down (out x r @ r x in) without forming it: up = Qu Ru, down^T = Qd Rd, so
    up @ down = Qu (Ru Rd^T) Qd^T and the r x r core carries every singular value."""
    _, ru = torch.linalg.qr(up.double())
    _, rd = torch.linalg.qr(down.double().T)
    return torch.linalg.svdvals(ru @ rd.T)


def weight_stats(desc, lora_path):
    """-> {"blocks": {block: {"norm", "retained": {k: share}}}, "retained": {k: share}, "max_rank", "modules",
    "rank_for": {0.9/0.95/0.99: k}}. Block ids follow the family's block map; modules outside it are OUTSIDE."""
    from safetensors import safe_open
    from fizgig.families.lorafile import (block_of, get_up, loha_delta, loha_modules, lokr_factors, lokr_modules,
                                          lora_pairs)
    blocks = block_of(desc)
    mods = []                                    # (block, singular values, rank)
    with safe_open(lora_path, "pt") as f:
        for mod, down, up, alpha in lora_pairs(desc, f.keys()):
            d = f.get_tensor(down).float()
            u = get_up(f, up).float()
            d, u = d.reshape(d.shape[0], -1), u.reshape(u.shape[0], -1)
            r = d.shape[0]
            a = float(f.get_tensor(alpha).float().reshape(-1)[0]) if alpha else float(r)
            mods.append((blocks.get(mod, OUTSIDE), _lowrank_svals(u, d) * (a / r), r))
        for mod, stem in lokr_modules(desc, f.keys()):
            w1, w2, scale = lokr_factors(f, stem)
            s = torch.outer(torch.linalg.svdvals(w1.double()), torch.linalg.svdvals(w2.double())).flatten()
            s = torch.sort(s, descending=True).values * scale
            mods.append((blocks.get(mod, OUTSIDE), s, int((s > s[0] * 1e-6).sum()) if len(s) else 0))
        for mod, stem in loha_modules(desc, f.keys()):
            s = torch.linalg.svdvals(loha_delta(f, stem).double())
            mods.append((blocks.get(mod, OUTSIDE), s, int((s > s[0] * 1e-6).sum()) if len(s) else 0))
    if not mods:
        raise RuntimeError(f"No LoRA modules found — is this a {desc.display_name} LoRA?")
    max_rank = max(r for _, _, r in mods)
    ks = [k for k in KS if k < max_rank] + [max_rank]

    def _curve(sel):
        tot = sum(float((s ** 2).sum()) for s in sel) or 1.0
        return {k: sum(float((s[:k] ** 2).sum()) for s in sel) / tot for k in ks}

    out = {}
    for b in dict.fromkeys(b for b, _, _ in mods):
        sel = [s for bb, s, _ in mods if bb == b]
        out[b] = {"norm": float(np.sqrt(sum(float((s ** 2).sum()) for s in sel))), "retained": _curve(sel)}
    all_s = [s for _, s, _ in mods]
    tot = sum(float((s ** 2).sum()) for s in all_s) or 1.0
    rank_for = {t: next((k for k in range(1, max_rank + 1)
                         if sum(float((s[:k] ** 2).sum()) for s in all_s) / tot >= t), max_rank)
                for t in (0.9, 0.95, 0.99)}
    return {"blocks": out, "retained": _curve(all_s), "max_rank": max_rank, "modules": len(mods),
            "rank_for": rank_for}


# ---- rendering --------------------------------------------------------------------------------------------------
def _small(img, side=128):
    from PIL import Image
    return np.asarray(img.convert("RGB").resize((side, side), Image.LANCZOS), dtype=np.float32) / 255.0


def _dist(a, b):
    """How different two renders look: mean absolute difference at 128 px (enough to rank conditions, cheap)."""
    return float(np.abs(a - b).mean())


def _thumb(img, side=220):
    from PIL import Image
    im = img.convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ablation_order(engine):
    """The primary's blocks in block-map order."""
    have = engine.primary_block_ids
    return [b.id for g in engine.block_groups() for b in g.blocks if b.id in have]


def windows(engine):
    """[(window id, label, [block ids])]: each big block group cut into about seven runs of neighbouring blocks,
    each small group (text fusion, input / output) whole."""
    have = engine.primary_block_ids
    out = []
    for g in engine.block_groups():
        bs = [b for b in g.blocks if b.id in have]
        if not bs:
            continue
        if len(bs) <= 6:
            out.append((g.label.lower().replace(" ", "_"), g.label, [b.id for b in bs]))
            continue
        size = max(2, round(len(bs) / 7))
        for i in range(0, len(bs), size):
            chunk = bs[i:i + size]
            label = chunk[0].label if len(chunk) == 1 else f"{chunk[0].label}-{chunk[-1].label.split()[-1]}"
            out.append((f"{chunk[0].id}..{chunk[-1].id}", label, [b.id for b in chunk]))
    return out


def conditions(engine, per_block=False):
    """[(id, label, blocks on, or None = the LoRA off)]: whole, off, each window alone, each window left out, and
    (per_block) each block left out."""
    blocks = ablation_order(engine)
    every = set(blocks)
    wins = windows(engine)
    conds = [("full", "Whole LoRA", every), ("none", "LoRA off", None)]
    conds += [(f"keep:{w}", f"Only {l}", set(bs)) for w, l, bs in wins]
    conds += [(f"drop:{w}", f"Without {l}", every - set(bs)) for w, l, bs in wins]
    if per_block:
        labels = {b.id: b.label for g in engine.block_groups() for b in g.blocks}
        conds += [(f"off:{b}", f"Without {labels.get(b, b)}", every - {b}) for b in blocks]
    return conds


def renders_needed(engine, n_seeds, with_class, per_block=False):
    return len(conditions(engine, per_block)) * n_seeds * (2 if with_class else 1)


def run_ablation(engine, *, prompt, class_prompt="", seeds=(1,), width=768, height=768, embed=None, baselines=(),
                 per_block=False, on_progress=None):
    """Render the primary (already loaded on `engine`) under every condition, on every seed, for the trigger prompt
    and (if given) the class prompt. embed(PIL) -> unit face embedding or None; baselines: unit embeddings of the
    subject. Raises the engine's RenderCancelled when request_cancel() is called."""
    from fizgig.families.workbench import PRIMARY
    conds = conditions(engine, per_block)
    prompts = [("trigger", prompt)] + ([("class", class_prompt)] if class_prompt.strip() else [])
    total = len(conds) * len(seeds) * len(prompts)
    done = 0
    base = np.stack(baselines) if len(baselines) else None

    def _score(img):
        if base is None or embed is None:
            return None
        e = embed(img)
        return None if e is None else float((base @ e).mean())

    res = {"blocks": ablation_order(engine), "windows": windows(engine), "conds": [(c, l) for c, l, _ in conds],
           "seeds": list(seeds), "size": [width, height], "prompt": prompt, "class_prompt": class_prompt,
           "thumbs": {}, "pics": {}, "scores": {}}
    state = engine.default_state(width, height)
    # exact attention for a measurement: comfy-kitchen's INT8 kernel (~1.6% per call) would ride on every score
    _i8a_was = getattr(engine, "int8_attention", False)
    engine.int8_attention = False
    try:
        for kind, text in prompts:
            res["pics"][kind], res["scores"][kind], res["thumbs"][kind] = {}, {}, {}
            for si, seed in enumerate(seeds):
                for cid, _label, on in conds:
                    for b, bs in state.blocks.items():
                        bs.primary_enabled = on is None or b in on
                    engine.net.set_enabled(PRIMARY, on is not None)
                    engine.net.set_outside(PRIMARY, not cid.startswith("keep:"))
                    img = engine.generate_preview(state, seed=int(seed), prompt=text, width=width, height=height)
                    res["pics"][kind].setdefault(cid, []).append(_small(img))
                    res["scores"][kind].setdefault(cid, []).append(_score(img))
                    if si == 0:
                        res["thumbs"][kind][cid] = _thumb(img)
                    done += 1
                    if on_progress:
                        on_progress(done, total, kind, cid)
    finally:
        engine.int8_attention = _i8a_was
        engine.net.set_enabled(PRIMARY, True)
        engine.net.set_outside(PRIMARY, True)
        for bs in state.blocks.values():
            bs.primary_enabled = True
        engine.apply_state(state)
    return summarise(res)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarise(res):
    """Per condition, as shares of what the whole LoRA does: picture change and (with faces) likeness / bleed -
    for a window ALONE (what it makes by itself) and LEFT OUT (what the rest cannot make without it)."""
    out = {k: v for k, v in res.items() if k != "pics"}
    out["alone"], out["lost"], out["gain"], out["score"] = {}, {}, {}, {}
    for kind, pics in res["pics"].items():
        full, none = pics["full"], pics["none"]
        whole = float(np.mean([_dist(a, b) for a, b in zip(full, none)])) or 1e-9
        sc = {c: _mean(v) for c, v in res["scores"][kind].items()}
        out["score"][kind] = sc
        gain = sc["full"] - sc["none"] if sc.get("full") is not None and sc.get("none") is not None else None
        out["gain"][kind] = gain
        faces = gain is not None and gain > 0.03
        alone, lost = {}, {}
        for cid, _ in res["conds"]:
            if cid in ("full", "none"):
                continue
            if cid.startswith("keep:"):
                alone[cid[5:]] = {
                    "change": float(np.mean([_dist(a, c) for a, c in zip(none, pics[cid])])) / whole,
                    "score": (sc[cid] - sc["none"]) / gain if faces and sc.get(cid) is not None else None}
            else:
                lost[cid] = {
                    "change": float(np.mean([_dist(a, c) for a, c in zip(full, pics[cid])])) / whole,
                    "score": (sc["full"] - sc[cid]) / gain if faces and sc.get(cid) is not None else None}
        out["alone"][kind], out["lost"][kind] = alone, lost
    return out


def verdicts(abl):
    """{window id: (tag, text)} from what each window gives alone and what is lost without it."""
    out = {}
    trig, cls = abl["alone"].get("trigger", {}), abl["alone"].get("class", {})
    lost = abl["lost"].get("trigger", {})
    for w, _l, _bs in abl["windows"]:
        a, c = trig.get(w, {}), cls.get(w, {})
        like, bleed, ch = a.get("score"), c.get("score"), a.get("change", 0.0)
        ld = lost.get(f"drop:{w}", {})
        if ch < 0.08 and ld.get("change", 0.0) < 0.08 and (like is None or like < 0.1):
            out[w] = ("quiet", "Little effect alone or missing: a candidate to switch off for a smaller file")
        elif bleed is not None and like is not None and bleed >= 0.25 and bleed > like + 0.15:
            out[w] = ("bleed", "Gives more bleed than likeness: turn this down first")
        elif like is not None and like >= 0.25 and (bleed is None or bleed < like - 0.15):
            out[w] = ("likeness", "Gives likeness with less bleed: keep")
        elif like is not None and like >= 0.25:
            out[w] = ("both", "Gives likeness and bleed together")
        elif ch >= 0.25:
            out[w] = ("look", "Shapes the picture (style, light, composition)")
        else:
            out[w] = ("minor", "Minor on its own")
    return out


def repair_settings(abl, down=0.5):
    """What the profile suggests for Repair Studio: ({block id: (enabled, strength)}, [notes]). Groups that give
    more bleed than likeness are turned down to `down`; groups with little effect are switched off; every other
    block stays as trained."""
    vd = verdicts(abl)
    blocks, notes = {}, []
    for w, label, bs in abl["windows"]:
        tag = vd[w][0]
        if tag == "bleed":
            blocks.update({b: (True, down) for b in bs})
            notes.append(f"{label} at {down:g} (more bleed than likeness)")
        elif tag == "quiet":
            blocks.update({b: (False, 1.0) for b in bs})
            notes.append(f"{label} off (little effect)")
    return blocks, notes


# ---- report -----------------------------------------------------------------------------------------------------
_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;background:#16181d;color:#e6e6e6;margin:0;padding:28px 34px;}
h1{font-size:22px;margin:0 0 4px;} h2{font-size:16px;margin:0 0 6px;} .sub{color:#9aa0aa;font-size:13px;margin-bottom:22px;}
.card{background:#20232b;border:1px solid #30343d;border-radius:12px;padding:18px 22px;margin-bottom:18px;}
.lead{color:#c3cdd9;font-size:13px;margin:0 0 12px;line-height:1.5;}
.note{color:#9aa0aa;font-size:12.5px;margin-top:10px;line-height:1.5;} .big{font-size:28px;font-weight:700;}
.kpis{display:flex;gap:16px;flex-wrap:wrap;} .kpi{background:#272b34;border-radius:10px;padding:12px 16px;min-width:170px;}
.kpi .lab{color:#9aa0aa;font-size:12px;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;}
.cell{background:#272b34;border-radius:10px;padding:8px;font-size:12px;border:2px solid transparent;}
.cell img{width:100%;border-radius:6px;display:block;margin-bottom:6px;} .cell b{font-size:12.5px;}
.tag-bleed{border-color:#ff6b8b;} .tag-likeness{border-color:#2ec4b6;} .tag-both{border-color:#ffd166;}
.tag-look{border-color:#4ea8ff;} .tag-quiet{opacity:.6;}
.bars{display:flex;flex-direction:column;gap:2px;margin-top:5px;} .bar{height:6px;border-radius:3px;background:#2c3038;overflow:hidden;}
.bar i{display:block;height:100%;} .lc{color:#2ec4b6;} .bc{color:#ff6b8b;} .cc{color:#4ea8ff;}
table{width:100%;border-collapse:collapse;font-size:13px;} td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #2c3038;}
th{color:#9aa0aa;font-weight:600;} .pill{display:inline-block;padding:2px 9px;border-radius:9px;font-size:12px;margin:2px;}
"""


_CAT_COLOR = {"identity": "#70AD47", "look": "#5B9BD5", "style_ident_overlap": "#5BB3A6",     # Repair Studio's
              "style_composition": "#5B9BD5", "ident_details_overlap": "#B8A547", "details": "#ED7D31"}


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def _bar(x, color):
    w = 0 if x is None else max(0, min(100, 100 * x))
    return f'<div class="bar"><i style="width:{w:.0f}%;background:{color}"></i></div>'


NOISE = 0.1          # shares this close to zero are within seed-to-seed noise


def _metric_rows(change, like, bleed, mode):
    """Each measurement as a plain sentence. mode "alone": what the group gives by itself; mode "lost": what
    happens to the whole LoRA when the group is switched off."""
    def line(cls, color, text, x):
        return f'<span class="{cls}">{text}</span>' + (_bar(x, color) if x is not None and x >= NOISE else "")

    def pc(x):
        return f"{100 * abs(x):.0f}%"
    rows = []
    if mode == "alone":
        if change is not None:
            rows.append(line("cc", "#4ea8ff", f"Changes the picture by {pc(change)}", change))
        if like is not None:
            rows.append(line("lc", "#2ec4b6", f"Gives {pc(like)} of the likeness" if like >= NOISE
                             else "Gives almost no likeness", like))
        if bleed is not None:
            rows.append(line("bc", "#ff6b8b", f"Gives {pc(bleed)} of the bleed" if bleed >= NOISE
                             else "Gives almost no bleed", bleed))
    else:
        if change is not None:
            rows.append(line("cc", "#4ea8ff", f"Picture changes by {pc(change)}", change))
        if like is not None:
            rows.append(line("lc", "#2ec4b6", f"Likeness drops by {pc(like)}" if like >= NOISE else
                             f"Likeness rises by {pc(like)}" if like <= -NOISE else "Likeness about the same", like))
        if bleed is not None:
            rows.append(line("bc", "#ff6b8b", f"Bleed drops by {pc(bleed)}" if bleed >= NOISE else
                             f"Bleed rises by {pc(bleed)}" if bleed <= -NOISE else "Bleed about the same", bleed))
    return f'<div class="bars">{"".join(rows)}</div>'


def _img(b64):
    return f'<img src="data:image/jpeg;base64,{b64}">' if b64 else ""


def write_report(desc, lora_path, stats, abl, output_html, labels):
    """HTML report + Repair Studio sidecar. abl may be None (weights only)."""
    from fizgig.utils.lora_files import compute_lora_hash
    name = os.path.basename(lora_path)
    lora_hash = compute_lora_hash(lora_path)
    lab = lambda b: labels.get(b, "Outside the blocks" if b == OUTSIDE else b)
    order = [b.id for g in desc.load_driver().block_map() for b in g.blocks]
    pos = lambda b: order.index(b) if b in order else 10 ** 6
    parts = []

    rf = stats["rank_for"]
    kpis = [("Rank in the file", str(stats["max_rank"])), ("Rank holding 95%", str(rf[0.95])),
            ("Rank holding 99%", str(rf[0.99]))]
    vd = {}
    if abl is not None:
        vd = verdicts(abl)
        g = abl["gain"].get("trigger")
        if g is not None and g > 0.03:
            kpis.append(("Likeness the LoRA adds", f"+{100 * g:.0f} pts"))
        gb = abl["gain"].get("class")
        if gb is not None:
            kpis.append(("Bleed into the plain prompt", f"+{100 * gb:.0f} pts" if gb > 0.03 else "none measurable"))
    parts.append('<div class="card"><div class="kpis">' + "".join(
        f'<div class="kpi"><div class="lab">{_html.escape(k)}</div><div class="big">{_html.escape(v)}</div></div>'
        for k, v in kpis) + "</div></div>")
    cats = dict(desc.block_categories)
    if cats:
        chips = "".join(f'<span title="{_html.escape(lab(b))}" style="display:inline-block;min-width:26px;padding:4px 0;'
                        f'margin:2px;border-radius:5px;text-align:center;font-size:11px;color:#111;'
                        f'background:{_CAT_COLOR.get(cats.get(b), "#555")}">{_html.escape(b.split("_")[-1])}</span>'
                        for b in order if b in cats)
        parts.append(f'<div class="card"><h2>{_html.escape(desc.display_name)} block map</h2><p class="lead">'
                     f'<span style="color:{_CAT_COLOR["identity"]}"><b>ID</b></span>: the blocks that carry a '
                     f'person\'s identity, measured with this Profiler across character LoRAs. '
                     f'<span style="color:{_CAT_COLOR["look"]}"><b>Look</b></span>: the blocks that shape the '
                     "picture (style, light, composition). Repair Studio colours its sliders the same way, and Fast "
                     "Identity Mode on the Training tab trains only the ID blocks.</p>" + chips + "</div>")

    # suggestions
    sug = []
    if abl is not None:
        wl = {w: l for w, l, _ in abl["windows"]}
        trig, cls = abl["alone"].get("trigger", {}), abl["alone"].get("class", {})
        bleeders = sorted((w for w in wl if vd[w][0] == "bleed"), key=lambda w: -(cls.get(w, {}).get("score") or 0))
        if bleeders:
            sug.append("To cut bleed, turn these down first in Repair Studio: " + ", ".join(
                f"<b>{_html.escape(wl[w])}</b> (alone: {_pct(cls[w]['score'])} of the bleed, "
                f"{_pct(trig.get(w, {}).get('score'))} of the likeness)" for w in bleeders[:3]) + ".")
        keepers = sorted((w for w in wl if vd[w][0] == "likeness"), key=lambda w: -(trig[w]["score"] or 0))
        if keepers:
            sug.append("Likeness with the least bleed comes from " + ", ".join(
                f"<b>{_html.escape(wl[w])}</b>" for w in keepers[:3]) + ": keep these.")
        if abl["gain"].get("class") is not None and abl["gain"]["class"] > 0.15 and not bleeders:
            sug.append("This LoRA bleeds, and the bleed sits in the same blocks as the likeness, so no block "
                       "setting removes one without the other. Training changes are the fix (captions that name "
                       "the person only by the trigger, or a regularisation set).")
        quiet = [w for w in wl if vd[w][0] == "quiet"]
        if quiet:
            sug.append(", ".join(f"<b>{_html.escape(wl[w])}</b>" for w in quiet) + " barely change the picture: "
                       "switching them off in Repair Studio gives a smaller file that looks the same.")
        if bleeders or quiet:
            sug.append("<b>Open in Repair Studio</b> on the Profiler tab loads the LoRA with these settings on the "
                       "sliders, side by side with the original, ready to save as a new file.")
    if rf[0.95] < stats["max_rank"]:
        sug.append(f"95% of this LoRA's change fits in rank <b>{rf[0.95]}</b> (99% in rank {rf[0.99]}): "
                   "Extract → Fast SVD at that rank gives a smaller file with almost nothing lost.")
    if sug:
        parts.append('<div class="card"><h2>What to do with it</h2><ul style="margin:6px 0 0;padding-left:18px;'
                     'line-height:1.7">' + "".join(f"<li>{s}</li>" for s in sug) + "</ul></div>")

    if abl is not None:
        th = abl["thumbs"]
        trig, cls = abl["alone"].get("trigger", {}), abl["alone"].get("class", {})
        lt, lc = abl["lost"].get("trigger", {}), abl["lost"].get("class", {})
        has_cls = "class" in th
        head = [f'<div class="cell">{_img(th["trigger"]["full"])}<b>Whole LoRA</b></div>',
                f'<div class="cell">{_img(th["trigger"]["none"])}<b>LoRA off</b></div>']
        cells = []
        for w, l, bs in abl["windows"]:
            tag, text = vd[w]
            cells.append(f'<div class="cell tag-{tag}">{_img(th["trigger"].get("keep:" + w))}'
                         f'<b>Only {_html.escape(l)}</b>'
                         + _metric_rows(trig[w]["change"], trig[w]["score"], cls.get(w, {}).get("score"), "alone")
                         + f'<div class="note" style="margin-top:6px">{_html.escape(text)}</div></div>')
        parts.append('<div class="card"><h2>Each group of blocks on its own</h2><p class="lead">Each picture is '
                     "the LoRA with <b>only this group switched on</b>. The percentages are shares of what the whole "
                     "LoRA does: <i>Gives 43% of the likeness</i> means these blocks alone get the face 43% of the way "
                     "to the whole LoRA's likeness; <i>Gives 31% of the bleed</i> means they alone make a prompt "
                     "without the trigger 31% as like your subject as the whole LoRA does. This section shows where "
                     "likeness and bleed come from.</p>"
                     f'<div class="grid">{"".join(head + cells)}</div></div>')
        cells = []
        for w, l, bs in abl["windows"]:
            d = lt.get("drop:" + w, {})
            cells.append(f'<div class="cell">{_img(th["trigger"].get("drop:" + w))}<b>Without {_html.escape(l)}</b>'
                         + _metric_rows(d.get("change"), d.get("score"), lc.get("drop:" + w, {}).get("score"),
                                        "lost")
                         + "</div>")
        parts.append('<div class="card"><h2>Each group left out</h2><p class="lead">Each picture is the whole LoRA '
                     "with <b>this group switched off</b> and everything else on, compared with the whole LoRA. "
                     "<i>Likeness drops by 19%</i> means the face loses 19% of what the LoRA added; <i>Bleed drops "
                     "by 24%</i> means a prompt without the trigger looks 24% less like your subject (the good "
                     "direction); <i>about the same</i> means the other blocks make up for this group.</p>"
                     f'<div class="grid">{"".join(cells)}</div></div>')
        offs = [c for c, _ in abl["conds"] if c.startswith("off:")]
        if offs:
            cells = []
            for c in offs:
                b = c[4:]
                d = lt.get(c, {})
                dot = (f'<span style="color:{_CAT_COLOR[cats[b]]}">● </span>' if cats.get(b) in _CAT_COLOR else "")
                cells.append(f'<div class="cell">{_img(th["trigger"].get(c))}<b>{dot}Without {_html.escape(lab(b))}</b>'
                             + _metric_rows(d.get("change"), d.get("score"), lc.get(c, {}).get("score"), "lost")
                             + "</div>")
            parts.append('<div class="card"><h2>Each block left out</h2><p class="lead">As above, one block at a '
                         "time: the whole LoRA with just this block switched off. Likeness often barely moves here, "
                         f'because the neighbouring blocks make up for any single one.</p><div class="grid">{"".join(cells)}'
                         "</div></div>")
        if has_cls:
            cc = [f'<div class="cell">{_img(th["class"]["none"])}<b>Base model</b></div>',
                  f'<div class="cell">{_img(th["class"]["full"])}<b>With the LoRA</b></div>']
            top = sorted((w for w, _, _ in abl["windows"]), key=lambda w: -(cls.get(w, {}).get("score") or 0))[:4]
            wl = {w: l for w, l, _ in abl["windows"]}
            cc += [f'<div class="cell">{_img(th["class"].get("keep:" + w))}<b>Only {_html.escape(wl[w])}</b>'
                   f'{_metric_rows(None, None, cls[w]["score"], "alone")}</div>' for w in top]
            parts.append(f'<div class="card"><h2>Bleed check: “{_html.escape(abl["class_prompt"])}”</h2>'
                         '<p class="lead">This prompt has no trigger word, so with the LoRA loaded it should still '
                         "look like the base model's picture. Any drift toward your subject is bleed. After the base "
                         "model and the whole LoRA come the groups that cause the most bleed on their own (only that "
                         f'group switched on).</p><div class="grid">{"".join(cc)}</div></div>')
        parts.append(f'<div class="note" style="margin:-6px 0 18px 4px">Prompt: {_html.escape(abl["prompt"])} · '
                     f'{len(abl["seeds"])} seed(s) · {abl["size"][0]}×{abl["size"][1]}. Likeness and bleed are measured by '
                     f"face recognition ({_html.escape(str(abl.get('embedder') or 'ArcFace'))}) against your subject "
                     "photos. Groups overlap in what they do, so "
                     "the percentages don't add up to 100%; anything under 10% either way is within "
                     "seed-to-seed noise.</div>")

    # rank card
    ks = list(stats["retained"].keys())
    head = "".join(f"<th>rank {k}</th>" for k in ks)
    whole = f"<tr><td><b>Whole LoRA</b></td>" + "".join(f"<td>{_pct(stats['retained'][k])}</td>" for k in ks) + "</tr>"
    rows = []
    for b in sorted(stats["blocks"], key=pos):
        rv = stats["blocks"][b]["retained"]
        rows.append(f"<tr><td>{_html.escape(lab(b))}</td>" + "".join(f"<td>{_pct(rv.get(k))}</td>" for k in ks) + "</tr>")
    parts.append(f'<div class="card"><h2>How much rank this LoRA uses</h2><p class="lead">The share of the LoRA\'s '
                 "change kept when it is cut to each rank: its real update, alpha included, through its singular "
                 "values. This is exactly what Extract → Fast SVD keeps at that rank.</p>"
                 f'<table><tr><th></th>{head}</tr>{whole}</table><details><summary class="note" style="cursor:pointer">'
                 f'Every block</summary><table><tr><th></th>{head}</tr>{"".join(rows)}</table></details></div>')

    tot = sum(v["norm"] for v in stats["blocks"].values()) or 1.0
    mx = max(v["norm"] for v in stats["blocks"].values()) or 1.0
    wr = "".join(f'<tr><td>{_html.escape(lab(b))}</td><td style="width:60%">{_bar(v["norm"] / mx, "#9b8cff")}</td>'
                 f'<td>{100 * v["norm"] / tot:.1f}%</td></tr>'
                 for b, v in sorted(stats["blocks"].items(), key=lambda kv: pos(kv[0])))
    parts.append('<div class="card"><h2>Where the weights are</h2><p class="lead">Size of each block\'s real update '
                 "(alpha included). Size is not effect: the rendered sections above show what the picture actually "
                 f'does.</p><details><summary class="note" style="cursor:pointer">Show</summary><table>{wr}</table>'
                 "</details></div>")

    title = f"{desc.display_name} LoRA profile"
    page = (f'<!doctype html><html><head><meta charset="utf-8">'
            + (f'<meta name="fizgig-lora-hash" content="{lora_hash}">' if lora_hash else "")
            + f"<title>{_html.escape(title)} — {_html.escape(name)}</title><style>{_CSS}</style></head><body>"
            f"<h1>{_html.escape(name)}</h1><div class=\"sub\">{_html.escape(title)} · {stats['modules']} modules · "
            f"{'measured by rendering' if abl is not None else 'weights only'} · "
            f"{datetime.datetime.now():%d %b %Y %H:%M}</div>{''.join(parts)}</body></html>")
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(page)

    # sidecar: Repair Studio's inline panel ranks blocks; per-block measurements when there are any, else weights
    if abl is not None and any(c.startswith("off:") for c, _ in abl["conds"]):
        score = {c[4:]: v["change"] for c, v in abl["lost"]["trigger"].items() if c.startswith("off:")}
    else:
        score = {b: v["norm"] for b, v in stats["blocks"].items() if b != OUTSIDE}
    s_tot = sum(max(v, 0.0) for v in score.values()) or 1.0
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    data = {
        "version": 2, "hash": lora_hash, "lora_path": lora_path, "lora_name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "architecture": desc.key, "profile_kind": "rendered" if abl is not None else "weights",
        "html_report": os.path.basename(output_html),
        "top_active_blocks": [{"name": b, "category": "", "pct": round(100 * max(v, 0.0) / s_tot, 1)}
                              for b, v in ranked[:8]],
        "top_static_blocks": [{"name": b, "norm": round(v["norm"], 5)}
                              for b, v in sorted(stats["blocks"].items(), key=lambda kv: -kv[1]["norm"])[:8]],
        "rank_for": {str(k): v for k, v in rf.items()},
    }
    if abl is not None:
        data["groups"] = [{"id": w, "label": l, "blocks": bs, "verdict": vd[w][0],
                           "alone": {k: abl["alone"][k].get(w) for k in abl["alone"]},
                           "left_out": {k: abl["lost"][k].get("drop:" + w) for k in abl["lost"]}}
                          for w, l, bs in abl["windows"]]
        data["gain"] = abl["gain"]
    sidecar = output_html.rsplit(".", 1)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_html, sidecar
