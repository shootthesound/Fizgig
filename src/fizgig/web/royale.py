"""The web UI's LoRA Royale (the desktop tab's): render every epoch of a run on one seed, crossfade between them, score
their likeness to the subject, export the morph, promote the winner. The renders run on the GPU worker
(fizgig.web.studio royale_render); everything here is CPU work in the server - scanning, picking the epochs, likeness
(ArcFace), the clips (fizgig.lora_royale.export), copies."""

import os
import re
import shutil
import threading

from fizgig.lora_royale.scan import run_name_for_folder, scan_checkpoints

_lock = threading.Lock()
_like = {"gen": 0, "scores": {}, "ref": "", "best": None}          # the last scoring, for its render generation


def families():
    from fizgig.families.registry import training_families
    return [d for d in training_families() if "royale" in (d.workbench or ()) and d.training_ready]


def scan(folder):
    """{items: [[label, path]], summary} - a run's epochs, or any LoRAs in the folder compared by file name."""
    cps = scan_checkpoints(folder)
    if not cps:
        return {"items": [], "summary": "No .safetensors checkpoints found in this folder.", "run": ""}
    labels = [e for e, _ in cps]
    summary = (f"{len(cps)} checkpoints found (epochs {labels[0]}–{labels[-1]})." if all(str(x).isdigit() for x in labels)
               else f"{len(cps)} LoRAs found — compared by filename.")
    return {"items": [[e, p] for e, p in cps], "summary": summary, "run": run_name_for_folder(folder)}


def pick(items, max_renders):
    """Max renders: that many, evenly spread across the run, the first and last always in (the desktop's rule)."""
    if str(max_renders).lower() == "all":
        return items
    n = int(max_renders)
    if len(items) <= n:
        return items
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def family_of(paths, candidates):
    """The one model these LoRAs are for (its key), None when unknown; raises on a mix (issue #62)."""
    from fizgig.families.registry import family_of_lora
    seen = []
    for p in paths:
        try:
            d = family_of_lora(p)
        except Exception:
            d = None
        if d is not None and d.key in candidates:
            seen.append((p, d))
    if not seen:
        return None
    first = seen[0][1]
    for p, d in seen[1:]:
        if d.key != first.key:
            raise ValueError(f"{os.path.basename(p)} was trained for {d.display_name}, but this selection also has "
                             f"{first.display_name} files (e.g. {os.path.basename(seen[0][0])}). Pick LoRAs of one model.")
    return first.key


def _images(rendered):
    from PIL import Image
    out = []
    for label, _path, png in rendered:
        with Image.open(png) as im:
            out.append((label, im.convert("RGB")))
    return out


def score(gen, rendered, refs):
    """Each render's face against up to 3 photos of the subject, the mean of its similarities (Image Prep's look
    match - with Auto, ArcFace for a person and DINOv2 for a furry / anime face - on the CPU, as the Samples gallery
    scores against its 3 baselines): {label: score or None (no face)}, and the best."""
    from fizgig.lora_royale import likeness
    if not likeness.available():
        raise ValueError("Likeness scoring needs the face tools - run install_fizgig.py to enable it.")
    refs = [str(r).strip() for r in ([refs] if isinstance(refs, str) else refs or []) if str(r).strip()][:3]
    if not refs:
        raise ValueError("Choose up to 3 photos of your subject to score likeness against.")
    gone = [os.path.basename(r) for r in refs if not os.path.isfile(r)]
    if gone:
        raise ValueError(f"{', '.join(gone)} isn't there any more - choose the photos again.")
    scored = likeness.score_renders(refs, _images(rendered))
    scores = {str(label): (None if sc != sc else float(sc)) for label, sc in scored}
    valid = [(lbl, sc) for lbl, sc in scores.items() if sc is not None]
    best = max(valid, key=lambda t: t[1])[0] if valid else None
    with _lock:
        _like.update(gen=gen, scores=scores, ref=refs[0], best=best)     # the clip shows the first photo
    return {"scores": scores, "best": best, "no_face": len(scores) - len(valid), "embedder": likeness.last_embedder_label}


def export(rendered, out_dir, run, fmt="MP4", speed="Normal", pingpong=True, show_epoch=True, brand=True,
           likeness_clip=False, gen=0):
    """The morph (or, likeness_clip, the subject beside each epoch with its score) as an MP4 or GIF in out_dir; its
    path and frame count. MP4 at the render's size, GIF capped at 768 (file size) - the desktop's rule."""
    from PIL import Image
    from fizgig.lora_royale import export as rexport
    fmt = "GIF" if str(fmt).upper() == "GIF" else "MP4"
    ext = ".gif" if fmt == "GIF" else ".mp4"
    images = _images(rendered)
    if not images:
        raise ValueError("Render some epochs first.")
    max_size = None if fmt == "MP4" else 768
    if likeness_clip:
        with _lock:
            like = dict(_like)
        if like["gen"] != gen or not like["scores"]:
            raise ValueError("Score likeness first - the clip shows each epoch's score beside your subject image.")
        with Image.open(like["ref"]) as ri:
            ref_img = ri.convert("RGB")
        scores = {lbl: (float("nan") if like["scores"].get(str(lbl)) is None else like["scores"][str(lbl)])
                  for lbl, _ in images}
        frames = rexport.build_likeness_frames(ref_img, images, scores, speed=speed, pingpong=pingpong, brand=brand,
                                               max_size=max_size)
        name = f"{run or 'lora'}-likeness{ext}"
    else:
        frames = rexport.build_frames(images, speed=speed, pingpong=pingpong, brand=brand, show_epoch=show_epoch,
                                      max_size=max_size)
        name = f"{run or 'lora'}-royale{ext}"
    if not frames:
        raise ValueError("No frames were produced.")
    os.makedirs(out_dir, exist_ok=True)
    out = _free(os.path.join(out_dir, name))
    (rexport.write_mp4 if fmt == "MP4" else rexport.write_gif)(frames, out, speed=speed)
    return {"path": out, "frames": len(frames)}


def save_stills(rendered, folder, run):
    """Every epoch's still as a full-size PNG, named so they sort in order (epochs zero-padded)."""
    os.makedirs(folder, exist_ok=True)
    saved = []
    for label, _path, png in rendered:
        name = (f"{run}-epoch{int(label):04d}.png" if str(label).isdigit()
                else f"{run}-{re.sub(r'[^A-Za-z0-9_-]', '_', str(label))}.png")
        dest = os.path.join(folder, name)
        shutil.copyfile(png, dest)
        saved.append(dest)
    return saved


def promote(path, label, run, out_dir, name=""):
    """A copy of the chosen epoch's LoRA to drop into ComfyUI (never over a file)."""
    if not path or not os.path.isfile(path):
        raise ValueError("Render epochs first, then slide to the one you want to promote.")
    name = (name or "").strip() or (f"{run or 'lora'}-epoch{label}-pick.safetensors" if str(label).isdigit()
                                    else f"{label}-pick.safetensors")
    if not name.lower().endswith(".safetensors"):
        name += ".safetensors"
    os.makedirs(out_dir, exist_ok=True)
    out = _free(os.path.join(out_dir, os.path.basename(name)))
    shutil.copy2(path, out)
    return out


def pick_columns(cols, spec):
    """The comparison sheet's epoch columns by the Epochs box: blank = all, "every N" / "/N" = every Nth with the last
    kept, "4,8,12" = those (the desktop's _royale_pick_epoch_columns)."""
    spec = (spec or "").strip().lower()
    if not spec or not cols:
        return cols
    m = re.fullmatch(r"(?:every\s*|/)(\d+)", spec)
    if m:
        n = max(1, int(m.group(1)))
        picked = cols[n - 1::n]
        if cols[-1] not in picked:
            picked.append(cols[-1])
        return picked or [cols[-1]]
    wanted = [t.strip() for t in spec.replace(" ", ",").split(",") if t.strip()]
    picked = [(lbl, p) for lbl, p in cols
              if any(str(lbl) == w or (str(lbl).isdigit() and w.isdigit() and int(lbl) == int(w)) for w in wanted)]
    return picked or cols


def save_travel(frames, out_dir, run, prefix, fmt="MP4", speed="Normal", pingpong=True, brand=True, badge=True,
                deflicker="None"):
    """A travel mode's frames as a clip, the options applied now (the desktop's _royale_sc_save: render once, save in
    either format with any options, no re-render)."""
    from PIL import Image
    from fizgig.lora_royale import export as rexport
    fmt = "GIF" if str(fmt).upper() == "GIF" else "MP4"
    imgs = []
    for png, _label in frames:
        with Image.open(png) as im:
            imgs.append(im.convert("RGB"))
    if not imgs:
        raise ValueError("Render the frames first.")
    if deflicker and deflicker != "None":
        imgs = rexport.deflicker_frames(imgs, sigma=(len(imgs) / 3.0) if deflicker == "Strong" else None)
    clip = rexport.frames_from_sequence(imgs, pingpong=pingpong, brand=brand,
                                        labels=[l for _p, l in frames] if badge else None,
                                        max_size=None if fmt == "MP4" else 768)
    os.makedirs(out_dir, exist_ok=True)
    out = _free(os.path.join(out_dir, f"{run or 'lora'}-{prefix}{'.mp4' if fmt == 'MP4' else '.gif'}"))
    (rexport.write_mp4 if fmt == "MP4" else rexport.write_gif)(clip, out, speed=speed)
    return {"path": out, "frames": len(clip)}


def copy_png(src, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    out = _free(os.path.join(out_dir, name))
    shutil.copyfile(src, out)
    return out


def _free(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"
