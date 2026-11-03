"""Fizgig's abilities as MCP tools. Actions go through the web UI's own API (the same checks as the page: a busy GPU,
the Start checks, folder rules); the tools run inside the web UI's process and call it over loopback with its token.

ctx (from the web UI): api(method, path, body=None, qs=None) -> dict, and log(n) -> the console's last n lines."""
from __future__ import annotations

import base64
import io
import os
import re
import shutil
import time

from fizgig.bridge import affinity, scores
from fizgig.bridge.mcp import Server, Tool, ToolError, image, text

PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
INSTRUCTIONS = (
    "Fizgig trains LoRAs (and fine-tunes) for image models on this computer, and renders with them. A training "
    "folder holds pictures with a caption .txt beside each. The usual order: make or pick a training folder, caption "
    "it, start training with a model's preset, watch the run, look at its previews, render with the finished LoRA. "
    "The affinity_* tools carry pictures between Fizgig and Affinity by Canva (layers out to a training folder, "
    "renders and previews back in as layers). Training and rendering use the GPU one at a time; check fizgig_status "
    "first. Never stop or overwrite a run the user didn't ask you to.")


def _shrink(path, side=768, quality=85):
    """A picture as base64 JPEG no larger than `side` (what a model looks at), and its full size."""
    from PIL import Image
    from fizgig.utils.image import flatten
    with Image.open(path) as im:
        size = im.size
        im = flatten(im)
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode(), size


def _need(v, what):
    if not v:
        raise ToolError(f"Give {what}.")
    return v


def build(ctx) -> Server:
    api, log = ctx.api, ctx.log
    srv = Server("fizgig", _version(), INSTRUCTIONS)

    def get(path, **qs):
        r = api("GET", path, qs=qs)
        if isinstance(r, dict) and r.get("error"):
            raise ToolError(r["error"])
        return r

    def post(path, body=None):
        r = api("POST", path, body=body or {})
        if isinstance(r, dict) and r.get("error"):
            raise ToolError(r["error"])
        return r

    def tool(name, description, params=None, required=()):
        def wrap(fn):
            srv.add(Tool(name, description, params or {}, fn, list(required)))
            return fn
        return wrap

    S = {"type": "string"}
    I = {"type": "integer"}
    N = {"type": "number"}
    B = {"type": "boolean"}

    # -- what's going on
    @tool("fizgig_status", "What Fizgig is doing now: a training run (its model, LoRA name, stage, epoch), Studio, "
                           "captioning and Image Prep jobs, and the training folder picked on the Start tab.")
    def _status():
        run = get("/api/run")
        keep = {k: run.get(k) for k in ("state", "stage", "lora", "family", "output_dir", "image_folder", "running",
                                        "epoch", "epochs", "resume") if run.get(k) not in (None, "")}
        from fizgig.web import captions, prep_jobs, studio
        st = studio.status()
        return {"training": keep, "console_tail": log(8),
                "studio": {k: st.get(k) for k in ("state", "family", "primary", "rendering", "status")},
                "captioning": _short(captions.status()), "image_prep": _short(prep_jobs.status(0)),
                "training_folder": get("/api/state").get("image_folder", "")}

    @tool("list_models", "The models Fizgig can train (key, name, kinds), each with its built-in presets.")
    def _models():
        from fizgig.families.registry import training_families
        out = []
        for d in training_families():
            if not d.training_ready:
                continue
            out.append({"key": d.key, "name": d.display_name, "kinds": list(getattr(d, "kinds", ()) or ()),
                        "presets": [n for n, _ in (d.presets or ())]})
        return out

    # -- training folders
    def training_folder(folder):
        """A folder's path, or a training folder's name (as list_training_folders shows it) made its path."""
        if os.path.isdir(folder or ""):
            return folder
        named = os.path.join(get("/api/upload/root")["root"], str(folder or "").strip(" /\\"))
        if folder and os.path.isdir(named):
            return named
        raise ToolError(f"{folder} isn't a folder: give its path, or its name from list_training_folders.")

    @tool("list_training_folders", "The training folders (datasets) on this computer: path, pictures, how many "
                                   "are captioned.")
    def _datasets():
        d = get("/api/datasets")
        return {"root": d.get("root"), "folders": [{k: x.get(k) for k in ("name", "path", "pictures", "clips",
                                                                           "songs", "captioned")}
                                                   for x in d.get("datasets", [])]}

    @tool("create_training_folder", "Make a new training folder in Fizgig's training data folder and copy pictures "
                                    "(and same-named caption .txt files) into it.",
          {"name": S, "pictures": {"type": "array", "items": S, "description": "picture files to copy in"}},
          ["name"])
    def _create(name, pictures=None):
        root = get("/api/upload/root")["root"]
        safe = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
        folder = os.path.join(root, _need(safe, "a folder name"))
        if os.path.exists(folder):
            raise ToolError(f"{folder} is already there - pick another name, or add_pictures to it.")
        os.makedirs(folder)
        n = _copy_pictures(pictures or [], folder)
        return {"folder": folder, "copied": n}

    @tool("add_pictures", "Copy pictures (and their caption .txt files) into an existing training folder.",
          {"folder": S, "pictures": {"type": "array", "items": S}}, ["folder", "pictures"])
    def _add(folder, pictures):
        folder = training_folder(folder)
        return {"folder": folder, "copied": _copy_pictures(pictures, folder)}

    @tool("set_training_folder", "Pick the training folder the Start tab, Image Prep and Captions work on.",
          {"folder": S}, ["folder"])
    def _set_folder(folder):
        folder = training_folder(folder)
        post("/api/state", {"image_folder": folder})
        return {"training_folder": folder}

    @tool("list_pictures", "A training folder's pictures with their captions (and masks, for masked training).",
          {"folder": S}, ["folder"])
    def _pictures(folder):
        folder = training_folder(folder)
        r = get("/api/captions", folder=folder)
        return [{"name": it.get("name"), "caption": it.get("caption"), **({"mask": it["mask"]} if it.get("mask") else {})}
                for it in r.get("items", [])]

    @tool("look_at_picture", "See a picture (a training picture, a preview, a render): it comes back as an image, "
                             "at most 768 px.", {"path": S, "side": I}, ["path"])
    def _look(path, side=768):
        if not os.path.isfile(path):
            raise ToolError(f"{path} isn't there.")
        data, size = _shrink(path, max(128, min(int(side), 1536)))
        return [text(f"{os.path.basename(path)}, {size[0]}x{size[1]}"), image(data, "image/jpeg")]

    @tool("write_caption", "Write one picture's caption (its .txt beside it).",
          {"folder": S, "name": {"type": "string", "description": "the picture's file name"}, "caption": S},
          ["folder", "name", "caption"])
    def _caption(folder, name, caption):
        folder = training_folder(folder)
        post("/api/captions/save", {"folder": folder, "name": name, "text": caption})
        return {"ok": True}

    @tool("caption_folder", "Caption a training folder's pictures with Fizgig's AI captioner (Qwen3-VL, on the GPU) "
                            "and wait until they're done. task: training (a person), object, concept or style. "
                            "trigger: the word the LoRA is called by, added to every caption.",
          {"folder": S, "trigger": S, "task": {"type": "string", "enum": ["training", "object", "concept", "style",
                                                                          "short", "detailed"]},
           "overwrite": B}, ["folder"])
    def _caption_all(folder, trigger="", task="training", overwrite=False):
        folder = training_folder(folder)
        from fizgig.web import captions
        opts = get("/api/captions/options")
        if not opts.get("qwen_available"):
            raise ToolError("The AI captioner needs its model file set in Preferences (Krea 2's text encoder).")
        tasks = {t["key"]: t for t in opts.get("qwen_tasks", [])}
        t = tasks.get(task or "training") or tasks.get("training")
        t0 = time.time()
        if trigger:                             # the folder's trigger word, as the Captions tab keeps it
            post("/api/trigger", {"folder": folder, "word": trigger})
        post("/api/captions/ai", {"folder": folder, "trigger": trigger, "model": opts.get("qwen_model"),
                                  "task": t["label"], "instruction": t["instruction"], "overwrite": bool(overwrite),
                                  "max_tokens": t.get("tokens") or 120})
        _wait(lambda: captions.status().get("state") in ("idle", "loaded"), 1800, first=2.0)
        st = captions.status()
        return {"captioned": st.get("ok"), "failed": st.get("fail"), "seconds": round(time.time() - t0),
                "last_lines": st.get("log", [])[-3:]}

    @tool("make_masks", "Make masks for masked training (the subject white, the rest black). kind: 'person' or "
                        "'sam' with words saying what to keep.",
          {"folder": S, "kind": S, "include": S, "exclude": S, "overwrite": B}, ["folder"])
    def _masks(folder, kind="person", include="", exclude="", overwrite=False):
        folder = training_folder(folder)
        sam = kind.lower().startswith("sam")
        return post("/api/masks/make", {"folder": folder, "kind": "" if sam else kind, "sam": sam,
                                        "include": include, "exclude": exclude, "overwrite": bool(overwrite)})

    # -- training
    @tool("start_training", "Start training a LoRA: a model (list_models key), a training folder and a preset; "
                            "overrides change single settings by their form key (LORA_NAME, MAX_TRAIN_EPOCHS, "
                            "NETWORK_DIM, LEARNING_RATE, CAPTION_TRIGGER_WORD ...). The Start checks' warnings come "
                            "back first; call again with anyway=true to start regardless. A person's likeness in "
                            "minutes: model qwen_image21, preset '✨ Qwen 2.1 Fast Identity Mode (rank 8) - very close "
                            "to full-model likeness, ~1.5x faster', overrides {\"MAX_TRAIN_EPOCHS\": \"20\", "
                            "\"SAVE_EVERY_N_EPOCHS\": \"5\", \"SAMPLE_ENABLED\": false}. Two or more people in one LoRA "
                            "who must not leak onto anyone else: subjects + people_mode isolation.",
          {"model": S, "folder": S, "preset": S, "lora_name": S, "trigger": S, "overrides": {"type": "object"},
           "subjects": {"type": "array", "items": S, "description": "more subjects (people) in the same LoRA: up to "
                        "3 more training folders, each captioned with its own trigger word"},
           "people_mode": {"type": "string", "enum": ["off", "leave_others_alone", "isolation"], "description":
                           "off: a normal LoRA. leave_others_alone: the subject only appears with its trigger word "
                           "(DOP). isolation: Fizgig LoRA Isolation - two or more subjects, every other person keeps "
                           "their own face; needs subjects"},
           "anyway": B}, ["model", "folder"])
    def _start(model, folder, preset="", lora_name="", trigger="", overrides=None, subjects=None, people_mode="",
               anyway=False):
        from fizgig.families import form
        from fizgig.families.registry import get as get_family
        folder = training_folder(folder)
        if trigger:
            post("/api/trigger", {"folder": folder, "word": trigger})
        st = get("/api/state")
        values = dict(((st.get("values") or {}).get(model)) or {})
        if preset:
            values = post("/api/preset", {"family": model, "name": preset, "values": values}).get("values", values)
        values.update(overrides or {})
        if lora_name:
            values["LORA_NAME"] = lora_name
        more = [training_folder(f) for f in (subjects or []) if str(f).strip()]
        if len(more) > 3:
            raise ToolError("Up to 4 subjects in all: the training folder and 3 more.")
        if people_mode == "isolation" and not more:
            raise ToolError("Fizgig LoRA Isolation needs two or more subjects: give the others' folders in subjects.")
        desc = get_family(model)
        if more:                                # the Subjects card's "Who or what is this LoRA of?"
            values.update(dict((desc.multi_concept_defaults if desc else ()) or ()))
            values.update(FAMILY_MULTICONCEPT=True, FAMILY_CONCEPT_COUNT=form.CONCEPT_COUNTS[len(more) - 1])
            for key, f in zip(("FAMILY_CONCEPT_DIR", "FAMILY_CONCEPT_DIR3", "FAMILY_CONCEPT_DIR4"), more):
                values[key] = f
        elif subjects is not None:
            values["FAMILY_MULTICONCEPT"] = False
        if people_mode:                         # ... and "Your subject and other people"
            values.update(FAMILY_DOP=people_mode != "off", FAMILY_MULTI_PERSON=people_mode == "isolation")
        for _ in range(3):                      # the form's own rules, as the page applies them (Isolation's must-haves)
            force = post("/api/form/state", {"family": model, "values": values, "image_folder": folder}).get("force")
            if not force:
                break
            values.update(force)
        r = post("/api/start", {"family": model, "values": values, "image_folder": folder, "anyway": bool(anyway)})
        if r.get("warnings"):
            return {"started": False, "warnings": r["warnings"],
                    "next": "Tell the user these, then call start_training again with anyway=true if they agree."}
        if r.get("problems"):
            return {"started": False, "problems": r["problems"]}
        run = get("/api/run")
        return {"started": True, "lora": run.get("lora"), "output_dir": run.get("output_dir"),
                "stages": r.get("stages"), **({"fetching": r["fetching"]} if r.get("fetching") else {})}

    @tool("training_status", "The current run's state and its console's last lines.", {"lines": I})
    def _train_status(lines=30):
        return _run_now() | {"console": log(max(1, min(int(lines), 200)))}

    def _run_now():
        run = get("/api/run")
        return {k: v for k, v in run.items() if k not in ("values", "web_queue", "pod_stop", "move", "slider_live")
                and v not in (None, "", [], {})}

    @tool("wait_for_training", "Wait for the current run to finish (or pause, or fail), up to `minutes`; then its "
                               "state and its newest LoRA file. Use it when the user wants the result in one go.",
          {"minutes": N})
    def _wait_train(minutes=30):
        end = time.time() + max(0.1, float(minutes)) * 60
        time.sleep(3)
        while time.time() < end:
            run = get("/api/run")
            if not run.get("running") and run.get("state") not in ("starting", "running", "caching"):
                break
            time.sleep(5)
        run = _run_now()
        return run | {"newest_lora": _newest_lora(run.get("output_dir") or "", run.get("lora") or ""),
                      "console": log(6), "finished": not run.get("running")}

    # -- training on a rented pod (RunPod): this computer's GPU stays free
    @tool("cloud_train", "Train a LoRA on a rented RunPod GPU instead of this computer's: Fizgig rents the cheapest "
                         "card that can train it at or under max_price dollars an hour, waits for it (a pod that "
                         "doesn't come up in boot_minutes is deleted and the next offer tried, up to 3), sends the "
                         "training folder, trains there with the model and preset (as start_training), brings every "
                         "LoRA file home and deletes the pod. Fizgig itself stops it at max_total dollars or "
                         "max_minutes. It runs in the background: follow it with cloud_status (wait=true). If it "
                         "fails, train here with start_training instead.",
          {"model": S, "folder": S, "preset": S, "lora_name": S, "trigger": S, "overrides": {"type": "object"},
           "max_price": N, "max_total": N, "boot_minutes": N, "max_minutes": N}, ["model", "folder"])
    def _cloud(model, folder, preset="", lora_name="", trigger="", overrides=None, max_price=2.0, max_total=4.0,
               boot_minutes=15, max_minutes=75):
        from fizgig.bridge import cloud
        from fizgig.web import helpers, pod_stop
        s = helpers._srv()
        folder = training_folder(folder)
        try:
            return cloud.start(pod_stop.api_key(s.load_prefs()), folder, model, preset=preset, overrides=overrides,
                               lora_name=lora_name, trigger=trigger, dest_root=s.default_output_dir(),
                               max_price=float(max_price), max_total=float(max_total),
                               boot_minutes=float(boot_minutes), max_minutes=float(max_minutes))
        except ValueError as e:
            raise ToolError(str(e))

    @tool("cloud_status", "The cloud training job: what it's doing, the pods it rented, money spent so far, and when "
                          "done the LoRA files here (newest first). wait=true waits (up to `minutes`) until it's done "
                          "or failed.", {"wait": B, "minutes": N})
    def _cloud_status(wait=False, minutes=60):
        from fizgig.bridge import cloud
        end = time.time() + max(0.1, float(minutes)) * 60
        while wait and time.time() < end and cloud.status().get("state") not in ("done", "failed", None):
            time.sleep(10)
        st = cloud.status()
        if not st:
            return {"state": "none"}
        st["log"] = st.get("log", [])[-14:]
        return st

    @tool("cloud_cancel", "Cancel the cloud training job: its pods are deleted (nothing is brought home).")
    def _cloud_cancel():
        from fizgig.bridge import cloud
        return cloud.cancel()

    @tool("release_gpu", "Take Fizgig's models off the GPU (Studio's render model and the captioner), so another "
                         "program can use it. A training run is not touched.")
    def _release():
        from fizgig.web import captions, studio
        t0 = time.time()
        held = studio.holding_gpu()
        if held:
            post("/api/studio/reset")
        try:
            captions.unload()
        except Exception:
            pass
        return {"released": bool(held), "seconds": round(time.time() - t0, 1)}

    @tool("pause_training", "Pause the run at the end of this epoch (it saves, frees the GPU, and can resume).")
    def _pause():
        return post("/api/pause")

    @tool("stop_training", "Stop the run now, without saving this epoch. Only when the user asks.")
    def _stop():
        return post("/api/stop")

    # -- results
    @tool("list_runs", "Past and current training runs, newest first: model, LoRA name, output folder, training "
                       "folder.", {"limit": I})
    def _runs(limit=15):
        r = get("/api/runs")
        runs = r if isinstance(r, list) else r.get("runs", [])
        return [{k: x.get(k) for k in ("id", "family", "lora", "output_dir", "image_folder", "state", "started",
                                       "epoch") if x.get(k) not in (None, "")} for x in runs[:max(1, int(limit))]]

    @tool("list_loras", "A run's saved LoRA files (each epoch's checkpoint), newest first.",
          {"output_dir": S, "lora": {"type": "string", "description": "the run's LoRA name"}}, ["output_dir"])
    def _loras(output_dir, lora=""):
        out = []
        try:
            for e in os.scandir(output_dir):
                if e.name.lower().endswith(".safetensors") and (not lora or e.name.startswith(lora)):
                    m = re.search(r"-(\d{6})\.safetensors$", e.name)
                    out.append({"path": e.path, "epoch": int(m.group(1)) if m else None, "time": e.stat().st_mtime})
        except OSError:
            raise ToolError(f"{output_dir} isn't readable.")
        out.sort(key=lambda x: x["time"], reverse=True)
        return out[:100]

    @tool("list_previews", "A run's preview pictures (made during training): path, epoch, prompt number, newest "
                           "first. Look at one with look_at_picture.",
          {"output_dir": S, "lora": S, "limit": I}, ["output_dir"])
    def _previews(output_dir, lora="", limit=24):
        r = get("/api/samples", dir=os.path.join(output_dir, "sample"), run=lora)
        items = r.get("items", [])
        return {"prompts": r.get("prompts", []),
                "previews": [{"path": it.get("path"), "epoch": it.get("epoch"), "prompt": it.get("idx")}
                             for it in items[:max(1, int(limit))]]}

    @tool("render", "Render a picture with a LoRA (a file, or a run's LoRA name) on Studio's preview model, on the GPU: prompt, seed, size and the "
                    "LoRA's strength. Several pictures in one call (variations, or prompts) load the model once: much faster than one call each. Takes Studio over while it works. Returns the pictures' paths and a look at them. A "
                    "see-through picture (a sticker, a cut-out, a logo) on Qwen Image 2.1: put the subject in its own "
                    "wording, 'This is an RGBA image with transparency. <subject>. The image has alpha channel and "
                    "the background is transparent.' - the PNG keeps its alpha and goes into Affinity as a cut-out. A "
                    "picture that isn't of the LoRA's subject (a sticker, a logo, a scene): any LoRA of that model at "
                    "strength 0 renders with the plain model.",
          {"lora": S, "prompt": S, "seed": I, "width": I, "height": I, "strength": N,
           "variations": {"type": "integer", "description": "how many pictures of this prompt (seeds seed, seed+1 ...)"},
           "prompts": {"type": "array", "items": S, "description": "several prompts in one go (instead of prompt)"},
           "model": {"type": "string", "description": "the LoRA's model key; found from the file when left out"},
           "reference": {"type": "string", "description": "an edit model's before-picture"}},
          ["lora"])
    def _render(lora, prompt="", seed=42, width=1024, height=1024, strength=1.0, variations=1, prompts=None,
                model="", reference=""):
        todo = list(prompts or []) or [prompt] * max(1, min(int(variations or 1), 8))
        if not any(p.strip() for p in todo):
            raise ToolError("Give a prompt.")
        if not os.path.isfile(lora):                # a run's LoRA name: its final file, else its newest checkpoint
            name = os.path.basename(lora).removesuffix(".safetensors")
            r = get("/api/runs")
            for x in (r if isinstance(r, list) else r.get("runs", [])):
                d = x.get("output_dir") or ""
                if x.get("lora") == name and d:
                    final = os.path.join(d, name + ".safetensors")
                    lora = final if os.path.isfile(final) else (_newest_lora(d, name) or lora)
                    break
        return _studio_render(api, post, lora, todo, seed, width, height, strength, model, reference)

    @tool("score_pictures", "Score pictures of a person before one is used (no GPU needed): likeness to the person's "
                            "training folder (ArcFace), eyes open, facing the camera, one main face, sharpness. Returns "
                            "each picture's scores and a pick: the best likeness among those that pass.",
          {"paths": {"type": "array", "items": S}, "likeness_folder": {"type": "string", "description":
           "the person's training folder (a name or a path)"}}, ["paths"])
    def _score(paths, likeness_folder=""):
        r = scores.score(paths, training_folder(likeness_folder) if likeness_folder else "")
        for row in r["pictures"]:
            row["file"] = os.path.basename(row.pop("path"))
        return r

    # -- Affinity
    @tool("affinity_status", "Is Affinity by Canva reachable (its MCP server on)? Its Desktop bridge folder.")
    def _a_status():
        return affinity.status()

    @tool("affinity_documents", "The documents open in Affinity, each with its layers (top first; index counts from "
                                "the bottom, 0). document (an id or title): just that one.", {"document": S})
    def _a_docs(document=""):
        docs = affinity.documents(document)
        if document or len(docs) < 3:
            return docs
        out = []                                # many open: newest first, the layers in full for the current one only
        for d in reversed(docs):
            if not d.get("current"):
                d = {**d, "layers": f"{len(d['layers'])} layers (affinity_documents with this document lists them)"}
            out.append(d)
        return out

    @tool("affinity_to_training_folder", "Export layers from an Affinity document as PNGs into a training folder: "
                                         "the given layer indexes, else (selected=true) the selected layers, else "
                                         "every visible picture layer; whole=true exports the flattened document "
                                         "instead. A new folder is made when `folder` is a name rather than a path.",
          {"folder": S, "document": S, "layers": {"type": "array", "items": I}, "whole": B, "selected": B},
          ["folder"])
    def _a_out(folder, document="", layers=None, whole=False, selected=False):
        if not os.path.isabs(folder):
            root = get("/api/upload/root")["root"]
            folder = os.path.join(root, re.sub(r'[\\/:*?"<>|]+', "_", folder).strip(" ."))
        paths = affinity.export_layers(document, layers, whole, tag=os.path.basename(folder), selected=selected)
        if not paths:
            raise ToolError("Affinity exported nothing (no layers selected or visible?).")
        names = affinity.copy_into(paths, folder)
        return {"folder": folder, "pictures": names}

    @tool("affinity_place", "Put a picture (a render, a preview, a training picture) into Affinity as a new layer: in "
                            "the given or current document, or a new document of its size. into_layer (a layer index "
                            "from affinity_documents, e.g. an empty photo box) fits the picture over that layer's "
                            "area; render at that area's shape first so nothing is cropped.",
          {"path": S, "document": S, "name": S, "new_document": B, "into_layer": I,
           "box": {"type": "array", "items": N, "description": "[x, y, width, height] in pixels"},
           "replace": {"type": "boolean", "description": "with into_layer: the new picture takes that layer's place "
                                                         "and the old layer is hidden (not deleted)"}}, ["path"])
    def _a_place(path, document="", name="", new_document=False, into_layer=None, box=None, replace=False):
        return affinity.place_image(path, document, name, new_document, into_layer, box, replace)

    @tool("affinity_new_document", "Make a new Affinity document of this many pixels (transparent, or on a "
                                   "background colour [r, g, b]), named and saved in the Desktop's Fizgig Bridge "
                                   "folder (a name that is taken gets a number: use the title it returns). Lay pictures out on it with "
                                   "affinity_place's box.",
          {"width": I, "height": I, "name": S, "background": {"type": "array", "items": I}}, ["width", "height"])
    def _a_new(width, height, name="", background=None):
        return affinity.new_document(width, height, name, background)

    colour = {"type": "array", "items": N, "description": "[r, g, b] or [r, g, b, a], 0-255"}

    @tool("affinity_add_shape", "Draw a rectangle layer in an Affinity document: a background, a panel, a band, or an "
                                "empty photo frame for affinity_place's into_layer (a faint fill, e.g. alpha 20, and "
                                "an outline). fill alone is solid; fill + fill_to is a gradient (angle 90 = top to "
                                "bottom, 0 = left to right). Give every layer a name.",
          {"document": S, "box": {"type": "array", "items": N, "description": "[x, y, width, height] in pixels"},
           "fill": colour, "fill_to": colour, "angle": N, "outline": colour, "outline_width": N, "name": S},
          ["box", "name"])
    def _a_shape(box, name, document="", fill=None, fill_to=None, angle=90, outline=None, outline_width=3):
        return affinity.add_shape(document, box, fill, fill_to, angle, outline, outline_width, name)

    @tool("affinity_add_text", "Write one line of text in an Affinity document: (x, y) is the left end of its "
                               "baseline, size in pixels (a headline ~90-110 on a 1080 px wide page, body ~30), bold "
                               "or not, a colour. One call per line; check the layout with affinity_view after.",
          {"document": S, "text": S, "x": N, "y": N, "size": N, "bold": B, "colour": colour,
           "font": {"type": "string", "description": "a PostScript font name; Arial when left out"}, "name": S},
          ["text", "x", "y", "size"])
    def _a_text(text, x, y, size, document="", bold=False, colour=None, font="", name=""):
        return affinity.add_text(document, text, x, y, size, bold, colour, font, name)

    @tool("affinity_hide_layer", "Hide a layer in an Affinity document (by index from affinity_documents or by "
                                 "name), e.g. a line of text to write again; it is hidden, not deleted.",
          {"document": S, "layer": I, "name": S})
    def _a_hide(document="", layer=None, name=""):
        if layer is None and not name:
            raise ToolError("Give the layer's index or name.")
        return affinity.hide_layer(document, layer, name)

    @tool("affinity_save", "Save an Affinity document (a new one goes into the Desktop's Fizgig Bridge folder, under "
                           "name or its title; a taken name gets a number).", {"document": S, "name": S})
    def _a_save(document="", name=""):
        return affinity.save(document, name)

    @tool("affinity_view", "See an Affinity document as it looks now (at most 1024 px).", {"document": S})
    def _a_view(document=""):
        data, mime = affinity.view(document)
        return [text("The document in Affinity:"), image(data, mime)]

    return srv


def _newest_lora(out_dir, name):
    best = None
    try:
        for e in os.scandir(out_dir):
            if name and e.name.lower().endswith(".safetensors") and e.name.startswith(name) \
                    and (best is None or e.stat().st_mtime > best[0]):
                best = (e.stat().st_mtime, e.path)
    except OSError:
        return ""
    return best[1] if best else ""


def _copy_pictures(paths, folder):
    n = 0
    for p in paths:
        if not os.path.isfile(p) or os.path.splitext(p)[1].lower() not in PICTURES:
            continue
        dest = os.path.join(folder, os.path.basename(p))
        if os.path.exists(dest):
            continue
        shutil.copy2(p, dest)
        cap = os.path.splitext(p)[0] + ".txt"
        if os.path.isfile(cap):
            shutil.copy2(cap, os.path.splitext(dest)[0] + ".txt")
        n += 1
    return n


def _short(st):
    if not isinstance(st, dict):
        return st
    return {k: v for k, v in st.items() if k in ("running", "state", "done", "total", "status", "error", "message",
                                                 "kind", "folder")}


def _studio_render(api, post, lora, prompts, seed, width, height, strength, model, reference):
    from fizgig.web import studio
    if not os.path.isfile(lora):
        raise ToolError(f"{lora} isn't there.")
    if not model:
        from fizgig.families.registry import family_of_lora
        d = family_of_lora(lora)
        if d is None:
            raise ToolError("Which model is this LoRA for? Give model (a list_models key).")
        model = d.key
    st = studio.status()
    if st.get("state") != "ready" or os.path.normcase(st.get("primary") or "") != os.path.normcase(lora) \
            or st.get("family") != model:
        if st.get("state") not in ("idle", "error", None, ""):
            try:
                post("/api/studio/reset")
            except ToolError:
                pass
            _wait(lambda: studio.status().get("state") in ("idle", "error"), 120)
        post("/api/studio/load", {"family": model, "primary": lora, "donor": "", "fast": True, "turbo": True})
        if not _wait(lambda: studio.status().get("state") in ("ready", "error", "idle"), 900):
            raise ToolError("Studio is still loading the model; try again in a minute.")
        st = studio.status()
        if st.get("state") != "ready":
            raise ToolError(f"Studio couldn't load: {st.get('error') or st.get('status')}")
    from fizgig.user_data import user_file
    out_dir = user_file("renders")
    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(lora))[0])[:40]
    jobs = [(p, int(seed) + i) for i, p in enumerate(prompts)]
    saved, looks = [], []
    for p, s in jobs:                       # one model load, then each render warm
        state = {"prompt": p, "seed": s, "preview_width": int(width), "preview_height": int(height),
                 "primary_scale": float(strength)}
        if reference:
            state["ref_image_path"] = reference
        gen = post("/api/studio/render", {"state": state}).get("gen")
        _wait(lambda: not studio.status().get("rendering"), 900, first=0.5)
        st = studio.status()
        if st.get("error"):
            raise ToolError(f"The render failed: {st['error']}")
        path = studio.image("tweaked", gen or 0) or studio.image("baseline", gen or 0)
        if not path:
            raise ToolError("The render finished but left no picture.")
        keep = os.path.join(out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}_s{s}.png")
        shutil.copy2(path, keep)            # the next render reuses Studio's file
        saved.append(keep)
        if len(looks) < 4:
            looks.append(_shrink(keep, 768 if len(jobs) == 1 else 448))
    sizes = ", ".join(f"{w}x{h}" for (_d, (w, h)) in looks[:1])
    return [text("Rendered " + ("\n".join(saved)) + f" ({sizes}).")] + [image(d, "image/jpeg") for d, _ in looks]


def _wait(cond, seconds, first=0.3):
    time.sleep(first)
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.5)
    return False


def _version():
    import subprocess
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "describe", "--tags", "--always"], cwd=here, capture_output=True, text=True,
                           timeout=5, creationflags=0x08000000 if os.name == "nt" else 0)
        return r.stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"
