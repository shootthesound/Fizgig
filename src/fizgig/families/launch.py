"""The launch plan for a described family: plain settings in, everything a run needs out.

One place turns the settings of a run into what starts it - the dataset TOML, the cache and training commands in
order, the prompt files the previews read, and the problems that would stop it - so every front end that starts a
run (the desktop app, a web UI, a queue, a command line) launches it the same way.

`inputs` is a plain dict: the run's settings (the same keys the desktop app saves: LORA_NAME, NETWORK_DIM,
FAMILY_EDIT, ...) plus the values the app reads from its other tabs, all lower case:

    python, repo_dir            interpreter and Fizgig checkout the commands run from
    models                      {pref_key: path} for the family's model files (Preferences)
    image_folder, caption_ext   the training folder (Start tab) and its caption extension
    batch_size, megapixels      Dataset settings
    enable_bucket, no_upscale
    cache_root                  the cache folder from Preferences (each dataset gets its own folder inside it)
    blocks_swap                 the Block Swap box as typed ("Auto (detect from GPU)", "8", ...)
    enable_cache, resuming      whether the cache stages run; a resume (or fine-tune continuation) skips them,
    recache                     unless its picture size changed (Target Megapixels): then they run, and keep what fits
    loss_watch                  {"detect", "per_image_lr", "warmup", "recaption": bool}
    captioner, caption_trigger, caption_overrides   auto-recaption: model path, trigger word, edited instructions
    caption_concept             the Captions tab's "The concept is" words, for a Concept task
    recaption_triggers          auto-recaption's trigger word for each other folder ({key prefix: word})
    recaption_tasks             auto-recaption's (first try, second try) Captions tab task keys
    keep_checkpointing          a helper is set to train a share: gradient checkpointing stays on (both halves match)
    samples                     {"enabled", "every", "width", "height", "steps", "cfg", "negative", "seed",
                                 "at_first": ..., "prompts": [lines as typed on the Samples tab]}
    samples_dir                 where preview prompt files go (the output folder's sample/)
    edit_caption                the edit instruction typed in the Edit card

Switches (FAMILY_EDIT, FAMILY_SLIDER, ADAPTIVE_LR, SAVE_STATE, enable_cache, samples["enabled"], loss_watch[...],
...) must be real booleans: the string "false" counts as on. Numbers may be strings or numbers. The run's
settings the commands need are LORA_NAME, LORA_OUTPUT_DIR, DATASET_CONFIG (the path the plan's TOML is written
to), NETWORK_DIM, NETWORK_ALPHA, LEARNING_RATE, MAX_TRAIN_EPOCHS, SAVE_EVERY_N_EPOCHS, SEED; the number checks
also read LORA_LR_RATIO, GRADIENT_ACCUMULATION, MAX_GRAD_NORM (and ADAPTIVE_LR_MIN / _MAX with
Adaptive LR on, LOKR_FACTOR with LoKR). A missing one is a problem, never a crash.

plan() returns problems only when there are any: then it has no stages and no files, and nothing may be written or
started. Otherwise the caller creates plan.dirs, writes plan.files (the dataset TOML first), then runs the stages in
order. What the desktop app also does at Start, which a caller has to do itself: refuse to start while a run is
active; delete a stale <LORA_OUTPUT_DIR>/.pause_requested (the trainer would stop after one epoch) and a stale
.sample_override.json (it changes the previews); warn when the output drive is nearly full, or when a resume is
already at Max Train Epochs.
"""
import os
import re
from dataclasses import dataclass, field
from fizgig.utils.upright import upright_size

from fizgig.dataset.masks import is_mask_file

PRECISION_LABELS = {"auto": "Auto (fits your free VRAM)", "bf16": "bf16 (full precision)",
                    "int8": "INT8 (8-bit, fastest)", "nf4": "4-bit NF4 (smallest)",
                    "hqq": "4-bit HQQ (more accurate 4-bit)"}
PAIR_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Stage:
    name: str
    cmd: list


@dataclass
class LaunchPlan:
    dataset_toml: str = ""                           # written to inputs["DATASET_CONFIG"] (also in files)
    stages: list = field(default_factory=list)      # Stage, in the order they run
    dirs: list = field(default_factory=list)        # folders to create before the stages run
    files: list = field(default_factory=list)       # (path, text) to write before the stages run
    console: list = field(default_factory=list)     # lines to show the user
    problems: list = field(default_factory=list)    # anything here and the run must not start


def _toml_str(x):
    """A TOML basic string's inside: backslashes and quotes escaped."""
    return str(x).replace("\\", "\\\\").replace('"', '\\"')


def _s(x):
    """A value from the inputs as trimmed text ("" for None) - they may arrive as numbers."""
    return "" if x is None else str(x).strip()


def _model(inputs, desc, role):
    return desc.model_path(role, (inputs.get("models") or {}).get)


def edit_on(desc, inputs):
    return bool(desc.edit_training and inputs.get("FAMILY_EDIT"))


# the Edit pairs card's "What the edit uses" choice (a family with edit_references > 1)
EDIT_PHOTO_ONLY, EDIT_WITH_EXTRAS = "The photo only", "The photo + extra pictures"


def edit_extras_on(desc, inputs):
    """An edit with extra pictures from their own folder (photo_1, photo_2 ... for photo.png)."""
    return (edit_on(desc, inputs) and getattr(desc, "edit_references", 1) > 1
            and _s(inputs.get("FAMILY_EDIT_PICTURES")) == EDIT_WITH_EXTRAS)


def preview_extras(desc, inputs, photo, exts=PAIR_EXTS):
    """The Extra pictures folder's pictures for `photo` (photo_1, photo_2 ... by their number), up to the model's
    count; [] outside a multi-picture edit."""
    xdir = _s(inputs.get("FAMILY_EDIT_EXTRAS_DIR"))
    if not edit_extras_on(desc, inputs) or not xdir or not os.path.isdir(xdir):
        return []
    stem = os.path.splitext(os.path.basename(photo))[0].casefold()
    found = []
    for f in os.listdir(xdir):
        b, e = os.path.splitext(f)
        head, _, n = b.rpartition("_")
        if e.lower() in exts and n.isdigit() and head.casefold() == stem:
            found.append((int(n), os.path.join(xdir, f)))
    return [p for _, p in sorted(found)][:max(0, desc.edit_references - 1)]


def edit_extras(after_dir, extras_dir, exts=PAIR_EXTS, test_photo=""):
    """({edited photo: its extra pictures}, extras with no edited photo, extras named exactly as a photo) for the
    Extra pictures folder: photo_1.png, photo_2.png ... belong to photo.png. The test photo's own extras (test_1 ...
    for the previews) are neither."""
    photos = {os.path.splitext(f)[0].casefold(): f for f in os.listdir(after_dir)
              if os.path.splitext(f)[1].lower() in exts}
    found, orphans, misnamed = {}, [], []
    for f in sorted(os.listdir(extras_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in exts:
            continue
        if b.casefold() in photos:
            misnamed.append(f)
            continue
        stem, _, n = b.rpartition("_")
        owner = photos.get(stem.casefold()) if n.isdigit() else None
        test_stem = os.path.splitext(os.path.basename(test_photo))[0].casefold() if test_photo else ""
        if owner is None and n.isdigit() and test_stem and stem.casefold() == test_stem:
            continue                      # the test photo's extras: its previews use them
        if owner is None:
            orphans.append(f)
        else:
            found.setdefault(owner, []).append(f)
    return found, orphans, misnamed


def slider_on(desc, inputs, source=None):
    """Slider mode for a family that offers it; source "pairs" / "prompts" narrows it to one kind."""
    on = bool(desc.slider_training and inputs.get("FAMILY_SLIDER"))
    if on and source:
        offered = tuple(getattr(desc, "slider_sources", ("pairs", "prompts"))) or ("pairs", "prompts")
        have = inputs.get("FAMILY_SLIDER_SOURCE") or "pairs"
        on = (have if have in offered else offered[0]) == source      # a family with one source always uses it
    return on


def ft_on(desc, inputs):
    """A full fine-tune of the base model, for a family whose driver offers one (description.finetune)."""
    return bool(desc.finetune and inputs.get("FAMILY_FT"))


# ---- a paused run's continuation point (both apps' Resume) -------------------------------------------------------
STATE_MARKERS = ("training_state.json", "random_states_0.pkl")   # written last: a state folder without one is a
                                                                  # half-written save


def latest_state(out_dir, name):
    """The run's newest complete state folder (<name>-NNNNNN-state with its commit marker), or ""."""
    try:
        found = [d for d in os.listdir(out_dir) if re.fullmatch(re.escape(name) + r"-\d{6}-state", d)
                 and any(os.path.isfile(os.path.join(out_dir, d, m)) for m in STATE_MARKERS)]
    except OSError:
        return ""
    return os.path.join(out_dir, max(found)) if found else ""


def latest_ft_checkpoint(out_dir, name):
    """A fine-tune's newest checkpoint (<name>-NNNNNN.safetensors - a fine-tune has no state folders), or ""."""
    try:
        found = [f for f in os.listdir(out_dir) if re.fullmatch(re.escape(name) + r"-\d{6}\.safetensors", f)]
    except OSError:
        return ""
    return os.path.join(out_dir, max(found)) if found else ""


def ft_continuation(path):
    """(next_window, epochs_done, n_windows) from a fine-tune checkpoint's metadata (the trainers stamp them); the
    -NNNNNN in the file name wins for the epochs. Anything missing reads as 0."""
    import json
    import struct
    md = {}
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            md = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        pass
    def num(k):
        try:
            return int(float(md.get(k) or 0))
        except (TypeError, ValueError):
            return 0
    epochs = num("fizgig_ft_epochs_done")
    m = re.search(r"-(\d{6})\.safetensors$", os.path.basename(path or ""))
    if m:
        epochs = int(m.group(1))
    return num("fizgig_next_start_window"), epochs, num("fizgig_ft_n_windows")


def ft_rotations_left(path, rotations, rotate_every):
    """What a family fine-tune paused at path has left of `rotations` (whole rotations of its cycle), and the
    continuation's FAMILY_FT_CONTINUE - as the desktop's Resume works it out."""
    nxt, done, nw = ft_continuation(path)
    per = max(1, _ft_int({"x": rotate_every}, "x", 1))
    rot = max(1, _ft_int({"x": rotations}, "x", 1))
    done_rot = round(done / (nw * per)) if nw else 0
    return rot - done_rot, done_rot, {"checkpoint": path, "start_window": nxt, "epochs_done": done}


def ft_reg_dir(desc, inputs):
    """A fine-tune's regularisation folder, when fine-tune is on and one is set (else "")."""
    return _s(inputs.get("FAMILY_FT_REG_DIR")) if ft_on(desc, inputs) and getattr(desc, "ft_regularisation", True)         else ""


def _ft_reg_mult(inputs):
    try:
        return max(0.0, float(str(inputs.get("FAMILY_FT_REG_MULT", "") or 0.2)))
    except ValueError:
        return None


def _ft_int(inputs, key, default):
    try:
        return max(1, int(float(str(inputs.get(key, "") or default))))
    except ValueError:
        return None


def area_blocks(desc, inputs):
    """The block ids the Model Area to Train setting trains (FAMILY_TRAIN_AREA; "Custom" = FAMILY_TRAIN_BLOCKS), or
    [] for every block."""
    name = _s(inputs.get("FAMILY_TRAIN_AREA"))
    if name == "Custom":
        return [b for b in (inputs.get("FAMILY_TRAIN_BLOCKS") or []) if b]
    area = next((a for a in desc.train_areas if a[0] == name), None)
    return list(area[1]) if area else []


def timestep_flags(inputs):
    """The Timestep Range (MIN/MAX_TIMESTEP, 0-1000 as the Training tab shows it) as the trainer's 0-1 window."""
    out = []
    for key, flag in (("MIN_TIMESTEP", "--min_timestep"), ("MAX_TIMESTEP", "--max_timestep")):
        try:
            v = float(_s(inputs.get(key)))
        except ValueError:
            continue
        out += [flag, f"{min(max(v, 0.0), 1000.0) / 1000.0:g}"]
    return out


def uncached_pictures(toml_text):
    """How many pictures in the dataset TOML's folders have no latent or no text cache in their cache folder (a
    resume skips caching, so a cache deleted by hand would leave it nothing to train on)."""
    import toml   # not tomllib: that's Python 3.11+, and 3.10 installs are supported
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    exts = {e.lower() for e in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}
    try:
        sets = toml.loads(toml_text or "").get("datasets") or []
    except toml.TomlDecodeError:
        return 0
    n = 0
    for ds in sets:
        img, cdir = ds.get("image_directory"), ds.get("cache_directory")
        if not img or not cdir or not os.path.isdir(img):
            continue
        names = os.listdir(cdir) if os.path.isdir(cdir) else []
        # <stem>_<WxH>_<arch>.safetensors and <stem>_<arch>_te.safetensors: the stem is all but the last two parts
        te = {"_".join(f[:-len(".safetensors")].split("_")[:-2]) for f in names if f.endswith("_te.safetensors")}
        lat = {"_".join(f.split("_")[:-2]) for f in names
               if f.endswith(".safetensors") and not f.endswith("_te.safetensors")}
        n += sum(1 for f in os.listdir(img) if os.path.splitext(f)[1].lower() in exts and not is_mask_file(f)
                 and (os.path.splitext(f)[0] not in lat or os.path.splitext(f)[0] not in te))
    return n


def caches(desc, inputs):
    """Whether the cache stages run: not on a resume (the cache is already built) unless it trains at a different
    size from now on (recache: the latents that still fit are kept), not for a prompt slider (no photos), not with
    Enable Cache off."""
    resuming = inputs.get("resuming") or bool(_s(inputs.get("RESUME_TRAINING")))
    if multi_person_on(desc, inputs) and not slider_on(desc, inputs, "prompts"):
        return True      # Fizgig LoRA Isolation builds its side-by-sides every start: always cache them (only what's new)
    return bool(inputs.get("enable_cache", True) and (not resuming or inputs.get("recache"))
                and not slider_on(desc, inputs, "prompts"))


# ---------------------------------------------------------------------------------------------------- pairs
def _pair_order(stem):
    """The dataset loader's order for a photo's partners: photo.<ext> first, then photo_1, photo_2 ..."""
    def key(name):
        bn = os.path.splitext(name)[0].casefold()
        if bn == stem.casefold():
            return 0
        suffix = bn.rsplit("_", 1)[-1]
        return int(suffix) + 1 if suffix.isdigit() else 999999
    return key


def pairs(after_dir, before_dir, exts=PAIR_EXTS, limit=1):
    """(photos without a partner, photos with more than `limit`, first partner's path) for an edit's or a slider's
    two folders, matched the way the dataset loader matches them: a partner of photo.png is photo.<ext> or
    photo_<anything>.<ext>, ignoring case. exts: what counts as an item (a clip family's sliders add .mp4). limit:
    partners a photo may have (a multi-reference edit: the family's edit_references)."""
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in exts
                     and not is_mask_file(f))
    missing, multiple, first = [], [], None
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in exts or is_mask_file(f):
            continue
        m = [x for x in befores if x.casefold().startswith(b.casefold() + ".")    # IMG_1 pairs with img_1
             or x.casefold().startswith(b.casefold() + "_")]
        if not m:
            missing.append(f)
        elif len(m) > limit:
            multiple.append(f)
        first = first or (os.path.join(before_dir, sorted(m, key=_pair_order(b))[0]) if m else None)
    return missing, multiple, first


def pair_problems(after_dir, before_dir, caption_ext=".txt", exts=PAIR_EXTS):
    """(photos in after_dir without a caption, pairs whose shapes differ - photos only; a clip pair's size and
    length are checked when it is cached)."""
    from PIL import Image
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in exts
                     and not is_mask_file(f))
    uncaptioned, shapes = [], []
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in exts or is_mask_file(f):
            continue
        if not os.path.exists(os.path.join(after_dir, b + caption_ext)):
            uncaptioned.append(f)
        m = sorted((x for x in befores if x.casefold().startswith(b.casefold() + ".")
                    or x.casefold().startswith(b.casefold() + "_")), key=_pair_order(b))
        if m and e.lower() in PAIR_EXTS:     # the original (the first): extra references may be any shape
            try:
                with Image.open(os.path.join(after_dir, f)) as ia, Image.open(os.path.join(before_dir, m[0])) as ib:
                    (wa, ha), (wb, hb) = upright_size(ia), upright_size(ib)
                    ra, rb = wa / ha, wb / hb
                if abs(ra - rb) / ra > 0.02:
                    shapes.append(f)
            except OSError:
                pass
    return uncaptioned, shapes


def edit_instruction(inputs):
    """The edit instruction for previews: the typed instruction, else the first edited photo's caption file."""
    text = _s(inputs.get("edit_caption"))
    if text:
        return text
    folder = _s(inputs.get("image_folder"))
    ext = inputs.get("caption_ext") or ".txt"
    try:
        for f in sorted(os.listdir(folder)):
            b, e = os.path.splitext(f)
            cap = os.path.join(folder, b + ext)
            if e.lower() in PAIR_EXTS and os.path.exists(cap):
                with open(cap, encoding="utf-8") as fh:
                    line = fh.read().strip().splitlines()
                if line and line[0].strip():
                    return line[0].strip()
    except OSError:
        pass
    return ""


# the Training tab's window-size choice -> the planner's cap on parts per window (0 = as many as fit)
FT_WINDOW_SIZES = (("Auto - as few windows as fit", 0), ("At most 3 parts per window", 3),
                   ("At most 2 parts per window", 2), ("One part per window (most headroom)", 1))


def ft_max_parts(inputs):
    return dict(FT_WINDOW_SIZES).get(_s(inputs.get("FAMILY_FT_MAX_PARTS")), 0)


# ---------------------------------------------------------------------------------------------------- problems
def problems(desc, inputs):
    """What the family's own checks refuse: an edit's or a slider's pairs, a prompt slider's prompts, and every
    required model file (plus the training adapter while it is on) set and on disk."""
    errors = []
    folder = _s(inputs.get("image_folder"))
    if ft_on(desc, inputs):
        for key, label in (("FAMILY_FT_ROTATIONS", "Rotations"), ("FAMILY_FT_SAVE_EVERY", "Save every"),
                           ("FAMILY_FT_ROTATE_EVERY", "Epochs per window")):
            if _ft_int(inputs, key, 1) is None:
                errors.append(f"Fine-tune: {label} must be a whole number of 1 or more")
        reg = ft_reg_dir(desc, inputs)
        if reg and not os.path.isdir(reg):
            errors.append(f"Fine-tune: the regularisation folder {reg} does not exist - clear the box to train "
                          f"without one")
        elif reg and _ft_reg_mult(inputs) is None:
            errors.append("Fine-tune: the regularisation LR multiplier must be a number")
    ext = inputs.get("caption_ext") or ".txt"
    if edit_on(desc, inputs) and slider_on(desc, inputs):
        errors.append("Edit LoRA and Slider are both on - a run is one kind of LoRA: pick Edit or Slider")
    if edit_on(desc, inputs):
        before = _s(inputs.get("FAMILY_EDIT_DIR"))
        ref = _s(inputs.get("FAMILY_EDIT_REF"))
        if not before or not os.path.isdir(before):
            errors.append("Edit LoRA is on: set the Originals folder (Training tab, Edit pairs)")
        elif os.path.isdir(folder):
            limit = 1          # one original per edited photo; extra pictures have their own folder
            missing, multiple, _ = pairs(folder, before, limit=limit)
            if missing:
                errors.append(f"Edit LoRA: {len(missing)} edited photo(s) have no original with the same file "
                              f"name in {before} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Edit LoRA: {len(multiple)} edited photo(s) match more than one original (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one original per edited photo, with the same "
                              "file name" + (" (extra pictures go in their own folder: pick The photo + extra "
                              "pictures)" if getattr(desc, "edit_references", 1) > 1 else ""))
            uncaptioned, shapes = pair_problems(folder, before, ext)
            if uncaptioned:
                errors.append(f"Edit LoRA: {len(uncaptioned)} edited photo(s) have no caption (e.g. "
                              f"{', '.join(uncaptioned[:3])}) - type the edit and press Write captions")
            if shapes:
                errors.append(f"Edit LoRA: {len(shapes)} edited photo(s) have a different crop or shape from their "
                              f"original (e.g. {', '.join(shapes[:3])}) - export both at the same crop")
        if ref and not os.path.isfile(ref):
            errors.append(f"Edit LoRA test photo for previews does not exist: {ref}")
        if edit_extras_on(desc, inputs):
            xdir = _s(inputs.get("FAMILY_EDIT_EXTRAS_DIR"))
            most = desc.edit_references - 1
            if not xdir or not os.path.isdir(xdir):
                errors.append("Edit LoRA with extra pictures: set the Extra pictures folder (Training tab, Edit pairs)")
            elif os.path.isdir(folder):
                found, orphans, misnamed = edit_extras(folder, xdir, test_photo=ref)
                over = [p for p, xs in found.items() if len(xs) > most]
                if not found:
                    errors.append(f"Edit LoRA: no extra pictures in {xdir} belong to an edited photo - name them "
                                  "after it: photo_1.png, photo_2.png ... for photo.png")
                if over:
                    errors.append(f"Edit LoRA: {len(over)} photo(s) have more than {most} extra pictures (e.g. "
                                  f"{', '.join(over[:3])}) - {desc.display_name} takes up to {most} per photo")
                if misnamed:
                    errors.append(f"Edit LoRA: {len(misnamed)} extra picture(s) are named exactly as a photo (e.g. "
                                  f"{', '.join(misnamed[:3])}) - add _1, _2 ... (photo_1.png for photo.png)")
                if orphans:
                    errors.append(f"Edit LoRA: {len(orphans)} extra picture(s) belong to no edited photo (e.g. "
                                  f"{', '.join(orphans[:3])}) - name each after its photo: photo_1.png for photo.png")
    if slider_on(desc, inputs, "pairs"):
        other = _s(inputs.get("FAMILY_SLIDER_DIR"))
        _ex = PAIR_EXTS | ({".mp4"} if "clip" in desc.media else set())    # H3: clip pairs too
        if not other or not os.path.isdir(other):
            errors.append("Slider: set the -1 end folder (Training tab, Slider)")
        elif os.path.isdir(folder):
            missing, multiple, _ = pairs(folder, other, _ex)
            if missing:
                errors.append(f"Slider: {len(missing)} +1 photo(s) have no -1 photo with the same file name in "
                              f"{other} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Slider: {len(multiple)} +1 photo(s) match more than one -1 photo (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one per photo, with the same file name")
            uncaptioned, shapes = pair_problems(folder, other, ext, _ex)
            if uncaptioned:
                errors.append(f"Slider: {len(uncaptioned)} photo(s) have no caption (e.g. "
                              f"{', '.join(uncaptioned[:3])}) - type what both ends share and press Write "
                              f"captions")
            if shapes:
                errors.append(f"Slider: {len(shapes)} photo(s) have a different crop or shape at the two ends "
                              f"(e.g. {', '.join(shapes[:3])}) - use the same framing for both")
    if slider_on(desc, inputs, "prompts"):
        for k, what in (("FAMILY_SLIDER_BASE", "what the picture is"), ("FAMILY_SLIDER_POS", "what the +1 end "
                        "adds"), ("FAMILY_SLIDER_NEG", "what the -1 end adds")):
            if not _s(inputs.get(k)):
                errors.append(f"Slider from prompts: fill in {what}")
        try:
            if float(inputs.get("FAMILY_SLIDER_GUIDANCE")) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Slider push strength must be a number above 0")
        if _s(inputs.get("FAMILY_SLIDER_ANCHORS")):
            try:
                if float(_s(inputs.get("FAMILY_SLIDER_ANCHOR_WEIGHT")) or "1") <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("Slider anchor strength must be a number above 0")
    need = [f for f in desc.model_files if f.required]
    if desc.training_adapter and inputs.get("FAMILY_TRAINING_ADAPTER", True):
        need += [f for f in desc.model_files if f.pref_key == desc.training_adapter]
    for f in need:
        path = (inputs.get("models") or {}).get(f.pref_key, "")
        if not path:
            errors.append(f"{f.label} path is empty (set it on the Preferences tab)")
        elif not os.path.exists(path):
            errors.append(f"{f.label} file does not exist: {path}")
    return errors


# ---------------------------------------------------------------------------------------------------- commands
def _media_count(folder):
    """Training items (photos, clips, songs) in a folder."""
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    exts = {e.lower() for e in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}
    try:
        return sum(1 for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in exts and not is_mask_file(n))
    except OSError:
        return 0


# Folders Fizgig itself makes inside a training folder: never trained on as sub-folders. Image Prep's Keep them safe
# (originals, then originals_2, _3... on later runs), Exclude by look and the desktop's Remove. Matched in any case.
OWN_SUBFOLDERS = {"excluded_by_look", "originals", "removed"}
_OWN_NUMBERED = re.compile(r"originals(_\d+)?", re.IGNORECASE)


def is_own_subfolder(name):
    """A folder Fizgig made inside a training folder (originals, originals_2, excluded_by_look, removed)."""
    n = os.path.basename(str(name).rstrip("\\/")).lower()
    return n in OWN_SUBFOLDERS or bool(_OWN_NUMBERED.fullmatch(n))


def subfolders_on(desc, st):
    """Also train on its sub-folders (the Training tab tick, off by default): a plain LoRA or a fine-tune - not with
    Subjects 2-4, DOP or Fizgig LoRA Isolation (one folder per subject), nor an edit or a slider (paired files)."""
    return bool(st.get("subfolders")) and not (st.get("multiconcept") or dop_on(desc, st) or multi_person_on(desc, st)
                                               or edit_on(desc, st) or slider_on(desc, st))


def subfolders(folder, skip=(), skipped=None):
    """[(relative path with "/", folder)] of every sub-folder at any depth holding training items (photos, clips,
    songs), sorted: Fizgig's own (originals, originals_2..., excluded_by_look, removed), hidden ones, and any folder in
    `skip` (the output, cache or regularisation folder when it sits inside), with everything under them, are left out.
    skipped: a list that gets the relative paths of the Fizgig folders left out (for the run's log)."""
    if not folder or not os.path.isdir(folder):
        return []
    norm = lambda f: os.path.normcase(os.path.abspath(f)).rstrip("\\/")
    skip = {norm(f) for f in skip if f}
    out = []
    for root, dirs, _files in os.walk(folder):
        if skipped is not None:
            skipped += sorted(os.path.relpath(os.path.join(root, d), folder).replace("\\", "/")
                              for d in dirs if not d.startswith(".") and is_own_subfolder(d))
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not is_own_subfolder(d)
                         and norm(os.path.join(root, d)) not in skip)
        if norm(root) != norm(folder) and _media_count(root):
            out.append((os.path.relpath(root, folder).replace("\\", "/"), root))
    return out


def run_subfolders(desc, st, skipped=None):
    """The run's sub-folders (subfolders), or [] when the tick is off or doesn't apply."""
    if not subfolders_on(desc, st):
        return []
    return subfolders(_s(st.get("image_folder")), (_s(st.get("LORA_OUTPUT_DIR")), _s(st.get("cache_root")),
                                                   ft_reg_dir(desc, st)), skipped)


def own_skipped_line(desc, st):
    """The run log's one line naming the Fizgig folders the sub-folder tick left out, or "" (none, or tick off)."""
    skipped = []
    run_subfolders(desc, st, skipped)
    if not skipped:
        return ""
    return "[sub-folders] Not trained (Fizgig's own folders): " + ", ".join(skipped[:6]) + (
        f" and {len(skipped) - 6} more" if len(skipped) > 6 else "")


def _balance_repeats(folders):
    """Multi Concept's balance: each folder repeated so its items come to about the largest folder's."""
    counts = {f: _media_count(f) for f in folders}
    top = max(counts.values() or [0])
    return {f: max(1, round(top / c)) for f, c in counts.items() if c}


def dataset_media(folder):
    """The media kinds in a training folder: {"photo", "clip", "voice"} (the dataset layer's extensions)."""
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    kinds = set()
    try:
        names = os.listdir(folder) if folder and os.path.isdir(folder) else []
    except OSError:
        names = []
    img, vid, aud = ({e.lower() for e in x} for x in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS))
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if is_mask_file(n):
            continue
        kinds |= {"photo"} if ext in img else {"clip"} if ext in vid else {"voice"} if ext in aud else set()
    return kinds


def option_applies(opt, media, values=None, options=(), finetune=None):
    """Whether a FamilyOption is live for a dataset with these media kinds: a clip-only option without clips, or a
    mixed-dataset one on a dataset that is not mixed, is neither shown nor sent (#136: a retired category restored
    from a mixed run must not reach a photo-only run)."""
    if opt.show_if_media and opt.show_if_media not in media:
        return False
    if finetune is not None and opt.mode and (opt.mode == "finetune") != bool(finetune):
        return False
    if opt.mixed_only and not ("voice" in media and media & {"photo", "clip"}):
        return False
    if opt.requires and values is not None:
        key, eq, want = opt.requires.partition("=")
        have = values.get(key, "")
        if eq:
            o = next((x for x in options if x.key == key), None)
            if (o.pick(have) if o is not None else have) != (o.pick(want) if o is not None else want):
                return False
        elif str(have) not in ("1", "True", "true"):
            return False
    return True


def option_tokens(desc, inputs, missing=None):
    """(trainer args, cache --aux args) from the family's FamilyOption values (inputs["FAMILY_OPTIONS"]: key ->
    value; an unset key takes the option's default). "pref:KEY" becomes that Preferences file's path; `missing`
    (a list) collects (pref key, option) for every such file that is not set or not on disk."""
    vals = inputs.get("FAMILY_OPTIONS") or {}
    if multi_person_on(desc, inputs):   # checks that can't run beside Fizgig LoRA Isolation (H3 reference distillation)
        vals = {**vals, **{o.key: "" for o in desc.options if getattr(o, "isolation_off", False)}}
    models = inputs.get("models") or {}

    def path(v):
        """'pref:KEY' -> the file; 'pref:KEY[OPT=LABEL|OPT2=1->ALT]' -> ALT's file when any condition holds."""
        if not v.startswith("pref:"):
            return v
        key, _, cond = v[5:].partition("[")
        if cond:
            tests, _, alt = cond.rstrip("]").partition("->")
            for t in tests.split("|"):
                ok, _, want = t.partition("=")
                o = next((x for x in desc.options if x.key == ok), None)
                have = vals.get(ok, o.default if o else "")
                if o is not None and o.kind == "choice":
                    have, want = o.pick(have), o.pick(want)
                if str(have) == want or (want == "1" and str(have) in ("True", "true")):
                    key = alt
                    break
        p = models.get(key, "")
        if missing is not None and (not p or not os.path.exists(p)):
            missing.append((key, cur[0]))
        return p

    train, aux = [], []
    cur = [None]                        # the option being resolved (for `missing`)
    media = dataset_media(_s(inputs.get("image_folder")))
    for opt in desc.options:
        if not option_applies(opt, media, vals, desc.options, finetune=ft_on(desc, inputs)):
            continue
        cur[0] = opt
        for tok in opt.resolve(vals.get(opt.key, opt.default)):
            if tok.startswith("aux:"):
                k, _, v = tok[4:].partition("=")
                aux += ["--aux", f"{k}={path(v)}"]
            elif tok.startswith("--"):
                k, eq, v = tok.partition("=")
                train += [k, path(v)] if eq else [k]
            else:
                k, _, v = tok.partition("=")
                train += ["--family_option", f"{k}={path(v)}"]
    return train, aux


def _dedupe_flags(cmd):
    """A flag an option sets (--dit, --training_adapter) replaces the one the generic builder put earlier: the last
    occurrence of a valued flag wins and the earlier pair goes."""
    valued = {"--dit", "--training_adapter", "--train_blocks", "--ema_decay", "--optimizer_type"}
    out, seen = [], set()
    i = len(cmd) - 1
    while i >= 0:
        tok = cmd[i]
        if i > 0 and cmd[i - 1] in valued:
            if cmd[i - 1] not in seen:
                out += [tok, cmd[i - 1]]
                seen.add(cmd[i - 1])
            i -= 2
            continue
        out.append(tok)
        i -= 1
    return out[::-1]


def cache_command(desc, inputs, stage):
    """families/cache.py: latents with the family's VAE, text with its text encoder."""
    model = _model(inputs, desc, "vae" if stage == "latents" else "text_encoder")
    cmd = [inputs["python"], os.path.join(inputs["repo_dir"], desc.cache_script), "--family", desc.key,
           "--stage", stage, "--dataset_config", inputs["DATASET_CONFIG"], "--model", model]
    cmd += option_tokens(desc, inputs)[1]
    if stage == "latents":
        cmd.append("--skip_existing")   # validated against the current bucket
    if slider_on(desc, inputs, "pairs"):
        cmd.append("--slider")          # the control folder is the -1 end: its latents, captions encoded plainly
    return cmd


def _state_flags(st):
    """Save-state flags. Keep-N is clamped to >= 1 here as well as in the trainer: a blank or zero box must never
    reach a prune that would take the state just written with it."""
    flags = []
    if st.get("SAVE_STATE", True):
        flags.append("--save_state")
    if st.get("SAVE_STATE_ON_TRAIN_END", True):
        flags.append("--save_state_on_train_end")
    if flags:
        try:
            keep_n = max(1, int(str(st.get("KEEP_LAST_N_STATES", 2)).strip()))
        except (TypeError, ValueError):
            keep_n = 2
        flags += ["--keep_last_n_states", str(keep_n)]
    return flags


def train_command(desc, inputs, plan):
    """families/train.py. Model paths from the description's roles, training settings from the run's settings,
    previews from the sample settings. Preview prompt files and console lines go into `plan`."""
    st = inputs
    cmd = [st["python"], os.path.join(st["repo_dir"], desc.train_script), "--family", desc.key,
           "--dit", _model(st, desc, "dit"),
           "--dataset_config", st["DATASET_CONFIG"],
           "--output_dir", st["LORA_OUTPUT_DIR"], "--output_name", st["LORA_NAME"],
           "--network_dim", str(st["NETWORK_DIM"]), "--network_alpha", str(st["NETWORK_ALPHA"]),
           "--learning_rate", str(st["LEARNING_RATE"]), "--max_train_epochs", str(st["MAX_TRAIN_EPOCHS"]),
           "--save_every_n_epochs", str(st["SAVE_EVERY_N_EPOCHS"]), "--seed", str(st["SEED"])]
    # How often to save, switched on (a fine-tune saves at whole rotations)
    later = [_s(st.get(k)) for k in ("SAVE_UNTIL_EPOCH", "SAVE_EVERY_AFTER")] \
        if st.get("SAVE_LATER") and not ft_on(desc, st) else []
    if later and all(x.isdigit() and int(x) > 0 for x in later):     # How often to save, switched on
        cmd += ["--save_until_epoch", later[0], "--save_every_n_epochs_after", later[1]]
    cmd += _state_flags(st)
    if _s(st.get("RESUME_TRAINING")) and not ft_on(desc, st):
        cmd += ["--resume", _s(st["RESUME_TRAINING"])]
    if desc.training_adapter and st.get("FAMILY_TRAINING_ADAPTER", True):
        cmd += ["--training_adapter", (st.get("models") or {}).get(desc.training_adapter, "")]
    ctx = _s(st.get("CONTEXT_LORA_PATH"))
    if ctx and not ft_on(desc, st):         # a fine-tune trains the base itself: there is no LoRA to stack on
        cmd += ["--context_lora_path", ctx,
                "--context_lora_strength", _s(st.get("CONTEXT_LORA_STRENGTH") or "1.0") or "1.0"]
    mp_subs = [x[1] for x in multi_person_subjects(desc, st) if x[1]] if multi_person_on(desc, st) else []
    if masked_on(desc, st):
        cmd += ["--masked_training", "--mask_floor", f"{mask_floor(st):g}"]
        if mask_prior(desc, st):
            cmd += ["--mask_prior", f"{mask_prior(desc, st):g}"]
    if dop_on(desc, st) or mp_subs:
        trig, cls = dop_rules(st)
        if mp_subs:          # the captions are read fused: DOP swaps "zwxemwoman" for "a woman"
            trig, cls = [s["fused"] for s in mp_subs], [s["phrase"] for s in mp_subs]
        # the tested recipe, always (Peter, 16 Oct 2026: every second step is proven, every step not needed): on half
        # the steps at twice the weight - the same preservation on average, about half the extra time (the weight
        # scales the DOP pass's loss, never the run's learning rate - its gradient joins the step's own)
        every, weight = 2, 1.0
        cmd += ["--dop", "--dop_weight", f"{weight * every:g}", "--dop_every", str(every)]
        if trig:
            cmd += ["--dop_trigger", ", ".join(trig), "--dop_class", ", ".join(cls)]
    if st.get("ADAPTIVE_LR") and desc.adaptive_lr and not ft_on(desc, st):   # a fine-tune trains at its own rate
        cmd += ["--adaptive_lr",
                "--adaptive_lr_min", str(st.get("ADAPTIVE_LR_MIN", "1e-4")).split(" ")[0],
                "--adaptive_lr_max", str(st.get("ADAPTIVE_LR_MAX", "2e-4")).split(" ")[0]]
    else:
        sched = _s(st.get("LR_SCHEDULER") or "constant") or "constant"
        if sched != "constant":
            cmd += ["--lr_scheduler", sched]
        try:
            if int(float(str(st.get("LR_WARMUP_STEPS", "") or 0))) > 0:
                cmd += ["--lr_warmup_steps", str(int(float(st["LR_WARMUP_STEPS"])))]
        except ValueError:
            pass
    if desc.compiles:
        cb = str(st.get("COMPILE_BLOCKS", "auto") or "auto").lower()
        if cb in ("auto", "on", "off", "outside"):
            cmd += ["--compile_blocks", cb]
    if st.get("keep_checkpointing"):
        cmd += ["--gradient_checkpointing", "on"]
    try:
        if int(float(str(st.get("GRADIENT_ACCUMULATION", "") or 1))) > 1:
            cmd += ["--gradient_accumulation_steps", str(int(float(st["GRADIENT_ACCUMULATION"])))]
    except ValueError:
        pass
    try:
        if abs(float(str(st.get("MAX_GRAD_NORM", "") or 1.0)) - 1.0) > 1e-9:
            cmd += ["--max_grad_norm", str(float(st["MAX_GRAD_NORM"]))]
    except ValueError:
        pass
    if len(desc.precisions) > 1:
        lab = str(st.get("FAMILY_PRECISION", "") or "")
        labels = {**PRECISION_LABELS, **desc.precision_labels}
        prec = next((k for k, v in labels.items() if v == lab), "auto")
        cmd += ["--precision", prec if prec == "auto" or prec in desc.precisions else "auto"]
    if ft_on(desc, st):
        # the length and cadence in whole rotations; a continuation starts from the paused run's checkpoint
        cmd += ["--finetune", "--ft_rotations", str(_ft_int(st, "FAMILY_FT_ROTATIONS", 10)),
                "--ft_save_every_rotations", str(_ft_int(st, "FAMILY_FT_SAVE_EVERY", 1)),
                "--ft_rotate_every", str(_ft_int(st, "FAMILY_FT_ROTATE_EVERY", 1))]
        if ft_max_parts(st):
            cmd += ["--ft_max_parts", str(ft_max_parts(st))]
        if st.get("FAMILY_FT_FUSED", True):
            cmd.append("--ft_fused_backward")
        if ft_reg_dir(desc, st):
            cmd += ["--reg_lr_multiplier", f"{_ft_reg_mult(st):g}"]
        cont = st.get("FAMILY_FT_CONTINUE") or {}
        if cont.get("checkpoint"):
            cmd += ["--ft_start_window", str(int(cont.get("start_window", 0))),
                    "--ft_epochs_done", str(int(cont.get("epochs_done", 0)))]
    # per-image loss watch (families/loss_watch.py), batch size 1 only
    try:
        bs1 = int(str(st.get("batch_size", 1)).strip() or 1) <= 1
    except ValueError:
        bs1 = True
    lw = (st.get("loss_watch") or {}) if desc.loss_watch else {}
    if multi_person_on(desc, st):    # the side-by-sides are hard on purpose: no verdicts, per-image LR or look warm-up
        lw = {**lw, "detect": False, "per_image_lr": False, "warmup": False}
    watch = [("detect", "--log_per_image_loss"), ("per_image_lr", "--per_image_lr"), ("warmup", "--warmup_look_outliers")]
    can_recaption = bool(st.get("captioner"))   # every family captions with the shared captioner
    # multi-person mode reads triggers fused as captions load; a new AI caption is encoded as written, unfused
    can_recaption = can_recaption and not multi_person_on(desc, st)
    if can_recaption:
        watch.append(("recaption", "--auto_recaption"))
    if bs1:
        cmd += [flag for key, flag in watch if lw.get(key)]
        if can_recaption and lw.get("recaption"):
            cmd += ["--captioner", st.get("captioner", "")]
            trig = _s(st.get("caption_trigger"))
            if trig and trig.lower() != "trigger_word":
                cmd += ["--trigger_word", trig]      # leads each AI caption, as the Captions tab writes it
            if st.get("recaption_triggers"):         # the other folders' own: {key prefix: trigger word}
                import json as _json
                cmd += ["--recaption_triggers", _json.dumps(st["recaption_triggers"])]
            from fizgig.krea2.caption_text import RECAPTION_DEFAULTS, recaption_instruction
            ovr = st.get("caption_overrides") or {}
            tasks = tuple(st.get("recaption_tasks") or RECAPTION_DEFAULTS)
            for key, (flag, task_flag) in zip(tasks, (("--recaption_instruction", "--recaption_task"),
                                                      ("--recaption_instruction_detailed", "--recaption_task_detailed"))):
                label, instr, edited = recaption_instruction(key, ovr, st.get("caption_concept") or "")
                cmd += [flag, instr, task_flag, f'"{label}"' + (" (your edit)" if edited else " (as shipped)")]
    elif any(lw.get(key) for key, _ in watch):
        plan.console.append("[loss-watch] per-image features skipped - they need Batch Size 1.\n")
    if any(lw.get(key) for key, _ in watch) and "--text_encoder" not in cmd:
        cmd += ["--text_encoder", _model(st, desc, "text_encoder")]   # caption repair re-encodes
    if ("lokr" in desc.network_types and str(st.get("NETWORK_TYPE", "")).startswith("LoKR")
            and not slider_on(desc, st)):        # a slider is always a plain LoRA
        cmd += ["--network_type", "lokr", "--lokr_factor", str(st.get("LOKR_FACTOR", 8))]
    raw_swap = str(st.get("blocks_swap") or "").strip()
    if raw_swap.lower().startswith("auto"):
        cmd += ["--blocks_to_swap", "-1"]            # the trainer sizes it for the precision that runs
    else:
        m = re.match(r"\d+", raw_swap)
        if m and int(m.group()) > 0:
            cmd += ["--blocks_to_swap", m.group()]
    if desc.ema_default:
        ema = str(st.get("FAMILY_EMA", "") or desc.ema_default).split(" ")[0]
        if ema != "Off":
            cmd += ["--ema_decay", ema]
    if str(st.get("OPTIMIZER_TYPE", "") or "").strip():
        cmd += ["--optimizer_type", str(st["OPTIMIZER_TYPE"]).strip()]
    if str(st.get("OPTIMIZER_ARGS", "") or "").strip():
        cmd += ["--optimizer_args", str(st["OPTIMIZER_ARGS"]).strip()]
    for key, flag in (("METADATA_TITLE", "--metadata_title"), ("METADATA_AUTHOR", "--metadata_author"),
                      ("METADATA_DESCRIPTION", "--metadata_description"),
                      ("METADATA_LICENSE", "--metadata_license"), ("METADATA_TAGS", "--metadata_tags"),
                      ("METADATA_THUMBNAIL", "--metadata_thumbnail")):
        val = str(st.get(key, "") or "").strip()
        if val:
            cmd += [flag, val]
    trig = _s(st.get("METADATA_TRIGGER_PHRASE")) or _s(st.get("caption_trigger"))
    if mp_subs:
        from fizgig.families import multi_person as _mp
        trig = ", ".join(s["fused"] for s in mp_subs)        # the LoRA answers to the fused words
        if any(s["kind"] == "person" for s in mp_subs):      # the side-by-side pictures' steps (people only)
            cmd += ["--lr_boost", f"{_mp.PREFIX}={_mp.LR_BOOST:g}"]
    if trig and trig.lower() != "trigger_word":
        cmd += ["--metadata_trigger_phrase", trig]
    cmd += _preview_flags(desc, st, plan, cmd)
    if (desc.identity_blocks and st.get("FAMILY_FAST_ID")
            and not (slider_on(desc, st) or edit_on(desc, st) or ft_on(desc, st))):
        cmd += ["--train_blocks", ",".join(desc.identity_blocks)]
    elif desc.train_areas and not (slider_on(desc, st) or ft_on(desc, st)):
        blocks = area_blocks(desc, st)
        if blocks:
            cmd += ["--train_blocks", ",".join(blocks)]
    if desc.train_areas and not ft_on(desc, st):
        cmd += timestep_flags(st)
    if slider_on(desc, st):
        if desc.slider_ultra_blocks and st.get("FAMILY_SLIDER_ULTRA"):
            cmd += ["--train_blocks", ",".join(desc.slider_ultra_blocks)]
        if slider_on(desc, st, "prompts"):
            base = _s(st.get("FAMILY_SLIDER_BASE"))
            cmd += ["--slider_prompts", base, f"{base} {_s(st.get('FAMILY_SLIDER_POS'))}",
                    f"{base} {_s(st.get('FAMILY_SLIDER_NEG'))}",       # the user's own comma, if any
                    "--slider_guidance", _s(st.get("FAMILY_SLIDER_GUIDANCE")) or f"{desc.slider_guidance:g}"]
            anchors = [a.strip() for a in str(st.get("FAMILY_SLIDER_ANCHORS") or "").splitlines() if a.strip()]
            if anchors:                  # prompts the dial must leave alone (#252)
                cmd += ["--slider_anchors", *anchors,
                        "--slider_anchor_weight", _s(st.get("FAMILY_SLIDER_ANCHOR_WEIGHT")) or "1"]
            try:        # the practice pictures (and so the training) at Target Megapixels, a square on the 16 px grid
                side = int((float(st.get("megapixels")) * 1_000_000) ** 0.5) // 16 * 16
                if side >= 256:
                    cmd += ["--slider_bank_res", str(side)]
            except (TypeError, ValueError):
                pass
            for flag, role in (("--text_encoder", "text_encoder"), ("--vae", "vae")):
                if flag not in cmd:          # the practice images and the three prompts need both
                    cmd += [flag, _model(st, desc, role)]
        else:
            cmd.append("--slider_pairs")
    if "--dop" in cmd and "--text_encoder" not in cmd:
        cmd += ["--text_encoder", _model(st, desc, "text_encoder")]   # the swapped captions are encoded at the start
    if desc.options:
        cmd = _dedupe_flags(cmd + option_tokens(desc, st)[0])
    cont = (st.get("FAMILY_FT_CONTINUE") or {}) if ft_on(desc, st) else {}
    if cont.get("checkpoint"):       # after the options: a model option's own --dit (H3's reference base) is what the
        cmd[cmd.index("--dit") + 1] = cont["checkpoint"]     # paused run started from; its checkpoint goes on
    return cmd


def _preview_flags(desc, st, plan, cmd):
    """The preview flags, and the prompt file they read (added to plan.files)."""
    sm = st.get("samples") or {}
    out = []
    if not sm.get("enabled"):
        return out
    sp = desc.preview_speed()
    speed_path = (st.get("models") or {}).get(sp.pref_key, "") if sp and sp.pref_key else ""
    turbo_steps = None
    if speed_path and os.path.exists(speed_path):
        dflt = desc.preview_speed_defaults()[1]
        try:
            if desc.samples_turbo_pace:        # H3's row: N steps at M% (the Steps box is the plain-model count)
                ts = float(str(st.get("FAMILY_TURBO_PACE", "") or round(100 * dflt))) / 100.0
                turbo_steps = str(st.get("FAMILY_TURBO_STEPS", "") or "").strip() or str(desc.preview_speed_defaults()[0])
            else:
                ts = float(str(st.get("FAMILY_TURBO_STRENGTH", "") or dflt))
        except ValueError:
            ts = dflt
        if ts > 0:
            out += ["--speed_lora", speed_path]
            if abs(ts - dflt) > 1e-9:
                out += ["--speed_lora_strength", f"{max(0.0, min(2.0, ts)):g}"]
        else:
            turbo_steps = None
            if not desc.samples_turbo_pace:     # strength 0: known to the run, loaded at the first preview that asks
                out += ["--speed_lora", speed_path, "--speed_lora_strength", "0"]
    samples_dir = st.get("samples_dir") or os.path.join(_s(st.get("LORA_OUTPUT_DIR")), "sample")
    prompts = None
    if edit_on(desc, st):              # edit previews: the edit instruction, not the Samples-tab prompts
        instr = edit_instruction(st)
        if instr:
            prompts = os.path.join(samples_dir, f"{desc.key}_edit_prompt.txt")
            plan.dirs.append(samples_dir)
            plan.files.append((prompts, instr + "\n"))
    elif slider_on(desc, st):          # slider previews: the dial on its own picture, not the Samples tab
        own = (st.get("FAMILY_SLIDER_BASE") if slider_on(desc, st, "prompts")
               else st.get("FAMILY_SLIDER_CAPTION"))
        prompts = os.path.join(samples_dir, f"{desc.key}_slider_prompt.txt")
        plan.dirs.append(samples_dir)
        plan.files.append((prompts, (str(own or "").strip() or "a photo") + "\n"))   # the trainer reads the captions
    else:
        lines = []
        for raw in sm.get("prompts") or []:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            ln = ln.split(" --")[0].strip()      # drop Klein-style inline flags
            if ln:
                lines.append(ln)
        if multi_person_on(desc, st):            # the previews read the prompts fused, as training reads the captions
            from fizgig.families import multi_person as mp
            rules = [r for _f, s, _w, _x in multi_person_subjects(desc, st) if s for r in s["rules"]]
            lines = [mp.fuse(ln, rules) for ln in lines]
        plan.dirs.append(samples_dir)            # the sample folder exists either way
        if lines or _sample_image(desc, sm):     # a reference alone previews too ('generate from this picture')
            prompts = os.path.join(samples_dir, f"{desc.key}_prompts.txt")
            plan.files.append((prompts, "\n".join(lines) + "\n"))
    every = str(sm.get("every") or "").strip()
    if every.isdigit() and int(every) > 0 and slider_on(desc, st):
        # a slider previews as set; only a cadence longer than the run (a family's long-run default - ACE-Step's 40 -
        # on a 20-epoch slider) would leave it with nothing after its epoch-0 preview: then once per saved checkpoint
        try:
            total = int(float(st.get("MAX_TRAIN_EPOCHS")))
            if int(every) > total:
                every = str(max(1, min(total, int(float(st.get("SAVE_EVERY_N_EPOCHS"))))))
        except (TypeError, ValueError):
            pass
    if prompts and every.isdigit() and int(every) > 0:
        later = [str(sm.get(k) or "").strip() for k in ("until", "after")] if sm.get("later") else []
        if later and all(x.isdigit() and int(x) > 0 for x in later):   # Then preview less often (off = nothing)
            out += ["--sample_until_epoch", later[0], "--sample_every_n_epochs_after", later[1]]
        out += ["--sample_prompts", prompts, "--sample_every_n_epochs", every,
                "--vae", _model(st, desc, "vae"),
                "--text_encoder", _model(st, desc, "text_encoder"),
                "--sample_width", str(sm.get("width") or "").strip() or str(desc.preview_width),
                "--sample_height", str(sm.get("height") or "").strip() or str(desc.preview_height)]
        if turbo_steps and turbo_steps.isdigit():
            out += ["--sample_steps", turbo_steps]
        elif str(sm.get("steps") or "").strip():
            out += ["--sample_steps", str(sm["steps"]).strip()]
        try:
            cfg = float(str(sm.get("cfg") or "").strip() or desc.preview_cfg)
        except ValueError:
            cfg = desc.preview_cfg
        out += ["--sample_cfg_scale", str(cfg)]
        neg = str(sm.get("negative") or "").strip()
        if neg and cfg > 1.0:
            out += ["--sample_negative", neg]
        try:
            out += ["--sample_seed", str(int(str(sm.get("seed")).strip()))]
        except ValueError:
            pass
        if sm.get("at_first"):
            out.append("--sample_at_first")
        strengths = [float(x) for x in (sm.get("strengths") or ()) if 0 < float(x) != 1.0]
        if strengths and not ft_on(desc, st) and not slider_on(desc, st) and not edit_on(desc, st):
            # each preview again at these LoRA strengths (a standard LoRA: a fine-tune has no strength to turn down)
            out += ["--sample_strengths", ",".join(f"{x:g}" for x in strengths)]
        if not edit_on(desc, st) and not slider_on(desc, st) and _sample_image(desc, sm):
            # the Samples tab's picture: seen through the vision path, or edited by every preview prompt
            out += ["--sample_image" if desc.reference_kind == "vision" else "--sample_reference",
                    _sample_image(desc, sm)]
        if edit_on(desc, st):
            ref = _s(st.get("FAMILY_EDIT_REF"))
            if not ref:
                try:
                    ref = pairs(_s(st.get("image_folder")), _s(st.get("FAMILY_EDIT_DIR")))[2] or ""
                except OSError:
                    ref = ""
            if ref:
                # a multi-picture edit previews with the photo's own extras (test.jpg -> test_1, test_2 ... in the
                # Extra pictures folder; the first original's when no test photo is set)
                out += ["--sample_reference", ref, *preview_extras(desc, st, ref)]
        ck = desc.preview_checkpoint()
        if (desc.train_preview_checkpoint and ck and sm.get("checkpoint") and not slider_on(desc, st)
                and not ft_on(desc, st)):      # a fine-tune previews the model it trains, not the Distilled
            path = _s((st.get("models") or {}).get(ck[0].pref_key))
            if path and os.path.exists(path):        # Klein's "Use Distilled model for samples"
                out += ["--preview_checkpoint", path,
                        "--preview_checkpoint_cache", _s(sm.get("checkpoint_cache")) or "auto"]
                if sm.get("int8"):
                    out.append("--preview_int8")
        if (desc.train_preview_checkpoint and ck and ck[1] and sm.get("checkpoint")
                and "--preview_checkpoint" not in out):
            # ticked, but this run previews on its own model (a fine-tune, a slider, no preview file): the boxes hold
            # the checkpoint's Steps and CFG (the tick's follow - Z-Image Turbo's 8 and 1), which render the training
            # model badly; its own come back for them, with its negative once CFG is above 1
            _swap_sample_value(out, "--sample_steps", ck[1].steps, desc.preview_steps)
            if _swap_sample_value(out, "--sample_cfg_scale", ck[1].cfg, desc.preview_cfg) and desc.preview_cfg > 1.0 \
                    and "--sample_negative" not in out:
                neg = _s(sm.get("negative")) or _s(desc.preview_negative)
                if neg:
                    out += ["--sample_negative", neg]
        look = desc.preview_look()
        if look is not None and st.get("FAMILY_PREVIEW_LOOK") and "--preview_checkpoint" not in out:
            # on the training model's previews only; a file still downloading is picked up at a later preview
            out += ["--preview_lora", look_path(desc, st), "--preview_lora_strength", f"{desc.preview_look_strength:g}"]
    return out


def _swap_sample_value(out, flag, was, now) -> bool:
    """The value after `flag` in a command list, replaced by `now` when it is `was` (as a number). True = replaced."""
    try:
        i = out.index(flag)
        if float(out[i + 1]) == float(was) and float(was) != float(now):
            out[i + 1] = f"{now:g}" if isinstance(now, float) else str(now)
            return True
    except (ValueError, IndexError):
        pass
    return False


def look_path(desc, st):
    """Where the preview look LoRA is: its Preferences path when set, else where its download puts it."""
    look = desc.preview_look()
    p = _s((st.get("models") or {}).get(look.pref_key))
    if p:
        return p
    from fizgig.user_data import MODELS_DIR
    return os.path.join(MODELS_DIR, look.local_name or os.path.basename(look.path))


def _sample_image(desc, sm):
    """The Samples tab's reference image, for families whose previews take one (reference_kind), when it exists."""
    ref = _s(sm.get("reference")) if desc.reference_kind else ""
    return ref if ref and os.path.isfile(ref) else ""


# ---------------------------------------------------------------------------------------------------- dataset
def cache_dir_for(cache_root, image_dir):
    """`<cache_root>/<folder name>-<hash of full path>`: one cache folder per image folder. The trainer builds its
    item list by globbing the cache folder, so two datasets sharing one would train on each other's leftovers; the
    hash keeps it stable per folder and unique across same-named folders (case and trailing slash normalised)."""
    import hashlib
    norm = image_dir.lower().replace("\\", "/").rstrip("/")
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    nm = "".join(c if (c.isalnum() or c in "-_") else "_"
                 for c in os.path.basename(image_dir.rstrip("/\\"))) or "dataset"
    return os.path.join(cache_root, f"{nm}-{h}")


def dataset_toml(desc, inputs):
    """The dataset TOML for the run: the training folder at Target Megapixels (a square of that area on the
    16-pixel grid), an edit's originals or a slider's -1 end as its control folder, and its own cache folder.
    Raises ValueError on a Target Megapixels that is not a number. Batch size is always 1 (one image a step)."""
    mp = float(inputs.get("megapixels"))
    if mp <= 0:
        raise ValueError(f"Target Megapixels {mp}")
    side = int((mp * 1_000_000) ** 0.5) // 16 * 16
    batch = 1           # every family trains one image at a time (families/train.py); Gradient Accumulation for more
    folder = _s(inputs.get("image_folder"))
    extra = [_s(f) for f in (inputs.get("extra_folders") or []) if _s(f)] if desc.multi_concept else []
    lines = ["[general]", f"resolution = [{side}, {side}]",
             f'caption_extension = "{_s(inputs.get("caption_ext", ".txt"))}"',
             f"batch_size = {batch}", "num_repeats = 1",
             f"enable_bucket = {'true' if inputs.get('enable_bucket', True) else 'false'}",
             f"bucket_no_upscale = {'true' if inputs.get('no_upscale', True) else 'false'}"]
    try:
        cmp_ = float(inputs.get("clip_megapixels") or 0)
    except (TypeError, ValueError):
        cmp_ = 0.0
    if cmp_ > 0 and "clip" in desc.media and any("clip" in dataset_media(f) for f in [folder] + extra):
        lines.append(f"clip_megapixels = {cmp_:g}")      # clips cache and train at their own size
    reps = _balance_repeats([folder] + extra) if extra and inputs.get("concept_balance") else {}
    mp = {f: (s, w) for f, s, _why, w in multi_person_subjects(desc, inputs) if s} \
        if multi_person_on(desc, inputs) else {}

    def _fuse(f):
        if f in mp:          # TOML arrays of strings read as JSON writes them
            import json as _json
            if mp[f][0]["rules"]:            # none when the captions are used as written
                lines.append(f"caption_replace = {_json.dumps(mp[f][0]['rules'])}")
            if isolation_photos_only(desc):  # a clip / voice family trains Isolation's subjects from their stills
                lines.append("photos_only = true")
    subs = run_subfolders(desc, inputs)
    head = len(lines)
    lines += ["", "[[datasets]]", f'image_directory = "{folder.replace(chr(92), "/")}"']
    if reps.get(folder, 1) > 1:
        lines.append(f"num_repeats = {reps[folder]}")
    _fuse(folder)
    if edit_on(desc, inputs):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_EDIT_DIR")).replace(chr(92), "/")}"')
        if edit_extras_on(desc, inputs):
            lines.append(f'extra_reference_directory = "{_s(inputs.get("FAMILY_EDIT_EXTRAS_DIR")).replace(chr(92), "/")}"')
    elif slider_on(desc, inputs, "pairs"):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_SLIDER_DIR")).replace(chr(92), "/")}"')
    root = _s(inputs.get("cache_root"))
    if root and folder:
        lines.append(f'cache_directory = "{cache_dir_for(root, folder).replace(chr(92), "/")}"')
    if subs and not _media_count(folder):     # everything is in the sub-folders: the training folder has no block
        del lines[head:]
    for rel, f in subs:                  # Also train on its sub-folders: each its own block and cache folder
        lines += ["", "[[datasets]]", f'image_directory = "{f.replace(chr(92), "/")}"',
                  f'item_prefix = "{_toml_str(rel + "/")}"']
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, f).replace(chr(92), "/")}"')
    for n, f in enumerate(extra, start=2):   # Multi Concept: each subject its own block and cache folder
        lines += ["", "[[datasets]]", f'image_directory = "{f.replace(chr(92), "/")}"', f'item_prefix = "s{n}:"']
        if reps.get(f, 1) > 1:
            lines.append(f"num_repeats = {reps[f]}")
        _fuse(f)
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, f).replace(chr(92), "/")}"')
    for f, (_s_, w) in mp.items():        # multi-person mode: each person's composites, already captioned fused
        if _s_["kind"] != "person":
            continue
        lines += ["", "[[datasets]]", f'image_directory = "{w.replace(chr(92), "/")}"']
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, w).replace(chr(92), "/")}"')
    reg = ft_reg_dir(desc, inputs)
    if reg:                              # a fine-tune's regularisation set: its own block (and cache folder)
        lines += ["", "[[datasets]]", f'image_directory = "{reg.replace(chr(92), "/")}"', 'item_prefix = "reg:"']
        if root:
            lines.append(f'cache_directory = "{cache_dir_for(root, reg).replace(chr(92), "/")}"')
        lines.append("is_reg = true")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- the plan
def multi_concept_problems(desc, inputs):
    """Multi Concept ticked (inputs["multiconcept"], the web form's): each extra subject folder becomes its own
    [[datasets]] block, so it has to be there, be its own folder and carry its own captions (the desktop's checks)."""
    if not (desc.multi_concept and inputs.get("multiconcept")):
        return []
    norm = lambda f: f.lower().replace("\\", "/").rstrip("/")
    seen, out = {norm(_s(inputs.get("image_folder")))}, []
    wanted = list(inputs.get("extra_folders") or [])
    for i, f in enumerate(wanted, start=2):
        if not _s(f):
            out.append(f"Subjects: set the subject {i} folder, or pick fewer subjects.")
    extra = [_s(f) for f in wanted if _s(f)]
    if not wanted:
        out.append("Subjects: no folder set for subject 2 - pick one, or choose 1 subject.")
    ext = _s(inputs.get("caption_ext")) or ".txt"
    for f in extra:
        if norm(f) in seen:
            out.append(f"Subjects: {f} is the same folder as another subject - each needs its own folder.")
            continue
        seen.add(norm(f))
        if not os.path.isdir(f):
            out.append(f"Subjects: folder does not exist: {f}")
        elif not any(fn.lower().endswith(ext.lower()) for fn in os.listdir(f)):
            out.append(f"Subjects: no {ext} captions in {f}. Caption every subject's folder before training - "
                       f"each needs its own trigger word in every caption, or they will blend.")
    return out


def automagic(inputs):
    return _s(inputs.get("OPTIMIZER_TYPE")).lower() == "automagic3"


def _dop_folders(st):
    """DOP's per-folder (name, words to remove, replacements): the training folder's, then subject 2's (Multi
    Concept). A folder's box left empty takes that folder's own trigger word (the Captions tab's, which the web
    server puts in DOP_FOLDER_TRIGGER / DOP_FOLDER_TRIGGER2)."""
    out = [("Subject 1", _s(st.get("FAMILY_DOP_TRIGGER")) or _s(st.get("DOP_FOLDER_TRIGGER"))
            or _s(st.get("CAPTION_TRIGGER_WORD")) or _s(st.get("caption_trigger")), _s(st.get("FAMILY_DOP_CLASS")))]
    if st.get("multiconcept"):
        for i, _f in enumerate(st.get("extra_folders") or [], start=2):
            out.append((f"Subject {i}", _s(st.get(f"FAMILY_DOP_TRIGGER{i}")) or _s(st.get(f"DOP_FOLDER_TRIGGER{i}")),
                        _s(st.get(f"FAMILY_DOP_CLASS{i}"))))
    rows = []
    for name, trig, cls in out:
        t = [x for x in (y.strip() for y in trig.split(",")) if x]
        c = [x.strip() for x in cls.split(",")]
        rows.append((name, t, c * len(t) if len(c) == 1 else c))     # one replacement (or none) for them all
    return rows


def dop_rules(st):
    """DOP's (words to remove, replacements) over every folder, in order."""
    trig, cls = [], []
    for _name, t, c in _dop_folders(st):
        if len(c) == len(t):
            trig += t
            cls += c
    return trig, cls


def dop_on(desc, st):
    """DOP (the Subjects card's "Leave the rest of the model alone", and under Fizgig LoRA Isolation): a family that
    offers it (description subject_isolation), never an edit or a slider."""
    return bool(st.get("FAMILY_DOP")) and desc.subject_isolation and not (slider_on(desc, st) or edit_on(desc, st))


MASK_FLOORS = ("0 - not at all", "0.05", "0.1 (recommended)", "0.2", "0.3", "0.5")
# masked prior preservation (OneTrainer's weight): outside the mask, the model's own picture at this weight
MASK_PRIORS = ("Off", "0.5", "1", "2")


def masked_on(desc, st):
    """Masked training (fizgig/dataset/masks.py): the tick on, a family whose driver takes masks, a run that trains on
    pictures (a prompt slider makes its own)."""
    return bool(st.get("MASKED_TRAINING")) and desc.masked_training and not slider_on(desc, st, "prompts")


def mask_floor(st):
    """How much outside a mask counts, from the Training tab's choice (0.1 unless it reads as a number 0-1)."""
    try:
        v = float(str(st.get("MASK_FLOOR") or "0.1").split()[0])
    except ValueError:
        return 0.1
    return min(1.0, max(0.0, v))


def mask_prior(desc, st):
    """Outside the mask, keep the model's own picture: the Training tab's weight, 0 when off - and for a slider or
    a fine-tune (it switches the LoRA off for a step: there's none to switch)."""
    if slider_on(desc, st) or ft_on(desc, st):
        return 0.0
    try:
        return max(0.0, float(str(st.get("MASK_PRIOR") or "Off").split()[0]))
    except ValueError:
        return 0.0


def mask_folders(desc, st):
    """The folders whose pictures train (and so whose masks count): the training folder, its sub-folders when
    they train, and Multi Concept's others."""
    extra = [_s(f) for f in (st.get("extra_folders") or []) if _s(f)] if desc.multi_concept else []
    return [f for f in [_s(st.get("image_folder"))] + [f for _r, f in run_subfolders(desc, st)] + extra if f]


def mask_report(desc, st):
    from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
    from fizgig.dataset.masks import folder_report
    return folder_report(mask_folders(desc, st), IMAGE_EXTENSIONS)


def multi_person_on(desc, st):
    """Fizgig LoRA Isolation (multi-person mode, families/multi_person.py): a standard LoRA run with the tick on, on a
    family that offers it."""
    return bool(st.get("FAMILY_MULTI_PERSON")) and desc.subject_isolation and         not (slider_on(desc, st) or edit_on(desc, st) or ft_on(desc, st))


def isolation_photos_only(desc):
    """Fizgig LoRA Isolation trains people from their stills: a family that also trains clips or voice (MiniMax H3)
    leaves those out of the subjects' folders - the side-by-sides are stills and DOP's preservation pass is a picture's."""
    return "photo" in desc.media and bool({"clip", "voice"} & set(desc.media))


def multi_person_subjects(desc, st):
    """[(folder, subject or None, problem, composites folder)] for each subject folder, in order: its one trigger
    word fused with its class ("zwxem" + "a woman" -> "zwxemwoman"), and where its composites are built."""
    from fizgig.families import multi_person as mp
    folders = [_s(st.get("image_folder"))] + ([_s(f) for f in st.get("extra_folders") or []]
                                              if st.get("multiconcept") else [])
    ext = _s(st.get("caption_ext")) or ".txt"
    root = _s(st.get("cache_root"))
    out = []
    for n, (folder, (name, t, c)) in enumerate(zip(folders, _dop_folders(st)), start=1):
        where = (cache_dir_for(root, folder) if root else folder.rstrip("/\\")) + "-multiperson"
        kind = mp.kind_key(st.get("FAMILY_SUBJECT_KIND" + ("" if n == 1 else str(n))))
        if len(t) != 1:
            out.append((folder, None, f"Fizgig LoRA Isolation: {name.lower()} needs exactly one trigger "
                                      f"word ({len(t)} given) - one person per folder", where))
            continue
        if st.get("FAMILY_MP_MANUAL"):   # the captions as written: the trigger is already the joined word
            sub, why = mp.subject_as_written(t[0], c[0] if c else "", kind)
        else:
            sub, why = mp.subject(folder, t[0], c[0] if c else "", ext, kind)
        out.append((folder, sub, f"Fizgig LoRA Isolation: {name.lower()} - {why}" if why else "", where))
    return out


def multi_person_problems(desc, v):
    if not multi_person_on(desc, v):
        return []
    out = [why for _f, _s_, why, _w in multi_person_subjects(desc, v) if why]
    if isolation_photos_only(desc):      # it trains people from their stills: a folder of only clips has none
        folders = [_s(v.get("image_folder"))] + ([_s(f) for f in v.get("extra_folders") or []]
                                                 if v.get("multiconcept") else [])
        for (name, _t, _c), f in zip(_dop_folders(v), folders):
            if f and "photo" not in dataset_media(f):
                out.append(f"Fizgig LoRA Isolation: {name.lower()} has no photos. It trains people from their photos "
                           f"(clips and voice are left out), so add some photos of them to its folder, or pick "
                           f"another option under Your subject and other people.")
    return out


def dop_problems(desc, v):
    if not (dop_on(desc, v) or multi_person_on(desc, v)):
        return []
    out = []
    for name, t, c in _dop_folders(v):
        if not t:
            out.append(f"Subjects: no trigger word found for {name.lower()} - tick "
                       "Correct a trigger word and type it, or caption the folder with one.")
        elif len(c) != len(t):
            out.append(f"Subjects: {name.lower()} has {len(t)} trigger words "
                       f"({', '.join(t)}) but {len(c)} 'what it is' words - give one for each, in the same order, one "
                       "for them all, or none.")
    return out


def start_problems(desc, inputs):
    """Everything Start refuses, in the order the desktop app lists it: the shared checks (fizgig.families.checks)
    with the family's own (problems) in their place. The dataset TOML is written by the plan itself, so only its
    path is required here."""
    from fizgig.families import checks
    v = inputs
    out = checks.learning_rate(v)
    out += checks.network(v, lokr="lokr" in desc.network_types and str(v.get("NETWORK_TYPE", "")).startswith("LoKR"))
    out += checks.numbers(v)
    out += checks.dataset_config(v.get("DATASET_CONFIG"), must_exist=False)
    out += problems(desc, inputs)
    out += checks.learning_rate_range(v)
    out += multi_concept_problems(desc, v)
    out += dop_problems(desc, v)
    out += multi_person_problems(desc, v)
    if automagic(v) and not v.get("ADAPTIVE_LR") and not ft_on(desc, v):
        try:
            lr = float(v.get("LEARNING_RATE"))
        except (TypeError, ValueError):
            lr = 0.0
        if lr > 1e-3:
            out.append(f"Automagic v3 starts at the Learning Rate you give and finds its own rate: {lr:g} is an AdamW "
                       f"number. Use 1e-6 (its default) or at most 0.001.")
    out += checks.context_lora(v)
    raw_swap = str(v.get("blocks_swap") or "").strip()
    m = re.match(r"\d+", raw_swap)
    swap = 0 if raw_swap.lower().startswith("auto") or not m else int(m.group())   # Auto always fits
    out += checks.run(v, blocks_swap=swap, swap_max=max(0, desc.n_blocks - 2), arch_label=desc.gui_label,
                      name_error=checks.tidy_name(v.get("LORA_NAME"))[1])
    # a Preferences file an option asks for (H3's ref2va base, an Ostris adapter): set and on disk
    miss = []
    option_tokens(desc, inputs, missing=miss)
    for key, opt in dict.fromkeys(miss):
        row = next((f.label for f in desc.model_files if f.pref_key == key), key)
        p = (inputs.get("models") or {}).get(key, "")
        why = f"'{opt.label}' on the Training tab needs it" if opt is not None and opt.label else "a setting needs it"
        out.append(f"{row} is {'empty' if not p else 'not where Preferences says (' + p + ')'} — {why}. Set it on "
                   f"the Preferences tab (its Download link, or the download button there).")
    # a typed block range (an option with counts_blocks) that does not parse, caught before any model loads
    vals = inputs.get("FAMILY_OPTIONS") or {}
    media = dataset_media(_s(v.get("image_folder")))
    for opt in desc.options:
        if not opt.counts_blocks or not option_applies(opt, media, vals, desc.options, finetune=ft_on(desc, inputs)):
            continue
        spec = str(vals.get(opt.key, opt.default) or "all").split("·")[0].strip() or "all"
        if spec.lower() != "all":
            from fizgig.utils.block_spec import parse_block_spec
            try:
                parse_block_spec(spec, opt.counts_blocks)
            except ValueError as e:
                out.append(f"{opt.label}: {e}")
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    if "clip" in desc.media:
        exts |= {e.lower() for e in VIDEO_EXTENSIONS}       # the clip formats the dataset layer reads
    if "voice" in desc.media:
        exts |= {e.lower() for e in AUDIO_EXTENSIONS}
    subs = run_subfolders(desc, v)
    if not (subs and not _media_count(_s(v.get("image_folder")))):   # all in the sub-folders: nothing to check here
        out += checks.training_folder(_s(v.get("image_folder")), _s(v.get("caption_ext", ".txt")),
                                      check_captions=not slider_on(desc, inputs, "prompts"), media_exts=exts)
        out += clip_problems(desc, _s(v.get("image_folder")))
    for rel, f in subs:                  # Also train on its sub-folders: each one's captions and clips too
        out += checks.training_folder(f, _s(v.get("caption_ext", ".txt")), media_exts=exts,
                                      label=f"the sub-folder {rel}")
        out += clip_problems(desc, f)
    if masked_on(desc, v) and not mask_report(desc, v)["with"]:
        out.append("Masked training is on, but no picture in the training folder has a mask it can use. Save each "
                   "mask beside its picture with -masklabel added to the name (photo.jpg -> photo-masklabel.png: "
                   "white = train, black = mostly ignore), or untick Masked training.")
    return out


_CLIP_CHECKED = {}          # (path, mtime, size) -> problem: a Start re-check does not probe an unchanged clip again


def clip_problems(desc, folder):
    """Every clip in the training folder that is off the family's ClipSpec, with the reason - before anything
    caches, so a wrong frame rate or frame count is refused at Start, not found an hour into caching. Plus the
    driver's own media checks (media_problem) for anything the spec cannot say."""
    if not ({"clip", "voice"} & set(desc.media)) or not folder or not os.path.isdir(folder):
        return []
    from fizgig.families import clips
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
    vid = {e.lower() for e in VIDEO_EXTENSIONS} if "clip" in desc.media else set()
    aud = {e.lower() for e in AUDIO_EXTENSIONS} if "voice" in desc.media else set()
    driver, out = None, []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in vid | aud or not os.path.isfile(path):
            continue
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
        if key not in _CLIP_CHECKED:
            why = clips.problem(path, desc.clip_spec, desc.display_name) if ext in vid and desc.clip_spec else ""
            if not why and desc.driver:
                if driver is None:
                    driver = desc.load_driver()
                why = driver.media_problem(path)
            _CLIP_CHECKED[key] = why
        if _CLIP_CHECKED[key]:
            out.append(_CLIP_CHECKED[key])
    return out


def plan(desc, inputs):
    """The whole launch: problems first (a plan with problems must not start), then the stages in order."""
    from fizgig.families.checks import tidy_name
    inputs = dict(inputs)
    folder = _s(inputs.get("image_folder"))
    if folder and not os.path.isabs(folder):
        inputs["image_folder"] = os.path.abspath(folder)     # the subprocesses run from the Fizgig folder
    inputs["LORA_NAME"] = tidy_name(_s(inputs.get("LORA_NAME")))[0]
    if automagic(inputs):            # it owns the rate: Adaptive LR, per-image LR and the look warm-up are off
        inputs["ADAPTIVE_LR"] = False
        inputs["loss_watch"] = {**(inputs.get("loss_watch") or {}), "per_image_lr": False, "warmup": False}
    p = LaunchPlan(problems=start_problems(desc, inputs))
    if not p.problems:
        try:
            p.dataset_toml = dataset_toml(desc, inputs)
        except (TypeError, ValueError):
            p.problems.append("Target Megapixels (Dataset) must be a number above 0")
    missing = [k for k in ("python", "repo_dir", "DATASET_CONFIG", "LORA_OUTPUT_DIR", "NETWORK_DIM", "NETWORK_ALPHA",
                           "LEARNING_RATE", "MAX_TRAIN_EPOCHS", "SAVE_EVERY_N_EPOCHS", "SEED") if not _s(inputs.get(k))]
    if missing and not p.problems:
        p.problems.append(f"Missing settings: {', '.join(missing)}")
    if p.problems:
        return p                        # nothing to write or run
    p.files.append((inputs["DATASET_CONFIG"], p.dataset_toml))
    own = own_skipped_line(desc, inputs)
    if own:
        p.console.append(own)
    train = train_command(desc, inputs, p)
    people = [(f, s, w) for f, s, _why, w in multi_person_subjects(desc, inputs) if s and s["kind"] == "person"] \
        if multi_person_on(desc, inputs) else []
    if people:                          # the side-by-side pictures: people only (an animal or a style has none)
        cmd = [inputs["python"], os.path.join(inputs["repo_dir"], "src", "fizgig", "scripts", "multi_person.py"),
               "--ext", _s(inputs.get("caption_ext")) or ".txt"]
        if not inputs.get("FAMILY_MP_FACECROPS_FIRST", True):
            cmd.append("--no_face_first")
        for f, s, w in people:
            cmd += ["--subject", f, s["fused"], w]
        p.stages.append(Stage("Multi-person composites", cmd))
    resuming = inputs.get("resuming") or bool(_s(inputs.get("RESUME_TRAINING")))
    if resuming and not inputs.get("recache") and inputs.get("enable_cache", True):
        n = uncached_pictures(p.dataset_toml)
        if n:                           # a resume expects its cache; deleted by hand, it's made again (only what's gone)
            inputs["recache"] = True
            p.console.append(f"[cache] {n} picture(s) have no cache - caching them before the run resumes.\n")
    if caches(desc, inputs):
        p.stages += [Stage("Cache Preparation", cache_command(desc, inputs, "latents")),
                     Stage("Text Encoder Caching", cache_command(desc, inputs, "text"))]
    p.stages.append(Stage("Training", train))
    return p
