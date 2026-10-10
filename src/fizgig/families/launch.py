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
    enable_cache, resuming      whether the cache stages run; a resume (or fine-tune continuation) skips them
    loss_watch                  {"detect", "per_image_lr", "warmup", "recaption": bool}
    captioner, caption_trigger, caption_overrides   auto-recaption: model path, trigger word, edited instructions
    samples                     {"enabled", "every", "width", "height", "steps", "cfg", "negative", "seed",
                                 "at_first": ..., "prompts": [lines as typed on the Samples tab]}
    samples_dir                 where preview prompt files go (the output folder's sample/)
    edit_caption                the edit instruction typed in the Edit card

Switches (FAMILY_EDIT, FAMILY_SLIDER, ADAPTIVE_LR, SAVE_STATE, enable_cache, samples["enabled"], loss_watch[...],
...) must be real booleans: the string "false" counts as on. Numbers may be strings or numbers. The run's
settings the commands need are LORA_NAME, LORA_OUTPUT_DIR, DATASET_CONFIG (the path the plan's TOML is written
to), NETWORK_DIM, NETWORK_ALPHA, LEARNING_RATE, MAX_TRAIN_EPOCHS, SAVE_EVERY_N_EPOCHS, SEED; the number checks
also read LORA_LR_RATIO, GRADIENT_ACCUMULATION, MAX_GRAD_NORM, NETWORK_DROPOUT (and ADAPTIVE_LR_MIN / _MAX with
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

PRECISION_LABELS = {"auto": "Auto (fits your free VRAM)", "bf16": "bf16 (full precision)",
                    "int8": "INT8 (8-bit, fastest)", "nf4": "4-bit NF4 (smallest)"}
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


def _s(x):
    """A value from the inputs as trimmed text ("" for None) - they may arrive as numbers."""
    return "" if x is None else str(x).strip()


def _model(inputs, desc, role):
    return (inputs.get("models") or {}).get(desc.pref_for(role), "")


def edit_on(desc, inputs):
    return bool(desc.edit_training and inputs.get("FAMILY_EDIT"))


def slider_on(desc, inputs, source=None):
    """Slider mode for a family that offers it; source "pairs" / "prompts" narrows it to one kind."""
    on = bool(desc.slider_training and inputs.get("FAMILY_SLIDER"))
    if on and source:
        on = (inputs.get("FAMILY_SLIDER_SOURCE") or "pairs") == source
    return on


def caches(desc, inputs):
    """Whether the cache stages run: not on a resume (the cache is already built), not for a prompt slider (no
    photos), not with Enable Cache off."""
    resuming = inputs.get("resuming") or bool(_s(inputs.get("RESUME_TRAINING")))
    return bool(inputs.get("enable_cache", True) and not resuming and not slider_on(desc, inputs, "prompts"))


# ---------------------------------------------------------------------------------------------------- pairs
def pairs(after_dir, before_dir):
    """(photos without a partner, photos with more than one, first partner's path) for an edit's or a slider's
    two folders, matched the way the dataset loader matches them: a partner of photo.png is photo.<ext> or
    photo_<anything>.<ext>, ignoring case."""
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in PAIR_EXTS)
    missing, multiple, first = [], [], None
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in PAIR_EXTS:
            continue
        m = [x for x in befores if x.casefold().startswith(b.casefold() + ".")    # IMG_1 pairs with img_1
             or x.casefold().startswith(b.casefold() + "_")]
        if not m:
            missing.append(f)
        elif len(m) > 1:
            multiple.append(f)
        first = first or (os.path.join(before_dir, m[0]) if m else None)
    return missing, multiple, first


def pair_problems(after_dir, before_dir, caption_ext=".txt"):
    """(photos in after_dir without a caption, pairs whose shapes differ)."""
    from PIL import Image
    befores = sorted(f for f in os.listdir(before_dir) if os.path.splitext(f)[1].lower() in PAIR_EXTS)
    uncaptioned, shapes = [], []
    for f in sorted(os.listdir(after_dir)):
        b, e = os.path.splitext(f)
        if e.lower() not in PAIR_EXTS:
            continue
        if not os.path.exists(os.path.join(after_dir, b + caption_ext)):
            uncaptioned.append(f)
        m = [x for x in befores if x.casefold().startswith(b.casefold() + ".")
             or x.casefold().startswith(b.casefold() + "_")]
        if len(m) == 1:
            try:
                with Image.open(os.path.join(after_dir, f)) as ia, Image.open(os.path.join(before_dir, m[0])) as ib:
                    ra, rb = ia.width / ia.height, ib.width / ib.height
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


# ---------------------------------------------------------------------------------------------------- problems
def problems(desc, inputs):
    """What the family's own checks refuse: an edit's or a slider's pairs, a prompt slider's prompts, and every
    required model file (plus the training adapter while it is on) set and on disk."""
    errors = []
    folder = _s(inputs.get("image_folder"))
    ext = inputs.get("caption_ext") or ".txt"
    if edit_on(desc, inputs) and slider_on(desc, inputs):
        errors.append("Edit LoRA and Slider are both on - a run is one kind of LoRA: pick Edit or Slider")
    if edit_on(desc, inputs):
        before = _s(inputs.get("FAMILY_EDIT_DIR"))
        ref = _s(inputs.get("FAMILY_EDIT_REF"))
        if not before or not os.path.isdir(before):
            errors.append("Edit LoRA is on: set the Originals folder (Training tab, Training Parameters)")
        elif os.path.isdir(folder):
            missing, multiple, _ = pairs(folder, before)
            if missing:
                errors.append(f"Edit LoRA: {len(missing)} edited photo(s) have no original with the same file "
                              f"name in {before} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Edit LoRA: {len(multiple)} edited photo(s) match more than one original (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one original per edited photo, with the "
                              f"same file name")
            uncaptioned, shapes = pair_problems(folder, before, ext)
            if uncaptioned:
                errors.append(f"Edit LoRA: {len(uncaptioned)} edited photo(s) have no caption (e.g. "
                              f"{', '.join(uncaptioned[:3])}) - type the edit and press Write captions")
            if shapes:
                errors.append(f"Edit LoRA: {len(shapes)} edited photo(s) have a different crop or shape from their "
                              f"original (e.g. {', '.join(shapes[:3])}) - export both at the same crop")
        if ref and not os.path.isfile(ref):
            errors.append(f"Edit LoRA test photo for previews does not exist: {ref}")
    if slider_on(desc, inputs, "pairs"):
        other = _s(inputs.get("FAMILY_SLIDER_DIR"))
        if not other or not os.path.isdir(other):
            errors.append("Slider: set the -1 end folder (Training tab, Training Parameters)")
        elif os.path.isdir(folder):
            missing, multiple, _ = pairs(folder, other)
            if missing:
                errors.append(f"Slider: {len(missing)} +1 photo(s) have no -1 photo with the same file name in "
                              f"{other} (e.g. {', '.join(missing[:3])})")
            if multiple:
                errors.append(f"Slider: {len(multiple)} +1 photo(s) match more than one -1 photo (e.g. "
                              f"{', '.join(multiple[:3])}) - keep one per photo, with the same file name")
            uncaptioned, shapes = pair_problems(folder, other, ext)
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
def cache_command(desc, inputs, stage):
    """families/cache.py: latents with the family's VAE, text with its text encoder."""
    model = _model(inputs, desc, "vae" if stage == "latents" else "text_encoder")
    cmd = [inputs["python"], os.path.join(inputs["repo_dir"], desc.cache_script), "--family", desc.key,
           "--stage", stage, "--dataset_config", inputs["DATASET_CONFIG"], "--model", model]
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
    cmd += _state_flags(st)
    if _s(st.get("RESUME_TRAINING")):
        cmd += ["--resume", _s(st["RESUME_TRAINING"])]
    if desc.training_adapter and st.get("FAMILY_TRAINING_ADAPTER", True):
        cmd += ["--training_adapter", (st.get("models") or {}).get(desc.training_adapter, "")]
    ctx = _s(st.get("CONTEXT_LORA_PATH"))
    if ctx:
        cmd += ["--context_lora_path", ctx,
                "--context_lora_strength", _s(st.get("CONTEXT_LORA_STRENGTH") or "1.0") or "1.0"]
    if st.get("ADAPTIVE_LR"):
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
    try:
        if abs(float(str(st.get("MAX_GRAD_NORM", "") or 1.0)) - 1.0) > 1e-9:
            cmd += ["--max_grad_norm", str(float(st["MAX_GRAD_NORM"]))]
    except ValueError:
        pass
    if len(desc.precisions) > 1:
        lab = str(st.get("FAMILY_PRECISION", "") or "")
        prec = next((k for k, v in PRECISION_LABELS.items() if v == lab), "auto")
        cmd += ["--precision", prec if prec == "auto" or prec in desc.precisions else "auto"]
    # per-image loss watch (families/loss_watch.py), batch size 1 only
    try:
        bs1 = int(str(st.get("batch_size", 1)).strip() or 1) <= 1
    except ValueError:
        bs1 = True
    lw = st.get("loss_watch") or {}
    watch = [("detect", "--log_per_image_loss"), ("per_image_lr", "--per_image_lr"), ("warmup", "--warmup_look_outliers")]
    can_recaption = bool(st.get("captioner"))   # every family captions with the shared captioner
    if can_recaption:
        watch.append(("recaption", "--auto_recaption"))
    if bs1:
        cmd += [flag for key, flag in watch if lw.get(key)]
        if can_recaption and lw.get("recaption"):
            cmd += ["--captioner", st.get("captioner", "")]
            trig = _s(st.get("caption_trigger"))
            if trig and trig.lower() != "trigger_word":
                cmd += ["--trigger_word", trig]      # leads each AI caption, as the Captions tab writes it
            ovr = st.get("caption_overrides") or {}
            for key, flag in (("training", "--recaption_instruction"), ("exhaustive", "--recaption_instruction_detailed")):
                instr = str(ovr.get(key, "") or "").strip()
                if instr:
                    cmd += [flag, instr]
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
    if trig and trig.lower() != "trigger_word":
        cmd += ["--metadata_trigger_phrase", trig]
    cmd += _preview_flags(desc, st, plan, cmd)
    if slider_on(desc, st):
        if slider_on(desc, st, "prompts"):
            base = _s(st.get("FAMILY_SLIDER_BASE"))
            cmd += ["--slider_prompts", base, f"{base} {_s(st.get('FAMILY_SLIDER_POS'))}",
                    f"{base} {_s(st.get('FAMILY_SLIDER_NEG'))}",       # the user's own comma, if any
                    "--slider_guidance", str(st.get("FAMILY_SLIDER_GUIDANCE") or "2")]
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
    return cmd


def _preview_flags(desc, st, plan, cmd):
    """The preview flags, and the prompt file they read (added to plan.files)."""
    sm = st.get("samples") or {}
    out = []
    if not sm.get("enabled"):
        return out
    sp = desc.preview_speed()
    speed_path = (st.get("models") or {}).get(sp.pref_key, "") if sp and sp.pref_key else ""
    if speed_path and os.path.exists(speed_path):
        dflt = desc.preview_speed_defaults()[1]
        try:
            ts = float(str(st.get("FAMILY_TURBO_STRENGTH", "") or dflt))
        except ValueError:
            ts = dflt
        if ts > 0:                              # strength 0 = previews without the turbo: don't load it
            out += ["--speed_lora", speed_path]
            if abs(ts - dflt) > 1e-9:
                out += ["--speed_lora_strength", f"{max(0.0, min(2.0, ts)):g}"]
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
        plan.dirs.append(samples_dir)            # the sample folder exists either way
        if lines:
            prompts = os.path.join(samples_dir, f"{desc.key}_prompts.txt")
            plan.files.append((prompts, "\n".join(lines) + "\n"))
    every = str(sm.get("every") or "").strip()
    if prompts and every.isdigit() and int(every) > 0:
        out += ["--sample_prompts", prompts, "--sample_every_n_epochs", every,
                "--vae", _model(st, desc, "vae"),
                "--text_encoder", _model(st, desc, "text_encoder"),
                "--sample_width", str(sm.get("width") or "").strip() or str(desc.preview_width),
                "--sample_height", str(sm.get("height") or "").strip() or str(desc.preview_height)]
        if str(sm.get("steps") or "").strip():
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
        if edit_on(desc, st):
            ref = _s(st.get("FAMILY_EDIT_REF"))
            if not ref:
                try:
                    ref = pairs(_s(st.get("image_folder")), _s(st.get("FAMILY_EDIT_DIR")))[2] or ""
                except OSError:
                    ref = ""
            if ref:
                out += ["--sample_reference", ref]
    return out


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
    Raises ValueError on a Target Megapixels or Batch Size that is not a number."""
    mp = float(inputs.get("megapixels"))
    if mp <= 0:
        raise ValueError(f"Target Megapixels {mp}")
    side = int((mp * 1_000_000) ** 0.5) // 16 * 16
    batch = int(inputs.get("batch_size"))
    folder = _s(inputs.get("image_folder"))
    lines = ["[general]", f"resolution = [{side}, {side}]",
             f'caption_extension = "{_s(inputs.get("caption_ext", ".txt"))}"',
             f"batch_size = {batch}", "num_repeats = 1",
             f"enable_bucket = {'true' if inputs.get('enable_bucket', True) else 'false'}",
             f"bucket_no_upscale = {'true' if inputs.get('no_upscale', True) else 'false'}",
             "", "[[datasets]]", f'image_directory = "{folder.replace(chr(92), "/")}"']
    if edit_on(desc, inputs):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_EDIT_DIR")).replace(chr(92), "/")}"')
    elif slider_on(desc, inputs, "pairs"):
        lines.append(f'control_directory = "{_s(inputs.get("FAMILY_SLIDER_DIR")).replace(chr(92), "/")}"')
    root = _s(inputs.get("cache_root"))
    if root and folder:
        lines.append(f'cache_directory = "{cache_dir_for(root, folder).replace(chr(92), "/")}"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- the plan
def start_problems(desc, inputs):
    """Everything Start refuses, in the order the desktop app lists it: the shared checks (fizgig.families.checks)
    with the family's own (problems) in their place. The dataset TOML is written by the plan itself, so only its
    path is required here."""
    from fizgig.families import checks
    v = inputs
    out = checks.learning_rate(v)
    out += checks.network(v, lokr=str(v.get("NETWORK_TYPE", "")).startswith("LoKR"))
    out += checks.numbers(v)
    out += checks.dataset_config(v.get("DATASET_CONFIG"), must_exist=False)
    out += problems(desc, inputs)
    out += checks.learning_rate_range(v)
    out += checks.context_lora(v)
    raw_swap = str(v.get("blocks_swap") or "").strip()
    m = re.match(r"\d+", raw_swap)
    swap = 0 if raw_swap.lower().startswith("auto") or not m else int(m.group())   # Auto always fits
    out += checks.run(v, blocks_swap=swap, swap_max=max(0, desc.n_blocks - 2), arch_label=desc.gui_label,
                      name_error=checks.tidy_name(v.get("LORA_NAME"))[1])
    try:
        from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
    except Exception:
        IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    out += checks.training_folder(_s(v.get("image_folder")), _s(v.get("caption_ext", ".txt")),
                                  check_captions=not slider_on(desc, inputs, "prompts"),
                                  media_exts={e.lower() for e in IMAGE_EXTENSIONS})
    return out


def plan(desc, inputs):
    """The whole launch: problems first (a plan with problems must not start), then the stages in order."""
    from fizgig.families.checks import tidy_name
    inputs = dict(inputs)
    folder = _s(inputs.get("image_folder"))
    if folder and not os.path.isabs(folder):
        inputs["image_folder"] = os.path.abspath(folder)     # the subprocesses run from the Fizgig folder
    inputs["LORA_NAME"] = tidy_name(_s(inputs.get("LORA_NAME")))[0]
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
    train = train_command(desc, inputs, p)
    if caches(desc, inputs):
        p.stages += [Stage("Cache Preparation", cache_command(desc, inputs, "latents")),
                     Stage("Text Encoder Caching", cache_command(desc, inputs, "text"))]
    p.stages.append(Stage("Training", train))
    return p
