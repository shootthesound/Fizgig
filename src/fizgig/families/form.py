"""The Training and Samples forms of a described family, as data: what a front end shows, in order, built from the
family description - so a family, an option or a preset added to a description reaches every front end that renders
this (the web UI today) with no change to that front end.

form(desc) -> {"sections": [{"id", "title", "tab", "rows": [row, ...]}, ...], "presets": [...]}. A row is a plain
dict: key, label, kind ("entry" | "choice" | "combo" (editable) | "check" | "text" (several lines) | "path" | "folder"
| "radio" | "note" | "shown": a read-only value the state fills), default, hint, choices (labels) and, for a family option, option=True. upload=True marks a file or a
folder of training data that may be on another computer than the one training (a browser front end offers to upload
it); output and resume folders never are. start ("data" | "output") is where a picker for it opens. Only rows the family can
ever show are in it (its abilities decide: Adaptive LR, loss watch, EMA, precisions, kinds of training, ...).

state(desc, values, image_folder) -> {"hidden": [keys], "disabled": [keys], "notes": {key: text}}: what depends on
the current values (LoKR hides rank and alpha, the kind of training picks its card, a family option's `requires` /
media / mode rules through launch.option_applies), so those rules live here once, in Python.

inputs(desc, values, prefs, ...) -> the launch plan's inputs (fizgig.families.launch), as the desktop app builds them
in _family_launch_inputs: the same keys, the same switches, the family options under FAMILY_OPTIONS.

The generic rows mirror the desktop Training and Samples tabs (lora_trainer_gui.py); their defaults are its
self.settings defaults, and a family's first preset is applied over them on a first visit, as the desktop does."""
import os

from fizgig.families import launch

ADAPTIVE_MIN = ("1e-5", "5e-5", "1e-4", "2e-4", "3e-4")
ADAPTIVE_MAX = ("1e-4", "2e-4", "3e-4", "4e-4")
MEGAPIXELS = ("0.25", "0.37", "0.5", "0.75", "1.0", "1.05", "1.5", "2.0", "2.4", "3.0", "4.2")
SWAP = ("Auto (detect from GPU)", "0  (No swap — 16GB fp8 / 24GB bf16)", "4  (Light — 14GB fp8 / 20GB bf16)",
        "8  (Moderate — 12GB fp8 / 16GB bf16)", "12 (Aggressive — 10GB fp8 / 12GB bf16)", "16 (Max — 8GB fp8 / 10GB bf16)",
        "1", "2", "3", "5", "6", "7", "9", "10", "11", "13", "14", "15")        # the desktop's Blocks Swap list
NETWORK_TYPES = {"lora": "LoRA (standard)", "lokr": "LoKR (Kronecker)"}
EMA = ("Off", "0.98 (recommended)", "0.99 (stronger)", "0.995 (long runs only)")
EMA_SHORT = "Short run (window = ¼ of the run)"
EMA_HINT = ("Saves and previews a running average of the LoRA instead of its latest step: smoother, less overbaked. "
            "0.98 averages over roughly the last 50 steps.")
SIZES = ("512", "640", "768", "1024", "1280", "1536")
KINDS = (("standard", "Standard LoRA"), ("edit", "Edit (original + edited photo pairs)"),
         ("slider", "Slider (a dial between two looks)"), ("finetune", "Fine-tune the whole model"))
FT_REG_MULT = ("0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0")

# the desktop app's self.settings defaults (lora_trainer_gui.py), for the keys these forms carry
DEFAULTS = {
    "LORA_NAME": "", "LEARNING_RATE": "4e-4", "LORA_LR_RATIO": "1", "NETWORK_DIM": "4", "NETWORK_ALPHA": "4",
    "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": "8", "MAX_TRAIN_EPOCHS": "12", "SAVE_EVERY_N_EPOCHS": "1",
    "SAVE_LATER": False, "SAVE_UNTIL_EPOCH": "10", "SAVE_EVERY_AFTER": "1",
    "SEED": "42", "BLOCKS_SWAP": "Auto (detect from GPU)", "RESUME_TRAINING": "", "OPTIMIZER_TYPE": "adamw8bit",
    "OPTIMIZER_ARGS": "", "GRADIENT_ACCUMULATION": "1", "MAX_GRAD_NORM": "1.0", "LR_SCHEDULER": "constant",
    "LR_WARMUP_STEPS": "", "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
    "CONTEXT_LORA_PATH": "", "CONTEXT_LORA_STRENGTH": "1.0", "FAMILY_DOP": False, "FAMILY_MULTI_PERSON": False, "FAMILY_DOP_TRIGGER": "",
    "FAMILY_DOP_CLASS": "", "FAMILY_DOP_TRIGGER2": "", "FAMILY_DOP_CLASS2": "",
    "FAMILY_DOP_TRIGGER3": "", "FAMILY_DOP_CLASS3": "", "FAMILY_DOP_TRIGGER4": "", "FAMILY_DOP_CLASS4": "",
    "COMPILE_BLOCKS": "Auto", "SAVE_STATE": True,
    "FAMILY_SUBJECTS": "1 subject", "FAMILY_PEOPLE_MODE": "Off - a normal LoRA", "FAMILY_SUBJECT_FIX": False,
    "FAMILY_MP_MANUAL": False, "FAMILY_MP_FACECROPS_FIRST": True, "FAMILY_SUBJECT_KIND": "From its captions", "FAMILY_SUBJECT_KIND2": "From its captions",
    "FAMILY_SUBJECT_KIND3": "From its captions", "FAMILY_SUBJECT_KIND4": "From its captions",
    "SAVE_STATE_ON_TRAIN_END": True, "KEEP_LAST_N_STATES": "2", "KREA2_LOSS_WATCH": False,
    "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False, "TRAIN_SUBFOLDERS": False, "CAPTION_TRIGGER_WORD": "",
    "RECAPTION_FIRST": "Training caption (person or character)", "RECAPTION_SECOND": "Exhaustive detail (person or character)", "KREA2_WARMUP_LOOK": False,
    "FAMILY_PRECISION": launch.PRECISION_LABELS["auto"], "FAMILY_EMA": "0.98 (recommended)",
    "FAMILY_TRAINING_ADAPTER": True, "FAMILY_KIND": "standard", "FAMILY_FAST_ID": False,
    "FAMILY_MULTICONCEPT": False, "FAMILY_CONCEPT_DIR": "", "FAMILY_CONCEPT_COUNT": "2 subjects",
    "FAMILY_CONCEPT_DIR3": "", "FAMILY_CONCEPT_DIR4": "", "FAMILY_CONCEPT_BALANCE": False,
    "FAMILY_EDIT_DIR": "", "FAMILY_EDIT_REF": "", "FAMILY_EDIT_CAPTION": "",
    "FAMILY_EDIT_PICTURES": launch.EDIT_PHOTO_ONLY, "FAMILY_EDIT_EXTRAS_DIR": "",
    "FAMILY_SLIDER_SOURCE": "pairs", "FAMILY_SLIDER_DIR": "", "FAMILY_SLIDER_CAPTION": "",
    "FAMILY_SLIDER_BASE": "", "FAMILY_SLIDER_POS": "", "FAMILY_SLIDER_NEG": "", "FAMILY_SLIDER_ANCHORS": "",
    "FAMILY_SLIDER_ANCHOR_WEIGHT": "1", "FAMILY_SLIDER_ULTRA": False,
    "FAMILY_FT_ROTATIONS": "10", "FAMILY_FT_SAVE_EVERY": "1", "FAMILY_FT_ROTATE_EVERY": "1",
    "FAMILY_FT_FUSED": True, "FAMILY_FT_REG_DIR": "", "FAMILY_FT_REG_MULT": "0.2",
    "DATASET_MEGAPIXELS": "0.25", "DATASET_CAPTION_EXT": ".txt", "ENABLE_BUCKET": True, "BUCKET_NO_UPSCALE": True,
    "ENABLE_CACHE": True, "MASKED_TRAINING": False, "MASK_FLOOR": "0.1 (recommended)", "MASK_PRIOR": "Off",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "", "METADATA_LICENSE": "",
    "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "", "METADATA_THUMBNAIL": "",
    "SAMPLE_ENABLED": True, "SAMPLE_PROMPT": "A high quality photo", "SAMPLE_SEED": "1234",
    "SAMPLE_EVERY_N_EPOCHS": "1", "SAMPLE_LATER": False, "SAMPLE_UNTIL_EPOCH": "10",
    "SAMPLE_EVERY_AFTER": "1", "SAMPLE_AT_FIRST": True, "SAMPLE_CHECKPOINT": True, "CACHE_SAMPLE_MODEL": "auto",
}


TRAINING_ORDER = ("model", "output", "params", "edit", "slider", "finetune", "memory", "optimizer", "other")
# any other Training card (Masked training, Subjects) comes after these, under the Extras divider
TRAINING_OPEN = ("output", "params")         # the Training tab's cards open when it opens; the rest start closed


def _row(key, label, kind="entry", hint="", choices=(), default=None, **extra):
    return {"key": key, "label": label, "kind": kind, "hint": hint, "choices": list(choices),
            "default": DEFAULTS.get(key, "") if default is None else default, **extra}


def _recaption_choices():
    from fizgig.krea2.caption_text import RECAPTION_CHOICES      # text only: no torch to load
    return RECAPTION_CHOICES


def _recaption_key(label, default):
    return next((k for k, lab in _recaption_choices() if lab == label), default)


def _optimizers():
    from fizgig.training.optimizers import available_optimizers
    try:
        return list(available_optimizers())
    except Exception:
        return ["adamw8bit", "adamw"]


_BLOCK_GROUPS = {}


def _block_groups(desc):
    """The family's block map as the Custom picker shows it: [{label, blocks: [{id, label}]}] (kept per family -
    loading the driver to read it takes a second)."""
    if desc.key not in _BLOCK_GROUPS:
        try:
            _BLOCK_GROUPS[desc.key] = [{"label": g.label.split()[0], "blocks": [
                {"id": b.id, "label": b.label.split()[-1]} for b in g.blocks]} for g in desc.load_driver().block_map()]
        except Exception:
            _BLOCK_GROUPS[desc.key] = []
    return _BLOCK_GROUPS[desc.key]


def _option_row(opt):
    """A family's own control (FamilyOption) as a row: its declared label, kind, choices, default and hint."""
    kind = {"choice": "choice", "check": "check"}.get(opt.kind, "combo" if opt.suggestions else
                                                      "split" if opt.split or opt.slide else "entry")
    labels = opt.choice_labels() if opt.kind == "choice" else []
    default = ((opt.default if opt.default in labels else labels[0]) if labels else
               opt.default == "1" if opt.kind == "check" else opt.default)
    return _row(opt.key, opt.label, kind, opt.hint, opt.choice_labels() if opt.kind == "choice" else
                [s.split(" · ")[0] for s in opt.suggestions], default, option=True, suffix=opt.suffix,
                **({"labels": list(opt.suggestions)} if kind == "combo" else {}),
                **({"warn": opt.warn, "warn_link": opt.warn_link} if opt.warn else {}),
                **({"featured": True} if opt.featured else {}),
                **({"split": list(opt.split), "split_hint": opt.split_hint} if kind == "split" and opt.split else {}),
                **({"slide": list(opt.slide), "slide_hint": opt.slide_hint} if kind == "split" and opt.slide else {}),
                **({"single": opt.single} if kind == "split" and opt.single else {}))


def model_keys(desc):
    """The Preferences files a run of this family reads, as the desktop's _family_launch_inputs collects them: its
    model files, its speed LoRA, its training adapter and every file an option names (pref:KEY, ...->ALT)."""
    import re
    keys = [f.pref_key for f in desc.model_files]
    sp = desc.preview_speed()
    if sp and sp.pref_key:
        keys.append(sp.pref_key)
    if desc.training_adapter:
        keys.append(desc.training_adapter)
    for o in desc.options:
        for tok in [t for c in o.choices for t in c[1].split()] + o.tokens.split():
            keys += re.findall(r"pref:([A-Za-z0-9_]+)", tok) + re.findall(r"->([A-Za-z0-9_]+)", tok)
    return list(dict.fromkeys(keys))


def kind_sets(desc, kind):
    """What picking a kind of training also sets (the desktop's _family_kind_chosen): Fine-tune takes the family's
    fine-tune learning rate with Adaptive LR off; a family's kind_values come too."""
    out = dict(dict(getattr(desc, "kind_values", ()) or ()).get(kind, ()))
    if kind == "finetune":
        out.update(LEARNING_RATE=f"{desc.ft_learning_rate:g}", ADAPTIVE_LR=False)
    return out


# what each kind is, under the choice (the desktop's _KIND_HINTS; a family words its own in kind_notes)
KIND_HINTS = {
    "standard": "A normal LoRA: a person, a style or a concept, learned from your captioned pictures.",
    "edit": "Teach the model an edit (a grade, a look, a relight) from pairs of the same photo: the original, and "
            "your edited version. The LoRA learns to make that edit to new photos.",
    "slider": "A dial: at strength +1 the LoRA moves a picture toward one look, at -1 toward the other, and "
              "anything in between. Previews show -1, 0 and +1 side by side.",
    "finetune": "The whole model learns your pictures, not a LoRA on top of it. The result is a full model file "
                "(large), loaded in place of the base. Needs plenty of system memory (the Log says how much).",
}
SLIDER_SOURCES = {"pairs": "Photo pairs", "prompts": "Prompts"}


def _kind_notes(desc):
    return {k: (label, hint) for k, label, hint in (getattr(desc, "kind_notes", ()) or ())}


def kinds(desc):
    """The kinds of training the family offers, in the desktop's order (Standard is always there), each labelled in
    the family's own words when it has them (description kind_notes)."""
    on = {"standard": True, "edit": desc.edit_training, "slider": desc.slider_training, "finetune": desc.finetune}
    kn = _kind_notes(desc)
    return [(k, kn.get(k, ("", ""))[0] or label) for k, label in KINDS if on[k]]


def slider_sources(desc):
    """Where the family's sliders come from (description slider_sources), in order; one = no chooser."""
    return [s for s in (getattr(desc, "slider_sources", ("pairs", "prompts")) or ("pairs", "prompts"))
            if s in SLIDER_SOURCES]


CONCEPT_COUNTS = ("2 subjects", "3 subjects", "4 subjects")
SUBJECT_COUNTS = ("1 subject",) + CONCEPT_COUNTS
# the Subjects card's "Your subject and other people": off / DOP / Fizgig LoRA Isolation (DOP + multi-person, the measured recipe)
PEOPLE_MODES = ("Off - a normal LoRA", "Leave the rest of the model alone",
                "Fizgig LoRA Isolation")
PEOPLE_HINT = ("Off: your subject can leak into prompts that don't use its trigger word (a face onto other "
               "people, a style onto every picture).\n"
               "Leave the rest of the model alone: ask for \"a man at a party\" without the trigger word and you get "
               "a stranger, just as the model made before you trained. Your subject only appears when you call for "
               "it. With one subject, use it when it must stay out of prompts without the trigger word; otherwise "
               "use Off.\n"
               "Fizgig LoRA Isolation: your subjects can stand together in one picture, or next to "
               "a person the model already knows well, and every face stays right. Best for people, and needs two or "
               "more subjects in the LoRA. "
               "Training takes about 1.5x as long. Prompt with the word shown at the bottom of this card for "
               "each subject, e.g. zwxemwoman.")
VIDEO = "https://youtu.be/8uW5NdL5b20"          # the Subjects card video: setting up Fizgig LoRA Isolation
GUIDE = "https://github.com/shootthesound/Fizgig/blob/master/docs/MULTI_PERSON.md"     # the multi-person captioning guide (docs/MULTI_PERSON.md)
GUIDE_MASKS = "https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md"   # making and using masks
from fizgig.families.multi_person import KINDS as SUBJECT_KINDS  # noqa: E402
KIND_HINT = ("What each subject is decides what Fizgig adds to its training.\n"
             "From its captions: Fizgig decides - a person if the captions say woman, man, girl or boy, otherwise an "
             "animal or thing.\n"
             "A person: also trains on side-by-side pictures next to strangers, so other people keep their own "
             "faces.\n"
             "An animal or thing: a dog, a cat, a car. Its word is joined like a person's (rexodog), with no "
             "side-by-side pictures.\n"
             "A style or look: paper cut, a lighting style. Its trigger word stays as written, and the look only "
             "appears when you use it.")
KIND_ROWS = ("FAMILY_SUBJECT_KIND", "FAMILY_SUBJECT_KIND2", "FAMILY_SUBJECT_KIND3", "FAMILY_SUBJECT_KIND4")
SUBJECTS_HINT = ("Each subject is a folder of its own training {files}, with its own trigger word in every caption - "
                 "that is how the LoRA tells them apart. Caption each folder first.")
# measured on people, so only where the family has the people modes (not a music model)
SUBJECTS_PEOPLE_HINT = ("For people: up to 4 work well, with 2 or 3 of them in one picture. With a person the model "
                        "already knows well in the picture too: 2 is reliable, 3 is hit and miss.")

def form(desc):
    opts = [o for o in desc.options if o.kind != "fixed"]
    pictures = bool({"photo", "clip"} & set(desc.media))     # sizes and megapixels only mean something for pictures
    by_section = lambda tab, section=None, mode=None: [_option_row(o) for o in opts if o.tab == tab and (
        section is None or o.section == section) and (mode is None or (o.mode == "finetune") == (mode == "finetune"))]
    sections = []

    def add(sid, title, tab, rows, collapsed=False, help=None):
        """help: {"href", "text", "sub"?} - the card's video, a link across its top"""
        rows = [r for r in rows if r]
        if tab == "training":                    # the Training tab opens on Output and Training Parameters only
            collapsed = sid not in TRAINING_OPEN
        if rows:
            sections.append({"id": sid, "title": title, "tab": tab, "rows": rows, "collapsed": collapsed,
                             **({"help": help} if help else {})})

    add("model", "Base Model", "training",
        by_section("model") + ([_row("_model_note", desc.model_note, "note")] if desc.model_note else []))
    add("output", "Output", "training", [
        _row("LORA_OUTPUT_DIR", "Output Directory", "folder", "Where the LoRA, its checkpoints and the previews go. A folder picked here is kept for "
             "this model; empty (or Reset to default) uses the LoRA output folder in Preferences.",
             default=""),
        _row("LORA_NAME", "LoRA Name", default=f"my_lora_{desc.lora_name_suffix}" if desc.lora_name_suffix else "my_lora"),
    ])
    ks = kinds(desc)
    params = [
        _row("_when_read", "These settings are read when a run starts. A paused run resumes with its own settings"
             + (" (only a new picture size is offered)" if pictures else "") + "; a dataset or caption change needs a "
             "fresh run." + (" The previews' prompts can change while it trains, on the Samples tab." if pictures else ""),
             "note"),
        _row("FAMILY_KIND", "Kind of training", "radio", choices=[label for _, label in ks], default=ks[0][1],
             sets={label: kind_sets(desc, k) for k, label in ks if kind_sets(desc, k)}) if len(ks) > 1 else None,
        _row("TRAIN_SUBFOLDERS", "Also train on its sub-folders", "check", SUBFOLDERS_HINT, default=False),
        _row("LEARNING_RATE", "Learning Rate", default=DEFAULTS["LEARNING_RATE"]),
    ]
    if desc.adaptive_lr:
        params += [
            _row("ADAPTIVE_LR", "Adaptive LR (auto-adjust based on loss, gradient clipping & weight-norm growth)",
                 "check", "When on, the Learning Rate box is ignored: the run starts at the geometric midpoint of "
                 "Min and Max and the watcher sets the rate from there."),
            _row("ADAPTIVE_LR_MIN", "Min LR", "choice", choices=ADAPTIVE_MIN),
            _row("ADAPTIVE_LR_MAX", "Max LR", "choice", choices=ADAPTIVE_MAX),
        ]
    if desc.train_areas:
        names = [a[0] for a in desc.train_areas]
        # a family whose areas already cover the whole model (desc.train_area_custom = False) is not offered the
        # per-block picker: the areas are the choice list, and "Custom" would only re-enter the same blocks by hand
        custom = bool(getattr(desc, "train_area_custom", True))
        params.append(_row("TARGET_LAYERS", "Model Area to Train", "choice",
                           "Which blocks train. Style also limits training to the late, clean timesteps."
                           + (" Custom picks the blocks yourself." if custom else " Pick one of the family's areas."),
                           names + (["Custom"] if custom else []),
                           names[0], timesteps={a[0]: [str(int(a[2][0] * 1000)), str(int(a[2][1] * 1000))]
                                                if a[2] else ["", ""] for a in desc.train_areas}))
        groups = _block_groups(desc) if custom else []
        if groups:                           # the desktop's Custom block picker: All / None, an area's blocks, ticks
            params.append(_row("TRAINING_BLOCKS", "Blocks to train", "blocks", "", default={}, groups=groups,
                               areas={a[0]: list(a[1]) for a in desc.train_areas if a[1]}))
        # the family's timesteps are the reference's sigma recipe: no range to set, so the boxes are gone
    params += [
        _row("NETWORK_DIM", "Network Dim (Rank)"),
        _row("NETWORK_ALPHA", "Network Alpha"),
        _row("MAX_TRAIN_EPOCHS", "Max Epochs"),
        # the same control as How often to preview (a cadence row): every N epochs, switched on another rate from
        # an epoch on; ticking fills in every 5, then from epoch 10 every 1
        _row("SAVE_EVERY_N_EPOCHS", "How often to save", "cadence",
             "Ticked: every N epochs up to that epoch, then the second rate (whole multiples). Every 5, then from "
             "epoch 10 every 1, saves epochs 5, 10, 11, 12, 13 and on. The last epoch is always saved.",
             tick_fills={"SAVE_EVERY_N_EPOCHS": "5"},
             parts=[{"key": "SAVE_LATER", "default": False}, {"key": "SAVE_UNTIL_EPOCH", "default": "10"},
                    {"key": "SAVE_EVERY_AFTER", "default": "1"}]),
        _row("SEED", "Seed"),
        _row("CONTEXT_LORA_PATH", "Context LoRA", "path",
             "Trains with an existing LoRA active on the base. Use the same LoRA and strength at inference.",
             upload=True, start="lora"),
        _row("CONTEXT_LORA_STRENGTH", "Context LoRA strength"),
        None if not pictures else _row("DATASET_MEGAPIXELS", "Target Megapixels", "combo",
             "Images are resized to this area, any aspect ratio: 0.25 = 496 x 496, 1.0 = 992 x 992, 1.05 = 1024 x 1024, 2.4 = 1536 x "
             "1536, 4.2 = 2048 x 2048. 4.2 MP wants 24-32 GB.", MEGAPIXELS),
        _row("CLIP_MEGAPIXELS", "Clip Target Megapixels", "combo",
             "The size your video clips are cached and trained at; photos keep the Target Megapixels. A clip takes "
             "far more memory and time than a photo of the same size, so e.g. photos at 1.0 with clips at 0.25 "
             "keeps caching and training fast. A clip that would not fit in your card's free memory is cached "
             "smaller still.", MEGAPIXELS, "0.25") if "clip" in desc.media else None,
    ]
    if len(desc.network_types) > 1:
        params += [_row("NETWORK_TYPE", "Network Type", "choice",
                        desc.network_hint or "LoKR: potentially higher quality · LoRA: ~20% faster training",
                        [NETWORK_TYPES.get(t, t) for t in desc.network_types]),
                   _row("LOKR_FACTOR", "LoKR Factor", hint="8 is the sweet spot · 4 = four times the parameters: "
                        "stronger, a bigger file, and ~5 GB more VRAM with adamw (use adamw8bit) · above 8: just use LoRA")]
    if desc.loss_watch:
        params += [
            _row("KREA2_LOSS_WATCH", "Track how each image trains (per-image loss)", "check",
                 "Follows every image's loss through the run. Most images do their work early and are simply learned "
                 "by the end, which is what you want; a few stay hard, often because the caption doesn't match the "
                 "picture. Each image's progress, with its thumbnail and its caption to edit, is under Per-image loss "
                 "on the run card while it trains, and in Runs & outputs."),
            _row("KREA2_PER_IMAGE_LR", "Per-image adaptive LR (throttle stuck images, boost healthy learned ones)",
                 "check", "Images that stay hard without improving train more gently (x0.7, then x0.5 down to x0.1); "
                 "mined-out ones ease off (x0.6) and healthy learned ones get a gentle boost (x1.1). Batch size 1."),
            _row("KREA2_WARMUP_LOOK", "Warm up look outliers (unusual angles ease in at low LR - needs a Look "
                 "Filter scan)", "check", "Person LoRAs: images the Image Prep Look Filter scored as look outliers (tight angles, "
                 "profiles - real but unusual) start at x0.4 and ramp to x1.0 over the first ~4 epochs, so they refine "
                 "the identity instead of fighting it while it forms. Run the Look Filter (with 3 baselines) on this "
                 "folder first: it saves the scores with the dataset. Without them this does nothing."),
            _row("KREA2_AUTO_RECAPTION", "Auto-recaption stuck images (Qwen3-VL rewrites the caption between "
                 "epochs)", "check", "A stuck image gets a new caption written from what is in it, and a fresh start; "
                 "a second try goes into more detail. Still stuck after that, it is left out of training - edit its "
                 "caption to bring it back. Needs the Krea 2 text encoder set in Preferences."),
            _row("RECAPTION_FIRST", "First try uses", "choice",
                 "The Captions tab's instruction for that task, as you've edited it there. The defaults are the "
                 "person or character versions: for an object or product, pick the (object or product) Training "
                 "caption here and its Exhaustive detail for the second try; for a pose, an action or an effect, "
                 "the concept task (it uses the words in the Captions tab's The concept is box); for a style, "
                 "Style for both; for anything else, Custom with your own instruction.",
                 [label for _key, label in _recaption_choices()]),
            _row("RECAPTION_SECOND", "Second try uses", "choice",
                 "Used when the first new caption didn't help. Set separately from the first try.",
                 [label for _key, label in _recaption_choices()]),
            _row("CAPTION_TRIGGER_WORD", "Trigger word", hint="Used by Auto-recaption (below): it adds this word at the "
                 "end of a new caption. The training folder's word, the same as the Captions tab's."),
        ]
    params += by_section("training", "", "lora")
    if desc.identity_blocks:
        a, b = desc.identity_blocks[0], desc.identity_blocks[-1]
        params.append(_row("FAMILY_FAST_ID", "Fast Identity Mode", "check",
                           f"For a person or character LoRA: trains only the identity blocks ({a}-{b}), about 1.5x "
                           "faster. Leave it off for a style or concept."))
    params += by_section("training", "after", "lora")
    # who the LoRA is of and how it treats other people: the Subjects card (after Training Parameters)
    mc = dict(desc.multi_concept_defaults or {})
    subjects = [
        _row("_subjects_intro", "With 1 subject and Off, this is ordinary LoRA training. The other choices keep your "
             "subjects apart from everyone else, using Fizgig's own techniques together with DOP (Differential "
             "Output Preservation) from @ostris's AI-Toolkit.", "note"),
        _row("FAMILY_SUBJECTS", "Who or what is this LoRA of?", "radio",
             " ".join(t for t in ("\n".join(t for t in (SUBJECTS_HINT.format(files="pictures" if pictures else
                                                                            "files"),
                                                      SUBJECTS_PEOPLE_HINT if desc.subject_isolation else "") if t),
                                  desc.multi_concept_hint) if t), SUBJECT_COUNTS,
             default=SUBJECT_COUNTS[0],
             sets={SUBJECT_COUNTS[0]: {"FAMILY_MULTICONCEPT": False},
                   **{c: {**mc, "FAMILY_MULTICONCEPT": True, "FAMILY_CONCEPT_COUNT": c} for c in CONCEPT_COUNTS}})
        if desc.multi_concept else None,
        _row("_subject1_folder", "Subject 1", "shown", "The training folder - change it on the Start tab."),
        _row("FAMILY_SUBJECT_KIND", "Subject 1 is", "choice", KIND_HINT, SUBJECT_KINDS, default=SUBJECT_KINDS[0],
             hint_labels=True),                 # one spaced line per choice, its name in bold (said once, here)
        *([_row("FAMILY_CONCEPT_DIR", "Subject 2", "folder", upload=True, start="data"),
           _row("FAMILY_SUBJECT_KIND2", "Subject 2 is", "choice", choices=SUBJECT_KINDS, default=SUBJECT_KINDS[0]),
           _row("FAMILY_CONCEPT_DIR3", "Subject 3", "folder", upload=True, start="data"),
           _row("FAMILY_SUBJECT_KIND3", "Subject 3 is", "choice", choices=SUBJECT_KINDS, default=SUBJECT_KINDS[0]),
           _row("FAMILY_CONCEPT_DIR4", "Subject 4", "folder", upload=True, start="data"),
           _row("FAMILY_SUBJECT_KIND4", "Subject 4 is", "choice", choices=SUBJECT_KINDS, default=SUBJECT_KINDS[0]),
           ] if desc.multi_concept else []),
        _row("FAMILY_PEOPLE_MODE", "Your subject and other people", "radio", PEOPLE_HINT, PEOPLE_MODES,
             default=PEOPLE_MODES[0], hint_labels=True,           # one line per choice, its name in bold
             sets={PEOPLE_MODES[0]: {"FAMILY_DOP": False, "FAMILY_MULTI_PERSON": False},
                   PEOPLE_MODES[1]: {"FAMILY_DOP": True, "FAMILY_MULTI_PERSON": False},
                   PEOPLE_MODES[2]: {"FAMILY_DOP": True, "FAMILY_MULTI_PERSON": True}}),
        _row("FAMILY_MP_MANUAL", "Use my captions exactly as written", "check",
             "No joining or guessing: Fizgig trains on your captions as they are. Write each subject's joined word "
             "(zwxemwoman) into the captions yourself. The side-by-side pictures and the training recipe stay on.",
             link={"href": GUIDE + "#captioning-it-yourself", "text": "How to caption for this"}),
        _row("FAMILY_MP_FACECROPS_FIRST", "Side-by-side pictures: face crops first", "check",
             "The 16 pictures of your subject beside a stranger use Image Prep's face crops (FaceCrop_ files) "
             "first, then photos where a face is found. Untick to pick from every photo, e.g. for more body "
             "shape. A face crop is never cropped further; other photos are framed on the face, never zoomed."),
        _row("FAMILY_SUBJECT_FIX", "Correct a trigger word, or what a subject is", "check",
             "Fizgig reads each subject's trigger word, and what they are, from their captions. Tick this only if "
             "the line under a subject shows the wrong word."),
        # beside the card's other ticks (the tiles flow together): Use my captions, face crops first, Correct a trigger
        *([_row("FAMILY_CONCEPT_BALANCE", "Balance the subjects (optional)", "check", "Repeats the smaller folders "
                "so each subject gets about as many steps an epoch as the largest. Epochs get longer by the repeats.")]
          if desc.multi_concept else []),
        *[r for n in (1, 2, 3, 4) for r in (
            _row("FAMILY_DOP_TRIGGER" + ("" if n == 1 else str(n)), f"Subject {n} - trigger word",
                 hint="The word your captions use for this subject, e.g. zwxem.\n"
                      "The box shows the word Fizgig found. Type over it only if that is wrong."),
            _row("FAMILY_DOP_CLASS" + ("" if n == 1 else str(n)), f"Subject {n} - what they are",
                 hint="One word: woman, man, girl, boy, dog, cat, car ...\n"
                      "The box shows what Fizgig found. Type over it only if that is wrong."))],
        # how often and how strongly the trigger-free pass runs is the tested recipe (every second step, x2): no control
        # last, under everything that changes them: what training will use, per subject (one of its own captions)
        *[_row(f"_subject{n}_as", f"Subject {n} - what training will use", "note", hint_labels=True)
          for n in ((1, 2, 3, 4) if desc.multi_concept else (1,))],
    ]
    if desc.training_adapter:
        params.append(_row("FAMILY_TRAINING_ADAPTER", "Training adapter (recommended)", "check",
                           desc.training_adapter_note))
    if desc.ema_default:
        params.append(_row("FAMILY_EMA", "Weight averaging (EMA)", "choice", desc.ema_hint or EMA_HINT,
                           EMA + ((EMA_SHORT,) if desc.ema_short_run else ())))
    # the on/off tiles together at the bottom of the card, under every plain box (Adaptive LR stays beside its Min/Max)
    TILES = ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK", "KREA2_AUTO_RECAPTION", "FAMILY_FAST_ID",
             "FAMILY_TRAINING_ADAPTER")
    UNDER = ("RECAPTION_FIRST", "RECAPTION_SECOND")      # Auto-recaption's pickers, straight under the tiles
    params = ([r for r in params if not (r and r["key"] in TILES + UNDER)] + [r for r in params if r and r["key"] in TILES]
              + [r for r in params if r and r["key"] in UNDER])
    add("params", "Training Parameters", "training", params)
    if desc.masked_training:
        add("masks", "Masked training", "training", [
            _row("MASKED_TRAINING", "Train on the parts your masks show", "check",
                 "For pictures where only part of the frame matters - the subject, not the room around it (a person, a "
                 "product, a pet).\n\n"
                 "Save a black-and-white mask beside a picture, named after it with -masklabel added: photo.jpg -> "
                 "photo-masklabel.png. White is trained fully, grey partly, black only as much as the setting "
                 "below. The picture itself still trains whole.\n\n"
                 "Pictures without a mask train as normal. Masks made in OneTrainer work as they are.",
                 link={"href": GUIDE_MASKS, "text": "How to make masks"}),
            _row("MASK_FLOOR", "How much outside the mask counts", "choice",
                 "0.1 keeps a little of the background, so the LoRA still knows your subject sits in a scene. "
                 "0 ignores it completely - then a very small mask trains hard on a few spots.", launch.MASK_FLOORS),
            _row("MASK_PRIOR", "Outside the mask, keep the model's own picture (prior preservation)", "choice",
                 "Off: outside the mask the LoRA still learns a little of your photos (as much as How much outside "
                 "the mask counts says).\n"
                 "0.5, 1 or 2: outside the mask it learns to draw what the model itself would, so backgrounds and the "
                 "model's look stay its own. The higher the number, the firmer.\n"
                 "Speed: each step takes up to about half as long again.",
                 launch.MASK_PRIORS, hint_labels=True),                 # one spaced line per choice, name in bold
        ])
    add("subjects", "Subjects", "training", subjects,
        help={"href": VIDEO, "text": "How to set up Fizgig LoRA Isolation",
              "sub": "A step-by-step video of this card · Watch on YouTube"})
    if desc.edit_training:
        add("edit", "Edit pairs", "training", [
            _row("_edit_note", desc.edit_note, "note") if desc.edit_note else None,
            # a multi-picture edit (description edit_references > 1): the choice, then its own folder (step 2)
            _row("FAMILY_EDIT_PICTURES", "What the edit uses", "radio", choices=[launch.EDIT_PHOTO_ONLY, launch.EDIT_WITH_EXTRAS], hint=
                 "The photo only: an edit that changes how a photo looks (a grade, a relight, a retouch). The photo + "
                 "extra pictures: an edit that brings something from other pictures into it (a person, an object, a "
                 "place).", default=launch.EDIT_PHOTO_ONLY) if desc.edit_references > 1 else None,
            _row("FAMILY_EDIT_DIR", "1. Originals folder (before editing)", "folder",
                 "Your original, unedited photos. No captions needed.", upload=True, start="data"),
            _row("FAMILY_EDIT_EXTRAS_DIR", "2. Extra pictures folder", "folder",
                 "The pictures to bring into the edits: a person, an object, a place.\n"
                 "Name each after the photo it belongs to, adding _1, _2 ...: for beach.jpg, its extras are "
                 "beach_1.jpg, beach_2.jpg.\n"
                 "The number is the order the model sees them in: beach_1 is the second picture after the original, "
                 "beach_2 the third. Say so in the edit instruction, e.g. \"Put the man from the second picture into "
                 "this scene.\"\n"
                 f"A photo can have none (it trains as a normal pair) or up to {desc.edit_references - 1}. Any shape "
                 "or file type.", upload=True, start="data")
            if desc.edit_references > 1 else None,
            _row("_edit_extras_found", "", "note") if desc.edit_references > 1 else None,
            _row("_edit_edited_dir", "2. Edited folder (after editing)", "shown", "The same photos after your edit, "
                 "with the same file names as the originals. It is the training folder on the Start tab: choosing it "
                 "here changes it there too.", sets_training_folder=True),
            _row("FAMILY_EDIT_CAPTION", "3. Edit instruction", hint="What the edit is, e.g. \"Apply my concert grade.\" "
                 "Write captions saves it as the caption of every photo in the Edited folder.",
                 action="write_captions"),
            _row("FAMILY_EDIT_REF", "4. Test photo (optional)", "path",
                 "Optional - you can leave this empty. A photo for the previews to edit, so you can watch the LoRA "
                 "learn: an original, unedited photo that is in neither folder. One it never trains on shows whether "
                 "it learned the edit itself. Any size.\n"
                 "Left empty, the previews use your first original - that works, but it is a photo the LoRA trains on, "
                 "so the previews can look better than the LoRA really is." + (
                     "\nWith extra pictures: put the test photo's own in the Extra pictures folder the same way "
                     "(test_1.jpg, test_2.jpg ... for test.jpg)." if desc.edit_references > 1 else ""),
                 upload=True, start="data"),
        ])
    if desc.slider_training:
        srcs = slider_sources(desc)
        pn = {k: (h, t) for k, h, t in (getattr(desc, "slider_prompt_notes", ()) or ())}
        # the desktop's wording (lora_trainer_gui's Slider card), unless the family says it its own way
        prompt_row = lambda key, label, hint: _row(key, pn.get(key, (label, hint))[0], hint=pn.get(key, (label, hint))[1])
        clips = "clip" in desc.media
        add("slider", "Slider", "training", [
            _row("FAMILY_SLIDER_SOURCE", "Slider from", "radio", choices=[SLIDER_SOURCES[x] for x in srcs],
                 default=SLIDER_SOURCES[srcs[0]]) if len(srcs) > 1 else None,
            _row("_slider_plus_dir", "1. +1 end folder", "shown", "Photos showing the +1 end of the dial (e.g. the "
                 "person smiling, or the room in warm light). The captions go in this folder (step 3). It is the training folder on the Start "
                 "tab: choosing it here changes it there too.", sets_training_folder=True),
            _row("FAMILY_SLIDER_DIR", "2. -1 end folder", "folder",
                 "The same shots showing the -1 end (e.g. not smiling), each with the same file name as its +1 photo. "
                 "No captions in this folder.", upload=True, start="data"),
            _row("FAMILY_SLIDER_CAPTION", "3. Caption for every pair", hint="Describe what the two photos of a pair have "
                 "in common, and leave out the difference. For a smile dial, where each +1 photo shows someone smiling "
                 "and its -1 photo the same person not smiling, the caption is \"a portrait photo of a person\" - not "
                 "\"a smiling person\", because the smile is the difference the dial learns from the photos. For a dial "
                 "about something else, describe the subject the same way, e.g. \"a photo of a room\" for a lighting "
                 "dial. Write captions saves the same line as the caption of every photo in the training (+1) folder, so "
                 "keep it general enough to fit every pair.", action="write_captions"),
            prompt_row("FAMILY_SLIDER_BASE", "1. What the picture is",
                       "The start of the prompt. Each end's words are added after it with a space, so end this line "
                       "with a comma if you want one. One person: describe them, e.g. \"a close-up photo of a young "
                       "woman with short dark hair,\" - the dial changes the expression and keeps her the same. Anyone: "
                       "keep it general, e.g. \"a close-up photo of a person who is\" - the practice pictures show a "
                       "different person each time, so the dial learns only the change and works on anyone. Something "
                       "else: describe it the same way, e.g. \"a photo of a living room,\" for a lighting dial."),
            prompt_row("FAMILY_SLIDER_POS", "2. The +1 end adds",
                       "e.g. \"happy\" - with the general line above the prompt reads \"a close-up photo of a person "
                       "who is happy\"."),
            prompt_row("FAMILY_SLIDER_NEG", "3. The -1 end adds",
                       "e.g. \"sad\". Words like this can bring more than a face (grey light, rain); if the whole scene "
                       "changes, try a narrower word such as \"unsmiling\"."),
            _row("FAMILY_SLIDER_GUIDANCE", "4. Push strength", default=f"{desc.slider_guidance:g}",
                 hint="How hard the ends are pushed apart. Higher gives a stronger dial but changes more than the one "
                      f"thing you asked for (a frown can turn into a different, older man). Start on "
                      f"{desc.slider_guidance:g}. You can change the prompts and this while it trains, on the Run card."),
            _row("FAMILY_SLIDER_ANCHORS", *pn.get("FAMILY_SLIDER_ANCHORS", ("5. Leave these alone (optional)", None))[:1],
                 "text", hint=pn.get("FAMILY_SLIDER_ANCHORS", (None, None))[1] or "Prompts the dial must not change, one a line. For a hair-length dial meant for women, \"a photo "
                      "of a man\" keeps men's hair as it is; for a smile dial, \"a photo of a dog\" keeps it to people. "
                      "Each step checks one of them in turn, and the previews show the first one beside the dial, so "
                      "you can see it stay put. Each one makes a step about half as long again. Empty = none."),
            _row("FAMILY_SLIDER_ANCHOR_WEIGHT", pn.get("FAMILY_SLIDER_ANCHOR_WEIGHT", ("6. How firmly",))[0],
                 default="1", hint=pn.get("FAMILY_SLIDER_ANCHOR_WEIGHT", (None, None))[1] or "How much keeping them as they are counts against the dial itself. 1 is equal weight; higher "
                      "holds them tighter but can weaken the dial; lower lets a little of the change through."),
            _row("FAMILY_SLIDER_ULTRA", "Ultra mode", "check",
                 hint="Allows use at a much higher range of strengths. Affects fine detail less than a regular slider "
                      "LoRA. Best for sliders from prompts; with photo pairs, best when the change is compositional "
                      "(pose, framing, layout).") if desc.slider_ultra_blocks else None,
            # how it works, at the bottom: the source's own help (the desktop's "Where the two ends come from")
            _row("_slider_help_pairs", "Photo pairs: two folders of the same shots, one for each end of the dial (e.g. "
                 "the same person smiling, and not smiling). 4 to 10 pairs, each pair framed the same. For a dial about "
                 "people, use a different person in each pair if you can: the dial then learns only the change, and "
                 "works on anyone. A dial about something else (lighting, colour, a scene) works the same way: the same "
                 "subject at both ends of a pair, and a different one in each pair." + (
                     " Clips work too: a pair can be two clips with the same file name, the +1 clip and the same clip "
                     "with the look you want at the other end (a frame-skip look, a visual effect, a rendering style). "
                     "Looks that only exist in motion train this way, and the previews play -1, 0 and +1 side by side "
                     "as one clip." if clips else ""), "note"),
            _row("_slider_help_prompts", "Prompts: no photos - you describe the picture and what each end adds, and "
                 "Fizgig renders its own practice pictures from the description and trains on them. The practice "
                 "pictures are the size Target Megapixels sets (0.5 = 704 x 704). The previews are strips of three "
                 "pictures on one seed, at -1, 0 and +1.", "note"),
        ])
    if desc.finetune:
        add("finetune", "Fine-tune", "training", [
            _row("FAMILY_FT_ROTATIONS", "Rotations", hint="Each rotation trains every part of the model once."),
            _row("FAMILY_FT_SAVE_EVERY", "Checkpoint + preview every N rotations"),
            _row("FAMILY_FT_ROTATE_EVERY", "Epochs per part before the next"),
            _row("FAMILY_FT_MAX_PARTS", "Window size", "choice", "How many parts train together. Auto packs as many as "
                 "your card holds - the whole model at once when it fits, so nothing rotates. A smaller cap means more "
                 "windows and more memory to spare, if Auto runs too close to your card's limit.",
                 [lab for lab, _ in launch.FT_WINDOW_SIZES], default=launch.FT_WINDOW_SIZES[0][0]),
            _row("FAMILY_FT_FUSED", "Free each gradient as soon as it's used (less memory - recommended)", "check"),
            *by_section("training", mode="finetune"),
            *([_row("FAMILY_FT_REG_DIR", "Regularisation images (optional)", "folder", "A folder of ordinary photos "
                    "of the broader class (people for a person, cars for a car, plain photos of the same kinds of "
                    "scene for a style) with normal detailed captions. They hold the model's "
                    "sense of that class in place while your images train - a full fine-tune moves every weight, so "
                    "nothing else bounds that drift. Use real photos, not model output: generated images anchor the "
                    "model to its own artifacts. Leave empty to train without one.", upload=True, start="data"),
               _row("FAMILY_FT_REG_MULT", "Regularisation LR ×", "combo", "The regularisation photos train at this "
                    "fraction of the LR: 0.1-0.3 keeps them a light anchor; at 1.0 they are a second subject set.",
                    choices=FT_REG_MULT)]
              if getattr(desc, "ft_regularisation", True) else []),
            _row("_ft_to_lora", "To share a fine-tune as ordinary LoRAs (several ranks at once), use Extract > From a "
                 "fine-tune with the checkpoint it saves.", "note"),
        ])
    precisions = ["auto"] + [p for p in ("bf16", "int8", "nf4", "hqq") if p in desc.precisions]
    labels = {**launch.PRECISION_LABELS, **dict(desc.precision_labels or ())}
    add("memory", "Memory & Precision", "training", [
        _row("BLOCKS_SWAP", "Blocks Swap", "combo", "Auto plans the fewest swapped blocks that fit your free VRAM.",
             SWAP),
        _row("FAMILY_PRECISION", desc.precision_label or "Base precision", "choice", desc.precision_hint,
             [labels[p] for p in precisions], default=labels["auto"]) if len(desc.precisions) > 1 else None,
        _row("COMPILE_BLOCKS", "Compile Blocks", "choice", desc.compile_hint, ["Auto", "On", "Off"])
        if desc.compiles else None,
        _row("RESUME_TRAINING", "Resume Training", "folder", "Empty for a normal run. The easy way to carry "
             "a run on is Continue on Runs & outputs: from its newest save, with its own settings (a run that trained "
             "all its epochs asks how far to go). Here, a state folder (myLora-000012-state) continues that run with "
             "the settings on this tab, so keep them as the run had them - its Optimizer Type especially - and raise "
             "Max Train Epochs to train a finished LoRA further, or it ends at once."),
        _row("SAVE_STATE", "Save state at each checkpoint", "check"),
        _row("SAVE_STATE_ON_TRAIN_END", "Save state at the end of training", "check"),
        _row("KEEP_LAST_N_STATES", "Keep last N states", hint="States are big (a LoRA plus its optimizer)."),
    ])
    add("optimizer", "Optimizer", "training", [
        _row("OPTIMIZER_TYPE", "Optimizer Type", "choice", choices=_optimizers()),
        _row("OPTIMIZER_ARGS", "Optimizer Args"),
        _row("GRADIENT_ACCUMULATION", "Gradient Accumulation", hint="Every model trains one image at a time. For a "
             "larger effective batch, raise this: the same averaged gradient as a bigger batch, at the memory of "
             "batch 1."),
        _row("MAX_GRAD_NORM", "Max Grad Norm"),
    ], collapsed=True)
    add("other", "Other Options", "training", [
        _row("DATASET_CAPTION_EXT", "Caption Extension"),
        _row("ENABLE_BUCKET", "Enable Bucket", "check"),
        _row("BUCKET_NO_UPSCALE", "No Upscale (keep small images at native size)", "check"),
        _row("LR_SCHEDULER", "LR Scheduler", "choice", choices=["constant", "constant_with_warmup", "cosine",
                                                                 "cosine_with_restarts", "linear", "polynomial"]),
        _row("LR_WARMUP_STEPS", "Warmup steps"),
        *by_section("training", "other", "lora"),
        _row("METADATA_TITLE", "Title"), _row("METADATA_AUTHOR", "Author"),
        _row("METADATA_DESCRIPTION", "Description"), _row("METADATA_LICENSE", "License"),
        _row("METADATA_TAGS", "Tags"),
        _row("METADATA_TRIGGER_PHRASE", "Trigger phrase", hint="Blank uses the captions' trigger word."),
        _row("METADATA_THUMBNAIL", "Thumbnail", "path", "Blank embeds the latest preview; type off for none.",
             upload=True, start="output"),
        _row("ENABLE_CACHE", "Prepare the cache before training", "check"),
    ], collapsed=True)

    # Samples tab
    neg = desc.preview_negative
    sp, spd = desc.preview_speed(), desc.preview_speed_defaults()
    turbo_rows = []
    if sp is not None and sp.pref_key and desc.samples_turbo_pace:     # H3's row: N steps at M%
        turbo_rows = [
            _row("FAMILY_TURBO_STEPS", "Turbo preview steps", default=str(spd[0])),
            _row("FAMILY_TURBO_PACE", "Turbo strength (%)", default=str(round(100 * spd[1])),
                 hint=f"Previews render in these few steps with the {sp.name} at this strength on top of your "
                      "training LoRA - previews only, never the saved LoRA. 6 steps at 75% is the tested "
                      "recommendation; without the Turbo, the Steps box applies."),
        ]
    elif sp is not None and sp.pref_key:
        live, plain = {"SAMPLE_STEPS": str(spd[0])}, {"SAMPLE_STEPS": str(desc.preview_steps)}
        if not desc.samples_cfg_free and sp.settings.cfg != desc.preview_cfg:
            live["SAMPLE_CFG_SCALE"], plain["SAMPLE_CFG_SCALE"] = f"{sp.settings.cfg:g}", f"{desc.preview_cfg:g}"
        cfg_on = "" if desc.samples_cfg_free else f", CFG {sp.settings.cfg:g}"
        cfg_off = "" if desc.samples_cfg_free else f", CFG {desc.preview_cfg:g}"
        turbo_rows = [_row(
            "FAMILY_TURBO_STRENGTH", "Turbo LoRA strength", default=f"{spd[1]:g}", boxed=True,
            hint=(f"0 = off: full-quality previews ({desc.preview_steps} steps{cfg_off}). 1 = on: the {sp.name} "
                  f"renders previews in {spd[0]} steps{cfg_on}, much faster but softer, so they undersell your LoRA. "
                  "Steps and CFG switch with it. Previews only, never the saved LoRA."
                  + (" It works with Kromoss as the preview model, and on previews of your Base, but not with SFT: "
                     "with SFT as the preview model this box is greyed out. To use it, set Kromoss as the preview "
                     "model in Preferences."
                     if desc.checkpoint_speed_except else
                     (" It works on previews of the training model only: untick "
                      f"\"{desc.preview_checkpoint_tick.split(' (')[0]}\" above to use it.")
                     if desc.train_preview_checkpoint else "")
                  + (" Default " + f"{spd[1]:g}." if spd[1] else "")),
            turbo={"on": live, "off": plain})]
    if sp is not None and sp.pref_key:
        turbo_rows.append(_row("TURBO_MISSING", "", "note",
                               f"Set the {sp.name} in Preferences for fast Turbo previews; without it previews use "
                               "the plain model."))
    add("samples", "Previews during training", "samples", [
        # an edit run's previews come from the Edit pairs card, not these prompts (shown only while Edit is picked)
        _row("_edit_samples_note", "Edit LoRA previews: each round makes one picture from your edit instruction and "
             "test photo (Training tab, Edit pairs" + (", with its extra pictures" if desc.edit_references > 1 else "")
             + "). The prompts below are not used - to try other instructions while it trains, type them below "
             "(one per line) and tick Change the previews of the run going now.", "note")
        if desc.edit_training else None,
        _row("SAMPLE_ENABLED", "Make previews while training", "check"),
        _row("_prompts_note", " ".join(t for k, t in desc.samples_text if k == "prompts"), "note")
        if any(k == "prompts" for k, _t in desc.samples_text) else None,
        _row("SAMPLE_PROMPT", "Prompts", "text", default=desc.sample_prompt or None, hint=" ".join(
             t for k, t in desc.samples_text if k == "banner") or
             "One prompt per line: Enter starts the next. Write them as your captions are, trigger word first, e.g. "
             "\"zwxem, a woman in a garden\" or \"zwxem, a harbour at dusk\". The numbers are the gallery's Prompt "
             "1, 2, 3…",
             lines=True),
        _row("_overtrain_tip", "Tip - catch overtraining early: give one prompt something your dataset doesn't have, "
             "a pose, angle, setting or subject none of your pictures show. Early epochs follow it. When the LoRA starts to "
             "overtrain, that's the first thing to fail, as the previews drift back to your photos. The last epoch that "
             "still follows it is a good one to keep.", "note") if pictures else None,
        # the switches side by side (the grid tiles ticks), then the Turbo box, then the boxes they change
        _row("__SAMPLE_LIVE", "Change the previews of the run going now", "check",
             "On by default. While it's ticked, a change you make here (prompts, seed, size, steps, CFG, Turbo "
             "strength) goes to the run training now, from its next preview; new prompts are encoded again then. "
             "Untick to put the run back on its own.") if pictures else None,
        _row("SAMPLE_AT_FIRST", "Preview before training starts", "check"),
        *([_row("SAMPLE_CHECKPOINT", desc.preview_checkpoint_tick, "check", default=desc.preview_checkpoint_on,
                **({"follow": {"on": {"SAMPLE_STEPS": str(desc.preview_checkpoint_sampling.steps),
                                      "SAMPLE_CFG_SCALE": f"{desc.preview_checkpoint_sampling.cfg:g}"},
                               "off": {"SAMPLE_STEPS": str(desc.preview_steps),
                                       "SAMPLE_CFG_SCALE": f"{desc.preview_cfg:g}"}}}
                   if desc.preview_checkpoint_sampling else {}))] if desc.train_preview_checkpoint else []),
        _row("FAMILY_PREVIEW_LOOK", desc.preview_look_tick, "check", desc.preview_look_hint, default=False,
             **({"bold": desc.preview_look_bold} if desc.preview_look_bold else {}))
        if desc.preview_look() is not None else None,
        _row("SAMPLE_STRENGTHS", "Also preview at strength " + ", ".join(f"{x:g}" for x in desc.preview_strengths),
             "check", "Each preview is rendered again at that strength (same prompt and seed), beside the "
             "full-strength one: how the LoRA will look or sound the way you'll use it.", default=True)
        if desc.preview_strengths else None,
        *turbo_rows,
        _row("SAMPLE_REFERENCE", "Reference", "path",
             f"Optional - fed through {desc.display_name}'s text encoder vision path so previews become visually "
             "aware of it ('prompt from a picture', not a pixel edit). Leave empty for normal previews."
             if desc.reference_kind == "vision" else
             "Optional - every preview edits this photo with its prompt, at the photo's shape. Leave empty for normal "
             "previews. An Edit LoRA run previews on the photo set on its own card instead.",
             upload=True, start="data") if desc.reference_kind else None,
        # a picture size only for a model that makes pictures (ACE-Step's previews are songs: the launcher then
        # uses the description's own preview size)
        _row("SAMPLE_WIDTH", "Width", "choice", choices=SIZES, default=str(desc.preview_width)) if pictures else None,
        _row("SAMPLE_HEIGHT", "Height", "choice", choices=SIZES, default=str(desc.preview_height)) if pictures else None,
        _row("SAMPLE_STEPS", "Steps", default=str(desc.preview_steps)),
        _row("SAMPLE_SEED", "Seed", hint="0 = random"),
        *by_section("samples"),
        # one control: how often previews run, and (ticked) another rate from an epoch on - its parts are their
        # own values (SAMPLE_LATER / SAMPLE_UNTIL_EPOCH / SAMPLE_EVERY_AFTER), greyed and hidden one by one; ticking
        # it fills in every 5, then from epoch 10 every 1 (few previews early, every epoch once it gets close)
        _row("SAMPLE_EVERY_N_EPOCHS", "How often to preview", "cadence",
             "Ticked: every N epochs up to that epoch, then the second rate (whole multiples). Every 5, then from "
             "epoch 10 every 1, previews epochs 5, 10, 11, 12, 13 and on.",
             default=str(getattr(desc, "preview_every", 1) or 1), tick_fills={"SAMPLE_EVERY_N_EPOCHS": "5"},
             parts=[{"key": "SAMPLE_LATER", "default": False}, {"key": "SAMPLE_UNTIL_EPOCH", "default": "10"},
                    {"key": "SAMPLE_EVERY_AFTER", "default": "1"}]),
        _row("CACHE_SAMPLE_MODEL", "Cache sample model in RAM", "choice", choices=["auto", "on", "off"],
             default="auto") if desc.train_preview_checkpoint else None,
    ])
    if not desc.samples_cfg_free:
        add("samples_adv", "Guidance", "samples", [
            _row("SAMPLE_CFG_SCALE", "CFG Scale", hint=desc.cfg_note(), default=str(desc.preview_cfg)),
            _row("SAMPLE_NEGATIVE", "Negative Prompt", "text", default=neg or "") if neg is not None else None,
        ])
    # the Training tab's order: what every run sets, then (after the Extras divider) the features you choose to use
    order = {sid: i for i, sid in enumerate(TRAINING_ORDER)}
    sections.sort(key=lambda sec: (sec["tab"] != "training", order.get(sec["id"], len(order))))
    first = next((sec for sec in sections if sec["tab"] == "training" and sec["id"] not in order), None)
    if first:
        first["extras"] = True               # the divider goes above it
    return {"sections": sections, "presets": [{"name": n, "values": dict(v)} for n, v in (desc.presets or ())]}


# what a preset never carries (the desktop's): the Samples tab and the Turbo boxes are the model's own, a resume path
# is one run's
NOT_IN_PRESETS = ("FAMILY_TURBO_STRENGTH", "FAMILY_TURBO_STEPS", "FAMILY_TURBO_PACE", "CACHE_SAMPLE_MODEL",
                  "FAMILY_PREVIEW_LOOK",
                  "RESUME_TRAINING", "CAPTION_TRIGGER_WORD")     # the trigger word is the training folder's
# boxes a preset can't put an unknown value in (the desktop's _STRICT_COMBO_KEYS): the current choice stays
STRICT_CHOICES = ("OPTIMIZER_TYPE", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX", "LR_SCHEDULER", "NETWORK_TYPE")
LEGACY_AREAS = {"All Layers": "Full Model", "Identity Blocks": "Identity",
                "Style+Composition Blocks": "Style+Composition", "Details Blocks": "Details"}


SLIDER_PREVIEW_EVERY = "5"           # a Slider preset's preview cadence (apply_preset)


def _not_in_preset(k):
    return k.startswith("SAMPLE_") or k in NOT_IN_PRESETS


def apply_preset(desc, values, preset, whole=False):
    """The form's values with a preset applied, as the desktop's _apply_preset_values does: the family's aliases
    mapped first, the kind of training from its switches, a model-card option never taken from a preset, nor the
    Samples tab or a resume path (whole: a run's own record, Samples included), a strict box only given a value it
    offers, and a Model Area's own timesteps unless the preset has some."""
    p = dict(preset)
    for new, old in dict(getattr(desc, "settings_aliases", None) or ()).items():    # {current key: old key}
        if old in p and new not in p:
            p[new] = p.pop(old)
    if "MINIMAX_MULTICONCEPT" in p and "FAMILY_MULTICONCEPT" not in p:
        p["FAMILY_MULTICONCEPT"] = p.pop("MINIMAX_MULTICONCEPT")
    if "MINIMAX_CONCEPT_DIRS" in p:                      # a list: subject 2, 3, 4
        dirs = p.pop("MINIMAX_CONCEPT_DIRS") or []
        dirs = [dirs] if isinstance(dirs, str) else list(dirs)
        for (d, _t, _c), f in zip(CONCEPT_ROWS, list(dirs) + [""] * len(CONCEPT_ROWS)):
            p[d] = str(f or "").strip()
        p.setdefault("FAMILY_CONCEPT_COUNT", CONCEPT_COUNTS[min(max(len(dirs), 1), len(CONCEPT_COUNTS)) - 1])
    if not whole:
        p = {k: v for k, v in p.items() if not _not_in_preset(k)}
        # the one Samples box a preset sets: a Slider preset previews its dial every 5 epochs, whatever the model's
        # own preview default (every epoch is too often for a slider, ACE-Step's 40 too rare)
        if str(p.get("FAMILY_SLIDER")).lower() in ("true", "1"):
            p["SAMPLE_EVERY_N_EPOCHS"] = SLIDER_PREVIEW_EVERY
    rows = {r["key"]: r for sec in form(desc)["sections"] for r in sec["rows"] if r}
    for k in STRICT_CHOICES:
        if k in p and k in rows and rows[k].get("choices"):
            opts, v = [str(c) for c in rows[k]["choices"]], str(p[k])
            v = v if v in opts else next((c for c in opts if c.split(" ")[0] == v.split(" ")[0]), None)
            if v is None:
                p.pop(k)                                 # not offered here: the current choice stays
            else:
                p[k] = v
    if "TARGET_LAYERS" in p and desc.train_areas:
        names = [a[0] for a in desc.train_areas]
        area = LEGACY_AREAS.get(str(p["TARGET_LAYERS"]), str(p["TARGET_LAYERS"]))
        p["TARGET_LAYERS"] = area if area in names + ["Custom"] else names[0]
        ts = rows.get("TARGET_LAYERS", {}).get("timesteps", {}).get(p["TARGET_LAYERS"])
        if ts:                                           # the area's window, unless the preset has its own
            p.setdefault("MIN_TIMESTEP", ts[0])
            p.setdefault("MAX_TIMESTEP", ts[1])
    model_keys = {o.key for o in desc.options if o.tab == "model"}
    out = {**values, **{k: v for k, v in p.items() if k not in model_keys}}
    labels = dict(kinds(desc))
    if any(k in p for k in ("FAMILY_EDIT", "FAMILY_SLIDER", "FAMILY_FT")):
        kind = ("finetune" if p.get("FAMILY_FT") else "slider" if p.get("FAMILY_SLIDER") else
                "edit" if p.get("FAMILY_EDIT") else "standard")
        out["FAMILY_KIND"] = labels.get(kind, labels["standard"])
    for o in desc.options:                       # an option's value under its old settings key
        if o.tab != "model" and o.key not in p and o.setting and o.setting in p:
            out[o.key] = p[o.setting]
    return out


def to_preset(desc, values):
    """The form's values as a saved preset, in the desktop's format (presets/<model>/<name>.json): its settings by
    the same keys, the kind of training as FAMILY_EDIT / FAMILY_SLIDER / FAMILY_FT, Multi Concept's folder as the
    desktop's list, and none of the Base Model card's options, the Samples tab or a resume path (as on the
    desktop)."""
    model_keys = {o.key for o in desc.options if o.tab == "model"}
    kind = _kind(desc, values)
    out = {k: v for k, v in values.items() if not k.startswith("__") and k not in model_keys and k != "FAMILY_KIND"
           and not _not_in_preset(k) and k not in {d for d, _t, _c in CONCEPT_ROWS}
           and not (k == "LORA_OUTPUT_DIR" and not str(v or "").strip())}      # an empty folder = the default
    out.update(FAMILY_EDIT=kind == "edit", FAMILY_SLIDER=kind == "slider", FAMILY_FT=kind == "finetune")
    if desc.multi_concept:
        out["MINIMAX_CONCEPT_DIRS"] = [str(values.get(d) or "").strip()
                                       for d, _t, _c in CONCEPT_ROWS[:concept_count(values) - 1]]
    return out


# subject 2, 3, 4: its folder, and its DOP word to remove / replacement
CONCEPT_ROWS = (("FAMILY_CONCEPT_DIR", "FAMILY_DOP_TRIGGER2", "FAMILY_DOP_CLASS2"),
                ("FAMILY_CONCEPT_DIR3", "FAMILY_DOP_TRIGGER3", "FAMILY_DOP_CLASS3"),
                ("FAMILY_CONCEPT_DIR4", "FAMILY_DOP_TRIGGER4", "FAMILY_DOP_CLASS4"))
DOP_EXTRA_ROWS = {k for _d, t, c in CONCEPT_ROWS for k in (t, c)}


def concept_count(values):
    """Subjects in a Multi Concept run, the training folder included (2-4)."""
    v = str(values.get("FAMILY_CONCEPT_COUNT") or CONCEPT_COUNTS[0])
    return next((i + 2 for i, c in enumerate(CONCEPT_COUNTS) if c == v or c.split(" ")[0] == v.split(" ")[0]), 2)


def _kind(desc, values):
    label = values.get("FAMILY_KIND", "")
    return next((k for k, lab in kinds(desc) if lab == label or k == label), "standard")


SUBFOLDERS_HINT = ("Off: only the pictures directly in the training folder are used.\nOn: the pictures in its "
                   "sub-folders are trained on too, at any depth, each sub-folder as its own set with its own captions "
                   "and cache, so a big dataset can stay organised. Every picture trains once an epoch, wherever it "
                   "is. Image Prep and Captions work on one folder at a time: to caption or mask a sub-folder, choose "
                   "it as the training folder on the Start tab.")


def subfolders_note(desc, values, image_folder):
    """(the tick's line, its Show all folders / Hide the folders button or None): what it will train and which
    folders still need captions (on), or what it is leaving out (off)."""
    if not image_folder or not os.path.isdir(image_folder):
        return "", None
    # the form's values name the kind of training (a label), not FAMILY_FT: a fine-tune's regularisation folder
    reg = _str(values.get("FAMILY_FT_REG_DIR")) if desc.finetune and _kind(desc, values) == "finetune" else ""
    subs = launch.subfolders(image_folder, (_str(values.get("LORA_OUTPUT_DIR")), reg))
    if not subs:
        return ("This training folder has no sub-folders with pictures." if values.get("TRAIN_SUBFOLDERS") else ""), None
    ext = _str(values.get("DATASET_CAPTION_EXT")) or ".txt"
    counts = [(rel, launch._media_count(f), _uncaptioned(f, ext)) for rel, f in subs]
    more = sum(n for _r, n, _u in counts)
    if not values.get("TRAIN_SUBFOLDERS"):
        return (f"{len(subs)} sub-folder{'s' if len(subs) != 1 else ''} with {more:,} more picture"
                f"{'s' if more != 1 else ''} {'is' if more == 1 else 'are'} not trained on. Tick this to include "
                f"them."), None
    own = launch._media_count(image_folder)
    rows = ([("the training folder", own, _uncaptioned(image_folder, ext))] if own else []) + counts
    line = f"Training on {len(rows)} folders, {own + more:,} pictures."
    bare = [(rel, u) for rel, _n, u in rows if u]
    if bare:      # Start refuses a run with uncaptioned pictures: these come first
        line += (" Not captioned yet, so Start will refuse them: "
                 + " · ".join(f"{rel} ({u:,})" for rel, u in bare[:8])
                 + (f" · and {len(bare) - 8} more folders" if len(bare) > 8 else "") + ".")
    show = bool(values.get("_SUBFOLDER_LIST"))
    if show:
        line += " All folders: " + " · ".join(f"{rel} {n:,}" for rel, n, _u in rows) + "."
    return line, {"label": "Hide the folders" if show else "Show all folders", "values": {"_SUBFOLDER_LIST": not show}}


def _str(x):
    return "" if x is None else str(x).strip()


def _uncaptioned(folder, ext):
    """Training items in a folder with no caption beside them."""
    from fizgig.dataset.image_dataset import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from fizgig.dataset.masks import is_mask_file
    exts = {e.lower() for e in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    caps = {os.path.splitext(n)[0] for n in names if n.lower().endswith(ext.lower())}
    return sum(1 for n in names if os.path.splitext(n)[1].lower() in exts and not is_mask_file(n)
               and os.path.splitext(n)[0] not in caps)


def state(desc, values, image_folder="", captioner="", speed_file=False, checkpoint_file=False, base_file="",
          checkpoint_path="", look_file=False, look_fetch=("", None)):
    """What the current values hide, grey out or note (see the module docstring), and force: values the page sets
    (as the desktop unticks what automagic3 owns). captioner: the shared captioner's file when it is set and on disk
    (auto-recaption needs it); speed_file: the preview Turbo LoRA is set and on disk; checkpoint_file: the preview
    checkpoint (Klein's Distilled) is."""
    hidden, disabled, notes, force, actions = set(), set(), {}, {}, {}
    disabled_choices = {}                         # {key: {choice: why}}: single choices greyed in a row of pills
    poll = False                                  # something is under way (a download): the page asks again soon
    ck_turbo = False                              # previews on a preview checkpoint that takes the Turbo LoRA
    hidden |= ({"TURBO_MISSING"} if speed_file else
               {"FAMILY_TURBO_STRENGTH", "FAMILY_TURBO_STEPS", "FAMILY_TURBO_PACE"})
    if speed_file and desc.samples_turbo_pace:     # H3 with its Turbo on renders at the Turbo row's steps
        try:
            pace = float(str(values.get("FAMILY_TURBO_PACE", "") or round(100 * desc.preview_speed_defaults()[1])))
        except ValueError:
            pace = 0.0
        if pace > 0:
            hidden.add("SAMPLE_STEPS")
    if not captioner:
        hidden |= {"KREA2_AUTO_RECAPTION", "CAPTION_TRIGGER_WORD"}
    if not captioner or not values.get("KREA2_AUTO_RECAPTION"):
        hidden |= {"RECAPTION_FIRST", "RECAPTION_SECOND"}
    kind = _kind(desc, values)
    # Also train on its sub-folders: a plain LoRA or a fine-tune (Subjects 2-4, DOP and Isolation keep one folder per
    # subject; an edit or a slider pairs files)
    if kind in ("edit", "slider") or values.get("FAMILY_MULTICONCEPT") or values.get("FAMILY_DOP")             or values.get("FAMILY_MULTI_PERSON"):
        hidden.add("TRAIN_SUBFOLDERS")
    else:
        notes["TRAIN_SUBFOLDERS"], act = subfolders_note(desc, values, image_folder)
        if act:
            actions["TRAIN_SUBFOLDERS"] = act
    if desc.masked_training:
        prompt_slider = kind == "slider" and launch.slider_on(desc, {"FAMILY_SLIDER": True, "FAMILY_SLIDER_SOURCE": (
            "prompts" if values.get("FAMILY_SLIDER_SOURCE") == "Prompts" else "pairs")}, "prompts")
        media = launch.dataset_media(image_folder) if image_folder else {"photo"}
        if prompt_slider or (media and "photo" not in media):
            hidden.add("masks")             # a prompt slider makes its own pictures; a folder of clips has none to mask
        elif not values.get("MASKED_TRAINING"):
            hidden |= {"MASK_FLOOR", "MASK_PRIOR"}
        else:
            note = _mask_note(desc, values, image_folder)
            if note:
                notes["MASKED_TRAINING"] = note
            if kind in ("slider", "finetune"):      # it switches the LoRA off for a step: a slider or a fine-tune has none
                disabled.add("MASK_PRIOR")
                notes["MASK_PRIOR"] = "For a LoRA: a slider or a fine-tune trains without it."
    lokr = str(values.get("NETWORK_TYPE", "")).startswith("LoKR") and kind not in ("slider", "finetune")
    hidden |= {"NETWORK_DIM", "NETWORK_ALPHA"} if lokr else {"LOKR_FACTOR"}
    if lokr:
        notes["NETWORK_TYPE"] = "LoKR selected: set your learning rate to 5e-5 - or with Adaptive LR, set Max 1e-4 " \
                                "and Min 5e-5."
    if not values.get("ADAPTIVE_LR") or kind in ("slider", "finetune"):   # these train at the Learning Rate box
        hidden |= {"ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"}
    else:
        disabled.add("LEARNING_RATE")
    if kind == "finetune":       # a fine-tune trains at its own fixed rate: no Adaptive LR, on any model
        hidden.add("ADAPTIVE_LR")
    if str(values.get("OPTIMIZER_TYPE", "")) == "automagic3":
        disabled |= {"ADAPTIVE_LR", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK"}
        force.update({k: False for k in ("ADAPTIVE_LR", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK") if values.get(k)})
        if values.get("ADAPTIVE_LR"):            # turned off now: as it shows with Adaptive LR off
            disabled.discard("LEARNING_RATE")
            hidden |= {"ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"}
        notes["OPTIMIZER_TYPE"] = "automagic3 sets its own learning rate: Adaptive LR, per-image LR and the look " \
                                  "warm-up are off."
    hidden |= {s for s, k in (("edit", "edit"), ("slider", "slider"), ("finetune", "finetune")) if k != kind}
    if kind == "finetune":
        disabled |= {"MAX_TRAIN_EPOCHS", "SAVE_EVERY_N_EPOCHS"}
        hidden |= {"CONTEXT_LORA_PATH", "CONTEXT_LORA_STRENGTH"}
    if kind != "standard":
        hidden.add("FAMILY_FAST_ID")
    if not desc.subject_isolation:                    # a family without it (a music model): a normal LoRA, always
        hidden |= {"FAMILY_DOP", "FAMILY_MULTI_PERSON", "FAMILY_PEOPLE_MODE"}
        force.update({k: False for k in ("FAMILY_DOP", "FAMILY_MULTI_PERSON") if values.get(k)})
        values = {**values, "FAMILY_DOP": False, "FAMILY_MULTI_PERSON": False}
    if kind not in ("standard", "finetune"):          # DOP keeps a subject apart: a LoRA's or a fine-tune's job
        hidden.add("FAMILY_DOP")
    # Fizgig LoRA Isolation needs two or more subjects (its recipe teaches a face to belong to one trigger, against
    # a second one): with one, Off and Leave the rest of the model alone stay, Isolation is greyed, and a saved
    # one-subject form on Isolation moves to Leave the rest of the model alone (the nearest: it keeps the DOP pass)
    one_sub = kind == "standard" and (concept_count(values) if values.get("FAMILY_MULTICONCEPT") else 1) == 1
    if one_sub:
        disabled_choices["FAMILY_PEOPLE_MODE"] = {PEOPLE_MODES[2]: "Needs two or more subjects: add one under Who or "
                                                                   "what is this LoRA of?"}
        if values.get("FAMILY_MULTI_PERSON"):
            force.update(FAMILY_MULTI_PERSON=False, FAMILY_DOP=True)
            values = {**values, "FAMILY_MULTI_PERSON": False, "FAMILY_DOP": True}
            notes["FAMILY_PEOPLE_MODE"] = (f"Moved to {PEOPLE_MODES[1]}: {PEOPLE_MODES[2]} needs two or more "
                                           "subjects.")
        else:                           # the greyed pill's reason, where it's seen (no hover on a phone)
            notes["FAMILY_PEOPLE_MODE"] = (f"{PEOPLE_MODES[2]} is greyed out because it needs two or more "
                                           "subjects. Pick 2 subjects above to use it.")
    if kind != "standard":                            # multi-person mode: a LoRA (its LR boost is per step)
        hidden.add("FAMILY_MULTI_PERSON")
    elif values.get("FAMILY_MULTI_PERSON"):           # it runs on DOP
        force["FAMILY_DOP"] = True
        disabled.add("FAMILY_DOP")
        if desc.loss_watch:             # a new AI caption would be encoded with the trigger unfused
            disabled.add("KREA2_AUTO_RECAPTION")
            force["KREA2_AUTO_RECAPTION"] = False
            notes["KREA2_AUTO_RECAPTION"] = "Off with Fizgig LoRA Isolation (Subjects card)."
            for k in ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK"):   # side-by-sides: hard on purpose
                disabled.add(k)
                force[k] = False
                notes[k] = "Off with Fizgig LoRA Isolation (Subjects card)."
        for o in desc.options:          # a family check that can't run beside it (H3 reference distillation)
            if getattr(o, "isolation_off", False):
                disabled.add(o.key)
                if values.get(o.key):
                    force[o.key] = False
                notes[o.key] = "Off with Fizgig LoRA Isolation (Subjects card)."
        values = {**values, "FAMILY_DOP": True}
        notes["SAMPLE_PROMPT"] =("Write prompts as your captions are (\"zwxem, a woman at a cafe\"): previews read "
                                  "them the way training reads the captions.")
    if kind in ("edit", "slider"):      # the first folder has a partner folder (originals, the -1 end): one subject
        hidden.add("subjects")
    n_sub = concept_count(values) if kind not in ("edit", "slider") and values.get("FAMILY_MULTICONCEPT") else 1
    # the card's two choices follow the settings under them (a preset, a saved run, an older setup)
    want = SUBJECT_COUNTS[n_sub - 1] if desc.multi_concept else SUBJECT_COUNTS[0]
    if str(values.get("FAMILY_SUBJECTS") or "") != want:
        force["FAMILY_SUBJECTS"] = want
    mode = PEOPLE_MODES[2 if values.get("FAMILY_MULTI_PERSON") and kind == "standard" else
                        1 if values.get("FAMILY_DOP") else 0]
    if str(values.get("FAMILY_PEOPLE_MODE") or "") != mode:
        force["FAMILY_PEOPLE_MODE"] = mode
    if kind == "finetune" and values.get("FAMILY_MULTI_PERSON"):
        notes["FAMILY_PEOPLE_MODE"] = "A fine-tune trains as On - only where you use its trigger word; the " \
                                      "group-picture option is for LoRAs."
    hidden.add("FAMILY_CONCEPT_COUNT")
    if n_sub == 1:
        hidden.add("FAMILY_CONCEPT_BALANCE")
    notes["_subject1_folder"] = image_folder or "Not chosen yet - pick it on the Start tab"
    dop = bool(values.get("FAMILY_DOP")) and kind in ("standard", "finetune")
    if not dop:
        hidden.add("FAMILY_SUBJECT_FIX")
    if not (kind == "standard" and values.get("FAMILY_MULTI_PERSON")):
        hidden |= {"FAMILY_MP_MANUAL", "FAMILY_MP_FACECROPS_FIRST"}
        hidden |= set(KIND_ROWS)            # what a subject is only changes the group-picture recipe
    if not (dop and values.get("FAMILY_SUBJECT_FIX")):
        hidden |= {"FAMILY_DOP_TRIGGER", "FAMILY_DOP_CLASS"} | DOP_EXTRA_ROWS
    if not (kind == "standard" and values.get("FAMILY_MULTI_PERSON")):   # "what they are" is the joined word's
        hidden |= {"FAMILY_DOP_CLASS"} | {c for _d, _t, c in CONCEPT_ROWS}
    for i, (d, t, c) in enumerate(CONCEPT_ROWS, start=2):     # subject i's folder, line and words, up to the count
        if i > n_sub:
            hidden |= {d, t, c, f"_subject{i}_as", KIND_ROWS[i - 1]}
    
    kn = _kind_notes(desc)
    if len(kinds(desc)) > 1:
        notes["FAMILY_KIND"] = kn.get(kind, ("", ""))[1] or KIND_HINTS.get(kind, "")
    hint = getattr(desc, "long_run_hint", ()) or ()
    try:
        long_run = bool(hint) and kind == "standard" and int(float(values.get("MAX_TRAIN_EPOCHS") or 0)) >= hint[0]
    except (TypeError, ValueError):
        long_run = False
    if long_run:
        notes["MAX_TRAIN_EPOCHS"] = f"💡 Hint: {hint[1]}. {hint[2]}"
    small = getattr(desc, "small_dataset_hint", ()) or ()
    n_pics = _picture_count(image_folder) if small and kind == "standard" and image_folder else None
    if n_pics and n_pics < small[0]:            # a small dataset on a family that converges slowly on one
        tip = "💡 Hint: " + small[1].format(n=n_pics)
        notes["MAX_TRAIN_EPOCHS"] = notes["MAX_TRAIN_EPOCHS"] + " " + tip if notes.get("MAX_TRAIN_EPOCHS") else tip
    lr_note = _lr_hint(desc, values, kind, image_folder)
    if lr_note:
        notes[lr_note[0]] = lr_note[1]
        actions[lr_note[0]] = ({"label": "Use 1e-4 / 2e-4", "values": {"ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "2e-4"}}
                               if values.get("ADAPTIVE_LR") else {"label": "Use 1e-4", "values": {"LEARNING_RATE": "1e-4"}})
    if kind in ("edit", "slider"):
        hidden.add("_overtrain_tip")
    if kind != "edit":
        hidden.add("_edit_samples_note")
    ts = _timestep_note(values)
    if ts:
        notes["MAX_TIMESTEP"] = ts
    if not values.get("SAMPLE_LATER"):        # Then preview less often: its two boxes wait for the tick
        disabled |= {"SAMPLE_UNTIL_EPOCH", "SAMPLE_EVERY_AFTER"}
    if not values.get("SAVE_LATER"):
        disabled |= {"SAVE_UNTIL_EPOCH", "SAVE_EVERY_AFTER"}
    if kind == "finetune":                     # a fine-tune previews and saves at whole rotations
        hidden |= {"SAMPLE_LATER", "SAMPLE_UNTIL_EPOCH", "SAMPLE_EVERY_AFTER", "SAVE_LATER", "SAVE_UNTIL_EPOCH",
                   "SAVE_EVERY_AFTER"}
    if kind in ("edit", "slider"):
        hidden.add("_prompts_note")     # the family's advice on writing prompts: these runs don't use them
    if kind == "edit":
        notes["SAMPLE_PROMPT"] = ("An Edit LoRA run doesn't use these prompts or the reference: its previews apply "
                                  "the edit instruction to the test photo on the Training tab's Edit pairs card.")
    elif kind == "slider":
        notes["SAMPLE_PROMPT"] = ("A slider run doesn't use these prompts: its previews are the slider's own -1 / 0 / "
                                  "+1 strip.")
    srcs = slider_sources(desc)
    pairs = (values.get("FAMILY_SLIDER_SOURCE") or SLIDER_SOURCES[srcs[0]]) != "Prompts" if len(srcs) > 1 \
        else srcs[0] == "pairs"
    hidden |= ({"FAMILY_SLIDER_BASE", "FAMILY_SLIDER_POS", "FAMILY_SLIDER_NEG", "FAMILY_SLIDER_GUIDANCE",
                "FAMILY_SLIDER_ANCHORS", "FAMILY_SLIDER_ANCHOR_WEIGHT", "_slider_help_prompts"} if pairs else {"_slider_plus_dir", "FAMILY_SLIDER_DIR", "FAMILY_SLIDER_CAPTION",
                                                       "_slider_help_pairs"})
    if pairs:
        notes["_slider_plus_dir"] = image_folder or "Not chosen yet - pick it on the Start tab"
    notes["_edit_edited_dir"] = image_folder or "Not chosen yet - pick it on the Start tab"
    media = launch.dataset_media(image_folder) if image_folder else set()
    if "clip" not in media:                          # clips' own size only when the folder holds clips
        hidden.add("CLIP_MEGAPIXELS")
    # what the run ignores (launch.plan): a slider or a fine-tune trains its own blocks (Model Area), and a
    # fine-tune every noise level (the timestep range)
    if kind in ("slider", "finetune"):
        hidden |= {"TARGET_LAYERS", "TRAINING_BLOCKS", "NETWORK_TYPE", "LOKR_FACTOR"}   # a slider is always a LoRA
    if kind == "slider":         # the trainer turns these off for a slider: a slider step has two targets, not one
        hidden |= {"ADAPTIVE_LR", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX", "FAMILY_EMA", "KREA2_LOSS_WATCH",
                   "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK", "KREA2_AUTO_RECAPTION", "CAPTION_TRIGGER_WORD",
                   "RECAPTION_FIRST", "RECAPTION_SECOND"}
    if kind == "finetune":       # the model itself trains: no LoRA (rank, alpha), every noise level, its own continue
        hidden |= {"MIN_TIMESTEP", "MAX_TIMESTEP", "NETWORK_DIM", "NETWORK_ALPHA", "RESUME_TRAINING"}
    if str(values.get("TARGET_LAYERS") or "") != "Custom":   # the block picker is Model Area's Custom
        hidden.add("TRAINING_BLOCKS")
    vals = _option_values(desc, values)
    for o in desc.options:
        if o.kind == "fixed" or o.tab == "model":
            continue
        if not launch.option_applies(o, media, vals, desc.options, finetune=kind == "finetune"):
            (disabled if o.always_shown else hidden).add(o.key)
            if o.always_shown and o.unmet_notes:      # the note for the choice that owns it now (the desktop's)
                key = o.requires.partition("=")[0]
                owner = next((x for x in desc.options if x.key == key), None)
                cur = owner.pick(vals.get(key, "")) if owner is not None else ""
                notes[o.key] = next((n for n in o.unmet_notes if n[0] == cur), o.unmet_notes[0])[1]
    try:                                         # CFG 1 or less renders without the negative: no box for it
        if not desc.samples_cfg_free and float(str(values.get("SAMPLE_CFG_SCALE") or desc.preview_cfg)) <= 1.0:
            hidden.add("SAMPLE_NEGATIVE")
    except ValueError:
        pass
    shown, labels = {}, {}
    if desc.edit_training and getattr(desc, "edit_references", 1) > 1:
        if kind == "edit" and str(values.get("FAMILY_EDIT_PICTURES") or "") == launch.EDIT_WITH_EXTRAS:
            labels = {"_edit_edited_dir": "3. Edited folder (after editing)", "FAMILY_EDIT_CAPTION": "4. Edit instruction",
                      "FAMILY_EDIT_REF": "5. Test photo (optional)"}
            notes["_edit_extras_found"] = _extras_found(desc, values, image_folder)
        else:
            hidden |= {"FAMILY_EDIT_EXTRAS_DIR", "_edit_extras_found"}
    for key, old, new, *when in desc.value_moves:     # a changed default, moved once in a saved form
        mark = f"_moved_{key}"
        if not values.get(mark):
            force[mark] = True
            try:                                  # a move tied to another setting: only while that one is on (> 0)
                on = not when or float(str(values.get(when[0]) or 0)) > 0
            except ValueError:
                on = False
            if on and str(values.get(key, "")).strip() == str(old):
                force[key] = new
    if desc.train_preview_checkpoint and desc.standard_base:
        # another Base than the released one (Kromoss, a fine-tune, SFT): previews on it by default, the model the LoRA
        # will be used on. The tick moves once per Base file, so a choice made after that stays
        from fizgig.families.base_check import is_standard_base
        std, mark = is_standard_base(desc, base_file), str(values.get("_SAMPLE_CHECKPOINT_BASE") or "")
        if std is False:
            notes["SAMPLE_CHECKPOINT"] = (f"Your Base DiT isn't {desc.display_name}'s own Base, so previews render on it, "
                                        "the model you'll use the LoRA on. Tick this to preview on the other model instead.")
            if mark != base_file:
                force["SAMPLE_CHECKPOINT"], force["_SAMPLE_CHECKPOINT_BASE"] = False, base_file
        elif std and mark:            # back on the released Base: its default again
            force["SAMPLE_CHECKPOINT"], force["_SAMPLE_CHECKPOINT_BASE"] = desc.preview_checkpoint_on, ""
    if desc.train_preview_checkpoint and checkpoint_file and values.get("SAMPLE_CHECKPOINT", True):
        # the Distilled: Steps and CFG switch to its own values and stay editable (the tick's follow), as the Turbo
        # LoRA's do; CFG 1 hides the negative (above)
        if desc.preview_checkpoint_sampling is None:
            disabled |= {"SAMPLE_STEPS", "SAMPLE_CFG_SCALE", "SAMPLE_NEGATIVE"}
        from fizgig.families.base_check import checkpoint_takes_speed
        if checkpoint_takes_speed(desc, checkpoint_path):
            ck_turbo = True                       # Kromoss as the preview model: the Turbo LoRA joins it
        elif speed_file and desc.preview_speed() is not None and not desc.samples_turbo_pace:
            # previews on this checkpoint never take the speed LoRA (it joins previews on the training model)
            disabled.add("FAMILY_TURBO_STRENGTH")
            notes["FAMILY_TURBO_STRENGTH"] = (
                f"Greyed out: the {desc.preview_speed().name} doesn't work with this preview model. Set Kromoss as the "
                "preview model in Preferences to use it." if desc.checkpoint_speed_except else
                f"The {desc.preview_speed().name} applies to previews on the training model: untick "
                f"\"{desc.preview_checkpoint_tick.split(' (')[0]}\" to use it.")
    if desc.preview_look() is not None:
        # the preview look LoRA (MOSS-Image's SFT look, Z-Image Base's Turbo look): on previews of the training model
        # only; its file comes down the first time it's ticked (look_fetch: that download's state)
        look_name = desc.preview_look().label.split(" (")[0]
        if desc.train_preview_checkpoint and checkpoint_file and values.get("SAMPLE_CHECKPOINT", True):
            disabled.add("FAMILY_PREVIEW_LOOK")
            notes["FAMILY_PREVIEW_LOOK"] = (
                f"Greyed out: previews render on the preview model, which has its own look. Untick "
                f"\"{desc.preview_checkpoint_tick.split(' (')[0]}\" to preview on your Base with this LoRA.")
        elif values.get("FAMILY_PREVIEW_LOOK"):
            state, gb = look_fetch
            if state == "running":
                got = f"{gb[0] * 1024:.0f} MB" if gb and gb[0] < 1 else f"{gb[0]:.1f} GB" if gb else ""
                notes["FAMILY_PREVIEW_LOOK"] = (f"Downloading the {look_name}: {got} of {gb[1]:.1f} GB. "
                                                if gb else f"Downloading the {look_name}. ") + \
                    "A run that's going picks it up at its next preview."
                poll = True
            elif state == "failed" and not look_file:
                notes["FAMILY_PREVIEW_LOOK"] = ("The download failed: check you're online, then tick this again, "
                                                "or download it from Preferences.")
            elif state == "done":
                notes["FAMILY_PREVIEW_LOOK"] = "Downloaded: previews use it from the next one."
    return {"hidden": sorted(hidden), "disabled": sorted(disabled), "notes": notes, "kind": kind,
            "media": sorted(media), "turbo_file": bool(speed_file), "force": force, "actions": actions,
            "shown": shown, "checkpoint_file": bool(checkpoint_file), "labels": labels, "checkpoint_turbo": ck_turbo,
            "disabled_choices": disabled_choices, "poll": poll}


def _extras_found(desc, values, image_folder):
    """The Extra pictures folder's readout: how many edited photos have extra pictures, and how many each."""
    xdir = str(values.get("FAMILY_EDIT_EXTRAS_DIR") or "").strip()
    if not xdir or not os.path.isdir(xdir) or not image_folder or not os.path.isdir(image_folder):
        return ""
    try:
        found, orphans, misnamed = launch.edit_extras(image_folder, xdir,
                                                      test_photo=str(values.get("FAMILY_EDIT_REF") or ""))
        total = sum(1 for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in launch.PAIR_EXTS
                    and not launch.is_mask_file(f))
    except OSError:
        return ""
    counts = sorted(len(v) for v in found.values())
    line = (f"Found: {total} edited photos, {len(found)} with extra pictures"
            + (f" ({counts[0]}-{counts[-1]} each)" if counts and counts[0] != counts[-1] else
               f" ({counts[0]} each)" if counts else ""))
    loose = len(orphans) + len(misnamed)
    return line + (f". {loose} picture(s) here match no photo's name." if loose else ".")


def _timestep_note(values):
    """What a Min / Max timestep range trains (the desktop's readout, the right way round: in Fizgig's flow matching 0
    is the clean picture and 1000 pure noise)."""
    lo, hi = str(values.get("MIN_TIMESTEP") or "").strip(), str(values.get("MAX_TIMESTEP") or "").strip()
    if not lo and not hi:
        return ""
    try:
        a, b = int(float(lo)) if lo else 0, int(float(hi)) if hi else 1000
    except ValueError:
        return "Timesteps are whole numbers from 0 to 1000."
    if not 0 <= a < b <= 1000:
        return "Min must be below Max, both from 0 to 1000."
    if a == 0 and b >= 1000:
        return f"{a}-{b}: the whole range, every noise level."
    if b <= 400:
        return f"{a}-{b}: the clean end - style, texture and fine detail."
    if a >= 600:
        return f"{a}-{b}: the noisy end - composition, layout and pose."
    return f"{a}-{b}: part of the range."


_lr_counts = {}                     # folder -> (modified time, pictures)


def _mask_note(desc, values, image_folder):
    """The line under Masked training: how many pictures have a mask, and any mask that won't be used."""
    rep = launch.mask_report(desc, {**values, "image_folder": image_folder,
                                    "extra_folders": [values.get(d) for d, _t, _c in CONCEPT_ROWS[:concept_count(values) - 1]]
                                    if values.get("FAMILY_MULTICONCEPT") and _kind(desc, values) not in ("edit", "slider")
                                    else []})
    if not rep["total"]:
        return ""
    if not rep["with"] and not rep["bad"]:
        line = ("No masks found in the training folder yet. Save one beside a picture as photo-masklabel.png "
                "(for photo.jpg). Start won't run with this ticked and no masks.")
    else:
        line = f"Masks found for {rep['with']} of {rep['total']} pictures. The rest train as normal."
    if rep["bad"]:
        shown = rep["bad"][:5]
        line += (f"\n\n{len(rep['bad'])} mask(s) won't be used, so their pictures train without one:\n"
                 + "\n".join(shown) + (f"\n... and {len(rep['bad']) - 5} more" if len(rep["bad"]) > 5 else ""))
    return line


def _picture_count(image_folder):
    """The pictures in a training folder, cached until the folder changes; None when it can't be read."""
    try:
        mt = os.stat(image_folder).st_mtime
    except (OSError, TypeError):
        return None
    hit = _lr_counts.get(image_folder)
    if not hit or hit[0] != mt:
        from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
        exts = {e.lower() for e in IMAGE_EXTENSIONS}
        try:
            n = sum(1 for e in os.scandir(image_folder) if e.is_file() and os.path.splitext(e.name)[1].lower() in exts
                    and not launch.is_mask_file(e.name))
        except OSError:
            n = 0
        hit = _lr_counts[image_folder] = (mt, n)
    return hit[1]


def _lr_hint(desc, values, kind, image_folder):
    """The desktop's large-dataset hint: a standard LoRA whose learning rate (adaptive: Min LR) is at or above the
    family's lr_hint rate, on more steps an epoch than its count (one picture a step), moves a long way each epoch -
    the best point can fall between two saves. -> (the box it goes under, the text) or None."""
    hint = getattr(desc, "lr_hint", (2e-4, 125))
    if not hint or kind != "standard" or not image_folder:
        return None
    adaptive = bool(values.get("ADAPTIVE_LR"))
    key = "ADAPTIVE_LR_MIN" if adaptive else "LEARNING_RATE"
    raw = str(values.get(key) or "").split(" ")[0].strip()
    try:
        if float(raw) < hint[0] * 0.999:
            return None
    except ValueError:
        return None
    n = _picture_count(image_folder)
    if n is None or n <= hint[1]:
        return None
    if adaptive:
        return key, (f"💡 Hint: a large dataset. With {n} images, a Min LR of {raw} moves the LoRA a long way each "
                     "epoch, so the best point can fall between two saves. A Min LR of 1e-4 and a Max LR of 2e-4 "
                     "suit a dataset this size.")
    return key, (f"💡 Hint: a large dataset. With {n} images, a learning rate of {raw} moves the LoRA a long way each "
                 "epoch, so the best point can fall between two saves. 1e-4 suits a dataset this size.")


def _option_values(desc, values):
    """The family options' values as the launch plan reads them (a tick as "1" / "", the rest as text)."""
    out = {}
    for o in desc.options:
        if o.key in values:
            v = values[o.key]
            out[o.key] = ("1" if v else "") if o.kind == "check" else str(v)
    return out


def inputs(desc, values, *, models, python, repo_dir, image_folder, cache_root, samples_dir, inference_int8=False,
           captioner="", caption_overrides=None, caption_concept=""):
    """The launch plan's inputs from the form's values, as the desktop's _family_launch_inputs builds them."""
    v = {**DEFAULTS, **({"SAMPLE_PROMPT": desc.sample_prompt} if desc.sample_prompt else {}), **values}
    kind = _kind(desc, v)
    b = lambda k: bool(v.get(k))
    s = lambda k: str(v.get(k, "") if v.get(k) is not None else "").strip()
    d = {k: val for k, val in v.items() if not k.startswith("_") and k not in ("FAMILY_KIND",)}
    # Start's own conversions (lora_trainer_gui._start_training_launch); text that does not convert stays text, so
    # the plan reports it as a problem
    for key, cast in (("LEARNING_RATE", float), ("LORA_LR_RATIO", int), ("NETWORK_DIM", int),
                      ("NETWORK_ALPHA", float), ("LOKR_FACTOR", int), ("MAX_TRAIN_EPOCHS", int),
                      ("SAVE_EVERY_N_EPOCHS", int), ("SEED", int)):
        try:
            d[key] = cast(s(key) or (8 if key == "LOKR_FACTOR" else ""))
        except ValueError:
            if key == "LEARNING_RATE" and b("ADAPTIVE_LR") and desc.adaptive_lr:
                d[key] = 1e-4                    # ignored by the trainer under Adaptive LR (as Start does)
    d.update(
        FAMILY_EDIT=kind == "edit", FAMILY_SLIDER=kind == "slider", FAMILY_FT=kind == "finetune",
        FAMILY_SLIDER_SOURCE=(("prompts" if s("FAMILY_SLIDER_SOURCE") == "Prompts" else "pairs")
                              if len(slider_sources(desc)) > 1 else slider_sources(desc)[0]),
        ADAPTIVE_LR=b("ADAPTIVE_LR") and desc.adaptive_lr, SAVE_STATE=b("SAVE_STATE"),
        SAVE_STATE_ON_TRAIN_END=b("SAVE_STATE_ON_TRAIN_END"), FAMILY_FT_FUSED=b("FAMILY_FT_FUSED"),
        FAMILY_SLIDER_ULTRA=b("FAMILY_SLIDER_ULTRA"),
        FAMILY_FAST_ID=b("FAMILY_FAST_ID") and kind == "standard" and bool(desc.identity_blocks),
        **({"FAMILY_TRAIN_AREA": s("TARGET_LAYERS") or desc.train_areas[0][0],
            "FAMILY_TRAIN_BLOCKS": [k for k, on in (v.get("TRAINING_BLOCKS") or {}).items() if on]}
           if desc.train_areas else {}),
        FAMILY_TRAINING_ADAPTER=b("FAMILY_TRAINING_ADAPTER"),
        FAMILY_TURBO_STRENGTH=s("FAMILY_TURBO_STRENGTH"), FAMILY_TURBO_STEPS=s("FAMILY_TURBO_STEPS"),
        FAMILY_TURBO_PACE=s("FAMILY_TURBO_PACE"), FAMILY_PREVIEW_LOOK=b("FAMILY_PREVIEW_LOOK"),
        MASKED_TRAINING=b("MASKED_TRAINING") and desc.masked_training, MASK_FLOOR=s("MASK_FLOOR"),
        MASK_PRIOR=s("MASK_PRIOR"),
        python=python, repo_dir=repo_dir, models=models, image_folder=image_folder,
        caption_ext=s("DATASET_CAPTION_EXT") or ".txt", batch_size=1, megapixels=s("DATASET_MEGAPIXELS"),
        clip_megapixels=s("CLIP_MEGAPIXELS") if "clip" in desc.media else "",
        enable_bucket=b("ENABLE_BUCKET"), no_upscale=b("BUCKET_NO_UPSCALE"), cache_root=cache_root,
        blocks_swap=s("BLOCKS_SWAP"), enable_cache=b("ENABLE_CACHE"), resuming=bool(s("RESUME_TRAINING")),
        loss_watch={"detect": b("KREA2_LOSS_WATCH") and desc.loss_watch,
                    "per_image_lr": b("KREA2_PER_IMAGE_LR") and desc.loss_watch,
                    "warmup": b("KREA2_WARMUP_LOOK") and desc.loss_watch,
                    "recaption": b("KREA2_AUTO_RECAPTION") and desc.loss_watch},
        captioner=captioner, caption_trigger=s("CAPTION_TRIGGER_WORD"),
        caption_overrides=caption_overrides, caption_concept=caption_concept,
        recaption_tasks=(_recaption_key(s("RECAPTION_FIRST"), "training"),
                         _recaption_key(s("RECAPTION_SECOND"), "exhaustive")),
        samples={"enabled": b("SAMPLE_ENABLED"), "every": s("SAMPLE_EVERY_N_EPOCHS"), "later": b("SAMPLE_LATER"),
                 "until": s("SAMPLE_UNTIL_EPOCH"), "after": s("SAMPLE_EVERY_AFTER"), "width": s("SAMPLE_WIDTH"),
                 "height": s("SAMPLE_HEIGHT"), "steps": s("SAMPLE_STEPS"), "cfg": s("SAMPLE_CFG_SCALE") or
                 str(desc.preview_cfg), "negative": s("SAMPLE_NEGATIVE"), "seed": s("SAMPLE_SEED"),
                 "at_first": b("SAMPLE_AT_FIRST"), "prompts": s("SAMPLE_PROMPT").splitlines(), "reference": s("SAMPLE_REFERENCE"),
                 "strengths": list(desc.preview_strengths) if desc.preview_strengths and
                 v.get("SAMPLE_STRENGTHS", True) else [],
                 **({"checkpoint": b("SAMPLE_CHECKPOINT"), "checkpoint_cache": s("CACHE_SAMPLE_MODEL") or "auto",
                     "int8": bool(inference_int8)} if desc.train_preview_checkpoint else {})},
        samples_dir=samples_dir, edit_caption=s("FAMILY_EDIT_CAPTION"),
        multiconcept=b("FAMILY_MULTICONCEPT") and desc.multi_concept and kind not in ("edit", "slider"),
        # subject 2 on, in order; a blank folder stays in its place so the plan can name it
        extra_folders=([s(d) for d, _t, _c in CONCEPT_ROWS[:concept_count(v) - 1]]
                       if b("FAMILY_MULTICONCEPT") and desc.multi_concept and kind not in ("edit", "slider") else []),
        concept_balance=b("FAMILY_CONCEPT_BALANCE"),
        subfolders=b("TRAIN_SUBFOLDERS"),
        FAMILY_OPTIONS=_option_values(desc, v),
    )
    return d


# ---- MiniMax H3 RefMod (fizgig/families/refmod.py): the desktop's RefMod card, on H3's files ----------------------
def refmod_form(h3, ref_dit_set=False):
    """RefMod's form: H3's Training Base (Reference first when the reference DiT is set, as the desktop switches it),
    Output, the RefMod card in the desktop's order, the cache tick; a note where the Samples tab would be."""
    from fizgig.families import refmod as rm
    model = [_option_row(o) for o in h3.options if o.tab == "model"]
    for r in model:
        if ref_dit_set and r["key"] == "H3_TRAIN_BASE":
            ref = next((c for c in r["choices"] if "ref2va" in c.lower()), None)
            if ref:
                r["default"] = ref           # a RefMod rides the ref2va path at generation
    sections = [
        {"id": "model", "title": "Base Model", "tab": "training", "rows": model + [_row(
            "_model_note", "A RefMod runs on MiniMax H3's files (set in Preferences under MiniMax H3). Reference "
            "(ref2va) is the model a RefMod rides at generation.", "note")]},
        {"id": "output", "title": "Output", "tab": "training", "rows": [
            _row("LORA_OUTPUT_DIR", "Output Directory", "folder", "Where the mod is saved: <name>.safetensors. Point "
                 "it at ComfyUI/models/refmods and the mod is ready to load. Empty (or Reset to default) uses the LoRA "
                 "output folder in Preferences.", default=""),
            _row("LORA_NAME", "Name", default="my_refmod")]},
        {"id": "refmod", "title": "RefMod", "tab": "training", "rows": [
            _row("MINIMAX_REFMOD_GRID", "Grid", "choice", "", rm.REFMOD_GRID_OPTIONS, rm.REFMOD_GRID_OPTIONS[0]),
            _row("MINIMAX_REFMOD_REFS", "References", "combo", "", rm.REFMOD_REFS_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_REFS"]),
            _row("REFMOD_TOKENS", "", "note"),
            _row("MINIMAX_REFMOD_STEPS", "Steps", "combo", "", rm.REFMOD_STEP_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_STEPS"]),
            _row("DATASET_MEGAPIXELS", "Target MP", "choice", "", rm.REFMOD_MP_OPTIONS, "0.25"),
            _row("MINIMAX_REFMOD_CLIPS", "Clips", "choice",
                 "How a clip in the dataset enters the mod. Sharpest still: one frame, the face — the identity choice. "
                 "Motion: every latent frame of the clip, the node pack's video reference (a dance, a camera move); "
                 "tokens are per frame, so a clip costs its length. Prepare clips with Gizmo first — cut to H3's "
                 "frame grid at the right size, they work best.", rm.REFMOD_CLIPS_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_CLIPS"]),
            _row("MINIMAX_REFMOD_DESC", "Description (mod hint)",
                 hint="A few words saying what the mod is — 'a ginger woman with messy hair', 'a 1970s film look'. "
                      "Stored in the file; RefMod Toolkit's '+ mod hints' and the pack's loader put 'concept: "
                      "description' into the prompt so the model knows what it is looking at.", default=""),
            _row("MINIMAX_REFMOD_CONCEPT", "Concept", "choice",
                 "What the mod is, in the pack's terms: identity (a person), style (a look), pose_motion (a dance, a "
                 "camera move), clothing, background, generic.", rm.REFMOD_CONCEPT_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_CONCEPT"]),
            _row("MINIMAX_REFMOD_TOKEN_CAP", "Clip token cap (stills untouched)", "choice",
                 "For clips as motion only: thin their frames to fit a token budget the way the pack's extractor "
                 "thins a clip. Only when the mod is over the cap, each clip loses the frames that are near-duplicates "
                 "of the last kept one (a held shot is mostly those), then what is left is spread evenly down to what "
                 "fits. Photos are never touched, whatever the cap; with no motion clips this does nothing. Off keeps "
                 "every frame.", rm.REFMOD_TOKEN_CAP_OPTIONS, rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_TOKEN_CAP"]),
            _row("MINIMAX_REFMOD_AUDIO", "Audio support", "choice",
                 "Also make an audio mod: the folder's sound (each clip's soundtrack and any audio file, in file "
                 "order) through the H3 audio VAE, the pack's audio RefMod. One file puts it in <name>.safetensors "
                 "beside the visual mod (the pack's bundle; needs the current pack). Two files writes it as "
                 "<name>_audio.safetensors, which any version reads. A clip Gizmo marked mute (_mute in its name) "
                 "lends no sound, the same rule as H3 training. A plain encode, no training. Needs the Audio VAE "
                 "path in Preferences. RefMod Toolkit renders visual mods only; audio mods play in ComfyUI.",
                 rm.REFMOD_AUDIO_OPTIONS, rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_AUDIO"]),
            _row("MINIMAX_REFMOD_AUDIO_CONCEPT", "Kind", "choice",
                 "What the sound is, in the pack's terms: voice, singing, music_style, sound_fx, ambience. Stored in "
                 "the file as its concept type.", rm.REFMOD_AUDIO_CONCEPT_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_AUDIO_CONCEPT"]),
            _row("MINIMAX_REFMOD_AUDIO_SECONDS", "Up to (seconds)", "combo", "", rm.REFMOD_AUDIO_SECONDS_OPTIONS,
                 rm.REFMOD_DEFAULTS["MINIMAX_REFMOD_AUDIO_SECONDS"]),
            _row("_refmod_note", "Steps 0 writes the plain encode (no captions needed). Steps above 0 optimise the mod "
                 "against H3: level on shots like your photos, well ahead on looks they never showed, cleaner skin in "
                 "both. RefMods make no previews: judge a mod in RefMod Toolkit.", "note"),
            _row("ENABLE_CACHE", "Prepare the cache before making the mod", "check", default=True)]},
        {"id": "samples_none", "title": "Previews", "tab": "samples", "rows": [_row(
            "_samples_note", "RefMods have no previews, so this tab isn't used. Judge a mod in the desktop app's "
            "RefMod Toolkit once it is made.", "note")]},
    ]
    return {"sections": sections,
            "presets": [{"name": n, "values": dict(v)} for n, v in rm.REFMOD_BUILT_IN_PRESETS.items()]}


def refmod_state(values, image_folder=""):
    """What RefMod's values hide and note: Kind and Up to only while Audio support is on; the token readout."""
    from fizgig.families import refmod as rm
    hidden, notes = set(), {}
    if not rm.refmod_audio_on(values.get("MINIMAX_REFMOD_AUDIO")):
        hidden |= {"MINIMAX_REFMOD_AUDIO_CONCEPT", "MINIMAX_REFMOD_AUDIO_SECONDS"}
    text, over = rm.token_text(values.get("MINIMAX_REFMOD_GRID"), values.get("MINIMAX_REFMOD_REFS"),
                               values.get("DATASET_MEGAPIXELS"))
    notes["REFMOD_TOKENS"] = ("⚠ " if over else "") + text
    from fizgig.families import launch
    media = launch.dataset_media(image_folder) if image_folder else set()
    return {"hidden": sorted(hidden), "disabled": [], "notes": notes, "kind": "standard", "media": sorted(media),
            "turbo_file": False}


def refmod_preset(values, preset):
    """A RefMod preset over the form's values: its keys as they are (the Training Base is never a preset's)."""
    return {**values, **{k: v for k, v in preset.items() if k != "H3_TRAIN_BASE"}}
