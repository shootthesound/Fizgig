"""Start's shared checks: what every family refuses before a run starts, as plain functions.

The desktop app runs them for every family, in this order, with each family's own checks slotted in between
(validate_inputs); a described family's launch plan (fizgig.families.launch) runs the same functions in the same
order, so both refuse the same runs with the same words.

`v` is a plain dict of the run's settings as typed (LEARNING_RATE, NETWORK_DIM, MAX_TRAIN_EPOCHS, ...), plus
batch_size and megapixels. Every function returns a list of messages; an empty list means nothing to refuse.
"""
import glob
import os
import re


def number(label, raw, cast, minimum=None):
    """A free-text number box: a message naming the field when it is not a number or below its minimum.
    A bare int()/float() deeper in the launch used to fail inside a Tk callback where nobody saw it."""
    raw = str(raw).strip()
    try:
        val = cast(raw)
    except (TypeError, ValueError):
        return [f"{label} must be a number (got {raw!r})"]
    if minimum is not None and val < minimum:
        return [f"{label} must be at least {minimum} (got {raw})"]
    return []


def learning_rate(v):
    """The Learning Rate box, unless Adaptive LR is on (the box is ignored and greyed then)."""
    if v.get("ADAPTIVE_LR"):
        return []
    return number("Learning Rate", v.get("LEARNING_RATE", ""), float, 0)


def network(v, lokr=False):
    out = number("Network Dim (Rank)", v.get("NETWORK_DIM", ""), int, 1)
    out += number("Network Alpha", v.get("NETWORK_ALPHA", ""), float, 0)
    if lokr:
        out += number("LoKR Factor", v.get("LOKR_FACTOR", ""), int, 2)
    return out


def numbers(v):
    """The run's other number boxes. Target Megapixels matters most: an unparseable one makes the dataset TOML
    skip its rewrite silently, so the run would train the last dataset the TOML pointed at."""
    out = []
    for label, key, cast, minimum in (("Max Train Epochs", "MAX_TRAIN_EPOCHS", int, 1),
                                      ("Save Every N Epochs", "SAVE_EVERY_N_EPOCHS", int, 1),
                                      ("Seed", "SEED", int, None),
                                      ("LoRA+ LR Ratio", "LORA_LR_RATIO", int, 1),
                                      ("Gradient Accumulation", "GRADIENT_ACCUMULATION", int, 1),
                                      ("Max Grad Norm", "MAX_GRAD_NORM", float, 0),
                                      ("Batch Size (Dataset)", "batch_size", int, 1),
                                      ("Target Megapixels (Dataset)", "megapixels", float, 0)):
        out += number(label, v.get(key, ""), cast, minimum)
    if "KEEP_LAST_N_STATES" in v:
        out += number("Keep Last (states)", v["KEEP_LAST_N_STATES"], int, 1)
    return out


def dataset_config(path, must_exist=True):
    if not path:
        return ["Dataset config file path is empty — set the training image folder on the Start tab"]
    if must_exist and not os.path.exists(path):
        return [f"Dataset config file does not exist: {path}"]
    return []


def learning_rate_range(v):
    """With Adaptive LR on only Min < Max matters (the run starts at their geometric midpoint); off, the
    Learning Rate must be positive."""
    if v.get("ADAPTIVE_LR"):
        try:
            hi = str(v["ADAPTIVE_LR_MAX"]).split(" ")[0]
            lo = str(v["ADAPTIVE_LR_MIN"]).split(" ")[0]
            if float(lo) >= float(hi):
                return [f"Adaptive Min LR ({lo}) must be lower than Max LR ({hi})."]
        except (ValueError, KeyError):
            return ["Adaptive Min/Max LR must be valid numbers."]
        return []
    try:
        if float(v.get("LEARNING_RATE")) <= 0:
            return ["Learning rate must be positive"]
    except (TypeError, ValueError):
        return ["Learning rate must be a valid number"]
    return []


def context_lora(v):
    """A Context LoRA must be a .safetensors file that exists, at a strength from 0 to 2."""
    path = str(v.get("CONTEXT_LORA_PATH") or "").strip()
    if not path:
        return []
    out = []
    if not os.path.exists(path):
        out.append(f"Context LoRA file does not exist: {path}")
    elif not path.lower().endswith(".safetensors"):
        out.append(f"Context LoRA must be a .safetensors file: {path}")
    try:
        strength = float(v["CONTEXT_LORA_STRENGTH"])
        if not (0.0 <= strength <= 2.0):
            out.append(f"Context LoRA Strength ({strength}) must be between 0.0 and 2.0")
    except (ValueError, KeyError):
        out.append("Context LoRA Strength must be a valid number")
    return out


def tidy_name(raw):
    """(name, error or None). A stray character in the LoRA name (a pasted newline) only fails at the first
    checkpoint save, an epoch in. What has one obvious intent is fixed - surrounding whitespace, control
    characters, trailing dots Windows drops anyway; anything else is refused by name."""
    name = "".join(c for c in (raw or "") if c >= " ").strip().rstrip(".").strip()
    if not name:
        return name, "LoRA name cannot be empty"
    bad = next((c for c in name if c in '<>:"|?*/\\'), None)
    if bad is not None:
        return name, (f"LoRA name cannot contain '{bad}' — file names can't include that "
                      f"character. Use letters, numbers, spaces, - _ or .")
    return name, None


def run(v, *, blocks_swap, swap_max, arch_label, name_error):
    """Rank, alpha, epochs, the save cadence, the per-category stop epoch, block swap, the LoRA name, the output
    folder and a resume path. `blocks_swap` is the resolved number (None: not a number); `name_error` comes from
    tidy_name."""
    out = []
    try:
        if int(v.get("NETWORK_DIM")) <= 0:
            out.append("Network dim must be a positive integer")
    except (TypeError, ValueError):
        out.append("Network dim must be a valid integer")
    try:
        if float(v.get("NETWORK_ALPHA")) < 0:
            out.append("Network alpha must be non-negative")
    except (TypeError, ValueError):
        out.append("Network alpha must be a valid number")
    try:
        if int(v.get("MAX_TRAIN_EPOCHS")) <= 0:
            out.append("Max train epochs must be a positive integer")
    except (TypeError, ValueError):
        out.append("Max train epochs must be a valid integer")
    try:
        if int(v.get("SAVE_EVERY_N_EPOCHS")) <= 0:
            out.append("Save every N epochs must be a positive integer")
        stop = str(v.get("MIXED_STOP_EPOCH") or "").strip()
        if stop and (not stop.isdigit() or int(stop) <= 0):
            out.append(f"'Finish one category early: after epoch' must be blank or "
                       f"a positive whole number, not {stop!r}")
    except (TypeError, ValueError):
        out.append("Save every N epochs must be a valid integer")
    if blocks_swap is None:
        out.append("Blocks swap must be a valid integer")
    elif blocks_swap < 0:
        out.append("Blocks swap must be non-negative")
    elif blocks_swap > swap_max:
        out.append(f"Blocks swap ({blocks_swap}) exceeds maximum for {arch_label} ({swap_max})")
    if name_error:
        out.append(name_error)
    if not v.get("LORA_OUTPUT_DIR"):
        out.append("LoRA output directory is empty")
    resume = str(v.get("RESUME_TRAINING") or "").strip()
    if resume and not os.path.exists(resume):
        out.append(f"Resume training path does not exist: {resume}")
    elif resume and os.path.isdir(resume):     # a file is a fine-tune's checkpoint (its own continuation)
        out += state_folder_problems(resume)
    lora_out = str(v.get("LORA_OUTPUT_DIR") or "").strip()
    if lora_out and _is_state(lora_out):
        # #352: a state folder picked as the output folder - the run's files would land inside that saved state
        out.append(f"The LoRA output folder is a saved training state ({os.path.basename(os.path.normpath(lora_out))}). "
                   f"Set the output folder to the folder that holds it ({os.path.dirname(os.path.normpath(lora_out))}), "
                   f"and put the state folder in Resume Training (Other Options) to carry that run on.")
    return out


def _is_state(path):
    """A saved training state folder: Fizgig's (training_state.json) or the old trainer's (model.safetensors, the
    epoch in the folder name)."""
    return os.path.isdir(path) and (os.path.isfile(os.path.join(path, "training_state.json")) or (
        os.path.isfile(os.path.join(path, "model.safetensors"))
        and re.search(r"-\d{6}-state$", os.path.basename(os.path.normpath(path))) is not None))


def state_folder_problems(path):
    """Resume Training must name the state folder itself (#352: the folder holding it was passed, and the trainer
    stopped on "not a saved training state"). Names the newest state inside when there is one."""
    if _is_state(path):
        return []
    try:
        inside = [e for e in os.scandir(path) if e.is_dir() and _is_state(e.path)]
    except OSError:
        inside = []
    if inside:
        newest = max(inside, key=lambda e: e.stat().st_mtime)
        return [f"Resume Training is set to {path}, the folder that holds the saved states, not a state itself. Pick "
                f"the state folder inside it, e.g. {newest.name} (the newest)."]
    return [f"Resume Training is set to {path}, which isn't a saved training state. Pick a folder named like "
            f"'<lora name>-000012-state'."]


def training_folder(image_dir, caption_ext, *, check_captions=True, media_exts=(), label="the training folder"):
    """The training folder must exist; its photos (and clips) need captions - none at all, or some missing, and
    the run would quietly leave them out. check_captions is off for runs that read none (a RefMod plain encode,
    a prompt slider)."""
    out = []
    if image_dir and not os.path.isdir(image_dir):
        out.append(f"Training image folder does not exist: {image_dir}")
    if image_dir and os.path.isdir(image_dir) and caption_ext and check_captions:
        # glob.escape: a folder like "[subject] photos" otherwise finds no captions at all
        caption_files = glob.glob(os.path.join(glob.escape(image_dir), "*" + caption_ext))
        if not caption_files:
            out.append(f"No caption files (*{caption_ext}) found in {image_dir if label == 'the training folder' else label}. "
                       f"Use the Captions tab to generate them first.")
        else:
            from fizgig.dataset.masks import is_mask_file      # photo-masklabel.png is a mask, not a picture
            stems = {os.path.splitext(os.path.basename(p))[0] for p in caption_files}
            uncaptioned = sorted(f for f in os.listdir(image_dir)
                                 if os.path.splitext(f)[1].lower() in media_exts
                                 and os.path.splitext(f)[0] not in stems and not is_mask_file(f))
            if uncaptioned:
                shown = ", ".join(uncaptioned[:6]) + (f" … and {len(uncaptioned) - 6} more"
                                                      if len(uncaptioned) > 6 else "")
                out.append(f"{len(uncaptioned)} file(s) in {label} have no {caption_ext} caption and "
                           f"would be left out of the run: {shown}. Caption them on the Captions tab (clips are "
                           f"captioned from their middle frame), or move them out of the folder.")
    return out
