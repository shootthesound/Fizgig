"""Discover LoRA checkpoints to compare in LoRA Royale."""

import os
import re
from typing import List, Tuple

# One pattern for all three of the trainer's checkpoint names. The run name is whatever comes before the
# suffix, and the suffix says which kind of save this is:
#   "<name>-000005.safetensors"         an epoch checkpoint (get_epoch_ckpt_name)
#   "<name>-step0004064.safetensors"    a "Save now" snapshot (families/train.py; _CKPT in web/server.py
#                                       parses the same form for Runs & outputs)
#   "<name>.safetensors"                the last checkpoint, written when the final epoch completes
#                                       (get_last_ckpt_name)
# The name has to be lazy: greedy would swallow the suffix and every file would read as a last checkpoint.
_CKPT_RE = re.compile(r"^(?P<name>.+?)(?:-(?:step(?P<step>\d{7})|(?P<epoch>\d{6})))?\.safetensors$")


def scan_checkpoints(folder: str):
    """Find LoRA checkpoints in `folder`, sorted ascending.

    Returns a list of (label, path):
      - **Clean training run** (exactly one `<name>-NNNNNN.safetensors` run, with nothing
        beside it but that run's own files) → `label` is the integer epoch number, the
        run's own **Save now** snapshots join it last (newest step last, each labelled
        `step 4064`), and its **last checkpoint** follows them - the bare
        `<run>.safetensors` written when the final epoch completes, labelled `last`.
      - **Anything else** (a bag of arbitrary LoRAs, multiple runs, a run mixed with
        stray files, or Save now snapshots of some *other* run) → EVERY .safetensors
        in the folder, labelled by its filename stem (string), sorted by name. This is
        the 'compare a group of random LoRAs' case — not just training sets.

    State directories (`<name>-NNNNNN-state/`) and non-safetensors are ignored.
    """
    if not folder or not os.path.isdir(folder):
        return []

    by_run = {}   # base name -> [(epoch, path)]
    saves = {}    # base name -> [(step, path)] - that run's Save now snapshots
    others = []   # no suffix at all - a run's last checkpoint, or something not from here
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".safetensors"):
            continue
        full = os.path.join(folder, fn)
        if not os.path.isfile(full):
            continue
        m = _CKPT_RE.match(fn)
        if m and m.group("step"):
            saves.setdefault(m.group("name"), []).append((int(m.group("step")), full))
        elif m and m.group("epoch"):
            by_run.setdefault(m.group("name"), []).append((int(m.group("epoch")), full))
        else:
            others.append(full)

    # Clean single training run → epoch labels (the classic LoRA Royale flow), its own snapshots and
    # last checkpoint after the epochs. Anything not named after this run is a stray and demotes it.
    if len(by_run) == 1:
        run = next(iter(by_run))
        own_saves = all(n == run for n in saves)  # another run's snapshot still counts as a stray
        own_last = all(os.path.splitext(os.path.basename(p))[0] == run for p in others)
        if own_saves and own_last:
            epochs = sorted(by_run[run], key=lambda t: t[0])
            snaps = sorted(saves.get(run, []), key=lambda t: t[0])
            return ([(e, p) for e, p in epochs] + [(f"step {s}", p) for s, p in snaps]
                    + [("last", p) for p in others])

    # Otherwise: compare every .safetensors, labelled by filename stem.
    all_paths = [p for bucket in (by_run, saves) for group in bucket.values() for _, p in group] + others
    all_paths = sorted(set(all_paths), key=lambda p: os.path.basename(p).lower())
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in all_paths]


def run_name_for_folder(folder: str) -> str:
    """Best-guess training-run base name in `folder` (the run with the most
    epoch checkpoints), or '' if none. Used for labelling / promote naming."""
    if not folder or not os.path.isdir(folder):
        return ""
    counts = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return ""
    for fn in names:
        m = _CKPT_RE.match(fn)
        if m and m.group("epoch"):
            counts[m.group("name")] = counts.get(m.group("name"), 0) + 1
    return max(counts, key=counts.get) if counts else ""
