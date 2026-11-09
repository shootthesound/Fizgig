"""Captioners: the models that write a picture's caption on the Captions tab, each in Fizgig's language.

A captioner is a description (what it is: its kind, its files, the memory it needs, what it accepts) plus a small class
with three jobs - load(), caption(picture, request) -> CaptionResult, unload(). Fizgig does everything else: the files'
download, the worker process, batching, progress and Stop, Overwrite, detail crops, the trigger word, a tagger's
threshold and left-out tags, and writing the caption files. A captioner never writes a file, reads Preferences or sees
a folder: it is handed one upright RGB picture and gives back words. See docs/ADDING_A_CAPTIONER.md.

THE MEMORY RULE (Peter, 9 Nov 2026 - not to be broken): a captioner here runs only in the Captions tab's own worker
process (fizgig.scripts.batch_caption), which exits to free its memory. None ever loads in a training run, so no
training memory plan, block swap or Studio plan has to know about it. Auto-recaption inside a run stays on the one
captioner the training plans already fit around - Krea 2's Qwen3-VL-4B, or a secondary captioning file of the same
architecture (web.captions.captioner_problem checks its header) - and calls it directly (families/loss_watch.py), not
through this registry. checks/test_captioners.py fails if training code imports this package.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

KINDS = ("sentences", "tags")


@dataclass(frozen=True)
class HFRepo:
    """Files a captioner downloads itself from Hugging Face, pinned so a change upstream can't swap them."""
    repo: str
    revision: str
    code_revision: str = ""          # trust_remote_code models: the pinned code revision, when it differs


@dataclass(frozen=True)
class PrefFile:
    """A file Preferences already holds (the Captions tab resolves it; web.server.captioner_for for "captioner")."""
    pref: str


@dataclass(frozen=True)
class CaptionerDescription:
    key: str                         # stable id: saved settings and the worker refer to it
    label: str                       # what the model menu shows
    kind: str                        # "sentences" or "tags"
    module: str                      # "fizgig.captioners.<file>:<Class>", imported only inside the worker
    files: HFRepo | PrefFile
    licence: str
    vram_gb: float                   # what it needs on the card (the Captions tab checks it beside a run)
    size_gb: float = 0.0             # its download, for the hint
    tasks: tuple = ()                # a fixed menu of tasks the model understands (Florence's <CAPTION>...), first = default
    default_task: str = ""
    instruction: bool = False        # follows a written instruction (the Captions tab's task menu + edit box)
    hint: str = ""                   # one line for the model menu's hint


@dataclass
class CaptionRequest:
    """What Fizgig asks for one picture."""
    task: str = ""                   # one of the description's tasks, when it has them
    instruction: str = ""            # the instruction, for a captioner that follows one
    max_new_tokens: int = 120


@dataclass
class CaptionResult:
    """What a captioner gives back: sentences as text, or tags with their scores (0-1)."""
    text: str = ""
    tags: list = field(default_factory=list)      # [(tag, score)], any order


class Captioner:
    """The class a captioner implements. Fizgig constructs it with its description, the device ("cuda" / "cpu") and
    its files ({"repo", "revision", "code_revision"} or {"path"}), calls load() once, caption() per picture, and
    unload() before the worker exits."""

    def __init__(self, desc: CaptionerDescription, device: str, files: dict):
        self.desc, self.device, self.files = desc, device, files

    def load(self) -> None:
        raise NotImplementedError

    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        raise NotImplementedError

    def unload(self) -> None:
        pass


def finish(result: CaptionResult, *, threshold: float = 0.35, leave_out=(), underscores: bool = False) -> str:
    """Fizgig's side: a result as caption text. Tags: those scoring at least `threshold`, best first, minus any in
    `leave_out` (case-insensitive), underscores as spaces unless asked to keep them. "" when nothing is left."""
    if result.tags:
        drop = {str(t).strip().lower().replace(" ", "_") for t in leave_out if str(t).strip()}
        keep = [t for t, s in sorted(result.tags, key=lambda ts: -float(ts[1]))
                if float(s) >= threshold and str(t).lower().replace(" ", "_") not in drop]
        return ", ".join(t if underscores else str(t).replace("_", " ") for t in keep)
    return " ".join(str(result.text or "").split())


def with_trigger(caption: str, trigger: str) -> str:
    """The trigger word first, as every Fizgig caption has it."""
    trigger = str(trigger or "").strip()
    return f"{trigger}, {caption}" if trigger and caption else (caption or "")


# ---- the registry --------------------------------------------------------------------------------------------
from fizgig.captioners.registry import CAPTIONERS  # noqa: E402  (descriptions only: no torch, no model code)

_BY_KEY = {d.key: d for d in CAPTIONERS}
_BY_LABEL = {d.label: d for d in CAPTIONERS}


def get(key_or_label: str) -> CaptionerDescription | None:
    return _BY_KEY.get(key_or_label) or _BY_LABEL.get(key_or_label)


def make(desc: CaptionerDescription, device: str, files: dict) -> Captioner:
    """The captioner's class, imported now (inside the worker) and constructed."""
    mod, _, cls = desc.module.partition(":")
    return getattr(importlib.import_module(mod), cls)(desc, device, files)
