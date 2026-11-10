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
    vram_gb: float                   # what it needs on the card at its smallest (checked beside a running run)
    size_gb: float = 0.0             # its download, for the hint
    tasks: tuple = ()                # a fixed menu of tasks the model understands (Florence's <CAPTION>...), first = default
    default_task: str = ""
    instruction: bool = False        # follows a written instruction (the Captions tab's task menu + edit box)
    hint: str = ""                   # one line for the model menu's hint
    blurb: str = ""                  # a few words on what it's good for (Preferences' Other captioning models)
    download_skip: tuple = ()        # repo files its class never loads (a second copy of the weights...): the
                                     # Download button leaves them out
    precisions: tuple = ()           # ((name, vram_gb), ...) best first, when it loads in more than one way: the
                                     # worker picks the first that fits the card's free memory (pick_precision)
    # An instruction captioner's own prompt menu, shown under Fizgig's instruction tasks (the page builds the text):
    prompt_types: tuple = ()         # ((label, any length, "{word_count}" words, "{length}" word), ...) templates
    prompt_lengths: tuple = ()       # the Length menu: words ("long") and word counts ("120"); first = "any"
    default_length: str = ""
    prompt_extras: tuple = ()        # ((label, sentence, ticked by default), ...) added to any instruction; "{name}"
                                     # in a sentence is the trigger word (that sentence left out when there is none)


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
    """The class a captioner implements. Fizgig constructs it with its description, the device ("cuda" / "cpu"), its
    files ({"repo", "revision", "code_revision"} or {"path"}) and, for one with precisions, the one to load in; calls
    load() once, caption() per picture, and unload() before the worker exits."""

    def __init__(self, desc: CaptionerDescription, device: str, files: dict, precision: str = ""):
        self.desc, self.device, self.files, self.precision = desc, device, files, precision

    @classmethod
    def fetch(cls, desc: "CaptionerDescription", files: dict) -> None:
        """Download everything load() will need, so it then works offline: by default the pinned snapshot without
        desc.download_skip. A captioner whose first use caches more than its own files - a trust_remote_code model's
        code, which can live in another repo - overrides this to fetch that too (FlorenceCaptioner)."""
        from huggingface_hub import snapshot_download
        snapshot_download(files["repo"], revision=files["revision"], ignore_patterns=list(desc.download_skip) or None)

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


def pick_precision(desc: CaptionerDescription, free_gb: float | None) -> str:
    """The best of a captioner's precisions that fits `free_gb` (None: not on a GPU - the first), else its smallest;
    "" for one with a single way to load."""
    if not desc.precisions:
        return ""
    if free_gb is None:
        return desc.precisions[0][0]
    return next((n for n, need in desc.precisions if need <= free_gb), desc.precisions[-1][0])


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


def captioner_class(desc: CaptionerDescription):
    """Its class, imported now (inside the worker)."""
    mod, _, cls = desc.module.partition(":")
    return getattr(importlib.import_module(mod), cls)


def make(desc: CaptionerDescription, device: str, files: dict, precision: str = "") -> Captioner:
    """The captioner's class, imported now (inside the worker) and constructed."""
    mod, _, cls = desc.module.partition(":")
    return getattr(importlib.import_module(mod), cls)(desc, device, files, precision)
