"""Training-progress parsing kept outside the individual trainers.

The GUI already receives each trainer's normal console stream. This module turns
those existing tqdm and preview lines into display state, so changing a trainer does
not require a second progress-reporting path inside its training loop.
"""

from __future__ import annotations

import math
import re


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TQDM_PROGRESS_RE = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+)\s+\[[^\]<]*<(?P<eta>[^,\]]+),\s*"
    r"(?P<speed>[0-9]+(?:\.[0-9]+)?)(?P<unit>it/s|s/it)"
    r"(?:,\s*avr_loss=(?P<loss>[-+0-9.eE]+))?"
)
# the postfix's other figures, read on their own (tqdm sorts its postfix keys): the run summary's Learning
_POSTFIX_RE = re.compile(r"[,\s](lr|gn)=([-+0-9.eE]+)")
# The cache stages' own count (cache_progress below), for the run card:  [cache] captions 40/120
_CACHE_RE = re.compile(r"\[cache\]\s+(?P<what>[a-z ]+?)\s+(?P<n>\d+)/(?P<total>\d+)")
_EPOCH_RE = re.compile(r"\bepoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)", re.I)
# The RefMod maker (minimax_refmod.py) prints its own step line, not a tqdm bar:
#   [refmod] step 50/200  loss 0.4312  drift 0.012
_REFMOD_STEP_RE = re.compile(r"\[refmod\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)(?:\s+loss\s+(?P<loss>[-+0-9.eE]+))?")
_REFMOD_PROGRESS_RE = re.compile(
    r"\[refmod\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"loss\s+(?P<loss>[-+0-9.eE]+).*?"
    r"(?P<speed>[0-9]+(?:\.[0-9]+)?)\s+s/step",
    re.I,
)


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def parse_refmod_progress_line(line: str):
    """Extract the maker's periodic optimisation report without changing its code."""
    clean = _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")
    match = _REFMOD_PROGRESS_RE.search(clean)
    if match is None:
        return None
    try:
        step = int(match.group("step"))
        total = int(match.group("total"))
        seconds_per_step = float(match.group("speed"))
    except (TypeError, ValueError):
        return None
    if total <= 0 or step < 0 or seconds_per_step < 0:
        return None
    return {
        "step": step,
        "total_steps": total,
        "eta_text": _format_eta(max(0, total - step) * seconds_per_step),
        "speed_text": f"{match.group('speed')} s/step",
        "average_loss_text": match.group("loss"),
    }


def parse_tqdm_progress_line(line: str):
    """Extract exactly the values displayed by a tqdm training console line."""
    clean = _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")
    # Preview generation has its own tqdm bar (``sampling: 3/8``). It must not
    # replace the training card while the card is deliberately showing the
    # preview phase. These are the descriptions used by the three trainers.
    if re.search(r"(?:minimax-h3|steps):\s", clean) is None:
        return None
    match = _TQDM_PROGRESS_RE.search(clean)
    if match is None:
        return None
    try:
        result = {
            "step": int(match.group("step")),
            "total_steps": int(match.group("total")),
            "eta_text": match.group("eta").strip(),
            "speed_text": f"{match.group('speed')} {match.group('unit')}",
            "average_loss_text": match.group("loss"),
            **{{"lr": "lr_text", "gn": "grad_norm_text"}[k]: v for k, v in _POSTFIX_RE.findall(clean)},
        }
    except (TypeError, ValueError):
        return None
    if result["total_steps"] <= 0:
        return None
    return result


def parse_refmod_step_line(line: str):
    """``{"step", "total_steps", "loss_text"}`` from a RefMod maker step line, else ``None``."""
    clean = _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")
    match = _REFMOD_STEP_RE.search(clean)
    if match is None:
        return None
    step, total = int(match.group("step")), int(match.group("total"))
    if total <= 0:
        return None
    return {"step": step, "total_steps": total, "loss_text": match.group("loss")}


def parse_epoch_line(line: str):
    """Return ``(epoch, total_epochs)`` from an existing trainer message."""
    clean = _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")
    match = _EPOCH_RE.search(clean)
    if match is None:
        return None
    epoch, total = int(match.group("epoch")), int(match.group("total"))
    if epoch <= 0 or total <= 0:
        return None
    return epoch, total


def parse_preview_phase(line: str):
    """Return ``(phase, epoch)`` for existing trainer preview messages."""
    clean = _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")
    lower = clean.lower()
    epoch_match = re.search(r"epoch[- ]?(\d+)", lower)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    if "[preview]" in lower and "preview failed" in lower:
        return "failed", epoch
    if "[preview]" in lower and "wrote" in lower and "sample" in lower:
        return "complete", epoch
    if "rendering preview" in lower:
        return "start", epoch
    if ("[preview]" in lower and
            ("sampling with" in lower or re.search(r"epoch\s+\d+:\s+prompt", lower))):
        return "start", epoch
    return None


class TrainingProgressTracker:
    """Convert the unmodified trainers' console output into GUI-ready state."""

    def __init__(self, total_epochs=1):
        self.reset(total_epochs)

    def reset(self, total_epochs=1):
        try:
            total_epochs = int(total_epochs)
        except (TypeError, ValueError):
            total_epochs = 1
        self.total_epochs = max(1, total_epochs)
        self.current_epoch = 1

    def consume(self, line: str):
        """Return an update dictionary for a relevant line, otherwise ``None``."""
        m = _CACHE_RE.search(line)
        if m:
            return {"kind": "cache", "what": m["what"], "n": int(m["n"]), "total": int(m["total"])}
        refmod = parse_refmod_progress_line(line)
        if refmod is not None:
            return {"kind": "refmod", **refmod}

        epoch = parse_epoch_line(line)
        if epoch is not None:
            self.current_epoch, self.total_epochs = epoch

        preview = parse_preview_phase(line)
        if preview is not None:
            phase, preview_epoch = preview
            if preview_epoch is not None:
                self.current_epoch = preview_epoch
            return {"kind": "preview", "phase": phase, "epoch": preview_epoch}

        displayed = parse_tqdm_progress_line(line)
        if displayed is None:
            return None

        # The trainers use one run-wide bar with a constant number of steps per
        # epoch. Deriving the epoch here avoids instrumentation in every trainer.
        step = displayed["step"]
        total_steps = displayed["total_steps"]
        if total_steps >= self.total_epochs:
            inferred = math.ceil(max(1, step) * self.total_epochs / total_steps)
            self.current_epoch = max(1, min(self.total_epochs, inferred))

        return {
            "kind": "training",
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            **displayed,
        }


def dataset_items(dataset) -> int:
    """How many items a dataset caches (0 when it can't say: then no count is shown)."""
    try:
        return len(dataset.datasource)
    except Exception:
        return 0


class CacheProgress:
    """A cache stage's count as a plain line the run card reads ("[cache] captions 40/120"): the stages' tqdm bars
    walk a stream with no length, so they only count batches, and a long encode looked stuck (#249). At most one
    line a second, and always the last."""

    def __init__(self, what: str, total: int):
        self.what, self.total, self.n, self._at = what, max(0, int(total)), 0, 0.0

    def add(self, k: int) -> None:
        import time
        self.n += k
        now = time.time()
        if self.total and (now - self._at >= 1.0 or self.n >= self.total):
            self._at = now
            print(f"[cache] {self.what} {min(self.n, self.total)}/{self.total}", flush=True)
