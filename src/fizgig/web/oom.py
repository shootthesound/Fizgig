"""Out-of-memory recovery for web runs: what ran out (classify), and what to do about it (decide) - the server
(server._run_stages) only carries the decision out.

GPU memory, with a save to carry on from: Continue once on the same settings (fresh memory: memory that fragments
over hours is the usual cause late in a run). Out again at about the same point (within an epoch of the last try):
Continue with one more step of block swap; with swap at its most, the next precision down - only when the run's
precision was Auto (a precision you picked is kept, and the run stops with advice). No save yet: one fresh restart a
step lighter. System memory (the computer killed the run - exit -9 / 137 on Linux - or MemoryError, Windows' "paging
file is too small"): with a save, one Continue on the same settings; then it stops with RAM advice (fewer swapped
blocks, which is the opposite of the GPU advice). At most MAX_TRIES automatic tries a run; then a summary of each.

Each try is kept on the run's history record ("oom"), so a server restart keeps the count and the run card can show
it. A fine-tune has no block swap to raise: it gets the plain Continue, then advice."""
import math
import re
import sys

MAX_TRIES = 3

GPU_MARKS = ("cuda out of memory", "torch.outofmemoryerror", "outofmemoryerror", "hip out of memory",
             "cublas_status_alloc_failed", "cudnn_status_alloc_failed")
RAM_MARKS = ("defaultcpuallocator: can't allocate memory", "defaultcpuallocator: not enough memory", "memoryerror",
             "paging file is too small", "os error 1455", "cannot allocate memory", "std::bad_alloc")
PLAN_LINE = re.compile(r"\[precision\] Auto plan: (\w+), block swap (-?\d+)")
KILLED = (-9, 137)                       # SIGKILL as Popen reports it, and as a shell (or a container) reports it


class Watch:
    """The lines of a stage as they stream: which kind of memory message it printed (GPU or system RAM)."""

    def __init__(self):
        self.gpu = self.ram = False
        self.plan = None                 # (precision, block swap) the trainer's Auto plan chose, when it said

    def feed(self, line):
        m = PLAN_LINE.search(str(line))
        if m:
            self.plan = (m.group(1), int(m.group(2)))
        low = str(line).lower()
        # the GPU's first: torch.OutOfMemoryError ends in "memoryerror" too
        if any(m in low for m in GPU_MARKS) or ("out of memory" in low and "cpu" not in low):
            self.gpu = True
        elif any(m in low for m in RAM_MARKS):
            self.ram = True


def classify(ret, watch, stopped=False, platform=None):
    """"gpu", "ram" or None for a stage that ended with exit code `ret` (its Watch fed every line). A run you
    stopped is never a memory failure. The last word wins: a RAM message beats an earlier GPU one (a preview's OOM,
    which the trainer survives, is printed long before the run dies of something else)."""
    if stopped or ret == 0:
        return None
    if watch.ram:
        return "ram"
    if watch.gpu:
        return "gpu"
    if (platform or sys.platform) != "win32" and ret in KILLED:
        return "ram"                     # Linux's OOM killer: no message, just SIGKILL
    return None


def used_plan(log_text):
    """(precision, block swap) the trainer's Auto plan chose, the last such line in the run's log; None without
    one (the run's own numbers were used as typed)."""
    hits = PLAN_LINE.findall(log_text or "")
    return (hits[-1][0], int(hits[-1][1])) if hits else None


def _swap_number(text):
    m = re.match(r"\s*(\d+)", str(text or ""))
    return int(m.group(1)) if m else None


def step_down(desc, values, plan, max_swap, precision_key, precision_labels):
    """The next lighter setup: (values, what changed) or (None, why not). One step of block swap (max / 8, at
    least 2) from the swap the run used; at the most swap, the next precision down - only from Auto."""
    v = dict(values)
    used_prec, used_swap = plan if plan else (precision_key(desc, v.get("FAMILY_PRECISION")), None)
    if used_swap is None:
        used_swap = _swap_number(v.get("BLOCKS_SWAP")) or 0
    if max_swap and used_swap < max_swap:
        step = max(2, math.ceil(max_swap / 8))
        new = min(max_swap, max(0, used_swap) + step)
        v["BLOCKS_SWAP"] = str(new)
        if plan and precision_key(desc, v.get("FAMILY_PRECISION")) == "auto":
            v["FAMILY_PRECISION"] = precision_labels.get(used_prec, v.get("FAMILY_PRECISION"))   # keep what fitted
        return v, f"block swap {used_swap} -> {new}"
    if precision_key(desc, v.get("FAMILY_PRECISION")) != "auto":
        return None, ("block swap is at its most and the precision is your own choice, so it's left as it is")
    order = [p for p in ("bf16", "int8", "nf4") if p in (desc.precisions or ())]
    lower = order[order.index(used_prec) + 1:] if used_prec in order else []
    if not lower:
        return None, "block swap is at its most and there's no lighter precision for this model"
    v["FAMILY_PRECISION"] = precision_labels.get(lower[0], lower[0])
    v["BLOCKS_SWAP"] = "Auto (detect from GPU)"
    return v, f"precision {used_prec} -> {lower[0]}"


def decide(kind, tries, has_save, step_now, steps_per_epoch, finetune, lighter):
    """What to do after a memory failure: {"do": "continue" | "restart" | "stop", "lighter": bool, "why": text}.
    tries: this run's earlier automatic tries ([{"kind", "step", ...}]); lighter(): (values, change) | (None, why)
    for one step lighter - asked only when it's needed."""
    if kind is None:
        return {"do": "stop", "why": ""}
    mine = [t for t in tries if t.get("kind") == kind]
    if len(tries) >= MAX_TRIES:
        return {"do": "stop", "why": f"{len(tries)} automatic tries already"}
    if kind == "ram":
        if has_save and not mine:
            return {"do": "continue", "lighter": False, "why": "carrying on from the newest save with fresh memory"}
        return {"do": "stop", "why": "system memory ran out again" if mine else "nothing saved to carry on from"}
    if not has_save:
        if mine:
            return {"do": "stop", "why": "it ran out of GPU memory again before saving anything"}
        if finetune:
            return {"do": "stop", "why": "a fine-tune has no block swap to raise"}
        return {"do": "restart", "lighter": True, "why": "nothing saved yet: starting again one step lighter"}
    if not mine:
        return {"do": "continue", "lighter": False, "why": "carrying on from the newest save with fresh memory"}
    near = step_now is not None and mine[-1].get("step") is not None and \
        step_now - int(mine[-1]["step"]) <= max(1, int(steps_per_epoch or 1))
    if not near:
        return {"do": "continue", "lighter": False,
                "why": "it trained on well past the last try, so fresh memory again"}
    if finetune:
        return {"do": "stop", "why": "out again at the same point, and a fine-tune has no block swap to raise"}
    return {"do": "continue", "lighter": True, "why": "out again at about the same point: one step lighter"}


def advice(kind, has_save, unit_done=""):
    """The plain-English note after a memory failure that recovery couldn't fix."""
    if kind == "ram":
        return ("\nSystem memory (RAM) ran out and the computer stopped the run. This is the computer's memory, not "
                "the graphics card's: swap fewer blocks (Blocks Swap, on the Training tab), turn off Faster block swap "
                "(Preferences > Training memory & speed), close other programs, or use a computer or pod with more "
                "RAM." + (f" The run is saved at {unit_done}: Continue on its card carries on from there."
                          if has_save and unit_done else "") + "\n")
    if has_save:
        return (f"\nThe GPU ran out of memory. The run is saved at {unit_done}: press Continue on its card in Runs & "
                f"outputs to carry on from there. More block swap or a smaller picture size (Training tab) uses "
                f"less, and closing other programs that use the GPU (a browser, a game) helps too.\n")
    return ("\nThe GPU ran out of memory before the run saved anything. Try a smaller picture size or more block swap "
            "(Memory & Precision, on the Training tab), and close other programs that use the GPU.\n")


def summary(tries):
    """Each automatic try, one line each, for the log once recovery stops."""
    if not tries:
        return ""
    lines = [f"  {i}. {'GPU' if t.get('kind') == 'gpu' else 'system RAM'} ran out at step {t.get('step', '?')}: "
             f"{t.get('action', '')}" for i, t in enumerate(tries, 1)]
    return "\nOut-of-memory recovery tried:\n" + "\n".join(lines) + "\n"


__all__ = ["MAX_TRIES", "Watch", "advice", "classify", "decide", "step_down", "summary", "used_plan"]
