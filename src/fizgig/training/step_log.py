"""The run's per-step loss for the web UI's loss graph, and the "Sample now" / "Save now" requests it can make.

Loss: `<output_dir>/.fizgig_loss_<output_name>.jsonl`, one line a step - {"s": step, "e": epoch, "l": loss, "lr": lr} -
written in small batches. A run starting at step N first keeps only the lines up to N: a fresh run (N = 0) starts the
file empty, a resumed one carries on its own curve without the steps it is about to redo.

Requests: the web UI (or anything) drops `.sample_now` / `.save_now` into the output folder, as `.pause_requested` is
dropped for Pause; the trainer takes each once, after an optimizer step.
"""
import json
import os

FLUSH_EVERY = 20


def loss_path(output_dir, output_name):
    return os.path.join(output_dir, f".fizgig_loss_{output_name}.jsonl")


class StepLog:
    def __init__(self, output_dir, output_name, start_step=0):
        self.path = loss_path(output_dir, output_name)
        self.buf = []
        kept = []
        if start_step > 0:
            try:
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            if int(json.loads(line)["s"]) <= start_step:
                                kept.append(line if line.endswith("\n") else line + "\n")
                        except (ValueError, KeyError, TypeError):
                            continue
            except OSError:
                pass
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(kept)
        except OSError:
            self.path = None                 # a read-only folder: the graph is simply not there

    def add(self, step, epoch, loss, lr=None):
        if self.path is None:
            return
        rec = {"s": int(step), "e": int(epoch), "l": round(float(loss), 6)}
        if lr is not None:
            rec["lr"] = float(f"{lr:.6g}")
        self.buf.append(json.dumps(rec) + "\n")
        if len(self.buf) >= FLUSH_EVERY:
            self.flush()

    def add_validation(self, step, epoch, loss, per_t=None, seconds=None, seg=0):
        """A fixed validation check: {"k": "val", "s", "e", "v": mean, "vt": [per timestep], "sec"}. It has no "l",
        so anything reading only step lines skips it. Written straight away (one a check)."""
        if self.path is None:
            return
        rec = {"k": "val", "s": int(step), "e": int(epoch), "v": round(float(loss), 6)}
        if per_t:
            rec["vt"] = [round(float(x), 6) for x in per_t]
        if seconds is not None:
            rec["sec"] = round(float(seconds), 1)
        if seg:                      # a new section of the validation line (what's checked changed on a resume)
            rec["seg"] = int(seg)
        self.buf.append(json.dumps(rec) + "\n")
        self.flush()

    def flush(self):
        if self.path is None or not self.buf:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.writelines(self.buf)
        except OSError:
            pass
        self.buf = []


def take_request(output_dir, what):
    """True once when `.sample_now` / `.save_now` (what = "sample" / "save") is waiting; the flag is removed."""
    p = os.path.join(output_dir, f".{what}_now")
    if not os.path.exists(p):
        return False
    try:
        os.remove(p)
    except OSError:
        return False                         # already taken (or not ours to remove): don't act twice
    return True


def epoch_means(rows):
    """[[last step of the epoch, epoch, mean raw loss], ...] - one point an epoch."""
    out, cur, acc = [], None, []
    for r in list(rows) + [None]:
        if r is None or r[1] != cur:
            if acc:
                out.append([acc[-1][0], cur, round(sum(a[2] for a in acc) / len(acc), 6)])
            if r is None:
                break
            cur, acc = r[1], []
        acc.append(r)
    return out


def running_avg(rows):
    """kohya's avr_loss (train_util LossRecorder): the first epoch the mean of every step so far; after that a window
    of one epoch's steps, each step position's newest loss replacing the one before. [[step, avg], ...]"""
    out, slots, total, first, cur, pos = [], [], 0.0, None, None, 0
    for r in rows:
        if first is None:
            first = r[1]
        if r[1] != cur:
            cur, pos = r[1], 0
        else:
            pos += 1
        if cur == first or not slots:
            slots.append(r[2])
            total += r[2]
        else:
            i = pos % len(slots)
            total += r[2] - slots[i]
            slots[i] = r[2]
        out.append([r[0], round(total / len(slots), 6)])
    return out


def _thin(seq, max_points):
    k = max(1, -(-len(seq) // max(1, max_points)))
    out = seq[::k]
    if seq and out[-1] is not seq[-1]:
        out.append(seq[-1])
    return out


def read(path, max_points=4000):
    """The graph's data: {"points": [[step, epoch, loss, lr], ...] (every k-th when longer than max_points, the
    last always kept), "epochs": [[epoch, first step], ...], "steps": total, and the trend views, worked out from
    every step: "epoch_mean", "running_avg" and "val"
    ([[step, epoch, loss, [per timestep]], ...], the fixed validation checks)}."""
    rows, val = [], []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("k") == "val":
                        val.append([int(d["s"]), int(d["e"]), float(d["v"]), d.get("vt") or []]
                                   + ([int(d["seg"])] if d.get("seg") else []))   # [.., seg] from a new section
                        continue
                    rows.append([int(d["s"]), int(d["e"]), float(d["l"]), d.get("lr")])
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    except OSError:
        return None
    epochs, last = [], None
    for r in rows:
        if r[1] != last:
            epochs.append([r[1], r[0]])
            last = r[1]
    k = max(1, -(-len(rows) // max_points))
    pts = rows[::k]
    if rows and pts[-1] is not rows[-1]:
        pts.append(rows[-1])
    return {"points": pts, "epochs": epochs, "steps": len(rows), "stride": k, "epoch_mean": epoch_means(rows),
            "running_avg": _thin(running_avg(rows), max_points), "val": val}
