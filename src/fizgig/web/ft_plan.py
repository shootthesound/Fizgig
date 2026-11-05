"""The fine-tune card's plan line (the desktop's _family_ft_plan_refresh): how a rotation splits on this card, worked
out from the model file's header and the free GPU memory now - no model loads. The planner imports torch and reads
the card, so it runs in a short subprocess (`python -m fizgig.web.ft_plan ...`), never in the web server; the answer
is kept per setup, and line() says "working it out" until it is in."""
import json
import os
import subprocess
import sys
import threading

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_lock = threading.Lock()
_cache = {}                     # key -> (windows, stream, free) | None (no plan) | "running"


def _num(values, key, default):
    try:
        return max(1, int(float(str(values.get(key) or default).strip())))
    except ValueError:
        return None


def _options(desc, values):
    """The family's own fine-tune settings (H3's Blocks range), as the desktop hands them to the planner."""
    from fizgig.families.form import _option_values
    vals, out = _option_values(desc, values), {}
    for o in desc.options:
        for t in o.resolve(vals.get(o.key, o.default)):
            k, eq, v = t.partition("=")
            if eq and not k.startswith(("--", "aux:")):
                out[k] = v
    return out


def _run(key, family, path, mp, options, max_parts, env, batch_gb=0.0):
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.ft_plan", family, path, str(mp or ""),
                            json.dumps(options), str(max_parts), f"{batch_gb:g}"], capture_output=True, text=True, timeout=120,
                           cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        got = json.loads((r.stdout.strip().splitlines() or ["null"])[-1])
    except (OSError, ValueError, subprocess.SubprocessError):
        got = None
    with _lock:
        _cache[key] = tuple(got) if got else None


def line(desc, values, model_path, env):
    """(text, pending) for the fine-tune card: pending while the plan is being worked out."""
    from fizgig.families.launch import ft_max_parts
    rot, every, per = (_num(values, "FAMILY_FT_ROTATIONS", 10), _num(values, "FAMILY_FT_SAVE_EVERY", 1),
                       _num(values, "FAMILY_FT_ROTATE_EVERY", 1))
    if None in (rot, every, per):
        return "Rotations, checkpoint cadence and epochs per part are whole numbers of 1 or more.", False
    fallback = (f"{rot} rotation(s), a checkpoint + preview every {every}. How many parts a rotation has is planned "
                "from the free memory at Start; the log shows the epochs.")
    if not model_path or not os.path.isfile(model_path):
        return fallback, False
    try:
        mp = float(str(values.get("DATASET_MEGAPIXELS") or "").split()[0])
    except (ValueError, IndexError):
        mp = None
    options = _options(desc, values)
    cap = ft_max_parts({"FAMILY_FT_MAX_PARTS": values.get("FAMILY_FT_MAX_PARTS", "")})
    from fizgig.families.launch import batch_size
    fixed, per_mp = getattr(desc, "batch_gb", None) or (0.0, 0.0)      # quant.batch_extra (no torch in the server)
    extra = max(0, batch_size(desc, values) - 1) * (float(fixed) + float(per_mp) * float(mp or 0.25))
    key = (desc.key, model_path, mp, tuple(sorted(options.items())), cap, round(extra, 2))
    with _lock:
        plan = _cache.get(key, "new")
        if plan == "new":
            _cache[key] = "running"
            threading.Thread(target=_run, args=(key, desc.key, model_path, mp, options, cap, env, extra),
                             daemon=True).start()
    if plan in ("new", "running"):
        return "Working out how a rotation splits on this card…", True
    if not plan or not plan[0]:
        return fallback, False
    windows, stream, free = plan
    cyc = len(windows) * per
    total = rot * cyc
    at = list(range(cyc * every, total + 1, cyc * every))
    if at and at[-1] != total:
        at.append(total)
    shown = ", ".join(str(e) for e in at[:5]) + (f", ... {at[-1]}" if len(at) > 5 else "")
    try:
        gb = os.path.getsize(model_path) / 1e9
    except OSError:
        gb = 0
    return ((f"On this card ({free:.0f} GB free): every part trains at once, nothing rotates - a rotation is "
             f"{per} epoch(s)" if len(windows) == 1 and isinstance(windows[0], list) else
             f"On this card ({free:.0f} GB free): a rotation is {len(windows)} window(s) x {per} epoch(s) = {cyc} "
             f"epochs" + (" (several parts train together in a window)" if any(isinstance(w, list) for w in windows)
                          else ""))
            + (" - parts outside the one training stream from system memory, so steps are slower" if stream else "")
            + f". {rot} rotation(s) = {total} epochs. Checkpoint + preview at epoch {shown}"
            + (f" (~{gb:.0f} GB each)" if gb else "") + ". The log confirms the plan at Start."), False


if __name__ == "__main__":        # the subprocess: [family, model path, megapixels, options json, max parts, batch GB]
    from fizgig.families.registry import get
    from fizgig.utils.device import plannable_free_vram
    fam, path, mp, opts, cap = sys.argv[1:6]
    free = plannable_free_vram()
    extra = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0      # Batch size: the pictures past the first
    plan = get(fam).load_driver().ft_card_plan(path, free - extra, mp=float(mp) if mp else None,
                                               options=json.loads(opts), max_parts=int(cap or 0))
    # a depth split (prefix, lo, hi) stays apart from a list of whole parts: JSON would make both lists
    wins = [{"split": list(w)} if isinstance(w, tuple) else w for w in plan[0]] if plan and plan[0] else None
    print(json.dumps([wins, plan[1], free] if wins else None))
