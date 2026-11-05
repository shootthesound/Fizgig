"""The Batch size row's plan note: what Auto would pick on this card at Batch size 1 and at the batch asked, from the
same planner the trainer runs at Start (quant.plan over the family's measured training memory, the free VRAM now,
Target Megapixels x pictures a step). The planner imports torch and reads the card, so it runs in a short subprocess
(`python -m fizgig.web.batch_plan ...`), never in the web server, as the fine-tune card's plan line does; answers are
kept a minute per setup, and line() says "working it out" until one is in."""
import json
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KEEP_S = 60                      # free VRAM moves (a game closes): a setup's answer is asked again after this
_lock = threading.Lock()
_cache = {}                      # key -> (time, answer | None | "running")


def request(desc, values):
    """(precision, blocks_to_swap, megapixels, batch) as Start would hand them to the trainer, or None when Auto
    plans nothing here (both chosen by hand, a batch of 1, or a family that doesn't batch)."""
    from fizgig.families import launch
    batch = launch.batch_size(desc, values)
    if batch <= 1:
        return None
    precision = launch.precision_key(desc, values.get("FAMILY_PRECISION")) if len(desc.precisions) > 1 else "bf16"
    swap = launch.swap_blocks(values.get("BLOCKS_SWAP"))
    if precision != "auto" and swap >= 0:
        return None
    try:
        mp = float(str(values.get("DATASET_MEGAPIXELS") or "").split()[0])
    except (ValueError, IndexError):
        return None
    if mp <= 0:
        return None
    side = launch.train_side(mp)
    return precision, swap, side * side / 1e6, batch


def note(desc, one, many, batch):
    """The row's note from the two plans ((precision, swap, free GB) at batch 1 and at the batch)."""
    labels = {"bf16": "bf16", "int8": "int8", "nf4": "4-bit NF4", "hqq": "4-bit HQQ", "fp8": "fp8"}
    (p1, s1, free), (pn, sn, _) = one, many
    where = f"On this card ({free:.0f} GB free)"
    if (p1, s1) == (pn, sn):
        return f"{where} batch {batch} fits as batch 1 does: {labels.get(p1, p1)}" + \
            (f", {s1} blocks swapped." if s1 else ", no block swap.")
    parts = []
    if pn != p1:
        parts.append(f"the base drops to {labels.get(pn, pn)} (batch 1: {labels.get(p1, p1)})")
    if sn != s1:
        parts.append(f"{sn} blocks are swapped (batch 1: {s1 or 'none'})")
    text = f"{where} batch {batch} needs more memory: " + " and ".join(parts) + "."
    if sn > s1:
        # a step's swap serves every picture in it: blocks swapped per picture is what sets the pace (A40, 5 Nov
        # 2026: Qwen batch 3 at 16 blocks 3.5 s a picture against batch 1 at 8 blocks 5.0 s, at none 1.2 s)
        per = sn / batch
        if per < s1 * 0.85:
            text += " Each swap serves every picture in the batch, so it should still be quicker per picture."
        elif per <= s1 * 1.15:
            text += " About as quick per picture as batch 1: the batch's extra swapping cancels its gain."
        else:
            text += " Batch 1 is likely faster here: the extra swapping costs more than batching saves."
    return text


def _run(key, family, req, env):
    precision, swap, mp, batch = req
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.batch_plan", family, precision, str(swap), str(mp),
                            str(batch)], capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        got = json.loads((r.stdout.strip().splitlines() or ["null"])[-1])
    except (OSError, ValueError, subprocess.SubprocessError):
        got = None
    with _lock:
        _cache[key] = (time.time(), got)


def line(desc, values, env, busy=False):
    """(note or "", pending) for the Batch size row. busy: a run is training on this card now, so its free memory
    says nothing about an empty card's - no note."""
    req = None if busy else request(desc, values)
    if req is None:
        return "", False
    key = (desc.key,) + req
    with _lock:
        at, got = _cache.get(key, (0, "new"))
        if got == "new" or (got != "running" and time.time() - at > KEEP_S):
            _cache[key] = (time.time(), "running")
            threading.Thread(target=_run, args=(key, desc.key, req, env), daemon=True).start()
            if got == "new":
                got = "running"
    if got == "running":
        return "Working out what this batch needs on this card…", True
    if not got:
        return "", False
    return note(desc, tuple(got[0]), tuple(got[1]), req[3]), False


def fit(desc, values, env):
    """The largest Batch size this card trains the run at with no more block swap than batch 1 and the same
    precision (a run moved to a pod that batches: worked out there, on its own card), 1 when none does or Auto
    plans nothing here. Blocks up to two minutes (the subprocess)."""
    from fizgig.families import launch
    if launch.batch_size(desc, {**values, "BATCH_SIZE": launch.BATCH_SIZES[-1]}) <= 1:
        return 1
    req = request(desc, {**values, "BATCH_SIZE": "2", "BLOCKS_SWAP": "Auto"})
    if req is None:
        return 1
    precision, swap, mp, _ = req
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.batch_plan", "fit", desc.key, precision, str(swap),
                            str(mp)], capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        return max(1, int(json.loads((r.stdout.strip().splitlines() or ["1"])[-1]) or 1))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return 1


if __name__ == "__main__":        # the subprocess: [family, precision, blocks_to_swap, megapixels, batch]
    import torch                  # or ["fit", family, precision, blocks_to_swap, megapixels]: the largest batch
    from fizgig.families import launch, quant
    from fizgig.families.registry import get
    fitting = sys.argv[1] == "fit"
    fam, precision, swap, mp, batch = (sys.argv[2:6] + ["1"]) if fitting else sys.argv[1:6]
    if not torch.cuda.is_available():
        print("null")
        raise SystemExit
    desc = get(fam)
    driver = desc.load_driver()
    free = quant.free_vram_gb()
    if fitting:
        p1, s1 = quant.plan(desc, driver, precision, int(swap), free_gb=free, megapixels=float(mp))[:2]
        best = 1
        for n in (int(b) for b in launch.BATCH_SIZES[1:]):
            pn, sn = quant.plan(desc, driver, precision, int(swap), free_gb=free, megapixels=float(mp) * n)[:2]
            if pn == p1 and sn <= s1:
                best = n
        print(json.dumps(best))
        raise SystemExit
    plans = [quant.plan(desc, driver, precision, int(swap), free_gb=free, megapixels=float(mp) * n)[:2]
             for n in (1, int(batch))]
    print(json.dumps([[*plans[0], free], [*plans[1], free]]))
