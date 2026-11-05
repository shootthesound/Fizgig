"""The Batch size row's plan note: what Auto would pick on this card at Batch size 1 and at the batch asked, from the
same planner the trainer runs at Start (quant.plan over the family's measured training memory, the free VRAM now,
Target Megapixels x pictures a step; a family with its own planner, description.own_plan, is asked through its driver's
plan_pictures with the run's model file and adapter settings). The planner imports torch and reads the card, so it runs in a short subprocess
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
    if batch <= 1 or not (desc.train_memory or desc.own_plan):     # no figures: no note
        return None
    if launch.ft_on(desc, values):        # a fine-tune plans its windows (the fine-tune card's line), not this
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


def models(desc, lookup):
    """{pref key: path} of the family's model files and training adapter, from lookup(pref key)."""
    keys = [f.pref_key for f in desc.model_files] + ([desc.training_adapter] if desc.training_adapter else [])
    return {k: lookup(k) or "" for k in keys}


def run_for(desc, values, files):
    """What an own_plan family's planner weighs (train.py's `run` for plan_run): the DiT file and the adapter's size
    settings as Start would send them; None for a family the shared figures plan, or when the DiT file isn't here."""
    if not desc.own_plan:
        return None
    dit = desc.model_path("dit", (files or {}).get)
    if not dit or not os.path.isfile(dit):
        return None
    ema = str(values.get("FAMILY_EMA", "") or desc.ema_default).split(" ")[0] if desc.ema_default else "Off"
    try:
        ema = 0.0 if ema in ("", "Off") else 0.98 if ema.lower().startswith("short") else float(ema)
    except ValueError:
        ema = 0.0
    lokr = "lokr" in desc.network_types and str(values.get("NETWORK_TYPE", "")).startswith("LoKR")
    adapter = desc.training_adapter and values.get("FAMILY_TRAINING_ADAPTER", True)
    return {"dit_path": dit, "network_type": "lokr" if lokr else "lora",
            "network_dim": int(values.get("NETWORK_DIM") or 16), "lokr_factor": int(values.get("LOKR_FACTOR") or 8),
            "optimizer_type": str(values.get("OPTIMIZER_TYPE") or "").strip(),
            "training_adapter": (files or {}).get(desc.training_adapter, "") if adapter else "",
            "context_lora_path": str(values.get("CONTEXT_LORA_PATH") or "").strip(), "ema_decay": ema}


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


def _run(key, family, req, env, run=None):
    precision, swap, mp, batch = req
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.batch_plan", family, precision, str(swap), str(mp),
                            str(batch)] + ([json.dumps(run)] if run else []), capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        got = json.loads((r.stdout.strip().splitlines() or ["null"])[-1])
    except (OSError, ValueError, subprocess.SubprocessError):
        got = None
    with _lock:
        _cache[key] = (time.time(), got)


def line(desc, values, env, busy=False, files=None):
    """(note or "", pending) for the Batch size row. busy: a run is training on this card now, so its free memory
    says nothing about an empty card's - no note. files: models(desc, ...), for an own_plan family."""
    req = None if busy else request(desc, values)
    run = run_for(desc, values, files) if req else None
    if req is None or (desc.own_plan and run is None):
        return "", False
    key = (desc.key,) + req + ((json.dumps(run, sort_keys=True),) if run else ())
    with _lock:
        at, got = _cache.get(key, (0, "new"))
        if got == "new" or (got != "running" and time.time() - at > KEEP_S):
            _cache[key] = (time.time(), "running")
            threading.Thread(target=_run, args=(key, desc.key, req, env, run), daemon=True).start()
            if got == "new":
                got = "running"
    if got == "running":
        return "Working out what this batch needs on this card…", True
    if not got:
        return "", False
    return note(desc, tuple(got[0]), tuple(got[1]), req[3]), False


def fit(desc, values, env, files=None):
    """The largest Batch size this card trains the run at with no more block swap than batch 1 and the same
    precision (a run moved to a pod that batches: worked out there, on its own card), 1 when none does or Auto
    plans nothing here. files: models(desc, ...), for an own_plan family. Blocks up to two minutes (the
    subprocess)."""
    from fizgig.families import launch
    if launch.batch_size(desc, {**values, "BATCH_SIZE": launch.BATCH_SIZES[-1]}) <= 1:
        return 1
    if launch.ft_on(desc, values):
        return _ft_fit(desc, values, env, files)
    if not (desc.train_memory or desc.own_plan):     # no figures to fit a batch by
        return 1
    req = request(desc, {**values, "BATCH_SIZE": "2", "BLOCKS_SWAP": "Auto"})
    if req is None:
        return 1
    precision, swap, mp, _ = req
    run = run_for(desc, values, files)
    if desc.own_plan and run is None:
        return 1
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.batch_plan", "fit", desc.key, precision, str(swap),
                            str(mp)] + ([json.dumps(run)] if run else []), capture_output=True, text=True,
                           timeout=120, cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        return max(1, int(json.loads((r.stdout.strip().splitlines() or ["1"])[-1]) or 1))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return 1


def _ft_fit(desc, values, env, files):
    """A fine-tune moved to a pod: the largest Batch size whose window plan on this card is batch 1's (the same
    windows, nothing newly streamed), each picture past the first planned off the free memory as the trainer does."""
    from fizgig.families import launch
    from fizgig.web import ft_plan
    dit = desc.model_path("dit", (files or {}).get)
    if not dit or not os.path.isfile(dit):
        return 1
    try:
        mp = float(str(values.get("DATASET_MEGAPIXELS") or "").split()[0])
    except (ValueError, IndexError):
        mp = 0.25
    cap = launch.ft_max_parts({"FAMILY_FT_MAX_PARTS": values.get("FAMILY_FT_MAX_PARTS", "")})
    try:
        r = subprocess.run([sys.executable, "-m", "fizgig.web.batch_plan", "ftfit", desc.key, dit, str(mp),
                            json.dumps(ft_plan._options(desc, values)), str(cap)], capture_output=True, text=True,
                           timeout=180, cwd=BASE_DIR, env=env,
                           **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        return max(1, int(json.loads((r.stdout.strip().splitlines() or ["1"])[-1]) or 1))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return 1


if __name__ == "__main__" and sys.argv[1] == "ftfit":      # [ftfit, family, model path, megapixels, options, cap]
    from fizgig.families import launch, quant
    from fizgig.families.registry import get
    from fizgig.utils.device import plannable_free_vram
    fam, path, mp, opts, cap = sys.argv[2:7]
    desc = get(fam)
    driver, free = desc.load_driver(), plannable_free_vram()

    def ft_plan_at(n):
        p = driver.ft_card_plan(path, free - quant.batch_extra(desc, float(mp), n), mp=float(mp),
                                options=json.loads(opts), max_parts=int(cap or 0))
        return json.dumps([[list(w) if isinstance(w, tuple) else w for w in p[0]], bool(p[1])]) if p and p[0] else None
    one, best = ft_plan_at(1), 1
    for n in (int(b) for b in launch.BATCH_SIZES[1:]):
        if one is not None and ft_plan_at(n) == one:
            best = n
    print(json.dumps(best))
    raise SystemExit


if __name__ == "__main__":        # the subprocess: [family, precision, blocks_to_swap, megapixels, batch, run?]
    import torch                  # or ["fit", family, precision, blocks_to_swap, megapixels, run?]: the largest batch
    from fizgig.families import launch, quant
    from fizgig.families.registry import get
    fitting = sys.argv[1] == "fit"
    fam, precision, swap, mp, batch = (sys.argv[2:6] + ["1"]) if fitting else sys.argv[1:6]
    run = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None      # an own_plan family's run_for()
    if not torch.cuda.is_available():
        print("null")
        raise SystemExit
    desc = get(fam)
    driver = desc.load_driver()
    free = quant.free_vram_gb()

    def plan(n):
        if run is not None:
            return tuple(driver.plan_pictures(precision, int(swap), megapixels=float(mp), pictures=n, run=run))
        return tuple(quant.plan(desc, driver, precision, int(swap), free_gb=free, megapixels=float(mp),
                                pictures=n)[:2])

    if fitting:
        p1, s1 = plan(1)
        best = 1
        for n in (int(b) for b in launch.BATCH_SIZES[1:]):
            pn, sn = plan(n)
            if pn == p1 and sn <= s1 and quant.fits(desc, pn, sn, free, float(mp), n):
                best = n
        print(json.dumps(best))
        raise SystemExit
    plans = [plan(n) for n in (1, int(batch))]
    print(json.dumps([[*plans[0], free], [*plans[1], free]]))
