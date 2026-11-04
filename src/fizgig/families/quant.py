"""Quantised frozen bases for described families: INT8 (W8A8) and 4-bit NF4, on the Linears of the driver's block map.

Reuses Fizgig's existing quantisers (modules/int8_train.py, modules/nf4.py): each keeps the nn.Linear and patches its
forward, so the family LoRA layer wraps a quantised Linear exactly like a bf16 one. What gets quantised is the block
map's modules (the same Linears a LoRA targets); everything outside it (modulation, embedders, norms, output) stays
bf16. The base is loaded on CPU first and quantised one Linear at a time on the GPU, so the full bf16 model is never
resident.

    bf16  the model as shipped
    int8  8-bit weights + int8 matmuls (forward int8, backward bf16) - about half the bf16 size, fastest
    nf4   4-bit weights dequantised per matmul - about a quarter, slowest, cannot block-swap
"""
import gc
import math
import os
import logging

import torch

logger = logging.getLogger(__name__)

PRECISIONS = ("bf16", "int8", "nf4", "hqq")       # hqq: a family whose driver loads it (H3)


def _targets(dit, driver):
    names = set(driver.quant_target_names(dit))
    return [(n, m) for n, m in dit.named_modules() if n in names and isinstance(m, torch.nn.Linear)]


@torch.no_grad()
def quantize(dit, driver, precision, compute_device, store_device=None):
    """Quantise the block map's Linears in place. Returns the number quantised (0 for bf16). INT8 weights are
    quantised on `compute_device` and kept on `store_device` when given (the CPU, for a block-swapped base: the swap
    streams them in, where a fully resident INT8 base would have to fit before swapping could start)."""
    if precision == "bf16":
        return 0
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    compute_device = torch.device(compute_device)
    targets = _targets(dit, driver)
    for _, m in targets:
        _quantize_one(m, driver, precision, compute_device, store_device)
    if precision == "int8":
        dit._int8_quantized = True
    else:
        dit._nf4_quantized = True
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[precision] {precision}: {len(targets)} block Linears quantised; everything else stays bf16")
    return len(targets)


@torch.no_grad()
def _quantize_one(m, driver, precision, compute_device, store_device=None):
    """One block-map Linear to `precision` ("int8" or "nf4"), in place - the step quantize() takes for each."""
    from fizgig.modules.nf4 import _dequantize_source_weight
    if precision == "int8":
        from fizgig.modules.int8_train import int8_train_forward
        w = _dequantize_source_weight(m).to(compute_device)
        # a driver may keep its original trainer's bf16 scales
        w = w.float() if getattr(driver, "int8_fp32_scales", True) else w.contiguous()
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        m.weight.requires_grad_(False)
        q = (w / scale).round_().clamp_(-127, 127).to(torch.int8).contiguous()
        m.weight.data = q if store_device is None else q.to(store_device)
        m.register_buffer("_int8_wscale", scale.reshape(1, -1).to(torch.float32), persistent=False)
        m._is_int8 = True
        m._int8_grad_mode = "bf16"
        m.forward = int8_train_forward.__get__(m, type(m))
        del w
    else:
        from bitsandbytes.functional import quantize_nf4
        from fizgig.modules.nf4 import nf4_linear_forward_patch
        packed, state = quantize_nf4(_dequantize_source_weight(m).to(compute_device).contiguous())
        m._nf4_packed, m._nf4_state, m._is_nf4 = packed, state, True
        m.weight.data = torch.empty(0, device=compute_device, dtype=torch.bfloat16)
        m.weight.requires_grad_(False)
        m.forward = nf4_linear_forward_patch.__get__(m, type(m))


STREAM_HEADROOM_GB = 6.0     # what the rest of the run takes in RAM beside the model file while it loads


def _ram_short(driver, path):
    """The bf16 model file would not fit in the RAM that's free (on a pod, its own limit, via container_ram): the
    one case the base is streamed. With RAM to spare the whole-file load runs as it always has."""
    try:
        import psutil
        return driver.dit_gb(path) + STREAM_HEADROOM_GB > psutil.virtual_memory().available / 1e9
    except Exception:
        return False


@torch.no_grad()
def _stream_load(driver, path, device, precision, swap):
    """The base built a tensor at a time, each block-map Linear quantised as soon as its weight is read, so the
    bf16 model is never whole in RAM. Same result as load_dit on the CPU + quantize(); None when the driver can't
    stream this file (it then loads the usual way)."""
    got = driver.stream_dit(path)
    if got is None:
        return None
    dit, f, dtype = got                    # the model on meta, the open file, the dtype load_dit gives its weights
    with f:
        _stream_fill(dit, driver, f, dtype, path, device, precision, swap)
    dit.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[precision] {precision}: {len(_targets(dit, driver))} block Linears quantised; everything else stays bf16")
    return dit


def _stream_fill(dit, driver, f, dtype, path, device, precision, swap):
    keys = f.keys()
    read = lambda k: f.get_tensor(k, device=torch.device("cpu"), dtype=dtype)    # noqa: E731 - as load_dit reads
    expected = set(dit.state_dict().keys())
    if set(keys) != expected:
        raise RuntimeError(f"{os.path.basename(path)}: {len(set(keys) - expected)} unexpected and "
                           f"{len(expected - set(keys))} missing weights for this model")
    targets = dict(_targets(dit, driver))
    owned = {}
    for k in keys:
        mod = k.rpartition(".")[0]
        if mod in targets:
            owned.setdefault(mod, []).append(k)
    rest = [k for k in keys if k.rpartition(".")[0] not in targets]
    logger.info(f"[precision] RAM is short for the whole {driver.dit_gb(path):.0f} GB model: "
                f"loading it a Linear at a time and quantising each to {precision} as it arrives")
    dit.load_state_dict({k: read(k) for k in rest}, strict=False, assign=True)
    compute_device, store_device = torch.device(device), ("cpu" if swap else None)
    for name, m in targets.items():
        m.load_state_dict({k.rpartition(".")[2]: read(k) for k in owned.get(name, ())}, strict=True, assign=True)
        _quantize_one(m, driver, precision, compute_device, store_device)
    if precision == "int8":
        dit._int8_quantized = True
    else:
        dit._nf4_quantized = True


def move(dit, device):
    """dit.to(device), plus NF4's packed weights (plain attributes .to() does not see)."""
    dit.to(device)
    if getattr(dit, "_nf4_quantized", False):
        from fizgig.modules.nf4 import move_nf4_to_device
        move_nf4_to_device(dit, device)


def load_base(driver, path, device, precision="bf16", blocks_to_swap=0, supports_backward=True):
    """The family's DiT at `precision`, optionally block-swapped. NF4 cannot swap (its weights are not .weight), so
    swap is dropped for it; a family whose driver has no block swap ignores the request. Returns (dit, swapped)."""
    swap = int(blocks_to_swap or 0)
    own = driver.load_planned(path, device, precision, swap)
    if own is not None:                               # the driver loads its own tiers (H3: ConvRot int8 / NF4 / HQQ rings)
        return own
    if swap and precision == "nf4":
        logger.info("[block swap] off: NF4 weights cannot stream (the 4-bit base is small enough not to need it)")
        swap = 0
    if swap:
        cap = driver.max_blocks_to_swap()
        if cap <= 0:
            logger.info("[block swap] this family has no block swap; loading without it")
            swap = 0
        elif swap > cap:
            logger.info(f"[block swap] {swap} requested, {cap} is the maximum")
            swap = cap
    if getattr(driver, "loads_quantized", False):     # the file is already the base precision
        dit = driver.load_dit(path, device)
        return dit, 0
    dit = None
    if precision != "bf16" and hasattr(driver, "stream_dit") and _ram_short(driver, path):
        dit = _stream_load(driver, path, device, precision, swap)
    if dit is None:
        dit = driver.load_dit(path, "cpu" if (swap or precision != "bf16") else device)
        quantize(dit, driver, precision, device, store_device="cpu" if swap else None)
    if swap:
        driver.enable_block_swap(dit, swap, device, supports_backward)
        logger.info(f"[block swap] {swap} blocks stream between CPU and GPU")
    elif precision != "bf16":
        move(dit, device)
    return dit, swap


def _peak(entry, megapixels):
    """Peak GB for a train_memory entry at `megapixels`: a number, or ((mp, GB), ...) points interpolated linearly
    (and extrapolated from the nearest pair) by the run's resolution."""
    peak = entry[0]
    if not isinstance(peak, (tuple, list)):
        return float(peak)
    pts = sorted(peak)
    if len(pts) == 1:
        return float(pts[0][1])
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if megapixels <= pts[1][0] else (pts[-2], pts[-1])
    for a, b in zip(pts, pts[1:]):
        if a[0] <= megapixels <= b[0]:
            (x0, y0), (x1, y1) = a, b
    return y0 + (y1 - y0) * (megapixels - x0) / (x1 - x0)


TIGHT_MARGIN_GB = 0.5      # nf4 with no swap beats a swapped int8 base down to this much headroom


def plan(desc, driver, precision="auto", blocks_to_swap=-1, free_gb=None, margin_gb=1.5, megapixels=1.0):
    """Resolve 'auto' precision and/or block swap (-1) from free VRAM and the description's measured training
    memory (desc.train_memory: {precision: (peak GB at no swap, GB saved per swapped block)}).

    Auto precision takes the most precise base that fits without swap (bf16, then int8, then nf4), since swap costs
    far more speed than quantisation. If none fits with the full margin, nf4 still goes first when it fits with a
    small one: swapping int8 instead cost 4.4x the step time and 4x the CPU on a 16 GB card (#216; measured July
    2026). Only if nothing fits does it swap (int8, the smaller swappable base). An explicit precision with auto swap
    gets the fewest blocks that fit. Returns (precision, blocks_to_swap, reason)."""
    if free_gb is None:
        free_gb = free_vram_gb()
    mem = {p: (_peak(v, megapixels), v[1]) for p, v in (desc.train_memory or {}).items()}
    offered = [p for p in PRECISIONS if p in desc.precisions]
    if precision == "auto" and getattr(desc, "auto_precisions", ()):
        offered = [p for p in desc.auto_precisions if p in desc.precisions]
    cap = driver.max_blocks_to_swap()
    budget = free_gb - margin_gb

    def swap_for(p):
        if p not in mem or p == "nf4":
            return 0
        need, per_block = mem[p]
        if need <= budget or per_block <= 0 or cap <= 0:
            return 0
        return min(cap, math.ceil((need - budget) / per_block))

    if precision == "auto":
        if not mem:
            return offered[0], max(0, blocks_to_swap), "no memory figures in the description: first precision"
        for p in offered:
            if p in mem and mem[p][0] <= budget:
                return p, 0 if blocks_to_swap < 0 else blocks_to_swap, f"{p} fits {free_gb:.1f} GB free"
        if "nf4" in offered and "nf4" in mem and mem["nf4"][0] <= free_gb - TIGHT_MARGIN_GB:
            return "nf4", 0 if blocks_to_swap < 0 else blocks_to_swap, (f"nf4 fits {free_gb:.1f} GB free, closely "
                                                                        "(block swap would be far slower)")
        swappable = [p for p in ("int8", "bf16") if p in offered and p in mem]
        if swappable and cap > 0:
            p = swappable[0]
            n = swap_for(p)
            if mem[p][0] - n * mem[p][1] <= budget:
                return p, n if blocks_to_swap < 0 else blocks_to_swap, f"{p} + {n} swapped blocks fit {free_gb:.1f} GB"
        p = "nf4" if "nf4" in offered else offered[-1]
        return p, 0, f"nothing fits {free_gb:.1f} GB comfortably: {p}, the smallest base"
    if precision not in offered:
        raise ValueError(f"{desc.display_name} offers {offered}, not {precision!r}")
    if blocks_to_swap < 0:
        n = swap_for(precision)
        return precision, n, (f"{n} swapped blocks to fit {free_gb:.1f} GB" if n else f"fits {free_gb:.1f} GB")
    return precision, blocks_to_swap, "as set"


def share_shortfall(desc, driver, precision, blocks_to_swap, megapixels, free_gb=None, margin_gb=0.5):
    """Why `precision` with `blocks_to_swap` swapped won't fit this card ("needs ~X GB, Y GB free"), or "" when it
    does or the description has no figure for it - read from the same train_memory figures Auto plans with."""
    if precision not in (desc.train_memory or {}):
        return ""
    need, per_block = _peak(desc.train_memory[precision], megapixels), desc.train_memory[precision][1]
    if free_gb is None:
        free_gb = free_vram_gb()
    left = need - max(0, blocks_to_swap) * per_block
    if left <= free_gb - margin_gb:
        return ""
    return (f"needs ~{left:.1f} GB with {max(0, blocks_to_swap)} blocks swapped at {megapixels:.2f} MP, "
            f"{free_gb:.1f} GB free")


NOGC_MIN_CARD_GB = 38.0   # big cards only (40 GB and up; a 40 GB A100 reports ~39.4 GiB) - 32 GB and smaller always checkpoint
NOGC_MARGIN_GB = 4.0     # over the measured no-checkpointing peak: rank, adapters, EMA, a Context LoRA, a busy desktop
NOGC_MAX_RANK = 32       # rank 32 measured +0.6-1.8 GB over the figures (inside the margin); higher ranks unmeasured


def card_gb():
    """The card's size in GiB - the simulated card's when FIZGIG_SIM_VRAM_GB is set."""
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if sim:
        return float(sim)
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def plan_checkpointing(desc, precision, blocks_to_swap, megapixels, free_gb=None, compiled=True, kind="", rank=8):
    """Gradient checkpointing on or off: (on, reason). Off only on a big card (NOGC_MIN_CARD_GB+), for a compiled run with no block swap whose measured
    no-checkpointing peak (desc.train_memory_nogc, at the largest bucket) fits free VRAM with NOGC_MARGIN_GB to spare.
    The figures are compiled runs - uncompiled blocks without checkpointing take more - and a swapped block's weights
    would stay pinned through backward without the recompute."""
    key = f"{precision}:{kind}" if kind else precision
    pts = (getattr(desc, "train_memory_nogc", None) or {}).get(key)
    if not pts:
        return True, f"no measured figure for {key} without it"
    if blocks_to_swap > 0:
        return True, "blocks are swapped"
    if not compiled and desc.compiles:       # a family that never compiles was measured uncompiled
        return True, "the blocks aren't compiled this run"
    if card_gb() < NOGC_MIN_CARD_GB:
        return True, f"a {card_gb():.0f} GB card (only {NOGC_MIN_CARD_GB:.0f} GB+ cards go without it)"
    if rank > NOGC_MAX_RANK:
        return True, f"on: rank {rank} is past the measured ranks (up to {NOGC_MAX_RANK})"
    if free_gb is None:
        free_gb = free_vram_gb()
    if megapixels > max(p[0] for p in pts) + 0.01:
        return True, f"on: {megapixels:.2f} MP is past the measured sizes"
    need = _peak((pts, 0.0), max(megapixels, min(p[0] for p in pts)))    # below the first point: that point's peak
    if need + NOGC_MARGIN_GB <= free_gb:
        return False, f"off: needs ~{need:.1f} GB at {megapixels:.2f} MP, {free_gb:.1f} GB free"
    return True, f"on: off would need ~{need + NOGC_MARGIN_GB:.1f} GB at {megapixels:.2f} MP, {free_gb:.1f} GB free"


def plan_partial_checkpointing(desc, precision, blocks_to_swap, megapixels, free_gb=None, compiled=True, kind="",
                               rank=8):
    """How many of the last blocks train without gradient checkpointing when full off doesn't fit a big card: (n, why).
    Read from desc.train_memory_partial {key: (blocks, ((megapixels, GB with every block checkpointed, GB per block
    without it), ...))} - reserved peaks, compiled with the checkpoint outside the graph, previews on - at the first
    measured size at or above the run's (never between or past them), with NOGC_MARGIN_GB to spare. 0 = all on."""
    key = f"{precision}:{kind}" if kind else precision
    fig = (getattr(desc, "train_memory_partial", None) or {}).get(key)
    if (not fig or blocks_to_swap > 0 or (not compiled and desc.compiles) or card_gb() < NOGC_MIN_CARD_GB
            or rank > NOGC_MAX_RANK):
        return 0, ""
    blocks, pts = fig
    at = [p for p in pts if p[0] + 0.01 >= megapixels]
    if not at:
        return 0, ""
    mp, on_gb, per_block = min(at)
    if free_gb is None:
        free_gb = free_vram_gb()
    n = max(0, min(blocks - 1, int((free_gb - NOGC_MARGIN_GB - on_gb) // per_block)))
    if not n:
        return 0, ""
    return n, (f"partly off: the last {n} of {blocks} blocks without it (~{on_gb + n * per_block:.1f} GB at "
               f"{megapixels:.2f} MP, {free_gb:.1f} GB free)")


_SIM_OTHERS_GB = 0.8     # a simulated card also loses this to the desktop / other apps, as real cards do


def apply_vram_cap():
    """FIZGIG_SIM_VRAM_GB=N: behave like an N GB card - cap this process's allocations at N minus what the desktop
    takes, so a plan that does not fit fails as it would on the real card. No-op without the variable."""
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if not sim or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(min(1.0, (float(sim) - _SIM_OTHERS_GB) / total))
    logger.info(f"[sim] behaving like a {float(sim):g} GB card ({float(sim) - _SIM_OTHERS_GB:.1f} GB for Fizgig)")


def free_vram_gb():
    """Free VRAM in GB - of the simulated card when FIZGIG_SIM_VRAM_GB is set."""
    if not torch.cuda.is_available():
        return 0.0
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if sim:
        return max(0.0, float(sim) - _SIM_OTHERS_GB - torch.cuda.memory_reserved() / 1024 ** 3)
    return torch.cuda.mem_get_info()[0] / 1024 ** 3
