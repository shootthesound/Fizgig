"""What this machine can actually do, probed once and cached.

Fizgig used to pick memory settings from VRAM alone, which produced a bad outcome on 16 GB
cards: fp8 doesn't fit, so it fell back to swapping 20 of 28 blocks to CPU every step. Measured
on an RTX 5090 (Krea 2, 36 images @ 0.25 MP, batch 1):

    fp8, no swap    0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20    3.09 s/it   12.3 GB   49.9% CPU     <- what 16 GB cards were getting
    NF4, no swap    0.70 s/it   13.8 GB   14.0% CPU

Block swap costs 4.4x the time and 4x the CPU, and NF4 fits the same card outright. So the
choice is a *strategy*, not a swap count — and it needs to know what the hardware supports,
because fp8 matmul is Ada+ while NF4 and int8 go back further.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

from .gpu_backend import is_rocm

logger = logging.getLogger(__name__)


# torch major.minor -> the triton major.minor releases it is known to work with (the
# version the Linux torch wheel depends on, and the one before it, which the field has
# proven). A triton built for a newer torch imports fine and then fails or hangs INSIDE
# torch.compile, so an import check alone lets it through — that is how a fresh install
# with an unpinned triton-windows 3.8 on torch 2.10 stalled Krea 2 previews with no log.
_TRITON_FOR_TORCH = {
    "2.8": ("3.3", "3.4"),
    "2.9": ("3.4", "3.5"),
    "2.10": ("3.5", "3.6"),
    "2.11": ("3.6", "3.7"),
    "2.12": ("3.7", "3.8"),
}


def _major_minor(v: str) -> str:
    parts = str(v or "").split("+")[0].split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(v or "")


def triton_matches_torch(triton_version=None, torch_version=None):
    """(ok, note): whether this triton is one torch is known to work with. Unknown torch
    versions are not gated (ok, ""). Versions default to the installed ones."""
    try:
        if torch_version is None:
            import torch as _t
            torch_version = _t.__version__
        if triton_version is None:
            import triton as _tr
            triton_version = getattr(_tr, "__version__", "")
    except Exception:
        return True, ""
    want = _TRITON_FOR_TORCH.get(_major_minor(torch_version))
    if not want:
        return True, ""
    have = _major_minor(triton_version)
    if have in want:
        return True, ""
    return False, (f"triton {triton_version} does not pair with torch {torch_version} "
                   f"(torch {_major_minor(torch_version)} needs triton {' or '.join(want)}; "
                   f"a mismatched triton fails inside torch.compile, sometimes silently) — "
                   f"reinstall with: pip install \"triton-windows>={want[0]}.0,<{float(want[1]) + 0.1:.1f}\"")


def has_host_c_compiler(platform: Optional[str] = None) -> bool:
    """POSIX: is a C compiler on PATH? inductor/triton build small host-side stubs with one at
    runtime, so torch.compile without it dies with "Failed to find C compiler" — which is what
    happened on RunPod, where the runtime image ships no toolchain. Windows always returns True
    here: MSVC lives outside PATH by design and the trainer's vcvars import handles it."""
    if (platform or os.name) == "nt":
        return True
    return any(shutil.which(c) for c in ("cc", "gcc", "clang"))


@dataclass
class Capabilities:
    has_cuda: bool = False
    is_rocm: bool = False
    device_name: str = "cpu"
    sm: tuple = (0, 0)
    vram_gb: float = 0.0        # card total, as reported
    vram_free_gb: float = 0.0   # actually available right now — what decisions must use
    fp8_matmul: bool = False       # torch._scaled_mm on fp8 — Ada (sm 8.9) and newer
    int8_matmul: bool = False      # torch._scaled_mm on int8 — NOT a thing; fp8-only API
    int8_matmul_train: bool = False  # torch._int_mm — the real int8 GEMM, Turing and newer
    cudnn_attention: bool = False  # PyTorch SDPA cuDNN backend
    flash_attn: bool = False       # the flash_attn package
    bitsandbytes: bool = False     # required for NF4
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.has_cuda:
            return "no GPU device"
        backend = "ROCm" if self.is_rocm else "CUDA"
        flags = [backend]
        if not self.is_rocm:
            flags.append(f"sm_{self.sm[0]}{self.sm[1]}")
        used = self.vram_gb - self.vram_free_gb
        vram = (f"{self.vram_free_gb:.1f} GB free of {self.vram_gb:.0f} GB"
                + (f" ({used:.1f} GB already in use)" if used > 1.0 else ""))
        for name, ok in (("fp8", self.fp8_matmul), ("int8", self.int8_matmul_train),
                         ("cuDNN-attn", self.cudnn_attention), ("flash", self.flash_attn),
                         ("nf4", self.bitsandbytes)):
            flags.append(f"{name} {'yes' if ok else 'no'}")
        return f"{self.device_name}, {vram} — " + " · ".join(flags)


def _probe_scaled_mm(dtype) -> bool:
    """Actually run a tiny _scaled_mm rather than trusting a compute-capability table."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        a = torch.zeros((16, 16), dtype=dtype, device="cuda")
        b = torch.zeros((16, 16), dtype=dtype, device="cuda").t()
        one = torch.ones((), dtype=torch.float32, device="cuda")
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


def _probe_int_mm() -> bool:
    """torch._int_mm is the int8 GEMM — a different API from _scaled_mm, which is fp8-only.
    Confusing the two is why int8 first looked unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        a = torch.zeros((32, 32), dtype=torch.int8, device="cuda")
        torch._int_mm(a, a.t().contiguous())
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def detect() -> Capabilities:
    caps = Capabilities()
    try:
        import torch
    except Exception:
        caps.notes.append("torch not importable")
        return caps

    if not torch.cuda.is_available():
        caps.notes.append("GPU unavailable")
        return caps

    caps.has_cuda = True
    caps.is_rocm = is_rocm()
    props = torch.cuda.get_device_properties(0)
    caps.device_name = props.name
    caps.sm = torch.cuda.get_device_capability(0)
    caps.vram_gb = props.total_memory / (1024 ** 3)
    try:
        # What is ACTUALLY available: a "16 GB" card reports ~15.9 GiB total, and a browser or
        # a running ComfyUI can be holding several more. Deciding from total would hand those
        # users a config that OOMs or silently falls back to swapping.
        free_b, _total_b = torch.cuda.mem_get_info(0)
        caps.vram_free_gb = free_b / (1024 ** 3)
    except Exception:
        caps.vram_free_gb = caps.vram_gb
        caps.notes.append("could not read free VRAM — using card total")

    caps.fp8_matmul = _probe_scaled_mm(torch.float8_e4m3fn) if not caps.is_rocm else False
    caps.int8_matmul = _probe_scaled_mm(torch.int8)     # expected False: _scaled_mm is fp8-only
    caps.int8_matmul_train = _probe_int_mm()

    if caps.is_rocm:
        caps.cudnn_attention = False
    else:
        try:    # cuDNN SDPA backend: present from PyTorch 2.5-ish, Ampere and newer
            from torch.backends.cuda import can_use_cudnn_attention  # noqa: F401
            caps.cudnn_attention = True
        except Exception:
            caps.cudnn_attention = hasattr(__import__("torch").backends.cuda, "cudnn_sdp_enabled")

    try:
        import flash_attn  # noqa: F401
        caps.flash_attn = True
    except Exception:
        pass

    try:
        import bitsandbytes  # noqa: F401
        caps.bitsandbytes = True
    except Exception:
        caps.notes.append("bitsandbytes missing — NF4 unavailable")

    return caps


# TRAINING-ONLY footprints at 0.25 MP, batch 1: the measured peaks (20.1 / 13.8 GB) were
# whole-GPU readings on a desktop already holding ~2.4 GB, so they overstate what training
# needs. Headroom then covers the user's own desktop plus allocator slack.
#
# The budget is FREE VRAM, not the number on the box: a "16 GB" card reports ~15.9 GiB total
# and may have several GB already held by a browser or a running ComfyUI. Deciding from total
# is how 16 GB cards ended up 4.4x slower — the config looked like it fit, and didn't.
# 18.7 is MEASURED, not derived: the real trainer on a real dataset under a ballast-constrained
# 14.5 GB budget gave 13.66 / 12.08 / 10.28 GB at swap 12 / 16 / 20, a straight line whose swap-0
# intercept is 18.7. The old 17.7 was a whole-GPU reading minus an assumed 2.4 GB desktop, and
# being 1 GB light is what made the auto plan choose 12 blocks where 12 does not actually fit.
_FP8_PEAK_GB = 18.7
_NF4_PEAK_GB = 11.4
# INT8 keeps the full 12.9B at one byte per weight, so it is ~5 GB above NF4 — measured 18.6 GB
# whole-GPU, ~16.2 GB training-only. It buys ~11% speed AND ~7x lower forward error than NF4
# (1.3e-02 vs 9.2e-02: 8-bit beats 4-bit), so it leads wherever it fits.
_INT8_PEAK_GB = 16.2
# Smaller than it looks: the budget is FREE VRAM, which already excludes whatever else is
# resident, so this only has to cover allocator slack and fragmentation.
_HEADROOM_GB = 1.5

# Run-shape terms, measured on a 5090 (36-image grid, 28 Jul 2026; whole-GPU peaks minus the
# ~1 GB desktop baseline; gradient checkpointing on, as the trainers force):
#   batch      +2.4 GB per extra image — flat across 0.25–1.05 MP, and by far the largest
#              term (the old single-constant budget's blind spot: batch 2 sailed through the
#              check and OOM'd).
#   resolution +0.15 GB from 0.25 → 1.05 MP at batch 1 (checkpointing absorbs it); budgeted
#              at 0.25 GB/MP for slack.
#   rank       +0.35 GB from r8 → r32 (~15 MB/rank); bases are measured AT rank 32.
_BATCH_GB_PER_IMAGE = 2.4
_RES_GB_PER_MP = 0.25
_RANK_GB_PER_RANK = 0.015

# Block swap: GB of peak removed per swapped block, measured on the real trainer under an
# emulated 16 GB budget (fp8, 0.25 MP, batch 1, rank 32; ballast-constrained 5090, 28 Jul):
#
#     swap 12 -> 13.66 GB     swap 16 -> 12.08 GB     swap 20 -> 10.28 GB
#
# A straight line through those: peak ~= 18.7 - 0.42 * swap, which also predicted the swap-16
# point to 0.1 GB before it was run. Krea 2 has 28 main blocks and the offloader keeps 2
# resident, so 26 is the ceiling.
_SWAP_GB_PER_BLOCK = 0.42
_MAX_SWAP_KREA2 = 26


def swap_for_budget(need_gb: float, free_gb: float, headroom_gb: float = None) -> int:
    """Fewest swapped blocks that fit `need_gb` into `free_gb`, keeping `headroom_gb` spare.

    Deliberately the FEWEST: every swapped block is a PCIe round-trip per step (~4x step time
    at heavy swap), so over-swapping is a real speed cost, not free safety. Returns 0 when it
    already fits, and the ceiling when even max swap cannot get there (the caller warns).
    """
    headroom = _HEADROOM_GB if headroom_gb is None else headroom_gb
    budget = free_gb - headroom
    if need_gb <= budget:
        return 0
    import math
    return min(_MAX_SWAP_KREA2, int(math.ceil((need_gb - budget) / _SWAP_GB_PER_BLOCK)))


def _lokr_extra_gb(factor: int) -> float:
    """Trainable-state cost of LoKR beyond the rank-32 LoRA baseline the peak constants were
    measured with. Full-matrix LoKR params scale ~1/factor² (measured: factor 8 ≈ 200M params,
    a 400 MB bf16 file on Krea 2's 264 Linears); param + grad (bf16) + two 8-bit Adam states
    ≈ 6 bytes/param. The baseline already carries ~0.6 GB of rank-32 LoRA state, so only the
    excess counts — zero at factor 16+, ~0.6 GB at the default factor 8, ~4 GB at factor 4,
    which is exactly the size that breaks a 16 GB NF4 fit if unmodelled."""
    f = int(factor) if factor and int(factor) >= 1 else 8
    params_m = 200.0 * (8.0 / f) ** 2
    return max(0.0, params_m * 6.0 / 1000.0 - 0.6)


def estimate_krea2_peak(base_gb: float, mp: float = 0.25, batch: int = 1,
                        rank: int = 32, network_type: str = "lora",
                        lokr_factor: int = 8) -> float:
    """Peak VRAM estimate for a Krea 2 run of this shape (base measured at 0.25 MP, b1, r32)."""
    return (base_gb
            + _BATCH_GB_PER_IMAGE * max(0, int(batch) - 1)
            + _RES_GB_PER_MP * max(0.0, float(mp) - 0.25)
            + _RANK_GB_PER_RANK * max(0, int(rank) - 32)
            + (_lokr_extra_gb(lokr_factor) if network_type == "lokr" else 0.0))


@dataclass
class MemoryStrategy:
    quant_4bit: bool
    blocks_to_swap: int
    reason: str
    quant_int8: str = ""     # "" | "bf16" — W8A8 base with exact bf16 gradients


# --- Rotating fine-tune window sizing ------------------------------------------------
# Measured peak reserved VRAM on a 5090, 130 steps (past a rotation boundary — component's
# peak IS the window switch, ~7 GB above steady state, so a shorter probe understates it
# badly). Requirements add ~2 GB on top for the CUDA context, which torch's `reserved`
# figure excludes, plus a little headroom.
#
#   component            27.67 GB   every window spans all 28 blocks (quality-preferred)
#   block 8 + streaming  20.71 GB
#   block 4 + streaming  18.70 GB
#   block 2 + streaming  17.62 GB
#
# HISTORY (27 Aug): this table used to drive the auto pick, and its component row
# excluded streaming on the reasoning that every block holds a trainable slice.
# Depth-split component windows (the small-card tiers) dissolved that premise — an
# out-of-window block is fully frozen and streamable — so auto now stays in component
# mode at every depth (see recommend_ft_rotation) and the trainer's window planner
# does the exact sizing at launch. The block rows remain for the explicit Window
# dropdown choice, quality-untested as ever.
FT_ROTATION_TIERS = [
    # (min_free_gb, mode,        blocks, stream, measured_peak_gb)
    (29.5, "component", 14, False, 27.67),
    (22.5, "block",      8, True,  20.71),
    (20.5, "block",      4, True,  18.70),
    (19.5, "block",      2, True,  17.62),
]

# Component-mode auto ladder: (min_free_gb, stream, narrative). The window planner in
# the trainer does the exact split arithmetic from the model's own Linear sizes; these
# thresholds only decide the MODE and set expectations in the console.
FT_COMPONENT_LADDER = [
    (29.5, False, "full-depth component windows (the classic 4-window cycle)"),
    (18.5, False, "component mode with depth-split windows — fat windows train in "
                  "slices, still full speed"),
    (12.5, True,  "component mode with depth-split windows + frozen-block streaming "
                  "from RAM — steps slower for the PCIe trips"),
]


def recommend_ft_rotation(free_gb: Optional[float] = None):
    """Pick a rotating fine-tune window config that fits the free VRAM.

    Returns (mode, blocks_per_window, stream, reasons) where `reasons` is a list of lines
    for the console. Budgets from FREE VRAM rather than card capacity, so another app
    holding the GPU is accounted for.
    """
    if free_gb is None:
        try:
            # plannable_free_vram honours FIZGIG_SIM_VRAM_GB, so FT sizing is testable on
            # the small-card simulator like every other planner (post-merge follow-up).
            from fizgig.utils.device import plannable_free_vram
            free_gb = plannable_free_vram()
        except Exception:
            free_gb = None
    if free_gb is None:
        return ("component", 14, False,
                ["could not read free VRAM — falling back to component mode"])
    try:
        if is_rocm():
            # Tiers were measured on a 5090; ROCm allocator behaviour differs enough that
            # the numbers are advisory there. Don't refuse — warn.
            logger.warning("[ft-rotation] tier table was measured on NVIDIA (5090); on ROCm "
                           "treat the picked window as a starting point and watch VRAM.")
    except Exception:
        pass

    # Component mode at every depth (27 Aug): depth-split windows made every card tier
    # a component tier — block mode remains an explicit Window-dropdown choice only.
    for min_free, stream, narrative in FT_COMPONENT_LADDER:
        if free_gb >= min_free:
            return ("component", 14, stream,
                    [f"{free_gb:.1f} GB free -> {narrative}; the window planner sizes "
                     "the exact splits at launch and prints them"])

    lo = FT_COMPONENT_LADDER[-1]
    return ("component", 14, True,
            [f"{free_gb:.1f} GB free is below the ~{lo[0]:.1f} GB the smallest streamed "
             "component plan was budgeted for — trying anyway; the window planner will "
             "refuse cleanly if it truly cannot fit."])


def _nvidia_smi_used_gb() -> Optional[float]:
    """Total VRAM in use per the DRIVER (every process), in GB. None when unreadable
    (no nvidia-smi on ROCm, or parsing failed) — callers treat None as 'guard is a no-op'."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        import subprocess
        out = subprocess.run([smi, "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def wait_for_gpu_handoff(threshold_gb: float = 6.0, timeout_s: float = 180.0,
                         poll_s: float = 5.0) -> None:
    """Driver-level VRAM handoff guard — call BEFORE a fine-tune claims the card.

    WDDM virtualizes memory per process, so the trainer's own mem_get_info can NEVER see a
    just-finished trainer that is still tearing down (unwinding a ~38 GB Python heap takes
    a while after the 'completed' line). Uploading a 21 GB base while the old process still
    holds its copy overcommits the card, and WDDM's demotion to shared memory is STICKY —
    the demoted blocks crawl when they first rotate in, so the slowdown surfaces mid-run
    (field: epochs 6-7 of a back-to-back fine-tune), not at load. nvidia-smi reads the
    driver's global view, which sees every process — so the guard runs there.

    Must run before this process's first CUDA call: after CUDA init, our own context is
    part of the total and the threshold arithmetic assumes we hold ~nothing yet."""
    used = _nvidia_smi_used_gb()
    if used is None or used < threshold_gb:
        return
    import time
    logger.info(
        "[ft-guard] another process is still holding ~%.1f GB of VRAM — waiting up to "
        "%.0f s for it to let go. A just-finished fine-tune can take a minute to fully "
        "exit; restarting Fizgig between fine-tunes always guarantees a clean handoff.",
        used, timeout_s)
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        time.sleep(poll_s)
        used = _nvidia_smi_used_gb()
        if used is None or used < threshold_gb:
            logger.info("[ft-guard] VRAM released after %.0f s — proceeding.",
                        time.monotonic() - start)
            return
    logger.warning(
        "[ft-guard] still ~%.1f GB held elsewhere after %.0f s — starting anyway. If this "
        "run trains slow, close other GPU apps, or restart Fizgig between fine-tunes.",
        used, timeout_s)


def _available_ram_gb():
    """(available_gb, total_gb) physical RAM, or (None, None) when unreadable."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS(dwLength=ctypes.sizeof(_MS))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / 2**30, ms.ullTotalPhys / 2**30
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        return info["MemAvailable"] / 2**20, info["MemTotal"] / 2**20
    except Exception:
        return None, None


def wait_for_ram_recovery(timeout_s: float = 180.0, poll_s: float = 5.0) -> None:
    """The RAM leg of the fine-tune handoff guard — call right after wait_for_gpu_handoff.

    A single H3 fine-tune commits ~120 GB (bf16 master + CUDA's system-RAM backing under
    WDDM + park arenas), and a just-finished one takes a while to hand that back. A second
    fine-tune started into that teardown gets its master evicted to the pagefile AS IT IS
    BUILT, and the evictions bite when each window first rotates in — measured in the field
    as a crawl through cycle 1, worst at its last windows (epochs 6-7). Physical
    availability climbs back within a minute or two of the old process dying, so waiting is
    both observable and sufficient. Threshold: 40%% of total RAM (a 128 GB box waits for
    ~51 GB — master + working margin; scales down for smaller boxes, where the run was
    always going to lean on the pagefile anyway)."""
    avail, total = _available_ram_gb()
    if avail is None or total is None:
        return
    need = 0.40 * total
    if avail >= need:
        return
    import time
    logger.info(
        "[ft-guard] only %.0f GB of %.0f GB RAM is available — a fine-tune wants ~%.0f GB "
        "before its master builds, so waiting up to %.0f s for Windows to hand memory back "
        "(a just-finished fine-tune releases ~120 GB and that takes a minute; restarting "
        "Fizgig between fine-tunes always guarantees a clean handoff).",
        avail, total, need, timeout_s)
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        time.sleep(poll_s)
        avail, _ = _available_ram_gb()
        if avail is None or avail >= need:
            logger.info("[ft-guard] RAM recovered (%.0f GB available) after %.0f s — "
                        "proceeding.", avail or 0.0, time.monotonic() - start)
            return
    logger.warning(
        "[ft-guard] still only %.0f GB of RAM available after %.0f s — starting anyway. If "
        "this run crawls in its first cycle, restart Fizgig between fine-tunes.",
        avail or 0.0, timeout_s)


def recommend_krea2_strategy(vram_gb: Optional[float] = None,
                             caps: Optional[Capabilities] = None,
                             mp: float = 0.25, batch: int = 1,
                             rank: int = 32,
                             force_quant: Optional[str] = None,
                             network_type: str = "lora",
                             lokr_factor: int = 8) -> MemoryStrategy:
    """Pick quantisation + swap for Krea 2 training on this machine.

    Preference: INT8 no-swap > NF4 no-swap > fp8 no-swap > swapping.

    INT8 leads where it fits — faster than NF4 AND ~7x more accurate (8-bit vs 4-bit), with
    exact gradients, at the cost of ~5 GB. NF4 comes next because it measured faster than fp8 as
    well as smaller (fused bitsandbytes dequant, where the fp8 path materialises a bf16 copy of
    every weight per forward). Swapping is always last: 4.4x slower and 4x the CPU load.
    """
    caps = caps or detect()
    # Decide on FREE memory, not the number on the box — read it FRESH, not from the
    # lru_cached detect() snapshot: a GUI session that started while a browser held 6 GB
    # would otherwise plan every later run from that stale reading.
    vram = vram_gb
    if vram is None:
        try:
            import torch
            free_b, _ = torch.cuda.mem_get_info(0)
            vram = free_b / (1024 ** 3)
        except Exception:
            vram = caps.vram_free_gb or caps.vram_gb

    if not caps.has_cuda:
        return MemoryStrategy(False, 0, "no CUDA device — settings left alone")

    # The user pinned the quantisation (the 4-bit control is not on Auto). Plan swap for the
    # footprint that will ACTUALLY run — the whole point, because the ladder below used to pick
    # a quant, size the swap for it, and then have the caller discard the quant and keep the
    # swap. That shipped fp8 (17.7 GB) on NF4's swap-0 plan and OOM'd 16 GB cards (issue #18).
    if force_quant:
        # A card without int8 tensor cores cannot run the INT8 path at all, so an explicit
        # INT8 request degrades to fp8 rather than launching something that will fail.
        if force_quant == "int8" and not caps.int8_matmul_train:
            force_quant = "fp8"
        # "no_4bit" prices as fp8 up front, then the branch below upgrades it to INT8 when that
        # fits — the fp8 figure is the fallback, not the decision. (Legacy: the old 4-bit
        # control's "Off". The Base-precision dropdown asks for int8/fp8 explicitly now.)
        _bases = {"nf4": _NF4_PEAK_GB, "int8": _INT8_PEAK_GB,
                  "fp8": _FP8_PEAK_GB, "no_4bit": _FP8_PEAK_GB}
        _base = _bases.get(force_quant)
        if _base is not None:
            need = estimate_krea2_peak(_base, mp, batch, rank, network_type, lokr_factor)
            if force_quant == "nf4":
                # NF4 cannot block-swap: the weights live in `_nf4_packed`, which the offloader
                # cannot move, and the trainer force-zeroes blocks_to_swap under 4-bit.
                fits = need + _HEADROOM_GB <= vram
                return MemoryStrategy(
                    True, 0,
                    f"NF4 4-bit as you set it (~{need:.0f} GB needed, {vram:.1f} GB free)"
                    + ("" if fits else " — this does NOT fit, and NF4 cannot block-swap; "
                                       "switch the 4-bit control to Auto or free some VRAM"))
            # "no_4bit" = the user turned the *4-bit* control off. That is a vote against NF4,
            # NOT against all quantisation: INT8 is 8-bit, faster than NF4 and ~7x more
            # accurate, so it still leads wherever it fits. Only when INT8 doesn't fit (or the
            # card lacks int8 cores) does this fall through to fp8.
            if force_quant == "no_4bit":
                _i8 = estimate_krea2_peak(_INT8_PEAK_GB, mp, batch, rank, network_type, lokr_factor)
                if caps.int8_matmul_train and vram >= _i8 + _HEADROOM_GB:
                    return MemoryStrategy(
                        False, 0,
                        f"INT8 W8A8, no block swap (~{_i8:.0f} GB needed at this run shape, "
                        f"{vram:.1f} GB free) — 4-bit is off as you set it, and INT8 is the "
                        "fastest thing that fits (8-bit, exact gradients)",
                        quant_int8="bf16")
                need = estimate_krea2_peak(_FP8_PEAK_GB, mp, batch, rank, network_type, lokr_factor)

            swap = swap_for_budget(need, vram)
            label = "INT8 W8A8" if force_quant == "int8" else "fp8"
            reason = (f"{label} as you set it (~{need:.0f} GB needed at this run shape, "
                      f"{vram:.1f} GB free) — "
                      + (f"{swap} blocks swapped to fit" if swap else "no block swap needed"))
            if swap >= _MAX_SWAP_KREA2 and need - swap * _SWAP_GB_PER_BLOCK + _HEADROOM_GB > vram:
                reason += " — even max swap may not fit; consider 4-bit on Auto"
            return MemoryStrategy(False, swap, reason,
                                  quant_int8="bf16" if force_quant == "int8" else "")

    # INT8 first where it fits: faster than NF4 *and* far more accurate, with exact gradients.
    # Needs int8 tensor cores, which torch._int_mm requires — present from Turing, so this is
    # not Blackwell-only (unlike fp8 _scaled_mm, which needs sm_89+).
    _int8_need = estimate_krea2_peak(_INT8_PEAK_GB, mp, batch, rank, network_type, lokr_factor)
    _nf4_need = estimate_krea2_peak(_NF4_PEAK_GB, mp, batch, rank, network_type, lokr_factor)
    _fp8_need = estimate_krea2_peak(_FP8_PEAK_GB, mp, batch, rank, network_type, lokr_factor)
    if caps.int8_matmul_train and vram >= _int8_need + _HEADROOM_GB:
        return MemoryStrategy(
            False, 0,
            f"INT8 W8A8, no block swap (~{_int8_need:.0f} GB needed at this run shape, {vram:.1f} GB free) — "
            "fastest measured, and ~7x more accurate than NF4 (8-bit vs 4-bit)",
            quant_int8="bf16")

    if caps.bitsandbytes and vram >= _nf4_need + _HEADROOM_GB:
        return MemoryStrategy(
            True, 0,
            f"NF4 4-bit, no block swap (~{_nf4_need:.0f} GB needed at this run shape, {vram:.1f} GB free) — "
            "fastest measured and leaves the most headroom")

    if vram >= _fp8_need + _HEADROOM_GB:
        return MemoryStrategy(
            False, 0, f"fp8, no block swap (~{_fp8_need:.0f} GB needed at this run shape, {vram:.1f} GB free)")

    if not caps.bitsandbytes:
        # Sized from the measured curve rather than coarse VRAM tiers: the old table gave a
        # 16 GB card 20 blocks where 16 fits with 2.4 GB to spare, and every extra block is a
        # PCIe round-trip per step.
        swap = swap_for_budget(_fp8_need, vram)
        return MemoryStrategy(
            False, swap,
            f"fp8 with {swap} blocks swapped — bitsandbytes is missing, so NF4 (which would "
            "avoid swapping entirely and run ~4x faster) is unavailable. Install it.")

    # Below NF4's own footprint. NF4 CANNOT swap (the trainer force-zeroes blocks_to_swap
    # under 4-bit — weights live in _nf4_packed, not .weight), so the old "NF4 + swap"
    # recommendation here was a configuration that cannot exist: on a 12 GB card it was
    # the only reachable tier, leaving the auto path with no working configuration at all.
    # fp8 + heavy swap is the one combination that actually runs at this size.
    swap = swap_for_budget(_fp8_need, vram)
    return MemoryStrategy(
        False, swap,
        f"fp8 with {swap} blocks swapped — {vram:.1f} GB free is below what Krea 2 needs "
        "resident even at 4-bit, and NF4 can't block-swap, so fp8+swap is the only "
        "combination that fits (slow: ~4x the step time)")


# torch.compile decision. Warm-up is the whole story: compiling costs ~90 s up front (one plan per
# distinct sequence shape) and then saves per step, so it is a straight loss on a short run and a
# clear win on a long one. Measured per step on an RTX 5090, Krea 2, rank 16:
#
#     INT8   0.5917 -> 0.292   saves 0.300 s/step   break-even ~300 steps
#     NF4    0.7092 -> 0.556   saves 0.153 s/step   break-even ~590 steps
#
# Doubled for margin, as with the attention backend: at break-even there is nothing to win, and
# being wrong should cost a few percent rather than a run.
_COMPILE_WARMUP_S = 90.0
_COMPILE_SAVING_S = {"int8": 0.300, "nf4": 0.153}
_COMPILE_MARGIN = 2.0
# INT8 + compile peaked at 21.7 GB against 17.8 GB for INT8 alone. NF4 + compile is VRAM-neutral
# (12.9 GB vs 13.6 GB) and completes under a hard 15.5 GB cap, so it fits a 16 GB card.
# BOTH figures are 0.25 MP measurements — see _COMPILE_GB_PER_MP for what happens above that.
_INT8_COMPILE_PEAK_GB = 20.0
_NF4_COMPILE_PEAK_GB = 13.0
# Resolution scaling UNDER COMPILE, and it is nothing like the eager path's 0.25 GB/MP.
# Eager, gradient checkpointing absorbs resolution (measured +0.15 GB from 0.25 -> 1.05 MP).
# Compiled, inductor's partitioner saves activations between the forward and backward graphs,
# and those scale with token count: a real 0.98 MP INT8+compile run on a 32 GB card reached
# 30.8 GB reserved and OOM'd on the first backward — implying >= ~12.6 GB/MP over the 0.25 MP
# baseline, and an OOM only bounds the true peak from BELOW. 15 adds slack in the only safe
# direction: over-declining runs uncompiled (slower), under-declining repeats the OOM.
# The NF4 figure is EXTRAPOLATED from that INT8 data point (the saved
# activations are bf16 either way, so the slope shouldn't depend on the weight format) — being
# wrong here declines compile and the run proceeds uncompiled, which costs speed, never the run.
_COMPILE_GB_PER_MP = 15.0
# The OUTSIDE boundary (#99): checkpoint kept OUTSIDE the compiled region, so inductor's
# partitioner stashes only what eager checkpointing stashes — recompute reruns the compiled
# graph. Measured (Krea 2 INT8, 46 imgs @ 1.05 MP, rank 32, RTX 5090): peak ~18.7 GB net vs
# eager's ~18.0, steady 2.4 s/step vs eager 3.30 (~27% faster), warm-up ~25 s (the per-block
# graph is reused across Krea 2's 28 identical blocks). The constant is the measured need at
# 1.05 MP; below that it is mildly conservative, which never matters — inside is preferred
# wherever IT fits, and it fits everywhere small. Beyond 1.05 MP the inside slope is borrowed
# as a deliberately-too-steep bound: over-declining runs eager (slower), never OOMs.
_INT8_COMPILE_OUTSIDE_PEAK_GB = 19.0
_COMPILE_OUTSIDE_ANCHOR_MP = 1.05


def compile_boundary(quant_4bit: bool, quant_int8: str, vram_gb=None, caps=None,
                     mp: float = 0.25, batch: int = 1) -> str:
    """'inside' | 'outside' — where the gradient checkpoint sits for an EXPLICIT
    Compile=On (#99). Never declines (On means on); it only places the boundary where
    it fits: inside-the-graph is 1.19x faster per block but its stashes scale hard with
    tokens (measured >32 GB at 1 MP on INT8), outside stays at eager-level stashes.
    When inside doesn't fit, outside is the strictly safer gamble even near the edge."""
    kind = "nf4" if quant_4bit else ("int8" if quant_int8 else None)
    if kind != "int8":
        return "inside"          # NF4/other: outside unmeasured — behave as before
    caps = caps or detect()
    vram = vram_gb if vram_gb is not None else (caps.vram_free_gb or caps.vram_gb)
    if not vram:
        # None OR the detect() default of 0.0 (no readable GPU): no basis to pick, and
        # 0.0 falling through would read as "inside doesn't fit -> outside", which is a
        # decision dressed as a default. Inside = today's behaviour.
        return "inside"
    _step_mp = float(mp) * max(1, int(batch))
    _res_gb = _COMPILE_GB_PER_MP * max(0.0, _step_mp - 0.25)
    if vram < _INT8_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        return "outside"
    return "inside"


def should_compile(total_steps: int, quant_4bit: bool, quant_int8: str,
                   blocks_to_swap: int, vram_gb: Optional[float] = None,
                   caps: Optional[Capabilities] = None, mp: float = 0.25,
                   batch: int = 1) -> tuple:
    """Decide whether torch.compile pays for itself on this run. Returns (bool, reason).

    `mp` is the run's largest bucket in megapixels, `batch` its batch size. What compiled
    activation stashes scale with is tokens PER STEP, and batch multiplies tokens exactly as
    resolution does — so the load term is mp x batch, priced at _COMPILE_GB_PER_MP over the
    0.25 baseline. At the defaults (0.25 MP, batch 1) the term is exactly zero, so the
    extensively-validated behaviour there cannot shift; eager checkpointing absorbs both knobs,
    which is why only the compile gate needs them at this strength.
    """
    caps = caps or detect()
    if caps.is_rocm:
        return False, (
            "ROCm/HIP PyTorch build — Auto leaves torch.compile off "
            "(recompiles per bucket shape on HIP; set Compile Blocks to On to override)"
        )
    vram = vram_gb if vram_gb is not None else (caps.vram_free_gb or caps.vram_gb)

    if blocks_to_swap:
        return False, "block swap is active — swapping moves weights between devices every step, " \
                      "which compiled graphs cannot tolerate"
    try:
        import triton  # noqa: F401
    except Exception:
        return False, "triton is not installed (pip install triton-windows on Windows)"
    _ok, _why = triton_matches_torch()
    if not _ok:
        return False, _why
    if not has_host_c_compiler():
        return False, ("no C compiler on this system — inductor/triton build host-side stubs "
                       "with one at runtime (on Debian/Ubuntu: apt install gcc); "
                       "running uncompiled")

    kind = "nf4" if quant_4bit else ("int8" if quant_int8 else None)
    if kind is None:
        return False, "only measured for the quantised paths (NF4 / INT8); not enabled for fp8 or bf16"
    _step_mp = float(mp) * max(1, int(batch))       # MP of latents per step
    _res_gb = _COMPILE_GB_PER_MP * max(0.0, _step_mp - 0.25)
    _shape = (f" at {mp:.2f} MP" + (f" x batch {batch}" if batch > 1 else "")) if _res_gb else ""
    _fix = (" (lower Target Megapixels or batch size to compile)" if _res_gb else "")
    _boundary = "inside"
    if kind == "int8" and vram < _INT8_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        # Inside-the-graph doesn't fit at this token load — try the OUTSIDE boundary
        # (#99): same fused kernels, eager-level stashes, measured ~27% faster than
        # eager at 1 MP. Only falls to uncompiled when even that can't fit.
        # Batch is charged at the measured EAGER term, not laundered through the step-MP
        # slope (which only starts at the anchor): the outside boundary's stashes ARE
        # eager-checkpointing stashes, and this module's own history says batch is the
        # single biggest term and the classic blind spot ("batch 2 sailed through the
        # check and OOM'd"). Resolution above the anchor keeps the deliberately-too-steep
        # inside slope, on mp alone.
        _out_need = (_INT8_COMPILE_OUTSIDE_PEAK_GB
                     + _COMPILE_GB_PER_MP * max(0.0, float(mp) - _COMPILE_OUTSIDE_ANCHOR_MP)
                     + _BATCH_GB_PER_IMAGE * max(0, int(batch) - 1))
        if vram >= _out_need + _HEADROOM_GB:
            _boundary = "outside"
        else:
            return False, (f"INT8 + compile peaks near {_INT8_COMPILE_PEAK_GB + _res_gb:.0f} "
                           f"GB{_shape} (checkpoint-outside still ~{_out_need:.0f} GB) and only "
                           f"{vram:.1f} GB is free — INT8 alone still fits, compile does not"
                           + _fix)
    if kind == "nf4" and _res_gb and vram < _NF4_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        return False, (f"NF4 + compile peaks near {_NF4_COMPILE_PEAK_GB + _res_gb:.0f} GB{_shape} "
                       f"and only {vram:.1f} GB is free — NF4 alone still fits, compile does not"
                       + _fix)

    needed = int(_COMPILE_WARMUP_S / _COMPILE_SAVING_S[kind] * _COMPILE_MARGIN)
    if total_steps < needed:
        return False, (f"{total_steps} steps is too short — compiling costs ~{_COMPILE_WARMUP_S:.0f} s "
                       f"up front and needs ~{needed} steps on the {kind.upper()} path to pay back")
    if _boundary == "outside":
        # Truthy like True, so bool-minded callers keep working; boundary-aware callers
        # (the Krea 2 trainer) pass it through to _compile_blocks.
        return "outside", (f"{total_steps} steps on the {kind.upper()} path — inside-the-graph "
                           f"doesn't fit{_shape}, compiling with the checkpoint OUTSIDE the "
                           f"region instead (measured ~27% faster than eager at 1 MP)")
    return True, (f"{total_steps} steps on the {kind.upper()} path — compile pays back within "
                  f"~{needed} steps and this run is longer")
