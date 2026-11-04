"""torch.compile for any described family's DiT blocks (FamilyDriver.compile_blocks): each block of the list the
driver names compiled, with gradient checkpointing inside the compiled region or around it, after checking a host C
compiler and a matching Triton. Krea 2 measured it first; every family that sets compiles=True runs the same code."""
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


class CheckpointedBlock(torch.nn.Module):
    """A transformer block that does its own gradient checkpointing.

    Exists so torch.compile can capture the checkpoint inside the graph. `_handles_checkpointing`
    tells the DiT forward not to wrap it a second time.
    """

    _handles_checkpointing = True

    def __init__(self, block, checkpointing: bool):
        super().__init__()
        self.block = block
        self.checkpointing = checkpointing

    def forward(self, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.block, *args, use_reentrant=False)
        return self.block(*args)


def find_host_compiler() -> bool:
    """Make sure a host C/C++ compiler exists before torch.compile runs; never crash the run.

    Inductor/triton build small host-side stubs at runtime, so compile without a compiler dies
    with "Failed to find C compiler". POSIX: check PATH for cc/gcc/clang — the RunPod image
    shipped without a toolchain, which crashed every compiled run there. Windows: `cl.exe` is
    installed by Visual Studio but only exposed inside a developer prompt, so launching Fizgig
    normally leaves compile dead on arrival — running vcvars64.bat and importing the environment
    it sets is what a developer prompt does; doing it here means the user does not have to know
    any of this.
    """
    import shutil
    import subprocess

    if os.name != "nt":
        from fizgig.utils.capabilities import has_host_c_compiler
        if has_host_c_compiler():
            return True
        logger.warning("[compile] no C compiler found — torch.compile needs one to build "
                       "inductor/triton host-side stubs (on Debian/Ubuntu: apt install gcc). "
                       "Training continues uncompiled.")
        return False
    if shutil.which("cl"):
        return True

    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    roots = []
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=30)
            roots += [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except Exception:
            pass
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"),
               os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        for year in ("2022", "2019"):
            for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
                roots.append(os.path.join(pf, "Microsoft Visual Studio", year, ed))

    for root in roots:
        vcvars = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.isfile(vcvars):
            continue
        try:
            # shell=True is intentional here: vcvars is a path just discovered via vswhere/
            # well-known VS install roots (not external input), and we need the shell's &&
            # to source the .bat file's env vars into `set`. cmd.exe /c would hit the same
            # interpreter anyway, so it'd be cosmetic, not safer.
            out = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True,
                                 text=True, timeout=120)
            if out.returncode != 0:
                continue
            for line in out.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k] = v
            if shutil.which("cl"):
                logger.info("[compile] MSVC found via %s", os.path.basename(root))
                return True
        except Exception:
            continue

    logger.warning("[compile] no MSVC C++ compiler found — torch.compile needs one on Windows to "
                   "build inductor's host-side code. Direct installer: "
                   "https://aka.ms/vs/17/release/vs_BuildTools.exe (tick the 'Desktop development "
                   "with C++' workload), or leave Compile Blocks off. Training continues uncompiled.")
    return False


_COMPILE_WORKS = None


def _compile_works() -> bool:
    """Compile and run one tiny function on the GPU: the whole chain a compiled block needs (inductor, Triton's
    kernel build, and Triton's own C helper, which it builds and links with the host compiler the first time a
    kernel runs). Finding a compiler is not enough - an old Visual Studio 2019 is found but fails that link, and the
    failure comes from the first kernel launch, outside dynamo's suppress_errors, so it ended the run before step
    one. Checked once per process; seconds the first time, cached by torch/Triton after that."""
    global _COMPILE_WORKS
    if _COMPILE_WORKS is not None:
        return _COMPILE_WORKS
    import torch
    if not torch.cuda.is_available():
        _COMPILE_WORKS = True
        return True
    import torch._dynamo
    try:
        f = torch.compile(lambda x: (x * 2.0).sin() + 1.0)
        f(torch.randn(64, device="cuda"))
        torch.cuda.synchronize()
        _COMPILE_WORKS = True
    except Exception as e:
        msg = str(e).strip().splitlines()
        logger.warning("[compile] Compile Blocks can't run on this PC: a test compile failed (%s: %s). Training "
                       "continues uncompiled, which is slower. To fix it on Windows (usually an old Visual Studio): "
                       "install Visual Studio 2022 Build Tools from https://aka.ms/vs/17/release/vs_BuildTools.exe, "
                       "tick 'Desktop development with C++' in the installer, then restart Fizgig. Or set Compile "
                       "Blocks to Off to skip this check.", type(e).__name__, msg[-1][:200] if msg else "")
        _COMPILE_WORKS = False
    finally:
        torch._dynamo.reset()
    return _COMPILE_WORKS


def ready_to_compile(blocks_to_swap: int = 0, fp8_scaled: bool = False) -> bool:
    """The machine and the run can torch.compile (no block swap, Triton matching torch, a host C compiler, fp8 only on
    SM 8.9+), with Fizgig's compile settings applied. False = say why in the log and run eager. A driver that compiles
    its own way (FamilyDriver.compile_blocks overridden, e.g. SDXL's blocks spread over several lists) calls this
    first."""
    if blocks_to_swap > 0:
        logger.warning("[compile] ignored — block swap moves weights between devices every step, "
                       "which invalidates compiled graphs. Quantise instead of swapping if you "
                       "want both.")
        return False
    if fp8_scaled:
        _cc = None
        try:
            # `import torch as _torch`, NOT the bare name: the `import torch._dynamo`
            # further down makes `torch` function-LOCAL, so referencing it here raises
            # UnboundLocalError — which the except below would silently eat, and the
            # guard would never fire (caught by the #97 regression test's tracer).
            import torch as _torch
            if _torch.cuda.is_available():
                _cc = _torch.cuda.get_device_capability()
        except Exception:
            pass
        if _cc is not None and _cc < (8, 9):
            logger.warning("[compile] ignored — the fp8 base needs fp8 Triton kernels "
                           "(fp8e4nv), which need SM 8.9+ (RTX 40-series or newer); this GPU "
                           "is SM %d.%d. Pick INT8 or NF4 Base Precision to compile on this "
                           "card. Training continues uncompiled.", _cc[0], _cc[1])
            return False
    try:
        import triton  # noqa: F401
    except Exception:
        logger.warning("[compile] ignored — triton is not installed (pip install triton-windows "
                       "on Windows, triton on Linux)")
        return False
    try:
        from fizgig.utils.capabilities import triton_matches_torch
        _ok, _why = triton_matches_torch()
    except Exception:
        _ok, _why = True, ""
    if not _ok:
        # A triton built for another torch imports fine and then fails or hangs INSIDE
        # torch.compile (a preview that never comes back, no log) — say so and run eager.
        logger.warning("[compile] ignored — %s. Training continues uncompiled.", _why)
        return False
    if not find_host_compiler():
        return False
    if not _compile_works():
        return False
    import torch._dynamo
    # Raises the recompile ceiling (default 8, which a bucketed dataset exhausts immediately —
    # after which dynamo silently runs eager) and works around a torch assertion that otherwise
    # aborts inductor mid-run. See fizgig/modules/compile_util.py.
    from fizgig.modules.compile_util import init_compile
    init_compile()
    # Settle the SDPA backend global BEFORE tracing: its lazy first-use probe (device alloc +
    # global write + logging) inside a compiled block is exactly what fullgraph=True raises on.
    from fizgig.modules import sdpa as _sdpa
    _sdpa.prime()
    # A compile failure must cost speed, not the run.
    torch._dynamo.config.suppress_errors = True
    # Two inductor notices that are expected here, not problems: TF32 stays off on purpose (the LoRA's fp32 maths
    # would change), and a complex-number op (Qwen's RoPE) runs uncompiled inside the compiled block.
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"TensorFloat32 tensor cores for float32 matrix multiplication available")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"Torchinductor does not support code generation for complex operators")
    return True


def compile_blocks(dit, blocks, blocks_to_swap: int = 0, fp8_scaled: bool = False,
                   boundary: str = "inside", fullgraph: bool = True) -> None:
    """Compile each block of `blocks` (the driver's ModuleList, replaced in place). The DiT's forward must call a
    block that has `_handles_checkpointing` directly, without checkpointing it again.

    The win is real on the quantised path (inductor fuses the per-matmul quantise/dequantise
    elementwise work that bounds INT8), and small on dense bf16. It costs compile time on the
    first step, and a recompile for every new latent shape a bucketed dataset presents.

    `boundary` places the gradient checkpoint relative to the compiled region (#99):
    "inside" (default) compiles the checkpoint INTO the graph — worth 1.19x per block, but
    inductor's partitioner stashes far more intermediates as tokens grow (measured >32 GB
    at 1 MP on the INT8 path, vs ~18 GB eager). "outside" compiles the raw block and keeps
    the checkpoint wrapper eager: recompute reruns the compiled graph, stashes stay at
    eager checkpointing's level, and the kernel-fusion win on the quantise/dequantise
    traffic survives — the high-resolution fit.

    Refused under block swap: compiled graphs assume their weights stay put, and swap moves them
    between CPU and GPU every step. Also refused for the fp8 base on pre-Ada GPUs: inductor
    lowers the fp8 dequant to an fp8e4nv Triton kernel that only SM 8.9+ silicon has, and the
    resulting ValueError escapes dynamo's suppress_errors and kills the run before step one
    (#97, RTX 3090).
    """
    if not ready_to_compile(blocks_to_swap, fp8_scaled):
        return
    # fullgraph=True refuses to compile around a graph break instead of quietly degrading. The
    # known break (attn_params.seqlens[0].item(), a device sync in the trim check) was fixed
    # earlier, so this should now hold — and if it does not, it says so instead of hiding.
    #
    # Each block is wrapped so the GRADIENT CHECKPOINT sits INSIDE the compiled region. Compiling
    # the raw block and checkpointing around it leaves the recompute outside the graph, and with
    # checkpointing the forward runs twice per step, so the boundary is worth 1.19x on a real block
    # (8.817 -> 7.428 ms/block-step).
    checkpointing = bool(getattr(dit, "gradient_checkpointing", False))
    # partial checkpointing (a big card's plan, quant.plan_partial_checkpointing): the last `gc_off_blocks` blocks run
    # without it, the rest keep it
    last_ck = len(blocks) - int(getattr(dit, "gc_off_blocks", 0) or 0)
    n = 0
    if boundary == "outside":
        for i, block in enumerate(blocks):
            blocks[i] = CheckpointedBlock(torch.compile(block, fullgraph=fullgraph), checkpointing and i < last_ck)
            n += 1
        logger.info("[compile] %d blocks compiled (checkpoint OUTSIDE the "
                    "compiled region — recompute reruns the compiled graph, so activation "
                    "stashes stay at eager level) — the first "
                    "step of each new shape pauses to compile", n)
        return
    for i, block in enumerate(blocks):
        blocks[i] = torch.compile(CheckpointedBlock(block, checkpointing and i < last_ck), fullgraph=fullgraph)
        n += 1
    logger.info("[compile] %d blocks compiled (checkpoint inside the graph, "
                "cache_size_limit=8192) — the first step of each new shape pauses to compile", n)
