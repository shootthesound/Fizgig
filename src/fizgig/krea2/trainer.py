"""Krea 2 LoRA training: full-model LoRA + flow-matching loss over a bucketed dataloader.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet — Krea2's
block semantics aren't mapped, so Identity/Style/Details presets come later). The base is frozen
(optionally fp8, QLoRA-style); only the LoRA trains in bf16. Uses Fizgig's bucketed multi-resolution
dataloader (same framework as Klein) over the krea2 latent/TE caches.
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
from multiprocessing import Value

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.krea2.utils import load_krea2_dit
from fizgig.krea2.sampling import gather_valid_text, prepare
from fizgig.modules.sdpa import consider_training_backend as _consider_training_backend
from fizgig.networks.lora import create_network
from fizgig.training.metadata import (
    ARCHITECTURE_KREA2, build_metadata, latest_sample_image, thumbnail_data_uri, resolve_title,
)
from fizgig.training.train_utils import LossRecorder, prune_state_dirs, validate_output_name

logger = logging.getLogger(__name__)


def _apply_context_lora(target, path, strength, *, device, dtype):
    """Load a context LoRA and apply it FROZEN + ACTIVE on `target` (the base DiT during
    training, or the Turbo at preview time). The context and the trainable/preview LoRA each
    wrap the forward and contribute additively; gradients never flow to the context. Returns
    the network so the caller can keep a reference (and free it after previews)."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    # Normalize foreign formats (PEFT / diffusers / ComfyUI `diffusion_model.*`, LyCORIS) to
    # kohya keys so create_network_from_weights' lora_down scan finds the modules — without
    # this a diffusers-format context LoRA yields 0 modules. Mirrors Klein's load_lora.
    sd = ensure_kohya_lora_state_dict(load_file(path))
    net = create_network_from_weights(None, float(strength), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    net.requires_grad_(False)
    return net


def _batch_is_reg(item_keys, reg_keys) -> bool:
    """True when EVERY item in the batch is a regularisation image.

    Batches are drawn from a single dataset's BucketBatchManager, so in practice a batch is
    wholly reg or wholly not (and FT runs at batch size 1 regardless). `all` rather than `any`
    is the safe reading of a mixed batch: throttling a subject image by mistake costs training
    signal, while missing a throttle on a reg image costs only a slightly firmer anchor.
    """
    if not item_keys or not reg_keys:
        return False
    return all(str(k) in reg_keys for k in item_keys)


def _apply_turbo_lora(dit, path, *, device, dtype):
    """Stage the Turbo distillation LoRA (rank 64) on the TRAINING DiT, disabled.

    RAW + this LoRA at strength 1.0 behaves as the Turbo model, so previews can render on the
    resident training DiT with the exact settings the Turbo path uses (8-step, CFG-free,
    mu=1.15). Between previews the net is disabled (a disabled module's forward is one flag
    check) and its params live on CPU, so training pays nothing for it.

    The file also carries `diff_b` BIAS deltas the kohya conversion cannot represent — on the
    I/O layers (first, last.linear, tmlp, tproj, txtmlp), the timestep embedding among them,
    where much of the 8-step distillation plausibly lives. They are resolved to the actual
    bias Parameters here, at setup, so a bad key fails loudly now rather than mid-run; the
    preview path applies them by exact snapshot/restore (bf16 `+=` then `-=` is not
    bit-clean; `copy_` of a clone is), so training weights are never altered.

    Returns (net, diffb) where diffb is a list of (bias_param, delta_cpu) pairs.
    """
    from safetensors.torch import load_file
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict

    raw_sd = load_file(path)
    diffb = []
    for key in sorted(raw_sd):
        if not key.endswith(".diff_b"):
            continue
        mod_path = key[:-len(".diff_b")]
        if mod_path.startswith("diffusion_model."):
            mod_path = mod_path[len("diffusion_model."):]
        delta = raw_sd[key]
        try:
            bias = dit.get_submodule(mod_path).bias
            if bias is None or tuple(bias.shape) != tuple(delta.shape):
                raise AttributeError("no bias / shape mismatch")
        except AttributeError as e:
            logger.warning("[turbo-lora] diff_b %s has no matching model bias (%s) — skipped", key, e)
            continue
        diffb.append((bias, delta.detach().to("cpu").clone()))

    sd = ensure_kohya_lora_state_dict(raw_sd)
    net = create_network_from_weights(None, 1.0, sd, None, dit, for_inference=True)
    net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    # Structure only until this lands — without it the LoRA sits at zero init and the
    # "turbo" previews would silently render the undistilled RAW at 8 steps (mush).
    net.load_state_dict(sd, strict=False)
    net.to(device="cpu", dtype=dtype).eval()
    net.requires_grad_(False)
    net.set_enabled(False)
    logger.info("[turbo-lora] %s staged on the training DiT: %d LoRA modules + %d bias deltas "
                "(disabled outside previews)", os.path.basename(path), len(net.unet_loras), len(diffb))
    return net, diffb


def load_dit_for_training(
    raw_path: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    network_type: str = "lora",     # "lora" | "lokr" — the trainable parametrization
    lokr_factor: int = 8,
    fp8_scaled: bool = True,
    quant_4bit: bool = False,
    quant_int8: str = "",          # "" | "bf16" | "int8" — W8A8 base, grad_mode of the same name
    blocks_to_swap: int = 0,
    gradient_checkpointing: bool = True,
    compile_blocks: bool = False,   # resolved by the caller; load_dit_ takes a plain bool
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    turbo_lora_path: str = None,    # Turbo distillation LoRA — staged disabled, for on-DiT previews
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    fp8_fast: bool = False,
):
    """Load the RAW DiT (frozen base, optionally fp8) and apply a trainable full-model LoRA.
    An optional frozen Context LoRA is applied to the base first, so the new LoRA learns to
    coexist with it (the context stays active during previews too).

    quant_4bit: QLoRA-style 4-bit (NF4) frozen base — halves DiT residency (~14 GB fp8 → ~5.6 GB)
    so a full LoRA trains on a 10-12 GB card with no block swap. Mutually exclusive with block
    swap (weights live in _nf4_packed, not .weight). Loads the base bf16 on CPU and NF4-quantizes
    the block Linears onto the GPU layer-by-layer (peak VRAM never holds the whole bf16 model).
    Reuses the same target/exclude keys as the fp8 path (`blocks.` minus mod./norm/txtfusion)."""
    if quant_int8:
        # INT8 W8A8: quantize from bf16 like NF4 (avoids fp8->int8 double-quant).
        fp8_scaled = False
        quant_4bit = False
        loading_device = "cpu"
    elif quant_4bit:
        # NF4 quantizes from bf16 (cleaner than fp8->NF4 double-quant), staged on CPU, and can't
        # coexist with block swap — force both here so callers can't misconfigure it.
        fp8_scaled = False
        blocks_to_swap = 0
        loading_device = "cpu"
    else:
        loading_device = "cpu" if blocks_to_swap > 0 else device
    dit = load_krea2_dit(raw_path, device=device, dtype=dtype, fp8_scaled=fp8_scaled,
                         loading_device=loading_device, fp8_fast=fp8_fast)
    dit.requires_grad_(False)  # frozen base (QLoRA-style)
    if quant_int8:
        from fizgig.krea2.utils import KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS
        from fizgig.modules.int8_train import apply_int8_training
        n_q = apply_int8_training(
            dit, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
            compute_device=torch.device(device), grad_mode=quant_int8)
        dit.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"INT8 W8A8 base active: {n_q} Linears; grad_mode={quant_int8}; resident on {device}.")
    if quant_4bit:
        from fizgig.krea2.utils import KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS
        from fizgig.modules.nf4 import apply_nf4_quantization
        n_q = apply_nf4_quantization(
            dit, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS, compute_device=torch.device(device))
        dit.to(device)  # move the remaining (non-quantized) bf16 modules to the GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"NF4 4-bit base active: {n_q} Linears quantized; DiT resident on {device}.")
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # Stacking order (deliberate): Turbo LoRA innermost, then Context, then the trainable LoRA
    # outermost. The deltas are additive so the sum is order-independent, but this order tells
    # the deployment story: base+turbo ≈ the Turbo model, context rides on that, and the LoRA
    # being trained samples on top of the whole stack — exactly how it will be used.

    # Turbo distillation LoRA (rank 64): staged DISABLED on the training DiT so previews can
    # render on the resident model (RAW + turbo@1.0 ≈ Turbo, same 8-step CFG-free settings)
    # instead of loading the ~13 GB Turbo checkpoint and parking the trainer to CPU.
    turbo_net = turbo_diffb = None
    if turbo_lora_path:
        turbo_net, turbo_diffb = _apply_turbo_lora(dit, turbo_lora_path, device=device, dtype=dtype)

    # Context LoRA: frozen + active on the base BEFORE the trainable LoRA, so the trainable
    # one wraps the context-included forward (both additive; grads only flow to the trainable).
    if context_lora_path:
        logger.info(f"context LoRA: {os.path.basename(context_lora_path)} @ {context_lora_strength} (frozen, active)")
        _apply_context_lora(dit, context_lora_path, context_lora_strength, device=device, dtype=dtype)

    if network_type == "lokr":
        from fizgig.networks.lora import LoKRModule
        logger.info(f"network: LoKR (Kronecker), factor {lokr_factor}, full-matrix w2 — "
                    "dim/alpha do not apply")
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 module_class=LoKRModule, module_kwargs={"factor": int(lokr_factor)})
    else:
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    network._network_type = network_type
    network._lokr_factor = int(lokr_factor)
    # Dotted module paths, for the LyCORIS-standard final save (diffusion_model.<path>.lokr_*).
    # Built from the DiT itself with the same flattening create_modules used, so the reverse
    # mapping is exact even where module names contain underscores.
    network._dotted_names = {
        f"lora_unet_{name.replace('.', '_')}": name
        for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
    }

    # torch.compile LAST — after the LoRAs have patched the forwards, so the compiled graph is
    # the one that actually runs. Per block, not whole-model: the 28 blocks share a graph
    # signature so inductor compiles once and reuses, and a failure is contained to one block.
    # With a turbo net staged, its enabled flag becomes a dynamo guard: the first preview
    # compiles a second graph variant (enabled=True), after which both states are cached.
    if compile_blocks:
        # fp8_scaled here is the RESOLVED value (int8/NF4 force it False above), so the
        # guard sees the base that actually loaded. compile_blocks may be the string
        # "outside" (high-res boundary, #99) — any other truthy value means "inside".
        _compile_blocks(dit, blocks_to_swap, fp8_scaled=fp8_scaled,
                        boundary=("outside" if str(compile_blocks).lower() == "outside"
                                  else "inside"))
    return dit, network, turbo_net, turbo_diffb


class _CheckpointedBlock(torch.nn.Module):
    """A transformer block that does its own gradient checkpointing.

    Exists so torch.compile can capture the checkpoint inside the graph. `_handles_checkpointing`
    tells the DiT forward not to wrap it a second time.
    """

    _handles_checkpointing = True

    def __init__(self, block, checkpointing: bool):
        super().__init__()
        self.block = block
        self.checkpointing = checkpointing

    def forward(self, x, vec, freqs, attn_params=None):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.block, x, vec, freqs, attn_params, use_reentrant=False)
        return self.block(x, vec, freqs, attn_params)


def _find_host_compiler() -> bool:
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


def _compile_blocks(dit, blocks_to_swap: int, fp8_scaled: bool = False,
                    boundary: str = "inside") -> None:
    """Compile each transformer block. Opt-in — see the roadmap for what it is and isn't worth.

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
    if blocks_to_swap > 0:
        logger.warning("[compile] ignored — block swap moves weights between devices every step, "
                       "which invalidates compiled graphs. Quantise instead of swapping if you "
                       "want both.")
        return
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
            return
    try:
        import triton  # noqa: F401
    except Exception:
        logger.warning("[compile] ignored — triton is not installed (pip install triton-windows "
                       "on Windows, triton on Linux)")
        return
    try:
        from fizgig.utils.capabilities import triton_matches_torch
        _ok, _why = triton_matches_torch()
    except Exception:
        _ok, _why = True, ""
    if not _ok:
        # A triton built for another torch imports fine and then fails or hangs INSIDE
        # torch.compile (a preview that never comes back, no log) — say so and run eager.
        logger.warning("[compile] ignored — %s. Training continues uncompiled.", _why)
        return
    if not _find_host_compiler():
        return
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

    # fullgraph=True refuses to compile around a graph break instead of quietly degrading. The
    # known break (attn_params.seqlens[0].item(), a device sync in the trim check) was fixed
    # earlier, so this should now hold — and if it does not, it says so instead of hiding.
    #
    # Each block is wrapped so the GRADIENT CHECKPOINT sits INSIDE the compiled region. Compiling
    # the raw block and checkpointing around it leaves the recompute outside the graph, and with
    # checkpointing the forward runs twice per step, so the boundary is worth 1.19x on a real block
    # (8.817 -> 7.428 ms/block-step).
    checkpointing = bool(getattr(dit, "gradient_checkpointing", False))
    n = 0
    if boundary == "outside":
        for i, block in enumerate(dit.blocks):
            dit.blocks[i] = _CheckpointedBlock(torch.compile(block, fullgraph=True),
                                               checkpointing)
            n += 1
        logger.info("[compile] %d blocks compiled (fullgraph, checkpoint OUTSIDE the "
                    "compiled region — recompute reruns the compiled graph, so activation "
                    "stashes stay at eager level; the high-resolution fit) — the first "
                    "step of each new shape pauses to compile", n)
        return
    for i, block in enumerate(dit.blocks):
        dit.blocks[i] = torch.compile(_CheckpointedBlock(block, checkpointing), fullgraph=True)
        n += 1
    logger.info("[compile] %d blocks compiled (fullgraph, checkpoint inside the graph, "
                "cache_size_limit=8192) — the first step of each new shape pauses to compile", n)


def _get_lin_function(x1, y1, x2, y2):
    """Linear map through (x1,y1)-(x2,y2): f(x) = m*x + b. Used to schedule the flow shift `mu`
    from image-token count (musubi's get_lin_function)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# Krea 2 resolution->mu schedule (musubi `krea2_shift`): token count maps to mu, shift = exp(mu).
# Endpoints match krea2 inference defaults (minres 256, maxres 1280 at align 16):
#   x1 = (256//16)**2 = 256, x2 = (1280//16)**2 = 6400, y1 = 0.5, y2 = 1.15.
_KREA2_MU = _get_lin_function(256, 0.5, 6400, 1.15)


def sample_krea2_timesteps(bsize: int, num_img_tokens: int, device, sigmoid_scale: float = 1.0,
                           min_timestep: float = 0.0, max_timestep: float = 1.0) -> torch.Tensor:
    """Krea 2 'krea2_shift' timestep sampling — a faithful port of the musubi krea2_train recipe.

    The base t is **logit-normal** (sigmoid of a standard normal), so timesteps concentrate near the
    middle instead of being uniform. Uniform sampling (the old code) dumps far too much mass on the
    high-noise end, where the flow-matching velocity is intrinsically hard to predict — that inflates
    the loss AND skews the training signal away from the validated reference recipe. The shift is
    resolution-dependent (shift = exp(mu), mu from the image-token count), not a fixed 2.5.

        t_base = sigmoid(randn * sigmoid_scale)
        t      = (t_base * shift) / (1 + (shift - 1) * t_base)
    """
    mu = _KREA2_MU(num_img_tokens)
    shift = math.exp(mu)
    t = (torch.randn(bsize, device=device) * sigmoid_scale).sigmoid()
    t = (t * shift) / (1.0 + (shift - 1.0) * t)
    # Optional timestep window (0-1 scale; t near 1 = high noise / structure, near 0 = detail).
    # Rescale INTO the window rather than clamp — clamping piles probability mass onto the two
    # endpoints, which trains those exact t values disproportionately.
    if min_timestep > 0.0 or max_timestep < 1.0:
        lo, hi = max(0.0, float(min_timestep)), min(1.0, float(max_timestep))
        t = lo + t * max(hi - lo, 1e-6)
    return t


def compute_loss(dit, latent, hidden_states, attention_mask, *, shift=2.5, dtype=torch.bfloat16,
                 device=None, control_latent=None,
                 min_timestep=0.0, max_timestep=1.0, motion_weight=0.0,
                 diff_ref_latent=None, diff_weight=0.0):
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)         — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask
    control_latent: optional (B, 16, h, w) — a CLEAN in-context reference (paired-image
        training, e.g. the source frame of a temporal-displacement pair). Injected as extra
        image tokens at RoPE frame=1 — the krea2_edit ecosystem's convention: the model tells
        source from target purely by that axis. Never noised; loss on target tokens only; the
        mu schedule stays on the target token count. Klein's control path is the template.
    motion_weight: 0..1 (paired path only) — upweight target tokens where the CLEAN pair
        actually differs (|x0 - source| per token). Uniform MSE rewards copying the source:
        most of a frame is static, so the copy shortcut wins the gradient. At m the per-token
        weight is (1-m) + m * diff/mean(diff), capped at 8x and renormalized to per-sample
        mean 1 — a fully static pair degrades to uniform weights and avr_loss stays on the
        same scale either way. 0 (default) = exact previous behaviour.

    `shift` is kept for signature compatibility but no longer used: krea2_shift derives the flow
    shift from the image resolution (see sample_krea2_timesteps), matching the musubi reference.
    """
    # The caller knows the compute device; only fall back to sniffing a parameter when it
    # doesn't say. Sniffing is fragile — under block swap some parameters are SUPPOSED to be on
    # CPU, so a stray placement silently drags the whole batch onto the wrong device instead of
    # failing loudly (which is exactly how the NF4 recaption bug presented).
    device = device if device is not None else next(p for p in dit.parameters()).device
    B = latent.shape[0]
    latent = latent.to(device=device, dtype=dtype)
    patch = dit.config.patch

    noise = torch.randn_like(latent)
    # krea2_shift: logit-normal base + resolution-dependent shift, over the image-token count
    # (latent grid // patch). Replaces the old uniform-u sampler that over-weighted high-noise t
    # and inflated the loss.
    num_img_tokens = (latent.shape[-2] // patch) * (latent.shape[-1] // patch)
    t = sample_krea2_timesteps(B, num_img_tokens, device,
                               min_timestep=min_timestep, max_timestep=max_timestep)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))

    if control_latent is None:
        img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
        target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)
        n_tgt = None
    else:
        # Paired-image (edit-style) sequence: [noisy target @ frame 0 | clean source @ frame 1
        # | text @ zeros]. Image tokens stay a contiguous all-valid prefix (the varlen
        # invariant); imglen inside the DiT covers target+source, so its output includes source
        # rows — sliced off below before the loss.
        from fizgig.krea2.sampling import patchify_block
        src = control_latent.to(device=device, dtype=dtype)
        tgt_tokens, tgt_pos, tgt_mask = patchify_block(noised, patch, frame=0.0)
        src_tokens, src_pos, src_mask = patchify_block(src, patch, frame=1.0)
        txtpos = torch.zeros(B, txt.shape[1], 3, device=device)
        img_tokens = torch.cat((tgt_tokens, src_tokens), dim=1)
        pos = torch.cat((tgt_pos, src_pos, txtpos), dim=1)
        mask = torch.cat((tgt_mask, src_mask, txtmask), dim=1)
        target_tokens, _, _ = patchify_block(target, patch, frame=0.0)
        n_tgt = tgt_tokens.shape[1]

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    if n_tgt is not None:
        pred = pred[:, :n_tgt]  # loss on target tokens only — source rows carry no target
    # Return the mean drawn timestep alongside the loss so the passive per-image loss logger can
    # normalize for noise level (the caller ignores it when logging is off).
    if control_latent is not None and motion_weight > 0.0:
        # Motion comes from the CLEAN latents (x0 vs source) — the velocity target carries
        # noise and would randomize the weights.
        diff_tokens, _, _ = patchify_block((latent - src).abs(), patch, frame=0.0)
        d = diff_tokens.float().mean(dim=-1)                                # (B, N)
        dm = d.mean(dim=1, keepdim=True)
        r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
        w = (1.0 - float(motion_weight)) + float(motion_weight) * r
        w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)                 # per-sample mean 1
        # A DEGENERATE pair (identical images) at weight 1.0 would zero every token's weight
        # and silently train nothing — fall back to uniform for that sample instead.
        w = torch.where(dm > 1e-6, w, torch.ones_like(w))
        se = (pred.float() - target_tokens.float()).pow(2).mean(dim=-1)     # (B, N)
        return (se * w).mean(), float(t.mean().item())
    if diff_ref_latent is not None and diff_weight > 0.0:
        # Slider training's disentanglement weight: identical formula to motion weighting,
        # but on the PLAIN (unpaired) sequence — the reference is the pair's other image,
        # never packed into the forward. The smile slider learns the mouth, not the haircut.
        from fizgig.krea2.sampling import patchify_block
        _ref = diff_ref_latent.to(device=device, dtype=dtype)
        diff_tokens, _, _ = patchify_block((latent - _ref).abs(), patch, frame=0.0)
        d = diff_tokens.float().mean(dim=-1)
        dm = d.mean(dim=1, keepdim=True)
        r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
        w = (1.0 - float(diff_weight)) + float(diff_weight) * r
        w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
        w = torch.where(dm > 1e-6, w, torch.ones_like(w))   # same degenerate-pair guard
        se = (pred.float() - target_tokens.float()).pow(2).mean(dim=-1)
        return (se * w).mean(), float(t.mean().item())
    return F.mse_loss(pred.float(), target_tokens.float()), float(t.mean().item())



class _BucketOrderSampler(torch.utils.data.Sampler):
    """Yield indices grouped by latent shape, shuffled within and across groups.

    Keeps an epoch random while making consecutive steps mostly share a shape, which is what
    lets shape-sensitive kernels (cuDNN attention, cuBLAS heuristics, torch.compile's shape
    cache) stay warm. Reshuffles every epoch so the order is never identical twice.
    """

    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        buckets = {}
        for i in range(len(dataset)):
            lat = dataset[i]["latents"]
            buckets.setdefault(tuple(lat.shape[-2:]), []).append(i)
        self.buckets = list(buckets.values())
        self.n = sum(len(b) for b in self.buckets)
        self.n_shapes = len(self.buckets)
        # What a plain shuffle would cost, for the log line: probability consecutive draws
        # differ, times the number of transitions.
        p_same = sum((len(b) / self.n) ** 2 for b in self.buckets)
        self.est_random_changes = int(round((1 - p_same) * (self.n - 1)))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        import random as _r
        rng = _r.Random(self.seed + self.epoch)
        order = []
        groups = [list(b) for b in self.buckets]
        for g in groups:
            rng.shuffle(g)
        rng.shuffle(groups)
        for g in groups:
            order.extend(g)
        return iter(order)


class _Krea2Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_training_state(output_dir, output_name, network, optimizer, *, epoch, global_step,
                         network_dim, network_alpha, dtype, extra=None):
    """Save a resumable training-state dir matching Klein's naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). The dir
    holds the LoRA weights, the optimizer state, RNG states, and a small JSON. The GUI's
    _detect_latest_state_dir finds the highest-numbered one and passes it to --resume."""
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    try:
        return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                  global_step=global_step, network_dim=network_dim,
                                  network_alpha=network_alpha, dtype=dtype, extra=extra)
    except Exception as _first:
        # Clean the partial dir (no training_state.json = no commit marker, but it would shadow
        # the previous good state in the GUI's latest-state scan), then retry ONCE after a short
        # pause. Network filesystems (RunPod volumes) throw transient stream errors that clear
        # in seconds — a real run lost its epoch-8 state to exactly one of those. If the retry
        # also fails it is not transient; re-raise and let the caller decide fatality.
        import shutil
        import time
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.warning("[state] save failed (%s: %s) — retrying once in 5s",
                       type(_first).__name__, _first)
        time.sleep(5)
        try:
            os.makedirs(state_dir, exist_ok=True)
            return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                      global_step=global_step, network_dim=network_dim,
                                      network_alpha=network_alpha, dtype=dtype, extra=extra)
        except Exception:
            shutil.rmtree(state_dir, ignore_errors=True)
            raise


def _write_state_files(state_dir, network, optimizer, *, epoch, global_step,
                       network_dim, network_alpha, dtype, extra=None):
    _save_lora(network, os.path.join(state_dir, "lora.safetensors"), network_dim, network_alpha, dtype)
    if optimizer is not None:   # None under fused backward (per-parameter optimizers)
        torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    if extra:
        meta.update(extra)
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    # training_state.json is written LAST on purpose: it is the commit marker. A save that
    # dies partway leaves no json, and both the resume validator and the GUI's latest-state
    # detection treat a json-less dir as not-a-state rather than resuming garbage.
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _validate_state_dir(state_dir):
    """Refuse anything that is not a saved training state, and say what to pick instead.

    Issue #48: choosing the OUTPUT directory rather than a state folder failed with a bare
    "lora.safetensors not found", and the obvious workaround — putting a LoRA there under that
    name — then appeared to work. It cannot: without training_state.json there is no epoch or
    step, and without optimizer.pt there is no Adam state, so the run silently starts over from
    epoch 0 while looking like a resume, and overwrites the finished LoRA on the way. Refusing
    is the only safe answer, and the message has to name the folder they actually wanted.
    """
    if os.path.isfile(state_dir):
        sib = ""
        base = os.path.dirname(state_dir)
        try:
            states = sorted(d for d in os.listdir(base) if d.endswith("-state")
                            and os.path.isfile(os.path.join(base, d, "training_state.json")))
            if states:
                sib = " Next to it: " + ", ".join(states[-3:])
        except OSError:
            pass
        raise RuntimeError(
            f"[resume] {os.path.basename(state_dir)} is a file — resume takes the saved-state "
            f"FOLDER (named like '<lora name>-000012-state'), not a .safetensors.{sib}")
    if not os.path.isdir(state_dir):
        raise RuntimeError(f"[resume] {state_dir} does not exist — was the state folder moved "
                           f"or renamed?")
    missing = [f for f in ("lora.safetensors", "training_state.json")
               if not os.path.isfile(os.path.join(state_dir, f))]
    if not missing:
        return
    lines = [
        f"[resume] {state_dir} is not a saved training state — missing {', '.join(missing)}.",
        "[resume] Pick the folder named like '<lora name>-000012-state'. Renaming a LoRA to "
        "lora.safetensors does not make one: there would be no optimizer state and no epoch "
        "to resume from, so the run would quietly start again from scratch.",
    ]
    try:
        # The usual mistake is picking the parent output directory, one level above the state
        # folders — so if they are sitting right there, name them.
        here = sorted(d for d in os.listdir(state_dir)
                      if d.endswith("-state")
                      and os.path.isfile(os.path.join(state_dir, d, "training_state.json")))
        if here:
            lines.append("[resume] That looks like your output directory. The saved states in "
                         "it are: " + ", ".join(here[-5:]))
    except OSError:
        pass
    raise RuntimeError(os.linesep.join(lines))


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
    _validate_state_dir(state_dir)
    from safetensors.torch import load_file
    # strict=False tolerates benign key drift, but if NOTHING matched the LoRA silently stays at
    # its zero init and the run "succeeds" while training from scratch — then overwrites the
    # finished LoRA with a no-op. That's most reachable via resume-a-finished-run, so refuse it.
    _incompat = network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
    _missing = getattr(_incompat, "missing_keys", [])
    if _missing and len(_missing) >= len(network.state_dict()):
        raise RuntimeError(
            f"[state] {state_dir} matched none of this network's {len(network.state_dict())} keys — "
            f"refusing to resume into a zero-initialised LoRA. The state was almost certainly saved "
            f"with a different network config (rank/alpha/target modules or a different Context LoRA).")
    opt_path = os.path.join(state_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    meta_path = os.path.join(state_dir, "training_state.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


class AdaptiveLR:
    """Bi-directional plateau LR tracker — a faithful port of Klein's adaptive_lr logic.

    Each epoch boundary: probe UP ×1.25 on steady loss descent (patience 2); reduce DOWN ×0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the
    LoRA weights 70/30 toward the previous epoch's snapshot and restores the optimizer state
    (kills bad Adam momentum). Klein's stability signals are grad-clip ratio + weight-norm
    growth; krea2 has no grad clipping, so weight-norm growth (>30%) is the stability signal.

    State (streaks/best_loss/prev_weight_norm) is JSON round-trippable for pause/resume; the
    CPU rollback snapshot is in-memory only (too big to persist) — so the first post-resume
    epoch can't roll back, exactly as in Klein. Call epoch_boundary() at each epoch end."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


def _build_bf16_master(raw_path: str, dit) -> dict:
    """CPU bf16 copy of every fp8-patched block Linear — the source of truth for rotation.

    Read straight from the RAW file (which is bf16 on disk) rather than dequantizing the GPU
    copy: the GPU weights have already been through fp8, so dequantizing them would bake the
    quantization error into the master and we'd fine-tune a degraded model.
    """
    from safetensors.torch import load_file
    from fizgig.krea2.rotation import is_rotatable_linear

    # Discovery must match what the rotator will actually target — an NF4 base has no
    # `scale_weight`, so an fp8-only test here would build an EMPTY master and the run
    # would train nothing while looking healthy.
    wanted = set()
    for bi, block in enumerate(dit.blocks):
        for name, m in block.named_modules():
            if is_rotatable_linear(m):
                wanted.add(f"blocks.{bi}.{name}.weight")
    # txtfusion sits outside dit.blocks, so rotation never reaches it — but it's the stack
    # that fuses the text embeddings, so it's held always-on rather than left frozen.
    txtf = getattr(dit, "txtfusion", None)
    if txtf is not None:
        for name, m in txtf.named_modules():
            if is_rotatable_linear(m):
                wanted.add(f"txtfusion.{name}.weight")

    sd = load_file(raw_path)          # mmap'd; we copy out only the keys we need
    master, missing = {}, []
    for key in sorted(wanted):
        t = sd.get(key)
        if t is None:
            missing.append(key)
            continue
        master[key] = t.to("cpu", dtype=torch.bfloat16).clone()
    del sd
    gc.collect()
    total_gb = sum(v.numel() * v.element_size() for v in master.values()) / 1e9
    logger.info("[ft-rotation] bf16 master: %d tensors, %.1f GB in CPU RAM%s",
                len(master), total_gb,
                f" ({len(missing)} keys missing from the RAW file — those stay frozen)" if missing else "")
    if missing:
        logger.warning("[ft-rotation] missing master keys, e.g. %s", missing[:3])
    return master


def _save_full_checkpoint(rotator, raw_path: str, path: str, extra_metadata=None):
    """Write the fine-tuned model: the RAW checkpoint with trained block weights replaced.

    Everything the rotator never touches (norms, embeddings, txtfusion, I/O layers) is copied
    through from the original, so the result is a complete, loadable Krea 2 checkpoint.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(raw_path)
    trained = rotator.master_state_dict()
    replaced = 0
    for k, v in trained.items():
        if k in sd:
            sd[k] = v.to(torch.bfloat16)
            replaced += 1
    meta = {"fizgig_finetune": "krea2-rotation", "fizgig_trained_tensors": str(replaced)}
    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write to a temp name and rename on completion. This file is ~24.5 GB and takes minutes;
    # Stop (a hard kill) or a power cut partway through used to leave a TRUNCATED checkpoint
    # sitting at the real filename — correct header, most of the tensors, and unloadable
    # ("MetadataIncompleteBuffer"). It looked like a valid save until ComfyUI refused it, which
    # could be days later. os.replace is atomic within a volume, so an interrupted write now
    # leaves only a .tmp you can delete, and the previous checkpoint stays intact.
    tmp = path + ".tmp"
    try:
        save_file(sd, tmp, metadata=meta)
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt/SystemExit are exactly the cases
        # that produce a half-written file, and they must not leave the .tmp behind either.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    size_gb = os.path.getsize(path) / 1e9
    logger.info("[ft-rotation] saved full checkpoint (%d/%d tensors trained, %.1f GB) -> %s",
                replaced, len(sd), size_gb, path)
    del sd, trained
    gc.collect()


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None,
               comfy_format=False):
    """Save the trainable network. `comfy_format` (final artifact only) rewrites a LoKR's keys
    to the LyCORIS standard (`diffusion_model.<dotted>.lokr_*`) — the format every ComfyUI LoKR
    in the wild uses. Internal saves (state dirs, preview temps) stay in native state_dict
    naming so resume's load_state_dict and the preview reload path work unchanged; our own
    loader ingests both via ensure_kohya_lora_state_dict."""
    is_lokr = getattr(network, "_network_type", "lora") == "lokr"
    if is_lokr:
        metadata = {
            "ss_network_module": "fizgig.krea2 (lokr, all-Linear)",
            "ss_lokr_factor": str(getattr(network, "_lokr_factor", "")),
            "ss_architecture": ARCHITECTURE_KREA2,
        }
    else:
        metadata = {
            "ss_network_module": "fizgig.krea2 (lora_unet, all-Linear)",
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": str(network_alpha),
            "ss_architecture": ARCHITECTURE_KREA2,
        }
    if extra_metadata:
        metadata.update(extra_metadata)
    if comfy_format and is_lokr:
        from fizgig.networks.lora import _precalculate_safetensors_hashes
        from safetensors.torch import save_file
        dotted = getattr(network, "_dotted_names", {})
        sd = {}
        for k, v in network.state_dict().items():
            mod, _, suffix = k.partition(".")
            path_dotted = dotted.get(mod)
            nk = f"diffusion_model.{path_dotted}.{suffix}" if path_dotted else k
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            sd[nk] = v
        model_hash, legacy_hash = _precalculate_safetensors_hashes(sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(sd, path, metadata)
        return
    network.save_weights(path, dtype, metadata)


# --- in-training previews (sample the fp8 Turbo with the live LoRA) -----------
def encode_sample_prompts(te_path, prompts, *, ref_image=None, vision_megapixels=1.0, device="cuda"):
    """Pre-encode the sample prompts once (Qwen3-VL), freeing the encoder afterwards.
    Returns a list of (txt, txtmask) on CPU, fed straight to sampling.sample at preview time.

    `ref_image` (a PIL image or path) routes a reference through Qwen3-VL's vision path so the
    samples become visually aware of it ('prompt from a picture' — Krea 2's reference mechanism)."""
    from fizgig.krea2.utils import load_krea2_text_encoder
    from fizgig.krea2 import sampling

    pil = None
    if ref_image:
        from PIL import Image
        pil = ref_image if hasattr(ref_image, "convert") else Image.open(ref_image)

    enc = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)
    out = []
    for p in prompts:
        images = [[pil]] if pil is not None else None
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [p], cfg=False,
                                                     images=images, vision_megapixels=vision_megapixels)
        out.append((txt.cpu(), txtmask.cpu()))
    del enc
    torch.cuda.empty_cache()
    return out


def _read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height, ref_image} while active, else None. ref_image (if set)
    is routed through Qwen3-VL's vision path (Krea 2's reference mechanism)."""
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        ref = str(d.get("ref_image", "")).strip()
        # Active on a prompt OR a reference — a reference with an empty prompt is a valid
        # 'generate from this picture' override (the Qwen3-VL vision path handles the rest).
        if prompt or ref:
            return {"prompt": prompt,
                    "seed": int(d.get("seed", 1234)),
                    "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024)),
                    "ref_image": ref}
    except Exception:
        pass
    return None


def _remove_claimed_queue(path: str) -> None:
    """Best-effort removal of the claimed caption queue (.processing).

    The Problem Images window polls this exact file every 4 s and Python's open()
    doesn't request FILE_SHARE_DELETE, so on Windows os.remove can raise
    PermissionError while the GUI holds it open. That must never propagate out of
    the epoch boundary — or (worse) trip the re-encode failure path AFTER a
    successful encode, re-queueing captions that were already applied. The file
    is consumed on the next boundary either way."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.debug("[caption-fix] could not remove %s (%s) — a reader holds it open; "
                     "it will be cleaned up next boundary", os.path.basename(path), e)


def _apply_caption_updates(output_dir, group, te_path, device, dit, blocks_to_swap, loss_watch, epoch,
                           *, auto_recaption=False, trigger_word=None, trigger_position="start",
                           recaptioned=None,
                           image_dir=None, caption_ext=".txt", recaption_instruction=None,
                           recaption_instruction_detailed=None):
    """Live caption repair (Problem Images window). Consume <output_dir>/loss_log/caption_updates.json
    ({item_key: new_caption}), re-encode each caption with Qwen3-VL, and OVERWRITE the item's
    text-embedding cache file — the collate re-reads that file from disk every step, so the very
    next epoch trains on the corrected caption. Also resets the image's loss-watch history (its
    stuck record reflects the old caption). Never raises into the training loop.

    auto_recaption: additionally re-caption CONFIRMED-STUCK images with the same Qwen3-VL (it's a
    full VLM with a real LM head — the captioner ships inside the training stack), appending
    "<trigger_word>, " (leading) when one is set. Max TWO attempts per image per run (`recaptioned` is a
    {key: attempts} dict): attempt 1 = standard caption; if the image re-confirms stuck after its
    history reset (~5-6 epochs later, i.e. the first caption demonstrably failed), attempt 2 =
    exhaustive-detail caption; after that it's permanently human-review. A manual edit already
    queued for a key always wins over the auto path. Both jobs share one DiT park + one
    text-encoder load.

    recaption_instruction / recaption_instruction_detailed: the Captions tab's Training-caption
    and Exhaustive-detail instructions, sent only when the user has edited that preset. Attempt 1
    uses the training one, attempt 2 the exhaustive one — so the escalation to exhaustive detail
    (the point of a second attempt, and what makes the two-attempt protocol a causal test
    separating caption-fixable images from entropy-limited ones) is preserved whether the user
    edited those presets or not. Either being None just means "use the built-in".

    The GUI separately rewrites the .txt for manual edits; the auto path writes the .txt itself
    (image_dir + caption_ext from the dataset TOML) so fixes survive future re-caches. The 8 GB
    text encoder won't co-fit with the resident training DiT on smaller cards, so the DiT is
    parked on CPU around the encode (same dance as previews)."""
    path = os.path.join(output_dir, "loss_log", "caption_updates.json")
    updates = {}
    processing = path + ".processing"
    if os.path.exists(path):
        try:
            os.replace(path, processing)  # atomic claim — GUI edits during processing land in a fresh file
            with open(processing, encoding="utf-8") as f:
                updates = {str(k): str(v).strip() for k, v in json.load(f).items() if str(v).strip()}
        except Exception:
            logger.warning("[caption-fix] could not read caption_updates.json — skipping", exc_info=True)
            return

    # Auto-recaption candidates: confirmed stuck, not already handled this run, not manually
    # queued (the human's edit wins), and the source image must be findable on disk.
    auto_todo = []
    if auto_recaption and loss_watch is not None and image_dir and os.path.isdir(image_dir):
        confirmed = {k for k, v in loss_watch.verdicts.items() if v == "stuck"}
        for k in sorted(confirmed):
            attempts = recaptioned.get(k, 0) if recaptioned is not None else 0
            if k in updates or attempts >= 2:
                continue
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                p = os.path.join(image_dir, os.path.basename(k) + ext)
                if os.path.exists(p):
                    auto_todo.append((k, p, attempts + 1))
                    break

    if not updates and not auto_todo:
        _remove_claimed_queue(processing)
        return
    if not te_path:
        logger.warning("[caption-fix] caption work is pending but no text encoder path was passed "
                       "(--text_encoder). Leaving the queue for a run with previews configured.")
        if updates:  # put the claim back (atomic + merged — the GUI may have queued more edits)
            try:
                newer = {}
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        newer = json.load(f)
                merged = {**updates, **newer}  # newer GUI edits win
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)
                os.replace(path + ".tmp", path)
                _remove_claimed_queue(processing)
            except Exception:
                pass
        return

    # item_key -> ItemInfo (training items come from the cache-driven path, so item_key is the
    # image basename without extension — same key the loss watch and the GUI use).
    items = {}
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        if bm is None:
            continue
        for bucket in bm.buckets.values():
            for it in bucket:
                items[str(it.item_key)] = it
    # todo entries: (key, ItemInfo, caption, attempt) — attempt 0 = manual edit, 1/2 = auto.
    todo = [(k, items[k], cap, 0) for k, cap in updates.items() if k in items]
    for k in updates:
        if k not in items:
            logger.warning(f"[caption-fix] '{k}' not found in the training set — skipped")
    auto_todo = [(k, p, a) for k, p, a in auto_todo if k in items]
    if not todo and not auto_todo:
        _remove_claimed_queue(processing)
        return

    logger.info(f"[caption-fix] epoch boundary {epoch}: {len(todo)} manual edit(s), "
                f"{len(auto_todo)} stuck image(s) to auto-recaption...")
    dit.to("cpu")
    if getattr(dit, "_nf4_quantized", False):
        from fizgig.modules.nf4 import move_nf4_to_device
        move_nf4_to_device(dit, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    ok = False
    try:
        from fizgig.krea2.utils import load_krea2_text_encoder
        from fizgig.krea2.caching import encode_and_save_text
        from fizgig.krea2.embedder import generate_caption
        encoder = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)

        # Auto-recaption: the SAME loaded VLM describes what's actually in the stuck image.
        # The trigger goes FIRST by default, matching what the Captions tab writes — a dataset
        # must not end up with the trigger leading on some images and trailing on others.
        # Leading is also the right call when the trigger is a real name (base-model
        # fine-tuning): the name is the subject, not an afterthought. trigger_position="end"
        # restores the weaker trailing claim, which suits a conditional trigger on a LoRA.
        # Attempt 2 (the first caption demonstrably failed) goes exhaustive-detail.
        if recaption_instruction or recaption_instruction_detailed:
            _which = " + ".join(
                w for w, v in (("Training caption", recaption_instruction),
                               ("Exhaustive detail", recaption_instruction_detailed)) if v)
            logger.info("[auto-recaption] using your edited instruction for: %s", _which)
        for k, img_path, attempt in auto_todo:
            try:
                # Attempt 1 uses the Training-caption instruction, attempt 2 the Exhaustive one —
                # the user's edited version of each where they have one, the built-in otherwise.
                # The escalation to exhaustive detail is preserved either way.
                _instr = (recaption_instruction_detailed if attempt >= 2
                          else recaption_instruction) or None
                cap = generate_caption(encoder, img_path, detailed=(attempt >= 2),
                                       instruction=_instr)
                if trigger_word:
                    cap = (f"{cap}, {trigger_word}" if str(trigger_position) == "end"
                           else f"{trigger_word}, {cap}")
                cap_path = os.path.join(image_dir, os.path.basename(k) + caption_ext)
                try:
                    with open(cap_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                except Exception:
                    logger.warning(f"[auto-recaption] could not write {cap_path} — the live run is "
                                   f"fixed but a future re-cache will use the old caption")
                todo.append((k, items[k], cap, attempt))
                logger.info(f"[auto-recaption] {os.path.basename(k)} (attempt {attempt}/2"
                            f"{', detailed' if attempt >= 2 else ''}): \"{cap[:110]}"
                            f"{'…' if len(cap) > 110 else ''}\"")
            except Exception:
                logger.warning(f"[auto-recaption] captioning failed for {os.path.basename(k)} — "
                               f"skipped (will retry next boundary)", exc_info=True)

        if not todo:
            del encoder
            _remove_claimed_queue(processing)
            return
        for _, item, cap, _auto in todo:
            item.caption = cap
        for i in range(0, len(todo), 4):  # small chunks — captions pad to the longest in the batch
            encode_and_save_text(encoder, [item for _, item, _, _ in todo[i:i + 4]])
        del encoder
        ok = True
        # Mark auto-recaptioned keys only AFTER a successful encode — a failed boundary must be
        # allowed to retry them (their captions are re-queued in the failure path below).
        if recaptioned is not None:
            for k, _, _, attempt in todo:
                if attempt > 0:
                    recaptioned[k] = max(recaptioned.get(k, 0), attempt)
        if loss_watch is not None:
            for k, _, _, _ in todo:
                loss_watch.reset_key(k)
            # After the 2nd (detailed) AI caption, the benefit of the doubt is spent: if the
            # image re-confirms stuck, it goes STRAIGHT to the LR floor — no escalation ladder.
            # reset_key cleared any prior mark, so a manual human edit (attempt 0) restores hope.
            for k, _, _, attempt in todo:
                if attempt >= 2:
                    loss_watch.mark_incorrigible(k)
        # Ack for the GUI (row badge "caption re-encoded @ epoch N" / "AI re-captioned").
        # Per-fix HISTORY list per key — last-writer-wins lost every fix but the final one,
        # so an image fixed twice replayed only its last reset on resume and the pre-first-fix
        # records (the old caption's, usually the worst in the run) skewed the thresholds
        # every other image is judged against. Older files carry a single dict per key.
        applied_path = os.path.join(output_dir, "loss_log", "caption_updates_applied.json")
        applied = {}
        try:
            if os.path.exists(applied_path):
                with open(applied_path, encoding="utf-8") as f:
                    applied = {k: (v if isinstance(v, list) else [v])
                               for k, v in json.load(f).items()}
        except Exception:
            applied = {}
        for k, _, cap, attempt in todo:
            applied.setdefault(k, []).append(
                {"epoch": epoch, "caption": cap, "auto": attempt > 0, "attempt": attempt})
        # Atomic write — the GUI polls this file for the row badges.
        with open(applied_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(applied, f, indent=2)
        os.replace(applied_path + ".tmp", applied_path)
        _remove_claimed_queue(processing)
        logger.info(f"[caption-fix] {len(todo)} caption(s) re-encoded — next epoch trains on the "
                    f"fixed text. Loss-watch history reset for: "
                    + ", ".join(os.path.basename(k) for k, _, _, _ in todo))
    except Exception:
        logger.warning("[caption-fix] re-encode failed — training continues on the old captions; "
                       "the edits stay queued and will be retried next epoch.", exc_info=True)
        # Put the claim back, merging any edits the GUI queued while we were processing. The
        # already-generated auto captions re-queue as if manual — no need to regenerate them.
        try:
            newer = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    newer = json.load(f)
            auto_caps = {k: cap for k, _, cap, attempt in todo if attempt > 0}
            merged = {**updates, **auto_caps, **newer}  # newer GUI edits win
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            os.replace(path + ".tmp", path)
            _remove_claimed_queue(processing)
        except Exception:
            pass
    finally:
        # Restore the training DiT's placement exactly as load_dit_for_training left it.
        # (A garbled duplicate of this block previously ran `dit.to(device)` on every
        # non-NF4 run, hoisting all 28 blocks onto the GPU and undoing the block-swap
        # placement two lines above it — OOM on small cards at the next step.)
        gc.collect()
        torch.cuda.empty_cache()
        # Placement first, THEN the NF4 packed weights — the two are complementary, not
        # alternatives: nn.Module.to() moves the ordinary params/buffers, move_nf4_to_device
        # moves `_nf4_packed`/`_nf4_state`, which are plain attributes .to() cannot see.
        # Making the NF4 case an `elif` stranded every ordinary parameter on CPU.
        if blocks_to_swap > 0:
            dit.move_to_device_except_swap_blocks(torch.device(device))
            dit.switch_block_swap_for_training()
        else:
            dit.to(device)
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.modules.nf4 import move_nf4_to_device
            move_nf4_to_device(dit, device)
        dit.train()
    return ok


def sample_previews(turbo_path, ae, encoded_prompts, lora_sd, out_dir, epoch, *,
                    output_name="krea2", steps=8, cfg_scale=1.0, neg=None,
                    width=512, height=512,
                    seed=42, context_lora_path=None, context_lora_strength=1.0,
                    blocks_to_swap=0, int8=False, device="cuda", prompts=None):
    """Load the (clean) pre-quant fp8 Turbo, apply the current LoRA LIVE (no merge -> no grid),
    and render each pre-encoded prompt. Turbo is freed afterwards.

    `blocks_to_swap` > 0 puts the Turbo on forward-only block swap so previews fit smaller cards
    (mirrors Klein's Distilled sample-model auto-swap). Order mirrors load_dit_for_training: load
    the base on CPU, apply the LoRA(s), then enable swap + place the resident blocks.

    Filenames follow the Fizgig samples-gallery pattern
    `{name}_e{epoch:06d}_{idx:02d}_{timestamp:14d}_{seed}.png` so the live preview gallery
    (which parses that exact format) picks them up — same as the Klein training path."""
    from fizgig.krea2.utils import load_krea2_dit
    from fizgig.networks.lora import create_network_from_weights

    _ld = "cpu" if blocks_to_swap > 0 else device
    turbo = load_krea2_dit(turbo_path, device=device, dtype=torch.bfloat16,
                           loading_device=_ld)  # prequant fp8 auto-detected
    if int8:
        # INT8 (W8A8) fast preview matmul — quantize the block Linears BEFORE the LoRA wraps them
        # (so the LoRA wraps the int8 forward) and before block swap (so the offloader stages int8).
        # Quantize on the load device so a swapped (CPU-loaded) model doesn't need the whole int8
        # model resident on GPU.
        from fizgig.modules.int8 import apply_int8_quantization
        from fizgig.krea2.utils import (KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                        KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS)
        apply_int8_quantization(turbo, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
                                compute_device=torch.device(_ld))
    # Context LoRA (frozen) goes on FIRST so previews match deployment: the trained LoRA runs
    # on top of the same context at the same strength it was trained with.
    ctx_net = None
    if context_lora_path:
        ctx_net = _apply_context_lora(turbo, context_lora_path, context_lora_strength,
                                      device=device, dtype=torch.bfloat16)
    net = create_network_from_weights(None, 1.0, lora_sd, None, turbo, for_inference=True)
    net.apply_to(text_encoders=None, unet=turbo, apply_text_encoder=False, apply_unet=True)
    # create_network_from_weights only builds the module STRUCTURE (sizes from dims/alphas);
    # the trained values must be loaded in, or the LoRA stays at its zero init (lora_up=0) and
    # contributes nothing — which made every epoch's preview identical. Mirrors the Klein path
    # (inference.py: apply_to -> load_state_dict(strict=False)).
    net.load_state_dict(lora_sd, strict=False)
    net.to(device=device, dtype=torch.bfloat16).eval()
    if blocks_to_swap > 0:
        from fizgig.krea2.offloading import BlockSwapConfig
        turbo.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=False))
        turbo.move_to_device_except_swap_blocks(torch.device(device))
        turbo.switch_block_swap_for_inference()
    turbo.eval()
    result = _render_prompt_set(turbo, ae, encoded_prompts, out_dir, epoch,
                                output_name=output_name, steps=steps, cfg_scale=cfg_scale,
                                neg=neg, width=width, height=height, seed=seed, device=device,
                                prompts=prompts)
    del turbo, net, ctx_net
    torch.cuda.empty_cache()
    return result


def _render_prompt_set(model, ae, encoded_prompts, out_dir, epoch, *, output_name, steps,
                       cfg_scale, neg, width, height, seed, device, prompts=None):
    """Shared preview render loop — identical settings (mu=1.15 pinned) and the exact gallery
    filename pattern for both the Turbo-model path and the turbo-LoRA-on-training-DiT path.

    `prompts` is the raw text parallel to `encoded_prompts` (same order, same length) — optional
    because a couple of call sites don't have it handy, but when it's there we hand back which
    prompt made the LAST image, so a checkpoint saved right after can default its description to
    it instead of shipping blank (same idea as the auto-thumbnail, which is already whatever
    sample happens to be newest on disk)."""
    import datetime
    from fizgig.krea2 import sampling
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 14-digit timestamp
    paths = []
    last_prompt = None
    # Negative prompt rides through the CFG path (untxt) — only when CFG is actually on.
    _untxt = _untxtmask = None
    if neg is not None and cfg_scale and cfg_scale > 1.0:
        _untxt, _untxtmask = neg
    for i, (txt, txtmask) in enumerate(encoded_prompts):
        with torch.no_grad():
            imgs = sampling.sample(model, ae, txt, txtmask, untxt=_untxt, untxtmask=_untxtmask,
                                   device=device, dtype=torch.bfloat16, width=width, height=height,
                                   steps=steps, cfg_scale=cfg_scale, mu=1.15, seed=seed + i)
        p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
        imgs[0].save(p)
        paths.append(p)
        if prompts and i < len(prompts):
            last_prompt = prompts[i]
    return paths, last_prompt


def sample_previews_on_dit(dit, turbo_net, turbo_diffb, ae, encoded_prompts, out_dir, epoch, *,
                           output_name="krea2", steps=8, cfg_scale=1.0, neg=None,
                           width=512, height=512, seed=42, blocks_to_swap=0, device="cuda",
                           prompts=None):
    """Render previews on the RESIDENT training DiT with the Turbo LoRA enabled at 1.0 —
    no Turbo checkpoint load, no parking the trainer to CPU.

    Stack order at render time is turbo (innermost) -> context -> trainable, i.e. the model
    behaves as Turbo, the context rides on it, and the LoRA being trained samples on top —
    matching deployment. The trainable network is live, so previews reflect current weights
    with no save/reload round-trip.

    Everything is reverted in the finally: turbo net disabled + back to CPU, bias deltas
    undone by exact snapshot restore, block swap returned to training mode, train() mode
    re-entered. Training state is untouched whether the render succeeds or raises.

    `blocks_to_swap` here is the TRAINING swap setting (this is the training model), not the
    preview-model swap the Turbo-checkpoint path uses.
    """
    saved_biases = []
    was_training = dit.training
    try:
        dit.eval()
        if blocks_to_swap > 0:
            dit.switch_block_swap_for_inference()
        turbo_net.to(device=device)
        turbo_net.set_enabled(True)
        for bias, delta in turbo_diffb:
            saved_biases.append((bias, bias.detach().clone()))
            bias.data.add_(delta.to(device=bias.device, dtype=bias.dtype))
        return _render_prompt_set(dit, ae, encoded_prompts, out_dir, epoch,
                                  output_name=output_name, steps=steps, cfg_scale=cfg_scale,
                                  neg=neg, width=width, height=height, seed=seed, device=device,
                                  prompts=prompts)
    finally:
        for bias, snap in saved_biases:
            bias.data.copy_(snap)
        turbo_net.set_enabled(False)
        turbo_net.to(device="cpu")
        if blocks_to_swap > 0:
            dit.switch_block_swap_for_training()
        if was_training:
            dit.train()
        torch.cuda.empty_cache()


def train_krea2(
    raw_path: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    network_type: str = "lora",     # "lora" | "lokr" (Kronecker, full-matrix w2)
    lokr_factor: int = 8,           # LoKR only: w1 is ~factor x factor; dim/alpha unused
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    # Resumable state dirs. Pause/Resume saves state regardless of these — they only govern the
    # automatic saves. Each dir is LoRA + optimizer moments (~474 MB at rank 32), hence keep_n.
    save_state: bool = False,
    save_state_on_train_end: bool = False,
    keep_last_n_states: int = 2,
    fp8_scaled: bool = True,
    fast_ft: bool = False,
    reg_lr_multiplier: float = 0.2,
    quant_4bit: bool = False,
    quant_int8: str = "",
    # FINE-TUNE ONLY. NF4 is the fine-tune default; this is the escape hatch back to an fp8
    # frozen trunk. It exists because at the CLI "fp8" and "said nothing" are the same thing
    # (fp8_scaled = not --no_fp8, i.e. True by default), so without an explicit signal the
    # new default would silently swallow a deliberate fp8 pick and turn the GUI's fp8 option
    # into a lie. Ignored outside rotation.
    ft_base_fp8: bool = False,
    blocks_to_swap: int = 0,
    shift: float = 2.5,
    # Timestep window (0-1 scale): restrict training to a noise band. High-t-only training
    # teaches structure/layout without touching the detail-rendering regime — the quality
    # protection when training on low-res data (e.g. temporal-displacement pairs).
    min_timestep: float = 0.0,
    max_timestep: float = 1.0,
    # Paired-image runs only: upweight target tokens where source and target actually differ
    # (kills the copy shortcut on mostly-static pairs). 0 = off; ~0.7 is a strong setting.
    motion_weighted_loss: float = 0.0,
    # Image-pair slider training: the adapter learns a signed attribute direction from
    # positive/negative image pairs (training image = positive, its control_directory match
    # = negative). Strength is the dial at inference: +N pushes toward the positive pole,
    # -N away. Captions must NOT name the attribute — the multiplier carries it.
    slider_pairs: bool = False,
    slider_diff_weight: float = 1.0,
    # Rotation FT resume: 0-based window the schedule starts at. A resumed run is a fresh
    # process, so without this it re-runs window 0 (attn) instead of the next unfinished one.
    finetune_start_window: int = 0,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    # Effective batch = batch_size (1) x this. Grads accumulate over N micro-batches, then one
    # optimizer step. Per-image LR still applies per micro-batch (each image scales its own loss).
    gradient_accumulation_steps: int = 1,
    # Optimizer family + free-form kwargs ("weight_decay=0.01 betas=0.9,0.99").
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    compile_blocks: str = "auto",   # "auto" | "on" | "off"
    # LR schedule (step-level). Ignored when adaptive_lr is on — that watcher owns the LR.
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    lr_decay_steps: int = 0,
    lr_scheduler_num_cycles: int = 1,
    lr_scheduler_power: float = 1.0,
    # in-training previews (sample the fp8 Turbo with the live LoRA — or, when
    # turbo_lora_path is set, the resident training DiT with the Turbo LoRA @1.0)
    sample_prompts: list = None,
    turbo_path: str = None,
    turbo_lora_path: str = None,
    vae_path: str = None,
    te_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_width: int = 512,
    sample_height: int = 512,
    sample_steps: int = 8,
    sample_cfg_scale: float = 1.0,   # >1 enables CFG on the Turbo (needs sample_negative for a real uncond)
    sample_negative: str = None,     # negative prompt, used only when sample_cfg_scale > 1
    sample_at_first: bool = False,   # render an epoch-0 preview before training starts
    sample_seed: int = 42,
    sample_ref_image: str = None,
    preview_blocks_to_swap: int = 0,
    preview_int8: bool = False,
    log_per_image_loss: bool = False,
    per_image_lr: bool = False,
    auto_recaption: bool = False,
    warmup_look_outliers: bool = False,
    trigger_word: str = None,
    # Captions-tab instructions for auto-recaption: the Training-caption preset for attempt 1,
    # the Exhaustive-detail preset for attempt 2. Each sent only when the user edited that preset.
    recaption_instruction: str = None,
    recaption_instruction_detailed: str = None,
    resume_state_dir: str = None,
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # Rotating-block FULL fine-tune (experimental). >0 trains that many DiT blocks at a
    # time in bf16 while the rest stay fp8-frozen, rotating the window every N epochs.
    # No LoRA is trained in this mode — the output is a full model checkpoint.
    # Where the trigger word lands in auto-generated captions: "start" (matches the
    # Captions tab, and right for a real-name trigger) or "end" (weaker claim).
    trigger_position: str = "start",
    finetune_rotation: int = 0,
    finetune_rotate_every: int = 1,
    # "block" = contiguous depth slices; "component" = attn across ALL blocks, then
    # mlp — same VRAM, but every window spans the model's full depth.
    finetune_rotation_mode: str = "block",
    # Step each parameter's optimizer inside backward and free its grad immediately, so the
    # whole active window's gradients never coexist. Saves roughly the gradient footprint.
    finetune_fused_backward: bool = False,
    # Output metadata (Other Options → Metadata in the GUI) — recorded in the saved LoRA.
    metadata_title: str = None,
    metadata_author: str = None,
    metadata_description: str = None,
    metadata_license: str = None,
    metadata_tags: str = None,
    metadata_trigger_phrase: str = None,
    metadata_thumbnail: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native Krea 2 LoRA training: bucketed multi-resolution dataloader over the krea2 caches ->
    flow-matching loss -> AdamW -> save a ComfyUI-compatible LoRA. In-training Turbo previews +
    GUI wiring are layered on elsewhere."""
    validate_output_name(output_name)     # before the model loads, not an epoch later (#70)
    torch.manual_seed(seed)

    # Updated at every sample render (see the sample_previews*/prompts= call sites below) so
    # _sai_metadata can default the description to whatever prompt made the newest thumbnail,
    # instead of shipping blank.
    _last_sample_prompt = None

    def _sai_metadata():
        """SAI ModelSpec block shared by every checkpoint (per-epoch and final), so an epoch
        picked over the last one is just as identifiable in ComfyUI. Thumbnail defaults to
        whatever preview training has produced so far — cosmetic, so a missing one is fine."""
        if metadata_thumbnail and metadata_thumbnail.lower() in ("off", "none"):
            thumb_source = None
        elif metadata_thumbnail:
            thumb_source = metadata_thumbnail
        else:
            thumb_source = latest_sample_image(output_dir)
        return build_metadata(
            None, ARCHITECTURE_KREA2, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author,
            description=(metadata_description if metadata_description is not None
                         else _last_sample_prompt),
            license=metadata_license, tags=metadata_tags,
            trigger_phrase=metadata_trigger_phrase,
            thumbnail=thumbnail_data_uri(thumb_source),
        )

    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run the krea2 cache scripts first.")
    logger.info(f"Krea 2 training: {group.num_train_items} items, {max_train_epochs} epochs")

    ft_rotation = max(0, int(finetune_rotation or 0))

    if ft_rotation:
        # Handoff guards, before our first CUDA call: a back-to-back fine-tune can start
        # while the previous trainer process is still tearing down — VRAM (WDDM demotion is
        # sticky) and RAM (the old process hands back a huge commit) both need to settle.
        # See the guards' docstrings in capabilities.py.
        from fizgig.utils.capabilities import wait_for_gpu_handoff, wait_for_ram_recovery
        wait_for_gpu_handoff()
        wait_for_ram_recovery()

    if slider_pairs:
        # Slider mode's structural coercions. FT and sliders are mutually exclusive (a slider
        # IS an adapter; FT has none). The per-image loss watch and adaptive LR both assume
        # one target per image — a slider step has two poles, so their signals would be
        # meaningless noise. Motion weighting is the paired-EDIT path's knob; the slider owns
        # its pairs and never packs them.
        if ft_rotation:
            raise RuntimeError("[slider] slider training and base-model fine-tuning are "
                               "mutually exclusive — turn one off.")
        if adaptive_lr:
            logger.info("[slider] adaptive LR disabled — its plateau signals assume one "
                        "target per image; a slider step has two poles.")
            adaptive_lr = False
        if log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers:
            logger.info("[slider] per-image loss watch disabled for the same reason.")
            log_per_image_loss = per_image_lr = auto_recaption = warmup_look_outliers = False
        motion_weighted_loss = 0.0
        logger.info("[slider] IMAGE-PAIR SLIDER TRAINING: each step trains the adapter at "
                    "+1 toward the training image and at -1 toward its paired control; "
                    "diff-weight %.2f concentrates the loss where the pair differs. Keep "
                    "captions neutral — the attribute must live in the multiplier, not the "
                    "text.", float(slider_diff_weight))

    # Fine-tune trains the BASE weights — the LoRA/LoKR network is built but inert, so a LoKR
    # request would only burn VRAM on parameters that are never trained or saved. Coerce with a
    # loud log (the GUI also hides the Network Type control under fine-tune).
    if ft_rotation and network_type == "lokr":
        logger.warning("[ft-rotation] --network_type lokr is ignored under base-model "
                       "fine-tuning (the adapter is inert) — proceeding as standard.")
        network_type = "lora"
    # Resume restores an INERT LoRA + an optimizer that may not even exist under fused
    # backward — it would silently restart the base from RAW while looking like a resume.
    # The full checkpoints are the continuation point: swap --dit to the saved checkpoint.
    if ft_rotation and resume_state_dir:
        raise RuntimeError(
            "--resume is not supported in fine-tune mode: state dirs hold the (inert) LoRA, "
            "not the base weights. To continue a fine-tune, point --dit at your last saved "
            "checkpoint (structurally identical to the RAW base) and set "
            "--finetune_start_window to the value printed at that save.")

    # Auto window mode: size the rotation window to the free VRAM. Resolved here rather than
    # in the GUI so headless runs get it too, and so it reads the VRAM actually available at
    # the moment training starts rather than whenever a dialog was last opened.
    if ft_rotation and str(finetune_rotation_mode).lower().startswith("auto"):
        from fizgig.utils.capabilities import recommend_ft_rotation
        _mode, _blocks, _stream, _why = recommend_ft_rotation()
        finetune_rotation_mode = _mode
        ft_rotation = _blocks
        if _stream and blocks_to_swap <= 0:
            blocks_to_swap = 12      # any positive value switches on rotation-aware streaming
        for _line in _why:
            logger.info("[ft-auto] %s", _line)

    # Continuation numbering: a fine-tune continued from a saved checkpoint is a fresh
    # process whose local epochs restart at 1, so `<name>-000001.safetensors` would silently
    # OVERWRITE the original run's first checkpoint. The --dit file IS the previous
    # checkpoint, and its trailing epoch number is the true count of epochs already trained —
    # offset every checkpoint filename by it. A renamed file just starts numbering at 1.
    ft_epoch_offset = 0
    if ft_rotation:
        _m = re.search(r"-(\d{6})\.safetensors$", os.path.basename(raw_path or ""))
        if _m:
            ft_epoch_offset = int(_m.group(1))
        else:
            # The FINAL checkpoint carries no epoch number in its name — its cumulative
            # count rides in the metadata instead (stamped at every FT save below).
            try:
                from safetensors import safe_open
                with safe_open(raw_path, framework="pt") as _f:
                    _md = _f.metadata() or {}
                ft_epoch_offset = int(_md.get("fizgig_ft_epochs_done", 0))
            except Exception:
                ft_epoch_offset = 0
        if ft_epoch_offset:
            logger.info("[ft-rotation] continuing from %s — checkpoint numbering starts at "
                        "epoch %d", os.path.basename(raw_path), ft_epoch_offset + 1)

    # --- Regularisation set (fine-tune only) ---------------------------------------------
    # Images from a dataset block marked `is_reg = true` — a PRIOR ANCHOR, not a subject.
    # Full fine-tuning moves every weight, so 40 epochs on a handful of subjects drifts the
    # model's whole notion of people, with no low-rank bound to limit it. Reg data pulls back,
    # and it only works if it stays a nudge: a fixed reduced LR.
    #
    # LoRA training doesn't have that problem — the update is rank-bounded — so reg images are
    # ignored outright rather than quietly trained as subjects at full LR.
    reg_keys = set()
    reg_mult = float(reg_lr_multiplier)
    if ft_rotation:
        for _ds in group.datasets:
            if not getattr(_ds, "is_reg", False):
                continue
            _bm = getattr(_ds, "batch_manager", None)
            if _bm is None:
                continue
            for _bucket in _bm.buckets.values():
                for _it in _bucket:
                    reg_keys.add(str(_it.item_key))
        if reg_keys:
            logger.info(f"[reg] {len(reg_keys)} regularisation image(s) at x{reg_mult:g} LR "
                        f"({group.num_train_items - len(reg_keys)} subject items).")
            if len(reg_keys) >= group.num_train_items - len(reg_keys):
                logger.warning("[reg] regularisation images are at least half the training set — "
                               "the multiplier only reads as an LR cut while they are the "
                               "minority, since Adafactor normalises by a second moment the "
                               "majority dominates.")
    elif any(getattr(_ds, "is_reg", False) for _ds in group.datasets):
        logger.warning("[reg] the dataset config has a regularisation block, but this is a LoRA "
                       "run — regularisation images are a fine-tune feature and are IGNORED "
                       "here. They will train as ordinary images at full LR; remove the "
                       "`is_reg` block from the TOML if that is not what you want.")
    ft_stream_frozen = False

    if ft_rotation:
        # Rotation owns the block weights: it swaps them between fp8-frozen and bf16-trainable
        # in place. Block swap moves whole blocks to CPU behind the offloader's back, and 4-bit
        # keeps weights packed in _nf4_packed — neither survives that swap, so both are off.
        # Resolved here, with the other quantisation/swap interactions below, so everything
        # downstream (compile decision, loader, offloader) reads settled values.
        if blocks_to_swap > 0:
            # Rotation brings its own swap policy (RotationOffloader): the trainable window is
            # pinned and every other block streams. The stock offloader can't express that —
            # it keeps a fixed contiguous prefix resident.
            logger.info("[ft-rotation] using rotation-aware block swap instead of the "
                        "fixed-prefix offloader (--blocks_to_swap value ignored).")
            blocks_to_swap = 0
            ft_stream_frozen = True
        # NF4 IS THE FINE-TUNE DEFAULT (Peter, 27 Aug: "lets make FT default to NF4").
        # Measured on the same dataset and the same budget, NF4 beats fp8 at BOTH tested
        # tiers, and at 24 GB it changes the plan shape rather than merely fitting:
        #   24 GB fp8 -> 8 windows, depth-split AND streamed, ~3.0 s/it
        #   24 GB NF4 -> 4 FULL-DEPTH windows, resident, ~1.02 s/it (peaks 15.7-16.0 GB)
        #   16 GB fp8 -> does not complete (allocator fragmentation)
        #   16 GB NF4 -> completes, peaks 8.6-11.0 GB
        # So it is ~3x the step speed AND half the cycle (4 epochs per full pass, not 8).
        # This applies to the FINE-TUNE path only — the LoRA recommender is untouched, and
        # is exactly why "Auto" could never reach 4-bit here: it is LoRA-shaped, has no
        # fine-tune awareness, and prefers INT8 (which rotation cannot use at all).
        # An EXPLICIT fp8 pin is still honoured; this only fills the gap where the user
        # expressed no usable preference.
        # NOTE ON WHY THIS IS NOT JUST `if not fp8_scaled`: fp8_scaled arrives as
        # `not args.no_fp8`, i.e. TRUE unless the user asked for a bf16 base. So "the user
        # wants fp8" and "the user said nothing" are the SAME value here, and a default
        # keyed on it would never fire. The explicit signal is ft_base_fp8.
        if quant_int8 and not ft_base_fp8:
            # int8 keeps its own packed weights + scales, which the bf16-master round-trip
            # would have to undo and redo every window. It is not a choice we can keep, so
            # substitute the best base that rotation CAN use rather than the middle one.
            logger.info("[ft-rotation] INT8 base is incompatible with rotation — using "
                        "4-bit NF4 instead (the fastest base rotation supports: measured "
                        "~3x fp8's step speed at 24 GB, and the only one that fits 16 GB).")
            quant_int8 = ""
            quant_4bit = True
        elif quant_int8:
            logger.info("[ft-rotation] INT8 base is incompatible with rotation — honouring "
                        "the explicit fp8 request instead.")
            quant_int8 = ""
        if not quant_4bit and not ft_base_fp8:
            logger.info("[ft-rotation] frozen base: 4-bit NF4 (the fine-tune default). It "
                        "keeps full-depth component windows resident on a 24 GB card — 4 "
                        "windows at ~3x fp8's step speed — and is what makes 16 GB "
                        "possible at all. Choose fp8 in Base precision (CLI --ft_base_fp8) "
                        "for the fp8 trunk instead.")
            quant_4bit = True
        elif ft_base_fp8 and not quant_4bit:
            logger.info("[ft-rotation] frozen base: fp8, by explicit request (the "
                        "fine-tune default is 4-bit NF4).")
            fp8_scaled = True
        if quant_4bit:
            # Announced AFTER resolution, not before it: the branches above can turn 4-bit
            # on, so a banner placed earlier would stay silent on exactly the runs that
            # took the new default — the class of lie this file has been fixing all day.
            # The rotator branches per module: activate reads the bf16 master and frees
            # `_nf4_packed`; deactivate re-encodes with quantize_nf4 and restores the
            # patched forward. The trade, as on H3: the frozen CONTEXT the active window
            # trains against carries NF4's error. The saved checkpoint does not — it is
            # written bf16 from the master, which never sees a quantizer.
            fp8_scaled = False
            logger.info("[ft-rotation] 4-bit base: the frozen trunk holds NF4 (~half the "
                        "fp8 footprint) while the trainable window runs bf16 from the "
                        "master. The window trains against a coarser frozen context; the "
                        "saved checkpoint is unaffected (bf16, straight from the master).")
        if fp8_scaled and not quant_4bit:
            # Only reachable now by an EXPLICIT fp8 choice — the defaults above never land
            # here. Measured bands, not guesses: the fp8 trunk (~13 GB) fine-tunes at 24 GB
            # but runs out of usable memory below ~20; NF4 (6.08 GB packed) completes a
            # 16 GB run with ~5 GB spare. Honour the pin, but say what it costs.
            try:
                from fizgig.utils.device import plannable_free_vram as _pfv_warn
                _free_now = _pfv_warn()
            except Exception:
                _free_now = None
            if _free_now is not None and _free_now < 20.0:
                logger.warning(
                    "[ft-rotation] fp8 frozen base was requested explicitly on a ~%.0f GB "
                    "card: measured, fp8 fine-tunes at 24 GB but runs out of usable memory "
                    "below ~20 GB. Drop the fp8 pick (or pass --quantize_4bit) to get the "
                    "4-bit default back — half the trunk, and it completes a 16 GB run.",
                    _free_now)

    # Resolve quantisation/swap interactions BEFORE anything reads blocks_to_swap —
    # should_compile used to be consulted with a swap value the NF4 branch zeroed a few
    # lines later, declining compile "because block swap is active" about a swap that no
    # longer existed (NF4 + compile is the one combination measured VRAM-neutral).
    if quant_4bit and blocks_to_swap > 0:
        logger.info("[nf4] 4-bit base is incompatible with block swap (weights live in _nf4_packed) "
                    "— forcing blocks_to_swap=0.")
        blocks_to_swap = 0
    if quant_int8 and blocks_to_swap > 0:
        # The int8 path stages on CPU then makes the whole quantised model resident —
        # swap could never engage before the full residency, so it OOM'd at load on
        # exactly the cards that asked for swapping. INT8 residency is ~its own budget;
        # cards that need swap should use fp8+swap or NF4 instead.
        logger.info("[int8] W8A8 base is fully resident (staged quantise -> GPU) — block swap "
                    "can't reduce its footprint; forcing blocks_to_swap=0.")
        blocks_to_swap = 0

    # torch.compile: "auto" weighs its ~90 s warm-up against how long this run actually is, which
    # is knowable here because the dataset is already built. Short runs are a straight loss, so the
    # default must not simply turn it on. "on"/"off" are the explicit overrides.
    _do_compile = str(compile_blocks).lower() in ("1", "true", "on", "yes")
    if str(compile_blocks).lower() == "outside":
        # Explicit high-res boundary (#99): checkpoint outside the compiled region.
        _do_compile = "outside"
    # The largest ACTUAL bucket, not the Target Megapixels box — bucket_no_upscale can land
    # buckets well below the target, and it's the real token count that sets compiled-path
    # VRAM. Batch rides along because it multiplies tokens per step the same way. Unreadable
    # values fall back to the defaults, i.e. the pre-shape-aware behaviour.
    _mp_max, _batch_max = 0.25, 1
    try:
        _mp_max = max(w * h / 1e6 for ds in group.datasets
                      for (w, h) in ds.batch_manager.bucket_resos)
        _batch_max = max(int(ds.batch_size) for ds in group.datasets)
    except Exception:
        pass
    if _do_compile is True:
        # Explicit On means ON — but the checkpoint boundary is still placed where it
        # fits (#99): forced inside-the-graph at 1 MP measured >32 GB and OOM'd on a
        # 32 GB card; outside completed in ~18.7 GB at ~27% faster than eager.
        from fizgig.utils.capabilities import compile_boundary
        _b = compile_boundary(quant_4bit, quant_int8, mp=_mp_max, batch=_batch_max)
        if _b == "outside":
            logger.info("[compile] on: inside-the-graph won't fit at this token load — "
                        "compiling with the checkpoint OUTSIDE the region instead.")
            _do_compile = "outside"
    if str(compile_blocks).lower() == "auto":
        from fizgig.utils.capabilities import should_compile
        _steps_est = group.num_train_items * max_train_epochs
        _do_compile, _why = should_compile(_steps_est, quant_4bit, quant_int8, blocks_to_swap,
                                           mp=_mp_max, batch=_batch_max)
        logger.info("[compile] auto: %s — %s",
                    ("ENABLED (checkpoint outside)" if _do_compile == "outside"
                     else ("ENABLED" if _do_compile else "off")), _why)
    if _do_compile and ft_rotation:
        # Rotation flips requires_grad and swaps weights between the bf16 master and the GPU
        # every window, which is exactly the kind of state change a compiled graph bakes in.
        # Untested together — take the safe side rather than debug it mid-run. Catches the
        # "outside" boundary too (truthy), on purpose: the boundary changes where the
        # checkpoint sits, not the fact that the compiled graph pins its weights.
        logger.info("[compile] disabled under rotating fine-tune (the trainable set changes "
                    "every window; compiled graphs assume it doesn't).")
        _do_compile = False

    # Preview setup: pre-encode prompts (frees the 8GB encoder) + load the VAE BEFORE the RAW DiT,
    # so the encoder never coexists with the resident base.
    # sample_at_first counts as wanting previews even without a per-epoch cadence.
    # A missing turbo-LoRA file must be caught BEFORE prompts are encoded, so the fallback
    # decision (Turbo checkpoint, or no previews at all) is made once, up front.
    if turbo_lora_path and not os.path.isfile(turbo_lora_path):
        logger.warning("[preview] turbo LoRA not found at %s — %s", turbo_lora_path,
                       "falling back to the Turbo checkpoint" if turbo_path
                       else "previews need it or a Turbo checkpoint; disabling previews")
        turbo_lora_path = None
    do_previews = bool((sample_every_n_epochs or sample_at_first)
                       and sample_prompts and (turbo_path or turbo_lora_path)
                       and vae_path and te_path)
    # Under a full fine-tune the trained weights live in the BASE, so the standalone Turbo
    # checkpoint (a different model) cannot show them — the only faithful preview renders on
    # the training DiT itself with the Turbo LoRA applied fresh inside a deactivate/reactivate
    # bracket (the H3 pattern). Without the Turbo LoRA, previews stay off.
    # MUST stay after every other do_previews decision — the FT gate wins.
    _ft_preview_gap_warned = False
    if do_previews and ft_rotation and not turbo_lora_path:
        logger.info("[ft-rotation] in-training previews need the Turbo LoRA (the standalone "
                    "Turbo checkpoint can't show fine-tuned weights) and none is configured — "
                    "previews off. Evaluate saved checkpoints in ComfyUI instead.")
        do_previews = False
        _ft_preview_gap_warned = True
    encoded_prompts = sample_ae = sample_dir = None
    encoded_negative = None
    if do_previews and network_type == "lokr" and not ft_rotation:
        # Deliberately a MESSAGE, not a behaviour change (Peter, 31 Aug): the preview
        # renders on the resident training DiT with the LoKR net live, and LoKR's
        # full-matrix w2 plus its per-layer GEMM transients cost more than a 16 GB card
        # has left at that moment. If the run stalls or OOMs at a preview, this line is
        # the explanation sitting right above it in the log.
        logger.warning("[preview] heads-up: preview samples with a LoKR network need more "
                       "than 16 GB of VRAM on Krea 2 (the render runs on the resident "
                       "training DiT with the LoKR live). On a 16 GB card, train LoKR "
                       "with previews off — or use a standard LoRA if you want previews.")
    if do_previews:
        from fizgig.krea2.vae_loader import load_vae
        logger.info(f"pre-encoding {len(sample_prompts)} sample prompt(s)"
                    f"{' with reference image' if sample_ref_image else ''}...")
        encoded_prompts = encode_sample_prompts(te_path, sample_prompts, ref_image=sample_ref_image, device=device)
        if sample_negative and sample_cfg_scale and sample_cfg_scale > 1.0:
            # One shared negative embedding; only meaningful with CFG active.
            encoded_negative = encode_sample_prompts(te_path, [sample_negative], device=device)[0]
        elif sample_negative:
            logger.info("[sample] negative prompt set but CFG Scale is 1.0 (CFG off) — it will "
                        "be ignored. Set Sample CFG Scale above 1 to use it.")
        sample_ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        sample_dir = os.path.join(output_dir, "sample")

    if fast_ft:
        # Fast FT only has meaning on the fp8 path (it swaps the block-64 scale layout for a
        # per-tensor one so _scaled_mm can take it). Say so plainly rather than no-op quietly.
        from fizgig.modules.fp8 import _train_scaled_mm_supported
        if not fp8_scaled:
            logger.warning("[fast-ft] requested, but the base is not fp8 "
                           f"({'INT8' if quant_int8 else '4-bit' if quant_4bit else 'bf16'}) "
                           "— Fast FT does nothing here and is ignored.")
            fast_ft = False
        elif not _train_scaled_mm_supported():
            logger.warning("[fast-ft] requested, but this GPU has no fp8 _scaled_mm support "
                           "(needs SM 8.9+) — falling back to the standard dequant path.")
            fast_ft = False
        else:
            logger.info("[fast-ft] ON — per-tensor fp8 scales + _scaled_mm on the frozen base. "
                        "Costs ~1.5x the per-Linear forward error of the default path "
                        "(3.7e-02 vs 2.5e-02) — mostly from quantising activations to fp8, which "
                        "the fp8 GEMM requires; the scale change alone is 1.10x. "
                        "Set FIZGIG_FP8_DIAG=1 to see per-Linear SCALED/DEQUANT decisions.")

    dit, network, turbo_net, turbo_diffb = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        network_type=network_type, lokr_factor=lokr_factor,
        fp8_scaled=fp8_scaled, quant_4bit=quant_4bit, quant_int8=quant_int8,
        blocks_to_swap=blocks_to_swap, compile_blocks=_do_compile, fp8_fast=fast_ft,
        context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
        # Under FT the Turbo LoRA is NOT staged at load: the rotation bracket applies it
        # FRESH against the deactivated model at each preview and restores every wrapped
        # forward exactly afterwards — a load-time wrap would end up stashed inside the
        # rotator's forward snapshots and double-applied.
        turbo_lora_path=(turbo_lora_path if (do_previews and not ft_rotation) else None),
        device=device, dtype=dtype)
    if turbo_net is not None:
        logger.info("[preview] turbo-LoRA mode: previews render on the resident training DiT "
                    "(RAW + Turbo LoRA @1.0, same 8-step CFG-free settings) — the Turbo "
                    "checkpoint is not loaded and the trainer is never parked to CPU")
    if blocks_to_swap > 0 and not quant_4bit and not quant_int8:
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=True))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()
    dit.train()
    network.train()

    rotator = rot_schedule = None
    if ft_rotation:
        from fizgig.krea2.rotation import RotationSchedule, BlockRotator
        # The LoRA network stays created but frozen and zero-init, so it contributes nothing
        # to the forward. We're training the base weights themselves.
        network.requires_grad_(False)
        master = _build_bf16_master(raw_path, dit)
        rotator = BlockRotator(dit.blocks, master, key_prefix="blocks", device=device,
                               quantization_mode="tensor" if fast_ft else "block")
        if getattr(dit, "txtfusion", None) is not None:
            rotator.activate_always("txtfusion", dit.txtfusion)
        # Component-window plan (the small-card tiers, mirrors H3): read free VRAM, add
        # back the fp8 trunk already resident, and let the shared planner decide how many
        # depth-splits the budget forces — and whether the frozen out-of-window blocks
        # must stream. Window sizes come from the model's own Linears, not a table.
        _k2_windows = None
        if str(finetune_rotation_mode).startswith("comp"):
            from fizgig.krea2.rotation import (component_gb_per_block,
                                               plan_krea2_ft_windows,
                                               K2FT_COMPONENT_PREFIXES,
                                               krea2_trunk_gb_per_block)
            _comp_gb = component_gb_per_block(dit.blocks[0], K2FT_COMPONENT_PREFIXES)
            _trunk_per_block = krea2_trunk_gb_per_block(bool(quant_4bit))
            try:
                from fizgig.utils.device import plannable_free_vram as _pfv0
                _usable = _pfv0() + _trunk_per_block * len(dit.blocks) - 1.5
            except Exception:
                _usable = 99.0
            _k2_windows, _k2_stream, _plan_why = plan_krea2_ft_windows(
                _usable, _comp_gb, n_blocks=len(dit.blocks),
                allow_stream=os.environ.get("FIZGIG_NO_FT_STREAM") != "1",
                nf4=bool(quant_4bit))
            for _line in _plan_why:
                logger.info("[ft-rotation] %s", _line)
            if _k2_windows is None:
                raise RuntimeError(
                    f"[ft-rotation] ~{_usable:.1f} GB of usable VRAM is below what the "
                    "fine-tune needs even with depth-split windows and streamed frozen "
                    "blocks. Close other GPU apps, or train a LoRA instead.")
            if _k2_stream:
                # The planner's verdict outranks how streaming was (or wasn't) requested:
                # split windows leave the out-of-window blocks fully frozen, so the
                # rotation-aware streamer carries them regardless of the swap box.
                ft_stream_frozen = True
        rot_schedule = RotationSchedule(len(dit.blocks), active=ft_rotation,
                                        rotate_every=finetune_rotate_every,
                                        mode=finetune_rotation_mode,
                                        components=(_k2_windows if _k2_windows
                                                    else ("attn", "mlp.gate",
                                                          "mlp.up", "mlp.down")),
                                        start_window=finetune_start_window)
        if finetune_start_window:
            logger.info(f"[ft-rotation] resuming mid-cycle: schedule starts at window "
                        f"{rot_schedule.window_at(0)} of {rot_schedule.n_windows}")
        # Continuation across a different card/plan: the stamped window index only lines
        # up when the window COUNT matches (mirrors H3).
        if ft_epoch_offset:
            try:
                from safetensors import safe_open as _so_w
                with _so_w(raw_path, framework="pt") as _fw:
                    _prev_nw = int((_fw.metadata() or {}).get("fizgig_ft_n_windows", 0))
            except Exception:
                _prev_nw = 0
            if _prev_nw and _prev_nw != rot_schedule.n_windows:
                logger.warning("[ft-rotation] this card's plan has %d windows; the "
                               "checkpoint was trained with %d — the rotation cycle "
                               "cannot line up exactly across the change, so expect one "
                               "cycle of mild imbalance while it settles.",
                               rot_schedule.n_windows, _prev_nw)

        def _ft_resident_blocks(spec):
            """Blocks holding trainable Linears under `spec` — block-index specs pass
            through; component entries translate (bare prefix = every block, a depth
            slice = its range). The streamer's resident set, per window."""
            from fizgig.krea2.rotation import is_component_spec
            if not is_component_spec(spec):
                return set(int(b) for b in spec)
            res = set()
            for _e in spec:
                if isinstance(_e, str):
                    res |= set(range(len(dit.blocks)))
                else:
                    _p, _lo, _hi = _e
                    res |= set(range(max(0, _lo), min(len(dit.blocks), _hi + 1)))
            return res

        _split_windows = any(isinstance(c, tuple) for c in rot_schedule.components)
        if rot_schedule.mode == "component" and ft_stream_frozen and not _split_windows:
            # Every block holds trainable Linears when component windows span full depth,
            # so nothing can be streamed out. (Depth-SPLIT windows are different — the
            # out-of-window blocks are fully frozen, which is the whole 16 GB tier.)
            logger.info("[ft-rotation] full-depth component windows train part of every "
                        "block — block streaming disabled.")
            ft_stream_frozen = False
        if ft_stream_frozen:
            from fizgig.krea2.rotation import RotationOffloader
            # Injected as the DiT's offloader: the forward already calls wait_for_block /
            # submit_move_blocks_forward whenever blocks_to_swap is truthy, so no model change.
            # Window 0 here; the epoch loop re-pins to the correct window on the first
            # iteration (and on resume, since want != rotator.active triggers a rotation).
            # Constructed claiming EVERYTHING is resident — which is the truth at this
            # moment, the whole fp8 base is on the card — and then narrowed. The
            # constructor only RECORDS the resident set; eviction is just-in-time during
            # the forward, so building it with the target set directly would leave the
            # full base resident until the first step. The window's bf16 then lands on
            # top of it and a 16 GB card OOMs inside the very first `rotate_to` (field,
            # 27 Aug: 14.30 GiB of a 14.83 GiB cap, before a single training step).
            # set_resident() does the up-front eviction with the one proven call.
            # H3 solves the same problem by rescoping its ring before activating.
            dit.offloader = RotationOffloader(dit.blocks, torch.device(device),
                                              range(len(dit.blocks)))
            dit.offloader.set_resident(_ft_resident_blocks(rot_schedule.active_at(0)))
            dit.blocks_to_swap = 1
            logger.info("[ft-rotation] streaming frozen blocks from CPU — only the trainable "
                        "window stays resident.")
        logger.info("[ft-rotation] FULL FINE-TUNE — %s", rot_schedule.describe())
        # Max epochs snaps UP to end on a cycle boundary — an off-cycle total leaves the
        # FINAL checkpoint (the one people keep) with some components trained one more
        # pass than others. Start-aware, so a resumed leg still lands on the original
        # total when that total was cycle-aligned. Before the too-short warning below,
        # which must judge the post-snap value. (H3's twin lives at its save-snap site.)
        from fizgig.krea2.rotation import snap_ft_epochs as _snap_ep
        _snapped_ep = _snap_ep(max_train_epochs, rot_schedule.cycle_epochs,
                               start_window=int(finetune_start_window or 0),
                               rotate_every=max(1, int(finetune_rotate_every or 1)))
        if _snapped_ep != max_train_epochs:
            logger.info("[ft-rotation] Max epochs %d would end mid-cycle (%d-epoch cycle) "
                        "— snapping to %d so the final checkpoint ends with every "
                        "component evenly trained.",
                        max_train_epochs, rot_schedule.cycle_epochs, _snapped_ep)
            max_train_epochs = _snapped_ep
        # Two pre-flight honesty checks (log-only — a power user gets the facts, not a gate):
        # (1) FT has never been run on AMD/ROCm, and the NF4 default leans on bitsandbytes
        # Linear4bit, whose ROCm wheel is the least-travelled part of that stack. Say so up
        # front rather than letting a default-config failure look like the user's fault.
        if getattr(torch.version, "hip", None):
            logger.warning("[ft-rotation] heads-up: fine-tuning is UNTESTED on AMD/ROCm — "
                           "every measured tier is NVIDIA. The NF4 default depends on "
                           "bitsandbytes 4-bit, the least-tested part of the ROCm stack. "
                           "It may work; if it does (or doesn't), a report on GitHub "
                           "genuinely helps.")
        # (2) The Krea 2 bf16 master (~24 GB) lives in system RAM with no disk spill (H3's
        # spills; Krea 2's does not). A short host leaves Windows paging, and paging
        # surfaces as a misleading 'CUDA error: out of memory' with the GPU nearly empty
        # (#94/#110's commit-charge trap). Warn while the user can still close things.
        try:
            import psutil as _ps
            _avail = _ps.virtual_memory().available / 1e9
            if _avail < 34.0:
                logger.warning("[ft-rotation] system RAM is tight for a Krea 2 fine-tune: "
                               "%.0f GB available, and the ~24 GB bf16 master plus staging "
                               "wants ~34 GB free. Expect paging (slow steps), and know "
                               "that running out surfaces as 'CUDA error: out of memory' "
                               "with the GPU nearly empty. Close other apps, or use a "
                               "machine with 48 GB+ of RAM for comfort.", _avail)
        except Exception:
            pass
        if rot_schedule.cycle_epochs > max_train_epochs:
            logger.warning("[ft-rotation] a full cycle needs %d epochs but max_train_epochs=%d — "
                           "blocks after window %d will NEVER train this run.",
                           rot_schedule.cycle_epochs, max_train_epochs,
                           max_train_epochs // finetune_rotate_every)
        # Checkpoints (and the previews that ride them) land at rotation-cycle boundaries
        # only — mirrors H3: every window must see the identical data mix for equal passes
        # before the mix changes, or checkpoints compare unlike-for-unlike. Snapped UP,
        # never down — the user asked for at least that much training between saves.
        _cyc = rot_schedule.cycle_epochs
        if save_every_n_epochs and save_every_n_epochs % _cyc:
            _snapped_save = ((save_every_n_epochs + _cyc - 1) // _cyc) * _cyc
            logger.info("[ft-rotation] checkpoint saves land at rotation-cycle boundaries — "
                        "save-every %d snaps to %d (%d-epoch cycle).",
                        save_every_n_epochs, _snapped_save, _cyc)
            save_every_n_epochs = _snapped_save
        if do_previews:
            logger.info("[ft-rotation] previews follow CHECKPOINT SAVES (every %d epoch(s), "
                        "plus the final one), overriding Sample-every-N — each sample is the "
                        "rehearsal of a checkpoint you can deploy, rendered on the training "
                        "DiT via a deactivate/reactivate bracket with the Turbo LoRA applied "
                        "fresh each time.",
                        save_every_n_epochs if save_every_n_epochs else max_train_epochs)
        elif sample_prompts and not _ft_preview_gap_warned:
            # H3's 7377f2c twin: a prompts file is the clearest statement the user WANTS
            # previews, so a silent off is the same class as the announce-then-never-render
            # lie fixed there — say which ingredient is missing instead (the cadence flag
            # is the one nobody guesses; a run with prompts but no cadence renders nothing).
            # The FT turbo-lora gate above prints its own line; don't double up on it.
            _missing = [m for m, ok in (
                ("a sample cadence — set Sample every N epochs "
                 "(--sample_every_n_epochs) or sample-at-first",
                 bool(sample_every_n_epochs or sample_at_first)),
                ("the Turbo LoRA", bool(turbo_lora_path)),
                ("the VAE path", bool(vae_path)),
                ("the text encoder path", bool(te_path))) if not ok]
            if _missing:
                logger.warning("[ft-rotation] previews are OFF for this run: a prompts "
                               "file is set, but missing %s. With that in place previews "
                               "ride the checkpoint saves.", "; ".join(_missing))
        if adaptive_lr:
            # The watcher reads epoch-to-epoch loss movement as signal. Rotation changes which
            # weights are trainable at the boundary, so every rotation looks like a step change
            # and would trigger spurious reductions/rollbacks. Off for now.
            logger.info("[ft-rotation] adaptive LR disabled — rotation boundaries look like "
                        "instability to the plateau watcher.")
            adaptive_lr = False
    else:
        network.requires_grad_(True)

    # Label recorded in the saved checkpoint's metadata (ss_optimizer).
    _optlabel = {"v": optimizer_type}

    def _make_optimizer(params_, quiet: bool = False):
        """Rotation-mode optimizer. Adafactor first: its factored state is ~10x smaller than
        Adam's, which is what keeps a full fine-tune inside 32 GB. (The LoRA path uses the
        shared catalog instead — see create_optimizer below — so the user's Optimizer Type
        choice applies there; here the memory constraint decides.)"""
        try:
            from transformers.optimization import Adafactor
            opt = Adafactor(params_, lr=learning_rate, scale_parameter=False,
                            relative_step=False, warmup_init=False)
            if not quiet:
                logger.info("optimizer: Adafactor (rotation)")
            _optlabel["v"] = "adafactor (rotation)"
            return opt
        except Exception as e:
            if not quiet:
                logger.warning("Adafactor unavailable (%s) — falling back to AdamW8bit", e)
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params_, lr=learning_rate)
            if not quiet:
                logger.info("optimizer: AdamW8bit")
            _optlabel["v"] = "adamw8bit (rotation)"
            return opt
        except Exception:
            if not quiet:
                logger.info("optimizer: AdamW (bitsandbytes unavailable)")
            _optlabel["v"] = "adamw (rotation)"
            return torch.optim.AdamW(params_, lr=learning_rate)

    # ---- optimizer-in-backward (fused) ----
    # Normally every active parameter's gradient exists simultaneously at the peak of backward.
    # With this on, each parameter's optimizer steps the moment its grad is ready and the grad
    # is dropped, so only one parameter's gradient is live at a time.
    fused_backward = bool(finetune_fused_backward and ft_rotation)
    _fused = {"opts": {}, "handles": []}
    if finetune_fused_backward and not ft_rotation:
        logger.info("[fused-backward] only applies to rotation fine-tuning — ignored.")
    if fused_backward:
        if int(gradient_accumulation_steps or 1) > 1:
            logger.info("[fused-backward] incompatible with gradient accumulation (grads are "
                        "consumed and freed per parameter) — forcing accumulation to 1.")
        if max_grad_norm > 0:
            logger.info("[fused-backward] global grad-norm clipping needs all grads at once — "
                        "clipping is disabled in this mode.")

    def _attach_fused(params_):
        """One single-parameter optimizer per tensor, stepped from its grad hook."""
        for h in _fused["handles"]:
            h.remove()
        _fused["handles"].clear()
        _fused["opts"].clear()
        for p in params_:
            _fused["opts"][p] = _make_optimizer([p], quiet=True)

        def _hook(param):
            opt = _fused["opts"].get(param)
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)

        for p in params_:
            _fused["handles"].append(p.register_post_accumulate_grad_hook(_hook))
        logger.info("[fused-backward] %d per-parameter optimizers attached", len(params_))

    if ft_rotation:
        rotator.rotate_to(rot_schedule.active_at(0))
        params = rotator.trainable_params()
        if fused_backward:
            _attach_fused(params)
            optimizer = None        # stepping happens in the backward hooks
        else:
            optimizer = _make_optimizer(params)
        optimizer_label = _optlabel["v"]
    else:
        # LoRA training: the shared catalog, so the Optimizer Type / Args the user picked
        # applies (and its family-appropriate LR warnings fire).
        params = list(network.get_trainable_params())
        from fizgig.training.optimizers import create_optimizer
        optimizer, optimizer_label = create_optimizer(
            optimizer_type, params, learning_rate, optimizer_args)

    collator = _Krea2Collator(shared_epoch, group)
    # Bucket-grouped ordering (OFF by default — measured, and it buys nothing today).
    #
    # OneTrainer groups batches by resolution (AspectBatchSorting); Fizgig shuffles freely,
    # which changes latent shape on ~43% of steps for a mixed-aspect dataset. That sounded
    # like it should matter — shape churn makes cuDNN re-plan, cuBLAS re-pick algorithms and
    # the allocator fragment. Measured on a 36-image set, 3 epochs, NF4:
    #
    #     shuffled ................ 0.7042 s/it
    #     bucket-grouped .......... 0.7042 s/it   (no change at all)
    #     bucket-grouped + cuDNN .. 1.29   s/it   (vs 1.57 unbucketed — helps, still 1.8x worse)
    #
    # So the default backend does not care, and it is not enough to rescue cuDNN either. Left
    # in because it should matter for torch.compile, which recompiles per shape — but not
    # enabled without evidence, since grouping correlates consecutive gradients (all one aspect
    # in a row), and that is a real if modest quality risk to take for nothing.
    _sampler = None
    if os.environ.get("FIZGIG_BUCKET_ORDER", "0") != "0":
        try:
            _sampler = _BucketOrderSampler(group, seed=seed)
            logger.info("[dataloader] bucket-grouped order: %d shapes, ~%d shape changes/epoch "
                        "(random shuffle would be ~%d)", _sampler.n_shapes,
                        _sampler.n_shapes, _sampler.est_random_changes)
        except Exception as e:
            logger.warning("[dataloader] bucket ordering unavailable (%s) — using plain shuffle", e)
    loader = DataLoader(group, batch_size=1, shuffle=(_sampler is None), sampler=_sampler,
                        collate_fn=collator, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        # The Learning Rate box is IGNORED while adaptive is on: start at the GEOMETRIC
        # MIDPOINT of Min/Max and let the watcher own the LR (matches Klein). Two knobs,
        # not three. A resumed run's optimizer restore below overwrites this with the
        # watcher's mid-flight LR, which is correct.
        _mid = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        if abs(learning_rate - _mid) > 1e-12:
            logger.info(f"[adaptive_lr] starting LR set to {_mid:.3e} — the geometric midpoint "
                        f"of Min/Max (the Learning Rate box is ignored while adaptive is on)")
        learning_rate = _mid
        for g in optimizer.param_groups:
            g["lr"] = _mid
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} "
                    f"min_lr={adaptive_lr_min:.3e} max_lr={adaptive_lr_max:.3e}")

    global_step = 0
    start_epoch = 0
    # Resume: restore LoRA + optimizer + RNG + (start_epoch, global_step) from a saved state dir.
    # `if resume_state_dir` — NOT `and os.path.isdir(...)`: a requested resume whose path is bad
    # (the .safetensors picked instead of its folder, a moved/typo'd dir) used to skip this block
    # silently and train from scratch. If a resume was asked for, it happens or the run refuses.
    if resume_state_dir:
        start_epoch, global_step, _resume_meta = _load_training_state(resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
            logger.info(f"[resume] adaptive_lr state restored: best_loss={adaptive.best_loss} "
                        f"streaks g/b/s={adaptive.good_streak}/{adaptive.bad_streak}/{adaptive.stability_streak} "
                        f"stability_triggered={adaptive.stability_triggered}")
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs} "
                    f"(global_step {global_step})")
        # Nothing left to train: the epoch loop below is empty and we fall through to writing the
        # final LoRA. That fall-through is deliberate — pausing ON the last epoch exits before the
        # final LoRA is written, so Resume is what completes it — hence a loud log, not an error.
        if start_epoch >= max_train_epochs:
            logger.warning(
                f"[resume] state is at epoch {start_epoch} of {max_train_epochs} — nothing left to "
                f"train. Writing the final LoRA from the restored state. To train further, raise "
                f"Max Train Epochs above {start_epoch} and resume again.")
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    # ---- Step-level LR scheduler (cosine / linear / warmup / ...) ----
    # Mutually exclusive with adaptive LR by design: both write optimizer.param_groups[*]["lr"],
    # so a live scheduler would stomp the watcher's epoch decisions every step. Adaptive wins
    # (same rule as Klein, whose GUI also forces "constant" when adaptive is on).
    accum_requested = max(1, int(gradient_accumulation_steps or 1))
    # Fused backward consumes and frees each grad as it lands, so there is nothing
    # left to accumulate across micro-batches.
    accum = 1 if fused_backward else accum_requested
    if accum > 1:
        logger.info(f"[grad_accum] {accum} micro-batches per optimizer step "
                    f"(effective batch {accum}); ~{max(1, steps_per_epoch // accum)} updates/epoch")

    _sched_total_steps = math.ceil(steps_per_epoch / accum) * max_train_epochs

    def _sched_position(gstep: int) -> int:
        """global_step counts MICRO-batches; a schedule's position is optimizer steps.

        The loop flushes a PARTIAL accumulation group at every epoch boundary (the scheduler
        steps there too), so updates/epoch = ceil(steps_per_epoch/accum) — a flat
        `global_step // accum` ignores those flushes and winds the schedule short, leaving the
        LR high for the whole remainder. Resume always lands on an epoch boundary; the
        leftover term covers a hand-rolled mid-epoch state anyway."""
        if gstep <= 0:
            return 0
        _epochs_done = gstep // steps_per_epoch
        _leftover = gstep % steps_per_epoch
        return _epochs_done * math.ceil(steps_per_epoch / accum) + _leftover // accum

    def _rebuild_scheduler(opt, position: int):
        """Build the configured schedule against `opt`, wound forward to `position`
        optimizer-steps. Used at startup, on resume, and after every rotation (which
        replaces the optimizer, so the old scheduler's parameter refs are dead)."""
        if adaptive or not lr_scheduler or lr_scheduler == "constant":
            return None
        from diffusers.optimization import get_scheduler
        kwargs = {}
        if lr_scheduler == "cosine_with_restarts":
            kwargs["num_cycles"] = int(lr_scheduler_num_cycles)
        elif lr_scheduler == "polynomial":
            kwargs["power"] = float(lr_scheduler_power)
        s = get_scheduler(lr_scheduler, opt,
                          num_warmup_steps=int(lr_warmup_steps or 0),
                          num_training_steps=_sched_total_steps, **kwargs)
        # These schedules are pure functions of the step count, so re-deriving the position
        # is exact and needs no persisted state. Setting last_epoch then stepping once lands
        # the LR exactly where `position` calls to step() would have.
        if position > 0:
            s.last_epoch = position - 1
            s.step()
        return s

    scheduler = None
    if adaptive:
        if lr_scheduler and lr_scheduler != "constant":
            logger.info(f"[lr_scheduler] '{lr_scheduler}' ignored — adaptive LR is enabled and owns the LR.")
    elif lr_scheduler and lr_scheduler != "constant":
        scheduler = _rebuild_scheduler(optimizer, _sched_position(global_step))
        logger.info(f"[lr_scheduler] {lr_scheduler} — warmup {int(lr_warmup_steps or 0)} / "
                    f"{_sched_total_steps} total steps, start lr={optimizer.param_groups[0]['lr']:.3e}"
                    + (f" (resumed at step {global_step})" if global_step > 0 else ""))
    elif lr_warmup_steps:
        logger.info("[lr_scheduler] warmup steps ignored — LR scheduler is 'constant'.")

    pause_flag = os.path.join(output_dir, ".pause_requested")
    # Progress + loss display exactly as Klein: one continuous tqdm bar over all steps with
    # a smoothed avr_loss in the postfix (the raw per-step loss is very noisy — batch size 1
    # plus a random flow-matching timestep each step — so the moving average is the signal).
    loss_recorder = LossRecorder()
    # Per-image loss watcher (experiment). Three tiers, all sharing one class:
    #   env FIZGIG_PERIMAGE_LOSS_LOG=1  -> passive JSONL log only (offline study)
    #   log_per_image_loss (GUI toggle) -> JSONL + per-epoch stuck-image detection report
    #   per_image_lr (GUI toggle)       -> detection + per-image loss multiplier (throttle stuck,
    #                                      boost healthy learned; safe per-image LR at batch size 1)
    from fizgig.training.loss_logger import PerImageLossWatch, is_enabled as _loss_log_env
    # Fresh (non-resume) run: clear the previous run's loss-log artifacts so the GUI's Problem
    # Images window never shows stale verdicts (problem_images.json only gets rewritten after the
    # new run's warmup — or never, if the toggles are off this run). The pending caption queue is
    # stale too (the .txt fixes are already applied by the startup text re-cache). The research
    # JSONL is rotated, not deleted — appending would mix runs and corrupt offline analysis.
    if not (resume_state_dir and os.path.isdir(resume_state_dir)):
        _ll = os.path.join(output_dir, "loss_log")
        for _f in ("problem_images.json", "problem_images.json.tmp",
                   "caption_updates_applied.json", "caption_updates_applied.json.tmp",
                   "caption_updates.json", "caption_updates.json.processing"):
            try:
                os.remove(os.path.join(_ll, _f))
            except OSError:
                pass
        _jsonl = os.path.join(_ll, "per_image_loss.jsonl")
        if os.path.exists(_jsonl):
            import time as _time
            try:
                os.replace(_jsonl, _jsonl + "." + _time.strftime("%Y%m%d%H%M%S") + ".bak")
            except OSError:
                pass
    # The watch + auto-recaption need the source images + caption extension — pull them from the
    # dataset TOML (recursive: the keys live under [general] / [[datasets]] depending on config).
    # Also used to load/store <image_dir>/fizgig_excluded.json (exclusions travel with the dataset).
    recaptioned = {}   # key -> AI recaption attempts used (max 2; 2nd is the detailed pass)
    ar_image_dir, ar_caption_ext = None, ".txt"
    if auto_recaption and ft_rotation:
        # The between-epoch recaption loads the VLM by moving the WHOLE DiT to CPU and
        # restoring it through a blocks_to_swap-aware path that knows nothing about the FT
        # rotation streamer — on the streamed tier the restore would hoist every streamed
        # block back onto the card behind the offloader's bookkeeping. The GUI hides the
        # checkbox under a fine-tune; this is the belt for CLI runs and stale configs.
        # The OTHER watch features stay: their multipliers ride the same loss-scaling the
        # FT regularisation path uses (fused backward consumes the scaled grads), and
        # detection judges each image against the cohort at the same epoch, so rotation's
        # boundary shifts cancel — unlike the global adaptive watcher, which reads
        # absolute movement and is disabled under FT for exactly that reason.
        logger.info("[ft-rotation] auto-recaption is not available under a base-model "
                    "fine-tune yet — detection, per-image LR and look-outlier warmup all "
                    "still run. Fix stuck captions from the Problem Images window instead; "
                    "manual edits queued there still apply at epoch boundaries in LoRA runs.")
        auto_recaption = False
    watch_enabled = (log_per_image_loss or per_image_lr or auto_recaption
                     or warmup_look_outliers or _loss_log_env())
    if watch_enabled:
        def _find_toml_key(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    r = _find_toml_key(v, key)
                    if r is not None:
                        return r
            elif isinstance(d, list):
                for v in d:
                    r = _find_toml_key(v, key)
                    if r is not None:
                        return r
            return None
        ar_image_dir = _find_toml_key(user_config, "image_directory")
        ar_caption_ext = _find_toml_key(user_config, "caption_extension") or ".txt"
        if not (ar_image_dir and os.path.isdir(ar_image_dir)):
            ar_image_dir = None
    if auto_recaption:
        if ar_image_dir:
            logger.info(f"[auto-recaption] ON — stuck images re-captioned by Qwen3-VL from "
                        f"{ar_image_dir}" + (f" (trigger: '{trigger_word}')" if trigger_word else ""))
        else:
            logger.warning("[auto-recaption] image_directory not found in the dataset config "
                           "— auto-recaption disabled")
            auto_recaption = False
    loss_watch = None
    if watch_enabled:
        # write_jsonl is ALWAYS on when the watch runs: the JSONL is the watch's persistence
        # layer, not a detection feature. Binding it to the detect toggle meant per-image LR
        # or auto-recaption without "Detect problem images" wrote no JSONL — and every resume
        # of such a run silently discarded the entire watch history while `recaptioned` WAS
        # restored from the ledger, pinning spent images on the stuck ladder with no way off.
        loss_watch = PerImageLossWatch(output_dir, apply_lr=per_image_lr,
                                       write_jsonl=True,
                                       dataset_dir=ar_image_dir, caption_ext=ar_caption_ext)
        # Reconcile persisted exclusions against the actual training set (prune entries for
        # images that left the dataset; refuse a file that would exclude everything).
        _dataset_keys = {str(it.item_key)
                         for ds in group.datasets
                         if getattr(ds, "batch_manager", None) is not None
                         for bucket in ds.batch_manager.buckets.values()
                         for it in bucket}
        loss_watch.preflight(_dataset_keys)
        logger.info(f"[loss-watch] per-image loss watch ON (per_image_lr={per_image_lr})")
        if warmup_look_outliers:
            # LR warm-up for Look Consistency Filter outliers (tight angles, unusual views):
            # they keep their unique information but ease in at x0.4 -> x1.0 over the first
            # epochs instead of fighting the forming identity core at full strength. Scores are
            # saved by the Image Prep tab's Look Filter into the dataset folder.
            _look_path = os.path.join(ar_image_dir or "", "fizgig_look_scores.json")
            try:
                with open(_look_path, encoding="utf-8") as _f:
                    _look = json.load(_f)
                _cut = _look.get("cutoff")
                _scores = _look.get("scores") or {}
                if _cut is None:
                    logger.warning("[look-warmup] no cutoff in fizgig_look_scores.json (too few "
                                   "scored faces) — warm-up disabled this run")
                else:
                    _outliers = {k for k, v in _scores.items()
                                 if isinstance(v, (int, float)) and v < float(_cut)}
                    # Outliers no longer in the dataset were most likely marked + moved to
                    # excluded_by_look/ in the Look Filter — that's the tool working, not an
                    # error. Warm up only what is actually being trained.
                    _gone = sorted(_outliers - _dataset_keys)
                    _keys = _outliers & _dataset_keys
                    if _gone:
                        logger.info(f"[look-warmup] {len(_gone)} scored outlier(s) not in the "
                                    f"dataset (moved/excluded via the Look Filter) — skipped: "
                                    + ", ".join(_gone[:8]) + ("…" if len(_gone) > 8 else ""))
                    if _keys:
                        loss_watch.set_warmup_keys(_keys)
                        logger.info(f"[look-warmup] {len(_keys)} look-outlier image(s) on LR "
                                    f"warm-up ×0.4→×1.0 over the first epochs (released early "
                                    f"on improvement): " + ", ".join(sorted(_keys)[:8])
                                    + ("…" if len(_keys) > 8 else ""))
                    else:
                        logger.info("[look-warmup] no look-outliers present in the dataset — "
                                    "nothing to warm up")
            except FileNotFoundError:
                logger.warning("[look-warmup] fizgig_look_scores.json not found in the dataset "
                               "folder — run the Look Consistency Filter (Image Prep tab, scan "
                               "with 3 baselines) first; warm-up disabled this run")
            except Exception as _e:
                logger.warning(f"[look-warmup] could not load look scores ({_e}) — warm-up "
                               f"disabled this run")
        if resume_state_dir and os.path.isdir(resume_state_dir) and start_epoch > 0:
            # Resumed run: rebuild the watch's history by replaying its own JSONL (it appends
            # across pause/resume). The applied-captions ledger supplies the reset/incorrigible
            # timeline: recaptioned images re-enter with post-fix history only, and images whose
            # 2 AI attempts are spent go back on the exclusion track instead of getting a free
            # third life. Also restores `recaptioned` so the max-2 attempt cap survives resume.
            _resets = {}
            try:
                with open(os.path.join(output_dir, "loss_log", "caption_updates_applied.json"),
                          encoding="utf-8") as _f:
                    for _k, _info in json.load(_f).items():
                        # Per-fix history list (older files: a single dict = last fix only).
                        _entries = _info if isinstance(_info, list) else [_info]
                        for _e in _entries:
                            _att = int(_e.get("attempt", 0) or 0)
                            _auto = bool(_e.get("auto"))
                            if _auto:
                                recaptioned[_k] = max(recaptioned.get(_k, 0), _att)
                            _resets.setdefault(_k, []).append(
                                (int(_e.get("epoch", 0) or 0), _att, _auto))
            except Exception:
                pass
            loss_watch.resume_from_jsonl(up_to_epoch=start_epoch, resets=_resets)
    # The preview bracket's scheduler hand-off: dropping the scheduler unpins the old
    # window's optimizer, and the next rotation rebuilds it here from the stashed position.
    _ft_sched_pos = {"pos": None}

    def _ft_bracket_preview(epoch1):
        """Fine-tune preview (the field-proven H3 bracket, krea2-shaped): deactivate the
        whole window so the model is a consistent all-fp8 checkpoint (the master holds
        every trained weight), apply the Turbo LoRA FRESH against it, render on the
        resident training DiT, then put every wrapped forward back exactly. The model is
        left DEACTIVATED on purpose: the next epoch's rotation check sees active=[] !=
        wanted and reactivates + rebuilds the optimizer through the one normal path — and
        on a failed render, only after the exception's tensors are gone (re-activating
        inside the exception's lifetime OOM'd in H3's field runs). Returns the last
        rendered prompt (for the status line) or None."""
        nonlocal optimizer, scheduler, params
        _act = list(rotator.active)
        if _act:
            rotator.deactivate(_act)
            # Drop every reference to the window's now-orphaned bf16 Parameters — the
            # fused per-parameter optimizers are KEYED on them, a plain optimizer's state
            # pins them just as hard, and the scheduler pins the optimizer.
            params = None
            if fused_backward:
                for _h in _fused["handles"]:
                    _h.remove()
                _fused["handles"].clear()
                _fused["opts"].clear()
            else:
                if scheduler is not None:
                    _ft_sched_pos["pos"] = scheduler.last_epoch
                optimizer = None
                scheduler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _t_net = _t_diffb = None
        try:
            # Override read + TE encode AFTER the deactivate, when the window's VRAM is
            # back — the ~8 GB Qwen3-VL is the render's one real memory risk.
            ov = _read_sample_override(output_dir)
            if ov:
                logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                            f"seed={ov['seed']} {ov['width']}x{ov['height']}"
                            f"{' +ref' if ov.get('ref_image') else ''}")
                _enc = encode_sample_prompts(te_path, [ov["prompt"]],
                                             ref_image=ov.get("ref_image") or None, device=device)
                _w, _h, _seed, _prompts = ov["width"], ov["height"], ov["seed"], [ov["prompt"]]
            else:
                _enc, _w, _h, _seed = encoded_prompts, sample_width, sample_height, sample_seed
                _prompts = sample_prompts
            if _seed == 0:
                _seed = random.randint(1, 2**31 - 1)
                logger.info(f"[sample] seed 0 -> random {_seed}")
            _t_net, _t_diffb = _apply_turbo_lora(dit, turbo_lora_path, device=device, dtype=dtype)
            _, _lp = sample_previews_on_dit(dit, _t_net, _t_diffb, sample_ae, _enc,
                                            sample_dir, epoch1, output_name=output_name,
                                            steps=sample_steps, cfg_scale=sample_cfg_scale,
                                            neg=encoded_negative, width=_w, height=_h,
                                            seed=_seed, blocks_to_swap=blocks_to_swap,
                                            device=device, prompts=_prompts)
            return _lp
        finally:
            if _t_net is not None:
                # Exact un-apply. The pre-Turbo forward here is an INSTANCE chain (the fp8
                # patch + the inert trainable wrap), so popping down to the class forward —
                # H3's move — would destroy it. Each LoRAInfModule kept the target module in
                # org_module_ref, so the captured bound forward is restored verbatim.
                for _l in _t_net.unet_loras:
                    _m = (_l.org_module_ref[0]
                          if getattr(_l, "org_module_ref", None) else None)
                    if _m is not None and getattr(_l, "org_forward", None) is not None:
                        _m.forward = _l.org_forward
                _t_net.to("cpu")
            _t_net = _t_diffb = None
            dit.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Sample at Start: an epoch-0 preview (base model + zero-init LoRA) so the run's
    # starting point is on record. Fresh runs only — a resume already has samples.
    if sample_at_first and do_previews and start_epoch == 0 and rotator is not None:
        # Fine-tune: the bracket renders the untouched base (nothing trained yet) on the
        # training DiT; the epoch-0 rotation check re-activates the first window after.
        logger.info("rendering epoch-0 preview (Sample at Start, fine-tune bracket)...")
        try:
            _last_p = _ft_bracket_preview(ft_epoch_offset)
            if _last_p:
                _last_sample_prompt = _last_p
        except Exception as _e0:
            logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                           f"continues; per-epoch previews will still be attempted.")
    elif sample_at_first and do_previews and start_epoch == 0 and turbo_net is not None:
        # Turbo-LoRA mode: render on the resident training DiT (live network — no save/reload,
        # no parking). sample_previews_on_dit reverts everything in its finally.
        logger.info("rendering epoch-0 preview (Sample at Start, on training DiT)...")
        try:
            _seed0 = sample_seed if sample_seed != 0 else random.randint(1, 2**31 - 1)
            _, _last_p = sample_previews_on_dit(dit, turbo_net, turbo_diffb, sample_ae, encoded_prompts,
                                   sample_dir, 0, output_name=output_name, steps=sample_steps,
                                   cfg_scale=sample_cfg_scale, neg=encoded_negative,
                                   width=sample_width, height=sample_height, seed=_seed0,
                                   blocks_to_swap=blocks_to_swap, device=device, prompts=sample_prompts)
            if _last_p:
                _last_sample_prompt = _last_p
        except Exception as _e0:
            logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                           f"continues; per-epoch previews will still be attempted.")
    elif sample_at_first and do_previews and start_epoch == 0:
        from safetensors.torch import load_file as _lf0
        _tmp0 = os.path.join(output_dir, "_sample_lora.safetensors")
        _save_lora(network, _tmp0, network_dim, network_alpha, dtype)
        logger.info("rendering epoch-0 preview (Sample at Start)...")
        dit.to("cpu")
        if getattr(dit, "_nf4_quantized", False):
            from fizgig.modules.nf4 import move_nf4_to_device
            move_nf4_to_device(dit, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            _seed0 = sample_seed if sample_seed != 0 else random.randint(1, 2**31 - 1)
            _, _last_p = sample_previews(turbo_path, sample_ae, encoded_prompts, _lf0(_tmp0), sample_dir, 0,
                            output_name=output_name, steps=sample_steps,
                            cfg_scale=sample_cfg_scale, neg=encoded_negative,
                            width=sample_width, height=sample_height, seed=_seed0,
                            context_lora_path=context_lora_path,
                            context_lora_strength=context_lora_strength,
                            blocks_to_swap=preview_blocks_to_swap, int8=preview_int8, device=device,
                            prompts=sample_prompts)
            if _last_p:
                _last_sample_prompt = _last_p
        except Exception as _e0:
            logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                           f"continues; per-epoch previews will still be attempted.")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            # Placement first, THEN the NF4 packed weights — the two are complementary, not
            # alternatives: nn.Module.to() moves the ordinary params/buffers, move_nf4_to_device
            # moves `_nf4_packed`/`_nf4_state`, which are plain attributes .to() cannot see.
            # Making the NF4 case an `elif` stranded every ordinary parameter on CPU.
            if blocks_to_swap > 0:
                dit.move_to_device_except_swap_blocks(torch.device(device))
                dit.switch_block_swap_for_training()
            else:
                dit.to(device)
            if getattr(dit, "_nf4_quantized", False):
                from fizgig.modules.nf4 import move_nf4_to_device
                move_nf4_to_device(dit, device)
            dit.train()

    # Return the load-time transients (quantise staging, resume's optimizer-state copy) to the
    # driver before stepping. Fresh runs with Sample at Start got this for free from the
    # preview's empty_cache; resumed runs skip that preview and sat ~4 GB high until the first
    # epoch-boundary preview cleared it (issue #24). Unconditional so every path starts clean.
    gc.collect()
    torch.cuda.empty_cache()

    # VRAM probe: capture what the load phase alone cost, then zero the high-water mark so the
    # training figure is training's own. Answers "is the peak the model coming in, or the steps?"
    # AFTER the cleanup above on purpose: the probe should read the settled figure, not the
    # transients that were about to be freed anyway.
    _probe_load_gb = 0.0
    if int(os.environ.get("FIZGIG_VRAM_PROBE", "0") or 0) and torch.cuda.is_available():
        torch.cuda.synchronize()
        _probe_load_gb = torch.cuda.max_memory_reserved() / 1024**3
        torch.cuda.reset_peak_memory_stats()

    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="steps", smoothing=0)
    pending_accum = 0  # micro-batches backward'd since the last optimizer step
    # Warm-up reassurance: the first two epochs start slowly (first-sight kernel planning,
    # cuBLAS algorithm picks, allocator + cache warm-up; the cuDNN switch at the epoch-1
    # boundary re-plans every shape in epoch 2). Users watching a crawling bar assume a
    # hang, so repeat a gentle note every ~30 s while it lasts.
    _warmup_note_last = 0.0
    # Per-epoch step rate. The progress bar runs with smoothing=0, so its it/s (and the ETA
    # derived from it) are a CUMULATIVE average over the whole run — every second not spent
    # iterating (checkpoint saves, the Qwen3-VL load at a recaption boundary, rotation window
    # switches) is amortised in permanently and can only push it up. That makes "is it actually
    # slowing down?" unanswerable from the bar. This measures each epoch on its own clock.
    _epoch_rate_prev = None
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        _epoch_t0, _epoch_step0 = time.time(), global_step
        if _sampler is not None:
            _sampler.set_epoch(epoch)      # reshuffle within/across buckets each epoch
        if rotator is not None:
            want = rot_schedule.active_at(epoch)
            if want != rotator.active:
                # New window: swap the blocks, then rebuild the optimizer. The old optimizer's
                # state refers to tensors that no longer require grad, and Adam moments for the
                # outgoing window are meaningless to the incoming one.
                #
                # Release the optimizer's grip BEFORE the swap (H3 field lesson): the old
                # optimizer/hooks/scheduler are keyed on the outgoing window's Parameters,
                # so without this every rotation transiently held BOTH windows at once —
                # the measured 27.7 GB component peak IS that boundary, ~7 GB over steady
                # state. Stash the schedule position first; the rebuild below re-attaches.
                _sched_pos_now = (scheduler.last_epoch
                                  if (scheduler is not None and not fused_backward)
                                  else _ft_sched_pos["pos"])
                params = None
                if fused_backward:
                    for _h in _fused["handles"]:
                        _h.remove()
                    _fused["handles"].clear()
                    _fused["opts"].clear()
                else:
                    optimizer = None
                    scheduler = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if ft_stream_frozen:
                    # Pin the incoming window BEFORE the weight swap: activate() reads from the
                    # master onto the GPU, and the outgoing window must rejoin the stream pool.
                    # Component specs translate to their resident block set.
                    dit.offloader.set_resident(_ft_resident_blocks(want))
                rotator.rotate_to(want)
                if torch.cuda.is_available():
                    # Per-window peak measurement starts here (logged at epoch end).
                    torch.cuda.reset_peak_memory_stats()
                _new_params = rotator.trainable_params()
                if fused_backward:
                    # Hooks and per-parameter optimizers belong to the OLD window's tensors —
                    # rebuild them or the incoming blocks would never step.
                    _attach_fused(_new_params)
                else:
                    optimizer = _make_optimizer(_new_params)
                params = _new_params
                if not fused_backward and _sched_pos_now is not None:
                    # Re-attach the schedule to the new optimizer at the stashed position
                    # (covers both the rotation drop above and the preview bracket's).
                    scheduler = _rebuild_scheduler(optimizer, _sched_pos_now)
                    _ft_sched_pos["pos"] = None
                logger.info("[ft-rotation] epoch %d: training blocks %s", epoch + 1, want)
        for i, batch in enumerate(loader):
            if epoch < 2:
                _now = time.time()
                if _now - _warmup_note_last > 30.0:
                    _warmup_note_last = _now
                    logger.info("[warm-up] Warm-up phase — the first two epochs start slowly "
                                "while the GPU plans kernels and fills its caches. Nothing is "
                                "stuck; full speed arrives from epoch 3.")
            # Excluded images (two failed AI recaptions, still stuck) are skipped ENTIRELY: no
            # forward, no gradient, and no loss recorded — avr_loss stops carrying their permanent
            # error term. Step accounting (bar + global_step) stays consistent for resume math.
            if loss_watch is not None and loss_watch.is_excluded(batch.get("item_keys")):
                loss_recorder.drop(step=i)  # the slot leaves avr_loss — no stale/zero padding
                global_step += 1
                progress_bar.update(1)
                continue
            if slider_pairs:
                # Image-pair slider step: the training image is the POSITIVE pole, its control
                # the NEGATIVE. The adapter trains at +1 toward the positive and at -1 toward
                # the negative — the SAME weights learn a signed direction, which is what makes
                # strength a dial at inference. The pair is never packed into one sequence
                # (that's edit-style training); each pole is a plain forward, diff-weighted so
                # the loss concentrates where the pair actually differs.
                _neg = batch.get("latents_control_0")
                if _neg is None:
                    raise RuntimeError(
                        "[slider] this item has no pair image. Slider training needs a "
                        "control_directory with a negative-pole image for every training "
                        "image (matched by filename stem).")
                network.set_multiplier(1.0)
                _l_pos, t_used = compute_loss(dit, batch["latents"], batch["hidden_states"],
                                              batch["attention_mask"], device=device,
                                              shift=shift, dtype=dtype,
                                              min_timestep=min_timestep, max_timestep=max_timestep,
                                              diff_ref_latent=_neg, diff_weight=slider_diff_weight)
                network.set_multiplier(-1.0)
                _l_neg, _ = compute_loss(dit, _neg, batch["hidden_states"],
                                         batch["attention_mask"], device=device,
                                         shift=shift, dtype=dtype,
                                         min_timestep=min_timestep, max_timestep=max_timestep,
                                         diff_ref_latent=batch["latents"],
                                         diff_weight=slider_diff_weight)
                network.set_multiplier(1.0)
                loss = 0.5 * (_l_pos + _l_neg)
            else:
                loss, t_used = compute_loss(dit, batch["latents"], batch["hidden_states"], batch["attention_mask"],
                                            device=device,
                                            shift=shift, dtype=dtype,
                                            control_latent=batch.get("latents_control_0"),
                                            min_timestep=min_timestep, max_timestep=max_timestep,
                                            motion_weight=motion_weighted_loss)
            # Per-image LR: scale THIS step's gradient by the image's multiplier (throttle stuck
            # images, boost healthy learned ones). Raw loss is still what gets recorded/averaged below,
            # so avr_loss and the global adaptive-LR watcher see unscaled numbers.
            if reg_keys and _batch_is_reg(batch.get("item_keys"), reg_keys):
                step_mult = reg_mult   # regularisation images: fixed nudge, never the watch's
            else:
                step_mult = loss_watch.multiplier(batch.get("item_keys")) if loss_watch is not None else 1.0
            # Divide by the accumulation count so N micro-batches AVERAGE into one update rather
            # than summing (which would scale the effective LR by N).
            _scaled = loss * step_mult if step_mult != 1.0 else loss
            (_scaled / accum if accum > 1 else _scaled).backward()
            pending_accum += 1
            if fused_backward:
                # The per-parameter hooks already stepped and freed each grad during backward.
                pending_accum = 0
            elif pending_accum >= accum:
                # Gradient clipping to match the musubi reference (max_grad_norm default 1.0). 0 disables.
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_accum = 0
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            if loss_watch is not None:
                loss_watch.observe(epoch=epoch + 1, step=global_step,
                                   item_keys=batch.get("item_keys"), timestep=t_used, loss=loss.item())
            # refresh=False so only update(1) draws the bar — otherwise set_postfix AND update each
            # force a refresh, which a captured (non-tty) stderr logs as two lines per step (the
            # "187, 187, 188, 188" doubling). Training itself is one step per iteration.
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)

            # VRAM probe (FIZGIG_VRAM_PROBE=N): run N steps, report peak memory, exit. For
            # answering "does this config fit card X" without a full run or a checkpoint write.
            # `reserved` is the number to compare against a card's capacity — it is what torch
            # holds from the driver; `allocated` is only what is live inside that. Neither
            # includes the ~0.5-1 GB CUDA context, so nvidia-smi always reads a little higher.
            _probe_n = int(os.environ.get("FIZGIG_VRAM_PROBE", "0") or 0)
            if _probe_n and global_step >= _probe_n:
                torch.cuda.synchronize()
                _alloc = torch.cuda.max_memory_allocated() / 1024**3
                _resv = torch.cuda.max_memory_reserved() / 1024**3
                _tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
                _overall = max(_resv, _probe_load_gb)
                logger.info("[vram-probe] steps=%d  load=%.2f GB  training=%.2f GB  overall=%.2f GB  "
                            "(peak is %s)  allocated=%.2f GB  card=%.1f GB  alloc_conf=%s",
                            global_step, _probe_load_gb, _resv, _overall,
                            "LOAD" if _probe_load_gb >= _resv else "TRAINING",
                            _alloc, _tot, os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"))
                print(f"VRAM_PROBE_RESULT reserved={_overall:.3f} allocated={_alloc:.3f} "
                      f"load={_probe_load_gb:.3f} train={_resv:.3f}", flush=True)
                sys.exit(0)
        # Flush a partial accumulation group at the epoch boundary: the epoch-end work (adaptive
        # LR decisions, rollback snapshot, state save) must see a settled optimizer, and leftover
        # grads must not leak into the next epoch.
        if pending_accum > 0 and not fused_backward:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_accum = 0
        _ep_steps = global_step - _epoch_step0
        _ep_secs = time.time() - _epoch_t0
        _rate = f"  {_ep_secs / _ep_steps:.2f}s/it" if _ep_steps > 0 else ""
        if _ep_steps > 0 and _epoch_rate_prev:
            # Only flag a real change; epoch-to-epoch jitter of a few percent is normal, and in
            # component rotation each window trains a different share of the model.
            _drift = (_ep_secs / _ep_steps) / _epoch_rate_prev
            if _drift >= 1.15 or _drift <= 0.87:
                _rate += f" ({_drift:.2f}x vs last epoch)"
        if _ep_steps > 0:
            _epoch_rate_prev = _ep_secs / _ep_steps
        # Cumulative across FT pause/resume, matching checkpoint numbering (H3 twin; the
        # offset is 0 on fresh and LoRA runs, so nothing changes there).
        logger.info(f"epoch {epoch + 1 + ft_epoch_offset}/{max_train_epochs + ft_epoch_offset}  avr_loss={loss_recorder.moving_average:.4f}  step={global_step}"
                    + _rate
                    + (f"  lr={optimizer.param_groups[0]['lr']:.3e}" if (scheduler is not None and optimizer is not None) else ""))
        if rotator is not None and torch.cuda.is_available():
            # Per-window peak, reset at each rotation — the H3 twin of this line is what
            # calibrated that family's window planner, and _K2FT_OVERHEAD_GB here is still
            # a DERIVED guess with no field measurement behind it. GB (1e9), never GiB:
            # the planner constants are GB, and mixing the two cost H3 a full GB of
            # headroom before it was caught (27 Aug).
            logger.info("[ft-rotation] window peak VRAM: %.1f GB",
                        torch.cuda.max_memory_allocated() / 1e9)

        # Attention backend: cuDNN's kernel is ~6% faster per step but costs ~1.3 s per distinct
        # sequence shape to plan, so it only wins on runs long enough to amortize that. After a
        # full epoch every shape the dataset produces has been seen, so this is arithmetic rather
        # than a guess — see fizgig/modules/sdpa.py.
        # SUPPRESSED under torch.compile: flipping the backend mid-run changes the branch every
        # compiled block traced, forcing a retrace whose worst case (fullgraph + an unprimed
        # path) killed the run at the epoch-2 boundary. Compile's own win is the larger one;
        # take cuDNN only on uncompiled runs.
        _switch = None if _do_compile else \
            _consider_training_backend(steps_per_epoch * (max_train_epochs - epoch - 1))
        if _switch:
            _n_shapes, _needed = _switch
            logger.info(f"[attention] switching to the cuDNN backend for the rest of the run — "
                        f"{_n_shapes} distinct sequence shape(s), which pays back within "
                        f"{_needed} steps and this run has more left. Expect a slower first pass "
                        f"over each shape while it plans, then ~6% faster steps.")

        # Adaptive LR: epoch-boundary plateau tracker (before save/preview so they reflect the
        # post-adjustment state). Uses the smoothed avr_loss as the signal, like Klein.
        if adaptive:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)

        # Per-image loss watch: reclassify images (stuck/learning/easy), refresh next epoch's
        # multipliers, write loss_log/problem_images.json (the GUI's Problem Images popup reads it).
        if loss_watch is not None:
            loss_watch.epoch_boundary(epoch + 1)

        # Live caption repair: apply caption edits queued from the Problem Images window, and
        # (when enabled) auto-recaption confirmed-stuck images with the same Qwen3-VL — both
        # re-encode in place so the next epoch trains on the fixed captions.
        # NOT under a fine-tune: the re-encode moves the whole DiT to CPU and restores it
        # through a blocks_to_swap-aware path the FT rotation streamer knows nothing about
        # (same reason auto-recaption is disarmed above). Manual edits stay queued and
        # apply in the next LoRA-mode run.
        if ft_rotation:
            _q = os.path.join(output_dir, "loss_log", "caption_updates.json")
            if os.path.exists(_q):
                logger.info("[ft-rotation] caption edits are queued but held — live caption "
                            "re-encode is not available under a fine-tune; they will apply "
                            "in the next LoRA-mode run on this dataset.")
        else:
            _apply_caption_updates(output_dir, group, te_path, device, dit, blocks_to_swap,
                               loss_watch, epoch + 1,
                               auto_recaption=auto_recaption, trigger_word=trigger_word,
                               trigger_position=trigger_position,
                               recaptioned=recaptioned, image_dir=ar_image_dir,
                               caption_ext=ar_caption_ext,
                               recaption_instruction=recaption_instruction,
                               recaption_instruction_detailed=recaption_instruction_detailed)

        state_saved_this_epoch = False
        ft_ckpt_saved_this_epoch = False
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            if rotator is not None:
                # The full checkpoint IS the resumable state under fine-tuning: the LoRA is
                # inert and the optimizer may be None (fused backward), so a state dir would
                # snapshot nothing real. To continue a fine-tune, point --dit at the saved
                # checkpoint (structurally identical to the RAW base). The next start window
                # rides in the metadata so the GUI's Resume can pick it up without the console.
                _ck = os.path.join(output_dir,
                                   f"{output_name}-{epoch + 1 + ft_epoch_offset:06d}.safetensors")
                _next_w = rot_schedule.window_at(epoch + 1)
                _save_full_checkpoint(rotator, raw_path, _ck, extra_metadata={
                    "fizgig_next_start_window": str(_next_w),
                    "fizgig_ft_n_windows": str(rot_schedule.n_windows),
                    "fizgig_ft_epochs_done": str(epoch + 1 + ft_epoch_offset)})
                ft_ckpt_saved_this_epoch = True
                logger.info("[ft-rotation] to continue from this checkpoint: --dit %s "
                            "--finetune_start_window %d", os.path.basename(_ck), _next_w)
                if save_state:
                    logger.info("[ft-rotation] state dirs are skipped — the full checkpoint "
                                "is the state; swap the base to continue training.")
            else:
                # comfy_format so a user's picked-best epoch is byte-format-identical to the final
                # artifact (LoKR: LyCORIS-standard keys). No-op for standard LoRA.
                _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                           network_dim, network_alpha, dtype, extra_metadata=_sai_metadata(),
                           comfy_format=True)
                # Resumable state rides the checkpoint cadence. Safe to snapshot here: pending_accum
                # was flushed above, the adaptive-LR watcher has already made its call for this epoch,
                # and any queued caption updates are applied — so the optimizer is settled.
                if save_state:
                    # NON-FATAL by design. A real run (7 Aug, RunPod) died at 27% of 55 hours
                    # because rng.pt hit a full network volume — for a file whose only job is to
                    # make resume nicer. The checkpoint itself had already saved. State saving must
                    # never cost a run; if the disk is truly full, the next EPOCH CHECKPOINT will
                    # fail and that one is rightly fatal.
                    try:
                        _save_training_state(output_dir, output_name, network, optimizer,
                                             epoch=epoch + 1, global_step=global_step,
                                             network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                             extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
                        state_saved_this_epoch = True
                    except Exception as _se:
                        logger.error("[state] saving the resume state FAILED (%s: %s). This is "
                                     "almost always the disk — on RunPod the volume quota is "
                                     "invisible from inside the pod (the dashboard is the only true "
                                     "reading), so writes fail with no warning. Training continues; "
                                     "this epoch has no resume point. The epoch checkpoint itself "
                                     "already saved.", type(_se).__name__, _se)
                    prune_state_dirs(output_dir, output_name, keep_last_n_states)

        if (rotator is not None and do_previews
                and (ft_ckpt_saved_this_epoch or (epoch + 1) == max_train_epochs)):
            # Fine-tune: previews ride the checkpoint saves (plus the final epoch) — each
            # sample is the rehearsal of a file you can deploy. Clear the last step's
            # loss/batch refs first: the loss tensor's graph metadata holds the window's
            # Parameters (H3 field lesson, ab04379) and would pin them through the render.
            loss = batch = None
            logger.info(f"rendering previews (epoch {epoch + 1 + ft_epoch_offset}) via the "
                        "fine-tune bracket (training DiT + fresh Turbo LoRA)...")
            try:
                _lp = _ft_bracket_preview(epoch + 1 + ft_epoch_offset)
                if _lp:
                    _last_sample_prompt = _lp
            except Exception as _prev_err:
                _oom = "out of memory" in str(_prev_err).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1 + ft_epoch_offset} preview failed "
                    f"({'CUDA OOM' if _oom else type(_prev_err).__name__}); "
                    f"disabling previews for the rest of the run. Training continues and "
                    f"checkpoints still save normally.")
                do_previews = False
        elif (do_previews and sample_every_n_epochs and (epoch + 1) % sample_every_n_epochs == 0
                and turbo_net is not None):
            # Turbo-LoRA mode: live network, resident DiT, no parking. The override TE encode
            # is the one VRAM risk (the ~8 GB Qwen3-VL loads alongside the resident trainer,
            # where the parked path had the DiT out of the way) — the except keeps an OOM from
            # killing the run, same as the Turbo-checkpoint path.
            logger.info(f"rendering previews (epoch {epoch + 1}) on the training DiT + Turbo LoRA...")
            try:
                ov = _read_sample_override(output_dir)
                if ov:
                    logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                                f"seed={ov['seed']} {ov['width']}x{ov['height']}"
                                f"{' +ref' if ov.get('ref_image') else ''}")
                    prev_enc = encode_sample_prompts(te_path, [ov["prompt"]],
                                                     ref_image=ov.get("ref_image") or None, device=device)
                    prev_w, prev_h, prev_seed = ov["width"], ov["height"], ov["seed"]
                    prev_prompts = [ov["prompt"]]
                else:
                    prev_enc, prev_w, prev_h, prev_seed = encoded_prompts, sample_width, sample_height, sample_seed
                    prev_prompts = sample_prompts
                if prev_seed == 0:
                    prev_seed = random.randint(1, 2**31 - 1)
                    logger.info(f"[sample] seed 0 -> random {prev_seed}")
                _, _last_p = sample_previews_on_dit(dit, turbo_net, turbo_diffb, sample_ae, prev_enc,
                                       sample_dir, epoch + 1, output_name=output_name,
                                       steps=sample_steps, cfg_scale=sample_cfg_scale,
                                       neg=encoded_negative, width=prev_w, height=prev_h,
                                       seed=prev_seed, blocks_to_swap=blocks_to_swap, device=device,
                                       prompts=prev_prompts)
                if _last_p:
                    _last_sample_prompt = _last_p
            except Exception as _prev_err:
                _oom = "out of memory" in str(_prev_err).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1} preview failed "
                    f"({'CUDA OOM' if _oom else type(_prev_err).__name__}); "
                    f"disabling previews for the rest of the run. Training continues and LoRAs still save normally."
                )
                do_previews = False
            network.train()
        elif (do_previews and sample_every_n_epochs and rotator is None
                and (epoch + 1) % sample_every_n_epochs == 0):
            from safetensors.torch import load_file
            tmp = os.path.join(output_dir, "_sample_lora.safetensors")
            _save_lora(network, tmp, network_dim, network_alpha, dtype)
            logger.info(f"rendering previews (epoch {epoch + 1}) on the fp8 Turbo...")
            # The preview loads the fp8 Turbo (~13 GB) on top of the resident training DiT
            # (~14 GB fp8) + the VAE — two full models won't fit (OOMs ~30 GB on a 32 GB card).
            # Park the training DiT on CPU for the preview, then restore it (and its block-swap
            # placement) before the next epoch. Costs one CPU<->GPU round-trip per preview.
            dit.to("cpu")
            if getattr(dit, "_nf4_quantized", False):
                # NF4's packed weights + quant state are plain attributes that .to("cpu") ignores
                # (~6 GB would stay on the GPU), so move them explicitly to free the VRAM the
                # preview needs — restored in the finally below.
                from fizgig.modules.nf4 import move_nf4_to_device
                move_nf4_to_device(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Live sample override (GUI status-bar panel) — model-agnostic prompt/seed/res
                # for the next preview. Encoded here (after the training DiT is on CPU) so the
                # text encoder has room. No override -> the configured pre-encoded prompts.
                ov = _read_sample_override(output_dir)
                if ov:
                    logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                                f"seed={ov['seed']} {ov['width']}x{ov['height']}"
                                f"{' +ref' if ov.get('ref_image') else ''}")
                    prev_enc = encode_sample_prompts(te_path, [ov["prompt"]],
                                                     ref_image=ov.get("ref_image") or None, device=device)
                    prev_w, prev_h, prev_seed = ov["width"], ov["height"], ov["seed"]
                    prev_prompts = [ov["prompt"]]
                else:
                    prev_enc, prev_w, prev_h, prev_seed = encoded_prompts, sample_width, sample_height, sample_seed
                    prev_prompts = sample_prompts
                # Seed 0 means "random": pick a fresh seed for this preview so 0 isn't a fixed seed
                # (each epoch's sample differs). Covers the Samples-tab field and a 0 in the override.
                if prev_seed == 0:
                    prev_seed = random.randint(1, 2**31 - 1)
                    logger.info(f"[sample] seed 0 -> random {prev_seed}")
                _, _last_p = sample_previews(turbo_path, sample_ae, prev_enc, load_file(tmp), sample_dir, epoch + 1,
                                output_name=output_name, steps=sample_steps,
                                cfg_scale=sample_cfg_scale, neg=encoded_negative, width=prev_w,
                                height=prev_h, seed=prev_seed,
                                context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
                                blocks_to_swap=preview_blocks_to_swap, int8=preview_int8, device=device,
                                prompts=prev_prompts)
                if _last_p:
                    _last_sample_prompt = _last_p
            except Exception as _prev_err:
                # A preview failure — almost always CUDA OOM (the ~13 GB Turbo + the Qwen3-VL
                # encoder won't fit alongside the parked training DiT on a small card) — must NEVER
                # kill the run. Training and LoRA saving are independent of previews, so we log,
                # disable previews for the rest of this run (so we don't re-OOM every sample epoch),
                # and carry on. The training DiT is restored in the finally below.
                _oom = "out of memory" in str(_prev_err).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1} preview failed "
                    f"({'CUDA OOM — this card is too small for the Turbo preview' if _oom else type(_prev_err).__name__}); "
                    f"disabling previews for the rest of the run. Training continues and LoRAs still save normally."
                )
                do_previews = False
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                if blocks_to_swap > 0:
                    # Re-establish the training placement (non-swap blocks -> GPU, swap blocks -> CPU).
                    dit.move_to_device_except_swap_blocks(torch.device(device))
                    dit.switch_block_swap_for_training()
                else:
                    dit.to(device)
                if getattr(dit, "_nf4_quantized", False):
                    # Restore the 4-bit packed weights + quant state to the GPU (they were parked
                    # on CPU above; .to(device) doesn't touch them). NF4 forces blocks_to_swap=0.
                    from fizgig.modules.nf4 import move_nf4_to_device
                    move_nf4_to_device(dit, device)
            dit.train()
            network.train()

        # Graceful pause (GUI wrote <output_dir>/.pause_requested): save a full resumable
        # state at this epoch boundary and exit cleanly so the GPU frees. The GUI detects the
        # clean exit, records the paused state, and offers Resume. Same contract as Klein.
        if os.path.exists(pause_flag):
            # Pause ALWAYS saves state, regardless of the save-state settings — that's the whole
            # contract of the button. But if the cadence already wrote this epoch's state above,
            # don't re-dump ~0.5 GB into the same dir: _save_training_state overwrites in place,
            # so a crash during the rewrite would destroy the very state we're pausing to keep.
            # The flag (not os.path.isdir) is the right check — a stale dir left by an earlier
            # run with the same name would wrongly suppress a real pause save.
            if rotator is not None:
                # Under fine-tuning the full checkpoint IS the state — a state dir would hold
                # only the inert LoRA (and the optimizer may be None under fused backward),
                # and worse, an off-cadence pause used to save NOTHING at all: every epoch of
                # dense training since the last cadence checkpoint silently vanished on exit.
                if ft_ckpt_saved_this_epoch:
                    logger.info(f"[pause] requested — epoch {epoch + 1 + ft_epoch_offset} checkpoint "
                                "already saved this epoch; exiting cleanly")
                    logger.info("[ft-rotation] paused. Continue with: --dit %s "
                                "--finetune_start_window %d",
                                f"{output_name}-{epoch + 1 + ft_epoch_offset:06d}.safetensors",
                                rot_schedule.window_at(epoch + 1))
                else:
                    _pp = os.path.join(output_dir,
                                       f"{output_name}-{epoch + 1 + ft_epoch_offset:06d}.safetensors")
                    _next_w = rot_schedule.window_at(epoch + 1)
                    try:
                        _save_full_checkpoint(rotator, raw_path, _pp, extra_metadata={
                            "fizgig_next_start_window": str(_next_w),
                            "fizgig_ft_n_windows": str(rot_schedule.n_windows),
                            "fizgig_ft_epochs_done": str(epoch + 1 + ft_epoch_offset)})
                        logger.info("[ft-rotation] paused. Continue with: --dit %s "
                                    "--finetune_start_window %d", os.path.basename(_pp), _next_w)
                    except Exception as _se:
                        logger.error("[pause] checkpoint save FAILED (%s: %s) — there is NO new "
                                     "continuation point for this pause. Free disk space and "
                                     "continue from the previous saved checkpoint.",
                                     type(_se).__name__, _se)
            elif state_saved_this_epoch:
                logger.info(f"[pause] requested — state for epoch {epoch + 1} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested — saving state at epoch {epoch + 1} and exiting cleanly")
                try:
                    _save_training_state(output_dir, output_name, network, optimizer,
                                         epoch=epoch + 1, global_step=global_step,
                                         network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                         extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
                except Exception as _se:
                    logger.error("[pause] state save FAILED (%s: %s) — there is NO new resume "
                                 "point for this pause. Free disk space (on RunPod: check the "
                                 "volume quota in the dashboard) and resume from the previous "
                                 "saved state.", type(_se).__name__, _se)
            try:
                os.remove(pause_flag)
            except Exception:
                pass
            progress_bar.close()
            logger.info("[pause] state saved — exiting (exit 0).")
            sys.exit(0)

    progress_bar.close()
    if loss_watch is not None:
        loss_watch.close()

    # End-of-run state, so a finished LoRA can be trained further by raising max_train_epochs.
    # Skipped when the run trained nothing (resumed from a state already at the final epoch) —
    # the only dir we'd write is the one we resumed FROM, and the save overwrites in place.
    # No state dir under fine-tuning: the LoRA is inert, the optimizer may be None (fused
    # backward), and the full checkpoint below IS the continuation point.
    if save_state_on_train_end and max_train_epochs > start_epoch and rotator is None:
        # Non-fatal: the final LoRA is already on disk; dying here would turn a finished run red.
        try:
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=max_train_epochs, global_step=global_step,
                                 network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                 extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
        except Exception as _se:
            logger.error("[state] end-of-run state save FAILED (%s: %s) — the finished LoRA is "
                         "saved and fine; only train-further-by-resume is affected. Free disk "
                         "space (RunPod: dashboard quota) and re-run the last epoch if you need "
                         "the state.", type(_se).__name__, _se)

    out = os.path.join(output_dir, f"{output_name}.safetensors")
    # Record the context LoRA in metadata so users know to pair it at the same strength at
    # inference (the trained LoRA is context-dependent — same contract as Klein).
    extra = {"ss_optimizer": optimizer_label}
    if slider_pairs:
        # Deploy contract: the strength dial IS the slider. Tools read these to default
        # their range (Repair Studio / Royale scrub ±).
        extra.update({"ss_slider": "image_pairs",
                      "ss_slider_diff_weight": f"{float(slider_diff_weight):g}"})
    if context_lora_path:
        extra.update({"ss_context_lora": os.path.basename(context_lora_path),
                      "ss_context_lora_strength": str(context_lora_strength)})
    # Full SAI ModelSpec block — same keys ComfyUI/model managers read (GUI: Other Options →
    # Metadata), plus trigger phrase and an auto-picked sample thumbnail.
    extra.update(_sai_metadata())
    if rotator is not None:
        # Fine-tune mode: the training lives in the base weights, not the (inert) LoRA.
        _next_w = rot_schedule.window_at(max_train_epochs)
        extra.update({"fizgig_next_start_window": str(_next_w),
                      "fizgig_ft_n_windows": str(rot_schedule.n_windows),
                      "fizgig_ft_epochs_done": str(max_train_epochs + ft_epoch_offset)})
        _save_full_checkpoint(rotator, raw_path, out, extra_metadata=extra)
        logger.info(f"saved fine-tuned checkpoint -> {out}")
        logger.info("[ft-rotation] to train it further: --dit %s --finetune_start_window %d",
                    os.path.basename(out), _next_w)
        return out
    _save_lora(network, out, network_dim, network_alpha, dtype, extra_metadata=extra,
               comfy_format=True)
    logger.info(f"saved final LoRA -> {out}")
    return out
