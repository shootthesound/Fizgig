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
from fizgig.training.train_utils import LossRecorder, prune_state_dirs

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
                         loading_device=loading_device)
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
        _compile_blocks(dit, blocks_to_swap)
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


def _compile_blocks(dit, blocks_to_swap: int) -> None:
    """Compile each transformer block. Opt-in — see the roadmap for what it is and isn't worth.

    The win is real on the quantised path (inductor fuses the per-matmul quantise/dequantise
    elementwise work that bounds INT8), and small on dense bf16. It costs compile time on the
    first step, and a recompile for every new latent shape a bucketed dataset presents.

    Refused under block swap: compiled graphs assume their weights stay put, and swap moves them
    between CPU and GPU every step.
    """
    if blocks_to_swap > 0:
        logger.warning("[compile] ignored — block swap moves weights between devices every step, "
                       "which invalidates compiled graphs. Quantise instead of swapping if you "
                       "want both.")
        return
    try:
        import triton  # noqa: F401
    except Exception:
        logger.warning("[compile] ignored — triton is not installed (pip install triton-windows "
                       "on Windows, triton on Linux)")
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


def sample_krea2_timesteps(bsize: int, num_img_tokens: int, device, sigmoid_scale: float = 1.0) -> torch.Tensor:
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
    return (t * shift) / (1.0 + (shift - 1.0) * t)


def compute_loss(dit, latent, hidden_states, attention_mask, *, shift=2.5, dtype=torch.bfloat16,
                 device=None):
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)         — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask

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
    t = sample_krea2_timesteps(B, num_img_tokens, device)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))
    img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
    target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    # Return the mean drawn timestep alongside the loss so the passive per-image loss logger can
    # normalize for noise level (the caller ignores it when logging is off).
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
    _save_lora(network, os.path.join(state_dir, "lora.safetensors"), network_dim, network_alpha, dtype)
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
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
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
                           *, auto_recaption=False, trigger_word=None, recaptioned=None,
                           image_dir=None, caption_ext=".txt", recaption_instruction=None,
                           recaption_instruction_detailed=None):
    """Live caption repair (Problem Images window). Consume <output_dir>/loss_log/caption_updates.json
    ({item_key: new_caption}), re-encode each caption with Qwen3-VL, and OVERWRITE the item's
    text-embedding cache file — the collate re-reads that file from disk every step, so the very
    next epoch trains on the corrected caption. Also resets the image's loss-watch history (its
    stuck record reflects the old caption). Never raises into the training loop.

    auto_recaption: additionally re-caption CONFIRMED-STUCK images with the same Qwen3-VL (it's a
    full VLM with a real LM head — the captioner ships inside the training stack), appending
    ", <trigger_word>" when one is set. Max TWO attempts per image per run (`recaptioned` is a
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

        # Auto-recaption: the SAME loaded VLM describes what's actually in the stuck image;
        # trigger word (if any) is appended at the END — per the conditional-trigger doctrine,
        # a trailing token is a far weaker identity claim than a leading one. Attempt 2 (the
        # first caption demonstrably failed) goes exhaustive-detail.
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
                    cap = f"{cap}, {trigger_word}"
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
    quant_4bit: bool = False,
    quant_int8: str = "",
    blocks_to_swap: int = 0,
    shift: float = 2.5,
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
    if str(compile_blocks).lower() == "auto":
        from fizgig.utils.capabilities import should_compile
        _steps_est = group.num_train_items * max_train_epochs
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
        _do_compile, _why = should_compile(_steps_est, quant_4bit, quant_int8, blocks_to_swap,
                                           mp=_mp_max, batch=_batch_max)
        logger.info("[compile] auto: %s — %s", "ENABLED" if _do_compile else "off", _why)

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
    encoded_prompts = sample_ae = sample_dir = None
    encoded_negative = None
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

    dit, network, turbo_net, turbo_diffb = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        network_type=network_type, lokr_factor=lokr_factor,
        fp8_scaled=fp8_scaled, quant_4bit=quant_4bit, quant_int8=quant_int8,
        blocks_to_swap=blocks_to_swap, compile_blocks=_do_compile,
        context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
        turbo_lora_path=(turbo_lora_path if do_previews else None),
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
    network.requires_grad_(True)

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
    if resume_state_dir and os.path.isdir(resume_state_dir):
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
    accum = max(1, int(gradient_accumulation_steps or 1))
    if accum > 1:
        logger.info(f"[grad_accum] {accum} micro-batches per optimizer step "
                    f"(effective batch {accum}); ~{max(1, steps_per_epoch // accum)} updates/epoch")

    scheduler = None
    if adaptive:
        if lr_scheduler and lr_scheduler != "constant":
            logger.info(f"[lr_scheduler] '{lr_scheduler}' ignored — adaptive LR is enabled and owns the LR.")
    elif lr_scheduler and lr_scheduler != "constant":
        from diffusers.optimization import get_scheduler
        # Schedules count OPTIMIZER steps, not micro-batches.
        total_steps = math.ceil(steps_per_epoch / accum) * max_train_epochs
        kwargs = {}
        if lr_scheduler == "cosine_with_restarts":
            kwargs["num_cycles"] = int(lr_scheduler_num_cycles)
        elif lr_scheduler == "polynomial":
            kwargs["power"] = float(lr_scheduler_power)
        scheduler = get_scheduler(
            lr_scheduler, optimizer,
            num_warmup_steps=int(lr_warmup_steps or 0),
            num_training_steps=total_steps,
            **kwargs,
        )
        # Resume: these schedulers are pure functions of the step count, and global_step is
        # already restored from the state dir — so re-deriving the position is exact and needs
        # no extra persisted state. Setting last_epoch then stepping once lands the LR exactly
        # where global_step calls to step() would have.
        if global_step > 0:
            # global_step counts micro-batches; the schedule's position is optimizer steps.
            # The loop flushes a PARTIAL accumulation group at every epoch boundary (the
            # scheduler steps there too), so updates/epoch = ceil(steps_per_epoch/accum) —
            # a flat `global_step // accum` ignored those flushes and restored the schedule
            # early, leaving the LR high for the whole remainder. Resume always lands on an
            # epoch boundary; the leftover term covers a hand-rolled mid-epoch state anyway.
            _epochs_done = global_step // steps_per_epoch
            _leftover = global_step % steps_per_epoch
            done_updates = (_epochs_done * math.ceil(steps_per_epoch / accum)
                            + _leftover // accum)
            scheduler.last_epoch = done_updates - 1
            scheduler.step()
        logger.info(f"[lr_scheduler] {lr_scheduler} — warmup {int(lr_warmup_steps or 0)} / "
                    f"{total_steps} total steps, start lr={optimizer.param_groups[0]['lr']:.3e}"
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
    # Sample at Start: an epoch-0 preview (base model + zero-init LoRA) so the run's
    # starting point is on record. Fresh runs only — a resume already has samples.
    if sample_at_first and do_previews and start_epoch == 0 and turbo_net is not None:
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

    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="steps", smoothing=0)
    pending_accum = 0  # micro-batches backward'd since the last optimizer step
    # Warm-up reassurance: the first two epochs start slowly (first-sight kernel planning,
    # cuBLAS algorithm picks, allocator + cache warm-up; the cuDNN switch at the epoch-1
    # boundary re-plans every shape in epoch 2). Users watching a crawling bar assume a
    # hang, so repeat a gentle note every ~30 s while it lasts.
    _warmup_note_last = 0.0
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        if _sampler is not None:
            _sampler.set_epoch(epoch)      # reshuffle within/across buckets each epoch
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
            loss, t_used = compute_loss(dit, batch["latents"], batch["hidden_states"], batch["attention_mask"],
                                        device=device,
                                        shift=shift, dtype=dtype)
            # Per-image LR: scale THIS step's gradient by the image's multiplier (throttle stuck
            # images, boost healthy learned ones). Raw loss is still what gets recorded/averaged below,
            # so avr_loss and the global adaptive-LR watcher see unscaled numbers.
            step_mult = loss_watch.multiplier(batch.get("item_keys")) if loss_watch is not None else 1.0
            # Divide by the accumulation count so N micro-batches AVERAGE into one update rather
            # than summing (which would scale the effective LR by N).
            _scaled = loss * step_mult if step_mult != 1.0 else loss
            (_scaled / accum if accum > 1 else _scaled).backward()
            pending_accum += 1
            if pending_accum >= accum:
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
        # Flush a partial accumulation group at the epoch boundary: the epoch-end work (adaptive
        # LR decisions, rollback snapshot, state save) must see a settled optimizer, and leftover
        # grads must not leak into the next epoch.
        if pending_accum > 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_accum = 0
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={loss_recorder.moving_average:.4f}  step={global_step}"
                    + (f"  lr={optimizer.param_groups[0]['lr']:.3e}" if scheduler is not None else ""))

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
        _apply_caption_updates(output_dir, group, te_path, device, dit, blocks_to_swap,
                               loss_watch, epoch + 1,
                               auto_recaption=auto_recaption, trigger_word=trigger_word,
                               recaptioned=recaptioned, image_dir=ar_image_dir,
                               caption_ext=ar_caption_ext,
                               recaption_instruction=recaption_instruction,
                               recaption_instruction_detailed=recaption_instruction_detailed)

        state_saved_this_epoch = False
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            # comfy_format so a user's picked-best epoch is byte-format-identical to the final
            # artifact (LoKR: LyCORIS-standard keys). No-op for standard LoRA.
            _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                       network_dim, network_alpha, dtype, extra_metadata=_sai_metadata(),
                       comfy_format=True)
            # Resumable state rides the checkpoint cadence. Safe to snapshot here: pending_accum
            # was flushed above, the adaptive-LR watcher has already made its call for this epoch,
            # and any queued caption updates are applied — so the optimizer is settled.
            if save_state:
                _save_training_state(output_dir, output_name, network, optimizer,
                                     epoch=epoch + 1, global_step=global_step,
                                     network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                     extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
                state_saved_this_epoch = True
                prune_state_dirs(output_dir, output_name, keep_last_n_states)

        if (do_previews and sample_every_n_epochs and (epoch + 1) % sample_every_n_epochs == 0
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
        elif do_previews and sample_every_n_epochs and (epoch + 1) % sample_every_n_epochs == 0:
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
            if state_saved_this_epoch:
                logger.info(f"[pause] requested — state for epoch {epoch + 1} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested — saving state at epoch {epoch + 1} and exiting cleanly")
                _save_training_state(output_dir, output_name, network, optimizer,
                                     epoch=epoch + 1, global_step=global_step,
                                     network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                     extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
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
    if save_state_on_train_end and max_train_epochs > start_epoch:
        _save_training_state(output_dir, output_name, network, optimizer,
                             epoch=max_train_epochs, global_step=global_step,
                             network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                             extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
        prune_state_dirs(output_dir, output_name, keep_last_n_states)

    out = os.path.join(output_dir, f"{output_name}.safetensors")
    # Record the context LoRA in metadata so users know to pair it at the same strength at
    # inference (the trained LoRA is context-dependent — same contract as Klein).
    extra = {"ss_optimizer": optimizer_label}
    if context_lora_path:
        extra.update({"ss_context_lora": os.path.basename(context_lora_path),
                      "ss_context_lora_strength": str(context_lora_strength)})
    # Full SAI ModelSpec block — same keys ComfyUI/model managers read (GUI: Other Options →
    # Metadata), plus trigger phrase and an auto-picked sample thumbnail.
    extra.update(_sai_metadata())
    _save_lora(network, out, network_dim, network_alpha, dtype, extra_metadata=extra,
               comfy_format=True)
    logger.info(f"saved final LoRA -> {out}")
    return out
