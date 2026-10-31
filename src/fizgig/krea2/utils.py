"""Shared loaders / helpers for the Krea 2 (K2) integration."""

from __future__ import annotations

import json
import logging
from typing import Optional, Union

import torch

from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
from fizgig.krea2.safetensors_utils import load_safetensors

# Text-encoder imports are lazy (see _encoder_api) so the DiT loader + config can be used
# without pulling in transformers' Qwen3-VL classes (and the heavy encoder module).

logger = logging.getLogger(__name__)


# Dynamic fp8 quantization scope for the DiT: the per-block (SingleStreamBlock) attention
# and SwiGLU Linear weights — the heavy, repeated compute, matching the LoRA target. The
# modulation (`mod.lin`) is a raw nn.Parameter and the RMSNorm scales must stay in compute
# dtype, so both are excluded (cf. Z-Image's split). `txtfusion` (the text-fusion transformer,
# whose submodule is also named `layerwise_blocks` and so matches "blocks.") is small and
# delicate, so it is kept in compute dtype too.
KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]

# --- Pre-quantized (ComfyUI) fp8 support -------------------------------------
# A ComfyUI fp8_scaled checkpoint (e.g. krea2_turbo_fp8_scaled) stores raw fp8 weights +
# per-Linear `.weight_scale` scalars (+ optional `.comfy_quant` markers). Musubi's monkey
# patch expects `.scale_weight` broadcastable against [out, in], so a load-time split hook
# renames + reshapes, and load is done with allow_prequantized_fp8=True (no re-quantization).
_FP8_SCALE_SUFFIX = ".weight_scale"
_COMFY_FP8_MARKER_SUFFIX = ".comfy_quant"


def _reshape_prequant_fp8_scale(scale: "torch.Tensor") -> "torch.Tensor":
    """Per-channel [out] -> [out, 1]; per-tensor scalar [] -> [1]; else unchanged."""
    if scale.ndim == 1:
        return scale.unsqueeze(1)
    if scale.ndim == 0:
        return scale.reshape(1)
    return scale


def _make_krea2_comfy_fp8_split_hook(compute_dtype: "torch.dtype", expected=None):
    """Split hook: `.weight_scale` -> `.scale_weight` (reshaped, cast to compute dtype);
    drop `.comfy_quant` markers; pass everything else through. With `expected` (the model's own
    keys), a "model.diffusion_model." prefix is stripped and tensors the model doesn't have are dropped."""
    def split_hook(key, value):
        if key.endswith(_COMFY_FP8_MARKER_SUFFIX):
            return [], None
        name = _strip_prefix(key)
        if key.endswith(_FP8_SCALE_SUFFIX):
            new_key = name[: -len(_FP8_SCALE_SUFFIX)] + ".scale_weight"
            if expected is not None and new_key[: -len(".scale_weight")] + ".weight" not in expected:
                return [], None
            if value is None:
                return [new_key], None
            return [new_key], [_reshape_prequant_fp8_scale(value).to(compute_dtype)]
        if expected is None:
            return None, None
        if name not in expected:
            return [], None
        if name == key:
            return None, None
        return [name], (None if value is None else [value])
    return split_hook


def is_prequantized_fp8(path: str) -> bool:
    """True if `path` is a ComfyUI-style pre-quantized fp8 checkpoint (has `.weight_scale` keys)."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    with MemoryEfficientSafeOpen(path) as f:
        return any(k.endswith(_FP8_SCALE_SUFFIX) for k in f.keys())


# --- Other ComfyUI builds (#273) ---------------------------------------------
# Community Krea 2 checkpoints come as int8_tensorwise (often with ConvRot: each weight stored times a block
# Hadamard rotation), as fp8 with more layers quantised than ours, as the older scaled_fp8 layout
# (`.scale_weight` + a `scaled_fp8` marker), and sometimes with a "model.diffusion_model." prefix or tensors
# Krea 2 doesn't have (ComfyUI ignores them). fp8 in the current layout stays fp8; the rest is decoded to bf16
# as it loads, and Fizgig's own precision (INT8 / NF4) then applies, as for the RAW file.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_SIDE_SUFFIXES = (_FP8_SCALE_SUFFIX, _COMFY_FP8_MARKER_SUFFIX, ".scale_weight", ".input_scale", ".scale_input")


def _strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def dit_layout(path: str) -> str:
    """How a Krea 2 DiT file stores its weights: "fp8" (ComfyUI fp8_scaled: fp8 weights + `.weight_scale`, run
    as stored), "decode" (int8, with or without ConvRot, or the older `.scale_weight` fp8: decoded to bf16 at
    load) or "" (plain weights)."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    with MemoryEfficientSafeOpen(path) as f:
        h = {k: v for k, v in f.header.items() if k != "__metadata__"}
    if any(v["dtype"] == "I8" for v in h.values()) or \
            any(k.endswith(".scale_weight") or _strip_prefix(k) == "scaled_fp8" for k in h):
        return "decode"
    return "fp8" if any(k.endswith(_FP8_SCALE_SUFFIX) for k in h) else ""


class ComfyDecodeReader:
    """A "decode" file (dit_layout) read as plain weights, under the model's own names: each quantised Linear
    comes back dense (int8 codes x per-row scale with ConvRot's rotation undone, fp8 x its scale); markers,
    scales and tensors the model doesn't have are left out. keys() / get_tensor() as MemoryEfficientSafeOpen.
    The decode runs on `calc_device` (the rotation is a matmul per weight: slow on a CPU)."""

    def __init__(self, path: str, expected, calc_device=None):
        from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
        self.f = MemoryEfficientSafeOpen(path)
        self.src = {_strip_prefix(k): k for k in self.f.keys()}
        expected = set(expected)
        self._keys = [k for k in self.src if k in expected]
        extra = sorted(k for k in self.src if k not in expected and not k.endswith(_SIDE_SUFFIXES)
                       and k != "scaled_fp8")
        if extra:
            logger.info(f"Krea 2 DiT: skipping {len(extra)} tensors this model doesn't have "
                        f"({', '.join(extra[:4])}{', ...' if len(extra) > 4 else ''})")
        meta = self.f.metadata().get("_quantization_metadata")
        try:
            self.layers = json.loads(meta).get("layers", {}) if meta else {}
        except ValueError:
            self.layers = {}
        self.calc = torch.device(calc_device if calc_device is not None else
                                 ("cuda" if torch.cuda.is_available() else "cpu"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.file.close()

    def keys(self):
        return list(self._keys)

    def _side(self, base: str, suffix: str):
        k = self.src.get(base + suffix)
        return None if k is None else self.f.get_tensor(k)

    def get_tensor(self, key: str, device=None, dtype=None):
        src = self.src[key]
        stored = self.f.header[src]["dtype"]
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        if base is None or stored not in ("I8", "F8_E4M3", "F8_E5M2"):
            return self.f.get_tensor(src, device=device, dtype=dtype)
        scale = self._side(base, _FP8_SCALE_SUFFIX)
        if scale is None:
            scale = self._side(base, ".scale_weight")
        if scale is None:
            raise ValueError(f"{key}: a {stored} weight with no scale beside it")
        w = self.f.get_tensor(src, device=self.calc)
        scale = scale.to(self.calc, torch.float32)
        scale = scale.reshape(-1, 1) if scale.numel() > 1 else scale.reshape(1, 1)
        if stored == "I8":
            from fizgig.minimax.convrot import dequantize_int8_convrot, parse_comfy_quant
            marker = self._side(base, _COMFY_FP8_MARKER_SUFFIX)
            conf = parse_comfy_quant(marker) if marker is not None else dict(self.layers.get(base, {}))
            conf.setdefault("format", "int8_tensorwise")      # older INT8-Fast exports leave it out
            out = dequantize_int8_convrot(w, scale, conf, out_dtype=torch.float32)
        else:
            out = w.to(torch.float32) * scale
        return out.to(device=device if device is not None else "cpu",
                      dtype=dtype if dtype is not None else torch.bfloat16)


# The single config shipped with the OSS checkpoints (single_mmdit_large_wide).
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    lora_weights: Optional[list] = None,
    lora_multipliers: Optional[list] = None,
    fp8_fast: bool = False,
) -> SingleStreamDiT:
    """Build the K2 single-stream MMDiT on meta and load weights (assign=True).

    When ``fp8_scaled`` is True, the per-block Linear weights are dynamically quantized to
    scaled fp8 at load time and the matching Linear forwards are monkey-patched to
    dequantize on the fly (cf. Z-Image / qwen_image). ``dtype`` is then ignored — non-target
    weights (norms, modulation, embedders, heads) keep their checkpoint dtype.

    ``lora_weights`` (a list of loaded LoRA state dicts, with optional ``lora_multipliers``)
    are merged into the base weights at load time. This is the only correct route under fp8
    (fp8-quantized weights cannot be post-hoc merged), and it also keeps loading uniform for
    block swap: the merged/quantized state dict is produced before the model is placed, so the
    offloader can stream blocks afterward without an external weight mutation.

    For block swap, pass ``loading_device="cpu"``: the weights stay on CPU (``move_to_device``
    is then False) and the caller's ``enable_block_swap`` / ``move_to_device_except_swap_blocks``
    places the resident blocks on ``device`` and keeps the swap blocks on CPU.
    """
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    has_lora = lora_weights is not None and len(lora_weights) > 0

    logger.info(
        f"Loading Krea 2 DiT weights from {dit_path}"
        + (" (fp8 scaled)" if fp8_scaled else "")
        + (f" (+{len(lora_weights)} LoRA merged)" if has_lora else "")
    )
    from fizgig.utils.safetensors import warm_file_cache
    warm_file_cache(dit_path)
    with torch.device("meta"):
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    layout = dit_layout(dit_path)
    if layout == "decode":
        if fp8_scaled or has_lora:
            raise ValueError(f"{dit_path}: an int8 / older fp8 ComfyUI build loads as plain weights only")
        logger.info(f"Loading Krea 2 DiT from a ComfyUI quantised build, decoded to {dtype}: {dit_path}")
        with ComfyDecodeReader(dit_path, dit.state_dict().keys(),
                               calc_device=device if device.type == "cuda" else None) as f:
            sd = {k: f.get_tensor(k, device=loading_device, dtype=dtype) for k in f.keys()}
        dit.load_state_dict(sd, strict=True, assign=True)
        return dit

    if (not has_lora) and layout == "fp8":
        # Pre-quantized ComfyUI fp8 (e.g. krea2_turbo_fp8_scaled): keep the weights fp8, normalize
        # the Comfy scale layout via the split hook, and monkey-patch each Linear to dequantize on
        # the fly. No re-quantization, so it loads fast — ideal for the per-epoch sample swap.
        from fizgig.krea2.fp8_optimization_utils import apply_fp8_monkey_patch
        from fizgig.krea2.lora_utils import load_safetensors_with_lora_and_fp8
        from fizgig.krea2.safetensors_utils import WeightTransformHooks

        logger.info(f"Loading Krea 2 DiT from pre-quantized fp8 {dit_path}")
        hooks = WeightTransformHooks(split_hook=_make_krea2_comfy_fp8_split_hook(
            dtype, expected=set(dit.state_dict().keys())))
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
            weight_transform_hooks=hooks,
            allow_prequantized_fp8=True,
        )
        apply_fp8_monkey_patch(dit, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=True, assign=True)
        return dit

    if fp8_scaled or has_lora:
        # Single load path that merges LoRA (if any) into the base weights and optionally
        # quantizes the per-block Linears to scaled fp8. fp8 targets/excludes only apply when
        # quantizing; without fp8 the weights are merged and cast to ``dtype`` as-is.
        # Lazy: fp8/LoRA support modules (ported later) are only needed on this path.
        from fizgig.krea2.fp8_optimization_utils import apply_fp8_monkey_patch
        from fizgig.krea2.lora_utils import load_safetensors_with_lora_and_fp8

        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None if fp8_scaled else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
            # Fast FT needs a per-tensor scale: _scaled_mm takes one scalar per operand,
            # so the shipped block-64 layout cannot feed it.
            quantization_mode="tensor" if (fp8_scaled and fp8_fast) else "block",
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(dit, sd, use_scaled_mm=False, fast=fp8_fast)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=True, assign=True)
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM (mmap page cache + materialized tensor), file locking, and lazy disk reads. Load
        # directly to the target device+dtype (assign=True) so the loaded tensors become the params.
        sd = load_safetensors(dit_path, device=loading_device, disable_mmap=True, dtype=dtype)
        expected = set(dit.state_dict().keys())
        if not set(sd) <= expected:                # a "model.diffusion_model." prefix, or extras ComfyUI ignores
            sd = {_strip_prefix(k): v for k, v in sd.items()}
            extra = sorted(set(sd) - expected)
            if extra:
                logger.info(f"Krea 2 DiT: skipping {len(extra)} tensors this model doesn't have ({', '.join(extra[:4])})")
            sd = {k: v for k, v in sd.items() if k in expected}
        dit.load_state_dict(sd, strict=True, assign=True)

    return dit


def load_krea2_dit_state_dict(
    dit_path: str,
    fp8_scaled: bool = False,
    calc_device: Union[str, torch.device] = "cpu",
    result_device: Union[str, torch.device] = "cpu",
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_fast: bool = False,
) -> dict:
    """Produce a Krea 2 DiT state dict matching a model loaded via ``load_krea2_dit``.

    Unlike ``load_krea2_dit`` this builds no ``nn.Module`` — it returns only the weights,
    for swapping the base weights of an already-built model in place (e.g. RAW-train /
    Turbo-sample). When ``fp8_scaled`` is True the per-block Linears are dynamically
    quantized exactly as in ``load_krea2_dit`` (quantization runs on ``calc_device``), so
    the returned keys include the matching ``.scale_weight`` entries and line up 1:1 with
    the live model's ``named_parameters()`` + ``named_buffers()``. The result is moved to
    ``result_device``.

    When ``result_device`` equals ``calc_device`` (e.g. both the GPU, used by the M2 turbo/raw
    swap), the dict is built straight on that device with no full intermediate CPU dict — the
    CPU peak stays at ~1 tensor. When ``result_device`` is CPU (e.g. the M1 resident stash),
    the fp8 path quantizes on ``calc_device`` and then lands the dict on CPU.
    """
    calc_dev = torch.device(calc_device)
    rd = torch.device(result_device)
    # Keep the fp8-quantized tensors on calc_device when that is also the result device, so the
    # dict never round-trips through a full CPU copy (the M2 GPU-direct swap path).
    move_to_device = calc_dev == rd

    if fp8_scaled:
        from fizgig.krea2.lora_utils import load_safetensors_with_lora_and_fp8

        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=calc_dev,
            move_to_device=move_to_device,
            dit_weight_dtype=None,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
            quantization_mode="tensor" if fp8_fast else "block",
        )
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM, file locking, and lazy disk reads. Load directly to result_device in bf16.
        sd = load_safetensors(dit_path, device=result_device, disable_mmap=True, dtype=torch.bfloat16)

    sd = {k: v.to(rd) for k, v in sd.items()}
    return sd


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: Optional[int] = None,
    select_layers: Optional[tuple] = None,
    tokenizer_repo: Optional[str] = None,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``path`` (local safetensors,
    ComfyUI or official key layout), tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    # Lazy: pulls in transformers' Qwen3-VL classes only when an encoder is actually loaded.
    from fizgig.krea2.embedder import (
        QWEN3_VL_4B_INSTRUCT_REPO_ID,
        TextEncoderConfig,
        load_qwen3_vl_conditioner,
    )

    if max_length is None:
        max_length = TextEncoderConfig.max_length
    if select_layers is None:
        select_layers = TextEncoderConfig.select_layers
    if tokenizer_repo is None:
        tokenizer_repo = QWEN3_VL_4B_INSTRUCT_REPO_ID
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hiddens, mask).

    hiddens: (B, seq, num_select_layers, hidden) stacked selected hidden states.
    mask:    (B, seq) bool attention mask (valid tokens incl. suffix, padding=False).
    """
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)
