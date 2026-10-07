"""Bake a SliderState into a new safetensors LoRA.

Supports:
- Primary-only bake: disabled blocks dropped; enabled blocks rescaled by
  `primary_strength` (scale + mult baked into `lora_up`; output alpha = rank
  so new scale = 1.0, equivalent to the live preview's contribution).
- Donor-blended bake (standard LoRA + standard LoRA): rank-concatenation.
  For a block with both primary and donor enabled, the output module's
  `lora_up` is `cat([up_p * mp * scale_p, up_d * md * scale_d], dim=-1)` and
  `lora_down` is `cat([down_p, down_d], dim=0)`. Alpha set to the new rank so
  the file's effective scale is 1.0; the original primary/donor multipliers
  and alphas are fully baked in.
  Verification: `baked_up @ baked_down` equals the live inference sum
  `mp * scale_p * up_p @ down_p + md * scale_d * up_d @ down_d`.
- LyCORIS (LoKR / LoHa): baked LOSSLESSLY in native format. Both forms are
  linear in their first factor — `m·kron(w1,w2) == kron(m·w1,w2)` and
  `(m·W1)⊙W2 == m·(W1⊙W2)` — so a slider multiplier absorbs into w1 exactly
  like it absorbs into lora_up, and a LoKR in stays a LoKR out. The ONLY case
  that still SVDs to standard LoRA is a donor blend on the same block
  (Kronecker/Hadamard products don't rank-concatenate), and only for the
  blocks actually blended. GLoRA is still refused.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from fizgig.repair_studio.state import SliderState

logger = logging.getLogger(__name__)


_BLOCK_KEY_RE = re.compile(r"(?:lora_unet_)?(double_blocks|single_blocks)_(\d+)_")
# Krea 2 + MiniMax H3 module naming (see repair_studio.h3_blocks). txtfusion
# and token_refiner are checked before main blocks. Krea 2 and H3 SHARE the raw
# `lora_unet_blocks_N_` key shape but use disjoint block-id namespaces (block_N vs h3blk_N),
# so the mapper is resolved against the STATE's own ids: whichever namespace the state
# carries is the family being baked. A key that maps to an id absent from state.blocks
# still falls into the keep-as-is branch.
_KREA2_TXT_KEY_RE = re.compile(r"txtfusion_(layerwise|refiner)_blocks_(\d+)_")
_H3_REFINER_KEY_RE = re.compile(r"token_refiner_blocks_(\d+)_")
_MAIN_BLOCKS_KEY_RE = re.compile(r"lora_unet_blocks_(\d+)_")


def _block_id_from_key(key: str, state_block_ids=None) -> Optional[str]:
    m = _BLOCK_KEY_RE.search(key)
    if m:
        kind = m.group(1).replace("_blocks", "")  # "double" / "single"
        return f"{kind}_{int(m.group(2))}"
    t = _KREA2_TXT_KEY_RE.search(key)
    if t:
        return f"txt_{'lw' if t.group(1) == 'layerwise' else 'rf'}_{int(t.group(2))}"
    r = _H3_REFINER_KEY_RE.search(key)
    if r:
        return f"h3_rf_{int(r.group(1))}"
    m = _MAIN_BLOCKS_KEY_RE.search(key)
    if m:
        n = int(m.group(1))
        if state_block_ids is not None and f"h3blk_{n}" in state_block_ids:
            return f"h3blk_{n}"
        return f"block_{n}"
    return None


def _group_by_module(sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """{module_name: {suffix: tensor, ...}} — splits state dict keys on the
    first dot after the lora_unet_... prefix."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        if "." not in key:
            continue
        mod, _, suffix = key.partition(".")
        out.setdefault(mod, {})[suffix] = tensor
    return out


def _module_alpha(mod_keys: Dict[str, torch.Tensor], fallback_rank: int) -> float:
    """Extract alpha as a float; fallback to rank so scale = 1.0."""
    alpha = mod_keys.get("alpha")
    if alpha is None:
        return float(fallback_rank)
    try:
        return float(alpha.item())
    except Exception:
        return float(fallback_rank)


def _bake_single_lycoris_contribution(
    mod_keys: Dict[str, torch.Tensor], multiplier: float,
) -> Optional[Dict[str, torch.Tensor]]:
    """Bake a multiplier into a LoKR/LoHa module IN NATIVE FORMAT — no SVD, no loss.

    Both forms are linear in their first factor, so `multiplier * scale` absorbs into
    lokr_w1 (or lokr_w1_a / hada_w1_a) the way _bake_single_contribution absorbs into
    lora_up. Output alpha is the >=1e6 sentinel meaning "scale already baked in"
    (the LyCORIS analogue of "alpha = rank"). A no-op (factor exactly 1.0) returns the
    module byte-identical, original alpha included. Returns None if the module isn't a
    recognised LyCORIS variant.
    """
    from fizgig.networks.lora import lycoris_scale_from_keys

    if mod_keys.get("lokr_w1_a") is not None or mod_keys.get("lokr_w1") is not None:
        first = "lokr_w1_a" if mod_keys.get("lokr_w1_a") is not None else "lokr_w1"
    elif mod_keys.get("hada_w1_a") is not None:
        first = "hada_w1_a"
    else:
        return None

    factor = float(multiplier) * lycoris_scale_from_keys(mod_keys)
    if factor == 1.0:
        return dict(mod_keys)
    out = dict(mod_keys)
    t = out[first]
    out[first] = (t.to(torch.float32) * factor).to(t.dtype).contiguous()
    out["alpha"] = torch.tensor(1e10)
    return out


def _materialize_lycoris_module(mod_keys: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    """Convert a single LoKR or LoHa module to standard lora_up/lora_down via SVD.

    Returns a new dict with lora_up.weight, lora_down.weight, alpha — or None
    if the module isn't a recognised LyCORIS variant.
    """
    from fizgig.networks.lora import lycoris_scale_from_keys

    # --- LoKR ---
    if mod_keys.get("lokr_w1") is not None or mod_keys.get("lokr_w1_a") is not None:
        if mod_keys.get("lokr_w1_a") is not None:
            w1 = (mod_keys["lokr_w1_a"].float() @ mod_keys["lokr_w1_b"].float())
        else:
            w1 = mod_keys["lokr_w1"].float()
        if mod_keys.get("lokr_w2_a") is not None:
            w2 = (mod_keys["lokr_w2_a"].float() @ mod_keys["lokr_w2_b"].float())
        else:
            w2 = mod_keys["lokr_w2"].float()
        from fizgig.utils.device import gpu_kron
        W = gpu_kron(w1, w2) * lycoris_scale_from_keys(mod_keys)

    # --- LoHa ---
    elif mod_keys.get("hada_w1_a") is not None:
        W1 = mod_keys["hada_w1_a"].float() @ mod_keys["hada_w1_b"].float()
        W2 = mod_keys["hada_w2_a"].float() @ mod_keys["hada_w2_b"].float()
        W = (W1 * W2) * lycoris_scale_from_keys(mod_keys)

    else:
        return None

    # SVD the dense delta to a standard LoRA at a reasonable rank
    # Use min(64, min_dim) — preserves most information without bloating
    min_dim = min(W.shape)
    target_rank = min(64, min_dim)

    try:
        from fizgig.utils.device import gpu_svd
        U, S, Vt = gpu_svd(W)
    except Exception:
        return None

    R = min(target_rank, S.shape[0])
    sqrt_S = torch.sqrt(S[:R])
    lora_up = (U[:, :R] * sqrt_S.unsqueeze(0)).to(torch.float16)
    lora_down = (sqrt_S.unsqueeze(1) * Vt[:R, :]).to(torch.float16)

    return {
        "lora_up.weight": lora_up.contiguous(),
        "lora_down.weight": lora_down.contiguous(),
        "alpha": torch.tensor(float(R), dtype=torch.float16),
    }


def _bake_single_contribution(
    mod_keys: Dict[str, torch.Tensor],
    multiplier: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Bake one contribution (primary OR donor) into up/down tensors with
    scale + multiplier absorbed into up. Returns (new_up, down, rank) or None
    if the module isn't a standard LoRA (no lora_up/lora_down keys)."""
    up = mod_keys.get("lora_up.weight")
    down = mod_keys.get("lora_down.weight")
    if up is None or down is None:
        return None
    rank = int(down.shape[0])  # lora_down is (rank, in_dim)
    alpha = _module_alpha(mod_keys, rank)
    scale = alpha / max(rank, 1)
    factor = multiplier * scale
    if factor == 1.0:
        new_up = up.clone()
    else:
        new_up = (up.to(torch.float32) * factor).to(up.dtype)
    return new_up, down.clone(), rank


def save_repaired_lora(
    primary_path: str,
    state: SliderState,
    out_path: str,
    donor_path: Optional[str] = None,
) -> dict:
    """Bake state into a new .safetensors. Returns a summary dict with
    dropped_blocks, rescaled_blocks, blended_blocks, keys_in, keys_out, use_at.

    Load strengths (state.primary_scale / donor_scale, H3's Strength boxes; 1.0 on Klein): a primary-only file
    leaves them out and is used at the primary's strength. Once a donor contributes, the two LoRAs share one file,
    so each is baked at its own strength (on its block modules, as the live preview applies it) and the file is
    used at 1.0."""
    from fizgig.networks.lora import ensure_kohya_lora_state_dict, detect_lora_format, UnsupportedLoRAFormat

    if not os.path.isfile(primary_path):
        raise FileNotFoundError(primary_path)

    # Load primary.
    sd_p_raw: Dict[str, torch.Tensor] = load_file(primary_path)
    sd_p = ensure_kohya_lora_state_dict(sd_p_raw)
    fmt_p = detect_lora_format(sd_p)
    if fmt_p == "glora":
        raise UnsupportedLoRAFormat(
            f"Primary LoRA format is GLoRA (4-matrix form); not supported for bake."
        )

    # Metadata from the primary file.
    metadata: Dict[str, str] = {}
    try:
        from safetensors import safe_open
        with safe_open(primary_path, framework="pt") as f:
            md = f.metadata() or {}
            metadata = {str(k): str(v) for k, v in md.items()}
    except Exception:
        logger.exception("Could not read primary metadata; saving without it")

    # Load donor (optional).
    sd_d: Optional[Dict[str, torch.Tensor]] = None
    if donor_path and os.path.isfile(donor_path):
        sd_d_raw = load_file(donor_path)
        sd_d = ensure_kohya_lora_state_dict(sd_d_raw)
        fmt_d = detect_lora_format(sd_d)
        if fmt_d == "glora":
            raise UnsupportedLoRAFormat(
                f"Donor LoRA format is GLoRA (4-matrix form); not supported for bake."
            )

    modules_p = _group_by_module(sd_p)
    modules_d = _group_by_module(sd_d) if sd_d is not None else {}

    # LyCORIS modules are NOT converted up front any more — drop / rescale / pass-through all
    # bake losslessly in native format. SVD happens per-module inside the blend branch, the one
    # place rank-concat genuinely needs standard up/down, and nowhere else.
    lycoris_converted = 0

    all_modules = sorted(set(modules_p) | set(modules_d))

    sd_out: Dict[str, torch.Tensor] = {}
    dropped_blocks: set = set()
    rescaled_blocks: set = set()
    blended_blocks: set = set()
    combined_ranks: Dict[str, int] = {}  # module_name → new rank

    keys_in = len(sd_p) + (len(sd_d) if sd_d is not None else 0)

    ps = float(getattr(state, "primary_scale", 1.0))
    ds = float(getattr(state, "donor_scale", 1.0))
    donor_blocks = {_block_id_from_key(m, state.blocks.keys()) for m in modules_d}
    fold = any(bs.donor_enabled and abs(float(bs.donor_strength)) > 1e-9 and b in donor_blocks
               for b, bs in state.blocks.items())
    bake_state = state
    if fold and (ps != 1.0 or ds != 1.0):
        bake_state = state.copy()
        for bs in bake_state.blocks.values():
            bs.primary_strength = float(bs.primary_strength) * ps
            bs.donor_strength = float(bs.donor_strength) * ds

    for mod_name in all_modules:
        block_id = _block_id_from_key(mod_name, state.blocks.keys())
        if block_id is None:
            # Not a transformer-block module — pass primary's keys through if present.
            if mod_name in modules_p:
                for suffix, tensor in modules_p[mod_name].items():
                    sd_out[f"{mod_name}.{suffix}"] = tensor
            continue

        bs = bake_state.blocks.get(block_id)
        if bs is None:
            # No slider state for this block — keep primary as-is.
            if mod_name in modules_p:
                for suffix, tensor in modules_p[mod_name].items():
                    sd_out[f"{mod_name}.{suffix}"] = tensor
            continue

        has_p = mod_name in modules_p
        has_d = mod_name in modules_d
        # A zero-strength side contributes nothing — treat it as off. donor_enabled
        # once defaulted True with donor_strength 0.0, so merely LOADING a donor used to
        # rank-concatenate every shared block with an all-zero donor half (~2x the file
        # in dead weights) and write all-zero donor-only modules, before the user
        # touched a single slider.
        p_on = bs.primary_enabled and has_p and abs(float(bs.primary_strength)) > 1e-9
        d_on = bs.donor_enabled and has_d and abs(float(bs.donor_strength)) > 1e-9

        if not p_on and not d_on:
            dropped_blocks.add(block_id)
            continue

        if p_on and d_on:
            # Rank-concat: both contributions baked and concatenated. A LyCORIS side can't
            # rank-concat, so it (and only it, and only here) is SVD-materialized first.
            p_keys, d_keys = modules_p[mod_name], modules_d[mod_name]
            if p_keys.get("lora_up.weight") is None:
                mat = _materialize_lycoris_module(p_keys)
                if mat is not None:
                    p_keys = mat
                    lycoris_converted += 1
            if d_keys.get("lora_up.weight") is None:
                mat = _materialize_lycoris_module(d_keys)
                if mat is not None:
                    d_keys = mat
                    lycoris_converted += 1
            p_baked = _bake_single_contribution(p_keys, bs.primary_strength)
            d_baked = _bake_single_contribution(d_keys, bs.donor_strength)
            if p_baked is None or d_baked is None:
                # Fall back to whichever side has valid keys (shouldn't happen for well-formed std LoRA).
                logger.warning(f"Concat bake skipped for {mod_name} — missing up/down in one side")
                if p_baked is not None:
                    new_up, down, rank = p_baked
                    sd_out[f"{mod_name}.lora_up.weight"] = new_up
                    sd_out[f"{mod_name}.lora_down.weight"] = down
                    sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
                    rescaled_blocks.add(block_id)
                elif d_baked is not None:
                    new_up, down, rank = d_baked
                    sd_out[f"{mod_name}.lora_up.weight"] = new_up
                    sd_out[f"{mod_name}.lora_down.weight"] = down
                    sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
                    rescaled_blocks.add(block_id)
                continue

            up_p, down_p, rank_p = p_baked
            up_d, down_d, rank_d = d_baked

            # Normalize dtypes so cat doesn't choke if donor was saved in a different dtype.
            common_dtype = up_p.dtype
            if up_d.dtype != common_dtype:
                up_d = up_d.to(common_dtype)
                down_d = down_d.to(common_dtype)
            if down_p.dtype != common_dtype:
                down_p = down_p.to(common_dtype)

            new_rank = rank_p + rank_d
            new_up = torch.cat([up_p, up_d], dim=-1)     # (out, rank_p + rank_d)
            new_down = torch.cat([down_p, down_d], dim=0)  # (rank_p + rank_d, in)
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = new_down
            # alpha = rank so inference scale = 1.0; mult + original scales already in new_up.
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(new_rank))
            blended_blocks.add(block_id)
            combined_ranks[mod_name] = new_rank

        elif p_on:
            baked = _bake_single_contribution(modules_p[mod_name], bs.primary_strength)
            if baked is None:
                # LyCORIS module: bake the multiplier in native format — lossless. (The old
                # code passed the raw keys through here, silently IGNORING the multiplier.)
                lyc = _bake_single_lycoris_contribution(modules_p[mod_name], bs.primary_strength)
                if lyc is not None:
                    for suffix, tensor in lyc.items():
                        sd_out[f"{mod_name}.{suffix}"] = tensor
                    if bs.primary_strength != 1.0:
                        rescaled_blocks.add(block_id)
                else:
                    # Genuinely unknown module type — pass raw keys (future-proofing).
                    for suffix, tensor in modules_p[mod_name].items():
                        sd_out[f"{mod_name}.{suffix}"] = tensor
                continue
            new_up, down, rank = baked
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = down
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
            if bs.primary_strength != 1.0:
                rescaled_blocks.add(block_id)

        elif d_on:
            baked = _bake_single_contribution(modules_d[mod_name], bs.donor_strength)
            if baked is None:
                # LyCORIS donor module: native-format bake. (The old code hit `continue` here,
                # silently DROPPING the module from the output.)
                lyc = _bake_single_lycoris_contribution(modules_d[mod_name], bs.donor_strength)
                if lyc is not None:
                    for suffix, tensor in lyc.items():
                        sd_out[f"{mod_name}.{suffix}"] = tensor
                    rescaled_blocks.add(block_id)
                continue
            new_up, down, rank = baked
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = down
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
            rescaled_blocks.add(block_id)

    # The metadata was copied wholesale from the primary — scrub what no longer describes
    # THIS file: content hashes (every repaired LoRA inherited the SOURCE's hash) and
    # ss_network_dim/alpha, which Fizgig's own Profiler reads and which a donor blend
    # changes per-block. Report the actual max rank; per-module alpha == rank (scale 1.0).
    for _stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
        metadata.pop(_stale, None)
    _has_lycoris_out = any(".lokr_" in k or ".hada_" in k for k in sd_out)
    try:
        _ranks = [int(t.shape[0]) for k, t in sd_out.items() if k.endswith(".lora_down.weight")]
        if _ranks:
            metadata["ss_network_dim"] = str(max(_ranks))
            metadata["ss_network_alpha"] = str(float(max(_ranks)))
        elif _has_lycoris_out:
            # Pure-LyCORIS output has no rank; inherited dim/alpha would describe a file
            # this one no longer is.
            metadata.pop("ss_network_dim", None)
            metadata.pop("ss_network_alpha", None)
    except Exception:
        pass

    # Record slider config + donor reference in metadata.
    try:
        metadata["ss_repair_studio_config"] = json.dumps(state.to_json(), separators=(",", ":"))
    except Exception:
        logger.exception("Could not serialize SliderState into metadata")
    if donor_path:
        metadata["ss_repair_studio_donor_path"] = os.path.basename(donor_path)
    if combined_ranks:
        metadata["ss_repair_studio_combined_ranks"] = json.dumps(
            {k: v for k, v in sorted(combined_ranks.items())}, separators=(",", ":"))
    if lycoris_converted:
        metadata["ss_repair_studio_lycoris_svd"] = str(lycoris_converted)
    use_at = 1.0 if fold else ps
    metadata["ss_repair_studio_use_at"] = f"{use_at:g}"

    # safetensors requires contiguous tensors
    sd_out = {k: v.contiguous() if v.is_floating_point() else v for k, v in sd_out.items()}
    save_file(sd_out, out_path, metadata=metadata)

    _has_std_out = any(k.endswith(".lora_down.weight") for k in sd_out)
    summary = {
        "dropped_blocks": sorted(dropped_blocks),
        "rescaled_blocks": sorted(rescaled_blocks - dropped_blocks),
        "blended_blocks": sorted(blended_blocks),
        "keys_in": keys_in,
        "keys_out": len(sd_out),
        "donor_path": donor_path,
        # What the OUTPUT file is, so the GUI can say "saved natively as LoKR" vs warn
        # about the SVD a blend forced.
        "format_out": ("mixed" if (_has_lycoris_out and _has_std_out)
                       else "lycoris" if _has_lycoris_out else "standard"),
        "lycoris_converted": lycoris_converted,
        "use_at": use_at,
        "strengths_baked": fold,
    }
    logger.info(
        "save_repaired_lora: primary_in=%d donor_in=%d out=%d dropped=%d rescaled=%d blended=%d → %s",
        len(sd_p), len(sd_d) if sd_d is not None else 0, summary["keys_out"],
        len(summary["dropped_blocks"]), len(summary["rescaled_blocks"]), len(summary["blended_blocks"]),
        out_path,
    )
    return summary
