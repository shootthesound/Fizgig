"""Extract a LoRA from the difference between two full checkpoints.

The companion to rotating-block fine-tuning: that produces a ~26 GB model, this turns the
difference from its starting base into an ordinary, shareable LoRA. Measured on a 3-subject
Krea 2 fine-tune, rank 64 is perceptually indistinguishable from the full checkpoint and
rank 8 still holds identity separation — see docs/FINETUNE_ROTATION.md.

Multi-rank is nearly free and is the point of the API: SVD returns singular values in
descending order, so a rank-r factorisation is a truncation of the same decomposition.
Asking for [16, 32, 64, 128] costs one SVD per layer, not four, and each result is
bit-identical to extracting that rank alone.

Architecture-agnostic: keys are flattened to the kohya convention
(``blocks.0.attn.wq.weight`` -> ``lora_unet_blocks_0_attn_wq``), which is what the Klein,
Krea 2 and MiniMax H3 LoRA loaders all expect. A checkpoint that names its model under a prefix
(Anima's ``net.``, SDXL's ``model.diffusion_model.``) has it dropped first, so ComfyUI maps the keys.

Pre-quantized checkpoints are decoded before the diff, not after — see `dense_weight`. That
matters for MiniMax H3, where 200 of the 264 comparable matrices are stored as int8 ConvRot
codes rather than weights.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Layers a LoRA meaningfully applies to: 2-D Linear weights, excluding norms/embeddings
# (1-D or not linear maps, and not what adapters target).
_SKIP_SUBSTRINGS = ("norm", "embed", "_scale", "modulation")


_HADAMARD = {}


def _unrotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Undo ConvRot's block regular-Hadamard rotation along the last dim.

    The matrix is symmetric and orthogonal, so it is its own inverse — applying it is undoing
    it. Built in fp32 with entries exactly +-1 through the Kronecker powers, then divided by
    its power-of-two root, so the construction is exact.
    """
    if rot_size <= 1:
        return x
    if x.shape[-1] % rot_size:
        raise RuntimeError(f"last dim {x.shape[-1]} is not a multiple of the rotation block "
                           f"{rot_size}")
    key = (rot_size, str(x.device), x.dtype)
    h = _HADAMARD.get(key)
    if h is None:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float32)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise RuntimeError(f"convrot group size {rot_size} is not a power of 4")
        h = (h / rot_size ** 0.5).to(device=x.device, dtype=x.dtype)
        _HADAMARD[key] = h
    return torch.matmul(x.reshape(-1, x.shape[-1] // rot_size, rot_size), h).reshape(x.shape)


def dense_weight(handle, key: str, keyset, device) -> torch.Tensor:
    """The real fp32 weight for `key`, decoding pre-quantized storage when it is present.

    MiniMax H3's checkpoints (and any other ComfyUI int8 ConvRot file) do not store weights —
    they store int8 CODES in a rotated basis, alongside a per-output-row scale and a
    `comfy_quant` blob describing the rotation. Reading `.weight` straight off such a file and
    subtracting gives the difference of two sets of quantisation codes, which is not the
    difference of two models: the codes live in a rotated basis and each file carries its own
    scales. It does not fail — it produces a confident, meaningless LoRA, which is worse.

    So: decode first, then diff. Files that store ordinary float weights are returned as-is.
    """
    w = handle.get_tensor(key)
    if w.dtype != torch.int8:
        return w.to(device, torch.float32)

    stem = key[: -len(".weight")]
    conf_key, scale_key = f"{stem}.comfy_quant", f"{stem}.weight_scale"
    if conf_key not in keyset or scale_key not in keyset:
        raise RuntimeError(
            f"{key} is int8 but has no {os.path.basename(conf_key)} / "
            f"{os.path.basename(scale_key)} beside it — cannot decode it to real weights, and "
            f"diffing the raw codes would be meaningless.")

    conf = json.loads(bytes(handle.get_tensor(conf_key).to(torch.uint8).numpy().tobytes())
                      .decode("utf-8"))
    if conf.get("format") != "int8_tensorwise":
        raise RuntimeError(f"{key}: unsupported quantization {conf.get('format')!r}")
    # Decode ON the target device: the inverse rotation is a matmul per 256-wide block and
    # fc1 is [28672, 5376] — on CPU that is minutes a layer, across 200 of them.
    scale = handle.get_tensor(scale_key).to(device, torch.float32).reshape(-1, 1)
    dense = w.to(device, torch.float32) * scale
    group = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
    return _unrotate(dense, group)


def is_lora_target(key: str, shape) -> bool:
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    # by path component: "embed" in a stack of transformer blocks (MOSS-Image's text_embedder.blocks.N.qkv) is not an
    # embedding - its Linears are the text adapter's, and a fine-tune changes them like any other block
    parts = key.lower().split(".")
    for i, p in enumerate(parts):
        for s in _SKIP_SUBSTRINGS:
            if s in p and not (s == "embed" and i + 1 < len(parts) and parts[i + 1] in ("blocks", "layers")):
                return False
    return True


def lora_key(model_key: str, prefix: str = "") -> str:
    """blocks.0.attn.wq.weight -> lora_unet_blocks_0_attn_wq; a checkpoint's own prefix (`prefix`) is dropped first"""
    if prefix and model_key.startswith(prefix):
        model_key = model_key[len(prefix):]
    return "lora_unet_" + model_key[: -len(".weight")].replace(".", "_")


def model_prefixes() -> list:
    """The prefixes a model file can name its diffusion model's weights under: each fine-tune family's own
    (FTSpec.file_prefix - Anima's "net.", a single-file SDXL checkpoint's "model.diffusion_model.") and ComfyUI's
    "diffusion_model.". Longest first."""
    out = {"diffusion_model."}
    try:
        from fizgig.families.registry import training_families
        for d in training_families():
            if d.finetune:
                try:
                    p = d.load_driver().ft_spec(None).file_prefix
                except Exception:
                    p = ""
                if p:
                    out.add(p)
    except Exception:
        pass
    return sorted(out, key=len, reverse=True)


def prefix_family(prefix):
    """The fine-tune family whose checkpoints name their model weights under `prefix` (its FTSpec.file_prefix), or
    None - ComfyUI's generic "diffusion_model." belongs to no one family."""
    if not prefix or prefix == "diffusion_model.":
        return None
    try:
        from fizgig.families.registry import training_families
        for d in training_families():
            if d.finetune:
                try:
                    if d.load_driver().ft_spec(None).file_prefix == prefix:
                        return d
                except Exception:
                    pass
    except Exception:
        pass
    return None


def model_prefix(keys) -> str:
    """The prefix most of a checkpoint's keys share from model_prefixes(), or "" (Klein, Krea 2, Qwen, H3 files)."""
    keys = list(keys)
    counts = {p: sum(k.startswith(p) for k in keys) for p in model_prefixes()}
    best = max(counts, key=counts.get) if counts else ""
    return best if counts.get(best, 0) else ""


def identity_layout(base_keys, prefix=""):
    """(model name, block numbers, pattern) for a checkpoint laid out as a model with identity blocks
    (description.identity_blocks): its block list under its fine-tune spec's name, as many blocks as its block map;
    the pattern's group 1 is a key's block number. None when no such model matches."""
    import re
    try:
        from fizgig.families.registry import training_families
        fams = [d for d in training_families() if d.finetune and d.identity_blocks and d.block_categories]
    except Exception:
        return None
    for d in fams:
        try:
            blocks = d.load_driver().ft_spec(None).blocks
        except Exception:
            continue
        pat = re.compile(rf"^{re.escape(prefix)}{re.escape(blocks)}\.(\d+)\.")
        found = {int(m.group(1)) for m in (pat.match(k) for k in base_keys) if m}
        if found and len(found) == len(d.block_categories):
            return d.display_name, {int(b.rsplit("_", 1)[1]) for b in d.identity_blocks}, pat
    return None


def factor_multi(delta: torch.Tensor, ranks: Iterable[int]) -> dict:
    """One SVD, sliced to every requested rank. Returns {rank: (up, down)} in fp16/CPU.

    Singular values are split evenly between the factors (sqrt on each side) so neither
    matrix carries the whole magnitude — the usual convention, and it keeps both halves in
    a sane numeric range for fp16 storage.
    """
    m, n = delta.shape
    cap = min(m, n)
    wanted = sorted({int(r) for r in ranks if int(r) >= 1})
    if not wanted or cap < 1:
        return {}
    top = min(max(wanted), cap)
    # NOTE: svd_lowrank returns V, not Vh — delta ~= U @ diag(S) @ V.T
    U, S, V = torch.svd_lowrank(delta, q=min(top + 8, cap), niter=4)
    out = {}
    for r in wanted:
        k = min(r, cap)
        s = torch.sqrt(S[:k])
        up = (U[:, :k] * s.unsqueeze(0)).to(torch.float16).cpu().contiguous()          # (m, k)
        down = (s.unsqueeze(1) * V[:, :k].T).to(torch.float16).cpu().contiguous()      # (k, n)
        out[r] = (up, down)
    return out


def extract_diff_loras(
    base_path: str,
    tuned_path: str,
    output_dir: str,
    ranks: Iterable[int],
    name: str = "extracted",
    device: Optional[str] = None,
    min_norm: float = 1e-4,
    progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    identity: bool = False,
) -> list:
    """Diff two checkpoints and write one LoRA per rank. Returns the paths written. identity: a model's identity
    blocks keep IDENTITY_RANK x each rank (families.extract; Extract's identity mode, #301).

    Streams tensor-by-tensor, so peak memory is a couple of layers regardless of how big
    the checkpoints are.
    """
    ranks = sorted({int(r) for r in ranks if int(r) >= 1})
    if not ranks:
        raise ValueError("no ranks requested")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    say = log or (lambda _m: None)
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    h_base = safe_open(base_path, framework="pt", device="cpu")
    h_tune = safe_open(tuned_path, framework="pt", device="cpu")
    base_keys, tune_keys = set(h_base.keys()), set(h_tune.keys())

    only_tuned = len(tune_keys - base_keys)
    if only_tuned:
        say(f"note: {only_tuned} key(s) exist only in the trained file and are ignored")

    prefix = model_prefix(base_keys)
    if prefix:
        # a checkpoint carrying more than the diffusion model (SDXL's text encoders and VAE): only the model's keys
        say(f"model weights under '{prefix}' - the rest of the file is not compared")
    keys = [k for k in sorted(base_keys & tune_keys)
            if k.startswith(prefix) and is_lora_target(k, h_base.get_slice(k).get_shape())]
    if not keys:
        raise RuntimeError("no comparable 2-D weights found — are these the same architecture?")
    say(f"{len(keys)} candidate matrices, ranks {ranks}, device {dev}")
    # a family with its own (non-kohya) LoRA names gets them, as its own trainer writes them - ACE-Step's PEFT
    # base_model.model.<path>.lora_A, which ComfyUI maps (it ignores lora_unet_ names there)
    fam = prefix_family(prefix)
    fmt = fam.lora if fam is not None and fam.lora is not None and not fam.lora.kohya else None
    if fmt is not None:
        say(f"{fam.display_name}: writing its own LoRA names ({fmt.file_prefix}<path>.{fmt.down})")

    ident = identity_layout(base_keys, prefix) if identity else None
    if identity:
        from fizgig.families.extract import IDENTITY_RANK
        if ident is None:
            say("identity mode: this model has no identity blocks (MOSS-Image, Qwen Image 2.1 have) - every block at "
                "the rank asked")
        else:
            say(f"identity mode: {ident[0]} blocks {min(ident[1])}-{max(ident[1])} at {IDENTITY_RANK}x the rank")

    def times(k):
        m = ident[2].match(k) if ident else None
        return IDENTITY_RANK if m and int(m.group(1)) in ident[1] else 1

    sd = {r: {} for r in ranks}
    changed = skipped = 0
    total_norm = 0.0

    for i, k in enumerate(keys):
        if should_stop is not None and should_stop():
            say("cancelled")
            return []
        b = dense_weight(h_base, k, base_keys, dev)
        d = dense_weight(h_tune, k, tune_keys, dev) - b
        del b
        nrm = d.norm().item()
        if nrm < min_norm:
            skipped += 1
            del d
            if progress:
                progress(i + 1, len(keys), k)
            continue
        total_norm += nrm
        if fmt is not None:
            lk = fmt.file_prefix + k[len(prefix):-len(".weight")]
            dn, upn, ak = f"{lk}.{fmt.down}.weight", f"{lk}.{fmt.up}.weight", fmt.alpha_key.format(prefix=lk)
        else:
            lk = lora_key(k, prefix)
            dn, upn, ak = f"{lk}.lora_down.weight", f"{lk}.lora_up.weight", f"{lk}.alpha"
        x = times(k)
        facs = factor_multi(d, [r * x for r in ranks])
        for r in ranks:
            up, down = facs[r * x]
            sd[r][upn] = up
            sd[r][dn] = down
            # alpha == rank -> scale 1.0, so up @ down reproduces the delta as-is.
            sd[r][ak] = torch.tensor(float(up.shape[1]))
        changed += 1
        del d
        if progress:
            progress(i + 1, len(keys), k)

    if changed == 0:
        raise RuntimeError("the two checkpoints are identical (no weight moved) — nothing to extract")
    say(f"{changed} matrices changed, {skipped} unchanged, mean |delta| {total_norm / changed:.3f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    written = []
    for r in ranks:
        path = os.path.join(output_dir, f"{name}_{stamp}_r{r}{'_id' if ident else ''}.safetensors")
        meta = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(r),
            "ss_network_alpha": str(float(r)),
            "fizgig_extraction": "checkpoint_diff_svd",
            "fizgig_extraction_rank": str(r),
            "fizgig_extraction_rank_set": ",".join(str(x) for x in ranks),
            "fizgig_source_base": os.path.basename(base_path),
            "fizgig_source_tuned": os.path.basename(tuned_path),
            "fizgig_source_layers": str(changed),
        }
        if ident:
            from fizgig.families.extract import IDENTITY_RANK
            meta["fizgig_extraction_identity"] = (f"{ident[0]} blocks {min(ident[1])}-{max(ident[1])} at rank "
                                                  f"{r * IDENTITY_RANK}")
        if fam is not None and getattr(fam, "modelspec_arch", ""):
            meta["modelspec.architecture"] = fam.modelspec_arch
        save_file(sd[r], path, metadata=meta)
        written.append(path)
        say(f"saved r{r}: {os.path.basename(path)} "
            f"({os.path.getsize(path) / (1024 ** 2):.0f} MB)")

    say(f"done in {time.time() - t0:.0f}s")
    return written
