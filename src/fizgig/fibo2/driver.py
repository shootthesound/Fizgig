"""Fibo 2.0 driver for Fizgig's standard layer (families/driver.py).

* checkpoint: the upstream diffusers repo (models--briaai--Fibo-2.0-dev/transformer/), a directory of
  sharded bf16 safetensors plus index.json - the keys are already the diffusers names the block map uses,
  so load_dit() takes them as they are.
* export_lora_state_dict(): the inverse — the trained diffusers LoRA re-fused to ComfyUI's layout
  (attention.qkv / cross_attention.to_kv / self_attention.to_kv / ffn.w13, w2 padded back to 9728) so the LoRA
  loads in ComfyUI over the checkpoint.
* conditioning: the text encoder's prompt output -> {"tokens": (6, L, 7680), "mask": (L,)} per caption, the six
  bundles of selected Qwen3-VL layers the pipeline stacks (the Fibo-2.0 pipeline).
* latents: the Fibo 2.0 autoencoder (the FLUX.2 one) -> (32, H/8, W/8), normalized with the VAE's BatchNorm
  over the 2x2-packed latents, exactly as the pipeline's prepare_image_latents does it; the transformer packs
  the 2x2 itself (x_embedder is Linear(4*32, dim)).
* training: flow matching, x_t = sigma*x0 + (1-sigma)*noise, target = noise - x0.
"""
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

logger = logging.getLogger(__name__)


def _is_upstream_repo(path: str) -> bool:
    """The upstream diffusers repo: a directory of shards plus index.json, or its index.json."""
    import os
    return os.path.isdir(path) or path.endswith(".json")


def _sdpa_log(n, q, k, v, mask):
    """The SDPA dtype diagnostic, emitted through a PLAIN FUNCTION so torch.compile can reorder it:
    torch._dynamo.config.reorderable_logging_functions only accepts FunctionType/BuiltinFunctionType objects
    (a logging.Logger bound method is a MethodType, so it never matches and dynamo aborts the graph instead),
    while a registered function becomes a DebuggingVariable whose call is deferred to run after the compiled
    region - so the dtype data still lands for compiled blocks. The logger method itself is additionally
    registered in ignore_logger_methods as the fallback (dynamo then skips any Logger call inside a traced
    region instead of aborting), and torch._dynamo.config.ignore_logger_methods is consulted for both the
    bound method and the unbound function, so either registration form works there."""
    logger.info("Fibo2 sdpa[%d]: q=%s k=%s v=%s mask=%s", n, q, k, v, mask)


N_BLOCKS = 44
N_REFINER = 2
N_PERCEIVER = 2
INJECTION = (6, 14, 22, 30, 38)
CAP_FEAT = 7680          # text/context feature dim
HIDDEN = 9536            # the diffusers model's own FFN hidden (what the live Linears train against)
HIDDEN_FUSED = 9728      # ComfyUI FFN hidden (zero-padded) - only the export pads back to it

# The reference trainer's own timestep recipe (bria-reference-lora-train.py sample_sigma): a uniform draw 10% of
# the steps, otherwise sigmoid(N(shift, 1)) with the shift rising from 0.5 to 1.15 across the token-count span -
# the same base/max shift the shipped pipeline's scheduler carries (pipeline_bria_fibo2.py calculate_shift).
# sigma is the NOISE level: x_t = sigma*noise + (1-sigma)*x0, and the model takes sigma itself as its timestep.
T_UNIFORM = 0.10
SHIFT_BASE, SHIFT_MAX, SEQ_BASE, SEQ_MAX = 0.5, 1.15, 256, 4096


def _t_cdf(x, mu):
    """P(sigma <= x) for sigma = sigmoid(Z) with Z ~ N(mu, 1) - the recipe's own shape, unit width."""
    return 0.5 * math.erfc(-((math.log(x / (1.0 - x)) - mu)) / math.sqrt(2.0))


def mu_for_clean_share(pct):
    """The Z mean that puts `pct`% of training steps below sigma 0.5 (the clean end), the 10% uniform included; the
    curve keeps the reference's shape and width. The model's own centre is 0.5 to 1.15 by token count, which lands
    the share at about 33% (256 tokens) down to 16% (4096 tokens) - the measured figures are in the pull request."""
    want = (float(pct) / 100.0 - T_UNIFORM * 0.5) / (1.0 - T_UNIFORM)
    want = min(max(want, 1e-4), 1.0 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(80):                       # the share rises with mu: bisect
        mid = 0.5 * (lo + hi)
        if _t_cdf(0.5, mid) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


DIM = 3584               # hidden size: each part's row count inside a fused attention tensor
# ComfyUI -> diffusers attention qkv split (rows), FFN w13 split (rows), cross-attn to_kv split (rows)
_QKV_ROWS = (3584, 3584, 3584)
_W13_ROWS = (9728, 9728)
_TOKV_ROWS = (3584, 3584)


def _split_rows(t, rows):
    """Split a (sum(rows), out) weight along dim 0 into the given row counts."""
    return [p.contiguous() for p in torch.split(t, list(rows), dim=0)]


# The ComfyUI fused layout -> the diffusers module names the block map uses, for load_dit's fused branch. Each rule
# matches on the fused tensor's own name (qkv, to_kv, w13 - always inside an attention or an ffn module, so the
# parent prefix the parts keep comes from the key itself) and names the diffusers leaves it splits into, with the
# row counts that place the parts. The FFN halves are stored zero-padded to 9728 (ComfyUI pads to a 128-multiple
# for its own kernels), so w13's parts are trimmed back to the model's real 9536; w2 is a trim of dim 1 rather than
# a split. A tensor that matches no rule is already in diffusers names and passes through unchanged - which is why
# one loader serves the fused build, a sharded diffusers single file, and any future repack of either.
_FUSED_SPLITS = (
    ("qkv.weight", ("to_q.weight", "to_k.weight", "to_v.weight"), _QKV_ROWS, None),
    ("to_kv.weight", ("to_k.weight", "to_v.weight"), _TOKV_ROWS, None),
    ("w13.weight", ("w1.weight", "w3.weight"), _W13_ROWS, HIDDEN),
)
_FUSED_TRIMS = (("w2.weight", 1, HIDDEN),)                  # (3584, 9728) -> (3584, 9536): drop the zero pad
_FUSED_RENAMES = (("attention.out.weight", "attention.to_out.0.weight"),
                  ("attention.out.bias", "attention.to_out.0.bias"),
                  ("to_out.weight", "to_out.0.weight"),
                  ("to_out.bias", "to_out.0.bias"),
                  (".q_norm.", ".norm_q."),
                  (".k_norm.", ".norm_k."))



def _self_check(fused):
    """Assert the exported LoRA is consistent for ComfyUI's LoRAAdapter (comfy/weight_adapter/lora.py), which
    computes (up @ down).reshape(weight.shape) * (alpha / down.shape[0]): every down needs a matching up with
    the same rank (so the fused pair must be matmul-compatible and land on the base tensor's shape), and the
    fused alpha must ride along or ComfyUI would scale the delta by a bare 1.0. Also asserts every stem is in
    the kohya form the loader registers - a bare dotted stem resolves to nothing."""
    missing_alpha = 0
    for k, v in fused.items():
        if not k.endswith(".lora_down.weight"):
            continue
        stem = k[: -len(".lora_down.weight")]
        u = fused.get(f"{stem}.lora_up.weight")
        assert u is not None, f"Fibo2 export self-check: {stem} has a lora_down with no lora_up"
        assert u.shape[1] == v.shape[0], (
            f"Fibo2 export self-check: {stem} rank mismatch: down {tuple(v.shape)} up {tuple(u.shape)}")
        if f"{stem}.alpha" not in fused:
            missing_alpha += 1
    if missing_alpha:
        logger.warning("Fibo2 export: %d modules carry no .alpha; ComfyUI would scale them by 1.0 "
                       "instead of alpha/rank", missing_alpha)


class Fibo2Driver(FamilyDriver):
    """The Fibo 2.0 family driver. The block map declares every Linear of the transformer (all 415, the I/O group
    included), so families/quant.py's runtime INT8/NF4 pass wraps the whole model: an INT8 base keeps its weights
    in .weight.data (the compute dtype therefore resolves past them in predict), an NF4 base keeps bf16 stubs.
    Decisions on the optional hooks:
      * block swap: fibo2/model.py's Fibo2DiT streams the 44 main layers through Fizgig's shared offloader, with
        the 2 noise refiners and the 2 perceiver layers resident. diffusers' base class has no swap hooks and the
        stream calls have to sit inside the block loop, so its forward is forked (see the module docstring).
      * int8_fp32_scales: left at the default True - Fibo 2.0 has no original INT8 trainer whose scale layout
        must be reproduced, so the more accurate fp32 quantisation math applies.
      * loads_quantized: not set - every Fibo 2.0 checkpoint is bf16 on disk; a pre-quantised repack would need
        the flag (or a load_planned) before the shared layer would leave the file alone.
    """

    _te_processor = None        # the Qwen3-VL processor, resolved when the encoder is loaded

    def _log_memory(self, stage, device):
        """One line per component load/unload: what the device holds now, the high-water mark it has reached since
        the last peak reset, the allocator's reservation, and the host's peak RSS. The memory-peak evidence (todo
        17's measurement), logged from inside a real run so the log itself is the measurement: a paste of the
        [memory] lines is what the figures need, with no separate profiler pass and no idle GPU handoff.
        """
        try:
            parts = []
            dev = str(device) if device is not None else ""
            if dev.startswith("cuda") and torch.cuda.is_available():
                gb = 2 ** 30
                d = torch.device(dev)
                # the planner's basis is the reserved peak (families/quant.py reads a family's figures as reserved
                # peaks at the run's megapixels), so it is logged beside the current reservation
                parts.append(f"gpu alloc {torch.cuda.memory_allocated(d) / gb:.1f} GB, "
                             f"peak reserved {torch.cuda.max_memory_reserved(d) / gb:.1f} GB, "
                             f"reserved {torch.cuda.memory_reserved(d) / gb:.1f} GB")
            import resource
            parts.append(f"host peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.1f} GB")
            logger.info("[memory] %s: %s", stage, "; ".join(parts))
        except Exception as e:                       # a diagnostic never breaks a run
            logger.debug("[memory] %s: unavailable (%s)", stage, e)

    def load_dit(self, path, device):
        from accelerate import init_empty_weights
        from fizgig.fibo2.model import Fibo2DiT

        # a bare model.safetensors carries no config.json, so the architecture is built from its constructor
        # defaults on the meta device (no allocation at all) and the real tensors are assigned into it
        with init_empty_weights():
            model = Fibo2DiT(
                in_channels=32, patch_size=2, dim=3584, n_heads=28, cap_feat_dim=7680,
                n_layers=44, n_refiner_layers=2, perceiver_num_layers=2,
                injection_layer_ids=(6, 14, 22, 30, 38), max_num_gist_tokens=512,
            )
        d = self._component_path(path, "transformer")
        if os.path.isdir(d):
            # the upstream diffusers repo (a directory of shards plus its index.json, or the index.json itself):
            # its keys are the diffusers names the block map uses, so nothing is remapped - the checkpoint is the
            # model.
            sd = self._load_upstream(path, device)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            logger.info("Fibo2 load_dit: %d tensors, missing=%d unexpected=%d", len(sd), len(missing), len(unexpected))
            if missing or unexpected:
                logger.warning("load_dit missing=%s unexpected=%s", missing[:10], unexpected[:10])
        elif os.path.isfile(d):
            # a single-file checkpoint: ComfyUI's fused build (or a diffusers repack of the sharded repo) - the
            # tensors stream through the layout normalization into the meta model one at a time
            self._load_fused(d, model, device)
        else:
            raise ValueError(f"Fibo2 load_dit: no checkpoint at {path} - expected the upstream repo (its "
                             f"transformer/ dir, a shard or the index.json) or a single-file build")
        self._log_memory("after load_dit", device)
        return model.eval().requires_grad_(False)

    # ---- block swap (the family's Fibo2DiT, Fizgig's shared offloader) ------------------------------------------
    def max_blocks_to_swap(self, dit=None):
        """The 44 main layers may stream; the 4 refiner/perceiver blocks stay resident."""
        return (len(dit.layers) if dit is not None else N_BLOCKS) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    @staticmethod
    def _weights_file(d, orig):
        """The weights file inside a component dir: the diffusers single-file names, else the first shard named
        in an index.json, else `orig` (a layout the loader below judges).
        """
        for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
        if os.path.isdir(d):
            idx = sorted(f for f in os.listdir(d) if f.endswith(".safetensors.index.json"))
            if idx:
                import json
                with open(os.path.join(d, idx[0]), encoding="utf-8") as fh:
                    shards = sorted(set(json.load(fh)["weight_map"].values()))
                if shards:
                    return os.path.join(d, shards[0])
        return orig

    @staticmethod
    def _component_path(path, sub, want="dir"):
        """One component of the upstream repo, from whatever path the user gave: the parent repo dir (steps into
        `sub`), the component dir itself, or a file inside either. A weights file that stands alone - a fetched
        single-file build in the models folder, with no config.json or shard index beside it - is its own
        component and comes back as itself. `want="file"` answers with the component's weights file when the dir
        carries one, or the first shard named in its index; the text encoder's loader takes a file. Nothing here
        raises: a path that resolves to nothing the loader can use comes back as it went in, and the loader says
        so.
        """
        if not path:
            return path
        if os.path.isdir(path):
            inner = os.path.join(path, sub)
            d = inner if os.path.isdir(inner) else path
            return d if want == "dir" else Fibo2Driver._weights_file(d, d)
        # a file: an index names the shard set around it; a weights file with a config.json or an index beside it
        # is part of a repo component (the dir is the loader's unit); anything else is a standalone file.
        dn = os.path.dirname(path) or "."
        if path.endswith(".json"):
            return dn if want == "dir" else Fibo2Driver._weights_file(dn, path)
        if os.path.isfile(os.path.join(dn, "config.json")) or \
           any(f.endswith(".safetensors.index.json") for f in os.listdir(dn)):
            return dn if want == "dir" else path
        return path

    def _load_upstream(self, path, device):
        """The upstream diffusers repo: the parent repo dir (which carries transformer/), the shard directory
        itself, its index.json, or one of its shard files - every shape lands on the same shard set.

        The index is trusted when present; without one the headers are read (cheap - a header is a few tens
        of KB regardless of shard size). Keys are the diffusers names the block map uses, so nothing is remapped.
        """
        d = self._component_path(path, "transformer")
        if not os.path.isdir(d):
            raise ValueError(f"Fibo2 load_dit: expected the upstream repo - its transformer/ dir, one of its "
                             f"shards or the index.json - got {path}")
        from fizgig.krea2.safetensors_utils import ShardedSafeOpen
        sharded = ShardedSafeOpen(d)
        sd = {k: sharded.get_tensor(k).to(device) for k in sharded.keys()}
        return sd

    def _unfuse(self, k, t):
        """One tensor of a single-file checkpoint -> (name, tensor) pairs in the diffusers layout the model
        expects: the fused tensors split along dim 0 into the parts their rule names (w13's halves are trimmed
        back from the 9728 padding to the model's own 9536), w2 is a trim of dim 1, the renames cover the output
        projections and the q/k norms, and anything else is already in diffusers names and passes through.
        """
        for leaf, parts, rows, keep in _FUSED_SPLITS:
            if k.endswith(leaf):
                b = k[: -len(leaf)]
                return [(b + n, p[:keep].contiguous() if keep else p)
                        for n, p in zip(parts, _split_rows(t, rows))]
        for leaf, dim, keep in _FUSED_TRIMS:
            if k.endswith(leaf):
                return [(k, t.narrow(dim, 0, keep).contiguous())]
        for a, b in _FUSED_RENAMES:
            if a.startswith("."):
                if a in k:
                    return [(k.replace(a, b), t)]
            elif k.endswith(a):
                return [(k[: -len(a)] + b, t)]
        return [(k, t)]

    @staticmethod
    def _assign(model, name, tensor):
        """Put one real tensor into the meta model under its diffusers name - replace the placeholder's slot with
        the tensor itself, exactly as load_state_dict's assign=True does: the incoming dtype and device stand
        (a meta placeholder is a shape and a dtype with no storage, so its set_data refuses a real tensor and the
        slot has to go instead), and the parameter keeps the placeholder's requires_grad. False when the model
        has no slot by that name, or the tensor does not fit it - the caller counts that as unexpected.
        """
        parent, _, leaf = name.rpartition(".")
        m = model.get_submodule(parent) if parent else model
        if leaf in m._parameters:
            q = m.get_parameter(leaf)
            if tuple(q.shape) != tuple(tensor.shape):
                return False          # the model rejects a tensor that does not fit its own architecture
            tensor.requires_grad_(q.requires_grad)
            m._parameters[leaf] = tensor
            return True
        if leaf in m._buffers:
            m._buffers[leaf] = tensor
            return True
        return False

    def _load_fused(self, path, model, device):
        """A single-file checkpoint assigned into the meta model tensor by tensor: the file's tensors stream
        through the layout normalization straight into the model's own parameters and onto the target device, so
        the host never holds the whole 16 GB checkpoint and the allocator's high-water mark is one tensor. The
        sharded repo reaches the same model by the branch above; both are the same weights, and the export
        re-fuses on the way out."""
        from safetensors import safe_open
        tgt = torch.device(str(device))
        filled, unknown = set(), []
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                for name, part in self._unfuse(k, f.get_tensor(k)):
                    if self._assign(model, name, part.to(tgt)):
                        filled.add(name)
                    else:
                        unknown.append(name)
        want = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
        missing = sorted(want - filled)
        logger.info("Fibo2 load_dit: %d tensors from %s (fused single file), missing=%d unexpected=%d",
                    len(filled), os.path.basename(path), len(missing), len(unknown))
        if missing or unknown:
            logger.warning("load_dit missing=%s unexpected=%s", missing[:10], unknown[:10])

    def load_vae(self, path, device):
        from fizgig.flux3.autoencoder import load_flux3_vae
        # the convention for a FLUX.2-family autoencoder (as MOSS-Image, Klein and Flux 3 load it): the shared
        # shape-built loader, which takes the weights file directly - a repo component dir, a file inside it,
        # or a fetched standalone file all reach the same tensors
        vae = load_flux3_vae(self._component_path(path, "vae", want="file"), device=device)
        self._log_memory("after load_vae", device)
        return vae

    # ---- the VAE round trip: the FLUX.2 autoencoder normalizes its latents with a BatchNorm over the 2x2-packed
    # ---- form, so both directions pack, touch the statistics, and unpack again (as the pipeline itself does)
    @staticmethod
    def _pack_latents(z):
        b, c, h, w = z.shape
        return z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpack_latents(z):
        b, c, h, w = z.shape
        return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)

    @torch.no_grad()
    def encode_images(self, vae, images):
        """uint8 (H, W, 3) arrays -> (32, H/8, W/8) latents, one per image, float32 on CPU.

        One image per forward, as the pipeline encodes them: the bucketed sizes can differ across a batch, and the
        BatchNorm statistics are per-channel, so a caption's latent must not depend on which images shared its batch.
        The shared autoencoder applies the pack and the normalization inside encode(); the latents are handed back
        unpacked, in the (32, H/8, W/8) layout this family's caches hold."""
        p = next(vae.parameters())
        out = []
        for a in images:
            x = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float().div(127.5).sub(1.0)
            x = x.contiguous().to(p.device, p.dtype)
            tokens, ids = vae.encode(x)
            h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
            z = tokens.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)          # (1, 128, h, w): packed, normalized
            out.append(self._unpack_latents(z)[0].float().cpu())                # (32, 2h, 2w): the cache layout
        return out

    def load_text_encoder(self, path, device):
        """Fibo 2.0 reads its text through Qwen3-VL (the same encoder Krea 2 and FLUX.3 use), loaded whole onto the
        target device, as the Fibo 2.0 pipeline loads it: encoding runs the full model with output_hidden_states=True
        and takes the selected layers' hidden states, so every part the forward needs must stay callable. The
        processor is resolved cache-first so a fully cached install runs offline."""
        from transformers import AutoProcessor
        from fizgig.krea2.embedder import (QWEN3_VL_4B_INSTRUCT_REPO_ID, _load_qwen3_vl_model,
                                           _resolve_tokenizer_source)
        from fizgig.utils.hf_cache import from_pretrained_cache_first
        # the parent repo dir, the text_encoder/ dir itself or a file inside either all resolve to the weights
        # file this encoder's loader takes (a bare dir reaches is_prequantized_fp8 as IsADirectoryError otherwise)
        path = self._component_path(path, "text_encoder", want="file")
        # the weights go straight onto the target device: staging a 4B encoder through CPU and copying it over
        # afterwards is what made the load crawl. lm_head stays: the Fibo 2.0 pipeline itself runs the full
        # model with output_hidden_states=True, so the forward needs it to be callable.
        model = _load_qwen3_vl_model(path, dtype=torch.bfloat16, device=device)
        self._te_processor = from_pretrained_cache_first(
            AutoProcessor, _resolve_tokenizer_source(path, QWEN3_VL_4B_INSTRUCT_REPO_ID))
        self._log_memory("after load_text_encoder", device)
        return model.to(device).eval().requires_grad_(False)

    def unload_text_encoder(self, te) -> None:
        device = te.device
        te.to("cpu")
        del te
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_memory("after unload_text_encoder (TE weights freed)", device)

    def encode_text(self, te, captions):
        """One conditioning dict per caption: the transformer's six text bundles as tokens (6, L, 7680) plus the
        mask (L,), on CPU. Built exactly as the Fibo 2.0 pipeline builds them - every selected layer scaled to
        unit RMS per token in float32, its three layers concatenated along the feature dim into one 7680-wide
        bundle, and the six bundles stacked behind the Perceiver's (bundle 0) and the injection blocks' (1..5)."""
        from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import PROMPT_TEMPLATE, TEXT_ENCODER_OUT_LAYERS
        proc = self._te_processor
        device = te.device
        used = sorted({i for group in TEXT_ENCODER_OUT_LAYERS for i in group})
        out = []
        for cap in captions:
            text = cap if isinstance(cap, str) else cap.get("text", "")
            enc = proc(text=[PROMPT_TEMPLATE.format(text)], images=None, padding="longest",
                       max_length=4096, truncation=True, return_tensors="pt").to(device)
            states = te(**enc, output_hidden_states=True).hidden_states
            norm = {}
            for i in used:
                s = states[i].float()
                norm[i] = (s * torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(te.dtype)
            bundles = torch.stack([torch.cat([norm[i] for i in group], dim=-1)
                                   for group in TEXT_ENCODER_OUT_LAYERS], dim=1)
            # one conditioning dict per caption: the transformer's encoder_hidden_states is (batch, 6, text_len,
            # cap_feat_dim), so the six bundles get the batch axis and the mask is (batch, text_len)
            out.append({"tokens": bundles[0].unsqueeze(0).cpu(), "mask": enc.attention_mask.bool().cpu()})
        return out

    def encode_text_with_references(self, te, captions, references):
        # edit_training is False for Fibo 2.0: reference images reach the encoder's vision path only when an
        # edit recipe is wired up, so until then the caption-only path is the same encoding
        return self.encode_text(te, captions)

    # ---- temporary diagnostics ---------------------------------------------------------------------------
    # The INT8 precision run crashed inside the DiT's scaled_dot_product_attention with mismatched dtypes
    # ("expected scalar type bfloat16 but got float"). These logs pin down, without guessing: what the shared
    # quant layer (families/quant.py) did to the base, what dtype predict() resolves and casts with, and the
    # exact q/k/v/mask dtypes every SDPA call receives. Remove once the crash is explained.

    _DIAG_PROBES = ("x_embedder", "t_embedder.mlp.0", "t_embedder.mlp.2",
                    "layers.0.attention.to_q", "layers.0.attention.to_k", "layers.0.attention.to_v",
                    "layers.0.attention.to_out.0", "layers.0.adaLN_modulation.0",
                    "layers.6.cross_attention.to_q", "layers.6.cross_attention.to_k",
                    "layers.6.cross_attention.to_v", "layers.6.cross_attention.to_out.0",
                    "noise_refiner.0.attention.to_v", "noise_refiner.0.adaLN_modulation.0",
                    "perceiver.layers.0.self_attention.to_v", "perceiver.layers.0.cross_attention.to_v",
                    "final_layer.linear", "final_layer.adaLN_modulation.1")
    _DIAG_SDPA_LIMIT = 40
    _INT8_MIN_TOKENS = 2048      # first token count where the int8 kernel measures faster on this card

    def _diag_dtypes(self, dit):
        """One line: the quant flags, how many of the model's Linears are quantised to int8, the compute dtype
        predict() resolves (the first parameter that is not int8), and the dtype of every probed Linear (an int8 or
        NF4 base shows a zero-element bf16 stub; the flag is the module's own _is_int8 / _is_nf4 marker)."""
        named = list(dit.named_parameters())
        mods = dict(dit.named_modules())
        # The count comes from the module flags, not from parameter dtypes: since the torch.compile fix the int8
        # bytes sit in a non-persistent buffer (modules/int8_train.py), so a quantised base carries no int8
        # parameter at all and a dtype count reports 0 while every block Linear is quantised.
        n_int8 = sum(1 for m in mods.values() if getattr(m, "_is_int8", False))
        n_lin = sum(1 for m in mods.values() if isinstance(m, torch.nn.Linear))
        parts = [f"int8={getattr(dit, '_int8_quantized', False)}",
                 f"nf4={getattr(dit, '_nf4_quantized', False)}",
                 f"n_int8={n_int8}/{n_lin}"]
        if named:
            c0, p0 = next((p for p in named if p[1].dtype != torch.int8), named[0])
            parts.append(f"compute={c0}:{str(p0.dtype).replace('torch.', '')}@{p0.device.type}")
        for probe in self._DIAG_PROBES:
            m = mods.get(probe)
            if m is None or not isinstance(m, torch.nn.Linear):
                continue
            flag = "i8" if getattr(m, "_is_int8", False) else ("nf4" if getattr(m, "_is_nf4", False) else "--")
            parts.append(f"{probe}:{str(m.weight.dtype).replace('torch.', '')}({flag})")
        return "  ".join(parts)

    def _install_sdpa_diag(self):
        """Wrap the fibo2 transformer module's dispatch_attention_fn so every SDPA call logs the exact q/k/v/mask
        dtypes it receives (the crash site). Installed once; logs the first _DIAG_SDPA_LIMIT calls.
        A raw logger.info inside a torch.compile region aborts the graph (dynamo: "logging.Logger method not
        supported for non-export cases"), so the logger method is first registered with
        torch._dynamo.config.reorderable_logging_functions - dynamo then reorders the side effect to run after
        the compiled region, so the dtype data still lands even for compiled blocks. If that config is not
        available, the call falls back to a torch.compiler.is_compiling() guard: the compiled graph omits the
        log (no abort) and eager still logs."""
        if getattr(self, "_sdpa_diag_installed", False):
            return
        self._sdpa_diag_installed = True
        self._sdpa_diag_count = 0
        self._sdpa_coerced = False
        from diffusers.models.transformers import transformer_bria_fibo2 as _tbf
        self._sdpa_diag_reorderable = False
        try:
            import torch._dynamo as _dynamo
            cfg = getattr(_dynamo.config, "reorderable_logging_functions", None)
            if isinstance(cfg, set):
                cfg.discard(logger.info)   # a bound Logger method can never be reorderable - drop the stale entry
                cfg.add(_sdpa_log)
                self._sdpa_diag_reorderable = True
            ign = getattr(_dynamo.config, "ignore_logger_methods", None)
            if isinstance(ign, set):
                ign.add(logger.info)
            logger.info("Fibo2 sdpa diag: _sdpa_log registered as dynamo-reorderable (the dtype logs run "
                        "after each compiled region); logger.info also in ignore_logger_methods as the "
                        "fallback (no abort, no data from compiled blocks)")
        except Exception:
            logger.info("Fibo2 sdpa diag: dynamo logging configs unavailable, falling back to the "
                        "is_compiling() guard (eager logs only)")
        orig = _tbf.dispatch_attention_fn
        from fizgig.modules import int8_attention as _i8a

        def _logged(query, key, value, *a, **kw):
            eager = not torch.compiler.is_compiling()
            if eager and self._sdpa_diag_count < self._DIAG_SDPA_LIMIT:
                self._sdpa_diag_count += 1
                n = self._sdpa_diag_count
                def d(t):
                    return "None" if t is None else str(t.dtype).replace("torch.", "")
                _sdpa_log(n, d(query), d(key), d(value), d(kw.get("attn_mask")))
            # int8 keeps a stray fp32 island in the value path (t_embedder -> adaLN -> modulation -> to_v);
            # SDPA rejects mixed q/k/v dtypes outright. Coerce everything to the attention input's own dtype
            # so the forward runs - if it does, the fp32 value was the whole problem and nothing else is.
            if query.dtype != key.dtype or query.dtype != value.dtype:
                if eager and not self._sdpa_coerced:
                    self._sdpa_coerced = True
                    logger.info("Fibo2 sdpa: coercing mixed q/k/v dtypes to query's %s (stray fp32 island)",
                                str(query.dtype).replace("torch.", ""))
                key = key.to(query.dtype)
                value = value.to(query.dtype)
            # The int8 attention kernel, for renders only. wanted() is a Python bool, so training - where the
            # switch is never turned on - never enters this branch at all, and attend() itself refuses a live
            # grad, a mask that excludes tokens, or a missing kernel. The 2048-token floor is this card's own
            # measurement (loss below ~1300 tokens, wins from 1664 up - table in the run sheet), so short
            # renders stay on PyTorch SDPA. diffusers hands the processor [B, S, H, D] and flattens dims 2-3
            # back after the call, while attend() works in [B, H, S, D] - hence the transposes on both sides.
            if _i8a.wanted() and key.shape[1] >= self._INT8_MIN_TOKENS:
                out = _i8a.attend(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                  kw.get("attn_mask"))
                if out is not None:
                    return out.transpose(1, 2)
            return orig(query, key, value, *a, **kw)

        _tbf.dispatch_attention_fn = _logged

    def predict(self, dit, state, cond):
        # the trainer's own loss paths call predict() and take its result apart as a tensor (they .float() it and
        # add it into guidance deltas), so the velocity comes back, not the output object. The cast to the model's
        # compute dtype is the driver's job (the trainer only moves tensors to the device): the cached latents and
        # the text bundles are fp32, the DiT runs bf16, and the reference Fibo2 recipe keeps the fp32 timestep
        # while casting the noised latents and the reference-image latents to bf16 for the forward.
        first = not getattr(self, "_shapes_logged", False)
        self._shapes_logged = True
        if first:
            def spec(v):
                if v is None:
                    return "None"
                if isinstance(v, (list, tuple)):
                    return "[" + ", ".join(spec(x) for x in v) + "]"
                if not torch.is_tensor(v):
                    return f"{type(v).__name__}:{v}"
                return f"{tuple(v.shape)}/{str(v.dtype).replace('torch.', '')}/{v.device}"
            logger.info("Fibo2 step inputs: x_t=%s  timestep=%s  noise=%s  tokens=%s  mask=%s  context=%s",
                        spec(state.get("hidden_states")), spec(state.get("timestep")), spec(state.get("noise")),
                        spec(cond.get("tokens")), spec(cond.get("mask")), spec(cond.get("context_latents")))
            logger.info("Fibo2 quant diag: %s", self._diag_dtypes(dit))
            self._install_sdpa_diag()
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        # The compute dtype is the first parameter that is not INT8. An INT8 base keeps its bytes in buffers now,
        # so this filter is a no-op for it too; it stays for a base whose driver quantises into the Parameter
        # (qwen_image21's embedder does), and an NF4 stub or unquantised base is bf16 either way.
        dtype = next(p.dtype for p in dit.parameters() if p.dtype != torch.int8)
        if first:
            # what predict casts WITH: the bf16 latents, the 7680-wide text bundles AND the timestep go to this
            # dtype - the log shows both sides of every cast the model receives. The timestep joins the compute
            # dtype (measured 25 Oct 2026): the checkpoint is uniformly BF16 (all 634 tensors, header-verified),
            # and an fp32 timestep creates an fp32 island (t_embedder -> adaLN -> modulation -> attention input
            # -> to_v) that the bf16-weighted RMSNorms then cast away from q/k, leaving value the odd dtype out
            # at the SDPA call ("query.dtype: BFloat16 key.dtype: BFloat16 and value.dtype: float").
            logger.info("Fibo2 predict dtype resolution: device=%s dtype=%s | x_t %s -> %s | tokens %s -> %s | "
                        "timestep %s -> %s",
                        device, dtype, state["hidden_states"].dtype, dtype,
                        cond["tokens"].dtype, dtype, state["timestep"].dtype, dtype)
        ctx = cond.get("context_latents")
        if ctx is not None:
            ctx = [c.to(device, dtype) for c in ctx]
        # the cached latents and the text bundles are fp32 and the DiT runs bf16: the family's conditioning stays on
        # the CPU (the cache holds it there) so the model's device must be resolved here, not only its dtype
        # the mask must stay bool - the transformer's compute_gist_count sums it over the text tokens and the sum
        # of a bool tensor is an int count, but a bf16 sum is a float that the transformer's ratio math rejects
        # the batch's extra axis: the dataset stacks the family's per-caption conditioning (1, 6, L, cap_feat_dim)
        # into (batch, 1, 6, L, cap_feat_dim) - the squeeze removes the family's per-caption axis so the DiT gets
        # (batch, 6, L, cap_feat_dim) as its contract declares (diffusers' prompt_embeds)
        tokens = cond["tokens"].squeeze(1)
        mask = cond.get("mask").squeeze(1)
        try:
            pred = dit(state["hidden_states"].to(device, dtype), state["timestep"].to(dtype),
                       tokens.to(device, dtype), mask.to(device), ctx).sample
        except Exception:
            logger.exception("Fibo2 predict crashed")
            try:
                logger.info("Fibo2 quant state at crash: %s", self._diag_dtypes(dit))
            except Exception as diag_exc:
                logger.info("Fibo2 quant state at crash: (diag failed: %s)", diag_exc)
            raise
        if first:
            logger.info("Fibo2 step output: pred=%s/%s", tuple(pred.shape), str(pred.dtype).replace("torch.", ""))
        return pred

    @staticmethod
    def _as4(latents):
        """The model wants (B, C, h, w). A cache entry written as a per-item (1, C, h, w) latent collates into a 5D
        (B, 1, C, h, w) batch, so that extra singleton comes off here, whichever vintage wrote the cache."""
        return latents.squeeze(1) if latents.ndim == 5 else latents

    @staticmethod
    def _grid(width, height):
        """The family's own bucketing: the model requires both sides to be a multiple of 16 (the VAE's 8x and the
        transformer's patch 2x)."""
        align = 16
        return ((width + align - 1) // align * align, (height + align - 1) // align * align)

    @staticmethod
    def _text(cond):
        """The family's conditioning: its cached {"tokens", "mask"} dicts - the six text bundles and the mask, as the
        pipeline's encode_prompt builds them."""
        return cond["tokens"], cond["mask"]

    def _mu_for(self, image_seq_len):
        """The shift the pipeline's scheduler applies to an image's token count - the same base/max shift the shipped
        pipeline's scheduler carries (calculate_shift with 256 / 4096 / 0.5 / 1.15, the reference trainer's own
        centre).

        The model's own token count is the latent's token count at the transformer's patch size: the 2x2 packed
        latent's (H/16, W/16) tokens."""
        m = (SHIFT_MAX - SHIFT_BASE) / (SEQ_MAX - SEQ_BASE)
        b = SHIFT_BASE - m * SEQ_BASE
        return image_seq_len * m + b

    def _schedule(self, steps, latents, sigmas=None, options=()):
        """The pipeline's sigma schedule for this size, with the terminal sigma 0 appended: an explicit schedule
        (a speed LoRA's SamplingSettings) is shifted the same way; otherwise sigmas from 1 down to 1 / steps through
        the exponential time shift.

        The schedule's sigmas are the noise levels the model takes as its timestep (sigma in [0, 1]), and each step
        moves the latent along the Euler direction by (next_sigma - sigma): the same math the pipeline's scheduler
        runs between its shifted sigmas."""
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / steps, steps)
        else:
            sigmas = np.array(sigmas, dtype=np.float32)
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = self._mu_for(image_seq_len)
        exp_mu = math.exp(mu)
        # the pipeline's scheduler: exp(mu) / (exp(mu) + (1/sigma - 1)) for each sigma, then the terminal 0
        # (sigma 0 is the terminal - the shift of 0 is 0, so it stays 0, no division)
        shifted = [float(exp_mu / (exp_mu + 1.0 / s - 1.0)) if s > 0 else 0.0 for s in sigmas]
        return shifted + [0.0]

    def _sample_sigma(self, generator, latents):
        """The reference trainer's draw, in its own order: the branch coin, then the value. One draw, used by every
        caller, and built on the model's device, since a CPU timestep tensor lands up crashing the CUDA matmul
        inside the timestep embedder."""
        n_tokens = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)   # the transformer packs the 2x2 itself
        if float(torch.rand(1, generator=generator)) < T_UNIFORM:
            return float(torch.rand(1, generator=generator))
        z = float(torch.randn(1, generator=generator)) + self._t_mu(n_tokens)
        return 1.0 / (1.0 + math.exp(-z))

    def _t_mu(self, n_tokens):
        """lownoise_pct=P (the share of steps below sigma 0.5) moves the curve's centre; unset = the model's own
        token-count shift. Split, "A:N:B": A% for the first N epochs, then B%. Slide, "A:H:B:E": A% until epoch H,
        then in a straight line down to B% by epoch E (and B% after); "A:H:B:E:R": and every R epochs after E, one
        epoch at A% again."""
        pct = str(self.options.get("lownoise_pct", "") or "").replace("%", "").strip()
        if not pct:
            return SHIFT_BASE + (SHIFT_MAX - SHIFT_BASE) * (n_tokens - SEQ_BASE) / (SEQ_MAX - SEQ_BASE)
        if ":" in pct:
            try:
                parts = [float(x) for x in pct.split(":")]
                if len(parts) not in (3, 4, 5):
                    raise ValueError
            except ValueError:
                raise RuntimeError(f"Clean-end share '{pct}': give share:epochs:share (75:5:40 = 75% for the first 5 "
                                   "epochs, then 40%) or share:epoch:share:epoch (75:6:40:10 = 75% until epoch 6, "
                                   "then down to 40% by epoch 10; a fifth number R = every R epochs after that, one "
                                   "epoch at the first share)") from None
            first, until, rest = parts[:3]
            end = parts[3] if len(parts) >= 4 else until
            every = int(parts[4]) if len(parts) == 5 else 0
            e = self.epoch
            share = first if e <= until else rest if e >= end else first + (rest - first) * (e - until) / (end - until)
            if every > 0 and e > end and (e - end) % every == 0:
                share = first                     # a refresh epoch at the first share
            pct = f"{round(share, 2):g}"
        cached = getattr(self, "_mu_cached", None)
        if cached is None or cached[0] != pct:
            cached = self._mu_cached = (pct, mu_for_clean_share(pct))
            logger.info(f"Fibo 2.0: {pct}% of training steps at the clean end (sigma < 0.5) from epoch {self.epoch}, "
                        f"Z mean {cached[1]:.2f} (the model's own centre is 0.5 to 1.15 by token count)")
        return cached[1]

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        latents = self._as4(latents)
        t = self._sample_sigma(generator, latents)
        t = min_t + (max_t - min_t) * t
        noise = torch.randn(latents.shape, generator=generator).to(latents.device, latents.dtype)
        # the timestep is built in the latents' own dtype: an fp32 scalar tensor promotes the whole interpolation
        # to float32, and the noised latents then reach the model's bf16 x_embedder as Float
        sigma = torch.full((latents.shape[0],), t, device=latents.device, dtype=latents.dtype)
        x_t = (1 - sigma).view(-1, 1, 1, 1) * latents + sigma.view(-1, 1, 1, 1) * noise
        return {"hidden_states": x_t, "timestep": sigma, "noise": noise, "t": t}

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None,
                      diff_ref=None, diff_weight=0.0):
        if not getattr(self, "_step_logged", False):
            self._step_logged = True
            logger.info("Fibo2 training_loss in: latents=%s/%s refs=%s", tuple(latents.shape),
                        str(latents.dtype).replace("torch.", ""),
                        None if not refs else [tuple(r.shape) for r in refs])
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._step_count == 1 or self._step_count % 100 == 0:
            # the trainer resets the peak stats at the epoch start and again at "preview start", so the epoch's
            # training high-water mark can only be read from inside the steps themselves
            self._log_memory(f"training step {self._step_count}", next(dit.parameters()).device)
        latents = self._as4(latents)
        state = self.noise_latents(latents, generator, min_t=min_t, max_t=max_t)
        pred = self.predict(dit, state, cond)
        loss = F.mse_loss(pred.float(), (state["noise"] - latents).float())
        return loss, {"t": state["t"], "loss": float(loss.detach())}

    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        """The seed's starting noise, in the driver's latent layout (1, 32, H/8, W/8) - the form generate() draws
        and decode() accepts, on CPU float32.

        The description's latent_channels=128 / spatial_factor=16 are the VAE's packed 2x2 form; the driver's latent
        is the unpacked (32, H/8, W/8) the transformer takes, so the noise is built there, exactly as generate()
        draws it for the same seed."""
        width, height = self._grid(width, height)
        dev = "cpu"
        g = torch.Generator(device=dev).manual_seed(int(seed))
        return torch.randn(1, 32, height // 8, width // 8, device=dev, dtype=torch.float32, generator=g)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None,
                 options=(), noise=None, on_step=None, refs=None, frames=1, audio=False, **kw):
        """Denoise one image, honouring the base's sampling contract: the device past the meta params, the gridded
        size, the family's own {"tokens", "mask"} conditioning, guidance only when cfg > 1.0 with a negative, the
        schedule from a speed LoRA's SamplingSettings (or the pipeline's, shifted by the image size), the seed's
        starting noise (or `noise`), on_step(done, total) before every step (it may raise to abort), refs as the
        perceiver's context latents, and latents in the driver's layout for decode.

        A stills family: frames / audio are accepted on the contract's signature and ignored - the family renders
        one still. The workbench's extras (cond_b, prompt_blend, source, refer, repaint) are its own surfaces and
        leave the driver's stills render alone."""
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        width, height = self._grid(width, height)
        tokens, mask = self._text(cond)
        guided = cfg > 1.0 and neg_cond is not None
        neg_tokens, neg_mask = (self._text(neg_cond) if guided else (None, None))
        x = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device, torch.bfloat16)
        if refs is not None:
            # the perceiver's context latents, as the pipeline's context_latents (edit references)
            cond = dict(cond)
            cond["context_latents"] = [r.to(torch.bfloat16) for r in refs]
        schedule = self._schedule(steps, x, sigmas=sigmas, options=options)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i, (s_cur, s_next) in enumerate(zip(schedule[:-1], schedule[1:])):
                if on_step is not None:
                    on_step(i, len(schedule) - 1)
                t = torch.full((x.shape[0],), float(s_cur), device=device, dtype=torch.bfloat16)
                pred = self.predict(dit, {"hidden_states": x, "timestep": t, "noise": None, "t": s_cur}, cond)
                if guided:
                    uncond = self.predict(dit, {"hidden_states": x, "timestep": t, "noise": None, "t": s_cur},
                                          {"tokens": neg_tokens, "mask": neg_mask})
                    pred = uncond + cfg * (pred - uncond)
                x = x + (s_next - s_cur) * pred
        return x

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        from fizgig.flux3.autoencoder import latent_ids
        z = latents if latents.dim() == 4 else latents.unsqueeze(0).float()
        z = self._pack_latents(z)                                    # (1, 128, h, w): packed, normalized
        h, w = z.shape[-2:]
        tokens = z.permute(0, 2, 3, 1).reshape(h * w, -1)
        p = next(vae.parameters())
        px = vae.decode(tokens.to(p.device), latent_ids(h, w, p.device)).float().cpu()   # (3, H, W): unbatched
        px = (px / 2 + 0.5).clamp(0, 1)
        return Image.fromarray((px.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    def compile_targets(self, dit):
        """The 44 transformer blocks torch.compile replaces in place (the main compute; the refiner and the
        perceiver are compiled by compile_blocks below).

        The model's forward calls every block of `layers` directly, so the compiled versions run in their place."""
        return dit.layers

    def compile_blocks(self, dit, boundary="inside", blocks_to_swap=0):
        """The three block lists (the 44 transformer blocks, the 2 noise refiners and the 2 perceiver layers)
        through the shared compile (families/compile.py) - 48 blocks in all, which is what a compiled run logs.
        The diffusers model has no context_embedder to compile: the perceiver IS its context path, so there is
        nothing missing from this list."""
        from fizgig.families.compile import compile_blocks
        for blocks in (dit.layers, dit.noise_refiner, dit.perceiver.layers):
            compile_blocks(dit, blocks, blocks_to_swap, boundary=boundary,
                           fullgraph=self.description.compile_fullgraph)

    def ft_spec(self, dit):
        """The checkpoint's own file layout, declared for the readers that never load the model (the Profiler and
        Extract, through families/lorafile.fused_parts): the file fuses the attention q/k/v into one qkv tensor,
        w1/w3 into one w13 tensor and every cross/self-attention k+v into one to_kv tensor, and `load_dit` splits
        them back into the Linears the block map names. A LoRA trained on the fused file tensor therefore carries
        the fused stem, and each part of the layout says which rows of the fused up matrix belong to which model
        Linear - without it those pairs cannot be placed and are reported as outside the blocks.

        Fibo 2.0 is a LoRA-only family (`finetune=False`): the fine-tune half of the spec (the windows, the
        planner's figures) is never read, so it stays at the dataclass defaults rather than claiming measurements
        the family has not made. The layout itself is the model's, verified against `load_dit`'s splits: qkv is 3
        equal row chunks (3x3584), w13 and to_kv are 2 equal row chunks (2x9728, 2x3584).
        """
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks=("layers", "noise_refiner"),
                      components=("attention", "feed_forward", "cross_attention", "self_attention", "ffn",
                                  "adaLN_modulation"),
                      file_prefix="",
                      file_layout=(
                          ("attention.to_q.weight", "attention.qkv.weight", 0, 3),
                          ("attention.to_k.weight", "attention.qkv.weight", 1, 3),
                          ("attention.to_v.weight", "attention.qkv.weight", 2, 3),
                          ("cross_attention.to_k.weight", "cross_attention.to_kv.weight", 0, 2),
                          ("cross_attention.to_v.weight", "cross_attention.to_kv.weight", 1, 2),
                          ("self_attention.to_k.weight", "self_attention.to_kv.weight", 0, 2),
                          ("self_attention.to_v.weight", "self_attention.to_kv.weight", 1, 2),
                          ("feed_forward.w1.weight", "feed_forward.w13.weight", 0, 2),
                          ("feed_forward.w3.weight", "feed_forward.w13.weight", 1, 2),
                          ("ffn.w1.weight", "ffn.w13.weight", 0, 2),
                          ("ffn.w3.weight", "ffn.w13.weight", 1, 2),
                      ))

    def alias_flat(self, flat):
        """The checkpoint's flattened name for one of the model's Linears -> the model's flattened name. Fibo 2.0's
        ComfyUI file calls the attention output `attention.out` (and `cross_attention.to_out` /
        `self_attention.to_out` where those exist), where the diffusers module the block map names is
        `...to_out.0`; a LoRA exported on the file's names arrives with the file's stem. The fused qkv / w13 /
        to_kv are not renames and go through ft_spec's file_layout instead.
        """
        for a, b in (("attention_out", "attention_to_out_0"),
                     ("cross_attention_to_out", "cross_attention_to_out_0"),
                     ("self_attention_to_out", "self_attention_to_out_0")):
            if flat.endswith(a):
                return flat[:-len(a)] + b
        return None

    def block_map(self, dit=None):
        """Every Linear the model's own recipe adapts, in the blocks it sits in - the reference trainer adapts all
        of them (415 for Fibo 2.0 Dev), so the map is the whole trainable set and families/driver.py's default
        `lora_target_names` needs no override. Each block carries its own adaLN modulation; the injection blocks
        carry their cross-attention output; the three I/O areas (the input projection, the timestep embedding and
        the output projection) sit in their own group.

        With a `dit` the modules are filtered to the ones that model actually has, as the base does - a name that
        is not in the DiT cannot be wrapped."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if m in names] if names is not None else list(mods)

        groups = []
        for i in range(N_BLOCKS):
            pre = f"layers.{i}"
            mods = [f"{pre}.attention.to_q", f"{pre}.attention.to_k", f"{pre}.attention.to_v",
                    f"{pre}.attention.to_out.0", f"{pre}.feed_forward.w1", f"{pre}.feed_forward.w2",
                    f"{pre}.feed_forward.w3", f"{pre}.adaLN_modulation.0"]
            if i in INJECTION:
                mods += [f"{pre}.cross_attention.to_q", f"{pre}.cross_attention.to_k",
                         f"{pre}.cross_attention.to_v", f"{pre}.cross_attention.to_out.0"]
            groups.append(BlockGroup(pre, [Block(pre, f"Block {i}", keep(mods))]))
        for i in range(N_REFINER):
            pre = f"noise_refiner.{i}"
            mods = [f"{pre}.attention.to_q", f"{pre}.attention.to_k", f"{pre}.attention.to_v",
                    f"{pre}.attention.to_out.0", f"{pre}.feed_forward.w1", f"{pre}.feed_forward.w2",
                    f"{pre}.feed_forward.w3", f"{pre}.adaLN_modulation.0"]
            groups.append(BlockGroup(pre, [Block(pre, pre, keep(mods))]))
        for i in range(N_PERCEIVER):
            pre = f"perceiver.layers.{i}"
            mods = [f"{pre}.self_attention.to_q", f"{pre}.self_attention.to_k", f"{pre}.self_attention.to_v",
                    f"{pre}.self_attention.to_out.0", f"{pre}.cross_attention.to_q",
                    f"{pre}.cross_attention.to_k", f"{pre}.cross_attention.to_v",
                    f"{pre}.cross_attention.to_out.0", f"{pre}.ffn.w1", f"{pre}.ffn.w2", f"{pre}.ffn.w3"]
            groups.append(BlockGroup(pre, [Block(pre, pre, keep(mods))]))
        groups.append(BlockGroup("I/O", [
            Block("x_embedder", "Input projection", keep(["x_embedder"])),
            Block("t_embedder", "Timestep embedding", keep(["t_embedder.mlp.0", "t_embedder.mlp.2"])),
            Block("final_layer", "Output projection",
                  keep(["final_layer.linear", "final_layer.adaLN_modulation.1"])),
        ]))
        return groups

    def _module_paths(self):
        """{declared stem: dotted module path} for every module the block map names.

        The export needs it to tell its fused areas apart: flattening a path to underscores is not invertible by
        rule (to_out.0, noise_refiner.0 and perceiver.layers.0 all land on underscores), so the block map is the
        authority on which trained parts make up a fused tensor - whichever stem form the description declares."""
        idx = getattr(self, "_stem_index", None)
        if idx is None:
            fmt = self.description.lora
            idx = {fmt.stem(m): m for g in self.block_map() for b in g.blocks for m in b.modules}
            self._stem_index = idx
        return idx

    def convert_lora_state_dict(self, sd):
        """The read-side inverse of export_lora_state_dict: a file in the checkpoint's fused layout (this family's
        own exported LoRAs, or another trainer's file written against the fused base) back onto the split Linears
        the live diffusers model trains against. A split file passes through untouched; a fused tensor is split
        exactly - the shared down hands back each part's own rank rows, the block-diagonal up each part's own
        rows and columns - and the FFN's 9728 lanes un-pad to the model's own 9536. The summed alpha goes back
        as equal parts, exact for what the trainer builds (one rank and one alpha per part)."""
        if not any(any(leaf in k for leaf in ("_attention_qkv.", "_feed_forward_w13.", "_ffn_w13.",
                                              "_cross_attention_to_kv.", "_self_attention_to_kv.",
                                              "_attention_out.", "_cross_attention_to_out.",
                                              "_self_attention_to_out.", "_feed_forward_w2.", "_ffn_w2."))
                   for k in sd):
            return sd
        fmt = self.description.lora
        # flat block id -> dotted block id, for every block the map names: a fused stem's parent is the BLOCK
        # (layers_0 -> layers.0), never one of its modules - the map is the authority, as in export_lora_state_dict
        parents = {b.id.replace(".", "_"): b.id for g in self.block_map() for b in g.blocks}

        def _ab_of(k):
            for ab, suf in (("down", ".lora_down.weight"), ("up", ".lora_up.weight"), ("alpha", ".alpha")):
                if k.endswith(suf):
                    return k[: -len(suf)], ab
            return None

        # fused leaf -> the split module paths it stands for, each with its row count in the fused tensor
        _GROUPS = {
            "attention_qkv": (("attention.to_q", "attention.to_k", "attention.to_v"), _QKV_ROWS),
            "feed_forward_w13": (("feed_forward.w1", "feed_forward.w3"), _W13_ROWS),
            "ffn_w13": (("ffn.w1", "ffn.w3"), _W13_ROWS),
            "cross_attention_to_kv": (("cross_attention.to_k", "cross_attention.to_v"), _TOKV_ROWS),
            "self_attention_to_kv": (("self_attention.to_k", "self_attention.to_v"), _TOKV_ROWS),
        }
        # fused leaf -> the one split module it renames to (no tensor change, except the w2 down's un-pad)
        _RENAMES = {
            "attention_out": "attention.to_out.0",
            "cross_attention_to_out": "cross_attention.to_out.0",
            "self_attention_to_out": "self_attention.to_out.0",
            "feed_forward_w2": "feed_forward.w2",
            "ffn_w2": "ffn.w2",
        }

        _LEAVES = tuple(sorted(list(_GROUPS) + list(_RENAMES), key=len, reverse=True))
        out = {}
        for k, v in sd.items():
            r = _ab_of(k)
            if r is None:
                out[k] = v
                continue
            stem, ab = r
            flat = stem[len("lora_unet_"):] if stem.startswith("lora_unet_") else stem
            leaf = next((l for l in _LEAVES if flat.endswith("_" + l)), None)
            if leaf is None:
                out[k] = v
                continue
            par = parents.get(flat[: -(len(leaf) + 1)])
            if par is None:
                out[k] = v
                continue
            if leaf in _GROUPS:
                parts, rows = _GROUPS[leaf]
                n = len(parts)
                if ab == "alpha":                      # the summed alpha, back as equal parts
                    for p in parts:
                        out[f"{fmt.stem(f'{par}.{p}')}.alpha"] = (v.float() / n).to(v.dtype)
                    continue
                rank = v.shape[0] // n if ab == "down" else v.shape[1] // n
                if ab == "down":                       # (n*rank, d) -> one (rank, d) per part
                    for i, p in enumerate(parts):
                        out[f"{fmt.stem(f'{par}.{p}')}.lora_down.weight"] = v[i * rank:(i + 1) * rank].contiguous()
                else:                                  # blockdiag (sum(rows), n*rank) -> each part's own block
                    for i, p in enumerate(parts):
                        r0 = sum(rows[:i])
                        t = v[r0:r0 + rows[i], i * rank:(i + 1) * rank]
                        if rows[i] > HIDDEN:           # an FFN lane: drop the 192 zero rows back off
                            t = t[:HIDDEN]
                        out[f"{fmt.stem(f'{par}.{p}')}.lora_up.weight"] = t.contiguous()
                continue
            if leaf in _RENAMES:
                p = _RENAMES[leaf]
                if ab == "down" and v.shape[1] == HIDDEN_FUSED:
                    v = v[:, :HIDDEN].contiguous()     # w2's contraction dim, un-padded
                tail = ".lora_down.weight" if ab == "down" else ".lora_up.weight" if ab == "up" else ".alpha"
                out[f"{fmt.stem(f'{par}.{p}')}{tail}"] = v
                continue
            out[k] = v
        return out

    def export_lora_state_dict(self, sd):
        """Re-fuse the trained LoRA back to the checkpoint's on-disk layout (the inverse of load_dit).

        `sd` is the family's LoRA in the form its description declares (families/fibo2.py LoRAFormat: the kohya
        stems ComfyUI's loader registers, written by families/lora.py through the same declaration). The output
        keys are the checkpoint's own tensor paths, stemmed by the same declaration - the export never builds a
        stem of its own, it only re-groups the tensors. What only the family knows is the fused layout:
        comfy/weight_adapter/lora.py LoRAAdapter computes (up @ down).reshape(weight.shape) *
        (alpha / down.shape[0]), so each fused group must reproduce the fused tensor's exact shape and
        scaling: the downs stack along dim 0 (the fused rank is the sum of the part ranks), the ups are
        block-diagonal (each part's delta stays inside its own rows), and the fused alpha is the sum of the
        part alphas - with one rank and one alpha per part (what the trainer builds) that keeps every part's
        own alpha/rank scaling exactly. The FFN hidden is padded back to 9728 (w13 up rows, w2 down cols):
        the checkpoint's extra lanes are zero, so the padded lanes of the delta stay zero too.
        """
        def _ab_of(k):
            for ab, suf in (("down", ".lora_down.weight"), ("up", ".lora_up.weight"), ("alpha", ".alpha")):
                if k.endswith(suf):
                    return k[: -len(suf)], ab
            return None

        fmt = self.description.lora
        inv = self._module_paths()
        mods = {}                                       # module path -> {down, up, alpha}
        arrival = {}                                    # module path -> the stem its keys came in on
        for k, v in sd.items():
            r = _ab_of(k)
            if r is not None:
                stem, ab = r
                m = inv.get(stem, stem)                 # the map's name for it, or the stem itself
                mods.setdefault(m, {})[ab] = v
                arrival[m] = stem

        def _pad_rows(t, rows):
            if t.shape[0] == rows:
                return t.contiguous()
            pad = torch.zeros(rows - t.shape[0], t.shape[1], dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=0)

        def _pad_cols(t, cols):
            if t.shape[1] == cols:
                return t.contiguous()
            pad = torch.zeros(t.shape[0], cols - t.shape[1], dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=1)

        def _block_diag(ts):
            """Diagonal block assembly of the per-part up matrices: (sum(rows), sum(ranks))."""
            m = torch.zeros((sum(t.shape[0] for t in ts), sum(t.shape[1] for t in ts)),
                            dtype=ts[0].dtype, device=ts[0].device)
            r0 = c0 = 0
            for t in ts:
                m[r0:r0 + t.shape[0], c0:c0 + t.shape[1]] = t
                r0 += t.shape[0]
                c0 += t.shape[1]
            return m

        out, used = {}, set()

        def fuse(fused_stem, srcs, part_rows=None, down_cols=None):
            """Emit the fused tensor `fused_stem` from the trained parts in `srcs`, in the fused tensor's
            row order. One part is a passthrough (optionally with its down padded to `down_cols`); several
            are rank-cat (down) + block-diagonal (up) + summed-alpha fused. The emitted keys are kohya stems,
            which is the form the loader looks for (see the docstring)."""
            got = [mods.get(s) for s in srcs]
            if not any(m and "down" in m for m in got):
                return                                  # nothing trained in this group
            ks = fmt.stem(fused_stem)
            if len(srcs) == 1:
                m = got[0]
                d = m["down"] if down_cols is None else _pad_cols(m["down"], down_cols)
                out[f"{ks}.lora_down.weight"] = d
                out[f"{ks}.lora_up.weight"] = m["up"]
                if "alpha" in m:
                    out[f"{ks}.alpha"] = m["alpha"]
                used.add(srcs[0])
                return
            ref = next(m for m in got if m and "down" in m)
            rank = ref["down"].shape[0]
            downs, ups, alpha = [], [], None
            for s, m in zip(srcs, got):
                if m and "down" in m:
                    downs.append(m["down"])
                    ups.append(m["up"] if part_rows is None else _pad_rows(m["up"], part_rows))
                    if "alpha" in m:
                        alpha = m["alpha"].clone() if alpha is None else alpha + m["alpha"]
                    used.add(s)
                else:
                    # an untrained part still occupies its rows of the fused tensor, with a zero delta
                    downs.append(torch.zeros((rank, ref["down"].shape[1]),
                                             dtype=ref["down"].dtype, device=ref["down"].device))
                    ups.append(torch.zeros((part_rows, rank), dtype=ref["up"].dtype, device=ref["up"].device))
            out[f"{ks}.lora_down.weight"] = torch.cat(downs, dim=0)
            out[f"{ks}.lora_up.weight"] = _block_diag(ups)
            if alpha is not None:
                out[f"{ks}.alpha"] = alpha

        for pre in [f"layers.{i}" for i in range(N_BLOCKS)] + [f"noise_refiner.{i}" for i in range(N_REFINER)]:
            fuse(f"{pre}.attention.qkv",
                 (f"{pre}.attention.to_q", f"{pre}.attention.to_k", f"{pre}.attention.to_v"), DIM)
            fuse(f"{pre}.attention.out", (f"{pre}.attention.to_out.0",))
            fuse(f"{pre}.cross_attention.to_kv",
                 (f"{pre}.cross_attention.to_k", f"{pre}.cross_attention.to_v"), DIM)
            fuse(f"{pre}.cross_attention.to_q", (f"{pre}.cross_attention.to_q",))
            fuse(f"{pre}.cross_attention.to_out", (f"{pre}.cross_attention.to_out.0",))
            fuse(f"{pre}.feed_forward.w13",
                 (f"{pre}.feed_forward.w1", f"{pre}.feed_forward.w3"), HIDDEN_FUSED)
            fuse(f"{pre}.feed_forward.w2", (f"{pre}.feed_forward.w2",), down_cols=HIDDEN_FUSED)

        for pre in [f"perceiver.layers.{i}" for i in range(N_PERCEIVER)]:
            fuse(f"{pre}.self_attention.to_q", (f"{pre}.self_attention.to_q",))
            fuse(f"{pre}.self_attention.to_kv",
                 (f"{pre}.self_attention.to_k", f"{pre}.self_attention.to_v"), DIM)
            fuse(f"{pre}.self_attention.to_out", (f"{pre}.self_attention.to_out.0",))
            fuse(f"{pre}.cross_attention.to_kv",
                 (f"{pre}.cross_attention.to_k", f"{pre}.cross_attention.to_v"), DIM)
            fuse(f"{pre}.cross_attention.to_q", (f"{pre}.cross_attention.to_q",))
            fuse(f"{pre}.cross_attention.to_out", (f"{pre}.cross_attention.to_out.0",))
            fuse(f"{pre}.ffn.w13", (f"{pre}.ffn.w1", f"{pre}.ffn.w3"), HIDDEN_FUSED)
            fuse(f"{pre}.ffn.w2", (f"{pre}.ffn.w2",), down_cols=HIDDEN_FUSED)

        # a trained module that no fused group covers is written back on the stem it arrived on: it is already a
        # checkpoint tensor path, so there is nothing for the family to re-group
        for m in sorted(mods):
            if m in used or "down" not in mods[m]:
                continue
            ks = arrival[m]
            out[f"{ks}.lora_down.weight"] = mods[m]["down"]
            out[f"{ks}.lora_up.weight"] = mods[m]["up"]
            if "alpha" in mods[m]:
                out[f"{ks}.alpha"] = mods[m]["alpha"]
            logger.debug("Fibo2 export: passthrough %s", m)

        _self_check(out)
        return out
