"""Load a real MiniMax H3 checkpoint into MiniMaxH3DiT, NF4-quantizing the big block linears.

Two checkpoint variants load through the same path (auto-detected from the file's own keys):

  * **bf16 full** (`minimax_h3_fl2va_bf16.safetensors`, 66 GB) — the released weights. ~33 B
    params, of which the full-width AdaLN modulation ([96768, 2688] x50) is ~40%.
  * **pruned int8 ConvRot** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, 21 GB) — what
    ComfyUI ships and what users actually run at inference. The AdaLN MLP is replaced by a
    sampled curve table (`adaln_t_table`, so `adaln_proj.linear` is [96768, **8**]) and the 200
    block matmul Linears are stored as int8 in a rotated basis. Training this one means the LoRA
    trains against the same weights it will be deployed on, and AdaLN becomes a sane LoRA target
    instead of an unusable one. See convrot.py for the decode.

Strategy (never holds the file whole): build the model on `meta`, then per parameter read just
that tensor, decode it if quantized, and place it —
  * big block/refiner Linear weights (attn qkv/out, mlp fc1/fc2, adaln)  -> NF4 (bitsandbytes)
  * everything else (norms, patch/condition/time proj, heads, rope, table) -> bf16 on GPU
The frozen base is what LoRA/FT trains on top of; the fp32 output-head island stays fp32.
"""

import os

import torch
import torch.nn as nn

# Not the official safe_open mmap path: on Windows, streaming a 66 GB file via
# safe_open(device="cpu").get_tensor() hard-crashes (access violation) in torch's storage
# mmap-slicing. MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile instead —
# the same reader every other large-model loader in the repo uses. See embedder.py.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, ShardedSafeOpen

from .convrot import ConvRotInt8Linear, dequantize_int8_convrot, parse_comfy_quant
from .model import MiniMaxH3DiT, MiniMaxH3Config

# Which Linear weights get NF4'd: the per-block matmul bulk AND the per-block AdaLN modulation
# projection. On the bf16 model adaln_proj.linear is [96768, 2688] — ~0.5 GB/block, the single
# biggest VRAM sink; on the pruned model it is [96768, 8] and NF4 barely matters either way.
# All are frozen base weights, so NF4 is fine for LoRA/FT on top.
_NF4_SUBSTRINGS = (".attn.qkv_proj.weight", ".attn.out_proj.weight",
                   ".mlp.fc1.weight", ".mlp.fc2.weight", ".adaln_proj.linear.weight")


def _is_nf4_target(name: str) -> bool:
    return name.startswith(("blocks.", "token_refiner.blocks.")) and name.endswith(_NF4_SUBSTRINGS)


def _owner_and_leaf(model, name: str):
    """(module that holds `name`, attribute name) — for a TOP-LEVEL entry the owner is the
    model itself. The pruned checkpoint's `adaln_t_table` is exactly that case: every other
    tensor in either file is nested, so a naive rsplit('.') works everywhere else and fails
    only here."""
    parent_path, _, leaf = name.rpartition(".")
    return (model.get_submodule(parent_path) if parent_path else model), leaf


def config_from_checkpoint(keys, table_shape=None) -> MiniMaxH3Config:
    """Build the config the file actually describes.

    The only structural difference between the two releases is the timestep path: a pruned file
    carries `adaln_t_table` and no `time_embedder.*`."""
    if "adaln_t_table" in keys:
        if table_shape is None:
            raise ValueError("pruned checkpoint: adaln_t_table shape is required")
        size, dim = int(table_shape[0]), int(table_shape[1])
        return MiniMaxH3Config(adaln_t_table_size=size, time_embed_dim=dim)
    return MiniMaxH3Config()


def load_minimax_h3_dit(path: str, device="cuda", compute_dtype=torch.bfloat16,
                        quantize=True, blocks_to_swap: int = 0, base_quant="auto",
                        adaln_fp32: bool = True) -> MiniMaxH3DiT:
    """Return a MiniMaxH3DiT with the real weights loaded, the base frozen.

    `path` is either a single .safetensors file or a directory of Hub shards — the Hub serves
    FL2VA/transformer as 13 shards + an index, which carry the same 535 key names the single
    ComfyUI-converted file does.

    adaln_fp32 keeps the AdaLN projections in fp32 on a PRUNED (curve-table) checkpoint, which
    is what ComfyUI does: `adaln_dtype = torch.float32 if use_adaln_curves else dtype`
    (comfy/ldm/minimax/model.py), and it never downcasts the lerped `t_emb` either. The file
    stores those weights FP16, so loading them at the bf16 compute dtype drops 3 mantissa bits
    on the shift/scale/GATE of every block — measured 0.22-0.30% RMS on the modulation vectors,
    with gate_msa deviating up to 0.084 at block 49. That is a deterministic, same-sign
    perturbation applied 100 times per forward, not zero-mean rounding, and the gate scales the
    whole attention/MLP branch — i.e. exactly where a LoRA's contribution lands.

    `_mod_scale_shift` / `_mod_gate` still cast to the activation dtype at the point of use, so
    only the modulation itself gains precision; the block matmuls stay bf16, as in the reference.

    Ignored on the full bf16 checkpoint, where ComfyUI also uses the compute dtype. Pass False
    when AdaLN is a LoRA target: the adapters are bf16 and would meet an fp32 activation.

    base_quant:
      "int8" — KEEP the checkpoint's int8 ConvRot weights (what the reference trainer does).
               ~0.17% error in the frozen base, ~21 GB resident. Needs a pre-quantized file.
      "nf4"  — decode to bf16 then NF4 (bitsandbytes). ~9.5% error, ~11 GB resident. The only
               option for the bf16 checkpoint, and the fallback for small cards.
      "hqq"  — decode to bf16 then HQQ 4-bit, group 8 (rintic-13, #102; needs `pip install
               hqq`). ~6.3% error — a third closer to int8 than NF4 — at ~0.75 B/param, so
               ~45% more resident than NF4. Explicit pick only; never chosen by "auto".
      "auto" — int8 when the file carries int8 ConvRot weights, else nf4. Matching the
               reference is worth the 10 GB: a LoRA fitted against a 9.5%-perturbed base spends
               capacity correcting quantization error that is absent at inference, and the
               effect compounds with depth (measured: our block 40-49 adapters moved 1.5-3.9x
               the reference's, blocks 0-19 about 0.75x).

    blocks_to_swap parks the LAST n blocks on CPU at load — pair it with
    model.enable_block_swap(n) so the forward moves them just-in-time."""
    from bitsandbytes.nn import Linear4bit, Params4bit

    opener = ShardedSafeOpen if os.path.isdir(path) else MemoryEfficientSafeOpen
    with opener(path) as f:
        keys = set(f.keys())
        table_shape = None
        if "adaln_t_table" in keys:
            table_shape = tuple(f.get_tensor("adaln_t_table").shape)
        cfg = config_from_checkpoint(keys, table_shape)
        # `<module>.comfy_quant` marks a pre-quantized Linear; the config rides in the blob.
        quant_conf = {k[: -len(".comfy_quant")]: parse_comfy_quant(f.get_tensor(k))
                      for k in keys if k.endswith(".comfy_quant")}

        mode = base_quant
        if mode == "auto":
            mode = "int8" if quant_conf else "nf4"
        if mode == "int8" and not quant_conf:
            raise ValueError("base_quant='int8' needs a pre-quantized checkpoint "
                             "(minimax_h3_*_pruned_int8_convrot.safetensors)")
        if mode == "hqq":
            from .hqq4 import HQQ4bitLinear, _quantizer
            _quantizer()                      # fail HERE with the install hint, not mid-stream
        if not quantize:
            mode = "none"
        # fp32 AdaLN only applies to a curve-table (pruned) file — see the docstring.
        _adaln32 = bool(adaln_fp32) and cfg.adaln_t_table_size is not None

        def _is_adaln(n: str) -> bool:
            return _adaln32 and ".adaln_proj.linear." in n
        # In int8 mode ONLY the checkpoint's own quantized linears stay quantized — the refiner,
        # AdaLN and the IO layers load dense, exactly as the reference leaves them (its
        # get_quantization_exclude_modules covers token_refiner and *adaln_proj*).

        with torch.device("meta"):
            model = MiniMaxH3DiT(cfg)

            # Swap the NF4-target Linears for Linear4bit shells — INSIDE the meta context.
            # Outside it, each Linear4bit constructor eagerly allocates a full fp32 weight on CPU
            # (nn.Linear's default), which across the 33 B of NF4-target Linears is ~118 GB of
            # throwaway tensors that the process allocator then holds for the whole run. On meta
            # the shells are 0 bytes; the real weights are streamed in below.
            hqq_targets = []                  # module paths that got an HQQ4bitLinear shell
            if mode != "none":
                for mod_name, module in list(model.named_modules()):
                    for child_name, child in list(module.named_children()):
                        full = f"{mod_name}.{child_name}" if mod_name else child_name
                        if not isinstance(child, nn.Linear):
                            continue
                        if mode == "int8":
                            conf = quant_conf.get(full)
                            if conf is not None:
                                rot = (int(conf.get("convrot_groupsize", 256))
                                       if conf.get("convrot") else 1)
                                setattr(module, child_name, ConvRotInt8Linear(
                                    child.in_features, child.out_features,
                                    bias=child.bias is not None, rot=rot,
                                    compute_dtype=compute_dtype))
                        elif (_is_nf4_target(f"{full}.weight")
                              and not _is_adaln(f"{full}.weight")):
                            # _is_adaln has to be tested HERE as well as at the weight below.
                            # The two decisions are one decision: an fp32 AdaLN that still got a
                            # Linear4bit shell is a 4-bit module holding a dense Parameter, and
                            # bitsandbytes only finds out on the first forward — deep inside a
                            # checkpointed block, as `assert module.weight.shape[1] == 1`.
                            if mode == "hqq":
                                n = child.in_features * child.out_features
                                if n % 32:      # 2*group_size — leave the odd one dense
                                    continue
                                q = HQQ4bitLinear(child.in_features, child.out_features,
                                                  bias=child.bias is not None,
                                                  compute_dtype=compute_dtype)
                                hqq_targets.append(full)
                            else:
                                q = Linear4bit(child.in_features, child.out_features,
                                               bias=child.bias is not None,
                                               compute_dtype=compute_dtype, quant_type="nf4")
                            setattr(module, child_name, q)

        dev = torch.device(device)
        if mode == "int8":
            print(f"[load] streaming the MiniMax H3 base, KEEPING its {len(quant_conf)} int8 "
                  "ConvRot linears (~0.17% base error, ~21 GB resident — the reference's own "
                  "storage; NF4 would be ~9.5% error at ~11 GB).", flush=True)
        elif mode == "hqq":
            print(f"[load] streaming the MiniMax H3 base and quantizing {len(hqq_targets)} "
                  "linears to HQQ 4-bit (group 8 — ~4.8% base error vs NF4's ~9.5%, ~45% "
                  "more resident than NF4). The proximal solver runs per weight on the GPU; "
                  "a few quiet minutes here is normal.", flush=True)
        elif quant_conf:
            print(f"[load] streaming the pre-quantized MiniMax H3 base ({len(quant_conf)} int8 "
                  "ConvRot linears decoded to bf16, then NF4) — a quiet minute here is normal "
                  "(as is a bitsandbytes 'expandable_segments not supported' warning on Windows).",
                  flush=True)
        else:
            print("[load] streaming the 66 GB MiniMax H3 base and quantizing to NF4 — a couple of "
                  "quiet minutes here is normal (as is a bitsandbytes 'expandable_segments not "
                  "supported' warning on Windows).", flush=True)

        # Block swap: params of the LAST n blocks are parked on CPU at load time — quantized on
        # the GPU (Params4bit quantizes on the .to(cuda) move) then immediately moved off, so the
        # packed weights never accumulate. Loading everything resident first and parking
        # afterwards would transiently need the full ~17 GB, OOMing exactly the cards that asked
        # for swap.
        n_layers = len(model.blocks)
        swap_from = n_layers - max(0, min(int(blocks_to_swap or 0), n_layers - 2))

        def _parked(name: str) -> bool:
            if not name.startswith("blocks."):
                return False
            return int(name.split(".")[1]) >= swap_from

        def _read_weight(name: str):
            """The dense tensor for `name`, decoding a pre-quantized Linear if that's what it is."""
            module_path = name[: -len(".weight")]
            conf = quant_conf.get(module_path) if name.endswith(".weight") else None
            if conf is None:
                return f.get_tensor(name)
            return dequantize_int8_convrot(f.get_tensor(name),
                                           f.get_tensor(f"{module_path}.weight_scale"),
                                           conf, out_dtype=compute_dtype)

        # int8 mode: the quantized linears hold BUFFERS, not parameters — fill them first.
        if mode == "int8":
            for mod_path in quant_conf:
                mod = model.get_submodule(mod_path)
                tgt = torch.device("cpu") if _parked(mod_path) else dev
                mod.qdata = f.get_tensor(f"{mod_path}.weight").to(torch.int8).to(tgt)
                mod.wscale = (f.get_tensor(f"{mod_path}.weight_scale")
                              .to(torch.float32).reshape(-1, 1).to(tgt))
        # hqq mode: same buffer story — the shells have no `weight` Parameter, so the
        # named_parameters loop below never sees them. Quantize on the GPU (the solver is
        # CPU-slow on fc1-sized weights), then park the packed buffers if the block is swapped.
        if mode == "hqq":
            for mod_path in hqq_targets:
                mod = model.get_submodule(mod_path)
                mod.load_dense(_read_weight(f"{mod_path}.weight").to(compute_dtype), dev)
                if _parked(mod_path):
                    mod.W_q, mod.scale, mod.zero, mod.qmeta = (
                        mod.W_q.cpu(), mod.scale.cpu(), mod.zero.cpu(), mod.qmeta.cpu())

        for name, param in model.named_parameters():
            if name not in keys:
                continue
            if mode == "int8" and name.rsplit(".", 1)[0] in quant_conf:
                continue                      # already placed as int8 buffers above
            w = _read_weight(name)
            parent, leaf = _owner_and_leaf(model, name)
            if mode == "nf4" and _is_nf4_target(name) and not _is_adaln(name):
                # NF4-quantize this weight onto the GPU; frozen (no grad).
                # NF4 quantization happens on the .to(cuda) move (Params4bit.cuda()).
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                if _parked(name):
                    p = p.to("cpu")            # stays packed (uint8) + keeps quant_state
                setattr(parent, leaf, p)
            else:
                keep = (w.to(torch.float32)
                        if (w.dtype == torch.float32 or _is_adaln(name))
                        else w.to(compute_dtype))
                target = torch.device("cpu") if _parked(name) else dev
                setattr(parent, leaf, nn.Parameter(keep.to(target), requires_grad=False))
        # buffers (rope.inv_freq, and adaln_t_table on a pruned file)
        for name, _ in model.named_buffers():
            if name in keys:
                parent, leaf = _owner_and_leaf(model, name)
                parent.register_buffer(leaf, f.get_tensor(name).to(torch.float32).to(dev))
    # tells the forward not to demote t_emb — an fp32 projection needs an fp32 input.
    model.adaln_fp32 = _adaln32
    model.eval()
    return model
