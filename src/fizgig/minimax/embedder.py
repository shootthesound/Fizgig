"""MiniMax H3 text encoder: Qwen3-VL-32B language model (truncated to 50 layers).

The DiT is conditioned on the **unnormalized hidden state after language layer 50** — the
last-layer output of the truncated checkpoint, with NO final norm (comfy applies none). We
build a transformers Qwen3Model (its layout matches the checkpoint's `model.layers.*` 1:1),
strip the `model.` prefix, skip the vision tower (`visual.*` — unused for text-only captions),
NF4-quantize the big linears so the ~32B fits a 24-32 GB card, and replace the final norm with
Identity so last_hidden_state IS the raw layer-50 output.

Image LoRA training uses text-only captions (no vision blocks), tokenized raw with NO special
tokens — the H3 convention. The tokenizer is the Qwen3-VL one Fizgig already bundles
(src/fizgig/assets/qwen3vl_tokenizer — shared vocab across the 4B/32B sizes).

Output: [1, L, 5120] bf16, exactly what the DiT's condition_proj expects.
"""

import logging
import os

import torch
import torch.nn as nn

# The official safetensors safe_open(device="cpu") + get_tensor path memory-maps the file and
# slices the torch storage per tensor — on Windows, reading a 48 GB file that way hard-crashes
# (access violation, exit 0xC0000005) deep in torch.storage.__getitem__. The repo's own
# MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile (no torch mmap-view), which
# is exactly why every large-model loader (krea2 / klein) uses it. Use it here too.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

# Derived from the checkpoint tensor shapes (U8 weights are 4-bit-packed: real in-dim = 2x).
_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120, num_hidden_layers=50, num_attention_heads=64, num_key_value_heads=8,
    head_dim=128, intermediate_size=25600, vocab_size=151936, max_position_embeddings=262144,
    rms_norm_eps=1e-6, rope_theta=5000000.0, attention_bias=False, tie_word_embeddings=False,
)

# The Qwen3 decoder Linears to NF4 (the matmul bulk). embed_tokens stays as-is (int8 in the
# checkpoint / bf16 after load); the tiny q/k norms + layernorms stay bf16.
_NF4_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                 ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                 ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


# ---------------------------------------------------------------------------
# ComfyUI "comfy_quant" dequant (nvfp4 + int8_tensorwise) — lets the small nvfp4-awq TE
# (~15 GB) stand in for the 48 GB bf16 file. We dequantize each weight back to bf16 per tensor
# at load, then NF4-quantize it on GPU exactly like the bf16 path, so nothing downstream changes.
#
# nvfp4 (comfy/float.py): W = fp4_e2m1 * block_scale * global_scale, block size 16 along the input
# dim. Stored as: weight U8 [out, in/2] (2 E2M1 values/byte, HIGH nibble = first/even element),
# weight_scale FP8-E4M3 [out, in/16] in ComfyUI's cuBLAS `to_blocked` swizzle (128x4 tiles ->
# (-1,32,16); see comfy/float.py::to_blocked — must be inverted to row-major before use),
# weight_scale_2 F32 scalar. AWQ layers (o_proj/down_proj) also carry pre_quant_scale [in]: comfy
# applies it as `input = input * pre_quant_scale` (ops.py), i.e. y=(x*s)@W^T = x@(W*s)^T — so we
# fold it into the weight columns by MULTIPLYING. For the other Linears (q/k/v, gate/up) the AWQ
# scale was folded into the PRECEDING norm weight at quantization time, so the checkpoint is
# self-consistent: load its norm weights as-is and no unfold is needed (validated per-column
# against the bf16 file: ratio W_dq/W_ref == ln_ref/ln_nvfp4 to <1%, all layer shapes).
# int8_tensorwise: W = int8 * weight_scale (per-tensor scalar).
# ---------------------------------------------------------------------------
# E2M1 magnitude for the low 3 bits (sign is bit 3): {0, .5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# The same table extended over the full nibble, sign bit included — so decoding is a single
# gather rather than a magnitude lookup plus a separate sign tensor (see _nvfp4_dequant).
_E2M1_SIGNED = torch.cat([_E2M1_MAG, -_E2M1_MAG])


def _from_blocked(blocked, rows, cols):
    """Invert ComfyUI's to_blocked (comfy/float.py): (-1, 32, 16) tiles -> row-major [rows, cols].
    Verified as an exact roundtrip of to_blocked for the shapes in this checkpoint."""
    nrb = -(-rows // 128)
    ncb = -(-cols // 4)
    x = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return x[:rows, :cols].contiguous()


def _nvfp4_dequant(packed, block_scale_fp8, global_scale):
    """packed U8 [out, in/2] -> bf16 [out, in].  W = e2m1(code) * block_scale * global_scale.

    Written for CPU throughput — this runs once per Linear across ~350 layers of a 32B model,
    so the temporaries dominate. Two things keep it lean vs the obvious formulation:
      * ONE gather through a 16-entry SIGNED table, instead of a magnitude gather plus a
        separate sign tensor (the sign bit is just the top bit of the nibble, so it can live
        in the table);
      * the block scale is BROADCAST over each group of 16, instead of repeat_interleave'd to
        full width — which never materialises an [out, in] float32 copy of the scales.
    Measured bit-identical to the naive version, ~3x faster (209 -> 70 ms on a [5120, 8192]
    layer, i.e. ~73s -> ~25s across the whole encoder)."""
    out, in2 = packed.shape
    inp = in2 * 2
    dev = packed.device
    # device follows `packed`: this began as a load-time CPU helper, but Nvfp4Linear now calls it
    # inside a GPU forward, and an un-placed scratch tensor silently lands on CPU.
    bs = _from_blocked(block_scale_fp8.to(torch.float32).reshape(-1, 32, 16), out, inp // 16)
    gs = global_scale.to(torch.float32)
    table = _E2M1_SIGNED.to(dev)

    # ROW-CHUNKED, because the whole-matrix formulation transiently held an int64 gather
    # index (8 B/element — over a GB on the big layers), the fp32 values, and two fp32
    # products all at once: ~1.7 GB per forward, which is exactly what pushed a 16 GB card's
    # text-caching over the edge (surfaced by the VRAM simulator; real cards limped through
    # it on WDDM paging — the issue-#71 freeze). Same ops per element in the same order, so
    # still bit-identical — the transients are just bounded to a chunk. The scalar multiply
    # is in-place for the same reason.
    w_out = torch.empty(out, inp, dtype=torch.bfloat16, device=dev)
    chunk = max(1, (32 << 20) // max(inp, 1))                     # ~32M elements per chunk
    for r0 in range(0, out, chunk):
        r1 = min(out, r0 + chunk)
        codes = torch.empty(r1 - r0, inp, dtype=torch.uint8, device=dev)
        codes[:, 0::2] = packed[r0:r1] >> 4                       # HIGH nibble first (verified)
        codes[:, 1::2] = packed[r0:r1] & 0x0F
        vals = table[codes.long()]                                # one signed gather
        w = (vals.view(r1 - r0, inp // 16, 16) * bs[r0:r1].unsqueeze(-1)).mul_(gs)
        w_out[r0:r1] = w.view(r1 - r0, inp).to(torch.bfloat16)
    return w_out


def _dequant_comfy_weight(f, file_mod, ckpt):
    """Dequantize one comfy-quant Linear weight (nvfp4 or int8_tensorwise) back to bf16 [out, in].

    The `.comfy_quant` blob names the scheme — checked explicitly, because e.g. the
    int8_convrot TE variant stores ROTATED weights that would sail through a naive
    weight*scale dequant and silently produce a garbage encoder."""
    import json as _json
    fmt = ""
    try:
        blob = bytes(f.get_tensor(file_mod + ".comfy_quant").tolist())
        fmt = _json.loads(blob.decode("utf-8")).get("format", "")
    except Exception:
        pass
    if fmt not in ("nvfp4", "int8_tensorwise"):
        raise NotImplementedError(
            f"Unsupported comfy-quant format '{fmt or 'unknown'}' on {file_mod} — Fizgig can load "
            "the nvfp4-awq or bf16 Qwen3-VL-32B TE. The int8_convrot variant stores rotated "
            "weights and is not supported; use qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "
            "(~15.7 GB) or the bf16 file instead.")
    if fmt == "nvfp4":
        packed = f.get_tensor(file_mod + ".weight")
        bscale = f.get_tensor(file_mod + ".weight_scale")
        gscale = f.get_tensor(file_mod + ".weight_scale_2")
        w = _nvfp4_dequant(packed, bscale, gscale)
        pqs_key = file_mod + ".pre_quant_scale"
        if pqs_key in ckpt:                               # AWQ: fold s into the weight columns
            pqs = f.get_tensor(pqs_key).to(torch.float32)
            w = (w.to(torch.float32) * pqs.unsqueeze(0)).to(torch.bfloat16)
        return w
    # int8_tensorwise: W = int8 * weight_scale (per-tensor scalar)
    w = f.get_tensor(file_mod + ".weight").to(torch.float32)
    s = f.get_tensor(file_mod + ".weight_scale").to(torch.float32)
    return (w * s).to(torch.bfloat16)


def _bundled_tokenizer_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "qwen3vl_tokenizer")


# H3 extends the stock Qwen3-VL vocabulary with these, in this order — MiniMax's own
# text_encoder/tokenizer_config.json lists them appended to Qwen's 13, which is what assigns them
# ids 151669..151675 (all inside the 151936-row embedding table, so they index real trained rows).
# ORDER IS LOAD-BEARING: they exist in neither vocab.json nor tokenizer.json, so transformers
# numbers them sequentially on load, and a reordering would silently shift every id.
_H3_SPECIAL_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>",
                      "<|caption_start|>", "<|caption_end|>")


def _add_h3_special_tokens(tok):
    """Teach a stock Qwen3-VL tokenizer H3's markup, unless it already knows it.

    The bundled tokenizer is redistributed unmodified from Qwen3-VL-4B-Instruct and is SHARED
    with the krea2 encoder, so the tokens are added here rather than to that asset: patching the
    asset would change tokenization on an unrelated model path.

    Without this, `<d>` splits into byte pairs ([90707, 30768, ...] rather than [151669]) and
    `<|lyrics_start|>hold on<|lyrics_end|>` goes from 4 tokens to 16. Ordinary prose is
    unaffected, which is why the omission is invisible until a prompt carries dialogue markup —
    and H3 wraps ALL dialogue in `<d>[Language] ...</d>`.
    """
    missing = [t for t in _H3_SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) is None]
    if missing:
        tok.add_tokens(missing, special_tokens=True)


def build_qwen3_te(config_overrides=None):
    """A Qwen3Model text encoder with no final norm (returns the raw layer-50 output).

    Text-only. For a caption with no reference image this is equivalent to the full Qwen3-VL
    stack: VL applies mrope, but every text token gets the SAME index on all three axes, which
    collapses mrope to ordinary 1-D rope. Reference images break that equality, which is why the
    r2v path needs build_qwen3vl_te() below rather than this."""
    from transformers import Qwen3Config, Qwen3Model
    cfg = dict(_QWEN3_32B_TRUNC50)
    if config_overrides:
        cfg.update(config_overrides)
    model = Qwen3Model(Qwen3Config(**cfg))
    model.norm = nn.Identity()          # comfy applies NO final norm to the layer-50 conditioning
    return model


# Qwen3-VL-32B vision tower. Every value here is transformers' own default for
# Qwen3VLVisionConfig EXCEPT out_hidden_size, and all of them were cross-checked against the
# shipped checkpoint's tensor shapes rather than taken on trust:
#   depth 27              <- 27 distinct visual.blocks.N
#   hidden_size 1152      <- visual.blocks.N.attn.proj.weight [1152, 1152]
#   intermediate_size 4304<- visual.blocks.N.mlp.linear_fc1.weight [4304, 1152]
#   patch_size 16,        <- visual.patch_embed.proj.weight [1152, 3, 2, 16, 16]
#   temporal_patch_size 2     (also gives in_channels 3 and the temporal patch)
#   spatial_merge_size 2  <- visual.merger.linear_fc1.weight in-features 4608 = 1152 * 2 * 2
#   out_hidden_size 5120  <- visual.merger.linear_fc2.weight [5120, 4608]; the DEFAULT IS 3584,
#                            which is the one value that would have been wrong left alone
#   num_position_embeddings 2304 <- visual.pos_embed.weight [2304, 1152]
#   deepstack_visual_indexes [8, 16, 24] <- 3 visual.deepstack_merger_list.N entries
# The config is hardcoded rather than fetched: ai-toolkit pulls it from the MiniMax HF repo,
# which is gated and geo-restricted, and Fizgig already hardcodes the text half the same way.
_QWEN3VL_VISION = dict(out_hidden_size=5120)

# Qwen3-VL is an mrope model, so the text stack needs a rope_scaling section. mrope_section sums
# to head_dim/2 (24+20+20 = 64 = 128/2).
_QWEN3VL_ROPE_SCALING = {"rope_type": "default", "mrope_section": [24, 20, 20],
                         "mrope_interleaved": True}


def build_qwen3vl_te(config_overrides=None):
    """The FULL Qwen3-VL stack — vision tower + language model — with no final norm.

    Needed only for reference (r2v) conditioning, where the prompt carries `<Picture i>:` vision
    blocks. Checked against the shipped nvfp4 checkpoint: all 902 of its base tensors map onto
    this module tree with nothing left over (see tests/test_minimax_ref_encoder.py). The three
    parameters with no checkpoint entry are `language_model.norm.weight` (replaced by Identity
    here, as comfy applies no final norm) and the two computed rope inv_freq buffers."""
    from transformers import Qwen3VLConfig, Qwen3VLModel
    txt = dict(_QWEN3_32B_TRUNC50)
    txt["rope_scaling"] = dict(_QWEN3VL_ROPE_SCALING)
    vis = dict(_QWEN3VL_VISION)
    if config_overrides:
        txt.update(config_overrides.get("text", {}))
        vis.update(config_overrides.get("vision", {}))
    model = Qwen3VLModel(Qwen3VLConfig(text_config=txt, vision_config=vis))
    model.language_model.norm = nn.Identity()
    return model


# AdaLN modality tags. The DiT selects its modulation by these, so a mis-tagged row is modulated
# as the wrong modality — it renders, it just renders wrong.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def build_image_processor(tokenizer_dir=None):
    """The Qwen3-VL image processor, from the config bundled in src/fizgig/assets.

    Built directly from the bundled preprocessor_config.json rather than downloaded: MiniMax's
    HF repo is gated and geo-restricted, and everything needed (patch 16, temporal patch 2,
    merge 2, mean/std 0.5) already ships with the tokenizer assets."""
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(tokenizer_dir or _bundled_tokenizer_dir())


def build_reference_tokens(tokenizer, image_processor, caption, images, max_length: int = None):
    """The r2v prompt presentation: `<Picture i>: <vision block>` per image, then the caption.

    Returns (input_ids [1, L], token_tags [L], pixel_values, image_grid_thw). Mirrors
    ai-toolkit's encode_minimax_h3_prompt and comfy's MiniMaxH3Tokenizer.tokenize_with_weights:
    raw tokens, no chat template, no special tokens.

    The tags are the load-bearing part. `<Picture 1>: ` is TEXT, but the WHOLE vision run —
    the <|vision_start|> and <|vision_end|> markers included, not just the image pads — is
    VIDEO. Tagging the markers as text would modulate two rows per image as the wrong modality.
    """
    pixel_values, image_grid_thw = None, None
    ids, tags = [], []
    if images:
        vision = image_processor(images=images, return_tensors="pt")
        pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
        merge = image_processor.merge_size ** 2
        v_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        v_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for i in range(len(images)):
            n_img = int(image_grid_thw[i].prod()) // merge
            label = tokenizer(f"<Picture {i + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = [v_start] + [img_pad] * n_img + [v_end]
            ids += label + vision_ids
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision_ids)

    _tk = dict(add_special_tokens=False)
    if max_length:                                  # None -> no cap, matching ComfyUI
        _tk.update(truncation=True, max_length=max_length)
    prompt_ids = tokenizer(caption, **_tk)["input_ids"]
    ids += prompt_ids
    tags += [TEXT_TAG] * len(prompt_ids)
    if not ids:                                     # empty prompt with no reference
        pid = getattr(tokenizer, "pad_token_id", None)
        ids, tags = [151643 if pid is None else int(pid)], [TEXT_TAG]
    return (torch.tensor([ids], dtype=torch.long), torch.tensor(tags, dtype=torch.long),
            pixel_values, image_grid_thw)


# ─── numbered references (the pack's H3 RefMod Text Encode presentation) ───────────────────
# ComfyUI's H3 tokenizer (comfy/text_encoders/minimax.py) presents saved references as
#   image -> "<Picture i>: " <vision block>            (the frame twice in the temporal patch)
#   audio -> "<Audio j>: "                             (a label only; audio never enters Qwen)
#   video -> "<Video k>: " then, per 2-frame block, "<t seconds>" <vision block>
# with its own resize / normalise / patch policy (process_qwen2vl_images with patch 16 and
# mean/std 0.5 for pictures, process_video_block for the pairs). Mirrored here so a prompt
# written against Fizgig's numbers means the same thing in ComfyUI.
QWEN_REF_MIN_PIXELS, QWEN_REF_MAX_PIXELS = 3136, 12845056


def reference_patches(frames: torch.Tensor, patch_size: int = 16, temporal_patch_size: int = 2,
                      merge_size: int = 2, min_pixels: int = QWEN_REF_MIN_PIXELS,
                      max_pixels: int = QWEN_REF_MAX_PIXELS):
    """[k, H, W, 3] float in [0, 1], k = 1 (a picture: repeated into the temporal patch) or 2 (a
    video block) -> (flatten [gh*gw, 3*2*16*16], grid_thw [1, 3] = (1, gh, gw)). ComfyUI's
    process_qwen2vl_images (k = 1) / process_video_block (k = 2), same arithmetic."""
    import math
    x = frames.detach().float().permute(0, 3, 1, 2)                       # [k, 3, H, W]
    if x.shape[0] == 1:
        x = x.repeat(temporal_patch_size, 1, 1, 1)
    assert x.shape[0] == temporal_patch_size, f"a reference block is 1 or 2 frames, got {x.shape[0]}"
    height, width = int(x.shape[2]), int(x.shape[3])
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    x = torch.nn.functional.interpolate(x, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = x.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size,
                        grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flatten.contiguous(), torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)


def frame_tensor(frame) -> torch.Tensor:
    """A PIL image or a [H, W, 3] tensor -> float [H, W, 3] in [0, 1]."""
    if isinstance(frame, torch.Tensor):
        t = frame.detach().float()
        return t / 255.0 if t.max() > 1.0 else t
    import numpy as np
    return torch.from_numpy(np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0)


def sample_video_frames(frames, reference_fps: float = 24.0):
    """The pack's 2 fps presentation of a decoded reference video: -> (data [k, H, W, 3],
    timestamps). Index by timestamp so a non-integer rate does not drift (their maths)."""
    import math
    n = len(frames)
    times = [i / 2 for i in range(math.ceil(n * 2 / float(reference_fps)))] or [0.0]
    idx = [min(round(t * float(reference_fps)), n - 1) for t in times]
    data = torch.stack([frame_tensor(frames[i]) for i in idx])
    return data, times


def build_numbered_reference_tokens(tokenizer, caption: str, items, max_length: int = None):
    """ComfyUI's tokenize_with_weights(minimax_ref_items=…), as ids: items are dicts
    {"type": "image", "data": frame} / {"type": "video", "data": [T, H, W, 3], "timestamps": […]}
    / {"type": "audio"}. Returns (input_ids [1, L], token_tags [L], pixel_values or None,
    image_grid_thw or None) — vision runs (markers included) tagged VIDEO, everything else TEXT."""
    v_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids, tags, pix, grids = [], [], [], []
    counters = {"image": 0, "audio": 0, "video": 0}

    def add_text(text):
        t = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(t)
        tags.extend([TEXT_TAG] * len(t))

    def add_vision(block):
        flatten, grid = reference_patches(block)
        n = int(grid[0].prod()) // 4                       # merge_size ** 2
        run = [v_start] + [img_pad] * n + [v_end]
        ids.extend(run)
        tags.extend([VIDEO_TAG] * len(run))
        pix.append(flatten)
        grids.append(grid)

    for item in items or []:
        kind = str(item.get("type", ""))
        if kind not in counters:
            raise ValueError(f"reference item type must be image, video or audio, got {kind!r}")
        counters[kind] += 1
        if kind == "image":
            add_text(f"<Picture {counters['image']}>: ")
            add_vision(frame_tensor(item["data"]).unsqueeze(0))
        elif kind == "audio":
            add_text(f"<Audio {counters['audio']}>: ")
        else:
            frames = item["data"]
            frames = torch.stack([frame_tensor(f) for f in frames]) if not isinstance(frames, torch.Tensor) else frame_tensor(frames)
            timestamps = list(item.get("timestamps") or [i / 2.0 for i in range(int(frames.shape[0]))])
            if int(frames.shape[0]) % 2 == 1:                 # repeat-pad to the temporal patch of 2
                frames = torch.cat([frames, frames[-1:]], dim=0)
                timestamps = timestamps + [timestamps[-1]]
            add_text(f"<Video {counters['video']}>: ")
            for i in range(0, int(frames.shape[0]), 2):
                add_text("<%.1f seconds>" % ((timestamps[i] + timestamps[i + 1]) / 2.0))
                add_vision(frames[i:i + 2])
    _tk = dict(add_special_tokens=False)
    if max_length:
        _tk.update(truncation=True, max_length=max_length)
    prompt_ids = tokenizer(caption, **_tk)["input_ids"]
    ids += prompt_ids
    tags += [TEXT_TAG] * len(prompt_ids)
    if not ids:
        pid = getattr(tokenizer, "pad_token_id", None)
        ids, tags = [151643 if pid is None else int(pid)], [TEXT_TAG]
    pixel_values = torch.cat(pix, dim=0) if pix else None
    grid_thw = torch.cat(grids, dim=0) if grids else None
    return (torch.tensor([ids], dtype=torch.long), torch.tensor(tags, dtype=torch.long), pixel_values, grid_thw)


def qwen3vl_key_map(key: str) -> str:
    """Checkpoint key -> Qwen3VLModel parameter name.

    The single-file checkpoint stores the language stack under `model.` and the vision tower
    under `visual.`; Qwen3VLModel wants `language_model.` and `visual.`."""
    return "language_model." + key[len("model."):] if key.startswith("model.") else key


def _cfg_attr(cfg, name, default):
    """A config attribute that lives at the top level (Qwen3Model) or under text_config
    (Qwen3VLModel — the vision build still runs the text path)."""
    for cand in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(cand, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


class MiniMaxH3TextEncoder:
    """Loads the bf16 TE, NF4 on GPU, and encodes captions to [1, L, 5120] bf16."""

    def __init__(self, model, tokenizer, device="cuda", compute_dtype=torch.bfloat16,
                 cpu_embed=False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.compute_dtype = compute_dtype
        self.cpu_embed = cpu_embed     # embed_tokens left on CPU; see _text_forward()
        self._cache = {}          # caption -> CPU embedding; see encode()
        self._image_processor = None   # built on first r2v encode; see encode_with_reference()

    def _text_forward(self, ids):
        """Token ids -> last_hidden_state, on whichever side the embedding table lives.

        cpu_embed=False is the original call verbatim: ids to the device, `input_ids=`.

        cpu_embed=True leaves the 151936 x 5120 table on the CPU — ~1.5 GB of VRAM that served
        one lookup per forward — so the gather happens there and only its `[B, L, 5120]` result
        crosses to the GPU, passed as `inputs_embeds=`. The decoder stack sees the identical
        tensor either way; a gather is memory-bound and trivial at caption lengths.

        Text-only path. The vision build scatters image embeddings into the `<|image_pad|>` slots
        from input_ids, so it never takes the cpu_embed branch (see load_minimax_h3_te).

        Long sequences on pre-sm_80 GPUs are forwarded in L-chunks (see _plan_chunk): SDPA
        there falls back to the math backend, which materializes the [B, H, L, L] score tensor
        per layer — ~800 MiB at L~2400 — and a 12.8 GB nvfp4 TE leaves only ~2.5 GB free on a
        16 GB card. Chunk k passes the past_key_values from chunk k-1; causal attention makes
        this EXACT (each query attends to the identical key set of the full forward, in the
        same single softmax), and right padding from encode_batch() stays safe (a real token
        only looks left)."""
        chunk = self._plan_chunk(ids)
        if chunk is None or chunk >= ids.shape[1]:            # fast path: the original call
            if not self.cpu_embed:
                return self.model(input_ids=ids.to(self.device)).last_hidden_state
            emb = self.model.embed_tokens(ids.to("cpu")).to(self.device)
            return self.model(inputs_embeds=emb).last_hidden_state

        outs, past = [], None
        for s in range(0, ids.shape[1], chunk):
            e = min(s + chunk, ids.shape[1])
            if self.cpu_embed:
                kw = dict(inputs_embeds=self.model.embed_tokens(ids[:, s:e].to("cpu")).to(self.device))
            else:
                kw = dict(input_ids=ids[:, s:e].to(self.device))
            kw["use_cache"] = True
            if past is not None:
                kw["past_key_values"] = past
            r = self.model(**kw)
            past = r.past_key_values
            outs.append(r.last_hidden_state)
        return torch.cat(outs, dim=1)                         # [B, L, 5120], as in the full pass

    def _plan_chunk(self, ids):
        """Prefill chunk size for this forward, or None for the single pass.

        None when: the device is not a CUDA card, the card is sm_80+ (flash / mem-efficient
        SDPA never materializes the score table — the original single call stands), or the
        batch is small: B*L^2 <= 1024^2. The per-layer math-SDPA peak is H*(B*L^2) at
        (2*itemsize + 4) bytes/element — the bf16 score matrix, its mask-add temporary and
        the fp32 softmax all live at once — so the threshold is a ~1 GB table at H=64,
        cheap anywhere the encoder itself fits. B*L^2, not L: a batch of short captions is
        as heavy as one long one (the original `L <= 1024` bail-out skipped the former).

        A chunk size otherwise — the single pass genuinely does NOT fit, measured on the
        dying card (B=11 L=560, 2.67 GB free): in-flight reached ~1.9 GB before the first
        fp32 softmax (bf16 scores + mask temporary + dense causal mask + projections +
        hidden copies), and the 844.00 MiB softmax then OOM'd at 722 MiB free. A plain
        forward in transformers 4.57 builds no KV cache at all — the working set is the
        attention temporaries themselves, so there is nothing to drop, only to bound.
        Chunking bounds the per-layer slice to [B, H, C, L]; C is computed live from the
        free VRAM (mem_get_info at call time) and the model config: the largest C keeping
        the slice, priced at (2*itemsize + 4) bytes/element, under
        (free - KV - MLP - hidden/q overhead - 10% margin). The KV is NOT optional in
        the chunked path —
        chunk k attends through past_key_values, that is the exactness mechanism — so it
        is priced in full. Clamped to [128, 2048]; C >= L collapses to the single pass
        in _text_forward."""
        B, L = ids.shape
        dev = torch.device(self.device)
        if dev.type != "cuda":
            return None                                        # non-CUDA: fits anywhere
        if torch.cuda.get_device_capability(dev.index)[0] >= 8:
            return None                                        # flash SDPA: original call stands
        if B * L * L <= 1024 * 1024:
            return None                                        # small table: single pass is cheap
        free_b, _ = torch.cuda.mem_get_info(dev.index)
        cfg = self.model.config
        heads = _cfg_attr(cfg, "num_attention_heads", 64)
        itemsize = self.compute_dtype.itemsize                  # bf16 -> 2 bytes
        kv_heads = _cfg_attr(cfg, "num_key_value_heads", heads)
        head_dim = _cfg_attr(cfg, "head_dim", _cfg_attr(cfg, "hidden_size", 5120) // heads)
        layers = _cfg_attr(cfg, "num_hidden_layers", 50)
        kv = B * kv_heads * L * head_dim * 2 * itemsize * layers          # k+v, bf16, all layers
        mlp = B * L * _cfg_attr(cfg, "intermediate_size", 0) * itemsize
        extra = B * L * _cfg_attr(cfg, "hidden_size", 5120) * itemsize * 4       # hidden copies + q proj
        margin = int(0.10 * free_b)
        budget = free_b - kv - mlp - extra - margin
        if budget <= 0:
            budget = int(0.25 * free_b)                        # pathological: keep some chunk
        c = max(128, min(2048, budget // max(1, B * heads * L * (2 * itemsize + 4))))
        if not getattr(self, "_chunk_logged", False):           # one line per process, no spam
            self._chunk_logged = True
            logger.info("TE forward: B=%d L=%d single-pass peak ~%.2f GB > %.2f GB free "
                        "(KV ~%.2f GB) — chunked prefill (C=%d, exact: causal attention "
                        "over past_key_values)",
                        B, L, (B * heads * L * L * (2 * itemsize + 4) + kv + mlp) / 2 ** 30,
                        free_b / 2 ** 30, kv / 2 ** 30, min(c, L))
        return min(c, L)

    def _pad_id(self) -> int:
        """The tokenizer's pad id, or Qwen's default. NOT `pad_token_id or <default>` — a pad id
        of 0 is falsy and would silently be replaced by the default, which is out of range for
        any tokenizer with a smaller vocab."""
        pid = getattr(self.tokenizer, "pad_token_id", None)
        return 151643 if pid is None else int(pid)

    @torch.no_grad()
    def encode(self, caption: str, max_length: int = None) -> torch.Tensor:
        """Encode one caption to [1, L, 5120]. Memoized by caption text for the caching pass.

        Text conditioning does not depend on resolution or on anything else that varies between
        dataset blocks, so a caption that appears more than once — repeated text, or several
        dataset entries over the same folder — is encoded once. Under the nvfp4-resident encoder
        a forward dequantizes 351 weights, so a repeat is far from free.

        max_length=None means NO TRUNCATION, matching ComfyUI (its MiniMax tokenizer sets
        max_length=99999999, pad_to_max_length=False). This used to cap at 512, which silently
        dropped the tail of a long prompt — and worse, the text length sets the media clock
        ORIGIN for the video rows, so a truncated prompt also shifted the render onto a
        different temporal grid than the same prompt in ComfyUI."""
        hit = self._cache.get(caption)
        if hit is not None:
            return hit.clone()                             # callers must not share storage
        # H3: raw prompt text, NO special tokens (no chat template).
        _tk = dict(add_special_tokens=False, return_tensors="pt")
        if max_length:                                     # None/0 -> no cap, as ComfyUI does
            _tk.update(truncation=True, max_length=max_length)
        ids = self.tokenizer(caption, **_tk)["input_ids"]
        if ids.shape[1] == 0:                              # empty caption -> single pad token
            ids = torch.tensor([[self._pad_id()]])
        # norm=Identity -> raw layer-50 output
        emb = self._text_forward(ids).to(self.compute_dtype)  # [1, L, 5120]
        # keep it on CPU: a few hundred KB per caption, and GPU memory is the scarce thing here
        self._cache[caption] = emb.detach().to("cpu")
        return emb

    @torch.no_grad()
    def encode_with_reference(self, caption: str, images, max_length: int = None):
        """r2v conditioning: `<Picture i>:` vision blocks + caption -> ([1, L, 5120], tags [L]).

        Requires the encoder to have been built with build_qwen3vl_te() — the text-only
        Qwen3Model has no vision tower and would silently ignore the pixels. NOT memoized: the
        cache is keyed by caption text alone, which would collide across different reference
        images for the same caption.
        """
        if not hasattr(self.model, "visual"):
            raise RuntimeError(
                "encode_with_reference needs the vision-capable encoder — load with "
                "with_vision=True (build_qwen3vl_te), not the text-only Qwen3Model.")
        if self._image_processor is None:
            self._image_processor = build_image_processor()
        ids, tags, pixel_values, grid = build_reference_tokens(
            self.tokenizer, self._image_processor, caption, images, max_length)
        kw = {}
        if pixel_values is not None:
            # the vision tower is bf16 in the checkpoint; feed it its own dtype
            kw["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            kw["image_grid_thw"] = grid.to(self.device)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=torch.ones_like(ids).to(self.device), **kw)
        # norm is Identity on this build, so last_hidden_state IS the raw layer-50 conditioning
        emb = out.last_hidden_state.to(self.compute_dtype)
        return emb, tags

    @torch.no_grad()
    def encode_with_items(self, caption: str, items, max_length: int = None):
        """Numbered references (the pack's Text Encode presentation): pictures, video blocks
        and audio labels ahead of the caption -> ([1, L, 5120], tags [L]). Vision build only;
        not memoized (the embedding depends on the frames)."""
        if not hasattr(self.model, "visual"):
            raise RuntimeError(
                "encode_with_items needs the vision-capable encoder — load with "
                "with_vision=True (build_qwen3vl_te), not the text-only Qwen3Model.")
        ids, tags, pixel_values, grid = build_numbered_reference_tokens(self.tokenizer, caption, items, max_length)
        kw = {}
        if pixel_values is not None:
            kw["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            kw["image_grid_thw"] = grid.to(self.device)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=torch.ones_like(ids).to(self.device), **kw)
        return out.last_hidden_state.to(self.compute_dtype), tags

    # NO encode_with_reference_batch. Batching the reference encodes was implemented and
    # MEASURED against the one-at-a-time path on the real encoder (tests/diag_ref_batch_encode.py)
    # and it is NOT equivalent: max|diff| 1.3e2 to 1.7e3 on conditioning whose values top out
    # around 2e4, i.e. up to ~8% — against a left-padding control of 1.8e4. A proper 0/1 attention
    # mask (Qwen3-VL derives its mrope positions from it) improved matters but did not fix them.
    #
    # Right padding IS exact on the caption path, for the causal-attention reason documented on
    # encode_batch. The vision path adds mrope positions computed from image_grid_thw and a
    # scatter into the <|image_pad|> slots, and something there does not survive padding. Rather
    # than ship conditioning that is subtly wrong in a way nothing downstream would reveal, the
    # reference pass stays one encode at a time. The cost is ~1.5 s per encode — about 2.5 min
    # for 46 images at 2 references, linear in (images x references), and cached afterwards.
    #
    # If this is ever worth revisiting, the diagnostic is the gate: it must reach the ~1e-7 the
    # caption path achieves, not merely "look close".

    @torch.no_grad()
    def encode_batch(self, captions, max_length: int = None, batch_size: int = 8):
        """Encode many captions, returning [1, L_i, 5120] each — same values as encode(), fewer
        forwards.

        Under the nvfp4-resident encoder a forward dequantizes 351 weights, and that cost is per
        FORWARD, not per token: batching amortizes it across the batch. The rest is exactness.

        RIGHT padding, no attention mask needed. The stack is causal, so a real token at position
        i attends only to 0..i and trailing pads cannot influence it; slicing each row back to its
        true length recovers the single-caption result. Verified on a real Qwen3 stack
        (tests/diag_batch_encode.py): max|diff| 4.5e-08 across batch sizes and pad ids, while the
        LEFT-padding control diverges by 1.9e-01 — the equivalence is a property of right padding
        specifically, not of batching in general."""
        out = [None] * len(captions)
        todo = [i for i, c in enumerate(captions) if c not in self._cache]
        for i, c in enumerate(captions):
            if c in self._cache:
                out[i] = self._cache[c].clone()

        pad_id = self._pad_id()
        for start in range(0, len(todo), batch_size):
            idxs = todo[start:start + batch_size]
            toks = []
            for i in idxs:
                _tk = dict(add_special_tokens=False, return_tensors="pt")
                if max_length:                      # None -> no cap, matching ComfyUI
                    _tk.update(truncation=True, max_length=max_length)
                t = self.tokenizer(captions[i], **_tk)["input_ids"][0]
                toks.append(t if t.numel() else torch.tensor([pad_id]))
            L = max(t.numel() for t in toks)
            ids = torch.full((len(toks), L), pad_id, dtype=torch.long)
            for r, t in enumerate(toks):
                ids[r, : t.numel()] = t
            hs = self._text_forward(ids)
            for r, i in enumerate(idxs):
                emb = hs[r, : toks[r].numel()].unsqueeze(0).to(self.compute_dtype)
                self._cache[captions[i]] = emb.detach().to("cpu")
                out[i] = emb
        return out


class Nvfp4Linear(nn.Linear):
    """A frozen Linear that KEEPS the checkpoint's nvfp4 weights, like the reference loader.

    The previous path dequantized nvfp4 -> bf16 and then re-quantized to NF4, which adds ~9%
    relative error to the encoder (measured per tensor on the real file) that the reference
    never incurs — it attaches the packed nvfp4 tensors and uses them directly. NF4 is also the
    coarser format here: one absmax per 64 elements against nvfp4's FP8 scale per 16, plus AWQ.

    Cost of keeping them: nothing. Packed nvfp4 is 0.5 byte/param, the same as NF4 — ~15.7 GB
    for this encoder either way. Dequantization moves to the matmul, which is fine because the
    TE runs once per caption during the caching pass, not per training step.

    AWQ note: `pre_quant_scale` multiplies the INPUT (comfy ops.py does `input * s`), so it is
    applied to the activation here rather than folded into the weight columns — exact, and it
    keeps the stored block scales untouched.
        Both scale tensors are held as their raw BYTES (uint8 views) and reinterpreted in the
    forward. The block scales are float8_e4m3fn and the global scale fp32; a stray
    `.to(dtype=...)` anywhere would silently cast either one — for the fp8 scales that is not
    lossy, it is garbage, since a cast reads them as numbers rather than reinterpreting. The
    byte view cannot be cast.
    """

    def __init__(self, in_features, out_features, bias=False, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        del self._parameters["weight"]
        self.compute_dtype = compute_dtype
        self.register_buffer("packed", torch.empty(out_features, in_features // 2,
                                                   dtype=torch.uint8), persistent=False)
        # fp8 block scales [out, in/16], stored as bytes
        self.register_buffer("bscale", torch.empty(out_features, in_features // 16,
                                                   dtype=torch.uint8), persistent=False)
        # fp32 scalar, stored as its 4 bytes
        self.register_buffer("gscale", torch.empty(4, dtype=torch.uint8), persistent=False)
        self.register_buffer("pre_quant_scale", None, persistent=False)

    def _scales(self):
        return self.bscale.view(torch.float8_e4m3fn), self.gscale.view(torch.float32)

    @property
    def weight(self):
        """The TRUE dense weight, materialized on demand (inspection only — the forward
        dequantizes inline). The AWQ scale is folded into the columns here so this matches the
        weight the reference produces; the forward applies it to the activation instead, which
        is algebraically identical and one rounding step shorter."""
        bs, gs = self._scales()
        w = _nvfp4_dequant(self.packed, bs, gs).to(self.compute_dtype)
        if self.pre_quant_scale is not None:
            w = w * self.pre_quant_scale.to(w.dtype).reshape(1, -1)
        return w

    def forward(self, x):
        dt = self.compute_dtype
        if self.pre_quant_scale is not None:
            x = x * self.pre_quant_scale.to(x.dtype)      # AWQ acts on the input
        bs, gs = self._scales()
        return torch.nn.functional.linear(x.to(dt), _nvfp4_dequant(self.packed, bs, gs).to(dt),
                                          self.bias)


def load_minimax_h3_te(path: str, device="cuda", compute_dtype=torch.bfloat16,
                       quantize=True, tokenizer_dir=None,
                       te_quant="auto", with_vision=False,
                       cpu_embed=True) -> MiniMaxH3TextEncoder:
    """Build the Qwen3-VL-32B TE. Language-only by default (visual.* skipped).

    cpu_embed keeps the ~1.5 GB embedding table in system RAM and gathers there, which is free
    at caption lengths and is the difference between fitting a 16 GB card and being paged out by
    the driver. It is forced OFF with the vision tower: that build scatters image embeddings into
    the `<|image_pad|>` slots from input_ids, and feeding it inputs_embeds would drop the pixels
    silently.

    with_vision=True builds the FULL stack instead — needed for r2v reference conditioning,
    where the prompt carries `<Picture i>` vision blocks. The vision tower is left in bf16: it
    is ~176 weights against the language stack's thousands, and none of its module names match
    the quantization suffixes (checked — `attn.qkv` / `mlp.linear_fc1`, not `self_attn.q_proj`
    / `mlp.gate_proj`), so it is excluded automatically rather than by a special case.

    te_quant:
      "nvfp4" — KEEP the checkpoint's packed nvfp4 weights (what the reference does). Same
                residency as NF4 (~0.5 byte/param) with none of the extra ~9% error.
      "nf4"   — dequantize to bf16 then 4-bit quantize with bitsandbytes. Required for the
                bf16 checkpoint, which has nothing to keep.
      "auto"  — nvfp4 when the file is nvfp4-awq, else nf4.
    """
    from bitsandbytes.nn import Linear4bit, Params4bit
    from transformers import AutoTokenizer

    cpu_embed = bool(cpu_embed) and not with_vision

    with MemoryEfficientSafeOpen(path) as _probe:
        _is_cq = any(k.endswith(".comfy_quant") for k in _probe.keys())
    mode = te_quant
    if mode == "auto":
        mode = "nvfp4" if _is_cq else "nf4"
    if mode == "nvfp4" and not _is_cq:
        raise ValueError("te_quant='nvfp4' needs the nvfp4-awq checkpoint")
    if not quantize:
        mode = "none"

    # Parameter name -> checkpoint key. The file stores the language stack under `model.` and
    # the vision tower under `visual.`; the VL module tree calls those `language_model.` and
    # `visual.`. Getting this wrong does not raise — the parameter simply keeps its random init.
    def _ck(name: str) -> str:
        if not with_vision:
            return "model." + name
        if name.startswith("language_model."):
            return "model." + name[len("language_model."):]
        return name

    with torch.device("meta"):
        model = build_qwen3vl_te() if with_vision else build_qwen3_te()

        # Swap the NF4-target Linears for Linear4bit shells INSIDE the meta context. Outside it,
        # each Linear4bit eagerly allocates a full fp32 CPU weight (nn.Linear default) — across the
        # 32B Qwen3-VL that is well over 100 GB of throwaway tensors the allocator then holds for
        # the whole caching pass. On meta the shells are 0 bytes; real weights stream in below.
        if mode != "none":
            for mod_name, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    full = f"{mod_name}.{child_name}" if mod_name else child_name
                    if not (isinstance(child, nn.Linear)
                            and (full + ".weight").endswith(_NF4_SUFFIXES)):
                        continue
                    if mode == "nvfp4":
                        setattr(module, child_name,
                                Nvfp4Linear(child.in_features, child.out_features,
                                            bias=child.bias is not None,
                                            compute_dtype=compute_dtype))
                    else:
                        q = Linear4bit(child.in_features, child.out_features,
                                       bias=child.bias is not None,
                                       compute_dtype=compute_dtype, quant_type="nf4")
                        setattr(module, child_name, q)

    dev = torch.device(device)
    print("[load] streaming the Qwen3-VL-32B text encoder — a couple of quiet minutes here is "
          "normal (nvfp4 dequant is the slower one; a bitsandbytes 'expandable_segments not "
          "supported' warning on Windows is harmless).", flush=True)
    model_keys = {n for n, _ in model.named_parameters()}
    with MemoryEfficientSafeOpen(path) as f:
        ckpt = set(f.keys())
        # The comfy nvfp4-awq TE (15 GB) stores packed quant weights + scales instead of plain
        # bf16 — detect once, then dequantize each Linear weight from its
        # `<mod>.weight/.weight_scale[/_2]/.pre_quant_scale` family back to bf16. Everything
        # else (norms, embed) loads as usual; the checkpoint's norm weights already carry the
        # folded AWQ scales, so the model function matches the bf16 TE up to 4-bit noise.
        is_comfy_quant = any(k.endswith(".comfy_quant") for k in ckpt)
        if mode == "nvfp4":
            print("[minimax-te] comfy-quant checkpoint (nvfp4-awq) — KEEPING the packed nvfp4 "
                  "weights (the reference's own storage; NF4 on top would add ~9% error)")
        elif is_comfy_quant:
            print("[minimax-te] comfy-quant checkpoint (nvfp4-awq) — dequantizing to bf16 per tensor")
        # nvfp4 mode: the quantized linears hold BUFFERS, not parameters — fill them first.
        if mode == "nvfp4":
            for mod_name, module in model.named_modules():
                if not isinstance(module, Nvfp4Linear):
                    continue
                fm = _ck(mod_name)
                module.packed = f.get_tensor(fm + ".weight").to(torch.uint8).to(dev)
                # byte views: fp8 block scales and the fp32 global scale must never be CAST
                module.bscale = f.get_tensor(fm + ".weight_scale").contiguous().view(
                    torch.uint8).to(dev)
                module.gscale = f.get_tensor(fm + ".weight_scale_2").to(torch.float32).reshape(
                    1).contiguous().view(torch.uint8).to(dev)
                pqs = fm + ".pre_quant_scale"
                module.pre_quant_scale = (f.get_tensor(pqs).to(compute_dtype).to(dev)
                                          if pqs in ckpt else None)

        for name in model_keys:
            src = _ck(name)
            file_mod = src.rsplit(".", 1)[0]
            leaf = name.rsplit(".", 1)[1]
            if is_comfy_quant and leaf == "weight" and (file_mod + ".comfy_quant") in ckpt:
                w = _dequant_comfy_weight(f, file_mod, ckpt)   # -> bf16 [out, in]
            elif src in ckpt:
                w = f.get_tensor(src)
            else:
                continue                                   # e.g. norm (Identity) has no params
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            if mode == "nvfp4" and name.endswith(_NF4_SUFFIXES):
                continue                                   # placed as nvfp4 buffers above
            if mode == "nf4" and (name.endswith(_NF4_SUFFIXES)):
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                # embed_tokens is excluded from _NF4_SUFFIXES on purpose, so it lands unquantized
                # — 151936 x 5120 in bf16 is ~1.5 GB. It serves one gather per forward, so on the
                # text-only path it stays in system RAM and never reaches the card at all. That
                # margin is what lets a 16 GB card clear the caching pass without the driver
                # paging the encoder back out over PCIe.
                tgt = torch.device("cpu") if (cpu_embed and name == "embed_tokens.weight") else dev
                setattr(parent, leaf, nn.Parameter(keep.to(tgt), requires_grad=False))

    # Computed (non-checkpoint) buffers stayed on meta from the meta-build. The rotary
    # embedding's inv_freq is the load-bearing one — rebuild it on the real device. A general
    # sweep materializes any other stray meta buffers to be safe.
    # These carry inv_freq, which the generic sweep below would otherwise ZERO — a rope table of
    # zeros is no positional information at all, and the model would still run.
    if with_vision:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextRotaryEmbedding,
                                                                    Qwen3VLVisionRotaryEmbedding)
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(model.config.text_config).to(dev)
        _vcfg = model.config.vision_config
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            _vcfg.hidden_size // _vcfg.num_heads // 2).to(dev)
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        model.rotary_emb = Qwen3RotaryEmbedding(model.config).to(dev)
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype, device=dev))
    model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir or _bundled_tokenizer_dir())
    _add_h3_special_tokens(tok)
    return MiniMaxH3TextEncoder(model, tok, device=device, compute_dtype=compute_dtype,
                                cpu_embed=cpu_embed)


# The text-only resident build measures 12.78 GB resident + a 1.6-1.9 GB forward peak
# (tests/diag_minimax_te_peak.py, quoted in minimax_cache_text.py) — it genuinely fits a
# 16 GB card, which keeps the shipped resident path there. Below this, streaming wins.
# 15, not 16: a 16 GB card reports ~15.9 GB total (same lesson as the block-swap
# thresholds), and the measured ~14.7 GB peak fits inside that.
_TEXT_RESIDENT_NEED_GB = 15.0
# The streamed build keeps the packed nvfp4 model (~19 GB) in system RAM; the pin
# degrades gracefully (#94) but a machine that can't even HOLD it would page-storm,
# which is worse than the slow resident path. @mabseyuk's own caveat on his streamer.
_TE_STREAM_RAM_NEED_GB = 22.0


def plan_text_te_build(free_gb, avail_ram_gb, is_nvfp4=True, kill_switch=False,
                       ram_ok=False):
    """The text-only TE build decision, as pure data: 'resident' | 'stream' |
    'resident-ram-short'.

    Kept side-effect-free (like plan_base_quant) so the truth table pins on CPU. Rules:
    the kill-switch or a non-nvfp4 checkpoint always build resident (the streamer only
    serves comfy-quant nvfp4 files); unknown free VRAM builds resident (the safe,
    shipped behaviour); a card below the resident need streams — unless system RAM
    can't hold the packed model, which reports as its own state so the caller can say
    WHY it stayed resident. ram_ok (FIZGIG_TE_RAM_OK, same override the vision rung
    honours) streams despite a short RAM reading — for boxes where 'available' is
    transiently low. Unknown RAM streams (the plan_base_quant H2D precedent: psutil is
    in requirements, unreadable is vanishingly rare)."""
    if kill_switch or not is_nvfp4:
        return "resident"
    if free_gb is None or free_gb >= _TEXT_RESIDENT_NEED_GB:
        return "resident"
    if (not ram_ok and avail_ram_gb is not None
            and avail_ram_gb < _TE_STREAM_RAM_NEED_GB):
        return "resident-ram-short"
    return "stream"


def load_minimax_h3_te_planned(path: str, device="cuda", **kw):
    """Resident or H2D layer-streamed TE, planned from free VRAM.

    Vision (reference) encodes: the resident build peaks at 25.8 GB; the streamed build
    (#79, rintic-13) peaks at 12.7 GB with bit-for-bit identical output, ~2% slower —
    streamed below 27 GB free. Text-only: the resident build fits 16 GB (12.78 GB
    resident + ~1.9 GB forward peak); below that the planner picks @mabseyuk's text-only
    layer streaming (~1.7 GB VRAM — the 12 GB tier's precache went from ~235 s/image to
    ~2.8 s/batch), gated on system RAM actually holding the ~19 GB packed model.
    FIZGIG_NO_TE_H2D=1 is the debug kill-switch back to always-resident."""
    if not kw.get("with_vision"):
        free_gb = None
        if torch.cuda.is_available() and str(device) != "cpu":
            try:
                from fizgig.utils.device import plannable_free_vram
                free_gb = plannable_free_vram(device)
            except Exception:
                free_gb = None
        avail_ram = None
        try:
            import psutil
            avail_ram = psutil.virtual_memory().available / 1e9
        except Exception:
            pass
        try:
            with MemoryEfficientSafeOpen(path) as _probe:
                _is_cq = any(k.endswith(".comfy_quant") for k in _probe.keys())
        except Exception:
            _is_cq = False
        _plan = plan_text_te_build(
            # quantize=False dequantizes everything to bf16 — no packed weights exist,
            # so the streamer can't serve it either (audit N2: the old check printed a
            # streaming line and then built resident-on-GPU anyway).
            free_gb, avail_ram, is_nvfp4=_is_cq and kw.get("quantize", True),
            kill_switch=os.environ.get("FIZGIG_NO_TE_H2D") == "1",
            ram_ok=os.environ.get("FIZGIG_TE_RAM_OK") == "1")
        if _plan == "stream":
            from fizgig.minimax.embedderH2D import load_minimax_h3_te as _load_h2d
            print(f"[minimax-te] {free_gb:.1f} GB free < {_TEXT_RESIDENT_NEED_GB:.0f} GB "
                  "the resident text encoder needs — streaming layers host-to-device "
                  "instead (~2 GB VRAM at caching batch sizes; @mabseyuk)", flush=True)
            return _load_h2d(path, device=device, layer_streaming=True, **kw)
        if _plan == "resident-ram-short":
            print(f"[minimax-te] {free_gb:.1f} GB free is tight for the resident text "
                  f"encoder, but streaming stages the ~19 GB packed model in system RAM "
                  f"and only {avail_ram:.0f} GB is available (other apps — or this run's "
                  "own parked model, mid-training) — loading resident instead; expect "
                  "it to be slow or to run out of memory. Freeing RAM helps, or set "
                  "FIZGIG_TE_RAM_OK=1 to stream regardless.", flush=True)
        return load_minimax_h3_te(path, device=device, **kw)
    need_gb = 27.0                                        # measured 25.8 peak + margin
    free_gb = None
    if torch.cuda.is_available() and str(device) != "cpu":
        try:
            from fizgig.utils.device import plannable_free_vram
            free_gb = plannable_free_vram(device)
        except Exception:
            free_gb = None
    if free_gb is not None and free_gb < need_gb:
        # The nvfp4 streamed build stages the packed ~19 GB model in system RAM — and on
        # Windows, GPU allocations are backed by commit charge, so a box that can't hold
        # the staging dies minutes later at the FIRST real GPU allocation, wearing a
        # misleading 'CUDA error: out of memory' with headroom free on the card (#95:
        # 3090 24 GB VRAM, 24 GB RAM — the pin fallback from #94 moved the crash from
        # load to encode). Refuse up front ONLY where it is hopeless: an nvfp4 file on a
        # small-TOTAL-RAM machine. A big-RAM box with a transiently busy moment (browser
        # holding 12 GB of a 64 GB machine) proceeds with a loud warning instead — the
        # #94 pin fallback and Windows standby reclaim handled those before, and a hard
        # refusal there would be a regression (audit, 25 Aug). Non-nvfp4 files never
        # stage packed weights at all, so they are exempt entirely.
        _is_cq_v = False
        try:
            with MemoryEfficientSafeOpen(path) as _probe:
                _is_cq_v = any(k.endswith(".comfy_quant") for k in _probe.keys())
        except Exception:
            pass
        _avail_ram = _total_ram = None
        try:
            import psutil
            _vm = psutil.virtual_memory()
            _avail_ram, _total_ram = _vm.available / 1e9, _vm.total / 1e9
        except Exception:
            pass
        if (_is_cq_v and kw.get("quantize", True)
                and _total_ram is not None and _total_ram < 28.0
                and os.environ.get("FIZGIG_TE_RAM_OK") != "1"
                # The refusal is about the STREAMED build's RAM staging — with the
                # kill-switch forcing resident, it must stand down (review 6).
                and os.environ.get("FIZGIG_NO_TE_H2D") != "1"):
            raise RuntimeError(
                f"[minimax-te] the reference (vision) encoder does not fit this machine: "
                f"{free_gb:.1f} GB VRAM free needs the streamed build, which stages "
                f"~19 GB of packed weights in system RAM — and this machine has "
                f"{_total_ram:.0f} GB of RAM in total. Options: (1) train WITHOUT "
                f"references (drop the reference/distill setting — plain caption caching "
                f"fits this machine fine), or (2) more system RAM (48 GB+ is "
                f"comfortable). Loading anyway would fail minutes from now with a "
                f"misleading CUDA out-of-memory at the first encode "
                f"(set FIZGIG_TE_RAM_OK=1 to attempt it regardless).")
        # Only an nvfp4 file under quantize can actually stream (the H2D file has no
        # resident nvfp4 path, and non-nvfp4 / --no_quantize builds through it land
        # RESIDENT — the review caught this rung printing 'streaming' and then building
        # the full model on the card). Anything else, and the kill-switch, take the
        # resident loader honestly; on a sub-27 GB card that will be tight, but tight
        # and truthful beats a lie followed by an OOM.
        if (_is_cq_v and kw.get("quantize", True)
                and os.environ.get("FIZGIG_NO_TE_H2D") != "1"):
            if _avail_ram is not None and _avail_ram < _TE_STREAM_RAM_NEED_GB:
                print(f"[minimax-te] heads-up: the streamed encoder stages ~19 GB in "
                      f"system RAM and only {_avail_ram:.0f} GB is available right now "
                      "— closing other apps first will make this faster and safer.",
                      flush=True)
            from fizgig.minimax.embedderH2D import load_minimax_h3_te as _load_h2d
            print(f"[minimax-te] {free_gb:.1f} GB free < {need_gb:.0f} GB the resident "
                  "encoder peaks at — streaming layers host-to-device instead "
                  "(identical output, ~2% slower; #79)")
            return _load_h2d(path, device=device, layer_streaming=True, **kw)
        print(f"[minimax-te] {free_gb:.1f} GB free is below the ~{need_gb:.0f} GB the "
              "resident reference encoder peaks at, and this configuration can't "
              "stream — loading resident anyway; expect it to be slow or to run out "
              "of memory.", flush=True)
    return load_minimax_h3_te(path, device=device, **kw)
