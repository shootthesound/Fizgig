"""MiniMax H3 DiT — pure-PyTorch port for image-only training.

Faithful, weight-name-compatible port of ComfyUI's comfy/ldm/minimax/model.py, with the
ComfyUI plumbing replaced by plain PyTorch so it trains under Fizgig's LoRA / rotating-FT
machinery:

  operations.Linear / RMSNorm      -> nn.Linear / RMSNorm
  comfy.quant_ops rms_rope_split_half -> pure-torch RMSNorm + split-half RoPE (below)
  comfy.ops.linear_input_act(swiglu)  -> fc2(silu(a) * b)
  optimized_attention                 -> F.scaled_dot_product_attention
  prefetch / patcher wrappers         -> dropped

Scope: IMAGE ONLY. A still image is a single video frame (T=1), so the packed sequence is
[text | audio | video] — the refs / keyframes / cond-row apparatus of the reference is omitted
(nothing conditions an image run), but the AUDIO ROWS ARE NOT: H3 always packs an audio block,
and for one still that is 4 rows of silence noised on the audio schedule. They carry no loss;
they exist so the frozen base runs in the layout it was trained in. `final_layer.audio_out` is
built for checkpoint compatibility and never run — we only read the video head.

Module + parameter names match the checkpoint exactly (blocks.N.attn.qkv_proj.weight,
adaln_proj.linear.weight, video_patch_proj, condition_proj, time_embedder.proj_in, rope.inv_freq,
token_refiner.blocks.N..., final_layer...), so a LoRA/FT trained here maps back onto the base.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --- config ------------------------------------------------------------------------------

@dataclass
class MiniMaxH3Config:
    """From FL2VA/transformer/config.json (MiniMaxH3DiTModel). Defaults are the real model;
    tests pass a tiny override so a forward runs on CPU in milliseconds."""
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    # PRUNED checkpoints (minimax_h3_*_pruned_*, what ComfyUI ships and users run at inference)
    # replace the timestep MLP with a lookup table sampled by linear interpolation: no
    # `time_embedder`, an `adaln_t_table` of [size, time_embed_dim] instead, time_embed_dim 8
    # rather than 2688, and NO silu in front of the AdaLN projections. Set this and the model
    # builds that variant; None = the full bf16 model.
    adaln_t_table_size: Optional[int] = None
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5


# --- helpers (ported verbatim in intent from the reference) ------------------------------

def patchify_video(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]  (row-major t,h,w; channel-major within a patch)."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows: torch.Tensor, t, h, w, c=24, patch_size=(1, 2, 2)) -> torch.Tensor:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

# r2v condition rows. The reference blends each condition latent with a trace of noise and pins
# its row timestep near clean — comfy/ldm/minimax/model.py VISUAL_COND_TIMESTEP.
VISUAL_COND_TIMESTEP = 0.999


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    """Area-normalized (h, w) coords of one latent frame's 2x2-patch rows: [(h//2)*(w//2), 2]."""
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


# --- audio rows --------------------------------------------------------------------------
# H3 is an audio+video model and its packed sequence ALWAYS carries an audio block — the
# reference packs one even for a single still. A still is 1 pixel frame at 24 fps against a
# 40 Hz audio latent clock, so that block is round(40/24) = 2 latents x 2 channels = 4 rows.
# We train and sample images, so those rows are silence (x0 = 0) noised at the audio schedule;
# they contribute no loss. They are packed anyway because leaving them out changes the
# attention context the base model was trained in — the whole point is to run the frozen base
# on-distribution, so the LoRA spends its capacity on the subject and not on compensating for
# a layout the model has never seen.
AUDIO_CHANNELS = 2
AUDIO_LATENTS_PER_SECOND = 40
FPS = 24
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0

VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2       # AdaLN modality rows — a checkpoint contract
MODALITY_NUM = 3


def audio_latents_for_frames(num_frames: int = 1) -> int:
    """Audio latents covering `num_frames` PIXEL frames (24 fps video, 40 Hz audio)."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def pixel_frames_for_latent(latent_t: int) -> int:
    """Latent frames -> the pixel frames they encode, summed off the (1, 4, 4, 4, 4) token
    grid: 5n+2 latents <-> 17n+5 pixels (the trained / ComfyUI grid: 1, 5, 22, 39, 56, ...).

    A PARTIAL temporal group is exact arithmetic too (3 -> 9, 4 -> 13, 6 -> 18) — the audio
    clock follows the summed pixel count, so nothing misaligns — but it is off-distribution:
    neither the reference trainer nor ComfyUI ever builds one. Only the Repair Studio's
    short-clip lengths ask for it (`latent_frames_for_pixels(..., exact=True)`); the trainer
    snaps down and never does."""
    t = int(latent_t)
    if t < 1:
        raise ValueError(f"latent_t={latent_t} must be >= 1")
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(t))


def latent_frames_for_pixels(num_frames: int, exact: bool = False) -> int:
    """Pixel frames -> latent frames, snapping DOWN onto the 17n+5 grid like the reference
    trainer does (align_num_frames_down): 5..21 -> 2 latents, 22..38 -> 7, 124 -> 37.
    num_frames == 1 is the still case -> 1 latent.

    exact=True (Repair Studio short clips): the count must sit on the token grid itself —
    1, 5, 9, 13, 17, 18, 22, 26, ... — and comes back as that many latents, partial temporal
    group included (9 -> 3, 13 -> 4); anything else raises instead of snapping."""
    if exact:
        n = int(num_frames)
        total, t = 0, 0
        while total < n:
            total += FRAME_PER_TOKEN[t % 5]
            t += 1
        if total != n:
            raise ValueError(f"num_frames={num_frames} is not on the (1, 4, 4, 4, 4) token "
                             f"grid (1, 5, 9, 13, 17, 18, 22, ...)")
        return max(1, t)
    if num_frames <= 1:
        return 1
    n = max(5, int(num_frames))
    n -= (n - 5) % 17                       # snap down onto 17n+5
    return 5 * (n - 5) // 17 + 2


def shift_sigma(sigma, shift: float):
    """Exponential timeshift: shift*sigma / (1 + (shift-1)*sigma)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def remap_sigma(sigma, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT):
    """A sigma on the `from_shift` schedule -> the same schedule POSITION on `to_shift`.

    The two streams denoise together: video runs shift 12, audio shift 3, and this closed form
    is what keeps them at the same underlying point. Used for the audio rows' timestep."""
    return shift_sigma(sigma / (from_shift + sigma * (1.0 - from_shift)), to_shift)


def sigma_remap_slope(sigma, from_shift: float = VIDEO_SIGMA_SHIFT,
                      to_shift: float = AUDIO_SIGMA_SHIFT):
    """d(sigma_to)/d(sigma_from) at the same base-grid point — comfy's `time_shift_slope`.

    Scaling the audio head's velocity by this puts the audio stream onto the VIDEO sigma grid,
    which is how the reference lets one sampler drive both streams: comfy packs audio and video
    into a single latent (NestedTensor) and returns `(-slope)*audio_out`, so res_multistep's
    second-order update applies to the audio rows too. Integrating the audio separately on its
    own grid is a first-order scheme for a stream the video attends to at every step."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return ((to_shift * (1.0 + (from_shift - 1.0) * base) ** 2)
            / (from_shift * (1.0 + (to_shift - 1.0) * base) ** 2))


def ref_row_count(refs) -> int:
    """Total packed rows contributed by a list of (latent_h, latent_w) reference images."""
    return sum((rh // 2) * (rw // 2) for rh, rw in (refs or ()))


def image_position_ids(text_len, latent_h, latent_w, num_audio_latents: int = 0,
                       refs=None, latent_t: int = 1, keyframes=None) -> torch.Tensor:
    """3-axis (t, h, w) position ids for a [text | keyframes | refs | audio | video] sequence.

    Text rows: t = 0..text_len-1, h=w=0 — so prompt length shifts the whole media clock.
    Keyframe rows (fl2va first/last-frame conditioning, `keyframes=[frame_index, ...]`): one
    condition frame each on the TARGET's own grid, pinned to the moment it conditions —
    t = target_origin + FRAME_RESCALE * frame_index, where target_origin is the cursor AFTER
    the references (the reference packs keyframe cond rows before the refs but anchors them
    on the target timeline: comfy/ldm/minimax/model.py::PackedLayout, cond_t). frame 0 is
    the clip's first pixel frame; sum(_video_t_spans(latent_t)) == FRAME_RESCALE * frames,
    so index frames-1 lands on the last pixel frame exactly.
    Reference rows (r2v): each reference image contributes its OWN area-normalized frame grid at
    t = cursor, and advances the cursor by 1.0. Ordered right after the text, matching the
    reference's segment order [text | cond | refs | target audio | target video]
    (comfy/ldm/minimax/model.py::PackedLayout).
    Audio rows: t = cursor + 0..A-1 repeated per channel, h = 0, w pinned to the frame grid's
    first column for channel 0 and its last for channel 1 (the reference's stereo convention).
    Video rows: latent_t frames, t-major (matching patchify_video's row order), each frame's
    rows at t = _video_t_grid(latent_t, cursor)[k] — the (1,4,4,4,4)x5/3 spans that keep the
    video clock aligned with the 40 Hz audio clock (17 pixel frames = 5 latents = 28.33 rotary
    units ~ 28 audio latents). latent_t=1 is the still case and reproduces the old layout
    exactly.

    NOTE the cursor: with no references it is text_len and this is exactly the old layout, but
    every reference image SHIFTS THE TARGET'S TEMPORAL ORIGIN by +1.0. Pinning the target at
    text_len while packing references would leave the target sitting on top of the first
    reference, which is the kind of error that produces a plausible-looking but wrong teacher.

    Returns [S, 3] float64."""
    frame = _frame_grid(latent_h, latent_w)                 # [(h//2)*(w//2), 2]
    frame_rows = frame.shape[0]
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)

    rows = [text]
    cursor = float(text_len)
    ref_rows = []
    for rh, rw in (refs or ()):
        r_frame = _frame_grid(rh, rw)
        g = torch.empty(r_frame.shape[0], 3, dtype=torch.float64)
        g[:, 0] = cursor
        g[:, 1:] = r_frame
        ref_rows.append(g)
        cursor += 1.0
    # Keyframe cond rows sit BETWEEN text and refs in the sequence, but their clock is the
    # target's (post-ref cursor) — build them once the cursor is final, insert them first.
    for idx in (keyframes or ()):
        g = torch.empty(frame_rows, 3, dtype=torch.float64)
        g[:, 0] = cursor + FRAME_RESCALE * float(idx)
        g[:, 1:] = frame
        rows.append(g)
    rows.extend(ref_rows)

    if num_audio_latents:
        w_axis = _axis_from_sqrt_area(latent_w, 2, math.sqrt(latent_h * latent_w))
        aud = torch.zeros(num_audio_latents * AUDIO_CHANNELS, 3, dtype=torch.float64)
        aud[:, 0] = (cursor + torch.arange(num_audio_latents, dtype=torch.float64)
                     ).repeat(AUDIO_CHANNELS)
        aud[:, 2] = torch.cat([torch.full((num_audio_latents,), float(w_axis[0]), dtype=torch.float64),
                               torch.full((num_audio_latents,), float(w_axis[-1]), dtype=torch.float64)])
        rows.append(aud)

    t_grid = _video_t_grid(latent_t, cursor)                # [latent_t], t_grid[0] = cursor
    vid = torch.empty(latent_t * frame_rows, 3, dtype=torch.float64)
    vid[:, 0] = t_grid.repeat_interleave(frame_rows)        # t-major, like patchify_video
    vid[:, 1:] = frame.repeat(latent_t, 1)
    rows.append(vid)
    return torch.cat(rows, dim=0)


# --- RoPE --------------------------------------------------------------------------------

def rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor):
    """position_ids [S,3] (t,h,w) x inv_freq[16] -> cos, sin each [S, 48].

    Split-half convention: the 48 angles are [t(16) | h(16) | w(16)]; the attention rope
    rotates head dims (0..47) against (48..95) using these, leaving dims 96..127 untouched."""
    pos = position_ids.to(torch.float32)
    ang = (pos.unsqueeze(-1) * inv_freq.to(torch.float32).view(1, 1, -1)).reshape(pos.shape[0], -1)  # [S,48]
    return torch.cos(ang), torch.sin(ang)


def apply_rope_split_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, heads, head_dim]; cos/sin [S, rot_half]. Rotates the first 2*rot_half dims in
    split-half style (dim i with dim i+rot_half); the tail passes through unrotated."""
    rot_half = cos.shape[-1]
    rot = 2 * rot_half
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :rot_half], xr[..., rot_half:]
    c = cos.unsqueeze(1)                                    # [S,1,rot_half] -> broadcast over heads
    s = sin.unsqueeze(1)
    rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([rotated, xp], dim=-1)


# --- blocks ------------------------------------------------------------------------------

class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out, bias=True)

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # The sinusoid is built fp32 for precision; the projections load at the compute
        # dtype (bf16) on a bf16 checkpoint — feed them their own dtype or F.linear
        # rejects the mix. (Pre-existing: the bf16-checkpoint path could never forward;
        # unnoticed because everyone trains the pruned int8 file, which has no
        # time_embedder at all. Found by the NF4-ring bf16-shape test, 25 Aug.)
        return self.proj_out(F.silu(self.proj_in(emb.to(self.proj_in.weight.dtype))))


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x, cos=None, sin=None):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
        v = v.view(s, self.heads, self.head_dim)
        if cos is not None:
            q = apply_rope_split_half(q, cos, sin)
            k = apply_rope_split_half(k, cos, sin)
        # [S, H, D] -> [1, H, S, D] for SDPA
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = h3_attention(q, k, v)                         # [1, H, S, D]
        out = out.squeeze(0).transpose(0, 1).reshape(s, self.heads * self.head_dim)
        return self.out_proj(out)


# ---- optional INT8 attention (comfy-kitchen) ------------------------------------------
# NVIDIA's pure-INT8 SDPA from the comfy-kitchen wheel (Apache-2.0): Q / K / V to int8 after a
# Hadamard rotation, P in uint8, softmax maths in fp32. Measured on a 5090 (5 Sep): 3.2x
# PyTorch's attention at 1.3k tokens, 6-7x at 9k-18k (a 1024² × 56-frame clip's attention
# goes from 6.3 s to 0.84 s per pass), at ~1.6% relative error per call against fp32 (the
# bf16 path is 0.23%). An opt-in the H3 Repair Studio sets per render; nothing else turns it
# on, and it is inference-only (never under grad). Three fallbacks, each announced once: the
# package missing / not importable (AMD builds skip it), the kernel unavailable on this GPU
# (compute < 7.5), a call raising at run time.
_INT8_ATTN = {"wanted": False, "checked": False, "fn": None}


def set_int8_attention(on: bool) -> None:
    _INT8_ATTN["wanted"] = bool(on)


def int8_attention_wanted() -> bool:
    return bool(_INT8_ATTN["wanted"])


def _int8_attention_fn():
    st = _INT8_ATTN
    if not st["checked"]:
        st["checked"] = True
        try:
            import comfy_kitchen as _ck
            if _ck.int8_attention_is_available():
                st["fn"] = _ck.int8_attention
                print("[h3] int8 attention: comfy-kitchen kernel active", flush=True)
            else:
                print("[h3] int8 attention: comfy-kitchen has no kernel for this GPU — "
                      "PyTorch attention instead", flush=True)
        except Exception as _e:
            print(f"[h3] int8 attention: comfy-kitchen not available "
                  f"({type(_e).__name__}) — PyTorch attention instead", flush=True)
    return st["fn"]


def int8_kernel_available() -> bool:
    """True when the kernel is really there (import + GPU check + no run-time failure) —
    what the studio's status line combines with its own tick. Not tied to the per-render
    switch, which is only on while a render runs."""
    return _int8_attention_fn() is not None


def h3_attention(q, k, v):
    """[1, H, S, D] SDPA — the int8 kernel when asked for and available, else PyTorch's."""
    if _INT8_ATTN["wanted"] and not torch.is_grad_enabled():
        fn = _int8_attention_fn()
        if fn is not None:
            try:
                return fn(q, k, v)
            except Exception as _e:
                _INT8_ATTN["fn"] = None
                print(f"[h3] int8 attention failed ({type(_e).__name__}: {_e}) — PyTorch "
                      "attention for the rest of the run", flush=True)
    return F.scaled_dot_product_attention(q, k, v)


class MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)                       # swiglu


class AdalnProj(nn.Module):
    """t_dim -> `expand` modulation tensors, each [M*modalities, hidden]."""
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, mod_row):
    """h [S,hidden] modulated per row: h*(1+scale[row]) + shift[row]. mod_row [S] long indexes
    the [modalities, hidden] modulation tensors. Fully out-of-place (autograd-safe)."""
    return h * (1.0 + scale[mod_row].to(h.dtype)) + shift[mod_row].to(h.dtype)


def _mod_gate(x, gate, other, mod_row):
    """Gated residual add: x + other*gate[row], per row. Out-of-place."""
    return x + other * gate[mod_row].to(x.dtype)


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu)

    def forward(self, x, t_emb, mod_row, cos, sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_row)
        x = _mod_gate(x, gate_msa, self.attn(h, cos=cos, sin=sin), mod_row)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_row)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_row)


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu)
        # fp32 output heads, matching the checkpoint's fp32 island
        self.video_out = nn.Linear(hidden, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def forward(self, x_video, t_emb, t_index: int = 0):
        """x_video [n_video, hidden] — just the video segment.

        modalities=1, so the table is one row per distinct timestep: `t_index` selects the
        video rows' own timestep."""
        shift, scale = self.adaln_proj(t_emb)
        hv = self.norm(x_video) * (1.0 + scale[t_index]) + shift[t_index]
        return self.video_out(hv.to(self.video_out.weight.dtype))

    def forward_audio(self, x_audio, t_emb, t_index: int = 0):
        """Same shared norm+modulation, but at the AUDIO rows' timestep and through the audio
        head — the reference's final layer modulates every row by its own timestep and then
        applies the per-modality head. Used by joint audio denoising in the sampler."""
        shift, scale = self.adaln_proj(t_emb)
        ha = self.norm(x_audio) * (1.0 + scale[t_index]) + shift[t_index]
        return self.audio_out(ha.to(self.audio_out.weight.dtype))


# --- model -------------------------------------------------------------------------------

class ForwardAborted(RuntimeError):
    """Raised between blocks when the module's `_abort_event` (set by the owner — the Repair
    Studio engine's cancel) is set: a cancel lands within one block (~0.15 s) instead of at
    the next sampler step (up to ~7 s at 56 frames full size)."""


def _run_block(blocks, i, swap_from, h, t_emb, mod_row, cos, sin):
    """One DiT block, optionally CPU-parked ('block swap').

    Lives at module level (not a closure) and is the unit torch.utils.checkpoint re-runs in
    backward — which is exactly what makes swap work with bnb NF4: the packed weights move to
    GPU just-in-time HERE, so the recompute pass re-fetches them too.

    TWO park sites, because non-reentrant checkpoint EARLY-STOPS its recompute the moment the
    needed tensors are recovered — the tail of this function never executes during backward
    (measured: relying on the tail alone re-accumulated every swapped block on GPU, +9 GB by
    step 3). Backward visits blocks in reverse, so when block i recomputes, block i+1's
    backward has already finished — parking the SUCCESSOR at entry is the site the early-stop
    can't skip. The self-park at the tail still runs in the plain forward pass (no early-stop
    there), keeping the forward's footprint at ~one swapped block too."""
    swapped = i >= swap_from
    offloader = getattr(blocks[i], "_h2d_offloader", None) if swapped else None
    if offloader is not None:
        # H2D-only streaming (int8 ConvRot, #73): the block's bytes are prefetched into a GPU
        # ring by a copy stream; this only WAITS for them. No parking dance — blocks never
        # physically move, so the checkpoint early-stop has nothing to skip, and a recompute
        # arriving after the ring rotated is healed inside wait_for_block.
        offloader.wait_for_block(i)
        return blocks[i](h, t_emb, mod_row, cos, sin)
    if swapped:
        blocks[i].to(h.device)
        if i + 1 < len(blocks):
            blocks[i + 1].to("cpu")            # successor's backward is done (reverse order)
    out = blocks[i](h, t_emb, mod_row, cos, sin)
    if swapped:
        blocks[i].to("cpu")                     # forward-pass park (skipped by recompute)
    return out


class MiniMaxH3DiT(nn.Module):
    """Image-only training forward for the MiniMax H3 DiT. Names match the bf16 checkpoint."""

    def __init__(self, config: Optional[MiniMaxH3Config] = None):
        super().__init__()
        c = config or MiniMaxH3Config()
        self.config = c
        self.hidden_size = c.hidden_size
        self.patch_size = tuple(c.patch_size)
        self.latents_dim = c.latents_dim
        video_patch_dim = c.latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, c.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(c.audio_latents_dim, c.hidden_size, bias=True)
        self.condition_proj = nn.Linear(c.text_dim, c.hidden_size, bias=True)
        # Full model: a timestep MLP. Pruned: a sampled curve table (see the config field).
        self.pruned_adaln = c.adaln_t_table_size is not None
        # Set by the loader when it keeps the AdaLN projections fp32 (pruned checkpoints, as
        # ComfyUI does). The forward then hands them an fp32 t_emb instead of demoting it.
        self.adaln_fp32 = False
        _silu = not self.pruned_adaln              # the table already absorbs the nonlinearity
        if self.pruned_adaln:
            self.time_embedder = None
            self.register_buffer("adaln_t_table",
                                 torch.zeros(c.adaln_t_table_size, c.time_embed_dim))
        else:
            self.time_embedder = TimeEmbedder(c.timestep_input_dim, c.time_embed_hidden_size,
                                              c.time_embed_dim)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq",
                                  1.0 / (10000.0 ** (torch.arange(0, c.rope_inv_freq_len, dtype=torch.float32)
                                                     / c.rope_inv_freq_len)))
        self.token_refiner = TokenRefiner(c.token_refiner_num_layers, c.hidden_size, c.num_attention_heads,
                                          c.attention_head_dim, c.ffn_hidden_size, c.norm_eps, c.qk_norm_eps,
                                          c.final_norm_eps)
        self.blocks = nn.ModuleList([
            DiTBlock(c.hidden_size, c.num_attention_heads, c.attention_head_dim, c.ffn_hidden_size,
                     c.time_embed_dim, c.norm_eps, c.qk_norm_eps, apply_silu=_silu)
            for _ in range(c.num_layers)])
        self.final_layer = FinalLayer(c.hidden_size, c.time_embed_dim, video_patch_dim,
                                      c.audio_latents_dim, c.final_norm_eps, apply_silu=_silu)
        self._gradient_checkpointing = False
        self._swap_from = len(self.blocks)          # blocks >= this index live on CPU between uses
        self._h2d_requested = False                 # H2D-only streaming (#73) — int8 bases
        self._h2d_offloader = None
        # Pack the reference's silence audio block (see the AUDIO_* constants above). Off is an
        # escape hatch for A/B only — the base was trained with these rows present.
        self.pack_audio_rows = True

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """[M] cleanness values in [0,1] -> [M, time_embed_dim] modulation input.

        Full model: the sinusoid+MLP time embedder. Pruned: linear interpolation into the
        curve table (t=0 is row 0, t=1 the last row)."""
        if not self.pruned_adaln:
            return self.time_embedder(t)
        table = self.adaln_t_table.float()
        pos = t.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        # the table can be CPU-parked under offloading while t arrives on cuda
        pos = pos.to(table.device)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=table.shape[0] - 1)
        frac = (pos - lo.float()).unsqueeze(1)
        return (table[lo] * (1.0 - frac) + table[hi] * frac).to(t.device)

    def enable_gradient_checkpointing(self, enabled: bool = True):
        """Recompute each block in backward instead of storing activations. Required when block
        swap is active (without it, autograd's saved tensors would pin every swapped block's GPU
        weights through the whole backward, saving nothing)."""
        self._gradient_checkpointing = enabled

    def enable_block_swap(self, blocks_to_swap: int, h2d_only=None, ring_size: int = 2):
        """Park the LAST n blocks on CPU; each is moved to GPU just-in-time for its forward
        (and again for its checkpoint recompute) then parked again. bnb NF4 weights stay packed
        (uint8, 0.5 B/param) through the moves, so a parked block costs ~¼ of its GPU footprint
        in system RAM and one PCIe round-trip per pass.

        h2d_only=True streams the frozen tensors host→device through a ring buffer on a
        copy stream instead — no writeback, the prefetch overlaps compute. The ring class
        is picked by MODULE TYPE in the swapped tail: ConvRot int8 → H3Int8H2DOffloader
        (#73 @rintic-13); bnb Linear4bit (NF4) → H3NF4H2DOffloader (@mabseyuk) — type,
        not residency, so a bare re-entry after a full CPU park can't mis-dispatch. Ring
        construction failure falls back to classic parking with a warning rather than
        killing the run. h2d_only=None KEEPS the previously requested mode: the
        preview-decode path parks the whole DiT and re-calls this bare, and it must come back
        in the mode it left, with the stale offloader rebuilt (a whole-model .to() replaces
        every tensor storage this machinery caches)."""
        if h2d_only is None:
            h2d_only = getattr(self, "_h2d_requested", False)
        self._h2d_requested = bool(h2d_only)

        # Any existing offloader is stale the moment blocks were moved wholesale — and even
        # when they weren't, rebuilding is cheap next to leaking a set of backward hooks.
        old = getattr(self, "_h2d_offloader", None)
        if old is not None:
            old.release()
            self._h2d_offloader = None
            for blk in self.blocks:
                if hasattr(blk, "_h2d_offloader"):
                    blk._h2d_offloader = None
            # Drop the reference NOW: the old ring's CPU staging dict would otherwise
            # stay alive through the new ring's construction below, doubling the pinned
            # staging at every preview rebuild (up to ~13 GB extra on a 40-block bf16
            # plan — audit, 25 Aug). The module bindings keep each old flat alive only
            # until its block rebinds, so the peak drops to ~one block's worth.
            old = None

        n = max(0, min(int(blocks_to_swap), len(self.blocks) - 2))   # keep >=2 resident
        self._swap_from = len(self.blocks) - n

        if self._h2d_requested and n > 0:
            # Dispatch by module TYPE in the swapped tail, not residency: after a full
            # park every tensor is CPU-resident, and the bare re-entry from
            # restore_parked_dit must still find the same ring class it left.
            _tail_types = {m.__class__.__name__
                           for i in range(self._swap_from, len(self.blocks))
                           for m in self.blocks[i].modules()}
            try:
                if "ConvRotInt8Linear" in _tail_types:
                    from .h3_h2d_offload import H3Int8H2DOffloader
                    _cls = H3Int8H2DOffloader
                    dev = next((m.qdata.device for b in self.blocks for m in b.modules()
                                if m.__class__.__name__ == "ConvRotInt8Linear"
                                and m.qdata.is_cuda), None) or torch.device("cuda")
                elif "Linear4bit" in _tail_types:
                    from .h3_nf4_h2d_offload import H3NF4H2DOffloader
                    _cls = H3NF4H2DOffloader
                    # Derive the device from the resident head rather than assuming
                    # device 0 (the int8 branch derives from live qdata for the same
                    # reason) — a non-default CUDA index would otherwise put the ring
                    # slots and copy stream on the wrong card.
                    dev = next((p.device for p in self.blocks[0].parameters()
                                if p.is_cuda), None) or torch.device("cuda")
                elif "HQQ4bitLinear" in _tail_types:
                    from .h3_hqq_h2d_offload import H3HQQH2DOffloader
                    _cls = H3HQQH2DOffloader
                    dev = next((m.W_q.device for b in self.blocks for m in b.modules()
                                if m.__class__.__name__ == "HQQ4bitLinear"
                                and m.W_q.is_cuda), None) or torch.device("cuda")
                else:
                    raise RuntimeError(
                        f"no ring-streamable modules in the swapped tail "
                        f"(saw {sorted(_tail_types)[:6]}...)")
                self._h2d_offloader = _cls(self.blocks, self._swap_from,
                                           device=dev, ring_size=ring_size)
                self._h2d_offloader.move_static_weights_to_gpu()
                self._h2d_offloader.prepare()
                for blk in self.blocks:
                    blk._h2d_offloader = self._h2d_offloader
                return n
            except Exception as _e:
                # The ring is strictly better when constructible; classic parking is the
                # safety net, never a crash. _h2d_requested resets so bare re-entries
                # don't retry a construction that already failed.
                logger.warning("[vram] H2D ring construction failed (%s: %s) — falling "
                               "back to classic parking swap for this run: blocks will "
                               "cross PCIe every step, several times slower. Lower "
                               "Target Megapixels or free VRAM to need less swap.",
                               type(_e).__name__, _e)
                # A failure AFTER construction (move_static/prepare OOM — the case this
                # fallback exists for) leaves a LIVE half-built ring: its backward hooks
                # would fire mid-training over classic-parked blocks and rebind their
                # weights to ring views (crash or silent corruption), and its pinned
                # staging would stay resident on the exact machine that just ran out.
                # release() removes the hooks, rebinds modules to their CPU masters, and
                # synchronizes the copy stream — which also makes the .to("cpu") parking
                # below race-free (review, 25 Aug).
                _bad = self._h2d_offloader
                self._h2d_offloader = None
                if _bad is not None:
                    try:
                        _bad.release()
                    except Exception:
                        pass
                    _bad = None
                self._h2d_requested = False

        self._h2d_offloader = None
        for i in range(self._swap_from, len(self.blocks)):
            self.blocks[i].to("cpu")
        return n

    def forward(self, video_latent: torch.Tensor, t: torch.Tensor,
                text_embeds: torch.Tensor, audio_noise: torch.Tensor = None, *,
                audio_rows: torch.Tensor = None, return_audio: bool = False,
                ref_latents=None, text_token_tags: torch.Tensor = None, seed: int = 0,
                visual_cond_noise_aug: float = VISUAL_COND_TIMESTEP, keyframes=None):
        """
        keyframes    : optional list of (frame_index, latent [1, C, 1, h, w]) — fl2va first /
                       last-frame conditioning. Each is packed as a condition frame on the
                       TARGET grid right after the text (before any refs), noise-augmented at
                       VISUAL_COND_TIMESTEP with a role-pinned seed (H3Studio's convention:
                       last frame 0, first frame 1, others 100+index — so the two never share
                       a noise field), tagged video, pinned near clean, never denoised, and
                       placed on the target clock at origin + FRAME_RESCALE * frame_index.
                       Full strength only (no blend dial) — Repair Studio's contract.
        video_latent : [1, C=latents_dim, T, H, W] — a still (T=1, the keyframe layout) or a
                       clip (T on the 5n+2 latent grid; position ids and audio rows follow).
        t            : scalar or [1] flow time in [0, 1] (the value fed to the time embedder;
                       the trainer owns noising and the flow target).
        text_embeds  : [1, L, text_dim] Qwen3-VL hidden states (or already [1, L, hidden]).
        audio_noise  : optional [A*2, audio_latents_dim] unit noise for SILENCE rows, scaled by
                       the audio sigma internally. None draws fresh — what training wants (the
                       reference redraws every step; image datasets have no soundtrack, so
                       silence noised at sigma_a rides along without contributing loss).
        audio_rows   : optional [A*2, audio_latents_dim] EXPLICIT row content, used verbatim
                       (no sigma scaling) — the sampler's joint audio denoising evolves these
                       across steps exactly like the reference pipeline. Overrides audio_noise.
        return_audio : also return the audio head's prediction for those rows, so a sampler
                       can step them.
        ref_latents  : optional list of [1, C, 1, h, w] NORMALIZED reference latents (r2v). Each
                       is packed as condition rows right after the text: noise-augmented, tagged
                       video, pinned near clean, and never denoised. Their presence shifts the
                       target's temporal origin (see image_position_ids).
        text_token_tags : optional [L] per-row modality tags for the text rows. Required when the
                       conditioning carries `<Picture i>` vision blocks, whose rows are VIDEO —
                       without it every text row is tagged TEXT and the vision rows are modulated
                       as the wrong modality.
        seed         : RNG seed for the condition noise augmentation. The reference restarts the
                       same stream for every condition, so this is deliberately not per-ref.
        returns      : [1, C, T, H, W] video prediction — or (video, audio [A*2, adim]) when
                       return_audio.
        """
        if video_latent.shape[0] != 1:
            raise ValueError("MiniMax H3 image training is batch size 1 (pack one image per step)")
        device = video_latent.device
        dtype = text_embeds.dtype
        _, _, latent_t, lat_h, lat_w = video_latent.shape
        text_len = text_embeds.shape[1]

        # text: condition_proj -> refiner (skip if already hidden-width)
        text_states = text_embeds[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))
        # video: patchify -> patch proj
        video_rows = patchify_video(video_latent.to(torch.float32), self.patch_size)
        # Rows are built fp32 for precision, but the projection loads at whatever dtype
        # the checkpoint stored (fp32 island on the pruned file, bf16 elsewhere) — feed
        # it its own dtype, same rule as TimeEmbedder (review, 25 Aug).
        video_embed = self.video_patch_proj(
            video_rows.to(self.video_patch_proj.weight.dtype)).to(dtype)

        # r2v reference condition rows. Same patchify + projection as the target, but blended
        # with a trace of noise first: r = aug*r + (1-aug)*noise. The reference restarts the SAME
        # CPU generator for every condition rather than drawing one continuous stream, so this
        # reseeds inside the loop — a shared stream would give different rows for ref 2 onward.
        ref_shapes, ref_embed = [], None
        if ref_latents:
            _rows = []
            for z in ref_latents:
                r = patchify_video(z.to(device=device, dtype=torch.float32), self.patch_size)
                if visual_cond_noise_aug < 1.0:
                    gen = torch.Generator("cpu").manual_seed(int(seed))
                    noise = torch.randn(r.shape, generator=gen, dtype=torch.float32).to(device)
                    r = visual_cond_noise_aug * r + (1.0 - visual_cond_noise_aug) * noise
                _rows.append(r)
                ref_shapes.append((z.shape[-2], z.shape[-1]))
            ref_embed = self.video_patch_proj(
                torch.cat(_rows, dim=0).to(self.video_patch_proj.weight.dtype)).to(dtype)

        # fl2va keyframe condition rows: the same row machinery as a reference, on the
        # target's own grid, with a role-pinned noise seed per frame (see the docstring).
        kf_indices, kf_embed = [], None
        if keyframes:
            _pixel_frames = pixel_frames_for_latent(latent_t)
            _rows = []
            for idx, z in keyframes:
                idx = int(idx)
                if z.shape[-2] != lat_h or z.shape[-1] != lat_w:
                    raise ValueError(f"keyframe latent is {tuple(z.shape[-2:])} but the target "
                                     f"grid is {(lat_h, lat_w)} — keyframes must be encoded at "
                                     f"the clip's own size")
                if not (0 <= idx < _pixel_frames):
                    raise ValueError(f"keyframe index {idx} is outside the clip's "
                                     f"{_pixel_frames} pixel frames")
                r = patchify_video(z.to(device=device, dtype=torch.float32), self.patch_size)
                if visual_cond_noise_aug < 1.0:
                    _role = 1 if idx == 0 else (0 if idx == _pixel_frames - 1 else 100 + idx)
                    gen = torch.Generator("cpu").manual_seed(_role)
                    noise = torch.randn(r.shape, generator=gen, dtype=torch.float32).to(device)
                    r = visual_cond_noise_aug * r + (1.0 - visual_cond_noise_aug) * noise
                _rows.append(r)
                kf_indices.append(idx)
            kf_embed = self.video_patch_proj(
                torch.cat(_rows, dim=0).to(self.video_patch_proj.weight.dtype)).to(dtype)

        # audio: silence (x0 = 0) noised on the audio schedule at the same schedule position as
        # the video rows. Present because the base model has never seen a pack without it.
        t_val = t.reshape(-1)[:1].to(torch.float32) if torch.is_tensor(t) else torch.tensor([float(t)], device=device)
        t_val = t_val.to(device)
        # PIXEL frames, not latent frames: a 37-latent clip is 124 pixel frames and needs
        # round(124/24*40)=207 audio latents, not round(37/24*40). Identical at T=1.
        n_audio_latents = (audio_latents_for_frames(pixel_frames_for_latent(latent_t))
                          if self.pack_audio_rows else 0)
        audio_embed = None
        if n_audio_latents:
            sigma_v = (1.0 - t_val).clamp(0.0, 1.0)
            sigma_a = remap_sigma(sigma_v)
            t_audio = 1.0 - sigma_a
            if audio_rows is not None:
                # Checked, because the sequence and its POSITION GRID are built from two
                # different numbers: the rows come from the cache, the grid from n_audio_latents
                # above. They agree only while the cache and this file agree about a clip's
                # length, and that is exactly the sort of contract that drifts across files. A
                # silent desync would train the audio stream against shifted positions.
                _want = n_audio_latents * AUDIO_CHANNELS
                if audio_rows.shape[0] != _want:
                    raise ValueError(
                        f"audio_rows has {audio_rows.shape[0]} rows but this {latent_t}-latent-"
                        f"frame clip ({pixel_frames_for_latent(latent_t)} pixel frames) packs "
                        f"{_want} — {n_audio_latents} latents x {AUDIO_CHANNELS} channels. The "
                        f"cached audio does not match the cached video.")
                _arows = audio_rows.to(device=device, dtype=torch.float32)
            else:
                eps = audio_noise
                if eps is None:
                    eps = torch.randn(n_audio_latents * AUDIO_CHANNELS, self.config.audio_latents_dim,
                                      device=device, dtype=torch.float32)
                _arows = sigma_a * eps.to(device=device, dtype=torch.float32)
            audio_embed = self.audio_patch_proj(
                _arows.to(self.audio_patch_proj.weight.dtype)).to(dtype)

        # pack [text | keyframes | refs | audio | video] — the reference's segment order
        parts = ([text_states.to(dtype)]
                 + ([kf_embed] if kf_embed is not None else [])
                 + ([ref_embed] if ref_embed is not None else [])
                 + ([audio_embed] if audio_embed is not None else [])
                 + [video_embed])
        h = torch.cat(parts, dim=0)
        seq_len = h.shape[0]
        n_video = video_embed.shape[0]
        n_kf = 0 if kf_embed is None else kf_embed.shape[0]
        n_ref = 0 if ref_embed is None else ref_embed.shape[0]
        n_cond = n_kf + n_ref                        # every condition row: keyframes then refs
        n_audio = 0 if audio_embed is None else audio_embed.shape[0]
        audio_start = text_len + n_cond
        video_start = audio_start + n_audio

        # One modulation row-set per DISTINCT timestep (video/text at t, audio at t_audio),
        # indexed per row as `timestep_index * 3 + modality_tag` — the reference's exact
        # (timestep, modality) table. Tags: video 0, text 1, audio 2.
        # Condition rows sit at max(t, aug) — near clean whatever the sampler is doing, so on a
        # late step they can share the video timestep and on an early one they need their own.
        t_parts = [t_val] + ([t_audio] if audio_embed is not None else [])
        if n_cond:
            t_parts.append(torch.maximum(t_val, torch.tensor([visual_cond_noise_aug],
                                                             device=device, dtype=torch.float32)))
        t_all = torch.cat(t_parts) if len(t_parts) > 1 else t_val
        uniq, inverse = torch.unique(t_all, sorted=True, return_inverse=True)
        # Kept fp32 when the loader kept the AdaLN projections fp32 (ComfyUI's curve-checkpoint
        # dtype). _mod_scale_shift / _mod_gate cast back to the activation dtype at the point of
        # use, exactly as the reference does, so only the modulation gains the precision.
        t_emb = self._time_embedding(uniq)                                        # [M, t_dim]
        if not self.adaln_fp32:
            t_emb = t_emb.to(dtype)
        tags = torch.full((seq_len,), VIDEO_TAG, dtype=torch.long, device=device)
        tags[:text_len] = TEXT_TAG
        if text_token_tags is not None:
            _tt = text_token_tags.reshape(-1).to(device=device, dtype=torch.long)
            if _tt.numel() != text_len:
                raise ValueError(f"text_token_tags has {_tt.numel()} rows for {text_len} text rows")
            tags[:text_len] = _tt                    # vision-block rows carry VIDEO_TAG
        row_t_index = torch.full((seq_len,), int(inverse[0]), dtype=torch.long, device=device)
        if n_cond:                                   # cond rows: video tag (already), cond timestep
            row_t_index[text_len:audio_start] = int(inverse[-1])
        if audio_embed is not None:
            tags[audio_start:video_start] = AUDIO_TAG
            row_t_index[audio_start:video_start] = int(inverse[1])
        mod_row = row_t_index * MODALITY_NUM + tags
        video_t_index = int(inverse[0])

        # rope
        pos = image_position_ids(text_len, lat_h, lat_w, n_audio_latents,
                                 refs=ref_shapes or None, latent_t=latent_t,
                                 keyframes=kf_indices or None).to(device)
        cos, sin = rope_cos_sin(pos, self.rope.inv_freq.to(device))
        cos, sin = cos.to(dtype), sin.to(dtype)

        # Gate on grad-enabled only, NOT self.training: the frozen base stays in eval() during
        # LoRA training (grads flow through it regardless), so a training-mode gate would
        # silently disable checkpointing for exactly the runs that need it.
        use_ckpt = self._gradient_checkpointing and torch.is_grad_enabled()
        for i in range(len(self.blocks)):
            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    _run_block, self.blocks, i, self._swap_from, h, t_emb, mod_row, cos, sin,
                    use_reentrant=False)
            else:
                _ev = getattr(self, '_abort_event', None)
                if _ev is not None and _ev.is_set():
                    raise ForwardAborted()
                h = _run_block(self.blocks, i, self._swap_from, h, t_emb, mod_row, cos, sin)
            # H2D streaming: the block's forward is done — free its ring slot and start the
            # copy for the block ring_size ahead, overlapping the next blocks' compute. Sits
            # OUTSIDE the checkpoint call: it must run once per forward pass, not again per
            # recompute (backward prefetch is the offloader's hooks' job).
            _off = getattr(self, "_h2d_offloader", None)
            if _off is not None and i >= self._swap_from:
                _off.submit_move_blocks_forward(i)

        v = self.final_layer(h[video_start:], t_emb, video_t_index)              # [n_video, video_patch_dim]
        out = unpatchify_video(v, latent_t, lat_h // self.patch_size[1], lat_w // self.patch_size[2],
                               self.latents_dim, self.patch_size)
        out = out.to(video_latent.dtype)
        if return_audio:
            if n_audio:
                audio_t_index = int(inverse[1])
                a = self.final_layer.forward_audio(h[audio_start:video_start], t_emb, audio_t_index)
            else:
                a = None
            return out, a
        return out

    # (Entry class lives at module level: H3ActivationCacheEntry, below the DiT.)
    def forward_cached(self, video_latent: torch.Tensor, t: torch.Tensor,
                       text_embeds: torch.Tensor, audio_rows: torch.Tensor = None,
                       return_audio: bool = False, *,
                       resume_from=None, cached=None, new_cache=None,
                       cache_device: str = "cpu", keyframes=None):
        """Inference forward with per-block activation caching (Repair Studio Turbo Preview).

        `resume_from` = the earliest changed main-block index (or None for a full pass). When
        resuming with a valid `cached`, the pre-block setup (token refiner, patchify, audio
        pack, modulation table, rope) and the cached INPUT to block `resume_from` are reused,
        so only blocks >= resume_from run. If `new_cache` is given, each executed block's
        input is stored into it (parked on `cache_device` — a 22-frame 768 clip's per-block
        h is ~46 MB, x50 blocks x6 steps ~11.6 GB, which lives in system RAM, never VRAM).

        Numerically identical to forward() for resume_from=None. Inference-only: no gradient
        checkpointing (previews run under no_grad), no refs/tags (the workbench preview path
        doesn't use them), and a live H2D offloader disables resume (its ring assumes a walk
        from _swap_from) — the pass silently runs in full instead.
        """
        if keyframes:
            # The cache entries key on a fixed sequence layout; a keyframe changes it, and
            # the resume is off on H3 anyway (measured misleading). Say so, don't guess.
            raise ValueError("forward_cached does not support keyframe conditioning — "
                             "use forward()")
        if video_latent.shape[0] != 1:
            raise ValueError("MiniMax H3 inference is batch size 1")
        device = video_latent.device
        dtype = text_embeds.dtype
        _, _, latent_t, lat_h, lat_w = video_latent.shape
        nblocks = len(self.blocks)

        can_resume = (resume_from is not None and cached is not None
                      and getattr(self, "_h2d_offloader", None) is None
                      and cached.block_inputs and 0 <= resume_from < nblocks
                      and cached.block_inputs[resume_from] is not None)
        if can_resume:
            t_emb = cached.t_emb.to(device)
            mod_row = cached.mod_row.to(device)
            cos, sin = cached.cos.to(device), cached.sin.to(device)
            audio_start, video_start = cached.audio_start, cached.video_start
            video_t_index, audio_t_index = cached.video_t_index, cached.audio_t_index
            n_audio = video_start - audio_start
            h = cached.block_inputs[resume_from].to(device, dtype)
            start = resume_from
            if new_cache is not None:
                new_cache.block_inputs = list(cached.block_inputs)   # unchanged prefix by ref
        else:
            # The preamble below mirrors forward() exactly for the no-ref, no-tags case —
            # verified bitwise by tests/test_h3_forward_cached.py. Touch both or neither.
            text_len = text_embeds.shape[1]
            text_states = text_embeds[0]
            if text_states.shape[-1] != self.hidden_size:
                text_states = self.token_refiner(self.condition_proj(text_states))
            video_rows = patchify_video(video_latent.to(torch.float32), self.patch_size)
            video_embed = self.video_patch_proj(
                video_rows.to(self.video_patch_proj.weight.dtype)).to(dtype)

            t_val = t.reshape(-1)[:1].to(torch.float32) if torch.is_tensor(t) else torch.tensor([float(t)], device=device)
            t_val = t_val.to(device)
            n_audio_latents = (audio_latents_for_frames(pixel_frames_for_latent(latent_t))
                              if self.pack_audio_rows else 0)
            audio_embed = None
            if n_audio_latents:
                sigma_v = (1.0 - t_val).clamp(0.0, 1.0)
                sigma_a = remap_sigma(sigma_v)
                t_audio = 1.0 - sigma_a
                if audio_rows is not None:
                    _want = n_audio_latents * AUDIO_CHANNELS
                    if audio_rows.shape[0] != _want:
                        raise ValueError(f"audio_rows has {audio_rows.shape[0]} rows, expected {_want}")
                    _arows = audio_rows.to(device=device, dtype=torch.float32)
                else:
                    _arows = sigma_a * torch.randn(n_audio_latents * AUDIO_CHANNELS,
                                                   self.config.audio_latents_dim,
                                                   device=device, dtype=torch.float32)
                audio_embed = self.audio_patch_proj(
                    _arows.to(self.audio_patch_proj.weight.dtype)).to(dtype)

            parts = ([text_states.to(dtype)]
                     + ([audio_embed] if audio_embed is not None else [])
                     + [video_embed])
            h = torch.cat(parts, dim=0)
            seq_len = h.shape[0]
            n_audio = 0 if audio_embed is None else audio_embed.shape[0]
            audio_start = text_len
            video_start = audio_start + n_audio

            t_parts = [t_val] + ([t_audio] if audio_embed is not None else [])
            t_all = torch.cat(t_parts) if len(t_parts) > 1 else t_val
            uniq, inverse = torch.unique(t_all, sorted=True, return_inverse=True)
            t_emb = self._time_embedding(uniq)
            if not self.adaln_fp32:
                t_emb = t_emb.to(dtype)
            tags = torch.full((seq_len,), VIDEO_TAG, dtype=torch.long, device=device)
            tags[:text_len] = TEXT_TAG
            row_t_index = torch.full((seq_len,), int(inverse[0]), dtype=torch.long, device=device)
            if audio_embed is not None:
                tags[audio_start:video_start] = AUDIO_TAG
                row_t_index[audio_start:video_start] = int(inverse[1])
            mod_row = row_t_index * MODALITY_NUM + tags
            video_t_index = int(inverse[0])
            audio_t_index = int(inverse[1]) if n_audio else 0

            pos = image_position_ids(text_len, lat_h, lat_w, n_audio_latents,
                                     latent_t=latent_t).to(device)
            cos, sin = rope_cos_sin(pos, self.rope.inv_freq.to(device))
            cos, sin = cos.to(dtype), sin.to(dtype)
            start = 0
            if new_cache is not None:
                new_cache.block_inputs = [None] * nblocks

        if new_cache is not None:
            new_cache.t_emb = t_emb.detach()
            new_cache.mod_row = mod_row
            new_cache.cos, new_cache.sin = cos, sin
            new_cache.audio_start, new_cache.video_start = audio_start, video_start
            new_cache.video_t_index, new_cache.audio_t_index = video_t_index, audio_t_index

        for i in range(start, nblocks):
            if new_cache is not None:
                new_cache.block_inputs[i] = h.detach().to(cache_device)
            _ev = getattr(self, '_abort_event', None)
            if _ev is not None and _ev.is_set():
                raise ForwardAborted()
            h = _run_block(self.blocks, i, self._swap_from, h, t_emb, mod_row, cos, sin)
            _off = getattr(self, "_h2d_offloader", None)
            if _off is not None and i >= self._swap_from:
                _off.submit_move_blocks_forward(i)

        v = self.final_layer(h[video_start:], t_emb, video_t_index)
        out = unpatchify_video(v, latent_t, lat_h // self.patch_size[1], lat_w // self.patch_size[2],
                               self.latents_dim, self.patch_size)
        out = out.to(video_latent.dtype)
        if return_audio:
            a = (self.final_layer.forward_audio(h[audio_start:video_start], t_emb, audio_t_index)
                 if n_audio else None)
            return out, a
        return out


class H3ActivationCacheEntry:
    """Per-denoising-step cache for MiniMaxH3DiT.forward_cached (Repair Studio Turbo
    Preview): the INPUT h to every block, plus the LoRA-invariant preamble so a resumed
    pass can skip the token refiner / patchify / audio pack / modulation table / rope.

    block_inputs live on the CACHE device (CPU by default - a 22-frame 768x768 clip holds
    ~46 MB per block, 2.3 GB per step, ~11.6 GB across a 6-step render; system RAM, never
    VRAM). Everything else is small and rides wherever it was computed.
    """

    __slots__ = ("block_inputs", "t_emb", "mod_row", "cos", "sin",
                 "audio_start", "video_start", "video_t_index", "audio_t_index")

    def __init__(self):
        self.block_inputs = []
        self.t_emb = None
        self.mod_row = None
        self.cos = None
        self.sin = None
        self.audio_start = 0
        self.video_start = 0
        self.video_t_index = 0
        self.audio_t_index = 0
