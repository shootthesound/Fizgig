# Unified attention function supporting various implementations
# Adapted from musubi-tuner (https://github.com/kohya-ss/musubi-tuner), Apache-2.0.
# Modified for Fizgig. See THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass
import logging
import os
import torch
from typing import Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


# Sequence lengths are rounded up to a multiple of this before attention, so a bucketed dataset
# presents a few distinct shapes rather than one per caption length. 1 disables the rounding
# (exact trim, the old behaviour). See AttentionParams.__post_init__.
# Guarded parse: this module is imported by the model, both trainers and the GUI — a typo'd
# env var used to surface as a bare ValueError traceback at app start that never named it.
try:
    _TRIM_MULTIPLE = int(os.environ.get("FIZGIG_ATTN_TRIM_MULTIPLE", "64") or 64)
except ValueError:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "FIZGIG_ATTN_TRIM_MULTIPLE=%r is not an integer — using the default 64",
        os.environ.get("FIZGIG_ATTN_TRIM_MULTIPLE"))
    _TRIM_MULTIPLE = 64


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    def __post_init__(self):
        # Every sequence the same length means attention can trim to it instead of attending over
        # padding. Deciding that requires reading a CUDA tensor on the CPU, so it is resolved here,
        # once per forward, rather than inside each of the 28 blocks. None = do not trim.
        #
        # FIZGIG_ATTN_TRIM=0 keeps the padded length and passes the key-padding mask instead.
        # Slower in raw compute (more tokens attend), but it is what makes SHAPES STABLE: the DiT
        # pads the sequence to a multiple of 256 explicitly to keep kernel shapes stable, and
        # trimming undoes that — on a 36-image set it turns 4 distinct shapes into 30, because the
        # trimmed length includes each caption's own token count. That matters to any backend that
        # plans or compiles per shape (cuDNN, torch.compile), so it needs to be measurable.
        self.uniform_seqlen = None
        self.uniform_exact = True     # False -> the trimmed window still contains padding
        if (self.seqlens is not None and self.attention_mask is not None and not self.split_attn
                and os.environ.get("FIZGIG_ATTN_TRIM", "1") != "0"
                and self.attn_mode not in ("flash", "sageattn")):
            if bool(torch.all(self.seqlens == self.seqlens[0])):
                valid = int(self.seqlens[0])
                # Round the trim length UP to a multiple, so sequence lengths land in a handful
                # of buckets instead of taking a distinct value per caption. Trimming to the
                # exact valid length sounds optimal and is not: the length carries each caption's
                # own token count, which turned 36 images into 30 distinct shapes and made every
                # shape-planning backend (cuDNN, torch.compile) pay a first-sight cost it could
                # never amortize. At x64 that is 10 shapes for +3.6% tokens.
                q = _TRIM_MULTIPLE
                if q > 1 and self.max_seqlen:
                    rounded = min(((valid + q - 1) // q) * q, int(self.max_seqlen))
                    self.uniform_exact = rounded == valid
                    self.uniform_seqlen = rounded
                else:
                    self.uniform_seqlen = valid
        # Tell the backend chooser which shape this forward will ACTUALLY attend — including
        # when trim is disabled (FIZGIG_ATTN_TRIM=0, which the docs recommend to help cuDNN)
        # or the batch is ragged: both attend the padded length. Recording only inside the
        # trim branch meant those runs recorded nothing and the cuDNN auto-switch could
        # never fire.
        if self.attention_mask is not None:
            attended = self.uniform_seqlen or (int(self.max_seqlen) if self.max_seqlen else None)
            if attended:
                _note_shape(attended)

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
            for i in range(batch_size):
                cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]  # end of valid tokens for query
                cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen  # end of all tokens for query

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

            # attention bias for xformers
            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen)


# Backend choice (cuDNN for inference, PyTorch's default while training) lives in
# fizgig.modules.sdpa so every attention site in the app makes the same decision — this one,
# Klein's DiT, and both VAEs. See that module for the measurements behind it.
from fizgig.modules.sdpa import note_shape as _note_shape  # noqa: E402
from fizgig.modules.sdpa import sdpa_backend_ctx as _sdpa_backend_ctx
from fizgig.modules.rdna2_attention import attention as _torch_attention

logger = logging.getLogger(__name__)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_param: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # GQA: q may carry more heads than k/v (e.g. Krea 2 = 48 query / 12 kv heads). Expanding
    # k/v stays the right call even on cuDNN — measured 0.128 ms (expand) vs 0.138 ms
    # (enable_gqa) at seq 1024, and enable_gqa on the DEFAULT backend is 1.802 ms. flash and
    # sageattn group heads natively inside the kernel (verified), so they ignore this. For the
    # torch (SDPA) path we expand k/v to q's head count below instead of passing enable_gqa=True,
    # because enable_gqa forces SDPA onto the slow math kernel (~7x slower than the fused kernels
    # at K2 scale); the repeat is numerically identical. Revert to enable_gqa=True once PyTorch's
    # fused (flash / mem-efficient) kernels gain native GQA support. (q/k/v here are [B, L, H, D].)
    enable_gqa = q.shape[-2] != k.shape[-2]

    # When every sequence is the same length, attend over that length instead of the padded
    # maximum. Whether that applies is decided once, in AttentionParams.__post_init__ — it needs
    # to read a CUDA tensor on the CPU, and doing that here meant a device sync inside every one
    # of the 28 blocks (56 with gradient checkpointing's recompute), plus a hard graph break that
    # is why compiling the blocks lost end to end while the same block compiled 1.37x faster in
    # isolation. flash/sageattn handle masks efficiently themselves and are excluded there.
    seqlen_trimmed = False
    seqlen = attn_params.uniform_seqlen
    if seqlen is not None:
        q = q[:, :seqlen]
        k = k[:, :seqlen]
        v = v[:, :seqlen]
        max_seqlen = attn_params.max_seqlen
        # The window is rounded up to a multiple, so unless it landed exactly on the valid length
        # it still contains padding — which must stay masked or those tokens join the attention.
        # The mask is [B, 1, 1, S] for the torch path, so it slices on the last axis.
        kept_mask = None if attn_params.uniform_exact else attn_params.attention_mask[..., :seqlen]
        attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
        attn_params.attention_mask = kept_mask
        attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
        seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                    g = qi.shape[1] // ki.shape[1]  # [B, H, L, D] -> heads at dim 1
                    ki = ki.repeat_interleave(g, dim=1)
                    vi = vi.repeat_interleave(g, dim=1)
                with _sdpa_backend_ctx():
                    x_i = _torch_attention(qi, ki, vi, dropout_p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                g = q.shape[1] // k.shape[1]  # [B, H, L, D] -> heads at dim 1
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            with _sdpa_backend_ctx():
                x = _torch_attention(
                    q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                    g = qi.shape[2] // ki.shape[2]
                    ki = ki.repeat_interleave(g, dim=2)
                    vi = vi.repeat_interleave(g, dim=2)
                x_i = xops.memory_efficient_attention(qi, ki, vi, p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                g = q.shape[2] // k.shape[2]
                k = k.repeat_interleave(g, dim=2)
                v = v.repeat_interleave(g, dim=2)
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = flash_attn_func(q, k, v, drop_rate)  # B, L, H, D
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            x = flash_attn_varlen_func(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen, drop_rate
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
