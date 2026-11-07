"""Unified attention function supporting multiple backend implementations.

Supports: PyTorch SDPA (torch), flash_attn, xformers, sageattention.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from fizgig.modules import int8_attention as _int8_attention
from fizgig.modules.sdpa import masked_ctx, sdpa_backend_ctx

try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
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
        # See the trim block in attention(): resolved once here, not once per block.
        self.uniform_seqlen = None
        if (self.seqlens is not None and self.attention_mask is not None and not self.split_attn
                and self.attn_mode not in ("flash", "sageattn")):
            if bool(torch.all(self.seqlens == self.seqlens[0])):
                self.uniform_seqlen = int(self.seqlens[0])

    @staticmethod
    def create(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)

        # attention_mask is for text tokens only, not including image tokens
        seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
        max_seqlen = attention_mask.shape[1] + img_len

        if split_attn:
            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

        # Build cumulative sequence lengths for flash attention
        batch_size = attention_mask.shape[0]
        cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
        for i in range(batch_size):
            cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]
            cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen

        # Expand attention mask to include image tokens
        attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

        # Build attention bias for xformers
        if attn_mode == "xformers":
            seqlens_list = seqlens.cpu().tolist()
            attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                seqlens_list, seqlens_list, device=attention_mask.device
            )
        elif attn_mode == "torch":
            attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

        return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """Compute scaled dot-product attention with variable sequence length support.

    Args:
        qkv_or_q: Query tensor [B, L, H, D] or list of [q, k, v].
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
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
        attn_params = AttentionParams.create("torch", False)

    # Trim to the common sequence length when every sequence has one, instead of attending over
    # padding. Whether that applies is decided once in AttentionParams.__post_init__: it has to
    # read a CUDA tensor on the CPU, and doing it here cost a device sync inside every block of
    # every forward. flash/sageattn handle masks efficiently themselves and are excluded.
    seqlen_trimmed = False
    seqlen = attn_params.uniform_seqlen
    if seqlen is not None:
        q = q[:, :seqlen]
        k = k[:, :seqlen]
        v = v[:, :seqlen]
        max_seqlen = attn_params.max_seqlen
        attn_params = AttentionParams.create(attn_params.attn_mode, False)
        attn_params.max_seqlen = max_seqlen
        seqlen_trimmed = True

    # Layout depends on backend
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D]
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D]
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Split attention: process each batch element separately
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            attn_params = AttentionParams.create(attn_params.attn_mode, True)
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    # --- Backend dispatch ---

    if attn_params.attn_mode == "torch":
        # cuDNN's kernel for inference, PyTorch's own choice while training — one shared
        # decision, see fizgig.modules.sdpa for the measurements. Klein's previews (Repair
        # Studio, Explorer, profiler, sampling) hold one resolution for a whole render, which
        # is the case cuDNN wins; bucketed training is the case it loses.
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                with sdpa_backend_ctx():
                    x_i = torch.nn.functional.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate)
                q[i], k[i], v[i] = None, None, None
                x.append(pad_fn(x_i, attn_params.max_seqlen))
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        else:
            x = _int8_attention.attend(q, k, v, attn_params.attention_mask)   # an opted-in render only
            if x is None:
                with masked_ctx(attn_params.attention_mask):
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
                    )
            q, k, v = None, None, None

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
                q[i], k[i], v[i] = None, None, None
                x.append(pad_fn(x_i, attn_params.max_seqlen))
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        else:
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = sageattn(q[i], k[i], v[i])
                q[i], k[i], v[i] = None, None, None
                x.append(pad_fn(x_i, attn_params.max_seqlen))
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:
            x = sageattn(q, k, v)
            q, k, v = None, None, None
        else:
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            q, k, v = None, None, None
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)
                q[i], k[i], v[i] = None, None, None
                x.append(pad_fn(x_i, attn_params.max_seqlen))
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:
            x = flash_attn_func(q, k, v, drop_rate)
            q, k, v = None, None, None
        else:
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])
            x = flash_attn_varlen_func(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen, drop_rate
            )
            q, k, v = None, None, None
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])

    else:
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)

    return x
