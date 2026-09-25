"""Bound attention's math-backend workspace on RDNA2 without dropping tokens."""
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fizgig.modules.rdna2_linear import _is_rdna2

logger = logging.getLogger(__name__)
_announced = False


def head_chunk_attention(q, k, v, attn_mask=None, heads_per_chunk=8):
    """Independent head groups, with recomputation instead of retained score matrices.

    q/k/v use [batch, heads, sequence, features] and equal head counts. The
    caller has already expanded grouped-query keys/values. No dropout is used.
    """
    outputs = []
    for start in range(0, q.shape[1], heads_per_chunk):
        stop = start + heads_per_chunk
        mask = attn_mask
        if mask is not None and mask.ndim >= 3 and mask.shape[-3] == q.shape[1]:
            mask = mask[..., start:stop, :, :]
        args = (q[:, start:stop], k[:, start:stop], v[:, start:stop], mask)
        if torch.is_grad_enabled() and any(t.requires_grad for t in args if t is not None):
            # Non-reentrant checkpointing also composes with the enclosing DiT
            # block checkpoint. Backward materializes only one group's scores.
            out = checkpoint(F.scaled_dot_product_attention, *args, use_reentrant=False)
        else:
            out = F.scaled_dot_product_attention(*args)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def attention(q, k, v, attn_mask=None, dropout_p=0.0):
    global _announced
    use_chunks = (os.environ.get("FIZGIG_RDNA2_ATTENTION", "auto") != "0"
                  and q.device.type == "cuda" and q.dtype == torch.bfloat16
                  and q.ndim == k.ndim == v.ndim == 4
                  and q.shape[1] == k.shape[1] == v.shape[1]
                  and q.shape[1] > 8 and q.shape[-2] >= 512
                  and dropout_p == 0.0 and _is_rdna2(q.device.index))
    if not use_chunks:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
    if not _announced:
        logger.info("[rdna2] attention uses 8-head groups with recomputation to limit "
                    "backward workspace; set FIZGIG_RDNA2_ATTENTION=0 to compare")
        _announced = True
    return head_chunk_attention(q, k, v, attn_mask)
