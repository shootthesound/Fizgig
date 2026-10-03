"""Krea 2 attention forwards installed only on a detected RDNA2 model."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from fizgig.krea2.attention import AttentionParams
from fizgig.krea2.model import Attention as KreaAttention
from fizgig.krea2.model import common_attention, ropeapply
from fizgig.modules.sdpa import sdpa_backend_ctx


def head_chunk_attention(q, k, v, attn_mask=None, heads_per_chunk=8):
    """Recompute one group of attention heads at a time during backward."""
    outputs = []
    for start in range(0, q.shape[1], heads_per_chunk):
        stop = start + heads_per_chunk
        mask = attn_mask
        if mask is not None and mask.ndim >= 3 and mask.shape[-3] == q.shape[1]:
            mask = mask[..., start:stop, :, :]
        args = (q[:, start:stop], k[:, start:stop], v[:, start:stop], mask)
        if torch.is_grad_enabled() and any(t.requires_grad for t in args if t is not None):
            out = checkpoint(F.scaled_dot_product_attention, *args, use_reentrant=False)
        else:
            out = F.scaled_dot_product_attention(*args)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def _sdpa(q, k, v, mask=None):
    if q.shape[1] > 8 and q.shape[-2] >= 512:
        return head_chunk_attention(q, k, v, mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def grouped_common_attention(q, k, v, attn_params=None):
    """The existing Krea 2 torch path with grouped SDPA; other backends delegate."""
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)
    if attn_params.attn_mode != "torch":
        return common_attention([q, k, v], attn_params=attn_params)

    seqlen_trimmed = False
    seqlen = attn_params.uniform_seqlen
    if seqlen is not None:
        q, k, v = q[:, :seqlen], k[:, :seqlen], v[:, :seqlen]
        max_seqlen = attn_params.max_seqlen
        kept_mask = None if attn_params.uniform_exact else attn_params.attention_mask[..., :seqlen]
        attn_params = AttentionParams.create_attention_params("torch", False)
        attn_params.attention_mask = kept_mask
        attn_params.max_seqlen = max_seqlen
        seqlen_trimmed = True

    if attn_params.split_attn:
        if attn_params.seqlens is None:
            attn_params = AttentionParams.create_attention_params("torch", True)
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        outputs = []
        for i in range(q.shape[0]):
            qi = q[i:i + 1, :attn_params.seqlens[i]].transpose(1, 2)
            ki = k[i:i + 1, :attn_params.seqlens[i]].transpose(1, 2)
            vi = v[i:i + 1, :attn_params.seqlens[i]].transpose(1, 2)
            if qi.shape[1] != ki.shape[1]:
                groups = qi.shape[1] // ki.shape[1]
                ki = ki.repeat_interleave(groups, dim=1)
                vi = vi.repeat_interleave(groups, dim=1)
            with sdpa_backend_ctx():
                out = _sdpa(qi, ki, vi)
            outputs.append(F.pad(out, (0, 0, 0, attn_params.max_seqlen - out.shape[-2])))
        x = torch.cat(outputs, dim=0)
    else:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if q.shape[1] != k.shape[1]:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        with sdpa_backend_ctx():
            x = _sdpa(q, k, v, attn_params.attention_mask)

    x = x.transpose(1, 2).reshape(x.shape[0], x.shape[-2], -1)
    if seqlen_trimmed:
        x = F.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]))
    return x


def rdna2_forward(self, qkv, freqs=None, attn_params=None):
    """The original Krea 2 Attention.forward with this model's grouped path."""
    q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
    q, k, v = (
        rearrange(q, "B L (H D) -> B H L D", H=self.heads),
        rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
        rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
    )
    q, k, v = self.qknorm(q, k, v)
    if freqs is not None:
        q, k = ropeapply(q, k, freqs)
    x = grouped_common_attention(q.transpose(1, 2), k.transpose(1, 2),
                                 v.transpose(1, 2), attn_params)
    return self.wo(x * F.sigmoid(gate))


def install_attention(model):
    """Bind optimized forwards to this model only, leaving shared functions untouched."""
    count = 0
    for module in model.modules():
        if isinstance(module, KreaAttention):
            module.forward = rdna2_forward.__get__(module, type(module))
            count += 1
    return count
