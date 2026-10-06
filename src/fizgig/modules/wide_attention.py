"""Attention for the VAEs' single wide head, correct on AMD ROCm (#179, reported by @ArchAngelAries).

On ROCm builds of PyTorch, scaled_dot_product_attention can return wrong values (NaN in bf16) when the head is wider
than 256 - and the VAEs' mid-block attention is one head of 384 (Qwen-Image / Krea 2, Qwen 2.1) or 512 (Klein, SDXL).
The encoder shares the block, so cached latents were affected as well as previews. On ROCm, a head wider than 256 is
computed here instead: softmax(q k^T / sqrt(d)) v in float32, a slice of query rows at a time so the full attention
matrix is never held. Everywhere else (every NVIDIA card, CPU, narrow heads) it is PyTorch's own call, unchanged.
"""
import torch
import torch.nn.functional as F

ROCM = bool(getattr(torch.version, "hip", None))
FAST_HEAD_DIM = 256                 # ROCm SDPA is correct up to here (head 64 / 128 / 256 measured in #179)
CHUNK_ELEMENTS = 1 << 27            # attention-matrix elements per slice (512 MB in float32)


def wide_head_attention(q, k, v):
    """q, k, v (..., S, D) -> (..., S, D), no mask, no dropout: PyTorch's SDPA, except a head wider than 256 on ROCm,
    which is computed directly in float32."""
    if not ROCM or q.shape[-1] <= FAST_HEAD_DIM:
        return F.scaled_dot_product_attention(q, k, v)
    dtype = q.dtype
    qf = q.float() * q.shape[-1] ** -0.5
    kt = k.float().transpose(-1, -2)
    vf = v.float()
    lead = max(1, int(torch.Size(q.shape[:-2]).numel()))
    rows = max(1, CHUNK_ELEMENTS // (lead * k.shape[-2]))
    out = [torch.softmax(qf[..., i:i + rows, :] @ kt, dim=-1) @ vf for i in range(0, q.shape[-2], rows)]
    return torch.cat(out, dim=-2).to(dtype)


def rocm_vae_processor():
    """diffusers' AttnProcessor2_0 with its attention call through wide_head_attention, for a diffusers VAE on ROCm
    (SDXL's AutoencoderKL). The VAE path only: self-attention, no mask."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    class _WideHeadVAEProcessor(AttnProcessor2_0):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                     *args, **kwargs):
            if encoder_hidden_states is not None or attention_mask is not None:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                                        *args, **kwargs)
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size = hidden_states.shape[0]
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            head_dim = key.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            hidden_states = wide_head_attention(query, key, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            return hidden_states / attn.rescale_output_factor

    return _WideHeadVAEProcessor()
