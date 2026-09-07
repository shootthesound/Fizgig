"""Fused W8A16 Triton BACKWARD GEMM for the int8-ConvRot base — by Dave Maybank (@mabseyuk).

The companion to rintic-13's forward kernel (convrot_w8a16_triton.py). The eager backward in
convrot.py computes the input gradient as

    gs = bf16(grad_out * row_scale)          # a [tokens, out] bf16 temporary
    gx = gs @ bf16(int8_weight)              # a full [out, in] bf16 weight, materialised

This kernel does both in one pass: the scale is applied in fp32 on the fly, the int8 weight
is cast tile by tile inside the kernel, accumulation is fp32 and there is a single bf16
rounding at the store. Neither temporary exists — measured on the real ConvRot shapes the
eager path's transient allocation is 16 MiB to 1.5 GiB per linear per step (it grows with
the token count), the fused path's 1 to 250 MiB.

Measured (5090, 7 Sep 2026, ConvRot shapes 2688x2688 / 8064x2688 / 2688x5376 / 5376x2688,
tokens 256 to 24576): every output within one bf16 ulp of the eager result, deterministic,
and against an fp32 ground truth marginally CLOSER than the eager path (0.150 vs 0.185 mean
abs, the one-rounding argument again). Per GEMM 1.0-1.9x the eager backward, typically
1.2-1.4x at clip-sized token counts. The Hadamard inverse rotation stays in convrot.py.

In a real run (Peter's recipe on the 23-clip / 23-still videotest set, int8 base streamed 8
blocks, TREAD on, adapter on, 5090): 1.11 s/step against 1.39 s/step for the eager backward
once tuned — a 20% faster training step, at the top of Dave's 5-15% estimate. The first
epoch pays ~24 autotune events (a few seconds) and is already level with eager.

Kernel, autotune configs and wrapper are Dave's (his int64-offset variant — the #89 v2
hardening for M*N > 2^31), with one change: the autotune key buckets the token count to its
power of 2 (see the kernel) so a training run tunes once per shape bucket, not once per
caption length. Opt in with FIZGIG_TRITON_W8A16_BACKWARD=1; the dispatch
lives in convrot.py and falls back to the eager path if the kernel ever raises.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_N": 32},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 32},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 32},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64},
                          num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64},
                          num_stages=3, num_warps=8),
        ],
        # M_KEY, not M: keyed on the exact token count, a training run re-tunes for every
        # caption length (measured: 1,488 config benchmarks in one 46-step epoch, 160 more in
        # the next as caption dropout shifted lengths — 3x the epoch). Keyed on the power-of-2
        # bucket of M there are ~6 keys per shape and the tuning is done in the first steps.
        key=["M_KEY", "N", "K"],
    )
    @triton.jit
    def _w8a16_backward_kernel(
        G_ptr, W_ptr, S_ptr, O_ptr,
        M, N, K, M_KEY,
        stride_gm, stride_gn,
        stride_wn, stride_wk,
        stride_om, stride_ok,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K
        offs_m_i64 = offs_m.to(tl.int64)          # #89 v2 hardening: M*N and M*K can exceed 2^31
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        # N is the forward layer's output dimension. Scaling occurs in fp32,
        # followed by the same BF16 rounding as the eager ``.to(ctx.dt)``.
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            g_ptrs = G_ptr + offs_m_i64[:, None] * stride_gm + offs_n[None, :] * stride_gn
            grad = tl.load(g_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
            scale = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            grad_scaled = (grad.to(tl.float32) * scale[None, :]).to(tl.bfloat16)

            w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            weight = tl.load(w_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0)
            weight = weight.to(tl.bfloat16)
            accumulator += tl.dot(grad_scaled, weight, out_dtype=tl.float32)

        out_ptrs = O_ptr + offs_m_i64[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, accumulator.to(tl.bfloat16),
                 mask=(mask_m[:, None] & mask_k[None, :]))


    def fused_w8a16_input_grad(grad_out, qdata, wscale):
        """Return the input gradient for a frozen W8A16 linear layer."""
        original_shape = grad_out.shape[:-1]
        grad2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        qdata = qdata.contiguous()
        scale = wscale.reshape(-1).contiguous()
        M, N = grad2d.shape
        weight_n, K = qdata.shape

        if not grad2d.is_cuda:
            raise RuntimeError("W8A16 backward kernel requires CUDA")
        if grad2d.dtype != torch.bfloat16:
            raise RuntimeError("W8A16 backward kernel expects BF16 gradients")
        if qdata.dtype != torch.int8:
            raise RuntimeError("W8A16 backward kernel expects INT8 qdata")
        if weight_n != N or scale.numel() != N:
            raise RuntimeError(
                f"W8A16 backward shape mismatch: grad={tuple(grad2d.shape)}, "
                f"weight={tuple(qdata.shape)}, scale={scale.numel()}")

        out = torch.empty((M, K), device=grad2d.device, dtype=torch.bfloat16)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(K, META["BLOCK_K"]),
        )
        _w8a16_backward_kernel[grid](
            grad2d, qdata, scale, out,
            M, N, K, 1 << max(0, M - 1).bit_length(),      # M_KEY: the power-of-2 bucket of M
            grad2d.stride(0), grad2d.stride(1),
            qdata.stride(0), qdata.stride(1),
            out.stride(0), out.stride(1),
        )
        return out.reshape(*original_shape, K)

else:

    def fused_w8a16_input_grad(grad_out, qdata, wscale):
        raise RuntimeError("Triton is required for fused W8A16 backward")
