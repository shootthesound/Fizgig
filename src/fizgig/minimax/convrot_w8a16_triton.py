"""Fused W8A16 Triton GEMM for the int8-ConvRot base — by rintic-13 (issue #89).

The successor to their #78 kernel, with the lesson applied: #78 quantized ACTIVATIONS to
int8 (W8A8) and the ~1% activation noise destroyed likeness at identical loss. This one
keeps activations bf16 (W8A16): int8 weights cast to bf16 inside the kernel, fp32
accumulation, per-row fp32 scale, single bf16 rounding at the store — measured 2e-05 from
the pure-fp32 reference, i.e. CLOSER to the true product than the eager path, whose two
roundings (torch's bf16 GEMM output, then the separate scale multiply) were a composition
artifact. rintic-13 supplied two epilogues (an eager-parity one that reproduces the double
rounding, and this single-round one); 40-epoch blind quality A/Bs in ComfyUI came out at
parity across eager, parity-epilogue and single-round, so the single-round ships — with
quality equal, fewer roundings win.

Measured speed: 1.14-1.42x per GEMM on real checkpoint weights across layers and shapes.

Kernel and autotune configs are rintic-13's, verbatim. The dispatch lives in convrot.py.
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
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 16},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                          num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                          num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                          num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
                          num_stages=3, num_warps=8),
        ],
        # M_KEY, not M (7 Sep 2026): keyed on the exact token count, every caption length
        # is a new key — measured 244 s for a 46-step first epoch against 53 s eager, with
        # tuning still leaking into later epochs as caption dropout shifts lengths. Keyed on
        # the power-of-2 bucket of M the tuning is ~6 keys per shape, done in the first steps.
        key=["M_KEY", "N", "K"],
    )
    @triton.jit
    def fused_w8a16_gemm_kernel(
        X_ptr, W_ptr, WS_ptr, B_ptr, O_ptr,
        M, N, K, M_KEY,
        stride_xm, stride_xk,
        stride_wm, stride_wk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # rintic-13's v2 "large M" hardening (#89): the row offsets go int64 in the X and
        # O pointer math. int32 wraps at M x stride_xm > 2^31 (M > ~150k tokens at
        # K=14336) — silent corruption, no error. Below that the addresses are identical,
        # so the kernel is bit-for-bit unchanged for every reachable shape (pinned by the
        # old-vs-new parity check that shipped this). The k-side stays int32, verbatim
        # from their file: K never exceeds ~14k.
        offs_m_i64 = offs_m.to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        offs_k = tl.arange(0, BLOCK_K)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            mask_k = offs_k < K - k_idx * BLOCK_K

            x_ptrs = (X_ptr + offs_m_i64[:, None] * stride_xm
                      + (offs_k[None, :] + k_idx * BLOCK_K) * stride_xk)
            x = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
            x = x.to(tl.bfloat16)

            w_ptrs = (W_ptr + offs_n[:, None] * stride_wm
                      + (offs_k[None, :] + k_idx * BLOCK_K) * stride_wk)
            w_int8 = tl.load(w_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0)
            w = w_int8.to(tl.bfloat16)

            accumulator += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

        # Single rounding, at the store only: the accumulator is scaled at full fp32 and
        # cast to bf16 once. The eager path rounds TWICE (torch's bf16 GEMM output, then
        # the separate fp32 scale multiply) — a composition artifact, not a design; this
        # is measurably closer to the true product (2e-05 from the fp32 reference).
        w_scale = tl.load(WS_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        out = accumulator * w_scale[None, :]

        if HAS_BIAS:
            bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            out += bias[None, :]

        out_ptrs = O_ptr + offs_m_i64[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=(mask_m[:, None] & mask_n[None, :]))

    def fused_w8a16_gemm(x, qdata, wscale, bias):
        """y = bf16(fp32(x @ q^T) * s) [+ bias] — single rounding, at the store."""
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        M, K = x2d.shape
        N = qdata.shape[0]

        if x2d.dtype != torch.bfloat16:
            raise RuntimeError("Fused W8A16 Triton kernel expects BF16 activation")
        if qdata.dtype != torch.int8:
            raise RuntimeError("Fused W8A16 Triton kernel expects INT8 qdata")
        if not x2d.is_cuda:
            raise RuntimeError("Fused W8A16 Triton kernel requires CUDA")

        qdata = qdata.contiguous()
        wscale = wscale.reshape(-1).contiguous()

        if bias is None:
            bias_ptr = torch.empty(1, device=x2d.device, dtype=torch.bfloat16)
            has_bias = False
        else:
            bias_ptr = bias.reshape(-1).contiguous()
            has_bias = True

        out = torch.empty((M, N), device=x2d.device, dtype=torch.bfloat16)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        fused_w8a16_gemm_kernel[grid](
            x2d, qdata, wscale, bias_ptr, out,
            M, N, K, 1 << max(0, M - 1).bit_length(),     # M_KEY: the power-of-2 bucket of M
            x2d.stride(0), x2d.stride(1),
            qdata.stride(0), qdata.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=has_bias,
        )
        return out.reshape(*x.shape[:-1], N)

else:

    def fused_w8a16_gemm(x, qdata, wscale, bias):
        raise RuntimeError("Triton is required for fused W8A16 GEMM")
