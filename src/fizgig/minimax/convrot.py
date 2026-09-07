"""ConvRot int8 decode — reads ComfyUI's pre-quantized MiniMax H3 checkpoints.

`minimax_h3_*_pruned_int8_convrot.safetensors` stores every big block Linear as int8 in a
ROTATED basis, marked by a `<module>.comfy_quant` blob:

    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

ConvRot (arXiv:2512.03673) folds a block regular-Hadamard rotation into the weight offline and
applies the same rotation to the activation at runtime; being orthogonal it cancels in the
matmul, and it spreads outliers so one scale per output row is safe (the SmoothQuant failure
mode is what it avoids). Storage is therefore:

    weight        int8   [out, in]   codes of  rotate(W, G)
    weight_scale  fp32   [out, 1]    per-output-row symmetric scale
    comfy_quant   uint8  [n]         the JSON above

Fizgig doesn't run an int8 kernel — it NF4s the base and trains LoRA on top — so all we need is
the inverse: undo the rotation and get W back. The regular Hadamard is symmetric and orthogonal,
i.e. its own inverse, so "unrotate" is the same operation as "rotate".

Verified against the bf16 release of the same model: 0.9% relative error per tensor (the ~1%
the method claims), vs cosine 0.06 if the rotation is skipped — the failure is loud, not subtle.
"""

import json

import torch

_hadamard_cache = {}


def regular_hadamard(rot_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Kronecker powers of the 4x4 regular Hadamard, orthonormal — symmetric and orthogonal,
    so it is its own inverse.

    Built in fp32 on purpose: entries stay exactly +-1 through the krons and rot_size is a power
    of 4, so dividing by its (power-of-two) root is exact."""
    key = (rot_size, str(device), dtype)
    if key not in _hadamard_cache:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float32)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"convrot group size {rot_size} is not a power of 4")
        _hadamard_cache[key] = (h / rot_size ** 0.5).to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Block regular-Hadamard rotation along the LAST dim. Self-inverse, so this both applies
    and undoes it."""
    if rot_size <= 1:
        return x
    if x.shape[-1] % rot_size:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of the rotation block {rot_size}")
    h = regular_hadamard(rot_size, x.device, x.dtype)
    return torch.matmul(x.reshape(-1, x.shape[-1] // rot_size, rot_size), h).reshape(x.shape)


def parse_comfy_quant(blob: torch.Tensor) -> dict:
    """The `<module>.comfy_quant` marker: a uint8 tensor holding JSON."""
    return json.loads(bytes(blob.to(torch.uint8).cpu().numpy().tobytes()).decode("utf-8"))


def dequantize_int8_convrot(qweight: torch.Tensor, scale: torch.Tensor, conf: dict,
                            out_dtype=torch.bfloat16) -> torch.Tensor:
    """int8 codes + per-row scale (+ the marker's config) -> the dense weight.

    scale arrives as [out, 1] (or [out]); the rotation is undone in fp32 and the result cast
    once, so no intermediate is held at full precision longer than one tensor."""
    fmt = conf.get("format")
    if fmt != "int8_tensorwise":
        raise ValueError(f"unsupported comfy quant format {fmt!r} (this decodes int8_tensorwise)")
    group = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
    w = qweight.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)
    return rotate(w, group).to(out_dtype)


def quantize_int8_convrot(w: torch.Tensor, rot: int = 256, stochastic: bool = False,
                          generator: torch.Generator = None):
    """The ENCODE direction — a TRUE-basis dense weight -> int8 codes + per-row scale.

    Exists for rotation fine-tuning: a trained bf16 window is written back into the frozen
    model's int8-ConvRot storage. Inverse of dequantize_int8_convrot: rotate into the storage
    basis (self-inverse Hadamard, same call), then symmetric per-output-row absmax scaling to
    the int8 grid. Round-trip property the tests pin: encoding a freshly DECODED weight
    reproduces the original codes exactly — the codes are integers on their own grid and the
    rowmax recovers the scale. clamp_min guards an all-zero row (a dead output would
    otherwise divide by zero).

    stochastic=True is the FT-save mode. Nearest rounding is systematically biased back to
    the base's codes: a training delta below the grid step rounds HOME, so a fine-tune's
    0.2-1% weight movement saved with nearest carries near-zero training signal (measured:
    direction-cosine 0.02 at a 0.3% delta). Stochastic rounding rounds each weight up/down
    with probability equal to its fractional position — per-weight ±1-code dither (the
    error class the base already carries) but the delta is preserved IN EXPECTATION, and
    the network integrates the coherent delta while averaging out the random dither
    (measured: cosine 0.36 at 0.3%, ~10x the weight-space SNR of the NF4 trunk that
    in-training previews render likeness through). Weights exactly ON the grid have zero
    fractional part, so untouched tensors still roundtrip byte-exact.

    Returns (codes int8 [out, in], scale fp32 [out, 1])."""
    wr = rotate(w.to(torch.float32), rot)
    s = wr.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    v = wr / s
    if stochastic:
        fl = torch.floor(v)
        rdev = generator.device if generator is not None else v.device
        r = torch.rand(v.shape, generator=generator, device=rdev).to(v.device)
        q = (fl + (r < (v - fl)).to(v.dtype)).clamp(-127, 127).to(torch.int8)
    else:
        q = torch.round(v).clamp(-127, 127).to(torch.int8)
    return q, s


class _Int8RotLinearFn(torch.autograd.Function):
    """linear(rotate(x), dequant(q, s)) that does NOT keep the dequantized weight alive.

    The naive `F.linear(xr, w)` hands autograd a bf16 `w` to save for backward — 308 MB for
    fc1, and ~19 GB across the 200 block linears once every layer holds one. NF4 never pays
    that (bitsandbytes re-dequantizes inside its own backward), so the int8 path OOM'd exactly
    when gradient checkpointing was OFF, which is the case at small bucket sizes.

    The base is frozen, so no weight gradient is needed and the INPUT need not be saved either:
    grad_x = rotate(grad_y @ W_rot), recomputed from the int8 buffers that are resident anyway.
    Net effect: this stores less than an ordinary Linear would.
    """

    # rintic-13's fused W8A16 Triton GEMM (#89) — test-branch default ON. Opt out with
    # FIZGIG_NO_TRITON_W8A16=1. Falls back to the eager path when Triton is missing, the
    # input is not CUDA-bf16, or the kernel ever raises (logged once).
    _w8a16_state = {"checked": False, "use": False, "announced": False}
    # Dave Maybank's (@mabseyuk) fused W8A16 BACKWARD GEMM — opt IN with
    # FIZGIG_TRITON_W8A16_BACKWARD=1 (convrot_w8a16_backward_triton.py). Same fallback rules.
    _w8a16_bwd_state = {"checked": False, "use": False, "announced": False}

    @classmethod
    def _use_w8a16_backward(cls, grad_out, dt):
        st = cls._w8a16_bwd_state
        if not st["checked"]:
            st["checked"] = True
            import os as _os
            if _os.environ.get("FIZGIG_TRITON_W8A16_BACKWARD") == "1":
                try:
                    from fizgig.minimax.convrot_w8a16_backward_triton import TRITON_AVAILABLE
                    st["use"] = bool(TRITON_AVAILABLE)
                except Exception:
                    st["use"] = False
        if not (st["use"] and dt == torch.bfloat16 and grad_out.is_cuda
                and grad_out.dtype == torch.bfloat16):
            return False
        if not st["announced"]:
            st["announced"] = True
            print("[convrot] fused W8A16 Triton BACKWARD kernel active (Dave Maybank, @mabseyuk)",
                  flush=True)
        return True

    @classmethod
    def _use_w8a16(cls, x, dt):
        st = cls._w8a16_state
        if not st["checked"]:
            st["checked"] = True
            import os as _os
            if _os.environ.get("FIZGIG_NO_TRITON_W8A16") == "1":
                st["use"] = False
            else:
                try:
                    from fizgig.minimax.convrot_w8a16_triton import TRITON_AVAILABLE
                    st["use"] = bool(TRITON_AVAILABLE)
                except Exception:
                    st["use"] = False
        # Training steps only: previews run under no_grad on razor-thin margins (56-frame
        # clips at 11 GB free), where Triton's first-use overhead — JIT cubins, autotune
        # benchmarking, context growth outside torch's allocator — tipped a working preview
        # into OOM (field). The kernel buys nothing in a 6-step preview anyway; the win is
        # the thousands of training forwards.
        if not (st["use"] and dt == torch.bfloat16 and x.is_cuda
                and torch.is_grad_enabled()):
            return False
        if not st["announced"]:
            st["announced"] = True
            print("[convrot] fused W8A16 Triton kernel active (issue #89, rintic-13)",
                  flush=True)
        return True

    @staticmethod
    def forward(ctx, x, qdata, wscale, bias, rot, dt):
        ctx.save_for_backward(qdata, wscale)
        ctx.rot, ctx.dt = rot, dt
        xr = rotate(x.to(dt), rot) if rot > 1 else x.to(dt)
        if _Int8RotLinearFn._use_w8a16(xr, dt):
            try:
                from fizgig.minimax.convrot_w8a16_triton import fused_w8a16_gemm
                return fused_w8a16_gemm(xr, qdata, wscale, bias)
            except Exception as _ke:
                _Int8RotLinearFn._w8a16_state["use"] = False
                print(f"[convrot] W8A16 kernel failed ({type(_ke).__name__}: {_ke}) — "
                      "falling back to the eager path for the rest of the run.", flush=True)
        # Scale on the OUTPUT, not the weight. wscale is per output ROW, so
        #     y[t,o] = sum_i xr[t,i]*q[o,i]*s[o] = (xr @ q^T)[t,o] * s[o]
        # is exact, and it buys two things. The int8 CODES are integers <= 127, so they survive
        # the bf16 cast bit-for-bit; the fp32 SCALES do not — measured 0.137% mean / 0.388% max
        # relative error per row on the real checkpoint, i.e. a systematic per-output-channel
        # GAIN error on all 200 block linears, every block, every step. Applying s in fp32 to a
        # [tokens, out] tensor removes it. It also means no dequantized [out, in] weight is
        # materialized at all, which is the 308 MB-per-layer transient this Function exists to
        # avoid. Structurally this is also what ComfyUI's int8 kernel does: accumulate, then
        # apply the fp32 scale.
        y = torch.nn.functional.linear(xr, qdata.to(dt))
        y = (y.float() * wscale.reshape(-1).float()).to(dt)
        return y if bias is None else y + bias.to(dt)

    @staticmethod
    def backward(ctx, grad_out):
        qdata, wscale = ctx.saved_tensors
        # grad_x = grad_out @ W = grad_out @ (q * s) = (grad_out * s) @ q — same identity, so
        # the [out, in] weight is never rebuilt here either.
        gx = None
        if _Int8RotLinearFn._use_w8a16_backward(grad_out, ctx.dt):
            try:
                from fizgig.minimax.convrot_w8a16_backward_triton import fused_w8a16_input_grad
                gx = fused_w8a16_input_grad(grad_out, qdata, wscale)
            except Exception as _ke:
                _Int8RotLinearFn._w8a16_bwd_state["use"] = False
                print(f"[convrot] W8A16 backward kernel failed ({type(_ke).__name__}: {_ke}) — "
                      "falling back to the eager backward for the rest of the run.", flush=True)
                gx = None
        if gx is None:
            gs = (grad_out.float() * wscale.reshape(-1).float()).to(ctx.dt)
            gx = gs @ qdata.to(ctx.dt)                     # [..., out] @ [out, in]
        if ctx.rot > 1:
            gx = rotate(gx, ctx.rot)                       # H is symmetric: R^T == R
        return gx, None, None, None, None, None


class ConvRotInt8Linear(torch.nn.Linear):
    """A frozen Linear that KEEPS the checkpoint's int8 ConvRot weights — the reference's storage.

    Why this exists: decoding these weights to bf16 and re-quantizing to NF4 (our previous base
    path) perturbs the frozen model by ~9.5% relative, against ~0.17% for the int8 weights the
    reference trains on directly — ~57x more error in the function the LoRA is fitted against.
    Measured per-tensor on the real checkpoint. The adapter then spends capacity correcting
    quantization error that will not exist at inference, and it compounds with depth (our
    block 40-49 adapters moved 1.5-3.9x more than the reference's; blocks 0-19 moved ~0.75x).

    Why it is affordable: dequantizing the WEIGHT needs the inverse rotation, an [out, in]
    matmul per step — far too expensive. So rotate the ACTIVATION instead, which is
    [tokens, in]:

        (x R) (W_rot)^T = x R R^T W^T = x W^T

    because the regular Hadamard is symmetric and orthogonal (its own inverse). The per-matmul
    cost is then one small rotation plus an elementwise `q * scale`. This is the same
    dequantized-matmul path ai-toolkit itself falls back to without Triton — which is what it
    used on this machine, so the comparison stays like-for-like.

    Storage is 1 byte/param (~19 GB for H3's 200 block linears) against NF4's ~0.5.
    """

    def __init__(self, in_features, out_features, bias=False, rot=256,
                 compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        del self._parameters["weight"]          # the int8 buffers replace it
        self.rot = int(rot)
        self.compute_dtype = compute_dtype
        self.register_buffer("qdata", torch.empty(out_features, in_features, dtype=torch.int8),
                             persistent=False)
        self.register_buffer("wscale", torch.empty(out_features, 1, dtype=torch.float32),
                             persistent=False)

    @property
    def weight(self):
        """The dense weight in the TRUE basis, materialized on demand (inspection only — the
        forward never calls this, it rotates the activation instead)."""
        return rotate(self.qdata.float() * self.wscale.float(), self.rot).to(self.compute_dtype)

    def forward(self, x):
        if self.bias is not None and self.bias.requires_grad:
            raise RuntimeError("ConvRotInt8Linear holds a FROZEN base weight; its bias must not "
                               "require grad (the custom backward returns no bias gradient)")
        return _Int8RotLinearFn.apply(x, self.qdata, self.wscale, self.bias,
                                      self.rot, self.compute_dtype)
