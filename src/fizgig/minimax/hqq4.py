"""HQQ 4-bit frozen base for MiniMax H3 — by rintic-13 (issue #102).

A third base precision beside the checkpoint's own int8 ConvRot and bitsandbytes NF4. HQQ
(Half-Quadratic Quantization, Badri & Shaji, Mobius Labs) fits the per-group scale/zero with
a proximal solver instead of taking min/max, and at group size 16 it lands roughly a third
closer to the int8 weights the LoRA is deployed on than NF4 does. Measured by rintic-13 on
the real pruned checkpoint against its int8 codes decoded to bf16:

    HQQ 4-bit g16   L2 error 6.29 %   MAE 6.56 %   max 0.227
    HQQ 4-bit g8    L2 error 4.80 %
    NF4             L2 error 9.21 %   MAE 9.14 %   max 0.547

with grad cosine 0.998 against the int8 reference on blocks.25.attn.out_proj, and an 80-epoch
A/B at identical settings finishing 0.2168 avr_loss vs NF4's 0.2318 at the same 1.32 s/it.

Shipped default (3 Sep 2026, #102): GROUP 8 with the per-group scale/zero stored as 8-bit
codes. Plain g8 doubles the group-vector overhead to 1.0 B/param — int8's own footprint, at
which point int8 (0.17 % error) wins outright — so the group vectors are quantized too: one
affine (lo, step) per OUTPUT ROW for scale and for zero, uint8 codes. Measured on a real fc1
and on random weights: 4.83 % vs plain g8's 4.80 %, at 0.75 B/param — the same footprint as
g16 for a third less error. rintic-13 measured g8 at ~6 % slower than g16 on a 5060 Ti (the
step there is PCIe-bound, so the dequant hides behind the streaming); on a 5090 at swap 0
the PyTorch dequant path shows as ~half NF4's step speed for either group size.

What lives here is Fizgig's OWN Linear over hqq's quantizer, not hqq's `HQQLinear`. That class
is not an nn.Linear (the LoRA scan would skip it), keeps scale/zero in a plain dict that
`Module.to()` never sees, and overrides `.to()`/`.cpu()` as no-ops — so the preview park,
`restore_parked_dit`, and the loader's block parking would all silently leave tensors behind.
Holding the packed codes and the two group vectors as ordinary buffers makes every existing
move path work unchanged, and the ring streamer (h3_hqq_h2d_offload.py) rebinds three named
buffers per module. The quantization itself IS hqq's (`Quantizer.quantize` — the proximal
optimizer is the whole win) and the dequant is hqq's too, so what trains here is exactly what
rintic-13 measured.

Cost: 0.5 B/param of codes plus two uint8 group vectors at 1/8 → 0.75 B/param (+ a
negligible [out, 4] fp32 row table), against NF4's ~0.52 (its double-quantized absmax at
block 64). Resident footprint is ~45 % larger than NF4 for the same model; the H2D ring
covers the difference on small cards.

hqq is pinned in requirements.txt; the install/update scripts set DISABLE_CUDA=1 so its
optional CUDA kernel is never built — Fizgig only uses the PyTorch path, which is the one
rintic-13 benchmarked.
"""

import torch
import torch.nn as nn

HQQ_NBITS = 4
HQQ_GROUP_SIZE = 8           # g8 + 8-bit group vectors: 4.83 % error at g16's 0.75 B/param
_PACKING = "4bit_u8"
_META_LEVELS = 255           # scale/zero codes: uint8, affine per output row


def _quantizer():
    try:
        from hqq.core.quantize import Quantizer
    except ImportError as e:            # the package is optional — name the fix, not a traceback
        raise ImportError(
            "Base Precision '4-bit HQQ' needs the hqq package, which this venv predates. "
            "Run the Fizgig updater (update_fizgig.bat / update_fizgig_rocm.bat, or the "
            "install script on Linux) and it installs with the rest of the requirements. "
            "Or pick '4-bit' (bitsandbytes NF4) / 'int8' instead.") from e
    return Quantizer


def hqq_available() -> bool:
    try:
        import hqq.core.quantize  # noqa: F401
        return True
    except ImportError:
        return False


def _meta(shape, scale, zero, compute_dtype, group_size):
    """The dict hqq's dequantize reads — rebuilt per call from the module's buffers so the
    tensors it sees are whatever the module is CURRENTLY bound to (ring slot or CPU master)."""
    return {"nbits": HQQ_NBITS, "group_size": int(group_size), "shape": tuple(shape),
            "scale": scale, "zero": zero, "axis": 1, "packing": _PACKING,
            "unpack_view_dtype": torch.uint8, "view_as_float": False,
            "compute_dtype": compute_dtype}


@torch.no_grad()
def _quantize_group_vectors(scale, zero, out_features):
    """bf16/fp32 [groups, 1] scale and zero -> uint8 codes + a per-ROW affine.

    hqq lays groups out row-major (W.reshape(-1, g)), so [groups, 1] reshapes to
    [out, groups_per_row]; one (lo, step) pair per row keeps the codes within a few
    hundredths of a percent of the bf16 originals (measured: 4.83 % vs 4.80 % base error).
    qmeta is [out, 4] = (scale_lo, scale_step, zero_lo, zero_step), fp32 values carried in an
    int32 buffer."""
    def _q(t):
        v = t.float().reshape(out_features, -1)
        lo = v.amin(1, keepdim=True)
        step = (v.amax(1, keepdim=True) - lo).clamp_min(1e-12) / _META_LEVELS
        codes = ((v - lo) / step).round().clamp(0, _META_LEVELS).to(torch.uint8)
        return codes.reshape(t.shape), lo.reshape(-1), step.reshape(-1)
    s_codes, s_lo, s_step = _q(scale)
    z_codes, z_lo, z_step = _q(zero)
    qmeta = torch.stack((s_lo, s_step, z_lo, z_step), dim=1).contiguous()   # [out, 4] fp32
    # Stored as the fp32 BIT PATTERNS in an int32 buffer: Module.to(dtype) casts every
    # floating buffer it finds, and a stray .to(bfloat16) on the DiT would otherwise degrade
    # the affine (measured: grad cosine 0.9986 vs 1.0). Integer buffers are never cast.
    return s_codes.contiguous(), z_codes.contiguous(), qmeta.view(torch.int32)


def _dequantize_group_vectors(scale_q, zero_q, qmeta, compute_dtype):
    """uint8 codes + [out, 4] row table -> [groups, 1] scale and zero in the compute dtype."""
    out_features = qmeta.shape[0]
    qm = qmeta.view(torch.float32) if qmeta.dtype == torch.int32 else qmeta.float()
    s = scale_q.reshape(out_features, -1).to(torch.float32) * qm[:, 1:2] + qm[:, 0:1]
    z = zero_q.reshape(out_features, -1).to(torch.float32) * qm[:, 3:4] + qm[:, 2:3]
    return s.reshape(scale_q.shape).to(compute_dtype), z.reshape(zero_q.shape).to(compute_dtype)


@torch.no_grad()
def quantize_hqq4(w: torch.Tensor, group_size: int = HQQ_GROUP_SIZE,
                  compute_dtype=torch.bfloat16, device="cuda"):
    """(W_q uint8 [numel/32, g], scale_q uint8 [numel/g, 1], zero_q uint8 [numel/g, 1],
    qmeta fp32 [out, 4]) for a dense weight.

    Exactly `BaseQuantizeConfig(nbits=4, group_size=g, axis=1)` as HQQLinear would apply it
    (channel_wise, proximal optimize, round_zero for 4-bit), then the group vectors are
    quantized to 8-bit per output row (see _quantize_group_vectors). Runs on `device`: the
    solver is ~20 iterations over the whole tensor and is CPU-slow for fc1-sized weights."""
    Q = _quantizer()
    dev = torch.device(device)
    if w.numel() % (2 * group_size):
        raise ValueError(f"HQQ 4-bit needs numel divisible by 2*group_size "
                         f"({2 * group_size}); got shape {tuple(w.shape)}")
    W_q, meta = Q.quantize(w.to(dev), nbits=HQQ_NBITS, channel_wise=True,
                           group_size=int(group_size), optimize=True, round_zero=True,
                           axis=1, bitpack=True, compute_dtype=compute_dtype,
                           view_as_float=False, device=str(dev))
    scale_q, zero_q, qmeta = _quantize_group_vectors(meta["scale"], meta["zero"], w.shape[0])
    return W_q.contiguous(), scale_q, zero_q, qmeta


def dequantize_hqq4(W_q, scale_q, zero_q, qmeta, shape, compute_dtype,
                    group_size=HQQ_GROUP_SIZE):
    """The dense [out, in] weight in the compute dtype — the group vectors' affine, then
    hqq's own unpack + affine."""
    Q = _quantizer()
    scale, zero = _dequantize_group_vectors(scale_q, zero_q, qmeta, compute_dtype)
    return Q.dequantize(W_q, _meta(shape, scale, zero, compute_dtype, group_size))


class _HQQ4LinearFn(torch.autograd.Function):
    """Dequantize-in-forward AND dequantize-again-in-backward, saving only the packed tensors
    — the same shape as hqq's own `HQQMatmulNoCacheMul` (its default PyTorch backend, what
    rintic-13 measured) and as ConvRot's `_Int8RotLinearFn`: no dense weight survives the
    forward, so nothing is retained per layer through backward. The dense transient
    (fc1: 28672x5376 bf16 = 308 MB) exists only inside each call."""

    @staticmethod
    def forward(ctx, x, W_q, scale_q, zero_q, qmeta, bias, shape, dt, group_size):
        ctx.save_for_backward(W_q, scale_q, zero_q, qmeta)
        ctx.shape, ctx.dt, ctx.gs = shape, dt, group_size
        W = dequantize_hqq4(W_q, scale_q, zero_q, qmeta, shape, dt, group_size)
        y = torch.nn.functional.linear(x.to(dt), W)
        del W
        return y if bias is None else y + bias.to(dt)

    @staticmethod
    def backward(ctx, grad_out):
        W_q, scale_q, zero_q, qmeta = ctx.saved_tensors
        W = dequantize_hqq4(W_q, scale_q, zero_q, qmeta, ctx.shape, ctx.dt, ctx.gs)
        gx = grad_out.to(ctx.dt) @ W                        # [..., out] @ [out, in]
        return gx, None, None, None, None, None, None, None, None


class HQQ4bitLinear(nn.Linear):
    """A frozen Linear holding HQQ 4-bit codes + 8-bit per-group scale/zero codes + their
    per-row affine table as BUFFERS (W_q, scale, zero, qmeta).

    nn.Linear subclass so the LoRA/LoKR module scan and `_dotted_names` (isinstance
    nn.Linear) see it; the class NAME is also listed in networks/lora.py's whitelist, as every
    quantized stand-in must be. The dense `weight` property materializes on demand for
    inspection only — the forward never calls it."""

    def __init__(self, in_features, out_features, bias=False, group_size=HQQ_GROUP_SIZE,
                 compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        del self._parameters["weight"]          # the packed buffers replace it
        self.group_size = int(group_size)
        self.compute_dtype = compute_dtype
        n = in_features * out_features
        if n % (2 * self.group_size):
            raise ValueError(f"HQQ4bitLinear: {out_features}x{in_features} is not divisible "
                             f"by 2*group_size ({2 * self.group_size})")
        groups = n // self.group_size
        # Shapes follow hqq's axis=1 layout: W.reshape(-1, g) → codes packed two-per-byte
        # along dim 0 (BitPack.pack_4bit_u8), scale/zero one per group.
        self.register_buffer("W_q", torch.empty(groups // 2, self.group_size, dtype=torch.uint8),
                             persistent=False)
        self.register_buffer("scale", torch.empty(groups, 1, dtype=torch.uint8),
                             persistent=False)
        self.register_buffer("zero", torch.empty(groups, 1, dtype=torch.uint8),
                             persistent=False)
        # int32 holding fp32 bit patterns — immune to Module.to(dtype), see _quantize_group_vectors
        self.register_buffer("qmeta", torch.empty(out_features, 4, dtype=torch.int32),
                             persistent=False)

    @property
    def weight(self):
        return dequantize_hqq4(self.W_q, self.scale, self.zero, self.qmeta,
                               (self.out_features, self.in_features),
                               self.compute_dtype, self.group_size)

    @torch.no_grad()
    def load_dense(self, w: torch.Tensor, device=None):
        """Quantize a dense weight into this module's buffers (on `device`, default: where
        the buffers already live — the loader passes the GPU and parks afterwards)."""
        dev = torch.device(device) if device is not None else self.W_q.device
        W_q, scale, zero, qmeta = quantize_hqq4(w, self.group_size, self.compute_dtype, dev)
        self.W_q, self.scale, self.zero, self.qmeta = W_q, scale, zero, qmeta

    def forward(self, x):
        if self.bias is not None and self.bias.requires_grad:
            raise RuntimeError("HQQ4bitLinear holds a FROZEN base weight; its bias must not "
                               "require grad (the custom backward returns no bias gradient)")
        return _HQQ4LinearFn.apply(x, self.W_q, self.scale, self.zero, self.qmeta, self.bias,
                                   (self.out_features, self.in_features),
                                   self.compute_dtype, self.group_size)
