"""Dave Maybank's fused W8A16 backward (convrot_w8a16_backward_triton.py) through the real
_Int8RotLinearFn: ON by default beside the forward kernel, OFF with
FIZGIG_NO_TRITON_W8A16_BACKWARD=1 or when the forward kernel is opted out, grad_x within one
bf16 ulp of the eager backward on the real ConvRot shapes (rotation included), sticky
fallback to eager when the kernel raises. Needs a CUDA card + Triton; skips otherwise.

Run: venv/Scripts/python.exe tests/test_convrot_w8a16_backward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


if not torch.cuda.is_available():
    print("no CUDA — skipped"); sys.exit(0)
try:
    import triton  # noqa: F401
except ImportError:
    print("no Triton — skipped"); sys.exit(0)

from fizgig.minimax import convrot as C  # noqa: E402
from fizgig.minimax.convrot import _Int8RotLinearFn as Fn  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


def reset(env):
    """env: None = defaults (both kernels on), "off" = backward opted out, "nofwd" = forward
    opted out (which must take the backward with it)."""
    Fn._w8a16_bwd_state.update({"checked": False, "use": False, "announced": False})
    Fn._w8a16_state.update({"checked": False, "use": False, "announced": False})
    os.environ.pop("FIZGIG_NO_TRITON_W8A16_BACKWARD", None)
    os.environ.pop("FIZGIG_NO_TRITON_W8A16", None)
    if env == "off":
        os.environ["FIZGIG_NO_TRITON_W8A16_BACKWARD"] = "1"
    elif env == "nofwd":
        os.environ["FIZGIG_NO_TRITON_W8A16"] = "1"


def grad_x(x, q, s, rot):
    x = x.detach().clone().requires_grad_(True)
    y = Fn.apply(x, q, s, None, rot, torch.bfloat16)
    y.float().square().mean().backward()
    return x.grad.detach()


def ulp_ok(a, b):
    # one bf16 ulp is 2^-7 of the magnitude; allow that plus a hair of absolute slack
    d = (a.float() - b.float()).abs()
    tol = b.float().abs() * (2 ** -7) + 1e-3
    return bool((d <= tol).all()), f"max|d|={d.max().item():.3e} over-tol={(d > tol).sum().item()}"


# real ConvRot shapes (out, in), rotation like the checkpoint's (rot=64 groups)
for (N, K), M in (((2688, 2688), 1666), ((8064, 2688), 300), ((2688, 5376), 4046), ((5376, 2688), 8806)):
    w = torch.randn(N, K, device=dev) * 0.02
    q, s = C.quantize_int8_convrot(w, rot=64)            # a power of 4 that divides every ConvRot dim
    x = torch.randn(1, M, K, device=dev, dtype=torch.bfloat16)
    reset("off")
    g_eager = grad_x(x, q, s, 64)
    ck(f"{N}x{K} M={M}: opted out — the backward stays eager", Fn._w8a16_bwd_state["use"] is False)
    reset(None)
    g_fused = grad_x(x, q, s, 64)
    ck(f"{N}x{K} M={M}: on by default beside the forward kernel", Fn._w8a16_bwd_state["use"] is True)
    ok, det = ulp_ok(g_fused, g_eager)
    ck(f"{N}x{K} M={M}: fused grad_x within one bf16 ulp of eager (rotation included)", ok, det)

reset("nofwd"); grad_x(x, q, s, 64)
ck("forward kernel opted out -> the backward never runs alone", Fn._w8a16_bwd_state["use"] is False)

# sticky fallback: a raising kernel -> eager result, kernel off for the rest of the run
import fizgig.minimax.convrot_w8a16_backward_triton as KB  # noqa: E402
real = KB.fused_w8a16_input_grad
KB.fused_w8a16_input_grad = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    reset(None)
    x = torch.randn(1, 300, 2688, device=dev, dtype=torch.bfloat16)
    q = torch.randint(-127, 128, (2688, 2688), dtype=torch.int8, device=dev); s = torch.rand(2688, 1, device=dev) * 0.02
    g = grad_x(x, q, s, 64)
    reset("off"); g_e = grad_x(x, q, s, 64)
    ck("a raising kernel falls back to the eager backward (same result)", torch.equal(g, g_e))
finally:
    KB.fused_w8a16_input_grad = real
reset(None); grad_x(x, q, s, 64)
KB.fused_w8a16_input_grad = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    grad_x(x, q, s, 64)
    ck("...and stays off afterwards", Fn._w8a16_bwd_state["use"] is False)
finally:
    KB.fused_w8a16_input_grad = real
    reset(None)

# --- the FORWARD kernel's gate (rintic-13, #89): it must engage on a training forward -----
# PyTorch runs a custom Function's forward with grad mode OFF, so the old
# torch.is_grad_enabled() gate was never true and the kernel silently never ran in training
# (v4.3.2 through v5.3.3). The signal is whether the input wants a gradient.
def fwd_reset():
    Fn._w8a16_state.update({"checked": False, "use": False, "announced": False})
m = C.ConvRotInt8Linear(2688, 2688, bias=False).to(dev)
m.qdata, m.wscale, m.rot, m.compute_dtype = q, s, 64, torch.bfloat16
xt = torch.randn(1, 300, 2688, device=dev, dtype=torch.bfloat16, requires_grad=True)
fwd_reset(); m(xt).float().sum().backward()
ck("forward kernel engages on a training forward (input wants a gradient)", Fn._w8a16_state["announced"])
fwd_reset(); torch.utils.checkpoint.checkpoint(m, xt, use_reentrant=False).float().sum().backward()
ck("...and under the model's non-reentrant checkpointing", Fn._w8a16_state["announced"])
fwd_reset()
with torch.no_grad():
    m(xt.detach())
ck("...but not on a preview forward (no_grad, nothing wants a gradient)", not Fn._w8a16_state["announced"])
fwd_reset()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}"); sys.exit(1)
print("ALL PASS")
