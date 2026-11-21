"""Measure what the Prompt Enhancer costs, and prove what must not drift.

    python -m fizgig.scripts.check_text_boost                    formula parity - any machine, seconds
    python -m fizgig.scripts.check_text_boost --fusion            adds a real text fusion at Krea 2's shape
    python -m fizgig.scripts.check_text_boost --weights <file>    the real fusion, from a Krea 2 DiT file
    python -m fizgig.scripts.check_text_boost --device cuda       where to build it

Reads no prefs and no dataset; `--weights` opens one safetensors file and takes only its
`txtfusion.*` tensors. Prints one table - paste it into docs/KREA2.md and krea2/text_boost.py's
docstring, with the date and the machine. A fusion built by `--fusion` is the family's SHAPE with
random weights: honest for cost and numerics, not a production render.
"""
from __future__ import annotations

import argparse
import time

import torch

from fizgig.krea2 import text_boost as tb
from fizgig.krea2.model import AttentionParams, TextFusionTransformer

# The text side of single_mmdit_large_wide, the config shipped with the OSS checkpoints (krea2/utils.py)
TAP_LAYERS, TXT_DIM, TXT_HEADS, TXT_KVHEADS, TXT_MULT = 12, 2560, 20, 20, 4


LOG = []      # every printed line, so --log can hand a run to someone else to read back later


def _row(label, value=""):
    LOG.append(f"{label:<52} {value}")
    print(f"  {label:<46} {value}")


def _rms(t):
    return t.float().pow(2).mean(-1).sqrt()


def formula_parity():
    """The maths the donor ships, checked against values computed by hand - no model involved."""
    print("\n[1] Formula parity (hand-computed references)")
    g = tb.gains(1.0, "cpu", torch.float32)
    assert torch.allclose(g, torch.tensor(list(tb.PROFILE_12) * 2, dtype=torch.float32))
    _row("gains at s=1 are the donor's profile, tiled", "ok")
    for s, want in ((0.5, 3.0), (2.0, 9.0)):
        got = tb.gains(s, "cpu", torch.float32)[8].item()
        assert abs(got - want) < 1e-6, (s, got, want)
    _row("gains[8] = 1 + s*(5-1) at s=0.5 and s=2", "ok")
    x = torch.ones(1, 1, tb.TAPS, tb.TAP_DIM)
    c8 = tb.scaled_context(x, 1.0)[0, 0].reshape(tb.CHUNKS, tb.CHUNK_DIM)
    assert abs(c8[8][0].item() - 5.0 * 15.0) < 1e-3 and abs(c8[0][0].item() - 1.0 * 15.0) < 1e-3
    _row("chunk 8 = gain 5.0 x global 15.0; chunk 0 = 1.0 x 15.0", "ok")
    assert tb.scaled_context(x, 0.0) is x
    _row("s=0 returns the conditioning itself, unmodified", "ok")
    gen = torch.Generator().manual_seed(11)
    worst = 0.0
    for scale in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4):
        ref = torch.randn(4, 5, tb.TAP_DIM, generator=gen)
        cand = ref + scale * torch.randn(4, 5, tb.TAP_DIM, generator=gen)
        out = tb.capped_residual(ref, cand)
        ratio = (_rms(out - ref) / _rms(ref).clamp_min(1e-8)).max().item()
        worst = max(worst, ratio)
        assert ratio <= 0.75 + 1e-5, (scale, ratio)
    _row(f"cap holds over 7 residual scales (worst {worst:.4f} <= 0.75)", "ok")
    assert tb.capped_residual(ref.to(torch.bfloat16), cand.to(torch.bfloat16)).dtype == torch.bfloat16
    _row("cap maths runs in float32, result back in the input dtype", "ok")
    for bad in (None, "", "nonsense", -1, float("nan")):
        assert tb.clamped(bad) == 0.0
    assert tb.clamped(50) == 2.0 and tb.text_scale(None) == 1.0 and tb.text_scale(50) == 4.0
    _row("strength / text_scale clamped to the donor's ranges", "ok")


def determinism():
    """Once per preview must equal per step, exactly. This is the design claim, not a hope."""
    print("\n[2] Once-per-preview vs the donor's per-step derivation")
    x = torch.randn(1, 8, tb.TAPS, tb.TAP_DIM, generator=torch.Generator().manual_seed(5))
    w = torch.randn(1, 1, tb.TAPS, 1, generator=torch.Generator().manual_seed(6))
    run = lambda c: (c * w).sum(-2)
    once = tb.enhance(run, x, 1.0)
    steps = [tb.enhance(run, x, 1.0) for _ in range(20)]
    identical = all(torch.equal(once, s) for s in steps)
    _row(f"20 recomputations bit-identical to computing once: {identical}")
    assert identical
    plain = run(x)
    ratio = (_rms(once - plain) / _rms(plain)).max().item()
    _row(f"residual admitted, max {ratio:.4f} of the reference's rms (cap 0.75)")
    assert ratio <= 0.75 + 1e-5


def _build_fusion(device, weights=None):
    """A real TextFusionTransformer at Krea 2's text shape: `--fusion` builds it fresh, `--weights`
    fills it from a DiT file (only its `txtfusion.*` tensors), inferring the MLP multiplier."""
    sd = None
    if weights:
        from safetensors import safe_open
        with safe_open(weights, framework="pt") as f:
            sd = {k[len("txtfusion."):]: f.get_tensor(k) for k in f.keys() if k.startswith("txtfusion.")}
        if not sd:
            raise SystemExit(f"{weights} has no txtfusion.* tensors - is it a Krea 2 DiT?")
    if sd is None:
        m = TextFusionTransformer(TAP_LAYERS, TXT_DIM, TXT_HEADS, TXT_MULT, False, TXT_KVHEADS)
        return m.to(device).eval(), f"random init at the shipped config's shape (multiplier {TXT_MULT})"
    for mult in range(1, 9):     # the file's own MLP width says which multiplier it was trained with
        m = TextFusionTransformer(TAP_LAYERS, TXT_DIM, TXT_HEADS, mult, False, TXT_KVHEADS)
        try:
            m.load_state_dict(sd, strict=True)
            return m.to(device).eval(), f"weights from {weights} (multiplier {mult})"
        except RuntimeError:
            continue     # load_state_dict's own word for "these shapes are not this model's"
    raise SystemExit("no MLP multiplier from 1 to 8 fits those txtfusion tensors")


def cost(args):
    """Time and memory for the enhancement, and what the donor's per-step derivation would cost."""
    print(f"\n[3] Cost ({args.device}, {args.tokens} text tokens, {args.reps} reps, {args.steps} steps/preview)")
    fusion, source = _build_fusion(args.device, args.weights)
    _row(f"text fusion: {source}")
    x = torch.randn(1, args.tokens, tb.TAPS, tb.TAP_DIM, device=args.device, dtype=torch.float32)
    mask = torch.ones(1, args.tokens, dtype=torch.bool, device=args.device)
    nomask = AttentionParams.create_attention_params_from_mask("torch", False, 0, None)
    params = AttentionParams.create_attention_params_from_mask("torch", False, 0, mask)
    run = lambda c: fusion(c, nomask, params)
    with torch.no_grad():
        plain = run(x)
        enh = tb.enhance(run, x, 1.0)
        ratio = (_rms(enh - plain) / _rms(plain)).max().item()
        _row(f"enhanced differs from plain by at most {ratio:.4f} rms (cap 0.75)")
        assert ratio <= 0.75 + 1e-5
        assert all(torch.equal(enh, tb.enhance(run, x, 1.0)) for _ in range(args.steps))
        _row(f"per-step recomputation on the real fusion: bit-identical {args.steps} times")

        def timed(fn):
            """Best / mean / worst of N, GPU-synchronised. Without a sync an eager launch returns before the
            card has done the work, so the timer reads the CPU's typing speed rather than the card's - which
            is what made a ~2 ms pass report 0.0021 s best against 0.0107 s mean. Best and worst still show
            whether the card was quiet."""
            cuda = args.device == "cuda" and torch.cuda.is_available()
            if cuda:
                torch.cuda.synchronize()
            for _ in range(2):
                fn()
            if cuda:
                torch.cuda.synchronize()
            marks = []
            for _ in range(args.reps):
                t = time.perf_counter()
                fn()
                if cuda:
                    torch.cuda.synchronize()
                marks.append(time.perf_counter() - t)
            return min(marks), sum(marks) / len(marks), max(marks)

        b_min, b_avg, b_max = timed(lambda: run(x))
        f_min, f_avg, f_max = timed(lambda: tb.enhance(run, x, 1.0))
        base, full = b_min, f_min
        _row(f"one fusion pass: best {b_min:.4f}s  mean {b_avg:.4f}s  worst {b_max:.4f}s")
        _row(f"enhanced (two passes + cap): best {f_min:.4f}s  mean {f_avg:.4f}s  worst {f_max:.4f}s")
        _row(f"spread worst/mean: pass {b_max / b_avg:.2f}x  enhancement {f_max / f_avg:.2f}x"
             + ("   (a quiet card is near 1.0x)" if max(b_max / b_avg, f_max / f_avg) < 1.3 else "   CONTENDED"))
        _row(f"ours, per preview: +{full - base:.4f}s once")
        _row(f"the donor's, per preview: +{full - base:.4f}s x {args.steps} steps "
             f"= {((full - base) * args.steps):.3f}s")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            tb.enhance(run, x, 1.0)
            _row(f"peak reserved for one enhancement: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fusion", action="store_true", help="also measure a real text fusion at Krea 2's shape")
    p.add_argument("--weights", help="a Krea 2 DiT safetensors file: use its txtfusion tensors")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tokens", type=int, default=512, help="text tokens in the conditioning")
    p.add_argument("--steps", type=int, default=20, help="steps per preview, for the donor's cost")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--log", help="append this run to a file - timestamp, torch, device and every figure - "
                                 "so an idle-card measurement can be taken offline and read back")
    args = p.parse_args()
    print("Prompt Enhancer - krea2/text_boost.py measurements")
    formula_parity()
    determinism()
    if args.fusion or args.weights:
        cost(args)
    else:
        print("\n[3] Cost: skipped (pass --fusion for the family's shape, --weights <file> for real weights)")
    print("\nAll checks held.")
    if args.log:
        import datetime
        dev = (torch.cuda.get_device_name(0) if args.device == "cuda" and torch.cuda.is_available()
               else "cpu")
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} · torch "
                    f"{torch.__version__} · {dev} · tokens={args.tokens} reps={args.reps} "
                    f"steps={args.steps}\n" + "\n".join(LOG) + "\n\n")
        print(f"(appended to {args.log})")


if __name__ == "__main__":
    main()
