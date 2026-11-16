"""Compare the original Krea 2 Sage wrapper with the ComfyUI-style wrapper on ROCm.

    venv/bin/python checks/bench_krea2_sageattention.py > krea2-sage-results.json

Synthetic attention only: no model download, denoising, weight offload or VAE.
Reports repeated timings, allocated VRAM, output difference, and errors against FP32 math attention.
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fizgig.krea2 import attention as A


def before(q, k, v, params):
    """The previous fixed-length split path: native GQA and Sage's default options."""
    outputs = []
    for i in range(q.shape[0]):
        length = params.seqlens[i]
        out = A.sageattn(q[i:i+1, :length].transpose(1, 2),
                        k[i:i+1, :length].transpose(1, 2),
                        v[i:i+1, :length].transpose(1, 2), tensor_layout="HND", is_causal=False)
        outputs.append(F.pad(out, (0, 0, 0, params.max_seqlen - out.shape[2])))
    return torch.cat(outputs).transpose(1, 2).flatten(2)


@torch.no_grad()
def reference(q, k, v, params):
    """FP32 math attention, chunked over queries to avoid a full quadratic VRAM allocation."""
    out = q.new_zeros((q.shape[0], q.shape[1], q.shape[2] * q.shape[3]), dtype=torch.float32)
    groups = q.shape[2] // k.shape[2]
    with sdpa_kernel(SDPBackend.MATH):
        for i, length in enumerate(params.seqlens.tolist()):
            kh = k[i:i+1, :length].transpose(1, 2).float().repeat_interleave(groups, dim=1)
            vh = v[i:i+1, :length].transpose(1, 2).float().repeat_interleave(groups, dim=1)
            for start in range(0, length, 256):
                end = min(start + 256, length)
                part = F.scaled_dot_product_attention(q[i:i+1, start:end].transpose(1, 2).float(), kh, vh)
                out[i:i+1, start:end] = part.transpose(1, 2).flatten(2)
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2560, 4352])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1 or any(n <= 32 for n in args.tokens):
        parser.error("iterations/repeats must be positive; tokens must exceed 32")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("Run on the target ROCm GPU")
    if not callable(A.sageattn):
        parser.error("Install the same SageAttention build as ComfyUI in this Python environment")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    rows = []
    for tokens in args.tokens:
        for batch, ragged in ((1, False), (2, False), (2, True)):
            q = torch.randn(batch, tokens, 48, 128, device=device, dtype=torch.bfloat16)
            k, v = [torch.randn(batch, tokens, 12, 128, device=device, dtype=q.dtype) for _ in range(2)]
            lengths = torch.tensor([tokens - 17 - (i * 7 if ragged else 0) for i in range(batch)], device=device)
            mask = torch.arange(32, device=device)[None, :] < (lengths - (tokens - 32))[:, None]
            params = A.AttentionParams.create_attention_params_from_mask("sageattn", True, tokens - 32, mask)
            variants = {"before": lambda: before(q, k, v, params),
                        "after": lambda: A.attention(q, k, v, params)}
            for run in variants.values():
                for _ in range(3):
                    run()
            torch.cuda.synchronize(device)
            for repeat in range(args.repeats):
                # Alternate order to reduce clock/temperature bias.
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for label in order:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        variants[label]()
                    torch.cuda.synchronize(device)
                    rows.append(dict(tokens=tokens, batch=batch, ragged=ragged, repeat=repeat, context=label,
                                     ms=(time.perf_counter() - start) * 1000 / args.iterations,
                                     peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20))
            old, new = variants["before"]().float(), variants["after"]().float()
            if not torch.isfinite(old).all() or not torch.isfinite(new).all():
                raise RuntimeError(f"Non-finite Sage output: tokens={tokens}, batch={batch}, ragged={ragged}")
            delta = ((new - old).norm() / old.norm().clamp_min(1e-12)).item()
            # Outside the timing/memory measurements: neither approximate kernel is ground truth.
            ref = reference(q, k, v, params)
            if not torch.isfinite(ref).all():
                raise RuntimeError("Non-finite FP32 reference output")
            denom = ref.norm().clamp_min(1e-12)
            errors = {"before": ((old - ref).norm() / denom).item(),
                      "after": ((new - ref).norm() / denom).item()}
            for row in rows[-2 * args.repeats:]:
                row["relative_l2_difference"] = delta
                row["relative_l2_error_vs_fp32"] = errors[row["context"]]
            del q, k, v, old, new, ref, denom
    try:
        sage_version = version("sageattention")
    except PackageNotFoundError:
        sage_version = "unknown"
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip, sageattention=sage_version,
                         gpu=torch.cuda.get_device_name(device), iterations=args.iterations, results=rows), indent=2))


if __name__ == "__main__":
    main()
