"""Synthetic inference attention benchmark; run on the target AMD GPU, not CI.

    venv/bin/python checks/bench_rocm_sdpa.py > sdpa-results.json

Reports before/after timing and peak allocated VRAM for each context, using the
same tensors. These are attention-only figures, not full-model preview results.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fizgig.utils.gpu_backend import fix_rocm_sdpa_masks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2560, 4352])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("Run this benchmark on the target ROCm GPU")
    if args.iterations < 1 or any(n < 1 for n in args.tokens):
        parser.error("Tokens and iterations must be positive")
    fix_rocm_sdpa_masks()  # Retain upstream's mask correctness fix in both cases.
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    rows = []
    for tokens in args.tokens:
        q, k, v = [torch.randn(1, 48, tokens, 128, device=device, dtype=torch.bfloat16) for _ in range(3)]
        for masked in (False, True):
            mask = None
            if masked:
                mask = torch.arange(tokens, device=device)[None, None, None, :] < tokens * .9
            backends = [SDPBackend.CUDNN_ATTENTION]
            if not masked:
                backends.append(SDPBackend.FLASH_ATTENTION)
            backends += [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            for label, context in (("before", lambda: sdpa_kernel(backends, set_priority=True)),
                                   ("after", nullcontext)):
                try:
                    with context():
                        for _ in range(3):
                            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                        torch.cuda.synchronize(device)
                        rows.append(dict(tokens=tokens, masked=masked, context=label,
                                         ms=(time.perf_counter() - start) * 1000 / args.iterations,
                                         peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20))
                except RuntimeError as exc:
                    rows.append(dict(tokens=tokens, masked=masked, context=label, error=str(exc)))
                    torch.cuda.empty_cache()
        del q, k, v
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip,
                         gpu=torch.cuda.get_device_name(device), results=rows), indent=2))


if __name__ == "__main__":
    main()
