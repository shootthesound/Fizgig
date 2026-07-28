"""CLI entry point for Klein 9B LoRA training.

Usage:
    accelerate launch src/fizgig/scripts/train.py --dit path/to/dit --vae path/to/ae ...

This is the script the GUI calls via subprocess.
"""

import sys
import os

# Ensure fizgig package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# CUDA allocator policy, set before torch is imported below — the backend is fixed at CUDA init.
# Training churns the allocator: large tensors allocated and freed every step, and on a rotating
# fine-tune whole windows swap between bf16 and fp8 each epoch. The default allocator carves from
# fixed-size segments, which fragments under that pattern and worsens as a run goes on.
# The GUI already sets this and the training subprocess inherits it; this covers headless runs.
# Respects an existing value, and FIZGIG_NO_EXPANDABLE=1 opts out for A/B testing.
#
# Not on Windows: the CUDA allocator there rejects the option outright ("expandable_segments not
# supported on this platform") and falls back to the default allocator, so setting it bought
# nothing and printed a warning on every process launch.
if (
    sys.platform != "win32"
    and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# OpenMP wait policy, before torch loads libiomp: Intel OpenMP keeps every pool thread
# actively spinning for 200 ms after each parallel region, so the small per-step CPU ops
# re-arm an all-core busy-spin for the whole run (issue #18 — 100% CPU on every core while
# the actual training is on the GPU). BLOCKTIME=0 measured: 14.8 spinning cores -> 0, no
# step-time cost. The GUI sets this too; this covers headless runs. setdefault, so an
# explicit user value wins.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fizgig.training.trainer import KleinTrainer, setup_parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    trainer = KleinTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
