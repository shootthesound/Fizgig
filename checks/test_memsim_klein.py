"""Simulated GPU memory (memsim.py), Klein 9B: the page's own launch - caching, training, previews - at full size on a
simulated card, in seconds on the CPU. A change that raises Klein's card memory fails here before a user's card runs
out: the training step's peak against the figure measured on a 5090, and Auto's plan on 16 and 12 GB cards (what a
Windows desktop leaves of them) completing with its previews.

Needs a torch built with CUDA (no card): skipped on the CPU-only build."""
import pytest

from memsim import CUDA_BUILT, run_launch

pytestmark = pytest.mark.skipif(not CUDA_BUILT, reason="the simulated card needs a torch built with CUDA (no card)")

DESKTOP_GB = 1.3            # what Windows' desktop holds of a card (a 16 GB card leaves 14.7)


def _say(r):
    stages = ", ".join(f"{k} {a:.2f} / {v:.2f} GB" for k, (a, v) in r.stages.items())
    return f"{stages} (allocated / reserved); error: {r.error!r}"


def _completes(r):
    """Every stage finished on the card less the margin (memsim.margin), no preview turned off by a failure."""
    assert r.error is None and r.ooms == 0, _say(r)
    assert not [ln for ln in r.lines if "preview failed" in ln], _say(r)
    for name, (_a, reserved) in r.stages.items():
        assert reserved <= r.usable + 1e-6, f"{name} reserved {reserved:.2f} GB of {r.usable:.2f}: {_say(r)}"


def test_training_step_memory_int8(tmp_path):
    """INT8, no swap, 1 MP, no previews: measured on a 5090 (Windows), peak VRAM 14.1 GB; simulated 13.3 allocated /
    14.1 reserved. A rise past 0.5 GB fails; so does a fall past 1.5 GB (the simulation no longer seeing the step)."""
    r = run_launch("klein", tmp_path, {"FAMILY_PRECISION": "INT8 (8-bit, fastest)", "BLOCKS_SWAP": "0",
                                       "SAMPLE_ENABLED": False}, card_gb=32, others_gb=1.9)
    _completes(r)
    allocated, reserved = r.stages["Training"]
    assert 13.26 - 1.5 < allocated < 13.26 + 0.5, _say(r)
    assert 14.06 - 1.5 < reserved < 14.06 + 0.5, _say(r)


@pytest.mark.parametrize("card,with_room", [(16, True), (12, False)])
def test_auto_plan_fits_with_previews(tmp_path, card, with_room):
    """Every setting at its default (Auto precision and swap, previews on), 1 MP: each stage stays inside the card,
    with no out-of-memory and no preview turned off. 16 GB fits with the margin to spare. 12 GB fits inside the margin:
    the caption caching (Qwen3-8B as fp8, any picture size) reserves 10.5 GB of the 10.7 a Windows desktop leaves,
    measured 10.54 on a 5090 - checked without the margin, so any rise there fails."""
    r = run_launch("klein", tmp_path, {"SAMPLE_ENABLED": True}, card_gb=card, others_gb=DESKTOP_GB,
                   with_room=with_room)
    _completes(r)

