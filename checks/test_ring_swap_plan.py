"""Auto's block-swap plan for a family that swaps through an H2D ring (MiniMax H3 Image): the ring's 2 card slots each
hold a streamed block, so the first 2 swapped blocks free nothing (measured by checks/test_simgpu_families.py
test_card_memory_saved_per_swapped_block). The planner used to count every swapped block, so a plan of 1-9 swapped
leaned on up to 0.58 GB the ring never gave back - inside the 1.5 GB margin, so no run ran out, but the margin was
half spent on exactly the cards that only just need swap."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import quant  # noqa: E402
from fizgig.families.registry import FAMILIES, get  # noqa: E402


@pytest.fixture
def h3i(monkeypatch):
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "0")        # the big-card figures, whatever card runs the check
    desc = get("minimax_image")
    return desc, desc.load_driver()


def test_the_ring_frees_nothing_for_its_first_two_blocks(h3i):
    desc, _ = h3i
    assert desc.swap_ring_slots == 2
    per = desc.train_memory["int8"][1]
    assert quant.swap_saved_gb(desc, per, 1) == 0 and quant.swap_saved_gb(desc, per, 2) == 0
    assert quant.swap_saved_gb(desc, per, 5) == pytest.approx(3 * per)
    need = quant._peak(desc.train_memory["int8"], 1.0)
    # 0.1 GB short of fitting unswapped: 2 swapped blocks (all ring slots) don't make it fit
    assert not quant.fits(desc, "int8", 2, need + 1.5 - 0.1, megapixels=1.0)
    assert quant.fits(desc, "int8", 3, need + 1.5 - 0.1, megapixels=1.0)


@pytest.mark.parametrize("free", [24.1, 24.0, 23.5, 23.0, 22.0, 20.0, 18.0, 16.0])
def test_a_ring_plan_frees_what_it_is_short(h3i, free):
    """Every swapped plan passes the ring's slots and frees what the card is short of."""
    desc, drv = h3i
    p, n, _ = quant.plan(desc, drv, "int8", -1, free_gb=free, megapixels=1.0)
    need, per = quant._peak(desc.train_memory["int8"], 1.0), desc.train_memory["int8"][1]
    short = need - (free - 1.5)
    assert n > desc.swap_ring_slots, (free, n)
    assert quant.swap_saved_gb(desc, per, n) >= short - 1e-9, (free, n)
    assert quant.swap_saved_gb(desc, per, n - 1) < short, (free, n)          # and is the fewest that do


def test_the_measured_point_still_fits(h3i):
    """10 swapped measured 19.8 GB (1 MP, int8, preview peak): a card with exactly that budget plans at least 10."""
    desc, drv = h3i
    _, n, _ = quant.plan(desc, drv, "int8", -1, free_gb=19.8 + 1.5, megapixels=1.0)
    assert n >= 10


def test_studio_swaps_past_the_ring_slots(h3i, monkeypatch):
    desc, drv = h3i
    from fizgig.families.workbench import WorkbenchEngine
    from fizgig.minimax_image import workbench as W
    got = {}
    monkeypatch.setattr(W, "_free_vram_gb", lambda: 13.5)
    monkeypatch.setattr(WorkbenchEngine, "ensure_pipeline",
                        lambda self, *a, precision="auto", blocks_to_swap=0, **kw: got.update(p=precision,
                                                                                               n=blocks_to_swap))
    eng = object.__new__(W.MiniMaxH3ImageWorkbenchEngine)
    eng.pipeline, eng.desc, eng.driver = None, desc, drv
    eng.ensure_pipeline("model.safetensors", precision="auto", blocks_to_swap=0)
    peaks, per = desc.train_memory[got["p"]]
    short = max(gb for _, gb in peaks) - (13.5 - 1.5)
    assert short > 0 and got["n"] > desc.swap_ring_slots
    assert quant.swap_saved_gb(desc, per, got["n"]) >= short


def test_only_ring_families_count_slots():
    """The exchange offloaders free a block per swapped block: no slots anywhere else."""
    assert {k for k, d in FAMILIES.items() if getattr(d, "swap_ring_slots", 0)} == {"minimax_image"}
