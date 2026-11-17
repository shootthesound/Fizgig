"""A 4-bit base that streams blocks (MiniMax H3 Image's NF4 ring: train_memory has a GB-per-block figure for it) gets
the blocks it needs from Auto, instead of a plan past the card's memory with none (#354: a 12 GB card planned NF4
with no swap, 12.8 GB, and trained at 40 s/step on Windows; 16 blocks swapped trained at 4.1). Every other family's
4-bit base still never swaps, and out-of-memory recovery raises block swap only where the base streams."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import launch, quant  # noqa: E402
from fizgig.families.registry import get, training_families  # noqa: E402
from fizgig.web import oom  # noqa: E402


def test_which_bases_stream():
    h3i = get("minimax_image")
    assert h3i.streams_blocks("nf4") and h3i.streams_blocks("hqq") and h3i.streams_blocks("int8")
    for desc in training_families():
        if desc.key != "minimax_image":
            assert not desc.streams_blocks("nf4"), desc.key       # every other nf4 figure saves 0 GB a block
        assert desc.streams_blocks("bf16") and desc.streams_blocks("int8")


def test_a_12gb_card_swaps_the_4bit_base(monkeypatch):
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "0")       # the big-card figures, whatever card runs the check
    desc = get("minimax_image")
    drv = desc.load_driver()
    peak, per_block = quant._peak(desc.train_memory["nf4"], 0.25), desc.train_memory["nf4"][1]
    for free in (10.0, 10.5, 11.0, 11.5):
        for mp in (0.25, 1.0):
            p, n, why = quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=mp)
            assert p == "nf4" and n > 0, (free, mp, n, why)
            assert peak - n * per_block <= free - 1.5, (free, n)      # fits with the usual margin
            assert f"{n} swapped blocks" in why
            p, n2, why = quant.plan(desc, drv, "nf4", -1, free_gb=free, megapixels=mp)   # 4-bit + Auto swap
            assert (p, n2) == ("nf4", n) and f"{n} swapped blocks" in why
    # past what the most swap can hold, it says so and still swaps the most
    p, n, why = quant.plan(desc, drv, "auto", -1, free_gb=4.0, megapixels=1.0)
    assert (p, n) == ("nf4", drv.max_blocks_to_swap()) and "nothing fits" in why and "(the most)" in why
    # what already fitted is unchanged: no swap where 4-bit fits, int8 where int8 fits
    assert quant.plan(desc, drv, "auto", -1, free_gb=14.6, megapixels=0.25)[:2] == ("nf4", 0)
    assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25)[:2] == ("int8", 0)


def test_small_card_figures_plan_fewer_blocks(monkeypatch):
    """Under 20 GB H3 Image plans from its small-card figures (training's peak; the 2048 preview parks tail blocks to
    decode), so the Standard preset (1.4 GB over the figures) on a 12 GB card swaps ~15 blocks, not 30, and Auto keeps
    the 4-bit base even where int8 with most blocks streamed would fit."""
    desc = get("minimax_image")
    drv = desc.load_driver()
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "0")
    assert quant.plan(desc, drv, "auto", -1, free_gb=11.2, megapixels=0.25, adapter_gb=1.4)[:2] == ("nf4", 31)
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "1")
    for free, mp, n in ((11.2, 0.25, 15), (11.2, 1.0, 16), (10.7, 0.25, 18), (15.2, 1.0, 0), (14.7, 0.25, 0)):
        assert quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=mp, adapter_gb=1.4)[:2] == ("nf4", n)
    for free in (4.0, 8.0, 10.0, 12.0, 16.0, 18.0):
        assert quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=1.0)[0] == "nf4"
    assert quant.plan(desc, drv, "int8", -1, free_gb=15.2, megapixels=1.0, adapter_gb=1.4)[:2] == ("int8", 26)


def test_other_families_4bit_never_swaps():
    for desc in training_families():
        if desc.key == "minimax_image" or desc.own_plan or not desc.train_memory or "nf4" not in desc.train_memory:
            continue
        drv = desc.load_driver()
        for free in (4.0, 8.0, 11.0):
            p, n, why = quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=1.0)
            if p == "nf4":
                assert n == 0, (desc.key, free, why)
            assert quant.plan(desc, drv, "nf4", -1, free_gb=free, megapixels=1.0)[1] == 0, desc.key


def test_out_of_memory_recovery_swaps_a_streaming_4bit_base():
    labels = launch.PRECISION_LABELS
    auto = {"FAMILY_PRECISION": labels["auto"], "BLOCKS_SWAP": "Auto (detect from GPU)"}
    h3i = get("minimax_image")
    v, why = oom.step_down(h3i, auto, ("nf4", 16), 40, launch.precision_key, labels)
    assert v["BLOCKS_SWAP"] == "21" and why == "block swap 16 -> 21"
    plain = SimpleNamespace(precisions=("bf16", "int8", "nf4"), precision_labels={}, train_memory={"nf4": (6.0, 0.0)})
    v, why = oom.step_down(plain, auto, ("nf4", 0), 26, launch.precision_key, labels)
    assert v is None and "can't swap blocks" in why
