"""The Auto plan on a card under 20 GB reads a family's small-card figures (train_memory_small), since the trainer
parks the model for each preview's decode there; other cards and families keep train_memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import quant  # noqa: E402
from fizgig.families.registry import get  # noqa: E402


def test_fibo_16gb_trains_int8_without_swap(monkeypatch):
    desc = get("fibo2")
    drv = desc.load_driver()
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "1")
    assert quant.plan(desc, drv, "auto", -1, free_gb=15.0, megapixels=0.25)[:2] == ("int8", 0)
    assert quant.plan(desc, drv, "auto", -1, free_gb=15.0, megapixels=1.0)[:2] == ("nf4", 0)
    assert quant.plan(desc, drv, "auto", -1, free_gb=11.2, megapixels=0.25)[:2] == ("nf4", 0)


def test_big_card_keeps_the_preview_peaks(monkeypatch):
    desc = get("fibo2")
    drv = desc.load_driver()
    monkeypatch.setenv("FIZGIG_PREVIEW_LOWMEM", "0")
    assert quant.plan(desc, drv, "auto", -1, free_gb=22.0, megapixels=0.25)[:2] == ("int8", 0)
    p, swap, _ = quant.plan(desc, drv, "int8", -1, free_gb=15.0, megapixels=0.25)
    assert p == "int8" and swap > 0                    # without the small-card treatment, 15 GB needs swap
