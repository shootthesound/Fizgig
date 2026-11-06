"""Batch size Auto (40 GB+ cards): the family's measured fastest batch, and gradient checkpointing off for a batch only
when one picture's figure plus each extra picture's measured cost fits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import launch, quant  # noqa: E402
from fizgig.families.registry import get  # noqa: E402
from fizgig.families.train import _apply_auto_batch  # noqa: E402

RUN = dict(slider=False, finetune=False, dop=False, share=False, stage=False, mask_prior=False, lora=True,
           accumulation=False, paired=False)


def _cfg(side):
    return {"general": {"resolution": [side, side], "batch_size": 1}, "datasets": [{"image_directory": "x"}]}


def test_sweet_spot_by_size():
    assert quant.auto_batch(get("sdxl"), 0.25) == 8 and quant.auto_batch(get("sdxl"), 1.0) == 4
    assert quant.auto_batch(get("sdxl"), 0.5) == 4                 # between points: the larger size's
    assert quant.auto_batch(get("sdxl"), 1.5) == 1                 # past the measured sizes
    assert quant.auto_batch(get("anima"), 1.0) == 1 and quant.auto_batch(get("klein"), 0.25) == 1


def test_auto_batch_only_on_big_cards(monkeypatch):
    desc = get("sdxl")
    drv = desc.load_driver()
    monkeypatch.setattr(quant, "free_vram_gb", lambda: 44.0)
    monkeypatch.setenv("FIZGIG_SIM_VRAM_GB", "48")
    cfg = _cfg(496)
    assert _apply_auto_batch(desc, drv, cfg, "auto", -1, RUN) == 8
    assert cfg["general"]["batch_size"] == 8 and cfg["datasets"][0]["batch_size"] == 8
    monkeypatch.setenv("FIZGIG_SIM_VRAM_GB", "32")
    cfg = _cfg(496)
    assert _apply_auto_batch(desc, drv, cfg, "auto", -1, RUN) == 1 and cfg["general"]["batch_size"] == 1
    monkeypatch.setenv("FIZGIG_SIM_VRAM_GB", "48")
    assert _apply_auto_batch(desc, drv, _cfg(496), "auto", -1, {**RUN, "dop": True}) == 1
    assert _apply_auto_batch(desc, drv, _cfg(496), "auto", -1, {**RUN, "lora": False}) == 1
    assert _apply_auto_batch(get("klein"), get("klein").load_driver(), _cfg(496), "auto", -1, RUN) == 1


def test_checkpointing_off_for_a_batch_needs_the_per_picture_figure(monkeypatch):
    monkeypatch.setenv("FIZGIG_SIM_VRAM_GB", "48")
    sdxl, klein = get("sdxl"), get("klein")
    on, why = quant.plan_checkpointing(sdxl, "bf16", 0, 0.25, free_gb=44.0, pictures=8)
    assert not on and "x 8 pictures" in why                       # 10.3 + 7 x 2.3 + 4 margin = 30.4 GB
    on, _ = quant.plan_checkpointing(sdxl, "bf16", 0, 1.0, free_gb=30.0, pictures=4)
    assert on                                                      # 11.8 + 3 x 5.5 + 4 = 32.3 GB > 30
    on, why = quant.plan_checkpointing(klein, "int8", 0, 0.25, free_gb=44.0, pictures=2)
    assert on and "no measured figure" in why                      # Klein has no per-picture figure
    assert not quant.plan_checkpointing(klein, "int8", 0, 0.25, free_gb=44.0)[0]


def test_launcher_offers_and_passes_auto():
    sdxl, klein = get("sdxl"), get("klein")
    assert "Auto" in launch.batch_choices(sdxl) and "Auto" not in launch.batch_choices(klein)
    assert launch.batch_size(sdxl, {"BATCH_SIZE": "Auto"}) == 1        # the dataset config says 1
    assert launch.batch_auto(sdxl, {"BATCH_SIZE": "Auto"})
    assert not launch.batch_auto(klein, {"BATCH_SIZE": "Auto"}) and not launch.batch_auto(sdxl, {"BATCH_SIZE": "4"})
    assert "8 pictures" in launch.batch_auto_note(sdxl, 0.25)


def test_note_says_what_this_card_does():
    sdxl = get("sdxl")
    assert launch.batch_auto_note(sdxl, 0.25, 23.5).startswith("Auto: 1 picture a step on this 24 GB card")
    assert launch.batch_auto_note(sdxl, 0.25, 79.2).startswith("Auto: 8 pictures a step at 0.25 MP on this 79 GB card")
    assert "40 GB+ card" in launch.batch_auto_note(sdxl, 1.0)          # no card read (no GPU here): the rule


def test_clip_studio_reuses_the_baseline_when_nothing_moved():
    """Studio's clip pair: the page starts every block's donor slider at 1, so with no donor loaded the default state
    is the baseline - one render for both panes; a moved block, or a donor that's loaded and on, renders the Tweaked
    clip."""
    from types import SimpleNamespace
    from fizgig.repair_studio.state import BlockState, SliderState
    from fizgig.scripts.studio_worker import _clip_same_as_baseline
    st = SliderState(prompt="x", seed=1)
    for b in ("double_blocks_0", "double_blocks_1"):
        st.blocks[b] = BlockState(primary_enabled=True, primary_strength=1.0, donor_enabled=True, donor_strength=1.0)
    no_donor = SimpleNamespace(donor_network=None, donor_path=None, donor_block_ids=set())
    assert _clip_same_as_baseline(no_donor, st)
    with_donor = SimpleNamespace(donor_network=object(), donor_path="d.safetensors",
                                 donor_block_ids={"double_blocks_0"})
    assert not _clip_same_as_baseline(with_donor, st)
    st.donor_scale = 0.0
    assert _clip_same_as_baseline(with_donor, st)
    st.blocks["double_blocks_1"].primary_strength = 0.5
    assert not _clip_same_as_baseline(no_donor, st)
