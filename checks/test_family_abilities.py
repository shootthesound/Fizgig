"""Every picture family has the abilities most picture families have, or names each one it leaves out in its
description's not_offered with the reason. A missing ability is a decision written down, never an oversight
(docs/drivers/CHECKLIST.md, "What a finished family includes")."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families.registry import training_families  # noqa: E402

# the user-facing abilities a description switches on (fields that are True when offered)
ABILITIES = ("masked_training", "slider_training", "finetune", "activation_cache", "int8_attention", "adaptive_lr",
             "loss_watch", "caption_dropout", "batch_training", "detail_crops", "resolution_stages", "edit_training",
             "preview_park_optimizer", "ema_short_run", "category_masters")


def _picture_families():
    return [d for d in training_families() if "photo" in (d.media or ())]


def expected():
    """The abilities at least half the picture families have: what a picture family is expected to offer."""
    fams = _picture_families()
    return {a for a in ABILITIES if sum(bool(getattr(d, a, False)) for d in fams) * 2 >= len(fams)}


def test_every_picture_family_offers_or_explains():
    want = expected()
    assert {"masked_training", "slider_training", "finetune", "activation_cache"} <= want
    gaps = {}
    for d in _picture_families():
        missing = [a for a in sorted(want) if not getattr(d, a, False) and a not in (d.not_offered or {})]
        if missing:
            gaps[d.key] = missing
    assert not gaps, ("these families lack abilities most picture families have, with no reason in their "
                      f"description's not_offered: {gaps}")


def test_not_offered_names_real_missing_abilities():
    for d in training_families():
        for a, why in (d.not_offered or {}).items():
            assert a in ABILITIES, f"{d.key}: not_offered names {a!r}, not an ability"
            assert not getattr(d, a, False), f"{d.key}: {a} is offered - drop it from not_offered"
            assert str(why).strip(), f"{d.key}: {a} needs a reason"
