"""Resume at another picture size: the run caches before it trains, and its record keeps the settings it last
trained with until that pass finishes - so a resume stopped mid-caching is asked again, and one whose cache is
already right goes straight to training."""
import os

import pytest

from fizgig.families.launch import LaunchPlan, Stage
from fizgig.web import history, server


def _plan(*names):
    return LaunchPlan(stages=[Stage(n, ["x"]) for n in names])


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_FILE", str(tmp_path / "runs.json"))
    out = str(tmp_path / "out")
    os.makedirs(out)
    rid = history.start("web", "moss_image", "mine", out, str(tmp_path / "data"),
                        {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": out, "DATASET_MEGAPIXELS": "1.0",
                         "CLIP_MEGAPIXELS": "1.0", "MAX_TRAIN_EPOCHS": "10"})
    history.end(rid, "paused")                        # a run you paused: its Resume carries on this record
    return {"id": rid, "out": out}


def test_plans_that_cache_are_rebuilding():
    assert server._rebuilding_cache(_plan("Cache Preparation", "Text Encoder Caching", "Training"))
    assert server._rebuilding_cache(_plan("Multi-person composites", "Cache Preparation (2.4 MP)",
                                         "Text Encoder Caching (2.4 MP)", "Training"))
    assert not server._rebuilding_cache(_plan("Training"))
    assert not server._rebuilding_cache(LaunchPlan())   # nothing to start


def test_resume_mid_caching_keeps_the_settings_it_trained_with(rec):
    big = {"DATASET_MEGAPIXELS": "2.4", "CLIP_MEGAPIXELS": "2.4"}
    assert server._size_changed("mine", rec["out"], big)
    rid = history.start("web", "moss_image", "mine", rec["out"], "", None, resume=True)
    assert rid == rec["id"]                            # the same record carries on
    assert history.get(rid)["values"]["DATASET_MEGAPIXELS"] == "1.0"
    history.end(rid, "stopped")                        # the caching pass never finished
    assert server._size_changed("mine", rec["out"], big)   # the next one caches again, instead of dying


def test_training_start_records_the_settings_it_cached_to(rec):
    big = {"DATASET_MEGAPIXELS": "2.4", "CLIP_MEGAPIXELS": "2.4"}
    rid = history.start("web", "moss_image", "mine", rec["out"], "", None, resume=True)
    history.set_values(rid, big)                       # as the Training stage starts
    assert history.get(rid)["values"] == {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": rec["out"],
                                         "DATASET_MEGAPIXELS": "2.4", "CLIP_MEGAPIXELS": "2.4",
                                         "MAX_TRAIN_EPOCHS": "10"}
    assert not server._size_changed("mine", rec["out"], big)   # proven: the next resume trains straight away


def test_resume_after_a_finished_run_starts_a_clean_record(rec):
    history.end(rec["id"], "finished")                 # not paused: Resume can't carry on that record
    big = {"DATASET_MEGAPIXELS": "2.4", "CLIP_MEGAPIXELS": "2.4"}
    rid = history.start("web", "moss_image", "mine", rec["out"], "", None, resume=True)
    assert rid != rec["id"] and history.get(rid)["values"] == {}
    assert server._size_changed("mine", rec["out"], big)   # nothing proven yet, so it caches first
    history.set_values(rid, big)
    assert not server._size_changed("mine", rec["out"], big)


def test_resume_with_an_intact_cache_records_at_once(rec):
    same = {"LORA_NAME": "mine", "LORA_OUTPUT_DIR": rec["out"], "DATASET_MEGAPIXELS": "1.0",
            "CLIP_MEGAPIXELS": "1.0", "MAX_TRAIN_EPOCHS": "16"}
    assert not server._size_changed("mine", rec["out"], same)
    rid = history.start("web", "moss_image", "mine", rec["out"], "", same, resume=True)
    assert rid == rec["id"] and history.get(rid)["values"]["MAX_TRAIN_EPOCHS"] == "16"


def test_a_run_fizgig_has_no_record_of_caches(rec):
    assert server._size_changed("other", rec["out"], {"DATASET_MEGAPIXELS": "2.4"})
    rid = history.start("web", "moss_image", "other", rec["out"], "", None)
    assert history.get(rid)["values"] == {}            # a new record: nothing proven yet
    assert server._size_changed("other", rec["out"], {"DATASET_MEGAPIXELS": "2.4"})
    history.set_values(rid, {"DATASET_MEGAPIXELS": "2.4"})
    assert not server._size_changed("other", rec["out"], {"DATASET_MEGAPIXELS": "2.4"})
