"""When the cache stages run: a fresh run with Enable Cache on, and a resume only when its picture size changed -
and then whatever Enable Cache says, because nothing is cached at the new size. Never for a prompt slider."""
import pytest

from fizgig.families.launch import caches
from fizgig.families.registry import get as get_family


@pytest.fixture
def desc():
    return get_family("moss_image")


def test_fresh_run_caches_when_enabled(desc):
    assert caches(desc, {"enable_cache": True})
    assert caches(desc, {})                        # on unless the run says otherwise
    assert not caches(desc, {"enable_cache": False})


def test_resume_caches_only_at_a_changed_size(desc):
    assert not caches(desc, {"resuming": True, "enable_cache": True})
    assert caches(desc, {"resuming": True, "recache": True, "enable_cache": True})


def test_resume_at_a_changed_size_caches_whatever_the_toggle_says(desc):
    # a resume at a bigger Target Megapixels with Enable Cache unticked went straight to training and found
    # nothing at that size to train on: the size change makes the cache pass a dependency, not a courtesy
    assert caches(desc, {"resuming": True, "recache": True, "enable_cache": False})
    assert not caches(desc, {"resuming": True, "recache": False, "enable_cache": False})


def test_desktop_resume_key_counts_as_resuming(desc):
    save = "/tmp/whatever-000005-state"
    assert caches(desc, {"RESUME_TRAINING": save, "recache": True, "enable_cache": False})
    assert not caches(desc, {"RESUME_TRAINING": save, "enable_cache": False})


def test_prompt_slider_never_caches(desc):
    slider = {"FAMILY_SLIDER": True, "FAMILY_SLIDER_SOURCE": "prompts"}
    assert not caches(desc, {**slider, "enable_cache": True})
    assert not caches(desc, {**slider, "resuming": True, "recache": True, "enable_cache": True})


def test_isolation_builds_its_side_by_sides_every_start(desc):
    assert caches(desc, {"FAMILY_MULTI_PERSON": True, "enable_cache": False})
