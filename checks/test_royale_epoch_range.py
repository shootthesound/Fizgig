"""LoRA Royale renders part of a run (#318): Epochs from / to beside Max renders. After rendering 12 of 90 epochs and
liking 83, the nearest were 75 and 90 - now 75 to 90 renders that stretch, Max renders spreading its picks inside it.
Empty boxes are the whole run, as before; a folder of unrelated LoRAs has no epochs and renders all of it."""
import pytest

from fizgig.web.royale import epoch_range, pick

RUN = [(e, f"r-{e:06d}.safetensors") for e in range(1, 91)]


def test_empty_boxes_are_the_whole_run():
    assert pick(RUN, "12") == pick(RUN, "12", "", "") == pick(RUN, "12", None, None)
    assert len(pick(RUN, "All", "", "")) == 90


def test_a_range_renders_that_stretch_with_its_ends_in():
    got = [e for e, _ in pick(RUN, "All", "75", "90")]
    assert got == list(range(75, 91))
    got = [e for e, _ in pick(RUN, "6", "75", "90")]
    assert got[0] == 75 and got[-1] == 90 and len(got) == 6
    assert [e for e, _ in pick(RUN, "All", "88", "")] == [88, 89, 90]        # one end empty: to the run's end
    assert [e for e, _ in pick(RUN, "All", "90", "88")] == [88, 89, 90]      # either way round


def test_snapshots_taken_in_the_range_come_along():
    items = [(1, "a"), (2, "b"), ("step 500", "s1"), (3, "c"), ("step 900", "s2"), (4, "d")]
    assert [e for e, _ in epoch_range(items, "3", "3")] == ["step 500", 3]   # taken during epoch 3
    assert [e for e, _ in epoch_range(items, "1", "2")] == [1, 2]


def test_a_folder_of_unrelated_loras_has_no_range():
    items = [("alpha", "a"), ("beta", "b")]
    assert epoch_range(items, "3", "9") == items


def test_bad_or_empty_ranges_say_so():
    with pytest.raises(ValueError, match="must be a number"):
        pick(RUN, "12", "seventy", "")
    with pytest.raises(ValueError, match="no epochs from 200"):
        pick(RUN, "12", "200", "300")
