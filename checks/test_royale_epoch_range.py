"""LoRA Royale's epoch-range slider: the knobs set a window into the scanned epochs, and Max renders spreads its
picks inside it. Without a window (the knobs at the ends, or no slider at all) it is the whole run, exactly as before.
"""
from fizgig.web.royale import pick


def items(n):
    return [(i, f"/run/mylora-{i:06d}.safetensors") for i in range(1, n + 1)]


def test_no_window_is_untouched():
    it = items(30)
    assert pick(it, "12") == pick(it, "12", None, None)
    assert pick(it, "all") == it
    assert pick(it, "all", 0, 29) == it                       # the knobs fully extended: every epoch
    assert pick(it, "12", 0, 29) == pick(it, "12")


def test_window_scopes_the_spread():
    it = items(30)
    got = pick(it, "12", 3, 17)                               # epochs 4..18
    labels = [l for l, _ in got]
    assert len(labels) == 12
    assert labels[0] == 4 and labels[-1] == 18                # the window's own ends always in
    assert all(4 <= l <= 18 for l in labels)
    assert labels == sorted(labels)


def test_a_short_window_renders_all_of_it():
    it = items(30)
    assert pick(it, "20", 4, 4) == [it[4]]                     # one epoch under the knobs
    assert pick(it, "6", 10, 12) == it[10:13]
    assert pick(it, "all", 10, 12) == it[10:13]


def test_a_window_beyond_the_run_is_clamped():
    it = items(6)
    assert pick(it, "all", 0, 99) == it
    assert pick(it, "all", -5, 3) == it[:4]
    assert pick(it, "2", 0, 3) == [it[0], it[3]]                # the ends of epochs 1..4


def test_crossed_knobs_are_taken_the_other_way():
    it = items(10)
    assert pick(it, "all", 6, 2) == pick(it, "all", 2, 6) == it[2:7]


def test_indices_arrive_as_text():
    it = items(30)
    assert pick(it, "12", "3", "17") == pick(it, "12", 3, 17)
    assert pick(it, "all", "0", "9") == it[:10]


def test_an_unclassified_folder_windows_by_position():
    loras = [(f"style-{i}", f"/x/style{i}.safetensors") for i in range(1, 11)]
    assert pick(loras, "all", 2, 5) == loras[2:6]
    assert pick(loras, "3", 0, 2) == loras[0:3]
    assert [l for l, _ in pick(loras, "4", 0, 8)] == ["style-1", "style-4", "style-6", "style-9"]
