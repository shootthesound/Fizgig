"""The cache stages' count reaches the run card (#249): CacheProgress prints "[cache] captions 40/120" lines, at most
one a second and always the last, and the run's tracker reads them as a cache update, not training steps."""
from fizgig.training.progress import CacheProgress, TrainingProgressTracker, dataset_items


def test_cache_line_is_a_cache_update():
    t = TrainingProgressTracker(10)
    assert t.consume("[cache] captions 40/120\n") == {"kind": "cache", "what": "captions", "n": 40, "total": 120}
    assert t.consume("[cache] pictures 3/3") == {"kind": "cache", "what": "pictures", "n": 3, "total": 3}
    # launch's own note has no count: not a cache update
    assert t.consume("[cache] 4 picture(s) have no cache - caching them before the run resumes.") is None


def test_counts_throttled_last_always(capsys):
    p = CacheProgress("captions", 20)
    for _ in range(5):
        p.add(4)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[cache] captions 4/20"          # the first at once
    assert lines[-1] == "[cache] captions 20/20"         # and the last, however soon after
    assert len(lines) == 2


def test_no_total_no_lines(capsys):
    p = CacheProgress("pictures", 0)
    p.add(8)
    assert capsys.readouterr().out == ""


def test_dataset_items():
    class DS:
        datasource = [1, 2, 3]
    assert dataset_items(DS()) == 3
    assert dataset_items(object()) == 0
