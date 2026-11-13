"""#327: a download Start is waiting on says how it's going - in the run's log as each file starts and at every tenth,
and with its speed and time left in the status the page shows (Training's box under Start, Preferences). Driven
through _fetch_watch with a stand-in download, so the log written from it can't deadlock on the server's lock."""
import threading

from fizgig.web import server as S


class _Proc:
    def __init__(self, lines):
        self.stdout = iter(lines)

    def wait(self):
        return 1                                  # "failed": nothing to start afterwards


def test_progress_logs_and_reports_speed(monkeypatch):
    logged = []
    monkeypatch.setattr(S, "append_log", lambda t: logged.append(t))
    clock = iter([100.0, 101.0, 104.0, 108.0, 112.0, 120.0])
    monkeypatch.setattr(S.time, "time", lambda: next(clock))
    gb = 1024 ** 3
    lines = [f"[progress] {d} {10 * gb} big.safetensors" for d in (0, gb // 2, 2 * gb, 4 * gb, 6 * gb)] + \
            ["  [done] big.safetensors"]
    S._fetch.update(state="running", file="", done=0, total=0, lines=[], then_start=("fam", {}, ""),
                    rate=0.0, eta=None, said=-1)
    t = threading.Thread(target=S._fetch_watch, args=(_Proc(lines),), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "the watcher hung (the lock taken twice?)"
    assert logged[0].startswith("Downloading big.safetensors (10.0 GB)")
    tenths = [x for x in logged if "% " in x or "%)" in x]
    assert any("(20%)" in x for x in tenths) and any("(40%)" in x for x in tenths) and any("(60%)" in x for x in tenths)
    assert any("MB/s" in x and "left" in x for x in tenths)
    assert len(set(tenths)) == len(tenths)                                  # each tenth said once
    st = S._fetch_status()
    assert st["rate"] > 0 and st["eta"] is not None and st["done"] == 6 * gb


def test_no_log_lines_from_a_preferences_download(monkeypatch):
    logged = []
    monkeypatch.setattr(S, "append_log", lambda t: logged.append(t))
    S._fetch.update(state="running", file="", done=0, total=0, lines=[], then_start=None, rate=0.0, eta=None, said=-1)
    S._fetch_watch(_Proc([f"[progress] {d} 1000 a.safetensors" for d in (0, 500, 900)]))
    assert logged == []                                                     # the Preferences card shows it instead
