"""#406: when the app window gives way to the browser, Launch Fizgig.bat (pythonw) leaves no window and no console, so
closing the browser must close Fizgig - it ran on until Task Manager. The server says how long since a page last asked
for anything; the launcher stops once that's 3 minutes and no run is training."""
import http.server
import json
import threading
import urllib.request

import pytest

from fizgig.web import app_window as aw


def test_stop_rules():
    done = aw.browser_done
    assert not done(None, 999)                                        # no answer: ask again
    assert not done({"training": False, "page_age": 1.0}, 600)        # a page is open
    assert not done({"training": False, "page_age": 70.0}, 600)       # a hidden tab, its timers slowed
    assert done({"training": False, "page_age": 200.0}, 600)          # the browser was closed
    assert not done({"training": True, "page_age": 900.0}, 1200)      # a run finishes first
    assert not done({"training": False, "page_age": 200.0}, 30)       # just opened: give the browser time
    assert not done({"training": False, "page_age": None}, 120)       # the page hasn't come yet
    assert done({"training": False, "page_age": None}, 300)           # and never did


def test_serve_for_browser_stops_after_settled_answers(monkeypatch):
    answers = iter([{"training": True, "page_age": 190.0}] * 2 + [{"training": False, "page_age": 190.0},
                   {"training": True, "page_age": 192.0}] + [{"training": False, "page_age": 195.0}] * 3)
    asked = []

    class Proc:
        def poll(self):
            return None
    monkeypatch.setattr(aw, "_ask_busy", lambda port, key: asked.append(1) or next(answers))
    monkeypatch.setattr(aw, "_log", lambda msg: None)
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(aw.time, "time", lambda: next(clock))
    monkeypatch.setattr(aw.time, "sleep", lambda s: None)
    aw._serve_for_browser(Proc(), 1, "k")
    assert len(asked) == 7            # one idle answer between runs (the queue's next starting) doesn't stop it


def test_server_reports_page_age(monkeypatch):
    server = pytest.importorskip("fizgig.web.server")
    monkeypatch.setenv("FIZGIG_APP_KEY", "the-key")
    monkeypatch.setattr(server, "_page_seen", [0.0])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def busy():
        req = urllib.request.Request(base + "/api/app/busy", data=b"{}", method="POST",
                                     headers={"X-Fizgig-App-Key": "the-key", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    try:
        assert busy()["page_age"] is None                 # no page yet (the window's own calls don't count)
        try:
            urllib.request.urlopen(base + "/api/run", timeout=10).read()
        except urllib.error.HTTPError:
            pass                                          # no token: refused, but a page is there
        age = busy()["page_age"]
        assert age is not None and age < 5
    finally:
        srv.shutdown()
