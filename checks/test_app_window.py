"""The app window's remembered size and place: closing it while minimized on Windows must not save the off-screen
position Windows reports then (-32000, scaled by the display), or the next start opens off screen (#205)."""


def test_app_window_ignores_minimized_coordinates(tmp_path, monkeypatch):
    from fizgig.web import app_window as aw
    monkeypatch.setattr(aw, "PREFS_FILE", str(tmp_path / "prefs.json"))
    aw._remember({"width": 1440, "height": 920, "x": 100, "y": 80})
    assert aw._sane_geom(aw._prefs().get("app_window")) == {"width": 1440, "height": 920, "x": 100, "y": 80}
    aw._remember({"width": 380, "height": 600, "x": -21332, "y": -21332})   # closed while minimized on Windows
    assert aw._prefs().get("app_window") == {"width": 1440, "height": 920, "x": 100, "y": 80}
    assert aw._sane_geom({"width": 380, "height": 600, "x": -21332, "y": -21332}) == {}


def test_a_bad_position_already_saved_is_ignored_at_start():
    from fizgig.web import app_window as aw
    assert aw._sane_geom(None) == {}
    assert aw._sane_geom({"width": 1440, "height": 920, "x": -32000, "y": -32000}) == {}
    assert aw._sane_geom({"width": 1440, "height": 920, "x": -1800, "y": 40}) == {"width": 1440, "height": 920,
                                                                                  "x": -1800, "y": 40}   # a left monitor


def test_webview2_failing_to_start_is_noticed_at_once():
    """#406: pywebview only logs WebView2 failing to start and leaves a blank window; the window listens on its log
    and gives way to the browser at once, saying why (it waited ~50 s, then blamed drawing or security software)."""
    import logging
    from fizgig.web import app_window as aw
    e = aw.EngineFailed()
    log = logging.getLogger("pywebview")
    log.addHandler(e.handler)
    try:
        log.warning("something else")
        log.error("an unrelated error")
        assert not e.event.is_set()
        log.error("WebView2 initialization failed with exception:\n\n Couldn't find a compatible Webview2 Runtime "
                  "installation to host WebViews. ---> System.IO.FileNotFoundException: The system cannot find the "
                  "file specified.\n   at Microsoft.Web.WebView2.Core.CoreWebView2Environment.MoveNext()")
        assert e.event.is_set()
        assert e.why == "Couldn't find a compatible Webview2 Runtime installation to host WebViews."
    finally:
        log.removeHandler(e.handler)
