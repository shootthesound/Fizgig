"""PR #382 fix-up: the app window's server output log (user_data/web_stdout.log) can't grow without end: past
SERVER_LOG_MAX it moves to web_stdout.log.1 and a new one begins."""
import os
from unittest.mock import patch

from fizgig.web import app_window


def test_rotates_past_the_cap(tmp_path):
    log = tmp_path / "web_stdout.log"
    with patch.object(app_window, "user_file", lambda name: str(tmp_path / name)), \
            patch.object(app_window, "SERVER_LOG_MAX", 10):
        log.write_bytes(b"small")
        assert app_window.server_log() == str(log) and log.read_bytes() == b"small"
        log.write_bytes(b"x" * 11)
        assert app_window.server_log() == str(log)
        assert not log.exists() and (tmp_path / "web_stdout.log.1").read_bytes() == b"x" * 11
