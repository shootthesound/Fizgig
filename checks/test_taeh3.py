"""MiniMax H3's quick-preview decoder (taeh3): missing until downloaded, a failed download (no connection) returns None
without raising, and load() then hands back None so Show early falls back to the full decoder."""
from fizgig.minimax import taeh3


def test_missing_file_is_none(tmp_path):
    assert taeh3.path(str(tmp_path)) is None


def test_a_file_of_the_wrong_size_doesnt_count(tmp_path):
    (tmp_path / taeh3.NAME).write_bytes(b"not the weights")
    assert taeh3.path(str(tmp_path)) is None


def test_failed_download_returns_none_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(taeh3, "URL", "http://127.0.0.1:9/taeh3.safetensors")   # nothing listens there
    msgs = []
    assert taeh3.fetch(str(tmp_path), timeout=5, log=msgs.append) is None
    assert not list(tmp_path.iterdir())                                       # no .part left behind
    assert any("full decoder" in m for m in msgs)


def test_load_without_the_file_is_none_and_starts_a_fetch(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(taeh3, "fetch_in_background", lambda d=None: started.append(d))
    assert taeh3.load("cpu", str(tmp_path)) is None
    assert started == [str(tmp_path)]
