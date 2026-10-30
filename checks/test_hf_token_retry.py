"""Public model downloads with an expired Hugging Face login: one anonymous retry, then a short friendly error."""
import os
import time

import pytest

# The CI checks install only checks/requirements.txt, so skip when the Hub client isn't there.
requests = pytest.importorskip("requests")
pytest.importorskip("huggingface_hub")
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

from fizgig.utils import hf_public

APP_JS = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _resp(code):
    r = requests.Response()
    r.status_code = code
    r.url = "https://huggingface.co/some/public-repo/resolve/x/model.onnx"
    return r


def expired():
    return RepositoryNotFoundError(
        "401 Client Error. Repository Not Found for url: https://huggingface.co/some/public-repo/resolve/x/model.onnx."
        ' OAuth token has expired: "exp" claim timestamp check failed', response=_resp(401))


class FakeHub:
    """hf_hub_download stand-in: refuses any call that may carry the saved login, writes the file otherwise."""

    def __init__(self, anon_error=None):
        self.calls = []
        self.anon_error = anon_error

    def __call__(self, repo_id, filename, local_dir=None, token=None, **kw):
        self.calls.append(token)
        if token is not False:
            raise expired()
        if self.anon_error is not None:
            raise self.anon_error
        d = local_dir or "."
        p = os.path.join(d, filename)
        os.makedirs(os.path.dirname(p) or d, exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_oauth_saved")
    monkeypatch.setattr(hf_public, "_noted", False)
    return fake


def test_expired_login_retries_once_without_it(hub, tmp_path):
    notes = []
    got = hf_public.download(repo_id="some/public-repo", filename="model.onnx", local_dir=str(tmp_path),
                             log=notes.append)
    assert os.path.isfile(got)
    assert hub.calls == [None, False]
    assert notes == [hf_public.NOTE]
    hf_public.download(repo_id="some/public-repo", filename="b.onnx", local_dir=str(tmp_path), log=notes.append)
    assert notes == [hf_public.NOTE]                      # said once, not per file


def test_note_is_short():
    assert len(hf_public.NOTE) < 140 and "hf auth login" in hf_public.NOTE
    assert len(hf_public.EXPIRED) < 140 and "hf auth login" in hf_public.EXPIRED


def test_explicit_token_retries_too(hub, tmp_path):
    hf_public.download(repo_id="r/x", filename="a.bin", local_dir=str(tmp_path), token="hf_old", log=lambda m: None)
    assert hub.calls == ["hf_old", False]


def test_no_saved_login_no_retry(hub, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(RepositoryNotFoundError):
        hf_public.download(repo_id="r/x", filename="a.bin")
    assert hub.calls == [None]


def test_token_false_no_retry(hub):
    hub.anon_error = expired()
    with pytest.raises(RepositoryNotFoundError):
        hf_public.download(repo_id="r/x", filename="a.bin", token=False)
    assert hub.calls == [False]


def test_other_errors_no_retry(monkeypatch):
    import huggingface_hub
    calls = []

    def boom(**kw):
        calls.append(kw.get("token"))
        raise EntryNotFoundError("404 Client Error. Entry Not Found", response=_resp(404))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_saved")
    with pytest.raises(EntryNotFoundError):
        hf_public.download(repo_id="r/x", filename="a.bin")
    assert calls == [None]


def test_gated_repo_still_needs_login(hub):
    hub.anon_error = GatedRepoError("401 Client Error. Cannot access gated repo", response=_resp(401))
    with pytest.raises(hf_public.LoginExpired) as ei:
        hf_public.download(repo_id="Comfy-Org/sam3.1", filename="a.safetensors")
    assert hub.calls == [None, False]                     # one retry only
    assert hf_public.friendly_error(ei.value) == hf_public.EXPIRED


def test_network_error_on_retry_is_kept(hub):
    hub.anon_error = requests.ConnectionError("no internet")
    with pytest.raises(requests.ConnectionError):
        hf_public.download(repo_id="r/x", filename="a.bin")


def test_friendly_error():
    assert hf_public.friendly_error(expired()) == hf_public.EXPIRED
    assert hf_public.friendly_error(ValueError("bad")) == "ValueError: bad"
    # an anonymous 401 (a repo that isn't there) isn't called a login problem
    assert hf_public.friendly_error(RepositoryNotFoundError("401 Client Error. Repository Not Found",
                                                            response=_resp(401))).startswith("RepositoryNotFound")


def test_snapshot_retries(monkeypatch, tmp_path):
    import huggingface_hub
    calls = []

    def snap(repo_id, token=None, **kw):
        calls.append(token)
        if token is not False:
            raise expired()
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_saved")
    monkeypatch.setattr(hf_public, "_noted", False)
    assert hf_public.snapshot(repo_id="r/x", repo_type="dataset", log=lambda m: None) == str(tmp_path)
    assert calls == [None, False]


def _wait(mod, models_dir):
    for _ in range(200):
        st = mod.fetch_state(models_dir)
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(0.02)
    raise AssertionError("download never finished")


@pytest.mark.parametrize("name", ["sammask"])
def test_download_cards_recover(hub, tmp_path, monkeypatch, name):
    import importlib
    mod = importlib.import_module(f"fizgig.dataset.{name}")
    monkeypatch.setattr(mod, "_fetch", {"state": "", "error": "", "at": 0.0})
    mod.download(str(tmp_path))
    st = _wait(mod, str(tmp_path))
    assert st["state"] == "done", st
    assert False in hub.calls


@pytest.mark.parametrize("name", ["sammask"])
def test_download_cards_short_error(hub, tmp_path, monkeypatch, name):
    import importlib
    mod = importlib.import_module(f"fizgig.dataset.{name}")
    monkeypatch.setattr(mod, "_fetch", {"state": "", "error": "", "at": 0.0})
    hub.anon_error = GatedRepoError("401 Client Error. Cannot access gated repo", response=_resp(401))
    mod.download(str(tmp_path))
    st = _wait(mod, str(tmp_path))
    assert st["state"] == "failed"
    assert st["error"] == hf_public.EXPIRED
    assert "RepositoryNotFound" not in st["error"] and "exp" not in st["error"].split()


def test_fetch_weight_says_login(hub, tmp_path, monkeypatch):
    from fizgig.scripts import fetch_models
    hub.anon_error = GatedRepoError("401 Client Error. Cannot access gated repo", response=_resp(401))
    monkeypatch.setattr(fetch_models, "EMIT_PROGRESS", False)

    class W:
        repo, path_in_repo, filename, gb, note, pref_key = "a/b", "m.safetensors", "m.safetensors", 0.001, "", "k"
        switch_from, replaces, gated = "", (), False
    lines = []
    assert fetch_models.fetch_weight(W, str(tmp_path), {}, log=lines.append) is False
    assert any(l.startswith("  [login]") and "hf auth login" in l for l in lines), lines


def test_ui_hint_reads_the_login_line():
    js = open(APP_JS, encoding="utf-8").read()
    assert "function downloadFailedText(err)" in js
    assert "/^Your Hugging Face login/" in js
    assert hf_public.EXPIRED.startswith("Your Hugging Face login")
    assert js.count("downloadFailedText(f.error)") == 1
    assert "` The download failed: ${f.error}`" not in js
