"""Caching runs offline once a family's tokenizer files are downloaded (#397): Anima's and ACE-Step's tokenizers and
helper files load from the Hugging Face cache without touching the network (a plain from_pretrained / hf_hub_download
asks the Hub on every load), and the updater fetches a repo's tokenizer files when only some of them are cached."""
import json
import os
import socket

import pytest

from fizgig.utils import hf_cache

ANIMA = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ACE = "ACE-Step/Ace-Step1.5"


def _tiny_tokenizer(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel({"<unk>": 0, "</s>": 1, "<pad>": 2, "a": 3, "cat": 4}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>").save_pretrained(str(folder))


def _snapshot(cache, repo):
    """An empty cached snapshot of `repo` in the Hugging Face cache layout; returns its folder."""
    root = cache / ("models--" + repo.replace("/", "--"))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text("0123abcd")
    snap = root / "snapshots" / "0123abcd"
    snap.mkdir(parents=True, exist_ok=True)
    return snap


@pytest.fixture
def cache(tmp_path, monkeypatch):
    import huggingface_hub.constants as C
    c = tmp_path / "hub"
    c.mkdir()
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(c))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return c


@pytest.fixture
def no_network(monkeypatch):
    tried = []

    def refuse(*a, **k):
        tried.append(a[1:] or a)
        raise OSError("no network in this check")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return tried


def test_anima_tokenizers_load_offline(cache, no_network):
    snap = _snapshot(cache, ANIMA)
    _tiny_tokenizer(snap / "tokenizer")
    _tiny_tokenizer(snap / "t5_tokenizer")
    from fizgig.anima import driver
    tok, t5 = driver._tokenizers()
    assert tok("a cat").input_ids == [3, 4] and t5("a cat").input_ids[:2] == [3, 4]
    assert not no_network, "asked the Hub with every file cached"


def test_acestep_tokenizer_and_silence_file_load_offline(cache, no_network):
    snap = _snapshot(cache, ACE)
    _tiny_tokenizer(snap / "Qwen3-Embedding-0.6B")
    (snap / "acestep-v15-turbo").mkdir()
    (snap / "acestep-v15-turbo" / "silence_latent.pt").write_bytes(b"x")
    from fizgig.acestep import driver
    assert driver._tokenizer()("a cat").input_ids == [3, 4]
    assert os.path.samefile(driver._helper_file(driver.SILENCE), snap / "acestep-v15-turbo" / "silence_latent.pt")
    assert not no_network, "asked the Hub with every file cached"


def test_sdxl_reads_its_config_from_the_cache(cache, monkeypatch):
    # diffusers' from_single_file with config=<repo id> looks the repo up on the Hub on every load
    from fizgig.sdxl import driver
    snap = _snapshot(cache, driver.CONFIG_REPO)
    (snap / "model_index.json").write_text("{}")
    calls = []

    class Fake:
        @classmethod
        def from_single_file(cls, path, **kw):
            calls.append(kw)
            return "loaded"
    assert hf_cache.single_file_cache_first(Fake, "ckpt.safetensors", driver.CONFIG_REPO, subfolder="vae") == "loaded"
    assert calls == [{"config": str(snap), "local_files_only": True, "subfolder": "vae"}]
    src = open(driver.__file__, encoding="utf-8").read()
    assert "config=CONFIG_REPO" not in src and src.count("single_file_cache_first(") == 3


def test_a_partly_cached_repo_is_fetched_by_the_updater(cache, tmp_path, monkeypatch):
    from fizgig.families.registry import FAMILIES
    from fizgig.scripts import fetch_turbo_lora as U
    from fizgig.utils import hf_public
    d = next(f for f in FAMILIES.values() if ACE in dict(f.helper_files))
    pats = dict(d.helper_files)[ACE]
    snap = _snapshot(cache, ACE)
    (snap / "acestep-v15-turbo").mkdir()
    (snap / "acestep-v15-turbo" / "silence_latent.pt").write_bytes(b"x")      # fetched on its own: no tokenizer
    assert not hf_cache.has_files(ACE, pats)
    repo = tmp_path / "fizgig"
    (repo / "user_data").mkdir(parents=True)
    (repo / "user_data" / "prefs.json").write_text(json.dumps({d.model_files[0].pref_key: "x.safetensors"}))
    monkeypatch.delenv("FIZGIG_USER_DATA", raising=False)
    got = []
    monkeypatch.setattr(hf_public, "snapshot", lambda **kw: got.append(kw["repo_id"]))
    U.ensure_text_helpers(str(repo), log=lambda m: None)
    assert ACE in got
    # everything there: skipped without asking the Hub
    _tiny_tokenizer(snap / "Qwen3-Embedding-0.6B")
    (snap / "Qwen3-Embedding-0.6B" / "x.txt").write_text("")
    (snap / "Qwen3-Embedding-0.6B" / "chat_template.jinja").write_text("")
    assert hf_cache.has_files(ACE, pats)
    got.clear()
    U.ensure_text_helpers(str(repo), log=lambda m: None)
    assert ACE not in got
