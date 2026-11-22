"""Cache-first HuggingFace loading — offline once warm.

`from_pretrained(repo_id)` phones the Hub for an etag check on EVERY call, even when the files
are fully cached — so a machine with flaky or no internet can't start training (or mid-run
auto-recaption) despite having every byte on disk (issue #26). Asking for the cache first makes
the network a first-run-only event: hit the local cache, and only fall back to the Hub when
something is genuinely missing. Also shaves the round-trip off every warm start.

`local_files_only=True` alone is not enough for a large-vocabulary tokenizer (every Qwen one): since
transformers 4.57 its loader asks the Hub whether the repo is a Mistral model (`model_info`) unless the
source is a local directory — so a cached Qwen tokenizer still failed offline (#174). The cached
snapshot's own folder is a local directory, so it is loaded from there first.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def cached_snapshot_dir(repo_id: str):
    """The local folder of `repo_id`'s cached snapshot (no network), or None when nothing is cached or
    `repo_id` is already a local path."""
    if os.path.isdir(repo_id):
        return None
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None
    return path if path and os.path.isdir(path) else None


def cached_file(repo_id: str, filename: str, **kwargs) -> str:
    """hf_hub_download(repo_id, filename) from the local cache, the Hub only when the file isn't cached (#397: a plain
    hf_hub_download asks the Hub on every call, so a cached file still needs the internet)."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True, **kwargs)
    except Exception:
        from fizgig.utils import hf_public
        return hf_public.download(repo_id=repo_id, filename=filename, **kwargs)


def single_file_cache_first(cls, path: str, config: str, **kwargs):
    """diffusers' `cls.from_single_file(path, config=config)` with the config repo's cached folder, the Hub only when it
    isn't cached: a config given by repo id is looked up on the Hub on every load (#397)."""
    local = cached_snapshot_dir(config)
    if local is not None:
        try:
            return cls.from_single_file(path, config=local, local_files_only=True, **kwargs)
        except Exception as e:
            logger.info(f"{cls.__name__}: the cached {config} config didn't load ({type(e).__name__}) - "
                        "asking the Hub")
    return cls.from_single_file(path, config=config, **kwargs)


def has_files(repo_id: str, patterns) -> bool:
    """Does the cached snapshot of `repo_id` hold a file for every allow-pattern? (No network.) A snapshot holding
    only some of a repo's files (one fetched on its own) isn't the tokenizer folder the patterns name."""
    import fnmatch
    local = cached_snapshot_dir(repo_id)
    if local is None:
        return False
    files = [os.path.relpath(os.path.join(r, f), local).replace(os.sep, "/")
             for r, _, fs in os.walk(local) for f in fs]
    return all(any(fnmatch.fnmatch(f, p) for f in files) for p in patterns)


def from_pretrained_cache_first(cls, repo_id: str, **kwargs):
    """`cls.from_pretrained(repo_id)`, trying the local cache before touching the network.

    Works for tokenizers, processors and configs alike; `repo_id` may also be a local
    directory (local_files_only is a no-op there). A partial or missing cache falls through
    to the normal network path, so first runs behave exactly as before.
    """
    local = cached_snapshot_dir(repo_id)
    if local is not None:
        try:
            return cls.from_pretrained(local, local_files_only=True, **kwargs)
        except Exception:
            pass                       # an incomplete snapshot: the repo-id paths below say what is missing
    try:
        return cls.from_pretrained(repo_id, local_files_only=True, **kwargs)
    except Exception:
        logger.info(f"{cls.__name__}: {repo_id} not in the local cache — fetching from the Hub "
                    "(one-time; later runs load offline)")
        return cls.from_pretrained(repo_id, **kwargs)
