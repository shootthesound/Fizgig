"""Compact text caches: only the real (non-padding) tokens are stored, the padded tensors rebuilt as they load.

Krea 2's text encoder pads every caption to 512 tokens (~31.5 MB a caption) and uses ~80; its trainer drops the
padding before every step (description.text_cache_trim), so the padding rows are never read. A compact file keeps,
for each `<base>attention_mask`, its tensors' masked rows only (the mask itself stays, 512 bytes), and its metadata
records which tensors were compacted and their full length. load() puts each one back at full length: the real tokens
at their own positions, zeros where the padding was - the same mask, the same real tokens, so the same training step
(checks/test_compact_text_cache.py). A file without the marker (every cache written before) loads as it is.

FIZGIG_COMPACT_TEXT=0 writes full files again.
"""
from __future__ import annotations

import json
import os


MARK = "fizgig_compact_text"        # metadata: {"<key>": [full length, "<its mask's key>"], ...}
MASK = "attention_mask"


def enabled() -> bool:
    return os.environ.get("FIZGIG_COMPACT_TEXT", "1") != "0"


def compact(sd: dict, metadata: dict | None = None):
    """(tensors, metadata) with every masked tensor cut to its real rows. A dict with no masks comes back as it is."""
    out, done = dict(sd), {}
    for mk, m in sd.items():
        if not mk.endswith(MASK) or m.dim() != 1:
            continue
        base, m = mk[:-len(MASK)], m.bool()
        if bool(m.all()):
            continue
        for k, v in sd.items():
            if (k != mk and k.startswith(base) and not k.endswith(MASK) and k not in done and v.dim() >= 1
                    and v.shape[0] == m.shape[0]):
                out[k] = v[m].contiguous()
                done[k] = [int(m.shape[0]), mk]
    md = dict(metadata or {})
    if done:
        md[MARK] = json.dumps(done)
    return out, md


def expand(sd: dict, metadata: dict | None) -> dict:
    """A compact file's tensors back at full length (a full file's as they are)."""
    raw = (metadata or {}).get(MARK)
    if not raw:
        return sd
    out = dict(sd)
    for k, (n, mk) in json.loads(raw).items():
        if k not in sd or mk not in sd:
            continue
        v, m = sd[k], sd[mk].bool()
        full = v.new_zeros((int(n), *v.shape[1:]))
        full[m] = v
        out[k] = full
    return out


def load(path: str, keys=None) -> dict:
    """A text cache file's tensors (all, or those starting with one of `keys`' prefixes), compact or not, at full
    length."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
        names = list(f.keys())
        if keys is not None:
            want = [n for n in names if any(n.startswith(p) for p in keys)]
        else:
            want = names
        sd = {n: f.get_tensor(n) for n in want}
    return expand(sd, md)


def save(sd: dict, path: str, metadata: dict | None = None, compact_it: bool = True) -> None:
    from safetensors.torch import save_file
    if compact_it and enabled():
        sd, metadata = compact(sd, metadata)
    save_file(sd, path, metadata=metadata)


def is_compact(path: str) -> bool:
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            return MARK in (f.metadata() or {})
    except Exception:
        return False


__all__ = ["MARK", "compact", "enabled", "expand", "is_compact", "load", "save"]
