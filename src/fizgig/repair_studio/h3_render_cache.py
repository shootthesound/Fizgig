"""Exact render cache for the MiniMax H3 Repair Studio — cache OUTCOMES, serve them exactly.

For one render setup (LoRA, donor, prompt, seed, length, canvas, regime, conditioning frames)
every rendered clip latent is stored under the SIGNATURE of the slider state that produced it.
A state that was rendered before comes back without touching the DiT (decode only); a
background builder pre-renders every block switched off — the "library" — so "what does block
N do" is answered instantly for all 52 blocks after a one-off build.

Nothing here approximates. An earlier version blended these entries in latent space to fake
previews; measured on the real 33B (3 Sep 2026) a blend predicted the exact render no better
than the untouched baseline, so it was removed for good. An entry is served only for exactly
the state that rendered it.

Storage: <cache_dir>/repair_cache/<setup_key>/ with <sig>.safetensors (latent fp16 + audio rows
fp32), <sig>.jpg (middle frame, for the block ticks and the history strip) and a manifest.json
written LAST — the trainer's gallery settle-guard discipline — so a half-written entry is never
trusted. Entries survive restarts; a different setup is a different key; old setups stay on
disk until "Clear cache".
"""

import hashlib
import json
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch

CACHE_FORMAT = 2
MANIFEST = "manifest.json"
BASE_SIG = "base"


def setup_key(*, primary_hash: str, donor_hash: str, prompt: str, seed: int, frames: int,
              width: int, height: int, steps: int, turbo_strength, keyframe_sig) -> str:
    """sha256 over everything a render depends on (plus the format version), 20 hex chars —
    the cache directory name."""
    payload = json.dumps({
        "fmt": CACHE_FORMAT, "primary": primary_hash or "", "donor": donor_hash or "",
        "prompt": prompt, "seed": int(seed), "frames": int(frames), "w": int(width),
        "h": int(height), "steps": int(steps), "turbo": turbo_strength,
        "kf": [list(map(str, k)) for k in (keyframe_sig or ())],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _row_tuple(bs):
    return (bool(bs.primary_enabled), round(float(bs.primary_strength), 4),
            bool(bs.donor_enabled), round(float(bs.donor_strength), 4))


_DEFAULT_ROW = (True, 1.0, True, 0.0)


def block_off_sig(block_id: str) -> str:
    return f"off:{block_id}"


def signature(state) -> str:
    """The slider state's cache signature. "base" when every row is at its default (primary
    1.0 on, donor 0.0); "off:<bid>" when exactly one block is at 0 with everything else
    default (a disabled block counts as 0 — same render); otherwise a hash of the non-default
    rows. Rows the LoRA doesn't touch still count: the signature describes the STATE, the
    setup key describes the LoRA."""
    moved = []
    for bid, bs in sorted(state.blocks.items()):
        row = _row_tuple(bs)
        if row != _DEFAULT_ROW:
            moved.append((bid, row))
    if not moved:
        return BASE_SIG
    if len(moved) == 1:
        bid, (p_on, p_str, d_on, d_str) = moved[0]
        eff = p_str if p_on else 0.0
        if abs(eff) < 1e-9 and (not d_on or abs(d_str) < 1e-9):
            return block_off_sig(bid)
    payload = json.dumps(moved, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_name(sig: str) -> str:
    return sig.replace(":", "__")


class RenderCache:
    """One render setup's cached clips, on disk and (lazily) in RAM.

    Thread-safe for one writer (the builder thread or the preview worker) and readers: an
    entry is written to a temp file and renamed, the manifest is rewritten after each put,
    and the in-RAM dict is only ever added to."""

    def __init__(self, root_dir: str, key: str, block_ids: Iterable[str]):
        self.root_dir = root_dir
        self.key = key
        self.dir = os.path.join(root_dir, key)
        self.block_ids: List[str] = list(block_ids)      # blocks the LoRA touches
        self._lock = threading.RLock()
        self._ram: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._index: Dict[str, dict] = {}               # sig -> meta (what the manifest vouches)
        self.meta: dict = {}
        self._read_manifest()

    # ----- disk -------------------------------------------------------------------------
    def _read_manifest(self):
        p = os.path.join(self.dir, MANIFEST)
        if not os.path.isfile(p):
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        if d.get("format") != CACHE_FORMAT:
            return
        self.meta = d.get("meta", {})
        self._index = {s: m for s, m in d.get("entries", {}).items()
                       if os.path.isfile(self._path(s))}

    def _write_manifest(self):
        os.makedirs(self.dir, exist_ok=True)
        d = {"format": CACHE_FORMAT, "key": self.key, "meta": self.meta, "entries": self._index}
        tmp = os.path.join(self.dir, MANIFEST + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, MANIFEST))

    def _path(self, sig: str) -> str:
        return os.path.join(self.dir, _safe_name(sig) + ".safetensors")

    def thumb_path(self, sig: str) -> str:
        return os.path.join(self.dir, _safe_name(sig) + ".jpg")

    def _load(self, sig: str):
        from safetensors.torch import load_file
        t = load_file(self._path(sig), device="cpu")
        return t["latent"].float(), (t["audio"].float() if "audio" in t else None)

    # ----- writes ---------------------------------------------------------------------
    def set_meta(self, **meta):
        self.meta.update(meta)

    def put(self, sig: str, latent, audio, middle=None, **meta):
        """Store a rendered clip under its signature. `middle` (PIL) becomes the jpg thumb."""
        from safetensors.torch import save_file
        with self._lock:
            os.makedirs(self.dir, exist_ok=True)
            lat = latent.detach().float().cpu()
            aud = audio.detach().float().cpu() if audio is not None else None
            tensors = {"latent": lat.to(torch.float16).contiguous()}
            if aud is not None:
                tensors["audio"] = aud.contiguous()
            tmp = self._path(sig) + ".tmp"
            save_file(tensors, tmp, metadata={"sig": sig})
            os.replace(tmp, self._path(sig))
            if middle is not None:
                try:
                    im = middle.convert("RGB")
                    im.thumbnail((256, 256))
                    im.save(self.thumb_path(sig), quality=88)
                except Exception:
                    pass
            self._ram[sig] = (lat, aud)
            self._index[sig] = {"when": time.time(), **meta}
            self._write_manifest()

    # ----- reads --------------------------------------------------------------------------
    def has(self, sig: str) -> bool:
        return sig in self._ram or sig in self._index

    def get(self, sig: str):
        """(latent fp32, audio rows or None) or None when not cached."""
        with self._lock:
            e = self._ram.get(sig)
            if e is None and sig in self._index:
                try:
                    e = self._load(sig)
                except Exception:
                    self._index.pop(sig, None)
                    return None
                self._ram[sig] = e
            return e

    def info(self, sig: str) -> dict:
        return dict(self._index.get(sig, {}))

    def entries(self) -> Dict[str, dict]:
        """sig -> meta for everything the manifest vouches for (newest last)."""
        return dict(sorted(self._index.items(), key=lambda kv: kv[1].get("when", 0)))

    def has_block_off(self, block_id: str) -> bool:
        return self.has(block_off_sig(block_id))

    def done_ids(self) -> List[str]:
        return [b for b in self.block_ids if self.has_block_off(b)]

    def missing(self) -> List[str]:
        return [b for b in self.block_ids if not self.has_block_off(b)]

    def complete(self) -> bool:
        return self.has(BASE_SIG) and not self.missing()

    def n_entries(self) -> int:
        return len(self._index)

    def size_bytes(self) -> int:
        return dir_size(self.dir)


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clear_render_cache(root_dir: str) -> Tuple[int, int]:
    """Delete every cached setup under root_dir. Returns (setups removed, bytes freed)."""
    import shutil
    if not os.path.isdir(root_dir):
        return 0, 0
    n, freed = 0, 0
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, MANIFEST)):
            freed += dir_size(p)
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n, freed


def build_order(block_ids: Iterable[str]) -> List[str]:
    """The order the library builder renders block-off entries: main blocks ASCENDING (each
    entry then differs from the previous one in two neighbouring blocks, so the exact pass-1
    resume skips everything before them), token-refiner blocks last."""
    main = sorted((b for b in block_ids if b.startswith("h3blk_")),
                  key=lambda b: int(b.split("_")[1]))
    refiners = sorted(b for b in block_ids if b.startswith("h3_rf_"))
    return main + refiners
