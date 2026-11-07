"""taeh3: madebyollin's tiny autoencoder for MiniMax H3 latents (22.7 MB, MIT - fizgig/minimax/taehv.py), used for
H3's quick previews (Studio's Show early) instead of the 4.5 GB video decoder. The final clip still decodes through
the full VAE.

Measured on an RTX 5090, a real H3 clip's middle frame: 0.14 s against the full decoder's 1.55 s, PSNR 30.2 dB to it
(7 Nov 2026) - a little softer, fine text garbled, composition / colour / faces all there.

The file is fetched with the H3 models (fetch_models, fetch_turbo_lora on update) and, for an install that doesn't
have it yet, in the background the first time a preview wants it; until it's there (or with no connection) the
previews use the full decoder as before. Pinned to one upstream commit and checked by hash, so only this exact file
is ever loaded."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

NAME = "taeh3.safetensors"
URL = ("https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/"
       "safetensors/taeh3.safetensors")
SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
SIZE = 22709752

_lock = threading.Lock()
_fetching = [False]


def models_dir() -> str:
    from fizgig.user_data import MODELS_DIR
    return MODELS_DIR


def path(directory: Optional[str] = None) -> Optional[str]:
    """The downloaded file, or None before it's there."""
    p = os.path.join(directory or models_dir(), NAME)
    return p if os.path.isfile(p) and os.path.getsize(p) == SIZE else None


def fetch(directory: Optional[str] = None, timeout: float = 120.0, log=logger.info) -> Optional[str]:
    """Download it (if missing) into the models folder; its path, or None when it couldn't (no connection, a slow or
    broken download, a hash that doesn't match) - never raises."""
    directory = directory or models_dir()
    have = path(directory)
    if have:
        return have
    dest = os.path.join(directory, NAME)
    part = dest + ".part"
    try:
        os.makedirs(directory, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Fizgig"})
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != SHA256:
            raise ValueError("the downloaded file doesn't match its checksum")
        os.replace(part, dest)
        log(f"[taeh3] downloaded the H3 quick-preview decoder ({SIZE / 1e6:.1f} MB) to {dest}")
        return dest
    except Exception as e:
        try:
            os.remove(part)
        except OSError:
            pass
        log(f"[taeh3] couldn't download the H3 quick-preview decoder ({type(e).__name__}: {e}) - previews use the "
            f"full decoder")
        return None


def fetch_in_background(directory: Optional[str] = None) -> None:
    """Start one background download (an install that doesn't have it yet); the caller carries on meanwhile."""
    with _lock:
        if _fetching[0] or path(directory):
            return
        _fetching[0] = True

    def work():
        try:
            fetch(directory)
        finally:
            _fetching[0] = False
    threading.Thread(target=work, daemon=True, name="taeh3-fetch").start()


def load(device, directory: Optional[str] = None):
    """The decoder (fp16, on `device`), or None when the file isn't there yet (a background download starts)."""
    p = path(directory)
    if p is None:
        fetch_in_background(directory)
        return None
    import torch
    from fizgig.minimax.taehv import TAEHV
    return TAEHV(p, arch_name="taeh3").to(device=device, dtype=torch.float16).eval()


def middle_frame(tae, latent):
    """The middle frame (a PIL image) of an H3 latent [B, C, T, H, W] in H3's normalised space. Decoded one latent
    frame at a time (parallel=False), so its memory stays small at any canvas or length."""
    import torch
    from PIL import Image
    with torch.no_grad():
        z = latent.to(device=next(tae.parameters()).device, dtype=torch.float16)
        v = tae.decode_video(z.permute(0, 2, 1, 3, 4), parallel=False, show_progress_bar=False)   # [B, F, 3, H, W]
    px = v[0, v.shape[1] // 2].float().clamp(0, 1)
    return Image.fromarray((px.permute(1, 2, 0) * 255).byte().cpu().numpy())
