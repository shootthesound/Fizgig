"""Large convolutions in bands of output rows on AMD (ROCm): the VAEs' decode and encode no longer take a work buffer
the size of the whole picture.

On ROCm the convolutions run as GEMMs with a large workspace sized to the whole output, so a VAE at preview size asks
for far more than the card has. #404: Anima's decode reserved 29.8 GB on a 16 GB RX 7800 XT at 1024x1536, and Windows
spilled it to shared memory until the PC froze. Measured on NVIDIA with cuDNN off (the nearest stand-in), the same holds
for every still family's VAE: 16-17 GB at 1024x1024 and 23-31 GB at 1024x1536 to decode (Klein, Flux 3, Fibo 2,
MOSS-Image, SDXL, Z-Image), 11-18 GB to encode for the cache.

A convolution's output row reads only the input rows under its kernel, so splitting the output into bands of rows and
giving each band exactly the input rows it needs (zero rows where the padding would be) computes the same values - not
tiles blended at their seams, the same convolution - with the work buffer bounded by the band. Only convolutions whose
im2col-sized buffer would pass BAND_BYTES are split, so the DiTs' small convolutions and everything else run as before.
ROCm only. Preferences > Speed-ups for your graphics card > "Split large convolutions" (on by default) turns it off,
and FIZGIG_ROCM_BANDS=0/1 wins over it. NVIDIA never installs it.
"""
import os

import torch
import torch.nn.functional as F

BAND_BYTES = 1 << 30            # a convolution's im2col-sized buffer past this is computed in bands of output rows
_orig = {}


def _pair(v, n):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n


def _banded(fn, nd):
    """F.conv2d / F.conv3d, split along the height (dim -2) when large. Same arguments, same result."""
    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dim() != nd + 2 or isinstance(padding, str) or any(d != 1 for d in _pair(dilation, nd)):
            return fn(input, weight, bias, stride, padding, dilation, groups)
        st, pd = _pair(stride, nd), _pair(padding, nd)
        k, sh, ph = weight.shape[-2], st[-2], pd[-2]
        h = input.shape[-2]
        h_out = (h + 2 * ph - k) // sh + 1
        per_row = input.shape[0] * input.shape[1] * weight[0, 0].numel() * input.element_size()
        for i in range(nd):
            if i != nd - 2:             # every output dim but the height
                per_row *= (input.shape[2 + i] + 2 * pd[i] - weight.shape[2 + i]) // st[i] + 1
        if h_out <= 1 or per_row * h_out <= BAND_BYTES:
            return fn(input, weight, bias, stride, padding, dilation, groups)
        rows = max(1, BAND_BYTES // per_row)
        pad_rest = pd[:-2] + (0,) + pd[-1:]          # the height's padding is made by hand below
        outs = []
        for a in range(0, h_out, rows):
            b = min(h_out, a + rows)
            lo, hi = a * sh - ph, (b - 1) * sh - ph + k          # input rows this band reads (may be past an edge)
            start = min(max(lo, 0), h)
            end = max(min(hi, h), start)
            x = input[..., start:end, :]
            top = max(0, min(start - lo, hi - lo))            # a band wholly in the padding is all zero rows
            bottom = (hi - lo) - top - (end - start)
            if top or bottom:
                x = F.pad(x, (0, 0, top, bottom))
            outs.append(fn(x, weight, bias, st, pad_rest, 1, groups))
        return torch.cat(outs, dim=-2)
    conv._fizgig_banded = True
    return conv


def wanted():
    """FIZGIG_ROCM_BANDS=0/1 when set, else Preferences' rocm_conv_bands (on unless switched off)."""
    env = os.environ.get("FIZGIG_ROCM_BANDS", "")
    if env in ("0", "1"):
        return env == "1"
    from fizgig.user_data import pref
    return pref("rocm_conv_bands", "1") != "0"


def install(force=False):
    """On ROCm (or force=True, for checks): F.conv2d / F.conv3d banded. Idempotent. nn.Conv2d / Conv3d call F.conv*
    at run time, so every VAE picks it up."""
    if not force and not (getattr(torch.version, "hip", None) and wanted()):
        return False
    for name, nd in (("conv2d", 2), ("conv3d", 3)):
        fn = getattr(F, name)
        if not getattr(fn, "_fizgig_banded", False):
            _orig[name] = fn
            setattr(F, name, _banded(fn, nd))
    return True


def uninstall():
    for name, fn in _orig.items():
        setattr(F, name, fn)
    _orig.clear()
