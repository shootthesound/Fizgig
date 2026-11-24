"""Detect whether PyTorch is running on NVIDIA CUDA or AMD ROCm (HIP).

ROCm builds expose ``torch.cuda`` (HIP backend) just like CUDA builds do,
so most device code works unchanged — but feature probes (fp8, cuDNN SDPA,
bitsandbytes wheels) need to know which stack is underneath.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def is_rocm() -> bool:
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    # torch 2.10+: torch.version.rocm; older builds only had .hip
    rocm = getattr(torch.version, "rocm", None)
    if rocm:
        return True
    hip = getattr(torch.version, "hip", None)
    if hip:
        return True
    ver = getattr(torch, "__version__", "") or ""
    return "+rocm" in ver.lower()


def pin_rocm_device() -> None:
    """AMD PCs with built-in graphics listed first (#186, #342): Launch Fizgig AMD.bat puts the graphics card's index
    in HIP_VISIBLE_DEVICES so only that card is visible. Some ROCm builds on Windows ignore it - torch still lists
    both, and "cuda" then means the built-in graphics (gfx1036 on a Ryzen 7000), whose kernels the card's PyTorch
    doesn't carry ("invalid kernel file"). When that happens, make the chosen card the current device, so every
    "cuda" tensor lands on it. When the variable worked (one device left), nothing changes."""
    import os
    want = (os.environ.get("HIP_VISIBLE_DEVICES") or "").strip()
    unset = not any((os.environ.get(k) or "").strip()
                    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    if not want.isdigit() and not unset:
        return
    try:
        import torch
        if not is_rocm():
            return
        n = torch.cuda.device_count()
        if unset:
            # nothing picked a card (#342: the launcher's pick never ran): pick it here by the launcher's rule
            idx = training_device(n)
            if idx:
                torch.cuda.set_device(idx)
                import sys
                print(f"[AMD-ROCm] nothing set HIP_VISIBLE_DEVICES: using device {idx}, "
                      f"{torch.cuda.get_device_name(idx)} (built-in graphics skipped)", file=sys.stderr, flush=True)
            return
        idx = int(want)
        if n > 1 and idx < n:
            torch.cuda.set_device(idx)
            import sys
            # stderr: this runs as fizgig is imported, before any logging is set up; the console and run logs show it
            print(f"[AMD-ROCm] HIP_VISIBLE_DEVICES={idx} was not applied by this ROCm build ({n} devices listed): "
                  f"using device {idx}, {torch.cuda.get_device_name(idx)}", file=sys.stderr, flush=True)
    except Exception:
        pass


def training_device(n) -> int:
    """The device to train on among n: not built-in graphics, the most VRAM of the rest (src/install/pick_gpu.py's
    rule, which the launcher runs); 0 when there's one device or every one is built in."""
    import torch
    if n < 2:
        return 0
    devs = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        built_in = bool(getattr(p, "is_integrated", False)) or (getattr(p, "name", "") or "").strip() == \
            "AMD Radeon(TM) Graphics"
        devs.append((i, int(getattr(p, "total_memory", 0)), built_in))
    cards = [d for d in devs if not d[2]] or devs
    return max(cards, key=lambda d: d[1])[0]


def report_rocm_devices() -> None:
    """One line per process on AMD (#342), on stderr so a stage's line lands in its run log: the variables that pick
    the card, the devices this process sees, and the one "cuda" means here."""
    import os
    import sys
    env = " ".join(f"{k}={os.environ.get(k) or '-'}"
                   for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    try:
        import torch
        if not is_rocm():
            return
        n = torch.cuda.device_count()
        cur = torch.cuda.current_device()
        names = ", ".join(f"{i}: {torch.cuda.get_device_name(i)}" for i in range(n))
        print(f"[AMD-ROCm] {env} - {n} device{'s' if n != 1 else ''} ({names}); using {cur}, "
              f"{torch.cuda.get_device_name(cur)}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[AMD-ROCm] {env} - couldn't list the devices: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def miopen_switch() -> None:
    """MIOpen off: convolutions skip it (torch.backends.cudnn.enabled = False) and use PyTorch's own kernels, as
    ComfyUI does by default on RDNA3 and newer. Preferences > Speed-ups for your graphics card > "Use AMD's MIOpen for
    convolutions" (on by default; rocm_miopen), FIZGIG_MIOPEN=0/1 winning over it. For #389, where a Windows RX 7900
    XTX froze in a VAE recache that slowed with each new picture shape (MIOpen searches kernels per shape). Said on
    stderr, in the log."""
    import os
    import sys
    env = os.environ.get("FIZGIG_MIOPEN", "")
    if env not in ("0", "1"):
        from fizgig.user_data import pref
        env = "0" if pref("rocm_miopen", "1") == "0" else "1"
    if env != "0":
        return
    try:
        import torch
    except Exception:
        return
    torch.backends.cudnn.enabled = False
    print("[AMD-ROCm] MIOpen off: convolutions use PyTorch's own kernels", file=sys.stderr, flush=True)


def fix_rocm_sdpa_masks() -> None:
    """ROCm's fused attention kernel attends wrongly when the mask broadcasts over the queries - the usual key-padding
    mask, [B, 1, 1, S]: measured on an MI350X (torch 2.14 / ROCm 10.1) against the math kernel, max error 0.1-0.2 at
    2560+ tokens (an RTX 5090: 6e-4), which turned LoKR training to NaN and changed renders above 512 px (#341, #343).
    The same mask expanded over the queries ([B, 1, L, S], a view: no memory) is right (6e-4). So on ROCm every
    scaled_dot_product_attention call gets its query-broadcast mask expanded first. Every family is covered: they all
    call it through torch.nn.functional. No-op on NVIDIA, and installed once."""
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return
    if not getattr(torch.version, "hip", None) or getattr(F.scaled_dot_product_attention, "_fizgig_rocm", False):
        return
    orig = F.scaled_dot_product_attention

    def scaled_dot_product_attention(query, key, value, attn_mask=None, *args, **kwargs):
        if (attn_mask is not None and attn_mask.dim() >= 2 and attn_mask.shape[-2] == 1
                and query.dim() >= 2 and query.shape[-2] > 1):
            attn_mask = attn_mask.expand(*attn_mask.shape[:-2], query.shape[-2], attn_mask.shape[-1])
        return orig(query, key, value, attn_mask, *args, **kwargs)

    scaled_dot_product_attention._fizgig_rocm = True
    scaled_dot_product_attention.__doc__ = orig.__doc__
    F.scaled_dot_product_attention = scaled_dot_product_attention


def backend_label() -> str:
    return "ROCm" if is_rocm() else "CUDA"
