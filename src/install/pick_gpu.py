# pick_gpu.py - which AMD GPU Fizgig should train on, for Launch Fizgig AMD.bat.
#
# On a PC with a processor's built-in graphics as well as a graphics card, ROCm can list the built-in one first
# (device 0), and torch then trains on it (#186: "AMD Radeon(TM) Graphics" ahead of an RX 7900 XT). This prints the
# index of the graphics card - not built in, the most VRAM of those - for the launcher to put in HIP_VISIBLE_DEVICES.
# It prints nothing when there is only one GPU, or when device 0 is already the one, so the usual case is untouched.
# Diagnostics go to stderr: the launcher's `for /f` reads stdout only.
#
# choose_training_gpu() is the shared rule for write_rocm_env.py (#188 / #192) so the baked FIZGIG_AMD_* identity
# matches the card the launcher trains on.

import sys


def is_built_in(props) -> bool:
    """True for processor graphics ROCm often lists ahead of the discrete card (#186)."""
    name = (getattr(props, "name", None) or "").strip()
    return bool(getattr(props, "is_integrated", False)) or name == "AMD Radeon(TM) Graphics"


def list_amd_gpus():
    """[(index, name, total_memory_bytes, built_in), ...] or None if torch/devices unavailable."""
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        return None
    if n < 1:
        return None
    gpus = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        name = (getattr(p, "name", None) or "").strip()
        gpus.append((i, name, int(p.total_memory), is_built_in(p)))
    return gpus


def choose_training_gpu(gpus=None):
    """(index, name, total_memory_bytes) for the discrete card, or None.

    Skip built-in graphics; among the rest take the most VRAM. If every device is
    built-in, fall back to that set (same as the launcher).
    """
    if gpus is None:
        gpus = list_amd_gpus()
    if not gpus:
        return None
    cards = [g for g in gpus if not g[3]] or gpus
    best = max(cards, key=lambda g: g[2])
    return best[0], best[1], best[2]


def main():
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception as e:
        print(f"[AMD-ROCm] GPU pick skipped: {e}", file=sys.stderr)
        return
    if n < 2:
        return
    gpus = list_amd_gpus()
    if not gpus:
        return
    for i, name, total, built_in in gpus:
        print(
            f"[AMD-ROCm] GPU {i}: {name}, {total / 2**30:.1f} GB"
            f"{' (built in)' if built_in else ''}",
            file=sys.stderr,
        )
    best = choose_training_gpu(gpus)
    if best is not None and best[0] != 0:
        print(best[0])


if __name__ == "__main__":
    main()
