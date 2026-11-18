"""Whether an AMD launch turns on PyTorch's expandable segments - decided before torch loads (the allocator reads its
setting once), so nothing here imports torch.

Only on AMD's datacenter cards (Instinct MI100 / MI200 / MI300 / MI350). On consumer Radeon cards (RDNA2 / 3 / 4) they
corrupt memory silently on current ROCm: on an RX 9070 XT (gfx1201) a live tensor read back as zeros, a model weight
turned to NaN and Krea 2 rendered garbage, with ComfyUI failing the same way once they were on (#363, @xGnome;
PyTorch issue 204212). A card it can't name counts as a consumer card: a picture of garbage is worse than running out
of memory. FIZGIG_NO_EXPANDABLE=1 keeps them off everywhere."""
import os

# Instinct: gfx908 (MI100), gfx90a (MI200), gfx94x (MI300), gfx95x (MI350). Where #341's 16 GB-capped MI350X needed
# them: a Krea 2 LoKR run ran out with 1.8 GB held in unusable pieces without.
DATACENTER = ("gfx908", "gfx90a", "gfx94", "gfx95")


def _arch(version):
    """The kernel driver's gfx_target_version (90402, 120001, 90010) as its name (gfx942, gfx1201, gfx90a)."""
    major, minor, step = version // 10000, version // 100 % 100, version % 100
    return f"gfx{major}{minor:x}{step:x}"


def amd_archs():
    """The AMD GPUs' gfx names: the installer's FIZGIG_AMD_GFX, else (Linux) the kernel driver's topology."""
    named = os.environ.get("FIZGIG_AMD_GFX", "").strip().lower()
    if named:
        return [named]
    out = []
    nodes = "/sys/class/kfd/kfd/topology/nodes"
    try:
        names = sorted(os.listdir(nodes))
    except OSError:                                # not Linux, or no AMD kernel driver
        names = []
    for name in names:
        path = os.path.join(nodes, name, "properties")
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("gfx_target_version"):
                        v = int(line.split()[1])
                        if v:                      # 0: a CPU node
                            out.append(_arch(v))
        except (OSError, ValueError, IndexError):
            continue
    return out


def expandable_ok(archs=None):
    """Expandable segments for this AMD machine: every GPU is a datacenter card, and FIZGIG_NO_EXPANDABLE isn't set."""
    if os.environ.get("FIZGIG_NO_EXPANDABLE") == "1":
        return False
    archs = amd_archs() if archs is None else archs
    return bool(archs) and all(a.startswith(DATACENTER) for a in archs)
