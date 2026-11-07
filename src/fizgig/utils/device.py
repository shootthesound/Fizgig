"""Device utilities for memory management and synchronization."""

import gc
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def fp8_scaled_mm_supported(device: Optional[Union[str, torch.device]] = None) -> bool:
    """True if the GPU has fp8 tensor cores usable by torch._scaled_mm.

    Requires compute capability >= 8.9 (Ada / Hopper / Blackwell). Older cards
    (Ampere sm_86 like the 3090, Turing, etc.) lack fp8 silicon and must fall
    back to the dequantize-to-bf16 path — for them this returns False and the
    fast path is never entered, so training/inference behaves exactly as today.
    """
    if not torch.cuda.is_available():
        return False
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        index = dev.index if dev.type == "cuda" else None
    else:
        index = None
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return (major, minor) >= (8, 9)


def card_free_vram_gb(index: int = 0) -> Optional[float]:
    """Free VRAM on the whole card in GB, as the NVIDIA driver counts it - including what other apps and the desktop
    hold, which this process's own CUDA view can't see on Windows (WDDM virtualizes memory per process). None when
    NVML isn't there. Matched to this process's CUDA device by UUID, so CUDA_VISIBLE_DEVICES is honoured."""
    try:
        import pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(index).uuid)
        h = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
        return pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**30
    except Exception:
        return None


def plannable_free_vram(device: Optional[Union[str, torch.device]] = None) -> float:
    """Free VRAM in GB for PLANNING decisions — honouring the small-card simulator.

    Set FIZGIG_SIM_VRAM_GB=16 and every planner behaves as though the machine had a 16 GB
    card: reported free becomes (simulated total − whatever Windows/desktop currently eat),
    the same view that card's real owner gets. A separate VRAM-hog process cannot do this
    job — WDDM virtualizes memory per process, so mem_get_info in Fizgig's processes never
    sees another process's ballast (the issue-#71 overcommit behaviour, met from the other
    side). Pair with apply_sim_vram_cap() so exceeding the budget genuinely OOMs too.
    """
    import os
    idx = None
    if device is not None:
        idx = torch.device(device).index
    free_b, total_b = torch.cuda.mem_get_info(idx if idx is not None else 0)
    free = free_b / 1e9
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if sim:
        try:
            reported_total = float(sim) * 0.995e9 / 1e9   # a "16 GB" card reports ~15.9
            deficit = (total_b - free_b) / 1e9            # the Windows/desktop tax
            free = min(free, max(0.0, reported_total - deficit))
        except ValueError:
            pass
    return free


def apply_sim_vram_cap(device: Optional[Union[str, torch.device]] = None):
    """The enforcement half of the simulator: cap this process's torch allocator at the
    simulated card size, so an allocation a real small card could not make OOMs here too
    instead of quietly spilling into the 5090's headroom. No-op without the env var."""
    import os
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if not sim or not torch.cuda.is_available():
        return
    try:
        idx = torch.device(device).index if device is not None else 0
        total = torch.cuda.mem_get_info(idx or 0)[1] / 1e9
        frac = min(1.0, (float(sim) * 0.995) / total)
        torch.cuda.set_per_process_memory_fraction(frac, idx or 0)
        logger.warning(f"[sim] FIZGIG_SIM_VRAM_GB={sim}: allocator capped at "
                       f"{frac * total:.1f} GB — this process behaves like a {sim} GB card.")
    except Exception as exc:
        logger.warning(f"[sim] could not cap the allocator: {exc}")


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    """Free cached memory on the specified device."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def gpu_svd(W: torch.Tensor) -> tuple:
    """SVD on GPU if available, CPU fallback. Returns (U, S, Vt) on CPU.

    A CUDA failure (usually OOM when the GPU is busy) falls back to CPU — but it's logged
    at WARNING so a slow, CPU-bound run is diagnosable instead of a silent mystery."""
    if torch.cuda.is_available():
        try:
            W_gpu = W.cuda()
            U, S, Vt = torch.linalg.svd(W_gpu, full_matrices=False)
            return U.cpu(), S.cpu(), Vt.cpu()
        except Exception as e:
            logger.warning("gpu_svd: CUDA SVD failed (%s: %s) for shape %s — falling back to CPU "
                           "(much slower). Free GPU memory to keep SVD on the GPU.",
                           type(e).__name__, e, tuple(W.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.linalg.svd(W, full_matrices=False)


def gpu_kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Kronecker product on GPU if available, CPU fallback. Returns result on CPU.

    Logs at WARNING on CUDA failure so a CPU fallback (usually OOM) is visible, not silent."""
    if torch.cuda.is_available():
        try:
            result = torch.kron(w1.cuda(), w2.cuda()).cpu()
            return result
        except Exception as e:
            logger.warning("gpu_kron: CUDA kron failed (%s: %s) for shapes %s x %s — falling back "
                           "to CPU. Free GPU memory to keep this on the GPU.",
                           type(e).__name__, e, tuple(w1.shape), tuple(w2.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.kron(w1, w2)


def synchronize_device(device: Optional[Union[str, torch.device]]):
    """Block until all pending operations on the device are complete."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def release_module_tensors(module: "torch.nn.Module") -> None:
    """Forcibly free a module's GPU memory even if the module object itself is leaked.

    An unload can null every owning reference and still not get the VRAM back if something
    outside the owner pins the module graph (seen in the field: a Repair Studio reset left
    the whole 21 GB DiT alive). The module object being pinned doesn't mean its STORAGE has
    to stay: replace every parameter/buffer with an empty tensor, drop known quant-side
    attrs (ConvRot qdata/scales), and pop instance-level `forward` patches (LoRA/AdaLN
    closures assigned as instance attrs — popping restores the class method and releases the
    closure's captured tensors). A leaked holder that later calls forward will crash loudly
    on the empty weights — strictly better than silently holding 21 GB."""
    if module is None:
        return
    try:
        for p in module.parameters():
            try:
                p.data = p.data.new_empty(0)
                if p.grad is not None:
                    p.grad = None
            except Exception:
                pass
        for b in module.buffers():
            try:
                b.data = b.data.new_empty(0)
            except Exception:
                pass
        for sub in module.modules():
            for attr in ("qdata", "scales", "weight_scale", "input_scale", "alpha"):
                v = getattr(sub, attr, None)
                if v is not None and torch.is_tensor(v):
                    try:
                        setattr(sub, attr, None)
                    except Exception:
                        pass
            # Instance-level forward patches (LoRA chains, turbo AdaLN injection) capture
            # weight tensors in their closures — pop them so those release too.
            sub.__dict__.pop("forward", None)
    except Exception:
        logger.exception("release_module_tensors: partial release only")


def flush_reserved_vram(tag: str = "", threshold_gb: float = 1.0) -> None:
    """Return cached-but-unallocated VRAM to the driver, and when segments stay pinned,
    census the stragglers pinning them.

    Field case (19 Aug): after a Repair Studio unload, allocated=0.01 GB but reserved=6.07 GB
    — a handful of tiny surviving tensors scattered inside big segments pinned ~6 GB of
    allocator cache. Steps: synchronize (blocks freed on side streams — the model loaders
    stream weights — are only returnable once their stream events settle), empty_cache, and
    if a gap remains, print EVERY surviving CUDA tensor with its shape so the pinning
    tensors can be identified and eliminated at the source."""
    import gc as _gc
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    _gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    if reserved - allocated < threshold_gb:
        return
    print(f"[vram-pin:{tag}] {reserved:.2f} GB reserved vs {allocated:.2f} GB allocated after "
          f"sync+flush — segments pinned by these survivors:", flush=True)
    count = 0
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                count += 1
                if count <= 40:
                    print(f"[vram-pin:{tag}]   {obj.numel()*obj.element_size()/2**20:9.3f} MB"
                          f"  {tuple(obj.shape)}  {obj.dtype}"
                          f"  {'grad' if obj.requires_grad else ''}", flush=True)
        except Exception:
            continue
    if count > 40:
        print(f"[vram-pin:{tag}]   … and {count - 40} more", flush=True)
    stats = torch.cuda.memory_stats()
    print(f"[vram-pin:{tag}] segments={stats.get('segment.all.current', '?')} "
          f"inactive_split={stats.get('inactive_split_bytes.all.current', 0)/2**30:.2f} GB",
          flush=True)

    # Name the holder: walk up from the largest survivor, closure-aware — cells and
    # functions are how a captured forward keeps a whole LoRA alive.
    biggest = None
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                sz = obj.numel() * obj.element_size()
                if biggest is None or sz > biggest[0]:
                    biggest = (sz, obj)
        except Exception:
            continue
    if biggest is not None:
        node = biggest[1]
        chain = []
        for _ in range(8):
            refs = [r for r in _gc.get_referrers(node)
                    if type(r).__name__ not in ("frame", "FrameType", "list_iterator")
                    and not (isinstance(r, tuple) and any(x is node for x in r))]
            if not refs:
                chain.append("(top: no python referrers)")
                break
            r = refs[0]
            tn = type(r).__name__
            if tn == "dict":
                keys = [k for k, v in r.items() if v is node][:3]
                owners = [o for o in _gc.get_referrers(r) if type(o).__name__ != "frame"]
                tn = f"dict{'' if not owners else ' of ' + type(owners[0]).__name__} keys={keys}"
            elif tn == "cell":
                fns = [o for o in _gc.get_referrers(r)
                       if callable(o) or type(o).__name__ == "function"]
                tn = "closure-cell" + (f" of {getattr(fns[0], '__qualname__', fns[0])}"
                                       if fns else "")
            elif tn == "function":
                tn = f"function {getattr(r, '__qualname__', '?')}"
            elif isinstance(r, torch.nn.Module):
                tn = f"Module:{type(r).__name__}"
            chain.append(tn)
            node = r
        print(f"[vram-pin:{tag}] largest survivor "
              f"({biggest[0]/2**20:.2f} MB {tuple(biggest[1].shape)}) held via: "
              + " <- ".join(chain), flush=True)


def report_cuda_leak(tag: str, threshold_gb: float = 2.0, top_n: int = 5,
                     orphan_min_mb: int = 128) -> float:
    """After an unload SHOULD have freed everything: if allocated VRAM is still above the
    threshold, name the holders. Walks gc for the largest live CUDA tensors and prints each
    one's referrer chain (a few levels of type names, dict keys, and owning nn.Module classes)
    so a leak report in a user's console identifies the exact holder instead of just the size.

    Returns the allocated GB either way. Prints nothing when clean. Costs one gc walk, only
    on the unload path. (Born from a real field leak: ~21 GB survived a Repair Studio reset
    with every engine attribute nulled — the holder was outside the engine.)"""
    import gc as _gc
    if not torch.cuda.is_available():
        return 0.0
    allocated = torch.cuda.memory_allocated() / 2**30
    if allocated < threshold_gb:
        return allocated
    print(f"[vram-leak:{tag}] {allocated:.2f} GB still allocated after unload — "
          f"hunting holders…", flush=True)

    tensors = []
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((obj.numel() * obj.element_size(), obj))
        except Exception:
            continue
    tensors.sort(key=lambda t: -t[0])
    print(f"[vram-leak:{tag}] {len(tensors)} live CUDA tensors, "
          f"{sum(s for s, _ in tensors)/2**30:.2f} GB gc-visible", flush=True)

    # Tensor-level chains proved useless in the field (every param is "held" by its module's
    # _parameters dict). The question is which ROOT MODULE is alive and WHO holds *it* — so
    # find every nn.Module that no other module contains, rank by resident CUDA bytes, and
    # print the roots' non-structural referrers.
    modules = [o for o in _gc.get_objects() if isinstance(o, torch.nn.Module)]
    children = set()
    for m in modules:
        for c in m._modules.values():
            if c is not None:
                children.add(id(c))

    def _cuda_bytes(mod):
        total = 0
        try:
            for p in mod.parameters():
                if p.is_cuda:
                    total += p.numel() * p.element_size()
            for b in mod.buffers():
                if b.is_cuda:
                    total += b.numel() * b.element_size()
            for sub in mod.modules():         # ConvRot int8 keeps qdata as a plain attr
                q = getattr(sub, "qdata", None)
                if q is not None and torch.is_tensor(q) and q.is_cuda:
                    total += q.numel() * q.element_size()
        except Exception:
            pass
        return total

    roots = sorted(((_cuda_bytes(m), m) for m in modules if id(m) not in children),
                   key=lambda t: -t[0])

    def _cell_owner(cell):
        """Name the code that owns a closure-cell. Two routes: a live FUNCTION reaches its
        cells through its __closure__ tuple; a SUSPENDED FRAME (generators — non-reentrant
        gradient checkpointing lives on these) holds cells directly and has no function
        object to find, so the frame's code name is the answer there."""
        try:
            for t in _gc.get_referrers(cell):
                if isinstance(t, tuple):
                    for f in _gc.get_referrers(t):
                        if callable(f) and getattr(f, "__closure__", None) is t:
                            return "fn " + getattr(f, "__qualname__", repr(f))
                elif type(t).__name__ == "frame":
                    code = t.f_code
                    return (f"frame {getattr(code, 'co_qualname', code.co_name)} "
                            f"({code.co_filename.rsplit(chr(92), 1)[-1]}:{t.f_lineno})")
            # Neither a function nor a frame — say what species DOES hold it, so the next
            # log narrows the search instead of printing another bare "closure-cell".
            kinds = [type(t).__name__ for t in _gc.get_referrers(cell)][:5]
            return f"?held-by {kinds}" if kinds else None
        except Exception:
            pass
        return None

    def _describe(holder, target):
        d = type(holder).__name__
        try:
            if isinstance(holder, dict):
                keys = [k for k, v in holder.items() if v is target][:4]
                owners = [o for o in _gc.get_referrers(holder)
                          if not isinstance(o, dict) and type(o).__name__ != "frame"]
                own = f" of {type(owners[0]).__name__}" if owners else ""
                d = f"dict{own} keys={keys}"
            elif type(holder).__name__ == "cell":
                fn = _cell_owner(holder)
                d = f"closure-cell of {fn}" if fn else "closure-cell"
            elif isinstance(holder, (list, tuple, set)):
                d += f" len={len(holder)}"
        except Exception:
            pass
        return d

    shown = 0
    for size, mod in roots:
        if size < 2**28 or shown >= top_n:      # only roots holding >=256 MB
            break
        shown += 1
        refs = [r for r in _gc.get_referrers(mod)
                if type(r).__name__ not in ("frame", "FrameType", "list_iterator")]
        descs = [_describe(r, mod) for r in refs[:6]]
        print(f"[vram-leak:{tag}]  ROOT {type(mod).__name__}: {size/2**30:.2f} GB resident"
              f"  held by: {' | '.join(descs) if descs else '(no python referrers)'}",
              flush=True)

    # ORPHANS: big CUDA tensors that belong to NO module — carried sampler state, stashed
    # activations, workspaces. A leak that isn't a module is invisible to the root scan.
    owned = set()
    for m in modules:
        try:
            for p in m.parameters(recurse=False):
                owned.add(id(p))
            for b in m.buffers(recurse=False):
                owned.add(id(b))
            q = getattr(m, "qdata", None)
            if q is not None:
                owned.add(id(q))
        except Exception:
            pass
    orphans = []
    for obj in _gc.get_objects():
        try:
            if (torch.is_tensor(obj) and obj.is_cuda and id(obj) not in owned
                    and obj.numel() * obj.element_size() >= orphan_min_mb * 2**20):
                orphans.append((obj.numel() * obj.element_size(), obj))
        except Exception:
            continue
    orphans.sort(key=lambda t: -t[0])
    for size, t in orphans[:top_n]:
        node, chain = t, []
        for _ in range(5):
            refs = [r for r in _gc.get_referrers(node)
                    if type(r).__name__ not in ("frame", "FrameType", "list_iterator")
                    and not (isinstance(r, tuple) and any(x is node for x in r))]
            if not refs:
                break
            # Prefer a CELL referrer when one exists — it leads to a nameable function,
            # where refs[0] is often gc-listing noise.
            r = next((x for x in refs if type(x).__name__ == "cell"), refs[0])
            tn = type(r).__name__
            if isinstance(r, dict):
                keys = [k for k, v in r.items() if v is node][:3]
                owners = [o for o in _gc.get_referrers(r) if type(o).__name__ != "frame"]
                tn = f"dict{'' if not owners else ' of ' + type(owners[0]).__name__} keys={keys}"
            elif tn == "cell":
                fn = _cell_owner(r)
                tn = f"closure-cell of {fn}" if fn else "closure-cell"
                chain.append(tn)
                break                      # the function name IS the answer — stop here
            chain.append(tn)
            node = r
        print(f"[vram-leak:{tag}]  ORPHAN {size/2**30:.2f} GB {tuple(t.shape)} {t.dtype}"
              f"  held via: {' <- '.join(chain) if chain else '(no python referrers)'}",
              flush=True)
    return allocated
