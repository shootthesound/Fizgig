"""Reuse unchanged blocks: the Repair Studio's activation cache for the standard-layer families.

For one render the DiT's blocks (each block-map id whose Linears sit under one ModuleList entry) are wrapped. A
render records every block's output; the next render with the same setup (LoRAs, prompt, seed, canvas, sampling)
where only sliders moved replays the outputs of the blocks that run before the earliest changed one instead of
running them. The run order is recorded, so a family whose text stack runs before its main blocks needs nothing
extra; a changed id with no single module (Krea 2's "Input and output") means a full render.

Step 1 only (both DiT calls when CFG is on): every block before the change sees the input it saw last time, so the
picture is identical to a full render; later steps run in full, their latent already carrying the change. Replaying
every step (old Klein's Turbo Preview, as it was called) was measured on 4 Oct 2026 (768², one tweak, Klein / Krea 2 / Qwen Image 2.1):
the preview showed only 6-33% of the tweak's visible effect, so it is not offered.
Any failure (out of memory included) drops the cache and the render carries on without it. The workbench marks each
step (`step(i)`, from the sampler's on_step, which every driver calls before a step's DiT calls).
"""
import contextlib
import logging

import torch

logger = logging.getLogger(__name__)

def _root(mods):
    """The ModuleList entry a block's Linears share ("double_blocks.3"), or None."""
    parts = [m.split(".") for m in mods]
    if not parts:
        return None
    pre = []
    for seg in zip(*parts):
        if any(s != seg[0] for s in seg):
            break
        pre.append(seg[0])
    while pre and not pre[-1].isdigit():
        pre.pop()
    return ".".join(pre) if len(pre) >= 2 else None


def block_modules(driver, dit):
    """{block id: the module to wrap} for the block-map ids that own one ModuleList entry each."""
    out, seen = {}, {}
    for g in driver.block_map(dit):
        for b in g.blocks:
            r = _root(b.modules)
            if r is None:
                continue
            try:
                m = dit.get_submodule(r)
            except AttributeError:
                continue
            if id(m) in seen:                     # two ids on one module: neither can be skipped on its own
                out.pop(seen[id(m)], None)
                continue
            seen[id(m)] = b.id
            out[b.id] = m
    return out


# what a render needs left on the card after the cache: the rest of step 1, the later steps and the decode. A stored block
# output that would cut into it goes to system RAM instead (a 1280x720 x 121 Vimoo clip stores ~7 GB: on the card it ran a
# 32 GB card out of memory, 6 Nov 2026) and comes back to the card when replayed - H3's engine does the same
# (_resume_cache_device)
KEEP_FREE_GB = 8.0


def _free_gb():
    try:
        return torch.cuda.mem_get_info()[0] / 2 ** 30
    except Exception:
        return float("inf")


def _store(x):
    """A block output to keep: a copy on the card while the card keeps KEEP_FREE_GB free past it, else in system RAM."""
    if torch.is_tensor(x) and x.is_cuda and _free_gb() - _nbytes(x) / 2 ** 30 < KEEP_FREE_GB:
        return x.to("cpu", copy=True)
    if isinstance(x, (tuple, list)):
        out = [_store(v) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x.clone() if torch.is_tensor(x) else x


def _back(x, device):
    """A stored output for replay, on the device the block runs on."""
    if torch.is_tensor(x):
        return x.to(device, copy=True) if x.device != device else x.clone()
    if isinstance(x, (tuple, list)):
        out = [_back(v, device) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x


def _arg_device(a, k):
    for v in list(a) + list(k.values()):
        if torch.is_tensor(v):
            return v.device
    return None


def _clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, tuple):
        return tuple(_clone(v) for v in x)
    if isinstance(x, list):
        return [_clone(v) for v in x]
    return x


def _nbytes(x):
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, (tuple, list)):
        return sum(_nbytes(v) for v in x)
    return 0


class ActivationCache:
    def __init__(self):
        self.last_replayed = 0          # blocks the last render replayed (0 = a full render)
        self.clear()

    def clear(self):
        self.key = None
        self.sig = None
        self.order = []                 # block ids in the order they ran (step 1's first call of a full render)
        self.out = {}                   # (step, call in the step, block id) -> output
        self._st = None

    @property
    def nbytes(self):
        return sum(_nbytes(v) for v in self.out.values())

    def plan(self, key, sig):
        """How many blocks (in run order) the next render may replay. key: everything but the sliders;
        sig: {block id: the sliders' values for it}."""
        if self.key != key or self.sig is None or not self.order:
            self.clear()
            self.key = key
            return 0
        changed = {b for b in set(sig) | set(self.sig) if sig.get(b) != self.sig.get(b)}
        if not changed:
            return len(self.order)
        if not changed <= set(self.order):
            return 0
        return min(self.order.index(b) for b in changed)

    def step(self, i):
        """The sampler is starting step i (0-based)."""
        st = self._st
        if st is not None:
            st["step"], st["call"] = int(i), -1

    @contextlib.contextmanager
    def render(self, dit, modules, key, sig):
        """Wraps the DiT and its blocks for one render. On success the cache holds this render's outputs;
        on any exception it is emptied."""
        n = self.plan(key, sig)
        replay = set(self.order[:n])
        st = self._st = {"step": 0, "call": -1, "ok": True}
        order = [] if not self.order else None
        patched = []

        def _fail(e):
            st["ok"] = False
            self.out.clear()
            logger.warning("Reuse unchanged blocks: cache dropped (%s: %s) - full renders until it fits", type(e).__name__, e)

        def wrap_call(orig):
            def fwd(*a, **k):
                st["call"] += 1
                return orig(*a, **k)
            return fwd

        def wrap_block(bid, orig):
            def fwd(*a, **k):
                at = (st["step"], st["call"], bid)
                if not st["ok"] or at[0] > 0:
                    return orig(*a, **k)
                if bid in replay:
                    hit = self.out.get(at)
                    if hit is not None:
                        dev = _arg_device(a, k)
                        return _back(hit, dev) if dev is not None else _clone(hit)
                y = orig(*a, **k)
                if order is not None and at[:2] == (0, 0):
                    order.append(bid)
                try:
                    self.out[at] = _store(y)
                except torch.OutOfMemoryError as e:
                    _fail(e)
                return y
            return fwd

        def patch(m, fn):
            had = "forward" in m.__dict__
            patched.append((m, had, m.__dict__.get("forward")))
            m.forward = fn(m.forward)

        try:
            patch(dit, wrap_call)
            for bid, m in modules.items():
                patch(m, lambda f, bid=bid: wrap_block(bid, f))
            yield n
            if st["ok"]:
                if order is not None:
                    self.order = order
                self.sig = dict(sig)
                self.last_replayed = n
            else:
                self.clear()
                self.last_replayed = 0
        except BaseException:
            self.clear()
            self.last_replayed = 0
            raise
        finally:
            self._st = None
            for m, had, f in reversed(patched):
                if had:
                    m.forward = f
                else:
                    del m.forward
