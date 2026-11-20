"""Simulated GPU memory: a real Fizgig run's card memory, measured on the CPU.

`run_launch(family, values, ...)` starts a run as the page does (form.inputs -> launch.plan -> each stage: caching, then
training with its previews, in this process), on full-size models made of fake tensors (FakeTensorMode: shapes and
dtypes, no data), on a simulated card. Every card allocation goes through a model of PyTorch's caching allocator, so
the run reports the card memory a real one allocates and reserves - and, on a card of a given size, runs out of memory
where a real one would (after giving its cache back, as PyTorch does). A run takes seconds.

Model files are their safetensors headers only (memsim_headers/: the tensor list, a few KB gzipped). Each is written
as a header-only file: Fizgig's own header readers read it as usual, and a tensor read gives a fake tensor.

Values: a fake tensor has none. That is right for weights and activations, but Fizgig computes shapes from small
tensors (position ids, indices), so every tensor of up to SHADOW_MAX elements made from known values (a factory, a real
tensor, other such tensors) keeps a real CPU copy computed alongside. Where a value is needed and unknown (a model's
numbers): a fixed stand-in - a scalar reads 0.5 / 1 / True, NaN and inf checks find none.

Checked against a 5090 (Windows, Klein 9B, the page's launch, each stage): the training peak 12.6 / 14.3 / 8.7 GB
(fp8 file / INT8 / NF4) vs 12.5 / 14.1 / 8.8 measured; previews' render and decode peaks within 0.3 GB; on the card
capped to 14.7 GB (a 16 GB card under Windows) training 9.1 vs 9.2 GB and the preview decode 11.9 vs 12.2 GB.

Needs a torch built with CUDA (no card): on the CPU-only build some fake card tensors can't be made."""
import bisect
import contextlib
import gzip
import importlib
import json
import logging
import os
import struct
import sys
import types
import weakref
from unittest import mock

import torch
from torch._subclasses.fake_tensor import (DataDependentOutputException, DynamicOutputShapeException, FakeTensor,
                                           FakeTensorMode, unset_fake_temporarily)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

GB = 1 << 30
MB = 1 << 20

# The margin every figure carries. Simulated minus measured on a 5090 (20 Nov 2026, Klein 9B, each stage and preview
# phase): allocated within 0.03 GB everywhere; reserved from -0.32 GB (the preview's render and decode, 2.6%: the real
# allocator fragments a little more there) to +0.6 GB (an INT8 epoch with previews: the simulation errs high). The margin
# is at least 1.5x the worst shortfall seen: a figure is its peak plus max(0.5 GB, 4%), and a run "fits" a card when it
# completes on that card less the margin.
MARGIN_GB, MARGIN_FRACTION = 0.5, 0.04


def margin(gb):
    return max(MARGIN_GB, MARGIN_FRACTION * gb)


def with_margin(gb):
    """A simulated peak as a figure to plan with."""
    return gb + margin(gb)
HEADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memsim_headers")
CUDA_BUILT = torch.backends.cuda.is_built()

# ---- the caching allocator (c10/cuda/CUDACachingAllocator.cpp) -------------------------------------------------------
# sizes rounded to 512 B; requests up to 1 MB from a small pool of 2 MB segments, larger ones from a large pool (a new
# segment is 20 MB for requests under 10 MB, else the request rounded up to 2 MB); best fit among cached free blocks; a
# block split when the rest is big enough (over 1 MB in the large pool, 512 B in the small one); freed blocks merge with
# free neighbours. Expandable segments (Linux): one segment per pool that grows at its end, in 2 MB pages. One stream.
SMALL, SMALL_BUFFER, LARGE_BUFFER, MIN_LARGE_ALLOC, ROUND_LARGE, PAGE = MB, 2 * MB, 20 * MB, 10 * MB, 2 * MB, 2 * MB


def _round(n):
    return 512 if n < 512 else (n + 511) // 512 * 512


class _Block:
    __slots__ = ("seg", "off", "size", "free", "prev", "next")

    def __init__(self, seg, off, size):
        self.seg, self.off, self.size, self.free, self.prev, self.next = seg, off, size, True, None, None


class _Pool:
    def __init__(self, small, expandable, room):
        self.small, self.expandable, self.room = small, expandable, room
        self.free = []                         # sorted (size, seg, off, id, block)
        self.nseg = 0
        self.tail = None                       # expandable: the segment's last block
        self.end = 0                           # expandable: mapped bytes

    @staticmethod
    def _key(b):
        return b.size, b.seg, b.off, id(b)

    def _add_free(self, b):
        bisect.insort(self.free, (*self._key(b), b))

    def _drop_free(self, b):
        i = bisect.bisect_left(self.free, self._key(b), key=lambda e: e[:4])
        assert self.free[i][4] is b
        del self.free[i]

    def _segment(self, size):
        return SMALL_BUFFER if self.small else (LARGE_BUFFER if size < MIN_LARGE_ALLOC else
                                                (size + ROUND_LARGE - 1) // ROUND_LARGE * ROUND_LARGE)

    def alloc(self, size):
        """-> (block, newly reserved bytes)"""
        i = bisect.bisect_left(self.free, (size,), key=lambda e: e[:1])
        if i >= len(self.free):                # nothing cached fits: the card must give more (or the cache back)
            if self.expandable:
                t = self.tail
                start = self.end if t is None else (t.off if t.free else t.off + t.size)
                self.room(max(0, (start + size - self.end + PAGE - 1) // PAGE * PAGE))
            else:
                self.room(self._segment(size))
            i = bisect.bisect_left(self.free, (size,), key=lambda e: e[:1])
        grown = 0
        if i < len(self.free):
            b = self.free[i][4]
            self._drop_free(b)
        elif self.expandable:
            t = self.tail
            if t is not None and t.free:       # grow the free tail block to fit
                self._drop_free(t)
                b = t
            else:
                b = _Block(0, self.end if t is None else t.off + t.size, 0)
                b.prev, self.tail = t, b
                if t is not None:
                    t.next = b
            need = b.off + size - self.end
            if need > 0:
                grown = (need + PAGE - 1) // PAGE * PAGE
                self.end += grown
            b.size = self.end - b.off
        else:
            self.nseg += 1
            b = _Block(self.nseg, 0, self._segment(size))
            grown = b.size
        rest = b.size - size
        if rest >= 512 if self.small else rest > SMALL:
            r = _Block(b.seg, b.off + size, rest)
            r.prev, r.next = b, b.next
            if b.next is not None:
                b.next.prev = r
            b.next, b.size = r, size
            if self.expandable and self.tail is b:
                self.tail = r
            self._add_free(r)
        b.free = False
        return b, grown

    def empty_cache(self):
        """torch.cuda.empty_cache: whole free segments go back (expandable: the free pages at the end). -> bytes"""
        back = 0
        if self.expandable:
            t = self.tail
            if t is not None and t.free:
                keep = (t.off + PAGE - 1) // PAGE * PAGE
                back, self.end = self.end - keep, keep
                self._drop_free(t)
                t.size = self.end - t.off
                if t.size:
                    self._add_free(t)
                else:
                    self.tail = t.prev
                    if t.prev is not None:
                        t.prev.next = None
            return back
        for e in [e for e in self.free if e[4].prev is None and e[4].next is None]:
            self._drop_free(e[4])
            back += e[4].size
        return back

    def release(self, b):
        b.free = True
        for nb in (b.prev, b.next):
            if nb is not None and nb.free:
                self._drop_free(nb)
                if nb is b.prev:               # merge into the lower block
                    nb.size += b.size
                    nb.next = b.next
                    if b.next is not None:
                        b.next.prev = nb
                    if self.tail is b:
                        self.tail = nb
                    b = nb
                else:
                    b.size += nb.size
                    b.next = nb.next
                    if nb.next is not None:
                        nb.next.prev = b
                    if self.tail is nb:
                        self.tail = b
        self._add_free(b)


class CardMeter(TorchDispatchMode):
    """Every card storage from the op that makes it until it dies, through the allocator model. now / peak are allocated
    bytes, reserved / peak_reserved the allocator's; reset_peaks() resets the peaks (torch.cuda.reset_peak_memory_stats),
    top / top_reserved are never reset. cap: bytes the card can give (None: no limit) - past it the allocator frees its
    cached free segments and tries again, then raises torch.OutOfMemoryError, as PyTorch does."""

    def __init__(self, expandable=False, cap=None):
        super().__init__()
        self.cap = cap
        self.pools = {True: _Pool(True, expandable, self._room), False: _Pool(False, expandable, self._room)}
        self.live = {}                         # id(storage) -> (weakref, bytes, block, small)
        self.now = self.peak = self.reserved = self.peak_reserved = self.top = self.top_reserved = 0
        self.ooms = 0

    def _gone(self, key):
        ent = self.live.pop(key, None)
        if ent is not None:
            self.now -= ent[1]
            self.pools[ent[3]].release(ent[2])

    def _room(self, nbytes):
        if self.cap is None or self.reserved + nbytes <= self.cap:
            return
        self.empty_cache()
        if self.reserved + nbytes > self.cap:
            self.ooms += 1
            raise torch.OutOfMemoryError(
                f"CUDA out of memory (simulated card). Tried to allocate {nbytes / MB:.0f} MiB: "
                f"{self.reserved / GB:.2f} GiB reserved, {self.now / GB:.2f} GiB allocated, card limit "
                f"{self.cap / GB:.2f} GiB")

    def add(self, t):
        if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
            return
        st = t.untyped_storage()
        key = id(st)
        if key in self.live and self.live[key][0]() is st:
            return
        if st.nbytes() == 0:
            return
        n = _round(st.nbytes())
        small = n <= SMALL
        blk, grown = self.pools[small].alloc(n)
        self.live[key] = (weakref.ref(st, lambda _, k=key: self._gone(k)), n, blk, small)
        self.now += n
        self.reserved += grown
        self.peak, self.top = max(self.peak, self.now), max(self.top, self.now)
        self.peak_reserved, self.top_reserved = max(self.peak_reserved, self.reserved), max(self.top_reserved,
                                                                                            self.reserved)

    def empty_cache(self):
        self.reserved -= sum(p.empty_cache() for p in self.pools.values())

    def reset_peaks(self):
        self.peak, self.peak_reserved = self.now, self.reserved

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            self.add(t)
        return out


# ---- values: shadows for the small tensors shapes come from, stand-ins elsewhere --------------------------------------
SHADOW_MAX = 1 << 20
_A = torch.ops.aten
_ASSUMED = {_A.isnan.default: False, _A.isinf.default: False, _A.isposinf.default: False,
            _A.isneginf.default: False, _A.isfinite.default: True}


class _IdMap:
    """A weak map keyed by tensor identity (WeakKeyDictionary compares keys with ==, elementwise on tensors)."""

    def __init__(self):
        self.d = {}

    def get(self, t, default=None):
        e = self.d.get(id(t))
        return e[1] if e is not None and e[0]() is t else default

    def __contains__(self, t):
        return self.get(t) is not None

    def __setitem__(self, t, v):
        k = id(t)
        self.d[k] = (weakref.ref(t, lambda _, k=k, d=self.d: d.pop(k, None)), v)

    def __delitem__(self, t):
        self.d.pop(id(t), None)


def _cpu_arg(x):
    if isinstance(x, torch.device) and x.type == "cuda":
        return torch.device("cpu")
    if isinstance(x, str) and x.startswith("cuda"):
        return "cpu"
    return x


class SimMeter(CardMeter):
    """CardMeter plus values (see the module docstring); stand_ins counts the ops whose output shape needed values that
    weren't known (computed on zeros)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shadow = _IdMap()
        self.stand_ins = {}

    def _sh(self, t):
        return self.shadow.get(t) if isinstance(t, FakeTensor) else t      # a real tensor is its own value

    def _on_shadows(self, func, args, kwargs):
        flat = tree_flatten((args, kwargs))[0]
        if any(isinstance(x, torch.Generator) for x in flat) or any(
                self._sh(t) is None for t in flat if isinstance(t, torch.Tensor)):
            return None
        conv = lambda x: self._sh(x) if isinstance(x, torch.Tensor) else _cpu_arg(x)     # noqa: E731
        sargs, skw = tree_map(conv, args), tree_map(conv, kwargs)
        if skw.get("pin_memory"):
            skw["pin_memory"] = False
        try:
            with unset_fake_temporarily():
                return func(*sargs, **skw)
        except Exception:                      # noqa: BLE001 - no shadow, not a failure
            return None

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _A._local_scalar_dense.default:
            s = self._sh(args[0])
            if s is not None:
                with unset_fake_temporarily():
                    return s.item()
            t = args[0]
            return True if t.dtype == torch.bool else 0.5 if (t.is_floating_point() or t.is_complex()) else 1
        if func in _ASSUMED:
            out = super().__torch_dispatch__(func, types_, args, kwargs)
            with unset_fake_temporarily():
                self.shadow[out] = torch.full(out.shape, _ASSUMED[func], dtype=torch.bool)
            return out
        try:
            out = super().__torch_dispatch__(func, types_, args, kwargs)
        except (DynamicOutputShapeException, DataDependentOutputException):
            return self._value_shaped(func, args, kwargs)
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        if outs and all(o.numel() <= SHADOW_MAX for o in outs):
            sout = self._on_shadows(func, args, kwargs)
            if sout is not None:
                for o, s in zip(outs, [t for t in tree_flatten(sout)[0] if isinstance(t, torch.Tensor)]):
                    if isinstance(o, FakeTensor):
                        self.shadow[o] = s
                return out
        for o in outs:                         # written by an op without shadows (in place): its values are gone
            if isinstance(o, FakeTensor) and o in self.shadow:
                del self.shadow[o]
        return out

    def _value_shaped(self, func, args, kwargs):
        """An output whose shape depends on values (unique, nonzero): computed on the shadows, else on zeros."""
        dev = next((t.device for t in tree_flatten((args, kwargs))[0] if isinstance(t, FakeTensor)), None)
        sout = self._on_shadows(func, args, kwargs)
        if sout is None:
            def zeros(t):
                if isinstance(t, FakeTensor):
                    with unset_fake_temporarily():
                        return torch.zeros(t.shape, dtype=t.dtype)
                return t
            with unset_fake_temporarily():
                sout = func(*tree_map(zeros, args), **tree_map(zeros, kwargs))
            self.stand_ins[str(func)] = self.stand_ins.get(str(func), 0) + 1

        def fake(s):
            if isinstance(s, torch.Tensor):
                f = torch.empty(s.shape, dtype=s.dtype, device=dev)        # the fake mode below makes it a fake
                self.add(f)
                self.shadow[f] = s
                return f
            return s
        return tree_map(fake, sout)


# ---- model files: headers only --------------------------------------------------------------------------------------
_FAKE = set()
_DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
       "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def _norm(p):
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def is_fake(p):
    try:
        return _norm(p) in _FAKE
    except TypeError:
        return False


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def save_header(header, name):
    """A model file's header as a fixture: memsim_headers/<name>.json.gz."""
    os.makedirs(HEADERS, exist_ok=True)
    with gzip.open(os.path.join(HEADERS, f"{name}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(header, f, separators=(",", ":"))


def load_header(name):
    with gzip.open(os.path.join(HEADERS, f"{name}.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def write_model_file(header, path):
    """A header-only safetensors file, registered as fake: real to every header reader, fake to tensor reads."""
    raw = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
    _FAKE.add(_norm(path))
    return str(path)


def _tensors(path):
    h = read_header(path)
    return {k: v for k, v in h.items() if k != "__metadata__"}, h.get("__metadata__", {}) or {}


def _empty(info, device=None, dtype=None):
    return torch.empty(info["shape"], dtype=dtype or _DT[info["dtype"]], device=device)


def _dev(device):
    return None if device is None or str(device) == "cpu" else device


class _FakeSafeOpen:
    """safetensors.safe_open on a header-only file."""

    def __init__(self, path, framework="pt", device="cpu"):
        self.t, self.meta = _tensors(path)
        self.device = _dev(device)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def keys(self):
        return list(self.t)

    offset_keys = keys

    def metadata(self):
        return self.meta

    def get_tensor(self, k):
        return _empty(self.t[k], self.device)

    def get_slice(self, k):
        return _FakeSlice(self.t[k])


class _FakeSlice:
    def __init__(self, info):
        self.info = info

    def get_dtype(self):
        return self.info["dtype"]

    def get_shape(self):
        return list(self.info["shape"])

    def __getitem__(self, idx):
        return _empty(self.info)[idx].clone()


def _materialise(x):
    """A fake tensor as real zeros of its shape and dtype (what a save writes)."""
    if isinstance(x, FakeTensor):
        with unset_fake_temporarily():
            return torch.zeros(x.shape, dtype=x.dtype)
    return x


# ---- the simulated card -----------------------------------------------------------------------------------------------
CURRENT = None
_real_swap = torch.utils.swap_tensors
_DATA = torch._C.TensorBase.data
_real_numpy, _real_tolist = torch.Tensor.numpy, torch.Tensor.tolist


def _swap_or_set(t1, t2):
    """Module._apply swaps parameters; a fake one carries a weakref (the fake mode's memo) and can't be swapped, so its
    data is set instead - the same tensor object afterwards, as a swap leaves it."""
    try:
        _real_swap(t1, t2)
    except RuntimeError:
        with torch.no_grad():
            _set_data(t1, t2)


def _set_data(self, v):
    """x.data = y (block swap moves weights this way): a fake tensor's device is its own attribute - it follows."""
    _DATA.__set__(self, v)
    if isinstance(self, FakeTensor) and isinstance(v, FakeTensor):
        self.fake_device = v.fake_device


def _values(t):
    s = CURRENT.shadow.get(t) if CURRENT is not None else None
    with unset_fake_temporarily():
        return s if s is not None else torch.zeros(t.shape, dtype=t.dtype)


def _numpy(self, *a, **k):
    """A fake tensor's values: its shadow, else zeros (a preview picture)."""
    if isinstance(self, FakeTensor):
        v = _values(self)
        with unset_fake_temporarily():
            return _real_numpy(v.float() if v.dtype == torch.bfloat16 else v)
    return _real_numpy(self, *a, **k)


def _tolist(self):
    if isinstance(self, FakeTensor):
        v = _values(self)
        with unset_fake_temporarily():
            return _real_tolist(v)
    return _real_tolist(self)


class _Stream:
    def __init__(self, *a, **k):
        pass

    def synchronize(self):
        pass

    def wait_event(self, e):
        pass

    def wait_stream(self, s):
        pass

    def record_event(self, e=None):
        return e or _Event()

    def query(self):
        return True


class _Event(_Stream):
    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def elapsed_time(self, other):
        return 1.0


class _Inline:
    """An offloader's mover thread, run inline (the fake mode and the meter are per thread)."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:             # noqa: BLE001 - surfaces at .result(), as a thread's would
            f.set_exception(e)
        return f

    def shutdown(self, wait=True, **k):
        pass


@contextlib.contextmanager
def simulated_card(card_gb=32.0, others_gb=0.0, capped=True, expandable=False, capability=(12, 0)):
    """Fake tensors on a simulated card; yields the meter. others_gb: held by other programs (Windows' desktop: about
    1.3-1.9 GB), left out of free memory. capped: the run gets card - others at most and runs out of memory past it;
    False measures the peak a card with room to spare reaches. expandable: PyTorch's expandable segments (Linux runs:
    families/train.py sets them off Windows)."""
    global CURRENT
    total, others = int(card_gb * GB), int(others_gb * GB)
    meter = CURRENT = SimMeter(expandable=expandable, cap=total - others if capped else None)
    props = types.SimpleNamespace(total_memory=total, name="Simulated GPU", major=capability[0], minor=capability[1],
                                  multi_processor_count=170)
    c = torch.cuda
    patches = [
        (c, "is_available", lambda: True), (c, "device_count", lambda: 1), (c, "current_device", lambda: 0),
        (c, "get_device_name", lambda *a: "Simulated GPU"), (c, "get_device_properties", lambda *a: props),
        (c, "get_device_capability", lambda *a: capability), (c, "is_bf16_supported", lambda *a, **k: True),
        (c, "Stream", _Stream), (c, "Event", _Event), (c, "current_stream", lambda *a: _Stream()),
        (c, "default_stream", lambda *a: _Stream()), (c, "stream", lambda s=None: contextlib.nullcontext()),
        (c, "synchronize", lambda *a: None), (c, "empty_cache", meter.empty_cache), (c, "ipc_collect", lambda: None),
        (c, "memory_allocated", lambda *a: meter.now), (c, "max_memory_allocated", lambda *a: meter.peak),
        (c, "memory_reserved", lambda *a: meter.reserved), (c, "max_memory_reserved", lambda *a: meter.peak_reserved),
        (c, "reset_peak_memory_stats", lambda *a: meter.reset_peaks()),
        (c, "reset_max_memory_allocated", lambda *a: meter.reset_peaks()),
        (c, "mem_get_info", lambda *a: (max(0, total - others - meter.reserved), total)),
        (c, "set_per_process_memory_fraction", lambda *a, **k: None),
        (c, "manual_seed", lambda *a: None), (c, "manual_seed_all", lambda *a: None),
        (torch.utils, "swap_tensors", _swap_or_set),
        (torch.Tensor, "data", property(_DATA.__get__, _set_data)),
        (torch.Tensor, "numpy", _numpy), (torch.Tensor, "tolist", _tolist),
        (torch.Tensor, "pin_memory", lambda self, *a, **k: self),
        (torch.Tensor, "is_pinned", lambda self, *a, **k: False),
    ]
    import safetensors
    import safetensors.torch as st
    real_open, real_load, real_save, real_tsave = safetensors.safe_open, st.load_file, st.save_file, torch.save

    class _RealOpen:
        """safe_open on a real file (a cache file): read on the CPU outside the fake mode (which can't wrap a file's
        storage), then moved as asked."""

        def __init__(self, path, framework="pt", device="cpu"):
            with unset_fake_temporarily():
                self.f = real_open(path, framework=framework, device="cpu")
            self.device = _dev(device)

        def __enter__(self):
            self.f.__enter__()
            return self

        def __exit__(self, *a):
            return self.f.__exit__(*a)

        def __getattr__(self, n):
            return getattr(self.f, n)

        def _move(self, t):
            return t if self.device is None else t.to(self.device)

        def get_tensor(self, k):
            with unset_fake_temporarily():
                t = self.f.get_tensor(k)
            return self._move(t)

        def get_slice(self, k):
            s, outer = self.f.get_slice(k), self

            class _S:
                def __getattr__(self, n):
                    return getattr(s, n)

                def __getitem__(self, idx):
                    with unset_fake_temporarily():
                        t = s[idx]
                    return outer._move(t)
            return _S()

    def safe_open(path, *a, **k):
        return _FakeSafeOpen(path, *a, **k) if is_fake(path) else _RealOpen(path, *a, **k)

    def load_file(path, device="cpu"):
        if is_fake(path):
            t, _ = _tensors(path)
            return {n: _empty(v, _dev(device)) for n, v in t.items()}
        with unset_fake_temporarily():
            sd = real_load(path)
        return sd if _dev(device) is None else {n: t.to(device) for n, t in sd.items()}

    def save_file(tensors, path, *a, **k):
        return real_save({n: _materialise(t) for n, t in tensors.items()}, path, *a, **k)

    def tsave(obj, *a, **k):
        return real_tsave(tree_map(_materialise, obj), *a, **k)

    reals = {real_open: safe_open, real_load: load_file, real_save: save_file, real_tsave: tsave}

    def reader_patch(cls):
        """Fizgig's own header readers: a tensor read on a fake file gives a fake; on a real one, read on the CPU."""
        real_get = cls.get_tensor

        def get_tensor(self, key, device=None, dtype=None, *a, **k):
            fn = getattr(self, "filename", None) or getattr(self, "path", None)
            if not isinstance(fn, (str, os.PathLike)):     # a reader over another reader (key renaming): as it is
                return real_get(self, key, device, dtype, *a, **k)
            if is_fake(fn):
                return _empty(self.header[key], _dev(device), dtype)
            with unset_fake_temporarily():
                t = real_get(self, key, None, dtype, *a, **k)
            return t if _dev(device) is None else t.to(device)
        return get_tensor

    from fizgig.utils import safetensors as fu
    readers = [fu.MemoryEfficientSafeOpen]
    try:
        from fizgig.krea2 import safetensors_utils as ku
        readers += [v for v in vars(ku).values() if isinstance(v, type) and v.__module__ == ku.__name__
                    and "get_tensor" in vars(v)]
    except ImportError:
        pass
    import concurrent.futures
    import fizgig.modules.offloading  # noqa: F401 - loaded so its mover thread is made inline below
    with contextlib.ExitStack() as stack:
        for obj, name, val in patches:
            stack.enter_context(mock.patch.object(obj, name, val))
        for cls in readers:
            stack.enter_context(mock.patch.object(cls, "get_tensor", reader_patch(cls)))
        stack.enter_context(mock.patch.object(torch, "save", tsave))
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith(("fizgig", "safetensors")):
                continue
            for attr, val in list(vars(mod).items()):
                try:
                    hit = val in reals
                except TypeError:
                    hit = False
                if hit:
                    stack.enter_context(mock.patch.object(mod, attr, reals[val]))
            if getattr(mod, "ThreadPoolExecutor", None) is concurrent.futures.ThreadPoolExecutor:
                stack.enter_context(mock.patch.object(mod, "ThreadPoolExecutor", _Inline))
        stack.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
        stack.enter_context(meter)
        try:
            yield meter
        finally:
            CURRENT = None


# ---- a run as the page launches it ------------------------------------------------------------------------------------
class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(record.getMessage())
        except Exception:                      # noqa: BLE001
            pass


def pictures(folder, n=3, side=1024, caption="a photo of a person standing in a field"):
    """n noise pictures (side x side) with captions: enough for the caching and a few training steps."""
    import numpy as np
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8)).save(os.path.join(folder, f"p{i}.png"))
        with open(os.path.join(folder, f"p{i}.txt"), "w", encoding="utf-8") as f:
            f.write(caption)


_STAGE_MODULES = {"cache.py": "fizgig.families.cache", "train.py": "fizgig.families.train"}


def run_launch(family, tmp, values=None, *, megapixels=1.0, card_gb=32.0, others_gb=0.0, capped=True,
               expandable=False, with_room=True):
    """A run of `family` as Start launches it, with the Training form's defaults plus `values` (form keys), one epoch
    on 3 pictures at `megapixels`, every stage on one simulated card (each starting with the card as a new process
    finds it: the previous stage's cache given back). Model files come from memsim_headers/<family>_<pref_key>.
    with_room: the card's usable memory less margin() - completing then means fitting with the margin to spare.

    -> SimpleNamespace(stages={name: (allocated peak, reserved peak) in GB}, lines=[log lines], ooms, stand_ins,
       error=the exception that ended it, or None)"""
    from fizgig.families import form, launch
    from fizgig.families.registry import get
    for m in _STAGE_MODULES.values():
        importlib.import_module(m)                     # before the card: their imports get the patched readers
    desc = get(family)
    tmp = str(tmp)
    models = {}
    for mf in desc.model_files:
        if os.path.exists(os.path.join(HEADERS, f"{family}_{mf.pref_key}.json.gz")):
            models[mf.pref_key] = write_model_file(load_header(f"{family}_{mf.pref_key}"),
                                                   os.path.join(tmp, f"{mf.pref_key}.safetensors"))
    img = os.path.join(tmp, "pictures")
    pictures(img, side=max(512, int(round((megapixels * 1024 * 1024) ** 0.5 / 64)) * 64))
    v = {**form.DEFAULTS, "LORA_NAME": "sim", "LORA_OUTPUT_DIR": os.path.join(tmp, "out"), "MAX_TRAIN_EPOCHS": "1",
         "DATASET_MEGAPIXELS": f"{megapixels:g}", "DATASET_CONFIG": os.path.join(tmp, "dataset.toml"),
         "COMPILE_BLOCKS": "Off", "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False, **(values or {})}
    inp = form.inputs(desc, v, models=models, python=sys.executable,
                      repo_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_folder=img,
                      cache_root=os.path.join(tmp, "cache"), samples_dir=os.path.join(tmp, "samples"))
    plan = launch.plan(desc, inp)
    if plan.problems:
        raise RuntimeError(f"the launch has problems: {plan.problems}")
    for path, text in plan.files:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    log = _Lines()
    root = logging.getLogger()
    root.addHandler(log)
    old_level = root.level
    root.setLevel(logging.INFO)
    out = types.SimpleNamespace(stages={}, lines=log.lines, ooms=0, stand_ins={}, error=None)
    if capped and with_room:
        others_gb += margin(card_gb - others_gb)       # the margin held back, as another program's share would be
    out.usable = card_gb - others_gb
    try:
        with simulated_card(card_gb, others_gb, capped, expandable) as m:
            for st in plan.stages:
                mod = _STAGE_MODULES[os.path.basename(st.cmd[1])]
                import gc
                gc.collect()
                m.empty_cache()                        # a new process: the last stage's cache is not held
                m.reset_peaks()
                m.top, m.top_reserved = m.now, m.reserved
                old_argv = sys.argv
                sys.argv = [st.cmd[1]] + list(st.cmd[2:])
                try:
                    importlib.import_module(mod).main()
                except BaseException as e:             # noqa: BLE001 - reported, the check decides
                    out.error = e
                finally:
                    sys.argv = old_argv
                    out.stages[st.name] = (m.top / GB, m.top_reserved / GB)
                if out.error is not None:
                    break
            out.ooms, out.stand_ins = m.ooms, dict(m.stand_ins)
    finally:
        root.removeHandler(log)
        root.setLevel(old_level)
    return out
