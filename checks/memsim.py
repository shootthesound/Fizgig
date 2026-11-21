"""Simulated GPU memory: a real Fizgig run's card memory, measured on the CPU.

`run_launch(family, values, ...)` starts a run as the page does (form.inputs -> launch.plan -> each stage: caching, then
training with its previews, in this process), on full-size models made of fake tensors (FakeTensorMode: shapes and
dtypes, no data), on a simulated card. Every card allocation goes through a model of PyTorch's caching allocator, so
the run reports the card memory a real one allocates and reserves - and, on a card of a given size, runs out of memory
where a real one would (after giving its cache back, as PyTorch does). A run takes seconds.

Model files are their safetensors headers only (memsim_headers/: the tensor list, a few KB gzipped). Each is written
as a header-only file: Fizgig's own header readers read it as usual, and a tensor read gives a fake tensor.

Values: a fake tensor has none. That is right for weights and activations, but Fizgig computes shapes from small
tensors (position ids, indices, a caption's mask), so every tensor made from known values (a factory up to SHADOW_MAX
elements, a real tensor such as a cache file's, other such tensors - up to SHADOW_MAX_DATA) keeps a real CPU copy
computed alongside. Where a value is needed and unknown (a model's
numbers): a fixed stand-in - a scalar reads 0.5 / 1 / True, NaN and inf checks find none.

Checked against a 5090 (Windows, Klein 9B, the page's launch, each stage): the training peak 12.6 / 14.3 / 8.7 GB
(fp8 file / INT8 / NF4) vs 12.5 / 14.1 / 8.8 measured; previews' render and decode peaks within 0.3 GB; on the card
capped to 14.7 GB (a 16 GB card under Windows) training 9.1 vs 9.2 GB and the preview decode 11.9 vs 12.2 GB.
Checked against an RTX PRO 4500 32 GB (Linux, expandable segments; 20 Nov 2026): every family, each precision, 45
points, each stage's peak, the previews' and the trainer's epoch figures - after modelling cuDNN's convolution workspace
(conv_workspace), the norms' input copy and expandable segments' holes, from 0.1 GB under to 0.5 GB over.

Needs a torch built with CUDA and an NVIDIA card present: torch's method bindings set a CUDA device guard on a
fake card tensor, so CUDA starts (a context on the card) - nothing runs or is allocated there (see REAL_CAP). On the
CPU-only build some fake card tensors can't be made at all."""
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
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

GB = 1 << 30
MB = 1 << 20

# The real card: torch's own method bindings (indexing, .to, copy_ ...) set a CUDA device guard on a fake card tensor's
# device, so a CUDA device must exist and CUDA starts (a context on the card, nothing else). Nothing runs or stays
# there: Triton launches are refused (a kernel launched from Python takes a fake tensor's address - an illegal memory
# access), Fizgig then takes its eager path as it does wherever the kernel fails; the process may allocate almost
# nothing on the real card (REAL_CAP), and run_launch checks it allocated nothing.
CARD_PRESENT = False
try:
    CARD_PRESENT = torch.backends.cuda.is_built() and torch.cuda.is_available()
except Exception:                              # noqa: BLE001 - no driver
    pass
REAL_CAP = 0.0005                              # of the real card: any real allocation past a few MB fails, loudly
_real_cuda = {n: getattr(torch.cuda, n) for n in ("max_memory_allocated", "set_per_process_memory_fraction",
                                                   "reset_peak_memory_stats")}

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
CAN_RUN = CARD_PRESENT      # a torch built with CUDA, and an NVIDIA card for the context (nothing runs on it)

# ---- the caching allocator (c10/cuda/CUDACachingAllocator.cpp) -------------------------------------------------------
# sizes rounded to 512 B; requests up to 1 MB from a small pool of 2 MB segments, larger ones from a large pool (a new
# segment is 20 MB for requests under 10 MB, else the request rounded up to 2 MB); best fit among cached free blocks; a
# block split when the rest is big enough (over 1 MB in the large pool, 512 B in the small one); freed blocks merge with
# free neighbours. Expandable segments (Linux): one segment per pool that grows at its end, in 2 MB pages. One stream.
SMALL, SMALL_BUFFER, LARGE_BUFFER, MIN_LARGE_ALLOC, ROUND_LARGE, PAGE = MB, 2 * MB, 20 * MB, 10 * MB, 2 * MB, 2 * MB


def _round(n):
    return 512 if n < 512 else (n + 511) // 512 * 512


class _Block:
    __slots__ = ("seg", "off", "size", "free", "prev", "next", "mapped")

    def __init__(self, seg, off, size, mapped=True):
        self.seg, self.off, self.size, self.free, self.prev, self.next = seg, off, size, True, None, None
        self.mapped = mapped                   # expandable: False for a hole empty_cache unmapped (free, no memory)


class _Pool:
    """One pool. Expandable (PyTorch's expandable segments): one segment, its blocks a list by address; empty_cache
    unmaps every whole free page anywhere in it (a hole: free address space, no memory); a request no mapped free block
    fits takes the lowest hole whose free run - with the free block just before it - is big enough, mapping the pages it
    needs, else grows the segment at its end (try_allocate_expandable_block / unmap_block in CUDACachingAllocator.cpp).
    Blocks only merge with neighbours in the same state."""

    def __init__(self, small, expandable, room):
        self.small, self.expandable, self.room = small, expandable, room
        self.free = []                         # mapped free blocks, sorted (size, seg, off, id, block)
        self.holes = []                        # expandable: unmapped blocks, sorted (off, id, block)
        self.nseg = 0
        self.tail = None                       # expandable: the segment's last block
        self.end = 0                           # expandable: the segment's used address space (mapped or not)

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

    # ---- expandable: holes ----
    def _hole_key(self, b):
        return b.off, id(b)

    def _add_hole(self, b):
        bisect.insort(self.holes, (*self._hole_key(b), b))

    def _drop_hole(self, b):
        i = bisect.bisect_left(self.holes, self._hole_key(b), key=lambda e: e[:2])
        assert self.holes[i][2] is b
        del self.holes[i]

    def _unlink(self, b):
        if b.prev is not None:
            b.prev.next = b.next
        if b.next is not None:
            b.next.prev = b.prev
        if self.tail is b:
            self.tail = b.prev

    def _split_at(self, b, at):
        """b becomes [b.off, at); -> the new block [at, old end), linked after it, same state."""
        r = _Block(b.seg, at, b.off + b.size - at, b.mapped)
        r.prev, r.next = b, b.next
        if b.next is not None:
            b.next.prev = r
        b.next, b.size = r, at - b.off
        if self.tail is b:
            self.tail = r
        return r

    def _span(self, size):
        """Where a request no mapped free block fits goes: (first block of the run or None for the end, pages to map).
        The lowest hole whose free run (from the free mapped block just before it) holds the request, else the end."""
        def run(c):
            got, need, x = 0, 0, c
            while x is not None and x.free and got < size:
                take = x.size if x.mapped else min(x.size, (size - got + PAGE - 1) // PAGE * PAGE)
                need += 0 if x.mapped else take
                got += take
                x = x.next
            if got < size and x is None:       # the run reaches the segment's end: it grows there
                more = (size - got + PAGE - 1) // PAGE * PAGE
                return got + more, need + more
            return got, need
        for _, _, h in self.holes:
            c = h.prev if h.prev is not None and h.prev.free and h.prev.mapped else h
            got, need = run(c)
            if got >= size:
                return c, need
        t = self.tail
        c = t if t is not None and t.free and t.mapped else None
        return c, run(c)[1] if c is not None else (size + PAGE - 1) // PAGE * PAGE

    def _take_span(self, c, size):
        """The run from c (None: the end) as one mapped block of at least size. -> (block, newly mapped bytes)"""
        grown = 0
        if c is None:
            t = self.tail
            c = _Block(0, self.end, 0)
            c.prev, self.tail = t, c
            if t is not None:
                t.next = c
        elif c.mapped:
            self._drop_free(c)
        else:
            self._drop_hole(c)
            m = min(c.size, (size + PAGE - 1) // PAGE * PAGE)
            grown += m
            if m < c.size:
                self._add_hole(self._split_at(c, c.off + m))
            c.mapped = True
        while c.size < size:
            x = c.next
            if x is None:                      # the end: grow the segment
                more = (size - c.size + PAGE - 1) // PAGE * PAGE
                c.size += more
                self.end += more
                grown += more
                break
            if x.mapped:
                self._drop_free(x)
                take = x.size
            else:
                self._drop_hole(x)
                take = min(x.size, (size - c.size + PAGE - 1) // PAGE * PAGE)
                grown += take
                if take < x.size:
                    self._add_hole(self._split_at(x, x.off + take))
            c.size += x.size if x.mapped else take
            self._unlink(x)
        x = c.next                             # a free mapped block after it joins (map_block's merge)
        if x is not None and x.free and x.mapped:
            self._drop_free(x)
            c.size += x.size
            self._unlink(x)
        return c, grown

    def alloc(self, size):
        """-> (block, newly reserved bytes)"""
        i = bisect.bisect_left(self.free, (size,), key=lambda e: e[:1])
        if i >= len(self.free):                # nothing cached fits: the card must give more (or the cache back)
            self.room(self._span(size)[1] if self.expandable else self._segment(size))
            i = bisect.bisect_left(self.free, (size,), key=lambda e: e[:1])
        grown = 0
        if i < len(self.free):
            b = self.free[i][4]
            self._drop_free(b)
        elif self.expandable:
            b, grown = self._take_span(self._span(size)[0], size)
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
        """torch.cuda.empty_cache: whole free segments go back (expandable: every whole free page, wherever it is - each
        becomes a hole). -> bytes"""
        back = 0
        if self.expandable:
            for e in list(self.free):
                b = e[4]
                lo, hi = (b.off + PAGE - 1) // PAGE * PAGE, (b.off + b.size) // PAGE * PAGE
                if hi <= lo:
                    continue
                self._drop_free(b)
                back += hi - lo
                if hi < b.off + b.size:        # the part page after the hole stays mapped and free
                    self._add_free(self._split_at(b, hi))
                if lo > b.off:                 # and the one before it
                    head, b = b, self._split_at(b, lo)
                    self._add_free(head)
                b.mapped = False
                for nb in (b.prev, b.next):    # holes side by side are one hole
                    if nb is not None and nb.free and not nb.mapped:
                        self._drop_hole(nb)
                        if nb is b.prev:
                            nb.size += b.size
                            self._unlink(b)
                            b = nb
                        else:
                            b.size += nb.size
                            self._unlink(nb)
                self._add_hole(b)
            t = self.tail
            if t is not None and not t.mapped:     # a hole at the end: the segment ends before it
                self._drop_hole(t)
                self.end = t.off
                self._unlink(t)
            return back
        for e in [e for e in self.free if e[4].prev is None and e[4].next is None]:
            self._drop_free(e[4])
            back += e[4].size
        return back

    def release(self, b):
        b.free = True
        for nb in (b.prev, b.next):
            if nb is not None and nb.free and nb.mapped:
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

    def transient(self, nbytes):
        """Card memory an op holds only while it runs (cuDNN's workspace): counted in the peaks, then given back. Where
        it doesn't fit, PyTorch's cuDNN path takes an algorithm that needs less instead of failing - so none here."""
        if nbytes <= 0:
            return
        n = _round(nbytes)
        small = n <= SMALL
        try:
            blk, grown = self.pools[small].alloc(n)
        except torch.OutOfMemoryError:
            self.ooms -= 1
            return
        self.reserved += grown
        self.peak, self.top = max(self.peak, self.now + n), max(self.top, self.now + n)
        self.peak_reserved, self.top_reserved = max(self.peak_reserved, self.reserved), max(self.top_reserved,
                                                                                            self.reserved)
        self.pools[small].release(blk)

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if func in _CONTIGUOUS_IN and isinstance(args[0], torch.Tensor) and args[0].device.type == "cuda" \
                and not args[0].is_contiguous():
            # CUDA's kernel copies its input to the standard layout first, and its output is in that layout (the fake
            # kernel keeps the input's - after attention's channels-last view, every op after a VAE's GroupNorm stayed
            # channels-last, and its convolutions took the wrong workspace)
            x = torch.ops.aten.clone.default(args[0], memory_format=torch.contiguous_format)
            self.add(x)
            args = (x, *args[1:])
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            self.add(t)
        if func is _CONV:
            self.transient(conv_workspace(args, out))
        return out


_CONV = torch.ops.aten.convolution.default
_CONTIGUOUS_IN = {torch.ops.aten.native_group_norm.default, torch.ops.aten.native_layer_norm.default}


def conv_workspace(args, out):
    """cuDNN's workspace for a convolution on the card, bytes. Measured per op on an RTX PRO 4500 (Blackwell, Linux,
    PyTorch 2.15, cuDNN's default heuristics; 20 Nov 2026: the VAE decodes of SDXL and Z-Image in fp32, SDXL's UNet in
    bf16, the Qwen-Image VAE's 2D and 3D convolutions in bf16, encode and decode). A 1x1 kernel takes none. Over 1x1:
    fp32 takes a copy of its input and weight, whatever the input's layout, and of its output when a channel count isn't
    a multiple of 8 (exact on the VAEs). 16-bit with its input in the standard layout: input, weight and output (on
    SDXL's UNet 11 of 17 shapes exact, the rest 10-40 MB more); channels-last (a picture permuted from height x width x
    channels): its weight only. Any other layout of a 16-bit input: a copy of the input first (a 1x1 kernel too). The
    decode's 256-channel convolution at 1024 x 1024 in fp32: 1026 MB, where a real preview peaks."""
    x, w = args[0], args[1]
    if getattr(getattr(x, "device", None), "type", None) != "cuda" or w.dim() < 3:
        return 0
    one = all(s == 1 for s in w.shape[2:])
    xb, wb, ob = x.numel() * x.element_size(), w.numel() * w.element_size(), out.numel() * out.element_size()
    transposed = len(args) > 6 and bool(args[6])
    cin, cout = (w.shape[0], w.shape[1]) if transposed else (w.shape[1] * (args[8] if len(args) > 8 else 1),
                                                              w.shape[0])
    if x.element_size() != 2:
        return 0 if one else xb + wb + (ob if cin % 8 or cout % 8 else 0)
    if x.is_contiguous():
        return 0 if one else xb + wb + ob
    last = torch.channels_last if x.dim() == 4 else torch.channels_last_3d if x.dim() == 5 else None
    if last is not None and x.is_contiguous(memory_format=last):
        return 0 if one else wb
    return xb + (0 if one else xb + wb + ob)


# ---- values: shadows for the small tensors shapes come from, stand-ins elsewhere --------------------------------------
SHADOW_MAX = 1 << 20            # elements, for a tensor made from nothing (a factory: torch.empty of a weight stays fake)
SHADOW_MAX_DATA = 1 << 24       # for one computed from known values (a cached caption's embedding, its mask)
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
        self.stand_ins, self.stand_in_where = {}, {}

    fake_mode = None                           # set by simulated_card

    def _fakeify(self, t):
        if isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor):
            f = self.fake_mode.from_tensor(t)
            f.constant = None
            if t.device.type == "cpu" and t.numel() <= SHADOW_MAX_DATA and f not in self.shadow:
                self.shadow[f] = t
            return f
        return t

    def _sh(self, t):
        if isinstance(t, FakeTensor):
            return self.shadow.get(t)
        return t if t.device.type == "cpu" else None      # a real tensor is its own value (never one on a real card)

    def _on_shadows(self, func, args, kwargs):
        flat = tree_flatten((args, kwargs))[0]
        if any(isinstance(x, torch.Generator) and x.device.type != "cpu" for x in flat) or any(
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
        if self.fake_mode is not None and any(isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor)
                                              for t in tree_flatten((args, kwargs))[0]):
            # a real input (a cache file's tensor, a scalar made from data): a fake before the fake mode sees it - with
            # real inputs it propagates constants, running the op for real (x.to("cuda") on the real card)
            args, kwargs = tree_map(self._fakeify, (args, kwargs))
        try:
            out = super().__torch_dispatch__(func, types_, args, kwargs)
        except (DynamicOutputShapeException, DataDependentOutputException):
            return self._value_shaped(func, args, kwargs)
        for o in tree_flatten(out)[0]:             # nor a fake's constant (the same propagation): kept as its values
            if isinstance(o, FakeTensor) and o.constant is not None:
                c = o.constant
                o.constant = None
                if c.device.type == "cpu" and o not in self.shadow:
                    self.shadow[o] = c
        flat = tree_flatten((args, kwargs))[0]
        ins = [t for t in flat if isinstance(t, torch.Tensor)]
        if any(isinstance(t, FakeTensor) and t not in self.shadow for t in ins):
            # an input without values (a model's numbers - most ops): none for the output either. In place, the
            # written tensor's values are gone
            schema = getattr(func, "_schema", None)
            if schema is not None and schema.is_mutable:
                for o in tree_flatten(out)[0]:
                    if isinstance(o, FakeTensor) and o in self.shadow:
                        del self.shadow[o]
            return out
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        limit = SHADOW_MAX_DATA if ins else SHADOW_MAX
        if outs and all(o.numel() <= limit for o in outs):
            sout = self._on_shadows(func, args, kwargs)
            if sout is not None:
                for o, s in zip(outs, [t for t in tree_flatten(sout)[0] if isinstance(t, torch.Tensor)]):
                    if isinstance(o, FakeTensor):
                        self.shadow[o] = s
                return out
        for o in outs:                         # written without values (in place, or too big): its values are gone
            if isinstance(o, FakeTensor) and o in self.shadow:
                del self.shadow[o]
        return out

    def _value_shaped(self, func, args, kwargs):
        """An output whose shape depends on values (unique, nonzero): computed on the shadows, else on zeros."""
        dev = next((t.device for t in tree_flatten((args, kwargs))[0] if isinstance(t, FakeTensor)), None)
        sout = self._on_shadows(func, args, kwargs)
        exact = sout is not None
        if sout is None:
            # each input's values where known, zeros where not: x[mask] with a known mask and unknown x has the right
            # shape. Counted as a stand-in only when what decides the shape is unknown (for indexing, the indices)
            decides = tree_flatten((args[1:], kwargs))[0] if "index" in str(func) else tree_flatten((args, kwargs))[0]
            if any(isinstance(t, FakeTensor) and t not in self.shadow for t in decides):
                key = str(func)
                if key not in self.stand_ins:          # where, for the report: the innermost Fizgig lines
                    import traceback
                    self.stand_in_where[key] = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                                                for f in traceback.extract_stack() if "fizgig" in f.filename][-3:]
                self.stand_ins[key] = self.stand_ins.get(key, 0) + 1

            def known(t):
                # unknown: a mask all True (everything kept - reads high, never low: Fibo 2's turbo LoRA keeps every
                # column of its fused parts, as the real file does), numbers 0 (a valid index anywhere)
                # unique's numbers all distinct (its longest result: indices into it stay in range)
                if isinstance(t, FakeTensor):
                    s = self.shadow.get(t)
                    with unset_fake_temporarily():
                        if s is not None:
                            return s
                        if t.dtype == torch.bool:
                            return torch.ones(t.shape, dtype=t.dtype)
                        if "unique" in str(func):
                            return torch.arange(t.numel()).reshape(t.shape).to(t.dtype)
                        return torch.zeros(t.shape, dtype=t.dtype)
                return _cpu_arg(t)
            with unset_fake_temporarily():
                sout = func(*tree_map(known, args), **tree_map(known, kwargs))

        def fake(s):
            if isinstance(s, torch.Tensor):
                f = torch.empty(s.shape, dtype=s.dtype, device=dev)        # the fake mode below makes it a fake
                self.add(f)
                if exact:                      # values from unknown inputs are not values
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


_SIZES = {}
_real_stat = os.stat


class _Stat:
    """A fake model file's stat: the real file's, with the size the real model file has."""

    def __init__(self, st, size):
        self._st, self.st_size = st, size

    def __getattr__(self, n):
        return getattr(self._st, n)

    def __getitem__(self, i):
        return self.st_size if i == 6 else self._st[i]


def _stat(p, *a, **k):
    st = _real_stat(p, *a, **k)
    if isinstance(p, (str, os.PathLike)):
        size = _SIZES.get(_norm(p))
        if size is not None and is_fake(p):
            return _Stat(st, size)
    return st


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


_VALUES = {}                                   # path -> {tensor name: bytes}: the few tensors whose values are read


def write_model_file(header, path):
    """A header-only safetensors file, registered as fake: real to every header reader, fake to tensor reads. A
    fixture's __memsim_data__ (the bytes of small tensors Fizgig reads as values: ConvRot's comfy_quant markers) is
    kept aside, and those tensors read as their real values."""
    header = dict(header)
    data = header.pop("__memsim_data__", None) or {}
    raw = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
    _FAKE.add(_norm(path))
    # the size the real file has (header + every tensor's bytes): Fizgig sizes models from their files (a preview
    # checkpoint's bf16-or-int8 choice) - a header-only file read as a few KB picked bf16 with nothing streamed
    _SIZES[_norm(path)] = 8 + len(raw) + max((v["data_offsets"][1] for k, v in header.items()
                                              if k != "__metadata__" and isinstance(v, dict) and "data_offsets" in v),
                                             default=0)
    import base64
    _VALUES[_norm(path)] = {k: base64.b64decode(v) for k, v in data.items()}
    return str(path)


def _tensors(path):
    h = read_header(path)
    return {k: v for k, v in h.items() if k != "__metadata__"}, h.get("__metadata__", {}) or {}


def _empty(info, device=None, dtype=None, path=None, name=None):
    raw = _VALUES.get(_norm(path), {}).get(name) if path is not None else None
    if raw is not None:                        # a value Fizgig reads: the real one (a CPU tensor, moved as asked)
        with _really():
            buf = bytearray(raw)               # frombuffer doesn't hold the buffer: kept until the copy is made
            t = torch.frombuffer(buf, dtype=_DT[info["dtype"]]).reshape(info["shape"]).clone()
            del buf
            if dtype is not None:
                t = t.to(dtype)
        return t if device is None else t.to(device)
    return torch.empty(info["shape"], dtype=dtype or _DT[info["dtype"]], device=device)


def _dev(device):
    return None if device is None or str(device) == "cpu" else device


class _FakeSafeOpen:
    """safetensors.safe_open on a header-only file."""

    def __init__(self, path, framework="pt", device="cpu"):
        self.path = path
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
        return _empty(self.t[k], self.device, path=self.path, name=k)

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


@contextlib.contextmanager
def _really():
    """Outside the simulation, for real tensors (a real file's read, a stored value): no fake mode, no meter, no function
    modes (safetensors' own torch.asarray(storage) fails through any)."""
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes(), torch._C.DisableTorchFunction():
        yield


def _materialise(x):
    """A fake tensor as a save writes it: its values where known (a caption's attention mask, which the trainer reads
    back for the sequence length), else zeros - a mask all True (every token counted: reads high, never low)."""
    if isinstance(x, FakeTensor):
        s = CURRENT.shadow.get(x) if CURRENT is not None else None
        with _really():
            if s is not None:
                return s.clone()
            return torch.ones(x.shape, dtype=x.dtype) if x.dtype == torch.bool else torch.zeros(x.shape, dtype=x.dtype)
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
    with _really():
        return s if s is not None else torch.zeros(t.shape, dtype=t.dtype)


def _numpy(self, *a, **k):
    """A fake tensor's values: its shadow, else zeros (a preview picture). A real tensor's with every mode off (inside
    them numpy() reads a fake alias's memory)."""
    v = _values(self) if isinstance(self, FakeTensor) else self
    with _really():
        return _real_numpy(v.float() if isinstance(self, FakeTensor) and v.dtype == torch.bfloat16 else v, *a, **k)


def _tolist(self):
    v = _values(self) if isinstance(self, FakeTensor) else self
    with _really():
        return _real_tolist(v)


class _DataOnCpu(TorchFunctionMode):
    """torch.tensor(data, device="cuda") makes a real tensor on the card before the fake mode sees it: made on the CPU
    instead, then moved to the simulated card (a fake, its values kept)."""
    FUNCS = {torch.tensor, torch.as_tensor, torch.asarray}

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        dev = kwargs.get("device")
        if func in self.FUNCS and dev is not None and torch.device(dev).type == "cuda":
            return func(*args, **dict(kwargs, device="cpu")).to(dev)
        return func(*args, **kwargs)


def _bnb_params_patch():
    """bitsandbytes' Params4bit / Int8Params wrap their data with torch.Tensor._make_subclass, which a fake tensor refuses
    (its object is already a FakeTensor). Around a fake, the fake itself becomes the parameter: a class that is both
    (FakeTensor first, so the fake mode still runs every op), with bitsandbytes' own methods - the quantising .to(cuda),
    quant_state, its __torch_function__ - unchanged; its ops have fake kernels, so their card memory is counted as usual.
    -> [(class, "__new__", replacement)] for the patches, or [] without bitsandbytes."""
    try:
        from bitsandbytes.nn import Int8Params, Params4bit
    except Exception:                          # noqa: BLE001 - not installed, or its library doesn't load
        return []
    made, out = {}, []

    def fake_class(base):
        if base not in made:
            made[base] = type("Fake" + base.__name__, (FakeTensor, base), {"__new__": base.__new__})
        return made[base]

    for base in (Params4bit, Int8Params):
        real_new = base.__new__

        def new(cls, data=None, requires_grad=False, *a, _base=base, _real=real_new, **k):
            if not isinstance(data, FakeTensor):
                return _real(cls, data, requires_grad, *a, **k)
            self = data.detach()
            self.__class__ = fake_class(_base)
            self.requires_grad_(bool(requires_grad))
            # the base's own attributes from its __init__-style __new__, called on a throwaway real tensor
            with _really():
                ref = _real(_base, torch.empty(0), False, *a, **k)
            for n, v in vars(ref).items():
                if n != "data":
                    setattr(self, n, v)
            return self
        out.append((base, "__new__", new))
    return out


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
        # weight initialisation: values only, and trunc_normal_ resamples until no value is out of bounds - with no
        # values to check, forever
        *[(torch.nn.init, n, (lambda tensor, *a, **k: tensor)) for n in dir(torch.nn.init)
          if n.endswith("_") and not n.startswith("_") and callable(getattr(torch.nn.init, n))],
        (torch.Tensor, "data", property(_DATA.__get__, _set_data)),
        (torch.Tensor, "numpy", _numpy), (torch.Tensor, "tolist", _tolist),
        (torch.Tensor, "pin_memory", lambda self, *a, **k: self),
        (torch.Tensor, "is_pinned", lambda self, *a, **k: False),
        *_bnb_params_patch(),
        (os, "stat", _stat),                   # os.path.getsize / Path.stat of a fake model file: the real size
    ]
    import safetensors
    import safetensors.torch as st
    real_open, real_load, real_save, real_tsave = safetensors.safe_open, st.load_file, st.save_file, torch.save

    class _RealOpen:
        """safe_open on a real file (a cache file): read on the CPU outside the fake mode (which can't wrap a file's
        storage), then moved as asked."""

        def __init__(self, path, framework="pt", device="cpu"):
            with _really():
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
            with _really():
                t = self.f.get_tensor(k)
            return self._move(t)

        def get_slice(self, k):
            s, outer = self.f.get_slice(k), self

            class _S:
                def __getattr__(self, n):
                    return getattr(s, n)

                def __getitem__(self, idx):
                    with _really():
                        t = s[idx]
                    return outer._move(t)
            return _S()

    def safe_open(path, *a, **k):
        return _FakeSafeOpen(path, *a, **k) if is_fake(path) else _RealOpen(path, *a, **k)

    def load_file(path, device="cpu"):
        if is_fake(path):
            t, _ = _tensors(path)
            return {n: _empty(v, _dev(device), path=path, name=n) for n, v in t.items()}
        with _really():
            sd = real_load(path)
        return sd if _dev(device) is None else {n: t.to(device) for n, t in sd.items()}

    def save_file(tensors, path, *a, **k):
        return real_save({n: _materialise(t) for n, t in tensors.items()}, path, *a, **k)

    def tsave(obj, *a, **k):
        return real_tsave(tree_map(_materialise, obj), *a, **k)

    real_tload = torch.load

    def tload(*a, **k):
        with _really():                        # a real file (ACE-Step's silence latent, a saved state)
            return real_tload(*a, **k)

    reals = {real_open: safe_open, real_load: load_file, real_save: save_file, real_tsave: tsave}
    try:                                       # diffusers' and transformers' loaders place weights through accelerate
        import accelerate.utils.modeling as _acc
        real_place = _acc.set_module_tensor_to_device

        def set_module_tensor_to_device(module, tensor_name, device, value=None, dtype=None, *a, **k):
            """accelerate rebuilds a parameter as type(old)(value): a fake parameter's type is FakeTensor, which
            can't be built so - placed directly instead, as accelerate would leave it."""
            try:
                return real_place(module, tensor_name, device, value, dtype, *a, **k)
            except TypeError as e:
                if "fake_mode" not in str(e):
                    raise
            for part in tensor_name.split(".")[:-1]:
                module = getattr(module, part)
            name = tensor_name.split(".")[-1]
            old = getattr(module, name)
            v = old if value is None else value
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
            elif dtype is None and old is not None and old.is_floating_point() and v.is_floating_point():
                v = v.to(old.dtype)
            v = v.to(device)
            if name in module._parameters:
                module._parameters[name] = torch.nn.Parameter(v, requires_grad=bool(getattr(old, "requires_grad", False)))
            else:
                module._buffers[name] = v
        reals[real_place] = set_module_tensor_to_device
    except ImportError:
        pass

    def reader_patch(cls):
        """Fizgig's own header readers: a tensor read on a fake file gives a fake; on a real one, read on the CPU."""
        real_get = cls.get_tensor

        def get_tensor(self, key, device=None, dtype=None, *a, **k):
            fn = getattr(self, "filename", None) or getattr(self, "path", None)
            if not isinstance(fn, (str, os.PathLike)):     # a reader over another reader (key renaming): as it is
                return real_get(self, key, device, dtype, *a, **k)
            if is_fake(fn):
                return _empty(self.header[key], _dev(device), dtype, path=fn, name=key)
            with _really():
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
    if CARD_PRESENT:                           # the real card: a context only, no memory (see REAL_CAP)
        _real_cuda["set_per_process_memory_fraction"](REAL_CAP)
        _real_cuda["reset_peak_memory_stats"]()   # run_launch checks this run's real peak

    def no_triton(*a, **k):
        raise RuntimeError("Triton kernels don't run on the simulated card (they would take a fake tensor's address)")
    try:
        import triton.runtime.jit as _tj
        patches.append((_tj.JITFunction, "run", no_triton))
    except ImportError:
        pass
    with contextlib.ExitStack() as stack:
        for obj, name, val in patches:
            stack.enter_context(mock.patch.object(obj, name, val))
        for cls in readers:
            stack.enter_context(mock.patch.object(cls, "get_tensor", reader_patch(cls)))
        stack.enter_context(mock.patch.object(torch, "save", tsave))
        stack.enter_context(mock.patch.object(torch, "load", tload))
        # a module imported after this point takes the inline one by name too (Krea 2's offloader loads with the DiT:
        # its mover ran on a real thread, on the same fake mode - "in kernel" from the other thread, half-moved weights)
        stack.enter_context(mock.patch.object(concurrent.futures, "ThreadPoolExecutor", _Inline))
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith(("fizgig", "safetensors", "diffusers", "accelerate",
                                                        "transformers")):
                continue
            for attr, val in list(vars(mod).items()):
                try:
                    hit = callable(val) and val in reals
                except Exception:              # noqa: BLE001 - an attribute that can't be hashed or compared
                    hit = False
                if hit:
                    stack.enter_context(mock.patch.object(mod, attr, reals[val]))
            if getattr(mod, "ThreadPoolExecutor", None) is concurrent.futures.ThreadPoolExecutor:
                stack.enter_context(mock.patch.object(mod, "ThreadPoolExecutor", _Inline))
        meter.fake_mode = stack.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
        stack.enter_context(meter)
        stack.enter_context(_DataOnCpu())
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
            msg = record.getMessage()
            if record.exc_info:
                import traceback
                msg += "\n" + "".join(traceback.format_exception(*record.exc_info))
            self.lines.append(msg)
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


class _RealCard:
    """The real card, measured the way the simulated one is (run_launch(real_models=...): a reference run): torch's
    own figures, the trainer's resets of its peaks folded into top / top_reserved first, so nothing it does erases a
    stage's peak."""

    def __init__(self):
        self.top = self.top_reserved = 0
        self.ooms, self.stand_ins, self.stand_in_where = 0, {}, {}

    @property
    def now(self):
        return torch.cuda.memory_allocated()

    @property
    def reserved(self):
        return torch.cuda.memory_reserved()

    def fold(self):
        self.top = max(self.top, torch.cuda.max_memory_allocated())
        self.top_reserved = max(self.top_reserved, torch.cuda.max_memory_reserved())

    def reset_peaks(self):
        self.fold()
        _real_cuda["reset_peak_memory_stats"]()

    def empty_cache(self):
        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def measuring(self):
        with mock.patch.object(torch.cuda, "reset_peak_memory_stats", lambda *a: self.reset_peaks()):
            yield self


def songs(folder, n=2, seconds=240, caption="an upbeat pop song with female vocals"):
    """n songs (a tone, 48 kHz stereo) with captions, as long as a song trains (ACE-Step cuts at 240 s)."""
    import wave
    import numpy as np
    os.makedirs(folder, exist_ok=True)
    t = np.arange(48000 * seconds) / 48000
    tone = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    frames = np.stack([tone, tone], 1).tobytes()
    for i in range(n):
        with wave.open(os.path.join(folder, f"s{i}.wav"), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(frames)
        with open(os.path.join(folder, f"s{i}.txt"), "w", encoding="utf-8") as f:
            f.write(caption)


_STAGE_MODULES = {"cache.py": "fizgig.families.cache", "train.py": "fizgig.families.train"}


def run_launch(family, tmp, values=None, *, megapixels=1.0, card_gb=32.0, others_gb=0.0, capped=True,
               expandable=False, with_room=True, real_models=None):
    """A run of `family` as Start launches it, with the Training form's defaults plus `values` (form keys), one epoch
    on 3 pictures at `megapixels`, every stage on one simulated card (each starting with the card as a new process
    finds it: the previous stage's cache given back). Model files come from memsim_headers/<family>_<pref_key>.
    with_room: the card's usable memory less margin() - completing then means fitting with the margin to spare.
    real_models: {pref_key: path} - the same launch on the real card with these files instead (a reference run to check
    the simulation against; card_gb / others_gb / capped / expandable are then the real card's own)

    -> SimpleNamespace(stages={name: (allocated peak, reserved peak) in GB}, peak_vram=[the trainer's own epoch
       figures, GB], preview_peaks=[each preview's reserved peak, GB], usable=GB the run had, lines=[log lines], ooms, stand_ins={op: (count, where)}, error=the
       exception that ended it, or None)"""
    from fizgig.families import form, launch
    from fizgig.families.registry import get
    for m in _STAGE_MODULES.values():
        importlib.import_module(m)                     # before the card: their imports get the patched readers
    desc = get(family)
    tmp = str(tmp)
    models = {}
    for mf in desc.model_files:
        if real_models is not None:
            if real_models.get(mf.pref_key):
                models[mf.pref_key] = real_models[mf.pref_key]
        elif os.path.exists(os.path.join(HEADERS, f"{family}_{mf.pref_key}.json.gz")):
            # the real file's name, in a folder of its own: Fizgig reads names (an int8 ConvRot file is twice its size
            # once decoded - named for its setting, MOSS's SFT preview read as half its size and loaded bf16)
            name = mf.local_name or os.path.basename(mf.path) or f"{mf.pref_key}.safetensors"
            os.makedirs(os.path.join(tmp, mf.pref_key), exist_ok=True)
            models[mf.pref_key] = write_model_file(load_header(f"{family}_{mf.pref_key}"),
                                                   os.path.join(tmp, mf.pref_key, name))
    img = os.path.join(tmp, "pictures")
    if "photo" in desc.media:
        pictures(img, side=max(512, int(round((megapixels * 1024 * 1024) ** 0.5 / 64)) * 64))
    else:                                      # a song family (ACE-Step)
        songs(img)
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
        with (_RealCard().measuring() if real_models is not None
              else simulated_card(card_gb, others_gb, capped, expandable)) as m:
            for st in plan.stages:
                mod = _STAGE_MODULES[os.path.basename(st.cmd[1])]
                import gc
                gc.collect()
                m.empty_cache()                        # a new process: the last stage's cache is not held
                m.reset_peaks()
                m.top, m.top_reserved = m.now, m.reserved
                cmd = list(st.cmd[2:])
                if "--sample_prompts" in cmd:          # previews in 2 steps: every step takes the same memory
                    if "--sample_steps" in cmd:
                        cmd[cmd.index("--sample_steps") + 1] = "2"
                    else:
                        cmd += ["--sample_steps", "2"]
                old_argv = sys.argv
                sys.argv = [st.cmd[1]] + cmd
                try:
                    importlib.import_module(mod).main()
                except BaseException as e:             # noqa: BLE001 - reported, the check decides
                    out.error = e
                    if isinstance(e, torch.OutOfMemoryError) and real_models is not None:
                        m.ooms += 1
                finally:
                    sys.argv = old_argv
                    if real_models is not None:
                        m.fold()
                    out.stages[st.name] = (m.top / GB, m.top_reserved / GB)
                if out.error is not None:
                    break
            out.ooms, out.stand_ins = m.ooms, {k: (n, m.stand_in_where.get(k)) for k, n in m.stand_ins.items()}
            import re
            out.peak_vram = [float(x) for ln in log.lines for x in re.findall(r"peak VRAM ([0-9.]+) GB", ln)]
            out.preview_peaks = [float(x) for ln in log.lines if "after preview cleanup" in ln
                                 for x in re.findall(r"\(peak ([0-9.]+) GB\)", ln)]
        if real_models is None and torch.cuda.is_initialized() and _real_cuda["max_memory_allocated"]() > 0:
            raise RuntimeError("the simulation allocated memory on the real card: something ran a real kernel")
    finally:
        root.removeHandler(log)
        root.setLevel(old_level)
    return out
