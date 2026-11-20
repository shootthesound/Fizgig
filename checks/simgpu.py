"""A simulated GPU on the CPU, for testing block swap without a card.

Inside `simulated_gpu()`:
- `torch.cuda` says a card is there (streams, events, synchronize, memory figures are stand-ins), and a tensor sent
  to "cuda" stays in CPU memory but is marked as on the card: `.device` / `.is_cuda` answer "cuda:0".
- Every operation (forward and backward, through a dispatch mode) checks its inputs: one on the card and one on the
  CPU raises the error a real GPU raises ("Expected all tensors to be on the same device ... mat2 is on cpu") - the
  class of bug block swap can cause (#335, #302).
- The block offloader's mover thread runs inline, so every move happens under the simulation and in program order.

What it measures: which blocks' weights are on the card (`blocks_on_card`) and the card bytes the marked weights hold.
It doesn't model timing, overlap or races between the mover and the compute - only where weights are when they are
used, which is what residency bugs are about.
"""
import concurrent.futures
import contextlib
from unittest import mock

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

CARD = torch.device("cuda", 0)
_on = set()                                    # storage addresses marked as on the card


class SimDeviceMismatch(RuntimeError):
    """What a real card raises when an operation mixes card and CPU tensors."""


def _ptr(t):
    with torch._C.DisableTorchFunction():
        try:
            return t.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError):
            return None


def on_card(t):
    return _ptr(t) in _on


def _mark(t, card):
    p = _ptr(t)
    if p is None or p == 0:
        return
    if card:
        _on.add(p)
    else:
        _on.discard(p)


def _tensors(x):
    return [t for t in tree_flatten(x)[0] if isinstance(t, torch.Tensor)]


def _device_of(args, kwargs):
    """The device a .to() call asks for, or None (a dtype-only call)."""
    for a in list(args[1:]) + [kwargs.get("device")]:
        if isinstance(a, (str, torch.device)):
            return torch.device(a)
        if isinstance(a, torch.Tensor):
            return CARD if on_card(a) else torch.device("cpu")
    return None


_REAL = {k: getattr(torch._C.TensorBase, k) for k in ("to", "cuda", "cpu", "clone")}
_REAL_DEVICE = torch._C.TensorBase.device.__get__


def _device(self):
    real = _REAL_DEVICE(self)
    return CARD if real.type == "cpu" and on_card(self) else real


def _is_cuda(self):
    return _device(self).type == "cuda"


def _move(self, want, dtype=None):
    out = _REAL["to"](self, dtype=dtype) if dtype is not None else self
    if want != on_card(self) and out is self:
        out = _REAL["clone"](self)            # a move is a copy, as on a real card
    _mark(out, want)
    return out


def _to(self, *args, **kwargs):
    dev = _device_of((self,) + args, kwargs)
    if dev is None:
        return _REAL["to"](self, *args, **kwargs)
    dtype = kwargs.get("dtype") or next((a for a in args if isinstance(a, torch.dtype)), None)
    return _move(self, dev.type == "cuda", dtype)


def _cuda(self, *a, **k):
    return _move(self, True)


def _cpu(self, *a, **k):
    return _move(self, False)


class _Fn(TorchFunctionMode):
    """Factories asked for the card (torch.zeros(..., device="cuda")), and what is made from nothing (on the CPU).
    Device queries and moves are patched on the tensor class itself (simulated_gpu), so they hold in the backward
    pass too, where function modes don't reach."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dev = kwargs.get("device")
        if dev is not None and not (args and isinstance(args[0], torch.Tensor)
                                    and getattr(func, "__name__", "") in ("to",)):
            want = torch.device(dev).type == "cuda"
            kw = dict(kwargs, device="cpu")
            if "pin_memory" in kw:
                kw["pin_memory"] = False
            out = func(*args, **kw)
            for t in _tensors(out):
                _mark(t, want)
            return out
        out = func(*args, **kwargs)
        if not _tensors((args, kwargs)):           # made from nothing (torch.randn(...)): on the CPU
            for t in _tensors(out):
                _mark(t, False)
        return out


_MOVES = ("aten.copy_", "aten._to_copy", "aten.set_", "aten.detach", "aten.alias", "aten.lift_fresh")


class _Ops(TorchDispatchMode):
    """Every operation: its inputs on one device (scalars may be anywhere, as on a real card); its outputs follow."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        ins = _tensors((args, kwargs))
        big = [t for t in ins if t.dim() > 0]
        where = {on_card(t) for t in big}
        if len(where) > 1 and not name.startswith(_MOVES):
            cpu = next((t for t in big if not on_card(t)), None)
            raise SimDeviceMismatch(f"Expected all tensors to be on the same device, but {name} got one on cpu "
                                    f"(shape {tuple(cpu.shape) if cpu is not None else '?'}) and others on cuda:0")
        out = func(*args, **kwargs)
        if name.startswith(("aten.copy_", "aten.set_")):
            return out                             # the destination keeps its place
        card = (True in where) if big else any(on_card(t) for t in ins)
        if ins:
            for t in _tensors(out):
                _mark(t, card)
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


class _Event:
    def __init__(self, *a, **k):
        pass

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True


class _Inline(concurrent.futures.Executor):
    """The offloader's mover, run at submit time on this thread (the modes are per thread)."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:                 # noqa: BLE001 - surfaces at .result(), as a thread's would
            f.set_exception(e)
        return f

    def shutdown(self, wait=True, **k):
        pass


def card_bytes(*modules):
    """Bytes of the modules' parameters and buffers marked as on the card."""
    seen, n = set(), 0
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            p = _ptr(t)
            if p in _on and p not in seen:
                seen.add(p)
                n += t.untyped_storage().nbytes()
    return n


def blocks_on_card(blocks):
    """For each block: True when every weight it holds is on the card, False when none is, None when split."""
    out = []
    for b in blocks:
        w = [on_card(p) for p in b.parameters()]
        out.append(True if w and all(w) else False if not any(w) else None)
    return out


@contextlib.contextmanager
def simulated_gpu(card_gb=24.0):
    """The simulated card (see the module docstring). card_gb: what torch.cuda reports as the card's size."""
    total = int(card_gb * 2**30)
    _on.clear()
    patches = [
        mock.patch.object(torch.cuda, "is_available", lambda: True),
        mock.patch.object(torch.cuda, "device_count", lambda: 1),
        mock.patch.object(torch.cuda, "current_device", lambda: 0),
        mock.patch.object(torch.cuda, "get_device_name", lambda *a: "Simulated GPU"),
        mock.patch.object(torch.cuda, "Stream", _Stream),
        mock.patch.object(torch.cuda, "Event", _Event),
        mock.patch.object(torch.cuda, "current_stream", lambda *a: _Stream()),
        mock.patch.object(torch.cuda, "stream", lambda s=None: contextlib.nullcontext()),
        mock.patch.object(torch.cuda, "synchronize", lambda *a: None),
        mock.patch.object(torch.cuda, "empty_cache", lambda: None),
        mock.patch.object(torch.cuda, "memory_allocated", lambda *a: 0),
        mock.patch.object(torch.cuda, "mem_get_info", lambda *a: (total, total)),
        mock.patch("fizgig.modules.offloading.ThreadPoolExecutor", _Inline),
        # where a tensor is, and moving it: on the tensor class, so the backward pass sees it too
        mock.patch.object(torch.Tensor, "device", property(_device)),
        mock.patch.object(torch.Tensor, "is_cuda", property(_is_cuda)),
        mock.patch.object(torch.Tensor, "to", _to),
        mock.patch.object(torch.Tensor, "cuda", _cuda),
        mock.patch.object(torch.Tensor, "cpu", _cpu),
        mock.patch.object(torch.Tensor, "pin_memory", lambda self, *a, **k: self),
        mock.patch.object(torch.Tensor, "is_pinned", lambda self, *a, **k: False),
        # checkpointing keeps the card's random-number state for its recompute: the simulated card has none (the
        # CPU's is kept as usual)
        mock.patch("torch.utils.checkpoint.get_device_states", lambda *a: ([], [])),
        mock.patch("torch.utils.checkpoint.set_device_states", lambda *a, **k: None),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(_Fn())
        stack.enter_context(_Ops())
        try:
            yield
        finally:
            _on.clear()
