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
import sys
import weakref
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


_empty_on = {}                                 # id(tensor) -> on the card, for tensors with no memory (empty ones)


def on_card(t):
    p = _ptr(t)
    if not p:                                  # an empty tensor has no storage address: tracked by the tensor itself
        return _empty_on.get(id(t), False)
    return p in _on


def _mark(t, card):
    p = _ptr(t)
    if p is None:
        return
    if p == 0:
        if id(t) not in _empty_on:
            weakref.finalize(t, _empty_on.pop, id(t), None)
        _empty_on[id(t)] = card
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
            for t in _tensors(out):              # Python asked for this device: it's there (the CPU too)
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
        dev = kwargs.get("device")
        if isinstance(dev, torch.device) and dev.type == "cuda":
            # made on the card (torch.zeros(..., device=q.device)): made on the CPU, marked as on the card - here, not
            # in the function layer, so a checkpointed block's recompute in the backward is caught too. An explicit
            # "cpu" here is torch's own (the backward restoring the real device of a tensor, which is always the CPU
            # in the simulation): that follows the inputs, as below. Python code asking for the CPU is marked in _Fn.
            kwargs = dict(kwargs, device=torch.device("cpu"))
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            out = func(*args, **kwargs)
            for t in _tensors(out):
                _mark(t, True)
            return out
        ins = _tensors((args, kwargs))
        # a card accepts CPU index tensors in indexing (x[:, mask] = v with a CPU bool mask): those don't count
        index_op = "index" in name or name.startswith(("aten.take", "aten.gather", "aten.scatter"))
        big = [t for t in ins if t.dim() > 0 and not (index_op and not t.is_floating_point() and not t.is_complex())]
        where = {on_card(t) for t in big}
        if len(where) > 1 and not name.startswith(_MOVES):
            cpu = next((t for t in big if not on_card(t)), None)
            raise SimDeviceMismatch(f"Expected all tensors to be on the same device, but {name} got one on cpu "
                                    f"({tuple(cpu.shape) if cpu is not None else '?'}, "
                                    f"{cpu.dtype if cpu is not None else '?'}) and others on cuda:0")
        if name.split(".")[0] in ("aten", "prim", "prims") or not ins:
            out = func(*args, **kwargs)
        else:
            # a library's own op (bitsandbytes' dequantize): its Python body runs here with the simulation's modes off,
            # so a torch.empty(..., device=A.device) inside it would ask for the card - the REAL card on a machine that
            # has one, handing a CPU kernel a GPU address. So it runs with both layers back on (every factory and op
            # inside it simulated like any other code), its kernel reached by redispatching below the Python key -
            # with its inputs' own keys, as torch would - so the op itself isn't caught here again.
            keys = torch._C._dispatch_keys(ins[0])
            for t in ins[1:]:
                keys = keys | torch._C._dispatch_keys(t)
            below = keys & torch._C._dispatch_keyset_full_after(torch._C.DispatchKey.Python)
            with _Fn(), self:
                out = func.redispatch(below, *args, **kwargs)
        for t in _tensors(out):
            if _REAL_DEVICE(t).type != "cpu":
                raise RuntimeError(f"the simulated card made a real {_REAL_DEVICE(t)} tensor in {name}: the simulation "
                                   "must stay on the CPU")
        if name.startswith(("aten.copy_", "aten.set_")):
            return out                             # the destination keeps its place
        card = (True in where) if big else any(on_card(t) for t in ins)
        if not ins:
            # made from nothing with no device asked (torch.zeros(3)): on the CPU. With a device asked at this level,
            # it's torch's own building from another tensor's options (pad_sequence's output, a gradient built from
            # scratch in the backward) - and every tensor's real device is the CPU here, so that "cpu" means "where
            # the computation is": the card. Python code that asks for the CPU is put right in _Fn after.
            card = isinstance(dev, torch.device)
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


_REAL_AUTOCAST = torch.autocast


class _Autocast(_REAL_AUTOCAST):
    """The card's autocast: the simulated card computes on the CPU, so it is the CPU's, at the dtype asked for."""

    def __init__(self, device_type, *a, **k):
        super().__init__("cpu" if device_type == "cuda" else device_type, *a, **k)


@contextlib.contextmanager
def simulated_gpu(card_gb=24.0):
    """The simulated card (see the module docstring). card_gb: what torch.cuda reports as the card's size."""
    total = int(card_gb * 2**30)
    _on.clear()
    _empty_on.clear()
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
        # where a tensor is, and moving it: on the tensor class, so the backward pass sees it too
        mock.patch.object(torch.Tensor, "device", property(_device)),
        mock.patch.object(torch.Tensor, "is_cuda", property(_is_cuda)),
        mock.patch.object(torch.Tensor, "to", _to),
        mock.patch.object(torch.Tensor, "cuda", _cuda),
        mock.patch.object(torch.Tensor, "cpu", _cpu),
        mock.patch.object(torch.Tensor, "pin_memory", lambda self, *a, **k: self),
        mock.patch.object(torch.Tensor, "is_pinned", lambda self, *a, **k: False),
        mock.patch.object(torch, "autocast", _Autocast),
        # checkpointing keeps the card's random-number state for its recompute: the simulated card has none (the
        # CPU's is kept as usual)
        mock.patch("torch.utils.checkpoint.get_device_states", lambda *a: ([], [])),
        mock.patch("torch.utils.checkpoint.set_device_states", lambda *a, **k: None),
    ]
    # attention kernels: a card-only choice (cuDNN, flash) has no CPU kernel - the simulated card lets torch pick.
    # Fizgig's modules only (torch's own sdpa_kernel is left alone: its compiler reads it at import)
    from torch.nn import attention as _att
    real_kernel = _att.sdpa_kernel
    any_kernel = lambda *a, **k: contextlib.nullcontext()
    import fizgig.modules.offloading  # noqa: F401 - the shared offloader, loaded so its mover is caught below
    for mod in list(sys.modules.values()):
        if not getattr(mod, "__name__", "").startswith("fizgig."):
            continue
        if getattr(mod, "sdpa_kernel", None) is real_kernel:
            patches.append(mock.patch.object(mod, "sdpa_kernel", any_kernel))
        # every offloader's mover thread (the shared one, Krea 2's ...) runs inline, in program order
        if getattr(mod, "ThreadPoolExecutor", None) is concurrent.futures.ThreadPoolExecutor:
            patches.append(mock.patch.object(mod, "ThreadPoolExecutor", _Inline))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(_Fn())
        stack.enter_context(_Ops())
        try:
            yield
        finally:
            _on.clear()
