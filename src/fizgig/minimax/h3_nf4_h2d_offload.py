"""H2D-only streaming for frozen MiniMax H3 NF4 blocks — by @mabseyuk.

The NF4 sibling of rintic-13's int8 ring (#73): the packed bitsandbytes weight and its
QuantState tensors remain authoritative in CPU RAM; two flat GPU ring slots are reused by
every streamed block; the frozen base never writes weights back to the host. This is what
retires the classic two-way parking swap for NF4 bases — the tier 12 GB cards land on —
measured on a 5070: 12-14 s/step under parking became ~1 s/step streamed, 2,800 steps in
48 minutes, with the resulting LoRA validated in ComfyUI.

Selected automatically by the VRAM planner (enable_block_swap dispatches by module type:
ConvRot -> the int8 ring, bnb Linear4bit -> this). FIZGIG_NO_NF4_H2D=1 is the debug
kill-switch back to classic parking; there is no opt-in — like every VRAM decision, the
planner owns it.

Interface contract matches H3Int8H2DOffloader exactly (wait_for_block /
submit_move_blocks_forward / unbind_to_cpu / release / _pin_failed), so park_dit_partial
and restore_parked_dit work on either ring unmodified.
"""

from __future__ import annotations

import logging

import torch
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit

logger = logging.getLogger(__name__)


def _state_tensors(state):
    out = [state.absmax, state.code]
    if state.nested:
        out.extend([state.offset, state.state2.absmax, state.state2.code])
    return out


def _state_from_views(source, views):
    it = iter(views)
    absmax, code = next(it), next(it)
    if source.nested:
        offset, nested_absmax, nested_code = next(it), next(it), next(it)
        state2 = QuantState(absmax=nested_absmax, shape=source.state2.shape,
                            code=nested_code, dtype=source.state2.dtype,
                            blocksize=source.state2.blocksize,
                            quant_type=source.state2.quant_type)
    else:
        offset = state2 = None
    return QuantState(absmax=absmax, shape=source.shape, code=code, dtype=source.dtype,
                      blocksize=source.blocksize, quant_type=source.quant_type,
                      offset=offset, state2=state2)


def _wrapped_weight(module, packed, state):
    p = Params4bit(packed, requires_grad=False, quant_state=state,
                   blocksize=state.blocksize, compress_statistics=state.nested,
                   quant_type=state.quant_type, quant_storage=packed.dtype,
                   module=module, bnb_quantized=True)
    module.quant_state = state
    return p


def bind_block_packed_to(block, device):
    """Move one block's packed NF4 weights (+ quant-state tensors) to `device` WITHOUT
    the bnb re-quantize trap: Params4bit re-quantizes on a cpu->cuda move, so an
    already-packed weight .to(cuda) turns into garbage. The raw byte tensors are copied
    and re-wrapped instead. Non-quantized params and buffers move normally. Used by the
    rotation fine-tune when a previously-streamed block becomes resident."""
    dev = torch.device(device)
    for module in block.modules():
        if isinstance(module, Linear4bit):
            weight = module.weight
            state = (getattr(weight, "quant_state", None)
                     or getattr(module, "quant_state", None))
            if state is not None:
                packed = weight.data.to(dev)
                views = [t.to(dev) for t in _state_tensors(state)]
                module.weight = _wrapped_weight(module, packed,
                                                _state_from_views(state, views))
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dev)
            continue
        for param in module.parameters(recurse=False):
            if param is not None:
                param.data = param.data.to(dev)
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None:
                module._buffers[name] = buf.to(dev)


class H3NF4H2DOffloader:
    """Streams frozen blocks' packed NF4 tensors host-to-device through a ring.

    `swap_from` is an int (stream the tail from that index — the classic LoRA-mode swap)
    OR an explicit iterable of block indices (an arbitrary streamed set — the rotation
    fine-tune's frozen out-of-window blocks, which form a head+tail around the active
    depth chunk). The forward visits streamed blocks in ascending order either way, so
    the ring walks by RANK — a block's position in the sorted streamed list — and the
    tail case is just the special ranking where rank == block_idx - swap_from."""

    kind = "nf4"

    def __init__(self, blocks, swap_from, device: torch.device, ring_size: int = 2):
        self.blocks = blocks
        if isinstance(swap_from, int):
            self.swap_list = list(range(swap_from, len(blocks)))
        else:
            self.swap_list = sorted(int(b) for b in swap_from)
        self.device = torch.device(device)
        self.ring_size = max(1, int(ring_size))
        self.stream = torch.cuda.Stream(device=self.device)
        self.specs = {}
        self.cpu_flat = {}
        self.cpu_bindings = {}
        self.gpu_bindings = {}
        self.ring_flat = None
        self.ring_views = None
        self.loaded_block = []
        self.free_event = []
        self.copy_done = {}
        self._layout = None
        self._pin_failed = False
        self.remove_handles = []

        self._collect_specs()
        self.n_swap = len(self.specs)
        if not self.n_swap:
            raise RuntimeError("H3NF4H2DOffloader: no swapped Linear4bit blocks found")
        # Rank = walk order. Built from the blocks that actually HAVE specs, so a block
        # with no Linear4bit in the requested set can never skew the arithmetic.
        self._ranked = sorted(self.specs)
        self._rank_of = {b: r for r, b in enumerate(self._ranked)}
        # RAM-aware pinning (audit, 25 Aug): the staged bytes must live in RAM whether
        # they ring-stream or classic-park — but they don't have to be PAGE-LOCKED. On
        # a bf16-checkpoint plan (adaln streams too, ~333 MB/block, up to ~13 GB at the
        # 40-block cap) pinning that much on a tight box starves Windows commit. When
        # available RAM barely covers the staging, start unpinned instead of failing
        # pin-by-pin: copies go synchronous, everything stays pageable, the run lives.
        try:
            import psutil
            _est = sum(t.numel() * t.element_size()
                       for t in self._source_tensors(min(self.specs))) * self.n_swap
            # Margin scales with the stage (review 6): a flat +10 GB disarmed pinning
            # for a ~1.6 GB NF4 stage on any box under ~11.6 GB available, where
            # page-locking that little is trivially safe — and the ring's prefetch
            # overlap (its whole point) was silently lost. 0.75x + 4 GB floor keeps
            # ~23 GB at the 13 GB bf16 cap this guard was written for.
            if psutil.virtual_memory().available < _est + max(4e9, 0.75 * _est):
                self._pin_failed = True
                logger.warning("[nf4-h2d] available RAM is tight for ~%.1f GB of "
                               "pinned staging — staging unpinned instead (copies "
                               "synchronous, memory stays pageable).", _est / 1e9)
        except Exception:
            pass
        self.ring_size = min(self.ring_size, self.n_swap)
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        for i, block in enumerate(self.blocks):
            hook = self._create_backward_hook(i)
            if hook is not None:
                self.remove_handles.append(block.register_full_backward_hook(hook))

    def _collect_specs(self):
        for block_idx in self.swap_list:
            block_specs = []
            for module in self.blocks[block_idx].modules():
                if not isinstance(module, Linear4bit):
                    continue
                weight = module.weight
                state = getattr(weight, "quant_state", None) or getattr(module, "quant_state", None)
                if state is None or not getattr(weight, "bnb_quantized", False):
                    raise RuntimeError(f"Block {block_idx}: Linear4bit is not quantized")
                block_specs.append((module, state, 1 + len(_state_tensors(state))))
            if block_specs:
                self.specs[block_idx] = block_specs

    @staticmethod
    def _compute_layout(tensors):
        offsets, total, align = [], 0, 256
        for tensor in tensors:
            total = (total + align - 1) // align * align
            offsets.append(total)
            total += tensor.numel() * tensor.element_size()
        return offsets, total

    def _flat_views(self, flat, tensors):
        offsets, _ = self._layout
        return [flat[o:o + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
                for o, t in zip(offsets, tensors)]

    def _source_tensors(self, block_idx):
        out = []
        for module, state, _ in self.specs[block_idx]:
            out.append(module.weight.data)
            out.extend(_state_tensors(state))
        return out

    def _make_bindings(self, block_idx, views):
        bindings, pos = [], 0
        for module, source_state, count in self.specs[block_idx]:
            packed = views[pos]
            state = _state_from_views(source_state, views[pos + 1:pos + count])
            bindings.append((module, _wrapped_weight(module, packed, state)))
            pos += count
        return bindings

    @staticmethod
    def _bind(bindings):
        for module, weight in bindings:
            module.weight = weight
            module.quant_state = weight.quant_state

    def move_static_weights_to_gpu(self):
        """Keep biases, norms, AdaLN and LoRA-side tensors resident; stream NF4 only."""
        for block_idx in self.swap_list:
            for module in self.blocks[block_idx].modules():
                if isinstance(module, Linear4bit):
                    if module.bias is not None:
                        module.bias.data = module.bias.data.to(self.device, non_blocking=True)
                    continue
                for param in module.parameters(recurse=False):
                    if param is not None:
                        param.data = param.data.to(self.device, non_blocking=True)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is not None:
                        module._buffers[name] = buf.to(self.device, non_blocking=True)

    def _ensure_cpu_flat(self, block_idx):
        if block_idx in self.cpu_flat:
            self._bind(self.cpu_bindings[block_idx])
            return
        tensors = self._source_tensors(block_idx)
        layout = self._compute_layout(tensors)
        if self._layout is None:
            self._layout = layout
        elif layout[1] != self._layout[1] or layout[0] != self._layout[0]:
            raise RuntimeError(f"Block {block_idx}: NF4 layout differs from the ring template")
        flat = torch.empty(self._layout[1], dtype=torch.uint8, device="cpu")
        if not self._pin_failed:
            try:
                flat = flat.pin_memory()
            except Exception as exc:
                self._pin_failed = True
                logger.warning("[nf4-h2d] CPU pinning failed (%s: %s); copies will be "
                               "synchronous", type(exc).__name__, exc)
        views = self._flat_views(flat, tensors)
        for view, source in zip(views, tensors):
            view.copy_(source)
        bindings = self._make_bindings(block_idx, views)
        self.cpu_flat[block_idx] = flat
        self.cpu_bindings[block_idx] = bindings
        self._bind(bindings)

    def _ensure_ring(self):
        if self.ring_flat is not None:
            return
        template = self._source_tensors(min(self.specs))
        self.ring_flat = [torch.empty(self._layout[1], dtype=torch.uint8,
                                      device=self.device) for _ in range(self.ring_size)]
        self.ring_views = [self._flat_views(flat, template) for flat in self.ring_flat]

    def _bindings_for_slot(self, block_idx, slot):
        key = (block_idx, slot)
        if key not in self.gpu_bindings:
            self.gpu_bindings[key] = self._make_bindings(block_idx, self.ring_views[slot])
        return self.gpu_bindings[key]

    def _load(self, rank, slot):
        block_idx = self._ranked[rank]
        if self.loaded_block[slot] == block_idx:
            self._bind(self._bindings_for_slot(block_idx, slot))
            return
        previous = self.loaded_block[slot]
        if previous is not None:
            self._bind(self.cpu_bindings[previous])
        self._ensure_cpu_flat(block_idx)
        self._ensure_ring()
        gate = self.free_event[slot]
        with torch.cuda.stream(self.stream):
            if gate is not None:
                self.stream.wait_event(gate)
            self.ring_flat[slot].copy_(self.cpu_flat[block_idx],
                                       non_blocking=not self._pin_failed)
            done = self.stream.record_event()
        self._bind(self._bindings_for_slot(block_idx, slot))
        self.loaded_block[slot] = block_idx
        self.copy_done[block_idx] = done

    def prepare(self):
        for block_idx in self.specs:
            self._ensure_cpu_flat(block_idx)
        self._ensure_ring()
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        self.copy_done.clear()
        for rank in range(self.ring_size):
            self._load(rank, rank)
        torch.cuda.current_stream().wait_event(self.stream.record_event())

    def wait_for_block(self, block_idx):
        if block_idx not in self.specs:
            return
        rank = self._rank_of[block_idx]
        slot = rank % self.ring_size
        if self.loaded_block[slot] != block_idx:
            self._load(rank, slot)
        event = self.copy_done.get(block_idx)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

    def submit_move_blocks_forward(self, block_idx):
        if block_idx not in self.specs:
            return
        rank = self._rank_of[block_idx]
        slot = rank % self.ring_size
        self.free_event[slot] = torch.cuda.current_stream().record_event()
        next_rank = rank + self.ring_size
        if next_rank < self.n_swap:
            self._load(next_rank, next_rank % self.ring_size)

    def _create_backward_hook(self, block_idx):
        if block_idx not in self.specs:
            return None
        rank = self._rank_of[block_idx]

        def hook(_module, _grad_input, _grad_output):
            slot = rank % self.ring_size
            self.free_event[slot] = torch.cuda.current_stream().record_event()
            previous = rank - self.ring_size
            if previous >= 0:
                self._load(previous, previous % self.ring_size)
            return None
        return hook

    @property
    def staged_gb(self):
        return sum(x.numel() for x in self.cpu_flat.values()) / 1e9

    def unbind_to_cpu(self):
        for block_idx in self.specs:
            # Guarded like release(): a park against a partially-prepared offloader
            # (construction failed mid-way) must not KeyError (review, 25 Aug).
            if block_idx in self.cpu_bindings:
                self._bind(self.cpu_bindings[block_idx])
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        self.copy_done.clear()

    def release(self):
        try:
            self.stream.synchronize()
        except Exception:
            pass
        for handle in self.remove_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.remove_handles = []
        for block_idx in self.specs:
            if block_idx in self.cpu_bindings:
                self._bind(self.cpu_bindings[block_idx])
        self.gpu_bindings.clear()
        self.ring_flat = self.ring_views = None
        # Reset the slot state too (the int8 twin does): stale loaded_block entries let
        # a post-release wait_for_block take the fast path and return with CPU-bound
        # weights — a silent device mismatch instead of a reload (review, 25 Aug).
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        self.copy_done.clear()
        torch.cuda.empty_cache()

    def __del__(self):
        # Belt-and-braces mirror of the int8 twin: if an owner drops the object without
        # release(), at least the backward hooks come off the modules.
        for handle in getattr(self, "remove_handles", []):
            try:
                handle.remove()
            except Exception:
                pass

