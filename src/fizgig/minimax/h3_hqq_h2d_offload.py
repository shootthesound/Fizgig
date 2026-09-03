"""H2D-only streaming for frozen MiniMax H3 HQQ 4-bit blocks — by rintic-13 (issue #102).

The HQQ sibling of rintic-13's int8 ConvRot ring (#73) and @mabseyuk's NF4 ring: the packed
4-bit codes and their per-group scale/zero stay authoritative in CPU RAM; `ring_size` GPU
slots are reused by every streamed block; the frozen base never writes back to the host.
Slot buffers are allocated PER SLOT and reused — the first cut keyed them by
(block_idx, slot) and grew a fresh GPU copy of every streamed block, a leak @mabseyuk caught
in review; the per-slot layout here is rintic-13's fix.

Each module contributes four tensors (W_q, scale, zero, qmeta — HQQ4bitLinear's buffers, see
hqq4.py); binding a block to a slot rebinds those three buffers by name, so the module's
forward reads whichever storage it is currently bound to. Selected automatically by the VRAM
planner: enable_block_swap dispatches by module type (ConvRot → int8 ring, Linear4bit → NF4
ring, HQQ4bitLinear → this). FIZGIG_NO_NF4_H2D=1 is the shared kill-switch back to classic
parking for both 4-bit rings.

Interface contract matches the other two rings exactly (kind / staged_gb / _pin_failed /
move_static_weights_to_gpu / prepare / wait_for_block / submit_move_blocks_forward /
unbind_to_cpu / release), so park_dit_partial and restore_parked_dit work unmodified.
"""

from __future__ import annotations

import logging

import torch

from .hqq4 import HQQ4bitLinear

logger = logging.getLogger(__name__)

_NAMES = ("W_q", "scale", "zero", "qmeta")


class H3HQQH2DOffloader:
    """Streams frozen blocks' HQQ tensors host-to-device through a ring.

    `swap_from` is an int (stream the tail from that index — the classic LoRA-mode swap) OR
    an explicit iterable of block indices, walked by RANK in ascending order exactly as the
    NF4 ring does."""

    kind = "hqq"

    def __init__(self, blocks, swap_from, device: torch.device, ring_size: int = 2):
        self.blocks = blocks
        if isinstance(swap_from, int):
            self.swap_list = list(range(swap_from, len(blocks)))
        else:
            self.swap_list = sorted(int(b) for b in swap_from)
        self.device = torch.device(device)
        self.ring_size = max(1, int(ring_size))
        self.stream = torch.cuda.Stream(device=self.device)
        self.specs = {}            # block_idx -> [HQQ4bitLinear, ...]
        self.cpu_data = {}         # block_idx -> [(W_q, scale, zero) on CPU, ...]
        self.gpu_data = None       # slot -> [(W_q, scale, zero) on GPU, ...]  (per SLOT)
        self.loaded_block = []
        self.free_event = []
        self.copy_done = {}
        self._pin_failed = False
        self.remove_handles = []

        self._collect_specs()
        self.n_swap = len(self.specs)
        if not self.n_swap:
            raise RuntimeError("H3HQQH2DOffloader: no swapped HQQ4bitLinear blocks found")
        self._ranked = sorted(self.specs)
        self._rank_of = {b: r for r, b in enumerate(self._ranked)}
        # RAM-aware pinning, as the NF4 ring does it: the staged bytes must live in RAM
        # either way, but page-locking ~14 GB on a tight box starves Windows commit. When
        # available RAM barely covers the stage, start unpinned (copies synchronous) rather
        # than failing pin-by-pin.
        try:
            import psutil
            _est = sum(t.numel() * t.element_size()
                       for m in self.specs[min(self.specs)]
                       for t in self._tensors(m)) * self.n_swap
            if psutil.virtual_memory().available < _est + max(4e9, 0.75 * _est):
                self._pin_failed = True
                logger.warning("[hqq-h2d] available RAM is tight for ~%.1f GB of pinned "
                               "staging — staging unpinned instead (copies synchronous, "
                               "memory stays pageable).", _est / 1e9)
        except Exception:
            pass
        self.ring_size = min(self.ring_size, self.n_swap)
        self.gpu_data = [None] * self.ring_size
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        for i, block in enumerate(self.blocks):
            hook = self._create_backward_hook(i)
            if hook is not None:
                self.remove_handles.append(block.register_full_backward_hook(hook))

    def _collect_specs(self):
        for block_idx in self.swap_list:
            mods = []
            for module in self.blocks[block_idx].modules():
                if not isinstance(module, HQQ4bitLinear):
                    continue
                for n in _NAMES:
                    t = getattr(module, n, None)
                    if not isinstance(t, torch.Tensor) or t.is_meta:
                        raise RuntimeError(f"Block {block_idx}: HQQ4bitLinear.{n} is not loaded")
                mods.append(module)
            if mods:
                self.specs[block_idx] = mods

    @staticmethod
    def _tensors(module):
        return tuple(getattr(module, n) for n in _NAMES)

    @staticmethod
    def _bind_one(module, triple):
        for n, t in zip(_NAMES, triple):
            setattr(module, n, t)           # registered buffer names → plain rebind

    def _copy_cpu_tensor(self, source):
        out = torch.empty_like(source, device="cpu")
        if not self._pin_failed:
            try:
                out = out.pin_memory()
            except Exception as exc:
                self._pin_failed = True
                logger.warning("[hqq-h2d] CPU pinning failed (%s: %s); copies will be "
                               "synchronous", type(exc).__name__, exc)
        out.copy_(source.detach())
        return out

    def _ensure_cpu_block(self, block_idx):
        if block_idx in self.cpu_data:
            return
        self.cpu_data[block_idx] = [tuple(self._copy_cpu_tensor(t) for t in self._tensors(m))
                                    for m in self.specs[block_idx]]

    def _ensure_ring_slot(self, slot, block_idx):
        if self.gpu_data[slot] is not None:
            return
        # Every streamed block has the same module layout (all H3 blocks are identical),
        # so a slot shaped from any one block serves them all.
        self.gpu_data[slot] = [tuple(torch.empty_like(t, device=self.device) for t in triple)
                               for triple in self.cpu_data[block_idx]]

    def _bind_cpu(self, block_idx):
        for module, triple in zip(self.specs[block_idx], self.cpu_data[block_idx]):
            self._bind_one(module, triple)

    def _bind_slot(self, block_idx, slot):
        for module, triple in zip(self.specs[block_idx], self.gpu_data[slot]):
            self._bind_one(module, triple)

    def move_static_weights_to_gpu(self):
        """Keep biases, norms, AdaLN and LoRA-side tensors resident; stream HQQ only."""
        for block_idx in self.swap_list:
            for module in self.blocks[block_idx].modules():
                if isinstance(module, HQQ4bitLinear):
                    if module.bias is not None:
                        module.bias.data = module.bias.data.to(self.device, non_blocking=True)
                    continue
                for param in module.parameters(recurse=False):
                    if param is not None:
                        param.data = param.data.to(self.device, non_blocking=True)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is not None:
                        module._buffers[name] = buf.to(self.device, non_blocking=True)

    def _load(self, rank, slot):
        block_idx = self._ranked[rank]
        if self.loaded_block[slot] == block_idx:
            self._bind_slot(block_idx, slot)
            return
        previous = self.loaded_block[slot]
        if previous is not None:
            self._ensure_cpu_block(previous)
            self._bind_cpu(previous)        # evicted block points at its CPU masters again
        self._ensure_cpu_block(block_idx)
        self._ensure_ring_slot(slot, block_idx)
        gate = self.free_event[slot]
        with torch.cuda.stream(self.stream):
            if gate is not None:
                self.stream.wait_event(gate)
            for src, dst in zip(self.cpu_data[block_idx], self.gpu_data[slot]):
                for cpu_t, gpu_t in zip(src, dst):
                    gpu_t.copy_(cpu_t, non_blocking=not self._pin_failed)
            done = self.stream.record_event()
        self._bind_slot(block_idx, slot)
        self.loaded_block[slot] = block_idx
        self.copy_done[block_idx] = done

    def prepare(self):
        for block_idx in self.specs:
            self._ensure_cpu_block(block_idx)
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
        return sum(t.numel() * t.element_size()
                   for triples in self.cpu_data.values()
                   for triple in triples for t in triple) / 1e9

    def unbind_to_cpu(self):
        for block_idx in self.specs:
            if block_idx in self.cpu_data:     # guarded: a park against a half-built ring
                self._bind_cpu(block_idx)
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
        self.unbind_to_cpu()
        if self.gpu_data is not None:
            self.gpu_data = [None] * self.ring_size
        torch.cuda.empty_cache()

    def __del__(self):
        for handle in getattr(self, "remove_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
