"""Async H2D-only block streaming for the int8 ConvRot base — contributed by @rintic-13 (#73).

The insight this is built on: the base is FROZEN during LoRA training, so the device-to-host
half of every classic block-swap round trip writes back weights that cannot have changed —
half the PCIe traffic was always waste. This offloader never copies device-to-host at all.

How it works:

  * Each swapped block's ConvRot tensors (`qdata` int8 + `wscale` f32 — 0.385 GB of the
    block's 0.39 GB) are coalesced into ONE pinned, 256-aligned flat CPU buffer per block.
    Everything else in a swapped block (norms, AdaLN, the LoRA adapters were never inside)
    stays resident on GPU permanently.
  * The GPU holds a small RING of identical flat buffers (2 slots). A dedicated copy stream
    prefetches block N+ring while block N computes; events guard both directions (compute
    waits for its block's copy; a slot is only overwritten after its previous owner's
    compute has been recorded complete).
  * Forward walks the ring forward; backward hooks prefetch in reverse. Under gradient
    checkpointing the recompute calls `wait_for_block`, which self-heals if the slot has
    rotated — blocks never physically move, so the non-reentrant-checkpoint early-stop that
    shaped the classic swap's two-park-site design has nothing to skip here.

Module attribute rebinding is what keeps a whole-model park safe: `_load` re-binds the
PREVIOUS slot owner to its CPU view before overwriting the slot, so at any instant a module's
`qdata`/`wscale` point either at its own CPU copy or at ring memory holding its own data —
`dit.to("cpu")` therefore always copies the right bytes. After such a park the tensor handles
this object caches ARE stale; `enable_block_swap` tears it down (`release()`) and builds a
fresh one, and `prepare()` accepts sources on any device (its `copy_` pulls GPU-resident
tensors into the pinned flats), which is what makes the rebuild correct.

ConvRot-specific by design. The NF4 path keeps the classic `.to()` swap; bnb's Params4bit
round-trips packed and could get the same treatment later if a use case appears.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class H3Int8H2DOffloader:
    """Streams the last N blocks' int8 tensors host→device through a ring buffer."""

    def __init__(self, blocks, swap_from: int, device: torch.device, ring_size: int = 2):
        self.blocks = blocks
        self.swap_from = swap_from
        self.device = device
        self.ring_size = max(1, int(ring_size))

        self.stream = torch.cuda.Stream(device=device)

        self.sources = {}                # block_idx -> [(module, attr, tensor), ...]
        self.cpu_flat = {}               # block_idx -> pinned flat uint8 buffer
        self.ring_flat = None            # [flat gpu buffer] * ring_size
        self.ring_views = None           # per-slot typed views into ring_flat
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        self.copy_done = {}              # block_idx -> event the compute stream must wait on
        self._layout = None              # (offsets, total_bytes), identical for every block
        # Page-locking every swapped block's CPU master wants n_swap x ~0.42 GB of pinned RAM
        # (31 blocks ≈ 13 GB on a 16 GB card's plan). Machines that can't lock that much get
        # "CUDA error: out of memory" out of cudaHostAlloc — host-side, before training ever
        # starts (field report: 16 GB 4090). Once one pin fails we stop trying and stage from
        # ordinary RAM: the async H2D copies quietly become synchronous — slower, not broken.
        self._pin_failed = False

        self._collect_sources()
        self.n_swap = len(self.sources)
        if self.n_swap == 0:
            raise RuntimeError("H3Int8H2DOffloader: no swapped ConvRot blocks found")
        self.ring_size = min(self.ring_size, self.n_swap)
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size

        # Backward prefetch: as each swapped block's backward completes, start the H2D for
        # the block ring_size positions EARLIER (backward walks in reverse).
        self.remove_handles = []
        for i, block in enumerate(self.blocks):
            hook = self._create_backward_hook(i)
            if hook is not None:
                self.remove_handles.append(block.register_full_backward_hook(hook))

    # -- static weights -----------------------------------------------------------------------
    def move_static_weights_to_gpu(self):
        """Everything in a swapped block EXCEPT the streamed int8 tensors lives on GPU
        permanently — norms and AdaLN are ~1.7 MB a block, not worth a single PCIe trip."""
        for block_idx in range(self.swap_from, len(self.blocks)):
            for module in self.blocks[block_idx].modules():
                if module.__class__.__name__ == "ConvRotInt8Linear":
                    continue
                for param in module.parameters(recurse=False):
                    if param is not None:
                        param.data = param.data.to(self.device, non_blocking=True)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None or name in ("qdata", "wscale"):
                        continue
                    module._buffers[name] = buf.to(self.device, non_blocking=True)

    # -- source discovery ---------------------------------------------------------------------
    def _collect_sources(self):
        for block_idx in range(self.swap_from, len(self.blocks)):
            entries = []
            for module in self.blocks[block_idx].modules():
                if module.__class__.__name__ != "ConvRotInt8Linear":
                    continue
                if not hasattr(module, "qdata") or not hasattr(module, "wscale"):
                    raise RuntimeError(
                        f"Block {block_idx}: ConvRotInt8Linear has no qdata/wscale")
                entries.append((module, "qdata", module.qdata))
                entries.append((module, "wscale", module.wscale))
            if entries:
                self.sources[block_idx] = entries

    # -- layout -------------------------------------------------------------------------------
    @staticmethod
    def _compute_layout(weights):
        """Byte offsets for each tensor in one flat buffer, 256-aligned so the typed views
        (int8 and f32 in the same flat) always start on a legal boundary."""
        align = 256
        offsets, total = [], 0
        for weight in weights:
            total = (total + align - 1) // align * align
            offsets.append(total)
            total += weight.numel() * weight.element_size()
        return offsets, total

    def _flat_views(self, flat, weights):
        offsets, _ = self._layout
        return [flat[offset: offset + w.numel() * w.element_size()]
                .view(w.dtype).view(w.shape)
                for offset, w in zip(offsets, weights)]

    # -- binding ------------------------------------------------------------------------------
    def _bind_cpu(self, block_idx):
        for module, attr, cpu_tensor in self.sources[block_idx]:
            setattr(module, attr, cpu_tensor)

    def _bind_gpu(self, block_idx, slot):
        for (module, attr, _), gpu_tensor in zip(self.sources[block_idx],
                                                 self.ring_views[slot]):
            setattr(module, attr, gpu_tensor)

    # -- load ---------------------------------------------------------------------------------
    def _ensure_cpu_flat(self, block_idx):
        """The block's pinned CPU master, built once. copy_ pulls from wherever the source
        tensors currently live (CPU after a fresh load, GPU after a whole-model unpark) —
        source-device-agnostic on purpose; it is what makes rebuild-after-restore correct."""
        entries = self.sources[block_idx]
        weights = [t for _, _, t in entries]
        if self._layout is None:
            self._layout = self._compute_layout(weights)
        if block_idx in self.cpu_flat:
            self._bind_cpu(block_idx)
            return
        flat = torch.empty(self._layout[1], dtype=torch.uint8, device="cpu")
        if not self._pin_failed:
            try:
                flat = flat.pin_memory(device=self.device)
            except Exception as e:
                self._pin_failed = True
                logger.warning(
                    "[h2d] could not page-lock the CPU staging buffers (%s: %s) — the OS "
                    "won't pin this much RAM. Continuing with ordinary (unpinned) staging: "
                    "H2D copies run synchronously, so steps are slower but the run works. "
                    "Freeing system RAM (or adding more) restores the fast path.",
                    type(e).__name__, e)
        cpu_views = self._flat_views(flat, weights)
        for (_, _, src), view in zip(entries, cpu_views):
            view.copy_(src)
        # The views become the canonical CPU tensors from here on.
        self.sources[block_idx] = [(m, a, v) for (m, a, _), v in zip(entries, cpu_views)]
        for module, attr, view in self.sources[block_idx]:
            setattr(module, attr, view)
        self.cpu_flat[block_idx] = flat

    def _ensure_ring(self):
        if self.ring_flat is not None:
            return
        self.ring_flat = [torch.empty(self._layout[1], dtype=torch.uint8, device=self.device)
                          for _ in range(self.ring_size)]
        template = [t for _, _, t in self.sources[min(self.sources)]]
        self.ring_views = [self._flat_views(flat, template) for flat in self.ring_flat]

    def _load(self, rank: int, slot: int):
        block_idx = self.swap_from + rank
        if self.loaded_block[slot] == block_idx:
            self._bind_gpu(block_idx, slot)
            return
        # The slot's previous owner goes back to its CPU view BEFORE the slot is overwritten —
        # the invariant that keeps a whole-model park copying the right bytes.
        previous = self.loaded_block[slot]
        if previous is not None:
            self._bind_cpu(previous)

        self._ensure_cpu_flat(block_idx)
        self._ensure_ring()

        gate = self.free_event[slot]
        with torch.cuda.stream(self.stream):
            if gate is not None:
                self.stream.wait_event(gate)          # never overwrite a slot still computing
            self.ring_flat[slot].copy_(self.cpu_flat[block_idx], non_blocking=True)
            done = self.stream.record_event()

        # The module points at ring memory immediately; actual compute waits on `done`
        # through wait_for_block().
        self._bind_gpu(block_idx, slot)
        self.loaded_block[slot] = block_idx
        self.copy_done[block_idx] = done

    # -- prepare ------------------------------------------------------------------------------
    def prepare(self):
        """Pin every block's CPU master, build the ring, preload the first ring_size blocks.
        Re-entrant: safe to call again after state was reset."""
        for block_idx in self.sources:
            self._ensure_cpu_flat(block_idx)
        self._ensure_ring()
        self.free_event = [None] * self.ring_size
        self.loaded_block = [None] * self.ring_size
        self.copy_done.clear()
        for rank in range(self.ring_size):
            self._load(rank, rank)
        event = self.stream.record_event()
        torch.cuda.current_stream().wait_event(event)

    # -- forward ------------------------------------------------------------------------------
    def wait_for_block(self, block_idx: int):
        """The compute stream holds until this block's bytes have landed. Self-healing: if
        the slot has rotated on (a checkpoint recompute arriving late), reload first."""
        if block_idx < self.swap_from or block_idx not in self.sources:
            return
        rank = block_idx - self.swap_from
        slot = rank % self.ring_size
        if self.loaded_block[slot] != block_idx:
            self._load(rank, slot)
        event = self.copy_done.get(block_idx)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

    def submit_move_blocks_forward(self, block_idx: int):
        """The block's forward is done: record its slot free and start the H2D for the block
        ring_size positions ahead — on the copy stream, overlapping the next computes."""
        if block_idx < self.swap_from:
            return
        rank = block_idx - self.swap_from
        slot = rank % self.ring_size
        self.free_event[slot] = torch.cuda.current_stream().record_event()
        next_rank = rank + self.ring_size
        if next_rank < self.n_swap:
            self._load(next_rank, next_rank % self.ring_size)

    # -- backward -----------------------------------------------------------------------------
    def _create_backward_hook(self, block_index):
        if block_index < self.swap_from:
            return None
        rank = block_index - self.swap_from

        def backward_hook(_module, _grad_input, _grad_output):
            slot = rank % self.ring_size
            self.free_event[slot] = torch.cuda.current_stream().record_event()
            prev_rank = rank - self.ring_size
            if prev_rank >= 0:
                self._load(prev_rank, prev_rank % self.ring_size)
            return None

        return backward_hook

    # -- teardown -----------------------------------------------------------------------------
    def release(self):
        """Full teardown: copy stream drained, hooks removed, every module re-bound to its
        pinned CPU view, ring memory dropped. enable_block_swap calls this before building a
        fresh offloader after a whole-model park replaced the tensor storages this object
        caches — without it, hooks would accumulate one set per preview cycle."""
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
        for block_idx in self.sources:
            self._bind_cpu(block_idx)
        self.loaded_block = [None] * self.ring_size
        self.copy_done.clear()
        self.ring_flat = None
        self.ring_views = None
        torch.cuda.empty_cache()

    def unbind_to_cpu(self):
        """Point every managed module back at its pinned CPU master, changing nothing else.

        park_dit_partial evicts CUDA buffers by resizing their storages to zero — correct
        for real weights, fatal for these bindings: after a forward, a swapped block's
        qdata/wscale still point at RING VIEWS, and every view of a slot shares ONE
        storage. Parking the first view resize_(0)s the whole slot, and the very next
        buffer copied from that storage dies with CUDA 'invalid argument' — a sticky
        context error that then kills the preview decode and the first training step
        (field: 24 GB card, auto plan swap 6, 56-frame clip preview at 4.9 GB free; run
        died at step 0/6900 in the rebuilt offloader's first ring copy). The swapped
        blocks' weights already live on CPU in the pinned flats, so re-binding lets the
        park's walk skip them exactly as its "already on CPU, skipped as not-cuda" rule
        assumes. Slot bookkeeping resets too, so a wait_for_block() that somehow runs
        before the next restore self-heals a reload instead of trusting stale bindings."""
        for block_idx in self.sources:
            self._bind_cpu(block_idx)
        self.loaded_block = [None] * self.ring_size
        self.free_event = [None] * self.ring_size
        self.copy_done.clear()

    def __del__(self):
        try:
            for handle in getattr(self, "remove_handles", []):
                handle.remove()
        except Exception:
            pass
