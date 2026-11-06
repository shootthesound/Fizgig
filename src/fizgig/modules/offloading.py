"""
Fizgig offloading utilities.

Self-contained module for swapping model block weights between CPU and GPU
during forward/backward passes. No external framework imports required.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import os
import time
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers (kept private to this module)
# ---------------------------------------------------------------------------

def _clean_memory_on_device(device: torch.device):
    """Clean memory on the specified device."""
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    """Block until all pending work on *device* has completed."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Standalone weight-transfer helpers
# ---------------------------------------------------------------------------

def _streamed_linears(layer: nn.Module):
    """{name: module} for the Linears a swap streams, keyed with LoRA wrappers' ".base" dropped, so a block whose
    Linears carry a LoRA pairs with one whose don't (#199)."""
    out = {}
    for name, m in layer.named_modules():
        if hasattr(m, "weight") and m.weight is not None and m.__class__.__name__.endswith("Linear"):
            out[name[:-5] if name.endswith(".base") else name] = m
    return out


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Swap weights between two layers without CUDA streams (fallback path).
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    modules_to_cuda = _streamed_linears(layer_to_cuda)
    for name, module_to_cpu in _streamed_linears(layer_to_cpu).items():
        module_to_cuda = modules_to_cuda.get(name)
        if module_to_cuda is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device -> cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu -> device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    """Move all Linear weights (and fp8 scale_weight buffers) in *layer* to *device*."""
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear"):
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")
            if hasattr(module, "scale_weight"):
                module.scale_weight.data = module.scale_weight.data.to(device, non_blocking=device.type != "cpu")


# ---------------------------------------------------------------------------
# Offloader
# ---------------------------------------------------------------------------

class Offloader:
    """Common offloading class for swapping transformer blocks between CPU and GPU."""

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        # Check debug flag from environment variable
        if not debug:
            debug = os.getenv("FIZGIG_OFFLOADER_DEBUG", "0") == "1"

        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for cuda offloading without large pinned memory.
        # Pinned memory buffers to speed up the transfer between CPU and GPU.
        # One staging buffer per transfer direction (A: GPU->CPU, B: CPU->GPU).
        self.staging_buffer_a = None
        self.staging_buffer_b = None

        # Pinned buffer for cuda offloading with pinned memory (one per layer transfer).
        self.pinned_buffer = None

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        debug_print = False
        if self.debug:
            debug_print = self.debug_block_count % 10 == 0
            self.debug_block_count += 1

        class Timer:
            def __init__(self, enabled=False):
                self.enabled = enabled
                self.totals = defaultdict(float)
                self.start_time = time.perf_counter()

            @contextmanager
            def section(self, name):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = Timer(enabled=debug_print)

        weight_swap_jobs = []

        with T.section("find modules"):
            modules_to_cpu = _streamed_linears(layer_to_cpu)
            for module_to_cuda_name, module_to_cuda in _streamed_linears(layer_to_cuda).items():
                module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                    weight_swap_jobs.append(
                        (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                    )
                else:
                    if module_to_cuda.weight.data.device.type != device.type:
                        module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        with T.section("synchronize before swap"):
            # Ensures the offloading layer's computation is done before we touch its buffers
            torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            # Minimize pinned memory usage for lower shared GPU RAM consumption
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                # Copy weights to staging buffers and record events
                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    # CUDA to staging buffer A, non-blocking copy
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                        event_a.record(stream)

                    # Wait for staging buffer B to be ready
                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()

                    # CPU to staging buffer B (synchronous copy, can overlap with CUDA->staging A)
                    with T.section("cpu to staging B"):
                        sbuf_b.copy_(module_to_cuda.weight.data)  # BOTTLENECK

                    # Wait for staging buffer A to be ready so cuda_data_view can be reused
                    with T.section("wait staging A"):
                        event_a.synchronize()

                    # Staging buffer B to CUDA, non-blocking copy
                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_data_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    # Staging buffer A to CPU (synchronous, can overlap with staging B->CUDA)
                    with T.section("staging A to CPU"):
                        cpu_data_view.copy_(sbuf_a)  # BOTTLENECK

            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                # Update references
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b  # final sync event for CPU->CUDA copy

        else:
            # Use pinned memory for faster transfer (requires more memory)
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            # Copy weights to CPU
            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                with torch.cuda.stream(self.stream):
                    with T.section("cuda to cpu"):
                        module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                        event.record(self.stream)

            # CPU to CUDA
            for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    with T.section("wait cpu"):
                        self.stream.wait_event(event)

                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            # Update references
            for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)

            # Reuse released pinned buffers
            if not released_pinned_buffer[0].is_pinned():
                # First time: offloaded weights are not yet pinned, so allocate new buffers
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        if debug_print:
            print(f"[{self.block_type}] Weight swap timing at {self.debug_block_count - 1}:")
            for name, total in T.totals.items():
                print(f"  {name}: {total * 1000:.2f}ms")
            print(
                f"Overall time: {(time.perf_counter() - T.start_time) * 1000:.2f}ms, "
                f"total time in sections: {sum(T.totals.values()) * 1000:.2f}ms"
            )

        return sync_event

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            sync_event = self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            sync_event = None
        return sync_event

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} "
                    f"to {'CUDA' if self.cuda_available else 'device'}"
                )

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} "
                    f"to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available and sync_event is not None:
            # Does not wait on the CPU side; the log below should be immediate when pinned memory is used
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# ModelOffloader  (supports forward + optional backward offloading)
# ---------------------------------------------------------------------------

class ModelOffloader(Offloader):
    """Offloader with forward (and optionally backward) block-swap support."""

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # can be toggled to True for inference

        self.remove_handles = []
        if self.supports_backward:
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)
        # a backward that stops early - a LoRA on some blocks only (Fast Identity Mode, a Model Area) leaves the blocks
        # before them without gradients, so their hooks never fire and the blocks they would bring back stay on the
        # CPU for the next forward ("mat2 is on cpu", #302). Counted here; the next forward puts every block back first.
        self._blocks = list(blocks)
        self._bw_expected = len(self.remove_handles)
        self._bw_fired = 0
        self._trained_forward = False
        self._said_restore = False

    def set_forward_only(self, forward_only: bool):
        # Switching must wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    def remove_hooks(self):
        """Detach this offloader's backward hooks. Idempotent.

        Must be called BEFORE a block's offloader is replaced (e.g. when block
        swap is re-enabled at a different level around sample generation). Relying
        on __del__ is unsafe — GC isn't prompt, so the old offloader's hooks linger
        and fire alongside the new ones, double-swapping blocks and leaving base
        weights stranded on CPU for the next backward ("mat2 is on cpu" crash /
        corrupted gradients)."""
        for handle in self.remove_handles:
            handle.remove()
        self.remove_handles = []

    def __del__(self):
        self.remove_hooks()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")
            self._bw_fired += 1

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._trained_forward, self._bw_fired = False, 0

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # ensure weights are on device

        cpu_device = torch.device("cpu")
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first (buffers/non-weights go to device)
            weighs_to_device(b, cpu_device)  # then move weights back to cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if block_idx == 0 and not self.forward_only and self._trained_forward:
            if self._bw_fired < self._bw_expected:
                # the last backward stopped before reaching every hooked block: put each block where a fresh forward
                # expects it (the moves already queued first)
                for idx in list(self.futures.keys()):
                    self._wait_blocks_move(idx)
                if not self._said_restore:
                    self._said_restore = True
                    print(f"[{self.block_type}] block swap: the backward stops before the first blocks (a LoRA on some "
                          "blocks only) - their places are restored before each step", flush=True)
                self.prepare_block_devices_before_forward(self._blocks)
            self._trained_forward, self._bw_fired = False, 0
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            self._trained_forward = True
            # With backward enabled, do not swap beyond blocks_to_swap (those must stay on GPU)
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        # Forward-only offloading uses two strategies:
        #
        # 1. blocks_to_swap < num_blocks // 2  -->  no wrap-around.
        #    Fewer swaps; good for small blocks_to_swap or lightweight models.
        #
        # 2. blocks_to_swap >= num_blocks // 2  -->  wrap-around.
        #    Transfers all blocks in a rotating manner; good for large blocks_to_swap
        #    or heavyweight models where transfer time is small vs. compute.

        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < (self.num_blocks // 2):
            # Strategy 1: no wrap-around
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            # Strategy 2: wrap-around
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
