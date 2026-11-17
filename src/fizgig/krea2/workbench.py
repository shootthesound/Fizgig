"""Krea 2 inference: reserve activation memory and offload between pipeline stages."""
import gc
import logging
import sys
from itertools import chain

import torch

from fizgig.families import quant
from fizgig.families.workbench import WorkbenchEngine, _free_vram_gb
from fizgig.krea2.safetensors_utils import release_mapped_pages

logger = logging.getLogger(__name__)
GIB = 1024 ** 3


def _block_weights(block):
    # Match the existing offloader; LoRAFactor adapters and scale buffers stay resident.
    seen = set()
    for module in block.modules():
        if module.__class__.__name__.endswith("Linear"):
            weight = getattr(module, "weight", None)
            if weight is not None and id(weight) not in seen:
                seen.add(id(weight))
                yield weight


class Krea2WorkbenchEngine(WorkbenchEngine):
    def ensure_pipeline(self, dit_path, *args, precision="auto", blocks_to_swap="auto", device="cuda", **kw):
        if self.pipeline is not None:
            return
        if not torch.version.hip:
            self._auto_swap = False
            self.driver._file_backed_inference = False
            return super().ensure_pipeline(dit_path, *args, precision=precision,
                                           blocks_to_swap=blocks_to_swap, device=device, **kw)
        self._auto_swap = str(blocks_to_swap).lower().startswith("auto") and torch.device(device).type == "cuda"
        self._reserve_extra = 0
        if self._auto_swap:
            if precision == "auto":
                from fizgig.krea2.utils import dit_layout
                # "bf16" means no additional quantisation; a stored FP8 checkpoint stays FP8.
                precision = "bf16" if dit_layout(dit_path) == "fp8" else "int8"
            blocks_to_swap = self.driver.max_blocks_to_swap() if precision != "nf4" else 0
        elif str(blocks_to_swap).lower().startswith("auto"):
            blocks_to_swap = 0
        if torch.device(device).type == "cuda":
            kw["lowmem"] = True  # Text encoding, sampling and VAE decoding get separate VRAM budgets.
        self.driver._file_backed_inference = (bool(torch.version.hip) and sys.platform == "linux"
                                              and torch.device(device).type == "cuda" and precision == "bf16")
        super().ensure_pipeline(dit_path, *args, precision=precision, blocks_to_swap=blocks_to_swap,
                                device=device, **kw)
        self._auto_swap = self._auto_swap and precision != "nf4" and not getattr(self.dit, "_nf4_quantized", False)
        release_mapped_pages(getattr(self.dit, "_cpu_weight_masters", {}).values())

    def _park_dit(self, where):
        if not torch.version.hip:
            return super()._park_dit(where)
        if self.dit is None:
            return
        if torch.device(where).type == "cpu":
            offloader = getattr(self.dit, "offloader", None)
            if offloader is not None:
                offloader.set_forward_only(True)  # Drain pending transfers before moving their weights.
            masters = getattr(self.dit, "_cpu_weight_masters", {})
            if masters:
                torch.cuda.synchronize()  # File pages must outlive all pending DMA reads.
                for parameter, master in masters.items():
                    parameter.data = master
            quant.move(self.dit, "cpu")
            release_mapped_pages(masters.values())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self.swapped:
            self.dit.move_to_device_except_swap_blocks(torch.device(where))
            self.driver.block_swap_mode(self.dit, inference=True)
        elif getattr(self.dit, "_cpu_weight_masters", None):
            self.dit.move_to_device_except_swap_blocks(torch.device(where))
        else:
            quant.move(self.dit, where)
        if torch.device(where).type != "cpu":
            release_mapped_pages(getattr(self.dit, "_cpu_weight_masters", {}).values())

    def _close_offloader(self):
        offloader = getattr(self.dit, "offloader", None)
        if offloader is not None:
            offloader.set_forward_only(True)
            offloader.remove_hooks()
            offloader.thread_pool.shutdown(wait=True)
            self.dit.offloader = None

    def _finish_decode(self):
        # Keep the DiT parked between images; the next render restores its streaming layout.
        if torch.version.hip and torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        else:
            super()._finish_decode()

    def _set_swap(self, count):
        if count == self.swapped:
            return
        self._invalidate_activation_cache()
        self._park_dit("cpu")
        self._close_offloader()
        self.dit.blocks_to_swap = 0
        self.swapped = count
        if count:
            self.driver.enable_block_swap(self.dit, count, self.device, supports_backward=False)
            self.driver.block_swap_mode(self.dit, inference=True)
        else:
            self._park_dit(self.device)

    def render(self, cond, width, height, seed, **kwargs):
        device = torch.device(self.device)
        rocm = device.type == "cuda" and bool(torch.version.hip)
        previous_fraction = allocator_limit = None
        try:
            if rocm and torch.cuda.is_available():
                if device.index is None:
                    device = torch.device("cuda", torch.cuda.current_device())
                torch.cuda.empty_cache()
                previous_fraction = torch.cuda.get_per_process_memory_fraction(device)
                free, total = torch.cuda.mem_get_info(device)
                # Respect other processes' allocations without reserving additional desktop VRAM.
                available = free + torch.cuda.memory_reserved(device)
                if available <= 0:
                    raise torch.OutOfMemoryError("Krea 2 has no available GPU memory")
                fraction = min(previous_fraction, available / total)
                torch.cuda.set_per_process_memory_fraction(fraction, device)
                allocator_limit = fraction * total
            if getattr(self, "_auto_swap", False):
                torch.cuda.empty_cache()
                sizes = [sum(w.numel() * w.element_size() for w in _block_weights(b)) for b in self.dit.blocks]
                resident = sum(w.numel() * w.element_size() for b in self.dit.blocks
                               for w in _block_weights(b) if w.device.type == "cuda")
                cpu_bytes = sum(t.numel() * t.element_size()
                                for t in chain(self.dit.parameters(), self.dit.buffers()) if t.device.type == "cpu")
                # Parked embedders, adapters and buffers also need room when sampling resumes.
                pending = max(0, cpu_bytes - (sum(sizes) - resident))
                width, height = self.driver._grid(width, height)
                # ComfyUI's efficient-attention estimate: latent area * dtype bytes * .01 * Krea's factor 2.2 MiB.
                # shortcut: this reserve is estimated; calibrate it with full-model RX 9070 XT measurements.
                reserve = int((width // 8) * (height // 8) * 2 * .01 * 2.2 * 1024 ** 2 + 1.5 * GIB)
                reserve += self._reserve_extra
                free = _free_vram_gb() * GIB
                if allocator_limit is not None:
                    free = min(free, max(0, allocator_limit - torch.cuda.memory_reserved(device)))
                budget = int(free) + resident - reserve - pending
                cap = self.driver.max_blocks_to_swap(self.dit)
                count = cap
                for n in range(cap + 1):
                    # The swap setup briefly places one extra block on the device.
                    if sum(sizes[:len(sizes) - n]) + (max(sizes) if n else 0) <= budget:
                        count = n
                        break
                logger.info("Krea 2 Auto: %dx%d, %.2f GiB activation reserve, %d/%d blocks offloaded",
                            width, height, reserve / GIB, count, len(sizes))
                self._set_swap(count)
            if rocm:
                self._park_dit(self.device)
            return super().render(cond, width, height, seed, **kwargs)
        finally:
            try:
                if rocm:
                    self._park_dit("cpu")
            finally:
                if previous_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(previous_fraction, device)

    def _sample_with_memory(self, sample):
        while True:
            try:
                return sample()
            except torch.OutOfMemoryError:
                cap = self.driver.max_blocks_to_swap(self.dit)
                if not getattr(self, "_auto_swap", False) or self.swapped >= cap:
                    raise
            # Retry outside the except block so the failed forward's tensors can be released.
            count = min(cap, self.swapped + 2)
            self._reserve_extra += sum(w.numel() * w.element_size()
                                       for b in self.dit.blocks[len(self.dit.blocks) - count:
                                                                len(self.dit.blocks) - self.swapped]
                                       for w in _block_weights(b))
            logger.warning("Krea 2 sampling ran out of VRAM; retrying the same seed with %d blocks offloaded", count)
            self._set_swap(count)

    def reset(self):
        self._close_offloader()
        super().reset()
