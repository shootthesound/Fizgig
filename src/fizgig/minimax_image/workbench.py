"""MiniMax H3 Image's Studio engine: the shared workbench engine, with Auto precision taken from the description's
measured peaks (train_memory, which include a 2048x2048 render).

The shared rule sizes precision from the file as if it were bf16 (int8 at half of it). H3 Image's file is int8
already (21 GB), so that rule put int8 with no block swap on a 16 GB card and the render spilled out of VRAM."""
import math

from fizgig.families.workbench import WorkbenchEngine, _free_vram_gb


class MiniMaxH3ImageWorkbenchEngine(WorkbenchEngine):

    def ensure_pipeline(self, *a, precision="auto", blocks_to_swap=0, **kw):
        """Auto: int8 where its measured peak fits the free VRAM, else 4-bit (quant.plan, as training's Auto), with
        the blocks that base still needs swapped on top of the given count (Preferences' inference block swap)."""
        if self.pipeline is None and precision == "auto":
            from fizgig.families import quant
            free = _free_vram_gb()
            precision, _, why = quant.plan(self.desc, self.driver, "auto", -1, free_gb=free)
            peaks, per = self.desc.train_memory.get(precision, ((), 0.0))
            if peaks and per > 0:
                peak = max(gb for _, gb in peaks)
                short = peak - (free - 1.5) - int(blocks_to_swap or 0) * per
                if short > 0:
                    blocks_to_swap = min(self.driver.max_blocks_to_swap(),
                                         int(blocks_to_swap or 0) + math.ceil(short / per))
            import logging
            logging.getLogger(__name__).info("MiniMax H3 Image Studio: %s, %d blocks swapped (%s)", precision,
                                             int(blocks_to_swap or 0), why)
        return super().ensure_pipeline(*a, precision=precision, blocks_to_swap=blocks_to_swap, **kw)
