"""Slider state data model for the Repair Studio.

A SliderState is the *complete* configuration of the studio at a point in time:
per-block primary/donor enables and strengths, plus the preview parameters
(seed, prompt, resolution). It serializes to/from JSON for preset storage.

The state is the SOLE input to RepairEngine.generate_preview(). Keeping it as
a single dataclass lets v2 diff successive states cheaply for cache invalidation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# Klein 9B block layout: 8 double + 24 single = 32 sliders.
DOUBLE_BLOCK_COUNT = 8
SINGLE_BLOCK_COUNT = 24


def all_block_ids() -> List[str]:
    ids = [f"double_{i}" for i in range(DOUBLE_BLOCK_COUNT)]
    ids += [f"single_{i}" for i in range(SINGLE_BLOCK_COUNT)]
    return ids


def block_regex(block_id: str) -> str:
    """Return a regex matching a LoRA module's lora_name for this block.

    LoRA module names follow `lora_unet_<double|single>_blocks_<N>_<linear>`.
    set_module_multiplier_by_pattern uses regex.search on lora_name, so anchoring
    on the underscore after the index is enough to avoid `single_1_` matching
    `single_10_*`.
    """
    kind, idx = block_id.split("_")
    return rf"{kind}_blocks_{idx}_"


@dataclass
class BlockState:
    primary_enabled: bool = True
    primary_strength: float = 1.0
    donor_enabled: bool = True
    donor_strength: float = 0.0


@dataclass
class SliderState:
    blocks: Dict[str, BlockState] = field(default_factory=dict)
    seed: int = 42
    prompt: str = ""
    preview_width: int = 512
    preview_height: int = 512
    # Reference image (Klein is an edit model): conditions the preview on a real
    # image. ref_megapixels caps the ref's encode resolution; ref_strength scales
    # the encoded ref latent (1.0 = stock, 0.85 ≈ Klein sweet spot, 0 = off).
    ref_image_path: str = ""
    ref_megapixels: float = 1.0
    ref_strength: float = 1.0
    # MiniMax H3 video mode (Repair Studio, 3 Sep 2026). preview_frames: pixel frames on the
    # 17n+5 grid (1 = still); 0 = the engine's default. keyframes: [(frame_index, latent)]
    # built by H3RepairEngine.encode_keyframe — TENSORS, so deliberately NOT serialised
    # (presets never carry a conditioning image); copied by reference.
    preview_frames: int = 0
    keyframes: Any = None

    @classmethod
    def default_klein9b(cls) -> "SliderState":
        return cls(blocks={bid: BlockState() for bid in all_block_ids()})

    @classmethod
    def default_krea2(cls) -> "SliderState":
        """Krea 2 layout: 28 main blocks + 4 txtfusion (see repair_studio.krea2_blocks).
        Same BlockState model; only the block-id set differs. Preview res 512 (keeps the
        Turbo Preview activation cache VRAM-feasible alongside the resident Turbo)."""
        from fizgig.repair_studio.krea2_blocks import all_block_ids_krea2
        return cls(blocks={bid: BlockState() for bid in all_block_ids_krea2()})

    @classmethod
    def default_h3(cls) -> "SliderState":
        """MiniMax H3 layout: 50 main blocks + 2 token-refiner (see repair_studio.h3_blocks).
        Preview res 768 — H3's native canvas short edge; the engine renders a 22-frame clip
        at this size and shows its middle frame (a still is H3's most out-of-distribution
        render, so the clip regime is the honest instrument)."""
        from fizgig.repair_studio.h3_blocks import all_block_ids_h3
        return cls(blocks={bid: BlockState() for bid in all_block_ids_h3()},
                   preview_width=768, preview_height=768)

    def to_json(self) -> Dict[str, Any]:
        return {
            "blocks": {bid: asdict(bs) for bid, bs in self.blocks.items()},
            "seed": self.seed,
            "prompt": self.prompt,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
            "ref_image_path": self.ref_image_path,
            "ref_megapixels": self.ref_megapixels,
            "ref_strength": self.ref_strength,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "SliderState":
        blocks = {bid: BlockState(**bs) for bid, bs in d.get("blocks", {}).items()}
        if not blocks:
            blocks = {bid: BlockState() for bid in all_block_ids()}
        return cls(
            blocks=blocks,
            seed=int(d.get("seed", 42)),
            prompt=str(d.get("prompt", "")),
            preview_width=int(d.get("preview_width", 512)),
            preview_height=int(d.get("preview_height", 512)),
            ref_image_path=str(d.get("ref_image_path", "")),
            ref_megapixels=float(d.get("ref_megapixels", 1.0)),
            ref_strength=float(d.get("ref_strength", 1.0)),
        )

    def copy(self) -> "SliderState":
        """Fast deep copy without JSON serialization."""
        return SliderState(
            blocks={bid: BlockState(bs.primary_enabled, bs.primary_strength,
                                    bs.donor_enabled, bs.donor_strength)
                    for bid, bs in self.blocks.items()},
            seed=self.seed, prompt=self.prompt,
            preview_width=self.preview_width, preview_height=self.preview_height,
            ref_image_path=self.ref_image_path, ref_megapixels=self.ref_megapixels,
            ref_strength=self.ref_strength,
            preview_frames=self.preview_frames, keyframes=self.keyframes,
        )

    def mutate(self, active_blocks: set, num_mutations: int = 3,
               intensity: float = 0.5, structure: float = 1.0,
               anchor: str = "double_0") -> "SliderState":
        """Return a new SliderState with random perturbations to num_mutations blocks.

        active_blocks: set of block_ids the LoRA touches (skip others).
        intensity: 0.0 = tiny nudges (±0.2), 1.0 = bold moves (±3.0).
        structure: 0.0 = no structural changes, 1.0 = full structural (off/invert/extreme).
                   Scales the magnitude of the guaranteed structural change on the first block.
        anchor: the composition-anchor block id that receives the structural change first and is
                never disabled (Klein: double_0; Krea 2: block_0).
        """
        import random
        new = self.copy()
        candidates = [bid for bid in new.blocks if bid in active_blocks]
        if not candidates:
            return new
        # When structure > 0, ensure the anchor block is first so it gets the structural change
        n = min(num_mutations, len(candidates))
        if structure > 0.05 and anchor in candidates:
            others = [bid for bid in candidates if bid != anchor]
            chosen = [anchor] + random.sample(others, min(n - 1, len(others)))
        else:
            chosen = random.sample(candidates, n)
        for i, bid in enumerate(chosen):
            bs = new.blocks[bid]
            magnitude = 0.2 + intensity * 2.8
            # First mutated block: structural change scaled by structure parameter
            # Never disables the anchor — only inverts or pushes to extreme
            if i == 0 and intensity > 0.3 and structure > 0.05:
                structural = random.choice(["invert", "extreme"])
                if structural == "invert":
                    # Scale the inversion by structure
                    bs.primary_strength = max(-3.0, min(3.0,
                        bs.primary_strength * (1.0 - 2.0 * structure)))
                else:
                    # Blend toward extreme
                    target = random.choice([-3.0, 3.0])
                    bs.primary_strength = max(-3.0, min(3.0,
                        bs.primary_strength + (target - bs.primary_strength) * structure))
            else:
                delta = random.uniform(-magnitude, magnitude)
                bs.primary_strength = max(-3.0, min(3.0, bs.primary_strength + delta))
                if random.random() < 0.35 * intensity:
                    bs.primary_enabled = not bs.primary_enabled
        return new

    def diff_blocks(self, other: "SliderState") -> List[str]:
        """Block ids whose primary/donor enable or strength differs from `other`.
        Used by the UI to feed RepairEngine.mark_blocks_changed (v2 hook)."""
        changed = []
        for bid, bs in self.blocks.items():
            ob = other.blocks.get(bid)
            if ob is None or bs != ob:
                changed.append(bid)
        return changed
