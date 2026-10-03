"""FamilyDriver: the one interface a new model family implements (the "driver" behind Fizgig's standard layer).

Fizgig's generic code - caching, training, previews, the LoRA layer, the fetcher, the GUI path and (later) the
workbench tools - talks to a family ONLY through its FamilyDescription (facts) and its FamilyDriver (model code).
A new family is: a description + a driver + its model package. Nothing else in Fizgig changes to add it.

Klein, Krea 2 and MiniMax H3 are not drivers; they keep their own code paths until this layer is proven.

Conventions every driver follows:
* Latents are (C, h, w) tensors in the family's normalised space (what the DiT is trained on).
* Conditioning is a dict of tensors per caption, keys chosen by the driver; the generic cache stores it as-is
  and hands the same dict back (batched, leading dim 1) to training_loss / generate.
* Images in and out are uint8 RGB numpy arrays (H, W, 3) / PIL images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    """One block of a model's LoRA map: a stable id (used in presets, sidecars and saved states), a display label
    in the model's own terms, and the dotted names (relative to the DiT) of the Linears it covers."""
    id: str
    label: str
    modules: list = field(default_factory=list)


@dataclass
class BlockGroup:
    """A named section of blocks (e.g. "Double blocks", "Text fusion", "Refiner", SDXL's "Input blocks")."""
    label: str
    blocks: list = field(default_factory=list)


class FamilyDriver:
    """Subclass per family. `description` is the family's FamilyDescription."""

    description = None

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path: str, device):
        """The diffusion transformer in bf16, frozen, ready for LoRA wrapping (gradient checkpointing available).
        INT8 / NF4 are applied afterwards by families/quant.py to the block map's Linears; the driver only loads."""
        raise NotImplementedError

    # ---- block swap (optional) --------------------------------------------------------------------
    def max_blocks_to_swap(self, dit=None) -> int:
        """How many blocks may stream between CPU and GPU; 0 = the family has no block swap."""
        return 0

    def enable_block_swap(self, dit, num_blocks: int, device, supports_backward: bool = True) -> None:
        """Stream `num_blocks` blocks: called with the model on CPU; leaves everything else on `device`."""
        raise NotImplementedError

    def block_swap_mode(self, dit, inference: bool) -> None:
        """Forward-only streaming for previews (inference=True), back to training layout after."""

    def load_vae(self, path: str, device):
        raise NotImplementedError

    def load_text_encoder(self, path: str, device):
        """The text encoder, for encode_text only - captioning is the shared captioner's job (Krea 2's Qwen3-VL-4B),
        so an encoder can leave out anything encoding does not use (an LM head, a vision tower)."""
        raise NotImplementedError

    def unload_text_encoder(self, te) -> None:
        """Free the encoder's VRAM (caching and preview-prompt encoding load it, then drop it)."""
        raise NotImplementedError

    def enable_gradient_checkpointing(self, dit, on: bool = True) -> None:
        raise NotImplementedError

    # ---- encoding (the generic cache calls these) -----------------------------------------------
    def encode_images(self, vae, images: list) -> list:
        """uint8 (H, W, 3) arrays, all the same size -> list of (C, h, w) latents."""
        raise NotImplementedError

    def encode_text(self, te, captions: list) -> list:
        """captions -> list of conditioning dicts (tensors on CPU)."""
        raise NotImplementedError

    # ---- edit training (optional) -----------------------------------------------------------------
    supports_references = False       # reference ("before") images: pair datasets and edit previews

    def load_reference_text_encoder(self, path: str, device):
        """The text encoder able to read reference images (encode_text_with_references)."""
        raise NotImplementedError

    def encode_text_with_references(self, te, captions: list, references: list) -> list:
        """captions + one list of uint8 (H, W, 3) reference images per caption, each already at the size its latent
        uses -> conditioning dicts. training_loss / generate get the references' latents as refs=."""
        raise NotImplementedError

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond: dict, generator, *, min_t: float = 0.0, max_t: float = 1.0,
                      refs=None, diff_ref=None, diff_weight: float = 0.0):
        """One training forward. latents (1, C, h, w) on device, cond = the cached dict (batched), refs = the pair's
        reference latents [(1, C, rh, rw), ...] (edit training) or None. diff_ref / diff_weight (image-pair
        sliders): weight each token's error by how much latents and diff_ref differ there, so the slider learns
        what changes between the poles and not what they share.
        Returns (loss tensor, info dict e.g. {"t": 0.63}). Owns the family's noise/target/timestep rules."""
        raise NotImplementedError

    # ---- prompt-pair sliders (optional) ------------------------------------------------------------
    def noise_latents(self, latents, generator, *, min_t: float = 0.0, max_t: float = 1.0) -> dict:
        """A noised training input for latents (1, C, h, w), drawn by the family's own timestep rule. Opaque to
        the caller; handed back to predict()."""
        raise NotImplementedError

    def predict(self, dit, state: dict, cond: dict):
        """The model's prediction at a noise_latents() state for this conditioning (batched dict)."""
        raise NotImplementedError

    # ---- sampling -------------------------------------------------------------------------------
    def initial_noise(self, seed: int, width: int, height: int):
        """The seed's starting noise (CPU float32), exactly what generate() draws for this seed. The workbench
        slerps two of these for seed travel and passes the result back as generate(noise=...)."""
        raise NotImplementedError

    def generate(self, dit, cond: dict, width: int, height: int, *, steps: int, seed: int, cfg: float = 1.0,
                 neg_cond: Optional[dict] = None, sigmas=None, options=(), noise=None, on_step=None, refs=None):
        """Denoise one image from noise (refs: reference latents for an edit, with conditioning from
        encode_text_with_references); returns latents in the driver's own layout (fed to decode).
        sigmas / options: an explicit schedule and driver-specific sampler options (e.g. from a speed LoRA's
        SamplingSettings); a driver ignores what it doesn't use. noise: a start from initial_noise() (or a blend of
        two) instead of the seed's. on_step(done, total): called before every step; it may raise to abort."""
        raise NotImplementedError

    def pad_conditioning(self, conds: list) -> list:
        """Optional (prompt travel): the conditioning dicts brought to one shape so they can be blended, e.g. padded
        with a validity mask. Tensors that are blended are floating point; booleans are combined as a union. A
        driver without it has no prompt travel."""
        raise NotImplementedError

    def decode(self, vae, latents, width: int, height: int):
        """-> PIL.Image (RGB)."""
        raise NotImplementedError

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None) -> list:
        """The model's LoRA structure in its OWN terms: ordered BlockGroups of Blocks. The workbench builds its
        slider panel, greying, presets and bake mapping from this, so a family with named areas (text fusion,
        refiners, SDXL-style input/middle/output blocks) overrides it. Default: one group of `n_blocks` numbered
        blocks from the description, each covering the description's LoRA modules (filtered to those in `dit`)."""
        d = self.description
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        blocks = []
        for i in range(d.n_blocks):
            mods = [f"{d.block_prefix}.{i}.{m}" for m in d.lora.block_modules]
            if names is not None:
                mods = [m for m in mods if m in names]
            blocks.append(Block(f"block_{i}", f"Block {i}", mods))
        return [BlockGroup("Blocks", blocks)]

    def alias_flat(self, flat: str):
        """Another trainer's name for one of this model's Linears (e.g. diffusers naming from OneTrainer or
        AI-Toolkit), flattened with dots as underscores -> this model's flattened name, or None. The loaders look the
        result up among the model's own Linears, so a family only lists its renames."""
        return None

    def lora_target_names(self, dit) -> list:
        """Dotted module names (relative to dit) of the Linears a LoRA wraps: every module in the block map."""
        return [m for g in self.block_map(dit) for b in g.blocks for m in b.modules]

    def encode_text_with_image(self, te, captions: list, image, megapixels: float = 1.0) -> list:
        """Captions conditioned on one PIL image through the text encoder's vision path (descriptions with
        preview_image=True) at about `megapixels` -> conditioning dicts, as encode_text."""
        raise NotImplementedError

    # ---- full fine-tune (optional: families/ft.py) ------------------------------------------------------------------
    def ft_spec(self, dit):
        """The family's fine-tune declaration (families.ft.FTSpec), or None: no fine-tune. Descriptions that set
        finetune=True return one."""
        return None

    def install_ft_streamer(self, dit, streamer) -> None:
        """Hand the fine-tune's block streamer to the model: it implements the block-swap interface the forward
        already calls (wait_for_block / submit_move_blocks_forward). The default suits the common
        `offloader` + `blocks_to_swap` convention; a family without block swap cannot stream (small cards then refuse
        up front)."""
        if not self.max_blocks_to_swap(dit):
            raise RuntimeError(f"{self.description.display_name} has no block swap, so a fine-tune cannot stream "
                               f"frozen blocks on this card - it needs more free VRAM")
        old = getattr(dit, "offloader", None)
        for fn in ("remove", "remove_hooks"):
            if old is not None and hasattr(old, fn):
                getattr(old, fn)()
        dit.offloader = streamer
        dit.blocks_to_swap = 1

    def compile_targets(self, dit):
        """The ModuleList of transformer blocks torch.compile replaces in place (descriptions with compiles=True).
        The DiT's forward must call a block that has `_handles_checkpointing` directly, without its own checkpoint."""
        raise NotImplementedError

    def compile_blocks(self, dit, boundary: str = "inside", blocks_to_swap: int = 0) -> None:
        """torch.compile the transformer blocks, in place, after every adapter has patched the forwards. `boundary`
        places the gradient checkpoint inside or outside the compiled region. Refuses (logs, runs eager) what it
        cannot compile: block swap, no triton, no host C compiler."""
        from fizgig.families.compile import compile_blocks
        compile_blocks(dit, self.compile_targets(dit), blocks_to_swap, boundary=boundary,
                       fullgraph=self.description.compile_fullgraph)

    def compile_plan(self, mode: str, total_steps: int, precision: str, blocks_to_swap: int,
                     mp: float = 0.25) -> tuple:
        """(False | "inside" | "outside", why) for Compile Blocks `mode` ("auto" / "on" / "outside"). On always
        compiles, at the description's boundary; Auto compiles once the run is longer than the description's
        measured payback for this base precision, and never where the machine cannot (compile_blocker)."""
        d = self.description
        if mode == "outside":
            return "outside", ""
        if mode != "auto":
            return d.compile_boundary, ""
        from fizgig.utils.capabilities import compile_blocker
        blocked = compile_blocker(blocks_to_swap)
        if blocked:
            return False, blocked
        pay = (d.compile_payback_steps or {}).get(precision)
        if not pay:
            return False, (f"not measured for the {precision.upper()} base on {d.display_name}; Compile Blocks On "
                           "still compiles")
        if total_steps < pay:
            return False, (f"{total_steps} steps on the {precision.upper()} path — compile pays back after ~{pay}, "
                           "so this run is quicker uncompiled")
        return d.compile_boundary, (f"{total_steps} steps on the {precision.upper()} path — compile pays back within "
                                    f"~{pay} steps and this run is longer")

    def on_base_loaded(self, dit, precision: str, device) -> None:
        """Called once after the base is loaded and quantised (families/quant.load_base): a driver can bind
        hardware-specific forwards to this model instance here. Default: nothing."""

    def quant_target_names(self, dit) -> list:
        """The Linears an INT8 / NF4 base quantises (families/quant.py). Default: the LoRA targets. A family whose
        LoRA reaches layers that must stay bf16 (Krea 2's text fusion and I/O layers) narrows it."""
        return self.lora_target_names(dit)

    def block_of(self, module_name: str) -> Optional[str]:
        """The block id a module belongs to, or None for modules outside the map (e.g. a speed LoRA's extras)."""
        idx = getattr(self, "_block_index", None)
        if idx is None:
            idx = self._block_index = {m: b.id for g in self.block_map() for b in g.blocks for m in b.modules}
        return idx.get(module_name)
