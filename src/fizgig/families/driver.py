"""FamilyDriver: the one interface a new model family implements (the "driver" behind Fizgig's standard layer).

Fizgig's generic code - caching, training, previews, the LoRA layer, the fetcher, the GUI path and the workbench
tools - talks to a family ONLY through its FamilyDescription (facts) and its FamilyDriver (model code).
A new family is: a description + a driver + its model package. Nothing else in Fizgig changes to add it. Every
family Fizgig trains (Klein 9B, MiniMax H3, Krea 2, Qwen Image 2.1, SDXL, Anima) is one; docs/drivers/ is the guide.

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

    # ---- files loaded by name (tokenizers, processors, configs): cache first, so a cached setup runs offline -----
    @staticmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        """`cls.from_pretrained(repo_id, **kwargs)` from the local cache, the Hub only when nothing is cached. Plain
        from_pretrained asks the Hub on every call, so a fully cached tokenizer still fails offline or when the Hub
        rate-limits the machine. List the repo and files in the description's `helper_files` so the model
        downloader fetches them up front."""
        from fizgig.utils.hf_cache import from_pretrained_cache_first
        return from_pretrained_cache_first(cls, repo_id, **kwargs)

    @staticmethod
    def helper_dir(repo_id: str) -> str:
        """The cached folder of a `helper_files` repo, for loaders that take a path or config repo (diffusers'
        from_single_file config=), else the repo id itself (the first run then fetches it)."""
        from fizgig.utils.hf_cache import cached_snapshot_dir
        return cached_snapshot_dir(repo_id) or repo_id

    def enable_gradient_checkpointing(self, dit, on: bool = True) -> None:
        raise NotImplementedError

    # ---- encoding (the generic cache calls these) -----------------------------------------------
    def encode_images(self, vae, images: list) -> list:
        """uint8 (H, W, 3) arrays, all the same size -> list of (C, h, w) latents."""
        raise NotImplementedError

    def encode_text(self, te, captions: list) -> list:
        """captions -> list of conditioning dicts (tensors on CPU)."""
        raise NotImplementedError

    def cache_stage(self, stage: str, datasets, args, device, aux: dict) -> bool:
        """Optional: cache `stage` ("latents" / "text") over the loaded datasets in the family's own layout and return
        True. args carries the shared cache flags (model, skip_existing, keep_cache, batch_size, num_workers, slider);
        aux the --aux KEY=VALUE extras. Default False: the shared encode_images / encode_text path runs."""
        return False

    def media_problem(self, path: str) -> str:
        """Why this clip or sound file cannot train as it is ("" = fine), beyond the description's clip_spec (the
        launch checks every clip against the spec itself, families/clips.py). Only called for families whose media
        include "clip" / "voice"."""
        return ""

    def clip_bucket_cap(self, free_gb: float, width: int, height: int) -> tuple:
        """The largest (width, height) of this shape a clip can be cached at in `free_gb` of free VRAM - encoding a
        clip holds many frames at once, so a size that suits the photos may not fit. Default: no cap."""
        return width, height

    # ---- family training options (optional) ------------------------------------------------------
    options = {}

    def set_options(self, options: dict) -> None:
        """The run's --family_option KEY=VALUE pairs, before the dataset is built (an option may shape it)."""
        self.options = dict(options)

    def prepare_training(self, dit, group, net=None) -> None:
        """After the DiT and the LoRA (`net`, the FamilyLoRA) are built, before the first step: install model-side
        training state (H3: TREAD, the caption-dropout embed, the preview Turbo). Default: nothing."""

    def step_frozen_blocks(self, batch: dict):
        """Block ids whose trainable weights sit out this item's step (per-modality routing). Default: none."""
        return ()

    def expand_train_blocks(self, items: list) -> Optional[list]:
        """--train_blocks as typed (block ids, or a family's own spec such as H3's "3-12, 22") -> block ids, or None
        for every block. Default: the ids as given."""
        return list(items) or None

    def legacy_state_order(self, dit) -> Optional[list]:
        """Module names in the order an older trainer for this family laid out its parameters (each module's down
        then up weight) - an untagged saved state's optimizer moments and EMA shadow are remapped from it into the
        family LoRA's own order (sorted module names). None = the states already share the family LoRA's order."""
        return None

    def after_optimizer_step(self) -> None:
        """Called after every optimizer step (the H3 adapter-relative LR ramp reads the adapter's new size)."""

    def run_metadata(self) -> dict:
        """Extra ss_* keys this run's family options put in every saved LoRA. Default: none."""
        return {}

    def step_policy(self, batch: dict, epoch: int) -> tuple:
        """(skip, lr_multiplier) for this item's step in 1-based `epoch`: skip = no forward, no loss, no record
        (H3's per-category retirement, "stop"); the multiplier scales the optimizer LR for the step (H3's "anchor",
        10%), averaged over an accumulation window. Default: (False, 1.0)."""
        return False, 1.0

    def batch_cond(self, batch: dict, device) -> dict:
        """The conditioning dict training_loss gets for one loaded item: the cached `cond__` entries, batched. A
        family that keeps its own cache layout (H3's hidden_states + audio rows) maps its keys here."""
        return {k[len("cond__"):]: v.to(device) for k, v in batch.items() if k.startswith("cond__")}

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
    def slider_setup(self, group) -> None:
        """A slider run's data is known (an image-pair group, or a prompt slider's practice bank)."""

    def still_renders(self):
        """A context in which generate() renders stills, whatever the previews are set to (a prompt slider's practice
        pictures). Families whose previews are always stills need nothing."""
        import contextlib
        return contextlib.nullcontext()

    def slider_preview(self, frames, multipliers):
        """One preview's decoded results at each strength -> what save_preview writes, or None for the shared
        side-by-side still strip (a clip family lays its clips side by side)."""
        return None

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
                 neg_cond: Optional[dict] = None, sigmas=None, options=(), noise=None, on_step=None, refs=None,
                 frames: int = 1, audio: bool = False):
        """Denoise one image from noise (refs: reference latents for an edit, with conditioning from
        encode_text_with_references); returns latents in the driver's own layout (fed to decode).
        sigmas / options: an explicit schedule and driver-specific sampler options (e.g. from a speed LoRA's
        SamplingSettings); a driver ignores what it doesn't use. noise: a start from initial_noise() (or a blend of
        two) instead of the seed's. on_step(done, total): called before every step; it may raise to abort.
        frames / audio (video families): a clip of `frames` frames, with sound - callers pass them only when they
        ask for more than a still, so a still-only driver never sees them."""
        raise NotImplementedError

    def pad_conditioning(self, conds: list) -> list:
        """Optional (prompt travel): the conditioning dicts brought to one shape so they can be blended, e.g. padded
        with a validity mask. Tensors that are blended are floating point; booleans are combined as a union. A
        driver without it has no prompt travel."""
        raise NotImplementedError

    def decode(self, vae, latents, width: int, height: int):
        """-> PIL.Image (RGB). A video family may return its own clip object instead (frames + sound), which its
        save_preview writes."""
        raise NotImplementedError

    # set by the workbench on a card with room: decode / decode_audio leave the VAE on the GPU between calls (training
    # leaves it False, so the VAE never sits in VRAM beside the training model between previews)
    keep_vae_resident = False

    def decode_audio(self, vae, audio):
        """A video family's sound on its own: generate()'s audio -> waveform [channels, L] (float, -1..1, on CPU), or
        None (no sound decoder configured). The workbench decodes a clip's frames and its sound separately. Default:
        no sound."""
        return None

    def save_preview(self, result, path: str) -> list:
        """Write one decoded preview at `path` (the gallery's <name>_e<epoch>_<idx>_<timestamp>_<seed>.png) and return
        the files written. Default: the PNG. A video family writes its clip contract around it (the PNG last, as
        the gallery's trigger)."""
        result.save(path)
        return [path]

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

    def convert_lora_state_dict(self, sd: dict) -> dict:
        """A LoRA file's tensors before the family LoRA reads them - a family whose other trainers write a layout
        the generic reader cannot map (Klein: diffusers split q/k/v fused into its combined qkv) converts here.
        Default: as is."""
        return sd

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

    def ft_backend(self, dit, device, src, group):
        """The fine-tune's model-side backend (families.ft.SharedBackend's surface), or None for the shared one. A
        family whose base file and trunk differ from the shared bf16 + NF4 path brings its own (H3)."""
        return None

    def ft_card_plan(self, path, free_gb, mp=None, options=None, max_parts=0):
        """The fine-tune card's estimate before anything loads: (windows, stream) for this file on a card with
        `free_gb` free, or None. Default: the shared planner over the file's header (families.ft.plan_from_file).
        max_parts: the Training tab's cap on parts per window (0 = as many as fit)."""
        from fizgig.families.ft import plan_from_file
        plan = plan_from_file(path, self.ft_spec(None), free_gb, mp=mp, max_parts=max_parts)
        return (plan[0], plan[1]) if plan else None

    def ft_cycle(self, cycle: int, offset: int, total: int) -> None:
        """The fine-tune's rotation cycle is known (`cycle` epochs; `offset` epochs done before this leg; `total`
        epochs this leg runs): a family whose options land on epochs (H3's retirement) snaps them here."""

    def ft_source_unfit(self, path):
        """Why `path` cannot be fine-tuned by this family, or None. Default: the shared rule (a bf16 file)."""
        from fizgig.families.ft import source_unfit_reason
        return source_unfit_reason(path)

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

    def frozen_file_added(self, dit, path: str, strength: float, role: str) -> None:
        """A frozen LoRA file (role "adapter": on for training, off in previews; "context": on for both; "speed": the
        preview speed LoRA, on for previews only) has been
        added to the family LoRA. A family whose LoRAs carry weights the family LoRA cannot wrap (H3's AdaLN rows on
        the pruned base, injected at run time) applies them here."""

    # ---- room beside the training model (preview decode, override encode) ----------------------------
    def park_for(self, dit, device, need_gb, purpose: str):
        """Make `need_gb` free next to the resident training model (None = the driver's own figure for a preview
        decode). None (the default) = the shared rule (small cards park the whole DiT on CPU for a decode; an encode
        parks it when the text encoder does not fit); anything else is a token for unpark, the driver having done
        (or decided against) its own park - H3 parks only as many tail blocks as are missing, ring-aware."""
        return None

    def unpark(self, dit, device, token) -> None:
        """Undo park_for."""

    # ---- training previews on the preview checkpoint (descriptions with train_preview_checkpoint) ----------------
    def park_for_preview(self, dit, device):
        """Free VRAM on the training model for the preview checkpoint; returns a token for unpark_after_preview."""
        raise NotImplementedError

    def load_preview_checkpoint(self, path, device, int8=False):
        """The preview checkpoint, frozen, ready to render beside the parked training model -> (model, swapped
        blocks)."""
        raise NotImplementedError

    def unpark_after_preview(self, dit, device, token) -> None:
        """Put the training model back exactly as training had it."""
        raise NotImplementedError

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

    def plan_run(self, precision: str, blocks_to_swap: int, *, group, run: dict) -> Optional[tuple]:
        """The family's own Auto plan, or None for the shared one (quant.plan over the description's train_memory).
        Called when the precision is "auto" or the swap is -1; `run` carries what the plan may weigh (dit_path,
        network_type / dim, lokr_factor, optimizer_type, training_adapter, context_lora_path, ema_decay); preview
        settings a family plans for reach it as its own options (set_options). Returns (precision, blocks_to_swap,
        why)."""
        return None

    def load_planned(self, path, device, precision: str, blocks_to_swap: int) -> Optional[tuple]:
        """Load the base at `precision` with `blocks_to_swap` streamed, the driver's own way - (dit, swapped) - or
        None for the shared load (quant.load_base: load_dit, quantise, enable_block_swap)."""
        return None

    def auto_uncompiled_precision(self, dit_path: str, precision: str) -> Optional[str]:
        """A precision that beats Auto's pick when this run is not compiled, or None. Klein: its INT8 base is slower
        than BFL's fp8 file uncompiled, so an fp8 file trains as it is (the old Klein trainer's behaviour)."""
        return None

    def compile_plan(self, mode: str, total_steps: int, precision: str, blocks_to_swap: int,
                     mp: float = 0.25) -> tuple:
        """(False | "inside" | "outside", why) for Compile Blocks `mode` ("auto" / "on" / "outside"). On always
        compiles, at the description's boundary; Auto compiles once the run is longer than the description's
        measured payback for this base precision, and never where the machine cannot (compile_blocker)."""
        d = self.description
        if mode == "outside":
            return "outside", ""
        if mode != "auto":
            b, why = self._compile_fit(precision, mp, d.compile_boundary)
            if not b:                       # On still compiles: the leanest boundary, with a warning
                return "outside", ("on: " + why.replace(" - running uncompiled", "") +
                                   " - compiling with the checkpoint outside the graph anyway (it may run out of "
                                   "memory; Off or Auto avoid that)")
            return b, (why if b != d.compile_boundary else "")
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
        b, fit = self._compile_fit(precision, mp, d.compile_boundary)
        if not b:
            return False, fit
        return b, (f"{total_steps} steps on the {precision.upper()} path — compile pays back within ~{pay} steps and "
                   f"this run is longer" + (f"; {fit}" if fit else ""))

    def _compile_fit(self, precision, mp, preferred):
        """(boundary, why) by the description's measured compiled peaks (compile_memory) against free VRAM: the
        preferred boundary if it fits, else the checkpoint outside the graph, else (False, why). A family without
        compiled figures keeps its boundary."""
        mem = (self.description.compile_memory or {}).get(precision)
        if not mem:
            return preferred, ""
        from fizgig.families.quant import _peak, free_vram_gb
        budget = free_vram_gb() - 1.5
        order = [preferred] + [b for b in ("inside", "outside") if b != preferred]
        for b in order:
            if b in mem and _peak((mem[b], 0.0), mp) <= budget:
                return b, ("" if b == preferred else
                           f"the checkpoint goes {b} the compiled region ({_peak((mem[b], 0.0), mp):.1f} GB at "
                           f"{mp:.2f} MP, measured)")
        need = min(_peak((v, 0.0), mp) for v in mem.values())
        return False, (f"compiled it needs ~{need:.1f} GB at {mp:.2f} MP and {budget + 1.5:.1f} GB is free - running "
                       f"uncompiled")

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
