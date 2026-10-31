"""The standard layer's workbench engine: Repair Studio and Royale for any described family.

One engine over a family's driver (model code) and the family LoRA layer (adapters, block controls, bake). It speaks
the same protocol as the H3 engine (repair_studio/h3_engine.py), so the tabs drive
it unchanged: ensure_pipeline, load_primary / load_donor / unload_donor, swap_primary_weights, apply_state,
generate_preview, generate_baseline, request_cancel / clear_cancel, reset, plus the primary_* / donor_* attributes.

Previews run with the family's speed LoRA (the description's preview_speed_lora) when its file is set and the user
picks it, else with the family's default sampling. Memory follows the trainer's rules: the text encoder loads only
when a prompt changes, and the DiT parks on CPU while it runs whenever both would not fit; on cards under 20 GB the
DiT also parks for the VAE decode.

Epoch blend / LoRA mix: several files as one LoRA, each at its own weight and all under that LoRA's block sliders -
exact, since each file is its own adapter, and the bake rank-concatenates them into one file. The files are a run's
saved epochs at EMA weights (find_epochs + ema_weights) or any LoRAs of the family at amounts the user sets
(set_sources, the primary's LoRA mix). They load as a mix first needs them (at most MAX_EPOCHS_LOADED per LoRA stay
on the card).

Model EQ: a LoRA made from the model's own weights (scripts/extract_model_lora.py, marker ss_fizgig_model_lora) maps
each block's slider s to (s - 1) x its gain - 1.0 and an unticked block are the stock model. Primary, donor or in a
mix alike; a save bakes the mapped strengths into an ordinary LoRA without the marker.

Turbo Preview (description.activation_cache, switched by the tab's tick through `turbo_preview`): a slider-only change
replays the blocks before the earliest changed one from the last render (families/act_cache.py).

A LoRA's load strength (state.primary_scale / donor_scale) scales the whole file and every block slider is relative
to it. The bake folds the sliders in; a primary-only file leaves the load strength out (it is used at that strength),
while a file with a donor in it bakes each LoRA at its own strength and is used at 1.0.
"""
import gc
import json
import logging
import os
import re
import threading
from typing import Optional

import torch

from fizgig.families.driver import FamilyDriver
from fizgig.families.lora import FamilyLoRA

logger = logging.getLogger(__name__)

PRIMARY, DONOR, SPEED, LOOK = "primary", "donor", "speed_lora", "preview_look"
MODEL_EQ_MARKER, MODEL_EQ_GAIN = "ss_fizgig_model_lora", "ss_fizgig_model_lora_gain"


def model_eq_gain(path):
    """A Model EQ LoRA's gain (scripts/extract_model_lora.py writes the marker), or None for any other file."""
    try:
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return None
    if meta.get(MODEL_EQ_MARKER) != "1":
        return None
    try:
        return float(meta.get(MODEL_EQ_GAIN, 0.25))
    except ValueError:
        return 0.25
EPOCH = "epoch_"                    # + the epoch number: one adapter per blended epoch (the primary keeps PRIMARY)

from fizgig.families.epochs import _EPOCH_RE, ema_weights, run_epochs   # noqa: F401 (the workbench's own names)
from fizgig.utils.upright import upright


class RenderCancelled(Exception):
    """A render aborted at a step boundary by request_cancel (a newer edit wants the engine)."""


class _Loaded:
    is_loaded = True


def _slerp(t, a, b):
    """Spherical interpolation of two noise tensors (keeps the norm, so a travel frame never goes mushy)."""
    af, bf = a.flatten().float(), b.flatten().float()
    na, nb = af.norm(), bf.norm()
    if na.item() == 0 or nb.item() == 0:
        return torch.lerp(a.float(), b.float(), t)
    dot = torch.dot(af / na, bf / nb).clamp(-1.0, 1.0)
    om = torch.acos(dot)
    so = torch.sin(om)
    if so.item() < 1e-6:
        return torch.lerp(a.float(), b.float(), t)
    return ((torch.sin((1 - t) * om) / so) * a.float() + (torch.sin(t * om) / so) * b.float())


def _blend(a, b, t, mode):
    af, bf = a.float(), b.float()
    if mode in (None, "lerp"):
        return torch.lerp(af, bf, t).to(a.dtype)
    eps = 1e-6
    na, nb = af.norm(dim=-1, keepdim=True), bf.norm(dim=-1, keepdim=True)
    target = (1 - t) * na + t * nb
    if mode == "norm":
        out = torch.lerp(af, bf, t)
        return (out * (target / out.norm(dim=-1, keepdim=True).clamp_min(eps))).to(a.dtype)
    ua, ub = af / na.clamp_min(eps), bf / nb.clamp_min(eps)
    dot = (ua * ub).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    om = torch.acos(dot)
    so = torch.sin(om)
    arc = (torch.sin((1 - t) * om) / so) * ua + (torch.sin(t * om) / so) * ub
    d = torch.where(so < 1e-4, torch.lerp(ua, ub, t), arc)
    return (d * target).to(a.dtype)


def _free_vram_gb():
    try:
        from fizgig.families.quant import free_vram_gb
        return free_vram_gb()
    except Exception:
        return 0.0


class WorkbenchEngine:
    def __init__(self, description):
        self.desc = description
        self.driver = description.load_driver()
        self.driver.keep_text_encoder = True
        self.pipeline = None
        self.dit = self.vae = self.net = None
        self.te_path = None
        self.device = "cuda"
        self.speed = None                   # the SpeedLoRA previews use, or None (default sampling)
        # comfy-kitchen's INT8 attention for this family's renders (description.int8_attention; the Profiler turns it
        # off for its measurements)
        self.int8_attention = bool(getattr(description, "int8_attention", False))
        self.lowmem = False

        self.primary_network = None         # the FamilyLoRA once a primary is attached (the tabs test for None)
        self.donor_network = None
        self.primary_path = self.donor_path = None
        self.primary_block_ids, self.donor_block_ids = set(), set()
        self.primary_hash = None
        self._clear_blend_state()

        self._prompt_cache = {}             # (prompt,) -> conditioning dict on CPU
        self._cancel_event = threading.Event()
        self.on_step = None                 # (done, total) per denoising step, from the render thread
        self._baseline_key = self._baseline_img = None
        # Turbo Preview: the activation cache (families/act_cache.py) when the description offers one and the tab's
        # tick is on
        self._turbo_enabled = False
        self._act = None
        self._act_modules = None
        self._act_ctx = None                # (key, sig) generate_preview hands its render; None = no cache
        self._act_bypass = False
        # the last render's clean latent (description.reference_strength: Royale's sequential travel edits it)
        self._last_frame_latent = None
        # description.workbench_follows_samples: the GUI keeps this dict current from the Samples tab - steps, cfg,
        # negative, turbo (strength; 0 = no speed LoRA). None = the family's fixed preview recipe.
        self.preview_settings = None
        self._speed_path = ""
        # Prompt B (a driver with supports_prompt_blend): the second prompt and how much of it (0 = prompt A only).
        # Applies to every render, the Baseline too, so Baseline vs Tweaked still shows only the sliders
        self.prompt_b = None
        self.prompt_blend = 0.0
        # audio to audio (a driver with supports_audio_source): the source file, the denoise (1 = start from noise)
        # and how much of the render follows its melody (0 = none); a reference recording to sound like
        self.source_path = None
        self.source_denoise = 1.0
        self.source_follow = 0.0
        self.source_repaint = None          # (start, end) seconds: remake only that section of the source (repaint;
        self.repaint_strength = 0.5         # past its end = extend), the rest kept; the handler's balanced strength
        self._source_cache = {}
        self.reference_path = None
        self._reference_cache = {}

    # ---- the block map ------------------------------------------------------------------------------
    def block_groups(self):
        return self.driver.block_map(self.dit)

    def block_ids(self):
        return [b.id for g in self.block_groups() for b in g.blocks]

    def default_state(self, width=None, height=None):
        """A SliderState over this family's blocks (every slider at its default)."""
        from fizgig.repair_studio.state import BlockState, SliderState
        s = SliderState(blocks={bid: BlockState() for bid in self.block_ids()})
        if width:
            s.preview_width, s.preview_height = int(width), int(height or width)
        return s

    # ---- models -------------------------------------------------------------------------------------
    def ensure_pipeline(self, dit_path, vae_path, text_encoder_path, speed_lora_path="", device="cuda",
                        lowmem=None, precision="auto", blocks_to_swap=0, preview_sampling=None,
                        speed_on_checkpoint=False, look_lora_path="", look_strength=1.0, **_ignored):
        """Load the DiT (resident) and the VAE once; the text encoder loads per new prompt. speed_lora_path: the
        family's speed LoRA file, attached unmerged and used for every preview ("" = default sampling).
        precision: "auto" = bf16 when the model file fits the free VRAM with 6 GB to spare (Qwen 2.1's 14 GB: 20 GB
        free), else INT8, else NF4 (whichever the family offers); blocks_to_swap streams blocks forward-only
        (previews never backprop). preview_sampling: dit_path is the family's preview checkpoint (a distilled model,
        description.preview_checkpoint) sampled this way, with no speed LoRA. look_lora_path: the Samples tab's
        preview look LoRA (MOSS-Image's SFT look), on for every render at look_strength - a fixed adapter like the
        speed LoRA: never under the block sliders, never in a save."""
        if self.pipeline is not None:
            return
        self.checkpoint_sampling = preview_sampling
        self.speed_on_checkpoint = bool(speed_on_checkpoint and preview_sampling is not None)
        if preview_sampling is not None and not self.speed_on_checkpoint:
            speed_lora_path = ""
        from fizgig.families.train import _small_card_previews
        self.device = device
        self.te_path = text_encoder_path
        self.lowmem = _small_card_previews() if lowmem is None else bool(lowmem)
        from fizgig.families import quant
        if precision == "auto":
            try:
                need = self.driver.dit_gb(dit_path) + 6.0
            except OSError:
                need = 20.0
            free = _free_vram_gb()
            precision = ("bf16" if free >= need or not {"int8", "nf4"} & set(self.desc.precisions) else
                         "int8" if "int8" in self.desc.precisions and free >= need / 2 else
                         "nf4" if "nf4" in self.desc.precisions else "int8")
        elif precision not in self.desc.precisions:
            precision = "bf16"
        self.dit, self.swapped = quant.load_base(self.driver, dit_path, device, precision, blocks_to_swap,
                                                 supports_backward=False)
        if self.swapped:
            self.driver.block_swap_mode(self.dit, inference=True)
        self.precision = precision
        self.vae = self.driver.load_vae(vae_path, "cpu" if self.lowmem else device)
        self.net = FamilyLoRA(self.dit, self.driver, device=device)
        sp = self.desc.preview_speed()
        self._speed_path = speed_lora_path or ""
        if speed_lora_path and sp is not None and os.path.exists(speed_lora_path):
            n = self.net.add_file(speed_lora_path, SPEED, sp.strength)
            self.net.move_adapter(SPEED, device)
            self.driver.frozen_file_added(self.dit, speed_lora_path, sp.strength, "speed")
            self.speed = sp
            logger.info("%s workbench: speed LoRA %s on %d Linears", self.desc.display_name, sp.name, n)
        if look_lora_path and os.path.exists(look_lora_path):
            n = self.net.add_file(look_lora_path, LOOK, float(look_strength))
            if n:
                self.net.move_adapter(LOOK, device)
                self.driver.frozen_file_added(self.dit, look_lora_path, float(look_strength), "speed")
                logger.info("%s workbench: look LoRA %s on %d Linears at %g", self.desc.display_name,
                            os.path.basename(look_lora_path), n, float(look_strength))
            else:
                self.net.remove(LOOK)
        self.pipeline = _Loaded()
        logger.info("%s workbench ready (%s, block swap %d, lowmem=%s)", self.desc.display_name, precision,
                    self.swapped, self.lowmem)

    def _attach(self, path, name, strength=1.0):
        n = self.net.add_file(path, name, strength)
        if n == 0:
            self.net.remove(name)
            raise ValueError(f"{os.path.basename(path)} adapts nothing in {self.desc.display_name} "
                             "(a LoRA for another model?)")
        self.net.move_adapter(name, self.device)
        gain = model_eq_gain(path)
        eq = self.__dict__.setdefault("_eq", {})
        eq.pop(name, None)
        if gain is not None:
            eq[name] = gain
            self.net.set_outside(name, False)          # an EQ acts through its block sliders only
        return n

    def is_model_eq(self, role=PRIMARY):
        """The role's file is a Model EQ LoRA (its sliders read 1.0 = the stock model)."""
        return role in getattr(self, "_eq", {})

    def load_primary(self, path):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        n = self._attach(path, PRIMARY)
        self.primary_network = self.net
        self._clear_blend_state()
        self.primary_path = path
        self.primary_block_ids = self.net.adapter_blocks(PRIMARY)
        self.primary_hash = self._hash(path)
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        logger.info("%s primary: %s (%d Linears, %d blocks)", self.desc.display_name, path, n,
                    len(self.primary_block_ids))

    def swap_primary_weights(self, path) -> bool:
        """Another checkpoint of the same run in place (Royale's epoch scrub). False = the caller should reset()
        and load_primary() instead."""
        if self.primary_network is None:
            raise RuntimeError("No primary loaded; call load_primary() first.")
        try:
            n = self.net.swap_file(PRIMARY, path)
        except Exception:
            logger.exception("swap_primary_weights: %s", path)
            return False
        if n == 0:
            return False
        self.unload_epochs(PRIMARY)
        self.net.move_adapter(PRIMARY, self.device)
        gain = model_eq_gain(path)                     # the swapped-in file may (or may not) be a Model EQ
        eq = self.__dict__.setdefault("_eq", {})
        eq.pop(PRIMARY, None)
        if gain is not None:
            eq[PRIMARY] = gain
            self.net.set_outside(PRIMARY, False)
        else:
            self.net.set_outside(PRIMARY, True)
        self.primary_path = path
        self.primary_block_ids = self.net.adapter_blocks(PRIMARY)
        self.primary_hash = self._hash(path)
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        return True

    def load_donor(self, path):
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — unload_donor() or reset() first.")
        self._attach(path, DONOR)
        # The donor comes in through its block sliders only (fully on by default); anything it adapts outside
        # the block map stays off, as on the other engines.
        self.net.set_outside(DONOR, False)
        self._clear_blend_state(DONOR)
        self.donor_network = self.net
        self.donor_path = path
        self.donor_block_ids = self.net.adapter_blocks(DONOR)
        self._invalidate_activation_cache()

    def unload_donor(self):
        if self.donor_network is not None:
            self.unload_epochs(DONOR)
            self.net.remove(DONOR)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()
            self._invalidate_activation_cache()

    @staticmethod
    def _hash(path):
        try:
            from fizgig.utils.lora_files import compute_lora_hash
            return compute_lora_hash(path)
        except Exception:
            return None

    # ---- epoch blend / LoRA mix -------------------------------------------------------------------------
    # Per LoRA (PRIMARY, DONOR): its sources (key -> file: a run's epochs, or the files of a LoRA mix), the weights
    # in use, all under that LoRA's block sliders and load strength. Loaded files are kept by path, so switching
    # between a run's epochs and a mix (or back) reuses what is already on the card.
    MAX_EPOCHS_LOADED = 12              # per LoRA

    @staticmethod
    def _np(path):
        return os.path.normcase(os.path.abspath(path))

    def _clear_blend_state(self, role=None):
        if role is None or not hasattr(self, "_eb"):
            self._eb = {}
            self._eb_seq = 0
        for r in ((PRIMARY, DONOR) if role is None else (role,)):
            # files: key -> file (an epoch number, or a mix row); adapters: normalised path -> loaded adapter (the
            # role's own file is the role's adapter and is not listed); own: the key of the role's own file;
            # blend: {key: weight} in use, else None; kind: "epochs" / "mix"
            self._eb[r] = {"files": {}, "adapters": {}, "own": None, "blend": None, "kind": "epochs"}

    def _role_path(self, role):
        return self.primary_path if role == PRIMARY else self.donor_path

    @property
    def primary_epoch(self):
        return self._eb[PRIMARY]["own"]

    def epoch_of(self, role=PRIMARY):
        """The key of the loaded file among its sources (its epoch in its run; None = unnumbered and alone)."""
        return self._eb[role]["own"]

    def _set_files(self, role, files, kind):
        path = self._role_path(role)
        if not path or not self.net or not self.net.has(role):
            raise RuntimeError(f"Load the {'primary' if role == PRIMARY else 'donor'} LoRA first.")
        eb = self._eb[role]
        eb["files"], eb["kind"], eb["blend"] = dict(files), kind, None
        own = [k for k, f in eb["files"].items() if self._np(f) == self._np(path)]
        eb["own"] = own[0] if own else None
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()

    def find_epochs(self, role=PRIMARY):
        """The run of the primary's (or the donor's) file as its sources: [epochs], oldest first (run_epochs).
        Nothing loads until a blend needs it."""
        self._set_files(role, run_epochs(self._role_path(role) or ""), "epochs")
        return sorted(self._eb[role]["files"])

    def set_sources(self, files, role=PRIMARY):
        """A LoRA mix: {key: file} (the role's own file among them). Weights come with set_blend."""
        self._set_files(role, {int(k): f for k, f in files.items()}, "mix")

    def _name_for(self, role, path):
        if self._np(path) == self._np(self._role_path(role) or ""):
            return role
        return self._eb[role]["adapters"].get(self._np(path))

    def _sync_epochs(self):
        """Attach the files each mix weights, and drop unused ones past MAX_EPOCHS_LOADED (renders call it; the
        engine is theirs)."""
        for role, eb in self._eb.items():
            if not eb["blend"] or not self.net or not self.net.has(role):
                continue
            wanted = set()
            for k in sorted(eb["blend"]):
                f = eb["files"].get(k)
                if f is None:
                    continue
                wanted.add(self._np(f))
                if self._name_for(role, f) is None:
                    self._eb_seq += 1
                    name = f"{EPOCH}{'d' if role == DONOR else 'p'}{self._eb_seq}"
                    self._attach(f, name)
                    if role == DONOR:
                        self.net.set_outside(name, False)     # as the donor itself: its blocks only
                    eb["adapters"][self._np(f)] = name
            spare = [np_ for np_ in eb["adapters"] if np_ not in wanted]
            while len(eb["adapters"]) > self.MAX_EPOCHS_LOADED and spare:
                self.net.remove(eb["adapters"].pop(spare.pop(0)))

    def _drop_epoch_adapters(self, role):
        for name in list(self._eb[role]["adapters"].values()):
            if self.net is not None and self.net.has(name):
                self.net.remove(name)
        self._eb[role]["adapters"] = {}

    def unload_epochs(self, role=PRIMARY):
        self._drop_epoch_adapters(role)
        self._clear_blend_state(role)
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()

    def set_blend(self, weights, role=PRIMARY):
        """{key: weight} over the role's sources (ema_weights for a run, the user's amounts for a mix), or None for
        its file alone. Takes effect at the next render; the Baseline follows it too (the mix, sliders at 1)."""
        eb = self._eb[role]
        weights = {int(k): float(w) for k, w in (weights or {}).items() if int(k) in eb["files"]} or None
        if weights != eb["blend"]:
            eb["blend"] = weights
            self._invalidate_baseline_cache()
            self._invalidate_activation_cache()

    def _blend_sig(self):
        return tuple((r, eb["kind"], tuple(sorted((self._np(eb["files"][k]), round(w, 5))
                                                  for k, w in eb["blend"].items())))
                     for r, eb in sorted(self._eb.items()) if eb["blend"]) or None

    def _role_names(self, role):
        """(adapter, weight) pairs that LoRA's sliders and load strength drive: its mix, or its file alone."""
        eb = self._eb[role]
        out = {role: 0.0}
        out.update({n: 0.0 for n in eb["adapters"].values()})
        if not eb["blend"]:
            out[role] = 1.0
        else:
            for k, w in eb["blend"].items():
                name = self._name_for(role, eb["files"][k])
                if name is not None:
                    out[name] = out.get(name, 0.0) + w
        return list(out.items())

    def blend_record(self, role=PRIMARY):
        """What a saved file records of the mix: {epoch: weight} for a run, {file name: amount} for a LoRA mix."""
        eb = self._eb[role]
        if not eb["blend"]:
            return None
        if eb["kind"] == "mix":
            return {os.path.basename(eb["files"][k]): round(w, 5) for k, w in sorted(eb["blend"].items())}
        return {str(k): round(w, 5) for k, w in sorted(eb["blend"].items())}

    # ---- slider state -------------------------------------------------------------------------------
    def apply_state(self, state):
        """Per-block strength / on-off and each LoRA's load strength, live (no reload)."""
        if self.primary_network is None:
            return
        targets = []
        for role, who in ((PRIMARY, "primary"), (DONOR, "donor")):
            scale = float(getattr(state, f"{who}_scale", 1.0))
            targets += [(n, who, scale * w) for n, w in self._role_names(role)]
        eq = getattr(self, "_eq", {})
        for name, who, strength in targets:
            if not self.net.has(name):
                continue
            mult = {b: float(getattr(bs, f"{who}_strength")) for b, bs in state.blocks.items()}
            if name in eq:                               # Model EQ: 1.0 = stock
                mult = {b: (m - 1.0) * eq[name] for b, m in mult.items()}
            self.net.set_blocks(
                name, mult=mult, enabled={b: bool(getattr(bs, f"{who}_enabled")) for b, bs in state.blocks.items()})
            self.net.set_strength(name, strength)

    def mark_blocks_changed(self, blocks):
        pass                                # the activation cache compares slider values itself

    # ---- Turbo Preview ------------------------------------------------------------------------------
    @property
    def turbo_preview(self):
        return bool(self._turbo_enabled and getattr(self.desc, "activation_cache", False))

    @turbo_preview.setter
    def turbo_preview(self, on):
        self._turbo_enabled = bool(on)
        self._invalidate_activation_cache()

    @staticmethod
    def _cache_sig(state):
        return {b: (round(float(bs.primary_strength), 4), bool(bs.primary_enabled),
                    round(float(bs.donor_strength), 4), bool(bs.donor_enabled)) for b, bs in state.blocks.items()}

    # ---- cancellation -------------------------------------------------------------------------------
    def request_cancel(self):
        self._cancel_event.set()

    def clear_cancel(self):
        self._cancel_event.clear()

    # ---- rendering ----------------------------------------------------------------------------------
    def _park_dit(self, where):
        if self.dit is not None and not getattr(self, "swapped", 0):   # a swapped DiT keeps its streaming layout
            from fizgig.families import quant
            quant.move(self.dit, where)
            if where == "cpu":
                gc.collect()
                torch.cuda.empty_cache()

    @property
    def reference_kind(self):
        """How a reference picture reaches this family's previews: "vision", "edit" or "" (the description's)."""
        return self.desc.reference_kind

    @staticmethod
    def _state_reference(state):
        """(path, megapixels) of the state's reference picture, or ("", 1.0) when it has none on disk."""
        path = (getattr(state, "ref_image_path", "") or "").strip()
        try:
            mp = float(getattr(state, "ref_megapixels", 1.0) or 1.0)
        except (TypeError, ValueError):
            mp = 1.0
        return (path, mp) if path and os.path.isfile(path) else ("", 1.0)

    @staticmethod
    def _fit_reference(path, width, height):
        """An edit reference as uint8 (H, W, 3), cropped and scaled to the preview's size."""
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return np.array(ImageOps.fit(upright(im).convert("RGB"), (int(width), int(height)), Image.LANCZOS))

    def _reference_latents(self, path, width, height):
        """The edit reference's latents (CPU, batch dim), cached per picture and size."""
        key = ("__ref_latents__", path, int(width), int(height))
        if key not in self._prompt_cache:
            if self.lowmem:
                self._park_dit("cpu")
                self.vae.to(self.device)
            try:
                z = self.driver.encode_images(self.vae, [self._fit_reference(path, width, height)])[0]
            finally:
                if self.lowmem:
                    self.vae.to("cpu")
                    self._park_dit(self.device)
            self._prompt_cache[key] = z[None].cpu()
        return self._prompt_cache[key]

    def encode(self, prompts, ref="", ref_mp=1.0, size=(768, 768)):
        """Conditioning for each prompt (CPU), through the family's text encoder, loaded for the call and freed.
        With a reference picture (`ref`), the prompt sees it the family's way (reference_kind): through the
        encoder's vision path at `ref_mp`, or as an edit of the picture at the preview `size`. The DiT parks on CPU
        while the encoder runs when both would not fit."""
        kind = self.reference_kind if ref else ""
        tag = ((ref, round(float(ref_mp), 4)) if kind == "vision" else
               (ref, int(size[0]), int(size[1])) if kind == "edit" else ()) + tuple(self.driver.text_cache_key())
        need = [p for p in prompts if (p,) + tag not in self._prompt_cache]
        if need:
            te_gb = self.driver.text_encoder_gb(self.te_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()             # the last render's cached blocks are not in use
            park = self.lowmem or _free_vram_gb() < te_gb + 2.0
            if park:
                self._park_dit("cpu")
            try:
                te = (self.driver.load_reference_text_encoder(self.te_path, self.device) if kind == "edit"
                      else self.driver.load_text_encoder(self.te_path, self.device))
                try:
                    if kind == "vision":
                        from PIL import Image
                        with Image.open(ref) as im:
                            conds = self.driver.encode_text_with_image(te, list(need), upright(im).convert("RGB"),
                                                                       megapixels=ref_mp)
                    elif kind == "edit":
                        pic = self._fit_reference(ref, *size)
                        conds = self.driver.encode_text_with_references(te, list(need), [[pic]] * len(need))
                    else:
                        conds = self.driver.encode_text(te, list(need))
                    for p, c in zip(need, conds):
                        self._prompt_cache[(p,) + tag] = c
                finally:
                    self.driver.unload_text_encoder(te)
                    del te
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                if park:
                    self._park_dit(self.device)
        return [self._prompt_cache[(p,) + tag] for p in prompts]

    def _cond_to_device(self, cond):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _follow_speed(self):
        """Samples-tab mode: the speed LoRA on at the tab's turbo strength (attached on first use), off at 0."""
        ps = self.preview_settings
        sp = self.desc.preview_speed()
        try:
            strength = float(ps.get("turbo") or 0.0)
        except (TypeError, ValueError):
            strength = 0.0
        on = strength > 0 and sp is not None and bool(self._speed_path) and os.path.exists(self._speed_path)
        if getattr(self, "_speed_follow", None) == (on, strength):
            return                          # unchanged: don't re-apply the adapter (its bias deltas included)
        self._speed_follow = (on, strength)
        if on:
            if not self.net.has(SPEED):
                self.net.add_file(self._speed_path, SPEED, strength)
                self.net.move_adapter(SPEED, self.device)
            self.net.set_strength(SPEED, strength)
            self.driver.frozen_file_added(self.dit, self._speed_path, strength, "speed")
            self.net.set_enabled(SPEED, True)
            self.speed = sp
        else:
            if self.net.has(SPEED):
                self.net.set_enabled(SPEED, False)
            self.speed = None

    def _settings_key(self):
        ps = self.preview_settings
        opts = tuple(sorted((self.driver.options or {}).items())) if self.driver is not None else ()
        return (None if ps is None else (ps.get("steps"), ps.get("cfg"), ps.get("negative"), ps.get("turbo"))), opts

    def sampling(self):
        """(steps, cfg, sigmas, options) previews use."""
        ps = self.preview_settings
        if ps is not None and (getattr(self, "checkpoint_sampling", None) is None
                               or getattr(self, "speed_on_checkpoint", False)):
            self._follow_speed()
            s = self.speed.settings if self.speed is not None else (
                getattr(self, "checkpoint_sampling", None) or self.desc.default_sampling())
            try:
                steps = int(ps.get("steps")) or s.steps
            except (TypeError, ValueError):
                steps = s.steps
            try:
                cfg = float(ps.get("cfg"))
            except (TypeError, ValueError):
                cfg = s.cfg
            return steps, cfg, s.sigmas, s.options          # the driver drops a schedule made for another step count
        if getattr(self, "checkpoint_sampling", None) is not None:
            s = self.checkpoint_sampling
            return s.steps, s.cfg, s.sigmas, s.options
        if self.speed is not None:
            s = self.speed.settings
            return s.steps, s.cfg, s.sigmas, s.options
        s = self.desc.default_sampling()
        return s.steps, s.cfg, s.sigmas, s.options

    @torch.no_grad()
    def _prompt_b_live(self):
        return bool(self.prompt_b and float(self.prompt_blend or 0) > 0
                    and getattr(self.driver, "supports_prompt_blend", False))

    def _prompt_b_key(self):
        return (self.prompt_b, round(float(self.prompt_blend or 0), 4)) if self._prompt_b_live() else None

    def _source_live(self):
        return bool(self.source_path and (float(self.source_denoise) < 1.0 or float(self.source_follow or 0) > 0
                                          or self.source_repaint is not None)
                    and getattr(self.driver, "supports_audio_source", False))

    @staticmethod
    def _file_stamp(path):
        try:
            return os.path.getsize(path), int(os.path.getmtime(path))
        except OSError:
            return None

    def _source_key(self):
        if not self._source_live():
            return None
        rp = self.source_repaint
        return (self.source_path, self._file_stamp(self.source_path), round(float(self.source_denoise), 4),
                round(float(self.source_follow or 0), 4),
                None if rp is None else (round(float(rp[0]), 3), round(float(rp[1]), 3),
                                         round(float(self.repaint_strength), 4)))

    def _source_latent(self, seconds):
        key = (self._source_key()[:2], round(float(seconds), 3))
        if key not in self._source_cache:
            if self.lowmem:
                self.vae.to(self.device)
            try:
                self._source_cache = {key: self.driver.encode_source(self.vae, self.source_path, seconds)}
                self._source_wave = None
            finally:
                if self.lowmem:
                    self.vae.to("cpu")
        return self._source_cache[key]

    def _source_wave_at(self, seconds):
        """The source's own waveform at the render's length (a repaint's splice), torch (2, n), cached."""
        key = (self._source_key()[:2], round(float(seconds), 3))
        if getattr(self, "_source_wave", None) is None or self._source_wave[0] != key:
            import torch as _t
            self._source_wave = (key, _t.from_numpy(self.driver.source_wave(self.source_path, seconds)))
        return self._source_wave[1]

    def _reference_live(self):
        return bool(self.reference_path and getattr(self.driver, "supports_audio_source", False))

    def _reference_key(self):
        return (self.reference_path, self._file_stamp(self.reference_path)) if self._reference_live() else None

    def _reference_latent(self):
        key = self._reference_key()
        if key not in self._reference_cache:
            if self.lowmem:
                self.vae.to(self.device)
            try:
                self._reference_cache = {key: self.driver.encode_reference(self.vae, self.reference_path)}
            finally:
                if self.lowmem:
                    self.vae.to("cpu")
        return self._reference_cache[key]

    def _audio_key(self):
        """What the audio inputs (source + reference) add to a render's cache keys; None without either."""
        s, r = self._source_key(), self._reference_key()
        return None if s is None and r is None else (s, r)

    def render(self, cond, width, height, seed, *, steps=None, noise=None, refs=None, neg_cond=None, cond_b=None,
               blend=0.0, source=None, refer=None):
        """One image from conditioning with the adapters as currently set (refs: an edit's reference latents;
        neg_cond: the negative, used when the CFG is above 1)."""
        d_steps, cfg, sigmas, options = self.sampling()
        steps = int(steps or d_steps)
        if cfg > 1.0 and neg_cond is not None:
            neg_cond = self._cond_to_device(neg_cond)
        else:
            neg_cond = None

        act = self._act_for_render()
        if act is not None:
            key, sig = self._act_ctx
            key = repr((key, steps, cfg, None if sigmas is None else [float(v) for v in sigmas], options,
                        neg_cond is not None, bool(self.int8_attention)))

        def _step(done, total):
            if act is not None:
                act.step(done)
            cb = self.on_step
            if cb is not None:
                try:
                    cb(done, total)
                except Exception:
                    pass
            if self._cancel_event.is_set():
                raise RenderCancelled()

        from fizgig.modules import int8_attention as _i8a
        meter = self._start_meter()

        def _generate():
            with _i8a.renders(self.int8_attention):
                return self.driver.generate(self.dit, self._cond_to_device(cond), width, height, steps=steps,
                                            seed=int(seed), cfg=cfg, sigmas=sigmas, options=options, noise=noise,
                                            on_step=_step, **({"neg_cond": neg_cond} if neg_cond is not None else {}),
                                            **({"cond_b": self._cond_to_device(cond_b), "prompt_blend": float(blend)}
                                               if cond_b is not None else {}),
                                            **({"source": source, "denoise": float(self.source_denoise),
                                                "follow": float(self.source_follow or 0)}
                                               if source is not None else {}),
                                            **({"repaint": tuple(self.source_repaint),
                                                "repaint_strength": float(self.repaint_strength)}
                                               if source is not None and self.source_repaint is not None else {}),
                                            **({"refer": refer} if refer is not None else {}),
                                            **({"refs": [r.to(self.device) for r in refs]} if refs else {}))
        try:
            if act is None:
                lat = _generate()
            else:
                try:
                    with act.render(self.dit, self._act_modules, key, sig):
                        lat = _generate()
                except torch.OutOfMemoryError:
                    # the cache's share of the card was the difference: give it back and render plainly
                    logger.warning("Turbo Preview: out of memory with the cache - rendering without it")
                    self._invalidate_activation_cache()
                    act = None
                    lat = _generate()
        finally:
            activity = self._stop_meter(meter)
        if getattr(self.desc, "reference_strength", False):
            self._last_frame_latent = lat[0].detach().to("cpu")     # the next travel frame may edit this one
        if self.lowmem:
            self._park_dit("cpu")
            self.vae.to(self.device)
        try:
            try:
                out = self.driver.decode(self.vae, lat, width, height)
            except torch.OutOfMemoryError:
                if self._act is None or not self._act.out:
                    raise
                logger.warning("Turbo Preview: out of memory decoding beside the cache - cache dropped")
                self._invalidate_activation_cache()
                out = self.driver.decode(self.vae, lat, width, height)
            if source is not None and self.source_repaint is not None:
                # the handler's waveform splice: outside the section, the source's own audio
                out.wave = self.driver.splice_source(out.wave, self._source_wave_at(width), self.source_repaint,
                                                     self.repaint_strength)
            if activity is not None:
                try:
                    out.activity = activity
                except AttributeError:
                    pass
            return out
        finally:
            if self.lowmem:
                self.vae.to("cpu")
                self._park_dit(self.device)

    def _start_meter(self):
        """Block activity (description block_activity): tag each wrapped Linear with its block - only those along
        the output's sequence (driver.activity_axis) - and start a BlockMeter. None for other families."""
        if not getattr(self.desc, "block_activity", False) or self.net is None:
            return None
        from fizgig.families import lora as _lora
        for full, mod in self.net.wrapped.items():
            mod._meter_block = self.driver.block_of(full) if self.driver.activity_axis(full) else None
        meter = _lora.BlockMeter(skip=(SPEED,))
        _lora.METER.on = meter
        return meter

    @staticmethod
    def _stop_meter(meter):
        """-> {block id: push along the sequence} (numpy), or None."""
        if meter is None:
            return None
        from fizgig.families import lora as _lora
        _lora.METER.on = None
        try:
            return meter.result() or None
        except Exception:
            return None

    def generate_preview(self, state, *, seed=None, prompt=None, width=None, height=None, steps=None,
                         seed_b=None, travel_t=0.0, override_ctx=None, override_neg_ctx=None,
                         prev_latent=None, prev_latent_strength=1.0):
        """The tabs' render call. seed_b / travel_t: seed travel by noise slerp. override_ctx: precomputed
        conditioning (prompt travel; the text only, as in the original Krea 2 engine; override_neg_ctx is accepted and
        ignored). The state's reference picture (ref_image_path, ref_megapixels) reaches the prompt the family's way;
        for an edit family with reference_strength it is scaled by state.ref_strength, and prev_latent (the previous
        travel frame's clean latent) joins as a second reference at prev_latent_strength."""
        self._sync_epochs()
        self.apply_state(state)
        seed = state.seed if seed is None else seed
        width = int(width or state.preview_width)
        height = int(height or state.preview_height)
        ref, ref_mp = self._state_reference(state) if self.reference_kind else ("", 1.0)
        if override_ctx is not None:
            cond = override_ctx
        else:
            cond = self.encode([prompt if prompt is not None else state.prompt], ref=ref, ref_mp=ref_mp,
                               size=(width, height))[0]
        refs = self._edit_refs(state, ref, width, height, prev_latent, prev_latent_strength) \
            if self.reference_kind == "edit" else None
        source = self._source_latent(width) if self._source_live() else None
        refer = self._reference_latent() if self._reference_live() else None
        cond_b = None
        if override_ctx is None and self._prompt_b_live():
            cond_b = self.encode([self.prompt_b], ref=ref, ref_mp=ref_mp, size=(width, height))[0]
        noise = None
        if seed_b is not None:
            noise = _slerp(float(travel_t or 0.0), self.driver.initial_noise(seed, width, height),
                           self.driver.initial_noise(seed_b, width, height))
        neg = None
        ps = self.preview_settings
        try:
            if float(self.sampling()[1] or 1.0) > 1.0:     # the CFG this render uses (a preview checkpoint has
                #                                            its own: Qwen's Turbo, CFG 1)
                # the Samples tab's negative (with no Samples settings, the family's default), seen the same way as
                # the prompt (an edit's reference included)
                text = ps.get("negative") if ps is not None else self.desc.preview_negative
                neg = self.encode([str(text or "")], ref=("" if override_ctx is not None else ref),
                                  ref_mp=ref_mp, size=(width, height))[0]
        except (TypeError, ValueError):
            neg = None
        self._act_ctx = None
        if override_ctx is None and noise is None and prev_latent is None and not self._act_bypass:
            self._act_ctx = ((id(self.dit), self.primary_path, self.primary_hash, self._blend_sig(), self.donor_path,
                              self._speed_path,
                              self._settings_key(), self._prompt_b_key(), self._audio_key(),
                              round(float(getattr(state, "primary_scale", 1.0)), 4),
                              round(float(getattr(state, "donor_scale", 1.0)), 4),
                              prompt if prompt is not None else state.prompt, ref, round(float(ref_mp), 4),
                              round(float(getattr(state, "ref_strength", 1.0) or 0.0), 4),
                              int(seed), width, height), self._cache_sig(state))
        try:
            return self.render(cond, width, height, seed, steps=steps, noise=noise, refs=refs, neg_cond=neg,
                               cond_b=cond_b, blend=float(self.prompt_blend or 0), source=source, refer=refer)
        finally:
            self._act_ctx = None

    def _edit_refs(self, state, ref, width, height, prev_latent=None, prev_strength=1.0):
        """An edit family's reference latents for one render: the picture (scaled by state.ref_strength where the
        family has reference_strength; 0 = left out) and, for a chained travel frame, the previous frame's clean
        latent at prev_strength. None when there is neither."""
        chain = bool(getattr(self.desc, "reference_strength", False))
        out = []
        s1 = 1.0
        if chain:
            try:
                s1 = float(getattr(state, "ref_strength", 1.0))
            except (TypeError, ValueError):
                s1 = 1.0
        if ref and s1 != 0.0:
            z = self._reference_latents(ref, width, height)
            out.append(z if s1 == 1.0 else z * s1)
        if chain and prev_latent is not None and float(prev_strength) != 0.0:
            z = prev_latent.detach().to("cpu")
            z = z[None] if z.dim() == 3 else z
            out.append(z if float(prev_strength) == 1.0 else z * float(prev_strength))
        return out or None

    def _act_for_render(self):
        """The activation cache for this render, or None (Turbo off, no family cache, a streamed DiT, or a render
        generate_preview did not set up: prompt / seed travel, the baseline)."""
        if not self.turbo_preview or self._act_ctx is None or getattr(self, "swapped", 0) or self.dit is None:
            return None
        if self._act is None:
            from fizgig.families.act_cache import ActivationCache
            self._act = ActivationCache()
        if self._act_modules is None:
            from fizgig.families.act_cache import block_modules
            self._act_modules = block_modules(self.driver, self.dit)
        return self._act if self._act_modules else None

    def generate_baseline(self, state):
        """The primary with every slider at 1.0 (at its load strength), donor off. Cached until the prompt, seed,
        size or load strength changes."""
        key = (self.primary_path, self._blend_sig(), self._prompt_b_key(), self._audio_key(), state.seed,
               state.prompt,
               state.preview_width, state.preview_height,
               round(float(getattr(state, "primary_scale", 1.0)), 4), self._state_reference(state),
               round(float(getattr(state, "ref_strength", 1.0) or 0.0), 4),
               self._settings_key())
        if self._baseline_key == key and self._baseline_img is not None:
            return self._baseline_img
        if self.same_as_baseline(state):
            # nothing moved: the tweaked render IS the baseline - one render for both panes, and it fills Turbo
            # Preview's cache for the first slider move
            img = self.generate_preview(state)
            self._baseline_key, self._baseline_img = key, img
            return img
        base = self.default_state(state.preview_width, state.preview_height)
        base.seed, base.prompt = state.seed, state.prompt
        base.ref_image_path = getattr(state, "ref_image_path", "")
        base.ref_megapixels = getattr(state, "ref_megapixels", 1.0)
        base.ref_strength = getattr(state, "ref_strength", 1.0)
        base.primary_scale = float(getattr(state, "primary_scale", 1.0))
        base.donor_scale = 0.0              # the primary alone: a loaded donor stays out of the left pane
        for bs in base.blocks.values():
            bs.donor_enabled = False
        self._act_bypass = True             # the baseline would overwrite the tweaked render's cache
        try:
            img = self.generate_preview(base)
        finally:
            self._act_bypass = False
        self._baseline_key, self._baseline_img = key, img
        return img

    def same_as_baseline(self, state):
        """True when the tweaked render would be the baseline: every primary block on at 1.0 and no donor block
        contributing (no donor, its load strength 0, or each of its blocks off or at 0)."""
        if any(not bs.primary_enabled or float(bs.primary_strength) != 1.0 for bs in state.blocks.values()):
            return False
        if self.donor_network is None or float(getattr(state, "donor_scale", 1.0)) == 0.0:
            return True
        return not any(bs.donor_enabled and float(bs.donor_strength) != 0.0
                       for b, bs in state.blocks.items() if b in self.donor_block_ids)

    def _invalidate_baseline_cache(self):
        self._baseline_key = self._baseline_img = None

    def _invalidate_activation_cache(self):
        if self._act is not None and self._act.out:
            self._act.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- prompt travel (Royale) -------------------------------------------------------------------------
    @property
    def supports_prompt_travel(self):
        return type(self.driver).pad_conditioning is not FamilyDriver.pad_conditioning

    def encode_travel_prompts(self, prompts):
        """Waypoint conditioning, padded to one shape by the driver. Returns (waypoints, None) - the second slot is
        the old engines' negative, unused here."""
        return self.driver.pad_conditioning(self.encode(list(prompts))), None

    @staticmethod
    def interp_waypoints(vecs, t, mode="lerp"):
        """Piecewise blend across the waypoint dicts (t 0..1 walks the whole chain): float tensors by lerp / norm /
        slerp along the feature axis; boolean masks become weights (a token only one side has fades in or out), so
        each endpoint is exactly its own prompt and nothing switches on mid-travel."""
        if len(vecs) == 1:
            return vecs[0]
        t = min(max(float(t), 0.0), 1.0)
        pos = t * (len(vecs) - 1)
        i = min(int(pos), len(vecs) - 2)
        local = pos - i
        out = {}
        for k, a in vecs[i].items():
            b = vecs[i + 1][k]
            if not torch.is_tensor(a):
                out[k] = a
            elif a.dtype == torch.bool:
                out[k] = a if local == 0.0 else torch.lerp(a.float(), b.float(), local)
            else:
                out[k] = _blend(a, b, local, mode)
        return out

    # ---- bake ---------------------------------------------------------------------------------------
    def save_repaired(self, out_path, state, include_donor=True):
        """Write the primary (and the donor's enabled blocks) as one LoRA in the family's format, block sliders
        folded in. Primary only: the load strength is left out (the file is used at it, as previewed). With a donor
        contributing, the two share one file, so each is baked at its own load strength and the file is used at
        1.0. Returns the summary dict the Repair Studio's save dialog reports (use_at = the strength to load it at)."""
        from safetensors import safe_open
        from safetensors.torch import save_file
        use_donor = include_donor and self.net.has(DONOR)
        blended = set()
        if use_donor:
            blended = {b for b, bs in state.blocks.items()
                       if b in self.donor_block_ids and bs.donor_enabled and bs.donor_strength != 0}
        flat = state.copy()
        if not blended:
            flat.primary_scale = flat.donor_scale = 1.0
        use_at = 1.0 if blended else float(getattr(state, "primary_scale", 1.0))
        self._sync_epochs()
        self.apply_state(flat)
        try:
            names = [n for n, w in self._role_names(PRIMARY) if w] + \
                ([n for n, w in self._role_names(DONOR) if w] if use_donor else [])
            sd, ranks = self.net.bake(names)
        finally:
            self.apply_state(state)
        try:
            with safe_open(self.primary_path, "pt") as f:
                metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
        except Exception:
            metadata = {}
        for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256", MODEL_EQ_MARKER, MODEL_EQ_GAIN):
            metadata.pop(stale, None)              # a baked Model EQ is an ordinary LoRA, used at 1.0
        if ranks:
            metadata["ss_network_dim"] = str(max(ranks.values()))
            metadata["ss_network_alpha"] = str(float(max(ranks.values())))
        metadata["ss_repair_studio_config"] = json.dumps(state.to_json(), separators=(",", ":"))
        metadata["ss_repair_studio_use_at"] = f"{use_at:g}"
        for role, who in ((PRIMARY, ""), (DONOR, "donor_")):
            rec = self.blend_record(role)
            if rec and (role == PRIMARY or use_donor):
                kind = "lora_mix" if self._eb[role]["kind"] == "mix" else "epoch_blend"
                metadata[f"ss_repair_studio_{who}{kind}"] = json.dumps(rec, separators=(",", ":"))
        if use_donor:
            metadata["ss_repair_studio_donor_path"] = os.path.basename(self.donor_path)
            combined = {m: r for m, r in ranks.items() if self.driver.block_of(m) in blended}
            if combined:
                metadata["ss_repair_studio_combined_ranks"] = json.dumps(dict(sorted(combined.items())),
                                                                         separators=(",", ":"))
        save_file(sd, out_path, metadata=metadata)
        kept = {self.driver.block_of(m) for m in ranks}
        dropped = sorted((self.primary_block_ids - kept) - blended)
        rescaled = sorted(b for b, bs in state.blocks.items()
                          if b in self.primary_block_ids and b not in dropped and bs.primary_enabled
                          and abs(bs.primary_strength - 1.0) > 1e-9)
        return {"dropped_blocks": dropped, "rescaled_blocks": rescaled, "blended_blocks": sorted(blended),
                "keys_in": 3 * len(self.net._frozen[PRIMARY]["alpha_rank"]), "keys_out": len(sd),
                "donor_path": self.donor_path if use_donor else None, "format_out": "standard",
                "lycoris_converted": 0, "use_at": use_at,
                "strengths_baked": bool(blended)}

    # ---- teardown -----------------------------------------------------------------------------------
    def forget_prompts(self):
        """Drop the cached prompt conditioning (a new prompt was applied)."""
        self._prompt_cache.clear()

    def reset(self):
        from fizgig.utils.device import release_module_tensors
        # a driver's VAE may be a dict of modules (a video family's: picture + sound decoders)
        vaes = list(self.vae.values()) if isinstance(self.vae, dict) else [self.vae]
        for m in [self.dit] + vaes:
            if isinstance(m, torch.nn.Module):
                try:
                    release_module_tensors(m)
                except Exception:
                    pass
        self.dit = self.vae = self.net = None
        self.pipeline = None
        self.speed = None
        self._speed_follow = None
        self._speed_path = ""
        self.primary_network = self.donor_network = None
        self.primary_path = self.donor_path = None
        self.primary_block_ids, self.donor_block_ids = set(), set()
        self.primary_hash = None
        self._clear_blend_state()
        self._eq = {}
        self._prompt_cache = {}
        getattr(self.driver, "drop_text_encoder", lambda: None)()
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        self._act_modules = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            from fizgig.utils.device import flush_reserved_vram, report_cuda_leak
            report_cuda_leak(f"{self.desc.key}-workbench-reset")
            flush_reserved_vram(f"{self.desc.key}-workbench-reset")
        except Exception:
            pass
