"""The workbench for video families: clips in Repair Studio and LoRA Royale.

Two pieces:

* `ClipContract` - what the tabs call on any video engine: render a clip for a slider state (decoded to every frame,
  the middle frame and the sound), the baseline and no-LoRA clips beside it, a clip straight from the render cache,
  and the keys / labels / regimes those use. It is written once, over five primitives an engine provides:
    render_latent(state, *, frames, steps, turbo_strength, on_denoised, no_lora, ...) -> (latent, audio_rows)
    decode_clip_frames(latent) -> [PIL]          decode_audio(audio_rows) -> waveform [2, L] or None
    decode_middle_frame_image(latent) -> PIL     _clip_base_state() -> a default SliderState
* `VideoWorkbenchEngine` - the generic engine (families/workbench.py WorkbenchEngine) with those primitives built on
  the family's driver: generate(frames=, audio=) and decode. A video family gets the tabs' clip features from its
  description and driver alone.

MiniMax H3's own engine (repair_studio/h3_engine.py) uses the same ClipContract over its own primitives (keyframes,
references, its exact pass-1 resume) - so the contract every video family relies on is the one H3 runs every day.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from fizgig.families.workbench import DONOR, PRIMARY, SPEED, RenderCancelled, WorkbenchEngine

logger = logging.getLogger(__name__)


class ClipContract:
    """The tabs' clip calls, over the engine's primitives (see the module docstring). Hooks an engine may override:
    default_clip_frames, _clip_regimes(), _speed_loaded(), _plain_steps(), block_label(), _int8_tag()."""

    video = True
    default_clip_frames = 22
    # what an engine adds beyond the contract (the tabs show those controls only for an engine that has them):
    # first / last frame keyframes and reference pictures (encode_keyframe / encode_reference_image), a base-model
    # and base-mode picker, the render library's block banks
    supports_keyframes = False
    keyframe_slots = ("first", "last")       # the frames a keyframe can pin (Vimoo: the first only)
    supports_show_early = False              # render_clip(on_early=) puts up a mid-render estimate (H3's engine)
    supports_base_modes = False
    supports_banks = False

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _clip_conditioning(self, state, cond, width, height, frames):
        """(cond, extra generate() arguments) for one clip: a family's engine adds what its keyframes or references
        need here (Vimoo: a start picture's latent and SigLIP tokens). Default: the prompt alone."""
        return cond, {}

    def _clip_regimes(self) -> dict:
        """{name: (steps, speed strength)}: the description's clip_regimes."""
        desc = getattr(self, "desc", None)
        return {n: (int(st), float(tu)) for n, st, tu in (getattr(desc, "clip_regimes", ()) or ())}

    def _speed_loaded(self) -> bool:
        return False

    def _plain_steps(self) -> int:
        return 20

    def block_label(self, bid: str) -> str:
        return bid

    def _int8_tag(self) -> bool:
        return bool(getattr(self, "int8_attention", False))

    def _clip_base_state(self):
        return self.default_state()

    # ---- the contract ------------------------------------------------------------------------------------------
    def regime_params(self, regime: str, steps=None, turbo_strength=None):
        """(steps, speed strength) for a regime name - the description's preset unless the caller dials its own:
        `steps` and `turbo_strength` override (0 = the speed LoRA switched off for the render). Without a speed LoRA
        loaded the strength has nothing to dial (None) and the steps default to the plain count."""
        if not self._speed_loaded():
            return (int(steps) if steps else self._plain_steps()), None
        regimes = self._clip_regimes()
        if regimes:
            st, tu = regimes.get(regime, regimes.get("confirm", next(iter(regimes.values()))))
        else:
            st, tu = self._plain_steps(), 1.0
        if steps:
            st = max(1, int(steps))
        if turbo_strength is not None:
            tu = float(turbo_strength)
        return st, tu

    @staticmethod
    def keyframe_signature(state):
        """A hashable stand-in for state.keyframes / state.references (index + tensor fingerprint per entry) - cheap
        enough for a cache key, specific enough that a different crop re-renders."""
        kf = getattr(state, "keyframes", None) or []
        refs = getattr(state, "references", None) or []
        if not kf and not refs:
            return ()
        sig = []
        for idx, lat in kf:
            t = (lat["latent"] if isinstance(lat, dict) else lat).float()
            sig.append((int(idx), tuple(t.shape), round(float(t.sum()), 3), round(float(t.abs().mean()), 5)))
        for i, (_img, lat) in enumerate(refs):
            t = lat.float()
            sig.append(("ref", i, tuple(t.shape), round(float(t.sum()), 3), round(float(t.abs().mean()), 5)))
        return tuple(sig)

    def clip_key(self, state, *, frames, steps, turbo_strength, with_audio):
        return (self.primary_path, self.donor_path, int(state.seed), state.prompt,
                int(state.preview_width), int(state.preview_height), int(frames), int(steps),
                turbo_strength, bool(with_audio), self.keyframe_signature(state),
                round(float(getattr(state, "primary_scale", 1.0)), 4),
                round(float(getattr(state, "donor_scale", 1.0)), 4),
                bool(getattr(self, "int8_attention", False)), bool(getattr(self, "speed_boost", False)))

    def cache_key_for(self, state, *, frames, regime, steps=None, turbo_strength=None, **_ignored):
        """The render library's setup key for this state's render setup, or None before a primary is loaded. Sound
        doesn't enter the key: audio rows are part of every entry. A family other than H3 adds its key, so two
        families never share an entry (H3's keys stay what its existing libraries were built under)."""
        if self.primary_network is None or not getattr(self, "primary_hash", None):
            return None
        from fizgig.repair_studio.h3_render_cache import setup_key
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = self._clip_frames(state, frames)
        fam = getattr(getattr(self, "desc", None), "key", "minimax")
        return setup_key(primary_hash=self.primary_hash, donor_hash=getattr(self, "donor_hash", "") or "",
                         prompt=state.prompt, seed=int(state.seed), frames=frames,
                         width=int(state.preview_width), height=int(state.preview_height),
                         steps=int(steps), turbo_strength=strength,
                         keyframe_sig=self.keyframe_signature(state),
                         int8_attention=bool(getattr(self, "int8_attention", False)),
                         primary_scale=float(getattr(state, "primary_scale", 1.0)),
                         donor_scale=float(getattr(state, "donor_scale", 1.0)),
                         dit=os.path.basename(getattr(self, "dit_path", "") or "") + ("" if fam == "minimax" else
                                                                                     f"|{fam}"))

    def _clip_frames(self, state, frames):
        return int(frames or getattr(state, "preview_frames", 0) or self.default_clip_frames)

    def render_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                    with_audio: bool = True, cache=None, early_step: int = 0,
                    on_early=None, no_lora: bool = False, steps=None, turbo_strength=None,
                    decode: bool = True, **_ignored) -> dict:
        """Render + decode one clip for the slider state. Returns
        {"latent", "audio_rows", "frames": [PIL...], "wav": [2, L] or None, "middle": PIL,
         "regime", "steps", "turbo_strength", "frames_n", "cached": bool}.

        cache: a RenderCache for this setup - a state rendered before is served from it (decode only, "cached":
        True); a fresh render is stored into it under the state's signature. early_step + on_early: "show early" -
        after pass `early_step` the clean-latent estimate's middle frame is decoded and handed to on_early(pil, step,
        n) while the remaining passes run. Never fires on a cache hit. no_lora renders the base model alone (see
        render_latent) under the "nolora" signature."""
        from fizgig.repair_studio.h3_render_cache import NOLORA_SIG, signature
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = self._clip_frames(state, frames)
        sig = NOLORA_SIG if no_lora else signature(state)
        hit = cache.get(sig) if cache is not None else None
        if hit is not None:
            lat, aud = hit
            cached = True
        else:
            def _on_denoised(step, n, x0):
                if on_early is None or step != int(early_step):
                    return
                on_early(self.early_frame_image(x0), step, n)

            lat, aud = self.render_latent(state, frames=frames, steps=steps, turbo_strength=strength,
                                          on_denoised=_on_denoised if early_step > 0 else None, no_lora=no_lora)
            cached = False
        if decode:
            imgs = self.decode_clip_frames(lat)
            wav = self.decode_audio(aud) if (with_audio and frames > 1) else None
            middle = imgs[len(imgs) // 2]
        else:
            # the library builder: the latent is the entry, a thumb is enough - and not even that if a cancel is
            # already waiting (the render itself is never thrown away)
            imgs, wav = [], None
            _ev = getattr(self, "_cancel_event", None)
            middle = None if (_ev is not None and _ev.is_set()) else self.decode_middle_frame_image(lat)
        clip = {"latent": lat, "audio_rows": aud, "frames": imgs, "wav": wav, "middle": middle, "regime": regime,
                "steps": steps, "turbo_strength": strength, "frames_n": frames, "cached": cached, "sig": sig,
                "int8_attention": self._int8_tag()}
        if cache is not None and not cached:
            try:
                cache.put(sig, lat, aud, middle=clip["middle"], regime=regime,
                          label="No LoRA" if no_lora else self.describe_state(state),
                          state=None if no_lora else state.to_json())
            except Exception:
                logger.exception("render cache: put failed (render still shown)")
        return clip

    def clip_from_cache(self, cache, sig: str, *, regime: str = "dial",
                        with_audio: bool = True, steps=None, turbo_strength=None) -> Optional[dict]:
        """A clip dict straight from a cached entry (history strip, peeks, pinned baseline) - decode only, no state
        needed. None when the entry isn't there."""
        hit = cache.get(sig) if cache is not None else None
        if hit is None:
            return None
        lat, aud = hit
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        imgs = self.decode_clip_frames(lat)
        wav = self.decode_audio(aud) if (with_audio and len(imgs) > 1) else None
        return {"latent": lat, "audio_rows": aud, "frames": imgs, "wav": wav, "middle": imgs[len(imgs) // 2],
                "regime": regime, "steps": steps, "turbo_strength": strength, "frames_n": len(imgs),
                "cached": True, "int8_attention": self._int8_tag(), "sig": sig,
                "label": cache.info(sig).get("label", "")}

    def describe_state(self, state) -> str:
        """A short human label for a slider state ("Block 30 off", "3 blocks moved")."""
        from fizgig.repair_studio.h3_render_cache import BASE_SIG, signature
        sig = signature(state)
        if sig == BASE_SIG:
            return "Baseline"
        if sig.startswith("off:"):
            return self.block_label(sig[4:]) + " off"
        if sig.startswith("bank:"):
            a, b = sig[5:].split("-")
            return f"Blocks {a}–{b} off"
        moved = [b for b, bs in state.blocks.items()
                 if not (bs.primary_enabled and abs(bs.primary_strength - 1.0) < 1e-6
                         and abs(bs.donor_strength) < 1e-6)]
        return f"{len(moved)} block{'s' if len(moved) != 1 else ''} moved"

    def _base_like(self, state, frames, with_scales):
        base = self._clip_base_state()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        base.preview_frames = frames
        base.keyframes = getattr(state, "keyframes", None)
        base.references = getattr(state, "references", None)
        if with_scales:
            base.primary_scale = float(getattr(state, "primary_scale", 1.0))
        base.donor_scale = 0.0              # the primary alone: a loaded donor stays out of the baseline clip
        for bs in base.blocks.values():
            bs.donor_enabled = False
        return base

    def baseline_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                      with_audio: bool = True, cache=None, steps=None, turbo_strength=None,
                      **_ignored) -> dict:
        """The clip for the primary at its load strength / all on, donor off - cached in memory on everything the
        render depends on (a slider move never re-renders it; a regime, length, size, seed, prompt, scale or keyframe
        change does) and, through `cache`, on disk."""
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = self._clip_frames(state, frames)
        key = self.clip_key(state, frames=frames, steps=steps, turbo_strength=strength, with_audio=with_audio)
        if getattr(self, "_baseline_clip_key", None) == key and getattr(self, "_baseline_clip", None) is not None:
            return self._baseline_clip
        clip = self.render_clip(self._base_like(state, frames, True), frames=frames, regime=regime,
                                with_audio=with_audio, cache=cache, steps=steps, turbo_strength=strength)
        self._baseline_clip_key = key
        self._baseline_clip = clip
        return clip

    def nolora_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                    with_audio: bool = True, cache=None, steps=None, turbo_strength=None,
                    **_ignored) -> dict:
        """The same seed / prompt / canvas / length / keyframes rendered by the base model with no LoRA at all - the
        player's third pane. Cached in memory like the baseline (a slider move never re-renders it) and on disk under
        "nolora"."""
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = self._clip_frames(state, frames)
        key = self.clip_key(state, frames=frames, steps=steps, turbo_strength=strength, with_audio=with_audio)
        if getattr(self, "_nolora_clip_key", None) == key and getattr(self, "_nolora_clip", None) is not None:
            return self._nolora_clip
        clip = self.render_clip(self._base_like(state, frames, False), frames=frames, regime=regime,
                                with_audio=with_audio, cache=cache, no_lora=True, steps=steps,
                                turbo_strength=strength)
        self._nolora_clip_key = key
        self._nolora_clip = clip
        return clip


class VideoWorkbenchEngine(ClipContract, WorkbenchEngine):
    """The generic video engine: the stills engine (loading, sliders, prompts, cancel, Reuse unchanged blocks, bake) plus
    clips, rendered and decoded by the family's driver. The driver's generate(frames=, audio=) returns
    {"latent", "audio"} and its decode turns that into {"frames": [3, F, H, W] in 0-1, "image": middle frame,
    "wave": [2, L] or None} for a clip (a PIL image for a still)."""

    def __init__(self, description):
        super().__init__(description)
        spec = getattr(description, "clip_spec", None)
        if spec is not None:
            from fizgig.families.clips import grid_frames
            grid = grid_frames(spec)
            self.default_clip_frames = grid[1] if len(grid) > 1 else grid[0]
        self._speed_strength = None

    def ensure_pipeline(self, dit_path, vae_path, text_encoder_path, speed_lora_path="", device="cuda", lowmem=None,
                        precision="auto", blocks_to_swap=0, preview_sampling=None, audio_vae_path="",
                        family_options=None, **_ignored):
        """The stills engine's load, plus what clips need: the driver gets the run's family options and the sound
        decoder (the model file whose role is "audio_vae", as options["audio_vae"] - what the cache stage gets as
        --aux audio_vae) before its VAE loads."""
        if self.pipeline is not None:
            return
        opts = dict(family_options or {})
        if audio_vae_path and os.path.exists(audio_vae_path):
            opts["audio_vae"] = audio_vae_path
        self.driver.set_options({**getattr(self.driver, "options", {}), **opts})
        self.dit_path = dit_path
        super().ensure_pipeline(dit_path, vae_path, text_encoder_path, speed_lora_path=speed_lora_path, device=device,
                                lowmem=lowmem, precision=precision, blocks_to_swap=blocks_to_swap,
                                preview_sampling=preview_sampling)
        # a big card (40 GB+) keeps the decoders on the GPU between renders (moving a video VAE each time costs
        # seconds); on 32 GB and below they go back to the CPU after each decode, so the card isn't left full while the
        # page plays the clip (Vimoo bf16 at 768x768 x 49 held 31.4 of 32 GB, 6 Nov 2026)
        from fizgig.families import quant
        big = quant.card_gb() >= quant.NOGC_MIN_CARD_GB
        self.driver.keep_vae_resident = not self.lowmem and big
        # and the model steps off the card for each decode there: a clip's decode beside a bf16 model peaked 30.6 GB on
        # a 32 GB card, which Windows pages into system RAM (a 768x768 x 49 pair took 286 s instead of ~45)
        self.park_for_decode = self.lowmem or not big
        self._speed_strength = self.speed.strength if self.speed is not None else None

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _speed_loaded(self):
        return self.net is not None and self.net.has(SPEED)

    def _plain_steps(self):
        s = self.desc.default_sampling()
        return s.steps if s else 20

    def block_label(self, bid):
        for g in self.block_groups():
            for b in g.blocks:
                if b.id == bid:
                    return b.label
        return bid

    def _set_speed_strength(self, strength):
        """The speed LoRA at this render's strength (0 = off), the driver told when it carries weights the LoRA
        layer cannot wrap (H3's AdaLN rows)."""
        if not self._speed_loaded() or strength is None or strength == self._speed_strength:
            return
        self._speed_strength = strength
        on = strength > 0
        self.net.set_enabled(SPEED, on)
        if on:
            self.net.set_strength(SPEED, strength)
        self.driver.frozen_file_added(self.dit, self._speed_path, strength if on else 0.0, "speed")

    # ---- primitives --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def render_latent(self, state, *, seed=None, prompt=None, width=None, height=None, frames=None, steps=None,
                      turbo_strength=None, on_denoised=None, no_lora=False, override_ctx=None, noise=None, **_ignored):
        """Apply the slider state and sample one clip through the driver -> (latent, audio rows), on CPU. no_lora:
        the base model alone for this render (the speed LoRA stays: it is the sampler, not the LoRA under test).
        on_denoised (show early) needs a driver hook the shared contract doesn't have yet, so it is not called."""
        self.apply_state(state)
        seed = state.seed if seed is None else seed
        prompt = state.prompt if prompt is None else prompt
        width = int(width or state.preview_width)
        height = int(height or state.preview_height)
        frames = self._clip_frames(state, frames)
        steps = int(steps or self._plain_steps())
        self._set_speed_strength(turbo_strength)
        cond = override_ctx if override_ctx is not None else self.encode([prompt], size=(width, height))[0]
        cond, extra = self._clip_conditioning(state, cond, width, height, frames)
        _d_steps, cfg, sigmas, options = self.sampling()
        if self._speed_loaded() and not self._speed_strength:
            # the speed LoRA off for this render: the base model's own guidance and schedule
            ds = self.desc.default_sampling()
            if ds is not None:
                cfg, sigmas, options = ds.cfg, None, ds.options
        neg = None
        if cfg > 1.0:
            # the page's Negative (sent with each clip render), else the model's own default
            text = getattr(self, "clip_negative", None)
            text = self.desc.preview_negative if text is None else text
            neg = self.encode([text or ""], size=(width, height))[0]
            extra["neg_cond"] = self._cond_to_device(neg)
        logger.info("[clip] %d steps, CFG %g%s, speed LoRA %s", steps, cfg, " with the negative" if neg is not None else "",
                    f"at {self._speed_strength:g}" if self._speed_loaded() and self._speed_strength else "off")
        names = [n for n in (PRIMARY, DONOR) if self.net.has(n)] if no_lora else []
        for n in names:
            self.net.set_enabled(n, False)
        # Reuse unchanged blocks: the same setup key generate_preview builds, plus the clip's length and speed strength
        self._act_ctx = None
        if not no_lora and not self._act_bypass and override_ctx is None and noise is None:
            self._act_ctx = ((id(self.dit), self.primary_path, self.primary_hash, self.donor_path, self._speed_path,
                              round(float(getattr(state, "primary_scale", 1.0)), 4),
                              round(float(getattr(state, "donor_scale", 1.0)), 4), prompt, int(seed), width, height,
                              frames, turbo_strength, self.keyframe_signature(state)), self._cache_sig(state))
        act = self._act_for_render()
        key = None
        if act is not None:
            k, sig = self._act_ctx
            key = repr((k, steps, cfg, bool(self.int8_attention)))

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

        def _generate():
            with _i8a.renders(self.int8_attention):
                return self.driver.generate(self.dit, self._cond_to_device(cond), width, height, steps=steps,
                                            seed=int(seed), cfg=cfg, sigmas=sigmas, options=options, on_step=_step,
                                            frames=frames, audio=True, **extra,
                                            **({"noise": noise} if noise is not None else {}))
        try:
            if act is None:
                out = _generate()
            else:
                with act.render(self.dit, self._act_modules, key, sig):
                    out = _generate()
        finally:
            self._act_ctx = None
            for n in names:
                self.net.set_enabled(n, True)
            if names:
                self.apply_state(state)
        lat = out["latent"].detach().float().cpu()
        aud = out.get("audio")
        return lat, (aud.detach().float().cpu() if aud is not None else None)

    def _preview_clip_frames(self):
        """The shortest clip the family offers (a still is not a clip): what a picture render stands on."""
        vc = getattr(self.desc, "video_controls", None)
        clips = sorted(int(n) for _l, n in (vc.lengths if vc else ()) if int(n) > 1)
        return clips[0] if clips else self.default_clip_frames

    @torch.no_grad()
    def generate_preview(self, state, *, seed=None, prompt=None, width=None, height=None, steps=None,
                         num_frames=None, override_ctx=None, seed_b=None, travel_t=0.0, **_ignored):
        """The picture tabs' render call (Profiler, LoRA Royale, Studio's stills paths), answered as H3's engine
        answers it: a short clip at the confirm regime, its middle frame. override_ctx (prompt travel) is the clip's
        conditioning; seed_b / travel_t (seed travel) slerp the two seeds' starting noise; a chained previous frame
        has no clip wiring and is ignored, as on H3."""
        from fizgig.families.workbench import _slerp
        frames = int(num_frames or self._preview_clip_frames())
        st, strength = self.regime_params("confirm", steps, None)
        noise = None
        if seed_b is not None:
            seed0 = state.seed if seed is None else seed
            w, h = int(width or state.preview_width), int(height or state.preview_height)
            noise = _slerp(float(travel_t or 0.0), self.driver.initial_noise(seed0, w, h, frames),
                           self.driver.initial_noise(seed_b, w, h, frames))
        lat, _aud = self.render_latent(state, seed=seed, prompt=prompt, width=width, height=height, frames=frames,
                                       steps=st, turbo_strength=strength, override_ctx=override_ctx, noise=noise)
        return self.decode_middle_frame_image(lat)

    def _decode(self, lat):
        park = getattr(self, "park_for_decode", self.lowmem)
        if park:
            self._park_dit("cpu")
        try:
            return self.driver.decode(self.vae, {"latent": lat, "audio": None}, None, None)
        finally:
            if park:
                self._park_dit(self.device)

    @torch.no_grad()
    def decode_clip_frames(self, latent):
        from PIL import Image
        out = self._decode(latent)
        if not isinstance(out, dict):
            return [out]
        px = (out["frames"].permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        return [Image.fromarray(px[i]) for i in range(px.shape[0])]

    @torch.no_grad()
    def decode_audio(self, audio_rows):
        if audio_rows is None:
            return None
        return self.driver.decode_audio(self.vae, audio_rows)

    @torch.no_grad()
    def early_frame_image(self, latent):
        """Show early's mid-render look; None skips it (a model whose decode won't fit beside the render)."""
        return self.decode_middle_frame_image(latent)

    def decode_middle_frame_image(self, latent):
        out = self._decode(latent)
        return out["image"] if isinstance(out, dict) else out
