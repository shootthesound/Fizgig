"""MiniMax H3's workbench engine for the family workbench tabs (Repair Studio, Explorer, Royale, Profiler, RefMod
Studio): the H3 Repair engine, unchanged - so every render is today's - with the generic workbench surface the tabs
use for any described family (families/workbench.py WorkbenchEngine) on top.

It is also the first VIDEO workbench engine. A family whose description has clips ("clip" in media) and names a
`workbench_engine` gets the tabs' clip features through these calls, which a future video family implements the same
way (render_clip, baseline_clip, nolora_clip, clip_from_cache, cache_key_for, encode_keyframe,
encode_reference_image, reference_items, render_refmod, decode_* - see H3RepairEngine for each one's contract).
"""
from fizgig.repair_studio.h3_engine import H3RepairEngine


class H3WorkbenchEngine(H3RepairEngine):
    video = True                       # the tabs' clip features (player, library, keyframes, references) apply

    def __init__(self, description):
        super().__init__()
        self.desc = description
        self.driver = description.load_driver()
        self.preview_settings = None   # the generic tabs set it for families that follow the Samples tab

    # ---- the generic surface -----------------------------------------------------------------------------------
    def block_groups(self):
        return self.driver.block_map(self.dit)

    def block_ids(self):
        return [b.id for g in self.block_groups() for b in g.blocks]

    def default_state(self, width=None, height=None):
        from fizgig.repair_studio.state import SliderState
        s = SliderState.default_h3()
        if width:
            s.preview_width, s.preview_height = int(width), int(height or width)
        return s

    @property
    def reference_kind(self):
        return ""                      # previews take no single reference picture (keyframes / refs are clip features)

    @property
    def supports_prompt_travel(self):
        return True                    # override_ctx drives H3's prompt travel

    def ensure_pipeline(self, dit_path, vae_path, text_encoder_path, speed_lora_path="", device="cuda", lowmem=None,
                        precision="auto", blocks_to_swap=0, preview_sampling=None, turbo_lora_path="",
                        turbo_lora_strength=0.75, te_cache_dir="", audio_vae_path="", base_mode="auto", **_ignored):
        """The generic call (the family's speed LoRA = H3's Turbo) mapped onto the H3 engine's own: base precision is
        H3's planner (base_mode auto / stream / nf4), not the generic precision preference."""
        out = super().ensure_pipeline(dit_path, vae_path, text_encoder_path, device=device,
                                      turbo_lora_path=speed_lora_path or turbo_lora_path,
                                      turbo_lora_strength=turbo_lora_strength, te_cache_dir=te_cache_dir,
                                      audio_vae_path=audio_vae_path, base_mode=base_mode)
        self.resume_enabled = self.turbo_preview          # the load turns the resume on; the tick has the last word
        return out

    # ---- Reuse unchanged blocks: the tick is the H3 engine's exact pass-1 resume (resume_enabled) ----------------------------
    @property
    def turbo_preview(self):
        return bool(getattr(self, "_turbo_wanted", True))

    @turbo_preview.setter
    def turbo_preview(self, on):
        self._turbo_wanted = bool(on)
        self.resume_enabled = bool(on)
        self._invalidate_activation_cache()

    # ---- the family LoRA's switches the Profiler's ablation uses (engine.net.set_enabled / set_outside) ------------
    @property
    def net(self):
        return _PrimarySwitches(self)

    def apply_state(self, state):
        """The H3 engine's slider push, then the two whole-LoRA switches on top: the primary fully off (every module,
        its AdaLN rows too), or only its modules outside the block map off."""
        super().apply_state(state)
        if self.primary_network is None:
            return
        on, outside = getattr(self, "_primary_on", True), getattr(self, "_outside_on", True)
        if on and outside:
            return
        import re
        from fizgig.repair_studio.h3_blocks import block_regex_h3
        pats = [block_regex_h3(b) for b in self.block_ids()]
        for m in self.primary_network.unet_loras:
            if not on or not any(re.search(p, m.lora_name) for p in pats):
                m.enabled = False
        if not on:
            self._reinstall_adaln(no_lora=True)

    def forget_prompts(self):
        """Drop the cached prompt conditioning (a new prompt was applied)."""
        self._prompt_cache_key = None
        self._prompt_cache = None

    def save_repaired(self, out_path, state, include_donor=True):
        """The file-based bake H3 has always used (repair_studio/bake.py), so a saved LoRA is today's byte for byte."""
        from fizgig.repair_studio.bake import save_repaired_lora
        return save_repaired_lora(self.primary_path, state, out_path,
                                  donor_path=self.donor_path if (include_donor and self.donor_path) else None)


class _PrimarySwitches:
    """engine.net for the Profiler: set_enabled(PRIMARY, on) / set_outside(PRIMARY, on) on the H3 engine's primary
    LoRA (the donor has no switch here - the Profiler never loads one)."""

    def __init__(self, engine):
        self.engine = engine

    def set_enabled(self, _name, enabled):
        self.engine._primary_on = bool(enabled)
        if getattr(self.engine, "_last_state", None) is not None:
            self.engine.apply_state(self.engine._last_state)

    def set_outside(self, _name, enabled):
        self.engine._outside_on = bool(enabled)
        if getattr(self.engine, "_last_state", None) is not None:
            self.engine.apply_state(self.engine._last_state)
