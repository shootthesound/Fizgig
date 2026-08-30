import json
import os
import sys
import threading

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY
from fizgig_gui.core.domain.architectures import ARCHITECTURES
from fizgig_gui.core.domain.minimax_math import minimax_lownoise_to_shift, MINIMAX_STRUCTURE_DEFAULT, \
    MINIMAX_STRUCTURE_OPTIONS, MINIMAX_STRUCTURE_DESC, minimax_block_spec, MINIMAX_NUM_BLOCKS, MINIMAX_LIKENESS_BLOCKS
from fizgig_gui.core.config.prefs import _auto_detect_blocks_to_swap

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class TrainingVisibilityMixin:
    def update_ui_for_architecture(self):
        """Update UI elements based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Window title is set once in __init__ and stays put — architecture is
        # hardcoded to Klein Base 9B, no need to mirror it in the title bar.

        # Show/hide CLIP model field
        if config["uses_clip"]:
            self.show_row("CLIP_MODEL")
        else:
            self.hide_row("CLIP_MODEL")

        # Show/hide T5 model field
        if config["uses_t5"]:
            self.show_row("T5_MODEL")
        else:
            self.hide_row("T5_MODEL")

        # Show/hide Text Encoder field (for Z-Image and Flux 2)
        if config["uses_text_encoder"]:
            self.show_row("TEXT_ENCODER")
            if "TEXT_ENCODER" in self.labels:  # Model Paths section may have been removed
                self.labels["TEXT_ENCODER"].config(text=f"{config['text_encoder_label']}:")
        else:
            self.hide_row("TEXT_ENCODER")

        # Show/hide Model Type dropdown (Wan only) and update options
        if config["uses_model_type"]:
            self.show_row("MODEL_TYPE")
            # Update MODEL_TYPE dropdown values for this architecture
            model_types = config.get("model_types", ["t2v-14B", "i2v-14B"])
            self.entries["MODEL_TYPE"]["values"] = model_types
            current_val = self.entries["MODEL_TYPE"].get()
            if current_val not in model_types:
                self.entries["MODEL_TYPE"].current(0)
        else:
            self.hide_row("MODEL_TYPE")

        # Update VAE label (Model Paths section may have been removed)
        if "VAE_MODEL" in self.labels:
            self.labels["VAE_MODEL"].config(text=f"{config['vae_label']}:")

        # Update FP8 text encoder checkbox label
        if arch.startswith("Wan"):
            self.fp8_text_encoder_check.config(text="Enable FP8 T5")
        elif arch.startswith("Z-Image"):
            self.fp8_text_encoder_check.config(text="Enable FP8 LLM")
        else:
            self.fp8_text_encoder_check.config(text="Enable FP8 Text Encoder")

        # Update blocks swap max (enforce limit). "Auto" is left ALONE: resolving it here
        # ran the auto strategy (a GPU probe that can flip the 4-bit toggle) as a side
        # effect of a bounds check, and writing the resolved number into the combobox
        # silently turned an auto choice into a permanent manual one. Auto strategies
        # already return in-range values; only explicit numbers need clamping.
        try:
            _raw_swap = self.entries["BLOCKS_SWAP"].get().strip()
            if not _raw_swap.lower().startswith("auto"):
                current_blocks = self._parse_blocks_swap()
                if current_blocks > config["blocks_swap_max"]:
                    self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                    self.entries["BLOCKS_SWAP"].insert(0, str(config["blocks_swap_max"]))
        except ValueError:
            pass

        # Update timestep section for architecture.
        # Values are only touched when the ARCHITECTURE actually changed: several callers
        # use this method as a pure visibility refresh, and unconditionally resetting the
        # section wiped the user's timestep settings on every call. On a real switch, the
        # outgoing family's values are stashed and restored when the user switches back.
        if hasattr(self, 'ts_sampling_var'):
            _prev_arch = getattr(self, "_ts_defaults_arch", None)
            if _prev_arch != arch:
                if not hasattr(self, "_arch_ts_stash"):
                    self._arch_ts_stash = {}
                if _prev_arch is not None:
                    self._arch_ts_stash[_prev_arch] = {
                        "sampling": self.ts_sampling_var.get(),
                        "shift": self.entries["DISCRETE_FLOW_SHIFT"].get(),
                        "min_ts": self.entries["MIN_TIMESTEP"].get(),
                        "max_ts": self.entries["MAX_TIMESTEP"].get(),
                        "preserve": self.preserve_dist_var.get(),
                        "weighting": self.weighting_scheme_var.get(),
                    }
                stash = self._arch_ts_stash.get(arch)
                if stash is not None:
                    # Returning to a family the user already configured — restore, don't reset.
                    self.ts_sampling_var.set(stash["sampling"])
                    self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                    self.entries["DISCRETE_FLOW_SHIFT"].delete(0, tk.END)
                    self.entries["DISCRETE_FLOW_SHIFT"].insert(0, stash["shift"])
                    self.entries["MIN_TIMESTEP"].delete(0, tk.END)
                    self.entries["MIN_TIMESTEP"].insert(0, stash["min_ts"])
                    self.entries["MAX_TIMESTEP"].delete(0, tk.END)
                    self.entries["MAX_TIMESTEP"].insert(0, stash["max_ts"])
                    self.preserve_dist_var.set(stash["preserve"])
                    self.weighting_scheme_var.set(stash["weighting"])
                else:
                    # First visit to this family — apply its defaults.
                    self.ts_sampling_var.set(config.get("timestep_sampling", "shift"))
                    default_shift = config.get("discrete_flow_shift")
                    if default_shift is not None:
                        self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                        self.entries["DISCRETE_FLOW_SHIFT"].delete(0, tk.END)
                        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, str(default_shift))
                    min_ts = config.get("min_timestep")
                    max_ts = config.get("max_timestep")
                    self.entries["MIN_TIMESTEP"].delete(0, tk.END)
                    self.entries["MAX_TIMESTEP"].delete(0, tk.END)
                    if min_ts is not None:
                        self.entries["MIN_TIMESTEP"].insert(0, str(min_ts))
                    if max_ts is not None:
                        self.entries["MAX_TIMESTEP"].insert(0, str(max_ts))
                    self.preserve_dist_var.set(config.get("preserve_distribution_shape", False))
                self._ts_defaults_arch = arch

            # Enable/disable states are pure display — refresh them on every call.
            supports_shift = config.get("supports_discrete_flow_shift", True)
            if supports_shift:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                self.ts_flow_shift_label.config(fg=COLORS["text_secondary"])
            else:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="disabled")
                self.ts_flow_shift_label.config(fg=COLORS["text_muted"])

            supports_weighting = config.get("supports_weighting_scheme", True)
            if supports_weighting:
                self.ts_weighting_combo.config(state="readonly")
                self.ts_weighting_label.config(fg=COLORS["text_secondary"])
            else:
                self.weighting_scheme_var.set("none")
                self.ts_weighting_combo.config(state="disabled")
                self.ts_weighting_label.config(fg=COLORS["text_muted"])

            # Refresh conditional field states
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()

        # Krea 2 hides Training-tab features that aren't wired into its native trainer yet.
        # These are DEFERRED, not removed — each is re-enabled simply by dropping it from the
        # hide lists in _apply_training_arch_visibility once krea2_train supports it.
        self._apply_training_arch_visibility(config.get("is_krea2", False))

    @staticmethod
    def _select_combo_by_token(combo, value):
        """Select the option whose FIRST whitespace-separated token equals `value`.

        Labelled dropdowns here carry a trailing note ("12 - reference default, 5% detail band")
        while settings/presets store only the bare token. Matching on the token means the note
        can be reworded without invalidating anyone's saved runs or queued items."""
        want = str(value).split(" ")[0]
        try:
            for opt in (combo.cget("values") or ()):
                if str(opt).split(" ")[0] == want:
                    combo.set(str(opt))
                    return True
        except tk.TclError:
            pass
        return False

    def _refresh_minimax_shift_match(self):
        """Show the schedule the typed percentage produces, or why it can't be read.

        The relationship is very non-linear at the ends — 5% is shift 19, 2% is shift 49 — so the
        resulting shift and median noise level are worth seeing next to the number you typed."""
        lbl = getattr(self, "_minimax_shift_match", None)
        ent = self.entries.get("MINIMAX_LOWNOISE_PCT")
        if lbl is None or ent is None or not lbl.winfo_exists():
            return
        shift = minimax_lownoise_to_shift(ent.get())
        if shift is None:
            lbl.config(text="✗ enter a number above 0 and below 100", fg="#E74C3C")
            return
        # The median is the shift map at the uniform base's median draw, so shift/(shift+1).
        med = shift / (shift + 1.0)
        lbl.config(text=f"→ shift {shift:.3g}, median noise {med:.2f}", fg="#27AE60")

    def _build_minimax_structure_row(self, parent):
        """Training Structure — the MiniMax timestep density, named.

        Rows 22-26 of Training Parameters, under Network Type. The structure dropdown is a VIEW of
        MINIMAX_LOWNOISE_PCT rather than a setting of its own, so every existing preset and saved
        run keeps working with no migration: 60 shows Face likeness, anything unrecognised shows
        Custom and reveals the box it came from.
        """
        self._minimax_structure_label = ttk.Label(parent, text="Training Structure:")
        self._minimax_structure_label.grid(row=22, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self.minimax_structure_var = tk.StringVar(value=MINIMAX_STRUCTURE_DEFAULT)
        self._minimax_structure_combo = ttk.Combobox(
            parent, textvariable=self.minimax_structure_var,
            values=list(MINIMAX_STRUCTURE_OPTIONS), state="readonly", width=36)
        self._minimax_structure_combo.grid(row=22, column=1, columnspan=2, sticky=tk.W,
                                           padx=5, pady=(8, 2))
        self._minimax_structure_combo.bind("<<ComboboxSelected>>",
                                           lambda _e: self._on_minimax_structure_changed())

        self._minimax_structure_desc = tk.Label(
            parent, text="", font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
            bg=COLORS["bg_surface"], justify=tk.LEFT, wraplength=700)
        self._minimax_structure_desc.grid(row=23, column=0, columnspan=3, sticky=tk.W,
                                          padx=(12, 5), pady=(0, 4))

        # Shown when the dataset carries voice recordings and the structure ISN'T Likeness —
        # A/B tested (Aug 2026): voices train much faster and sound better at Likeness and
        # Style than at Model default, for the same reason faces do: identity lives at the
        # clean end, and the audio schedule is chained to the video one. Managed by
        # _refresh_audio_only_ui.
        self._minimax_structure_voice_note = tk.Label(
            parent, text="🎙 Voice recordings in this dataset — Likeness and Style trains "
                         "voices much faster than Model default (tested). Consider switching.",
            font=(FONT_FAMILY, 9), fg="#F59E0B", bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=700)

        # Per-category retirement — MIXED datasets only (managed by _refresh_audio_only_ui).
        # Visuals and voice need not converge together (a much smaller category can finish,
        # or start to overbake, well before the larger one),
        # so each can retire at its own epoch. "Anchor" keeps the finished category training at
        # a REAL 10% LR (it multiplies the optimizer's lr — a loss multiplier would be an Adam
        # no-op) as a drift guard, with its epoch ledger staying live as the drift alarm;
        # "stop" skips its steps outright for faster epochs.
        self._mixed_stop_label = ttk.Label(parent, text="Finish one category early:")
        self._mixed_stop_frame = ttk.Frame(parent)
        _msf = self._mixed_stop_frame
        _RETIRE_MODES = ["anchor at 10% LR (recommended)", "stop completely (faster)"]
        self.entries["MIXED_STOP_CATEGORY"] = ttk.Combobox(
            _msf, values=["voice", "photos & clips"], width=14, state="readonly")
        self.entries["MIXED_STOP_CATEGORY"].set(
            str(self.settings.get("MIXED_STOP_CATEGORY", "")) or "voice")
        self.entries["MIXED_STOP_CATEGORY"].pack(side=tk.LEFT)
        ttk.Label(_msf, text=" after epoch ").pack(side=tk.LEFT)
        self.entries["MIXED_STOP_EPOCH"] = ttk.Entry(_msf, width=5)
        self.entries["MIXED_STOP_EPOCH"].insert(
            0, str(self.settings.get("MIXED_STOP_EPOCH", "")))
        self.entries["MIXED_STOP_EPOCH"].pack(side=tk.LEFT, padx=(0, 8))
        # Under FT the hint shows live where a typed epoch will land (cycle-boundary snap).
        self.entries["MIXED_STOP_EPOCH"].bind(
            "<KeyRelease>", lambda _e: self._refresh_mixed_stop_hint())
        self.entries["MIXED_STOP_MODE"] = ttk.Combobox(_msf, values=_RETIRE_MODES, width=26,
                                                       state="readonly")
        self.entries["MIXED_STOP_MODE"].set(
            str(self.settings.get("MIXED_STOP_MODE", "")) or _RETIRE_MODES[0])
        self.entries["MIXED_STOP_MODE"].pack(side=tk.LEFT)
        self._mixed_stop_hint = tk.Label(
            parent, text="If one category is a substantially different size from the other, "
                         "it may be done (or start to overbake) well before the rest — finish "
                         "it early instead of overtraining it. Blank = both train to the end. "
                         "Anchor keeps the finished category at a true 10% learning rate — "
                         "holding its quality against drift from the still-training category, "
                         "with its epoch report staying live as the drift alarm. Stop skips "
                         "its steps entirely: faster epochs, but that category goes unwatched.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=720)
        self._MIXED_STOP_HINT_LORA = self._mixed_stop_hint.cget("text")
        # FT text is rebuilt live by _refresh_mixed_stop_hint (the cycle length rides on
        # Rotate every); this placeholder is only ever shown for a frame at build time.
        self._MIXED_STOP_HINT_FT = ""

        # The raw share, revealed only under Custom — the named options are the point.
        self._minimax_shift_label = ttk.Label(parent, text="Clean-end share:")
        self._minimax_shift_label.grid(row=24, column=0, sticky=tk.W, padx=5, pady=2)
        self._minimax_shift_frame = ttk.Frame(parent)
        self._minimax_shift_frame.grid(row=24, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.entries["MINIMAX_LOWNOISE_PCT"] = ttk.Entry(self._minimax_shift_frame, width=8)
        self.entries["MINIMAX_LOWNOISE_PCT"].insert(
            0, str(self.settings.get("MINIMAX_LOWNOISE_PCT", "60")))
        self.entries["MINIMAX_LOWNOISE_PCT"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_shift_frame, text="% of steps").pack(side=tk.LEFT, padx=(4, 0))
        # Live readout: the typed number is what you care about, but the schedule it produces is
        # worth seeing — a couple of percent swings the shift enormously at the ends.
        self._minimax_shift_match = tk.Label(self._minimax_shift_frame, text="",
                                             font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        self._minimax_shift_match.pack(side=tk.LEFT, padx=(10, 0))
        self.entries["MINIMAX_LOWNOISE_PCT"].bind(
            "<KeyRelease>", lambda _e: self._refresh_minimax_shift_match())

        # Always visible: a preset recommends a value, the user can override it without that
        # counting as a different structure.
        self._minimax_hnlr_label = ttk.Label(parent, text="Medium to High LR:")
        self._minimax_hnlr_label.grid(row=25, column=0, sticky=tk.W, padx=5, pady=(2, 8))
        self._minimax_hnlr_frame = ttk.Frame(parent)
        self._minimax_hnlr_frame.grid(row=25, column=1, columnspan=2, sticky=tk.W,
                                      padx=5, pady=(2, 8))
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"] = ttk.Entry(self._minimax_hnlr_frame, width=8)
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"].insert(
            0, str(self.settings.get("MINIMAX_HIGHNOISE_LR_PCT", "100")))
        self.entries["MINIMAX_HIGHNOISE_LR_PCT"].pack(side=tk.LEFT)
        tk.Label(self._minimax_hnlr_frame,
                 text="%  — best left at 100 unless you are experimenting.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(4, 0))
        # Says what it does and what was measured, so lowering it is a decision rather than a
        # guess: across five datasets, at both densities, 0% and 100% render cleanly at 20 steps
        # without the Turbo LoRA and 100% holds face SHAPE better every time.
        self._minimax_hnlr_hint = tk.Label(
            parent,
            text="What the noisier steps — where pose, framing and face shape are decided — do to "
                 "the learning rate. Lowering it biases the run toward surface detail at the cost "
                 "of shape — useful for a skin-texture LoRA, not for a likeness one. Across five "
                 "datasets 100 held face shape better, and nothing distorted at any setting.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT, wraplength=700)
        self._minimax_hnlr_hint.grid(row=26, column=0, columnspan=3, sticky=tk.W,
                                     padx=(12, 5), pady=(0, 8))

        self._sync_minimax_structure_from_pct()
        self._refresh_minimax_shift_match()

    def _on_minimax_structure_changed(self):
        """A named option writes the numbers behind it; Custom just reveals them."""
        vals = MINIMAX_STRUCTURE_OPTIONS.get(self.minimax_structure_var.get())
        if vals is not None:
            pct, hnlr = vals
            for key, value in (("MINIMAX_LOWNOISE_PCT", pct), ("MINIMAX_HIGHNOISE_LR_PCT", hnlr)):
                ent = self.entries.get(key)
                if ent is not None:
                    ent.delete(0, tk.END)
                    ent.insert(0, str(value))
        self._refresh_minimax_structure_ui()
        self._refresh_minimax_shift_match()
        self._refresh_audio_only_ui()      # the voice hint clears the moment Likeness is picked

    def _sync_minimax_structure_from_pct(self):
        """Pick the dropdown entry the current percentage corresponds to, else Custom.

        Derived rather than stored, which is what lets every preset written before this control
        existed keep working untouched.
        """
        try:
            pct = float(str(self.entries["MINIMAX_LOWNOISE_PCT"].get()).strip().rstrip("%"))
        except (KeyError, TypeError, ValueError, tk.TclError):
            pct = None
        name = "Custom"
        if pct is not None:
            for label, vals in MINIMAX_STRUCTURE_OPTIONS.items():
                if vals is not None and abs(vals[0] - pct) < 1e-9:
                    name = label
                    break
        self.minimax_structure_var.set(name)
        self._refresh_minimax_structure_ui()

    def _refresh_minimax_structure_ui(self):
        """Description text, and the raw share shown only under Custom."""
        name = self.minimax_structure_var.get()
        desc = getattr(self, "_minimax_structure_desc", None)
        if desc is not None and desc.winfo_exists():
            desc.config(text=MINIMAX_STRUCTURE_DESC.get(name, ""))
        custom = MINIMAX_STRUCTURE_OPTIONS.get(name) is None
        for w in (getattr(self, "_minimax_shift_label", None),
                  getattr(self, "_minimax_shift_frame", None)):
            if w is None or not w.winfo_exists():
                continue
            if custom and self._is_minimax_arch():
                w.grid()
            else:
                w.grid_remove()

    def _refresh_minimax_blocks_count(self):
        """Say how many blocks the Blocks to Train box currently means, or why it can't be read.

        A typed spec fails silently in the worst way: "3-12, 4" trains 11 blocks and looks like a
        run, and nothing downstream ever says otherwise. This turns that into a number you can
        see before you launch."""
        lbl = getattr(self, "_minimax_blocks_count", None)
        combo = self.entries.get("MINIMAX_BLOCKS")
        if lbl is None or combo is None or not lbl.winfo_exists():
            return
        spec = minimax_block_spec(combo.get())
        if spec.lower() == "all":
            lbl.config(text="all 50 blocks", fg=COLORS["text_explain"])
            return
        try:
            from fizgig.minimax.trainer import parse_block_spec
            idx = parse_block_spec(spec, MINIMAX_NUM_BLOCKS)
        except ValueError as e:
            lbl.config(text=f"✗ {e}", fg="#E74C3C")
            return
        except ImportError:
            lbl.config(text="")
            return
        lbl.config(text=f"✓ {len(idx)} of {MINIMAX_NUM_BLOCKS} blocks", fg="#27AE60")

    # The Blocks to Train hint in both of its states — module-level truth so the greying
    # handler can swap them without duplicating the strings inline.
    _MINIMAX_BLOCKS_HINT = ("Train only a subset of the 50 blocks. Type ranges and single "
                            "blocks, comma-separated, like 3-12, 22, 31-33 (blocks 0-49). "
                            "Measured answers: 20-49 for likeness — sharper, more "
                            "prompt-responsive, better sound, faster and smoother to converge "
                            "(Optimised Likeness Learning applies it to photos automatically) — "
                            "and 0-3, 6-47 for style (the Style preset sets it). Full write-up "
                            "in the README.")
    _MINIMAX_BLOCKS_HINT_LOCKED = ("Disabled by Optimised Likeness Learning above — untick it "
                                   "to hand-pick blocks. While it's on, photos train "
                                   f"{MINIMAX_LIKENESS_BLOCKS}; video follows the restriction tickbox.")

    def _sync_minimax_likeness_state(self):
        """Grey Blocks to Train while Optimised Likeness Learning owns the block choice.

        The combobox VALUE is deliberately preserved — a hand-typed spec survives a toggle
        round-trip; only the widget state and the hint change. Driven by the checkbox trace
        (fires on preset loads too) and by arch switches."""
        combo = self.entries.get("MINIMAX_BLOCKS")
        hint = getattr(self, "_minimax_blocks_hint", None)
        if combo is None or hint is None or not combo.winfo_exists():
            return
        locked = self._is_minimax_arch() and bool(
            self.entries["MINIMAX_LIKENESS_OPT"].get())
        # The video-restriction sub-tick shows only where it means something: MiniMax
        # family, Fine-tune ON, likeness ON. (LoRA-mode clips keep whole-model behaviour;
        # the builder only emits --clip_blocks under FT regardless, so this is
        # presentation — the flag gate is the guard.)
        _clip_cb = getattr(self, "_minimax_ft_clip_cb", None)
        if _clip_cb is not None and _clip_cb.winfo_exists():
            _show = locked and bool(getattr(self, "minimax_finetune_var", None)
                                    and self.minimax_finetune_var.get())
            self._set_widget_visible(_clip_cb, _show)
        if locked:
            combo.config(state="disabled")
            hint.config(text=self._MINIMAX_BLOCKS_HINT_LOCKED)
            lbl = getattr(self, "_minimax_blocks_count", None)
            if lbl is not None and lbl.winfo_exists():
                lbl.config(text=f"photos: {MINIMAX_LIKENESS_BLOCKS} · clips: see video restriction",
                           fg=COLORS["text_explain"])
        else:
            combo.config(state="")               # editable, the widget's natural state
            hint.config(text=self._MINIMAX_BLOCKS_HINT)
            self._refresh_minimax_blocks_count()

    def _is_krea2_arch(self) -> bool:
        return ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_krea2", False)

    def _is_minimax_arch(self) -> bool:
        return ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_minimax", False)

    def _set_widget_visible(self, w, show: bool):
        """Show/hide a single widget, working for both grid- and pack-managed widgets.
        grid widgets use grid_remove()/grid() (position preserved); pack widgets stash their
        pack_info on hide and restore it on show (with the 'in'→'in_' kwarg fix)."""
        if w is None:
            return
        try:
            if show:
                saved = getattr(w, "_fizgig_pack_info", None)
                if saved is not None:
                    w.pack(**saved)
                    w._fizgig_pack_info = None
                elif w.winfo_manager() == "":   # grid_remove'd → restore remembered slot
                    w.grid()
            else:
                mgr = w.winfo_manager()
                if mgr == "pack":
                    info = {k: v for k, v in w.pack_info().items()}
                    if "in" in info:
                        info["in_"] = info.pop("in")
                    w._fizgig_pack_info = info
                    w.pack_forget()
                elif mgr == "grid":
                    w.grid_remove()
        except Exception:
            pass

    def _set_training_section_visible(self, key: str, before_key: str, visible: bool):
        """Show/hide a whole collapsible section, preserving the canonical pack order.
        When showing, pack it before `before_key` (which must be a currently-packed section)
        so it lands back in the right place rather than at the bottom of the tab."""
        sec = self.collapsible_sections.get(key)
        if sec is None:
            return
        try:
            if visible:
                before = self.collapsible_sections.get(before_key)
                if before is not None and before.winfo_manager() == "pack":
                    sec.pack(fill=tk.X, padx=36, pady=(0, 16), before=before)
                else:
                    sec.pack(fill=tk.X, padx=36, pady=(0, 16))
            else:
                sec.pack_forget()
        except Exception:
            pass

    # Recommended base-model fine-tune setup. Applied when the checkbox is ticked so the
    # whole recipe comes as one decision instead of six. Values come from the measured runs
    # on this branch; the LR especially — LoRA rates (1e-4+) destroy a base model.
    KREA2_FT_DEFAULTS = {
        "LEARNING_RATE": "1e-5",
        "MAX_TRAIN_EPOCHS": "40",         # 10 full 4-window cycles — an overnight run you can
                                          # scrub through; nobody has tuned this recipe on a
                                          # diffusion DiT, so compare checkpoints to find where
                                          # it peaks rather than trusting the number
        "SAVE_EVERY_N_EPOCHS": "4",       # one per full cycle: every component has had the same
                                          # number of passes, so checkpoints are comparable
                                          # like-for-like. ~26 GB each -> 10 files / ~260 GB
                                          # over a 40-epoch run
        "GRADIENT_ACCUMULATION": "1",     # fused backward consumes grads as they land
        "MAX_GRAD_NORM": "0",             # global clipping is impossible under fused backward
        "NETWORK_TYPE": "LoRA (standard)",  # FT trains the BASE — a LoKR adapter would sit
                                            # inert burning VRAM, so the recipe resets it
    }

    def _on_krea2_ft_toggle(self):
        """User ticked/unticked base-model fine-tuning. Only push the recipe on the way ON,
        so re-showing the tab never stomps values the user has since tuned."""
        self._apply_krea2_ft_visibility()
        # The regularisation block is fine-tune-only, so the TOML changes with this toggle.
        self.auto_save_dataset_config_silent()
        if bool(self.krea2_finetune_var.get()):
            self._apply_krea2_ft_defaults()

    def _apply_krea2_ft_defaults(self):
        """Set the whole fine-tune recipe in one go, and say what changed."""
        changed = []
        # Adaptive LR goes off FIRST: the trainer disables it anyway (rotation boundaries read
        # as instability), and while it's on the Learning Rate box is greyed out — writing the
        # recipe's 1e-5 into a disabled Entry silently does nothing, which left fine-tune runs
        # starting at a LoRA-grade LR.
        if getattr(self, "adaptive_lr_var", None) is not None and self.adaptive_lr_var.get():
            self.adaptive_lr_var.set(False)
            try:
                self._on_adaptive_lr_toggle()
            except Exception:
                pass
            changed.append("Adaptive LR: on -> off (incompatible with rotation)")
        for key, val in self.KREA2_FT_DEFAULTS.items():
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                before = entry.get()
                if str(before).strip() == val:
                    continue
                # Belt and braces: a disabled Entry rejects delete/insert, so re-enable it for
                # the write and put its state back.
                _was = str(entry.cget("state"))
                if _was == "disabled":
                    entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, val)
                if _was == "disabled":
                    entry.config(state=_was)
                changed.append(f"{key.replace('_', ' ').title()}: {before} -> {val}")
            except Exception:
                pass
        if getattr(self, "lr_scheduler_var", None) is not None and self.lr_scheduler_var.get() != "constant":
            was = self.lr_scheduler_var.get()
            self.lr_scheduler_var.set("constant")
            changed.append(f"LR Scheduler: {was} -> constant")
        if changed:
            self.update_console("[fine-tune] applied the recommended base-model setup:\n  "
                                + "\n  ".join(changed) + "\n")

    def _browse_krea2_reg_dir(self):
        """Pick the optional regularisation image folder (Krea 2 fine-tune)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Regularisation images (optional)",
                                    initialdir=self.krea2_reg_dir_var.get() or None)
        if d:
            self.krea2_reg_dir_var.set(d)
            self.auto_save_dataset_config_silent()   # the TOML carries the reg block

    def _browse_minimax_reg_dir(self):
        """Pick the optional regularisation image folder (MiniMax H3 fine-tune)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Regularisation images (optional)",
                                    initialdir=self.minimax_reg_dir_var.get() or None)
        if d:
            self.minimax_reg_dir_var.set(d)
            self.auto_save_dataset_config_silent()   # the TOML carries the reg block

    def _apply_krea2_ft_visibility(self):
        """Show the fine-tune knobs only when base-model fine-tuning is on, and the
        blocks-per-window picker only in block mode (component windows are fixed)."""
        if not hasattr(self, "_krea2_ft_frame"):
            return
        on = bool(self.krea2_finetune_var.get())
        for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_fast_ft_cb,
                  self._krea2_reg_frame, self._krea2_ft_hint):
            self._set_widget_visible(w, on)
        # Auto-recaption is hidden under a fine-tune: its between-epoch VLM load moves the
        # whole DiT off the card through a blocks_to_swap-aware restore that knows nothing
        # about the FT rotation streamer — on the 16 GB streamed tier the restore would
        # hoist every streamed block back onto the card behind the offloader's bookkeeping.
        # The other three watch toggles stay: their multipliers ride the same loss-scaling
        # the FT regularisation path uses, and detection compares each image against the
        # cohort at the same epoch, so rotation's boundary shifts cancel. The trainer
        # disarms a ticked-but-hidden box too, so this is presentation, not the guard.
        if hasattr(self, "_krea2_autorecap_cb"):
            self._set_widget_visible(self._krea2_autorecap_cb, not on)
        # Network Type (LoRA/LoKR) is meaningless under a base-model fine-tune — the adapter
        # is inert. Hide the row while FT is on; restore the normal swap when it goes off.
        if hasattr(self, "_network_type_rowf"):
            self._set_widget_visible(self.labels["NETWORK_TYPE"], not on)
            self._set_widget_visible(self._network_type_rowf, not on)
            if on:
                self.hide_row("LOKR_FACTOR")
                self.show_row("NETWORK_DIM")
                self.show_row("NETWORK_ALPHA")
            else:
                self._on_network_type_changed()
        block_mode = str(self.krea2_ft_mode_var.get()) == "block"   # Auto picks its own
        if on and block_mode:
            self._krea2_ft_blocks_lbl.pack(side=tk.LEFT, padx=(14, 4))
            self._krea2_ft_blocks_cb.pack(side=tk.LEFT)
        else:
            self._krea2_ft_blocks_lbl.pack_forget()
            self._krea2_ft_blocks_cb.pack_forget()

    # --- MiniMax H3 rotation fine-tune (mirrors the Krea 2 card) --------------------------
    MINIMAX_FT_DEFAULTS = {
        "LEARNING_RATE": "1e-5",          # a starting point, NOT a calibrated H3 recipe —
                                          # nobody has tuned FT rates on this model yet
        "MAX_TRAIN_EPOCHS": "100",        # a realistic fine-tune length (Peter, 29 Aug:
                                          # 26 was "far too small"; his field A/Bs ran 64
                                          # and kept improving). Clean at BOTH full-speed
                                          # plans — 25 cycles at 4 windows, 20 at 5 — and
                                          # the trainer now snaps any total UP to a cycle
                                          # boundary at launch anyway (snap_ft_epochs),
                                          # so odd window counts still end evenly trained.
        # SAVE_EVERY_N_EPOCHS is NOT a static recipe value: the cycle length depends on the
        # window mode/size, so _refresh_minimax_ft_save_box keeps the box in step live.
        "GRADIENT_ACCUMULATION": "1",     # fused backward consumes grads as they land
        "MAX_GRAD_NORM": "0",             # global clipping is impossible under fused backward
        "NETWORK_TYPE": "LoRA (standard)",  # FT trains the BASE — reset the adapter selector
    }

    def _on_minimax_ft_toggle(self):
        """Recipe pushed on the way ON only, so re-showing the tab never stomps tuned values.

        The likeness tickbox needs NO bridging here: --photo_blocks (and --audio_blocks)
        travel under FT and the TRAINER resolves them — the cycle tightens to the union of
        what the dataset trains, and each modality is confined to its own blocks per batch.
        The Blocks field stays purely manual."""
        self._apply_minimax_ft_visibility()
        # The video-restriction sub-tick lives with likeness but only under FT — re-sync
        # so toggling FT shows/hides it without touching the likeness box itself.
        self._sync_minimax_likeness_state()
        if bool(self.minimax_finetune_var.get()):
            self._apply_minimax_ft_defaults()
            self._refresh_minimax_ft_save_box()

    def _minimax_ft_cycle_estimate(self):
        """Epochs per full rotation cycle — the 32 GB BASELINE of 4 component windows
        (qkv / out / fc1 / fc2) x rotate-every. Since the small-card tiers landed this is
        an estimate, not exact: the trainer's window planner depth-splits fat windows on
        24 GB (5 windows) and streams on 16 GB (more), resolved from free VRAM at LAUNCH —
        unknowable here. The trainer's own cycle snap stays authoritative and logs when
        it corrects the Save-every box."""
        try:
            _every = max(1, int(str(self.minimax_ft_every_var.get()).strip() or 1))
        except ValueError:
            _every = 1
        return 4 * _every

    def _refresh_minimax_ft_save_box(self):
        """Keep Save-every in step with the cycle the FT controls imply.

        Ownership decides what may be rewritten: a value the GUI itself wrote (tracked in
        _minimax_ft_save_autoset) is only ever a suggestion and always follows the cycle —
        without this, rotate-every 1 -> 2 -> 1 stranded the box at 8, because 8 is a
        multiple of 4 and looked like a user choice (field). A USER-typed value is kept
        when it's 0 (final-only) or a non-zero multiple of the cycle (a deliberate sparser
        cadence); anything else is rewritten to the suggestion: EVERY SECOND CYCLE (8 on
        the 4-window baseline, 10 on a 5-window plan) — saves are ~21 GB each and previews
        ride them, so once-per-cycle doubled the disk and preview cost for no gain (Peter,
        29 Aug: ~10 epochs is the right feel). Trainer-side snap stays authoritative at
        launch."""
        # The stop-epoch hint quotes the same cycle length — keep the two in step (cheap,
        # and this refresh fires on every cycle-affecting control).
        try:
            self._refresh_mixed_stop_hint()
        except Exception:
            pass
        if not bool(getattr(self, "minimax_finetune_var", None)
                    and self.minimax_finetune_var.get()):
            return
        entry = self.entries.get("SAVE_EVERY_N_EPOCHS")
        if entry is None:
            return
        cyc = self._minimax_ft_cycle_estimate()
        try:
            cur = int(str(entry.get()).strip() or 0)
        except ValueError:
            cur = -1
        if cur == 2 * cyc:
            self._minimax_ft_save_autoset = 2 * cyc  # already right — claim it as ours
            return
        if cur != getattr(self, "_minimax_ft_save_autoset", None):
            if cur == 0 or (cur > 0 and cur % cyc == 0):
                return                              # the user's own deliberate cadence
        # A user-typed non-multiple snaps UP to the next cycle multiple (10 on a 4-cycle
        # -> 12): the typed number expressed how SPARSE they want 20 GB saves — and,
        # since previews follow saves, previews — so one-per-cycle would be 2.5x what
        # they asked for. A GUI-owned value just tracks the cycle itself.
        _target = (((cur + cyc - 1) // cyc) * cyc
                   if cur > 0 and cur != getattr(self, "_minimax_ft_save_autoset", None)
                   else 2 * cyc)
        try:
            _was = str(entry.cget("state"))
            if _was == "disabled":
                entry.config(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, str(_target))
            if _was == "disabled":
                entry.config(state=_was)
            self._minimax_ft_save_autoset = _target
        except Exception:
            pass

    def _apply_minimax_ft_defaults(self):
        """Same shape as _apply_krea2_ft_defaults — one recipe write, with a console report."""
        changed = []
        if getattr(self, "adaptive_lr_var", None) is not None and self.adaptive_lr_var.get():
            self.adaptive_lr_var.set(False)
            try:
                self._on_adaptive_lr_toggle()
            except Exception:
                pass
            changed.append("Adaptive LR: on -> off (incompatible with rotation)")
        for key, val in self.MINIMAX_FT_DEFAULTS.items():
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                before = entry.get()
                if str(before).strip() == val:
                    continue
                _was = str(entry.cget("state"))
                if _was == "disabled":
                    entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, val)
                if _was == "disabled":
                    entry.config(state=_was)
                changed.append(f"{key.replace('_', ' ').title()}: {before} -> {val}")
            except Exception:
                pass
        if changed:
            self.update_console("[fine-tune] applied the recommended base-model setup:\n  "
                                + "\n  ".join(changed) + "\n")

    def _launch_diff_to_lora(self):
        """Open the Checkpoint to LoRA tool (its own window) from inside the app.

        One click matters most on a RunPod pod, where the GUI in a browser tab is all the
        user has — hunting for run_diff_to_lora.bat/.sh in a noVNC desktop is exactly how
        a field user ended up invoking venv/bin/python by hand (29 Aug). Windows prefers
        pythonw so no console flashes; everywhere else the venv python running this GUI
        launches it directly."""
        import subprocess
        exe = sys.executable
        if os.name == "nt":
            _w = os.path.join(_FIZGIG_DIR, "venv", "Scripts", "pythonw.exe")
            if os.path.exists(_w):
                exe = _w
        try:
            subprocess.Popen([exe, os.path.join(_FIZGIG_DIR, "diff_to_lora_gui.py")],
                             cwd=_FIZGIG_DIR)
        except Exception as e:
            messagebox.showerror("Checkpoint to LoRA", f"Could not launch the tool: {e}")


    def _apply_minimax_ft_visibility(self):
        """FT sub-controls only while the checkbox is on. Everything that is ADAPTER machinery
        hides under FT rather than sitting there silently ignored — Network Type, Optimised
        Likeness Learning, and Blocks to Train (the FT card's own Blocks field is the
        fine-tune's block restriction)."""
        if not hasattr(self, "_minimax_ft_frame"):
            return
        on = bool(self.minimax_finetune_var.get())
        for w in (self._minimax_ft_frame, self._minimax_ft_fused_cb,
                  self._minimax_reg_frame, self._minimax_ft_hint):
            self._set_widget_visible(w, on)
        # The likeness tickbox STAYS — same meaning, different mechanism: under FT it drives
        # the Blocks field (whole fine-tune on the identity blocks) instead of masking photo
        # steps. Its hint swaps to say so. Blocks to Train is adapter-only and hides.
        if hasattr(self, "_minimax_likeness_hint"):
            self._minimax_likeness_hint.config(
                text=self._MINIMAX_LIKENESS_HINT_FT if on else self._MINIMAX_LIKENESS_HINT_LORA)
        for w in (getattr(self, "_minimax_blocks_label", None),
                  getattr(self, "_minimax_blocks_frame", None),
                  getattr(self, "_minimax_blocks_hint", None),
                  # Medium to High LR is a LoRA-mode knob (it rewrites the optimizer's
                  # param-group LR at boundary steps — machinery FT doesn't have). Hidden
                  # under FT; the builder also suppresses the flag.
                  getattr(self, "_minimax_hnlr_label", None),
                  getattr(self, "_minimax_hnlr_frame", None),
                  getattr(self, "_minimax_hnlr_hint", None)):
            if w is not None:
                self._set_widget_visible(w, not on)
        if hasattr(self, "_network_type_rowf"):
            self._set_widget_visible(self.labels["NETWORK_TYPE"], not on)
            self._set_widget_visible(self._network_type_rowf, not on)
            if on:
                self.hide_row("LOKR_FACTOR")
                self.show_row("NETWORK_DIM")
                self.show_row("NETWORK_ALPHA")
            else:
                self._on_network_type_changed()
        # 'Finish one category early' STAYS under FT (retirement works there now, stop-only
        # at cycle boundaries) — its mode picker hides and its hint swaps.
        self._refresh_mixed_stop_hint()
        # A restored session can come up with FT already ON and a stale non-multiple in
        # the save box (field: 10 survived an app restart and the trainer silently snapped
        # it) — visibility runs on every restore/arch-switch, so re-snap here too. No-op
        # when FT is off (the refresh early-returns).
        self._refresh_minimax_ft_save_box()

    def _refresh_mixed_stop_hint(self):
        """Swap the 'Finish one category early' hint and hide the anchor/stop picker under
        FT — retirement there is stop-only and lands on rotation-cycle boundaries. The FT
        text is rebuilt live: the cycle length rides on Rotate every, and a typed epoch
        gets its snap target spelled out (the trainer's snap stays authoritative)."""
        if not hasattr(self, "_mixed_stop_hint"):
            return
        on = bool(getattr(self, "minimax_finetune_var", None)
                  and self.minimax_finetune_var.get())
        _mode = self.entries.get("MIXED_STOP_MODE")
        if _mode is not None:
            self._set_widget_visible(_mode, not on)
        if not on:
            self._mixed_stop_hint.config(text=self._MIXED_STOP_HINT_LORA)
            return
        cyc = self._minimax_ft_cycle_estimate()
        try:
            _n = int(str(self.entries["MIXED_STOP_EPOCH"].get()).strip() or 0)
        except (ValueError, KeyError, tk.TclError):
            _n = 0
        _snap = ((_n + cyc - 1) // cyc) * cyc if _n > 0 else 0
        _ex = (f" Your epoch {_n} lands at {_snap}."
               if _n > 0 and _snap != _n else "")
        self._MIXED_STOP_HINT_FT = (
            "Under fine-tune the finished category STOPS outright (no anchor mode), and "
            "the stop lands on a rotation-cycle boundary — epochs snap UP to the next "
            f"multiple of the {cyc}-epoch cycle, so every window sees the same data mix "
            f"for equal passes before it changes.{_ex} Great for a polish tail: stop "
            "photos & clips and let the voice keep refining its own blocks, or the "
            "reverse.")
        self._mixed_stop_hint.config(text=self._MIXED_STOP_HINT_FT)

    def _refresh_optimizer_choices(self, is_krea2: bool):
        """Point the Optimizer Type dropdown at the selected family's catalog."""
        combo = self.entries.get("OPTIMIZER_TYPE")
        if combo is None:
            return
        choices = self.krea2_optimizer_types if is_krea2 else self.optimizer_types
        combo["values"] = choices
        if combo.get() not in choices:
            combo.set("adamw8bit" if "adamw8bit" in choices else choices[0])

    def _apply_training_arch_visibility(self, is_krea2: bool):
        """Hide Training-tab controls not yet wired into the Krea 2 native trainer; re-show for Klein.

        Deferred-for-Krea-2 feature groups (re-enable by removing from these lists as they land):
          • Model Area to Train (dropdown + desc + Custom panel) — no Krea 2 block map yet
          • Network Dropout                                     — not implemented for krea2
            (the rest of the Optimizer section is wired: Type, Args, Gradient Accumulation,
             Max Grad Norm. The Type dropdown re-populates per family — see
             _refresh_optimizer_choices — because the two resolve names differently.)
          • LR Decay steps                                      — Klein-only warmup_stable_decay
          • Timestep & Noise section                            — krea2 uses a fixed shift schedule
          • FP8 Scaled (in Memory & FP8)                        — krea2's fp8 path is always scaled
          • FP8 Text Encoder (in Memory & FP8)                  — krea2 caches the TE in bf16
          • Gradient Checkpointing (in Memory & FP8)            — krea2_train hardcodes it ON
          • FP8 Base (in Memory & FP8)                          — see below; an OOM trap on krea2
        Kept (model-agnostic / wired): the full live "Override next sample" status-bar panel
        including its Reference image — krea2 reads the override sentinel for previews
        (prompt/seed/resolution) and routes the reference through the Qwen3-VL vision path.

        FP8 Base is hidden for Krea 2 (29 Jul) because unticking it was a guaranteed OOM, not a
        useful option. It sends --no_fp8, i.e. a bf16 base: 25.8 GB of weights alone, ~28 GB in
        total, which no consumer card can hold. The auto swap planner never saw it — the plan is
        identical ticked or unticked — so the run got a swap count sized for fp8/INT8/NF4 and
        then loaded something twice that size. The command builder's elif chain also meant
        unticking it silently dropped the INT8 flag the planner had just chosen. Krea 2's real
        base-precision choices all live on the 4-bit control (Auto / On / Off -> NF4 / INT8 /
        fp8), which the planner does see.
        """
        # Guard: this may run via update_ui_for_architecture before the Training tab is built.
        if not hasattr(self, "_adaptive_cb"):
            return
        # MiniMax H3 is a THIRD, even-more-minimal native family. It shares Krea 2's "hide the
        # Klein-only controls" set, and ALSO hides Krea 2's own extras (base-precision dropdown,
        # per-image loss watch, torch.compile, LoKR network type) — it's LoRA-over-NF4 only, no
        # samples. `native` = "not Klein" (hide Klein-only); `is_krea2` still gates the Krea-2-only
        # widgets, so MiniMax (is_krea2 False) hides them too.
        is_minimax = self._is_minimax_arch()
        native = is_krea2 or is_minimax

        # The single-frame preview caveat belongs to MiniMax only — show it under the Base Model
        # selector when that family is picked, hide it otherwise.
        _note = getattr(self, "_minimax_sample_note", None)
        if _note is not None:
            if is_minimax:
                if not _note.winfo_manager():
                    _note.pack(anchor=tk.W, pady=(10, 0))
            elif _note.winfo_manager():
                _note.pack_forget()
        # Training Base rides in the same card, above the note (before= keeps the order when
        # the note is already on screen).
        _brow = getattr(self, "_minimax_base_frame", None)
        if _brow is not None:
            _bhint = self._minimax_base_hint
            if is_minimax:
                if not _brow.winfo_manager():
                    _kw = {"before": _note} if (_note is not None
                                                and _note.winfo_manager()) else {}
                    _brow.pack(anchor=tk.W, pady=(10, 0), **_kw)
                    _bhint.pack(anchor=tk.W, pady=(2, 0), **_kw)
            elif _brow.winfo_manager():
                _brow.pack_forget()
                _bhint.pack_forget()

        # The live-override REFERENCE image is a Klein edit-model feature. Neither native family
        # is an edit model, and their trainers ignore the field — so hide the picker rather than
        # leave a control that silently does nothing.
        for _n in ("_override_ref_browse_btn", "_override_ref_label", "_override_ref_clear_btn"):
            _w = getattr(self, _n, None)
            if _w is None:
                continue
            if native and _w.winfo_manager():
                _w.pack_forget()
            elif not native and not _w.winfo_manager():
                _w.pack(side=tk.LEFT, **({"padx": (6, 2)} if _n.endswith("label") else {}))
        if native:
            try:
                self.sample_override_ref_var.set("")
            except Exception:
                pass
        # Per-widget groups across the Training Parameters + Memory & FP8 sections.
        widgets = [
            self._modelarea_label, self._modelarea_combo, self._modelarea_desc_label,
            # The whole Weight Optimization row: label, both checkboxes, and the hint under
            # them. Hiding only the controls left an orphaned label and a paragraph of text
            # explaining something no longer on screen.
            self._fp8_row_label, self.fp8_check, self.scaled_check, self._fp8_hint,
            self.fp8_text_encoder_label, self.fp8_text_encoder_check,
            self._grad_checkpoint_label, self.grad_checkpoint_check, self._grad_checkpoint_hint,
            # LR Decay steps: Klein-only (warmup_stable_decay). LR Scheduler + Warmup ARE wired
            # for krea2 (--lr_scheduler / --lr_warmup_steps) so they stay visible.
            self._lr_decay_label, self.entries.get("LR_DECAY_STEPS"),
            # Optimizer Type / Args are wired for BOTH now (krea2 -> --optimizer_type /
            # --optimizer_args); only network dropout has no krea2 equivalent.
            self.labels.get("NETWORK_DROPOUT"), self.entries.get("NETWORK_DROPOUT"),
        ]
        for w in widgets:
            self._set_widget_visible(w, not native)

        # Base precision is the inverse: Krea 2 ONLY. Its options (Auto / INT8 / NF4 / fp8) and
        # the memory strategy behind them are entirely krea2_train's; Klein's trainer has no
        # INT8 path and no auto strategy, so offering the dropdown there would list options
        # Klein cannot run. (Klein's --quant_4bit still exists on its CLI.)
        for w in (self._quant_4bit_label, self.quant_4bit_check, self._quant_4bit_hint):
            self._set_widget_visible(w, is_krea2)

        # The two families resolve optimizer names differently, so the dropdown's contents follow
        # the selector. A name valid for one may not exist in the other (Klein takes module paths;
        # krea2 takes catalog names), so fall back to the shared default rather than carrying a
        # value across that the trainer would then have to reject.
        # MiniMax resolves optimizer names the same catalog-based way Krea 2 does (its trainer
        # takes catalog names too), so it shares Krea 2's dropdown contents.
        self._refresh_optimizer_choices(native)

        # Krea 2-ONLY controls (inverse of the above): the per-image loss watch toggles are only
        # wired into krea2_train for now — hide them under Klein.
        for w in (self._krea2_losswatch_frame, self._krea2_perimglr_cb,
                  self._krea2_autorecap_cb, self._krea2_warmuplook_cb,
                  self._krea2_losswatch_hint, self._krea2_ft_cb,
                  # torch.compile is wired into krea2_train only.
                  self._compile_blocks_label, self.compile_blocks_check, self._compile_blocks_hint):
            self._set_widget_visible(w, is_krea2)
        # The FT sub-controls are gated by the checkbox as well as by the family. Gate on
        # the family, NOT native: each family's FT visibility logic also swaps the Network
        # Type rows, which the other family's logic must never touch. Away from a family,
        # its FT sub-widgets hide outright (the family loop above only covers the checkbox).
        if is_krea2:
            self._apply_krea2_ft_visibility()
        elif hasattr(self, "_krea2_ft_frame"):
            for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_fast_ft_cb,
                      self._krea2_reg_frame, self._krea2_ft_hint):
                self._set_widget_visible(w, False)
        if is_minimax:
            self._apply_minimax_ft_visibility()
        elif hasattr(self, "_minimax_ft_frame"):
            for w in (self._minimax_ft_frame, self._minimax_ft_fused_cb,
                      self._minimax_reg_frame, self._minimax_ft_hint):
                self._set_widget_visible(w, False)
        # Network Type (LoRA/LoKR) is wired for BOTH native families (krea2_train and
        # minimax_train take --network_type/--lokr_factor); Klein trains standard only.
        # The row frame carries the combo + hint together. The speed note is Krea 2-only:
        # on MiniMax the governor holds both types at the same movement rate, so the ~20%
        # LoRA speed edge measured on Krea 2 doesn't translate.
        for w in (self.labels["NETWORK_TYPE"], self._network_type_rowf):
            self._set_widget_visible(w, native)
        self._network_type_hint.config(
            text="LoRA recommended for MiniMax" if is_minimax
            else "LoKR: higher quality · LoRA: ~20% faster training")

        # Detail Focus is the inverse: MiniMax ONLY. Klein and Krea 2 already derive their shift
        # from the sample's token count, so there is nothing to dial there.
        for w in (self._minimax_structure_label, self._minimax_structure_combo,
                  self._minimax_structure_desc,
                  self._minimax_hnlr_label, self._minimax_hnlr_frame, self._minimax_hnlr_hint,
                  self._minimax_blocks_label, self._minimax_blocks_frame, self._minimax_blocks_hint,
                  self._minimax_likeness_cb, self._minimax_likeness_hint,
                  self._minimax_distill_frame, self._minimax_distill_hint,
                  self._minimax_quant_label, self._minimax_quant_frame,
                  self._minimax_quant_hint,
                  self._minimax_smooth_label, self._minimax_smooth_frame,
                  self._minimax_smooth_hint,
                  self._minimax_ramp_label, self._minimax_ramp_frame, self._minimax_ramp_hint,
                  self._minimax_capdrop_label, self._minimax_capdrop_frame,
                  self._minimax_capdrop_hint,
                  self._minimax_mc_frame,
                  self._minimax_ft_cb,
                  ):
            self._set_widget_visible(w, is_minimax)
        # The clean-end box answers to BOTH the family and the dropdown: visible only for MiniMax,
        # and only when the structure is Custom.
        self._refresh_minimax_structure_ui()
        # Blocks to Train greys while Optimised Likeness Learning owns it — arch-dependent, so
        # re-sync on every family switch (a Klein session must not leave it locked).
        self._sync_minimax_likeness_state()
        # The Multi Concept sub-rows are owned by its own toggle handler (they are hidden even
        # under MiniMax until the box is ticked), so route them through it rather than the loop.
        if is_minimax:
            self._on_minimax_multiconcept_toggle()
            self._sync_distill_weight_state()
        else:
            for w in (self._minimax_mc_dir_frame, self._minimax_mc_hint,
                      self._minimax_mc_nodistill_hint):
                self._set_widget_visible(w, False)
        # Retired MiniMax controls — never shown under any family. AdaLN can't deploy on the
        # pruned builds; depth-split LR was superseded by the limiter (9 Aug). The per-step clip
        # and LR warmup joined them 10 Aug: the Adapter-relative LR ramp addresses the same
        # epoch-1 overshoot at its root by holding the step/size ratio steady, so a movement cap
        # and a fixed warmup count are both guesses at a problem that no longer needs them. The
        # command builder locks every one of these regardless of saved settings.
        for w in (self._minimax_adaln_cb, self._minimax_adaln_hint,
                  self._minimax_slow_label, self._minimax_slow_frame, self._minimax_slow_hint,
                  self._minimax_limiter_label, self._minimax_limiter_frame,
                  self._minimax_limiter_hint):
            self._set_widget_visible(w, False)
        if is_minimax:
            for _k, _off in (("MINIMAX_BLOCK_LIMIT", "Off"), ("MINIMAX_LR_WARMUP", "Off")):
                if str(self.entries[_k].get()) != _off:
                    self.entries[_k].set(_off)     # a preset or saved config must not revive it
        # Adaptive LR is hidden under MiniMax: ticking it silently disabled the governor +
        # warmup (they defer to it). The var is forced off so the greyed-LR-box state and the
        # curated launch dict can't carry a stale True into a run.
        for w in (self._adaptive_cb, self._adaptive_frame, self._adaptive_desc_label):
            self._set_widget_visible(w, not is_minimax)
        if is_minimax:
            if self.adaptive_lr_var.get():
                self.adaptive_lr_var.set(False)
                try:
                    self._on_adaptive_lr_toggle()      # un-grey the Learning Rate box
                except Exception:
                    pass
            # Optimizer locked to adamw (the likeness finding) — hide the dropdown row.
            self.hide_row("OPTIMIZER_TYPE")
        else:
            self.show_row("OPTIMIZER_TYPE")

        # Context LoRA is wired for Klein and Krea 2 but NOT MiniMax — hide the whole row there
        # rather than show a picker the trainer silently ignores.
        for w in (self._contextlora_label, self._contextlora_frame,
                  self._contextlora_desc_label, self._contextlora_warn_label):
            self._set_widget_visible(w, not is_minimax)
        if native:
            # Restore the rank/alpha <-> factor row swap for the current selection.
            self._on_network_type_changed()
        else:
            for w in (self._krea2_ft_frame, self._krea2_ft_fused_cb, self._krea2_ft_hint):
                self._set_widget_visible(w, False)
            # Klein always shows rank/alpha and never the factor, whatever the combo holds.
            self.show_row("NETWORK_DIM")
            self.show_row("NETWORK_ALPHA")
            self.hide_row("LOKR_FACTOR")

        # Custom block picker: always hidden under the native families (no Krea 2 / MiniMax block
        # map); under Klein, let the Model-Area dropdown decide (only shown when preset = "Custom").
        try:
            if native:
                self._training_custom_frame.grid_remove()
            else:
                self._on_training_preset_changed()
        except Exception:
            pass

        # Whole collapsible sections. Re-show in canonical order (Timestep before Optimizer,
        # Optimizer before Other Options) — show Optimizer first so Timestep's anchor is packed.
        # Optimizer section now stays visible for Krea 2 (Gradient Accumulation + Max Grad Norm
        # are wired); its unwired fields are hidden individually above.
        self._set_training_section_visible("optimizer", "scheduler", True)
        self._set_training_section_visible("timestep", "optimizer", not native)

    # ── Problem Images window (per-image loss watch) ────────────────────

    def _loss_log_dir(self) -> str:
        """<output_dir>/loss_log from the LIVE Output Directory field (settings only refresh at
        start_training, so a user who edits the field pre-launch would otherwise see stale data)."""
        out = ""
        try:
            if hasattr(self, "entries") and "LORA_OUTPUT_DIR" in self.entries:
                out = self.entries["LORA_OUTPUT_DIR"].get().strip()
        except Exception:
            out = ""
        out = out or (self.settings.get("LORA_OUTPUT_DIR", "") or "")
        return os.path.join(out, "loss_log") if out else ""

    def _problem_images_json_path(self) -> str:
        d = self._loss_log_dir()
        return os.path.join(d, "problem_images.json") if d else ""

    def _find_dataset_image(self, key: str):
        """Resolve a loss-watch item key (image basename, no extension) to a file in the training
        image folder. Returns a path or None."""
        folder = self.image_folder_var.get().strip() if hasattr(self, "image_folder_var") else ""
        if not folder or not os.path.isdir(folder):
            return None
        base = os.path.basename(key)
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            p = os.path.join(folder, base + ext)
            if os.path.exists(p):
                return p
        return None

    def _load_thumbs_async(self, jobs, cache):
        """Decode row thumbnails OFF the Tk main thread. Decoding a whole dataset of
        full-resolution PNGs inline froze the Problem Images / Look Filter windows for
        seconds; rows now show a placeholder and fill in as each decode completes.
        jobs: [(image_path, placeholder_label)]; cache: path -> PhotoImage (holds refs)."""
        def work():
            done = []
            for p, lbl in jobs:
                try:
                    # with-block: PIL otherwise keeps the file handle open until GC, and an
                    # open handle makes Windows fail a later move of that image (Look Filter's
                    # "Move Marked" raced this and left a copy behind). thumbnail() forces a
                    # full decode, so the raster stays usable after close.
                    with Image.open(p) as im:
                        im.thumbnail((96, 96), Image.LANCZOS)
                        done.append((p, im, lbl))
                except Exception:
                    done.append((p, None, lbl))
            def apply():
                for p, im, lbl in done:
                    ph = cache.get(p)
                    if ph is None and im is not None:
                        try:
                            ph = ImageTk.PhotoImage(im)
                            cache[p] = ph
                        except Exception:
                            ph = None
                    try:
                        if not lbl.winfo_exists():
                            continue
                        if ph is not None:
                            lbl.config(image=ph, text="")
                        else:
                            lbl.config(text="no\npreview")
                    except Exception:
                        pass
            try:
                self.master.after(0, apply)
            except Exception:
                pass   # GUI torn down mid-decode
        threading.Thread(target=work, daemon=True).start()

    def _open_problem_images_window(self):
        """Live viewer for the per-image loss watch — thumbnails + verdicts, auto-refreshing
        during training from <output_dir>/loss_log/problem_images.json."""
        win = getattr(self, "_problem_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            self._refresh_problem_images(force=True)
            return
        win = tk.Toplevel(self.master)
        win.title("Problem Images — per-image loss watch")
        win.geometry("1010x640")
        win.configure(bg=COLORS["bg_deep"])
        self._problem_win = win
        self._problem_mtime = None
        self._problem_thumbs = {}  # path -> PhotoImage, cached across refreshes (and kept alive)
        self._problem_row_ui = {}  # key -> persistent row widgets (in-place refresh; new window = fresh)
        self._problem_last_order = []
        self._problem_img_paths = getattr(self, "_problem_img_paths", {})  # key -> resolved image path

        head = tk.Frame(win, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=14, pady=(12, 6))
        tk.Label(head, text="Problem Images", font=(FONT_FAMILY, 15, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh", command=lambda: self._refresh_problem_images(force=True)).pack(side=tk.RIGHT)
        self._problem_status = tk.Label(win, text="", font=(FONT_FAMILY, 9),
                                        fg=COLORS["text_muted"], bg=COLORS["bg_deep"],
                                        justify=tk.LEFT, anchor="w")
        self._problem_status.pack(fill=tk.X, padx=14)

        # Wrap the status text to the live window width — without a wraplength, long lines
        # (plateau banners especially) clip off the right edge. The label is packed, so extra
        # wrapped lines push the rows list down cleanly rather than overlapping it.
        def _status_wrap(e):
            wl = max(300, e.width - 32)
            if getattr(self._problem_status, "_wl", None) != wl:
                self._problem_status._wl = wl
                self._problem_status.config(wraplength=wl)
        win.bind("<Configure>", lambda e: _status_wrap(e) if e.widget is win else None, add="+")

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 12))
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vbar = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        rows_id = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(rows_id, width=e.width))
        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.
        self._problem_rows = rows
        self._problem_canvas = canvas

        self._refresh_problem_images(force=True)

        def _tick():
            # Close over THIS window: an orphaned timer from a closed popup must never latch
            # onto a reopened one (each reopen starts its own loop — they'd multiply).
            if not win.winfo_exists() or getattr(self, "_problem_win", None) is not win:
                return
            try:
                self._refresh_problem_images()
            except Exception:
                pass  # one bad refresh (e.g. odd JSON shape) must not kill the auto-refresh loop
            win.after(4000, _tick)
        win.after(4000, _tick)

    def _refresh_problem_images(self, force: bool = False):
        """Re-read problem_images.json (only rebuilds when the file changed, unless forced)."""
        win = getattr(self, "_problem_win", None)
        if win is None or not win.winfo_exists():
            return
        path = self._problem_images_json_path()
        data = None
        if path and os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if not force and mtime == self._problem_mtime:
                    return  # unchanged
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
                    data = None
                self._problem_mtime = mtime  # only after a good parse — a bad read must retry next tick
            except Exception:
                data = None
        elif self._problem_mtime is not None:
            self._problem_mtime = None  # file removed (fresh run wiped it) — fall through to clear the rows
        elif not force:
            return  # still no file; nothing to redraw

        if not data or not data.get("images"):
            for w in self._problem_rows.winfo_children():
                w.destroy()
            self._problem_row_ui = {}
            self._problem_last_order = []
            self._problem_status.config(text="No data yet. Enable “Detect problem images” on the Training tab, "
                                             "start a Krea 2 run, and give it 3+ epochs of warmup.")
            return

        images = data["images"]
        counts = {}
        for s in images.values():
            v = s.get("verdict", "mid")
            counts[v] = counts.get(v, 0) + 1
        _known = ("excluded", "stuck", "suspect", "watch", "warmup", "exhausted", "learning", "mid", "easy")
        tally = "  ·  ".join([f"{counts.get(v, 0)} {v}" for v in _known]
                             + [f"{n} {v}" for v, n in sorted(counts.items()) if v not in _known])
        mode = ("per-image LR active (stuck ×0.5→×0.1 escalating, suspect ×0.7, mined-out ×0.6, learned ×1.1 boost)"
                if data.get("apply_lr") else "detection only")
        imp = data.get("improving_count")
        pend = int(data.get("pending_count") or 0)
        if data.get("plateaued") and data.get("best_epoch_estimate"):
            be = int(data["best_epoch_estimate"])
            if pend:
                progress = (f"⏳ Plateau (provisional) — the settled images finished ≈ epoch {be}, "
                            f"but {pend} image(s) are still being adjudicated (throttled or freshly "
                            f"recaptioned). If they resolve, training may get a second wind and a "
                            f"LATER epoch may be the better checkpoint — wait for the confirmed "
                            f"plateau (0 pending) before stopping.")
            else:
                progress = (f"📍 TRAINING PLATEAUED — best checkpoint ≈ epoch {be}. "
                            f"Scrub epochs {max(1, be - 2)}–{be + 2} in LoRA Royale to pick by eye; "
                            f"later epochs mainly add overbake risk.")
        elif data.get("plateaued"):
            progress = ("⏳ Plateau (provisional) — nothing improving, but "
                        f"{pend} image(s) still being adjudicated." if pend else
                        "📍 TRAINING PLATEAUED — no image is still improving.")
        elif imp is not None:
            progress = f"{imp} image(s) still improving" + (
                f"  ·  best checkpoint so far ≈ epoch {int(data['best_epoch_estimate'])}"
                if data.get("best_epoch_estimate") else "")
        else:
            progress = ""
        self._problem_status.config(
            text=f"Epoch {data.get('epoch', '?')}  ·  {len(images)} images tracked  ·  {mode}\n"
                 f"{tally}" + (f"\n{progress}" if progress else "") + "\n"
                 f"Residual = loss vs. the average at the same noise level (higher = harder than typical). "
                 f"Stuck = hard AND not improving → check the image + caption.")

        # Caption-fix queue/ack state for the row badges: queued = edit waiting for the next epoch
        # boundary (or mid-re-encode), applied = trainer re-encoded it (with the epoch number).
        loss_log_dir = os.path.dirname(path) if path else ""
        queued_keys, applied_info = set(), {}
        for qname in ("caption_updates.json", "caption_updates.json.processing"):
            try:
                qp = os.path.join(loss_log_dir, qname)
                if os.path.exists(qp):
                    with open(qp, encoding="utf-8") as f:
                        queued_keys.update(json.load(f).keys())
            except Exception:
                pass
        try:
            ap = os.path.join(loss_log_dir, "caption_updates_applied.json")
            if os.path.exists(ap):
                with open(ap, encoding="utf-8") as f:
                    applied_info = json.load(f)
        except Exception:
            pass

        style = {
            "excluded": ("#7F8C8D", "EXCLUDED from training — two AI captions couldn't fix it. Edit the caption to re-admit it, or remove it from the dataset."),
            "stuck":    ("#E74C3C", "STUCK — persistently hard, not improving. Review this image/caption."),
            "suspect":  ("#D35400", "Suspect — extremely hard from the start; provisionally slowed while the trend confirms. Worth a caption check now."),
            "watch":    ("#E67E22", "Watching — looked stuck this epoch; needs more epochs to confirm."),
            "warmup":   ("#8E7CC3", "Look-filter outlier easing in — unusual view on an LR ramp toward ×1.0 while the identity core forms; releases early once it starts improving."),
            "exhausted": ("#16A085", "Fully mined — improved a lot, then plateaued. Caption is fine; LR eased to prevent overbake."),
            "learning": ("#5B9BD5", "Learning — hard but improving. Leave it alone."),
            "mid":      ("#95A5A6", "Normal."),
            "easy":     ("#70AD47", "Learned — consistently easy. Gets a gentle ×1.1 boost to keep the healthy signal strong."),
        }
        order = {"excluded": 0, "stuck": 1, "suspect": 2, "watch": 3, "warmup": 4, "exhausted": 5,
                 "learning": 6, "mid": 7, "easy": 8}
        items = sorted(images.items(),
                       key=lambda kv: (order.get(kv[1].get("verdict", "mid"), 2),
                                       -float(kv[1].get("mean_residual", 0.0))))

        # Persistent rows: refreshes UPDATE existing rows in place instead of destroying and
        # recreating hundreds of widgets on the main thread every epoch boundary — that rebuild
        # was the window's remaining lag source. Only appearing/disappearing images create or
        # destroy widgets, and the list only re-packs when the sort order actually changed.
        new_keys = [key for key, _ in items]
        key_set = set(new_keys)
        for k in list(self._problem_row_ui):
            if k not in key_set:
                ui = self._problem_row_ui.pop(k)
                try:
                    ui["frame"].destroy()
                except Exception:
                    pass
        thumb_jobs = []
        for key, s in items:
            ui = self._problem_row_ui.get(key)
            if ui is None:
                ui = self._problem_build_row(key, thumb_jobs)
                self._problem_row_ui[key] = ui
            self._problem_update_row(ui, key, s, data, queued_keys, applied_info, style)
        if new_keys != self._problem_last_order:
            try:
                scroll_pos = self._problem_canvas.yview()[0]
            except Exception:
                scroll_pos = 0.0
            for key in new_keys:
                self._problem_row_ui[key]["frame"].pack_forget()
            for key in new_keys:
                self._problem_row_ui[key]["frame"].pack(fill=tk.X, pady=4)
            self._problem_last_order = new_keys
            try:
                self._problem_rows.update_idletasks()
                self._problem_canvas.yview_moveto(scroll_pos)
            except Exception:
                pass
        if thumb_jobs:
            self._load_thumbs_async(thumb_jobs, self._problem_thumbs)

    def _problem_build_row(self, key, thumb_jobs):
        """Create one persistent Problem Images row (static widgets only — per-refresh state is
        painted by _problem_update_row)."""
        row = tk.Frame(self._problem_rows, bg=COLORS["bg_surface"],
                       highlightbackground=COLORS["border"], highlightthickness=2)
        row.pack(fill=tk.X, pady=4)

        thumb_holder = tk.Frame(row, width=100, height=100, bg=COLORS["bg_surface"])
        thumb_holder.pack_propagate(False)
        thumb_holder.pack(side=tk.LEFT, padx=8, pady=8)
        thumb_lbl = tk.Label(thumb_holder, text="…", font=(FONT_FAMILY, 8),
                             fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        thumb_lbl.pack(expand=True)
        # Path resolution cached per key — probing 5 extensions per image per refresh added
        # hundreds of stat() calls against (often network/spinning) dataset drives.
        img_path = self._problem_img_paths.get(key)
        if img_path is None:
            img_path = self._find_dataset_image(key)
            if img_path:
                self._problem_img_paths[key] = img_path
        if img_path:
            ph = self._problem_thumbs.get(img_path)
            if ph is not None:
                thumb_lbl.config(image=ph, text="")
            else:
                thumb_jobs.append((img_path, thumb_lbl))
        else:
            thumb_lbl.config(text="no\npreview")

        info = tk.Frame(row, bg=COLORS["bg_surface"])
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)
        name_row = tk.Frame(info, bg=COLORS["bg_surface"])
        name_row.pack(fill=tk.X, anchor="w")
        tk.Label(name_row, text=os.path.basename(key), font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        verdict_lbl = tk.Label(name_row, text="", font=(FONT_FAMILY, 9, "bold"),
                               bg=COLORS["bg_surface"])
        verdict_lbl.pack(side=tk.LEFT)
        badge_lbl = tk.Label(name_row, text="", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        badge_lbl.pack(side=tk.LEFT)
        ttk.Button(name_row, text="✏ Edit Caption", width=14,
                   command=lambda k=key: self._open_caption_editor(k)).pack(side=tk.RIGHT, padx=(8, 0))
        stats_lbl = tk.Label(info, text="", font=(FONT_FAMILY, 9),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        stats_lbl.pack(anchor="w", pady=(2, 0))
        blurb_lbl = tk.Label(info, text="", font=(FONT_FAMILY, 8, "italic"),
                             fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        blurb_lbl.pack(anchor="w", pady=(2, 0))
        return {"frame": row, "verdict": verdict_lbl, "badge": badge_lbl,
                "stats": stats_lbl, "blurb": blurb_lbl}

    def _problem_update_row(self, ui, key, s, data, queued_keys, applied_info, style):
        """Paint one row's per-refresh state (verdict colour, badges, stats) in place."""
        verdict = s.get("verdict", "mid")
        color, blurb = style.get(verdict, style["mid"])
        try:
            ui["frame"].config(highlightbackground=color)
            ui["verdict"].config(text=f"  {verdict.upper()}", fg=color)
            if key in queued_keys:
                ui["badge"].config(text="  ✏ fix queued", fg="#F1C40F")
            elif key in applied_info:
                # Ledger entries are per-fix history lists (older files carry a single dict);
                # the badge shows the LATEST fix.
                _entry = applied_info[key]
                if isinstance(_entry, list):
                    _entry = _entry[-1] if _entry else {}
                _ai = _entry.get("auto")
                _att = int(_entry.get("attempt", 1) or 1)
                _ep = _entry.get("epoch", "?")
                if _ai and _att >= 2:
                    _txt = f"  🤖 AI re-captioned ×2 (detailed) @ epoch {_ep} — last chance"
                elif _ai:
                    _txt = f"  🤖 AI re-captioned @ epoch {_ep}"
                else:
                    _txt = f"  ✓ caption re-encoded @ epoch {_ep}"
                ui["badge"].config(text=_txt, fg="#2ECC71")
            else:
                ui["badge"].config(text="")
            # Trend shows the DECISION metric (the half-window drop test the verdicts actually
            # use), not the raw slope — the old slope arrow could say "improving" while the
            # decision bar said otherwise, which read as a contradiction next to a stuck badge.
            slope = float(s.get("slope", 0.0))
            if "improving" in s:
                trend = "↓ improving" if s["improving"] else ("↑ worsening" if slope > 1e-4 else "→ plateau")
            else:
                trend = "↓ improving" if slope < -1e-4 else ("↑ worsening" if slope > 1e-4 else "→ flat")
            # Stuck + improving = release countdown; make the state legible instead of confusing.
            rv = int(s.get("release_votes", 0))
            if verdict == "stuck" and s.get("improving"):
                trend += f" — releasing ({rv}/3 clean epochs)"
            elif verdict == "stuck" and s.get("stuck_epochs"):
                trend += f" — stuck {int(s['stuck_epochs'])} epochs"
            mult = s.get("multiplier", 1.0)
            ui["stats"].config(text=(
                f"difficulty {float(s.get('mean_residual', 0.0)):+.4f}   ·   trend {trend}   ·   "
                f"mean loss {float(s.get('mean_loss', 0.0)):.4f}   ·   "
                f"{int(s.get('epochs', 0))} epochs tracked"
                + (f"   ·   LR ×{mult:g}" if data.get("apply_lr") and mult != 1.0 else "")))
            ui["blurb"].config(text=blurb, fg=COLORS["text_muted"])
        except Exception:
            pass   # a dying widget mid-refresh must not kill the loop

    def _find_dataset_caption(self, key: str):
        """Caption file for a loss-watch item key: <image_folder>/<basename><caption_ext>."""
        folder = self.image_folder_var.get().strip() if hasattr(self, "image_folder_var") else ""
        if not folder or not os.path.isdir(folder):
            return None
        ext = ".txt"
        try:
            ext = self.dataset_caption_ext_var.get().strip() or ".txt"
        except Exception:
            pass
        return os.path.join(folder, os.path.basename(key) + ext)

    def _queue_caption_update(self, key: str, caption: str) -> bool:
        """Merge one caption edit into <output_dir>/loss_log/caption_updates.json (atomic write).
        The trainer consumes it at the next epoch boundary and re-encodes the embedding."""
        d = self._loss_log_dir()
        if not d:
            return False
        try:
            os.makedirs(d, exist_ok=True)
            qp = os.path.join(d, "caption_updates.json")
            updates = {}
            if os.path.exists(qp):
                try:
                    with open(qp, encoding="utf-8") as f:
                        updates = json.load(f)
                    if not isinstance(updates, dict):
                        updates = {}
                except Exception:
                    # An unreadable EXISTING queue means other pending edits we can't see —
                    # writing just this key would silently discard them. Fail instead.
                    print("[caption-fix] queue exists but could not be read — not overwriting it")
                    return False
            updates[str(key)] = caption
            tmp = qp + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(updates, f, indent=2)
            os.replace(tmp, qp)  # atomic — the trainer never sees a half-written file
            return True
        except Exception as e:
            print(f"[caption-fix] queue failed: {e}")
            return False

    def _open_caption_editor(self, key: str):
        """Standalone caption editor for one problem image. Deliberately a SEPARATE Toplevel from
        the auto-refreshing list, so rows can rebuild/reorder underneath without eating your edit."""
        editors = getattr(self, "_caption_editors", None)
        if editors is None:
            editors = self._caption_editors = {}
        existing = editors.get(key)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return

        win = tk.Toplevel(self.master)
        win.title(f"Edit Caption — {os.path.basename(key)}")
        win.geometry("560x680")
        win.minsize(520, 560)
        win.configure(bg=COLORS["bg_deep"])
        editors[key] = win
        win.bind("<Destroy>", lambda e: editors.pop(key, None) if e.widget is win else None)

        # Bottom bar FIRST with side=BOTTOM so the buttons can never be clipped off the window,
        # whatever the thumbnail aspect/caption length pushes the middle content to.
        btns = tk.Frame(win, bg=COLORS["bg_deep"])
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(6, 12))
        status = tk.Label(win, text="Tip: caption what the image actually shows — viewpoint "
                                    "(“from behind”, “side profile”), pose, occlusions.",
                          font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_deep"],
                          wraplength=520, justify=tk.LEFT)
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=14)

        img_path = self._find_dataset_image(key)
        if img_path:
            try:
                im = Image.open(img_path)
                im.thumbnail((280, 280), Image.LANCZOS)
                ph = ImageTk.PhotoImage(im)
                win._thumb_ref = ph  # keep alive
                tk.Label(win, image=ph, bg=COLORS["bg_deep"]).pack(pady=(14, 6))
            except Exception:
                img_path = None
        if not img_path:
            tk.Label(win, text="(image preview unavailable)", font=(FONT_FAMILY, 9),
                     fg=COLORS["text_muted"], bg=COLORS["bg_deep"]).pack(pady=(14, 6))

        cap_path = self._find_dataset_caption(key)
        caption = ""
        cap_read_failed = False
        if cap_path and os.path.exists(cap_path):
            try:
                # utf-8-sig strips a BOM (which would otherwise ride into the embedding);
                # errors="replace" keeps a legacy-ANSI caption editable instead of blank.
                with open(cap_path, encoding="utf-8-sig", errors="replace") as f:
                    caption = f.read().strip()
            except Exception:
                cap_read_failed = True

        tk.Label(win, text=os.path.basename(key), font=(FONT_FAMILY, 11, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack()
        txt = tk.Text(win, height=6, wrap=tk.WORD, font=(FONT_FAMILY, 10),
                      bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                      insertbackground=COLORS["text_primary"], relief=tk.FLAT, padx=8, pady=8)
        txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        txt.insert("1.0", caption)
        if cap_read_failed:
            status.config(fg="#E74C3C", text="Couldn't read the existing caption file — the box "
                          "starts empty. Saving will OVERWRITE the .txt with what you type.")

        def _save():
            new_cap = txt.get("1.0", tk.END).strip()
            if not new_cap:
                status.config(text="Caption is empty — not saved.", fg="#E74C3C")
                return
            wrote_txt = False
            if cap_path:
                try:
                    with open(cap_path, "w", encoding="utf-8") as f:
                        f.write(new_cap)
                    wrote_txt = True
                except Exception as e:
                    print(f"[caption-fix] .txt write failed: {e}")
            queued = self._queue_caption_update(key, new_cap)
            if queued:
                status.config(fg="#2ECC71", text="Saved & queued ✓ — the trainer re-encodes it at the next "
                              "epoch boundary; this image's history resets and it should turn blue "
                              "(learning) if the fix worked."
                              + ("" if wrote_txt else "  (Note: couldn't write the .txt — the live run is "
                                 "fixed, but re-caching later will use the old caption.)"))
            else:
                status.config(fg="#E74C3C", text="Could not queue the update (no output directory?). "
                              + ("The .txt was updated for future runs." if wrote_txt else ""))

        ttk.Button(btns, text="Save & Queue for Re-encode", command=_save).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    # ── Timestep section helpers ────────────────────────────────────────

    def _on_adaptive_lr_toggle(self):
        """Enable/disable the Min/Max LR dropdowns based on the Adaptive LR checkbox,
        and grey out the Learning Rate box — it is IGNORED while adaptive is on (the run
        starts at the geometric midpoint of Min/Max; the watcher owns the LR from there)."""
        if not hasattr(self, 'entries') or "ADAPTIVE_LR_MIN" not in self.entries:
            return
        on = self.adaptive_lr_var.get()
        # Comboboxes: "readonly" when enabled (dropdown active, no free typing), "disabled" when not
        combo_state = "readonly" if on else "disabled"
        btn_state = "normal" if on else "disabled"
        self.entries["ADAPTIVE_LR_MIN"].config(state=combo_state)
        self.entries["ADAPTIVE_LR_MAX"].config(state=combo_state)
        if hasattr(self, '_adaptive_reset_btn'):
            self._adaptive_reset_btn.config(state=btn_state)
        # The LR box is the inverse: live when adaptive is OFF, greyed when ON.
        try:
            if "LEARNING_RATE" in self.entries:
                self.entries["LEARNING_RATE"].config(state="disabled" if on else "normal")
            if hasattr(self, "labels") and "LEARNING_RATE" in self.labels:
                self.labels["LEARNING_RATE"].config(
                    fg=COLORS["text_muted"] if on else COLORS["text_secondary"])
        except Exception:
            pass

    def _parse_blocks_swap(self) -> int:
        """Extract integer from the BLOCKS_SWAP combobox value.
        'Auto' resolves to a value based on GPU VRAM (training needs more headroom than inference)."""
        import re as _re
        raw = self.entries["BLOCKS_SWAP"].get().strip()
        if raw.lower().startswith("auto"):
            cfg = ARCHITECTURES.get(self.architecture_var.get(), {})
            if cfg.get("is_krea2"):
                return self._auto_krea2_strategy()
            return self._auto_training_blocks_swap()
        # Explicit swap value: any INT8 pick from a PREVIOUS auto pass must not leak into
        # this launch (stale --quant_int8 alongside --blocks_to_swap N OOM'd small cards).
        self._auto_quant_int8 = ""
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _auto_krea2_strategy(self) -> int:
        """Choose Krea 2 quantisation AND swap together, then return the swap count.

        Picking a swap count from VRAM alone produced the worst possible outcome on 16 GB
        cards: fp8 doesn't fit, so it swapped 20 of 28 blocks every step. Measured on a 5090
        (Krea 2, 36 imgs @ 0.25 MP, batch 1):

            fp8, no swap   0.85 s/it   20.1 GB   12.5% CPU
            fp8, swap 20   3.09 s/it   12.3 GB   49.9% CPU
            NF4, no swap   0.70 s/it   13.8 GB   14.0% CPU

        NF4 is both faster and smaller, so it leads. Only touches the 4-bit toggle when the
        user has left block swap on Auto — an explicit swap choice is left alone.
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(_REPO_ROOT, "src"))
            from fizgig.utils.capabilities import detect, recommend_krea2_strategy
        except Exception:
            self._auto_quant_int8 = ""   # no strategy ran — a stale INT8 pick must not leak
            return self._auto_krea2_blocks_swap()

        try:
            caps = detect()
            # Budget for THIS run's shape — batch size is the largest term (+2.4 GB/image);
            # a single-constant budget let batch 2 sail through the check and OOM.
            try:
                _mp = float(self.dataset_megapixels_var.get().strip() or 0.25)
            except (ValueError, AttributeError):
                _mp = 0.25
            try:
                _bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
            except (ValueError, AttributeError):
                _bs = 1
            try:
                _rk = int(self.entries["NETWORK_DIM"].get().strip() or 32)
            except (ValueError, KeyError, AttributeError):
                _rk = 32
            # LoKR: the hidden rank box is meaningless (baseline it at 32) and the factor
            # carries the real state cost — params scale 1/factor², so factor 4 is ~+4 GB
            # the budget must know about on tight cards.
            _ntype = "lokr" if self._network_type_is_lokr() else "lora"
            if _ntype == "lokr":
                _rk = 32
            try:
                _lf = int(self.entries["LOKR_FACTOR"].get().strip() or 8)
            except (ValueError, KeyError, AttributeError):
                _lf = 8
            # If the user pinned the 4-bit control, the plan must be built AROUND that choice —
            # otherwise the swap count is sized for a quantisation that will not run. That
            # exact mismatch (fp8 given NF4's swap-0 plan) OOM'd 16 GB cards; reproduced and
            # fixed 28 Jul.
            #
            # "Off" maps to no_4bit, not fp8: the control is labelled *4-bit Base*, so turning
            # it off is a vote against NF4, not against every quantisation. INT8 is 8-bit,
            # faster than NF4 and far more accurate, so it still applies where it fits —
            # briefly making Off mean plain fp8 cost 20 GB+ cards the fastest path for nothing.
            _force = self._krea2_force_quant() if hasattr(self, "quant_4bit_mode_var") else None
            plan = recommend_krea2_strategy(caps=caps, mp=_mp, batch=_bs, rank=_rk,
                                            force_quant=_force,
                                            network_type=_ntype, lokr_factor=_lf)
        except Exception:
            self._auto_quant_int8 = ""   # no strategy ran — a stale INT8 pick must not leak
            return self._auto_krea2_blocks_swap()

        try:
            self.update_console(f"[auto] {caps.summary()}\n[auto] {plan.reason}\n")
        except Exception:
            pass
        # INT8 has no GUI toggle (it is newer than the 4-bit control) — carry it on the
        # instance so the krea2 command builder can pass --quant_int8.
        self._auto_quant_int8 = getattr(plan, "quant_int8", "") or ""
        # The plan only drives the NF4 flag when Base precision is on Auto — an explicit
        # choice is the user's call and the strategy must not override it. (Compared via the
        # canonical key, not the display label, which is why the label can change freely.)
        _q4_auto = (not hasattr(self, "quant_4bit_mode_var")
                    or self._base_precision() == "auto")
        if (_q4_auto and hasattr(self, "quant_4bit_var")
                and bool(self.quant_4bit_var.get()) != plan.quant_4bit):
            self.quant_4bit_var.set(plan.quant_4bit)
            try:
                self._on_quant_4bit_toggle()
            except Exception:
                pass
            try:
                self.update_console(
                    f"[auto] 4-bit NF4 base turned {'ON' if plan.quant_4bit else 'OFF'} "
                    "(block swap is on Auto — set it explicitly to control this yourself)\n")
            except Exception:
                pass
        return int(plan.blocks_to_swap)

    def _auto_krea2_blocks_swap(self) -> int:
        """Pick Krea 2 training block swap from GPU VRAM. Krea 2's RAW DiT is ~14 GB in fp8,
        so the training step fits a 32 GB card with no swap (fastest — no PCIe transfers); the
        in-training preview parks the training DiT on CPU separately, so swap only governs the
        training step. Smaller cards swap progressively. Max swap is 26 (28 main blocks − 2)."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 30:
                    return 0    # 32 GB — no swap; fp8 base (~14 GB) trains resident
                if vram_gb >= 22:
                    return 12   # 24 GB
                if vram_gb >= 15:
                    return 20   # 16 GB
                return 26       # <16 GB — maximum
        except Exception:
            pass
        return 12  # safe default for an unknown smaller card

    def _auto_krea2_inference_blocks_swap(self) -> int:
        """Pick Krea 2 INFERENCE/preview block swap from GPU VRAM, tuned for the fp8 Turbo.
        Measured: the Turbo peaks ~22.6 GB at swap 0 (DiT + the transient Qwen3-VL encode
        spike) and drops ~0.43 GB per swapped block — heavier than Klein's ~9 GB Distilled, so
        reusing the Klein inference preset would under-swap and OOM smaller cards. This adapts to
        the actual card so the workbench + previews 'just work'. Forward-only (lighter than the
        training step); max swap is 26 (28 main blocks − 2)."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 30:
                    return 0    # 32 GB — Turbo (~22.6 GB peak) fits resident, fastest
                if vram_gb >= 22:
                    return 4    # 24 GB — light swap for headroom over the encode spike
                if vram_gb >= 18:
                    return 12   # 20 GB
                if vram_gb >= 15:
                    return 20   # 16 GB
                return 26       # <16 GB — maximum
        except Exception:
            pass
        return 20  # safe default for an unknown smaller card

    def _auto_training_blocks_swap(self) -> int:
        """Pick training block swap based on GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # 16 GB and up → no swap. The fp8 Base is only ~9.6 GB resident, so
                # 16 GB cards train without swapping, and skipping swap is faster (no
                # PCIe block transfers). Threshold is 15, not 16: a 16 GB card reports
                # ~15.9 GiB total (drivers reserve a little), so a >=16 gate would
                # wrongly exclude it. Only genuinely smaller cards (<15 GB) swap.
                if vram_gb >= 15:
                    return 0   # 16 GB / 24 GB / 32 GB — no swap needed
                if vram_gb >= 10:
                    return 12  # 12 GB cards
                return 16      # <10 GB — maximum swap
        except Exception:
            pass
        return 0  # safe fallback — avoid the buggy swap path on detection failure

    def _on_gpu_choice(self, _event=None):
        """Save the picked GPU by UUID. Label -> UUID, so CUDA_VISIBLE_DEVICES is immune to
        NVML vs. CUDA index reordering (issue #104)."""
        _picked = self._gpu_choice_var.get()
        _uuid = ""
        for k, v in self._gpu_choice_labels.items():
            if v == _picked:
                _uuid = self._gpu_info.get(k, (None, None, None, ""))[3] or ""
                break
        self.prefs_vars["cuda_device"].set(_uuid)          # trace writes prefs.json
        self.update_console(
            f"[gpu] training will use {_picked}. Restart Fizgig to move the workbench tools too.\n"
            if _uuid else "[gpu] back to the system default GPU.\n")

    def _cuda_env_for_subprocess(self, env):
        """Stamp the chosen GPU onto a subprocess environment.

        The child would inherit our own CUDA_VISIBLE_DEVICES anyway, but only the value set at
        startup - so without this, changing the pref would not reach a run until the app was
        restarted, which is the one place it easily can take effect immediately."""
        _want = str(self.prefs_vars["cuda_device"].get()).strip() if hasattr(
            self, "prefs_vars") else ""
        if _want and not getattr(self, "_cuda_device_env_locked", False):
            env["CUDA_VISIBLE_DEVICES"] = _want
        return env

    def _get_inference_blocks_to_swap(self) -> int:
        """Resolve the Preferences inference_blocks_to_swap pref to an int.

        'Auto (detect from GPU)' resolves from VRAM at call time (same as the
        training Blocks Swap setting). Labeled options like '16 (Max — …)' store
        as the full string; we take the leading integer. Returns 0 on failure."""
        import re as _re
        raw = ""
        try:
            raw = str(self.prefs_vars["inference_blocks_to_swap"].get()).strip()
        except Exception:
            return 0
        if raw.lower().startswith("auto"):
            return _auto_detect_blocks_to_swap()
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _get_inference_int8(self) -> bool:
        """Resolve the Preferences 'INT8 fast inference' toggle (workbench + previews) to a bool."""
        try:
            return str(self.prefs_vars["inference_int8"].get()).strip() in ("1", "True", "true")
        except Exception:
            return False

    def _resolve_script(self, config: dict, script_key: str) -> str:
        """Resolve an absolute script path from an architecture config entry.

        Klein lives under FIZGIG_DIR — strip any legacy "FizgigIndependent/" prefix
        on the config value (back-compat with older config strings) and join onto
        FIZGIG_DIR.
        """
        rel = config[script_key]
        if rel.startswith("FizgigIndependent/"):
            rel = rel[len("FizgigIndependent/"):]
        return os.path.join(_FIZGIG_DIR, rel)

    def _get_path(self, key: str) -> str:
        """Resolve a model/path setting from the current source of truth.

        Pulls from prefs_vars (model paths) or from the hidden _dataset_config_var
        (dataset config).
        """
        pref_map = {
            "VAE_MODEL": "vae",
            "DIT_MODEL": "base_dit",
            "TEXT_ENCODER": "text_encoder",
            "LORA_OUTPUT_DIR": "lora_output_dir",
        }
        pref_key = pref_map.get(key)
        if pref_key and pref_key in self.prefs_vars:
            return self.prefs_vars[pref_key].get()
        if key == "DATASET_CONFIG":
            return self._dataset_config_var.get() if hasattr(self, "_dataset_config_var") else ""
        return ""

    def _reset_adaptive_lr_defaults(self):
        """Reset Learning Rate, Min LR, and Max LR to adaptive-mode defaults."""
        # Learning Rate is a free-text entry; Min/Max LR are comboboxes.
        lr_entry = self.entries.get("LEARNING_RATE")
        if lr_entry is not None:
            lr_entry.delete(0, tk.END)
            lr_entry.insert(0, "4e-4")
        for key, value in (("ADAPTIVE_LR_MIN", "1e-5"), ("ADAPTIVE_LR_MAX", "4e-4")):
            entry = self.entries.get(key)
            if entry is not None:
                entry.config(state="readonly")
                entry.set(value)
        # Re-apply enabled/disabled state on the adaptive fields
        self._on_adaptive_lr_toggle()

    def _on_timestep_sampling_changed(self, event=None):
        """Enable/disable sigmoid_scale based on selected sampling method."""
        sampling = self.ts_sampling_var.get()
        uses_sigmoid = sampling in ("sigmoid", "shift")
        state = "normal" if uses_sigmoid else "disabled"
        color = COLORS["text_secondary"] if uses_sigmoid else COLORS["text_muted"]
        self.entries["SIGMOID_SCALE"].config(state=state)
        self.ts_sigmoid_label.config(fg=color)

    def _on_weighting_scheme_changed(self, event=None):
        """Enable/disable logit_mean/std and mode_scale based on weighting scheme."""
        scheme = self.weighting_scheme_var.get()

        # Logit Normal params
        is_logit = (scheme == "logit_normal")
        logit_state = "normal" if is_logit else "disabled"
        logit_color = COLORS["text_secondary"] if is_logit else COLORS["text_muted"]
        self.entries["LOGIT_MEAN"].config(state=logit_state)
        self.entries["LOGIT_STD"].config(state=logit_state)
        self.ts_logit_label.config(fg=logit_color)

        # Mode Scale param
        is_mode = (scheme == "mode")
        mode_state = "normal" if is_mode else "disabled"
        mode_color = COLORS["text_secondary"] if is_mode else COLORS["text_muted"]
        self.entries["MODE_SCALE"].config(state=mode_state)
        self.ts_mode_label.config(fg=mode_color)

    def _update_noise_range_label(self):
        """Update the dynamic noise range description label."""
        if not hasattr(self, 'noise_range_label'):
            return
        min_str = self.entries["MIN_TIMESTEP"].get().strip()
        max_str = self.entries["MAX_TIMESTEP"].get().strip()

        if not min_str and not max_str:
            self.noise_range_label.config(text="Full range (default) - All noise levels",
                                          fg=COLORS["accent"])
            return

        try:
            min_val = int(min_str) if min_str else 0
            max_val = int(max_str) if max_str else 1000
        except ValueError:
            self.noise_range_label.config(text="Invalid timestep values", fg=COLORS["error"])
            return

        if min_val == 0 and max_val >= 1000:
            self.noise_range_label.config(text=f"Full range ({min_val}-{max_val}) - All noise levels",
                                          fg=COLORS["accent"])
        elif max_val <= 300:
            self.noise_range_label.config(text=f"High noise ({min_val}-{max_val}) - Composition/structure",
                                          fg=COLORS["success"])
        elif min_val >= 700:
            self.noise_range_label.config(text=f"Low noise ({min_val}-{max_val}) - Details/textures",
                                          fg="#B388FF")  # purple
        elif min_val >= 300 and max_val <= 700:
            self.noise_range_label.config(text=f"Mid noise ({min_val}-{max_val}) - Features/characteristics",
                                          fg=COLORS["warning"])
        else:
            self.noise_range_label.config(text=f"Custom range ({min_val}-{max_val})",
                                          fg=COLORS["text_secondary"])

    def _ts_preset_full_range(self):
        """Timestep preset: Full Range"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_structure(self):
        """Timestep preset: Structure Focus (high noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "0")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "300")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_detail(self):
        """Timestep preset: Detail Focus (low noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "700")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "1000")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_sigmoid(self):
        """Timestep preset: Balanced Sigmoid"""
        self.ts_sampling_var.set("sigmoid")
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def show_row(self, key):
        """Show a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid()
            row_info["entry"].grid()
            if row_info["browse"]:
                row_info["browse"].grid()

    def hide_row(self, key):
        """Hide a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid_remove()
            row_info["entry"].grid_remove()
            if row_info["browse"]:
                row_info["browse"].grid_remove()

    def _network_type_is_lokr(self) -> bool:
        try:
            return str(self.entries["NETWORK_TYPE"].get()).startswith("LoKR")
        except (KeyError, tk.TclError):
            return False

    def _on_network_type_changed(self):
        """LoKR has no rank/alpha — a single Factor dial replaces them, so the rows swap.
        Only meaningful under Krea 2; the arch-visibility pass calls this on family switch."""
        if self._network_type_is_lokr():
            self.hide_row("NETWORK_DIM")
            self.hide_row("NETWORK_ALPHA")
            self.show_row("LOKR_FACTOR")   # hint lives inside the row frame, rides along
        else:
            self.show_row("NETWORK_DIM")
            self.show_row("NETWORK_ALPHA")
            self.hide_row("LOKR_FACTOR")
        self._save_last_used_paths()

    def toggle_scaled(self):
        """Enable or disable the Scaled checkbox based on FP8 checkbox state"""
        if self.fp8_var.get():
            self.scaled_check.config(state=tk.NORMAL)
        else:
            self.scaled_check.config(state=tk.DISABLED)
            self.scaled_var.set(False)

    def _refresh_perimage_toggle_state(self, *args):
        """Grey out the four per-image watch toggles whenever Batch Size > 1.

        Every per-image feature (detection, per-image LR, auto-recaption, look warm-up)
        rests on attributing one step's loss to one image; the trainer already disables
        the LR side loudly at batch > 1, but the tickboxes stayed live and looked like
        they'd work. Values are preserved — dropping batch back to 1 re-enables them."""
        try:
            bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
        except (ValueError, AttributeError):
            bs = 1
        state = tk.DISABLED if bs > 1 else tk.NORMAL
        for cb in (getattr(self, "_krea2_detect_cb", None),
                   getattr(self, "_krea2_perimglr_cb", None),
                   getattr(self, "_krea2_autorecap_cb", None),
                   getattr(self, "_krea2_warmuplook_cb", None)):
            if cb is not None:
                try:
                    cb.configure(state=state)
                except Exception:
                    pass
        note = getattr(self, "_krea2_perimage_batch_note", None)
        if note is not None:
            try:
                if bs > 1:
                    note.grid()
                else:
                    note.grid_remove()
            except Exception:
                pass

    # Base precision (Krea 2). Canonical key -> the label shown in the combobox. Stored as the
    # KEY so the saved value stays stable if the wording changes.
    _BASE_PRECISION_LABELS = {
        "auto": "Auto (recommended)",
        "int8": "INT8 — 8-bit, fastest",
        "nf4":  "4-bit NF4 — smallest",
        "fp8":  "fp8 — least compressed",
    }

    @classmethod
    def _normalize_base_precision(cls, value) -> str:
        """Canonical key from a stored value, a display label, or a legacy Auto/On/Off.

        Legacy migration preserves BEHAVIOUR, not the old label's wording. "Off" meant "not
        4-bit", which the strategy resolved to INT8 wherever it fits — so Off maps to int8,
        and int8 degrades to fp8 by itself on a card that cannot run it or has too little
        VRAM. Mapping Off to fp8 instead would silently drop 20 GB+ cards off the INT8 path,
        which is precisely the regression v2.8.7 had to hotfix.
        """
        v = str(value or "").strip()
        if v in cls._BASE_PRECISION_LABELS:
            return v
        for key, label in cls._BASE_PRECISION_LABELS.items():
            if v == label:
                return key
        return {"Auto": "auto", "On": "nf4", "Off": "int8", "no_4bit": "int8"}.get(v, "auto")

    def _base_precision(self) -> str:
        """Canonical base-precision key currently selected."""
        return self._normalize_base_precision(self.quant_4bit_mode_var.get())

    def _krea2_force_quant(self):
        """force_quant for recommend_krea2_strategy — None when Auto (let the ladder choose)."""
        key = self._base_precision()
        return None if key == "auto" else key

    def _on_quant_4bit_mode_changed(self):
        """Derive quant_4bit_var from the selected base precision. Only NF4 sets it True; Auto
        rests at False and the launch-time strategy sets it (Krea 2 + Blocks Swap on Auto)."""
        self.quant_4bit_var.set(self._base_precision() == "nf4")
        self._on_quant_4bit_toggle()

    def _on_quant_4bit_toggle(self):
        """4-bit (NF4) base forces block swap off (NF4 weights live in
        module._nf4_packed, not .weight, so they can't be swapped) and supersedes
        the fp8 Base options. Grey those controls while it's on."""
        on = self.quant_4bit_var.get()
        try:
            # Show what will actually run: the trainer forces swap to 0 under 4-bit, but
            # the greyed-out box kept displaying the old count. Remember the user's value
            # and restore it when 4-bit is toggled off.
            if on:
                _cur = self.entries["BLOCKS_SWAP"].get()
                if _cur != "0":
                    self._blocks_swap_before_4bit = _cur
                    self.entries["BLOCKS_SWAP"].set("0")
                self.entries["BLOCKS_SWAP"].configure(state="disabled")
            else:
                self.entries["BLOCKS_SWAP"].configure(state="normal")
                _prev = getattr(self, "_blocks_swap_before_4bit", None)
                if _prev is not None and self.entries["BLOCKS_SWAP"].get() == "0":
                    self.entries["BLOCKS_SWAP"].set(_prev)
                    self._blocks_swap_before_4bit = None
        except Exception:
            pass
        for chk in (getattr(self, "fp8_check", None), getattr(self, "scaled_check", None)):
            if chk is not None:
                try:
                    chk.configure(state=tk.DISABLED if on else tk.NORMAL)
                except Exception:
                    pass
        # 4-bit base REQUIRES gradient checkpointing. Its dequant forward (like the
        # old fp8 dequant) materializes a bf16 weight per matmul; without GC,
        # autograd pins all ~112 of them (~18 GB) for the backward — instant OOM on
        # the 10-12 GB cards NF4 targets. There's no frugal-dequant Function for NF4
        # (only fp8), so force GC on and lock the checkbox while NF4 is selected.
        gc_chk = getattr(self, "grad_checkpoint_check", None)
        if gc_chk is not None:
            try:
                if on:
                    self.grad_checkpoint_var.set(True)
                    gc_chk.configure(state=tk.DISABLED)
                else:
                    gc_chk.configure(state=tk.NORMAL)
            except Exception:
                pass
        if not on:
            # restore the Scaled checkbox's dependent-disabled state
            self.toggle_scaled()

    def _on_grad_checkpoint_toggle(self):
        """Warn (VRAM-aware) when gradient checkpointing is switched OFF. Turning
        it on is always safe; off greatly increases activation VRAM, so it only
        fits on big cards with no block swap."""
        if self.grad_checkpoint_var.get():
            return  # ON is always safe — no warning
        # Switched OFF — assess the card.
        try:
            import torch
            vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                       if torch.cuda.is_available() else 0.0)
        except Exception:
            vram_gb = 0.0
        # Is block swap likely active? Auto on a <24GB card resolves to >0; an
        # explicit non-zero value also means swapping.
        swap_raw = str(self.settings.get("BLOCKS_SWAP", "auto")).strip().lower()
        swap_active = False
        if swap_raw.startswith("auto"):
            swap_active = bool(vram_gb) and vram_gb < 23
        else:
            try:
                swap_active = int(''.join(ch for ch in swap_raw if ch.isdigit()) or "0") > 0
            except ValueError:
                swap_active = False

        if (vram_gb and vram_gb < 23) or swap_active:
            vram_txt = f"~{vram_gb:.0f} GB" if vram_gb else "this card"
            messagebox.showwarning(
                "Gradient checkpointing off — likely to run out of memory",
                f"Your GPU reports {vram_txt}"
                + (" and block swap is active" if swap_active else "") + ".\n\n"
                "With gradient checkpointing OFF, a 9B LoRA holds every block's activations for the "
                "backward pass — that usually won't fit under ~24 GB and training will likely hit CUDA "
                "out-of-memory.\n\nThis option is meant for 24 GB+ cards (ideally 32 GB) with Blocks Swap "
                "set to 0. On your setup, leave it ON unless you know you have the headroom.")
        else:
            messagebox.showinfo(
                "Gradient checkpointing off",
                "Gradient checkpointing is now OFF — training runs ~20–30% faster but uses much more VRAM.\n\n"
                "For it to fit, set Blocks Swap to 0 and keep resolution/batch modest. If you hit CUDA "
                "out-of-memory, switch it back on.")

    def _populate_other_options(self, parent, start_row=0):
        """Populate Attention / Logging / Memory / Metadata fields onto the given parent.
        Used to inline these into the Other Options section on the Training tab."""
        row = start_row

        # Attention Mechanism
        ttk.Label(parent, text="Attention Mechanism:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.attention_var = tk.StringVar(value=self.settings["ATTENTION_MECHANISM"])
        attention_options = ["sdpa", "flash3"]
        self.entries["ATTENTION_MECHANISM"] = ttk.Combobox(parent, textvariable=self.attention_var, values=attention_options, state="readonly")
        self.entries["ATTENTION_MECHANISM"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1
        ttk.Label(parent, text="sdpa works on all GPUs. flash3 requires pip install flash-attn and an "
                  "NVIDIA Hopper/Blackwell GPU (H100, RTX 5090, etc.).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        # Logging
        ttk.Label(parent, text="Logging Directory:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOGGING_DIR"] = ttk.Entry(parent, width=40)
        self.entries["LOGGING_DIR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.browse_directory("LOGGING_DIR")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent, text="Log With:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.log_with_var = tk.StringVar(value=self.settings["LOG_WITH"])
        log_with_options = ["none", "tensorboard", "wandb", "all"]
        self.entries["LOG_WITH"] = ttk.Combobox(parent, textvariable=self.log_with_var, values=log_with_options, state="readonly")
        self.entries["LOG_WITH"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Log Prefix:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOG_PREFIX"] = ttk.Entry(parent, width=40)
        self.entries["LOG_PREFIX"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # img_in/txt_in offloading — no-op for Klein 9B, widget kept for preset compat
        self.img_in_txt_in_offloading_var = tk.BooleanVar(value=self.settings["IMG_IN_TXT_IN_OFFLOADING"])
        self.entries["IMG_IN_TXT_IN_OFFLOADING"] = self.img_in_txt_in_offloading_var

        # Metadata
        ttk.Label(parent, text="Metadata Title:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TITLE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TITLE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Author:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_AUTHOR"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_AUTHOR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Description:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_DESCRIPTION"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_DESCRIPTION"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata License:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_LICENSE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_LICENSE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Tags:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TAGS"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TAGS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Trigger Phrase:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TRIGGER_PHRASE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TRIGGER_PHRASE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1
        ttk.Label(parent, text="Blank uses the Captions tab's trigger word.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent, text="Metadata Thumbnail:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_THUMBNAIL"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_THUMBNAIL"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.browse_file("METADATA_THUMBNAIL", "file")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        ttk.Label(parent, text="Blank auto-embeds the latest sample preview; type 'off' to disable.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        return row