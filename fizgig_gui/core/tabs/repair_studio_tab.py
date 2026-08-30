import os
import sys
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY
from fizgig_gui.core.config.presets import SEED_TRAVEL_PRESETS
from fizgig_gui.core.ui_base.widgets import ToolTip

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class RepairStudioTabMixin:
    # Color palette mirrors src/fizgig/profiler/visualize.py 5-bucket scheme
    _REPAIR_CAT_COLOR = {
        "style_composition": "#5B9BD5",
        "style_ident_overlap": "#5BB3A6",
        "identity": "#70AD47",
        "ident_details_overlap": "#B8A547",
        "details": "#ED7D31",
    }
    _REPAIR_CAT_SHORT = {
        "style_composition": "Style+Comp",
        "style_ident_overlap": "Style/ID",
        "identity": "Identity",
        "ident_details_overlap": "ID/Detail",
        "details": "Details",
    }

    @staticmethod
    def _repair_category_for_block(block_id: str) -> str:
        # Klein's 5-bucket map. Non-Klein ids (block_N, h3blk_N, h3_rf_N — the last has THREE
        # underscore parts, which the old two-way unpack crashed on) get a neutral bucket:
        # their families have no semantic block map, and their master controls are hidden.
        kind, _, idx_s = block_id.rpartition("_")
        if kind not in ("double", "single") or not idx_s.isdigit():
            return "identity"
        idx = int(idx_s)
        if kind == "double":
            return "style_composition"
        if idx == 0:
            return "style_composition"
        if idx == 1:
            return "style_ident_overlap"
        if 2 <= idx <= 11:
            return "identity"
        if 12 <= idx <= 16:
            return "ident_details_overlap"
        return "details"

    def create_repair_studio_tab(self):
        """Per-block LoRA repair with side-by-side preview (Start-tab styled)."""
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.state import SliderState

        # Outer master scroll: the whole tab scrolls vertically when its
        # content exceeds the window height (e.g. when Res=768 blows up the
        # preview panel). The inner sliders panel has its own scroll too —
        # mousewheel hand-off is handled by <Enter>/<Leave> bind_all swapping
        # on each scrollable canvas independently.
        frame = self._build_repair_outer_scroll(self.repair_studio_tab)

        # Bg_deep container — all cards pack into this.
        outer = tk.Frame(frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Model family (Klein 9B / Krea 2 / MiniMax H3), restored from last_used.
        _fam = "klein"
        try:
            if str(self.last_used.get("repair_family", "klein")) in ("krea2", "minimax"):
                _fam = str(self.last_used.get("repair_family"))
        except Exception:
            pass
        self.repair_family_var = tk.StringVar(value=_fam)
        # Engine + state — lazy
        self.repair_engine = None
        self.repair_state = (SliderState.default_krea2() if _fam == "krea2"
                             else SliderState.default_h3() if _fam == "minimax"
                             else SliderState.default_klein9b())
        self.repair_block_vars = {}   # block_id -> dict(primary_chk, primary_scale, primary_lbl, donor_chk, donor_scale, donor_lbl)
        self.repair_thumbnails = {}   # GC-safe ImageTk.PhotoImage refs
        self.repair_pil_images = {"baseline": None, "tweaked": None}  # raw PIL for resize-to-fit
        self._repair_preview_redraw_after = {"baseline": None, "tweaked": None}
        self.repair_profile_match = None  # dict payload from profile sidecar, or None
        self._repair_preview_after_id = None
        self._repair_preview_in_flight = False
        self._repair_preview_dirty = False
        self._repair_donor_loaded = False

        self._add_tab_banner(
            outer,
            "Repair Studio",
            "Tweak each block's contribution live with side-by-side preview. "
            "Optional donor LoRA blends in via rank concatenation. Save the repaired result as a new .safetensors. "
            "Turbo Preview is on by default for faster updates — turn it off if VRAM is tight.",
        )

        # Card 1: Setup (DiT, Primary, Donor, Preview params, Preset)
        setup_card = self._start_section_card(
            outer, "Setup",
            "Paths come from Preferences. Load the primary LoRA first; donor is optional. "
            "Changing prompt / seed / resolution triggers a fresh baseline render.",
        )
        setup_card.columnconfigure(1, weight=1)
        self._build_repair_top_controls(setup_card)

        # Profile-match info panel (populated when a matching Profiler sidecar
        # exists for the primary's content hash). Packs directly into outer so
        # pack_forget/pack(before=…) slots it cleanly between Status and Preview.
        self.repair_profile_frame = tk.Frame(outer, bg=COLORS["bg_surface"],
                                             highlightbackground=COLORS["accent"],
                                             highlightthickness=1, bd=0)
        # Deliberately not packed yet — shown only when a match is found.

        # Card 2: Preview
        preview_card = self._start_section_card(
            outer, "Preview",
            "Left side is the baseline (LoRA at its original strengths); right side is the tweaked render using your slider state.",
        )
        # Anchor so _render_repair_profile_panel can pack(before=…) into the right slot.
        self._repair_profile_anchor = preview_card.master.master
        self._build_repair_preview_panel(preview_card)

        # Card 3: Master Controls
        master_card = self._start_section_card(
            outer, "Master Controls",
            "Bulk-tune by category. Flip the target radio to switch between primary and donor. "
            "Category toggles next to the donor ones bulk on/off the donor's contribution per bucket.",
        )
        self._build_repair_master_controls(master_card)
        self._repair_master_container = master_card.master.master  # for hide/show by family

        # Card 4: Per-Block Sliders
        sliders_card = self._start_section_card(
            outer, "Per-Block Sliders",
            "Range ±3.0. Greyed-out rows are blocks the LoRA doesn't touch. "
            "Colour bands match the Profiler's 5-bucket scheme: blue Style+Comp, teal Style/ID, green Identity, olive ID/Detail, orange Details.",
        )
        self._build_repair_slider_panel(sliders_card)
        self._repair_sliders_container = sliders_card.master.master

        # Card 5: Actions
        actions_card = self._start_section_card(outer, "Actions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(fill=tk.X)
        ttk.Button(action_row, text="Save Repaired LoRA…",
                   command=self._save_repaired_lora_action, style="Primary.TButton").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Reset All Sliders",
                   command=self._reset_repair_sliders).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Reset Session (unload models)",
                   command=self._reset_repair_session).pack(side=tk.LEFT)
        tk.Button(action_row, text="Explore this in LoRA the Explorer \u2192",
                  font=(FONT_FAMILY, 10, "bold"),
                  fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
                  relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                  command=self._repair_explore_in_explorer).pack(side=tk.RIGHT)

        self._add_youtube_help_button(outer, "repair_studio")
        # Sync DiT-radio labels + Master Controls visibility for the restored family.
        self._apply_repair_family_ui()

    def _apply_repair_family_ui(self):
        """Relabel the DiT radios per family and hide Master Controls for the no-block-map
        families (Krea 2 + MiniMax H3 — the per-block sliders stay as the discovery
        instrument). `krea2` below means "any no-map family" (historical naming)."""
        fam = self.repair_family_var.get()
        krea2 = fam in ("krea2", "minimax")
        if fam == "minimax":
            # H3 has no DiT choice: base precision is auto-planned from free VRAM and the
            # Turbo LoRA (6-step) applies whenever it's set in Preferences.
            self._repair_dit_radio_a.configure(text="Auto (int8/NF4 by VRAM + Turbo LoRA)",
                                               state="disabled")
            # H3 has ONE base plan — a second radio labeled "—" reads as a broken control.
            self._repair_dit_radio_b.pack_forget()
        elif fam == "krea2":
            self._repair_dit_radio_a.configure(text="Turbo (8-step, default)", state="normal")
            self._repair_dit_radio_b.configure(text="RAW (slow, precise)", state="normal")
            if not self._repair_dit_radio_b.winfo_manager():
                self._repair_dit_radio_b.pack(side=tk.LEFT)
        else:
            self._repair_dit_radio_a.configure(text="Distilled (4-step, fast)", state="normal")
            self._repair_dit_radio_b.configure(text="Base (20-step, precise but slow)", state="normal")
            if not self._repair_dit_radio_b.winfo_manager():
                self._repair_dit_radio_b.pack(side=tk.LEFT)
        try:
            if krea2:
                self._repair_master_container.pack_forget()
            elif self._repair_master_container.winfo_manager() == "":
                self._repair_master_container.pack(fill=tk.X, padx=36, pady=(0, 16),
                                                   before=self._repair_sliders_container)
        except Exception:
            pass
        # Turbo Preview (activation cache) is Klein-only. Krea 2 and H3 always full-forward:
        # the per-step resume compounds across a multi-step chain, and on H3 it was MEASURED
        # (18 Aug, real 33B): a resumed render retains ~6% of a block tweak's visible effect —
        # a preview that lies about the bake. forward_cached stays on the model as the
        # building block for a future multi-step-aware cache.
        try:
            if krea2:
                self._repair_turbo_chk.pack_forget()
            elif self._repair_turbo_chk.winfo_manager() == "":
                self._repair_turbo_chk.pack(side=tk.RIGHT)
        except Exception:
            pass
        # Preset list is family-dependent (Krea 2 has no semantic block map → Reset All only).
        try:
            self._refresh_repair_preset_combo()
            if getattr(self, "repair_preset_var", None) is not None and \
                    self.repair_preset_var.get() not in self._repair_preset_list():
                self.repair_preset_var.set("")
        except Exception:
            pass
        # Reference Strength is a Klein edit-conditioning knob; Krea 2's vision-path reference
        # has no strength dial, so hide it there (the MP cap still applies).
        for _w in (getattr(self, "_repair_ref_strength_label", None),
                   getattr(self, "_repair_ref_strength_entry", None)):
            try:
                if _w is None:
                    continue
                if krea2:
                    _w.pack_forget()
                elif _w.winfo_manager() == "":
                    _w.pack(side=tk.LEFT, **({"padx": (0, 2)} if _w is self._repair_ref_strength_label else {}))
            except Exception:
                pass
        # H3's engine has no reference path at all (r2v conditioning is out of the workbench's
        # scope) — a visible row the engine ignores is a lie, and editing it forced a re-render
        # that changed nothing. The WHOLE row hides under MiniMax; Klein and Krea 2 keep it.
        for _w in (getattr(self, "_repair_ref_label", None),
                   getattr(self, "_repair_ref_entry", None),
                   getattr(self, "_repair_ref_params", None)):
            try:
                if _w is None:
                    continue
                if fam == "minimax":
                    _w.grid_remove()
                else:
                    _w.grid()
            except Exception:
                pass
        if fam == "minimax" and self.repair_ref_path_var.get().strip():
            # A path carried over from a Klein session must not sit invisibly in the state.
            self.repair_ref_path_var.set("")
            self.repair_state.ref_image_path = ""

    def _on_repair_family_changed(self):
        """Family toggle: reset any loaded session (engine type changes), reset the slider
        state + rebuild the panel for the new block layout, relabel the DiT toggle, hide/show
        Master Controls, and persist the choice."""
        from fizgig.repair_studio.state import SliderState
        try:
            self._reset_repair_session()
        except Exception:
            pass
        fam = self.repair_family_var.get()
        self.repair_state = (SliderState.default_krea2() if fam == "krea2"
                             else SliderState.default_h3() if fam == "minimax"
                             else SliderState.default_klein9b())
        # 512 default for Klein/Krea 2 (keeps the Turbo Preview activation cache VRAM-feasible);
        # H3 previews at 768 — its native canvas, rendered as a 22-frame clip's middle frame.
        self.repair_res_var.set("768" if fam == "minimax" else "512")
        self._build_repair_slider_panel(self._repair_sliders_parent)
        self._apply_repair_family_ui()
        try:
            self.last_used["repair_family"] = fam
            self._save_last_used_paths()
        except Exception:
            pass
        from fizgig.networks.lora import FAMILY_DISPLAY_NAMES as _FDN
        self.repair_status_var.set(
            f"Switched to {_FDN.get(fam, fam)}. Set a LoRA path and click Start.")

    def _build_repair_outer_scroll(self, tab):
        """Wrap the Repair Studio tab in a vertical scrolling canvas. Returns the
        inner Frame into which all tab content should be placed."""
        outer_canvas = tk.Canvas(tab, highlightthickness=0, bg=COLORS["bg_deep"])
        outer_scroll = ttk.Scrollbar(tab, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=outer_scroll.set)
        outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        outer_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        inner = tk.Frame(outer_canvas, bg=COLORS["bg_deep"])
        inner_id = outer_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            outer_canvas.configure(scrollregion=outer_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            outer_canvas.itemconfigure(inner_id, width=e.width)
            # Window resize also drives preview re-render (preview scales with width).
            self._schedule_repair_preview_redraws()
        outer_canvas.bind("<Configure>", _on_canvas_config)

        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.

        self.repair_outer_canvas = outer_canvas
        return inner

    def _schedule_repair_preview_redraws(self):
        """Debounced redraw trigger for both preview sides. Called on window resize."""
        for which in ("baseline", "tweaked"):
            if self._repair_preview_redraw_after.get(which) is not None:
                try:
                    self.master.after_cancel(self._repair_preview_redraw_after[which])
                except Exception:
                    pass
            self._repair_preview_redraw_after[which] = self.master.after(
                60, lambda w=which: self._repair_redraw_preview(w))

    def _repair_redraw_preview(self, which: str):
        """Rescale stored PIL image to fit current holder box, preserving aspect ratio."""
        pil = self.repair_pil_images.get(which)
        if pil is None:
            return
        if which == "baseline":
            label = self.repair_baseline_label
            holder = self.repair_base_holder
        else:
            label = self.repair_tweaked_label
            holder = self.repair_tweaked_holder
        # Ensure layout is current before we query sizes (first redraw can
        # fire before Tk has finished laying things out).
        try:
            holder.update_idletasks()
        except Exception:
            pass
        # Floor of 256 so pre-layout reads still produce a usable image.
        box_w = max(256, holder.winfo_width() - 8)
        box_h = max(256, holder.winfo_height() - 8)
        src_w, src_h = pil.size
        scale = min(box_w / src_w, box_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        from PIL import Image as _PILImage
        img = pil.resize((new_w, new_h), _PILImage.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.repair_thumbnails[which] = photo  # keep ref so Tk doesn't GC it
        label.configure(image=photo, text="")

    def _build_repair_top_controls(self, parent):
        r = 0
        # Model family selector (Klein 9B / Krea 2). Krea 2 swaps the engine + model paths,
        # rebuilds the sliders for its block layout, and hides Master Controls (no block map).
        ttk.Label(parent, text="Model:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        fam_frame = ttk.Frame(parent)
        fam_frame.grid(row=r, column=1, columnspan=3, sticky=tk.W, padx=4, pady=2)
        ttk.Radiobutton(fam_frame, text="Klein 9B", variable=self.repair_family_var, value="klein",
                        style="Surface.TRadiobutton", command=self._on_repair_family_changed).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(fam_frame, text="Krea 2", variable=self.repair_family_var, value="krea2",
                        style="Surface.TRadiobutton", command=self._on_repair_family_changed).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(fam_frame, text="MiniMax H3", variable=self.repair_family_var, value="minimax",
                        style="Surface.TRadiobutton", command=self._on_repair_family_changed).pack(side=tk.LEFT)
        r += 1
        # DiT toggle (relabelled per family — Distilled/Base for Klein, Turbo/RAW for Krea 2)
        ttk.Label(parent, text="DiT:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_dit_choice_var = tk.StringVar(value="distilled")
        choice_frame = ttk.Frame(parent)
        choice_frame.grid(row=r, column=1, columnspan=3, sticky=tk.W, padx=4, pady=2)
        self._repair_dit_radio_a = ttk.Radiobutton(choice_frame, text="Distilled (4-step, fast)",
                        variable=self.repair_dit_choice_var, value="distilled",
                        style="Surface.TRadiobutton")
        self._repair_dit_radio_a.pack(side=tk.LEFT, padx=(0, 12))
        self._repair_dit_radio_b = ttk.Radiobutton(choice_frame, text="Base (20-step, precise but slow)",
                        variable=self.repair_dit_choice_var, value="base",
                        style="Surface.TRadiobutton")
        self._repair_dit_radio_b.pack(side=tk.LEFT)
        r += 1

        # Primary LoRA
        ttk.Label(parent, text="Primary LoRA:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_primary_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.repair_primary_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(parent, text="Browse",
                   command=self._browse_and_load_primary).grid(
            row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        r += 1

        # Donor LoRA
        ttk.Label(parent, text="Donor LoRA (optional):").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_donor_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.repair_donor_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        donor_btn_frame = ttk.Frame(parent)
        donor_btn_frame.grid(row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        ttk.Button(donor_btn_frame, text="Browse",
                   command=self._browse_and_load_donor).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(donor_btn_frame, text="Unload Donor",
                   command=self._unload_repair_donor).pack(side=tk.LEFT)
        r += 1

        # Prompt row
        ttk.Label(parent, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_prompt_var = tk.StringVar(value="")
        prompt_entry = ttk.Entry(parent, textvariable=self.repair_prompt_var)
        prompt_entry.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        self.repair_prompt_var.trace_add("write", lambda *_: self._repair_mark_update_needed())
        params_frame = ttk.Frame(parent)
        params_frame.grid(row=r, column=2, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(params_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 4))
        self.repair_seed_var = tk.StringVar(value="42")
        seed_entry = ttk.Entry(params_frame, textvariable=self.repair_seed_var, width=10)
        seed_entry.pack(side=tk.LEFT, padx=(0, 2))
        self.repair_seed_var.trace_add("write", lambda *_: self._repair_mark_update_needed())
        seed_entry.bind("<Return>", self._repair_seed_committed)
        tk.Button(params_frame, text="\u21bb", font=(FONT_FAMILY, 9),
                  bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                  activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, padx=4, pady=0, cursor="hand2",
                  command=self._repair_randomize_seed
                  ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(params_frame, text="Res:").pack(side=tk.LEFT, padx=(0, 4))
        self.repair_res_var = tk.StringVar(value="512")
        res_combo = ttk.Combobox(params_frame, textvariable=self.repair_res_var,
                                 values=["256", "384", "512", "768"], state="readonly", width=6)
        res_combo.pack(side=tk.LEFT)
        res_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preview_param_changed())
        # Turbo Preview toggle (Klein activation cache). Hidden in Krea 2 mode — krea's 8-step
        # denoise makes the per-step cache too lossy, so Krea 2 always does the full forward.
        self.repair_turbo_var = tk.BooleanVar(value=True)
        self._repair_turbo_chk = ttk.Checkbutton(params_frame, text="Turbo Preview",
                                     variable=self.repair_turbo_var,
                                     command=self._on_turbo_toggled)
        self._repair_turbo_chk.pack(side=tk.RIGHT)
        r += 1

        # Reference image row (Klein is an edit model — condition the preview on a
        # real image). Path + MP cap (downscale-only) + strength (1.0 stock,
        # ~0.85 Klein sweet spot, 0 = off). Carried in SliderState so it survives
        # the Explorer ↔ Repair handover.
        self._repair_ref_label = ttk.Label(parent, text="Reference:")
        self._repair_ref_label.grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_ref_path_var = tk.StringVar(value="")
        self._repair_ref_entry = ttk.Entry(parent, textvariable=self.repair_ref_path_var,
                                           state="readonly")
        self._repair_ref_entry.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        ref_params = ttk.Frame(parent)
        self._repair_ref_params = ref_params
        ref_params.grid(row=r, column=2, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(ref_params, text="Browse", command=self._browse_repair_ref).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(ref_params, text="Clear", command=self._clear_repair_ref).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(ref_params, text="MP:").pack(side=tk.LEFT, padx=(0, 2))
        self.repair_ref_mp_var = tk.StringVar(value="1.0")
        mp_combo = ttk.Combobox(ref_params, textvariable=self.repair_ref_mp_var,
                                values=["0.25", "0.5", "1.0", "2.0"], state="readonly", width=5)
        mp_combo.pack(side=tk.LEFT, padx=(0, 10))
        mp_combo.bind("<<ComboboxSelected>>", lambda e: self._on_repair_ref_changed())
        self._repair_ref_strength_label = ttk.Label(ref_params, text="Strength:")
        self._repair_ref_strength_label.pack(side=tk.LEFT, padx=(0, 2))
        self.repair_ref_strength_var = tk.StringVar(value="1.0")
        self._repair_ref_strength_entry = ttk.Entry(ref_params, textvariable=self.repair_ref_strength_var, width=6)
        self._repair_ref_strength_entry.pack(side=tk.LEFT)
        self.repair_ref_strength_var.trace_add("write", lambda *_: self._on_repair_ref_changed())
        r += 1

        # Preset row
        ttk.Label(parent, text="Preset:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_preset_var = tk.StringVar()
        self.repair_preset_combo = ttk.Combobox(parent, textvariable=self.repair_preset_var,
                                                values=self._repair_preset_list(), state="readonly")
        self.repair_preset_combo.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        self.repair_preset_combo.bind("<<ComboboxSelected>>",
                                      lambda e: self._load_repair_preset(self.repair_preset_var.get()))
        ttk.Button(parent, text="Save Preset…",
                   command=self._save_repair_preset).grid(row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        r += 1

        # Status + Start button row
        status_row = tk.Frame(parent, bg=COLORS["bg_surface"])
        status_row.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=(6, 0))
        self.repair_status_var = tk.StringVar(value="Set a LoRA path and prompt, then click Start.")
        tk.Label(status_row, textvariable=self.repair_status_var,
                 font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        self._repair_start_btn = tk.Button(
            status_row, text="Start", font=(FONT_FAMILY, 11, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=24, pady=6, cursor="hand2",
            command=self._repair_start)
        self._repair_start_btn.pack(side=tk.RIGHT)
        # Reset All up here too — the Actions card has one, but it sits below the full slider
        # panel, which on a 50-block family is a long scroll away from where you tweak.
        ttk.Button(status_row, text="Reset All Sliders",
                   command=self._reset_repair_sliders).pack(side=tk.RIGHT, padx=(0, 12))
        # The pop-out also opens by clicking either preview image, but nothing on screen SAYS
        # that — a named, coloured button plus the caption under the previews is how anyone
        # finds the compare view and its metrics (Peter, 22 Aug: plain ttk wasn't enough).
        _cmp_btn = tk.Button(
            status_row, text="⧉ Compare + Metrics", font=(FONT_FAMILY, 10, "bold"),
            fg="#FFFFFF", bg="#3B6FA0", activeforeground="#FFFFFF", activebackground="#2E5780",
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
            command=self._repair_popout_preview)
        _cmp_btn.pack(side=tk.RIGHT, padx=(0, 12))
        ToolTip(_cmp_btn, "Full-size side-by-side of baseline vs tweaked, with likeness and "
                          "quality metrics. Clicking either preview image opens it too.")
        # Render progress. H3 and Krea 2 report real denoising steps (determinate); Klein's
        # denoise loop has no hook, so the bar sweeps as a marquee there — and everywhere
        # until the first step lands, so model loads and TE encodes still show life.
        self._repair_progress = ttk.Progressbar(status_row, mode="indeterminate", length=200,
                                                style="Green.Horizontal.TProgressbar")
        self._repair_progress_det = False
        r += 1

    def _build_repair_preview_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Baseline (LoRA at default 1.0)",
                  font=(FONT_FAMILY, 9, "bold")).grid(row=0, column=0, padx=4, pady=(2, 0))
        ttk.Label(parent, text="Tweaked (current sliders)",
                  font=(FONT_FAMILY, 9, "bold")).grid(row=0, column=1, padx=4, pady=(2, 0))

        # Pixel-size holders per side. pack_propagate(False) stops the holder
        # from shrinking to the label's text size — without this, the frames
        # collapse to tiny squares because the default "(no preview yet)" text
        # is small and the packed label would otherwise dictate the frame size.
        # sticky="nsew" + columnconfigure(weight=1) lets the holder stretch
        # wider when the window is enlarged; pack_propagate(False) keeps the
        # minimum height at 512.
        base_holder = tk.Frame(parent, width=512, height=512, bg="#1c1c1c",
                               highlightthickness=0)
        base_holder.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        base_holder.pack_propagate(False)
        tweaked_holder = tk.Frame(parent, width=512, height=512, bg="#1c1c1c",
                                  highlightthickness=0)
        tweaked_holder.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        tweaked_holder.pack_propagate(False)

        self.repair_baseline_label = ttk.Label(base_holder, text="(no baseline yet)",
                                               anchor=tk.CENTER, background="#1c1c1c",
                                               cursor="hand2")
        self.repair_baseline_label.pack(fill=tk.BOTH, expand=True)
        self.repair_tweaked_label = ttk.Label(tweaked_holder, text="(no preview yet)",
                                              anchor=tk.CENTER, background="#1c1c1c",
                                              cursor="hand2")
        self.repair_tweaked_label.pack(fill=tk.BOTH, expand=True)
        # Either image opens the compare pop-out — clicking the baseline should not be a dead
        # zone when the tweaked side isn't.
        self.repair_baseline_label.bind("<Button-1>", lambda e: self._repair_popout_preview())
        self.repair_tweaked_label.bind("<Button-1>", lambda e: self._repair_popout_preview())
        # Spell the click affordance out — cursor changes alone weren't discoverable.
        ttk.Label(parent,
                  text="🔍 Click either image for the full-size side-by-side compare with "
                       "likeness + quality metrics",
                  font=(FONT_FAMILY, 9), foreground=COLORS["text_secondary"],
                  ).grid(row=2, column=0, columnspan=2, pady=(0, 4))
        self.repair_base_holder = base_holder
        self.repair_tweaked_holder = tweaked_holder
        self._repair_popout_window = None
        self._repair_popout_label = None
        self._repair_popout_tk_img = None
        # Metrics strip state: reference photo for likeness scoring (remembered via the
        # workbench table), chip labels, and a generation counter so a slow ArcFace pass
        # can never paint a stale result over a newer render's numbers.
        self.repair_metrics_ref_var = tk.StringVar(value="")
        self._repair_popout_metric_lbls = {}
        self._repair_metrics_gen = 0

        # Redraw on resize. Debounced so a drag doesn't spam Lanczos.
        def _mk_config_cb(which):
            def _cb(_e):
                if self._repair_preview_redraw_after.get(which) is not None:
                    try:
                        self.master.after_cancel(self._repair_preview_redraw_after[which])
                    except Exception:
                        pass
                self._repair_preview_redraw_after[which] = self.master.after(
                    60, lambda w=which: self._repair_redraw_preview(w))
            return _cb
        base_holder.bind("<Configure>", _mk_config_cb("baseline"))
        tweaked_holder.bind("<Configure>", _mk_config_cb("tweaked"))

    _REPAIR_MASTER_CATS = [
        ("style_composition", "Style+Comp"),
        ("style_ident_overlap", "Style/ID"),
        ("identity", "Identity"),
        ("ident_details_overlap", "ID/Detail"),
        ("details", "Details"),
    ]

    def _repair_quickset_buttons(self, parent, var, row, col_start, balance_cb=None):
        """Create [0] [1] [±] [⚖] quick-set buttons for a repair slider.

        Returns a list of button widgets (for greying in _refresh_block_slider_activity).
        balance_cb: optional callback for the balance button (sets complement on the other target).
        """
        btn_font = (FONT_FAMILY, 8)
        btn_kw = dict(bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                      activebackground=COLORS["bg_surface"],
                      activeforeground=COLORS["text_primary"],
                      relief="flat", bd=0, padx=2, pady=0, width=2, font=btn_font)
        b0 = tk.Button(parent, text="0", command=lambda: var.set(0.0), **btn_kw)
        b0.grid(row=row, column=col_start, padx=1, pady=1)
        b1 = tk.Button(parent, text="1", command=lambda: var.set(1.0), **btn_kw)
        b1.grid(row=row, column=col_start + 1, padx=1, pady=1)
        bn = tk.Button(parent, text="\u00b1",
                       command=lambda: var.set(-var.get() or 0.0), **btn_kw)
        bn.grid(row=row, column=col_start + 2, padx=1, pady=1)
        btns = [b0, b1, bn]
        if balance_cb is not None:
            bb = tk.Button(parent, text="\u2696", command=balance_cb, **btn_kw)
            bb.grid(row=row, column=col_start + 3, padx=1, pady=1)
            btns.append(bb)
        return btns

    def _build_repair_master_controls(self, parent):
        """Target radio + 5 category master sliders + 5 donor category toggles."""
        # State vars
        self.repair_master_target_var = tk.StringVar(value="primary")
        self.repair_master_strength_vars = {
            cat: tk.DoubleVar(value=1.0) for cat, _ in self._REPAIR_MASTER_CATS
        }
        self.repair_master_strength_labels = {}
        self.repair_donor_category_vars = {
            cat: tk.BooleanVar(value=False) for cat, _ in self._REPAIR_MASTER_CATS
        }
        # Suppression flag: when a master slider moves, we set N per-block vars.
        # Each per-block trace would otherwise fire _schedule_preview individually
        # (cheap but noisy in logs). Keep the trace firing — it updates state —
        # but mute the preview-schedule during bulk updates, then fire ONE
        # preview at the end.
        self._repair_master_mutating = False

        r = 0
        # Target radio
        target_frame = ttk.Frame(parent)
        target_frame.grid(row=r, column=0, columnspan=6, sticky=tk.W, padx=6, pady=(4, 6))
        ttk.Label(target_frame, text="Master sliders affect:",
                  font=(FONT_FAMILY, 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(target_frame, text="Primary",
                        variable=self.repair_master_target_var, value="primary",
                        command=self._on_master_target_changed,
                        style="Surface.TRadiobutton").pack(side=tk.LEFT, padx=(0, 8))
        self._repair_master_donor_radio = ttk.Radiobutton(
            target_frame, text="Donor", variable=self.repair_master_target_var, value="donor",
            command=self._on_master_target_changed, style="Surface.TRadiobutton")
        self._repair_master_donor_radio.pack(side=tk.LEFT)
        self._repair_master_donor_radio.state(["disabled"])  # enabled when donor loads
        r += 1

        # 5 category sliders
        parent.columnconfigure(1, weight=1)
        for cat, short in self._REPAIR_MASTER_CATS:
            color = self._REPAIR_CAT_COLOR[cat]
            tk.Label(parent, text=short, fg=color, bg=COLORS["bg_surface"],
                     width=11, anchor=tk.W,
                     font=(FONT_FAMILY, 9, "bold")).grid(
                row=r, column=0, sticky=tk.W, padx=(10, 4), pady=1)
            var = self.repair_master_strength_vars[cat]
            scale = ttk.Scale(parent, from_=-3.0, to=3.0, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=1)
            val_lbl = ttk.Label(parent, text="1.00", width=5, anchor=tk.E)
            val_lbl.grid(row=r, column=2, padx=(4, 4), pady=1)
            self.repair_master_strength_labels[cat] = val_lbl
            self._repair_quickset_buttons(parent, var, r, 3,
                balance_cb=lambda c=cat: self._repair_balance_master(c))

            def _mk_trace(_var, _lbl, _cat):
                def _cb(*_a):
                    v = float(_var.get())
                    _lbl.configure(text=f"{v:+.2f}" if v != 1.0 else "1.00")
                    self._on_master_strength_changed(_cat, v)
                return _cb
            var.trace_add("write", _mk_trace(var, val_lbl, cat))
            r += 1

        # Donor category toggles now live in the Per-Block Sliders card (created in create_repair_studio_tab)

    def _repair_balance_master(self, category: str):
        """Balance master slider: set the other target's category to 1.0 - current value."""
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        target = self.repair_master_target_var.get()
        current = self.repair_master_strength_vars[category].get()
        if current < 0 or current > 1.0:
            return
        complement = 1.0 - current
        # Set the other target's blocks
        other_key = "donor_strength" if target == "primary" else "primary_strength"
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category]
        self._repair_master_mutating = True
        try:
            for bid in affected:
                self.repair_block_vars[bid][other_key].set(complement)
                if other_key == "donor_strength":
                    self.repair_block_vars[bid]["donor_enabled"].set(True)
        finally:
            self._repair_master_mutating = False
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _repair_balance_block(self, block_id: str, source: str):
        """Balance a single block: set the other side to 1.0 - current value."""
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        v = self.repair_block_vars[block_id]
        if source == "primary":
            current = v["primary_strength"].get()
            if current < 0 or current > 1.0:
                return
            v["donor_strength"].set(1.0 - current)
            v["donor_enabled"].set(True)
        else:
            current = v["donor_strength"].get()
            if current < 0 or current > 1.0:
                return
            v["primary_strength"].set(1.0 - current)

    def _on_master_strength_changed(self, category: str, value: float):
        """Mirror master slider value to per-block strength vars for affected blocks."""
        # If the master var is being set programmatically (e.g. on target switch
        # to reflect current per-block values), skip the mirror — otherwise we'd
        # flatten the very diversity we're trying to display.
        if getattr(self, "_repair_master_mutating", False):
            return
        target = self.repair_master_target_var.get()
        # Only touch blocks the target LoRA actually contains: disabled Scales don't block
        # DoubleVar.set(), so bulk-setting greyed-out absent rows committed phantom
        # strengths into saved presets and ss_repair_studio_config — values that become
        # real when that preset is later applied to a full-model LoRA.
        present = None
        eng = getattr(self, "repair_engine", None)
        if eng is not None:
            present = (getattr(eng, "primary_block_ids", None) if target == "primary"
                       else getattr(eng, "donor_block_ids", None))
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category
                    and (present is None or bid in present)]
        if not affected:
            return
        # Bulk-set: each per-block strength trace will update state, but we
        # want ONE preview schedule for the whole batch. Mark mutating so
        # _on_block_changed can short-circuit the preview schedule.
        self._repair_master_mutating = True
        try:
            key = "primary_strength" if target == "primary" else "donor_strength"
            for bid in affected:
                self.repair_block_vars[bid][key].set(value)
        finally:
            self._repair_master_mutating = False
        # Fire one preview for the whole batch.
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _on_master_target_changed(self):
        """When target radio flips (Primary↔Donor), refresh master sliders to
        show the current average per-block strength for the new target. Without
        this, the master sliders would display stale values from the previous
        target and mislead the user."""
        if self._repair_is_krea2():
            return   # no-map families: master controls are hidden, nothing to refresh
        target = self.repair_master_target_var.get()
        key = "primary_strength" if target == "primary" else "donor_strength"
        self._repair_master_mutating = True
        try:
            for cat, _ in self._REPAIR_MASTER_CATS:
                affected = [bid for bid in self.repair_block_vars
                            if self._repair_category_for_block(bid) == cat]
                if not affected:
                    continue
                values = [float(self.repair_block_vars[bid][key].get()) for bid in affected]
                avg = sum(values) / len(values) if values else 1.0
                self.repair_master_strength_vars[cat].set(round(avg, 3))
        finally:
            self._repair_master_mutating = False

    def _on_donor_category_toggled(self, category: str):
        """Mirror donor category toggle to donor_enabled vars for affected blocks."""
        on = bool(self.repair_donor_category_vars[category].get())
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category]
        if not affected:
            return
        self._repair_master_mutating = True
        try:
            for bid in affected:
                self.repair_block_vars[bid]["donor_enabled"].set(on)
        finally:
            self._repair_master_mutating = False
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _build_repair_slider_panel(self, parent):
        # Rebuildable: clear any previous rows + their tk vars so a Klein<->Krea2 family
        # switch rebuilds the panel cleanly (the two families have different block ids).
        self._repair_sliders_parent = parent
        for child in parent.winfo_children():
            child.destroy()
        self.repair_block_vars = {}
        if getattr(self, "repair_family_var", None) is not None and self.repair_family_var.get() == "krea2":
            self._build_repair_slider_panel_krea2(parent)
            return
        if getattr(self, "repair_family_var", None) is not None and self.repair_family_var.get() == "minimax":
            self._build_repair_slider_panel_h3(parent)
            return
        # Scrollable canvas (vertical) holding two columns: double on left, single on right.
        # Bounded height (500px) so the panel stays compact inside the outer scroll
        # and the user can independently scroll all 32 rows without losing the preview.
        canvas = tk.Canvas(parent, highlightthickness=0, bg=COLORS["bg_surface"], height=500)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            canvas.itemconfigure(inner_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_config)

        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # Two balanced columns: doubles + singles 0-7 on left, singles 8-23 on right
        col_left = ttk.Frame(inner)
        col_left.grid(row=0, column=0, sticky=tk.NSEW, padx=4)
        col_right = ttk.Frame(inner)
        col_right.grid(row=0, column=1, sticky=tk.NSEW, padx=4)

        # Left column: double blocks, then single 0-7
        r = 0
        ttk.Label(col_left, text="Double Blocks", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(8):
            self._build_repair_block_row(col_left, f"double_{i}", r)
            r += 1
        ttk.Label(col_left, text="Single Blocks 0\u20137", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(8, 4), sticky=tk.W)
        r += 1
        for i in range(8):
            self._build_repair_block_row(col_left, f"single_{i}", r)
            r += 1

        # Right column: single 8-23
        r = 0
        ttk.Label(col_right, text="Single Blocks 8\u201323", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(8, 24):
            self._build_repair_block_row(col_right, f"single_{i}", r)
            r += 1

    def _build_repair_slider_panel_krea2(self, parent):
        """Krea 2 layout: 28 main blocks (0-13 left, 14-27 right) + the 4 txtfusion blocks.
        Generic per-block (no semantic bucket colouring \u2014 that map doesn't exist yet)."""
        canvas = tk.Canvas(parent, highlightthickness=0, bg=COLORS["bg_surface"], height=500)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(inner_id, width=e.width))

        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        col_left = ttk.Frame(inner)
        col_left.grid(row=0, column=0, sticky=tk.NSEW, padx=4)
        col_right = ttk.Frame(inner)
        col_right.grid(row=0, column=1, sticky=tk.NSEW, padx=4)

        r = 0
        ttk.Label(col_left, text="Main Blocks 0\u201313", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(14):
            self._build_repair_block_row(col_left, f"block_{i}", r)
            r += 1

        r = 0
        ttk.Label(col_right, text="Main Blocks 14\u201327", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(14, 28):
            self._build_repair_block_row(col_right, f"block_{i}", r)
            r += 1
        ttk.Label(col_right, text="Text-Fusion Blocks", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(8, 4), sticky=tk.W)
        r += 1
        for bid in ("txt_lw_0", "txt_lw_1", "txt_rf_0", "txt_rf_1"):
            self._build_repair_block_row(col_right, bid, r)
            r += 1

    def _build_repair_slider_panel_h3(self, parent):
        """MiniMax H3 layout: 50 main blocks (0-25 left, 26-49 right) + the 2 token-refiner
        blocks. Generic per-block (no semantic bucket colouring — that map doesn't exist yet;
        these sliders + the weight-only Profiler are the instrument to build it)."""
        canvas = tk.Canvas(parent, highlightthickness=0, bg=COLORS["bg_surface"], height=500)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(inner_id, width=e.width))

        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        col_left = ttk.Frame(inner)
        col_left.grid(row=0, column=0, sticky=tk.NSEW, padx=4)
        col_right = ttk.Frame(inner)
        col_right.grid(row=0, column=1, sticky=tk.NSEW, padx=4)

        r = 0
        ttk.Label(col_left, text="Blocks 0–25", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(26):
            self._build_repair_block_row(col_left, f"h3blk_{i}", r)
            r += 1

        r = 0
        ttk.Label(col_right, text="Blocks 26–49", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(26, 50):
            self._build_repair_block_row(col_right, f"h3blk_{i}", r)
            r += 1
        ttk.Label(col_right, text="Token Refiner", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(8, 4), sticky=tk.W)
        r += 1
        for bid in ("h3_rf_0", "h3_rf_1"):
            self._build_repair_block_row(col_right, bid, r)
            r += 1

    def _repair_block_display(self, block_id: str):
        """(label, colour, category_short_or_None) for a block row. Klein ids are
        category-coloured; Krea 2 (block_N / txt_*) and H3 (h3blk_N / h3_rf_N) ids are
        generic (no semantic bucket map yet) → neutral colour, no category tag."""
        if block_id.startswith("h3blk_"):
            return (f"Block {block_id.split('_')[1]}", COLORS["text_secondary"], None)
        if block_id.startswith("h3_rf_"):
            return (f"Refiner {block_id.split('_')[2]}", COLORS["text_secondary"], None)
        if block_id.startswith("block_"):
            return (f"Block {block_id.split('_')[1]}", COLORS["text_secondary"], None)
        if block_id.startswith("txt_"):
            _, kind, n = block_id.split("_")
            return (f"Txt {kind.upper()} {n}", COLORS["text_secondary"], None)
        cat = self._repair_category_for_block(block_id)
        kind, idx = block_id.split("_")
        return (f"{kind} {idx}", self._REPAIR_CAT_COLOR[cat], self._REPAIR_CAT_SHORT[cat])

    def _build_repair_block_row(self, parent, block_id: str, row: int):
        lbl_text, color, cat_short = self._repair_block_display(block_id)

        rowf = ttk.Frame(parent)
        rowf.grid(row=row, column=0, sticky=tk.EW, pady=1)
        rowf.columnconfigure(3, weight=1)

        # Primary checkbox + label + slider + value
        primary_enabled = tk.BooleanVar(value=True)
        primary_strength = tk.DoubleVar(value=1.0)
        donor_enabled = tk.BooleanVar(value=True)
        donor_strength = tk.DoubleVar(value=0.0)

        chk_p = ttk.Checkbutton(rowf, variable=primary_enabled,
                                command=lambda b=block_id: self._on_block_changed(b))
        chk_p.grid(row=0, column=0, padx=(2, 4))
        # Block label (category-coloured for Klein; neutral for Krea 2)
        lbl = tk.Label(rowf, text=lbl_text, fg=color, bg=COLORS["bg_surface"],
                       width=10, anchor=tk.W, font=(FONT_FAMILY, 9, "bold"))
        lbl.grid(row=0, column=1, padx=(0, 2))
        cat_lbl = None
        if cat_short:
            cat_lbl = tk.Label(rowf, text=f"[{cat_short}]", fg=color, bg=COLORS["bg_surface"],
                               width=11, anchor=tk.W, font=(FONT_FAMILY, 8))
            cat_lbl.grid(row=0, column=2, padx=(0, 4))

        scale_p = ttk.Scale(rowf, from_=-3.0, to=3.0, variable=primary_strength, orient=tk.HORIZONTAL)
        scale_p.grid(row=0, column=3, sticky=tk.EW, padx=2)

        val_lbl_p = ttk.Label(rowf, text="1.00", width=5, anchor=tk.E)
        val_lbl_p.grid(row=0, column=4, padx=(2, 2))
        btns_p = self._repair_quickset_buttons(rowf, primary_strength, 0, 5,
            balance_cb=lambda b=block_id: self._repair_balance_block(b, "primary"))

        # Donor row (hidden until donor is loaded)
        donor_rowf = ttk.Frame(rowf)
        donor_rowf.grid(row=1, column=0, columnspan=9, sticky=tk.EW, padx=(20, 0))
        donor_rowf.columnconfigure(2, weight=1)
        donor_rowf.grid_remove()
        chk_d = ttk.Checkbutton(donor_rowf, variable=donor_enabled,
                                command=lambda b=block_id: self._on_block_changed(b))
        chk_d.grid(row=0, column=0, padx=(2, 4))
        donor_tag_lbl = ttk.Label(donor_rowf, text="donor", foreground="#888",
                                  font=(FONT_FAMILY, 8, "italic"),
                                  width=11, anchor=tk.W)
        donor_tag_lbl.grid(row=0, column=1, padx=(0, 4))
        scale_d = ttk.Scale(donor_rowf, from_=-3.0, to=3.0, variable=donor_strength, orient=tk.HORIZONTAL)
        scale_d.grid(row=0, column=2, sticky=tk.EW, padx=2)
        val_lbl_d = ttk.Label(donor_rowf, text="1.00", width=5, anchor=tk.E)
        val_lbl_d.grid(row=0, column=3, padx=(2, 2))
        btns_d = self._repair_quickset_buttons(donor_rowf, donor_strength, 0, 4,
            balance_cb=lambda b=block_id: self._repair_balance_block(b, "donor"))

        # Bind variable traces to mirror into self.repair_state and live-update labels
        def _mk_strength_trace(var, lbl, bid, which):
            def _cb(*_a):
                v = float(var.get())
                lbl.configure(text=f"{v:+.2f}" if v != 1.0 else "1.00")
                bs = self.repair_state.blocks[bid]
                if which == "primary":
                    bs.primary_strength = v
                else:
                    bs.donor_strength = v
                self._on_block_changed(bid)
            return _cb

        def _mk_enabled_trace(var, bid, which):
            def _cb(*_a):
                bs = self.repair_state.blocks[bid]
                if which == "primary":
                    bs.primary_enabled = bool(var.get())
                else:
                    bs.donor_enabled = bool(var.get())
                self._on_block_changed(bid)
            return _cb

        primary_strength.trace_add("write", _mk_strength_trace(primary_strength, val_lbl_p, block_id, "primary"))
        donor_strength.trace_add("write", _mk_strength_trace(donor_strength, val_lbl_d, block_id, "donor"))
        primary_enabled.trace_add("write", _mk_enabled_trace(primary_enabled, block_id, "primary"))
        donor_enabled.trace_add("write", _mk_enabled_trace(donor_enabled, block_id, "donor"))

        self.repair_block_vars[block_id] = {
            # StringVars / BooleanVars
            "primary_enabled": primary_enabled,
            "primary_strength": primary_strength,
            "donor_enabled": donor_enabled,
            "donor_strength": donor_strength,
            # value readouts
            "primary_lbl": val_lbl_p,
            "donor_lbl": val_lbl_d,
            # row frames
            "donor_rowf": donor_rowf,
            # widget handles (for _refresh_block_slider_activity greying)
            "chk_p": chk_p,
            "scale_p": scale_p,
            "block_lbl": lbl,
            "cat_lbl": cat_lbl,
            "chk_d": chk_d,
            "scale_d": scale_d,
            "donor_tag_lbl": donor_tag_lbl,
            # quick-set buttons (for greying)
            "btns_p": btns_p,
            "btns_d": btns_d,
            # category color (for restore after greying)
            "cat_color": color,
        }

    # ------------------------------------------------------------
    # Repair Studio actions
    # ------------------------------------------------------------

    def _pref_initialdir(self, pref_key: str) -> str:
        """Return the preferred starting folder for a Browse dialog, or "" to let
        Tk use the last-used folder. Reads the prefs var named by pref_key (e.g.
        'input_lora_dir', 'input_ref_dir', 'lora_output_dir') and validates it."""
        try:
            d = self.prefs_vars[pref_key].get().strip()
        except (KeyError, AttributeError):
            return ""
        return d if d and os.path.isdir(d) else ""

    def _lora_initialdir(self) -> str:
        """Start folder for every LoRA-loading Browse dialog: the input_lora_dir pref
        when set, else wherever trained LoRAs are written. Without the fallback a fresh
        session's dialog opens in the process cwd — on a pod that's the git clone, which
        contains no .safetensors and made the picker look broken (found on RunPod)."""
        d = self._pref_initialdir("input_lora_dir")
        if d:
            return d
        out = ""
        try:
            if hasattr(self, "entries") and "LORA_OUTPUT_DIR" in self.entries:
                out = self.entries["LORA_OUTPUT_DIR"].get().strip()
        except Exception:
            out = ""
        out = out or (self.settings.get("LORA_OUTPUT_DIR", "") or "")
        return out if out and os.path.isdir(out) else ""

    def _browse_repair_lora(self, var):
        filepath = filedialog.askopenfilename(
            title="Select LoRA file",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
            initialdir=self._lora_initialdir(),
        )
        if filepath:
            var.set(filepath)

    def _repair_engine_plan(self):
        """Main-thread half of the engine load: validate paths (messageboxes live here),
        construct the right engine class, and return the ensure_pipeline kwargs for a worker
        thread to run. Returns {} when the pipeline is already loaded (nothing to do) and
        None on a validation failure (already shown to the user). Split this way because
        ensure_pipeline loads a 10-20+ GB DiT — running it on the Tk thread froze the whole
        GUI for the duration (Peter hit it loading a Krea 2 LoRA)."""
        if self.repair_engine is not None and self.repair_engine.pipeline is not None and self.repair_engine.pipeline.is_loaded:
            return {}

        if self.repair_family_var.get() == "krea2":
            return self._repair_engine_plan_krea2()
        if self.repair_family_var.get() == "minimax":
            return self._repair_engine_plan_h3()

        dit_choice = self.repair_dit_choice_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        if not dit_path or not os.path.exists(dit_path):
            messagebox.showerror("Error", f"{dit_choice.capitalize()} DiT path not set or not found.\nConfigure on Preferences tab.")
            return False
        if not vae_path or not os.path.exists(vae_path):
            messagebox.showerror("Error", "VAE path not set or not found.\nConfigure on Preferences tab.")
            return False
        if not te_path or not os.path.exists(te_path):
            messagebox.showerror("Error", "Text encoder path not set or not found.\nConfigure on Preferences tab.")
            return False

        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.engine import RepairEngine

        if self.repair_engine is None:
            self.repair_engine = RepairEngine()
        self.repair_engine._turbo_enabled = self.repair_turbo_var.get()

        # Auto-detect fp8 + model_version from filename, mirroring profiler.
        dit_basename = os.path.basename(dit_path).lower()
        model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
        is_fp8_model = "fp8" in dit_basename
        self.repair_status_var.set(f"Loading models ({model_version})…")
        return dict(
            dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            model_version=model_version, device="cuda",
            fp8_scaled=False if is_fp8_model else True,
            blocks_to_swap=self._get_inference_blocks_to_swap(),
            int8=self._get_inference_int8(),
        )

    def _repair_engine_plan_krea2(self):
        """Lazy-load the Krea 2 Repair engine. Turbo (8-step, default) or RAW (slow). The DiT
        radio's distilled/base values map to turbo/raw here."""
        dit_choice = self.repair_dit_choice_var.get()  # 'distilled'->turbo, 'base'->raw
        is_raw = (dit_choice == "base")
        dit_key = "krea2_raw_dit" if is_raw else "krea2_turbo_dit"
        dit_path = self.prefs_vars.get(dit_key, tk.StringVar()).get()
        vae_path = self.prefs_vars.get("krea2_vae", tk.StringVar()).get()
        te_path = self.prefs_vars.get("krea2_text_encoder", tk.StringVar()).get()
        for label, p in ((f"Krea 2 {'RAW' if is_raw else 'Turbo'} DiT", dit_path),
                         ("Qwen-Image VAE", vae_path), ("Qwen3-VL TE (bf16)", te_path)):
            if not p or not os.path.exists(p):
                messagebox.showerror("Error", f"{label} path not set or not found.\nConfigure on Preferences tab.")
                return False

        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.krea2_engine import Krea2RepairEngine
        if self.repair_engine is None or not isinstance(self.repair_engine, Krea2RepairEngine):
            self.repair_engine = Krea2RepairEngine()
        self.repair_status_var.set(f"Loading Krea 2 models ({'RAW' if is_raw else 'Turbo'})…")
        return dict(
            turbo_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            device="cuda", model_kind="raw" if is_raw else "turbo",
            blocks_to_swap=self._auto_krea2_inference_blocks_swap(),
            int8=self._get_inference_int8())

    def _repair_engine_plan_h3(self):
        """Lazy-load the MiniMax H3 Repair engine. No DiT choice: base precision is planned
        from free VRAM inside the engine (int8 on big cards, NF4-of-pruned otherwise, never
        swapped), and the Turbo LoRA (6-step @ 75%) applies whenever it's set in Preferences.
        Previews render a 22-frame 768x768 clip and show its middle frame."""
        dit_path = self.prefs_vars.get("minimax_dit", tk.StringVar()).get()
        vae_path = self.prefs_vars.get("minimax_vae", tk.StringVar()).get()
        te_path = self.prefs_vars.get("minimax_text_encoder", tk.StringVar()).get()
        for label, p in (("MiniMax H3 DiT", dit_path), ("MiniMax H3 video VAE", vae_path),
                         ("Qwen3-VL-32B text encoder", te_path)):
            if not p or not os.path.exists(p):
                messagebox.showerror("Error", f"{label} path not set or not found.\nConfigure on Preferences tab.")
                return False
        turbo_path = self.prefs_vars.get("minimax_turbo_lora", tk.StringVar()).get().strip()
        cache_dir = self.prefs_vars.get("cache_dir", tk.StringVar()).get().strip()
        te_cache = os.path.join(cache_dir, "te_prompts") if cache_dir else ""

        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.h3_engine import H3RepairEngine
        if self.repair_engine is None or not isinstance(self.repair_engine, H3RepairEngine):
            self.repair_engine = H3RepairEngine()
        # _turbo_enabled stays False: the activation-cache resume was measured to under-apply
        # tweaks ~16x on H3 (see _apply_repair_family_ui) — previews always full-forward.
        self.repair_status_var.set("Loading MiniMax H3 (the 33B base takes a minute)…")
        return dict(
            dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            device="cuda", turbo_lora_path=turbo_path,
            turbo_lora_strength=0.75, te_cache_dir=te_cache)

    # ------------------------------------------------------------------
    # LoRA Royale — render every epoch on one seed, crossfade to the sweet spot
    # ------------------------------------------------------------------
    def create_lora_royale_tab(self):
        self.royale_engine = None
        self._royale_checkpoints = []
        self._royale_images = []      # [(label, full-res PIL)]
        self._royale_thumbs = []      # keep ImageTk refs alive
        self._royale_preview_imgtk = None
        self._royale_rendering = False
        self._royale_render_gen = 0   # bumps each render so epoch-ref temp files get fresh paths
        self._royale_paths = {}       # label -> checkpoint path (for promote)
        self._royale_scores = {}      # label -> likeness cosine (Phase 3)
        self._royale_best_label = None
        self._royale_scoring = False
        self._royale_exporting = False
        self._royale_traveling = False
        self._royale_pt_running = False
        self._royale_lora_running = False

        frame, _canvas = self.create_scrollable_frame(self.lora_royale_tab)
        outer = tk.Frame(frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)
        self._add_tab_banner(outer, "LoRA Royale",
                             "Render every epoch of a training run on one seed, then crossfade between them to find the sweet spot.")

        # Model family selector. Klein renders on the Distilled 4-step; Krea 2 on the fp8 Turbo
        # 8-step. Seed/prompt/strength travel work for both; Krea 2 travel morphs come from the
        # seed slerp / prompt interpolation with the vision-path image as the per-frame anchor
        # (no Klein reference-latent chaining).
        _rfam = str(self.last_used.get("royale_family", "klein"))
        if _rfam not in ("klein", "krea2", "minimax"):
            _rfam = "klein"
        self.royale_family_var = tk.StringVar(value=_rfam)
        rfam_card = self._start_section_card(
            outer, "Model Family",
            "Klein 9B (Distilled previews), Krea 2 (fp8 Turbo previews) or MiniMax H3 (22-frame "
            "clip previews, middle frame shown — slower per epoch). Epoch comparison works for "
            "all three; the travel modes are Klein and Krea 2.")
        _rf = tk.Frame(rfam_card, bg=COLORS["bg_surface"])
        _rf.pack(anchor=tk.W)
        ttk.Radiobutton(_rf, text="Klein 9B", variable=self.royale_family_var, value="klein",
                        command=self._on_royale_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_rf, text="Krea 2", variable=self.royale_family_var, value="krea2",
                        command=self._on_royale_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_rf, text="MiniMax H3", variable=self.royale_family_var, value="minimax",
                        command=self._on_royale_family_changed).pack(side=tk.LEFT)

        setup = self._start_section_card(outer, "Setup",
                                         "Point at a training output folder. Renders use the Distilled 4-step model.")
        setup.columnconfigure(1, weight=1)
        _sbg = COLORS["bg_surface"]
        r = 0
        # Source: a training folder (compare epochs) or a single LoRA file.
        ttk.Label(setup, text="Source:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_mode_var = tk.StringVar(value=self.last_used.get("royale_mode", "folder"))
        _mr = tk.Frame(setup, bg=_sbg); _mr.grid(row=r, column=1, columnspan=2, sticky=tk.W, pady=4)
        ttk.Radiobutton(_mr, text="Training folder (compare epochs)", value="folder",
                        variable=self.royale_mode_var, command=self._royale_apply_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(_mr, text="Single LoRA", value="single",
                        variable=self.royale_mode_var, command=self._royale_apply_mode).pack(side=tk.LEFT, padx=(14, 0))
        r += 1

        self._royale_folder_lbl = ttk.Label(setup, text="Checkpoint folder:")
        self._royale_folder_lbl.grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_folder_var = tk.StringVar(
            value=self.last_used.get("royale_folder", "") or self.settings.get("LORA_OUTPUT_DIR", ""))
        self._royale_folder_row = tk.Frame(setup, bg=_sbg)
        self._royale_folder_row.grid(row=r, column=1, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Entry(self._royale_folder_row, textvariable=self.royale_folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self._royale_folder_row, text="Browse…", command=self._royale_browse_folder).pack(side=tk.LEFT, padx=(6, 0))
        r += 1

        self._royale_single_lbl = ttk.Label(setup, text="LoRA file:")
        self._royale_single_lbl.grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_single_var = tk.StringVar(value=self.last_used.get("royale_single", ""))
        self._royale_single_row = tk.Frame(setup, bg=_sbg)
        self._royale_single_row.grid(row=r, column=1, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Entry(self._royale_single_row, textvariable=self.royale_single_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self._royale_single_row, text="Browse…", command=self._royale_browse_single).pack(side=tk.LEFT, padx=(6, 0))
        r += 1

        self.royale_scan_var = tk.StringVar(value="")
        self._royale_scan_lbl = tk.Label(setup, textvariable=self.royale_scan_var, font=(FONT_FAMILY, 9, "italic"),
                                         fg=COLORS["accent"], bg=_sbg)
        self._royale_scan_lbl.grid(row=r, column=1, columnspan=2, sticky=tk.W)
        self.royale_folder_var.trace_add("write", lambda *a: (self._royale_scan(), self._save_last_used_paths()))
        self.royale_single_var.trace_add("write", lambda *a: self._save_last_used_paths())
        r += 1

        ttk.Label(setup, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_prompt_var = tk.StringVar(value=self.last_used.get("royale_prompt", ""))
        ttk.Entry(setup, textvariable=self.royale_prompt_var).grid(row=r, column=1, columnspan=2, sticky=tk.EW, pady=4)
        r += 1

        ttk.Label(setup, text="Seed:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        _pr = tk.Frame(setup, bg=_sbg); _pr.grid(row=r, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.royale_seed_var = tk.StringVar(value=self.last_used.get("royale_seed", "42"))
        ttk.Entry(_pr, textvariable=self.royale_seed_var, width=10).pack(side=tk.LEFT)
        tk.Label(_pr, text="shared by the crossfade, prompt travel and strength travel",
                 bg=_sbg, fg=COLORS["text_muted"], font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(10, 0))
        # Size and Max renders USED to sit here, beside the seed. They only ever drove the
        # crossfade - every travel card and the comparison sheet carry their own - so in a card
        # called Setup they read as global and were silently ignored by four of the six modes.
        # They now live in Crossfade, next to the thing they size.
        r += 1

        ttk.Label(setup, text="Reference:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_ref_var = tk.StringVar(value=self.last_used.get("royale_ref", ""))
        _rr = tk.Frame(setup, bg=_sbg); _rr.grid(row=r, column=1, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Entry(_rr, textvariable=self.royale_ref_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(_rr, text="Browse…", command=self._royale_browse_ref).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(_rr, text="Clear", command=lambda: self.royale_ref_var.set("")).pack(side=tk.LEFT, padx=(4, 0))
        self._royale_ref_strength_lbl = tk.Label(_rr, text="Strength", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_ref_strength_lbl.pack(side=tk.LEFT, padx=(8, 3))
        self.royale_ref_strength_var = tk.StringVar(value=self.last_used.get("royale_ref_strength", "1.0"))
        _rse = ttk.Entry(_rr, textvariable=self.royale_ref_strength_var, width=5)
        _rse.pack(side=tk.LEFT)
        self._royale_ref_strength_entry = _rse
        ToolTip(_rse, "How strongly the reference anchors each epoch render.\n"
                      "1.0 = full edit-model anchor (the reference holds the composition while each epoch's "
                      "LoRA renders the prompt), lower lets the prompt vary more, 0 = off.")
        r += 1

        # Remember the render inputs across sessions. Size and Max renders are bound with the
        # Crossfade card below, where they are now built.
        for _v in (self.royale_prompt_var, self.royale_seed_var,
                   self.royale_ref_var, self.royale_ref_strength_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())

        _br = tk.Frame(setup, bg=_sbg); _br.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        self._royale_render_row = _br
        self._royale_render_btn = tk.Button(_br, text="Render epochs", font=(FONT_FAMILY, 11, "bold"),
                                            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF",
                                            activebackground="#256F46", relief="flat", bd=0, padx=24, pady=6,
                                            cursor="hand2", command=self._royale_render)
        self._royale_render_btn.pack(side=tk.LEFT)
        self.royale_status_var = tk.StringVar(value="Pick a folder, set a prompt, then render.")
        tk.Label(_br, textvariable=self.royale_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))

        cf = self._start_section_card(outer, "Crossfade",
                                      "Drag to blend between consecutive epochs — stop where it looks best.")
        _cfr = tk.Frame(cf, bg=_sbg); _cfr.pack(anchor=tk.W, pady=(0, 8))
        tk.Label(_cfr, text="Size", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_w_var = tk.StringVar(value=self.last_used.get("royale_w", "512"))
        ttk.Combobox(_cfr, textvariable=self.royale_w_var,
                     values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_cfr, text="x", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(6, 6))
        self.royale_h_var = tk.StringVar(value=self.last_used.get("royale_h", "512"))
        ttk.Combobox(_cfr, textvariable=self.royale_h_var,
                     values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_cfr, text="Max renders", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 6))
        self.royale_max_var = tk.StringVar(value=self.last_used.get("royale_max", "12"))
        ttk.Combobox(_cfr, textvariable=self.royale_max_var,
                     values=["All", "6", "8", "10", "12", "16", "20"],
                     state="readonly", width=6).pack(side=tk.LEFT)
        tk.Label(_cfr, text="how many epochs to render, newest first",
                 bg=_sbg, fg=COLORS["text_muted"], font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(10, 0))
        for _v in (self.royale_w_var, self.royale_h_var, self.royale_max_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())
        holder = tk.Frame(cf, width=512, height=512, bg="#1c1c1c", highlightthickness=0)
        holder.pack(pady=(0, 8))
        holder.pack_propagate(False)
        self._royale_preview_label = ttk.Label(holder, text="(render to begin)", anchor=tk.CENTER, background="#1c1c1c")
        self._royale_preview_label.pack(fill=tk.BOTH, expand=True)
        self._royale_holder = holder
        self.royale_scrub_label_var = tk.StringVar(value="")
        tk.Label(cf, textvariable=self.royale_scrub_label_var, font=(FONT_FAMILY, 11, "bold"),
                 fg=COLORS["text_primary"], bg=_sbg).pack()
        self.royale_scrub_var = tk.DoubleVar(value=0.0)
        self._royale_scale = ttk.Scale(cf, from_=0.0, to=0.0, variable=self.royale_scrub_var,
                                       orient=tk.HORIZONTAL, command=lambda v: self._royale_scrub())
        self._royale_scale.pack(fill=tk.X, padx=20, pady=(4, 8))

        exp = self._start_section_card(outer, "Export the morph",
                                       "Save the crossfade as a looping clip — face resolving epoch by epoch, "
                                       "with a Fizgig · LoRA Royale tag. Made to share.")
        _er1 = tk.Frame(exp, bg=_sbg); _er1.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_er1, text="Format", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_export_format_var = tk.StringVar(value="MP4")
        _fmt_combo = ttk.Combobox(_er1, textvariable=self.royale_export_format_var, values=["MP4", "GIF"],
                                  state="readonly", width=6)
        _fmt_combo.pack(side=tk.LEFT)
        ToolTip(_fmt_combo, "MP4 is full colour, smaller, faster to write, and autoplays on X / Reddit / Instagram.\n"
                            "GIF embeds anywhere but is limited to 256 colours (slight banding on faces),\n"
                            "takes noticeably longer to export, and makes a larger file.")
        tk.Label(_er1, text="Speed", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.royale_export_speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(_er1, textvariable=self.royale_export_speed_var, values=["Slow", "Normal", "Fast"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        _er2 = tk.Frame(exp, bg=_sbg); _er2.pack(anchor=tk.W, pady=(0, 8))
        self.royale_export_loop_var = tk.BooleanVar(value=True)
        self.royale_export_epoch_var = tk.BooleanVar(value=True)
        self.royale_export_wm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(_er2, text="Loop (ping-pong)", variable=self.royale_export_loop_var).pack(side=tk.LEFT)
        ttk.Checkbutton(_er2, text="Epoch ticker", variable=self.royale_export_epoch_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(_er2, text="Fizgig tag", variable=self.royale_export_wm_var).pack(side=tk.LEFT, padx=(14, 0))
        _er3 = tk.Frame(exp, bg=_sbg); _er3.pack(anchor=tk.W)
        self._royale_export_btn = tk.Button(_er3, text="Export clip…", font=(FONT_FAMILY, 10, "bold"),
                                            fg="#FFFFFF", bg="#8E44AD", activeforeground="#FFFFFF",
                                            activebackground="#763A91", relief="flat", bd=0, padx=18, pady=5,
                                            cursor="hand2", command=self._royale_export)
        self._royale_export_btn.pack(side=tk.LEFT)
        self._royale_save_stills_btn = tk.Button(_er3, text="Save all stills…", font=(FONT_FAMILY, 10, "bold"),
                                                 fg="#FFFFFF", bg="#34495E", activeforeground="#FFFFFF",
                                                 activebackground="#2C3E50", relief="flat", bd=0, padx=18, pady=5,
                                                 cursor="hand2", command=self._royale_save_all_stills)
        self._royale_save_stills_btn.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(self._royale_save_stills_btn,
                "Save every rendered epoch still to a folder you pick, as full-res PNGs\n"
                "(named by epoch so they sort in order). The render results otherwise live\n"
                "only in memory and are lost when you close the app.")
        self.royale_export_status_var = tk.StringVar(value="")
        tk.Label(_er3, textvariable=self.royale_export_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))

        trav = self._start_section_card(outer, "Seed travel",
                                        "Take the epoch on the crossfade and morph it smoothly between two seeds "
                                        "(slerp through noise space) — shows the LoRA's range, saved as a clip.")
        _tps = tk.Frame(trav, bg=_sbg); _tps.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_tps, text="Preset", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_travel_preset_var = tk.StringVar(value="")
        _tpcb = ttk.Combobox(_tps, textvariable=self.royale_travel_preset_var,
                             values=list(SEED_TRAVEL_PRESETS.keys()), state="readonly", width=20)
        _tpcb.pack(side=tk.LEFT)
        _tpcb.bind("<<ComboboxSelected>>", lambda e: self._royale_travel_apply_preset())
        ToolTip(_tpcb,
                "One-click recipes for the seed-travel mechanics (reference / anchor / journey length /\n"
                "identity-lock). They don't touch your prompt — seed travel uses the Setup prompt.\n"
                "Tune anything after picking.")

        _tr1 = tk.Frame(trav, bg=_sbg); _tr1.pack(anchor=tk.W, pady=(0, 2))
        tk.Label(_tr1, text="Seeds", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_travel_seed_a_var = tk.StringVar(value=self.last_used.get("royale_travel_seed_a", "42"))
        ttk.Entry(_tr1, textvariable=self.royale_travel_seed_a_var, width=9).pack(side=tk.LEFT)
        tk.Label(_tr1, text="→", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 8))
        self.royale_travel_seed_b_var = tk.StringVar(value=self.last_used.get("royale_travel_seed_b", "4242"))
        ttk.Entry(_tr1, textvariable=self.royale_travel_seed_b_var, width=9).pack(side=tk.LEFT)
        tk.Button(_tr1, text="🎲", font=(FONT_FAMILY, 12), relief="flat", bd=0,
                  fg=COLORS["text_primary"], bg=COLORS["bg_deep"],
                  activebackground=COLORS["border"], activeforeground=COLORS["text_primary"],
                  cursor="hand2", padx=5, pady=0,
                  command=self._royale_travel_randomize_seeds).pack(side=tk.LEFT, padx=(6, 0))
        for _sv in (self.royale_travel_seed_a_var, self.royale_travel_seed_b_var):
            _sv.trace_add("write", lambda *a: self._save_last_used_paths())
        # (royale_travel_w/h var traces added after they're created below.)
        tk.Label(_tr1, text="Waypoints", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 4))
        self.royale_travel_waypoints_var = tk.StringVar(
            value=self.last_used.get("royale_travel_waypoints", "2"))
        _twcb = ttk.Combobox(_tr1, textvariable=self.royale_travel_waypoints_var,
                             values=["2", "3", "4", "5", "6", "8"], state="readonly", width=4)
        _twcb.pack(side=tk.LEFT)
        _twcb.bind("<<ComboboxSelected>>", lambda e: self._save_last_used_paths())
        ToolTip(_twcb,
                "Seeds in the journey. 2 = a straight Start→End morph. More = a flowing tour through\n"
                "extra seeds in between (Start → … → End), all slerped so it stays smooth. The\n"
                "intermediates derive from the Start seed, so the journey is reproducible.")
        tk.Label(_tr1, text="Frames", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 4))
        self.royale_travel_frames_var = tk.StringVar(value="24")
        _frcb = ttk.Combobox(_tr1, textvariable=self.royale_travel_frames_var,
                             values=["16", "24", "36", "48", "64", "96", "128", "192", "256"],
                             state="readonly", width=5)
        _frcb.pack(side=tk.LEFT)
        ToolTip(_frcb, "Each frame is a fresh 4-step render — more frames = smoother but slower.\n"
                       "More journey waypoints want more frames to stay smooth.")
        tk.Label(trav, text="Waypoints = seeds in the journey; 🎲 rerolls Start/End. With a reference holding "
                            "the subject, more waypoints = a longer tour through compositions.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))

        _trr = tk.Frame(trav, bg=_sbg); _trr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(_trr, text="Reference", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_travel_ref_var = tk.StringVar(value=self.last_used.get("royale_travel_ref", ""))
        self._royale_travel_ref_entry = ttk.Entry(_trr, textvariable=self.royale_travel_ref_var, state="readonly")
        self._royale_travel_ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._royale_travel_ref_browse = ttk.Button(_trr, text="Browse…", command=self._royale_travel_browse_ref)
        self._royale_travel_ref_browse.pack(side=tk.LEFT, padx=(6, 0))
        self._royale_travel_ref_clear = ttk.Button(_trr, text="Clear", command=lambda: self.royale_travel_ref_var.set(""))
        self._royale_travel_ref_clear.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(self._royale_travel_ref_entry,
                "Klein is an edit model — a reference image anchors the subject, so the seed morph stays "
                "more stable. Optional. Auto-resized to ~0.2 MP. Falls back to the Setup reference if left empty.")
        self.royale_travel_use_epoch_ref_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_travel_use_epoch_ref", False)))
        self.royale_travel_seq_ref_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_travel_seq_ref", False)))
        _tru = tk.Frame(trav, bg=_sbg); _tru.pack(anchor=tk.W, pady=(0, 4))
        self._royale_travel_ref_row = _tru
        self._royale_travel_useepoch_cb = ttk.Checkbutton(
            _tru, text="Use the rendered epoch as the reference",
            variable=self.royale_travel_use_epoch_ref_var,
            command=self._royale_travel_toggle_ref_widgets)
        self._royale_travel_useepoch_cb.pack(side=tk.LEFT)
        self._royale_travel_ref_strength_lbl = tk.Label(_tru, text="Strength", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_travel_ref_strength_lbl.pack(side=tk.LEFT, padx=(16, 3))
        self.royale_travel_ref_strength_var = tk.StringVar(
            value=self.last_used.get("royale_travel_ref_strength", "0.25"))
        _trse = ttk.Entry(_tru, textvariable=self.royale_travel_ref_strength_var, width=5)
        _trse.pack(side=tk.LEFT)
        self._royale_travel_ref_strength_entry = _trse
        ToolTip(_trse, "How strongly the reference anchors the morph.\n"
                       "0.1–0.4 is the sweet range for seed travel — enough to hold the subject, loose "
                       "enough to let the seeds actually travel. 1.0 clamps too hard here, 0 = off.")
        _tseq = ttk.Checkbutton(_tru, text="Sequential reference", variable=self.royale_travel_seq_ref_var)
        self._royale_travel_seq_cb = _tseq
        _tseq.pack(side=tk.LEFT, padx=(16, 0))
        ToolTip(_tseq, "Feedback chain: frame 1 uses your reference, each later frame edits the previous one "
                       "(the morph compounds and evolves).\n"
                       "⚠ With this ON, keep Strength low — high strength compounds every frame and causes "
                       "distortion / low-quality results. Recommended max ≈ 0.4 in this mode.")
        self._royale_travel_maxmp_lbl = tk.Label(_tru, text="Max MP", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_travel_maxmp_lbl.pack(side=tk.LEFT, padx=(16, 3))
        self.royale_travel_ref_mp_var = tk.StringVar(value=self.last_used.get("royale_travel_ref_mp", "0.2"))
        _trmp = ttk.Entry(_tru, textvariable=self.royale_travel_ref_mp_var, width=5)
        _trmp.pack(side=tk.LEFT)
        ToolTip(_trmp, "Reference encode resolution cap (megapixels), 0.05–1.0.\n"
                       "Higher carries more detail (useful for sequential reference) at a little more VRAM.")
        tk.Label(trav, text="Sequential: frame 1 uses your reference, each frame after edits the previous one "
                            "(a feedback chain — compounds and evolves, but can drift over a long run, especially "
                            "at high Strength). Off = every frame uses the same reference: cleaner and more predictable.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))
        for _v in (self.royale_travel_ref_var, self.royale_travel_use_epoch_ref_var,
                   self.royale_travel_ref_strength_var, self.royale_travel_ref_mp_var,
                   self.royale_travel_seq_ref_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())
        self._royale_travel_toggle_ref_widgets()

        _trf = tk.Frame(trav, bg=_sbg); _trf.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_trf, text="Speed", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_travel_speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(_trf, textvariable=self.royale_travel_speed_var, values=["Slow", "Normal", "Fast"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        tk.Label(_trf, text="W", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 3))
        self.royale_travel_w_var = tk.StringVar(
            value=self.last_used.get("royale_travel_w", self.last_used.get("royale_w", "512")))
        ttk.Combobox(_trf, textvariable=self.royale_travel_w_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_trf, text="H", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 3))
        self.royale_travel_h_var = tk.StringVar(
            value=self.last_used.get("royale_travel_h", self.last_used.get("royale_h", "512")))
        ttk.Combobox(_trf, textvariable=self.royale_travel_h_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        for _v in (self.royale_travel_w_var, self.royale_travel_h_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())
        _tro = tk.Frame(trav, bg=_sbg); _tro.pack(anchor=tk.W, pady=(0, 8))
        self.royale_travel_loop_var = tk.BooleanVar(value=True)
        self.royale_travel_epoch_var = tk.BooleanVar(value=True)
        self.royale_travel_wm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(_tro, text="Loop (ping-pong)", variable=self.royale_travel_loop_var).pack(side=tk.LEFT)
        ttk.Checkbutton(_tro, text="Epoch badge", variable=self.royale_travel_epoch_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(_tro, text="Fizgig tag", variable=self.royale_travel_wm_var).pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(_tro, text="Deflicker", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.royale_travel_deflicker_var = tk.StringVar(value="None")
        _tdf = ttk.Combobox(_tro, textvariable=self.royale_travel_deflicker_var,
                            values=["None", "Normal", "Strong"], state="readonly", width=10)
        _tdf.pack(side=tk.LEFT)
        ToolTip(_tdf,
                "Timelapse-style luminance deflicker (post-process, like DaVinci Resolve).\n"
                "• None — off.\n"
                "• Normal — removes frame-to-frame jitter, keeps the intended slow brightness arc.\n"
                "• Strong — wider window, flattens harder (also smooths slower changes).")

        _tr2 = tk.Frame(trav, bg=_sbg); _tr2.pack(anchor=tk.W)
        self._royale_travel_btn = tk.Button(_tr2, text="Render seed-travel…", font=(FONT_FAMILY, 10, "bold"),
                                            fg="#FFFFFF", bg="#C0392B", activeforeground="#FFFFFF",
                                            activebackground="#A03124", relief="flat", bd=0, padx=18, pady=5,
                                            cursor="hand2", command=self._royale_seed_travel)
        self._royale_travel_btn.pack(side=tk.LEFT)
        self.royale_travel_status_var = tk.StringVar(value="")
        tk.Label(_tr2, textvariable=self.royale_travel_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))
        # Render -> scrub -> save (options applied at save).
        self._royale_make_scrubber(
            trav, "seed", "seedtravel", _sbg, self.royale_travel_status_var,
            lambda: dict(speed=self.royale_travel_speed_var.get(),
                         pingpong=bool(self.royale_travel_loop_var.get()),
                         brand=bool(self.royale_travel_wm_var.get()),
                         badge=bool(self.royale_travel_epoch_var.get()),
                         deflicker=self.royale_travel_deflicker_var.get()))

        import sys as _sys
        _sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import prompt_travel as _ptlib
        ptrav = self._start_section_card(outer, "Prompt travel",
                                         "Morph the parked epoch through a series of prompt variations on a fixed "
                                         "seed — it interpolates the text embedding, so the same subject flows "
                                         "(e.g. dawn → night). Saved as a clip.")
        _pps = tk.Frame(ptrav, bg=_sbg); _pps.pack(fill=tk.X, pady=(0, 4))
        tk.Label(_pps, text="Preset", bg=_sbg, fg=COLORS["text_muted"], width=12,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_preset_var = tk.StringVar(value="")
        _ppcb = ttk.Combobox(_pps, textvariable=self.royale_pt_preset_var,
                             values=list(_ptlib.PRESET_NAMES), state="readonly",
                             width=min(30, max(len(n) for n in _ptlib.PRESET_NAMES) + 2))
        _ppcb.pack(side=tk.LEFT)
        _ppcb.bind("<<ComboboxSelected>>", lambda e: self._royale_pt_apply_preset())
        ToolTip(_ppcb,
                "One-click starting points tuned from real runs. Each fills the whole card —\n"
                "dimension, references, seed/drift, interpolation — and builds a prompt you can\n"
                "edit. image 1 = your original reference, image 2 = the previous frame; the\n"
                "prompt steers them by index. Age morphs the face (low anchor); Era and the\n"
                "world/lighting presets hold the person (full anchor) while the scene changes.")
        tk.Label(_pps, text="Subject", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.royale_pt_subject_var = tk.StringVar(
            value=self.last_used.get("royale_pt_subject", "Woman"))
        _pscb = ttk.Combobox(_pps, textvariable=self.royale_pt_subject_var,
                             values=list(_ptlib.SUBJECT_LABELS), state="readonly", width=12)
        _pscb.pack(side=tk.LEFT)
        _pscb.bind("<<ComboboxSelected>>", lambda e: self._royale_pt_apply_subject())
        ToolTip(_pscb,
                "Who/what the preset is about — fills the subject in the prompt (and the matching\n"
                "pronoun, so 'change her/his/its age' stays correct). Picking a preset jumps this\n"
                "to a sensible default (Age → Female); change it any time, then fine-tune the box.")

        _pp = tk.Frame(ptrav, bg=_sbg); _pp.pack(fill=tk.X, pady=(0, 4))
        tk.Label(_pp, text="Prompt", bg=_sbg, fg=COLORS["text_muted"], width=12,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_prompt_var = tk.StringVar(value=self.last_used.get("royale_pt_prompt", ""))
        _ppe = ttk.Entry(_pp, textvariable=self.royale_pt_prompt_var)
        _ppe.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._royale_pt_prompt_entry = _ppe
        ToolTip(_ppe, "Put {x} where the travel word should go, e.g.\n"
                      "  a portrait of sks man, {x}\n"
                      "If you omit {x}, the word is appended to the end.")
        ttk.Button(_pp, text="Insert {x}", command=self._royale_pt_insert_slot).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(ptrav, text="Type {x} where the travel word goes — e.g.  a portrait of sks man, {x} light. "
                             "No {x}? the word is appended to the end.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        # — Travel definition: what the clip morphs through (sits right under the prompt) —
        # Hidden dimension state — set by the Preset dropdown above (incl. "Custom").
        self.royale_pt_dim_var = tk.StringVar(value=self.last_used.get("royale_pt_dim", "Age"))
        _pcr = tk.Frame(ptrav, bg=_sbg); _pcr.pack(fill=tk.X, pady=(0, 0))
        tk.Label(_pcr, text="Custom words", bg=_sbg, fg=COLORS["text_muted"], width=12,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_custom_var = tk.StringVar(value=self.last_used.get("royale_pt_custom", ""))
        ttk.Entry(_pcr, textvariable=self.royale_pt_custom_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(ptrav, text="Comma-separated — only used when Preset = Custom words.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 6))

        _prg = tk.Frame(ptrav, bg=_sbg); _prg.pack(fill=tk.X, pady=(0, 0))
        tk.Label(_prg, text="Travel", bg=_sbg, fg=COLORS["text_muted"], width=12,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_start_var = tk.StringVar(value=self.last_used.get("royale_pt_start", ""))
        self._royale_pt_start_combo = ttk.Combobox(_prg, textvariable=self.royale_pt_start_var,
                                                    state="readonly", width=24)
        self._royale_pt_start_combo.pack(side=tk.LEFT)
        tk.Label(_prg, text="→", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 8))
        self.royale_pt_end_var = tk.StringVar(value=self.last_used.get("royale_pt_end", ""))
        self._royale_pt_end_combo = ttk.Combobox(_prg, textvariable=self.royale_pt_end_var,
                                                  state="readonly", width=24)
        self._royale_pt_end_combo.pack(side=tk.LEFT)
        tk.Label(_prg, text="Frames", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(18, 4))
        self.royale_pt_frames_var = tk.StringVar(value=self.last_used.get("royale_pt_frames", "32"))
        _pfcb = ttk.Combobox(_prg, textvariable=self.royale_pt_frames_var,
                             values=["24", "32", "48", "64", "96", "128", "192", "256"],
                             state="readonly", width=5)
        _pfcb.pack(side=tk.LEFT)
        ToolTip(_pfcb, "Each frame is a fresh 4-step render — more frames = smoother but slower.\n"
                       "Frames are spread across the Start→End waypoint span.")
        tk.Label(ptrav, text="Start/End pick which waypoints to span — e.g. start Age at the subject's current age so "
                             "it matches the reference, then travel onward (the loop ping-pongs back). Frames spread "
                             "across that span.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 4))
        for _v in (self.royale_pt_start_var, self.royale_pt_end_var):
            _v.trace_add("write", lambda *a: (self._royale_pt_refresh_words(), self._save_last_used_paths()))

        self.royale_pt_words_var = tk.StringVar(value="")
        tk.Label(ptrav, textvariable=self.royale_pt_words_var, font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["accent"], bg=_sbg, wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        _ptr = tk.Frame(ptrav, bg=_sbg); _ptr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(_ptr, text="Reference", bg=_sbg, fg=COLORS["text_muted"], width=12,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_ref_var = tk.StringVar(value=self.last_used.get("royale_pt_ref", ""))
        self._royale_pt_ref_entry = ttk.Entry(_ptr, textvariable=self.royale_pt_ref_var, state="readonly")
        self._royale_pt_ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._royale_pt_ref_browse = ttk.Button(_ptr, text="Browse…", command=self._royale_pt_browse_ref)
        self._royale_pt_ref_browse.pack(side=tk.LEFT, padx=(6, 0))
        self._royale_pt_ref_clear = ttk.Button(_ptr, text="Clear", command=lambda: self.royale_pt_ref_var.set(""))
        self._royale_pt_ref_clear.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(self._royale_pt_ref_entry,
                "Klein is an edit model — a reference image anchors the subject, so the prompt morph "
                "stays much more stable (composition holds while the words change). Optional but recommended. "
                "Auto-resized to ~0.2 MP. Falls back to the Setup reference if left empty.")
        self.royale_pt_use_epoch_ref_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_pt_use_epoch_ref", False)))
        self.royale_pt_seq_ref_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_pt_seq_ref", False)))
        _ptu = tk.Frame(ptrav, bg=_sbg); _ptu.pack(anchor=tk.W, pady=(0, 4))
        self._royale_pt_ref_row = _ptu
        self._royale_pt_useepoch_cb = ttk.Checkbutton(
            _ptu, text="Use the rendered epoch as the reference",
            variable=self.royale_pt_use_epoch_ref_var,
            command=self._royale_pt_toggle_ref_widgets)
        self._royale_pt_useepoch_cb.pack(side=tk.LEFT)
        self._royale_pt_ref_strength_lbl = tk.Label(_ptu, text="Strength", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_pt_ref_strength_lbl.pack(side=tk.LEFT, padx=(16, 3))
        self.royale_pt_ref_strength_var = tk.StringVar(
            value=self.last_used.get("royale_pt_ref_strength", "1.0"))
        _ptse = ttk.Entry(_ptu, textvariable=self.royale_pt_ref_strength_var, width=5)
        _ptse.pack(side=tk.LEFT)
        self._royale_pt_ref_strength_entry = _ptse
        ToolTip(_ptse, "How strongly the reference anchors the morph.\n"
                       "1.0 is the right default for prompt travel — this is the edit model working as "
                       "intended: the reference holds the subject at full strength while the prompt does "
                       "the editing. Lower only if you want the prompt to override the reference more. 0 = off.")
        _pseq = ttk.Checkbutton(_ptu, text="Sequential reference", variable=self.royale_pt_seq_ref_var)
        self._royale_pt_seq_cb = _pseq
        _pseq.pack(side=tk.LEFT, padx=(16, 0))
        ToolTip(_pseq, "Feedback chain: frame 1 uses your reference, each later frame edits the previous one.\n"
                       "⚠ With this ON, high Strength compounds every frame and can cause distortion / low quality "
                       "over a long run. Keep Strength modest (≈0.4) — or turn on 'Anchor to original' below, which "
                       "re-injects the clean reference each frame and lets you run higher Strength safely.")
        self._royale_pt_maxmp_lbl = tk.Label(_ptu, text="Max MP", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_pt_maxmp_lbl.pack(side=tk.LEFT, padx=(16, 3))
        self.royale_pt_ref_mp_var = tk.StringVar(value=self.last_used.get("royale_pt_ref_mp", "0.2"))
        _ptmp = ttk.Entry(_ptu, textvariable=self.royale_pt_ref_mp_var, width=5)
        _ptmp.pack(side=tk.LEFT)
        ToolTip(_ptmp, "Reference encode resolution cap (megapixels), 0.05–1.0.\n"
                       "Higher carries more detail (useful for sequential reference) at a little more VRAM.")
        _ptsq = tk.Frame(ptrav, bg=_sbg); _ptsq.pack(anchor=tk.W, pady=(0, 4))
        self.royale_pt_anchor_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_pt_anchor", True)))
        ttk.Checkbutton(_ptsq, text="Anchor to original",
                        variable=self.royale_pt_anchor_var).pack(side=tk.LEFT, padx=(14, 0))
        self._royale_pt_anchor_str_lbl = tk.Label(_ptsq, text="Anchor str", bg=_sbg, fg=COLORS["text_muted"])
        self._royale_pt_anchor_str_lbl.pack(side=tk.LEFT, padx=(8, 3))
        self.royale_pt_anchor_str_var = tk.StringVar(
            value=self.last_used.get("royale_pt_anchor_str", "1.0"))
        _ptast = ttk.Entry(_ptsq, textvariable=self.royale_pt_anchor_str_var, width=5)
        self._royale_pt_anchor_str_entry = _ptast
        _ptast.pack(side=tk.LEFT)
        ToolTip(_ptast, "With 'Anchor to original' on, every frame also references the ORIGINAL image at this "
                        "strength alongside the previous frame. The original re-injects clean identity/detail "
                        "each frame, so the feedback chain can't drift (VAE degradation). 1.0 = full anchor.")
        tk.Label(ptrav, text="Sequential: frame 1 uses your reference, each frame after edits the previous one "
                             "(a feedback chain — the subject smoothly evolves). Anchor to original keeps the "
                             "pristine reference in every frame to stop drift. Recommended: strength ~0.7, "
                             "Max MP ~0.5, anchor 1.0.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))
        for _v in (self.royale_pt_ref_mp_var, self.royale_pt_seq_ref_var,
                   self.royale_pt_anchor_var, self.royale_pt_anchor_str_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())

        _pof = tk.Frame(ptrav, bg=_sbg); _pof.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_pof, text="Speed", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(_pof, textvariable=self.royale_pt_speed_var, values=["Slow", "Normal", "Fast"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        tk.Label(_pof, text="W", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 3))
        self.royale_pt_w_var = tk.StringVar(
            value=self.last_used.get("royale_pt_w", self.last_used.get("royale_w", "512")))
        ttk.Combobox(_pof, textvariable=self.royale_pt_w_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_pof, text="H", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 3))
        self.royale_pt_h_var = tk.StringVar(
            value=self.last_used.get("royale_pt_h", self.last_used.get("royale_h", "512")))
        ttk.Combobox(_pof, textvariable=self.royale_pt_h_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)

        _pin = tk.Frame(ptrav, bg=_sbg); _pin.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_pin, text="Interpolation", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_pt_interp_var = tk.StringVar(
            value=self.last_used.get("royale_pt_interp", "Linear"))
        _ptin = ttk.Combobox(_pin, textvariable=self.royale_pt_interp_var,
                             values=["Linear", "Norm-preserved", "Slerp"],
                             state="readonly", width=15)
        _ptin.pack(side=tk.LEFT)
        ToolTip(_ptin,
                "How the text embedding morphs between waypoints (test with Vary seed OFF to see it):\n"
                "• Linear — original. Conditioning weakens mid-way between words → brightness/contrast can dip there.\n"
                "• Norm-preserved — keeps the linear blend but holds conditioning strength constant → flatter,\n"
                "  more even brightness with minimal change to the look. Low risk.\n"
                "• Slerp — constant-speed spherical glide at full conditioning strength → smoothest morph,\n"
                "  a bigger departure from Linear.")
        tk.Label(_pin, text="Seed drift", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 6))
        self.royale_pt_drift_var = tk.StringVar(value=self.last_used.get("royale_pt_drift", "0.0"))
        self._royale_pt_drift_entry = ttk.Entry(_pin, textvariable=self.royale_pt_drift_var, width=5)
        self._royale_pt_drift_entry.pack(side=tk.LEFT)
        ToolTip(self._royale_pt_drift_entry,
                "Fixes the static-seed fixed point. On a fixed seed the image can settle so hard that the\n"
                "prompt (e.g. age) never fully expresses. Seed drift slerps the noise smoothly from the base\n"
                "seed toward a second seed across the sweep, injecting fresh structure so the prompt takes\n"
                "hold — with none of the jumpiness of Vary seed (which re-rolls randomly every frame).\n"
                "0 = off (pure static seed).  0.1–0.3 = gentle.  0.5–1.0 = strong (the last frame ends up\n"
                "half / fully a different seed).  Only applies when Vary seed is OFF.")

        _poo = tk.Frame(ptrav, bg=_sbg); _poo.pack(anchor=tk.W, pady=(0, 8))
        self.royale_pt_loop_var = tk.BooleanVar(value=True)
        self.royale_pt_word_var = tk.BooleanVar(value=True)
        self.royale_pt_wm_var = tk.BooleanVar(value=True)
        self.royale_pt_vary_seed_var = tk.BooleanVar(
            value=bool(self.last_used.get("royale_pt_vary_seed", False)))
        ttk.Checkbutton(_poo, text="Loop (ping-pong)", variable=self.royale_pt_loop_var).pack(side=tk.LEFT)
        ttk.Checkbutton(_poo, text="Word badge", variable=self.royale_pt_word_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(_poo, text="Fizgig tag", variable=self.royale_pt_wm_var).pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(_poo, text="Deflicker", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.royale_pt_deflicker_var = tk.StringVar(value="None")
        _pdf = ttk.Combobox(_poo, textvariable=self.royale_pt_deflicker_var,
                            values=["None", "Normal", "Strong"], state="readonly", width=10)
        _pdf.pack(side=tk.LEFT)
        ToolTip(_pdf,
                "Timelapse-style luminance deflicker (post-process, like DaVinci Resolve).\n"
                "Smooths each frame's brightness toward a low-frequency baseline.\n"
                "• None — off.\n"
                "• Normal — removes frame-to-frame jitter, keeps the intended slow brightness arc\n"
                "  (safe for Time-of-day / Lighting — the deliberate change is preserved).\n"
                "• Strong — wider window, flattens harder; also smooths slower brightness changes.\n"
                "  Use when even the gentle arc wobbles, at the cost of some intended drift.")
        _vs = ttk.Checkbutton(_poo, text="Vary seed", variable=self.royale_pt_vary_seed_var,
                              command=self._royale_pt_sync_seed_widgets)
        _vs.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(_vs, "Give each frame its own seed instead of one fixed seed, so the image\n"
                     "re-rolls per frame and the prompt (e.g. age) expresses more strongly.\n"
                     "Seeds advance deterministically (base, base+1, base+2 …), so the whole\n"
                     "clip is reproducible: same base seed → same result every render.\n"
                     "Disables Seed drift (that's the smooth alternative for a fixed seed).")
        _pb = tk.Frame(ptrav, bg=_sbg); _pb.pack(anchor=tk.W)
        self._royale_pt_btn = tk.Button(_pb, text="Render prompt-travel…", font=(FONT_FAMILY, 10, "bold"),
                                        fg="#FFFFFF", bg="#B7791F", activeforeground="#FFFFFF",
                                        activebackground="#9A6518", relief="flat", bd=0, padx=18, pady=5,
                                        cursor="hand2", command=self._royale_prompt_travel)
        self._royale_pt_btn.pack(side=tk.LEFT)
        self.royale_pt_status_var = tk.StringVar(value="")
        tk.Label(_pb, textvariable=self.royale_pt_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))
        # Render -> scrub -> save (options applied at save).
        self._royale_make_scrubber(
            ptrav, "prompt", "prompttravel", _sbg, self.royale_pt_status_var,
            lambda: dict(speed=self.royale_pt_speed_var.get(),
                         pingpong=bool(self.royale_pt_loop_var.get()),
                         brand=bool(self.royale_pt_wm_var.get()),
                         badge=bool(self.royale_pt_word_var.get()),
                         deflicker=self.royale_pt_deflicker_var.get()))
        for _v in (self.royale_pt_dim_var, self.royale_pt_custom_var):
            _v.trace_add("write", lambda *a: (self._royale_pt_refresh_range(),
                                              self._royale_pt_refresh_words(), self._save_last_used_paths()))
        for _v in (self.royale_pt_prompt_var, self.royale_pt_frames_var, self.royale_pt_ref_var,
                   self.royale_pt_w_var, self.royale_pt_h_var, self.royale_pt_ref_strength_var,
                   self.royale_pt_drift_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())
        self.royale_pt_use_epoch_ref_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self.royale_pt_vary_seed_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self.royale_pt_interp_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self._royale_pt_refresh_range()
        self._royale_pt_refresh_words()
        self._royale_pt_toggle_ref_widgets()
        self._royale_pt_sync_seed_widgets()

        # ----- LoRA strength travel -----
        ltrav = self._start_section_card(outer, "LoRA strength travel",
                                         "Hold the prompt and seed fixed and ramp the LoRA's strength from one value "
                                         "to another — watch the effect fade in (0 = base model) through to full "
                                         "strength and beyond. Saved as a clip.")
        _ls1 = tk.Frame(ltrav, bg=_sbg); _ls1.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_ls1, text="Strength", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_lora_start_var = tk.StringVar(value=self.last_used.get("royale_lora_start", "0.0"))
        ttk.Entry(_ls1, textvariable=self.royale_lora_start_var, width=6).pack(side=tk.LEFT)
        tk.Label(_ls1, text="→", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 8))
        self.royale_lora_end_var = tk.StringVar(value=self.last_used.get("royale_lora_end", "1.0"))
        ttk.Entry(_ls1, textvariable=self.royale_lora_end_var, width=6).pack(side=tk.LEFT)
        tk.Label(_ls1, text="Frames", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 4))
        self.royale_lora_frames_var = tk.StringVar(value=self.last_used.get("royale_lora_frames", "24"))
        _lfcb = ttk.Combobox(_ls1, textvariable=self.royale_lora_frames_var,
                             values=["16", "24", "36", "48", "64", "96", "128", "192", "256"],
                             state="readonly", width=5)
        _lfcb.pack(side=tk.LEFT)
        ToolTip(_lfcb, "Each frame is a fresh 4-step render — more frames = smoother ramp but slower.")
        tk.Label(ltrav, text="0 = base model (no LoRA); 1.0 = trained strength; >1 over-drives it. Uses the Setup "
                             "prompt and seed, fixed — only the LoRA strength changes.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))

        _lsf = tk.Frame(ltrav, bg=_sbg); _lsf.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_lsf, text="Speed", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.royale_lora_speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(_lsf, textvariable=self.royale_lora_speed_var, values=["Slow", "Normal", "Fast"],
                     state="readonly", width=8).pack(side=tk.LEFT)
        tk.Label(_lsf, text="W", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 3))
        self.royale_lora_w_var = tk.StringVar(
            value=self.last_used.get("royale_lora_w", self.last_used.get("royale_w", "512")))
        ttk.Combobox(_lsf, textvariable=self.royale_lora_w_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_lsf, text="H", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 3))
        self.royale_lora_h_var = tk.StringVar(
            value=self.last_used.get("royale_lora_h", self.last_used.get("royale_h", "512")))
        ttk.Combobox(_lsf, textvariable=self.royale_lora_h_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)

        _lso = tk.Frame(ltrav, bg=_sbg); _lso.pack(anchor=tk.W, pady=(0, 8))
        self.royale_lora_loop_var = tk.BooleanVar(value=True)
        self.royale_lora_badge_var = tk.BooleanVar(value=True)
        self.royale_lora_wm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(_lso, text="Loop (ping-pong)", variable=self.royale_lora_loop_var).pack(side=tk.LEFT)
        _lbg = ttk.Checkbutton(_lso, text="Strength badge", variable=self.royale_lora_badge_var)
        _lbg.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(_lbg, "Burn the current strength (e.g. 0.80×) into each frame, ticking as it ramps — "
                      "makes the clip self-explanatory.")
        ttk.Checkbutton(_lso, text="Fizgig tag", variable=self.royale_lora_wm_var).pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(_lso, text="Deflicker", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.royale_lora_deflicker_var = tk.StringVar(value="None")
        ttk.Combobox(_lso, textvariable=self.royale_lora_deflicker_var,
                     values=["None", "Normal", "Strong"], state="readonly", width=10).pack(side=tk.LEFT)

        _lsb = tk.Frame(ltrav, bg=_sbg); _lsb.pack(anchor=tk.W)
        self._royale_lora_btn = tk.Button(_lsb, text="Render strength-travel…", font=(FONT_FAMILY, 10, "bold"),
                                          fg="#FFFFFF", bg="#B7791F", activeforeground="#FFFFFF",
                                          activebackground="#9A6518", relief="flat", bd=0, padx=18, pady=5,
                                          cursor="hand2", command=self._royale_lora_travel)
        self._royale_lora_btn.pack(side=tk.LEFT)
        self.royale_lora_status_var = tk.StringVar(value="")
        tk.Label(_lsb, textvariable=self.royale_lora_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))
        for _v in (self.royale_lora_start_var, self.royale_lora_end_var, self.royale_lora_frames_var,
                   self.royale_lora_w_var, self.royale_lora_h_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())

        # Render -> scrub -> save (centered 512 preview; options applied at save).
        self._royale_make_scrubber(
            ltrav, "strength", "strengthtravel", _sbg, self.royale_lora_status_var,
            lambda: dict(speed=self.royale_lora_speed_var.get(),
                         pingpong=bool(self.royale_lora_loop_var.get()),
                         brand=bool(self.royale_lora_wm_var.get()),
                         badge=bool(self.royale_lora_badge_var.get()),
                         deflicker=self.royale_lora_deflicker_var.get()))
        tk.Label(ltrav, text="Render once, scrub to review, then save — Speed / Loop / Strength badge / Fizgig tag / "
                             "Deflicker all apply at save, so you can re-save in either format without re-rendering.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        # ----- Comparison sheet (before/after grid) -----
        cmpc = self._start_section_card(outer, "Comparison sheet",
                                        "The share image people actually post for a new LoRA: one row per prompt, "
                                        "one column per condition, same seed across a row so only the LoRA changes. "
                                        "Saved as a single labelled PNG.")
        _cmphow = tk.Frame(cmpc, bg=COLORS["bg_deep"])
        _cmphow.pack(fill=tk.X, pady=(0, 8))
        tk.Label(_cmphow, text="How to use it", bg=COLORS["bg_deep"], fg=COLORS["accent"],
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 2))
        tk.Label(_cmphow, bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                 font=(FONT_FAMILY, 9), justify=tk.LEFT, wraplength=740,
                 text=("Without / with LoRA  —  two columns, showing what your LoRA adds.\n"
                       "    1. Load a LoRA:  Single-LoRA mode (pick the file), or Folder mode → Render,\n"
                       "        then slide the crossfade to the epoch you want to show off.\n"
                       "    2. Type your prompts below, one per line.\n"
                       "    3. Fill in Trigger so the no-LoRA column can drop it (the base model has\n"
                       "        never seen that word — leaving it in makes the comparison unfair).\n"
                       "    4. Render.\n\n"
                       "Every epoch  —  one column per epoch, showing the LoRA learning.\n"
                       "    1. Folder mode → pick your training output folder → Scan → Render.\n"
                       "        (This is the main render at the top of the tab, not this card — the\n"
                       "        sheet reuses those epochs, so it must happen first.)\n"
                       "    2. Set Epochs to keep it readable: blank = all, \"every 4\", or \"4,8,12\".\n"
                       "    3. Type your prompts, then Render.")
                 ).pack(anchor=tk.W, padx=10, pady=(0, 8))
        tk.Label(cmpc, text="Prompts (one per line — each becomes a row):", bg=_sbg,
                 fg=COLORS["text_muted"], font=(FONT_FAMILY, 9)).pack(anchor=tk.W)
        self.royale_cmp_prompts = tk.Text(cmpc, height=4, width=90, wrap=tk.WORD,
                                          bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                          insertbackground=COLORS["text_primary"], relief="flat",
                                          font=(FONT_FAMILY, 9))
        self.royale_cmp_prompts.pack(fill=tk.X, pady=(2, 6))
        _cp = self.last_used.get("royale_cmp_prompts", "")
        if _cp:
            self.royale_cmp_prompts.insert("1.0", _cp)

        _cr1 = tk.Frame(cmpc, bg=_sbg); _cr1.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(_cr1, text="Columns", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_cmp_mode_var = tk.StringVar(
            value=self.last_used.get("royale_cmp_mode", "Without / with LoRA"))
        _cmb = ttk.Combobox(_cr1, textvariable=self.royale_cmp_mode_var,
                            values=["Without / with LoRA", "Every epoch"],
                            state="readonly", width=22)
        _cmb.pack(side=tk.LEFT)
        ToolTip(_cmb, "Without / with LoRA: two columns on the current LoRA.\n"
                      "Every epoch: one column per epoch of the scanned run (Folder mode).")
        tk.Label(_cr1, text="Epochs", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 4))
        self.royale_cmp_epochs_var = tk.StringVar(value=self.last_used.get("royale_cmp_epochs", ""))
        _cep = ttk.Entry(_cr1, textvariable=self.royale_cmp_epochs_var, width=14)
        _cep.pack(side=tk.LEFT)
        ToolTip(_cep, "Which epochs become columns (Every epoch mode only).\n"
                      "Blank = all of them.\n"
                      "\"every 4\" = every 4th, always including the last.\n"
                      "\"4,8,12\" = just those.\n"
                      "A 40-epoch run as 40 columns is unreadable — pick 4-6.")
        tk.Label(_cr1, text="Trigger", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(16, 4))
        self.royale_cmp_trigger_var = tk.StringVar(value=self.last_used.get("royale_cmp_trigger", ""))
        _ctg = ttk.Entry(_cr1, textvariable=self.royale_cmp_trigger_var, width=14)
        _ctg.pack(side=tk.LEFT)
        ToolTip(_ctg, "Removed from the prompt for the no-LoRA column, so the base model isn't fed a\n"
                      "token it doesn't know. Also names the with-LoRA column header.")

        _cr2 = tk.Frame(cmpc, bg=_sbg); _cr2.pack(anchor=tk.W, pady=(0, 8))
        tk.Label(_cr2, text="Seed", bg=_sbg, fg=COLORS["text_muted"], width=10,
                 anchor="w").pack(side=tk.LEFT, padx=(0, 6))
        self.royale_cmp_seed_var = tk.StringVar(value=self.last_used.get("royale_cmp_seed", "42"))
        ttk.Entry(_cr2, textvariable=self.royale_cmp_seed_var, width=8).pack(side=tk.LEFT)
        tk.Label(_cr2, text="W", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 3))
        self.royale_cmp_w_var = tk.StringVar(value=self.last_used.get("royale_cmp_w", "512"))
        ttk.Combobox(_cr2, textvariable=self.royale_cmp_w_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(_cr2, text="H", bg=_sbg, fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(8, 3))
        self.royale_cmp_h_var = tk.StringVar(value=self.last_used.get("royale_cmp_h", "512"))
        ttk.Combobox(_cr2, textvariable=self.royale_cmp_h_var, values=["384", "512", "768", "1024", "1280", "1536", "2048"],
                     state="readonly", width=5).pack(side=tk.LEFT)
        self.royale_cmp_rowlabels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(_cr2, text="Row captions", variable=self.royale_cmp_rowlabels_var).pack(side=tk.LEFT, padx=(16, 0))
        self.royale_cmp_brand_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(_cr2, text="Fizgig tag", variable=self.royale_cmp_brand_var).pack(side=tk.LEFT, padx=(12, 0))

        _cb = tk.Frame(cmpc, bg=_sbg); _cb.pack(anchor=tk.W)
        self._royale_cmp_btn = tk.Button(_cb, text="Render comparison sheet…", font=(FONT_FAMILY, 10, "bold"),
                                         fg="#FFFFFF", bg="#B7791F", activeforeground="#FFFFFF",
                                         activebackground="#9A6518", relief="flat", bd=0, padx=18, pady=5,
                                         cursor="hand2", command=self._royale_comparison_sheet)
        self._royale_cmp_btn.pack(side=tk.LEFT)
        self.royale_cmp_status_var = tk.StringVar(value="")
        tk.Label(_cb, textvariable=self.royale_cmp_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(cmpc, text="Every cell is a fresh render — rows x columns images — so 3 prompts across 5 epochs "
                            "is 15 renders. Keep the prompt list short the first time. The finished sheet opens "
                            "in a window to review, then you choose whether to save it.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=_sbg,
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        for _v in (self.royale_cmp_mode_var, self.royale_cmp_trigger_var, self.royale_cmp_seed_var,
                   self.royale_cmp_w_var, self.royale_cmp_h_var, self.royale_cmp_epochs_var):
            _v.trace_add("write", lambda *a: self._save_last_used_paths())

        grid_card = self._start_section_card(outer, "All epochs",
                                             "Click a thumbnail to jump the crossfade there.")
        self._royale_grid = tk.Frame(grid_card, bg=_sbg)
        self._royale_grid.pack(fill=tk.X)

        like = self._start_section_card(outer, "Likeness score",
                                        "Pick a training image of your subject — Fizgig scores each epoch's face "
                                        "against it (ArcFace, CPU) and highlights the closest match in gold.")
        like.columnconfigure(1, weight=1)
        lr = 0
        ttk.Label(like, text="Subject image:").grid(row=lr, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.royale_like_ref_var = tk.StringVar(value=self.last_used.get("royale_like_ref", ""))
        self.royale_like_ref_var.trace_add("write", lambda *a: self._save_last_used_paths())
        _lrr = tk.Frame(like, bg=_sbg); _lrr.grid(row=lr, column=1, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Entry(_lrr, textvariable=self.royale_like_ref_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(_lrr, text="Browse…", command=self._royale_browse_like_ref).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(_lrr, text="Clear", command=lambda: self.royale_like_ref_var.set("")).pack(side=tk.LEFT, padx=(4, 0))
        lr += 1
        _lbr = tk.Frame(like, bg=_sbg); _lbr.grid(row=lr, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        self._royale_score_btn = tk.Button(_lbr, text="Score likeness", font=(FONT_FAMILY, 10, "bold"),
                                           fg="#FFFFFF", bg="#3A6EA5", activeforeground="#FFFFFF",
                                           activebackground="#2F5A86", relief="flat", bd=0, padx=18, pady=5,
                                           cursor="hand2", command=self._royale_score_likeness)
        self._royale_score_btn.pack(side=tk.LEFT)
        self._royale_jump_best_btn = ttk.Button(_lbr, text="Jump to best", command=self._royale_jump_best,
                                                state="disabled")
        self._royale_jump_best_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._royale_like_export_btn = tk.Button(_lbr, text="Export likeness clip…", font=(FONT_FAMILY, 10, "bold"),
                                                 fg="#FFFFFF", bg="#8E44AD", activeforeground="#FFFFFF",
                                                 activebackground="#763A91", relief="flat", bd=0, padx=18, pady=5,
                                                 cursor="hand2", command=self._royale_export_likeness)
        self._royale_like_export_btn.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(self._royale_like_export_btn,
                "Score likeness first, then export a side-by-side clip: your subject image next to\n"
                "each epoch with its likeness score, morphing epoch by epoch. Uses the Format /\n"
                "Speed / Loop / Fizgig-tag settings from the 'Export the morph' card above.")
        self.royale_like_status_var = tk.StringVar(value="")
        tk.Label(_lbr, textvariable=self.royale_like_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))

        promote = self._start_section_card(outer, "Promote winner",
                                           "Copy the epoch currently shown on the crossfade to a new .safetensors "
                                           "you can drop straight into ComfyUI.")
        _pbr = tk.Frame(promote, bg=_sbg); _pbr.pack(anchor=tk.W)
        self._royale_promote_btn = tk.Button(_pbr, text="Promote current epoch…", font=(FONT_FAMILY, 10, "bold"),
                                             fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF",
                                             activebackground="#256F46", relief="flat", bd=0, padx=18, pady=5,
                                             cursor="hand2", command=self._royale_promote)
        self._royale_promote_btn.pack(side=tk.LEFT)
        self.royale_promote_status_var = tk.StringVar(value="")
        tk.Label(_pbr, textvariable=self.royale_promote_status_var, font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=_sbg).pack(side=tk.LEFT, padx=(12, 0))

        # Final display order. _royale_apply_mode packs these (skipping the folder-only
        # cards in Single-LoRA mode), so this list IS the canonical order.
        # NOTE: every card built above MUST appear here — this list is what gets packed.
        # A card left out keeps its build-time slot while these re-pack after it, so it
        # jumps to the very top (above Setup) on the first mode apply.
        self._royale_cards_in_order = [setup, cf, exp, grid_card, like, trav, ptrav, ltrav, cmpc, promote]
        # Cards that only make sense with a folder of epochs (hidden in Single-LoRA mode).
        self._royale_folder_only_cards = {cf, exp, grid_card, like, promote}
        self._royale_apply_mode()

        # Apply the persisted family (krea2 hides the Klein-only reference-latent knobs).
        self._apply_royale_family_ui(str(self.royale_family_var.get()) == "krea2")

        # Scan the pre-filled output folder so the count shows on first open.
        try:
            self._royale_scan()
        except Exception:
            pass

    def _royale_browse_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select training output folder",
                                    initialdir=self.royale_folder_var.get() or self.settings.get("LORA_OUTPUT_DIR", ""))
        if d:
            self.royale_folder_var.set(d)

    def _royale_browse_single(self):
        from tkinter import filedialog
        cur = self.royale_single_var.get()
        init = os.path.dirname(cur) if cur else self.settings.get("LORA_OUTPUT_DIR", "")
        p = filedialog.askopenfilename(title="Select a LoRA .safetensors", initialdir=init,
                                       filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
        if p:
            self.royale_single_var.set(p)

    def _royale_apply_mode(self, *_):
        """Show the folder picker + epoch-comparison cards (Crossfade, Export, All
        epochs, Likeness, Promote) in folder mode; in Single-LoRA mode swap to the
        single-file picker and hide all of those, leaving just the travel/export tools."""
        single = (self.royale_mode_var.get() == "single")
        if single:
            for w in (self._royale_folder_lbl, self._royale_folder_row,
                      self._royale_scan_lbl, self._royale_render_row):
                w.grid_remove()
            self._royale_single_lbl.grid(); self._royale_single_row.grid()
        else:
            self._royale_single_lbl.grid_remove(); self._royale_single_row.grid_remove()
            for w in (self._royale_folder_lbl, self._royale_folder_row,
                      self._royale_scan_lbl, self._royale_render_row):
                w.grid()
        # The 'use rendered epoch as reference' toggle is meaningless in Single-LoRA
        # mode (nothing renders epochs) — hide it and force it OFF so the travel cards
        # use the file/Setup reference (Browse) instead.
        for cb, row, var, toggle in (
            (getattr(self, "_royale_travel_useepoch_cb", None), getattr(self, "_royale_travel_ref_row", None),
             getattr(self, "royale_travel_use_epoch_ref_var", None), getattr(self, "_royale_travel_toggle_ref_widgets", None)),
            (getattr(self, "_royale_pt_useepoch_cb", None), getattr(self, "_royale_pt_ref_row", None),
             getattr(self, "royale_pt_use_epoch_ref_var", None), getattr(self, "_royale_pt_toggle_ref_widgets", None)),
        ):
            if cb is None:
                continue
            cb.pack_forget()
            if single:
                if var is not None and var.get():
                    var.set(False)
            else:
                slaves = row.pack_slaves() if row is not None else []
                if slaves:
                    cb.pack(side=tk.LEFT, before=slaves[0])
                else:
                    cb.pack(side=tk.LEFT)
            if toggle is not None:
                toggle()          # re-grey the file-ref row to match use-epoch state

        # Re-pack the cards in canonical order, skipping folder-only ones when single.
        for content in self._royale_cards_in_order:
            content.master.master.pack_forget()
        for content in self._royale_cards_in_order:
            if single and content in self._royale_folder_only_cards:
                continue
            content.master.master.pack(fill=tk.X, padx=36, pady=(0, 16))
        self._save_last_used_paths()

    def _royale_browse_ref(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(title="Reference image",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
                                       initialdir=self._pref_initialdir("input_ref_dir"))
        if p:
            self.royale_ref_var.set(p)

    def _royale_pt_browse_ref(self):
        from tkinter import filedialog
        init = self.royale_pt_ref_var.get() or self.royale_ref_var.get() or self._pref_initialdir("input_ref_dir")
        p = filedialog.askopenfilename(title="Prompt-travel reference image",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
                                       initialdir=init if os.path.isdir(init) else os.path.dirname(init) if init else "")
        if p:
            self.royale_pt_ref_var.set(p)

    def _royale_travel_browse_ref(self):
        from tkinter import filedialog
        init = self.royale_travel_ref_var.get() or self.royale_ref_var.get() or self._pref_initialdir("input_ref_dir")
        p = filedialog.askopenfilename(title="Seed-travel reference image",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
                                       initialdir=init if os.path.isdir(init) else os.path.dirname(init) if init else "")
        if p:
            self.royale_travel_ref_var.set(p)

    def _royale_toggle_ref_widgets(self, entry, browse, clear, use_epoch):
        """Grey out a file-reference row (entry/browse/clear) when 'use rendered
        epoch' is ticked for that travel card."""
        state = "disabled" if use_epoch else "normal"
        try:
            entry.configure(state=("disabled" if use_epoch else "readonly"))
        except Exception:
            pass
        for w in (browse, clear):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _royale_pt_toggle_ref_widgets(self):
        self._royale_toggle_ref_widgets(self._royale_pt_ref_entry, self._royale_pt_ref_browse,
                                        self._royale_pt_ref_clear,
                                        bool(self.royale_pt_use_epoch_ref_var.get()))

    def _royale_pt_current_preset(self):
        import sys, os
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import prompt_travel as pt
        return pt, pt.PRESETS_BY_NAME.get(self.royale_pt_preset_var.get())

    def _royale_pt_apply_preset(self):
        """Fill the Prompt-travel card from a built-in preset. Jumps Subject to the
        preset's sensible default (e.g. Age → Female), composes the prompt into the
        editable box, and sets all the reference/seed/interp knobs to tuned values."""
        pt, preset = self._royale_pt_current_preset()
        if not preset:
            return
        self.royale_pt_subject_var.set(preset.get("subject", "Woman"))
        self.royale_pt_prompt_var.set(
            pt.fill_subject(preset["prompt"], self.royale_pt_subject_var.get()))
        self.royale_pt_dim_var.set(preset["dim"])
        self.royale_pt_interp_var.set(preset["interp"])
        self.royale_pt_drift_var.set(preset["drift"])
        self.royale_pt_vary_seed_var.set(bool(preset["vary_seed"]))
        self.royale_pt_seq_ref_var.set(bool(preset["sequential"]))
        self.royale_pt_anchor_var.set(bool(preset["anchor"]))
        self.royale_pt_anchor_str_var.set(preset["anchor_str"])
        self.royale_pt_ref_strength_var.set(preset["ref_strength"])
        self.royale_pt_ref_mp_var.set(preset["ref_mp"])
        # Reflect dependent UI state (dimension waypoints, drift enable).
        self._royale_pt_refresh_range()
        self._royale_pt_refresh_words()
        self._royale_pt_sync_seed_widgets()

    def _royale_pt_apply_subject(self):
        """Subject changed — re-compose the prompt text from the selected preset's
        template (knobs untouched) and refresh the waypoints, since some dimensions
        (Age) have subject-specific waypoints (a car has no 'baby' stage)."""
        pt, preset = self._royale_pt_current_preset()
        self._save_last_used_paths()
        if preset:
            self.royale_pt_prompt_var.set(
                pt.fill_subject(preset["prompt"], self.royale_pt_subject_var.get()))
        # Waypoints can depend on the subject — refresh regardless of preset.
        self._royale_pt_refresh_range()
        self._royale_pt_refresh_words()

    def _royale_pt_sync_seed_widgets(self):
        """'Seed drift' only applies while 'Vary seed' is OFF — it's the smooth
        alternative to a per-frame seed walk, so disable it when Vary seed is on."""
        on = bool(self.royale_pt_vary_seed_var.get())
        if hasattr(self, "_royale_pt_drift_entry"):
            try:
                self._royale_pt_drift_entry.configure(state=("disabled" if on else "normal"))
            except Exception:
                pass

    def _royale_travel_toggle_ref_widgets(self):
        self._royale_toggle_ref_widgets(self._royale_travel_ref_entry, self._royale_travel_ref_browse,
                                        self._royale_travel_ref_clear,
                                        bool(self.royale_travel_use_epoch_ref_var.get()))

    def _royale_current_epoch_image(self):
        """PIL image of the epoch the crossfade is parked on, or None."""
        if not self._royale_images:
            return None
        idx = int(round(float(self.royale_scrub_var.get())))
        idx = max(0, min(idx, len(self._royale_images) - 1))
        return self._royale_images[idx][1]

    def _royale_epoch_ref_tempfile(self):
        """Save the parked epoch's render to a temp PNG and return its path
        (so it can be used as edit-conditioning reference), or '' on failure.

        The filename encodes the render generation + epoch label so the path
        changes whenever the underlying image does — otherwise the engine's
        ref-token cache (keyed on path) would reuse a stale earlier render."""
        imgs = getattr(self, "_royale_images", None)
        if not imgs:
            return ""
        idx = int(round(float(self.royale_scrub_var.get())))
        idx = max(0, min(idx, len(imgs) - 1))
        label, img = imgs[idx]
        import tempfile, re
        gen = getattr(self, "_royale_render_gen", 0)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
        tmp = os.path.join(tempfile.gettempdir(), f"fizgig_royale_epoch_ref_{gen}_{safe}.png")
        try:
            img.save(tmp)
            return tmp
        except Exception:
            return ""

    def _royale_resolve_travel_ref(self, use_epoch, file_ref):
        """Resolve a travel card's reference to a path: the parked-epoch render
        if `use_epoch`, else the card's file ref, else the Setup reference."""
        if use_epoch:
            return self._royale_epoch_ref_tempfile()
        return (file_ref or "").strip() or self.royale_ref_var.get().strip()

    @staticmethod
    def _royale_parse_ref_strength(text, default=1.0):
        """Parse a reference-strength entry, clamped to [0, 2]."""
        try:
            return max(0.0, min(2.0, float(text)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _royale_parse_drift(text, default=0.0):
        """Parse a seed-drift amount, clamped to [0, 1] (0 = static seed)."""
        try:
            return max(0.0, min(1.0, float(text)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _royale_parse_ref_mp(text, default=0.2):
        """Parse a reference max-megapixels entry, clamped to [0.05, 1.0]."""
        try:
            return max(0.05, min(1.0, float(text)))
        except (TypeError, ValueError):
            return default

    def _royale_apply_travel_ref(self, st, p, i):
        """Set the reference(s) on a travel frame's SliderState and return the strength
        to apply to the PREVIOUS frame's latent (0.0 when this frame uses no prev latent).

        Non-sequential: the selected reference on every frame; no prev latent.
        Sequential: frame 0 uses the selected reference. Later frames edit the PREVIOUS
        frame via its cached clean latent (no VAE decode→PNG→encode round trip — the
        worker passes RepairEngine._last_frame_latent as prev_latent), which is what
        keeps the chain sharp at high strength. With 'Anchor to original', later frames
        ALSO keep the original as a clean image reference (identity anchor):
        original = image ref, previous = latent ref."""
        import os
        st.ref_megapixels = p.get("ref_mp", 0.2)
        st.ref2_path = ""
        orig = p.get("ref", "")
        orig_ok = orig if (orig and os.path.exists(orig)) else ""
        if p.get("sequential") and i > 0:
            if p.get("anchor"):
                # Dual reference: original (clean image anchor) + previous frame (latent).
                st.ref_image_path = orig_ok
                st.ref_strength = p.get("anchor_str", 1.0)
            else:
                # Previous frame only — supplied as the latent ref by the worker.
                st.ref_image_path = ""
                st.ref_strength = 0.0
            return float(p.get("ref_strength", 1.0))
        # Frame 0 or non-sequential: the selected image reference, no prev latent.
        st.ref_image_path = orig_ok
        st.ref_strength = p.get("ref_strength", 1.0)
        return 0.0

    @staticmethod
    def _royale_label_disp(label):
        """Display text for a checkpoint label: 'Epoch N' for trainer epochs, or the
        LoRA's filename for arbitrary (non-epoch) LoRAs."""
        s = str(label)
        return f"Epoch {s}" if s.isdigit() else s

    def _royale_scan(self):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale.scan import scan_checkpoints
        cps = scan_checkpoints(self.royale_folder_var.get().strip())
        self._royale_checkpoints = cps
        if cps:
            labels = [e for e, _ in cps]
            if all(str(x).isdigit() for x in labels):
                self.royale_scan_var.set(f"{len(cps)} checkpoints found (epochs {labels[0]}–{labels[-1]}).")
            else:
                self.royale_scan_var.set(f"{len(cps)} LoRAs found — compared by filename.")
        else:
            self.royale_scan_var.set("No .safetensors checkpoints found in this folder.")

    def _royale_family(self):
        fam = str(getattr(self, "royale_family_var", None) and self.royale_family_var.get())
        return fam if fam in ("klein", "krea2", "minimax") else "klein"

    def _royale_is_krea2(self):
        return self._royale_family() == "krea2"

    def _royale_default_state(self):
        """A default SliderState for the active family (block ids differ: Klein double/single,
        Krea 2 block_/txt_, H3 h3blk_/h3_rf_)."""
        from fizgig.repair_studio.state import SliderState
        fam = self._royale_family()
        return (SliderState.default_krea2() if fam == "krea2"
                else SliderState.default_h3() if fam == "minimax"
                else SliderState.default_klein9b())

    def _on_royale_family_changed(self):
        """Family toggle: the engine type changes, so unload any loaded engine + clear rendered
        frames, then persist."""
        fam = self._royale_family()
        self.last_used["royale_family"] = fam
        self._save_last_used_paths()
        if self._royale_is_busy():
            return
        self._royale_unload()
        self.royale_engine = None
        self._apply_royale_family_ui(fam != "klein")
        _names = {"krea2": "Krea 2 (Turbo previews)",
                  "minimax": "MiniMax H3 (22-frame clip previews)"}
        self.royale_status_var.set(
            f"Switched to {_names.get(fam, 'Klein 9B (Distilled previews)')}. "
            f"Pick a source and render.")

    def _apply_royale_family_ui(self, is_krea2):
        """Krea 2's reference goes through the Qwen3-VL vision path — a static per-frame anchor with
        no strength dial and no latent feedback chain. So hide the Klein-only reference-latent knobs
        (ref Strength on Setup + both travel cards, the Sequential-reference checkboxes, and the
        prompt-travel Anchor-strength) in Krea 2 mode. The reference picker, Max MP and 'Anchor to
        original' (which maps to 'keep the original as each frame's vision reference') stay.

        Specs are (widget_attr, pack_kwargs, before_anchor_attr) restored verbatim on the klein
        path; ordered so each before-anchored group re-packs in its original left-to-right order."""
        specs = [
            ("_royale_ref_strength_lbl", dict(side=tk.LEFT, padx=(8, 3)), None),
            ("_royale_ref_strength_entry", dict(side=tk.LEFT), None),
            ("_royale_travel_ref_strength_lbl", dict(side=tk.LEFT, padx=(16, 3)), "_royale_travel_maxmp_lbl"),
            ("_royale_travel_ref_strength_entry", dict(side=tk.LEFT), "_royale_travel_maxmp_lbl"),
            ("_royale_travel_seq_cb", dict(side=tk.LEFT, padx=(16, 0)), "_royale_travel_maxmp_lbl"),
            ("_royale_pt_ref_strength_lbl", dict(side=tk.LEFT, padx=(16, 3)), "_royale_pt_maxmp_lbl"),
            ("_royale_pt_ref_strength_entry", dict(side=tk.LEFT), "_royale_pt_maxmp_lbl"),
            ("_royale_pt_seq_cb", dict(side=tk.LEFT, padx=(16, 0)), "_royale_pt_maxmp_lbl"),
            ("_royale_pt_anchor_str_lbl", dict(side=tk.LEFT, padx=(8, 3)), None),
            ("_royale_pt_anchor_str_entry", dict(side=tk.LEFT), None),
        ]
        for attr, kwargs, anchor_attr in specs:
            w = getattr(self, attr, None)
            if w is None:
                continue
            if is_krea2:
                w.pack_forget()
            elif w.winfo_manager() == "":
                anchor = getattr(self, anchor_attr, None) if anchor_attr else None
                kw = dict(kwargs)
                if anchor is not None and anchor.winfo_manager() != "":
                    kw["before"] = anchor
                w.pack(**kw)
        # Krea 2 has no sequential latent chain — force the (now-hidden) flags off so each travel
        # frame takes the non-sequential path in _royale_apply_travel_ref (the selected image as a
        # per-frame vision reference) instead of dropping the reference for a prev-latent that the
        # Krea 2 engine ignores.
        if is_krea2:
            for v in ("royale_travel_seq_ref_var", "royale_pt_seq_ref_var"):
                var = getattr(self, v, None)
                if var is not None:
                    var.set(False)

    def _royale_validate_models(self):
        """Fast main-thread pre-flight before a render: verify the model paths exist,
        make sure the engine object exists, and stash the pipeline kwargs for the worker
        to load with. Does NOT load anything (no blocking) — the worker thread does the
        heavy load via _royale_ensure_pipeline_loaded(). Shows a messagebox + returns
        False on a missing path."""
        if self._royale_is_krea2():
            return self._royale_validate_models_krea2()
        if self._royale_family() == "minimax":
            return self._royale_validate_models_h3()
        dit_path = self.prefs_vars["distilled_dit"].get() if "distilled_dit" in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")
        for label, p in (("Distilled DiT", dit_path), ("VAE", vae_path), ("Text encoder", te_path)):
            if not p or not os.path.exists(p):
                messagebox.showerror("Error", f"{label} path not set or not found.\nConfigure on Preferences tab.")
                return False
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.engine import RepairEngine
        if not isinstance(self.royale_engine, RepairEngine):
            self.royale_engine = RepairEngine()      # cheap constructor — no model load
        is_fp8 = "fp8" in os.path.basename(dit_path).lower()
        # Stash kwargs so the worker thread can load (and rebuild after reset() on a
        # different-rank swap) without touching Tk.
        self._royale_pipeline_kwargs = dict(
            dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            model_version="klein-9b", device="cuda",
            fp8_scaled=False if is_fp8 else True,
            blocks_to_swap=self._get_inference_blocks_to_swap(),
            int8=self._get_inference_int8())
        return True

    def _royale_validate_models_krea2(self):
        """Krea 2 pre-flight: the fp8 Turbo + Qwen-Image VAE + bf16 Qwen3-VL TE from Preferences,
        and a Krea2RepairEngine. Stashes Krea2-shaped pipeline kwargs for the worker."""
        dit_path = self.prefs_vars.get("krea2_turbo_dit", tk.StringVar()).get()
        vae_path = self.prefs_vars.get("krea2_vae", tk.StringVar()).get()
        te_path = self.prefs_vars.get("krea2_text_encoder", tk.StringVar()).get()
        for label, p in (("Krea 2 Turbo DiT", dit_path), ("Qwen-Image VAE", vae_path),
                         ("Qwen3-VL TE (bf16)", te_path)):
            if not p or not os.path.exists(p):
                messagebox.showerror("Error", f"{label} path not set or not found.\nConfigure on Preferences tab.")
                return False
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.krea2_engine import Krea2RepairEngine
        if not isinstance(self.royale_engine, Krea2RepairEngine):
            self.royale_engine = Krea2RepairEngine()
        self._royale_pipeline_kwargs = dict(
            turbo_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            device="cuda", model_kind="turbo",
            blocks_to_swap=self._auto_krea2_inference_blocks_swap(),
            int8=self._get_inference_int8())
        return True

    def _royale_validate_models_h3(self):
        """MiniMax H3 pre-flight: the H3 DiT + video VAE + Qwen3-VL-32B TE from Preferences,
        an H3RepairEngine, and H3-shaped pipeline kwargs (auto-planned base + Turbo LoRA +
        prompt disk cache — same recipe as the Repair Studio). Epoch previews render a
        22-frame clip's middle frame."""
        dit_path = self.prefs_vars.get("minimax_dit", tk.StringVar()).get()
        vae_path = self.prefs_vars.get("minimax_vae", tk.StringVar()).get()
        te_path = self.prefs_vars.get("minimax_text_encoder", tk.StringVar()).get()
        for label, p in (("MiniMax H3 DiT", dit_path), ("MiniMax H3 video VAE", vae_path),
                         ("Qwen3-VL-32B text encoder", te_path)):
            if not p or not os.path.exists(p):
                messagebox.showerror("Error", f"{label} path not set or not found.\nConfigure on Preferences tab.")
                return False
        turbo_path = self.prefs_vars.get("minimax_turbo_lora", tk.StringVar()).get().strip()
        cache_dir = self.prefs_vars.get("cache_dir", tk.StringVar()).get().strip()
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.h3_engine import H3RepairEngine
        if not isinstance(self.royale_engine, H3RepairEngine):
            self.royale_engine = H3RepairEngine()
        self._royale_pipeline_kwargs = dict(
            dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
            device="cuda", turbo_lora_path=turbo_path, turbo_lora_strength=0.75,
            te_cache_dir=os.path.join(cache_dir, "te_prompts") if cache_dir else "")
        return True

    def _royale_is_krea2_engine(self):
        """True if the current royale_engine is a Krea2RepairEngine (so we know to rebuild it on a
        family switch)."""
        eng = getattr(self, "royale_engine", None)
        return eng is not None and type(eng).__name__ == "Krea2RepairEngine"

    def _royale_check_lora_families(self, *paths):
        """Main-thread pre-flight for issue #62: auto-follow the selector to match the batch's
        detected family BEFORE any worker thread starts, so picking a file never means eating
        the 9 GB pipeline load just to be told it doesn't match. Uses the exact same
        _on_royale_family_changed() a manual radio click would, so the engine ends up in the
        same state either way, and the switch is visible (radio button + status line) rather
        than silent. Call BEFORE _royale_validate_models(), not after — a switch here resets
        royale_engine to None, and _royale_validate_models() is what recreates it (and stashes
        the family-shaped pipeline kwargs); reversing the order leaves the worker thread calling
        ensure_pipeline() on a None engine with kwargs stashed for the family being switched
        away from. Both run before the threading.Thread(...).start() that kicks off the heavy
        work — _royale_load_or_swap_primary still carries a raise-based version of this check
        for mid-run epoch swaps on the worker thread, where touching Tk state isn't safe.

        Accepts multiple paths (e.g. a whole epoch selection). The target family is decided from
        the WHOLE batch first — not path-by-path, which would let an already-matching first path
        mask a mismatched later one when a switch fires mid-loop — so a genuinely mixed-family
        selection is always caught rather than depending on which order the paths happen to be
        in. None paths are skipped so callers can pass tuples straight from
        _royale_current_epoch(). Only shows a dialog when it can't resolve things automatically
        (a second family in the same selection); an unrecognized file is simply ignored."""
        from fizgig.networks.lora import lora_family_from_file, FAMILY_DISPLAY_NAMES
        seen = []          # (path, family) for every path with a determinable family
        checked = set()
        for path in paths:
            if not path or path in checked:
                continue
            checked.add(path)
            detected = lora_family_from_file(path)
            if detected is not None:
                seen.append((path, detected))
        if not seen:
            return True  # nothing recognizable — fail open, let the real loader be the judge
        target_path, target = seen[0]
        for path, detected in seen[1:]:
            if detected != target:
                messagebox.showerror(
                    "Mixed LoRA families",
                    f"{os.path.basename(path)} was trained for {FAMILY_DISPLAY_NAMES.get(detected, detected)}, "
                    f"but this selection also includes {FAMILY_DISPLAY_NAMES.get(target, target)} files "
                    f"(e.g. {os.path.basename(target_path)}). Pick LoRAs from a single family.")
                return False
        from fizgig.networks.lora import INFERENCE_FAMILIES
        if target not in INFERENCE_FAMILIES:
            messagebox.showerror(
                "Unsupported family",
                f"{os.path.basename(target_path)} was trained for "
                f"{FAMILY_DISPLAY_NAMES.get(target, target)}, but Royale doesn't support "
                f"{FAMILY_DISPLAY_NAMES.get(target, target)} LoRAs yet.")
            return False
        selected = str(self.royale_family_var.get())
        if target != selected:
            target_name = FAMILY_DISPLAY_NAMES.get(target, target)
            self.royale_family_var.set(target)
            self._on_royale_family_changed()
            self.royale_status_var.set(
                f"Switched family selector to {target_name} to match {os.path.basename(target_path)}.")
        return True

    def _royale_ensure_pipeline_loaded(self):
        """Worker-thread: load the preview pipeline if it isn't already. No Tk calls;
        raises on failure (callers route it to their finish handler). The same load
        already runs on a worker thread in _royale_load_or_swap_primary, so it's
        proven-safe off the main thread."""
        eng = self.royale_engine
        if eng is not None and eng.pipeline is not None and eng.pipeline.is_loaded:
            return
        eng.ensure_pipeline(**self._royale_pipeline_kwargs)

    def _royale_load_or_swap_primary(self, eng, path):
        """Point `eng` at `path`. Fast in-place weight swap when the structure matches
        (same-rank epochs); otherwise reset() + rebuild the pipeline + load_primary so a
        different-rank / unrelated LoRA loads cleanly. Worker-thread safe (no Tk).

        Checks the file's family against the selector BEFORE any of that (issue #62):
        every Royale entry point (render, comparison, seed-travel, LoRA-strength-travel)
        already calls this one function inside its own try/except, so one guard here
        covers all of them without touching each worker."""
        from fizgig.networks.lora import assert_lora_family_matches
        assert_lora_family_matches(path, str(self.royale_family_var.get()), "Royale")
        if eng.primary_network is None:
            eng.load_primary(path)
            return
        if eng.primary_path == path:
            return
        if eng.swap_primary_weights(path):
            return
        eng.reset()
        eng.ensure_pipeline(**self._royale_pipeline_kwargs)
        eng.load_primary(path)

    def _royale_unload(self):
        # Same internal guard as the Repair/Explorer unloads: resetting under a live CUDA
        # worker hard-hangs, and callers (tab switch, pre-training hygiene) can't know.
        if self._royale_is_busy():
            return
        if getattr(self, "royale_engine", None) is not None:
            try:
                self.royale_engine.reset()
            except Exception:
                pass

    def _royale_render(self):
        if self._royale_is_busy():
            return
        cps = self._royale_checkpoints
        if not cps:
            messagebox.showinfo("LoRA Royale", "No checkpoints found — pick a folder with trained LoRAs.")
            return
        prompt = self.royale_prompt_var.get().strip()
        if not prompt:
            messagebox.showinfo("LoRA Royale", "Enter a prompt (include your trigger word).")
            return
        # Subset to 'Max renders', evenly spaced across the run (always keep first + last).
        sel = cps
        mx = self.royale_max_var.get()
        if mx != "All":
            n = int(mx)
            if len(cps) > n:
                idx = sorted({round(i * (len(cps) - 1) / (n - 1)) for i in range(n)})
                sel = [cps[i] for i in idx]
        if not self._royale_check_lora_families(*[p for _, p in sel]):
            return
        if not self._royale_validate_models():
            return
        self._royale_rendering = True
        self._royale_render_btn.configure(state="disabled")
        self.royale_status_var.set("Loading model…")
        import threading
        threading.Thread(target=self._royale_render_worker, args=(sel, prompt), daemon=True).start()

    def _royale_release_vram(self):
        """Return the CUDA allocator's reserved-but-unused VRAM to the driver after a
        render. The Klein pipeline stays resident (model params aren't freed), but the
        transient working set (encode/decode spikes, intermediate tensors) is released
        so the Windows desktop compositor and hardware video decoders can get surfaces
        again. Without this, after several renders unrelated videos play as a black
        frame until the app frees VRAM (e.g. on tab switch) — the allocator was holding
        the spike memory the OS needed for decode surfaces."""
        try:
            eng = getattr(self, "royale_engine", None)
            if eng is None or eng.pipeline is None:
                return
            from fizgig.utils.device import clean_memory_on_device
            clean_memory_on_device(eng.pipeline.device)
        except Exception:
            pass

    def _royale_render_worker(self, sel, prompt):
        from fizgig.repair_studio.state import SliderState
        try:
            self._royale_ensure_pipeline_loaded()      # heavy load, off the main thread
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_render_load_failed(e))
            return
        try:
            seed = int(self.royale_seed_var.get() or "42")
        except ValueError:
            seed = 42
        try:
            width = int(self.royale_w_var.get()); height = int(self.royale_h_var.get())
        except ValueError:
            width = height = 512
        ref = self.royale_ref_var.get().strip()
        ref_strength = self._royale_parse_ref_strength(self.royale_ref_strength_var.get())
        results = []
        paths = {}
        eng = self.royale_engine
        total = len(sel)
        for i, (label, path) in enumerate(sel):
            self.master.after(0, lambda l=label, i=i: self.royale_status_var.set(
                f"Rendering {self._royale_label_disp(l)} ({i + 1}/{total})…"))
            try:
                # First-ever load patches the DiT; everything after (including a
                # re-kicked render that reuses the loaded engine) swaps weights.
                self._royale_load_or_swap_primary(eng, path)
                st = self._royale_default_state()
                st.prompt = prompt
                st.seed = seed
                st.preview_width = width
                st.preview_height = height
                if ref and os.path.exists(ref):
                    st.ref_image_path = ref
                    st.ref_megapixels = 0.2
                    st.ref_strength = ref_strength
                img = eng.generate_preview(st)
                results.append((label, img.copy()))
                paths[label] = path
            except Exception:
                import traceback
                print(f"[royale] render failed for {path}:\n{traceback.format_exc()}")
        self._royale_release_vram()
        self.master.after(0, lambda: self._royale_finish(results, paths))

    def _royale_render_load_failed(self, err):
        """Pipeline load failed on the render worker thread — re-enable the button and
        report (main thread)."""
        self._royale_rendering = False
        self._royale_render_btn.configure(state="normal")
        self.royale_status_var.set("Failed to load models — see console.")
        messagebox.showerror("Error", f"Failed to load models:\n{err}")

    def _royale_finish(self, results, paths=None):
        self._royale_rendering = False
        self._royale_render_btn.configure(state="normal")
        self._royale_images = results
        self._royale_paths = paths or {}
        self._royale_render_gen += 1   # new images -> epoch-ref temp files get a fresh path
        # New renders invalidate any prior likeness scores.
        self._royale_scores = {}
        self._royale_best_label = None
        if hasattr(self, "royale_like_status_var"):
            self.royale_like_status_var.set("")
        if not results:
            self.royale_status_var.set("No renders produced — see console.")
            return
        self.royale_status_var.set(f"Rendered {len(results)} epochs. Drag the crossfade slider, or click a thumbnail.")
        self._royale_scale.configure(to=float(len(results) - 1))
        self.royale_scrub_var.set(0.0)
        self._royale_fit_holder()
        self._royale_build_grid()
        self._royale_scrub()

    def _royale_scrub(self):
        from PIL import Image, ImageTk
        imgs = self._royale_images
        if not imgs:
            return
        p = float(self.royale_scrub_var.get())
        lo = max(0, min(int(p), len(imgs) - 1))
        hi = min(lo + 1, len(imgs) - 1)
        alpha = p - lo
        a_label, a_img = imgs[lo]
        b_label, b_img = imgs[hi]
        if alpha <= 0.01 or lo == hi:
            blended = a_img
            self.royale_scrub_label_var.set(self._royale_label_disp(a_label))
        else:
            if a_img.size != b_img.size:
                b_img = b_img.resize(a_img.size)
            blended = Image.blend(a_img, b_img, alpha)
            self.royale_scrub_label_var.set(
                f"{self._royale_label_disp(a_label)}  →  {self._royale_label_disp(b_label)}    ({alpha:.0%})")
        hw, hh = getattr(self, "_royale_holder_box", (512, 512))
        disp = blended.copy()
        disp.thumbnail((max(64, hw), max(64, hh)), Image.LANCZOS)
        self._royale_preview_imgtk = ImageTk.PhotoImage(disp)
        self._royale_preview_label.configure(image=self._royale_preview_imgtk, text="")

    def _royale_fit_holder(self):
        """Size the crossfade preview holder to the rendered image's aspect
        (max 512 on the long side), so portrait/landscape renders aren't
        letterboxed in a square box."""
        imgs = getattr(self, "_royale_images", None)
        m = 512
        if not imgs:
            box = (m, m)
        else:
            iw, ih = imgs[0][1].size
            if iw >= ih:
                box = (m, max(64, round(m * ih / iw)))
            else:
                box = (max(64, round(m * iw / ih)), m)
        self._royale_holder_box = box
        try:
            self._royale_holder.configure(width=box[0], height=box[1])
        except Exception:
            pass

    def _royale_build_grid(self):
        from PIL import Image, ImageTk
        for w in self._royale_grid.winfo_children():
            w.destroy()
        self._royale_thumbs = []
        cols = 10
        scores = getattr(self, "_royale_scores", {})
        best = getattr(self, "_royale_best_label", None)
        for i, (label, img) in enumerate(self._royale_images):
            t = img.copy()
            t.thumbnail((92, 92), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(t)
            self._royale_thumbs.append(tk_img)
            row = (i // cols) * 2
            col = i % cols
            is_best = (best is not None and label == best)
            bd = 3 if is_best else 0
            holder = tk.Frame(self._royale_grid, bg=("#FFD24A" if is_best else COLORS["bg_surface"]),
                              padx=bd, pady=bd)
            holder.grid(row=row, column=col, padx=4, pady=(4, 0))
            lbl = tk.Label(holder, image=tk_img, cursor="hand2", bg=COLORS["bg_surface"])
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, idx=i: (self.royale_scrub_var.set(float(idx)), self._royale_scrub()))
            # Caption: epoch number (or filename for arbitrary LoRAs) + likeness score.
            base = f"e{label}" if str(label).isdigit() else str(label)[:16]
            cap = base
            sc = scores.get(label)
            if sc is not None:
                cap = f"{base}  {sc:.2f}" if sc == sc else f"{base}  —"  # NaN check
            tk.Label(self._royale_grid, text=cap, font=(FONT_FAMILY, 8, "bold" if is_best else "normal"),
                     fg=("#FFD24A" if is_best else COLORS["text_muted"]),
                     bg=COLORS["bg_surface"]).grid(row=row + 1, column=col, pady=(0, 4))

    # ----- Likeness scoring (Phase 3) -----
    def _royale_browse_like_ref(self):
        from tkinter import filedialog
        # Default to the dataset/training image folder if we know it.
        init = (self.image_folder_var.get() if hasattr(self, "image_folder_var") else "") \
            or self._pref_initialdir("input_ref_dir")
        p = filedialog.askopenfilename(title="Subject image (a clean training shot)",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
                                       initialdir=init)
        if p:
            self.royale_like_ref_var.set(p)

    def _royale_score_likeness(self):
        if getattr(self, "_royale_scoring", False):
            return
        if not self._royale_images:
            messagebox.showinfo("LoRA Royale", "Render some epochs first, then score them.")
            return
        ref = self.royale_like_ref_var.get().strip()
        if not ref or not os.path.exists(ref):
            messagebox.showinfo("LoRA Royale", "Pick a subject image to score likeness against.")
            return
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import likeness
        if not likeness.available():
            messagebox.showerror("Likeness scoring unavailable",
                                 "InsightFace / OpenCV aren't installed.\nRun install_fizgig.py to enable face scoring.")
            return
        self._royale_scoring = True
        self._royale_score_btn.configure(state="disabled")
        self.royale_like_status_var.set("Scoring… (first run downloads the face model)")
        import threading
        threading.Thread(target=self._royale_score_worker, args=(ref, list(self._royale_images)),
                         daemon=True).start()

    def _royale_score_worker(self, ref, images):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import likeness
        try:
            scored = likeness.score_renders(ref, images)
        except Exception:
            import traceback
            err = traceback.format_exc()
            print(f"[royale] likeness scoring failed:\n{err}")
            self.master.after(0, lambda: self._royale_score_finish(None))
            return
        self.master.after(0, lambda: self._royale_score_finish(scored))

    def _royale_score_finish(self, scored):
        self._royale_scoring = False
        self._royale_score_btn.configure(state="normal")
        if scored is None:
            self.royale_like_status_var.set("Scoring failed — see console.")
            return
        self._royale_scores = {label: sc for label, sc in scored}
        valid = [(label, sc) for label, sc in scored if sc == sc]  # drop NaN (no face)
        if not valid:
            self._royale_best_label = None
            self._royale_jump_best_btn.configure(state="disabled")
            self.royale_like_status_var.set("No faces detected in the renders or subject image.")
            self._royale_build_grid()
            return
        best_label, best_sc = max(valid, key=lambda t: t[1])
        self._royale_best_label = best_label
        self._royale_jump_best_btn.configure(state="normal")
        n_noface = len(scored) - len(valid)
        extra = f"  ({n_noface} no-face)" if n_noface else ""
        self.royale_like_status_var.set(f"Best: epoch {best_label} ({best_sc:.3f}){extra}")
        self._royale_build_grid()
        # Park the crossfade on the winner so it's selected, not just highlighted.
        self._royale_jump_best()

    def _royale_jump_best(self):
        best = getattr(self, "_royale_best_label", None)
        if best is None:
            return
        for idx, (label, _img) in enumerate(self._royale_images):
            if label == best:
                self.royale_scrub_var.set(float(idx))
                self._royale_scrub()
                break

    def _royale_current_epoch(self):
        """(label, path) for the LoRA the travel tools should use. In Single-LoRA mode
        that's the chosen file; otherwise the epoch the crossfade is parked on."""
        if getattr(self, "royale_mode_var", None) is not None and self.royale_mode_var.get() == "single":
            p = self.royale_single_var.get().strip()
            if p and os.path.exists(p):
                return os.path.splitext(os.path.basename(p))[0], p
            return None, None
        if not self._royale_images:
            return None, None
        idx = int(round(float(self.royale_scrub_var.get())))
        idx = max(0, min(idx, len(self._royale_images) - 1))
        label = self._royale_images[idx][0]
        return label, self._royale_paths.get(label)

    def _royale_promote(self):
        label, path = self._royale_current_epoch()
        if path is None or not os.path.exists(path):
            messagebox.showinfo("LoRA Royale", "Render epochs first, then slide to the one you want to promote.")
            return
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import run_name_for_folder
        run = run_name_for_folder(self.royale_folder_var.get().strip()) or "lora"
        default_name = (f"{run}-epoch{label}-pick.safetensors" if str(label).isdigit()
                        else f"{label}-pick.safetensors")
        from tkinter import filedialog
        out = filedialog.asksaveasfilename(
            title="Promote epoch — save as",
            defaultextension=".safetensors",
            initialfile=default_name,
            initialdir=self.settings.get("LORA_OUTPUT_DIR", "") or os.path.dirname(path),
            filetypes=[("Safetensors", "*.safetensors")])
        if not out:
            return
        try:
            import shutil
            shutil.copy2(path, out)
        except Exception as e:
            messagebox.showerror("Promote failed", f"Could not copy checkpoint:\n{e}")
            return
        self.royale_promote_status_var.set(f"Saved {self._royale_label_disp(label)} → {os.path.basename(out)}")

    def _royale_save_all_stills(self):
        """Save every rendered epoch still (the all-epochs view) to a chosen folder as
        full-res PNGs. Epoch labels are zero-padded so the files sort in render order;
        arbitrary-LoRA labels use the LoRA filename. The render results otherwise live
        only in memory (`self._royale_images`) and are lost on app close.

        Runs on a worker thread (mirrors Export clip) so the blue status label animates
        and the UI stays responsive while a large batch of full-res PNGs is written."""
        if getattr(self, "_royale_exporting", False):
            return
        if not self._royale_images:
            messagebox.showinfo("LoRA Royale", "Render epochs first, then save the stills.")
            return
        from tkinter import filedialog
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import run_name_for_folder
        run = run_name_for_folder(self.royale_folder_var.get().strip()) or "lora"
        folder = filedialog.askdirectory(
            title="Save all epoch stills to folder",
            initialdir=self.settings.get("LORA_OUTPUT_DIR", ""))
        if not folder:
            return
        self._royale_exporting = True
        self._royale_save_stills_btn.configure(state="disabled")
        self._royale_export_btn.configure(state="disabled")
        self.royale_export_status_var.set("Saving stills…")
        images = list(self._royale_images)
        threading.Thread(target=self._royale_save_stills_worker,
                         args=(images, run, folder), daemon=True).start()

    def _royale_save_stills_worker(self, images, run, folder):
        import re, traceback
        saved = 0
        for i, (label, img) in enumerate(images):
            if str(label).isdigit():
                name = f"{run}-epoch{int(label):04d}.png"
            else:
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
                name = f"{run}-{safe}.png"
            try:
                img.save(os.path.join(folder, name))
                saved += 1
            except Exception:
                traceback.print_exc()
            self.master.after(0, lambda i=i: self.royale_export_status_var.set(
                f"Saving stills… {i + 1}/{len(images)}"))
        self.master.after(0, lambda: self._royale_save_stills_finish(saved, len(images), folder))

    def _royale_save_stills_finish(self, saved, total, folder):
        self._royale_exporting = False
        self._royale_save_stills_btn.configure(state="normal")
        self._royale_export_btn.configure(state="normal")
        self.royale_export_status_var.set(
            f"Saved {saved}/{total} stills → {os.path.basename(folder)}")

    # ----- Export the morph as a shareable clip -----
    def _royale_export(self):
        if getattr(self, "_royale_exporting", False):
            return
        if not self._royale_images:
            messagebox.showinfo("LoRA Royale", "Render some epochs first, then export the morph.")
            return
        fmt = self.royale_export_format_var.get().upper()
        ext = ".mp4" if fmt == "MP4" else ".gif"
        from tkinter import filedialog
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import run_name_for_folder
        run = run_name_for_folder(self.royale_folder_var.get().strip()) or "lora"
        out = filedialog.asksaveasfilename(
            title="Export morph clip",
            defaultextension=ext,
            initialfile=f"{run}-royale{ext}",
            initialdir=self.settings.get("LORA_OUTPUT_DIR", ""),
            filetypes=[("MP4 video", "*.mp4")] if fmt == "MP4" else [("Animated GIF", "*.gif")])
        if not out:
            return
        params = dict(
            images=list(self._royale_images),
            fmt=fmt, out=out,
            speed=self.royale_export_speed_var.get(),
            pingpong=bool(self.royale_export_loop_var.get()),
            brand=bool(self.royale_export_wm_var.get()),
            show_epoch=bool(self.royale_export_epoch_var.get()),
        )
        self._royale_exporting = True
        self._royale_export_btn.configure(state="disabled")
        self.royale_export_status_var.set("Building frames…")
        import threading
        threading.Thread(target=self._royale_export_worker, args=(params,), daemon=True).start()

    def _royale_export_worker(self, p):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import export as rexport
        try:
            # MP4 exports at the actual generated resolution; GIF stays capped at 768 (file size).
            max_size = None if p["fmt"] == "MP4" else 768
            frames = rexport.build_frames(p["images"], speed=p["speed"], pingpong=p["pingpong"],
                                          brand=p["brand"], show_epoch=p["show_epoch"], max_size=max_size)
            if not frames:
                raise RuntimeError("No frames were produced.")
            if p["fmt"] == "MP4":
                rexport.write_mp4(frames, p["out"], speed=p["speed"])
            else:
                rexport.write_gif(frames, p["out"], speed=p["speed"])
            self.master.after(0, lambda: self._royale_export_finish(p["out"], len(frames), None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_export_finish(p["out"], 0, e))

    def _royale_export_finish(self, out, n_frames, err):
        self._royale_exporting = False
        self._royale_export_btn.configure(state="normal")
        if err is not None:
            self.royale_export_status_var.set("Export failed — see console.")
            msg = str(err)
            if "codec" in msg.lower() or "writer" in msg.lower():
                msg += "\n\nTry the GIF format instead."
            messagebox.showerror("Export failed", msg)
            return
        self.royale_export_status_var.set(f"Saved {n_frames} frames → {os.path.basename(out)}")
        self._royale_reveal(out)

    # ----- Export the likeness comparison (subject vs each epoch, side by side) -----
    def _royale_export_likeness(self):
        if getattr(self, "_royale_exporting", False):
            return
        if not self._royale_images:
            messagebox.showinfo("LoRA Royale", "Render epochs first.")
            return
        scores = getattr(self, "_royale_scores", None)
        if not scores:
            messagebox.showinfo("LoRA Royale",
                                "Score likeness first — the clip shows each epoch's score beside your subject image.")
            return
        ref = self.royale_like_ref_var.get().strip()
        if not ref or not os.path.exists(ref):
            messagebox.showinfo("LoRA Royale", "Pick a subject image to show alongside the epochs.")
            return
        fmt = self.royale_export_format_var.get().upper()
        ext = ".mp4" if fmt == "MP4" else ".gif"
        from tkinter import filedialog
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import run_name_for_folder
        run = run_name_for_folder(self.royale_folder_var.get().strip()) or "lora"
        out = filedialog.asksaveasfilename(
            title="Export likeness clip",
            defaultextension=ext,
            initialfile=f"{run}-likeness{ext}",
            initialdir=self.settings.get("LORA_OUTPUT_DIR", ""),
            filetypes=[("MP4 video", "*.mp4")] if fmt == "MP4" else [("Animated GIF", "*.gif")])
        if not out:
            return
        params = dict(
            ref=ref, images=list(self._royale_images), scores=dict(scores),
            fmt=fmt, out=out,
            speed=self.royale_export_speed_var.get(),
            pingpong=bool(self.royale_export_loop_var.get()),
            brand=bool(self.royale_export_wm_var.get()))
        self._royale_exporting = True
        self._royale_like_export_btn.configure(state="disabled")
        self.royale_like_status_var.set("Building likeness clip…")
        threading.Thread(target=self._royale_export_likeness_worker, args=(params,), daemon=True).start()

    def _royale_export_likeness_worker(self, p):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import export as rexport
        from PIL import Image
        try:
            # MP4 at full panel res; GIF capped at 768 (file size) — mirrors the morph export.
            max_size = None if p["fmt"] == "MP4" else 768
            with Image.open(p["ref"]) as ri:
                ref_img = ri.convert("RGB")
            frames = rexport.build_likeness_frames(
                ref_img, p["images"], p["scores"], speed=p["speed"],
                pingpong=p["pingpong"], brand=p["brand"], max_size=max_size)
            if not frames:
                raise RuntimeError("No frames were produced.")
            if p["fmt"] == "MP4":
                rexport.write_mp4(frames, p["out"], speed=p["speed"])
            else:
                rexport.write_gif(frames, p["out"], speed=p["speed"])
            self.master.after(0, lambda: self._royale_export_likeness_finish(p["out"], len(frames), None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_export_likeness_finish(p["out"], 0, e))

    def _royale_export_likeness_finish(self, out, n_frames, err):
        self._royale_exporting = False
        self._royale_like_export_btn.configure(state="normal")
        if err is not None:
            self.royale_like_status_var.set("Likeness export failed — see console.")
            msg = str(err)
            if "codec" in msg.lower() or "writer" in msg.lower():
                msg += "\n\nTry the GIF format instead."
            messagebox.showerror("Export failed", msg)
            return
        self.royale_like_status_var.set(f"Saved {n_frames} frames → {os.path.basename(out)}")
        self._royale_reveal(out)

    def _royale_reveal(self, path):
        try:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        except Exception:
            pass

    @staticmethod
    def _royale_journey_seeds(start, end, n):
        """Ordered seed list for a journey: [start, …n-2 deterministic mids…, end].
        Mids derive from `start`, so the same Start seed reproduces the same journey."""
        n = max(2, int(n))
        start, end = int(start), int(end)
        if n == 2:
            return [start, end]
        import random as _r
        rng = _r.Random(start * 2654435761 + 12345)
        mids = [rng.randint(0, 2**31 - 1) for _ in range(n - 2)]
        return [start] + mids + [end]

    def _royale_travel_randomize_seeds(self):
        """Reroll Start/End seeds (and thus the whole journey, since mids derive from Start)."""
        import random
        self.royale_travel_seed_a_var.set(str(random.randint(0, 999999)))
        self.royale_travel_seed_b_var.set(str(random.randint(0, 999999)))

    def _royale_travel_apply_preset(self):
        """Fill the seed-travel mechanics knobs from a preset (no prompt involved)."""
        preset = SEED_TRAVEL_PRESETS.get(self.royale_travel_preset_var.get())
        if not preset:
            return
        self.royale_travel_ref_strength_var.set(preset["ref_strength"])
        self.royale_travel_ref_mp_var.set(preset["ref_mp"])
        self.royale_travel_seq_ref_var.set(bool(preset["sequential"]))
        self.royale_travel_waypoints_var.set(preset["waypoints"])

    # ----- Seed travel: morph one epoch between two seeds (slerp) -----
    def _royale_seed_travel(self):
        if self._royale_is_busy():
            return
        label, path = self._royale_current_epoch()
        if path is None or not os.path.exists(path):
            messagebox.showinfo("LoRA Royale", "Pick a LoRA file (Single-LoRA mode), or render epochs and slide "
                                               "to the one you want (Folder mode), then seed-travel.")
            return
        prompt = self.royale_prompt_var.get().strip()
        if not prompt:
            messagebox.showinfo("LoRA Royale", "Enter a prompt (include your trigger word).")
            return
        try:
            seed_a = int(self.royale_travel_seed_a_var.get())
            seed_b = int(self.royale_travel_seed_b_var.get())
        except ValueError:
            messagebox.showinfo("LoRA Royale", "Start and end seed must be whole numbers.")
            return
        try:
            waypoints = int(self.royale_travel_waypoints_var.get())
        except ValueError:
            waypoints = 2
        if seed_a == seed_b and waypoints <= 2:
            messagebox.showinfo("LoRA Royale", "Start and end seed are the same — pick two different seeds, "
                                               "or raise Waypoints for a longer journey.")
            return
        try:
            frames = int(self.royale_travel_frames_var.get())
        except ValueError:
            frames = 24
        if not self._royale_check_lora_families(path):
            return
        if not self._royale_validate_models():
            return
        try:
            width = int(self.royale_travel_w_var.get()); height = int(self.royale_travel_h_var.get())
        except ValueError:
            width = height = 512
        params = dict(
            label=label, path=path, prompt=prompt, seed_a=seed_a, seed_b=seed_b,
            waypoints=max(2, waypoints),
            frames=max(2, frames), width=width, height=height,
            ref=self._royale_resolve_travel_ref(self.royale_travel_use_epoch_ref_var.get(),
                                                self.royale_travel_ref_var.get()),
            ref_strength=self._royale_parse_ref_strength(self.royale_travel_ref_strength_var.get()),
            ref_mp=self._royale_parse_ref_mp(self.royale_travel_ref_mp_var.get()),
            sequential=bool(self.royale_travel_seq_ref_var.get()),
            anchor=False,
        )
        self._royale_traveling = True
        self._royale_travel_btn.configure(state="disabled")
        self.royale_travel_status_var.set("Loading model…")
        import threading
        threading.Thread(target=self._royale_travel_worker, args=(params,), daemon=True).start()

    def _royale_travel_worker(self, p):
        import sys, os
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.state import SliderState
        eng = self.royale_engine
        try:
            self._royale_ensure_pipeline_loaded()      # heavy load, off the main thread
            # Make sure the engine holds the parked epoch's weights.
            self._royale_load_or_swap_primary(eng, p["path"])
            n = p["frames"]
            seeds = self._royale_journey_seeds(p["seed_a"], p["seed_b"], p.get("waypoints", 2))
            nseg = len(seeds) - 1
            imgs, labels = [], []
            prev_latent = None
            for i in range(n):
                t = i / float(n - 1)
                self.master.after(0, lambda i=i: self.royale_travel_status_var.set(
                    f"Rendering frame {i + 1}/{n}…"))
                # Map global t onto the journey: which consecutive seed pair + local fraction.
                pos = t * nseg
                si = min(int(pos), nseg - 1)
                st = self._royale_default_state()
                st.prompt = p["prompt"]; st.seed = seeds[si]
                st.preview_width = p["width"]; st.preview_height = p["height"]
                pls = self._royale_apply_travel_ref(st, p, i)
                img = eng.generate_preview(
                    st, seed_b=seeds[si + 1], travel_t=pos - si,
                    prev_latent=prev_latent, prev_latent_strength=pls)
                imgs.append(img.copy())
                labels.append(self._royale_label_disp(p["label"]).upper())
                if p.get("sequential"):
                    prev_latent = eng._last_frame_latent
            self.master.after(0, lambda: self._royale_travel_finish(imgs, labels, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_travel_finish([], [], e))
        finally:
            self._royale_release_vram()

    def _royale_travel_finish(self, frames, labels, err):
        self._royale_traveling = False
        self._royale_travel_btn.configure(state="normal")
        if err is not None:
            self.royale_travel_status_var.set("Seed-travel failed — see console.")
            messagebox.showerror("Seed-travel failed", str(err))
            return
        if not frames:
            self.royale_travel_status_var.set("No frames produced — see console.")
            return
        self._royale_sc_populate("seed", frames, labels)
        self.royale_travel_status_var.set(f"Rendered {len(frames)} frames — scrub to review, then save.")

    # ----- LoRA strength travel: ramp the LoRA multiplier on a fixed prompt+seed -----
    # Render produces the raw frames into a scrubber; save (MP4/GIF/frame) is deferred
    # and applies the cosmetic/encode options at save time (no re-render).
    def _royale_lora_travel(self):
        if self._royale_is_busy():
            return
        label, path = self._royale_current_epoch()
        if path is None or not os.path.exists(path):
            messagebox.showinfo("LoRA Royale", "Pick a LoRA file (Single-LoRA mode), or render epochs and slide "
                                               "to the one you want (Folder mode), then strength-travel.")
            return
        prompt = self.royale_prompt_var.get().strip()
        if not prompt:
            messagebox.showinfo("LoRA Royale", "Enter a prompt (include your trigger word).")
            return
        try:
            seed = int(self.royale_seed_var.get() or "42")
        except ValueError:
            seed = 42
        try:
            s_start = float(self.royale_lora_start_var.get())
            s_end = float(self.royale_lora_end_var.get())
        except ValueError:
            messagebox.showinfo("LoRA Royale", "Start and end strength must be numbers.")
            return
        if s_start == s_end:
            messagebox.showinfo("LoRA Royale", "Start and end strength are the same — pick two different values.")
            return
        try:
            frames = int(self.royale_lora_frames_var.get())
        except ValueError:
            frames = 24
        if not self._royale_check_lora_families(path):
            return
        if not self._royale_validate_models():
            return
        try:
            width = int(self.royale_lora_w_var.get()); height = int(self.royale_lora_h_var.get())
        except ValueError:
            width = height = 512
        params = dict(
            label=label, path=path, prompt=prompt, seed=seed,
            s_start=s_start, s_end=s_end, frames=max(2, frames),
            width=width, height=height,
        )
        self._royale_lora_running = True
        self._royale_lora_btn.configure(state="disabled")
        self.royale_lora_status_var.set("Loading model…")
        import threading
        threading.Thread(target=self._royale_lora_travel_worker, args=(params,), daemon=True).start()

    def _royale_lora_travel_worker(self, p):
        import sys, os
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.state import SliderState
        eng = self.royale_engine
        try:
            self._royale_ensure_pipeline_loaded()      # heavy load, off the main thread
            self._royale_load_or_swap_primary(eng, p["path"])
            n = p["frames"]
            imgs, labels = [], []
            for i in range(n):
                t = i / float(n - 1)
                strength = p["s_start"] + (p["s_end"] - p["s_start"]) * t
                self.master.after(0, lambda i=i: self.royale_lora_status_var.set(
                    f"Rendering frame {i + 1}/{n}…"))
                st = self._royale_default_state()
                st.prompt = p["prompt"]; st.seed = p["seed"]
                st.preview_width = p["width"]; st.preview_height = p["height"]
                for bs in st.blocks.values():       # uniform LoRA strength across all blocks
                    bs.primary_strength = strength
                img = eng.generate_preview(st)
                imgs.append(img.copy())
                labels.append(f"{strength:.2f}×")
            self.master.after(0, lambda: self._royale_lora_travel_finish(imgs, labels, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_lora_travel_finish([], [], e))
        finally:
            self._royale_release_vram()

    def _royale_lora_travel_finish(self, frames, labels, err):
        self._royale_lora_running = False
        self._royale_lora_btn.configure(state="normal")
        if err is not None:
            self.royale_lora_status_var.set("Strength-travel failed — see console.")
            messagebox.showerror("Strength-travel failed", str(err))
            return
        if not frames:
            self.royale_lora_status_var.set("No frames produced — see console.")
            return
        self._royale_sc_populate("strength", frames, labels)
        self.royale_lora_status_var.set(f"Rendered {len(frames)} frames — scrub to review, then save.")

    # ----- Comparison sheet (labelled before/after grid) -----
    @staticmethod
    def _royale_pick_epoch_columns(cols, spec):
        """Subset (label, path) epoch columns by a user spec.

        '' -> all;  'every N' / '/N' -> every Nth, last always kept;  '4,8,12' -> those labels.
        A 40-epoch run as 40 columns is unreadable, so this is how you get a shareable sheet.
        Unparseable specs fall back to everything rather than failing the run.
        """
        spec = (spec or "").strip().lower()
        if not spec or not cols:
            return cols
        import re as _re
        m = _re.fullmatch(r"(?:every\s*|/)(\d+)", spec)
        if m:
            n = max(1, int(m.group(1)))
            picked = cols[n - 1::n]
            if cols[-1] not in picked:      # the final epoch is the one people most want
                picked.append(cols[-1])
            return picked or [cols[-1]]
        wanted = [t.strip() for t in spec.replace(" ", ",").split(",") if t.strip()]
        if not wanted:
            return cols
        picked = []
        for lbl, path in cols:
            s = str(lbl)
            for w in wanted:
                if s == w or (s.isdigit() and w.isdigit() and int(s) == int(w)):
                    picked.append((lbl, path))
                    break
        return picked or cols

    @staticmethod
    def _royale_strip_trigger(prompt: str, trigger: str) -> str:
        """Drop the trigger token from a prompt for the no-LoRA column. The base model has never
        seen it, so leaving it in feeds the baseline a junk token and makes the comparison unfair."""
        trigger = (trigger or "").strip()
        if not trigger:
            return prompt
        tl = trigger.lower()
        parts = [p for p in (seg.strip() for seg in prompt.split(","))
                 if p and p.lower() != tl]
        out = ", ".join(parts)
        if out.lower() == prompt.strip().lower():   # trigger wasn't its own comma segment
            import re as _re
            out = _re.sub(rf"\b{_re.escape(trigger)}\b", "", prompt, flags=_re.IGNORECASE)
            out = _re.sub(r"\s{2,}", " ", out).strip(" ,")
        return out or prompt

    def _royale_comparison_sheet(self):
        if self._royale_is_busy():
            return
        prompts = [ln.strip() for ln in self.royale_cmp_prompts.get("1.0", tk.END).splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
        if not prompts:
            messagebox.showinfo("LoRA Royale", "Enter at least one prompt — each line becomes a row of the sheet.")
            return
        mode = self.royale_cmp_mode_var.get()
        label, path = self._royale_current_epoch()
        if mode == "Every epoch":
            cols = [(lbl, self._royale_paths.get(lbl)) for lbl, _ in (self._royale_images or [])]
            cols = [(l, p) for l, p in cols if p and os.path.exists(p)]
            if not cols:
                messagebox.showinfo(
                    "Render the epochs first",
                    "\"Every epoch\" reuses the epochs from the main render at the top of this tab.\n\n"
                    "1. Switch to Folder mode and pick your training output folder\n"
                    "2. Click Render (the epoch/crossfade render)\n"
                    "3. Come back here and render the sheet\n\n"
                    "Or set Columns to \"Without / with LoRA\", which needs no epoch render.")
                return
            cols = self._royale_pick_epoch_columns(cols, self.royale_cmp_epochs_var.get())
        else:
            if path is None or not os.path.exists(path):
                messagebox.showinfo("LoRA Royale", "Pick a LoRA file (Single-LoRA mode), or render epochs and slide "
                                                   "to the one you want to showcase.")
                return
            cols = [(None, path), (label, path)]      # column 0 renders at strength 0 (base model)
        if not self._royale_check_lora_families(*[p for _, p in cols]):
            return
        if not self._royale_validate_models():
            return
        try:
            seed = int(self.royale_cmp_seed_var.get() or "42")
        except ValueError:
            seed = 42
        try:
            width = int(self.royale_cmp_w_var.get()); height = int(self.royale_cmp_h_var.get())
        except ValueError:
            width = height = 512
        total = len(prompts) * len(cols)
        if total > 24 and not messagebox.askyesno(
                "Large sheet",
                f"That's {total} renders ({len(prompts)} prompts x {len(cols)} columns).\n\n"
                "It will take a while. Continue?"):
            return
        params = dict(prompts=prompts, cols=cols, mode=mode, seed=seed, width=width, height=height,
                      trigger=self.royale_cmp_trigger_var.get().strip(),
                      row_labels=bool(self.royale_cmp_rowlabels_var.get()),
                      brand=bool(self.royale_cmp_brand_var.get()),
                      out_dir=os.path.dirname(path) if path else os.getcwd())
        self._royale_cmp_running = True
        self._royale_cmp_btn.configure(state="disabled")
        self.royale_cmp_status_var.set("Loading model…")
        import threading
        threading.Thread(target=self._royale_comparison_worker, args=(params,), daemon=True).start()

    def _royale_comparison_worker(self, p):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        eng = self.royale_engine
        try:
            self._royale_ensure_pipeline_loaded()
            n_cols, n_rows = len(p["cols"]), len(p["prompts"])
            # Column-major: each LoRA loads once, then renders every row.
            grid = [[None] * n_cols for _ in range(n_rows)]
            done = 0
            for ci, (clabel, cpath) in enumerate(p["cols"]):
                base_col = (p["mode"] != "Every epoch" and ci == 0)   # the no-LoRA column
                self._royale_load_or_swap_primary(eng, cpath)
                for ri, prompt in enumerate(p["prompts"]):
                    done += 1
                    self.master.after(0, lambda d=done, t=n_rows * n_cols:
                                      self.royale_cmp_status_var.set(f"Rendering {d}/{t}…"))
                    st = self._royale_default_state()
                    st.prompt = self._royale_strip_trigger(prompt, p["trigger"]) if base_col else prompt
                    st.seed = p["seed"]                     # same seed down a row: only the LoRA differs
                    st.preview_width = p["width"]; st.preview_height = p["height"]
                    for bs in st.blocks.values():
                        bs.primary_strength = 0.0 if base_col else 1.0
                    grid[ri][ci] = eng.generate_preview(st).copy()

            if p["mode"] == "Every epoch":
                headers = [f"epoch {l}" if str(l).isdigit() else str(l) for l, _ in p["cols"]]
            else:
                trig = p["trigger"] or "LoRA"
                headers = ["no lora", f"{trig} lora"]
            row_caps = p["prompts"] if p["row_labels"] else None

            from fizgig.lora_royale.export import build_comparison_grid
            sheet = build_comparison_grid(grid, headers, row_labels=row_caps,
                                          cell=(p["width"], p["height"]), brand=p["brand"])
            self.master.after(0, lambda: self._royale_comparison_finish(sheet, None, p["out_dir"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_comparison_finish(None, e, None))
        finally:
            self._royale_release_vram()

    def _royale_comparison_finish(self, sheet, err, out_dir):
        self._royale_cmp_running = False
        self._royale_cmp_btn.configure(state="normal")
        if err is not None:
            self.royale_cmp_status_var.set("Comparison sheet failed — see console.")
            messagebox.showerror("Comparison sheet failed", str(err))
            return
        self._royale_cmp_sheet = sheet
        self.royale_cmp_status_var.set(f"Sheet ready ({sheet.width}x{sheet.height}) — review, then save.")
        self._royale_cmp_popup(sheet, out_dir)

    def _royale_cmp_popup(self, sheet, out_dir):
        """Review window for a finished sheet: scaled-to-fit preview + Save / Close.

        Nothing is written to disk until Save is clicked — a sheet is a few minutes of
        renders, so it's worth looking at before deciding to keep it."""
        from PIL import ImageTk
        win = tk.Toplevel(self.master)
        win.title("LoRA Royale — Comparison sheet")
        win.configure(bg="#101010")
        # Open at a sensible fraction of the screen; the image scales to whatever size you drag to.
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        scale = min(1.0, (sw * 0.75) / sheet.width, (sh * 0.75) / sheet.height)
        win.geometry(f"{max(320, int(sheet.width * scale))}x{max(240, int(sheet.height * scale) + 46)}")
        win.minsize(320, 240)

        lbl = tk.Label(win, bg="#101010")
        lbl.pack(fill=tk.BOTH, expand=True)
        bar = tk.Frame(win, bg=COLORS["bg_surface"])
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        state = {"img": None, "after": None}

        def _render():
            state["after"] = None
            try:
                bw = max(64, lbl.winfo_width()); bh = max(64, lbl.winfo_height())
            except Exception:
                return
            r = min(bw / sheet.width, bh / sheet.height)
            w, h = max(1, int(sheet.width * r)), max(1, int(sheet.height * r))
            state["img"] = ImageTk.PhotoImage(sheet.resize((w, h), Image.LANCZOS))
            lbl.configure(image=state["img"])

        def _on_configure(event):
            if event.widget is not win:
                return
            if state["after"] is not None:
                try:
                    win.after_cancel(state["after"])
                except Exception:
                    pass
            state["after"] = win.after(80, _render)     # debounce drags

        def _save():
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                parent=win, title="Save comparison sheet",
                initialdir=out_dir or os.getcwd(),
                initialfile="fizgig_comparison.png",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
            if not path:
                return
            try:
                img = sheet
                if path.lower().endswith((".jpg", ".jpeg")):
                    img = sheet.convert("RGB")
                img.save(path)
            except Exception as e:
                messagebox.showerror("Save failed", str(e), parent=win)
                return
            self.royale_cmp_status_var.set(f"Saved {os.path.basename(path)}")
            self._royale_reveal(path)

        tk.Button(bar, text="Save sheet…", font=(FONT_FAMILY, 10, "bold"), fg="#FFFFFF",
                  bg="#B7791F", activeforeground="#FFFFFF", activebackground="#9A6518",
                  relief="flat", bd=0, padx=16, pady=4, cursor="hand2",
                  command=_save).pack(side=tk.LEFT, padx=10, pady=6)
        tk.Label(bar, text=f"{sheet.width} x {sheet.height}px", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        tk.Button(bar, text="Close", font=(FONT_FAMILY, 10), relief="flat", bd=0,
                  padx=14, pady=4, cursor="hand2",
                  command=win.destroy).pack(side=tk.RIGHT, padx=10, pady=6)

        win.bind("<Configure>", _on_configure)
        win.after(60, _render)
        win.lift()

    # ----- Shared render->scrub->save preview used by all three travel modes -----
    def _royale_make_scrubber(self, parent, mode, save_prefix, bg, status_var, options_getter):
        """Build a centered 512 preview scrubber + Save MP4/GIF/frame block. State lives
        in self._royale_sc[mode]. options_getter() -> dict(speed, pingpong, brand, badge,
        deflicker), read at save time so export settings apply without re-rendering."""
        if not hasattr(self, "_royale_sc"):
            self._royale_sc = {}
        sc = {"frames": [], "labels": [], "save_prefix": save_prefix, "options": options_getter,
              "status": status_var, "box": (512, 512), "imgtk": None}
        _p = tk.Frame(parent, bg=bg); _p.pack(fill=tk.X, pady=(8, 0))
        holder = tk.Frame(_p, bg="#1c1c1c", width=512, height=512, highlightthickness=0)
        holder.pack_propagate(False); holder.pack(pady=(0, 8))
        sc["holder"] = holder
        sc["preview"] = ttk.Label(holder, anchor=tk.CENTER, background="#1c1c1c",
                                  text="Render to preview the frames here, then save.")
        sc["preview"].pack(fill=tk.BOTH, expand=True)
        sc["label_var"] = tk.StringVar(value="")
        tk.Label(_p, textvariable=sc["label_var"], font=(FONT_FAMILY, 11, "bold"),
                 fg=COLORS["text_primary"], bg=bg).pack()
        sc["scrub_var"] = tk.DoubleVar(value=0.0)
        sc["scale"] = ttk.Scale(_p, from_=0, to=1, orient=tk.HORIZONTAL, variable=sc["scrub_var"],
                                command=lambda e, m=mode: self._royale_sc_scrub(m))
        sc["scale"].pack(fill=tk.X, padx=20, pady=(4, 8))
        sc["scale"].configure(state="disabled")
        _v = tk.Frame(parent, bg=bg); _v.pack(pady=(0, 0))
        sc["save_mp4"] = ttk.Button(_v, text="Save MP4", state="disabled",
                                    command=lambda m=mode: self._royale_sc_save(m, "MP4"))
        sc["save_mp4"].pack(side=tk.LEFT)
        sc["save_gif"] = ttk.Button(_v, text="Save GIF", state="disabled",
                                    command=lambda m=mode: self._royale_sc_save(m, "GIF"))
        sc["save_gif"].pack(side=tk.LEFT, padx=(6, 0))
        sc["save_frame_btn"] = ttk.Button(_v, text="Save frame", state="disabled",
                                          command=lambda m=mode: self._royale_sc_save_frame(m))
        sc["save_frame_btn"].pack(side=tk.LEFT, padx=(6, 0))
        self._royale_sc[mode] = sc
        return sc

    def _royale_sc_fit(self, mode):
        sc = self._royale_sc[mode]; m = 512
        frames = sc["frames"]
        if not frames:
            box = (m, m)
        else:
            iw, ih = frames[0].size
            box = (m, max(64, round(m * ih / iw))) if iw >= ih else (max(64, round(m * iw / ih)), m)
        sc["box"] = box
        try:
            sc["holder"].configure(width=box[0], height=box[1])
        except Exception:
            pass

    def _royale_sc_scrub(self, mode):
        from PIL import Image, ImageTk
        sc = self._royale_sc[mode]; frames = sc["frames"]
        if not frames:
            return
        i = max(0, min(int(round(float(sc["scrub_var"].get()))), len(frames) - 1))
        hw, hh = sc["box"]
        disp = frames[i].copy(); disp.thumbnail((max(64, hw), max(64, hh)), Image.LANCZOS)
        sc["imgtk"] = ImageTk.PhotoImage(disp)
        sc["preview"].configure(image=sc["imgtk"], text="")
        lbl = sc["labels"][i] if i < len(sc["labels"]) else ""
        sc["label_var"].set(f"Frame {i + 1}/{len(frames)}    {lbl}")

    def _royale_sc_populate(self, mode, frames, labels):
        sc = self._royale_sc[mode]
        sc["frames"] = frames; sc["labels"] = labels
        sc["scale"].configure(to=float(len(frames) - 1), state="normal")
        sc["scrub_var"].set(0.0)
        for k in ("save_mp4", "save_gif", "save_frame_btn"):
            sc[k].configure(state="normal")
        self._royale_sc_fit(mode)
        self._royale_sc_scrub(mode)

    def _royale_sc_save(self, mode, fmt):
        """Encode the rendered frames to MP4/GIF, applying the export options live."""
        sc = self._royale_sc[mode]; frames = sc["frames"]
        if not frames:
            return
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import export as rexport, run_name_for_folder
        from tkinter import filedialog
        ext = ".mp4" if fmt == "MP4" else ".gif"
        run = run_name_for_folder(self.royale_folder_var.get().strip()) or "lora"
        out = filedialog.asksaveasfilename(
            title=f"Save {sc['save_prefix']} {fmt}", defaultextension=ext,
            initialfile=f"{run}-{sc['save_prefix']}{ext}",
            initialdir=self.settings.get("LORA_OUTPUT_DIR", ""),
            filetypes=[("MP4 video", "*.mp4")] if fmt == "MP4" else [("Animated GIF", "*.gif")])
        if not out:
            return
        opts = sc["options"]()
        try:
            imgs = list(frames)
            dfm = opts.get("deflicker", "None")
            if dfm and dfm != "None":
                imgs = rexport.deflicker_frames(imgs, sigma=(len(imgs) / 3.0) if dfm == "Strong" else None)
            labels = sc["labels"] if opts.get("badge") else None
            max_size = None if fmt == "MP4" else 768
            clip = rexport.frames_from_sequence(imgs, pingpong=opts.get("pingpong", True),
                                                brand=opts.get("brand", True), labels=labels, max_size=max_size)
            if fmt == "MP4":
                rexport.write_mp4(clip, out, speed=opts.get("speed", "Normal"))
            else:
                rexport.write_gif(clip, out, speed=opts.get("speed", "Normal"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            if "codec" in msg.lower() or "writer" in msg.lower():
                msg += "\n\nTry GIF instead."
            messagebox.showerror("Save failed", msg)
            return
        sc["status"].set(f"Saved {os.path.basename(out)}")
        self._royale_reveal(out)

    def _royale_sc_save_frame(self, mode):
        """Save the currently-scrubbed frame as a PNG."""
        sc = self._royale_sc[mode]; frames = sc["frames"]
        if not frames:
            return
        from tkinter import filedialog
        i = max(0, min(int(round(float(sc["scrub_var"].get()))), len(frames) - 1))
        out = filedialog.asksaveasfilename(
            title="Save frame", defaultextension=".png",
            initialfile=f"{sc['save_prefix']}-frame{i + 1:03d}.png",
            initialdir=self.settings.get("LORA_OUTPUT_DIR", ""),
            filetypes=[("PNG image", "*.png")])
        if not out:
            return
        try:
            frames[i].save(out)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        sc["status"].set(f"Saved frame {i + 1} → {os.path.basename(out)}")
        self._royale_reveal(out)

    # ----- Prompt travel: morph one epoch through a prompt dimension -----
    def _royale_pt_words(self):
        """Resolve the full ordered waypoint list for the current dimension/custom."""
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.lora_royale import prompt_travel as pt
        dim = self.royale_pt_dim_var.get()
        if dim == "Custom":
            return pt.parse_custom(self.royale_pt_custom_var.get())
        return pt.waypoints_for(dim, self.royale_pt_subject_var.get())

    def _royale_pt_refresh_range(self):
        """Populate the Start/End waypoint dropdowns for the current dimension,
        resetting the selection to the full range when it no longer applies."""
        words = self._royale_pt_words()
        if hasattr(self, "_royale_pt_start_combo"):
            self._royale_pt_start_combo["values"] = words
            self._royale_pt_end_combo["values"] = words
        if not words:
            return
        if self.royale_pt_start_var.get() not in words:
            self.royale_pt_start_var.set(words[0])
        if self.royale_pt_end_var.get() not in words:
            self.royale_pt_end_var.set(words[-1])

    def _royale_pt_ranged_words(self):
        """The waypoint sub-sequence the travel will actually cover, honoring the
        Start/End selection (reversed if Start sits after End, e.g. de-aging)."""
        words = self._royale_pt_words()
        if len(words) < 2:
            return words
        s, e = self.royale_pt_start_var.get(), self.royale_pt_end_var.get()
        si = words.index(s) if s in words else 0
        ei = words.index(e) if e in words else len(words) - 1
        if si <= ei:
            return words[si:ei + 1]
        return list(reversed(words[ei:si + 1]))

    def _royale_pt_insert_slot(self):
        """Insert the {x} travel slot at the cursor in the prompt entry (append
        if it isn't focused)."""
        entry = getattr(self, "_royale_pt_prompt_entry", None)
        if entry is None:
            return
        try:
            entry.insert(entry.index(tk.INSERT), "{x}")
        except Exception:
            self.royale_pt_prompt_var.set((self.royale_pt_prompt_var.get() + " {x}").strip())
        entry.focus_set()

    def _royale_pt_refresh_words(self):
        words = self._royale_pt_ranged_words()
        if words and len(words) >= 2:
            self.royale_pt_words_var.set("Waypoints:  " + "  →  ".join(words))
        elif words:
            self.royale_pt_words_var.set("Pick a Start/End that span at least two waypoints.")
        else:
            self.royale_pt_words_var.set("Add at least two comma-separated custom words to travel between.")

    def _royale_prompt_travel(self):
        if self._royale_is_busy():
            return
        label, path = self._royale_current_epoch()
        if path is None or not os.path.exists(path):
            messagebox.showinfo("LoRA Royale", "Pick a LoRA file (Single-LoRA mode), or render epochs and slide "
                                               "to the one you want (Folder mode), then prompt-travel.")
            return
        words = self._royale_pt_ranged_words()
        if len(words) < 2:
            messagebox.showinfo("LoRA Royale", "Prompt travel needs at least two waypoints "
                                               "(pick a Travel dimension, or enter 2+ Custom words).")
            return
        base = self.royale_pt_prompt_var.get().strip()
        if not base:
            messagebox.showinfo("LoRA Royale", "Enter a base prompt (include your trigger word, and {x} where the "
                                               "travel word goes).")
            return
        try:
            frames = int(self.royale_pt_frames_var.get())
        except ValueError:
            frames = 32
        try:
            seed = int(self.royale_seed_var.get() or "42")
        except ValueError:
            seed = 42
        try:
            width = int(self.royale_pt_w_var.get()); height = int(self.royale_pt_h_var.get())
        except ValueError:
            width = height = 512
        if not self._royale_check_lora_families(path):
            return
        if not self._royale_validate_models():
            return
        params = dict(
            label=label, path=path, base=base, words=words,
            frames=max(2, frames), seed=seed, width=width, height=height,
            ref=self._royale_resolve_travel_ref(self.royale_pt_use_epoch_ref_var.get(),
                                                self.royale_pt_ref_var.get()),
            ref_strength=self._royale_parse_ref_strength(self.royale_pt_ref_strength_var.get()),
            ref_mp=self._royale_parse_ref_mp(self.royale_pt_ref_mp_var.get()),
            sequential=bool(self.royale_pt_seq_ref_var.get()),
            anchor=bool(self.royale_pt_anchor_var.get()),
            anchor_str=self._royale_parse_ref_strength(self.royale_pt_anchor_str_var.get()),
            vary_seed=bool(self.royale_pt_vary_seed_var.get()),
            drift=self._royale_parse_drift(self.royale_pt_drift_var.get()),
            interp=self.royale_pt_interp_var.get(),
        )
        self._royale_pt_running = True
        self._royale_pt_btn.configure(state="disabled")
        self.royale_pt_status_var.set("Loading model…")
        import threading
        threading.Thread(target=self._royale_pt_worker, args=(params,), daemon=True).start()

    def _royale_pt_worker(self, p):
        import sys, os, random
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.state import SliderState
        from fizgig.lora_royale import prompt_travel as pt
        eng = self.royale_engine
        try:
            self._royale_ensure_pipeline_loaded()      # heavy load, off the main thread
            self._royale_load_or_swap_primary(eng, p["path"])
            self.master.after(0, lambda: self.royale_pt_status_var.set("Encoding prompts…"))
            wp_prompts = pt.build_waypoint_prompts(p["base"], p["words"])
            ctx_list, neg = eng.encode_travel_prompts(wp_prompts)
            interp_mode = {"Linear": "lerp", "Norm-preserved": "norm",
                           "Slerp": "slerp"}.get(p.get("interp", "Linear"), "lerp")
            drift = float(p.get("drift", 0.0) or 0.0)
            # Deterministic second seed for smooth noise drift (uncorrelated with base).
            drift_seed = (int(p["seed"]) + 1013904223) % (2**31) if drift > 0 else None
            n = p["frames"]
            imgs, labels = [], []
            prev_latent = None
            for i in range(n):
                t = i / float(n - 1)
                self.master.after(0, lambda i=i: self.royale_pt_status_var.set(
                    f"Rendering frame {i + 1}/{n}…"))
                ctx = eng.interp_waypoints(ctx_list, t, mode=interp_mode)
                st = self._royale_default_state()
                st.prompt = p["base"]
                # Vary seed: deterministic sequential walk (base, base+1, …) so the
                # image re-rolls per frame yet the whole clip stays reproducible.
                st.seed = (p["seed"] + i) if p.get("vary_seed") else p["seed"]
                st.preview_width = p["width"]; st.preview_height = p["height"]
                pls = self._royale_apply_travel_ref(st, p, i)
                if drift > 0 and not p.get("vary_seed"):
                    # Smooth noise drift: slerp base seed -> drift_seed across the sweep,
                    # breaking the static-seed fixed point so the prompt expresses.
                    img = eng.generate_preview(st, seed_b=drift_seed, travel_t=drift * t,
                                               override_ctx=ctx, override_neg_ctx=neg,
                                               prev_latent=prev_latent, prev_latent_strength=pls)
                else:
                    img = eng.generate_preview(st, override_ctx=ctx, override_neg_ctx=neg,
                                               prev_latent=prev_latent, prev_latent_strength=pls)
                imgs.append(img.copy())
                labels.append(pt.dominant_word(p["words"], t).upper())
                if p.get("sequential"):
                    prev_latent = eng._last_frame_latent
            self.master.after(0, lambda: self._royale_pt_finish(imgs, labels, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda e=e: self._royale_pt_finish([], [], e))
        finally:
            self._royale_release_vram()

    def _royale_pt_finish(self, frames, labels, err):
        self._royale_pt_running = False
        self._royale_pt_btn.configure(state="normal")
        if err is not None:
            self.royale_pt_status_var.set("Prompt-travel failed — see console.")
            messagebox.showerror("Prompt-travel failed", str(err))
            return
        if not frames:
            self.royale_pt_status_var.set("No frames produced — see console.")
            return
        self._royale_sc_populate("prompt", frames, labels)
        self.royale_pt_status_var.set(f"Rendered {len(frames)} frames — scrub to review, then save.")

    def _repair_start(self):
        """Smart Start: load/swap primary, load/swap donor, or regenerate."""
        primary_path = self.repair_primary_var.get().strip()
        donor_path = self.repair_donor_var.get().strip()

        if not primary_path:
            messagebox.showerror("Error", "Set a primary LoRA path first.")
            return

        if not os.path.exists(primary_path):
            messagebox.showerror("Error", f"Primary LoRA not found:\n{primary_path}")
            return

        if getattr(self, "_repair_loading", False):
            return   # a load is already in flight

        # The loads run on worker threads (a 10-20 GB pipeline load froze the UI here), so
        # the primary → donor → preview order is kept by chaining completions rather than by
        # falling through this function.
        current_primary = self.repair_engine.primary_path if self.repair_engine else None

        def _then_donor_then_preview():
            if donor_path and os.path.exists(donor_path):
                self._load_repair_donor(on_done=lambda: self._schedule_preview(force=True))
            else:
                self._schedule_preview(force=True)

        if current_primary != primary_path or self.repair_engine is None \
                or self.repair_engine.primary_network is None:
            # New or changed primary — reset and reload
            if self.repair_engine is not None and self.repair_engine.primary_network is not None:
                self._reset_repair_session()
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._load_repair_primary(on_done=_then_donor_then_preview)
        else:
            # Primary unchanged — swap the donor if it changed, then regenerate.
            current_donor = self.repair_engine.donor_path
            if donor_path and os.path.exists(donor_path) and current_donor != donor_path:
                if self.repair_engine.donor_network is not None:
                    self._unload_repair_donor()
                self._load_repair_donor(on_done=self._force_regenerate_preview)
            else:
                self._force_regenerate_preview()

        # Reset button text back to Start
        self._repair_reset_start_button()

    def _load_repair_primary(self, on_done=None):
        """Load (or reload) the primary LoRA. The heavy work — ensure_pipeline's 10-20+ GB
        DiT load plus the LoRA network build — runs on a worker thread so the UI never
        freezes; completion (slider refresh, profile match, preview) lands back on the Tk
        thread. on_done, when given, replaces the default schedule-preview completion so a
        caller can chain a donor load first."""
        path = self.repair_primary_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid primary LoRA file first.")
            return
        # Auto-follow the file's family (issue #62 nice-to-have): detected from the header
        # alone, microseconds, so do it BEFORE _repair_engine_plan() commits to loading the
        # wrong family's DiT/VAE/TE. Only a genuinely unrecognized file falls through to the
        # generic error below, same as before.
        from fizgig.networks.lora import lora_family_from_file, FAMILY_DISPLAY_NAMES, INFERENCE_FAMILIES
        detected = lora_family_from_file(path)
        if detected is not None and detected not in INFERENCE_FAMILIES:
            # Setting the var to a family with no radio leaves all radios blank instead of
            # following it (issue #62). Refuse rather than land on a family this tab has no
            # engine for.
            messagebox.showerror(
                "Unsupported family",
                f"{os.path.basename(path)} was trained for {FAMILY_DISPLAY_NAMES.get(detected, detected)}, "
                f"but Repair Studio doesn't support {FAMILY_DISPLAY_NAMES.get(detected, detected)} LoRAs yet.")
            return
        selected = self.repair_family_var.get()
        if selected not in INFERENCE_FAMILIES:
            selected = "klein"
        if detected is not None and detected != selected:
            self.repair_family_var.set(detected)
            self._on_repair_family_changed()
            self.repair_status_var.set(
                f"Switched family selector to {FAMILY_DISPLAY_NAMES.get(detected, detected)} "
                f"to match {os.path.basename(path)}.")
        plan = self._repair_engine_plan()
        if plan is None:
            return
        if getattr(self, "_repair_loading", False):
            return   # a load is already in flight — ignore the double-click
        self._repair_loading = True
        self._repair_start_btn.configure(state="disabled")
        self._repair_progress_marquee_on()
        if self.repair_engine.pipeline is None or not getattr(self.repair_engine.pipeline,
                                                              "is_loaded", False):
            pass   # status already set by the plan builder ("Loading … models")
        else:
            self.repair_status_var.set("Loading primary LoRA…")
        engine = self.repair_engine

        def _work():
            try:
                if plan:
                    engine.ensure_pipeline(**plan)
                engine.load_primary(path)
                err = None
            except Exception as ex:
                import traceback
                err = (ex, traceback.format_exc())
            self.master.after(0, lambda: _finish(err))

        def _finish(err):
            self._repair_loading = False
            self._repair_start_btn.configure(state="normal")
            self._repair_progress_end()
            if err is not None:
                from fizgig.networks.lora import UnsupportedLoRAFormat
                ex, tb = err
                if isinstance(ex, UnsupportedLoRAFormat):
                    messagebox.showerror("Unsupported LoRA format", str(ex))
                    self.repair_status_var.set(f"Unsupported format: {os.path.basename(path)}.")
                else:
                    messagebox.showerror("Error", f"Failed to load primary:\n{tb}")
                    self.repair_status_var.set("Error loading primary.")
                return
            self._refresh_block_slider_activity()
            n_active = len(engine.primary_block_ids)
            # LyCORIS loads and saves natively — no popup on open; the save dialog
            # states the format (and the donor-blend SVD case warns at donor load).
            # Look up a matching Profiler sidecar by content hash and render
            # the inline info panel if one exists.
            self._find_repair_profile_match()
            self.repair_status_var.set(
                f"Primary loaded: {os.path.basename(path)} "
                f"({n_active}/{len(self.repair_state.blocks)} blocks). Generating preview…")
            if on_done is not None:
                on_done()          # a chained donor load schedules the preview itself
            else:
                self._schedule_preview(force=True)

        import threading
        threading.Thread(target=_work, daemon=True).start()

    def _load_repair_donor(self, on_done=None):
        """Async like _load_repair_primary — the donor's LoRA network build runs off the Tk
        thread. Requires the primary (and thus the pipeline) to be loaded already."""
        path = self.repair_donor_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid donor LoRA file first.")
            return
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA before adding a donor.")
            return
        if getattr(self, "_repair_loading", False):
            return
        self._repair_loading = True
        self._repair_start_btn.configure(state="disabled")
        self._repair_progress_marquee_on()
        self.repair_status_var.set("Loading donor LoRA…")
        engine = self.repair_engine

        def _work():
            try:
                engine.load_donor(path)
                err = None
            except Exception as ex:
                import traceback
                err = (ex, traceback.format_exc())
            self.master.after(0, lambda: _finish(err))

        def _finish(err):
            self._repair_loading = False
            self._repair_start_btn.configure(state="normal")
            self._repair_progress_end()
            if err is not None:
                from fizgig.networks.lora import UnsupportedLoRAFormat
                ex, tb = err
                if isinstance(ex, UnsupportedLoRAFormat):
                    messagebox.showerror("Unsupported LoRA format", str(ex))
                    self.repair_status_var.set(f"Unsupported format: {os.path.basename(path)}.")
                else:
                    messagebox.showerror("Error", f"Failed to load donor:\n{tb}")
                    self.repair_status_var.set("Error loading donor.")
                return
            # No popup on open (LyCORIS donors save natively; only blended blocks SVD, and
            # the save dialog reports exactly how many were).
            self._repair_donor_loaded = True
            # Show donor sub-rows + master section toggles + enable the "Donor" master target radio
            for vars_ in self.repair_block_vars.values():
                vars_["donor_rowf"].grid()
            self._repair_master_donor_radio.state(["!disabled"])
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
            self._refresh_block_slider_activity()
            n_donor = len(engine.donor_block_ids)
            self.repair_status_var.set(
                f"Donor loaded: {os.path.basename(path)} "
                f"({n_donor}/{len(self.repair_state.blocks)} blocks). Enable per-block to mix in.")
            if on_done is not None:
                on_done()

        import threading
        threading.Thread(target=_work, daemon=True).start()

    def _repair_mark_update_needed(self):
        """Prompt or seed changed — show 'Update' on the Start button instead of auto-regenerating."""
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self._repair_start_btn.configure(text="Update")

    def _repair_randomize_seed(self):
        """Randomize seed — and on a live session, regenerate immediately.

        Setting the var flips the button to 'Update' via its trace (only when a primary is
        loaded); if that's the state we're in, the click's intent is unambiguous and making
        the user walk to the button is a wasted step. Before Start it just marks, as before."""
        import random
        self.repair_seed_var.set(str(random.randint(1, 99999)))
        self._repair_mark_update_needed()
        if self._repair_start_btn.cget("text") == "Update":
            self._on_preview_param_changed()

    def _repair_seed_committed(self, _event=None):
        """Enter in the seed box = 'go' — same live-session-only rule as the ↻ button.
        Per-keystroke regen would render every partial number, so typing only marks."""
        if self._repair_start_btn.cget("text") == "Update":
            self._on_preview_param_changed()

    def _repair_reset_start_button(self):
        """Reset the Start button text back to 'Start'."""
        self._repair_start_btn.configure(text="Start")

    def _browse_and_load_primary(self):
        """Browse for a primary LoRA. Picking a file changes NOTHING — no reload, no render
        (Peter, 22 Aug: the user may want to set the prompt/seed/sliders first). The Start
        button arms as Update; its click does the swap-and-render (_repair_start already
        handles a changed primary with a full reset + reload)."""
        self._browse_repair_lora(self.repair_primary_var)
        path = self.repair_primary_var.get().strip()
        if not path or not os.path.exists(path):
            return
        if self.repair_engine is not None and self.repair_engine.primary_network is not None \
                and self.repair_engine.primary_path != path:
            self._repair_mark_update_needed()

    def _browse_and_load_donor(self):
        """Browse for a donor LoRA. Same contract as the primary: picking a file only arms
        the Update button — the swap happens on the user's click."""
        self._browse_repair_lora(self.repair_donor_var)
        path = self.repair_donor_var.get().strip()
        if not path or not os.path.exists(path):
            return
        if self.repair_engine is not None and self.repair_engine.primary_network is not None \
                and self.repair_engine.donor_path != path:
            self._repair_mark_update_needed()

    def _unload_repair_donor(self):
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        self.repair_engine.unload_donor()
        self._repair_donor_loaded = False
        # Hide donor sub-rows + master section toggles + revert donor master radio
        for vars_ in self.repair_block_vars.values():
            vars_["donor_rowf"].grid_remove()
            vars_["donor_enabled"].set(True)
            vars_["donor_strength"].set(0.0)
        # donor toggles removed — donor blocks managed via master sliders
        self._repair_master_donor_radio.state(["disabled"])
        if self.repair_master_target_var.get() == "donor":
            self.repair_master_target_var.set("primary")
        self._refresh_block_slider_activity()
        self.repair_status_var.set("Donor unloaded.")
        self._schedule_preview(force=True)

    def _on_block_changed(self, block_id: str):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return
        # During a master-slider bulk update, skip the per-block preview schedule;
        # the master handler fires ONE preview at the end of the batch.
        if getattr(self, "_repair_master_mutating", False):
            return
        # v2 hook
        self.repair_engine.mark_blocks_changed([block_id])
        self._schedule_preview()

    def _on_preview_param_changed(self):
        # seed/prompt/resolution change → invalidate baseline cache and regen.
        print(f"[repair] param change: res={self.repair_res_var.get()!r} "
              f"seed={self.repair_seed_var.get()!r} prompt={self.repair_prompt_var.get()!r}")
        if self.repair_engine is not None:
            self.repair_engine._invalidate_baseline_cache()
        self._repair_preview_dirty = True
        self._schedule_preview(force=True)

    def _browse_repair_ref(self):
        """Pick a reference image for edit-conditioning the preview."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.repair_ref_path_var.set(path)
            self._on_repair_ref_changed()

    def _clear_repair_ref(self):
        """Remove the reference image — preview reverts to text-only."""
        self.repair_ref_path_var.set("")
        self._on_repair_ref_changed()

    def _on_repair_ref_changed(self):
        """Reference path/MP/strength changed — push into state and regenerate.

        The ref conditions BOTH the baseline and tweaked previews, so this
        invalidates the baseline cache (same as a seed/res change)."""
        self.repair_state.ref_image_path = self.repair_ref_path_var.get().strip()
        try:
            self.repair_state.ref_megapixels = float(self.repair_ref_mp_var.get())
        except (ValueError, AttributeError):
            self.repair_state.ref_megapixels = 1.0
        try:
            self.repair_state.ref_strength = float(self.repair_ref_strength_var.get())
        except (ValueError, AttributeError):
            self.repair_state.ref_strength = 1.0
        self._on_preview_param_changed()

    def _on_turbo_toggled(self):
        """Sync Turbo Preview checkbox to the engine and invalidate cache on toggle."""
        if self.repair_engine is not None:
            self.repair_engine._turbo_enabled = self.repair_turbo_var.get()
            self.repair_engine._invalidate_activation_cache()

    def _force_regenerate_preview(self):
        if self.repair_engine is not None:
            self.repair_engine._invalidate_baseline_cache()
        self._repair_preview_dirty = True
        self._schedule_preview(force=True)

    def _schedule_preview(self, force: bool = False):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return
        if self._repair_preview_after_id is not None:
            try:
                self.master.after_cancel(self._repair_preview_after_id)
            except Exception:
                pass
        turbo_on = getattr(self, "repair_turbo_var", None) and self.repair_turbo_var.get()
        if force:
            delay = 100
        elif turbo_on:
            delay = 150
        else:
            delay = 400
        print(f"[repair] schedule preview: force={force} in_flight={self._repair_preview_in_flight} "
              f"delay={delay}ms dirty={self._repair_preview_dirty}")
        self._repair_preview_after_id = self.master.after(delay, self._run_preview_async)

    def _run_preview_async(self):
        self._repair_preview_after_id = None
        if self._repair_preview_in_flight:
            # A preview is running. Mark dirty so its completion hook
            # (_set_repair_preview_images) fires a fresh preview once it lands.
            self._repair_preview_dirty = True
            # Abort the in-flight pass (krea engine) so this newest edit restarts NOW instead of
            # queueing behind the full 8-step render.
            if hasattr(self.repair_engine, "request_cancel"):
                self.repair_engine.request_cancel()
            print("[repair] run_async: in-flight, requested cancel + marked dirty; will refire")
            return
        # Sync state from UI to repair_state (prompt/seed/res live in entries, not bound)
        prompt_text = self.repair_prompt_var.get().strip()
        if not prompt_text:
            # Distilled with empty conditioning → 4 steps of unguided denoising =
            # blocky noise (VAE decoding pure latent noise). Require a prompt.
            self.repair_status_var.set("Enter a prompt (include the LoRA trigger word) to generate previews.")
            return
        try:
            self.repair_state.seed = int(self.repair_seed_var.get() or "42")
        except ValueError:
            self.repair_state.seed = 42
        self.repair_state.prompt = prompt_text
        try:
            res = int(self.repair_res_var.get())
        except ValueError:
            res = 512
        self.repair_state.preview_width = res
        self.repair_state.preview_height = res
        # Reference-image fields (Klein edit conditioning).
        self.repair_state.ref_image_path = self.repair_ref_path_var.get().strip()
        try:
            self.repair_state.ref_megapixels = float(self.repair_ref_mp_var.get())
        except ValueError:
            self.repair_state.ref_megapixels = 1.0
        try:
            self.repair_state.ref_strength = float(self.repair_ref_strength_var.get())
        except ValueError:
            self.repair_state.ref_strength = 1.0

        # Snapshot for thread (dataclass copy via JSON round-trip)
        from fizgig.repair_studio.state import SliderState
        snapshot = self.repair_state.copy()

        # Clear the dirty flag NOW; any param change after this point will
        # set it again and trigger a re-fire when this preview completes.
        self._repair_preview_dirty = False
        self._repair_preview_in_flight = True
        print(f"[repair] run_async: starting worker w={snapshot.preview_width} "
              f"h={snapshot.preview_height} seed={snapshot.seed} prompt={snapshot.prompt!r}")
        self.repair_status_var.set("Generating preview…")
        self._repair_progress_begin()
        import threading
        thread = threading.Thread(target=self._repair_preview_worker, args=(snapshot,), daemon=True)
        thread.start()

    def _repair_progress_begin(self):
        """Show the render progress bar. Starts as a marquee; the first on_step report from a
        determinate engine (H3/Krea 2) flips it to real step counting."""
        bar = self._repair_progress
        self._repair_progress_det = False
        if not bar.winfo_manager():
            bar.pack(side=tk.RIGHT, padx=(12, 12))
        bar.configure(mode="indeterminate")
        bar.start(60)
        eng = self.repair_engine
        if eng is not None:
            try:
                eng.on_step = self._repair_progress_step
            except Exception:
                pass

    def _repair_progress_step(self, done, total):
        """Engine progress hook — called from the render thread, once per denoising step."""
        def _apply():
            if not self._repair_preview_in_flight:
                return
            bar = self._repair_progress
            if not self._repair_progress_det:
                bar.stop()
                bar.configure(mode="determinate")
                self._repair_progress_det = True
            bar.configure(maximum=max(int(total), 1), value=int(done))
        try:
            self.master.after(0, _apply)
        except Exception:
            pass

    def _print_gpu_process_breakdown(self):
        """One console line saying who holds the GPU right now, per process. Settles the
        'unload freed torch but the status bar still shows N GB' question with data instead
        of guesses — the status bar reads the WHOLE card, so this names our share vs
        everyone else's."""
        try:
            import re
            import subprocess
            me = os.getpid()
            mine, others = 0.0, {}
            # No console flash: a GUI (pythonw) process spawning powershell/nvidia-smi pops
            # a black window without this.
            _nowin = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                      if sys.platform == "win32" else {})
            if sys.platform == "win32":
                # WDDM hides per-process memory from nvidia-smi ([N/A]); the OS performance
                # counters carry it, instance names like "pid_1234_luid_...".
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage')"
                     ".CounterSamples | ForEach-Object "
                     "{ $_.InstanceName + '|' + $_.CookedValue }"],
                    capture_output=True, text=True, timeout=15, **_nowin).stdout
                for ln in out.splitlines():
                    m = re.match(r"pid_(\d+)_.*\|(\d+)", ln.strip())
                    if not m:
                        continue
                    pid, b = int(m.group(1)), float(m.group(2))
                    if b <= 0:
                        continue
                    if pid == me:
                        mine += b
                    else:
                        others[pid] = others.get(pid, 0.0) + b
            else:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, **_nowin).stdout
                for ln in out.splitlines():
                    parts = [p.strip() for p in ln.split(",")]
                    try:
                        pid, mb = int(parts[0]), float(parts[-1])
                    except (ValueError, IndexError):
                        continue
                    if pid == me:
                        mine += mb * 2**20
                    else:
                        others[pid] = others.get(pid, 0.0) + mb * 2**20

            def _name(pid):
                try:
                    import psutil
                    return psutil.Process(pid).name()
                except Exception:
                    return f"pid {pid}"

            top = sorted(others.items(), key=lambda kv: -kv[1])[:3]
            oth = ", ".join(f"{_name(p)} {b/2**30:.1f} GB" for p, b in top if b > 100 * 2**20)
            print(f"[repair] GPU by process: this app {mine/2**30:.2f} GB"
                  + (f" — others: {oth}" if oth else " — nothing sizeable elsewhere"),
                  flush=True)
        except Exception:
            pass

    def _repair_progress_marquee_on(self):
        """Bare marquee for model/LoRA loads — no step reports, just visible life."""
        bar = self._repair_progress
        self._repair_progress_det = False
        if not bar.winfo_manager():
            bar.pack(side=tk.RIGHT, padx=(12, 12))
        bar.configure(mode="indeterminate")
        bar.start(60)

    def _repair_progress_end(self):
        bar = getattr(self, "_repair_progress", None)
        if bar is None:
            return
        try:
            bar.stop()
        except Exception:
            pass
        if bar.winfo_manager():
            bar.pack_forget()

    def _repair_preview_worker(self, snapshot):
        from fizgig.krea2.sampling import SampleAborted
        from fizgig.minimax.sampling import PreviewAborted
        try:
            if self.repair_engine is None:
                self._repair_preview_in_flight = False
                return
            # Fresh render cycle — clear any pending cancel from a previous aborted pass.
            if hasattr(self.repair_engine, "clear_cancel"):
                self.repair_engine.clear_cancel()
            print(f"[repair] worker: generating baseline at "
                  f"{snapshot.preview_width}x{snapshot.preview_height}")
            baseline = self.repair_engine.generate_baseline(snapshot)
            print(f"[repair] worker: baseline done, size={baseline.size}")
            print(f"[repair] worker: generating tweaked at "
                  f"{snapshot.preview_width}x{snapshot.preview_height}")
            tweaked = self.repair_engine.generate_preview(snapshot)
            print(f"[repair] worker: tweaked done, size={tweaked.size}")
            self.master.after(0, lambda: self._set_repair_preview_images(baseline, tweaked))
        except (SampleAborted, PreviewAborted):
            # Cancelled mid-pass by a newer edit — quietly re-fire with the latest state.
            print("[repair] worker: aborted; re-firing with newest state")
            def _refire():
                self._repair_progress_end()
                self._repair_preview_in_flight = False
                self._repair_preview_dirty = False
                if getattr(self, "_repair_unload_wanted", False):
                    return   # the abort came from a pending tab-switch unload — don't re-fire
                self._schedule_preview(force=True)
            self.master.after(0, _refire)
        except Exception:
            import traceback
            err = traceback.format_exc()
            def _show():
                self._repair_progress_end()
                self.repair_status_var.set("Preview error — see console.")
                print(err)
                self._repair_preview_in_flight = False
                # If params changed while we were erroring, still re-fire.
                if self._repair_preview_dirty:
                    self._repair_preview_dirty = False
                    self._schedule_preview(force=True)
            self.master.after(0, _show)

    def _set_repair_preview_images(self, baseline_img, tweaked_img):
        try:
            # Store the raw PIL so <Configure> resize can re-render at any size.
            self.repair_pil_images["baseline"] = baseline_img
            self.repair_pil_images["tweaked"] = tweaked_img
            self._repair_redraw_preview("baseline")
            self._repair_redraw_preview("tweaked")
            self._repair_update_popout()
            self._repair_metrics_refresh()
            self.repair_status_var.set("Ready.")
            print(f"[repair] preview displayed: baseline={baseline_img.size} tweaked={tweaked_img.size}")
        finally:
            self._repair_progress_end()
            self._repair_preview_in_flight = False
            # Dirty flag was set during the in-flight preview → re-fire with
            # fresh state (pulls newest res/seed/prompt/slider values).
            if getattr(self, "_repair_unload_wanted", False):
                self._repair_preview_dirty = False   # unload pending — nothing to re-fire for
            elif self._repair_preview_dirty:
                self._repair_preview_dirty = False
                print("[repair] dirty flag set during preview — refiring")
                self._schedule_preview(force=True)

    def _repair_popout_compose(self):
        """Baseline and tweaked side by side on one canvas \u2014 same left/right order as the
        main panel, 8 px seam. Falls back to whichever image exists alone."""
        base = self.repair_pil_images.get("baseline")
        tweak = self.repair_pil_images.get("tweaked")
        if base is None and tweak is None:
            return None
        if base is None or tweak is None:
            return tweak or base
        from PIL import Image as _Image
        gap = 8
        h = max(base.height, tweak.height)
        canvas = _Image.new("RGB", (base.width + gap + tweak.width, h), (0, 0, 0))
        canvas.paste(base, (0, (h - base.height) // 2))
        canvas.paste(tweak, (base.width + gap, (h - tweak.height) // 2))
        return canvas

    def _repair_popout_preview(self):
        """Open (or raise) a resizable pop-out showing baseline and tweaked side by side."""
        if self._repair_popout_window is not None:
            try:
                if self._repair_popout_window.winfo_exists():
                    self._repair_popout_window.lift()
                    self._repair_update_popout()
                    return
            except Exception:
                pass
            self._repair_popout_window = None

        pil_img = self._repair_popout_compose()
        if pil_img is None:
            return

        win = tk.Toplevel(self.master)
        win.title("Repair Studio \u2014 Baseline vs Tweaked")
        win.configure(bg="#000000")
        # Native size, capped to the screen so a 768 pair doesn't open off-monitor.
        _w = min(pil_img.width, max(640, int(win.winfo_screenwidth() * 0.9)))
        _h = min(pil_img.height, max(360, int(win.winfo_screenheight() * 0.85)))
        win.geometry(f"{_w}x{_h}")
        win.minsize(128, 128)

        # Metrics strip along the bottom — packed FIRST so the image label can never
        # squeeze it out; the fit math below sizes from the LABEL so the image never
        # overflows behind it.
        bar = tk.Frame(win, bg=COLORS["bg_deep"])
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(bar, text="📷 Reference…", font=(FONT_FAMILY, 9),
                  bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                  activebackground=COLORS["bg_hover"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, padx=8, pady=2, cursor="hand2",
                  command=self._browse_repair_metrics_ref).pack(side=tk.LEFT, padx=(8, 2), pady=4)
        self._repair_popout_ref_lbl = tk.Label(bar, text="", font=(FONT_FAMILY, 9),
                                               fg=COLORS["text_explain"], bg=COLORS["bg_deep"])
        self._repair_popout_ref_lbl.pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(bar, text="✕", font=(FONT_FAMILY, 9), bg=COLORS["bg_deep"],
                  fg=COLORS["text_muted"], activebackground=COLORS["bg_deep"],
                  activeforeground=COLORS["text_primary"], relief="flat", bd=0,
                  padx=4, pady=0, cursor="hand2",
                  command=self._clear_repair_metrics_ref).pack(side=tk.LEFT, padx=(0, 10))
        self._repair_popout_metric_lbls = {}
        for key in ("likeness", "grid", "texture", "clip", "sat"):
            c = tk.Label(bar, text="", font=(FONT_FAMILY, 9),
                         fg=COLORS["text_explain"], bg=COLORS["bg_deep"])
            c.pack(side=tk.LEFT, padx=(0, 14), pady=4)
            self._repair_popout_metric_lbls[key] = c
        self._repair_popout_refresh_ref_label()

        lbl = tk.Label(win, bg="#000000")
        lbl.pack(fill=tk.BOTH, expand=True)

        self._repair_popout_window = win
        self._repair_popout_label = lbl

        def _on_close():
            self._repair_popout_window = None
            self._repair_popout_label = None
            self._repair_popout_tk_img = None
            self._repair_popout_metric_lbls = {}
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        def _on_resize(event):
            if event.widget == win:
                self._repair_update_popout()

        win.bind("<Configure>", _on_resize)
        # The fit math sizes from the LABEL, and at this point the label hasn't been laid out
        # (winfo 1x1) — without these two lines the pop-out opened BLACK and stayed black
        # until a manual resize. update_idletasks gives the label its real size for the first
        # paint, and the label's own <Configure> repaints whenever layout hands it new space.
        lbl.bind("<Configure>", lambda e: self._repair_update_popout())
        win.update_idletasks()
        self._repair_update_popout()
        self._repair_metrics_refresh()

    def _repair_update_popout(self):
        """Push the current baseline+tweaked pair to the pop-out window, scaled to fit."""
        if self._repair_popout_window is None or self._repair_popout_label is None:
            return
        try:
            if not self._repair_popout_window.winfo_exists():
                self._repair_popout_window = None
                return
        except Exception:
            self._repair_popout_window = None
            return

        pil_img = self._repair_popout_compose()
        if pil_img is None:
            return

        from PIL import ImageTk
        # Size from the LABEL, not the Toplevel — the metrics bar owns part of the window
        # height, and window-based math scaled the image to overflow behind it.
        w = self._repair_popout_label.winfo_width()
        h = self._repair_popout_label.winfo_height()
        if w < 10 or h < 10:
            return

        # Scale to fit window, preserving aspect ratio (upscale allowed)
        img_w, img_h = pil_img.size
        scale = min(w / img_w, h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = pil_img.resize((new_w, new_h), resample=3)  # LANCZOS=3
        self._repair_popout_tk_img = ImageTk.PhotoImage(resized)
        self._repair_popout_label.configure(image=self._repair_popout_tk_img)

    # ----- pop-out metrics strip ---------------------------------------------------------
    # Overbake instrumentation for the side-by-side view: likeness against a user-chosen
    # reference photo (ArcFace, the app's shared embedder), plus the paired metrics from
    # repair_studio.metrics (patch grid, face texture, clipping/saturation). Paired deltas
    # on a same-seed pair are the one honest use of no-reference image metrics.

    def _browse_repair_metrics_ref(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Reference photo for likeness scoring",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.repair_metrics_ref_var.set(path)
            self._repair_popout_refresh_ref_label()
            self._repair_metrics_refresh()

    def _clear_repair_metrics_ref(self):
        self.repair_metrics_ref_var.set("")
        self._repair_popout_refresh_ref_label()
        self._repair_metrics_refresh()

    def _repair_popout_refresh_ref_label(self):
        lbl = getattr(self, "_repair_popout_ref_lbl", None)
        if lbl is None or not lbl.winfo_exists():
            return
        p = self.repair_metrics_ref_var.get().strip()
        lbl.config(text=os.path.basename(p) if p else "(none)")

    def _repair_metrics_refresh(self):
        """Kick the metrics worker for the current image pair, if the pop-out is open."""
        win = self._repair_popout_window
        try:
            if win is None or not win.winfo_exists() or not self._repair_popout_metric_lbls:
                return
        except Exception:
            return
        base = self.repair_pil_images.get("baseline")
        tweak = self.repair_pil_images.get("tweaked")
        if base is None or tweak is None:
            return
        self._repair_metrics_gen += 1
        gen = self._repair_metrics_gen
        for c in self._repair_popout_metric_lbls.values():
            c.config(text="…", fg=COLORS["text_muted"])
        fam = (self.repair_family_var.get()
               if getattr(self, "repair_family_var", None) is not None else "klein")
        ref = self.repair_metrics_ref_var.get().strip()
        threading.Thread(target=self._repair_metrics_worker,
                         args=(base.copy(), tweak.copy(), fam, ref, gen),
                         daemon=True).start()

    def _repair_metrics_worker(self, base, tweak, fam, ref_path, gen):
        try:
            import numpy as np
            from fizgig.repair_studio.metrics import PATCH_PITCH, compare
            base_emb, base_bbox = self._repair_embed_pil(base)
            tweak_emb, tweak_bbox = self._repair_embed_pil(tweak)
            ref_emb = (self._ff_embed_cached(ref_path)
                       if ref_path and os.path.isfile(ref_path) else None)
            m = compare(np.array(base.convert("RGB")), np.array(tweak.convert("RGB")),
                        PATCH_PITCH.get(fam, 16), base_bbox, tweak_bbox)
            m["ref_set"] = bool(ref_path)
            m["ref_face"] = ref_emb is not None
            m["like_base"] = (float(np.dot(ref_emb, base_emb))
                              if ref_emb is not None and base_emb is not None else None)
            m["like_tweak"] = (float(np.dot(ref_emb, tweak_emb))
                               if ref_emb is not None and tweak_emb is not None else None)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
        try:
            self.master.after(0, lambda: self._repair_metrics_apply(m, gen))
        except Exception:
            pass    # app shutting down mid-computation — nowhere to paint, nothing to do

    def _repair_metrics_apply(self, m, gen):
        """Paint the chips — only if these numbers still describe the images on screen."""
        if gen != self._repair_metrics_gen:
            return
        lbls = self._repair_popout_metric_lbls
        try:
            if not lbls or not self._repair_popout_window.winfo_exists():
                return
        except Exception:
            return
        GOOD, BAD, WARM, DIM = "#2ECC71", "#E74C3C", "#F39C12", COLORS["text_explain"]
        if "error" in m:
            lbls["likeness"].config(text=f"metrics failed: {m['error'][:60]}", fg=BAD)
            for k in ("grid", "texture", "clip", "sat"):
                lbls[k].config(text="", fg=DIM)
            return
        # Likeness vs the reference photo
        if not m["ref_set"]:
            lbls["likeness"].config(text="Likeness: set a reference photo →", fg=DIM)
        elif not m["ref_face"]:
            lbls["likeness"].config(text="Likeness: no face in reference", fg=BAD)
        elif m["like_base"] is None or m["like_tweak"] is None:
            lbls["likeness"].config(text="Likeness: no face in render", fg=DIM)
        else:
            lb, lt = m["like_base"] * 100, m["like_tweak"] * 100
            d = lt - lb
            arrow = "▲" if d > 0.5 else ("▼" if d < -0.5 else "→")
            lbls["likeness"].config(
                text=f"Likeness {lb:.0f}% {arrow} {lt:.0f}%",
                fg=GOOD if d > 0.5 else (BAD if d < -0.5 else DIM))
        # Patch grid: rising = the model's lattice is showing through (bad)
        gd = m["grid_delta"]
        lbls["grid"].config(
            text=f"Grid {m['grid_base']:.2f} → {m['grid_tweak']:.2f}",
            fg=BAD if gd > 0.05 else (GOOD if gd < -0.05 else DIM))
        # Face texture: direction is information, not verdict (plastic vs fried)
        tb, tt = m["texture_base"], m["texture_tweak"]
        td = (tt - tb) / max(tb, 1e-6)
        lbls["texture"].config(
            text=f"Detail {tb:.0f} → {tt:.0f}",
            fg=WARM if abs(td) > 0.10 else DIM)
        # Clipping: blown pixels appearing is the earliest overbake tell
        cd = m["clip_delta"]
        lbls["clip"].config(
            text=f"Clipped {m['clip_base']:.1f}% → {m['clip_tweak']:.1f}%",
            fg=BAD if cd > 0.2 else (GOOD if cd < -0.2 else DIM))
        sd = m["sat_delta"]
        lbls["sat"].config(
            text=f"Sat {m['sat_base']:.0f} → {m['sat_tweak']:.0f}",
            fg=WARM if abs(sd) > 12 else DIM)

    def _save_repaired_lora_action(self):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA first.")
            return
        # Does the current slider state enable any donor blocks?
        donor_enabled_bids = [bid for bid, bs in self.repair_state.blocks.items() if bs.donor_enabled]
        donor_loaded = (self.repair_engine.donor_network is not None)
        donor_path = self.repair_engine.donor_path if donor_loaded else None

        primary_stem = os.path.splitext(os.path.basename(self.repair_engine.primary_path))[0]
        if donor_enabled_bids and donor_loaded:
            donor_stem = os.path.splitext(os.path.basename(donor_path))[0]
            default_name = f"{primary_stem}_with_{donor_stem}.safetensors"
        else:
            default_name = f"{primary_stem}_repaired.safetensors"

        out = filedialog.asksaveasfilename(
            title="Save Repaired LoRA",
            defaultextension=".safetensors",
            filetypes=[("SafeTensors", "*.safetensors")],
            initialfile=default_name,
        )
        if not out:
            return
        from fizgig.repair_studio.bake import save_repaired_lora
        from fizgig.networks.lora import UnsupportedLoRAFormat
        try:
            summary = save_repaired_lora(
                self.repair_engine.primary_path,
                self.repair_state,
                out,
                donor_path=donor_path if donor_enabled_bids else None,
            )
            msg = (
                f"Saved: {out}\n\n"
                f"Keys: {summary['keys_in']} → {summary['keys_out']}\n"
                f"Dropped blocks ({len(summary['dropped_blocks'])}): "
                f"{', '.join(summary['dropped_blocks']) or 'none'}\n"
                f"Rescaled blocks ({len(summary['rescaled_blocks'])}): "
                f"{', '.join(summary['rescaled_blocks']) or 'none'}\n"
                f"Donor-blended blocks ({len(summary['blended_blocks'])}): "
                f"{', '.join(summary['blended_blocks']) or 'none'}"
            )
            if summary['blended_blocks']:
                msg += "\n\nNote: blended blocks have rank = rank_primary + rank_donor. File size grows proportionally."
            if summary.get('format_out') == 'lycoris':
                msg += "\n\nSaved natively in LyCORIS format (LoKR/LoHa) — lossless, no conversion."
            elif summary.get('lycoris_converted'):
                msg += (f"\n\n{summary['lycoris_converted']} blended LyCORIS module(s) were "
                        f"converted to standard LoRA via SVD; everything else stayed native.")
            messagebox.showinfo("Repaired LoRA saved", msg)
        except UnsupportedLoRAFormat as ex:
            messagebox.showerror("Bake not supported for this LoRA format", str(ex))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _reset_repair_sliders(self):
        from fizgig.repair_studio.state import SliderState
        # Family-correct layout — the Klein default's block ids match nothing on Krea 2 / H3
        # panels, making Reset All a silent no-op there (same trap as GitHub #12).
        _fam = self.repair_family_var.get() if getattr(self, "repair_family_var", None) else "klein"
        defaults = (SliderState.default_krea2() if _fam == "krea2"
                    else SliderState.default_h3() if _fam == "minimax"
                    else SliderState.default_klein9b())
        # Suppress per-block preview spam while bulk-resetting.
        self._repair_master_mutating = True
        try:
            for bid, bs in defaults.blocks.items():
                v = self.repair_block_vars.get(bid)
                if v is None:
                    continue
                v["primary_enabled"].set(bs.primary_enabled)
                v["primary_strength"].set(bs.primary_strength)
                v["donor_enabled"].set(bs.donor_enabled)
                v["donor_strength"].set(bs.donor_strength)
            # Reset master sliders + donor category toggles too.
            for cat in self.repair_master_strength_vars:
                self.repair_master_strength_vars[cat].set(1.0)
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
        finally:
            self._repair_master_mutating = False
        self._schedule_preview(force=True)

    def _unload_repair_studio_models(self):
        """Unload Repair Studio pipeline + networks to free VRAM when leaving the tab.

        Unlike _reset_repair_session, this preserves all UI state (slider
        positions, loaded paths, etc.) so re-entering the tab and clicking
        Load again is seamless.  The engine is fully reset so the next load
        rebuilds the pipeline from scratch.
        """
        # Internal guard (all call sites): resetting under a live CUDA worker hard-hangs.
        # But a silent bail here LEAKED the whole pipeline (Peter: 22.6 GB held until app
        # quit) — an H3 preview runs ~12s, plenty of time to switch tabs mid-render, and
        # nothing ever retried the unload. Now: cancel the render and retry until the worker
        # lands, then unload for real.
        if (getattr(self, "_repair_preview_in_flight", False)
                or getattr(self, "_repair_loading", False)):
            self._repair_unload_wanted = True     # stops the abort path re-firing a preview
            eng = self.repair_engine
            if eng is not None and hasattr(eng, "request_cancel"):
                try:
                    eng.request_cancel()
                except Exception:
                    pass
            tries = getattr(self, "_repair_unload_tries", 0) + 1
            self._repair_unload_tries = tries
            if tries <= 120:                      # 60 s of patience, then give up loudly
                self.master.after(500, self._unload_repair_studio_models)
            else:
                self._repair_unload_tries = 0
                self._repair_unload_wanted = False
                print("[repair] unload gave up waiting for the render worker — "
                      "models are still loaded")
            return
        self._repair_unload_tries = 0
        self._repair_unload_wanted = False
        if self.repair_engine is not None and self.repair_engine.pipeline is not None:
            print("[repair] tab-switch unload: freeing models…", flush=True)
            try:
                self.repair_engine.reset()
            except Exception:
                import traceback
                print("[repair] tab-switch unload: engine reset RAISED —\n"
                      + traceback.format_exc(), flush=True)
            self.repair_engine = None
            self.repair_status_var.set("Models unloaded (tab switch). Load a LoRA to resume.")

    def _repair_explore_in_explorer(self):
        """Send current Repair Studio slider state to the Explorer for evolutionary discovery."""
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA first.")
            return
        # Mirror of the Explorer-side guard: _reset_repair_session below refuses to tear down
        # mid-preview, but it returns into THIS method which would then build the Explorer
        # pipeline alongside the still-rendering preview — two pipelines, two threads, one
        # GPU, GUI thread stuck in a CUDA call. Stop the whole handoff instead.
        if getattr(self, "_repair_preview_in_flight", False):
            self.repair_status_var.set(
                "A preview is still rendering — wait for it to finish, then Explore.")
            return

        # Warn if LyCORIS — saving from Explorer will require SVD
        lora_path = self.repair_engine.primary_path
        try:
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt = _df(_ek(_lf(lora_path)))
            # LyCORIS saves natively now (lossless) — no SVD gate needed on the handoff.
        except Exception:
            pass

        # Warn if donor is loaded — Explorer only supports primary
        if self.repair_engine.donor_network is not None:
            proceed = messagebox.askyesno(
                "Donor LoRA loaded",
                "The Explorer only works with a single primary LoRA — "
                "donor blending isn't supported there.\n\n"
                "Continue with just the primary LoRA's slider state?\n\n"
                "Tip: you can Save Repaired LoRA first to bake the primary+donor "
                "blend into a single file, then explore that.")
            if not proceed:
                return

        from fizgig.repair_studio.state import SliderState

        # Capture current state
        lora_path = self.repair_engine.primary_path
        current_state = self.repair_state.copy()
        prompt = self.repair_prompt_var.get()
        seed = self.repair_seed_var.get()
        res = self.repair_res_var.get()

        # Reset Repair Studio (frees VRAM)
        self._reset_repair_session()

        # Set up Explorer fields
        self.explorer_lora_var.set(lora_path)
        self.explorer_prompt_var.set(prompt)
        self.explorer_seed_var.set(seed)
        self.explorer_res_var.set(res)
        # Carry the reference image (path, MP, strength) across the handover.
        if hasattr(self, "explorer_ref_path_var"):
            self.explorer_ref_path_var.set(self.repair_ref_path_var.get().strip())
            self.explorer_ref_mp_var.set(self.repair_ref_mp_var.get())
            self.explorer_ref_strength_var.set(self.repair_ref_strength_var.get())

        # Handoff inherits the Repair Studio's family — switch the Explorer to match (so it loads
        # the right engine + hides the DiT radio/ref-strength for krea2).
        target_family = "krea2" if self.repair_family_var.get() == "krea2" else "klein"
        if self.explorer_family_var.get() != target_family:
            self.explorer_family_var.set(target_family)
            self.last_used["explorer_family"] = target_family
            self._apply_explorer_family_ui(target_family == "krea2")

        # Switch to Explorer tab
        self.notebook.select(self.explorer_tab)

        # Load LoRA in Explorer
        if not self._explorer_ensure_engine():
            return
        try:
            self.explorer_status_var.set("Loading from Repair Studio...")
            self.master.update_idletasks()
            if self._explorer_engine.primary_network is not None:
                if getattr(self, "_explorer_generating", False):
                    messagebox.showinfo("Busy", "The Explorer is mid-render — try again when "
                                        "the current preview finishes.")
                    return
                self._explorer_engine.reset()
                self._explorer_engine = None
                if not self._explorer_ensure_engine():
                    return
            self._explorer_engine.load_primary(lora_path)

            # Set the Explorer baseline to the Repair Studio's slider state
            self._explorer_baseline_state = current_state
            self._explorer_baseline_state.prompt = prompt
            self._explorer_baseline_state.seed = int(seed or 42)
            r = int(res or 512)
            self._explorer_baseline_state.preview_width = r
            self._explorer_baseline_state.preview_height = r
            self._explorer_history.clear()
            self._explorer_locked_blocks.clear()
            self._explorer_last_pick_blocks.clear()
            self._explorer_baseline_image = None
            self._explorer_undo_btn.configure(state="disabled")
            self._explorer_save_btn.configure(state="disabled")
            self._explorer_refine_btn.configure(state="disabled")
            self._explorer_freeze_btn.configure(state="disabled")
            self._explorer_roll_btn.configure(state="normal")

            # Set low intensity + structure for refinement (subtle variants)
            self.explorer_intensity_var.set(0.25)   # ±0.9
            self.explorer_structure_var.set(0.15)    # 15%

            n_active = len(self._explorer_engine.primary_block_ids)
            self.explorer_status_var.set(
                f"Loaded from Repair Studio: {os.path.basename(lora_path)} "
                f"({n_active}/{len(self._explorer_baseline_state.blocks) if self._explorer_baseline_state else 32} blocks). "
                f"Refining with low intensity. Generating variants...")
            self._explorer_generate_baseline_and_roll()

            self.master.after(500, lambda: messagebox.showinfo(
                "Refinement Mode",
                "Your Repair Studio slider state is now the Explorer baseline.\n\n"
                "Intensity and Structure have been set low so variants are subtle "
                "refinements of your current settings. Increase them if you want "
                "bolder exploration."))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load in Explorer:\n{traceback.format_exc()}")

    def _reset_repair_session(self):
        # Never tear down the engine while a preview worker is mid-forward through it.
        if getattr(self, "_repair_preview_in_flight", False):
            messagebox.showinfo("Busy", "A preview is still rendering — wait for it to finish "
                                "before resetting the session.")
            return
        # Close pop-out preview window if open
        if self._repair_popout_window is not None:
            try:
                self._repair_popout_window.destroy()
            except Exception:
                pass
            self._repair_popout_window = None
            self._repair_popout_label = None
            self._repair_popout_tk_img = None
        if self.repair_engine is not None:
            print("[repair] reset session: freeing models…", flush=True)
            try:
                self.repair_engine.reset()
            except Exception:
                import traceback
                print("[repair] reset session: engine reset RAISED —\n"
                      + traceback.format_exc(), flush=True)
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _free, _total = _t.cuda.mem_get_info()
                    print(f"[repair] reset session done — "
                          f"allocated {_t.cuda.memory_allocated()/2**30:.2f} GB, "
                          f"reserved {_t.cuda.memory_reserved()/2**30:.2f} GB, "
                          f"whole GPU in use {(_total-_free)/2**30:.2f} GB", flush=True)
                self._print_gpu_process_breakdown()
            except Exception:
                pass
        self.repair_engine = None
        self._repair_donor_loaded = False
        self._repair_preview_in_flight = False
        self._repair_preview_dirty = False
        for vars_ in self.repair_block_vars.values():
            vars_["donor_rowf"].grid_remove()
        # Hide master donor toggles + disable donor radio
        try:
            # donor toggles removed — donor blocks managed via master sliders
            self._repair_master_donor_radio.state(["disabled"])
            self.repair_master_target_var.set("primary")
        except Exception:
            pass
        # Reset master sliders to defaults (no preview — nothing loaded)
        self._repair_master_mutating = True
        try:
            for cat in self.repair_master_strength_vars:
                self.repair_master_strength_vars[cat].set(1.0)
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
        finally:
            self._repair_master_mutating = False
        # Restore all sliders to the pre-load visual default.
        self._refresh_block_slider_activity()
        self.repair_thumbnails.clear()
        self.repair_pil_images["baseline"] = None
        self.repair_pil_images["tweaked"] = None
        self.repair_baseline_label.configure(image="", text="(no baseline yet)")
        self.repair_tweaked_label.configure(image="", text="(no preview yet)")
        # Clear any profile-match panel.
        self.repair_profile_match = None
        try:
            self.repair_profile_frame.pack_forget()
        except Exception:
            pass
        self.repair_status_var.set("Session reset. Load a primary LoRA to start.")

    def _find_repair_profile_match(self):
        """Look up a Profiler sidecar whose hash matches the loaded primary.
        If found, render the info panel; otherwise hide it."""
        self.repair_profile_match = None
        if self.repair_engine is None or not self.repair_engine.primary_hash:
            self._render_repair_profile_panel()
            return
        from fizgig.repair_studio.engine import find_profile_for_hash
        profiles_dir = self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars else ""
        try:
            match = find_profile_for_hash(profiles_dir, self.repair_engine.primary_hash)
        except Exception:
            match = None
        self.repair_profile_match = match
        self._render_repair_profile_panel()

    def _render_repair_profile_panel(self):
        """Populate (or hide) the profile-match info panel."""
        frame = self.repair_profile_frame
        # Clear previous children.
        for child in list(frame.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        match = self.repair_profile_match
        if not match:
            frame.pack_forget()
            return

        lora_name = match.get("lora_name") or "(unknown)"
        created = match.get("created") or ""
        top = match.get("top_active_blocks") or []
        if not top:
            frame.pack_forget()
            return

        # Ensure visible in the correct slot (between Status line and Preview card).
        frame.pack(fill=tk.X, padx=36, pady=(0, 16), before=self._repair_profile_anchor)

        tk.Label(frame, text="Profile found for this LoRA",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(
            anchor=tk.W, padx=20, pady=(16, 2)
        )
        tk.Label(
            frame,
            text=f"{lora_name}  ·  profiled {created[:10] if created else ''}  ·  most active blocks:",
            font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
        ).pack(anchor=tk.W, padx=20, pady=(0, 8))

        body = tk.Frame(frame, bg=COLORS["bg_surface"])
        body.pack(fill=tk.X, padx=20, pady=(0, 16))

        pills_frame = tk.Frame(body, bg=COLORS["bg_surface"])
        pills_frame.pack(side=tk.LEFT, anchor=tk.W)
        for idx, b in enumerate(top[:8]):
            name = b.get("name", "")
            category = b.get("category", "identity")
            pct = b.get("pct", 0)
            color = self._REPAIR_CAT_COLOR.get(category, "#888")
            pill = tk.Label(
                pills_frame, text=f"{name}  {pct:.1f}%",
                fg="#FFFFFF", bg=color,
                font=(FONT_FAMILY, 9, "bold"),
                padx=8, pady=3,
            )
            pill.pack(side=tk.LEFT, padx=(0, 4))

        # Open Report button — launches the HTML in the system browser.
        sidecar_path = match.get("_sidecar_path")
        html_name = match.get("html_report") or ""
        html_path = os.path.join(os.path.dirname(sidecar_path), html_name) if (sidecar_path and html_name) else None
        if html_path and os.path.isfile(html_path):
            ttk.Button(body, text="📊 Open full report",
                       command=lambda p=html_path: self._open_repair_profile_report(p)).pack(
                side=tk.RIGHT, padx=(8, 0))

    def _open_repair_profile_report(self, html_path: str):
        import webbrowser
        try:
            webbrowser.open(html_path)
        except Exception:
            messagebox.showerror("Error", f"Could not open report:\n{html_path}")

    def _refresh_block_slider_activity(self):
        """Grey out primary/donor rows for blocks the loaded LoRA doesn't touch.

        Called after primary/donor load, donor unload, and session reset.
        When no engine is loaded, everything is restored to normal.
        """
        primary_ids = (
            self.repair_engine.primary_block_ids
            if self.repair_engine is not None and self.repair_engine.primary_network is not None
            else None
        )
        donor_ids = (
            self.repair_engine.donor_block_ids
            if self.repair_engine is not None and self.repair_engine.donor_network is not None
            else None
        )
        grey_fg = "#555"

        for block_id, v in self.repair_block_vars.items():
            # Primary activity
            p_active = primary_ids is None or block_id in primary_ids
            p_state = ["!disabled"] if p_active else ["disabled"]
            try:
                v["chk_p"].state(p_state)
                v["scale_p"].state(p_state)
            except Exception:
                pass
            for btn in v.get("btns_p", []):
                btn.configure(state="normal" if p_active else "disabled")
            p_color = v["cat_color"] if p_active else grey_fg
            v["block_lbl"].configure(fg=p_color)
            if v.get("cat_lbl") is not None:  # Krea 2 rows have no category label
                v["cat_lbl"].configure(fg=p_color)
            v["primary_lbl"].configure(foreground=p_color if not p_active else "")
            if not p_active:
                # Reset var to default so an absent block never carries stale edits.
                v["primary_enabled"].set(True)
                v["primary_strength"].set(1.0)

            # Donor activity (only meaningful when donor row is visible)
            d_active = donor_ids is None or block_id in donor_ids
            d_state = ["!disabled"] if d_active else ["disabled"]
            try:
                v["chk_d"].state(d_state)
                v["scale_d"].state(d_state)
            except Exception:
                pass
            for btn in v.get("btns_d", []):
                btn.configure(state="normal" if d_active else "disabled")
            d_color = "#888" if d_active else grey_fg
            v["donor_tag_lbl"].configure(foreground=d_color)
            v["donor_lbl"].configure(foreground=d_color if d_active else grey_fg)
            if donor_ids is not None and not d_active:
                v["donor_enabled"].set(True)
                v["donor_strength"].set(0.0)

    # ---------------- Presets (built-in + user JSON) -----------------

    _REPAIR_BUILTIN_PRESETS = {
        "✨Reset All": "reset",
        "✨Identity Only": "identity",
        "✨Style+Composition Only": "style",
        "✨Details Only": "details",
    }

    def _repair_preset_dir(self) -> str:
        """Per-family folder — a preset is a set of per-block sliders, and block ids only mean
        anything on the family they were saved from (a Klein 32-block state applied to H3's 52
        sliders matches nothing and silently does nothing). Separate folders keep each family's
        dropdown honest."""
        fam = (self.repair_family_var.get()
               if getattr(self, "repair_family_var", None) is not None else "klein")
        d = os.path.join(_REPO_ROOT, "presets", "repair_studio", fam)
        os.makedirs(d, exist_ok=True)
        return d

    def _repair_is_krea2(self) -> bool:
        """Historical name — True for ANY no-block-map family (Krea 2 or MiniMax H3), which
        is what every caller actually means: no category presets, no master sliders."""
        return (getattr(self, "repair_family_var", None) is not None
                and self.repair_family_var.get() in ("krea2", "minimax"))

    def _repair_preset_list(self) -> list:
        if self._repair_is_krea2():
            # No Krea 2 / H3 semantic block map yet — only Reset All is meaningful there.
            names = ["✨Reset All"]
        else:
            names = list(self._REPAIR_BUILTIN_PRESETS.keys())
        try:
            for fn in sorted(os.listdir(self._repair_preset_dir())):
                if fn.lower().endswith(".json"):
                    names.append(os.path.splitext(fn)[0])
        except Exception:
            pass
        return names

    def _refresh_repair_preset_combo(self):
        if hasattr(self, "repair_preset_combo"):
            self.repair_preset_combo.configure(values=self._repair_preset_list())

    def _apply_repair_blocks_to_widgets(self, state):
        """Sliders only — the shape user presets restore. Prompt, seed, res, reference and
        the loaded LoRAs are session context, not part of a block recipe (Peter, 19 Aug)."""
        for bid, bs in state.blocks.items():
            v = self.repair_block_vars.get(bid)
            if v is None:
                continue
            v["primary_enabled"].set(bs.primary_enabled)
            v["primary_strength"].set(bs.primary_strength)
            v["donor_enabled"].set(bs.donor_enabled)
            v["donor_strength"].set(bs.donor_strength)

    def _apply_repair_state_to_widgets(self, state):
        self._apply_repair_blocks_to_widgets(state)
        self.repair_seed_var.set(str(state.seed))
        self.repair_prompt_var.set(state.prompt)
        self.repair_res_var.set(str(state.preview_width))
        if hasattr(self, "repair_ref_path_var"):
            self.repair_ref_path_var.set(getattr(state, "ref_image_path", "") or "")
            self.repair_ref_mp_var.set(str(getattr(state, "ref_megapixels", 1.0)))
            self.repair_ref_strength_var.set(str(getattr(state, "ref_strength", 1.0)))

    def _repair_builtin_state(self, kind: str):
        from fizgig.repair_studio.state import SliderState
        # Family-correct layout: a Klein-shaped state applied to Krea 2 / H3 widgets matches
        # no slider vars and silently does nothing (GitHub #12).
        _fam = self.repair_family_var.get() if getattr(self, "repair_family_var", None) else "klein"
        s = (SliderState.default_krea2() if _fam == "krea2"
             else SliderState.default_h3() if _fam == "minimax"
             else SliderState.default_klein9b())
        s.seed = self.repair_state.seed
        s.prompt = self.repair_state.prompt
        s.preview_width = self.repair_state.preview_width
        s.preview_height = self.repair_state.preview_height
        if kind == "reset" or self._repair_is_krea2():
            return s
        if kind == "identity":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("identity", "style_ident_overlap", "ident_details_overlap")
            return s
        if kind == "style":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("style_composition", "style_ident_overlap")
            return s
        if kind == "details":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("details", "ident_details_overlap")
            return s
        return s

    def _save_repair_preset(self):
        name = simpledialog.askstring("Save Repair Studio Preset", "Preset name:")
        if not name:
            return
        # Sanitize
        name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not name or name.startswith("✨"):
            messagebox.showerror("Error", "Invalid preset name.")
            return
        path = os.path.join(self._repair_preset_dir(), f"{name}.json")
        if os.path.exists(path):
            if not messagebox.askokcancel("Overwrite?", f"Overwrite existing preset '{name}'?"):
                return
        try:
            with open(path, "w", encoding="utf-8") as f:
                import json as _json
                # SLIDERS ONLY. A preset is a block recipe — prompt, seed, resolution, the
                # reference image and the loaded LoRAs are the session it gets applied TO,
                # and saving them meant loading a preset yanked all of them out from under
                # the user (Peter, 19 Aug).
                _d = {"blocks": self.repair_state.to_json()["blocks"]}
                # Self-describing: which family's block ids these are. The folder already
                # scopes the dropdown; this makes a shared/copied file readable on its own.
                _d["family"] = (self.repair_family_var.get()
                                if getattr(self, "repair_family_var", None) is not None
                                else "klein")
                _json.dump(_d, f, indent=2)
            self._refresh_repair_preset_combo()
            self.repair_preset_var.set(name)
            messagebox.showinfo("Saved", f"Saved preset: {name}")
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _load_repair_preset(self, name: str):
        if not name:
            return
        from fizgig.repair_studio.state import SliderState
        if name in self._REPAIR_BUILTIN_PRESETS:
            state = self._repair_builtin_state(self._REPAIR_BUILTIN_PRESETS[name])
            self._apply_repair_state_to_widgets(state)
            self._schedule_preview(force=True)
            return
        path = os.path.join(self._repair_preset_dir(), f"{name}.json")
        if not os.path.exists(path):
            return
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                d = _json.load(f)
            state = SliderState.from_json(d)
            # Blocks only — a preset saved by an older build carries prompt/seed/res too;
            # they are deliberately ignored so loading never disturbs the live session.
            self._apply_repair_blocks_to_widgets(state)
            self._schedule_preview(force=True)
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Load failed:\n{traceback.format_exc()}")
