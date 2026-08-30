import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class ExplorerTabMixin:
    def create_explorer_tab(self):
        """LoRA the Explorer — evolutionary LoRA discovery via human-guided selection."""
        scrollable_frame, _ = self.create_scrollable_frame(self.explorer_tab)
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer, "LoRA the Explorer",
            "The computer randomly adjusts blocks and shows you 4 variants — pick your favourite and it evolves. "
            "Find a direction you like? Reduce Structure to stabilise composition, then Freeze to lock tweaked blocks in place.",
        )

        # State
        self._explorer_engine = None
        self._explorer_baseline_state = None
        self._explorer_baseline_image = None
        self._explorer_history = []  # stack of (SliderState, PIL.Image) for undo
        self._explorer_variant_states = []  # 4 SliderState objects
        self._explorer_variant_images = []  # 4 PIL.Image objects
        self._explorer_generating = False
        self._explorer_thumbnails = {}  # keep refs to prevent GC
        self._explorer_locked_blocks = set()  # Freeze: blocks locked at their current value
        self._explorer_last_pick_blocks = set()  # blocks changed in the most recent pick

        # Model family selector. Krea 2 explores on the fp8 Turbo (always) and has no ref-strength
        # dial (vision-path reference), so the DiT radio + ref Strength are hidden in Krea 2 mode.
        _xfam = str(self.last_used.get("explorer_family", "klein"))
        if _xfam not in ("klein", "krea2", "minimax"):
            _xfam = "klein"
        self.explorer_family_var = tk.StringVar(value=_xfam)
        xfam_card = self._start_section_card(
            outer, "Model Family",
            "Klein 9B (Distilled/Base), Krea 2 (fp8 Turbo, 8-step) or MiniMax H3 (22-frame clip "
            "previews, middle frame shown). Block roles are mapped for Klein; the other two "
            "explore their blocks generically.",
        )
        _xf = tk.Frame(xfam_card, bg=COLORS["bg_surface"])
        _xf.pack(anchor=tk.W)
        ttk.Radiobutton(_xf, text="Klein 9B", variable=self.explorer_family_var, value="klein",
                        command=self._on_explorer_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_xf, text="Krea 2", variable=self.explorer_family_var, value="krea2",
                        command=self._on_explorer_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_xf, text="MiniMax H3", variable=self.explorer_family_var, value="minimax",
                        command=self._on_explorer_family_changed).pack(side=tk.LEFT)

        # Card 1: Setup
        setup_card = self._start_section_card(
            outer, "Setup",
            "Load a LoRA and configure the exploration parameters.",
        )
        setup_card.columnconfigure(1, weight=1)

        r = 0
        self._explorer_dit_label = ttk.Label(setup_card, text="DiT:")
        self._explorer_dit_label.grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_dit_var = tk.StringVar(value="distilled")
        dit_frame = ttk.Frame(setup_card)
        dit_frame.grid(row=r, column=1, sticky=tk.W, pady=2)
        self._explorer_dit_frame = dit_frame
        ttk.Radiobutton(dit_frame, text="Distilled (4-step, fast)",
                        variable=self.explorer_dit_var, value="distilled",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(dit_frame, text="Base (20-step, precise)",
                        variable=self.explorer_dit_var, value="base",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT)
        r += 1

        ttk.Label(setup_card, text="LoRA:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_lora_var = tk.StringVar()
        ttk.Entry(setup_card, textvariable=self.explorer_lora_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        btn_frame = ttk.Frame(setup_card)
        btn_frame.grid(row=r, column=2, pady=2)
        ttk.Button(btn_frame, text="Browse",
                   command=lambda: self._browse_repair_lora(self.explorer_lora_var)).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="Strength:").pack(side=tk.LEFT, padx=(12, 4))
        self.explorer_strength_var = tk.StringVar(value="1.0")
        ttk.Entry(btn_frame, textvariable=self.explorer_strength_var, width=5).pack(side=tk.LEFT)
        r += 1

        ttk.Label(setup_card, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_prompt_var = tk.StringVar()
        ttk.Entry(setup_card, textvariable=self.explorer_prompt_var).grid(
            row=r, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        r += 1

        # Reference image (Klein edit conditioning). Carried in the baseline
        # SliderState, so a ref set here follows the LoRA into Repair Studio.
        ref_frame = ttk.Frame(setup_card)
        ref_frame.grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=(2, 2))
        ttk.Label(ref_frame, text="Reference:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_ref_path_var = tk.StringVar(value="")
        ttk.Entry(ref_frame, textvariable=self.explorer_ref_path_var, state="readonly",
                  width=28).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(ref_frame, text="Browse", command=self._browse_explorer_ref).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(ref_frame, text="Clear", command=self._clear_explorer_ref).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(ref_frame, text="MP:").pack(side=tk.LEFT, padx=(0, 2))
        self.explorer_ref_mp_var = tk.StringVar(value="1.0")
        ttk.Combobox(ref_frame, textvariable=self.explorer_ref_mp_var,
                     values=["0.25", "0.5", "1.0", "2.0"], state="readonly", width=5).pack(side=tk.LEFT, padx=(0, 10))
        self._explorer_ref_strength_label = ttk.Label(ref_frame, text="Strength:")
        self._explorer_ref_strength_label.pack(side=tk.LEFT, padx=(0, 2))
        self.explorer_ref_strength_var = tk.StringVar(value="1.0")
        self._explorer_ref_strength_entry = ttk.Entry(ref_frame, textvariable=self.explorer_ref_strength_var, width=6)
        self._explorer_ref_strength_entry.pack(side=tk.LEFT)
        r += 1

        params_frame = ttk.Frame(setup_card)
        params_frame.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
        ttk.Label(params_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_seed_var = tk.StringVar(value="42")
        ttk.Entry(params_frame, textvariable=self.explorer_seed_var, width=8).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(params_frame, text="\u21bb", font=(FONT_FAMILY, 9),
                  bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                  activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, padx=4, pady=0, cursor="hand2",
                  command=self._explorer_randomize_seed
                  ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(params_frame, text="Res:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_res_var = tk.StringVar(value="512")
        ttk.Combobox(params_frame, textvariable=self.explorer_res_var,
                     values=["256", "384", "512", "768"], state="readonly", width=5).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(params_frame, text="Intensity:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_intensity_var = tk.DoubleVar(value=0.964)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.explorer_intensity_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 4))
        self._explorer_intensity_lbl = ttk.Label(params_frame, text="\u00b12.9", width=5)
        self._explorer_intensity_lbl.pack(side=tk.LEFT, padx=(0, 16))
        self._explorer_intensity_debounce_id = None
        def _update_intensity_lbl(*_):
            mag = 0.2 + self.explorer_intensity_var.get() * 2.8
            self._explorer_intensity_lbl.configure(text=f"\u00b1{mag:.1f}")
            # Debounced re-roll when intensity changes
            if self._explorer_baseline_state is not None and not self._explorer_generating:
                if self._explorer_intensity_debounce_id is not None:
                    try:
                        self.master.after_cancel(self._explorer_intensity_debounce_id)
                    except Exception:
                        pass
                self._explorer_intensity_debounce_id = self.master.after(
                    750, self._explorer_reroll)
        self.explorer_intensity_var.trace_add("write", _update_intensity_lbl)
        ttk.Label(params_frame, text="Mutations:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_mutations_var = tk.StringVar(value="8")
        ttk.Combobox(params_frame, textvariable=self.explorer_mutations_var,
                     values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16"], state="readonly", width=3).pack(side=tk.LEFT, padx=(0, 16))
        # Hold Mode removed — replaced by Freeze Tweaked Blocks button
        r += 1

        struct_frame = ttk.Frame(setup_card)
        struct_frame.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))
        ttk.Label(struct_frame, text="Structure change:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_structure_var = tk.DoubleVar(value=1.0)
        ttk.Scale(struct_frame, from_=0.0, to=1.0, variable=self.explorer_structure_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 4))
        self._explorer_structure_lbl = ttk.Label(struct_frame, text="100%", width=5)
        self._explorer_structure_lbl.pack(side=tk.LEFT, padx=(0, 8))
        self._explorer_structure_debounce_id = None
        def _update_structure_lbl(*_):
            val = self.explorer_structure_var.get()
            self._explorer_structure_lbl.configure(text=f"{int(val * 100)}%")
            if self._explorer_baseline_state is not None and not self._explorer_generating:
                if self._explorer_structure_debounce_id is not None:
                    try:
                        self.master.after_cancel(self._explorer_structure_debounce_id)
                    except Exception:
                        pass
                self._explorer_structure_debounce_id = self.master.after(
                    750, self._explorer_reroll)
        self.explorer_structure_var.trace_add("write", _update_structure_lbl)
        r += 1

        status_row = tk.Frame(setup_card, bg=COLORS["bg_surface"])
        status_row.grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))
        tk.Label(status_row,
                 text="Increase Structure if variants look too similar to baseline.",
                 font=(FONT_FAMILY, 8, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        self._explorer_start_btn = tk.Button(
            status_row, text="Start", font=(FONT_FAMILY, 11, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=24, pady=6, cursor="hand2",
            command=self._explorer_start)
        self._explorer_start_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.explorer_status_var = tk.StringVar(value="Set a LoRA path and prompt, then click Start.")
        tk.Label(status_row, textvariable=self.explorer_status_var,
                 font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=COLORS["bg_surface"]).pack(side=tk.RIGHT)

        # Card 2: Baseline
        baseline_card = self._start_section_card(
            outer, "Current Baseline",
            "Your current best. Pick a favourite below to evolve it, or save it as a LoRA.",
        )
        baseline_inner = tk.Frame(baseline_card, bg=COLORS["bg_surface"])
        baseline_inner.pack(fill=tk.X)

        # Baseline image
        self._explorer_baseline_holder = tk.Frame(baseline_inner, bg="#000000",
                                                   width=512, height=512)
        self._explorer_baseline_holder.pack(side=tk.LEFT, padx=(0, 16), pady=4)
        self._explorer_baseline_holder.pack_propagate(False)
        self._explorer_baseline_label = tk.Label(self._explorer_baseline_holder,
                                                  text="(no baseline yet)",
                                                  fg=COLORS["text_muted"], bg="#000000")
        self._explorer_baseline_label.pack(expand=True)

        # Baseline info + buttons
        baseline_right = tk.Frame(baseline_inner, bg=COLORS["bg_surface"])
        baseline_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(baseline_right, bg=COLORS["bg_surface"])
        btn_row.pack(fill=tk.X, pady=(4, 8))
        btn_row.columnconfigure(0, weight=2)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=1)
        self._explorer_save_btn = ttk.Button(btn_row, text="Save Baseline as LoRA...",
                                              command=self._explorer_save, state="disabled")
        self._explorer_save_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        self._explorer_undo_btn = ttk.Button(btn_row, text="Undo",
                                              command=self._explorer_undo, state="disabled")
        self._explorer_undo_btn.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(btn_row, text="Restart",
                   command=self._explorer_restart).grid(row=0, column=2, sticky=tk.EW, padx=(4, 0))

        handoff_row = tk.Frame(baseline_right, bg=COLORS["bg_surface"])
        handoff_row.pack(fill=tk.X, pady=(0, 8))
        handoff_row.columnconfigure(0, weight=1)
        handoff_row.columnconfigure(1, weight=1)
        self._explorer_freeze_btn = tk.Button(
            handoff_row, text="Freeze tweaked blocks",
            font=(FONT_FAMILY, 10, "bold"),
            fg="#FFFFFF", bg=COLORS["accent"], activeforeground="#FFFFFF",
            activebackground=COLORS["accent_hover"],
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2", state="disabled",
            command=self._explorer_freeze_tweaked)
        self._explorer_freeze_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        self._explorer_refine_btn = tk.Button(
            handoff_row, text="Refine this baseline in Repair Studio \u2192",
            font=(FONT_FAMILY, 10, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2", state="disabled",
            command=self._explorer_refine_in_repair)
        self._explorer_refine_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))

        # Collapsed slider state display
        state_frame = tk.Frame(baseline_right, bg=COLORS["bg_deep"])
        state_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True, pady=(0, 4))
        self._explorer_state_text = tk.Text(state_frame, height=14, width=60,
                                             bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                                             font=(FONT_FAMILY, 8), wrap="word",
                                             state="disabled", relief="flat",
                                             highlightthickness=1,
                                             highlightbackground=COLORS["border"])
        state_scroll = ttk.Scrollbar(state_frame, orient="vertical",
                                      command=self._explorer_state_text.yview)
        self._explorer_state_text.configure(yscrollcommand=state_scroll.set)
        self._explorer_state_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        state_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Card 3: Gallery
        gallery_card = self._start_section_card(
            outer, "Variants",
            "4 random mutations of the current baseline. Click your favourite to evolve.",
        )

        gallery_btn_row = tk.Frame(gallery_card, bg=COLORS["bg_surface"])
        gallery_btn_row.pack(anchor=tk.W, pady=(0, 8))
        self._explorer_roll_btn = ttk.Button(gallery_btn_row, text="Re-roll",
                                              command=self._explorer_reroll, state="disabled")
        self._explorer_roll_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._explorer_progress_var = tk.StringVar(value="")
        tk.Label(gallery_btn_row, textvariable=self._explorer_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        self._explorer_gallery_frame = tk.Frame(gallery_card, bg=COLORS["bg_surface"])
        self._explorer_gallery_frame.pack(fill=tk.X)

        # 2x2 grid of clickable images
        self._explorer_gallery_labels = []
        for row_idx in range(2):
            for col_idx in range(2):
                idx = row_idx * 2 + col_idx
                holder = tk.Frame(self._explorer_gallery_frame, bg="#000000",
                                  highlightbackground=COLORS["border"], highlightthickness=2)
                holder.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky=tk.NSEW)
                lbl = tk.Label(holder, text=f"(variant {idx + 1})",
                               fg=COLORS["text_muted"], bg="#000000", cursor="hand2")
                lbl.pack(expand=True, fill=tk.BOTH)
                lbl.bind("<Button-1>", lambda e, i=idx: self._explorer_pick(i))
                lbl.bind("<Enter>", lambda e, h=holder: h.configure(highlightbackground=COLORS["accent"]))
                lbl.bind("<Leave>", lambda e, h=holder: h.configure(highlightbackground=COLORS["border"]))
                # Seed cycle button overlaid in top-right corner
                seed_btn = tk.Button(holder, text="\u21bb", font=(FONT_FAMILY, 10, "bold"),
                                     bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                     activebackground=COLORS["bg_surface"],
                                     activeforeground=COLORS["text_primary"],
                                     relief="flat", bd=0, padx=4, pady=2, cursor="hand2",
                                     command=lambda: self._explorer_cycle_seed())
                seed_btn.place(relx=1.0, x=-4, y=4, anchor="ne")
                self._explorer_gallery_labels.append(lbl)
        self._explorer_gallery_frame.columnconfigure(0, weight=1)
        self._explorer_gallery_frame.columnconfigure(1, weight=1)

        # Apply the persisted family (krea2 hides the DiT radio + ref Strength).
        self._apply_explorer_family_ui(str(self.explorer_family_var.get()) == "krea2")

        self._add_youtube_help_button(outer, "explorer")

    # ------------------------------------------------------------------
    # Explorer actions
    # ------------------------------------------------------------------

    def _explorer_family(self):
        fam = str(getattr(self, "explorer_family_var", None) and self.explorer_family_var.get())
        return fam if fam in ("klein", "krea2", "minimax") else "klein"

    def _explorer_is_krea2(self):
        return self._explorer_family() == "krea2"

    def _explorer_default_state(self):
        from fizgig.repair_studio.state import SliderState
        fam = self._explorer_family()
        return (SliderState.default_krea2() if fam == "krea2"
                else SliderState.default_h3() if fam == "minimax"
                else SliderState.default_klein9b())

    def _explorer_anchor_block(self):
        """The structural-composition anchor block — never locked/disabled, only inverted/pushed.
        Klein: double_0. Krea 2: block_0. MiniMax H3: h3blk_0 (each family's first block)."""
        fam = self._explorer_family()
        return {"krea2": "block_0", "minimax": "h3blk_0"}.get(fam, "double_0")

    def _on_explorer_family_changed(self):
        fam = self._explorer_family()
        self.last_used["explorer_family"] = fam
        self._save_last_used_paths()
        # Switching family is a hard reset — the engine + loaded LoRA belong to the old family.
        if self._explorer_engine is not None or self._explorer_baseline_state is not None:
            self._explorer_full_reset()
        self._apply_explorer_family_ui(fam != "klein")

    def _apply_explorer_family_ui(self, is_krea2):
        """Krea 2 / MiniMax H3: hide the Distilled/Base DiT radio (Krea 2 is always Turbo; H3
        auto-plans) and the ref Strength control (no reference-latent strength). Klein
        restores both. (`is_krea2` is historical naming: True = any non-Klein family.)"""
        dit_label = getattr(self, "_explorer_dit_label", None)
        dit_frame = getattr(self, "_explorer_dit_frame", None)
        strength_lbl = getattr(self, "_explorer_ref_strength_label", None)
        strength_entry = getattr(self, "_explorer_ref_strength_entry", None)
        if is_krea2:
            if dit_label is not None:
                dit_label.grid_remove()
            if dit_frame is not None:
                dit_frame.grid_remove()
            for w in (strength_lbl, strength_entry):
                if w is not None:
                    w.pack_forget()
        else:
            if dit_label is not None:
                dit_label.grid()
            if dit_frame is not None:
                dit_frame.grid()
            for w in (strength_lbl, strength_entry):
                if w is not None:
                    w.pack(side=tk.LEFT, padx=(0, 2) if w is strength_lbl else 0)

    def _explorer_ensure_engine(self):
        """Lazy-load engine + pipeline for the Explorer. Returns True on success."""
        if self._explorer_engine is not None and self._explorer_engine.pipeline is not None and self._explorer_engine.pipeline.is_loaded:
            return True

        if self._explorer_is_krea2():
            return self._explorer_ensure_engine_krea2()
        if self._explorer_family() == "minimax":
            return self._explorer_ensure_engine_h3()

        dit_choice = self.explorer_dit_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", f"{name} not found:\n{path}\n\nCheck Preferences tab.")
                return False

        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
        from fizgig.repair_studio.engine import RepairEngine

        if self._explorer_engine is None:
            self._explorer_engine = RepairEngine()
        self._explorer_engine._turbo_enabled = True  # always use Turbo for Explorer

        dit_basename = os.path.basename(dit_path).lower()
        model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
        is_fp8_model = "fp8" in dit_basename
        try:
            self.explorer_status_var.set(f"Loading models ({model_version})...")
            self.master.update_idletasks()
            self._explorer_engine.ensure_pipeline(
                dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
                model_version=model_version, device="cuda",
                fp8_scaled=False if is_fp8_model else True,
                blocks_to_swap=self._get_inference_blocks_to_swap(),
                int8=self._get_inference_int8(),
            )
            self.explorer_status_var.set("Models loaded.")
            return True
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load models:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading models.")
            return False

    def _explorer_ensure_engine_krea2(self):
        """Lazy-load the Krea 2 Repair engine for the Explorer (always fp8 Turbo, 8-step)."""
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
        if self._explorer_engine is None or not isinstance(self._explorer_engine, Krea2RepairEngine):
            self._explorer_engine = Krea2RepairEngine()
        try:
            self.explorer_status_var.set("Loading Krea 2 models (Turbo)...")
            self.master.update_idletasks()
            self._explorer_engine.ensure_pipeline(
                turbo_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
                device="cuda", model_kind="turbo",
                blocks_to_swap=self._auto_krea2_inference_blocks_swap(),
                int8=self._get_inference_int8())
            self.explorer_status_var.set("Models loaded.")
            return True
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load Krea 2 models:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading models.")
            return False

    def _explorer_ensure_engine_h3(self):
        """Lazy-load the MiniMax H3 engine for the Explorer — same auto-planned base + Turbo
        LoRA + prompt disk cache as the Repair Studio (see _ensure_repair_engine_h3)."""
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
        if self._explorer_engine is None or not isinstance(self._explorer_engine, H3RepairEngine):
            self._explorer_engine = H3RepairEngine()
        try:
            self.explorer_status_var.set("Loading MiniMax H3 (the 33B base takes a minute)…")
            self.master.update_idletasks()
            self._explorer_engine.ensure_pipeline(
                dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
                device="cuda", turbo_lora_path=turbo_path,
                turbo_lora_strength=0.75, te_cache_dir=te_cache)
            self.explorer_status_var.set("Models loaded.")
            return True
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load MiniMax H3 models:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading models.")
            return False

    def _explorer_load_lora(self):
        # Never tear down an engine a worker thread is mid-forward through (hard-hangs the app).
        if getattr(self, "_explorer_generating", False):
            messagebox.showinfo("Busy", "A preview is still rendering — wait for it to finish.")
            return
        path = self.explorer_lora_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid LoRA file first.")
            return
        # Auto-follow the file's family (issue #62 nice-to-have): detected from the header
        # alone, microseconds, so do it BEFORE _explorer_ensure_engine() commits to loading
        # the wrong family's DiT/VAE/TE. Only a genuinely unrecognized file falls through to
        # the generic error below, same as before.
        from fizgig.networks.lora import lora_family_from_file, FAMILY_DISPLAY_NAMES
        detected = lora_family_from_file(path)
        from fizgig.networks.lora import INFERENCE_FAMILIES
        if detected is not None and detected not in INFERENCE_FAMILIES:
            messagebox.showerror(
                "Unsupported family",
                f"{os.path.basename(path)} was trained for {FAMILY_DISPLAY_NAMES.get(detected, detected)}, "
                f"but the Explorer doesn't support {FAMILY_DISPLAY_NAMES.get(detected, detected)} LoRAs yet.")
            return
        selected = self._explorer_family()
        if detected is not None and detected != selected:
            self.explorer_family_var.set(detected)
            self._on_explorer_family_changed()
            self.explorer_status_var.set(
                f"Switched family selector to {FAMILY_DISPLAY_NAMES.get(detected, detected)} "
                f"to match {os.path.basename(path)}.")
        if not self._explorer_ensure_engine():
            return
        try:
            self.explorer_status_var.set("Loading LoRA...")
            self.master.update_idletasks()
            # Reset if something was already loaded
            if self._explorer_engine.primary_network is not None:
                self._explorer_engine.reset()
                self._explorer_engine = None
                if not self._explorer_ensure_engine():
                    return
            self._explorer_engine.load_primary(path)
            n_active = len(self._explorer_engine.primary_block_ids)
            # LyCORIS loads and saves natively — nothing to announce on open; the save
            # dialog states the format.
            # Initialize baseline state with user-specified LoRA strength
            self._explorer_baseline_state = self._explorer_default_state()
            self.explorer_status_var.set(
                f"Loaded: {os.path.basename(path)} "
                f"({n_active}/{len(self._explorer_baseline_state.blocks)} blocks). "
                f"Click Re-roll to start exploring.")
            try:
                base_strength = float(self.explorer_strength_var.get())
            except ValueError:
                base_strength = 1.0
            for bid, bs in self._explorer_baseline_state.blocks.items():
                bs.primary_strength = base_strength
            self._explorer_baseline_state.prompt = self.explorer_prompt_var.get()
            self._explorer_baseline_state.seed = int(self.explorer_seed_var.get() or 42)
            res = int(self.explorer_res_var.get() or 512)
            self._explorer_baseline_state.preview_width = res
            self._explorer_baseline_state.preview_height = res
            self._explorer_sync_ref_into(self._explorer_baseline_state)
            self._explorer_history.clear()
            self._explorer_locked_blocks.clear()
            self._explorer_last_pick_blocks.clear()

            self._explorer_baseline_image = None
            self._explorer_undo_btn.configure(state="disabled")
            self._explorer_save_btn.configure(state="disabled")
            self._explorer_refine_btn.configure(state="disabled")
            self._explorer_freeze_btn.configure(state="disabled")
            self._explorer_roll_btn.configure(state="normal")
            # Generate initial baseline image
            self._explorer_generate_baseline_and_roll()
        except Exception as ex:
            from fizgig.networks.lora import UnsupportedLoRAFormat
            if isinstance(ex, UnsupportedLoRAFormat):
                messagebox.showerror("Unsupported LoRA format", str(ex))
            else:
                import traceback
                messagebox.showerror("Error", f"Failed to load LoRA:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading LoRA.")

    def _explorer_generate_baseline_and_roll(self):
        """Generate baseline image then 4 variants in a background thread."""
        if self._explorer_generating or self._explorer_engine is None:
            return
        # Parse the free-text fields BEFORE setting the busy flag: a ValueError from a bad
        # Seed/Resolution used to fire inside the Tk callback with the flag already set —
        # never cleared — and the flag disables every other notebook tab, locking the whole
        # app to this tab until restart.
        try:
            _seed = int(self.explorer_seed_var.get() or 42)
            _res = int(self.explorer_res_var.get() or 512)
        except ValueError:
            messagebox.showerror(
                "Invalid value",
                f"Seed and Resolution must be whole numbers.\n\n"
                f"Seed: {self.explorer_seed_var.get()!r}  "
                f"Resolution: {self.explorer_res_var.get()!r}")
            return
        self._explorer_generating = True
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_progress_var.set("Generating baseline...")
        self.master.update_idletasks()

        # Sync prompt/seed/res into baseline state
        state = self._explorer_baseline_state
        state.prompt = self.explorer_prompt_var.get()
        state.seed = _seed
        res = _res
        state.preview_width = res
        state.preview_height = res
        self._explorer_sync_ref_into(state)

        # Generate mutations (exclude locked blocks)
        active = self._explorer_engine.primary_block_ids - self._explorer_locked_blocks
        # Composition anchor (double_0 / block_0) as structural anchor — only if not explicitly frozen
        anchor = self._explorer_anchor_block()
        if anchor in self._explorer_engine.primary_block_ids and anchor not in self._explorer_locked_blocks:
            active.add(anchor)
        intensity = self.explorer_intensity_var.get()
        structure = self.explorer_structure_var.get()
        n_muts = int(self.explorer_mutations_var.get() or 5)
        # Variant 1 & 2: structural (composition-anchor block)
        # Variant 3: pure random
        # Variant 4: random but avoids last pick's blocks (protects recent changes)
        variant_states = []
        for vi in range(4):
            vs_structure = structure if vi < 2 else 0.0
            if vi == 3 and self._explorer_last_pick_blocks:
                # Variant 4: exclude last pick's changed blocks
                protected_active = active - self._explorer_last_pick_blocks
                if len(protected_active) < 2:
                    protected_active = active  # fallback if too few left
                vs = state.mutate(protected_active, num_mutations=n_muts,
                                  intensity=intensity, structure=vs_structure, anchor=anchor)
            else:
                vs = state.mutate(active, num_mutations=n_muts, intensity=intensity,
                                  structure=vs_structure, anchor=anchor)
            variant_states.append(vs)

        import threading
        thread = threading.Thread(
            target=self._explorer_worker,
            args=(state, variant_states),
            daemon=True,
        )
        thread.start()

    def _explorer_worker(self, baseline_state, variant_states):
        """Background: generate baseline + 4 variants."""
        try:
            engine = self._explorer_engine
            if engine is None:
                return

            # Generate baseline (full forward, populates activation cache)
            engine._changed_blocks = set(baseline_state.blocks.keys())
            baseline_img = engine.generate_preview(baseline_state)
            # Show the baseline the moment it exists — the image the user actually loaded
            # must not wait behind four variant renders. Picking stays disabled until
            # _explorer_on_results flips _explorer_generating, so early display is safe.
            self.master.after(0, lambda: (
                self._explorer_show_baseline(baseline_img),
                self._explorer_update_state_text(baseline_state),
                self._explorer_progress_var.set("Baseline ready — generating variant 1/4...")))

            # Generate 4 variants — each runs a full forward (invalidate activation
            # cache between variants so they don't contaminate each other).
            # The prompt cache is still shared, saving ~300-500ms per variant.
            # Each variant appears in the gallery as soon as it renders.
            variant_images = []
            for i, vs in enumerate(variant_states):
                self.master.after(0, lambda i=i: self._explorer_progress_var.set(
                    f"Generating variant {i + 1}/4..."))
                engine._invalidate_activation_cache()
                engine._changed_blocks = set(vs.blocks.keys())
                img = engine.generate_preview(vs)
                variant_images.append(img)
                self.master.after(0, lambda i=i, im=img: self._explorer_show_variant(i, im))

            self.master.after(0, lambda: self._explorer_on_results(
                baseline_state, baseline_img, variant_states, variant_images))
        except Exception:
            import traceback
            err = traceback.format_exc()
            self.master.after(0, lambda: self._explorer_on_error(err))

    def _explorer_on_results(self, baseline_state, baseline_img, variant_states, variant_images):
        """Main-thread callback: update UI with results."""
        self._explorer_baseline_state = baseline_state
        self._explorer_baseline_image = baseline_img
        self._explorer_variant_states = variant_states
        self._explorer_variant_images = variant_images
        self._explorer_generating = False

        # Update baseline display
        self._explorer_show_baseline(baseline_img)
        self._explorer_update_state_text(baseline_state)

        # Update gallery
        for i, img in enumerate(variant_images):
            self._explorer_show_variant(i, img)

        self._explorer_save_btn.configure(state="normal")
        self._explorer_refine_btn.configure(state="normal")
        self._explorer_freeze_btn.configure(state="normal")
        # Update freeze button appearance based on locked state
        if self._explorer_locked_blocks:
            self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"])
        else:
            self._explorer_freeze_btn.configure(bg=COLORS["accent"])
        self._explorer_progress_var.set("")

        # Check if all blocks are frozen
        active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
        unlocked = active - self._explorer_locked_blocks
        if not unlocked:
            self._explorer_roll_btn.configure(state="disabled")
            self.explorer_status_var.set(
                f"All {len(active)} blocks frozen! Save your LoRA, or click Freeze to unlock.")
        else:
            self._explorer_roll_btn.configure(state="normal")
            locked_msg = f" ({len(self._explorer_locked_blocks)} frozen)" if self._explorer_locked_blocks else ""
            self.explorer_status_var.set(f"Pick a favourite or re-roll.{locked_msg}")

    def _explorer_on_error(self, err):
        self._explorer_generating = False
        self._explorer_roll_btn.configure(state="normal")
        self._explorer_progress_var.set("")
        self.explorer_status_var.set("Error — see console.")
        print(err)

    def _explorer_show_baseline(self, pil_img):
        """Display a PIL image in the baseline holder."""
        from PIL import ImageTk
        holder = self._explorer_baseline_holder
        w, h = holder.winfo_width(), holder.winfo_height()
        if w < 10:
            w, h = 512, 512
        resized = pil_img.copy()
        resized.thumbnail((w, h))
        tk_img = ImageTk.PhotoImage(resized)
        self._explorer_thumbnails["baseline"] = tk_img
        self._explorer_baseline_label.configure(image=tk_img, text="")

    def _explorer_show_variant(self, idx, pil_img):
        """Display a PIL image in gallery slot idx (0-3)."""
        from PIL import ImageTk
        lbl = self._explorer_gallery_labels[idx]
        parent = lbl.master
        w = parent.winfo_width()
        if w < 10:
            w = 256
        resized = pil_img.copy()
        resized.thumbnail((w, w))
        tk_img = ImageTk.PhotoImage(resized)
        self._explorer_thumbnails[f"variant_{idx}"] = tk_img
        lbl.configure(image=tk_img, text="")

    @staticmethod
    def _explorer_block_sort_key(b):
        """Stable ordering for block ids of either family. Klein ids are `<prefix>_<n>`
        (double_0, single_23); Krea 2 ids can be `<prefix>_<prefix>_<n>` (txt_lw_0, txt_rf_1)
        or `block_0`, plus the odd non-numeric `io`. Sort by the textual prefix then the trailing
        integer (0 when there isn't one) — naive `int(b.split('_')[1])` crashes on 'txt_lw_0'."""
        parts = str(b).split("_")
        try:
            num = int(parts[-1])
            prefix = "_".join(parts[:-1])
        except ValueError:
            num, prefix = 0, str(b)
        return (prefix, num)

    def _explorer_update_state_text(self, state):
        """Show the baseline's slider state as read-only text, with lock indicators."""
        lines = []
        for bid in sorted(state.blocks.keys(), key=self._explorer_block_sort_key):
            bs = state.blocks[bid]
            if not bs.primary_enabled or bs.primary_strength != 1.0:
                en = "ON" if bs.primary_enabled else "OFF"
                lock = " [LOCKED]" if bid in self._explorer_locked_blocks else ""
                lines.append(f"{bid}: {en} @ {bs.primary_strength:+.2f}{lock}")
        if not lines:
            lines = ["All blocks at default (1.0)"]
        self._explorer_state_text.configure(state="normal")
        self._explorer_state_text.delete("1.0", tk.END)
        self._explorer_state_text.insert("1.0", "\n".join(lines))
        self._explorer_state_text.configure(state="disabled")

    def _explorer_pick(self, idx):
        """User picked variant idx as the new baseline."""
        if self._explorer_generating or idx >= len(self._explorer_variant_images):
            return
        # Push current baseline to undo stack
        if self._explorer_baseline_state is not None and self._explorer_baseline_image is not None:
            self._explorer_history.append(
                (self._explorer_baseline_state.copy(), self._explorer_baseline_image,
                 set(self._explorer_locked_blocks)))
            self._explorer_undo_btn.configure(state="normal")

        picked_state = self._explorer_variant_states[idx]

        # Track which blocks changed in this pick (for variant 4 protection)
        if self._explorer_baseline_state is not None:
            self._explorer_last_pick_blocks = set(picked_state.diff_blocks(self._explorer_baseline_state))
        else:
            self._explorer_last_pick_blocks = set()

        # New baseline = the picked variant
        self._explorer_baseline_state = picked_state
        self._explorer_baseline_image = self._explorer_variant_images[idx]

        locked_msg = f" ({len(self._explorer_locked_blocks)} blocks locked)" if self._explorer_locked_blocks else ""
        self._explorer_show_baseline(self._explorer_baseline_image)
        self._explorer_update_state_text(self._explorer_baseline_state)
        self.explorer_status_var.set(
            f"Variant {idx + 1} selected as new baseline{locked_msg}. Generating new mutations...")

        # Roll new variants from the new baseline
        self._explorer_generate_baseline_and_roll()

    def _explorer_cycle_seed(self):
        """New random seed, regenerate all variants + baseline with current slider states."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        import random
        new_seed = random.randint(1, 99999)
        self.explorer_seed_var.set(str(new_seed))
        # Update baseline and all variant states to the new seed
        self._explorer_baseline_state.seed = new_seed
        for vs in self._explorer_variant_states:
            vs.seed = new_seed
        # Regenerate with same slider states but new seed
        self._explorer_generating = True
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_progress_var.set("Cycling seed...")
        self.master.update_idletasks()

        import threading
        thread = threading.Thread(
            target=self._explorer_seed_cycle_worker,
            args=(self._explorer_baseline_state, list(self._explorer_variant_states)),
            daemon=True,
        )
        thread.start()

    def _explorer_seed_cycle_worker(self, baseline_state, variant_states):
        """Background: regenerate baseline + existing variants at a new seed."""
        try:
            engine = self._explorer_engine
            if engine is None:
                return
            # Generate baseline at new seed
            engine._invalidate_activation_cache()
            engine._changed_blocks = set(baseline_state.blocks.keys())
            baseline_img = engine.generate_preview(baseline_state)
            # Generate each variant at new seed (same slider states)
            variant_images = []
            for i, vs in enumerate(variant_states):
                self.master.after(0, lambda i=i: self._explorer_progress_var.set(
                    f"Seed cycling variant {i + 1}/4..."))
                engine._invalidate_activation_cache()
                engine._changed_blocks = set(vs.blocks.keys())
                img = engine.generate_preview(vs)
                variant_images.append(img)
            self.master.after(0, lambda: self._explorer_on_results(
                baseline_state, baseline_img, variant_states, variant_images))
        except Exception:
            import traceback
            self.master.after(0, lambda: self._explorer_on_error(traceback.format_exc()))

    def _explorer_randomize_seed(self):
        """Randomize seed and regenerate (same as Apply but for seed)."""
        import random
        self.explorer_seed_var.set(str(random.randint(1, 99999)))
        if self._explorer_baseline_state is not None and not self._explorer_generating:
            self._explorer_generate_baseline_and_roll()

    def _explorer_sync_ref_into(self, state):
        """Copy the Explorer reference-image widgets into a SliderState."""
        state.ref_image_path = self.explorer_ref_path_var.get().strip()
        try:
            state.ref_megapixels = float(self.explorer_ref_mp_var.get())
        except (ValueError, AttributeError):
            state.ref_megapixels = 1.0
        try:
            state.ref_strength = float(self.explorer_ref_strength_var.get())
        except (ValueError, AttributeError):
            state.ref_strength = 1.0

    def _browse_explorer_ref(self):
        """Pick a reference image to edit-condition the Explorer previews."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.explorer_ref_path_var.set(path)
            if self._explorer_baseline_state is not None and not self._explorer_generating:
                self._explorer_generate_baseline_and_roll()

    def _clear_explorer_ref(self):
        """Remove the Explorer reference image."""
        self.explorer_ref_path_var.set("")
        if self._explorer_baseline_state is not None and not self._explorer_generating:
            self._explorer_generate_baseline_and_roll()

    def _explorer_start(self):
        """Start button: load LoRA if not loaded, or regenerate with current settings."""
        if self._explorer_generating:
            return
        if self._explorer_engine is None or self._explorer_engine.primary_network is None:
            # Not loaded yet — load the LoRA
            self._explorer_load_lora()
        else:
            # Already loaded — invalidate prompt cache and regenerate
            self._explorer_apply_prompt()

    def _explorer_apply_prompt(self):
        """Apply a new prompt — invalidates prompt cache and regenerates."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        if self._explorer_engine is not None:
            self._explorer_engine._prompt_cache_key = None
            self._explorer_engine._prompt_cache = None
        self._explorer_generate_baseline_and_roll()

    def _explorer_reroll(self):
        """Re-roll: generate 4 new mutations from the same baseline."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        self._explorer_generate_baseline_and_roll()

    def _explorer_undo(self):
        """Pop the history stack and restore the previous baseline + locked blocks."""
        if not self._explorer_history or self._explorer_generating:
            return
        prev_state, prev_img, prev_locked = self._explorer_history.pop()
        self._explorer_baseline_state = prev_state
        self._explorer_baseline_image = prev_img
        self._explorer_locked_blocks = prev_locked
        self._explorer_show_baseline(prev_img)
        self._explorer_update_state_text(prev_state)
        self._explorer_roll_btn.configure(state="normal")
        self._explorer_freeze_btn.configure(
            bg=COLORS["accent_hover"] if self._explorer_locked_blocks else COLORS["accent"])
        if not self._explorer_history:
            self._explorer_undo_btn.configure(state="disabled")
        locked_msg = f" ({len(self._explorer_locked_blocks)} frozen)" if self._explorer_locked_blocks else ""
        self.explorer_status_var.set(f"Undone{locked_msg}. Re-rolling...")
        self._explorer_generate_baseline_and_roll()

    def _explorer_restart(self):
        """Unlock all blocks and restart exploration — ask whether from defaults or current baseline."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        choice = messagebox.askyesnocancel(
            "Restart Exploration",
            "Unlock all blocks and restart.\n\n"
            "Yes = restart from default values\n"
            "No = restart from current baseline (keep slider positions)\n"
            "Cancel = don't restart",
        )
        if choice is None:
            return  # Cancel

        # Push current state to undo stack
        self._explorer_history.append(
            (self._explorer_baseline_state.copy(), self._explorer_baseline_image,
             set(self._explorer_locked_blocks)))
        self._explorer_undo_btn.configure(state="normal")
        self._explorer_locked_blocks.clear()
        self._explorer_walk_index = 0

        if choice:
            # Yes = reset to default values
            self._explorer_baseline_state = self._explorer_default_state()
            try:
                base_strength = float(self.explorer_strength_var.get())
            except ValueError:
                base_strength = 1.0
            for bid, bs in self._explorer_baseline_state.blocks.items():
                bs.primary_strength = base_strength
            self._explorer_baseline_state.prompt = self.explorer_prompt_var.get()
            self._explorer_baseline_state.seed = int(self.explorer_seed_var.get() or 42)
            res = int(self.explorer_res_var.get() or 512)
            self._explorer_baseline_state.preview_width = res
            self._explorer_baseline_state.preview_height = res
            self.explorer_status_var.set("Restarted from defaults — all blocks unlocked.")
        else:
            # No = keep current baseline, just unlock
            self.explorer_status_var.set("All blocks unlocked — continuing from current baseline.")

        self._explorer_roll_btn.configure(state="normal")
        self._explorer_generate_baseline_and_roll()

    def _explorer_full_reset(self):
        """Unload LoRA and pipeline, return to initial state."""
        if self._explorer_generating:
            return
        if self._explorer_engine is not None:
            try:
                self._explorer_engine.reset()
            except Exception:
                pass
            self._explorer_engine = None
        self._explorer_baseline_state = None
        self._explorer_baseline_image = None
        self._explorer_history.clear()
        self._explorer_locked_blocks.clear()
        self._explorer_walk_index = 0
        self._explorer_variant_states.clear()
        self._explorer_variant_images.clear()
        self._explorer_thumbnails.clear()
        self._explorer_baseline_label.configure(image="", text="(no baseline yet)")
        for lbl in self._explorer_gallery_labels:
            lbl.configure(image="", text="(variant)")
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_save_btn.configure(state="disabled")
        self._explorer_undo_btn.configure(state="disabled")
        self._explorer_progress_var.set("")
        self.explorer_status_var.set("Load a LoRA to begin exploring.")

    def _explorer_freeze_tweaked(self):
        """Freeze all blocks that differ from default — they won't be mutated on re-roll."""
        if self._explorer_baseline_state is None or self._explorer_generating:
            return

        # Find blocks that differ from default (strength != starting strength)
        try:
            base_strength = float(self.explorer_strength_var.get())
        except ValueError:
            base_strength = 1.0

        tweaked = set()
        for bid, bs in self._explorer_baseline_state.blocks.items():
            if not bs.primary_enabled or abs(bs.primary_strength - base_strength) > 0.01:
                tweaked.add(bid)
        # When explicitly freezing, include double_0 — stops the structural anchor behaviour

        if self._explorer_locked_blocks:
            # Already have frozen blocks — ask what to do
            active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
            unlocked = active - self._explorer_locked_blocks - {self._explorer_anchor_block()}

            if not unlocked:
                # All locked — offer unlock all or undo last
                choice = messagebox.askyesnocancel(
                    "All blocks frozen",
                    f"All {len(active)} blocks are already frozen.\n\n"
                    "Yes = Unlock all blocks\n"
                    "No = Undo last freeze (restore previous unlocked set)\n"
                    "Cancel = Keep as is",
                )
                if choice is True:
                    self._explorer_locked_blocks.clear()
                    self._explorer_freeze_btn.configure(bg=COLORS["accent"])
                    self._explorer_roll_btn.configure(state="normal")
                    self._explorer_update_state_text(self._explorer_baseline_state)
                    self.explorer_status_var.set("All blocks unlocked. Re-rolling...")
                    self._explorer_generate_baseline_and_roll()
                elif choice is False and hasattr(self, "_explorer_prev_locked"):
                    self._explorer_locked_blocks = self._explorer_prev_locked
                    if hasattr(self, "_explorer_prev_baseline") and self._explorer_prev_baseline is not None:
                        self._explorer_baseline_state = self._explorer_prev_baseline
                    self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"] if self._explorer_locked_blocks else COLORS["accent"])
                    self._explorer_roll_btn.configure(state="normal")
                    self._explorer_show_baseline(self._explorer_baseline_image)
                    self._explorer_update_state_text(self._explorer_baseline_state)
                    n = len(self._explorer_locked_blocks)
                    self.explorer_status_var.set(f"Last freeze undone. {n} blocks frozen. Re-rolling...")
                    self._explorer_generate_baseline_and_roll()
                return
            else:
                # Some locked, some not — ask to lock additions or unlock all
                choice = messagebox.askyesnocancel(
                    "Freeze tweaked blocks",
                    f"Currently {len(self._explorer_locked_blocks)} blocks frozen.\n"
                    f"{len(tweaked - self._explorer_locked_blocks)} new tweaked blocks to add.\n\n"
                    "Yes = Lock the additions too\n"
                    "No = Unlock all\n"
                    "Cancel = Keep as is",
                )
                if choice is True:
                    self._explorer_prev_locked = set(self._explorer_locked_blocks)
                    self._explorer_prev_baseline = self._explorer_baseline_state.copy()
                    self._explorer_locked_blocks |= tweaked
                elif choice is False:
                    self._explorer_prev_locked = set(self._explorer_locked_blocks)
                    self._explorer_prev_baseline = self._explorer_baseline_state.copy()
                    self._explorer_locked_blocks.clear()
                    self._explorer_freeze_btn.configure(bg=COLORS["accent"])
                else:
                    return
        else:
            # No existing frozen blocks — freeze the tweaked ones
            if not tweaked:
                messagebox.showinfo("Nothing to freeze",
                    "No blocks have been changed from their starting values yet.")
                return
            self._explorer_prev_locked = set()
            self._explorer_prev_baseline = self._explorer_baseline_state.copy()
            self._explorer_locked_blocks = tweaked

        # Update UI
        if self._explorer_locked_blocks:
            self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"])
        else:
            self._explorer_freeze_btn.configure(bg=COLORS["accent"])

        # Check if all blocks now locked (Freeze can lock double_0 too)
        active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
        unlocked = active - self._explorer_locked_blocks
        if not unlocked:
            self._explorer_roll_btn.configure(state="disabled")
            self.explorer_status_var.set(
                f"All {len(active)} blocks frozen! Save your LoRA, Restart, or click Freeze to unlock.")
        else:
            n = len(self._explorer_locked_blocks)
            self.explorer_status_var.set(f"{n} blocks frozen. {len(unlocked)} still explorable.")

        self._explorer_update_state_text(self._explorer_baseline_state)

        # Re-roll variants to respect the new freeze state
        if unlocked:
            self._explorer_generate_baseline_and_roll()

    def _explorer_refine_in_repair(self):
        """Send the current Explorer baseline to the Repair Studio for manual editing."""
        if self._explorer_engine is None or self._explorer_baseline_state is None:
            return
        # Mid-generation handoff hard-hangs: the unload below no-ops (its own worker guard),
        # then Repair Studio loads a SECOND pipeline on the GUI thread while the Explorer
        # worker is mid-CUDA on the first — two pipelines, two threads, one GPU, and the GUI
        # thread blocks inside a CUDA call. Same guard every other Explorer action uses.
        if getattr(self, "_explorer_generating", False):
            self.explorer_status_var.set(
                "Still generating variants — wait for them to finish, then Refine.")
            return
        lora_path = self._explorer_engine.primary_path
        if not lora_path:
            return

        # LyCORIS files now save natively (lossless) — the old SVD-warning gate is gone.

        baseline = self._explorer_baseline_state

        # Handoff inherits the Explorer's family — switch Repair Studio to match (rebuilds the
        # slider panel for the right block layout so the value-push loop below finds the block ids).
        target_family = "krea2" if self._explorer_is_krea2() else "klein"
        if hasattr(self, "repair_family_var") and self.repair_family_var.get() != target_family:
            self.repair_family_var.set(target_family)
            self._on_repair_family_changed()

        # Set the LoRA path in Repair Studio
        self.repair_primary_var.set(lora_path)

        # Unload Explorer engine to free VRAM
        self._unload_explorer_models()

        # Switch to Repair Studio tab
        self.notebook.select(self.repair_studio_tab)

        # Load the LoRA in Repair Studio — through the async loader (the pipeline load is a
        # 10-20 GB affair; it used to run right here on the Tk thread and freeze the GUI).
        def _push_baseline():
            # Push the Explorer baseline slider values into Repair Studio
            self._repair_master_mutating = True
            try:
                for bid, bs in baseline.blocks.items():
                    if bid in self.repair_block_vars:
                        self.repair_block_vars[bid]["primary_enabled"].set(bs.primary_enabled)
                        self.repair_block_vars[bid]["primary_strength"].set(bs.primary_strength)
            finally:
                self._repair_master_mutating = False

            # Set the prompt, seed, resolution, and reference image to match Explorer
            self.repair_prompt_var.set(baseline.prompt)
            self.repair_seed_var.set(str(baseline.seed))
            self.repair_res_var.set(str(baseline.preview_width))
            # Carry the reference image (path, MP, strength) across the handover,
            # reading the Explorer widgets (the authoritative source).
            if hasattr(self, "repair_ref_path_var"):
                self.repair_ref_path_var.set(self.explorer_ref_path_var.get().strip())
                self.repair_ref_mp_var.set(self.explorer_ref_mp_var.get())
                self.repair_ref_strength_var.set(self.explorer_ref_strength_var.get())

            n_active = len(self.repair_engine.primary_block_ids)
            self.repair_status_var.set(
                f"Loaded from Explorer: {os.path.basename(lora_path)} "
                f"({n_active}/{len(self.repair_state.blocks)} blocks). "
                f"Sliders set to Explorer baseline. Generating preview...")
            self._schedule_preview(force=True)

        self._load_repair_primary(on_done=_push_baseline)

    def _explorer_save(self):
        """Save the current baseline as a baked LoRA."""
        if self._explorer_engine is None or self._explorer_baseline_state is None:
            return
        primary_path = self._explorer_engine.primary_path
        if not primary_path:
            return
        stem = os.path.splitext(os.path.basename(primary_path))[0]
        default_name = f"{stem}_explored.safetensors"
        out = filedialog.asksaveasfilename(
            title="Save Explored LoRA",
            defaultextension=".safetensors",
            filetypes=[("SafeTensors", "*.safetensors")],
            initialfile=default_name,
        )
        if not out:
            return
        from fizgig.repair_studio.bake import save_repaired_lora
        from fizgig.networks.lora import UnsupportedLoRAFormat
        try:
            summary = save_repaired_lora(primary_path, self._explorer_baseline_state, out)
            _fmt_note = ("\n\nSaved natively in LyCORIS format — lossless, no conversion."
                         if summary.get('format_out') == 'lycoris' else "")
            messagebox.showinfo("Explored LoRA saved",
                                f"Saved: {out}\n\nKeys: {summary['keys_in']} -> {summary['keys_out']}"
                                + _fmt_note)
        except UnsupportedLoRAFormat as ex:
            messagebox.showerror("Unsupported LoRA format", str(ex))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _unload_explorer_models(self):
        """Unload Explorer pipeline when leaving the tab."""
        # Internal guard (all call sites): resetting under a live CUDA worker hard-hangs.
        if getattr(self, "_explorer_generating", False):
            return
        if self._explorer_engine is not None and self._explorer_engine.pipeline is not None:
            try:
                self._explorer_engine.reset()
            except Exception:
                pass
            self._explorer_engine = None
            self.explorer_status_var.set("Models unloaded (tab switch). Load a LoRA to resume.")