import tkinter as tk
from tkinter import ttk

from fizgig_gui.core.domain.architectures import _canon_arch, ARCHITECTURE_LIST
from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, FONT_MONO
from fizgig_gui.core.domain.minimax_math import MINIMAX_TRAIN_BASE_OPTIONS, minimax_train_base, MINIMAX_LIKENESS_BLOCKS, \
    MINIMAX_AUDIO_BLOCKS, MINIMAX_BASE_QUANT_OPTIONS, MINIMAX_BLOCK_OPTIONS
from fizgig_gui.core.ui_base.widgets import ToolTip, CollapsibleFrame


class TrainingUiMixin:
    def create_training_settings(self):
        """Create the Training tab (Start-tab styled)."""
        scrollable_frame, self.training_canvas = self.create_scrollable_frame(self.training_tab)

        # Outer bg_deep container — all sections pack into this so the banner
        # and collapsibles share the same horizontal alignment.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Store collapsible sections for later access
        self.collapsible_sections = {}

        self._add_tab_banner(
            outer,
            "Training",
            "Pick a preset or dial in a custom run. Sections below collapse to reduce clutter — "
            "click any header to toggle. Dataset-config knobs live in Other Options.",
        )

        # Model family selector. Restore the last choice; fall back to Klein if the
        # saved value isn't a known architecture (e.g. a removed/renamed entry).
        _saved_arch = "Flux 2 Klein Base 9B"
        try:
            _candidate = _canon_arch(self.last_used.get("architecture", _saved_arch))
            if _candidate in ARCHITECTURE_LIST:
                _saved_arch = _candidate
        except Exception:
            pass
        self.architecture_var = tk.StringVar(value=_saved_arch)
        # Seeded so _on_architecture_selected can tell a real family change from the user
        # re-picking the entry that's already selected (both fire <<ComboboxSelected>>).
        self._arch_last_selected = _saved_arch
        # Per-family settings memory, session-scoped: architecture -> the Training-tab
        # snapshot it had when you last switched away from it, and the preset label it was
        # showing. First visit to a family falls back to that family's default preset.
        self._arch_settings_memory = {}
        self._arch_preset_name_memory = {}

        # === Base Model card (only shown when more than one architecture is available) ===
        if len(ARCHITECTURE_LIST) > 1:
            model_card = self._start_section_card(
                outer, "Base Model",
                "Pick the model family to train. Krea 2 needs its model paths set on the "
                "Preferences tab before training.",
            )
            model_row = tk.Frame(model_card, bg=COLORS["bg_surface"])
            model_row.pack(anchor=tk.W)
            tk.Label(
                model_row, text="Model:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            arch_combo = ttk.Combobox(
                model_row, textvariable=self.architecture_var, state="readonly",
                width=28, values=ARCHITECTURE_LIST,
            )
            arch_combo.pack(side=tk.LEFT)
            arch_combo.bind("<<ComboboxSelected>>", self._on_architecture_selected)
            ToolTip(arch_combo, "Model family to train (Klein 9B, Krea 2 or MiniMax H3)")

            # Training Base (MiniMax only) — which H3 fine-tune the run trains against, right
            # where the family was just chosen. A dedicated var kept OUT of self.entries and
            # never collected, so presets can't flip it; last-train and the queue carry it
            # explicitly. Shown/hidden by _apply_training_arch_visibility alongside the note.
            self.minimax_train_base_var = tk.StringVar(
                value=MINIMAX_TRAIN_BASE_OPTIONS[
                    1 if minimax_train_base(self.settings.get("MINIMAX_TRAIN_BASE")) == "ref2va"
                    else 0])
            self._minimax_base_frame = tk.Frame(model_card, bg=COLORS["bg_surface"])
            tk.Label(
                self._minimax_base_frame, text="Training Base:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            self._minimax_base_combo = ttk.Combobox(
                self._minimax_base_frame, textvariable=self.minimax_train_base_var,
                values=list(MINIMAX_TRAIN_BASE_OPTIONS), state="readonly", width=36)
            self._minimax_base_combo.pack(side=tk.LEFT)
            self._minimax_base_hint = tk.Label(
                model_card,
                text="Pick the H3 model you deploy on. First/last frame (fl2va) is the standard "
                     "model most workflows run. Reference (ref2va) is the Reference-to-Video "
                     "fine-tune — choose it if your LoRA's home is the r2v workflow (needs 'DiT "
                     "(reference)' set in Preferences). Presets never change this; reference "
                     "distillation always trains on ref2va regardless.",
                font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                bg=COLORS["bg_surface"], wraplength=760, justify=tk.LEFT)
            self._minimax_base_frame.pack(anchor=tk.W, pady=(10, 0))
            self._minimax_base_hint.pack(anchor=tk.W, pady=(2, 0))
            if not self._is_minimax_arch():
                self._minimax_base_frame.pack_forget()
                self._minimax_base_hint.pack_forget()

            # Previews track likeness honestly but are not the place to compare quality — say
            # so where the family is chosen, along with the Pause/Resume route that makes
            # judging in ComfyUI practical on one GPU.
            self._minimax_sample_note = tk.Label(
                model_card,
                text=("⏱ Previews track LIKENESS, not quality. Judge quality in ComfyUI — Pause "
                      "frees the GPU, so you can check an epoch there and Resume.\n"
                      "Defaults are 768×768 56-frame clips with sound; Sample length has "
                      "stills and other lengths. 📖 Full write-ups in the README."),
                font=(FONT_FAMILY, 9), fg=COLORS["warning"], bg=COLORS["bg_surface"],
                wraplength=760, justify=tk.LEFT,
            )
            self._minimax_sample_note.pack(anchor=tk.W, pady=(10, 0))
            if not self._is_minimax_arch():
                self._minimax_sample_note.pack_forget()

        # === Presets card ===
        preset_card = self._start_section_card(
            outer, "Presets",
            "Save the current settings under a name, load a saved preset, or restore the exact "
            "configuration from your last training run.",
        )
        # Row 1: Save + Load Preset
        preset_row1 = tk.Frame(preset_card, bg=COLORS["bg_surface"])
        preset_row1.pack(anchor=tk.W, pady=(0, 8))
        save_preset_btn = ttk.Button(preset_row1, text="Save Preset", command=self.save_custom_preset)
        save_preset_btn.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(save_preset_btn, "Save current training parameters as a named preset")
        tk.Label(
            preset_row1, text="Load Preset:",
            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.custom_preset_var = tk.StringVar()
        self.custom_preset_combo = ttk.Combobox(preset_row1, textvariable=self.custom_preset_var, state="readonly", width=28)
        self.custom_preset_combo.pack(side=tk.LEFT)
        self.custom_preset_combo.bind("<<ComboboxSelected>>", self.load_custom_preset)
        ToolTip(self.custom_preset_combo, "Your saved training presets")
        # Bracketed nudge for the rank-16 recipe, shown only while the MiniMax Defaults preset
        # is selected: Fast is the default now, and this says when the bigger one earns its keep.
        self._preset_hint_label = tk.Label(
            preset_row1, text="(more suitable for larger datasets with longer trains)",
            font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
        )
        self.custom_preset_var.trace_add("write", lambda *_: self._update_preset_hint())

        # Row 2: Load Settings From Last Train
        load_last_btn = ttk.Button(preset_card, text="Load Settings From Last Train",
                                   command=self._load_last_train_settings, width=32)
        load_last_btn.pack(anchor=tk.W)
        ToolTip(load_last_btn, "Restore the exact settings used in your most recent training launch")

        # === Output Section (Expanded by default) ===
        output_section = CollapsibleFrame(outer,"Output", default_expanded=True)
        output_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["output"] = output_section

        output_content = output_section.get_content_frame()
        output_content.columnconfigure(1, weight=1)

        self._add_field_to_section(output_content, "LORA_OUTPUT_DIR", "Output Directory", "directory", 0)
        self._add_field_to_section(output_content, "LORA_NAME", "LoRA Name", "text", 1)

        # Save LoRA output directory when it changes
        self.entries["LORA_OUTPUT_DIR"].bind("<FocusOut>", lambda e: self._save_last_used_paths())
        self.entries["LORA_OUTPUT_DIR"].bind("<Return>", lambda e: self._save_last_used_paths())

        # === Training Parameters Section (Expanded by default) ===
        training_section = CollapsibleFrame(outer,"Training Parameters", default_expanded=True)
        training_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["training"] = training_section

        training_content = training_section.get_content_frame()
        training_content.columnconfigure(1, weight=1)

        self._add_field_to_section(training_content, "MODEL_TYPE", "Model Type", "dropdown", 0)
        self._add_field_to_section(training_content, "LEARNING_RATE", "Learning Rate", "float", 1)

        # --- Adaptive LR (bi-directional plateau tracker) — placed under Learning Rate so both bracket the starting LR
        self.adaptive_lr_var = tk.BooleanVar(value=False)
        adaptive_cb = ttk.Checkbutton(
            training_content, text="Adaptive LR (auto-adjust based on loss, gradient clipping & weight-norm growth)",
            variable=self.adaptive_lr_var, command=self._on_adaptive_lr_toggle,
        )
        adaptive_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(4, 0))
        self._adaptive_cb = adaptive_cb

        adaptive_frame = ttk.Frame(training_content)
        adaptive_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 2))
        self._adaptive_frame = adaptive_frame
        ttk.Label(adaptive_frame, text="Min LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MIN"] = ttk.Combobox(adaptive_frame, width=28, values=["1e-5", "5e-5", "1e-4", "2e-4 - rank 4/8 only", "3e-4 - low-rank only"], state="readonly")
        self.entries["ADAPTIVE_LR_MIN"].set("1e-5")
        self.entries["ADAPTIVE_LR_MIN"].pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(adaptive_frame, text="Max LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MAX"] = ttk.Combobox(adaptive_frame, width=8, values=["1e-4", "2e-4", "3e-4", "4e-4"], state="readonly")
        self.entries["ADAPTIVE_LR_MAX"].set("4e-4")
        self.entries["ADAPTIVE_LR_MAX"].pack(side=tk.LEFT, padx=(0, 12))
        self._adaptive_reset_btn = ttk.Button(adaptive_frame, text="Reset Defaults", command=self._reset_adaptive_lr_defaults)
        self._adaptive_reset_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._adaptive_desc_label = ttk.Label(training_content,
                  text="When on, the Learning Rate box is IGNORED (greyed out) — the run starts at the geometric "
                       "midpoint of Min/Max (e.g. 1e-4 & 4e-4 → 2e-4) and the watcher owns the LR from there: "
                       "probes UP on steady loss descent; reduces DOWN on loss plateau, heavy gradient clipping, "
                       "or runaway weight-norm growth (with a rollback to the previous epoch's weights on "
                       "stability events).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._adaptive_desc_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 6))
        self._on_adaptive_lr_toggle()  # sync initial enabled/disabled state

        # LoRA LR Ratio — hidden, always 1 (LoRA+ default). Widget exists for preset/save compat.
        self.entries["LORA_LR_RATIO"] = ttk.Entry(training_content, width=12)
        self.entries["LORA_LR_RATIO"].insert(0, str(self.settings["LORA_LR_RATIO"]))
        self._add_field_to_section(training_content, "NETWORK_DIM", "Network Dim (Rank)", "int", 5)
        self._add_field_to_section(training_content, "NETWORK_ALPHA", "Network Alpha", "float", 6)
        self._add_field_to_section(training_content, "MAX_TRAIN_EPOCHS", "Max Epochs", "int", 7)
        self._add_field_to_section(training_content, "SAVE_EVERY_N_EPOCHS", "Save Every N Epochs", "int", 8)
        self._add_field_to_section(training_content, "SEED", "Seed", "int", 9)

        # Network Type (Krea 2 only): standard LoRA or LoKR (Kronecker). Rows 18/19 sit between
        # the Target Megapixels hint (17) and the Krea 2 loss-watch block (20). LoKR replaces
        # rank/alpha with a single Factor dial, so the rows swap with the selection.
        _nt_label = tk.Label(training_content, text="Network Type:", font=(FONT_FAMILY, 10),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        _nt_label.grid(row=18, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels["NETWORK_TYPE"] = _nt_label
        # Widget + hint share a row frame so the hint hugs the control instead of being
        # pushed to the far column edge by the full-width rows above.
        self._network_type_rowf = tk.Frame(training_content, bg=COLORS["bg_surface"])
        self._network_type_rowf.grid(row=18, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        self.entries["NETWORK_TYPE"] = ttk.Combobox(
            self._network_type_rowf, values=["LoRA (standard)", "LoKR (Kronecker)"],
            state="readonly", width=18)
        self.entries["NETWORK_TYPE"].set(self.settings.get("NETWORK_TYPE", "LoRA (standard)"))
        self.entries["NETWORK_TYPE"].pack(side=tk.LEFT)
        self.entries["NETWORK_TYPE"].bind("<<ComboboxSelected>>",
                                          lambda e: self._on_network_type_changed())
        self._network_type_hint = tk.Label(
            self._network_type_rowf,
            text="LoKR: higher quality · LoRA: ~20% faster training",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT)
        self._network_type_hint.pack(side=tk.LEFT, padx=(10, 0))
        # rows entry is the FRAME (the gridded thing show_row/hide_row must toggle).
        self.rows["NETWORK_TYPE"] = {"row": 18, "label": _nt_label,
                                     "entry": self._network_type_rowf,
                                     "browse": None, "parent": training_content}

        _lf_label = tk.Label(training_content, text="LoKR Factor:", font=(FONT_FAMILY, 10),
                             fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        _lf_label.grid(row=19, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels["LOKR_FACTOR"] = _lf_label
        self._lokr_factor_rowf = tk.Frame(training_content, bg=COLORS["bg_surface"])
        self._lokr_factor_rowf.grid(row=19, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        self.entries["LOKR_FACTOR"] = ttk.Entry(self._lokr_factor_rowf, width=8)
        self.entries["LOKR_FACTOR"].insert(0, str(self.settings.get("LOKR_FACTOR", 8)))
        self.entries["LOKR_FACTOR"].pack(side=tk.LEFT)
        self._lokr_factor_hint = tk.Label(
            self._lokr_factor_rowf,
            text="8 is the sweet spot · 4 = stronger, bigger files · above 8: just use LoRA",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            justify=tk.LEFT)
        self._lokr_factor_hint.pack(side=tk.LEFT, padx=(10, 0))
        self.rows["LOKR_FACTOR"] = {"row": 19, "label": _lf_label,
                                    "entry": self._lokr_factor_rowf,
                                    "browse": None, "parent": training_content}

        self._build_minimax_structure_row(training_content)

        # Model Area to Train dropdown (blocks + timestep auto-fill)
        self._modelarea_label = ttk.Label(training_content, text="Model Area to Train:")
        self._modelarea_label.grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
        self.training_preset_var = tk.StringVar(value="Full Model")
        training_preset_combo = ttk.Combobox(
            training_content, textvariable=self.training_preset_var,
            values=["Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom"],
            state="readonly", width=20
        )
        training_preset_combo.grid(row=10, column=1, sticky=tk.W, padx=5, pady=2)
        training_preset_combo.bind("<<ComboboxSelected>>", self._on_training_preset_changed)
        self._modelarea_combo = training_preset_combo
        self._modelarea_desc_label = ttk.Label(training_content,
                  text="Identity = single 1-16  |  Style = style+comp blocks @ late ts (0-400)  |  Style+Composition = double 0-7 + single 0-1  |  Details = single 12-23",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"))
        self._modelarea_desc_label.grid(row=11, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Custom block picker panel (hidden unless preset == Custom)
        self._training_custom_frame = ttk.Frame(training_content)
        self._training_custom_frame.grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=15, pady=(4, 4))
        self.training_block_vars = {}  # block_name -> BooleanVar

        tc_header = ttk.Frame(self._training_custom_frame)
        tc_header.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        ttk.Label(tc_header, text="Select blocks to train:",
                  font=(FONT_FAMILY, 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tc_header, text="All", width=5,
                   command=lambda: self._set_all_training_blocks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="None", width=5,
                   command=lambda: self._set_all_training_blocks(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Identity", width=8,
                   command=lambda: self._set_category_training_blocks("identity")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Style+Comp", width=10,
                   command=lambda: self._set_category_training_blocks("style_composition")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Details", width=8,
                   command=lambda: self._set_category_training_blocks("details")).pack(side=tk.LEFT, padx=2)

        tc_double = ttk.Frame(self._training_custom_frame)
        tc_double.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_double, text="Double:", width=8, foreground="#5B9BD5").pack(side=tk.LEFT)
        for i in range(8):
            key = f"double_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_double, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single1 = ttk.Frame(self._training_custom_frame)
        tc_single1.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single1, text="Single:", width=8, foreground="#70AD47").pack(side=tk.LEFT)
        for i in range(12):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single1, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single2 = ttk.Frame(self._training_custom_frame)
        tc_single2.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single2, text="Single:", width=8, foreground="#ED7D31").pack(side=tk.LEFT)
        for i in range(12, 24):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single2, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        ttk.Label(self._training_custom_frame,
                  text="double + single 0-1 = style+composition  |  single 1-16 = identity (overlaps at 1 and 12-16)  |  single 12-23 = details  |  edit MIN/MAX_TIMESTEP on Advanced tab",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic")).pack(anchor=tk.W, pady=(4, 0))

        self._training_custom_frame.grid_remove()  # hidden until preset == Custom

        # Context LoRA (optional) — train new LoRA with an existing one frozen + active on the base
        self._contextlora_label = ttk.Label(training_content, text="Context LoRA:")
        self._contextlora_label.grid(row=13, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        ctx_frame = ttk.Frame(training_content)
        ctx_frame.grid(row=13, column=1, sticky=tk.W, padx=5, pady=(8, 2))
        self._contextlora_frame = ctx_frame
        self.entries["CONTEXT_LORA_PATH"] = ttk.Entry(ctx_frame, width=42)
        self.entries["CONTEXT_LORA_PATH"].pack(side=tk.LEFT)
        ttk.Button(ctx_frame, text="Browse",
                   command=lambda: self._browse_context_lora()).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(ctx_frame, text="Strength:").pack(side=tk.LEFT, padx=(4, 4))
        self.entries["CONTEXT_LORA_STRENGTH"] = ttk.Entry(ctx_frame, width=6)
        self.entries["CONTEXT_LORA_STRENGTH"].insert(0, "1.0")
        self.entries["CONTEXT_LORA_STRENGTH"].pack(side=tk.LEFT)
        self._contextlora_desc_label = ttk.Label(training_content,
                  text="Train this LoRA with an existing LoRA already active on the base model. "
                       "Pair with same context+strength at inference.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"))
        self._contextlora_desc_label.grid(row=14, column=0, columnspan=2, sticky=tk.W, padx=5)
        self._contextlora_warn_label = ttk.Label(training_content,
                  text="⚠ Context LoRAs usually look better in ComfyUI than in training samples — "
                       "don't worry if previews look rough, test the output LoRA in ComfyUI.",
                  foreground="#E67E22", font=(FONT_FAMILY, 9, "italic"))
        self._contextlora_warn_label.grid(row=15, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Target Megapixels (training resolution) — moved here from Other Options
        ttk.Label(training_content, text="Target Megapixels:").grid(row=16, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        mp_frame = ttk.Frame(training_content)
        mp_frame.grid(row=16, column=1, sticky=tk.W, padx=5, pady=(8, 2))
        self._mp_combo = ttk.Combobox(
            mp_frame, textvariable=self.dataset_megapixels_var,
            values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4", "3.0", "4.2"], width=8)
        self._mp_combo.pack(side=tk.LEFT, padx=(0, 10))
        # Shown (and the combo greyed) when the training folder is voice recordings only —
        # there are no pixels for this number to size. Managed by _refresh_audio_only_ui.
        self._mp_audio_note = ttk.Label(mp_frame, text="— audio-only dataset: nothing to size",
                                        foreground="#F59E0B", font=(FONT_FAMILY, 9))
        ttk.Label(mp_frame,
                  text="MP  (0.25 ≈ 512², 1.0 ≈ 1024², 2.4 ≈ 1536², 4.2 ≈ 2048²)   "
                       "example: 512² = 512×512 pixels, or any other width × height with a similar pixel area",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9), wraplength=620,
                  justify=tk.LEFT).pack(side=tk.LEFT)
        ttk.Label(training_content,
                  text="Images are automatically resized to fit this target area — no need to resize your dataset "
                       "beforehand. 0.25 MP ≈ 512×512 of pixel area, and your images do NOT have to be square: any "
                       "aspect ratio works (bucketing handles mixed shapes). Higher = more detail, but more VRAM per "
                       "step: 4.2 MP is 4x the pixels of 1.0 and realistically wants 24-32 GB (or heavy block swap) — "
                       "a 16 GB card will OOM well before it.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720).grid(
            row=17, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Per-image loss watch (Krea 2 only for now — hidden under Klein via
        # _apply_training_arch_visibility). Two tiers sharing one watcher in the trainer:
        # detection reports stuck images; per-image LR additionally throttles them.
        self.krea2_loss_watch_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_LOSS_WATCH", False)))
        self._krea2_losswatch_frame = ttk.Frame(training_content)
        self._krea2_losswatch_frame.grid(row=20, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._krea2_detect_cb = ttk.Checkbutton(
            self._krea2_losswatch_frame,
            text="Detect problem images (per-image loss tracking)",
            variable=self.krea2_loss_watch_var,
        )
        self._krea2_detect_cb.pack(side=tk.LEFT)
        ttk.Button(self._krea2_losswatch_frame, text="👁 View Problem Images",
                   command=self._open_problem_images_window).pack(side=tk.LEFT, padx=(12, 0))
        self.krea2_per_image_lr_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_PER_IMAGE_LR", False)))
        self._krea2_perimglr_cb = ttk.Checkbutton(
            training_content,
            text="Per-image adaptive LR (throttle stuck images, boost healthy learned ones) — experimental",
            variable=self.krea2_per_image_lr_var,
        )
        self._krea2_perimglr_cb.grid(row=21, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        self.krea2_auto_recaption_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_AUTO_RECAPTION", False)))
        self._krea2_autorecap_cb = ttk.Checkbutton(
            training_content,
            text="Auto-recaption stuck images (Qwen3-VL rewrites the caption between epochs) — experimental",
            variable=self.krea2_auto_recaption_var,
        )
        self._krea2_autorecap_cb.grid(row=22, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        self.krea2_warmup_look_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_WARMUP_LOOK", False)))
        self._krea2_warmuplook_cb = ttk.Checkbutton(
            training_content,
            text="Warm up look outliers (unusual angles ease in at low LR — needs a Look Filter scan) — experimental",
            variable=self.krea2_warmup_look_var,
        )
        self._krea2_warmuplook_cb.grid(row=23, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        # All four features attribute a step's loss to ONE image — a batch-mean isn't a
        # per-image signal, so they grey out whenever Batch Size > 1 (note shown instead).
        self._krea2_perimage_batch_note = tk.Label(
            training_content,
            text="Per-image features need Batch Size 1 (Dataset section) — a batch-mean loss "
                 "isn't a per-image signal, so these are disabled at the current batch size.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=680, justify=tk.LEFT)
        self._krea2_perimage_batch_note.grid(row=24, column=0, columnspan=2, sticky=tk.W,
                                             padx=5, pady=(2, 0))
        self._krea2_perimage_batch_note.grid_remove()
        try:
            self.dataset_batch_size_var.trace_add(
                "write", lambda *_a: self._refresh_perimage_toggle_state())
        except Exception:
            pass
        self._refresh_perimage_toggle_state()
        self._krea2_losswatch_hint = ttk.Label(training_content,
                  text="Tracks each image's loss (normalized for the random noise level) across epochs. Detection "
                       "flags images that stay hard without improving — usually mislabeled/off-concept data — in the "
                       "console, the Problem Images window, and loss_log/problem_images.json. Per-image LR also "
                       "throttles them (suspects ×0.7 from ~epoch 3, confirmed stuck ×0.5 from ~epoch 5 escalating "
                       "to ×0.1), eases off mined-out images (×0.6) and gives healthy learned ones a gentle boost (×1.1). Auto-recaption goes "
                       "further: when an image is confirmed stuck, the Qwen3-VL text encoder looks at it and rewrites "
                       "its caption from what's actually visible (appending your Captions-tab trigger word, if set), "
                       "re-encodes it, and gives the image a fresh start (a 2nd attempt goes extra-detailed; still "
                       "stuck after that = excluded from training entirely — edit its caption to re-admit it). "
                       "Warm-up: images the Image Prep Look Filter scored as look-outliers (tight angles, "
                       "profiles — real but unusual) start at ×0.4 LR and ramp to ×1.0 over the first ~4 epochs, "
                       "so they refine the identity instead of fighting it while it forms; released early the "
                       "moment they start improving. Run the Look Filter (scan with 3 baselines) first — it saves "
                       "the scores with your dataset. Batch size 1.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._krea2_losswatch_hint.grid(row=24, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Full fine-tune (rotating windows) — Krea 2 only, experimental branch.
        # Trains the BASE MODEL directly instead of a LoRA, a window of weights at a time so
        # a 12.9B full fine-tune fits a consumer card. Output is a full checkpoint.
        self.krea2_finetune_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FINETUNE", False)))
        self._krea2_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚗ Fine-tune the BASE MODEL instead of training a LoRA (experimental)",
            variable=self.krea2_finetune_var,
            command=lambda: self._on_krea2_ft_toggle(),
        )
        self._krea2_ft_cb.grid(row=60, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))

        self._krea2_ft_frame = ttk.Frame(training_content)
        self._krea2_ft_frame.grid(row=61, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        ttk.Label(self._krea2_ft_frame, text="Window:").pack(side=tk.LEFT, padx=(16, 4))
        self.krea2_ft_mode_var = tk.StringVar(
            value=str(self.settings.get("KREA2_FT_MODE", "Auto (by VRAM)")))
        _ftm = ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_mode_var,
                            values=["Auto (by VRAM)", "component", "block"],
                            state="readonly", width=15)
        _ftm.pack(side=tk.LEFT)
        _ftm.bind("<<ComboboxSelected>>", lambda e: self._apply_krea2_ft_visibility())
        ToolTip(_ftm, "Auto (recommended): sizes the window to the VRAM free at launch and prints "
                      "what it picked and why. Component needs ~29.5 GB free; below that it drops to "
                      "block mode with frozen-block streaming, which fits a 24 GB card.  |  "
                      "component: attention, then each MLP matrix, across ALL 28 blocks — "
                      "every window trains the model's full depth, so a concept is learned by every "
                      "layer at once. 4 windows per cycle.  |  "
                      "block: contiguous slices of blocks. Fewer windows, but each trains only part "
                      "of the depth at a time.")
        self._krea2_ft_blocks_lbl = ttk.Label(self._krea2_ft_frame, text="Blocks/window:")
        self._krea2_ft_blocks_lbl.pack(side=tk.LEFT, padx=(14, 4))
        self.krea2_ft_blocks_var = tk.StringVar(value=str(self.settings.get("KREA2_FT_BLOCKS", "14")))
        self._krea2_ft_blocks_cb = ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_blocks_var,
                                                values=["4", "8", "12", "14", "18"], state="readonly", width=4)
        self._krea2_ft_blocks_cb.pack(side=tk.LEFT)
        ToolTip(self._krea2_ft_blocks_cb, "How many of the 28 blocks train at once (block mode). "
                                          "Measured on a 32 GB card: 4 -> 24.8 GB, 8 -> 24.2 GB, "
                                          "14 -> 27.5 GB, 18 -> 29.5 GB. More blocks = fewer windows "
                                          "= each block gets a bigger share of the run.")
        ttk.Label(self._krea2_ft_frame, text="Rotate every:").pack(side=tk.LEFT, padx=(14, 4))
        self.krea2_ft_every_var = tk.StringVar(value=str(self.settings.get("KREA2_FT_EVERY", "1")))
        ttk.Combobox(self._krea2_ft_frame, textvariable=self.krea2_ft_every_var,
                     values=["1", "2", "3", "5"], state="readonly", width=4).pack(side=tk.LEFT)
        ttk.Label(self._krea2_ft_frame, text="epoch(s)").pack(side=tk.LEFT, padx=(4, 0))
        # One-click launcher for the extractor — the H3 card's twin (pod users only have
        # the browser GUI; hunting for run_diff_to_lora.bat/.sh there is real friction).
        self._krea2_c2l_btn = ttk.Button(self._krea2_ft_frame, text="Checkpoint to LoRA…",
                                         command=self._launch_diff_to_lora)
        self._krea2_c2l_btn.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(self._krea2_c2l_btn,
                "Open the Checkpoint to LoRA tool (its own window): diff a fine-tuned "
                "checkpoint against its base and extract a shareable LoRA at several ranks "
                "at once. Same tool as run_diff_to_lora.bat / .sh in the Fizgig folder.")

        self.krea2_ft_fused_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FT_FUSED", True)))
        self._krea2_ft_fused_cb = ttk.Checkbutton(
            training_content,
            text="Free each gradient as it lands (saves ~5 GB; disables gradient clipping)",
            variable=self.krea2_ft_fused_var,
        )
        self._krea2_ft_fused_cb.grid(row=62, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(2, 0))

        self.krea2_fast_ft_var = tk.BooleanVar(value=bool(self.settings.get("KREA2_FAST_FT", False)))
        self._krea2_fast_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚡ Fast FT — per-tensor fp8 + _scaled_mm on the frozen base (experimental)",
            variable=self.krea2_fast_ft_var,
        )
        self._krea2_fast_ft_cb.grid(row=63, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(2, 0))
        ToolTip(self._krea2_fast_ft_cb,
                "Runs the frozen base through torch._scaled_mm instead of dequantising every "
                "weight on every forward. Needs an RTX 40-series or newer (SM 8.9+); silently "
                "falls back to the normal path per-Linear if anything doesn't fit, and the "
                "console says so.\n\n"
                "Costs accuracy: ~1.5x the per-Linear forward error of the default path "
                "(3.7e-02 vs 2.5e-02, measured on real Krea 2 weights). Most of that is NOT the "
                "scale change (only 1.10x) — _scaled_mm needs the activations in fp8 too, and the "
                "default path keeps them in bf16. That is the price of the fp8 GEMM.\n\n"
                "Off by default so the default path stays exactly as it was. The saved checkpoint "
                "comes from the bf16 master either way, so this never changes what lands on disk — "
                "only the frozen forward the trainable window sees.\n\n"
                "Measured 1.14x on a 5090 (0.849 -> 0.742 s/it, component mode, epoch 4) — about "
                "12% off the wall clock. Loss runs ~1.4% higher at the same step, as a lossier "
                "frozen base predicts. Whether that costs output quality is NOT established — "
                "compare checkpoints before trusting it on a real run.")

        # Optional regularisation set. Real photos, not model output: anchoring to the model's
        # own samples distils its artifacts back in, and a full fine-tune moves every weight so
        # there is nothing bounding the drift. Trained at a fixed low LR so it tethers the prior
        # rather than teaching a new one.
        self._krea2_reg_frame = ttk.Frame(training_content)
        self._krea2_reg_frame.grid(row=64, column=0, columnspan=2, sticky=tk.W, padx=(21, 5), pady=(6, 0))
        ttk.Label(self._krea2_reg_frame, text="Regularisation images (optional):").pack(side=tk.LEFT)
        self.krea2_reg_dir_var = tk.StringVar(value=str(self.settings.get("KREA2_REG_DIR", "")))
        _regent = ttk.Entry(self._krea2_reg_frame, textvariable=self.krea2_reg_dir_var, width=40)
        _regent.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(self._krea2_reg_frame, text="Browse", width=8,
                   command=self._browse_krea2_reg_dir).pack(side=tk.LEFT)
        # Typed paths need the same TOML rewrite the Browse button triggers.
        self.krea2_reg_dir_var.trace_add(
            "write", lambda *_a: self.auto_save_dataset_config_silent())
        ttk.Label(self._krea2_reg_frame, text="LR ×").pack(side=tk.LEFT, padx=(14, 2))
        self.krea2_reg_mult_var = tk.StringVar(value=str(self.settings.get("KREA2_REG_MULT", "0.2")))
        ttk.Combobox(self._krea2_reg_frame, textvariable=self.krea2_reg_mult_var,
                     values=["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"],
                     state="normal", width=5).pack(side=tk.LEFT)
        ToolTip(_regent,
                "A folder of ordinary photos of the broader class — men, women, people — with "
                "normal detailed captions. Leave empty to train without one.\n\n"
                "Why real photos and not model output: generated regularisation images anchor "
                "the model to its own artifacts, and a full fine-tune moves every weight, so "
                "there is nothing bounding that drift. Real photos are an external reference.\n\n"
                "They train at the LR multiplier beside this box. 0.1-0.3 keeps them a nudge — "
                "tethering the model's prior rather than replacing it. Higher values train them "
                "more like real data: at 1.0 they are simply a second subject set, which is a "
                "different (valid) thing — class-balanced training rather than a light anchor.\n\n"
                "Captions matter: anything you leave unsaid gets attributed to the class word "
                "itself. Caption them as you would any training image.")

        self._krea2_ft_hint = ttk.Label(training_content,
                  text="Trains the base model's own weights, not an adapter — no rank bottleneck, so concepts "
                       "don't compete for the same directions. Only part of the model is trainable at a time and "
                       "the window rotates, which is what makes a 12.9B fine-tune fit; a full cycle is 4 epochs in "
                       "component mode, so run at least that many or some weights never train (the console warns "
                       "you). Use a LOW learning rate — 1e-5 or below; LoRA rates will wreck a base model. "
                       "Network Rank/Alpha are ignored. Adaptive LR and in-training previews are turned off "
                       "automatically, so there are no in-training previews — judge the saved checkpoints in "
                       "ComfyUI. EACH SAVE IS A FULL ~26 GB CHECKPOINT; saving every 4 epochs lands one per "
                       "full cycle, when every component has had the same number of passes and checkpoints "
                       "compare like-for-like (~260 GB over a 40-epoch run). Checkpoints are written to the Output Directory above "
                       "(the usual LoRA folder) — point it somewhere with room, e.g. your ComfyUI models/unet. "
                       "Test the result in ComfyUI as a normal Krea 2 model.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720)
        self._krea2_ft_hint.grid(row=65, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))

        # --- Full fine-tune (rotating windows) — MiniMax H3. Same idea as the Krea 2 card:
        # trains the BASE (the int8 checkpoint's own weights), a window of blocks at a time.
        # Output is a full ~21 GB checkpoint per save. Rows 66-69 (Krea's card is 60-65; each
        # family's card hides under the other).
        self.minimax_finetune_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FINETUNE", False)))
        self._minimax_ft_cb = ttk.Checkbutton(
            training_content,
            text="⚗ Fine-tune the BASE MODEL instead of training a LoRA (experimental)",
            variable=self.minimax_finetune_var,
            command=lambda: self._on_minimax_ft_toggle(),
        )
        self._minimax_ft_cb.grid(row=66, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10, 0))

        self._minimax_ft_frame = ttk.Frame(training_content)
        self._minimax_ft_frame.grid(row=67, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(2, 0))
        # Component windows are THE mode (24 Aug: the old Blocks/window picker with its
        # 4/6/8-block windows is gone — block mode never matched component's likeness speed).
        # Each window is one matmul (attention qkv/out, MLP fc1/fc2) across EVERY block:
        # full model depth per window, 4 windows per cycle, on an NF4-resident base.
        ttk.Label(self._minimax_ft_frame, text="Rotate every:").pack(side=tk.LEFT, padx=(16, 4))
        self.minimax_ft_every_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_EVERY", "1")))
        ttk.Combobox(self._minimax_ft_frame, textvariable=self.minimax_ft_every_var,
                     values=["1", "2", "3"], state="readonly", width=4).pack(side=tk.LEFT)
        ttk.Label(self._minimax_ft_frame, text="epoch(s)").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(self._minimax_ft_frame, text="Train on:").pack(side=tk.LEFT, padx=(14, 4))
        self.minimax_ft_scope_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_SCOPE", "All media")))
        _mfts = ttk.Combobox(self._minimax_ft_frame, textvariable=self.minimax_ft_scope_var,
                             values=["All media", "Photos only"], state="readonly", width=11)
        _mfts.pack(side=tk.LEFT)
        ToolTip(_mfts, "A dataset FILTER, not a mode. All media (default) fine-tunes on "
                       "everything in the folder — photos, clips, voice. Photos only is the "
                       "override for a mixed folder: clips and voice are skipped and the run "
                       "behaves as if the dataset were photos-only (with Optimised Likeness "
                       "on, the cycle then tightens to the identity blocks). On a dataset "
                       "that's already just photos this choice changes nothing.")
        ttk.Label(self._minimax_ft_frame, text="Blocks:").pack(side=tk.LEFT, padx=(14, 4))
        self.minimax_ft_blockspec_var = tk.StringVar(
            value=str(self.settings.get("MINIMAX_FT_BLOCKSPEC", "")))
        _mftbs = ttk.Entry(self._minimax_ft_frame, textvariable=self.minimax_ft_blockspec_var,
                           width=10)
        _mftbs.pack(side=tk.LEFT)
        ToolTip(_mftbs, "Optional: restrict the rotation cycle to a block range — the whole "
                        "fine-tune touches only these blocks. '20-49' is the measured likeness "
                        "recipe (protects the fragile 0-19 trunk) and roughly halves the "
                        "system-RAM master copy. Empty = the full model.")
        # One-click launcher for the extractor — pod users only have the browser GUI, so
        # "find run_diff_to_lora.bat in the folder" is real friction there (field, 29 Aug).
        self._minimax_c2l_btn = ttk.Button(self._minimax_ft_frame, text="Checkpoint to LoRA…",
                                           command=self._launch_diff_to_lora)
        self._minimax_c2l_btn.pack(side=tk.LEFT, padx=(14, 0))
        ToolTip(self._minimax_c2l_btn,
                "Open the Checkpoint to LoRA tool (its own window): diff a fine-tuned "
                "checkpoint against its base and extract a shareable LoRA at several ranks "
                "at once. Same tool as run_diff_to_lora.bat / .sh in the Fizgig folder.")

        # Save-every follows the CYCLE, and the cycle follows these controls — so the box is
        # kept live rather than seeded with a stale constant (it used to sit at 13, the old
        # full-model N=4 cycle, whatever mode was picked; the trainer's launch-time snap then
        # silently corrected it and the box looked ignored).
        for _v in (self.minimax_ft_every_var, self.minimax_ft_blockspec_var):
            _v.trace_add("write", lambda *_a: self._refresh_minimax_ft_save_box())

        self.minimax_ft_fused_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FT_FUSED", True)))
        self._minimax_ft_fused_cb = ttk.Checkbutton(
            training_content,
            text="Free each gradient as it lands (fits the window; disables gradient clipping)",
            variable=self.minimax_ft_fused_var,
        )
        self._minimax_ft_fused_cb.grid(row=68, column=0, columnspan=2, sticky=tk.W,
                                       padx=(21, 5), pady=(2, 0))

        # Optional regularisation set — same doctrine as the Krea 2 FT card (real photos of
        # the broader class, trained at a fixed low LR as a prior anchor), with H3-specific
        # lifecycle: reg stills follow the photo routing (likeness window) and stop with the
        # visual category under 'Finish one category early'. Entirely optional — empty = off.
        self._minimax_reg_frame = ttk.Frame(training_content)
        self._minimax_reg_frame.grid(row=69, column=0, columnspan=2, sticky=tk.W,
                                     padx=(21, 5), pady=(6, 0))
        ttk.Label(self._minimax_reg_frame, text="Regularisation images (optional):").pack(side=tk.LEFT)
        self.minimax_reg_dir_var = tk.StringVar(value=str(self.settings.get("MINIMAX_REG_DIR", "")))
        _mregent = ttk.Entry(self._minimax_reg_frame, textvariable=self.minimax_reg_dir_var, width=40)
        _mregent.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(self._minimax_reg_frame, text="Browse", width=8,
                   command=self._browse_minimax_reg_dir).pack(side=tk.LEFT)
        self.minimax_reg_dir_var.trace_add(
            "write", lambda *_a: self.auto_save_dataset_config_silent())
        ttk.Label(self._minimax_reg_frame, text="LR ×").pack(side=tk.LEFT, padx=(14, 2))
        self.minimax_reg_mult_var = tk.StringVar(value=str(self.settings.get("MINIMAX_REG_MULT", "0.2")))
        ttk.Combobox(self._minimax_reg_frame, textvariable=self.minimax_reg_mult_var,
                     values=["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"],
                     state="normal", width=5).pack(side=tk.LEFT)
        ToolTip(_mregent,
                "A folder of ordinary REAL photos of the broader class — people, faces — with "
                "normal detailed captions. Leave empty to train without one.\n\n"
                "A full fine-tune moves the base weights with nothing bounding the drift; these "
                "anchor the model's prior while your subject data pulls it. Real photos, not "
                "model output — generated images anchor the model to its own artifacts.\n\n"
                "They train at the LR multiplier beside this box (0.1-0.3 keeps them a nudge; "
                "keep them the MINORITY of the dataset or the nudge stops reading as one), "
                "follow the same photo routing as your subject stills, and stop when photos & "
                "clips stop under 'Finish one category early' — once subject pressure ends, the "
                "counter-pressure ends with it. Stills only: they tether the visual prior; the "
                "audio prior is protected by voice routing instead.\n\n"
                "With Optimised Likeness on, the anchor pulls only on the likeness blocks — "
                "the same territory your subject photos train, which is the point. On an "
                "audio-only dataset, adding reg stills widens the rotation cycle to include "
                "the photo blocks (the console prints the new span).")

        self._minimax_ft_hint = ttk.Label(training_content,
                  text="Trains the base model's own weights, not an adapter — Network Type "
                       "and Blocks to Train hide while this is on (they're LoRA machinery); "
                       "Optimised Likeness Learning keeps working with its usual meaning. "
                       "Each window is one matmul (attention qkv/out, MLP fc1/fc2) across "
                       "every block — full model depth per window, 4 windows per cycle, on "
                       "an NF4-resident base (the saved checkpoint is still exact int8). "
                       "The Blocks field above is an optional manual restriction of the "
                       "whole fine-tune. Needs a 32 GB card and ~64 GB of system RAM. "
                       "A full ~21 GB checkpoint saves once per COMPLETED CYCLE (every "
                       "block equally trained — the save box snaps to the cycle "
                       "automatically), and previews ride along with each save (plus the "
                       "final one), so every sample matches a checkpoint you can deploy. "
                       "Photos-only dataset? Leave 'Train on' "
                       "alone — there's nothing to skip. Use a LOW learning rate (1e-5 to "
                       "start; H3 is uncalibrated — compare checkpoints). Point the Output "
                       "Directory somewhere with room, judge results in ComfyUI, and distil "
                       "to a shareable LoRA with Checkpoint to LoRA.",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720)
        self._minimax_ft_hint.grid(row=70, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))
        # --- Per-step movement clip (MiniMax only) -----------------------------------------
        # Whichever block sits LAST in the trained range absorbs 2-4x the median block's
        # movement from epoch 1 (measured across four runs; cutting blocks just moves the hot
        # spot to the new last block). This caps any block's movement WITHIN A SINGLE STEP at
        # N x the median block's step. The 3.5.0 version capped CUMULATIVE movement instead,
        # which also scaled down everything the block had legitimately learned — measured as a
        # real likeness ceiling (on was visibly worse than off, off corrupted). Clipping the
        # step prevents the overshoot instead of undoing history, so there is nothing to trade.
        self._minimax_limiter_label = ttk.Label(training_content, text="Per-step movement clip:")
        self._minimax_limiter_label.grid(row=37, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_limiter_frame = ttk.Frame(training_content)
        self._minimax_limiter_frame.grid(row=37, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_BLOCK_LIMIT"] = ttk.Combobox(
            self._minimax_limiter_frame, values=["Off", "1.1 x median (tightest)",
                                                 "1.25 x median (default)",
                                                 "1.5 x median",
                                                 "2.0 x median (loose)",
                                                 "2.5 x median",
                                                 "3.0 x median (safety net only)"],
            width=26, state="readonly")
        self.entries["MINIMAX_BLOCK_LIMIT"].set(
            str(self.settings.get("MINIMAX_BLOCK_LIMIT", "Off")))
        self.entries["MINIMAX_BLOCK_LIMIT"].pack(side=tk.LEFT)
        self._minimax_limiter_hint = ttk.Label(
            training_content,
            text="STRONGLY RECOMMENDED ON — stops any single block overshooting in a step, the "
                 "classic source of distortion. Only the offending step is shortened, so it "
                 "costs nothing that was already learned. Full write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_limiter_hint.grid(row=38, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Reference distillation (MiniMax only, experimental) ---------------------------
        # No picker: the dataset IS the reference pool, so there is nothing to choose.
        self.minimax_distill_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_DISTILL", False)))
        self._minimax_distill_frame = ttk.Frame(training_content)
        self._minimax_distill_frame.grid(row=35, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_distill_cb = ttk.Checkbutton(
            self._minimax_distill_frame, text="Learn identity from my dataset (reference distillation)",
            variable=self.minimax_distill_var, command=self._on_minimax_distill_clicked)
        self._minimax_distill_cb.pack(side=tk.LEFT)
        # Multi Concept shows a warning while identity-learn is OFF (no reference steering), so
        # that hint has to refresh when this checkbox moves, not only when the mode is toggled.
        self.minimax_distill_var.trace_add(
            "write", lambda *_a: self._on_minimax_multiconcept_toggle())
        ttk.Label(self._minimax_distill_frame, text="   teacher ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_WEIGHT"] = ttk.Combobox(
            # 0.4/0.5 added 11 Aug — an even split is a reasonable thing to want and the list
            # stopped at 0.6, so it could not be asked for. 1.0 removes the photo term entirely,
            # which caps the LoRA at what reference mode can already do.
            self._minimax_distill_frame,
            values=["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], width=5)
        self.entries["MINIMAX_DISTILL_WEIGHT"].set(
            str(self.settings.get("MINIMAX_DISTILL_WEIGHT", "0.8")))
        self.entries["MINIMAX_DISTILL_WEIGHT"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_distill_frame, text="   references each ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_REFS"] = ttk.Combobox(
            self._minimax_distill_frame, values=["1", "2", "3", "4"], width=4)
        self.entries["MINIMAX_DISTILL_REFS"].set(
            str(self.settings.get("MINIMAX_DISTILL_REFS", "2")))
        self.entries["MINIMAX_DISTILL_REFS"].pack(side=tk.LEFT)
        # Identity-first: teacher-ONLY for the first stretch, then photos-only. A hard switch,
        # not a blend — the point is where the adapter STARTS, so what phase 2 forgets about the
        # teacher does not matter. Auto sizes phase 1 from the dataset (~650 steps, which is
        # where the teacher error was measured to converge on a real run).
        ttk.Label(self._minimax_distill_frame, text="   identity-first ").pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_PHASE1"] = ttk.Combobox(
            self._minimax_distill_frame, state="readonly", width=22,
            values=["Auto (from dataset size)", "Off — blend throughout",
                    "2 epochs", "4 epochs", "8 epochs", "16 epochs", "30 epochs"])
        self.entries["MINIMAX_DISTILL_PHASE1"].set(
            str(self.settings.get("MINIMAX_DISTILL_PHASE1", "Auto (from dataset size)")))
        self.entries["MINIMAX_DISTILL_PHASE1"].pack(side=tk.LEFT)
        self.entries["MINIMAX_DISTILL_PHASE1"].bind(
            "<<ComboboxSelected>>", lambda _e: self._sync_distill_weight_state())
        self._minimax_distill_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — teaches your LoRA to reproduce identity from the trigger word "
                 "the way H3 does when shown a photo, using your own dataset as the "
                 "references. Needs the ref2va model in Preferences. Full write-up in the "
                 "README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_distill_hint.grid(row=36, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))





        # --- Multi Concept (MiniMax only) ---------------------------------------------------
        # Two subjects in ONE folder get cross-referenced by reference distillation: the pairing
        # rotation runs per [[datasets]] block, so a single block marks subject A's answers
        # against photos of subject B — which blends them rather than separating them. Giving
        # each subject its own folder makes the rotation per-subject for free.
        self.minimax_multiconcept_var = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_MULTICONCEPT", False)))
        self._minimax_mc_frame = ttk.Frame(training_content)
        self._minimax_mc_frame.grid(row=49, column=0, columnspan=2, sticky=tk.W, padx=5,
                                    pady=(8, 0))
        self._minimax_mc_cb = ttk.Checkbutton(
            self._minimax_mc_frame, text="Multi Concept — a second subject in its own folder",
            variable=self.minimax_multiconcept_var,
            command=self._on_minimax_multiconcept_clicked)
        self._minimax_mc_cb.pack(side=tk.LEFT)

        # A LIST even though the UI shows one — a third concept is then a widget, not a rewrite.
        self._concept_folder_vars = [tk.StringVar(
            value=str(self.last_used.get("image_folder2", "") or ""))]
        self._minimax_mc_dir_frame = ttk.Frame(training_content)
        self._minimax_mc_dir_frame.grid(row=50, column=0, columnspan=2, sticky=tk.EW, padx=5,
                                        pady=(2, 0))
        ttk.Label(self._minimax_mc_dir_frame, text="Subject 2 folder:").pack(side=tk.LEFT,
                                                                             padx=(20, 6))
        self._minimax_mc_entry = ttk.Entry(self._minimax_mc_dir_frame,
                                           textvariable=self._concept_folder_vars[0],
                                           state="readonly", width=52)
        self._minimax_mc_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self._minimax_mc_dir_frame, text="Browse…",
                   command=self._browse_concept_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(self._minimax_mc_dir_frame, text="Clear",
                   command=lambda: self._concept_folder_vars[0].set("")).pack(side=tk.LEFT,
                                                                             padx=(4, 0))
        self._minimax_mc_hint = ttk.Label(
            training_content,
            text="Each folder needs its OWN trigger word, in every caption — that is the only "
                 "thing telling the two apart. Caption and prep both folders yourself first; "
                 "this box is training-only. Ticking the mode also sets the settings that suit "
                 "it (identity-learn on, 4 references, identity-first 2 epochs, dropout 0.10, "
                 "adapter-relative LR off) — all still yours to change.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT,
            wraplength=720)
        self._minimax_mc_hint.grid(row=51, column=0, columnspan=2, sticky=tk.W, padx=5,
                                   pady=(0, 4))
        # Only shown when Multi Concept is on AND identity-learn is off — see the toggle handler.
        self._minimax_mc_nodistill_hint = ttk.Label(
            training_content,
            text="Identity-learn is off, so the reference steering that keeps two subjects "
                 "apart is not running — separation rests on your trigger words alone.",
            foreground=COLORS["warning"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT,
            wraplength=720)
        self._minimax_mc_nodistill_hint.grid(row=52, column=0, columnspan=2, sticky=tk.W,
                                             padx=5, pady=(0, 4))
        # These change the [[datasets]] blocks, so the TOML has to be rewritten when they move.
        # Without this the mode looks enabled and silently trains the old single-folder config.
        if hasattr(self, "_auto_save_ds"):
            self.minimax_multiconcept_var.trace_add("write", self._auto_save_ds)
            for _cv in self._concept_folder_vars:
                _cv.trace_add("write", self._auto_save_ds)
                _cv.trace_add("write", lambda *_a: self._save_last_used_paths())

        # --- Slow blocks (MiniMax only, experimental): depth-dependent LR -------------------
        self._minimax_slow_label = ttk.Label(training_content, text="Slower LR for blocks:")
        self._minimax_slow_label.grid(row=33, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self._minimax_slow_frame = ttk.Frame(training_content)
        self._minimax_slow_frame.grid(row=33, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(8, 2))
        self.entries["MINIMAX_SLOW_BLOCKS"] = ttk.Entry(self._minimax_slow_frame, width=22)
        self.entries["MINIMAX_SLOW_BLOCKS"].insert(
            0, str(self.settings.get("MINIMAX_SLOW_BLOCKS", "") or ""))
        self.entries["MINIMAX_SLOW_BLOCKS"].pack(side=tk.LEFT)
        ttk.Label(self._minimax_slow_frame, text="  at ×").pack(side=tk.LEFT)
        self.entries["MINIMAX_SLOW_LR_SCALE"] = ttk.Combobox(
            self._minimax_slow_frame, values=["0.1", "0.2", "0.3", "0.5", "0.7"],
            state="normal", width=6)
        self.entries["MINIMAX_SLOW_LR_SCALE"].set(
            str(self.settings.get("MINIMAX_SLOW_LR_SCALE", "0.2")))
        self.entries["MINIMAX_SLOW_LR_SCALE"].pack(side=tk.LEFT, padx=(2, 0))
        self._minimax_slow_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — leave blank for one learning rate everywhere (normal). A change in "
                 "a late block goes almost straight to the output, while a change early on gets "
                 "smoothed out by the 40-odd blocks after it — so the same learning rate is "
                 "gentle at the front of the model and violent at the back. If the later blocks "
                 "wreck your samples at a rate the early ones handle fine, put those blocks here "
                 "with a multiplier instead of dropping them: 21-49 at ×0.2 trains them at a "
                 "fifth the rate. Same syntax as Blocks to Train, and only blocks you're actually "
                 "training count. Adaptive LR still works — it moves both rates together and "
                 "keeps the ratio.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_slow_hint.grid(row=34, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Train AdaLN (MiniMax only, experimental) --------------------------------------
        # A BooleanVar kept in self.entries so the preset/queue machinery picks it up for free.
        self.entries["MINIMAX_TRAIN_ADALN"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_TRAIN_ADALN", False)))
        self._minimax_adaln_cb = ttk.Checkbutton(
            training_content, text="Train AdaLN (timestep modulation)",
            variable=self.entries["MINIMAX_TRAIN_ADALN"])
        self._minimax_adaln_cb.grid(row=31, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_adaln_hint = ttk.Label(
            training_content,
            text="EXPERIMENT — off by default (the reference trainer leaves it on). AdaLN is "
                 "the part of the model that decides how strongly each block fires at each noise "
                 "level. It only ever sees the noise level: not your image, not your prompt. So it "
                 "CAN'T learn who someone is — but on this base it soaks up roughly 45% of "
                 "everything your LoRA learns. Turning it off hands that capacity to the parts "
                 "that do see the image. It may sharpen likeness, or it may cost you the timing "
                 "control that makes the rest work — run it both ways on the same dataset. Only "
                 "applies to the pruned int8 base; the bf16 one never trains AdaLN anyway.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_adaln_hint.grid(row=32, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # Optimised Likeness Learning — photo steps train the identity blocks only; clips train
        # the full model. BooleanVar in self.entries so presets/queue/last-train carry it free.
        self.entries["MINIMAX_LIKENESS_OPT"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_LIKENESS_OPT", True)))
        # Under FT the tickbox changes the rotation-cycle length (50 blocks -> the 20-49
        # tighten), so the Save-every suggestion follows it live.
        self.entries["MINIMAX_LIKENESS_OPT"].trace_add(
            "write", lambda *_a: self._refresh_minimax_ft_save_box())
        self._minimax_likeness_cb = ttk.Checkbutton(
            training_content, text="Optimised Likeness Learning",
            variable=self.entries["MINIMAX_LIKENESS_OPT"])
        self._minimax_likeness_cb.grid(row=39, column=0, columnspan=2, sticky=tk.W,
                                       padx=5, pady=(8, 0))
        self._minimax_likeness_hint = ttk.Label(
            training_content,
            text=f"Photos train the identity blocks ({MINIMAX_LIKENESS_BLOCKS}) only and "
                 f"voice recordings train the audio zone ({MINIMAX_AUDIO_BLOCKS}) only — "
                 "protecting the base model's rendering, anatomy and prompt following — while "
                 "video clips train the full model. Measured result: sharper, more "
                 "prompt-responsive, better sound, faster to converge. Untick for style or "
                 "scene training (the Style preset does).",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_likeness_hint.grid(row=40, column=0, columnspan=2, sticky=tk.W,
                                         padx=5, pady=(0, 4))
        self._MINIMAX_LIKENESS_HINT_LORA = self._minimax_likeness_hint.cget("text")
        self._MINIMAX_LIKENESS_HINT_FT = (
            f"Under fine-tune this keeps its exact LoRA meaning: photos feed only the "
            f"identity blocks ({MINIMAX_LIKENESS_BLOCKS}), voice feeds only the audio zone "
            f"({MINIMAX_AUDIO_BLOCKS}) — and video follows the restriction tickbox below "
            f"(on: clips train {MINIMAX_LIKENESS_BLOCKS} too; off: clips train the full "
            f"model). The rotation cycle tightens automatically to the union of what your "
            f"dataset actually trains. Untick for style/scene fine-tunes — voice still "
            f"routes to its zone either way. An explicit Blocks range above always wins.")
        # Restrict video to likeness blocks — FT-only sub-tick of likeness mode (Peter,
        # 29 Aug: a confined overnight video run trained perfectly well; on by default,
        # untick for whole-model video). Emitted as --clip_blocks by the FT builder only;
        # shown only when the family is MiniMax AND Fine-tune AND likeness are all on
        # (managed by _sync_minimax_likeness_state, which fires on all three).
        self.entries["MINIMAX_FT_CLIP_LIKENESS"] = tk.BooleanVar(
            value=bool(self.settings.get("MINIMAX_FT_CLIP_LIKENESS", True)))
        self._minimax_ft_clip_cb = ttk.Checkbutton(
            training_content,
            text=f"Restrict video to likeness blocks ({MINIMAX_LIKENESS_BLOCKS}) — in our "
                 "tests this trains video just as well, and it makes clips far lighter on "
                 "VRAM. Untick to train video on the whole model.",
            variable=self.entries["MINIMAX_FT_CLIP_LIKENESS"])
        self._minimax_ft_clip_cb.grid(row=41, column=0, columnspan=2, sticky=tk.W,
                                      padx=(21, 5), pady=(0, 4))
        # trace, not command=: preset loads set the var programmatically and must re-grey too.
        self.entries["MINIMAX_LIKENESS_OPT"].trace_add(
            "write", lambda *_a: self._sync_minimax_likeness_state())

        # Answers "when do changes take effect?" (issue #40) right where people wonder it.
        ttk.Label(training_content,
                  text="When do changes apply? Settings are read when a run launches — changing "
                       "them mid-run does nothing. Pause → Resume relaunches with your current "
                       "settings, so these can be changed at a pause. Dataset/caption changes "
                       "need a fresh run (Resume skips re-caching).",
                  foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"),
                  justify=tk.LEFT, wraplength=720).grid(
            row=30, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 6))

        # === Optimizer Section (Collapsed by default) ===
        optimizer_section = CollapsibleFrame(outer,"Optimizer", default_expanded=False)
        optimizer_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["optimizer"] = optimizer_section

        optimizer_content = optimizer_section.get_content_frame()
        optimizer_content.columnconfigure(1, weight=1)

        self._add_field_to_section(optimizer_content, "OPTIMIZER_TYPE", "Optimizer Type", "dropdown", 0)
        self._add_field_to_section(optimizer_content, "OPTIMIZER_ARGS", "Optimizer Args", "text", 1)
        self._add_field_to_section(optimizer_content, "GRADIENT_ACCUMULATION", "Gradient Accumulation", "int", 2)
        self._add_field_to_section(optimizer_content, "MAX_GRAD_NORM", "Max Grad Norm", "float", 3)
        self._add_field_to_section(optimizer_content, "NETWORK_DROPOUT", "Network Dropout", "float", 4)

        # === Scheduler Section (Collapsed by default) ===
        scheduler_section = CollapsibleFrame(outer,"Other Options", default_expanded=False)
        scheduler_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["scheduler"] = scheduler_section

        scheduler_content = scheduler_section.get_content_frame()
        scheduler_content.columnconfigure(1, weight=1)

        # === Dataset subsection (migrated from the removed Dataset tab) ===
        tk.Label(
            scheduler_content,
            text="Dataset",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(8, 4))

        ttk.Label(scheduler_content, text="Caption Extension:").grid(row=1, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        ttk.Entry(scheduler_content, textvariable=self.dataset_caption_ext_var, width=16).grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)
        tk.Label(scheduler_content, text="(default .txt)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=1, column=2, sticky=tk.W, padx=5)

        # (Target Megapixels moved to the Training Parameters section — it's a core
        # training setting, so it lives with the rest of them rather than buried here.)

        ttk.Label(scheduler_content, text="Batch Size:").grid(row=4, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bs_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bs_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(bs_frame, textvariable=self.dataset_batch_size_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(bs_frame, text="(recommended: 1 — higher values need more VRAM)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        ttk.Label(scheduler_content, text="Bucket Options:").grid(row=5, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bucket_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bucket_frame.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Checkbutton(bucket_frame, text="Enable Bucket", variable=self.dataset_enable_bucket_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(bucket_frame, text="No Upscale (keep small images at native size)",
                        variable=self.dataset_no_upscale_var).pack(side=tk.LEFT)

        ttk.Separator(scheduler_content, orient="horizontal").grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 6))

        # === LR Scheduler ===
        tk.Label(
            scheduler_content,
            text="LR Scheduler",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(4, 4))

        self._add_field_to_section(scheduler_content, "LR_SCHEDULER", "LR Scheduler", "dropdown", 8)

        # Warmup/Decay steps in a sub-frame
        lr_steps_label = tk.Label(
            scheduler_content,
            text="Warmup / Decay:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        lr_steps_label.grid(row=9, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        lr_steps_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        lr_steps_frame.grid(row=9, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(lr_steps_frame, text="Warmup:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_WARMUP_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_WARMUP_STEPS"].pack(side=tk.LEFT, padx=(0, 16))

        # Decay is Klein-only (its warmup_stable_decay path); krea2's scheduler set has no
        # decay-steps knob, so this pair is hidden under Krea 2 rather than silently ignored.
        self._lr_decay_label = tk.Label(lr_steps_frame, text="Decay:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self._lr_decay_label.pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_DECAY_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_DECAY_STEPS"].pack(side=tk.LEFT)

        # Separator before the migrated Advanced fields
        ttk.Separator(scheduler_content, orient="horizontal").grid(row=10, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 4))
        # Inline Attention / Logging / Memory / Metadata fields (formerly the Advanced tab)
        self._populate_other_options(scheduler_content, start_row=11)

        # === Memory & precision section ===
        # Header names every path the section now offers, INT8 included — it is the default on
        # 20 GB+ cards and had no presence anywhere in the UI.
        memory_section = CollapsibleFrame(outer, "Memory & Precision (INT8 / FP8 / NF4)",
                                          default_expanded=True)
        memory_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["memory"] = memory_section

        memory_content = memory_section.get_content_frame()
        memory_content.columnconfigure(1, weight=1)

        # Blocks Swap dropdown — labeled VRAM presets first, then leftover numbers (Klein 9B max=16)
        ttk.Label(memory_content, text="Blocks Swap:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        # VRAM guidance is per Base DiT precision: fp8 Base (~9.6 GB resident,
        # recommended) needs far less than bf16 Base (~18 GB). fp8 fits 16 GB cards
        # with no swap at all.
        blocks_swap_options = [
            "Auto (detect from GPU)",
            "0  (No swap — 16GB fp8 / 24GB bf16)",
            "4  (Light — 14GB fp8 / 20GB bf16)",
            "8  (Moderate — 12GB fp8 / 16GB bf16)",
            "12 (Aggressive — 10GB fp8 / 12GB bf16)",
            "16 (Max — 8GB fp8 / 10GB bf16)",
            "1", "2", "3", "5", "6", "7", "9", "10", "11", "13", "14", "15",
        ]
        _bs_max_len = max(len(v) for v in blocks_swap_options)
        self.entries["BLOCKS_SWAP"] = ttk.Combobox(memory_content, values=blocks_swap_options, width=_bs_max_len + 2)
        self.entries["BLOCKS_SWAP"].grid(row=0, column=1, sticky=tk.W, padx=5, pady=4)
        try:
            _bs_val = self.settings.get("BLOCKS_SWAP", "auto")
            if str(_bs_val).lower() == "auto":
                self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])
            else:
                _bs_int = int(_bs_val)
                _label_map = {0: blocks_swap_options[1], 4: blocks_swap_options[2], 8: blocks_swap_options[3],
                              12: blocks_swap_options[4], 16: blocks_swap_options[5]}
                self.entries["BLOCKS_SWAP"].set(_label_map.get(_bs_int, str(_bs_int)))
        except (ValueError, TypeError):
            self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])

        self._add_field_to_section(memory_content, "RESUME_TRAINING", "Resume Training", "directory", 1)
        # The field had no explanation at all, which made the whole resume feature invisible
        # unless you already knew a state dir was a folder (not a .safetensors) and where it lived.
        tk.Label(memory_content,
                 text="Leave empty for a normal run. To carry on from a saved state, Browse to a folder named "
                      "like myLora-000012-state in your LoRA output folder — the number is the epoch it "
                      "finished. Training continues at the next epoch with the optimizer, learning rate and "
                      "seed exactly as they were, so it picks up mid-run rather than starting over. To train a "
                      "FINISHED LoRA further, pick its highest-numbered state and raise Max Train Epochs first "
                      "— otherwise there are no epochs left to run. Pausing writes one of these for you, and "
                      "the Resume button fills this in automatically; you only need Browse for an older "
                      "checkpoint or a run from a previous session.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=2, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # FP8 Checkboxes. Row label + hint are captured on self (not locals) so
        # _apply_training_arch_visibility can hide them alongside the checkboxes for Krea 2 —
        # otherwise hiding the controls leaves their label and explanation floating.
        self._fp8_row_label = tk.Label(
            memory_content,
            text="Weight Optimization:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        self._fp8_row_label.grid(row=3, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        fp8_frame = tk.Frame(memory_content, bg=COLORS["bg_surface"])
        fp8_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=4)

        self.fp8_var = tk.BooleanVar(value=self.settings["FP8"])
        self.scaled_var = tk.BooleanVar(value=self.settings["SCALED"])

        self.fp8_check = ttk.Checkbutton(fp8_frame, text="FP8 Base", variable=self.fp8_var, command=self.toggle_scaled, style="Surface.TCheckbutton")
        self.fp8_check.pack(side=tk.LEFT, padx=(0, 16))

        self.scaled_check = ttk.Checkbutton(fp8_frame, text="FP8 Scaled", variable=self.scaled_var, state=tk.DISABLED if not self.fp8_var.get() else tk.NORMAL, style="Surface.TCheckbutton")
        self.scaled_check.pack(side=tk.LEFT)
        self._fp8_hint = tk.Label(
            memory_content,
            text="Converts a bf16 model to fp8 at load time. If your Base DiT is already fp8 "
                 "(e.g. flux-2-klein-base-9b-fp8), leave this unchecked — Fizgig detects "
                 "pre-quantised fp8 files automatically.",
            font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=600, justify=tk.LEFT)
        self._fp8_hint.grid(row=4, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # FP8 Text Encoder
        self.fp8_text_encoder_label = tk.Label(
            memory_content,
            text="FP8 Text Encoder:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        self.fp8_text_encoder_label.grid(row=5, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.fp8_text_encoder_var = tk.BooleanVar(value=self.settings["FP8_TEXT_ENCODER"])
        self.fp8_text_encoder_check = ttk.Checkbutton(memory_content, text="Enable FP8 T5/LLM", variable=self.fp8_text_encoder_var, style="Surface.TCheckbutton")
        self.fp8_text_encoder_check.grid(row=5, column=1, sticky=tk.W, padx=5, pady=4)

        # Base precision (Krea 2 only — hidden for Klein by _apply_training_arch_visibility).
        #
        # Was "4-bit Base: Auto / On / Off", which was actively misleading: "Off" meant "not
        # NF4", and on a capable card you would then silently get INT8 — a quantisation the UI
        # never named anywhere. Reading Off as "no quantisation" is what produced the v2.8.6
        # regression; the UI invited that misreading. Every path is now named and selectable,
        # INT8 included, and each one is planned for properly by the memory strategy.
        #
        # quant_4bit_var stays the BooleanVar every downstream consumer reads (True only for
        # NF4); quant_4bit_mode_var holds the canonical key and the combobox shows its label.
        self._quant_4bit_label = tk.Label(memory_content, text="Base precision:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._quant_4bit_label.grid(row=6, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.quant_4bit_var = tk.BooleanVar(value=False)
        self.quant_4bit_mode_var = tk.StringVar(value=self._BASE_PRECISION_LABELS[
            self._normalize_base_precision(
                self.settings.get("QUANT_4BIT_MODE", "")
                or ("On" if self.settings.get("QUANT_4BIT", False) else "auto"))])
        _q4_row = ttk.Frame(memory_content)
        _q4_row.grid(row=6, column=1, sticky=tk.W, padx=5, pady=4)
        self.quant_4bit_check = ttk.Combobox(
            _q4_row, textvariable=self.quant_4bit_mode_var,
            values=list(self._BASE_PRECISION_LABELS.values()), state="readonly", width=34)
        self.quant_4bit_check.pack(side=tk.LEFT)
        self.quant_4bit_check.bind("<<ComboboxSelected>>",
                                   lambda e: self._on_quant_4bit_mode_changed())
        self._quant_4bit_hint = tk.Label(memory_content,
                 text="Auto (recommended) picks the fastest option that fits your FREE VRAM, and sizes block "
                      "swap to match. INT8 is 8-bit — fastest, and ~7x more accurate than 4-bit, but needs "
                      "~18 GB free. 4-bit NF4 is the smallest (~5.6 GB base) so it fits 10–12 GB cards with "
                      "no swap, at a slight quality cost. fp8 is the least compressed of the three and needs "
                      "the most VRAM, so it swaps blocks to fit. Anything you pick explicitly is planned "
                      "for — swap is sized for the option that will actually run.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._quant_4bit_hint.grid(row=7, column=1, sticky=tk.W, padx=5, pady=(0, 4))
        self._on_quant_4bit_mode_changed()  # derive the boolean + sync dependent locks

        # Gradient checkpointing — trades compute for VRAM.
        self._grad_checkpoint_label = tk.Label(memory_content, text="Grad Checkpoint:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._grad_checkpoint_label.grid(row=8, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.grad_checkpoint_var = tk.BooleanVar(value=self.settings.get("GRADIENT_CHECKPOINTING", True))
        self.grad_checkpoint_check = ttk.Checkbutton(
            memory_content, text="Gradient checkpointing (recommended — lower VRAM)",
            variable=self.grad_checkpoint_var, command=self._on_grad_checkpoint_toggle,
            style="Surface.TCheckbutton")
        self.grad_checkpoint_check.grid(row=8, column=1, sticky=tk.W, padx=5, pady=4)
        self._grad_checkpoint_hint = tk.Label(memory_content,
                 text="On (default) recomputes activations during the backward pass to save VRAM — it's what lets "
                      "a 9B LoRA fit on a 16 GB card. Turning it OFF makes training ~20–30% faster but uses far more "
                      "VRAM, so it's only for big cards (24 GB+, ideally 32 GB) with Blocks Swap at 0. On 16 GB, or "
                      "with block swap on, leave it ON.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._grad_checkpoint_hint.grid(row=9, column=1, sticky=tk.W, padx=5, pady=(0, 4))
        # torch.compile (Krea 2 only — hidden under Klein by _apply_training_arch_visibility).
        self._compile_blocks_label = tk.Label(memory_content, text="Compile Blocks:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self._compile_blocks_label.grid(row=10, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        # Auto / On / Off rather than a checkbox: compile is a long-run win and a short-run loss,
        # so the right default is a judgement the trainer makes from the actual run length, the
        # same way Blocks Swap defaults to "Auto (detect from GPU)".
        self.compile_blocks_var = tk.StringVar(value=str(self.settings.get("COMPILE_BLOCKS", "auto")).capitalize())
        self.compile_blocks_check = ttk.Combobox(
            memory_content, textvariable=self.compile_blocks_var, state="readonly", width=36,
            values=["Auto", "On", "Off"])
        self.compile_blocks_check.grid(row=10, column=1, sticky=tk.W, padx=5, pady=4)
        self._compile_blocks_hint = tk.Label(memory_content,
                 text="Auto (recommended) turns torch.compile on only when this run is long enough to repay it. "
                      "It fuses the per-matmul quantise/dequantise work that bounds the INT8 and NF4 paths — "
                      "2.0× per step on INT8 (0.59 → 0.29 s/step, matching OneTrainer) and 1.28× on "
                      "NF4 (0.71 → 0.56) — but costs a ~90 s compile pause first, so a short run is SLOWER "
                      "overall. Break-even is around 600 steps on INT8, 1200 on NF4. NF4 + compile still fits a 16 GB "
                      "card (verified under a 13.5 GB cap). INT8 + compile fits from ~22 GB free: at high resolution "
                      "the checkpoint automatically moves outside the compiled region, which keeps memory at eager "
                      "levels (~18 GB at 1024px, measured ~27% faster than uncompiled). Requires Triton and, on "
                      "Windows, a C++ compiler (VS Build Tools) — both located automatically. Never used with "
                      "Blocks Swap, since swapping moves weights and compiled graphs assume they stay put.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self._compile_blocks_hint.grid(row=11, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Save State ------------------------------------------------------------------
        # Deliberately next to Resume Training (row 1): what writes state and what reads it back
        # belong together. Both families, so NOT added to either list in
        # _apply_training_arch_visibility.
        tk.Label(memory_content, text="Save State:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]
                 ).grid(row=12, column=0, sticky=tk.NW, padx=(12, 8), pady=(10, 4))
        _state_frame = tk.Frame(memory_content, bg=COLORS["bg_surface"])
        _state_frame.grid(row=12, column=1, sticky=tk.W, padx=5, pady=(10, 4))
        self.save_state_var = tk.BooleanVar(value=self.settings.get("SAVE_STATE", True))
        ttk.Checkbutton(_state_frame, text="At each checkpoint", variable=self.save_state_var,
                        style="Surface.TCheckbutton").pack(anchor=tk.W)
        self.save_state_on_train_end_var = tk.BooleanVar(
            value=self.settings.get("SAVE_STATE_ON_TRAIN_END", True))
        ttk.Checkbutton(_state_frame, text="At end of training",
                        variable=self.save_state_on_train_end_var,
                        style="Surface.TCheckbutton").pack(anchor=tk.W)
        tk.Label(memory_content,
                 text="A state dir holds the LoRA plus the optimizer, so a run can pick up exactly where it "
                      "left off — after a crash, or to train a finished LoRA further by raising Max Train "
                      "Epochs and resuming. \"At each checkpoint\" follows Save Every N Epochs. Pause always "
                      "saves state whether these are ticked or not.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=13, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        tk.Label(memory_content, text="Keep Last:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]
                 ).grid(row=14, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.entries["KEEP_LAST_N_STATES"] = ttk.Entry(memory_content, width=40)
        self.entries["KEEP_LAST_N_STATES"].insert(0, str(self.settings.get("KEEP_LAST_N_STATES", 2)))
        self.entries["KEEP_LAST_N_STATES"].grid(row=14, column=1, sticky=tk.W, padx=5, pady=4)
        tk.Label(memory_content,
                 text="States are big — roughly 470 MB at rank 32, 240 MB at rank 16 — so older ones are "
                      "deleted as new ones are written. Only state dirs for THIS LoRA name are touched, and "
                      "the newest is always kept.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(row=15, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # Re-sync now that the GC checkbox exists: the earlier _on_quant_4bit_toggle
        # call ran before it was created, so this applies the NF4→force-GC-on lock
        # when a saved config has 4-bit already enabled.
        self._on_quant_4bit_toggle()

        # === MiniMax H3 rows that belong in OTHER sections ===============================
        # Created here rather than up in Training Parameters because Tkinter cannot re-parent
        # a widget, and these two content frames do not exist until this point in the method.
        # Section display order follows CollapsibleFrame construction order, so the sections
        # themselves must not be reordered to make the parents available earlier.
        # Base Precision -> Memory & Precision; the rest -> Other Options.

        # --- Base Precision (MiniMax only) -------------------------------------------------
        # Auto picks the quantisation and the block-swap count TOGETHER. Deciding swap alone,
        # with the precision already fixed by which file you loaded, gives mid-range cards the
        # worst of both: the int8 base is ~21 GB, so a 24 GB card parks 38 of 50 blocks on CPU
        # and crosses PCIe every step for ~4x the runtime, when the same file loaded 4-bit is
        # ~11 GB and needs no swap at all.
        self._minimax_quant_label = ttk.Label(memory_content, text="Base Precision:")
        self._minimax_quant_label.grid(row=16, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_quant_frame = ttk.Frame(memory_content)
        self._minimax_quant_frame.grid(row=16, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_BASE_QUANT"] = ttk.Combobox(
            self._minimax_quant_frame, values=list(MINIMAX_BASE_QUANT_OPTIONS), width=30,
            state="readonly")
        self.entries["MINIMAX_BASE_QUANT"].set(
            str(self.settings.get("MINIMAX_BASE_QUANT", MINIMAX_BASE_QUANT_OPTIONS[0])))
        self.entries["MINIMAX_BASE_QUANT"].pack(side=tk.LEFT)
        self._minimax_quant_hint = ttk.Label(
            memory_content,
            text="Auto reads your FREE VRAM at launch and picks the base precision and block "
                 "swap together — int8 is the most accurate, 4-bit fits smaller cards. Full "
                 "write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_quant_hint.grid(row=17, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Weight averaging (MiniMax only): EMA ------------------------------------------
        # Damage at a high static LR comes from oversized Adam strides: worst at epoch 1
        # (zero-init adapters, steepest surface) and rough thereafter (the weights zigzag around
        # the good solution). EMA addresses the second by saving a smoothed average.
        #
        # WARMUP is retired (Peter, 10 Aug): the Adapter-relative LR ramp is the better answer to
        # the epoch-1 problem — it holds the step/size RATIO steady instead of guessing an epoch
        # count, so it eases in by construction and keeps doing so. The widget is kept (the
        # launch dict and presets still carry the key) but is never packed and is forced Off.
        self._minimax_smooth_label = ttk.Label(scheduler_content, text="Weight averaging (EMA):")
        self._minimax_smooth_label.grid(row=25, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_smooth_frame = ttk.Frame(scheduler_content)
        self._minimax_smooth_frame.grid(row=25, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_LR_WARMUP"] = ttk.Combobox(     # retired — never packed
            self._minimax_smooth_frame, values=["Off", "1 epoch", "2 epochs", "3 epochs"],
            width=10, state="readonly")
        self.entries["MINIMAX_LR_WARMUP"].set("Off")
        self.entries["MINIMAX_EMA"] = ttk.Combobox(
            self._minimax_smooth_frame, values=["Off", "0.98 (light)", "0.99 (recommended)",
                                                "0.995 (strong)"],
            width=18, state="readonly")
        self.entries["MINIMAX_EMA"].set(str(self.settings.get("MINIMAX_EMA", "Off")))
        self.entries["MINIMAX_EMA"].pack(side=tk.LEFT)
        self._minimax_smooth_hint = ttk.Label(
            scheduler_content,
            text="Saves a smoothed average of the weights, so checkpoints come out crisper "
                 "when you push the LR hard. Costs no speed. Full write-up in the README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_smooth_hint.grid(row=26, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Adapter-relative LR ramp (MiniMax only, EXPERIMENT, default Off) ---------------
        # From a real run: an adapter at ||dW||~53 took a full 2e-4 for ten epochs with no
        # distortion and gave the best likeness of the project, while a fresh adapter is
        # visibly damaged by half that. Same step, 9% perturbation vs 150% — a LoRA starts at
        # zero, so step/size is worst at step 1 and improves from there. This holds that RATIO
        # steady, which ramps the LR up toward the box value as the adapter grows.
        self._minimax_ramp_label = ttk.Label(scheduler_content, text="Adapter-relative LR:")
        self._minimax_ramp_label.grid(row=27, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_ramp_frame = ttk.Frame(scheduler_content)
        self._minimax_ramp_frame.grid(row=27, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_ADAPTER_RAMP"] = ttk.Combobox(
            self._minimax_ramp_frame, values=["Off", "0.003 (slow build)",
                                              "0.005 (recommended)", "0.01 (fast build)"],
            width=24, state="readonly")
        self.entries["MINIMAX_ADAPTER_RAMP"].set(
            str(self.settings.get("MINIMAX_ADAPTER_RAMP", "Off")))
        self.entries["MINIMAX_ADAPTER_RAMP"].pack(side=tk.LEFT)
        self._minimax_ramp_hint = ttk.Label(
            scheduler_content,
            text="Makes the Learning Rate box a CEILING the run climbs toward instead of a rate "
                 "it starts at, so set the LR to where you want to end up. Full write-up in the "
                 "README.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_ramp_hint.grid(row=28, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        # --- Caption dropout (MiniMax only) -------------------------------------------------
        # A fraction of steps train the image against the EMPTY prompt instead of its caption.
        # On a single-concept set that is healthy regularisation — it stops the LoRA leaning on
        # the trigger token alone. On a MULTI-concept set it is the opposite: those steps teach
        # the model to produce the concept with no trigger at all, which is exactly how one
        # subject leaks into the other's prompts. Was hardcoded at 0.05 (the CLI default) with
        # no way to change it from the GUI.
        self._minimax_capdrop_label = ttk.Label(scheduler_content, text="Caption dropout:")
        self._minimax_capdrop_label.grid(row=29, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self._minimax_capdrop_frame = ttk.Frame(scheduler_content)
        self._minimax_capdrop_frame.grid(row=29, column=1, sticky=tk.W, padx=5, pady=(8, 0))
        self.entries["MINIMAX_CAPTION_DROPOUT"] = ttk.Combobox(
            self._minimax_capdrop_frame,
            values=["Off", "0.05 (default)", "0.10 (strong)"],
            width=24, state="readonly")
        self.entries["MINIMAX_CAPTION_DROPOUT"].set(
            str(self.settings.get("MINIMAX_CAPTION_DROPOUT", "0.05 (default)")))
        self.entries["MINIMAX_CAPTION_DROPOUT"].pack(side=tk.LEFT)
        self._minimax_capdrop_hint = ttk.Label(
            scheduler_content,
            text="Trains a few percent of steps with no caption, so the LoRA does not lean "
                 "entirely on the trigger word.",
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_capdrop_hint.grid(row=30, column=0, columnspan=2, sticky=tk.W, padx=5,
                                        pady=(0, 4))

        # --- Blocks to Train (MiniMax only, experimental) ---------------------------------
        self._minimax_blocks_label = ttk.Label(scheduler_content, text="Blocks to Train:")
        self._minimax_blocks_label.grid(row=31, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self._minimax_blocks_frame = ttk.Frame(scheduler_content)
        self._minimax_blocks_frame.grid(row=31, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(8, 2))
        # Editable, not readonly — the presets are starting points and the real control is typing
        # a spec. Anything the trainer's parser takes is legal here.
        self.entries["MINIMAX_BLOCKS"] = ttk.Combobox(
            self._minimax_blocks_frame, values=MINIMAX_BLOCK_OPTIONS, width=34)
        self.entries["MINIMAX_BLOCKS"].pack(side=tk.LEFT)
        self._select_combo_by_token(self.entries["MINIMAX_BLOCKS"],
                                    self.settings.get("MINIMAX_BLOCKS", "all"))
        # Live readout: a typed spec is easy to fat-finger, and "trained 3 blocks when you meant
        # 30" is invisible in the output. Says how many blocks the box currently means.
        self._minimax_blocks_count = tk.Label(self._minimax_blocks_frame, text="",
                                              font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"])
        self._minimax_blocks_count.pack(side=tk.LEFT, padx=(10, 0))
        self.entries["MINIMAX_BLOCKS"].bind(
            "<KeyRelease>", lambda _e: self._refresh_minimax_blocks_count())
        self.entries["MINIMAX_BLOCKS"].bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_minimax_blocks_count())
        self._minimax_blocks_hint = ttk.Label(
            scheduler_content,
            text=self._MINIMAX_BLOCKS_HINT,
            foreground=COLORS["text_explain"], font=(FONT_FAMILY, 9, "italic"), justify=tk.LEFT, wraplength=720)
        self._minimax_blocks_hint.grid(row=32, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))
        self._refresh_minimax_blocks_count()

        # Training Structure lives in Training Parameters now — see _build_minimax_structure_row,
        # called from that section. It used to sit here in Other Options, collapsed, which is
        # where the single most consequential MiniMax setting was least likely to be found.

        # === Timestep & Noise Schedule Section (Collapsed by default) ===
        timestep_section = CollapsibleFrame(outer,"Timestep & Noise Schedule", default_expanded=False)
        timestep_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["timestep"] = timestep_section

        ts_content = timestep_section.get_content_frame()
        ts_content.columnconfigure(1, weight=1)

        ts_row = 0

        # Quick Preset Buttons
        preset_btn_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        preset_btn_frame.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(8, 4))

        tk.Label(preset_btn_frame, text="Quick Presets:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))

        for preset_name, preset_fn in [
            ("Full Range", self._ts_preset_full_range),
            ("Structure Focus", self._ts_preset_structure),
            ("Detail Focus", self._ts_preset_detail),
            ("Balanced Sigmoid", self._ts_preset_sigmoid),
        ]:
            btn = ttk.Button(preset_btn_frame, text=preset_name, command=preset_fn)
            btn.pack(side=tk.LEFT, padx=2)

        ts_row += 1

        # Timestep Sampling (editable dropdown)
        ts_sampling_label = tk.Label(ts_content, text="Timestep Sampling:",
                                     font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_sampling_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.ts_sampling_var = tk.StringVar(value=self.settings["TIMESTEP_SAMPLING"])
        # Must mirror the trainer's argparse choices exactly — "qwen_shift" was offered here
        # but rejected by the trainer at launch; "qinglong_flux" existed but wasn't offered.
        ts_sampling_options = ["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "logsnr", "qinglong_flux"]
        self.ts_sampling_combo = ttk.Combobox(ts_content, textvariable=self.ts_sampling_var,
                                               values=ts_sampling_options, state="readonly", width=20)
        self.ts_sampling_combo.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        self.ts_sampling_combo.bind("<<ComboboxSelected>>", self._on_timestep_sampling_changed)
        self.entries["TIMESTEP_SAMPLING"] = self.ts_sampling_combo
        ts_row += 1

        # Discrete Flow Shift — not used by Klein 9B (uses flux2_shift automatic).
        # Widget created but not gridded so presets/save still work.
        self.ts_flow_shift_label = tk.Label(ts_content, text="Discrete Flow Shift:",
                                            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.entries["DISCRETE_FLOW_SHIFT"] = ttk.Entry(ts_content, width=12)
        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, self.settings["DISCRETE_FLOW_SHIFT"])

        # Sigmoid Scale
        self.ts_sigmoid_label = tk.Label(ts_content, text="Sigmoid Scale:",
                                         font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_sigmoid_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.entries["SIGMOID_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["SIGMOID_SCALE"].insert(0, self.settings["SIGMOID_SCALE"])
        self.entries["SIGMOID_SCALE"].grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        ts_row += 1

        # Min / Max Timestep on one row
        ts_range_label = tk.Label(ts_content, text="Timestep Range:",
                                  font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_range_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        ts_range_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        ts_range_frame.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(ts_range_frame, text="Min:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MIN_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MIN_TIMESTEP"].insert(0, self.settings["MIN_TIMESTEP"])
        self.entries["MIN_TIMESTEP"].pack(side=tk.LEFT, padx=(0, 16))
        self.entries["MIN_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        tk.Label(ts_range_frame, text="Max:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MAX_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MAX_TIMESTEP"].insert(0, self.settings["MAX_TIMESTEP"])
        self.entries["MAX_TIMESTEP"].pack(side=tk.LEFT)
        self.entries["MAX_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        ts_row += 1

        # Noise range description label
        self.noise_range_label = tk.Label(ts_content, text="", font=(FONT_FAMILY, 9),
                                          fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.noise_range_label.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=(0, 4))
        self._update_noise_range_label()
        ts_row += 1

        # Preserve Distribution Shape
        self.preserve_dist_var = tk.BooleanVar(value=self.settings["PRESERVE_DISTRIBUTION"])
        self.preserve_dist_check = ttk.Checkbutton(ts_content, text="Preserve Distribution Shape",
                                                    variable=self.preserve_dist_var, style="Surface.TCheckbutton")
        self.preserve_dist_check.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=4)
        ToolTip(self.preserve_dist_check, "Use rejection sampling to preserve the original\n"
                "distribution shape within the min/max range.\n"
                "Only effective when min/max timestep is set.")
        self.entries["PRESERVE_DISTRIBUTION"] = self.preserve_dist_var
        ts_row += 1

        # --- Hidden fields (not used by Klein 9B) ---
        # Widgets created but not gridded so presets/save/command-building still work.

        # Weighting Scheme
        self.ts_weighting_label = tk.Label(ts_content, text="Weighting Scheme:")
        self.weighting_scheme_var = tk.StringVar(value=self.settings["WEIGHTING_SCHEME"])
        weighting_options = ["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"]
        self.ts_weighting_combo = ttk.Combobox(ts_content, textvariable=self.weighting_scheme_var,
                                                values=weighting_options, state="readonly", width=20)
        self.ts_weighting_combo.bind("<<ComboboxSelected>>", self._on_weighting_scheme_changed)
        self.entries["WEIGHTING_SCHEME"] = self.ts_weighting_combo

        # Logit Mean / Std
        self.ts_logit_label = tk.Label(ts_content, text="Logit Normal:")
        logit_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        self.entries["LOGIT_MEAN"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_MEAN"].insert(0, self.settings["LOGIT_MEAN"])
        self.entries["LOGIT_STD"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_STD"].insert(0, self.settings["LOGIT_STD"])

        # Mode Scale
        self.ts_mode_label = tk.Label(ts_content, text="Mode Scale:")
        self.entries["MODE_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["MODE_SCALE"].insert(0, self.settings["MODE_SCALE"])

        # Initial state for conditional fields
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()

        # === Reorder collapsible sections: Training → Memory & FP8 → Timestep → Optimizer → Other Options ===
        # Sections were created in declaration order; re-pack in the desired display order.
        # Training Parameters and Memory & FP8 are open by default since most users need both.
        for _sec in (training_section, memory_section, timestep_section, optimizer_section, scheduler_section):
            try:
                _sec.pack_forget()
                _sec.pack(fill=tk.X, padx=36, pady=(0, 16))
            except Exception:
                pass

        # === Run card — Enable Cache + Start/Pause/Resume/Stop buttons ===
        run_card = self._start_section_card(outer, "Run", None)

        self.enable_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_card, text="Enable Cache Preparation", variable=self.enable_cache_var).pack(anchor=tk.W, pady=(0, 12))

        button_frame = tk.Frame(run_card, bg=COLORS["bg_surface"])
        button_frame.pack(anchor=tk.W)

        self._start_training_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training, style="Primary.TButton")
        self._start_training_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._pause_training_btn = ttk.Button(button_frame, text="Pause Training", command=self._pause_training)
        self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._pause_training_btn.pack_forget()  # hidden until training is running

        self._resume_training_btn = ttk.Button(button_frame, text="Resume Training", command=self._resume_training, style="Primary.TButton")
        self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._resume_training_btn.pack_forget()  # hidden until paused state exists

        stop_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, style="Danger.TButton")
        stop_btn.pack(side=tk.LEFT, padx=(0, 24))

        ttk.Button(button_frame, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT)

        # === Console Output card ===
        console_card = self._start_section_card(outer, "Console Output", None)
        self.console_frame = tk.Frame(console_card, bg=COLORS["bg_surface"])
        self.console_frame.pack(fill=tk.BOTH, expand=True)

        self.console_output = tk.Text(
            self.console_frame,
            height=12,
            width=80,
            bg=COLORS["bg_deep"],
            fg=COLORS["text_primary"],
            font=(FONT_MONO, 9),
            wrap="word",
            state="disabled",
            selectbackground=COLORS["accent"],
            selectforeground="white",
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border_focus"],
            padx=12,
            pady=8
        )
        self.console_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console_scrollbar = ttk.Scrollbar(
            self.console_frame,
            orient="vertical",
            command=self.console_output.yview,
            style="Vertical.TScrollbar"
        )
        self.console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_output.configure(yscrollcommand=self.console_scrollbar.set)

        self.console_output.bind("<MouseWheel>", self.on_mousewheel)
        self.console_output.bind("<Button-4>", self.on_mousewheel)
        self.console_output.bind("<Button-5>", self.on_mousewheel)
        self.console_output.bind("<Button-3>", self.show_context_menu)

        # Initial UI update based on architecture
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()

        self._add_youtube_help_button(outer, "training")

