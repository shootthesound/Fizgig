import os

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class ExtractTabMixin:
    def create_extract_tab(self):
        """Create the Extract tab (Start-tab styled) — activation-based LoRA extraction."""
        scrollable_frame, _ = self.create_scrollable_frame(self.extract_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Extract",
            "Distill an existing LoRA down to a lower rank. Klein: block + timestep targeting, optional "
            "activation-weighted SVD. Krea 2 and MiniMax H3: pure weight SVD over all blocks (no block map yet).",
        )

        # Model family selector. Krea 2 / MiniMax H3 = pure weight SVD over all blocks (no pipeline /
        # prompt / timesteps / block presets), so those cards are hidden for both.
        _efam = str(self.last_used.get("extract_family", "klein"))
        if _efam not in ("klein", "krea2", "minimax"):
            _efam = "klein"
        self.extract_family_var = tk.StringVar(value=_efam)
        efam_card = self._start_section_card(
            outer, "Model Family",
            "Klein 9B (full extractor), Krea 2 or MiniMax H3 (weight-only SVD; block-targeting presets "
            "come once each block map exists). Browsing a LoRA auto-switches to its family.",
        )
        _ef = tk.Frame(efam_card, bg=COLORS["bg_surface"])
        _ef.pack(anchor=tk.W)
        ttk.Radiobutton(_ef, text="Klein 9B", variable=self.extract_family_var, value="klein",
                        command=self._on_extract_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_ef, text="Krea 2", variable=self.extract_family_var, value="krea2",
                        command=self._on_extract_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_ef, text="MiniMax H3", variable=self.extract_family_var, value="minimax",
                        command=self._on_extract_family_changed).pack(side=tk.LEFT)

        # Card 1: Source & Output
        io_card = self._start_section_card(
            outer, "Source & Output",
            "Choose the source LoRA and name the extraction — it will land in your LoRA output folder.",
        )
        io_card.grid_columnconfigure(1, weight=1)

        ttk.Label(io_card, text="Source LoRA:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_source_var = tk.StringVar()
        ttk.Entry(io_card, textvariable=self.extract_source_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(io_card, text="Browse", command=self._browse_extract_source).grid(row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4)

        ttk.Label(io_card, text="Output Name:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_output_var = tk.StringVar()
        ttk.Entry(io_card, textvariable=self.extract_output_var, width=50).grid(row=1, column=1, sticky=tk.EW, pady=4)
        tk.Label(io_card, text="(will be saved in your LoRA output folder)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=2, column=1, sticky=tk.W, pady=(0, 4)
        )

        # Card 2: Preset
        preset_card = self._start_section_card(
            outer, "Preset",
            "Fast * = pure weight SVD (samples=0, no GPU probes).  |  Identity = single 1-16.  |  "
            "Style = style+comp @ late timesteps.  |  Style+Composition = double 0-7 + single 0-1 + single 2 @ 0.5.  |  "
            "Details = single 12-23.",
        )
        self._extract_preset_container = preset_card.master.master
        preset_row = tk.Frame(preset_card, bg=COLORS["bg_surface"])
        preset_row.pack(anchor=tk.W)
        ttk.Label(preset_row, text="Extract Preset:").pack(side=tk.LEFT, padx=(0, 10))
        self.extract_preset_var = tk.StringVar(value="Identity")
        preset_combo = ttk.Combobox(
            preset_row, textvariable=self.extract_preset_var,
            values=["All Blocks", "Fast SVD", "Identity", "Fast Identity", "Style",
                    "Style+Composition", "Fast Style+Composition", "Details", "Fast Details", "Custom"],
            state="readonly", width=24,
        )
        preset_combo.pack(side=tk.LEFT)
        preset_combo.bind("<<ComboboxSelected>>", self._on_extract_preset_changed)

        # Card 3: Custom Blocks — pack-managed so pack_forget/pack drives visibility
        self._extract_custom_frame = tk.Frame(outer, bg=COLORS["bg_deep"])
        self._extract_custom_frame.pack(fill=tk.X, padx=36, pady=(0, 16))
        custom_card = tk.Frame(self._extract_custom_frame, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["border"], highlightthickness=1, bd=0)
        custom_card.pack(fill=tk.X)
        tk.Label(custom_card, text="Custom Blocks",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(16, 2))
        tk.Label(custom_card,
                 text="Pick individual blocks to target. Only shown when preset = Custom.",
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

        custom_content = tk.Frame(custom_card, bg=COLORS["bg_surface"])
        custom_content.pack(fill=tk.X, padx=20, pady=(0, 16))

        self.extract_block_vars = {}

        header_row = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        header_row.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(header_row, text="Select individual blocks:",
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(header_row, text="All", width=5,
                   command=lambda: self._set_all_extract_blocks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="None", width=5,
                   command=lambda: self._set_all_extract_blocks(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Identity", width=8,
                   command=lambda: self._set_category_extract_blocks("identity")).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Style+Comp", width=10,
                   command=lambda: self._set_category_extract_blocks("style_composition")).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Details", width=8,
                   command=lambda: self._set_category_extract_blocks("details")).pack(side=tk.LEFT, padx=2)

        double_row = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        double_row.pack(anchor=tk.W, pady=2)
        tk.Label(double_row, text="Double:", width=8, fg="#5B9BD5", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(8):
            key = f"double_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(double_row, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        single_row1 = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        single_row1.pack(anchor=tk.W, pady=2)
        tk.Label(single_row1, text="Single:", width=8, fg="#70AD47", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(12):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(single_row1, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        single_row2 = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        single_row2.pack(anchor=tk.W, pady=2)
        tk.Label(single_row2, text="Single:", width=8, fg="#ED7D31", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(12, 24):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(single_row2, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tk.Label(custom_content,
                 text="double + single 0-1 = style+composition  |  single 1-16 = identity (overlaps at 1 and 12-16)  |  single 12-23 = details",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(4, 0))

        self._extract_custom_frame.pack_forget()  # hidden until preset = Custom

        # Card 4: Options (rank / timesteps / forward passes)
        options_card = self._start_section_card(
            outer, "Options",
            "Timesteps: 'all' for general, 'late' for style, 'early' for composition.  |  "
            "Forward Passes: 0 = pure weight SVD (fastest, timestep-agnostic). Higher = better activation-weighted accuracy.",
        )
        # Anchor for the Custom Blocks card's re-pack so it reappears between Preset and Options.
        self._extract_options_anchor = options_card.master.master
        options_row = tk.Frame(options_card, bg=COLORS["bg_surface"])
        options_row.pack(anchor=tk.W)

        ttk.Label(options_row, text="Target Rank:").pack(side=tk.LEFT, padx=(0, 6))
        self.extract_rank_var = tk.StringVar(value="4")
        rank_combo = ttk.Combobox(options_row, textvariable=self.extract_rank_var,
                     values=["1", "2", "4", "8", "16"], state="readonly", width=4)
        rank_combo.pack(side=tk.LEFT, padx=(0, 20))
        rank_combo.bind("<<ComboboxSelected>>", lambda e: self._update_extract_output_name())

        self._extract_timesteps_label = ttk.Label(options_row, text="Timesteps:")
        self._extract_timesteps_label.pack(side=tk.LEFT, padx=(0, 6))
        self.extract_timesteps_var = tk.StringVar(value="all")
        self._extract_timesteps_combo = ttk.Combobox(
            options_row, textvariable=self.extract_timesteps_var,
            values=["all", "early", "mid", "late"], state="readonly", width=8,
        )
        self._extract_timesteps_combo.pack(side=tk.LEFT, padx=(0, 20))

        self._extract_samples_label = ttk.Label(options_row, text="Forward Passes:")
        self._extract_samples_label.pack(side=tk.LEFT, padx=(0, 6))
        self.extract_samples_var = tk.StringVar(value="16")
        self._extract_samples_combo = ttk.Combobox(
            options_row, textvariable=self.extract_samples_var,
            values=["0", "8", "16", "32"], state="readonly", width=4,
        )
        self._extract_samples_combo.pack(side=tk.LEFT)
        self._extract_samples_combo.bind("<<ComboboxSelected>>", self._on_extract_samples_changed)

        # Card 5: Prompt
        prompt_card = self._start_section_card(
            outer, "Prompt",
            "Used during the GPU probe forward passes. Include the source LoRA's trigger word for best results.",
        )
        self._extract_prompt_container = prompt_card.master.master
        prompt_card.grid_columnconfigure(1, weight=1)
        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_prompt_var = tk.StringVar(value="")
        ttk.Entry(prompt_card, textvariable=self.extract_prompt_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)

        # Card 6: Run
        run_card = self._start_section_card(
            outer, "Run",
            "Extraction runs SVD on each block and can take several minutes depending on rank and block count.",
        )
        self._extract_run_container = run_card.master.master
        run_row = tk.Frame(run_card, bg=COLORS["bg_surface"])
        run_row.pack(anchor=tk.W)
        self.extract_run_btn = ttk.Button(run_row, text="Extract LoRA", command=self._run_extract, style="Primary.TButton")
        self.extract_run_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.extract_open_btn = ttk.Button(run_row, text="Open Output Folder", command=self._open_extract_folder, state="disabled")
        self.extract_open_btn.pack(side=tk.LEFT)

        self.extract_progress_var = tk.StringVar(value="")
        tk.Label(run_card, textvariable=self.extract_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(10, 0))

        # Family-aware "how long this takes" note (set by _apply_extract_family_ui).
        self.extract_time_note_var = tk.StringVar(value="")
        tk.Label(run_card, textvariable=self.extract_time_note_var,
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

        # Card 7: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)
        self.extract_log = scrolledtext.ScrolledText(
            log_card, height=14, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.extract_log.pack(fill=tk.BOTH, expand=True)

        self._extract_output_path = None

        # Apply the persisted family (krea2 hides the Klein-only block/prompt/probe controls).
        self._apply_extract_family_ui(str(self.extract_family_var.get()) == "krea2")

        self._add_youtube_help_button(outer, "extract")

    def _on_mining_preset_changed(self, *args):
        """Fill gradient mining controls from the selected preset."""
        name = self._mining_preset_var.get()
        preset = self._mining_presets.get(name)
        if not preset:
            return
        for key, entry_key in [("snr", "GRADIENT_MINING_THRESHOLD"),
                                ("ortho", "GRADIENT_MINING_ORTHO")]:
            entry = self.entries.get(entry_key)
            if entry and key in preset:
                entry.delete(0, tk.END)
                entry.insert(0, preset[key])

    def _on_training_preset_changed(self, *args):
        """Auto-fill MIN/MAX_TIMESTEP and show/hide custom block picker based on training preset."""
        preset = self.training_preset_var.get()

        # Show/hide custom block panel
        if preset == "Custom":
            self._training_custom_frame.grid()
        else:
            self._training_custom_frame.grid_remove()

        # Skip timestep auto-fill for Custom (user-driven)
        if preset == "Custom":
            self._last_area_applied = preset
            return

        # Only rewrite MIN/MAX when the Model Area actually CHANGED. This handler is also
        # invoked as a visibility refresh (arch switches, tab builds) — unconditionally
        # auto-filling wiped the user's hand-set timesteps on every one of those calls.
        if getattr(self, "_last_area_applied", None) == preset:
            return
        self._last_area_applied = preset

        # Auto-fill MIN/MAX_TIMESTEP entries
        min_entry = self.entries.get("MIN_TIMESTEP")
        max_entry = self.entries.get("MAX_TIMESTEP")
        if min_entry is None or max_entry is None:
            return

        if preset == "Style":
            min_val, max_val = "0", "400"
        else:  # Full Model, Identity, Style+Composition, Details
            min_val, max_val = "", ""

        min_entry.delete(0, tk.END)
        if min_val:
            min_entry.insert(0, min_val)
        max_entry.delete(0, tk.END)
        if max_val:
            max_entry.insert(0, max_val)
        self.settings["MIN_TIMESTEP"] = min_val
        self.settings["MAX_TIMESTEP"] = max_val
        if hasattr(self, "_update_noise_range_label"):
            try:
                self._update_noise_range_label()
            except Exception:
                pass

    def _set_all_training_blocks(self, value: bool):
        """Tick/untick all per-block checkboxes in Custom training mode."""
        for var in self.training_block_vars.values():
            var.set(value)

    def _set_category_training_blocks(self, category: str):
        """Select all blocks in a category (identity/style_composition/details). Clears others."""
        for key, var in self.training_block_vars.items():
            var.set(False)
        if category == "style_composition":
            for i in range(8):
                self.training_block_vars[f"double_blocks.{i}"].set(True)
            for i in (0, 1):
                self.training_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "identity":
            for i in range(1, 17):
                self.training_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "details":
            for i in range(12, 24):
                self.training_block_vars[f"single_blocks.{i}"].set(True)

    def _build_custom_training_patterns(self):
        """Build a list of include_patterns regexes from the Custom block checkboxes."""
        selected = [key for key, var in self.training_block_vars.items() if var.get()]
        if not selected:
            return None
        patterns = []
        for key in selected:
            # key is "double_blocks.N" or "single_blocks.N"
            kind, idx = key.split(".")
            patterns.append(rf".*{kind}\.{idx}\..*")
        return patterns

    def _on_extract_preset_changed(self, *args):
        """Show/hide custom block checkboxes and auto-switch timesteps/samples based on preset."""
        preset = self.extract_preset_var.get()
        if preset == "Custom":
            # Pack before Options so the custom card appears between Preset and Options.
            self._extract_custom_frame.pack(fill=tk.X, padx=36, pady=(0, 16),
                                             before=self._extract_options_anchor)
        else:
            self._extract_custom_frame.pack_forget()

        # Fast * presets force samples=0 (pure weight SVD)
        is_fast = preset in ("Fast SVD", "Fast Identity", "Fast Style+Composition", "Fast Details")
        if is_fast:
            self.extract_samples_var.set("0")
        elif preset in ("All Blocks", "Identity", "Style", "Style+Composition", "Details"):
            # Activation-weighted presets need samples>0; bump off 0 if user is coming from a Fast variant
            if self.extract_samples_var.get() == "0":
                self.extract_samples_var.set("16")

        # Timestep auto-fill
        if preset == "Style":
            self.extract_timesteps_var.set("late")
        elif preset in ("All Blocks", "Fast SVD", "Identity", "Fast Identity",
                        "Style+Composition", "Fast Style+Composition", "Details", "Fast Details"):
            self.extract_timesteps_var.set("all")

        # Reflect timestep combo state (samples may have just changed above)
        self._apply_extract_samples_state()

        # Update suggested output filename to match new preset
        self._update_extract_output_name()

    def _on_extract_samples_changed(self, *args):
        """Grey out timesteps when samples=0; map presets to their Fast variant where applicable."""
        if self.extract_samples_var.get() == "0":
            preset = self.extract_preset_var.get()
            # Map activation-weighted presets to their Fast equivalent so the UI reflects reality
            fast_map = {
                "All Blocks": "Fast SVD",
                "Identity": "Fast Identity",
                "Style": "Fast Style+Composition",   # Style is inherently timestep-based; Fast loses that meaning
                "Style+Composition": "Fast Style+Composition",
                "Details": "Fast Details",
            }
            if preset in fast_map:
                self.extract_preset_var.set(fast_map[preset])
                self.extract_timesteps_var.set("all")
        self._apply_extract_samples_state()

    def _apply_extract_samples_state(self):
        """Timesteps dropdown is meaningful only when forward passes > 0."""
        if self.extract_samples_var.get() == "0":
            self._extract_timesteps_combo.configure(state="disabled")
        else:
            self._extract_timesteps_combo.configure(state="readonly")

    def _set_all_extract_blocks(self, value: bool):
        """Tick/untick all per-block checkboxes in Custom mode."""
        for var in self.extract_block_vars.values():
            var.set(value)

    def _set_category_extract_blocks(self, category: str):
        """Select all blocks in a category (identity/style_composition/details). Clears others."""
        for key, var in self.extract_block_vars.items():
            var.set(False)
        if category == "style_composition":
            # double 0-7 + single 0-1 + single 2 (at full strength in Custom mode)
            for i in range(8):
                self.extract_block_vars[f"double_blocks.{i}"].set(True)
            for i in (0, 1, 2):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "identity":
            for i in range(1, 17):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "details":
            for i in range(12, 24):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)

    def _update_extract_output_name(self):
        """Regenerate the suggested output filename from source + preset + rank."""
        source = self.extract_source_var.get().strip()
        if not source:
            return
        base = os.path.splitext(os.path.basename(source))[0]
        preset_slug = self.extract_preset_var.get().lower().replace("+", "_").replace(" ", "_")
        self.extract_output_var.set(f"{base}_{preset_slug}_r{self.extract_rank_var.get()}.safetensors")

    def _browse_extract_source(self):
        filepath = filedialog.askopenfilename(
            title="Select source LoRA",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
            initialdir=self._lora_initialdir(),
        )
        if filepath:
            self.extract_source_var.set(filepath)
            self._update_extract_output_name()
            # Header-only family sniff — picking a LoRA from the wrong family auto-switches
            # instead of erroring at run time (same as Explorer / Profiler).
            try:
                from fizgig.networks.lora import lora_family_from_file
                fam = lora_family_from_file(filepath)
                if fam in ("klein", "krea2", "minimax") and fam != self.extract_family_var.get():
                    self.extract_family_var.set(fam)
                    self._on_extract_family_changed()
            except Exception:
                pass

    def _extract_log(self, text):
        """Append to extract log (preserves user scroll position). Marshals to the main
        thread — the extract worker calls this from its own thread."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._extract_log, text)
            return
        self._append_global_log(text)
        self._smart_text_insert(self.extract_log, text)

    def _open_extract_folder(self):
        if self._extract_output_path and os.path.exists(self._extract_output_path):
            folder = os.path.dirname(self._extract_output_path)
            self._open_in_file_manager(folder)

    def _run_extract(self):
        """Start extraction in a background thread."""
        if str(self.extract_family_var.get()) in ("krea2", "minimax"):
            self._run_extract_krea2()      # weight-only path; model-agnostic, serves H3 too
            return

        source = self.extract_source_var.get()
        if not source or not os.path.exists(source):
            messagebox.showerror("Error", "Please select a valid source LoRA.")
            return

        output_name = self.extract_output_var.get().strip()
        if not output_name:
            messagebox.showerror("Error", "Please enter an output name.")
            return
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"

        # Fast (weight-only, samples=0) presets run entirely from the safetensors file:
        # no pipeline is loaded, so don't demand a DiT/VAE/TE or a prompt they never use
        # (the CLI already worked without them).
        try:
            _samples = int(self.extract_samples_var.get().strip() or 0)
        except (ValueError, AttributeError):
            _samples = 0

        prompt = self.extract_prompt_var.get().strip()
        dit_path = self.prefs_vars["distilled_dit"].get()
        vae_path = self.prefs_vars["vae"].get()
        te_path = self.prefs_vars["text_encoder"].get()

        if _samples > 0:
            if not prompt:
                messagebox.showerror("Error", "Please enter a prompt (trigger word recommended).")
                return
            for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
                if not path or not os.path.exists(path):
                    messagebox.showerror("Error", f"{name} not found:\n{path}\n\nCheck Preferences tab.")
                    return

        # Build block list from preset (or custom individual blocks)
        preset = self.extract_preset_var.get()
        custom_blocks = None  # Per-block list for Custom mode
        if preset in ("Identity", "Fast Identity"):
            blocks = ["identity"]
        elif preset in ("Style", "Style+Composition", "Fast Style+Composition"):
            blocks = ["style_composition"]
        elif preset in ("Details", "Fast Details"):
            blocks = ["details"]
        elif preset in ("All Blocks", "Fast SVD"):
            blocks = ["all"]
        else:  # Custom
            selected = [key for key, var in self.extract_block_vars.items() if var.get()]
            if not selected:
                messagebox.showerror("Error", "Custom mode: please select at least one block.")
                return
            blocks = ["custom"]
            custom_blocks = selected

        output_dir = self.prefs_vars["lora_output_dir"].get()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        # Never silently overwrite: the name is built from source+preset+rank only, so two
        # different Custom selections at the same rank land on the same filename.
        if os.path.exists(output_path):
            _stem, _ext = os.path.splitext(output_path)
            _n = 2
            while os.path.exists(f"{_stem}_{_n}{_ext}"):
                _n += 1
            output_path = f"{_stem}_{_n}{_ext}"

        # Disable button, clear log
        self.extract_run_btn.configure(state="disabled")
        self.extract_open_btn.configure(state="disabled")
        self.extract_log.configure(state="normal")
        self.extract_log.delete(1.0, tk.END)
        self.extract_log.configure(state="disabled")
        self.extract_progress_var.set("Loading models...")

        import threading
        thread = threading.Thread(
            target=self._extract_worker,
            args=(source, output_path, dit_path, vae_path, te_path, blocks, prompt, custom_blocks),
            daemon=True,
        )
        thread.start()

    def _extract_worker(self, source, output_path, dit_path, vae_path, te_path, blocks, prompt, custom_blocks=None):
        """Background worker for extraction."""
        try:
            import sys
            sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

            from fizgig.extraction.extractor import LoRAExtractor, ExtractionConfig

            timestep_presets = {
                "all": (0.0, 1.0),
                "early": (0.6, 1.0),
                "mid": (0.3, 0.7),
                "late": (0.0, 0.4),
            }

            rank = int(self.extract_rank_var.get())
            samples = int(self.extract_samples_var.get())
            timesteps = timestep_presets[self.extract_timesteps_var.get()]

            config = ExtractionConfig(
                source_lora_path=source,
                output_lora_path=output_path,
                target_rank=rank,
                timestep_range=timesteps,
                include_blocks=blocks,
                custom_blocks=custom_blocks,
                num_samples=samples,
                prompt=prompt,
                width=1024,
                height=1024,
                seed=42,
            )

            def progress(stage, current, total):
                def _update():
                    self.extract_progress_var.set(f"{stage}: {current+1}/{total}")
                self.master.after(0, _update)

            pipeline = None
            if samples == 0:
                # Pure weight SVD — no pipeline needed, no GPU models loaded
                self.master.after(0, lambda: self._extract_log(f"Pure weight SVD: blocks={blocks}, rank={rank}\n"))
                result = LoRAExtractor.extract_weight_only(config, progress_callback=progress)
            else:
                # Activation-weighted SVD — needs full pipeline for forward passes
                from fizgig.klein.inference import KleinInferencePipeline

                # Auto-detect model version and fp8 from filename
                dit_basename = os.path.basename(dit_path).lower()
                model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
                is_fp8_model = "fp8" in dit_basename

                self.master.after(0, lambda: self._extract_log("Loading models...\n"))

                pipeline = KleinInferencePipeline()
                pipeline.load_models(
                    dit_path=dit_path,
                    vae_path=vae_path,
                    text_encoder_path=te_path,
                    model_version=model_version,
                    device="cuda",
                    fp8_scaled=not is_fp8_model,
                    fp8_text_encoder=True,
                    blocks_to_swap=self._get_inference_blocks_to_swap(),
                )

                self.master.after(0, lambda: self._extract_log(
                    f"Starting extraction: blocks={blocks}, rank={rank}, samples={samples}\n"))

                extractor = LoRAExtractor(pipeline)
                result = extractor.extract(config, progress_callback=progress)

            if pipeline is not None:
                pipeline.unload_models()

            # A valid-but-empty artifact is a failure, not a success: a Details preset on a
            # Style LoRA used to produce a ~340-byte tensorless file and "Extraction
            # complete! Layers extracted: 0".
            if result.num_layers_extracted == 0:
                try:
                    os.remove(result.output_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Extraction produced 0 layers — the selected blocks ({blocks}) don't "
                    f"exist in this LoRA. Run it through the Profiler to see which blocks "
                    f"it actually trains, then pick a matching preset.")

            summary = f"\nExtraction complete!\n"
            summary += f"  Output: {result.output_path}\n"
            summary += f"  Layers extracted: {result.num_layers_extracted}\n"
            summary += f"  Target rank: {result.target_rank}\n"
            summary += f"  Total params: {result.total_params:,}\n"
            summary += f"  Time: {result.elapsed_seconds:.1f}s\n"

            self._extract_output_path = output_path

            def _update_ui():
                self._extract_log(summary)
                self.extract_progress_var.set("Done!")
                self.extract_run_btn.configure(state="normal")
                self.extract_open_btn.configure(state="normal")

            self.master.after(0, _update_ui)

        except Exception as e:
            import traceback
            error_msg = f"Extraction failed:\n{traceback.format_exc()}"
            def _show_error():
                self._extract_log(error_msg)
                self.extract_progress_var.set("Error")
                self.extract_run_btn.configure(state="normal")
            self.master.after(0, _show_error)

    # --- Krea 2 extract (weight-only SVD over all blocks; no pipeline / prompt / block map) ---

    def _on_extract_family_changed(self):
        fam = str(self.extract_family_var.get())
        if fam not in ("klein", "krea2", "minimax"):
            fam = "klein"
        self.last_used["extract_family"] = fam
        self._save_last_used_paths()
        self._apply_extract_family_ui(fam != "klein")

    def _apply_extract_family_ui(self, is_krea2):
        """Krea 2 / MiniMax H3 mode: pure weight SVD over all blocks. Hide the block-preset,
        custom-block, prompt and activation-probe (timesteps + forward passes) controls — only
        Target Rank plus Source/Output/Run remain. Klein mode restores everything.
        (`is_krea2` is historical naming: True means any weight-only family.)"""
        # (widget, original padx) — restored verbatim so the klein branch is idempotent.
        probe_widgets = [
            (getattr(self, "_extract_timesteps_label", None), (0, 6)),
            (getattr(self, "_extract_timesteps_combo", None), (0, 20)),
            (getattr(self, "_extract_samples_label", None), (0, 6)),
            (getattr(self, "_extract_samples_combo", None), (0, 0)),
        ]
        if is_krea2:
            for c in (getattr(self, "_extract_preset_container", None),
                      getattr(self, "_extract_prompt_container", None)):
                if c is not None:
                    c.pack_forget()
            if getattr(self, "_extract_custom_frame", None) is not None:
                self._extract_custom_frame.pack_forget()
            for w, _ in probe_widgets:
                if w is not None:
                    w.pack_forget()
            # Force weight-only all-blocks regardless of stale Klein selections.
            self.extract_samples_var.set("0")
            self.extract_timesteps_var.set("all")
            if str(self.extract_family_var.get()) == "minimax":
                self.extract_time_note_var.set(
                    "MiniMax H3 is a 33B model - weight SVD runs over every trained module "
                    "(208+ Linears, up to 5376 wide). Expect several minutes on a free GPU. "
                    "If the GPU is busy (a training run, ComfyUI, another preview), each SVD "
                    "falls back to the CPU and runs much slower - free up VRAM first.")
                return
            self.extract_time_note_var.set(
                "⏱ Krea 2 is a 12.9B model — weight SVD runs over all 264 modules, several of "
                "them very large (e.g. 36864×6144). Expect roughly 5–10 minutes on a free GPU. "
                "If the GPU is busy (a training run, ComfyUI, another preview), each SVD falls back "
                "to the CPU and the whole run can take 60 min+ — free up VRAM first for the fast path.")
        else:
            anchor = getattr(self, "_extract_options_anchor", None)
            if getattr(self, "_extract_preset_container", None) is not None and anchor is not None:
                self._extract_preset_container.pack(fill=tk.X, padx=36, pady=(0, 16), before=anchor)
            run_anchor = getattr(self, "_extract_run_container", None)
            if getattr(self, "_extract_prompt_container", None) is not None and run_anchor is not None:
                self._extract_prompt_container.pack(fill=tk.X, padx=36, pady=(0, 16), before=run_anchor)
            # Re-pack probe widgets in their original order/padding after the rank combo.
            for w, padx in probe_widgets:
                if w is not None:
                    w.pack(side=tk.LEFT, padx=padx)
            # Custom-block frame visibility is owned by the preset combo; refresh it.
            if hasattr(self, "_on_extract_preset_changed"):
                self._on_extract_preset_changed()
            self.extract_time_note_var.set(
                "⏱ Fast SVD presets (weight-only) finish in well under a minute. Activation-weighted "
                "presets load the full pipeline and run probe forward passes — budget a few minutes. "
                "If the GPU is busy, SVD falls back to the CPU and runs much slower (a WARNING is logged).")

    def _run_extract_krea2(self):
        """Krea 2: pure weight SVD over all blocks — no pipeline, prompt, or block targeting."""
        source = self.extract_source_var.get()
        if not source or not os.path.exists(source):
            messagebox.showerror("Error", "Please select a valid source LoRA.")
            return

        output_name = self.extract_output_var.get().strip()
        if not output_name:
            messagebox.showerror("Error", "Please enter an output name.")
            return
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"

        output_dir = self.prefs_vars["lora_output_dir"].get()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        # Never silently overwrite (same rule as the Klein path).
        if os.path.exists(output_path):
            _stem, _ext = os.path.splitext(output_path)
            _n = 2
            while os.path.exists(f"{_stem}_{_n}{_ext}"):
                _n += 1
            output_path = f"{_stem}_{_n}{_ext}"

        self.extract_run_btn.configure(state="disabled")
        self.extract_open_btn.configure(state="disabled")
        self.extract_log.configure(state="normal")
        self.extract_log.delete(1.0, tk.END)
        self.extract_log.configure(state="disabled")
        self.extract_progress_var.set("Extracting...")

        import threading
        thread = threading.Thread(
            target=self._extract_worker_krea2,
            args=(source, output_path),
            daemon=True,
        )
        thread.start()

    def _extract_worker_krea2(self, source, output_path):
        """Background worker: weight-only SVD, model-agnostic (extract_weight_only with all blocks)."""
        try:
            import sys
            sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
            from fizgig.extraction.extractor import LoRAExtractor, ExtractionConfig

            rank = int(self.extract_rank_var.get())
            config = ExtractionConfig(
                source_lora_path=source,
                output_lora_path=output_path,
                target_rank=rank,
                timestep_range=(0.0, 1.0),
                include_blocks=["all"],
                custom_blocks=None,
                num_samples=0,
                prompt="",
                width=1024,
                height=1024,
                seed=42,
            )

            def progress(stage, current, total):
                self.master.after(0, lambda: self.extract_progress_var.set(f"{stage}: {current+1}/{total}"))

            _fam_label = ("MiniMax H3" if str(self.extract_family_var.get()) == "minimax"
                          else "Krea 2")
            self.master.after(0, lambda: self._extract_log(
                f"{_fam_label} weight-only SVD (all blocks), rank={rank}\n"))
            result = LoRAExtractor.extract_weight_only(config, progress_callback=progress)

            # Empty artifact = failure, not success (same rule as the Klein worker).
            if result.num_layers_extracted == 0:
                try:
                    os.remove(result.output_path)
                except OSError:
                    pass
                raise RuntimeError("Extraction produced 0 layers — the source file contains "
                                   "no LoRA modules this extractor recognises.")

            summary = (f"\nExtraction complete!\n"
                       f"  Output: {result.output_path}\n"
                       f"  Layers extracted: {result.num_layers_extracted}\n"
                       f"  Target rank: {result.target_rank}\n"
                       f"  Total params: {result.total_params:,}\n"
                       f"  Time: {result.elapsed_seconds:.1f}s\n")
            self._extract_output_path = output_path

            def _update_ui():
                self._extract_log(summary)
                self.extract_progress_var.set("Done!")
                self.extract_run_btn.configure(state="normal")
                self.extract_open_btn.configure(state="normal")
            self.master.after(0, _update_ui)

        except Exception:
            import traceback
            error_msg = f"Extraction failed:\n{traceback.format_exc()}"
            def _show_error():
                self._extract_log(error_msg)
                self.extract_progress_var.set("Error")
                self.extract_run_btn.configure(state="normal")
            self.master.after(0, _show_error)
