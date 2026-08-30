import os

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY
from fizgig_gui.core.config.last_used import OUTPUT_LORAS_DIR

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class ProfilerTabMixin:
    def create_profiler_tab(self):
        """Create the LoRA Profiler tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.profiler_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Profiler",
            "Analyze a LoRA's per-block signature. Klein: full activation profile (5-bucket report). "
            "Krea 2 and MiniMax H3: weight-only profile (flat per-block — no block-role map yet). "
            "All write a sidecar the Repair Studio reads inline.",
        )

        # Model family selector. Krea 2 and MiniMax H3 are weight-only profiles — no pipeline,
        # prompt, resolution or stages — so those cards are hidden for both.
        _pfam = str(self.last_used.get("profiler_family", "klein"))
        if _pfam not in ("klein", "krea2", "minimax"):
            _pfam = "klein"
        self.profiler_family_var = tk.StringVar(value=_pfam)
        fam_card = self._start_section_card(
            outer, "Model Family",
            "Klein 9B (activation profile), Krea 2 or MiniMax H3 (weight-only — the instrument to "
            "discover each family's block roles). Browsing a LoRA auto-switches to its family.",
        )
        _pf = tk.Frame(fam_card, bg=COLORS["bg_surface"])
        _pf.pack(anchor=tk.W)
        ttk.Radiobutton(_pf, text="Klein 9B", variable=self.profiler_family_var, value="klein",
                        command=self._on_profiler_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_pf, text="Krea 2", variable=self.profiler_family_var, value="krea2",
                        command=self._on_profiler_family_changed).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(_pf, text="MiniMax H3", variable=self.profiler_family_var, value="minimax",
                        command=self._on_profiler_family_changed).pack(side=tk.LEFT)

        # Card 1: Model selection
        model_card = self._start_section_card(
            outer, "Model",
            "Paths are set on the Preferences tab. Distilled is a few seconds per probe and fine for most scans; "
            "Base produces the authoritative report but is slower.",
        )
        self._profiler_model_container = model_card.master.master
        self.profiler_dit_choice_var = tk.StringVar(value="distilled")
        dit_choice_frame = tk.Frame(model_card, bg=COLORS["bg_surface"])
        dit_choice_frame.pack(anchor=tk.W)
        ttk.Radiobutton(dit_choice_frame, text="Distilled (fast, ~4-step probes)",
                        variable=self.profiler_dit_choice_var, value="distilled").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(dit_choice_frame, text="Base (precise)",
                        variable=self.profiler_dit_choice_var, value="base").pack(side=tk.LEFT)

        # Card 2: LoRA
        lora_card = self._start_section_card(
            outer, "LoRA File",
            "Select the LoRA you want to profile. PEFT and LyCORIS (LoKR / LoHa) are auto-converted on load.",
        )
        self._profiler_lora_container = lora_card.master.master
        lora_card.grid_columnconfigure(1, weight=1)
        ttk.Label(lora_card, text="LoRA File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.profiler_lora_var = tk.StringVar()
        ttk.Entry(lora_card, textvariable=self.profiler_lora_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(lora_card, text="Browse", command=self._browse_profiler_lora).grid(row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4)

        # Card 3: Prompt
        prompt_card = self._start_section_card(
            outer, "Prompt",
            "Include the LoRA's trigger word so the profile captures its active pathways, e.g.: "
            "`zwxem, a portrait photo of a woman`.",
        )
        self._profiler_prompt_container = prompt_card.master.master
        prompt_card.grid_columnconfigure(1, weight=1)
        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.profiler_prompt_var = tk.StringVar(value="")
        ttk.Entry(prompt_card, textvariable=self.profiler_prompt_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)

        # Card 4: Options
        options_card = self._start_section_card(
            outer, "Options",
            "Resolution controls the probe render size; Stages is the number of denoising buckets the profiler measures.",
        )
        self._profiler_options_container = options_card.master.master
        options_row = tk.Frame(options_card, bg=COLORS["bg_surface"])
        options_row.pack(anchor=tk.W)

        ttk.Label(options_row, text="Resolution:").pack(side=tk.LEFT, padx=(0, 6))
        self.profiler_res_var = tk.StringVar(value="1024")
        ttk.Combobox(options_row, textvariable=self.profiler_res_var,
                     values=["512", "768", "1024"], state="readonly", width=6).pack(side=tk.LEFT, padx=(0, 24))

        ttk.Label(options_row, text="Stages:").pack(side=tk.LEFT, padx=(0, 6))
        self.profiler_stages_var = tk.StringVar(value="5")
        ttk.Combobox(options_row, textvariable=self.profiler_stages_var,
                     values=["3", "5", "8", "10"], state="readonly", width=4).pack(side=tk.LEFT)

        # Card 5: Run
        run_card = self._start_section_card(outer, "Run", None)
        self._profiler_run_container = run_card.master.master
        run_row = tk.Frame(run_card, bg=COLORS["bg_surface"])
        run_row.pack(anchor=tk.W)
        self.profiler_run_btn = ttk.Button(run_row, text="Profile LoRA", command=self._run_profiler, style="Primary.TButton")
        self.profiler_run_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.profiler_open_btn = ttk.Button(run_row, text="Open Report", command=self._open_profiler_report, state="disabled")
        self.profiler_open_btn.pack(side=tk.LEFT)

        self.profiler_progress_var = tk.StringVar(value="")
        tk.Label(run_card, textvariable=self.profiler_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(10, 0))

        # Card 6: Results
        results_card = self._start_section_card(
            outer, "Results",
            "Summary text lands here during profiling; the full heat-mapped report opens in your browser via Open Report.",
        )
        self.profiler_results = scrolledtext.ScrolledText(
            results_card, height=18, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.profiler_results.pack(fill=tk.BOTH, expand=True)

        self._profiler_report_path = None

        self._add_youtube_help_button(outer, "profiler")
        self._apply_profiler_family_ui()

    def _apply_profiler_family_ui(self):
        """Krea 2 / MiniMax H3 profiling is weight-only — hide the activation-probe cards
        (Model/Prompt/Options). Re-show (Klein) uses before= anchors so the cards land back
        in their canonical order."""
        krea2 = (self.profiler_family_var.get() in ("krea2", "minimax"))

        def _show(cont, before):
            try:
                if cont is not None and cont.winfo_manager() == "":
                    if before is not None and before.winfo_manager() == "pack":
                        cont.pack(fill=tk.X, padx=36, pady=(0, 16), before=before)
                    else:
                        cont.pack(fill=tk.X, padx=36, pady=(0, 16))
            except Exception:
                pass

        if krea2:
            for cont in (getattr(self, "_profiler_model_container", None),
                         getattr(self, "_profiler_prompt_container", None),
                         getattr(self, "_profiler_options_container", None)):
                try:
                    if cont is not None:
                        cont.pack_forget()
                except Exception:
                    pass
        else:
            # Order matters: options before run, prompt before options, model before lora.
            _show(getattr(self, "_profiler_options_container", None), getattr(self, "_profiler_run_container", None))
            _show(getattr(self, "_profiler_prompt_container", None), getattr(self, "_profiler_options_container", None))
            _show(getattr(self, "_profiler_model_container", None), getattr(self, "_profiler_lora_container", None))

    def _on_profiler_family_changed(self):
        self._apply_profiler_family_ui()
        try:
            self.last_used["profiler_family"] = self.profiler_family_var.get()
            self._save_last_used_paths()
        except Exception:
            pass

    def _browse_profiler_lora(self):
        filepath = filedialog.askopenfilename(
            title="Select LoRA file",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
            initialdir=self._lora_initialdir(),
        )
        if filepath:
            self.profiler_lora_var.set(filepath)
            # Same auto-switch the Explorer does: a header-only sniff, so picking a LoRA from
            # the wrong family Just Works instead of erroring at run time.
            try:
                from fizgig.networks.lora import lora_family_from_file
                fam = lora_family_from_file(filepath)
                if fam in ("klein", "krea2", "minimax") and fam != self.profiler_family_var.get():
                    self.profiler_family_var.set(fam)
                    self._on_profiler_family_changed()
            except Exception:
                pass

    def _browse_profiler_file(self, var):
        filepath = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if filepath:
            var.set(filepath)

    def _open_profiler_report(self):
        if self._profiler_report_path and os.path.exists(self._profiler_report_path):
            import webbrowser
            webbrowser.open(self._profiler_report_path)

    def _profiler_log(self, text):
        """Append to profiler log (preserves user scroll position). Marshals to the main
        thread — the profiler worker calls this from its own thread."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._profiler_log, text)
            return
        self._append_global_log(text)
        self._smart_text_insert(self.profiler_results, text)

    def _run_profiler(self):
        """Start profiling in a background thread."""
        lora_path = self.profiler_lora_var.get()
        if not lora_path or not os.path.exists(lora_path):
            messagebox.showerror("Error", "Please select a valid LoRA file.")
            return

        if self.profiler_family_var.get() == "krea2":
            return self._run_profiler_krea2(lora_path)
        if self.profiler_family_var.get() == "minimax":
            return self._run_profiler_h3(lora_path)

        prompt = self.profiler_prompt_var.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt (include the LoRA trigger word).")
            return

        # Resolve model paths from Preferences (single source of truth)
        dit_choice = self.profiler_dit_choice_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        if not dit_path:
            messagebox.showerror(
                "Error",
                f"{dit_choice.capitalize()} DiT path not set.\nConfigure it on the Preferences tab.",
            )
            return
        if not vae_path or not te_path:
            messagebox.showerror("Error", "VAE and Text Encoder paths not set.\nConfigure them on the Preferences tab.")
            return

        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
            if not os.path.exists(path):
                messagebox.showerror("Error", f"{name} file not found:\n{path}")
                return

        res = int(self.profiler_res_var.get())
        stages = int(self.profiler_stages_var.get())

        # Disable button during profiling
        self.profiler_run_btn.configure(state="disabled")
        self.profiler_open_btn.configure(state="disabled")
        self.profiler_results.configure(state="normal")
        self.profiler_results.delete(1.0, tk.END)
        self.profiler_results.configure(state="disabled")
        self.profiler_progress_var.set("Loading models...")

        import threading
        thread = threading.Thread(
            target=self._profiler_worker,
            args=(lora_path, prompt, dit_path, vae_path, te_path, res, stages),
            daemon=True,
        )
        thread.start()

    def _run_profiler_krea2(self, lora_path):
        """Krea 2 weight-only profile — no pipeline, fast. Runs in a thread (LoRA load + norms)."""
        import threading
        self.profiler_run_btn.configure(state="disabled")
        self.profiler_open_btn.configure(state="disabled")
        self.profiler_results.configure(state="normal")
        self.profiler_results.delete(1.0, tk.END)
        self.profiler_results.configure(state="disabled")
        self.profiler_progress_var.set("Profiling (Krea 2, weight-only)…")

        def worker():
            try:
                import sys
                sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
                from fizgig.profiler.krea2_profile import profile_krea2_weight_only
                profiles_dir = (self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars
                                else os.path.join(OUTPUT_LORAS_DIR, "profiles"))
                os.makedirs(profiles_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(lora_path))[0]
                out_html = os.path.join(profiles_dir, f"{stem}_krea2_profile.html")
                html, sidecar = profile_krea2_weight_only(lora_path, out_html, profiles_dir=profiles_dir)
                self._profiler_report_path = html
                self.master.after(0, lambda: self._profiler_krea2_done(html, sidecar))
            except Exception:
                import traceback
                err = traceback.format_exc()
                def _fail():
                    self._profiler_log(err + "\n")
                    self.profiler_progress_var.set("Error — see results.")
                    self.profiler_run_btn.configure(state="normal")
                self.master.after(0, _fail)
        threading.Thread(target=worker, daemon=True).start()

    def _run_profiler_h3(self, lora_path):
        """MiniMax H3 weight-only profile — no pipeline, fast. Runs in a thread."""
        import threading
        self.profiler_run_btn.configure(state="disabled")
        self.profiler_open_btn.configure(state="disabled")
        self.profiler_results.configure(state="normal")
        self.profiler_results.delete(1.0, tk.END)
        self.profiler_results.configure(state="disabled")
        self.profiler_progress_var.set("Profiling (MiniMax H3, weight-only)…")

        def worker():
            try:
                import sys
                sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
                from fizgig.profiler.h3_profile import profile_h3_weight_only
                profiles_dir = (self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars
                                else os.path.join(OUTPUT_LORAS_DIR, "profiles"))
                os.makedirs(profiles_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(lora_path))[0]
                out_html = os.path.join(profiles_dir, f"{stem}_h3_profile.html")
                html, sidecar = profile_h3_weight_only(lora_path, out_html, profiles_dir=profiles_dir)
                self._profiler_report_path = html
                self.master.after(0, lambda: self._profiler_h3_done(html, sidecar))
            except Exception:
                import traceback
                err = traceback.format_exc()
                def _fail():
                    self._profiler_log(err + "\n")
                    self.profiler_progress_var.set("Error — see results.")
                    self.profiler_run_btn.configure(state="normal")
                self.master.after(0, _fail)
        threading.Thread(target=worker, daemon=True).start()

    def _profiler_h3_done(self, html, sidecar):
        try:
            import json as _json
            d = _json.load(open(sidecar, encoding="utf-8"))
            lines = ["MiniMax H3 weight-only profile complete.\n",
                     f"Report: {html}\n\nTop blocks by weight:\n"]
            for b in d.get("top_active_blocks", []):
                lines.append(f"  {b['name']:<12} {b['pct']:.1f}%\n")
            lines.append("\nH3's 50 block roles aren't mapped yet — this is the weight signature "
                         "to discover them. Found a pattern? Share it on GitHub Issues.\n")
            self._profiler_log("".join(lines))
        except Exception:
            self._profiler_log(f"Profile complete: {html}\n")
        self.profiler_progress_var.set("Done.")
        self.profiler_run_btn.configure(state="normal")
        self.profiler_open_btn.configure(state="normal")

    def _profiler_krea2_done(self, html, sidecar):
        try:
            import json as _json
            d = _json.load(open(sidecar, encoding="utf-8"))
            lines = ["Krea 2 weight-only profile complete.\n",
                     f"Report: {html}\n\nTop blocks by weight:\n"]
            for b in d.get("top_active_blocks", []):
                lines.append(f"  {b['name']:<12} {b['pct']:.1f}%\n")
            lines.append("\nNo semantic block map yet — this is the weight signature to discover it.\n")
            self._profiler_log("".join(lines))
        except Exception:
            self._profiler_log(f"Profile complete: {html}\n")
        self.profiler_progress_var.set("Done.")
        self.profiler_run_btn.configure(state="normal")
        self.profiler_open_btn.configure(state="normal")

    def _profiler_worker(self, lora_path, prompt, dit_path, vae_path, te_path, res, stages):
        """Background worker for profiling."""
        try:
            import sys
            sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

            from fizgig.klein.inference import KleinInferencePipeline
            from fizgig.profiler.profiler import LoRAProfiler
            from fizgig.profiler.visualize import plot_profile_heatmap, print_profile_summary

            self.master.after(0, lambda: self._profiler_log("Loading models...\n"))

            # Auto-detect model version and fp8 from filename
            dit_basename = os.path.basename(dit_path).lower()
            model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
            is_fp8_model = "fp8" in dit_basename

            pipeline = KleinInferencePipeline()
            pipeline.load_models(
                dit_path=dit_path,
                vae_path=vae_path,
                text_encoder_path=te_path,
                model_version=model_version,
                device="cuda",
                fp8_scaled=False if is_fp8_model else True,  # Don't apply fp8_scaled to already-fp8 models
                fp8_text_encoder=self.settings.get("FP8_TEXT_ENCODER", True),
                blocks_to_swap=self._get_inference_blocks_to_swap(),
            )

            self.master.after(0, lambda: self._profiler_log("Models loaded. Starting profiling...\n"))
            self.master.after(0, lambda: self.profiler_progress_var.set(f"Profiling: 0 of {stages} stages..."))

            profiler = LoRAProfiler(pipeline)

            # Patch the profiler to report progress
            original_profile = profiler.profile
            _self = self

            result = profiler.profile(
                lora_path=lora_path,
                num_samples=4,
                num_bins=stages,
                width=res,
                height=res,
                prompt=prompt,
                seed=42,
            )

            self.master.after(0, lambda: self.profiler_progress_var.set("Generating report..."))

            # Save report into the dedicated Profiles directory from prefs
            # (defaults to FizgigIndependent/profiles/).
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            profiles_dir = self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars else os.path.join(OUTPUT_LORAS_DIR, "profiles")
            os.makedirs(profiles_dir, exist_ok=True)
            report_path = os.path.join(profiles_dir, f"{lora_name}_profile.html")

            plot_profile_heatmap(result, report_path)
            self._profiler_report_path = report_path

            # Build summary text
            from fizgig.profiler.visualize import _category_totals, _short_name, _get_category
            cat_totals = _category_totals(result)
            grand_total = sum(cat_totals.values()) or 1.0

            summary = f"LoRA Profile: {os.path.basename(lora_path)}\n"
            summary += f"Prompt: {prompt}\n"
            summary += f"Resolution: {res}x{res}\n\n"
            summary += f"Category Breakdown:\n"
            summary += f"  Style+Composition:        {cat_totals['style_composition']/grand_total*100:5.1f}%  (double 0-7 + single 0)\n"
            summary += f"  ↔ style↔identity:         {cat_totals['style_ident_overlap']/grand_total*100:5.1f}%  (single 1)\n"
            summary += f"  Identity:                 {cat_totals['identity']/grand_total*100:5.1f}%  (single 2-11)\n"
            summary += f"  ↔ identity↔details:       {cat_totals['ident_details_overlap']/grand_total*100:5.1f}%  (single 12-16)\n"
            summary += f"  Details:                  {cat_totals['details']/grand_total*100:5.1f}%  (single 17-23)\n\n"
            summary += f"Most Active Blocks:\n"
            for name, total in result.get_top_blocks(10):
                cat = _get_category(name)
                pct = total / grand_total * 100
                summary += f"  {_short_name(name):14s}  {pct:5.1f}%  [{cat}]\n"
            summary += f"\nReport saved: {report_path}\n"

            pipeline.unload_models()

            def _update_ui():
                self._profiler_log(summary)
                self.profiler_progress_var.set("Done!")
                self.profiler_run_btn.configure(state="normal")
                self.profiler_open_btn.configure(state="normal")

            self.master.after(0, _update_ui)

        except Exception as e:
            import traceback
            error_msg = f"Profiling failed:\n{traceback.format_exc()}"
            def _show_error():
                self._profiler_log(error_msg)
                self.profiler_progress_var.set("Error")
                self.profiler_run_btn.configure(state="normal")
            self.master.after(0, _show_error)
