import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fizgig_gui.core.config.constants import FONT_FAMILY, COLORS
from fizgig_gui.core.config.prefs import save_prefs

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class StartTabMixin:
    def create_start_tab(self):
        """Welcome screen + Training image folder picker — the single source of truth shared with
        the Image Prep / Captions tabs and the Fizgig_train.toml auto-saver."""
        scrollable_frame, _ = self.create_scrollable_frame(self.start_tab)

        container = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        container.pack(fill=tk.BOTH, expand=True, padx=36, pady=(28, 0))

        # Title + subtitle
        tk.Label(container, text="Welcome to Fizgig",
                 font=(FONT_FAMILY, 22, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor=tk.W)
        tk.Label(container,
                 text="A focused, local trainer and workbench for Flux 2 Klein 9B, Krea 2 and "
                      "MiniMax H3 LoRAs — train, profile, repair, explore, and extract, all in "
                      "one place.",
                 font=(FONT_FAMILY, 11),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_deep"],
                 wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 24))

        # Workflow card
        workflow_card = tk.Frame(container, bg=COLORS["bg_surface"],
                                 highlightbackground=COLORS["border"],
                                 highlightthickness=1, bd=0)
        workflow_card.pack(fill=tk.X, pady=(0, 20))

        # Use grid on workflow_card: col 0 = title+steps, col 1 = logo (no padding)
        workflow_card.columnconfigure(0, weight=1)

        tk.Label(workflow_card, text="Training Workflow",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).grid(
            row=0, column=0, sticky=tk.W, padx=20, pady=(16, 10))

        steps_frame = tk.Frame(workflow_card, bg=COLORS["bg_surface"])
        steps_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=20, pady=(0, 16))

        # Logo — spans both rows, fills full card height, zero padding
        try:
            from PIL import Image as _PILImage, ImageTk as _PILImageTk
            _logo_path = os.path.join(_REPO_ROOT, "logo.jpg")
            if os.path.exists(_logo_path):
                _logo_pil_src = _PILImage.open(_logo_path)
                _logo_label = tk.Label(workflow_card, bg=COLORS["bg_surface"])
                _logo_label.grid(row=0, column=1, rowspan=2, sticky=tk.NS, padx=0, pady=0)

                def _fit_logo(_event=None, _src=_logo_pil_src, _lbl=_logo_label):
                    h = workflow_card.winfo_height()
                    if h < 20:
                        h = 260
                    w = int(_src.width * h / _src.height)
                    resized = _src.resize((w, h), _PILImage.LANCZOS)
                    self._start_logo_tk = _PILImageTk.PhotoImage(resized)
                    _lbl.configure(image=self._start_logo_tk)

                self.master.after(100, _fit_logo)
        except Exception:
            pass  # PIL not available or logo missing — skip silently

        steps = [
            ("1", "Start",      "Choose your training image folder below.",                     False),
            ("2", "Image Prep", "Resize, convert to PNG, or face-crop. (Video Prep for MiniMax)", True),  # optional
            ("3", "Captions",   "Write trigger-word captions or generate them with AI.",        False),
            ("4", "Samples",    "Configure in-training preview prompts.",                       False),
            ("5", "Training",   "Pick a preset, tune settings, click Start Training.",          False),
        ]

        for num, tab_name, desc, is_optional in steps:
            row = tk.Frame(steps_frame, bg=COLORS["bg_surface"])
            row.pack(fill=tk.X, pady=4)

            # Step number badge
            tk.Label(row, text=num,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["accent"], bg=COLORS["bg_surface"],
                     width=2).pack(side=tk.LEFT, padx=(0, 12))

            # Tab name
            tk.Label(row, text=tab_name,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
                     width=12, anchor="w").pack(side=tk.LEFT)

            # Optional badge
            if is_optional:
                tk.Label(row, text="OPTIONAL",
                         font=(FONT_FAMILY, 8, "bold"),
                         fg=COLORS["warning"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 10))

            # Description
            tk.Label(row, text=desc,
                     font=(FONT_FAMILY, 10),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                     anchor="w", justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Folder picker card
        picker_card = tk.Frame(container, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["accent"],
                               highlightthickness=1, bd=0)
        picker_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(picker_card, text="Training image folder",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(8, 2))
        tk.Label(picker_card,
                 text="This is the single place you set your dataset folder. Image Prep, Captions, "
                      "and Training all read from it automatically.",
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 12))

        row = tk.Frame(picker_card, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X, padx=20, pady=(0, 8))
        ttk.Entry(row, textvariable=self.image_folder_var, width=70, font=(FONT_FAMILY, 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=4
        )
        ttk.Button(row, text="Browse…", command=self._browse_image_folder).pack(side=tk.LEFT)

        # Setup prompt — shown when model paths are not configured yet
        self._setup_prompt_frame = tk.Frame(container, bg=COLORS["warning"],
                                             highlightbackground=COLORS["warning"],
                                             highlightthickness=1, bd=0)
        setup_inner = tk.Frame(self._setup_prompt_frame, bg="#2A2200")
        setup_inner.pack(fill=tk.X, padx=1, pady=1)
        tk.Label(setup_inner, text="\u26a0  Model files not configured",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["warning"], bg="#2A2200").pack(anchor=tk.W, padx=20, pady=(12, 4))
        tk.Label(setup_inner,
                 text="Head to the Preferences tab to set your model paths before training or using the tools. "
                      "Each model row has a Download link that opens the correct HuggingFace page.",
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_primary"], bg="#2A2200",
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 4))
        _setup_btn_row = tk.Frame(setup_inner, bg="#2A2200")
        _setup_btn_row.pack(anchor=tk.W, padx=20, pady=(4, 12))
        ttk.Button(_setup_btn_row, text="Open Preferences",
                   command=lambda: self.notebook.select(self.prefs_tab)).pack(side=tk.LEFT)

        def _dismiss_setup_prompt():
            # Permanent, by request: a Krea-only user never fills the Klein paths (or vice
            # versa, or skips the Turbo checkpoint entirely) and shouldn't be nagged forever.
            self.prefs["setup_prompt_dismissed"] = True
            save_prefs(self.prefs)
            self._setup_prompt_frame.pack_forget()

        _dismiss_lbl = tk.Label(_setup_btn_row, text="Don't show this again",
                                font=(FONT_FAMILY, 9, "underline"),
                                fg=COLORS["text_secondary"], bg="#2A2200", cursor="hand2")
        _dismiss_lbl.pack(side=tk.LEFT, padx=(16, 0))
        _dismiss_lbl.bind("<Button-1>", lambda e: _dismiss_setup_prompt())

        def _check_model_paths(*_args):
            # Hidden forever once dismissed; otherwise satisfied by EITHER family being
            # usable — Klein's four paths, or Krea 2's training trio (the Turbo checkpoint
            # is optional now that previews default to the Turbo LoRA).
            if self.prefs.get("setup_prompt_dismissed"):
                self._setup_prompt_frame.pack_forget()
                return
            klein_ok = all(self.prefs_vars[k].get().strip()
                           for k in ("base_dit", "distilled_dit", "vae", "text_encoder"))
            krea_ok = all(self.prefs_vars[k].get().strip()
                          for k in ("krea2_raw_dit", "krea2_vae", "krea2_text_encoder"))
            if klein_ok or krea_ok:
                self._setup_prompt_frame.pack_forget()
            else:
                self._setup_prompt_frame.pack(fill=tk.X, pady=(20, 0),
                                               before=tools_card)

        # Re-check whenever a model path (either family) changes
        for _mk in ("base_dit", "distilled_dit", "vae", "text_encoder",
                    "krea2_raw_dit", "krea2_vae", "krea2_text_encoder"):
            self.prefs_vars[_mk].trace_add("write", _check_model_paths)

        # Initial check (deferred so tools_card exists)
        self.master.after(100, _check_model_paths)

        def _check_minimax_extras():
            # An H3 user (DiT set) missing the NEW files — the Audio VAE and the Turbo LoRA
            # both arrived after most people set up their paths, so nothing else would ever
            # tell them these exist (Peter). One popup, dismissable forever.
            if self.prefs.get("minimax_extras_prompt_dismissed"):
                return
            if not str(self.prefs.get("minimax_dit", "") or "").strip():
                return
            missing = [label for key, label in (
                ("minimax_audio_vae",
                 "Audio VAE (~605 MB) — train on the sound in video clips, and on voices"),
                ("minimax_turbo_lora",
                 "Turbo LoRA (~780 MB) — fast 6-step in-training previews"))
                if not str(self.prefs.get(key, "") or "").strip()]
            if not missing:
                return
            win = tk.Toplevel(self.master)
            win.title("MiniMax H3 — new model files")
            win.configure(bg=COLORS["bg_deep"], padx=20, pady=16)
            win.transient(self.master)
            win.resizable(False, False)
            tk.Label(win, text="Your MiniMax H3 setup is missing the new files",
                     font=(FONT_FAMILY, 12, "bold"), bg=COLORS["bg_deep"],
                     fg=COLORS["text_primary"]).pack(anchor=tk.W)
            tk.Label(win, text="Fizgig can now train on video, sound and voices, and render "
                               "fast Turbo previews. Your H3 model paths are set, but these "
                               "are not:\n\n"
                               + "\n".join(f"  •  {m}" for m in missing)
                               + "\n\nPreferences has a download link on each row — or press "
                                 "Download models for me and point it at your models folder.",
                     font=(FONT_FAMILY, 10), justify=tk.LEFT, wraplength=520,
                     bg=COLORS["bg_deep"], fg=COLORS["text_explain"]).pack(
                anchor=tk.W, pady=(8, 12))
            row = tk.Frame(win, bg=COLORS["bg_deep"])
            row.pack(anchor=tk.E)

            def _to_prefs():
                win.destroy()
                self.notebook.select(self.prefs_tab)

            def _never():
                self.prefs["minimax_extras_prompt_dismissed"] = True
                save_prefs(self.prefs)
                win.destroy()

            ttk.Button(row, text="Open Preferences", command=_to_prefs).pack(side=tk.LEFT)
            ttk.Button(row, text="Later", command=win.destroy).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(row, text="Don't ask again", command=_never).pack(side=tk.LEFT,
                                                                         padx=(8, 0))
        self._check_minimax_extras = _check_minimax_extras     # bound for tests / re-checks
        self.master.after(700, _check_minimax_extras)
        # The audio-aware Training-tab rows (grey-outs, the voice-structure hint, the
        # per-category retirement row) refresh on folder-change traces — but the RESTORED
        # folder's trace fires during startup, before those widgets exist, and nothing
        # re-fires after the tab is built. One deferred pass covers the restored state.
        self.master.after(150, self._refresh_audio_only_ui)

        # Tools card — highlights the post-training workbench tabs
        tools_card = tk.Frame(container, bg=COLORS["bg_surface"],
                              highlightbackground=COLORS["border"],
                              highlightthickness=1, bd=0)
        tools_card.pack(fill=tk.X, pady=(20, 0))

        tk.Label(tools_card, text="Post-Training Tools",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(16, 2))
        tk.Label(tools_card,
                 text="Fizgig is more than a trainer — these tabs let you understand and tune any Klein LoRA "
                      "you've made (or downloaded).",
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 12))

        tools = [
            ("Profiler",
             "Analyze a LoRA's per-block activation profile and produce an HTML report."),
            ("Repair Studio",
             "Live per-block sliders with side-by-side preview. Blend in a donor LoRA and "
             "bake the result to a new .safetensors."),
            ("LoRA the Explorer",
             "Evolutionary discovery — the computer proposes random mutations, you pick favourites, "
             "and the LoRA evolves. Seamlessly connected to Repair Studio."),
            ("LoRA Royale",
             "Compare epochs (or any LoRAs) on one seed, then export share-ready clips — seed, "
             "prompt, and LoRA-strength travels, deflickered and ready for social."),
            ("Extract",
             "Distill a LoRA to a lower rank with optional block- and timestep-targeted presets. "
             "Supports LyCORIS (LoKR / LoHa) sources."),
        ]

        for name, desc in tools:
            row = tk.Frame(tools_card, bg=COLORS["bg_surface"])
            row.pack(fill=tk.X, padx=20, pady=4)

            tk.Label(row, text=name,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["accent"], bg=COLORS["bg_surface"],
                     width=16, anchor="w").pack(side=tk.LEFT, padx=(0, 12))
            tk.Label(row, text=desc,
                     font=(FONT_FAMILY, 10),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                     anchor="w", justify=tk.LEFT, wraplength=620).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Frame(tools_card, bg=COLORS["bg_surface"], height=12).pack()

        self._add_youtube_help_button(scrollable_frame, "start", prominent=True)

    def _browse_concept_folder(self):
        """Folder picker for the second Multi Concept subject.

        Deliberately NOT wired to image_folder_var: that one is the Start tab's, and Captions,
        Image Prep, the Look filter, the gallery scorer and the loss watch all follow it."""
        folder = filedialog.askdirectory(
            initialdir=(self._concept_folder_vars[0].get()
                        or self.image_folder_var.get()
                        or self._pref_initialdir("input_dataset_dir")
                        or os.getcwd()))
        if folder:
            self._concept_folder_vars[0].set(folder)

    def _sync_distill_weight_state(self):
        """Grey the teacher-weight box while identity-first is running the show.

        In that mode phase 1 is teacher-ONLY (weight forced to 1.0) and phase 2 is
        photographs-only, so the box changes nothing. Leaving it live invites people to tune a
        dial that is not connected to anything."""
        w = self.entries.get("MINIMAX_DISTILL_WEIGHT")
        p1 = self.entries.get("MINIMAX_DISTILL_PHASE1")
        if w is None or p1 is None:
            return
        _blended = str(p1.get()).startswith("Off")
        try:
            if _blended:
                w.config(state="readonly")
                # Hand back whatever the user had before identity-first took the dial over.
                _stash = getattr(self, "_distill_weight_stash", None)
                if _stash:
                    w.set(_stash)
                    self._distill_weight_stash = None
            else:
                # Show 1.0, because that is what phase 1 ACTUALLY runs at — leaving 0.8 sitting
                # there greyed out told the user something untrue about their own run. (Phase 2
                # then drops the teacher entirely; the log line says so at the switch.)
                if str(w.get()) != "1.0" and not getattr(self, "_distill_weight_stash", None):
                    self._distill_weight_stash = str(w.get())
                w.set("1.0")
                w.config(state="disabled")
        except tk.TclError:
            pass

    # The recipe Multi Concept switches you into (Peter, 11 Aug). Applied ONCE, when the box is
    # ticked — not locked. Locking caption dropout to a theory is what broke the last version of
    # this mode, so these are starting points the user can still argue with.
    # Deliberately NOT here: Adapter-relative LR. It is an LR strategy, and the two presets own
    # that choice (Defaults on, Fast flat) - a box describing how your DATA is laid out has no
    # business overruling the preset the user just loaded.
    _MULTICONCEPT_DEFAULTS = {
        "MINIMAX_CAPTION_DROPOUT": "0.10 (strong)",
        "MINIMAX_DISTILL_REFS": "4",
        "MINIMAX_DISTILL_PHASE1": "2 epochs",
    }

    def _warn_if_no_ref_dit(self):
        """Identity-learn runs on ref2va. Say so when it is switched ON, not at Start.

        validate_inputs already blocks the launch, but by then the user has captioned, cached
        and pressed Start - and the remedy is a 21 GB download, so an hour of setup can be spent
        before anything says the run cannot happen."""
        if self._krea2_pref("minimax_ref_dit"):
            return
        messagebox.showinfo(
            "Identity mode needs one more model",
            "Learning identity from your dataset runs on the ref2va model - a separate 21 GB "
            "file, and the only H3 build that accepts reference images. Fizgig does not "
            "download it by default.\n\n"
            "Preferences → MiniMax H3 → DiT (reference): paste the path if you "
            "already have the file, or tick \"Include the reference DiT\" beside \"Download "
            "models for me\" and let Fizgig fetch it.\n\n"
            "Carry on setting the run up either way - Start will stop and remind you if the "
            "path is still empty.")

    def _on_minimax_distill_clicked(self):
        """Only on a real click: setting the var programmatically must stay silent."""
        if self.minimax_distill_var.get():
            self._warn_if_no_ref_dit()

    def _on_minimax_multiconcept_clicked(self):
        """User CLICKED the box — apply the recipe, then refresh the rows.

        Separate from _on_minimax_multiconcept_toggle because that one also runs on an
        architecture switch and on every preset load; applying the recipe there would silently
        overwrite settings the user had changed, every time they visited the tab."""
        if self.minimax_multiconcept_var.get():
            _changed = []
            for _k, _v in self._MULTICONCEPT_DEFAULTS.items():
                _w = self.entries.get(_k)
                if _w is not None and str(_w.get()) != _v:
                    _w.set(_v)
                    _changed.append(f"{_k.replace('MINIMAX_', '').lower()}={_v}")
            if hasattr(self, "minimax_distill_var") and not self.minimax_distill_var.get():
                self.minimax_distill_var.set(True)
                _changed.append("identity-learn=on")
            if _changed:
                self.update_console("[multi concept] applied: " + ", ".join(_changed)
                                    + "  (all still editable)\n")
        self._on_minimax_multiconcept_toggle()
        self._sync_distill_weight_state()
        if self.minimax_multiconcept_var.get() and self.minimax_distill_var.get():
            self._warn_if_no_ref_dit()

    def _on_minimax_multiconcept_toggle(self):
        """Show the extra folder row. Caption dropout is deliberately NOT touched.

        It used to be forced off here, on the theory that training a few percent of steps against
        the EMPTY prompt teaches the model to produce a subject with no trigger — the mechanism by
        which two subjects bleed. Peter's own A/B said otherwise (11 Aug): with distillation off,
        one folder WITH dropout beat two folders without it. Whatever dropout costs in bleed, it
        appears to be worth more as regularisation at this scale. The dial goes back to the user
        rather than being locked to a theory the data does not support."""
        on = bool(self.minimax_multiconcept_var.get()) and self._is_minimax_arch()
        for w in (getattr(self, "_minimax_mc_dir_frame", None),
                  getattr(self, "_minimax_mc_hint", None)):
            if w is not None:
                self._set_widget_visible(w, on)
        # The "no reference steering" warning only makes sense in multi-concept with distill off.
        _nd = getattr(self, "_minimax_mc_nodistill_hint", None)
        if _nd is not None:
            _distill = bool(getattr(self, "minimax_distill_var", None)
                            and self.minimax_distill_var.get())
            self._set_widget_visible(_nd, on and not _distill)

    def _browse_image_folder(self):
        """Folder picker for the Start tab (unified image folder).

        Falls back through: the folder you last used, then the Preferences default (which the pod
        image seeds to /workspace/datasets so Browse opens where uploads land), then cwd."""
        folder = filedialog.askdirectory(
            initialdir=(self.image_folder_var.get()
                        or self._pref_initialdir("input_dataset_dir")
                        or os.getcwd()))
        if folder:
            self.image_folder_var.set(folder)
