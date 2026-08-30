import json
import os
import socket
import threading
import time
import webbrowser

import tkinter as tk
from http.server import SimpleHTTPRequestHandler, HTTPServer
from tkinter import ttk

from face_utils import FaceEmbedder
from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, _REPO_ROOT, SAMPLE_RESOLUTIONS, FG_COLOR, FACE_DETECTION_AVAILABLE
from fizgig_gui.core.domain.architectures import ARCHITECTURES, ARCHITECTURE_LIST, LORA_NAME_SUFFIXES
from fizgig_gui.core.ui_base.widgets import ToolTip


class SamplesTabMixin:
    def create_samples_settings(self):
        """Create the Samples tab with sample generation settings (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.samples_tab)

        # Outer bg_deep container so the card stack sits on a consistent background.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Subtitle kept for reconfiguring per family — the "(Distilled 4-step)" parenthetical is
        # Klein's preview stack and means nothing on the other two.
        self._samples_banner = self._add_tab_banner(
            outer,
            "Sample Previews",
            "Preview prompts rendered periodically during training (Distilled 4-step). "
            "Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.",
        )
        _bkids = self._samples_banner.winfo_children()
        self._samples_banner_sub = _bkids[1] if len(_bkids) > 1 else None

        # === Base Model chooser (Peter) — the tab's options are family-shaped (Sample length
        # with sound, Turbo pace are MiniMax-only), so choose the family HERE rather than
        # round-tripping to the Training tab. Same StringVar as the Training-tab combobox, so
        # the two can never disagree; the bind is required because architecture changes ride
        # <<ComboboxSelected>>, not a var trace. Sits OUTSIDE sample_settings_frame on purpose:
        # it must survive the master-enable toggle and the enable/disable widget walk.
        if len(ARCHITECTURE_LIST) > 1:
            arch_card = self._start_section_card(
                outer, "Base Model",
                "Pick what you're training — the sample options below match it. Same setting "
                "as the Training tab.",
            )
            arch_row = tk.Frame(arch_card, bg=COLORS["bg_surface"])
            arch_row.pack(anchor=tk.W)
            tk.Label(
                arch_row, text="Model:",
                font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            ).pack(side=tk.LEFT, padx=(0, 8))
            samples_arch_combo = ttk.Combobox(
                arch_row, textvariable=self.architecture_var, state="readonly",
                width=28, values=ARCHITECTURE_LIST,
            )
            samples_arch_combo.pack(side=tk.LEFT)
            samples_arch_combo.bind("<<ComboboxSelected>>", self._on_architecture_selected)
            ToolTip(samples_arch_combo, "Model family to train (Klein 9B, Krea 2 or MiniMax H3)")
            self._samples_arch_combo = samples_arch_combo

        # Grid holder — video warning / master checkbox / settings block all row-managed
        # so update_samples_ui_for_architecture() can still .grid() / .grid_remove() them.
        grid_holder = tk.Frame(outer, bg=COLORS["bg_deep"])
        grid_holder.pack(fill=tk.X)
        grid_holder.grid_columnconfigure(0, weight=1)

        # --- Video model warning (hidden by default; grid_remove'd at the end) ---
        self.video_model_warning_frame = ttk.Frame(grid_holder)
        self.video_model_warning_frame.grid(row=0, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        ttk.Label(
            self.video_model_warning_frame,
            text="Sample generation is not available for video models (t2v, i2v).",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(
            self.video_model_warning_frame,
            text="Video sampling during training is too slow and memory-intensive.",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 15))
        video_viewer_frame = ttk.Frame(self.video_model_warning_frame)
        video_viewer_frame.pack(anchor=tk.W, pady=10)
        ttk.Button(video_viewer_frame, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_viewer_frame, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT, padx=5)
        self.video_model_warning_frame.grid_remove()

        # --- Master Enable card ---
        self.sample_enabled_var = tk.BooleanVar(value=self.settings["SAMPLE_ENABLED"])
        enable_card_outer = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        enable_card_outer.grid(row=1, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        enable_card = tk.Frame(enable_card_outer, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["border"], highlightthickness=1, bd=0)
        enable_card.pack(fill=tk.X)
        # Use ttk.Checkbutton — themed; sits on bg_surface via style inheritance
        self.sample_enabled_check = ttk.Checkbutton(
            enable_card, text="Enable Sample Generation", variable=self.sample_enabled_var,
            command=self.toggle_sample_settings,
        )
        self.sample_enabled_check.pack(anchor=tk.W, padx=20, pady=14)


        # --- Sample settings container (the 4 cards live inside this) ---
        self.sample_settings_frame = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        self.sample_settings_frame.grid(row=2, column=0, sticky=tk.EW)

        # Card 1: Prompt & Dimensions
        prompt_card = self._start_section_card(
            self.sample_settings_frame, "Prompt & Dimensions",
            "One prompt per line — every line renders its own sample each epoch. "
            "Output size, steps and seed apply to all of them.",
        )
        prompt_card.grid_columnconfigure(1, weight=1)

        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.NW, padx=(0, 10), pady=4)
        self.sample_prompt_text = tk.Text(
            prompt_card, height=3, width=50, bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"], font=(FONT_FAMILY, 10), wrap="word",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.sample_prompt_text.insert("1.0", self.last_used.get("sample_prompt", self.settings["SAMPLE_PROMPT"]))
        self.sample_prompt_text.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self.sample_prompt_text.bind("<KeyRelease>", lambda e: self._save_last_used_paths())
        # Issue #49: "Multi-line prompt" read as ONE prompt that may contain line breaks — two
        # users only discovered multiple prompts by accident. Say what a line actually does.
        tk.Label(prompt_card,
                 text="Each line is a SEPARATE prompt — press Enter to add another sample per "
                      "epoch. Keep a single prompt on one line (long ones wrap by themselves).",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=520, justify=tk.LEFT).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 6))

        ttk.Label(prompt_card, text="Width:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_width_var = tk.StringVar(value=str(self.settings["SAMPLE_WIDTH"]))
        self.sample_width_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_width_var,
            values=SAMPLE_RESOLUTIONS, state="readonly", width=10,
        )
        self.sample_width_combo.grid(row=2, column=1, sticky=tk.W, pady=4)

        ttk.Label(prompt_card, text="Height:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_height_var = tk.StringVar(value=str(self.settings["SAMPLE_HEIGHT"]))
        self.sample_height_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_height_var,
            values=SAMPLE_RESOLUTIONS, state="readonly", width=10,
        )
        self.sample_height_combo.grid(row=3, column=1, sticky=tk.W, pady=4)

        self.sample_steps_label = ttk.Label(prompt_card, text="Steps:")
        self.sample_steps_label.grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_STEPS"]))
        _steps_frame = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        _steps_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_steps_entry = ttk.Entry(_steps_frame, textvariable=self.sample_steps_var, width=10)
        self.sample_steps_entry.pack(side=tk.LEFT)
        self.sample_steps_note = tk.Label(_steps_frame, text="Base samples only — Distilled is locked at 4 steps",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.sample_steps_note.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(prompt_card, text="Seed:").grid(row=5, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_seed_var = tk.StringVar(value=str(self.settings["SAMPLE_SEED"]))
        self.sample_seed_entry = ttk.Entry(prompt_card, textvariable=self.sample_seed_var, width=10)
        self.sample_seed_entry.grid(row=5, column=1, sticky=tk.W, pady=4)
        tk.Label(prompt_card, text="(0 = random)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=5, column=2, sticky=tk.W, padx=(10, 0)
        )

        # Reference image (Klein edit conditioning) — the persistent default for
        # samples; the status-bar override can swap it live mid-run.
        self.sample_ref_label = ttk.Label(prompt_card, text="Reference:")
        self.sample_ref_label.grid(row=6, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_ref_image_var = tk.StringVar(value=self.last_used.get("sample_ref_image", ""))
        _ref_row = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        _ref_row.grid(row=6, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self._sample_ref_row = _ref_row        # hidden wholesale for MiniMax (no ref path)
        self.sample_ref_entry = ttk.Entry(_ref_row, textvariable=self.sample_ref_image_var, state="readonly")
        self.sample_ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sample_ref_browse_btn = ttk.Button(_ref_row, text="Browse…", command=self._browse_sample_ref)
        self.sample_ref_browse_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.sample_ref_clear_btn = ttk.Button(_ref_row, text="Clear", command=lambda: self.sample_ref_image_var.set(""))
        self.sample_ref_clear_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.sample_ref_image_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self.sample_ref_note = tk.Label(prompt_card,
                 text="Optional — Klein is an edit model, so samples can be conditioned on a real image (they edit "
                      "it rather than generate from scratch). Auto-resized to ~0.20 MP so any size is safe. Leave "
                      "empty for normal samples.",
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=560, justify=tk.LEFT)
        self.sample_ref_note.grid(row=7, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))

        # --- Sample length (MiniMax only) — still vs scrubbable clip -----------------------
        # H3 is a video model whose trained range is ~124-362 frames; a single still is out of
        # distribution and previews look worse than the same LoRA rendered as video in ComfyUI.
        # Shown/hidden by update_samples_ui_for_architecture.
        self.sample_frames_label = ttk.Label(prompt_card, text="Sample length:")
        self.sample_frames_label.grid(row=8, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        # Default: 56-FRAME CLIP WITH SOUND (Peter, 17 Aug — reversing the 11 Aug stills
        # call). What changed: the Turbo makes a 6-step clip render affordable, the sampler's
        # audio is finally trustworthy, and a clip IS the regime H3 was trained in — a
        # picture-and-sound preview is now the honest default heartbeat. Without the audio
        # VAE set it degrades to a silent clip with a console note; Still stays in the
        # dropdown for anyone who wants seconds-per-preview.
        self.sample_frames_var = tk.StringVar(
            value=self.last_used.get("sample_frames", "56 frames with sound (~2.3s)"))
        self.sample_frames_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_frames_var, state="readonly", width=34,
            values=["Still (1 frame)", "22 frames (~1s)", "56 frames (~2.3s)",
                    "124 frames (~5s — trained minimum)", "141 frames (~6s)",
                    "22 frames with sound (~1s)",
                    "56 frames with sound (~2.3s)",
                    "124 frames with sound (~5s)"])
        self.sample_frames_combo.grid(row=8, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_frames_var.trace_add("write", lambda *a: self._save_last_used_paths())
        self._sample_frames_hint = tk.Label(prompt_card,
                 text="Samples render as short clips you can scrub in the gallery — the regime "
                      "H3 was trained in. Longer clips cost minutes each, so set the cadence "
                      "with 'Generate every N epochs'. On 16 GB cards previews cap at 768×640 "
                      "and 22 frames (sound kept) — longer or larger picks clamp automatically. "
                      "Full write-up in the README.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=560, justify=tk.LEFT)
        self._sample_frames_hint.grid(row=9, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))
        # hidden until a MiniMax family is selected
        for _w in (self.sample_frames_label, self.sample_frames_combo, self._sample_frames_hint):
            _w.grid_remove()

        # --- Turbo preview pace (MiniMax only) ---------------------------------------------
        # Live only when the Turbo LoRA is set in Preferences; 6 steps at 75% is the tested
        # recommendation (Peter). Shown/hidden with the sample-length row above.
        self.turbo_pace_label = ttk.Label(prompt_card, text="Turbo preview:")
        self.turbo_pace_label.grid(row=10, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._turbo_pace_row = tk.Frame(prompt_card, bg=COLORS["bg_surface"])
        self._turbo_pace_row.grid(row=10, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.turbo_steps_entry = ttk.Entry(self._turbo_pace_row, width=5)
        self.turbo_steps_entry.insert(0, str(self.settings.get("MINIMAX_TURBO_STEPS", 6)))
        self.turbo_steps_entry.pack(side=tk.LEFT)
        ttk.Label(self._turbo_pace_row, text="steps at").pack(side=tk.LEFT, padx=6)
        self.turbo_strength_entry = ttk.Entry(self._turbo_pace_row, width=5)
        self.turbo_strength_entry.insert(0, str(self.settings.get("MINIMAX_TURBO_STRENGTH", 75)))
        self.turbo_strength_entry.pack(side=tk.LEFT)
        ttk.Label(self._turbo_pace_row, text="% strength").pack(side=tk.LEFT, padx=(6, 0))
        self._turbo_pace_hint = tk.Label(prompt_card,
                 text="Used when the Turbo LoRA is set in Preferences: previews render in "
                      "these few steps with the Turbo at this strength on top of your "
                      "training LoRA — previews only, never the saved LoRA. 6 steps at 75% "
                      "is the tested recommendation; without the Turbo, the Steps box above "
                      "applies as before.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=560, justify=tk.LEFT)
        self._turbo_pace_hint.grid(row=11, column=1, columnspan=2, sticky=tk.W, pady=(0, 4))
        for _w in (self.turbo_pace_label, self._turbo_pace_row, self._turbo_pace_hint):
            _w.grid_remove()

        # Card 2: Generation Frequency
        freq_card = self._start_section_card(
            self.sample_settings_frame, "Generation Frequency",
            "How often preview renders fire during training. Set either value to 0 to disable that cadence.",
        )
        freq_card.grid_columnconfigure(1, weight=1)

        ttk.Label(freq_card, text="Every N Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_epochs_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_EPOCHS"]))
        self.sample_every_n_epochs_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_epochs_var, width=10)
        self.sample_every_n_epochs_entry.grid(row=0, column=1, sticky=tk.W, pady=4)

        ttk.Label(freq_card, text="Every N Steps:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_STEPS"]))
        self.sample_every_n_steps_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_steps_var, width=10)
        self.sample_every_n_steps_entry.grid(row=1, column=1, sticky=tk.W, pady=4)
        tk.Label(freq_card, text="(0 = disabled)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=2, sticky=tk.W, padx=(10, 0)
        )

        self.sample_at_first_var = tk.BooleanVar(value=self.settings["SAMPLE_AT_FIRST"])
        ttk.Checkbutton(
            freq_card, text="Sample at Start", variable=self.sample_at_first_var
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        self.use_distilled_samples_var = tk.BooleanVar(value=True)
        self.use_distilled_check = ttk.Checkbutton(
            freq_card, text="Use Distilled model for samples (4-step, matches ComfyUI)",
            variable=self.use_distilled_samples_var,
            command=self._on_distilled_samples_toggled,
        )
        self.use_distilled_check.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        # Cache the Distilled sample model in CPU RAM between epochs (skip disk reload).
        self.cache_sample_model_label = ttk.Label(freq_card, text="Cache sample model in RAM:")
        self.cache_sample_model_label.grid(
            row=4, column=0, sticky=tk.W, padx=(0, 10), pady=(8, 0))
        # Allow-list on restore: a saved value outside auto/on/off lands in the readonly
        # combobox without complaint otherwise.
        _csm = str(self.settings.get("CACHE_SAMPLE_MODEL", "auto"))
        self.cache_sample_model_var = tk.StringVar(value=_csm if _csm in ("auto", "on", "off") else "auto")
        self.cache_sample_model_combo = ttk.Combobox(freq_card, textvariable=self.cache_sample_model_var,
                     values=["auto", "on", "off"], state="readonly", width=8)
        self.cache_sample_model_combo.grid(row=4, column=1, sticky=tk.W, pady=(8, 0))
        self.cache_sample_model_note = tk.Label(freq_card,
                 text="Keeps the ~10 GB Distilled model resident in system RAM between epochs so it isn't "
                      "re-read from disk every sample (~3–4 s/epoch saved). auto = only when free RAM is "
                      "comfortable; off = reload each time. Only applies when sampling isn't block-swapping the "
                      "Distilled — i.e. 24 GB+ cards, where the sample peaks around ~18 GB).",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT)
        self.cache_sample_model_note.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))

        # Krea 2 preview engine — lives HERE with the other sample-model choices (the Distilled
        # toggle above is Klein's equivalent choice). Shown only in Krea 2 mode, via
        # _apply_samples_klein_only. raw_lora: previews render on the resident training DiT
        # with the Turbo LoRA @1.0 — identical 8-step CFG-free settings, no ~13 GB Turbo load,
        # no parking the trainer to CPU per preview. turbo_model: classic checkpoint path.
        self._KREA2_ENGINE_LABELS = {
            "raw_lora": "RAW + Turbo LoRA (no model swapping — recommended)",
            "turbo_model": "Turbo model (classic — loads the fp8 Turbo each preview)",
        }
        _eng_saved = str(self.last_used.get("krea2_preview_engine", "raw_lora"))
        if _eng_saved not in self._KREA2_ENGINE_LABELS:
            _eng_saved = "raw_lora"
        self.krea2_preview_engine_var = tk.StringVar(value=self._KREA2_ENGINE_LABELS[_eng_saved])
        self.krea2_engine_frame = tk.Frame(freq_card, bg=COLORS["bg_surface"])
        tk.Label(self.krea2_engine_frame, text="Krea 2 preview engine:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))
        _eng_combo = ttk.Combobox(self.krea2_engine_frame, textvariable=self.krea2_preview_engine_var,
                                  state="readonly", width=52,
                                  values=list(self._KREA2_ENGINE_LABELS.values()))
        _eng_combo.pack(side=tk.LEFT)
        _eng_combo.bind("<<ComboboxSelected>>", lambda e: self._save_last_used_paths())
        ToolTip(_eng_combo,
                "RAW + Turbo LoRA renders previews on the training model itself with the Turbo\n"
                "distillation LoRA applied at 1.0 — same 8-step CFG-free settings as the Turbo\n"
                "model, but nothing is loaded or moved between epochs. The LoRA auto-downloads\n"
                "(~470 MB) if missing. The classic mode loads the fp8 Turbo checkpoint per preview.")
        self.krea2_engine_note = tk.Label(
            freq_card,
            text="Renders previews on the model already training, with the official Turbo LoRA "
                 "switched on just for the render — nothing loaded or moved between epochs. The "
                 "classic mode loads the ~13 GB Turbo checkpoint per preview instead.",
            font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=600, justify=tk.LEFT)
        # Gridded (rows 6-7) / removed by _apply_samples_klein_only; hidden by default (Klein).

        # Card 3: Architecture-Specific (Flow Shift / Guidance / Negative / CFG)
        arch_card = self._start_section_card(
            self.sample_settings_frame, "Advanced",
            "Architecture-specific knobs. Distilled models disable Negative Prompt; "
            "non-distilled models disable CFG Scale.",
        )
        self._samples_arch_card = arch_card       # description reworded per family
        # The whole card is hidden for MiniMax (every row in it is inapplicable), so keep the
        # OUTER frame — _start_section_card returns the inner content frame, and it is the outer
        # that was packed into sample_settings_frame.
        self._samples_arch_outer = arch_card.master.master
        arch_card.grid_columnconfigure(1, weight=1)

        def _arch_note(parent, text):
            return tk.Label(parent, text=text, font=(FONT_FAMILY, 9),
                            fg=COLORS["text_muted"], bg=COLORS["bg_surface"])

        self.sample_flow_shift_label = ttk.Label(arch_card, text="Flow Shift:")
        self.sample_flow_shift_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_flow_shift_var = tk.StringVar(value=str(self.settings["SAMPLE_FLOW_SHIFT"]))
        _flow_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _flow_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_flow_shift_entry = ttk.Entry(_flow_frame, textvariable=self.sample_flow_shift_var, width=10)
        self.sample_flow_shift_entry.pack(side=tk.LEFT)
        self._sample_flow_note = _arch_note(
            _flow_frame, "Base samples only — Distilled uses its own schedule")
        self._sample_flow_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_flow_shift_row = 0

        # (Guidance Scale removed — Klein Base has no guidance embed and Distilled
        # is locked at 1.0, so the field did nothing for Klein. CFG Scale is the
        # only real steering knob for Base samples.)

        self.sample_negative_label = ttk.Label(arch_card, text="Negative Prompt:")
        self.sample_negative_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_negative_var = tk.StringVar(value=self.settings["SAMPLE_NEGATIVE"])
        _neg_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _neg_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_negative_entry = ttk.Entry(_neg_frame, textvariable=self.sample_negative_var, width=50)
        self.sample_negative_entry.pack(side=tk.LEFT)
        self._sample_neg_note = _arch_note(_neg_frame, "Base samples only — Distilled ignores it")
        self._sample_neg_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_negative_row = 1

        self.sample_cfg_label = ttk.Label(arch_card, text="CFG Scale:")
        self.sample_cfg_label.grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_cfg_scale_var = tk.StringVar(value=str(self.settings["SAMPLE_CFG_SCALE"]))
        _cfg_frame = tk.Frame(arch_card, bg=COLORS["bg_surface"])
        _cfg_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.sample_cfg_scale_entry = ttk.Entry(_cfg_frame, textvariable=self.sample_cfg_scale_var, width=10)
        self.sample_cfg_scale_entry.pack(side=tk.LEFT)
        self._sample_cfg_note = _arch_note(_cfg_frame, "Base samples only — Distilled uses no CFG")
        self._sample_cfg_note.pack(side=tk.LEFT, padx=(10, 0))
        self.sample_cfg_row = 2

        # Card 4: Viewer
        viewer_card = self._start_section_card(
            self.sample_settings_frame, "Viewer",
            "Browse generated samples without leaving the app, or open the folder in Explorer.",
        )
        # Anchor for re-showing the Advanced card: pack() alone would re-add it at the BOTTOM,
        # below Viewer, so it has to go back in before this one.
        self._samples_viewer_outer = viewer_card.master.master

        viewer_buttons = tk.Frame(viewer_card, bg=COLORS["bg_surface"])
        viewer_buttons.pack(anchor=tk.W)
        ttk.Button(viewer_buttons, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(viewer_buttons, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT)

        self.sample_output_label = tk.Label(
            viewer_card, text="Sample output: <output_dir>/sample/",
            font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
        )
        self.sample_output_label.pack(anchor=tk.W, pady=(10, 0))

        # Store entries for saving/loading
        self.entries["SAMPLE_ENABLED"] = self.sample_enabled_var
        self.entries["SAMPLE_WIDTH"] = self.sample_width_combo
        self.entries["SAMPLE_HEIGHT"] = self.sample_height_combo
        self.entries["SAMPLE_STEPS"] = self.sample_steps_entry
        self.entries["SAMPLE_SEED"] = self.sample_seed_entry
        self.entries["SAMPLE_EVERY_N_EPOCHS"] = self.sample_every_n_epochs_entry
        self.entries["SAMPLE_EVERY_N_STEPS"] = self.sample_every_n_steps_entry
        self.entries["SAMPLE_AT_FIRST"] = self.sample_at_first_var
        self.entries["SAMPLE_FLOW_SHIFT"] = self.sample_flow_shift_entry
        self.entries["SAMPLE_NEGATIVE"] = self.sample_negative_entry
        self.entries["SAMPLE_CFG_SCALE"] = self.sample_cfg_scale_entry
        self.entries["SAMPLE_FRAMES"] = self.sample_frames_combo
        self.entries["MINIMAX_TURBO_STEPS"] = self.turbo_steps_entry
        self.entries["MINIMAX_TURBO_STRENGTH"] = self.turbo_strength_entry

        # Initial UI state based on current architecture
        self.update_samples_ui_for_architecture()

        self._add_youtube_help_button(outer, "samples")

    def toggle_sample_settings(self):
        """Enable or disable sample settings based on the enable checkbox"""
        state = tk.NORMAL if self.sample_enabled_var.get() else tk.DISABLED

        def _apply(widget):
            try:
                if isinstance(widget, tk.Text):
                    widget.configure(state=state if state == tk.NORMAL else tk.DISABLED)
                else:
                    widget.configure(state=state)
            except tk.TclError:
                pass  # Some widgets don't support state

        def _walk(parent):
            for child in parent.winfo_children():
                _apply(child)
                if child.winfo_children():
                    _walk(child)

        _walk(self.sample_settings_frame)

        # The walk above re-enabled every child uniformly — re-assert the Klein-only
        # greying so the Distilled / Cache-model / Reference controls stay disabled in
        # Krea 2 mode (and the distilled-toggle-driven field states).
        if self.sample_enabled_var.get():
            try:
                cfg = ARCHITECTURES.get(self.architecture_var.get(), {})
                self._apply_samples_klein_only(cfg.get("is_krea2", False))
                self._on_distilled_samples_toggled()
            except Exception:
                pass

    def _on_distilled_samples_toggled(self):
        """Grey out fields that Distilled overrides when the checkbox is ticked."""
        use_distilled = self.use_distilled_samples_var.get()
        state = "disabled" if use_distilled else "normal"
        grey = COLORS["text_muted"] if use_distilled else COLORS["text_primary"]

        # Steps (overridden to 4)
        self.sample_steps_entry.configure(state=state)
        self.sample_steps_label.configure(foreground=grey)
        # Flow Shift (overridden to auto)
        self.sample_flow_shift_entry.configure(state=state)
        self.sample_flow_shift_label.configure(foreground=grey)
        # Negative Prompt (not used by Distilled)
        self.sample_negative_entry.configure(state=state)
        self.sample_negative_label.configure(foreground=grey)
        # CFG Scale (not used by Distilled)
        self.sample_cfg_scale_entry.configure(state=state)
        self.sample_cfg_label.configure(foreground=grey)

    def _lora_name_rename_blocked(self) -> bool:
        """True when the LoRA name must not be touched: a run is live, or one is paused.

        The name is a filename PREFIX, not a label — _detect_latest_state_dir rebuilds
        f'{name}-NNNNNN-state' from it to find a resumable run, and checkpoint discovery
        rebuilds '{name}-NNNNNN.safetensors'. Renaming the box would orphan a paused run in a
        way the user wouldn't notice until they tried to resume it."""
        proc = getattr(self, "current_process", None)
        try:
            if proc is not None and proc.poll() is None:
                return True
        except Exception:
            pass
        try:
            return os.path.exists(self._paused_sidecar_path())
        except Exception:
            return False

    def _apply_lora_name_suffix(self, arch: str):
        """Retag the LoRA name for `arch` — myface_k9b -> myface_krea2 and back.

        Only rewrites a trailing tag that is itself a known family suffix, so names that don't
        follow the convention (bobs_dog, portrait_v2) are never touched. Idempotent: a name that
        already carries the right suffix is left alone, which is what makes it safe to call both
        at startup and after every family switch."""
        entry = self.entries.get("LORA_NAME") if hasattr(self, "entries") else None
        want = ARCHITECTURES.get(arch, {}).get("lora_name_suffix")
        if entry is None or not want or self._lora_name_rename_blocked():
            return
        try:
            name = entry.get().strip()
        except (AttributeError, tk.TclError):
            return
        for suffix in LORA_NAME_SUFFIXES:
            if suffix != want and name.endswith("_" + suffix):
                new = name[: -len(suffix)] + want
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, new)
                except (AttributeError, tk.TclError):
                    pass
                return

    def _on_architecture_selected(self, event=None):
        """Model-family selector changed — refresh sample defaults + presets + persist."""
        # Snapshot the family we're LEAVING before anything reshapes the tab. The combobox
        # has already moved to the new value by the time this fires, so the outgoing family
        # is _arch_last_selected and the widgets still hold its values — but only until
        # update_ui_for_architecture() runs below. Hence: first thing, before any UI work.
        _arch_new = self.architecture_var.get()
        _arch_old = getattr(self, "_arch_last_selected", None)
        _arch_changed = _arch_new != _arch_old
        if _arch_changed and _arch_old:
            try:
                self._arch_settings_memory[_arch_old] = self._collect_preset_values()
                # Capture the preset LABEL here too — refresh_preset_combobox() below
                # rewrites it, so this is the last moment it still names the old family's.
                self._arch_preset_name_memory[_arch_old] = self.custom_preset_var.get()
            except Exception:
                pass

        try:
            self.update_samples_ui_for_architecture()
        except Exception:
            pass
        # Refresh Training-tab field/section visibility for the new architecture
        # (hides Krea 2-unsupported controls; re-shows them for Klein).
        try:
            self.update_ui_for_architecture()
        except Exception:
            pass
        # Pause/Resume availability is architecture-dependent (Krea 2: Start/Stop only).
        try:
            self._refresh_training_buttons()
        except Exception:
            pass
        # Swap the preset dropdown to this family's presets, and actually load values into
        # the fields — either what this family last had, or its default preset.
        #
        # Naming a preset without applying it is a lie the user acts on: switching to Krea 2
        # left Klein's 55 epochs / rank 16 sitting in the fields while the dropdown read
        # "Krea 2 Defaults (rank 32, full model)". Those values don't transfer — Klein's
        # rank/epoch/block-targeting recipe is meaningless for Krea 2.
        #
        # Per-family memory: first visit to a family gets its default preset, every later
        # visit gets that family's own settings back, so flipping over to check something
        # doesn't cost you your tuning. Session-scoped — a restart starts fresh.
        #
        # Only on a REAL family change: re-selecting the entry that's already active fires
        # <<ComboboxSelected>> too, and that must not touch the fields at all.
        try:
            self._arch_last_selected = _arch_new
            self.refresh_preset_combobox()
            builtins = self._builtins_for_arch(_arch_new)
            _default_name = next(iter(builtins)) if builtins else None
            if _arch_changed:
                _remembered = self._arch_settings_memory.get(_arch_new)
                if _remembered:
                    self._apply_preset_values(_remembered)
                    self.custom_preset_var.set(self._arch_preset_name_memory.get(_arch_new, ""))
                    self.update_console(f"[preset] {_arch_new} selected — restored your settings "
                                        f"from earlier this session\n")
                elif _default_name:
                    self._apply_preset_values(builtins[_default_name])
                    self.custom_preset_var.set(_default_name)
                    self.update_console(f"[preset] {_arch_new} selected — applied {_default_name}\n")
            elif _default_name and not self.custom_preset_var.get():
                self.custom_preset_var.set(_default_name)
        except Exception:
            pass
        # Retag the LoRA name LAST — _apply_preset_values above rewrites every field including
        # LORA_NAME, so doing this earlier would just be clobbered. On a return visit the
        # restored name already carries the right suffix and this is a no-op; on a first visit
        # it converts the outgoing family's name (which the built-in presets don't set).
        try:
            self._apply_lora_name_suffix(_arch_new)
        except Exception:
            pass
        try:
            self._save_last_used_paths()
        except Exception:
            pass
        # Audio-only greying depends on BOTH the folder and the family — re-check on a switch.
        try:
            self._refresh_audio_only_ui()
        except Exception:
            pass

    def update_samples_ui_for_architecture(self):
        """Update samples tab UI based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        supports_samples = config.get("supports_samples", False)

        if not supports_samples:
            # Show warning, hide settings
            self.video_model_warning_frame.grid()
            self.sample_enabled_check.grid_remove()
            self.sample_settings_frame.grid_remove()
        else:
            # Hide warning, show settings
            self.video_model_warning_frame.grid_remove()
            self.sample_enabled_check.grid()
            self.sample_settings_frame.grid()

            # Apply this architecture's sample defaults ONLY when the architecture actually
            # changed. This method fires on every Base Model combobox event — including
            # re-picking the same family — and unconditionally overwriting CFG/flow-shift/
            # steps/width/height silently reverted the user's preview config (e.g. a
            # 1024x1024 preview back to 768x768). On a real switch, the outgoing family's
            # values are stashed and restored when the user switches back.
            _prev = getattr(self, "_sample_defaults_arch", None)
            if _prev != arch:
                if not hasattr(self, "_arch_sample_stash"):
                    self._arch_sample_stash = {}
                if _prev is not None:
                    self._arch_sample_stash[_prev] = {
                        "cfg": self.sample_cfg_scale_var.get(),
                        "shift": self.sample_flow_shift_var.get(),
                        "steps": self.sample_steps_var.get(),
                        "w": self.sample_width_var.get(),
                        "h": self.sample_height_var.get(),
                    }
                stash = self._arch_sample_stash.get(arch)
                if stash is not None:
                    self.sample_cfg_scale_var.set(stash["cfg"])
                    self.sample_flow_shift_var.set(stash["shift"])
                    self.sample_steps_var.set(stash["steps"])
                    self.sample_width_var.set(stash["w"])
                    self.sample_height_var.set(stash["h"])
                else:
                    if config.get("sample_cfg_default") is not None:
                        self.sample_cfg_scale_var.set(str(config["sample_cfg_default"]))
                    if config.get("sample_flow_shift_default") is not None:
                        self.sample_flow_shift_var.set(str(config["sample_flow_shift_default"]))
                    if config.get("sample_steps_default") is not None:
                        self.sample_steps_var.set(str(config["sample_steps_default"]))
                    if config.get("sample_width_default") is not None:
                        self.sample_width_var.set(str(config["sample_width_default"]))
                    if config.get("sample_height_default") is not None:
                        self.sample_height_var.set(str(config["sample_height_default"]))
                self._sample_defaults_arch = arch

            # Enable/disable flow shift based on architecture
            if config.get("sample_flow_shift_default") is None:
                self.sample_flow_shift_entry.configure(state=tk.DISABLED)
                self.sample_flow_shift_label.configure(foreground="gray")
            else:
                self.sample_flow_shift_entry.configure(state=tk.NORMAL)
                self.sample_flow_shift_label.configure(foreground=FG_COLOR)

            # Handle distilled models (no negative prompts)
            if config.get("sample_is_distilled", False):
                self.sample_negative_entry.configure(state=tk.DISABLED)
                self.sample_negative_label.configure(foreground="gray")
            else:
                self.sample_negative_entry.configure(state=tk.NORMAL)
                self.sample_negative_label.configure(foreground=FG_COLOR)

            # Handle fixed steps/cfg for distilled models
            if config.get("sample_steps_fixed", False):
                self.sample_steps_entry.configure(state=tk.DISABLED)
            else:
                self.sample_steps_entry.configure(state=tk.NORMAL)

            if config.get("sample_cfg_fixed", False):
                self.sample_cfg_scale_entry.configure(state=tk.DISABLED)
            else:
                self.sample_cfg_scale_entry.configure(state=tk.NORMAL)

            # Grey out / relabel the Klein-only sample controls when Krea 2 is selected.
            self._apply_samples_klein_only(config.get("is_krea2", False))
            # ...then let MiniMax override the wording that is still Klein's. Runs AFTER, so the
            # Klein/Krea 2 paths above are untouched.
            self._apply_samples_minimax(bool(config.get("is_minimax")))

            # Sample length (clip) row — MiniMax only: the other families' preview stacks are
            # image pipelines with no frames axis.
            _mm = bool(config.get("is_minimax"))
            for _w in (getattr(self, "sample_frames_label", None),
                       getattr(self, "sample_frames_combo", None),
                       getattr(self, "_sample_frames_hint", None),
                       getattr(self, "turbo_pace_label", None),
                       getattr(self, "_turbo_pace_row", None),
                       getattr(self, "_turbo_pace_hint", None)):
                if _w is None:
                    continue
                (_w.grid if _mm else _w.grid_remove)()

        # Update sample output path label
        self.update_sample_output_label()

    def _apply_samples_minimax(self, is_minimax):
        """Reword the Samples tab for MiniMax H3, and hide what belongs to the other families.

        The tab was written for Klein and still says so: the banner advertises a Distilled
        4-step preview, and every Advanced row is annotated 'Base samples only — Distilled ...'.
        None of that exists on H3, which renders clips on the model being trained, CFG-free, on
        a fixed shift-12 schedule. Runs after _apply_samples_klein_only so Klein and Krea 2 keep
        exactly the text they had; this only rewrites when MiniMax is the selected family."""
        _MM = {
            "banner": "Preview prompts rendered periodically during training, as short clips on "
                      "the model being trained. Samples land in <output_dir>/sample/ and the "
                      "Gallery button below opens the viewer.",
            "advanced": "H3 renders CFG-free on a fixed schedule, so the knobs it does not use "
                        "are greyed out.",
            "flow": "Fixed at 12 for H3 — the schedule every shipped workflow uses",
            "neg": "Unused — H3 samples render without CFG",
            "cfg": "H3 renders without CFG; the shipped workflow does the same",
            "steps": "20 steps matches the shipped H3 workflow",
        }
        _KLEIN = {
            "banner": "Preview prompts rendered periodically during training (Distilled 4-step). "
                      "Samples land in <output_dir>/sample/ and the Gallery button below opens "
                      "the viewer.",
            "advanced": "Architecture-specific knobs. Distilled models disable Negative Prompt; "
                        "non-distilled models disable CFG Scale.",
            "flow": "Base samples only — Distilled uses its own schedule",
            "neg": "Base samples only — Distilled ignores it",
            "cfg": "Base samples only — Distilled uses no CFG",
        }
        _t = _MM if is_minimax else _KLEIN

        if getattr(self, "_samples_banner_sub", None) is not None:
            self._samples_banner_sub.configure(text=_t["banner"])
        _adv = getattr(self, "_samples_arch_card", None)
        if _adv is not None and getattr(_adv, "_desc_label", None) is not None:
            _adv._desc_label.configure(text=_t["advanced"])

        # The whole Advanced card goes for MiniMax: Flow Shift is fixed at 12, and Negative
        # Prompt and CFG Scale are both inert on a CFG-free family — three greyed rows under a
        # heading is just a card asking to be misread.
        _adv_outer = getattr(self, "_samples_arch_outer", None)
        if _adv_outer is not None:
            if is_minimax:
                _adv_outer.pack_forget()
            elif not _adv_outer.winfo_manager():
                _anchor = getattr(self, "_samples_viewer_outer", None)
                if _anchor is not None and _anchor.winfo_manager():
                    _adv_outer.pack(fill=tk.X, padx=36, pady=(0, 16), before=_anchor)
                else:
                    _adv_outer.pack(fill=tk.X, padx=36, pady=(0, 16))
        for _attr, _key in (("_sample_flow_note", "flow"), ("_sample_neg_note", "neg"),
                            ("_sample_cfg_note", "cfg")):
            _w = getattr(self, _attr, None)
            if _w is not None:
                _w.configure(text=_t[_key])
        # Steps note: _apply_samples_klein_only owns the Klein/Krea 2 wording, so only override.
        if is_minimax and hasattr(self, "sample_steps_note"):
            self.sample_steps_note.configure(text=_MM["steps"])

        # Klein's sample-model choice and its RAM cache have no MiniMax equivalent — previews
        # always render on the resident training DiT. The Reference row goes too: it exists
        # because Klein is an EDIT model (and Krea 2 has a vision path); H3 has neither, and its
        # own r2v reference conditioning is a training feature, not a sample one. Hide rather
        # than grey — a disabled control still reads as "something I could turn on".
        for _attr in ("use_distilled_check", "cache_sample_model_label",
                      "cache_sample_model_combo",
                      "sample_ref_label", "_sample_ref_row", "sample_ref_note"):
            _w = getattr(self, _attr, None)
            if _w is None:
                continue
            try:
                (_w.grid_remove if is_minimax else _w.grid)()
            except tk.TclError:
                pass          # packed, not gridded — leave it alone

    def _apply_samples_klein_only(self, is_krea2):
        """Mark the Klein-only sample controls when Krea 2 is selected.

        Krea 2 always renders previews on its fp8 Turbo (8-step, CFG-free, no edit
        reference, model reloaded per pass), so the 'Use Distilled model' toggle, the
        'Cache sample model in RAM' dropdown, and the Klein edit-Reference image don't
        apply. Disable them and relabel as 'Klein only' so it's clear, rather than
        silently leaving live controls that do nothing in Krea 2 mode."""
        muted = COLORS["text_muted"]
        secondary = COLORS["text_secondary"]
        label_fg = muted if is_krea2 else secondary

        # "Use Distilled model for samples" checkbox — Klein's sample-model choice. Krea 2's
        # equivalent choice is the Preview engine dropdown, shown right below in Krea 2 mode.
        if hasattr(self, "use_distilled_check"):
            self.use_distilled_check.configure(
                state=(tk.DISABLED if is_krea2 else tk.NORMAL),
                text=("Use Distilled model for samples — Klein only (Krea 2: pick a preview engine below)"
                      if is_krea2 else
                      "Use Distilled model for samples (4-step, matches ComfyUI)"))

        # Steps note
        if hasattr(self, "sample_steps_note"):
            self.sample_steps_note.configure(
                text=("Klein only — Krea 2 previews are 8-step either way (engine choice below)"
                      if is_krea2 else
                      "Base samples only — Distilled is locked at 4 steps"))

        # Krea 2 preview engine dropdown + note (rows 6-7 of the cadence card) — Krea 2 only.
        if hasattr(self, "krea2_engine_frame"):
            if is_krea2:
                self.krea2_engine_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
                self.krea2_engine_note.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
            else:
                self.krea2_engine_frame.grid_remove()
                self.krea2_engine_note.grid_remove()

        # Reference image — supported by BOTH families now (Klein: edit conditioning; Krea 2:
        # Qwen3-VL vision path). Always enabled; only the note differs. No strength dial on this
        # row in either mode (Krea 2's vision-path reference has no strength; Klein auto-caps).
        if hasattr(self, "sample_ref_entry"):
            self.sample_ref_entry.configure(state="readonly")
        for attr in ("sample_ref_browse_btn", "sample_ref_clear_btn"):
            w = getattr(self, attr, None)
            if w is not None:
                w.configure(state=tk.NORMAL)
        if hasattr(self, "sample_ref_label"):
            self.sample_ref_label.configure(foreground=secondary)
        if hasattr(self, "sample_ref_note"):
            self.sample_ref_note.configure(
                text=("Optional — fed through Krea 2's Qwen3-VL vision path so samples become visually "
                      "aware of it ('prompt from a picture', not a pixel edit). Downscaled to a cap; leave "
                      "empty for normal samples."
                      if is_krea2 else
                      "Optional — Klein is an edit model, so samples can be conditioned on a real image (they edit "
                      "it rather than generate from scratch). Auto-resized to ~0.20 MP so any size is safe. Leave "
                      "empty for normal samples."))

        # "Cache sample model in RAM" dropdown (not implemented for Krea 2)
        if hasattr(self, "cache_sample_model_combo"):
            self.cache_sample_model_combo.configure(state=(tk.DISABLED if is_krea2 else "readonly"))
        if hasattr(self, "cache_sample_model_label"):
            self.cache_sample_model_label.configure(
                text=("Cache sample model in RAM (Klein only):" if is_krea2 else "Cache sample model in RAM:"),
                foreground=label_fg)

    def update_sample_output_label(self):
        """Update the sample output path label to show actual path"""
        samples_dir = self.get_samples_dir()
        self.sample_output_label.config(text=f"Sample Output: {samples_dir}")

    def generate_sample_prompt_file(self):
        """Generate prompt file for sample generation"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        prompt_file = os.path.join(samples_dir, "prompts.txt")

        # Build prompt with options
        prompt = self.sample_prompt_text.get("1.0", tk.END).strip()

        # Add options if not already in prompt
        if "--w" not in prompt:
            prompt += f" --w {self.sample_width_var.get()}"
        if "--h" not in prompt:
            prompt += f" --h {self.sample_height_var.get()}"
        if "--f" not in prompt:
            prompt += " --f 1"  # Always 1 for images
        if "--s" not in prompt:
            prompt += f" --s {self.sample_steps_var.get()}"

        seed = self.sample_seed_var.get()
        if seed and seed != "0" and "--d" not in prompt:
            prompt += f" --d {seed}"

        # Add flow shift if set
        flow_shift = self.sample_flow_shift_var.get()
        if flow_shift and "--fs" not in prompt:
            prompt += f" --fs {flow_shift}"

        # Add negative prompt if set
        negative = self.sample_negative_var.get().strip()
        if negative and "--n" not in prompt:
            prompt += f" --n {negative}"

        # Always add CFG scale (omitting --l causes fallback to 4.0 in Z-Image)
        cfg = self.sample_cfg_scale_var.get()
        if cfg and "--l" not in prompt:
            prompt += f" --l {cfg}"

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-generated by Fizgig LoRA Trainer GUI\n{prompt}\n")

        return prompt_file

    def update_gallery_html(self):
        """Update gallery HTML and files.json for live updates"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        if not self._gallery_owns(samples_dir):
            return   # another Fizgig session owns this gallery now — don't fight over sidecars

        # Find all images
        images = []
        if os.path.exists(samples_dir):
            for f in os.listdir(samples_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    images.append(f)

        # Sort alphabetically (which sorts by epoch due to naming convention)
        images.sort()

        # Write files.json for HTTP fetch (live updates)
        files_json_path = os.path.join(samples_dir, "files.json")
        try:
            with open(files_json_path, 'w', encoding='utf-8') as f:
                json.dump(images, f)
        except Exception:
            pass

        # Clip frames for the gallery scrubber: <stem>.clip/ dirs written by MiniMax clip
        # previews, mapped against their contract PNG. Additive — a gallery with no clips gets
        # an empty map and behaves exactly as before.
        try:
            clips = {}
            for f in os.listdir(samples_dir):
                if f.endswith(".clip") and os.path.isdir(os.path.join(samples_dir, f)):
                    frames = sorted(x for x in os.listdir(os.path.join(samples_dir, f))
                                    if x.lower().endswith((".jpg", ".jpeg", ".png")))
                    if frames:
                        clips[f[:-len(".clip")] + ".png"] = [f + "/" + x for x in frames]
            with open(os.path.join(samples_dir, "clips.json"), 'w', encoding='utf-8') as f:
                json.dump(clips, f)
            # Sample sound (previews with audio): <stem>.wav beside the contract PNG. Its own
            # map so clips.json keeps its shape for older galleries.
            sounds = {f[:-4] + ".png": f for f in os.listdir(samples_dir)
                      if f.lower().endswith(".wav")}
            with open(os.path.join(samples_dir, "sounds.json"), 'w', encoding='utf-8') as f:
                json.dump(sounds, f)
            # Playable clips (frames + sound muxed): <stem>.mp4 — the lightbox plays these
            # instead of the scrub slider + separate audio player.
            videos = {f[:-4] + ".png": f for f in os.listdir(samples_dir)
                      if f.lower().endswith(".mp4")}
            with open(os.path.join(samples_dir, "videos.json"), 'w', encoding='utf-8') as f:
                json.dump(videos, f)
        except Exception:
            pass

        # Map epoch -> checkpoint filename so each gallery entry can offer a "Download LoRA" link.
        # Computed here (Python knows the exact {name}-{epoch:06d}.safetensors naming) rather than
        # guessed in JS. Only epochs with a saved checkpoint appear — matches save_every_n_epochs.
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "") or os.path.dirname(samples_dir)
        lora_map = {}
        try:
            if output_dir and os.path.isdir(output_dir):
                import re as _re_ck
                # Key on THIS run's name, not epoch alone: in a reused output folder
                # (which the gallery supports) the last file listdir returned used to win,
                # serving another run's checkpoint from the Download button.
                _run_name = str(self.settings.get("LORA_NAME", "") or "").strip()
                if _run_name:
                    _ck_pat = _re_ck.compile(
                        r'^' + _re_ck.escape(_run_name) + r'-(\d{6})\.safetensors$')
                else:
                    _ck_pat = _re_ck.compile(r'-(\d{6})\.safetensors$')
                for f in os.listdir(output_dir):
                    if f.endswith(".safetensors"):
                        m = _ck_pat.search(f)
                        if m:
                            lora_map[str(int(m.group(1)))] = f
                # The final LoRA is {LORA_NAME}.safetensors (no epoch suffix) — surface it as a
                # header button under the reserved "final" key (epoch keys are numeric, so no clash).
                final_name = str(self.settings.get("LORA_NAME", "") or "").strip()
                if final_name and os.path.exists(os.path.join(output_dir, final_name + ".safetensors")):
                    lora_map["final"] = final_name + ".safetensors"
            with open(os.path.join(samples_dir, "loras.json"), 'w', encoding='utf-8') as f:
                json.dump(lora_map, f)
        except Exception:
            pass

        # Dataset image list for the likeness baseline picker (images served via /dataset/).
        # The folder path is included so the picker can SHOW which folder it's listing —
        # "why is that image here?" is answered by looking at the folder, not guessing.
        try:
            dfolder = getattr(self, "_gal_dataset_dir", "") or ""
            dimgs = []
            if dfolder and os.path.isdir(dfolder):
                dimgs = sorted(f for f in os.listdir(dfolder)
                               if os.path.splitext(f)[1].lower() in self._FF_EXTS)
            with open(os.path.join(samples_dir, "dataset.json"), 'w', encoding='utf-8') as f:
                json.dump({"folder": dfolder, "images": dimgs}, f)
        except Exception:
            pass

        # Also update embedded data in gallery.html (for fallback)
        gallery_path = os.path.join(samples_dir, "gallery.html")
        if os.path.exists(gallery_path):
            try:
                with open(gallery_path, 'r', encoding='utf-8') as f:
                    html = f.read()

                # Find and replace the embedded JSON. The replacement goes through a
                # FUNCTION, never the template parser: json.dumps escapes non-ASCII as
                # \uXXXX (one Chinese character in a sample filename), which re's template
                # parser rejects as "bad escape \u" — swallowed below, so the fallback
                # data silently stopped updating.
                import re
                new_json = json.dumps(images)
                pattern = r'(<script id="files-data" type="application/json">).*?(</script>)'
                new_html = re.sub(pattern, lambda m: m.group(1) + new_json + m.group(2),
                                  html, flags=re.DOTALL)

                if new_html != html:
                    with open(gallery_path, 'w', encoding='utf-8') as f:
                        f.write(new_html)
            except Exception:
                pass  # Don't fail if gallery update fails

    # region Gallery likeness scoring (CPU ArcFace vs 3 baselines)

    def _gallery_claim(self, samples_dir):
        """Claim gallery-sidecar ownership of this samples dir for THIS app session. Two Fizgig
        instances pointed at one output folder otherwise fight over likeness.json / dataset.json
        every 5s — caught live 2026-07-10: a stale instance's scorer (different baselines)
        alternated everyone's likeness scores down to ~5% each time a new sample landed."""
        if not getattr(self, "_gal_session_token", None):
            self._gal_session_token = f"{os.getpid()}-{int(time.time() * 1000)}"
        path = os.path.join(samples_dir, "gallery.owner")
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"token": self._gal_session_token}, f)
            os.replace(path + ".tmp", path)
        except Exception:
            pass

    def _gallery_owns(self, samples_dir):
        """True while this session still owns the samples dir (an unclaimed dir is claimed).
        The most recent session to open the gallery (or start scoring) wins; older sessions'
        watchers and scorers stand down the moment they see a foreign token."""
        path = os.path.join(samples_dir, "gallery.owner")
        try:
            with open(path, encoding="utf-8") as f:
                tok = json.load(f).get("token")
        except Exception:
            self._gallery_claim(samples_dir)
            return True
        return tok == getattr(self, "_gal_session_token", None)

    def _gallery_set_baselines(self, names):
        """Start (or clear, with an empty list) likeness scoring of the sample gallery.
        Called from the gallery HTTP server thread — plain attributes only, no Tk. Returns
        (ok, message) for the JSON response."""
        self._gal_gen = getattr(self, "_gal_gen", 0) + 1   # invalidates any running worker
        samples_dir = self.get_samples_dir()
        if not names:
            self._gal_baselines = []
            self._gallery_write_likeness(samples_dir, [], "cleared", {})
            return True, "cleared"
        if not FACE_DETECTION_AVAILABLE or FaceEmbedder is None:
            return False, "Face tools not installed — run install_fizgig.py."
        folder = getattr(self, "_gal_dataset_dir", "") or ""
        if not folder or not os.path.isdir(folder):
            return False, "Training image folder not set — set it on the Start tab, then reopen the gallery."
        if len(names) != 3:
            return False, f"Pick exactly 3 baseline images (got {len(names)})."
        paths = [os.path.join(folder, os.path.basename(n)) for n in names]
        missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
        if missing:
            return False, "Not in the dataset folder: " + ", ".join(missing)
        self._gal_baselines = paths
        self._gallery_claim(samples_dir)   # starting scoring (re)claims the dir for this session
        threading.Thread(target=self._gallery_likeness_worker,
                         args=(self._gal_gen, paths, samples_dir), daemon=True).start()
        return True, "scoring started"

    def _gallery_resume_likeness(self):
        """Resume scoring after a GUI restart — likeness.json persists the chosen baselines,
        so reopening the gallery picks up where the last session left off."""
        if getattr(self, "_gal_baselines", None):
            return   # already active this session
        try:
            with open(os.path.join(self.get_samples_dir(), "likeness.json"), encoding="utf-8") as f:
                names = json.load(f).get("baselines") or []
        except Exception:
            return
        if len(names) == 3:
            self._gallery_set_baselines(names)

    @staticmethod
    def _gallery_write_likeness(samples_dir, base_names, status, scores):
        payload = {"baselines": base_names, "status": status, "scores": scores}
        path = os.path.join(samples_dir, "likeness.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)   # atomic — the gallery polls this file
        except Exception:
            pass

    def _gallery_likeness_worker(self, gen, baselines, samples_dir):
        """Score every sample image's face against the 3 baselines (averaged) on CPU — zero
        GPU contention with training. Newest samples first, so the CURRENT run's epochs score
        immediately even when the folder holds hundreds of old samples; then keeps watching
        for new files as training produces them."""
        import numpy as np
        base_names = [os.path.basename(b) for b in baselines]
        scores = {}
        try:   # scores for the same baselines survive GUI restarts — no pointless rescoring
            with open(os.path.join(samples_dir, "likeness.json"), encoding="utf-8") as f:
                old = json.load(f)
            # Order-insensitive: the picker records baselines in CLICK order, and re-picking
            # the same 3 in a different order must not throw the whole score set away.
            if (sorted(old.get("baselines") or []) == sorted(base_names)
                    and isinstance(old.get("scores"), dict)):
                scores = old["scores"]
        except Exception:
            pass
        self._gallery_write_likeness(samples_dir, base_names, "loading face model…", scores)
        base_embs = [self._ff_embed_cached(b) for b in baselines]
        missing = [n for n, e in zip(base_names, base_embs) if e is None]
        if missing:
            self._gallery_write_likeness(samples_dir, base_names,
                                         "error: no face found in " + ", ".join(missing), scores)
            return
        last_status = None
        scored_mtimes = {}   # filename -> mtime at scoring time (rescore no-face on change)
        while gen == getattr(self, "_gal_gen", 0):
            if not self._gallery_owns(samples_dir):
                print("[likeness] another Fizgig session took over this samples folder — "
                      "this scorer is standing down.")
                return
            try:
                files = [f for f in os.listdir(samples_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            except OSError:
                files = []
            scores = {k: v for k, v in scores.items() if k in set(files)}   # drop deleted samples

            def _mt(f):
                try:
                    return os.path.getmtime(os.path.join(samples_dir, f))
                except OSError:
                    return 0.0
            # Settle guard: a sample the trainer is STILL WRITING decodes as a truncated image
            # and scores a spurious "no face" that then sticks. Skip anything modified in the
            # last few seconds — the next pass (~4 s) picks it up complete. A null that slipped
            # through anyway (e.g. scored by an older Fizgig) rescores when its mtime moves.
            now = time.time()
            todo = [f for f in files
                    if (now - _mt(f)) > 5.0
                    and (f not in scores
                         or (scores[f] is None and scored_mtimes.get(f) != _mt(f)))]
            todo.sort(key=_mt, reverse=True)   # current run scores first
            for i, f in enumerate(todo, 1):
                if gen != getattr(self, "_gal_gen", 0):
                    return
                scored_mtimes[f] = _mt(f)
                emb = self._ff_embed_cached(os.path.join(samples_dir, f))
                scores[f] = None if emb is None else round(
                    float(np.mean([float(np.dot(be, emb)) for be in base_embs])), 4)
                if i % 2 == 0 or i == len(todo):
                    self._gallery_write_likeness(samples_dir, base_names,
                                                 f"scoring… {len(scores)}/{len(files)}", scores)
            status = f"live — {len(scores)} sample(s) scored"
            if todo or status != last_status:
                self._gallery_write_likeness(samples_dir, base_names, status, scores)
                last_status = status
            for _ in range(16):   # ~4 s between folder checks, responsive to clear/re-pick
                if gen != getattr(self, "_gal_gen", 0):
                    return
                time.sleep(0.25)

    # endregion

    def start_samples_watcher(self):
        """Start background thread to update files.json for live gallery"""
        if self.samples_watcher_running:
            return

        self.samples_watcher_running = True

        def watcher_loop():
            while self.samples_watcher_running:
                try:
                    self.update_gallery_html()
                except Exception:
                    pass
                # Update every 5 seconds
                for _ in range(50):  # 5 seconds in 0.1s increments
                    if not self.samples_watcher_running:
                        break
                    time.sleep(0.1)

        self.samples_watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
        self.samples_watcher_thread.start()

    def stop_samples_watcher(self):
        """Stop the samples watcher thread"""
        was_running = self.samples_watcher_running
        self.samples_watcher_running = False
        if was_running:
            # One last refresh: the final epoch's samples and the final .safetensors land
            # seconds before the trainer exits, almost always inside the watcher's 5 s
            # sleep — without this the gallery never gains the final-epoch previews or
            # the Download Final LoRA button (loras.json "final" key).
            def _final_refresh():
                try:
                    self.update_gallery_html()
                except Exception:
                    pass
            threading.Thread(target=_final_refresh, daemon=True).start()

    def parse_sample_filename(self, filename):
        """Parse epoch, step, seed from sample filename"""
        import re
        info = {"epoch": None, "step": None, "seed": None, "prompt_idx": None}

        # Pattern: {name}_e{epoch:06d}_{promptIdx}_{timestamp}_{seed}.png
        # or: {name}_{step:06d}_{promptIdx}_{timestamp}_{seed}.png
        epoch_match = re.search(r'_e(\d{6})_', filename)
        if epoch_match:
            info["epoch"] = int(epoch_match.group(1))

        step_match = re.search(r'_(\d{6})_(\d{2})_\d{14}', filename)
        if step_match and not epoch_match:
            info["step"] = int(step_match.group(1))
            info["prompt_idx"] = int(step_match.group(2))
        elif epoch_match:
            prompt_match = re.search(r'_e\d{6}_(\d{2})_', filename)
            if prompt_match:
                info["prompt_idx"] = int(prompt_match.group(1))

        seed_match = re.search(r'_(\d+)\.\w+$', filename)
        if seed_match:
            info["seed"] = int(seed_match.group(1))

        return info

    def generate_gallery_html(self, images):
        """Generate the gallery HTML content"""
        import time

        # Parse image info and build items
        image_data = []
        for filename, mtime in images:
            info = self.parse_sample_filename(filename)
            info["filename"] = filename
            info["mtime"] = mtime
            info["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            image_data.append(info)

        image_items = ""
        for img in image_data:
            filename = img["filename"]
            timestamp = img["timestamp"]

            # Build epoch/step badge
            if img["epoch"] is not None:
                badge = f'<span class="epoch-badge">Epoch {img["epoch"]}</span>'
            elif img["step"] is not None:
                badge = f'<span class="step-badge">Step {img["step"]}</span>'
            else:
                badge = ''

            # Build seed info
            seed_info = f'<span class="seed">Seed: {img["seed"]}</span>' if img["seed"] else ''

            image_items += f'''
            <div class="gallery-item" onclick="openLightbox('{filename}')">
                <div class="image-container">
                    <img src="{filename}" alt="{filename}" loading="lazy">
                    {badge}
                </div>
                <div class="image-info">
                    <span class="filename">{filename}</span>
                    <div class="meta-row">
                        <span class="timestamp">{timestamp}</span>
                        {seed_info}
                    </div>
                </div>
            </div>'''

        if not image_items:
            image_items = '<div class="no-images">No sample images found yet. Start training to generate samples.</div>'

        # Build image data for JavaScript
        js_image_data = []
        for img in image_data:
            js_image_data.append({
                "filename": img["filename"],
                "epoch": img["epoch"],
                "step": img["step"],
                "seed": img["seed"]
            })

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }}
        header {{ background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }}
        header h1 {{ color: #ECF0F1; font-size: 24px; margin-bottom: 15px; }}
        .controls {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
        .controls label {{ display: flex; align-items: center; gap: 8px; }}
        .controls input[type="number"], .controls select {{ padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }}
        .controls button {{ padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background-color: #3498DB; }}
        #last-update {{ color: #95A5A6; font-size: 14px; }}
        main {{ padding: 20px; }}
        #gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .gallery-item {{ background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .gallery-item:hover {{ transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }}
        .image-container {{ position: relative; }}
        .gallery-item img {{ width: 100%; height: 250px; object-fit: contain; display: block; background-color: #1B2A38; }}
        .epoch-badge, .step-badge {{ position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .epoch-badge {{ background-color: #27AE60; color: white; }}
        .step-badge {{ background-color: #E67E22; color: white; }}
        .image-info {{ padding: 12px; display: flex; flex-direction: column; gap: 6px; }}
        .filename {{ font-weight: 500; font-size: 13px; word-break: break-all; color: #BDC3C7; }}
        .meta-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        .timestamp {{ color: #95A5A6; font-size: 12px; }}
        .seed {{ color: #3498DB; font-size: 12px; font-family: monospace; }}
        .no-images {{ grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; font-size: 18px; }}
        #lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }}
        #lightbox.active {{ display: flex; }}
        #lightbox img {{ max-width: 90%; max-height: 80%; object-fit: contain; }}
        #lightbox .close-btn {{ position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }}
        #lightbox .close-btn:hover {{ color: #E74C3C; }}
        #lightbox .nav-btn {{ position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }}
        #lightbox .nav-btn:hover {{ color: #2980B9; }}
        #lightbox .prev-btn {{ left: 20px; }}
        #lightbox .next-btn {{ right: 20px; }}
        #lightbox .image-details {{ margin-top: 15px; text-align: center; }}
        #lightbox .image-name {{ color: #ECF0F1; font-size: 16px; }}
        #lightbox .image-meta {{ color: #95A5A6; font-size: 14px; margin-top: 5px; }}
    </style>
</head>
<body>
    <header>
        <h1>Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Sort by: <select id="sort-select" onchange="sortGallery()">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High to Low)</option>
                <option value="epoch-asc">Epoch (Low to High)</option>
            </select></label>
            <label>Auto-refresh: <input type="number" id="refresh-interval" value="30" min="5" max="300"> sec</label>
            <button onclick="refreshGallery()">Refresh Now</button>
            <span id="last-update"></span>
        </div>
    </header>
    <main><div id="gallery">{image_items}</div></main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <script>
        const imageData = {json.dumps(js_image_data)};
        let refreshInterval = localStorage.getItem('fizgig-refresh') || 30;
        let refreshTimer = null;
        let currentImageIndex = 0;
        const images = imageData.map(d => d.filename);
        document.getElementById('refresh-interval').value = refreshInterval;
        updateLastRefresh();
        startRefreshTimer();
        const savedSort = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('sort-select').value = savedSort;
        document.getElementById('refresh-interval').addEventListener('change', (e) => {{
            refreshInterval = Math.max(5, Math.min(300, parseInt(e.target.value) || 30));
            e.target.value = refreshInterval;
            localStorage.setItem('fizgig-refresh', refreshInterval);
            startRefreshTimer();
        }});
        function startRefreshTimer() {{ if (refreshTimer) clearInterval(refreshTimer); refreshTimer = setInterval(refreshGallery, refreshInterval * 1000); }}
        function refreshGallery() {{ location.reload(); }}
        function updateLastRefresh() {{ document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString(); }}
        function sortGallery() {{
            const sortBy = document.getElementById('sort-select').value;
            localStorage.setItem('fizgig-sort', sortBy);
            const gallery = document.getElementById('gallery');
            const items = Array.from(gallery.querySelectorAll('.gallery-item'));
            items.sort((a, b) => {{
                const aFile = a.querySelector('img').alt;
                const bFile = b.querySelector('img').alt;
                const aData = imageData.find(d => d.filename === aFile) || {{}};
                const bData = imageData.find(d => d.filename === bFile) || {{}};
                switch(sortBy) {{
                    case 'newest': return images.indexOf(aFile) - images.indexOf(bFile);
                    case 'oldest': return images.indexOf(bFile) - images.indexOf(aFile);
                    case 'epoch-desc': return (bData.epoch || 0) - (aData.epoch || 0);
                    case 'epoch-asc': return (aData.epoch || 0) - (bData.epoch || 0);
                    default: return 0;
                }}
            }});
            items.forEach(item => gallery.appendChild(item));
        }}
        function getImageMeta(filename) {{
            const data = imageData.find(d => d.filename === filename);
            if (!data) return '';
            const parts = [];
            if (data.epoch !== null) parts.push('Epoch ' + data.epoch);
            if (data.step !== null) parts.push('Step ' + data.step);
            if (data.seed !== null) parts.push('Seed: ' + data.seed);
            return parts.join(' | ');
        }}
        function openLightbox(filename) {{
            currentImageIndex = images.indexOf(filename);
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }}
        function navigateLightbox(direction) {{
            if (images.length === 0) return;
            currentImageIndex = (currentImageIndex + direction + images.length) % images.length;
            const filename = images[currentImageIndex];
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
        }}
        document.addEventListener('keydown', (e) => {{
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        }});
        document.getElementById('lightbox').addEventListener('click', (e) => {{ if (e.target.id === 'lightbox') closeLightbox(); }});
        sortGallery();
    </script>
</body>
</html>'''
        return html

