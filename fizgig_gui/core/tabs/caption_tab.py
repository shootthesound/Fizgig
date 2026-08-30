import glob
import json
import os
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR, ENTRY_BG, FG_COLOR
from fizgig_gui.core.config.prefs import FLORENCE_TASKS, FLORENCE_MODELS, QWEN_CAPTION_MODEL, FLORENCE_DEFAULT_MODEL, \
    QWEN_CUSTOM_TASK, save_prefs, FLORENCE_REVISIONS, FLORENCE_CODE_REVISIONS
from fizgig_gui.core.ui_base.widgets import ToolTip

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class CaptionTabMixin:
    def create_caption_generator(self):
        """Create the Captions tab (Start-tab styled)."""
        scrollable_frame, self.caption_canvas = self.create_scrollable_frame(self.caption_gen_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Captions",
            "Write trigger-word captions or generate them with AI. "
            "You can optionally skip this tab if your images already have .txt caption files.",
        )

        # Card 1: Captioning Settings
        settings_card = self._start_section_card(
            outer, "Captioning Settings",
            "Trigger word is prepended to every caption. Qwen3-VL follows a captioning instruction "
            "you can read and edit, and is the better fit for training data — it needs the Krea 2 "
            "text encoder set in Preferences, and captions any dataset, Klein included. Florence-2 "
            "is smaller and downloads itself on first use. Static Caption writes the trigger word "
            "only.",
        )
        settings_card.grid_columnconfigure(1, weight=1)

        ttk.Label(settings_card, text="Image Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._caption_folder_display = tk.Label(settings_card, textvariable=self.image_folder_var,
                                                font=(FONT_FAMILY, 10),
                                                fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                                                anchor="w")
        self._caption_folder_display.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=4)
        tk.Label(settings_card, text="(set on the Start tab)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 8)
        )

        ttk.Label(settings_card, text="Trigger Word:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_trigger_var = tk.StringVar(value=self.settings.get("CAPTION_TRIGGER_WORD", ""))
        ttk.Entry(settings_card, textvariable=self.caption_trigger_var, width=40).grid(row=2, column=1, sticky=tk.W, pady=4)
        tk.Label(settings_card, text="(prepended to all captions)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=2, column=2, sticky=tk.W, padx=(10, 0)
        )

        ttk.Label(settings_card, text="Model:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        # Populated properly by _restore_caption_selection() once both vars exist.
        self.caption_model_var = tk.StringVar(value=self._initial_caption_model())
        self._caption_task_memory = {}
        self._caption_model_last = None
        self.caption_model_combo = ttk.Combobox(
            settings_card, textvariable=self.caption_model_var,
            values=self._caption_model_values(), state="readonly", width=37,
        )
        self.caption_model_combo.grid(row=3, column=1, sticky=tk.W, pady=4)
        self.caption_model_combo.bind("<<ComboboxSelected>>",
                                      lambda e: self._on_caption_model_changed())
        self.caption_model_hint_label = tk.Label(
            settings_card, text=self._qwen_captioner_hint(),
            font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.caption_model_hint_label.grid(row=3, column=2, sticky=tk.W, padx=(10, 0))
        # The Qwen3-VL entry appears as soon as the Krea 2 text encoder path is filled in on
        # Preferences — no restart. It's a captioner for ANY dataset, Klein included; the file
        # just happens to ship with the Krea 2 models.
        try:
            self.prefs_vars["krea2_text_encoder"].trace_add(
                "write", lambda *_: self._refresh_caption_model_values())
        except Exception:
            pass

        ttk.Label(settings_card, text="Task:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_task_var = tk.StringVar(value="<DETAILED_CAPTION>")
        _task_row = tk.Frame(settings_card, bg=COLORS["bg_surface"])
        _task_row.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=4)
        self.caption_task_combo = ttk.Combobox(
            _task_row, textvariable=self.caption_task_var,
            values=FLORENCE_TASKS, state="readonly", width=37,
        )
        self.caption_task_combo.pack(side=tk.LEFT)
        self.caption_task_combo.bind("<<ComboboxSelected>>",
                                     lambda e: self._on_caption_task_changed())
        self.caption_edit_instr_btn = ttk.Button(
            _task_row, text="Edit instructions…",
            command=self._open_caption_instruction_editor)
        self.caption_edit_instr_btn.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(self.caption_edit_instr_btn,
                "See and edit the exact instruction sent to the vision model for this task.\n"
                "Saving edits THIS preset — each of the four keeps its own wording, and your\n"
                "edit persists between sessions, including for the trainer's auto-recaption.\n"
                "'Restore default' puts the shipped text back.")

        ttk.Label(settings_card, text="Max Tokens:").grid(row=5, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_max_tokens_var = tk.StringVar(
            value=str(self.last_used.get("caption_max_tokens",
                                         self.settings.get("CAPTION_MAX_TOKENS", 256))))
        ttk.Entry(settings_card, textvariable=self.caption_max_tokens_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=4)
        # Without this the value only reached disk when some OTHER control happened to trigger a
        # save, so a hand-typed budget was usually lost on restart.
        self.caption_max_tokens_var.trace_add("write", lambda *_: self._save_last_used_paths())

        ttk.Checkbutton(
            settings_card, text="Overwrite existing caption files", variable=self.overwrite_captions_var,
        ).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        tk.Label(settings_card,
                 text="Untick to caption ONLY images that don't have a .txt yet — e.g. after "
                      "adding new images or face-cropping, existing captions stay untouched.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT).grid(
            row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))

        # Card 2: Generate Captions
        actions_card = self._start_section_card(outer, "Generate Captions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(anchor=tk.W)
        # Kept on self so a running job can grey them out — clicking Caption All twice used to
        # start a SECOND worker over the same files, which reads as the job having been queued.
        self.caption_all_btn = ttk.Button(action_row, text="Caption All Images (AI)",
                                          command=self.caption_all_florence)
        self.caption_all_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.caption_static_btn = ttk.Button(action_row, text="Static Caption All",
                                             command=self.generate_captions)
        self.caption_static_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.caption_stop_btn = ttk.Button(action_row, text="Stop", command=self.stop_captioning, state=tk.DISABLED)
        self.caption_stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Unload Model", command=self.unload_florence_model).pack(side=tk.LEFT)

        # Card 3: Bilingual translation
        bilingual_card = self._start_section_card(
            outer, "Bilingual Translation (English + Chinese)",
            "Translates each English caption to Chinese via Helsinki-NLP/opus-mt-en-zh (~300MB, auto-downloaded "
            "on first use) and appends as `english - chinese`. Trigger word preserved if it's the first "
            "comma-separated token. Hypothesis: dual-language signal may improve LoRA convergence — "
            "empirical test needed.",
        )
        bilingual_row = tk.Frame(bilingual_card, bg=COLORS["bg_surface"])
        bilingual_row.pack(anchor=tk.W)
        ttk.Checkbutton(
            bilingual_row, text="Skip files that already contain Chinese",
            variable=self.skip_bilingual_var,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(
            bilingual_row, text="Translate Captions in Folder",
            command=self._translate_captions_in_folder,
        ).pack(side=tk.LEFT)

        # Card 4: Find & Replace
        fr_card = self._start_section_card(
            outer, "Find & Replace",
            "Bulk-edit every `.txt` caption file in the image folder. Preview first to see which files change.",
        )
        fr_card.grid_columnconfigure(1, weight=1)

        ttk.Label(fr_card, text="Find:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.find_text_var = tk.StringVar()
        ttk.Entry(fr_card, textvariable=self.find_text_var, width=40).grid(row=0, column=1, sticky=tk.EW, pady=4)

        ttk.Label(fr_card, text="Replace:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.replace_text_var = tk.StringVar()
        ttk.Entry(fr_card, textvariable=self.replace_text_var, width=40).grid(row=1, column=1, sticky=tk.EW, pady=4)

        fr_buttons = tk.Frame(fr_card, bg=COLORS["bg_surface"])
        fr_buttons.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        ttk.Button(fr_buttons, text="Replace in All .txt Files", command=self.find_replace_in_captions).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(fr_buttons, text="Preview Changes", command=self.preview_find_replace).pack(side=tk.LEFT)

        # Card 5: Image Preview
        preview_card = self._start_section_card(
            outer, "Image Preview",
            "Browse the training folder and pick individual images to caption or inspect.",
        )

        # Voice recordings never appear in the grid — their captions are written in Gizmo's
        # audio tab, where you can hear what you are describing. This banner is how the tab
        # says so instead of silently showing fewer items than the folder holds. Text set (and
        # the label shown/hidden) per-refresh in refresh_caption_images.
        self._caption_audio_banner = tk.Label(
            preview_card, text="", font=(FONT_FAMILY, 10),
            fg=COLORS["accent"], bg=COLORS["bg_surface"],
            wraplength=760, justify=tk.LEFT)

        self.caption_grid_frame = tk.Frame(preview_card, bg=COLORS["bg_surface"])
        self.caption_grid_frame.pack(fill=tk.BOTH, expand=True)
        for _c in range(4):
            self.caption_grid_frame.columnconfigure(_c, weight=1)

        pagination_frame = tk.Frame(preview_card, bg=COLORS["bg_surface"])
        pagination_frame.pack(pady=(10, 0))
        ttk.Button(pagination_frame, text="<< Prev", command=self.caption_prev_page).pack(side=tk.LEFT, padx=(0, 8))
        self.caption_page_label = tk.Label(pagination_frame, text="Page 0 of 0",
                                           font=(FONT_FAMILY, 10),
                                           fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.caption_page_label.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(pagination_frame, text="Next >>", command=self.caption_next_page).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(pagination_frame, text="Refresh", command=self.refresh_caption_images).pack(side=tk.LEFT)

        # Card 6: Progress
        progress_card = self._start_section_card(outer, "Progress", None)
        progress_row = tk.Frame(progress_card, bg=COLORS["bg_surface"])
        progress_row.pack(fill=tk.X)
        self.caption_progress_var = tk.DoubleVar(value=0)
        self.caption_progress_bar = ttk.Progressbar(
            progress_row, variable=self.caption_progress_var, maximum=100, length=300,
        )
        self.caption_progress_bar.pack(side=tk.LEFT, padx=(0, 12))
        self.caption_progress_label = tk.Label(progress_row, text="",
                                               font=(FONT_FAMILY, 10),
                                               fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.caption_progress_label.pack(side=tk.LEFT)

        # Card 7: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)
        self.caption_log = scrolledtext.ScrolledText(
            log_card, height=10, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.caption_log.pack(fill=tk.BOTH, expand=True)

        # Apply the saved model + per-model task, then sync the task list / editor button.
        self._restore_caption_selection()

        self._add_youtube_help_button(outer, "captions")

    def browse_caption_folder_and_refresh(self):
        """Browse for caption folder and refresh image grid"""
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_var.set(folder)
            self.refresh_caption_images()

    # Video clips are training items only for MiniMax H3, and only there does the dataset glob
    # pick them up — so only there do they need captions.
    TRAINING_VIDEO_EXTENSIONS = {'.mp4'}

    @staticmethod
    def _read_middle_clip_frame(path):
        """One frame from the middle of a clip WITHOUT decoding the whole file. cv2 seeks the
        container to the midpoint and decodes from the nearest keyframe — milliseconds,
        against read_frames' full decode of every frame at native resolution (seconds per
        clip, and it was running on the GUI thread at every Captions tab refresh — the tab
        froze for the sum of it, Peter). None on any failure; the caller falls back."""
        try:
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            try:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
                ok, frame = cap.read()
                if not ok and n > 1:           # an odd container refused the seek — frame 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                return Image.fromarray(frame[:, :, ::-1])      # BGR -> RGB
            finally:
                cap.release()
        except Exception:
            return None

    def _open_training_frame(self, path):
        """A PIL image for any training item. A video clip gives up its middle frame.

        Clips are training items like every other, so they need a caption, a thumbnail and a face
        score like every other — and the middle frame is the fairest single representative of
        one. There is deliberately NO still written beside a clip on disk to serve this: the
        latent cache keys on the filename stem with the extension stripped, so a sidecar
        walk_03.png would land on walk_03.mp4's own cache file and one would silently overwrite
        the other.

        Clip frames are cached by (path, mtime) at a bounded size, so a Captions tab revisit
        costs nothing. The returned image is always a COPY — callers thumbnail() it in place,
        which would shrink the cached original for everyone after them. The clip's true
        resolution rides along as `fizgig_source_size`, because the cached frame is capped and
        a resolution label lying about the file would be worse than the wait was.
        """
        from PIL import Image
        if os.path.splitext(path)[1].lower() not in self.TRAINING_VIDEO_EXTENSIONS:
            return Image.open(path)
        cache = getattr(self, "_clip_frame_cache", None)
        if cache is None:
            cache = self._clip_frame_cache = {}
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            key = (path, 0)
        hit = cache.get(key)
        if hit is None:
            img = self._read_middle_clip_frame(path)
            if img is None:                    # cv2 refused the file — the slow, sure way
                from fizgig.minimax.clip import read_frames
                frames = read_frames(path)
                img = Image.fromarray(frames[len(frames) // 2])
            true_size = img.size
            img.thumbnail((1280, 1280), Image.LANCZOS)
            for k in [k for k in cache if k[0] == path]:      # a re-exported clip re-reads
                del cache[k]
            while len(cache) >= 64:            # bounded: ~2-3 MB a frame, oldest out first
                del cache[next(iter(cache))]
            cache[key] = hit = (img, true_size)
        frame, true_size = hit
        out = frame.copy()
        out.fizgig_source_size = true_size
        return out

    TRAINING_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a'}

    def _count_training_audio_files(self):
        """Voice recordings in the training folder — MiniMax only, 0 elsewhere."""
        folder = self.image_folder_var.get()
        if not folder or not os.path.isdir(folder) or not self._is_minimax_arch():
            return 0
        return sum(1 for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in self.TRAINING_AUDIO_EXTENSIONS)

    def _refresh_audio_only_ui(self, *_a):
        """Grey the image-shaped training controls when the dataset is voice recordings only.

        Only what is STRUCTURALLY meaningless goes grey: Target Megapixels (no pixels to
        size) and reference distillation (the teacher pairs photographs — with none, there is
        nothing to learn identity from). Schedule and LR controls stay live: the audio stream
        trains on the noise schedule like everything else. Disabled, not hidden — the user
        should see the controls exist and read why they are off.
        """
        audio_only = self._training_folder_audio_only()
        state = "disabled" if audio_only else "normal"
        try:
            if hasattr(self, "_mp_combo"):
                self._mp_combo.configure(state=state)
                if audio_only:
                    self._mp_audio_note.pack(side=tk.LEFT, padx=(8, 0))
                else:
                    self._mp_audio_note.pack_forget()
            if hasattr(self, "_minimax_distill_frame"):
                for w in self._minimax_distill_frame.winfo_children():
                    try:
                        w.configure(state=state)
                    except tk.TclError:
                        pass
                if audio_only and self.minimax_distill_var.get():
                    self.minimax_distill_var.set(False)
            # The voice-structure hint: ANY audio in the dataset (mixed counts too — its voice
            # steps benefit the same), family is MiniMax, and the structure is not already
            # Likeness. A/B tested: voices train much faster there than at Model default.
            _has_audio = self._is_minimax_arch() and self._count_training_audio_files() > 0
            if hasattr(self, "_minimax_structure_voice_note"):
                _wants_note = (_has_audio
                               and not str(self.minimax_structure_var.get()).startswith(
                                   "Likeness"))
                if _wants_note:
                    self._minimax_structure_voice_note.grid(
                        row=27, column=0, columnspan=3, sticky=tk.W, padx=(12, 5), pady=(0, 4))
                else:
                    self._minimax_structure_voice_note.grid_remove()
            # Per-category retirement rows: only when the dataset is genuinely MIXED — with
            # one category there is nothing to finish separately.
            if hasattr(self, "_mixed_stop_label"):
                _mixed = _has_audio and not audio_only
                if _mixed:
                    self._mixed_stop_label.grid(row=28, column=0, sticky=tk.W, padx=5,
                                                pady=(8, 2))
                    self._mixed_stop_frame.grid(row=28, column=1, columnspan=2, sticky=tk.W,
                                                padx=5, pady=(8, 2))
                    self._mixed_stop_hint.grid(row=29, column=0, columnspan=3, sticky=tk.W,
                                               padx=(12, 5), pady=(0, 4))
                else:
                    self._mixed_stop_label.grid_remove()
                    self._mixed_stop_frame.grid_remove()
                    self._mixed_stop_hint.grid_remove()
        except tk.TclError:
            pass

    def _training_folder_audio_only(self):
        """True when the training folder holds voice recordings and nothing visual — the state
        in which image-shaped controls (sizing, bucketing, face teachers) mean nothing."""
        folder = self.image_folder_var.get().strip() if hasattr(self, "image_folder_var") else ""
        if not folder or not os.path.isdir(folder) or not self._is_minimax_arch():
            return False
        if not self._count_training_audio_files():
            return False
        try:
            from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
            visual = {e.lower() for e in IMAGE_EXTENSIONS} | {".mp4"}
            return not any(os.path.splitext(f)[1].lower() in visual
                           for f in os.listdir(folder))
        except OSError:
            return False

    def get_caption_image_files(self):
        """Get list of image files in caption folder"""
        folder = self.image_folder_var.get()
        if not folder or not os.path.isdir(folder):
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        if self._is_minimax_arch():
            image_extensions = image_extensions | self.TRAINING_VIDEO_EXTENSIONS
        images = []

        for filename in os.listdir(folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                images.append(os.path.join(folder, filename))

        return sorted(images)

    def refresh_caption_images(self):
        """Refresh the image grid with thumbnails"""
        # Mark as loaded for this folder
        self.caption_images_loaded = True

        # Clear existing thumbnails
        for widget in self.caption_grid_frame.winfo_children():
            widget.destroy()
        self.caption_thumbnails.clear()

        images = self.get_caption_image_files()
        total_images = len(images)

        # Audio never enters the grid; count it so the tab explains itself rather than showing
        # fewer items than the folder holds. Wordings per Peter: all-audio vs mixed.
        _n_audio = self._count_training_audio_files()
        if _n_audio and not total_images:
            self._caption_audio_banner.config(
                text="🎙 Audio-only training set — captions are written in Gizmo's audio tab.")
            self._caption_audio_banner.pack(anchor=tk.W, pady=(0, 8),
                                            before=self.caption_grid_frame)
        elif _n_audio:
            self._caption_audio_banner.config(
                text=f"🎙 {_n_audio} audio file(s) in this training set — their captions are "
                     f"handled in Gizmo; the grid below shows only images and clips.")
            self._caption_audio_banner.pack(anchor=tk.W, pady=(0, 8),
                                            before=self.caption_grid_frame)
        else:
            self._caption_audio_banner.pack_forget()

        total_pages = max(1, (total_images + self.images_per_page - 1) // self.images_per_page)

        # Clamp current page
        self.current_caption_page = min(self.current_caption_page, total_pages - 1)
        self.current_caption_page = max(0, self.current_caption_page)

        # Update page label
        self.caption_page_label.config(text=f"Page {self.current_caption_page + 1} of {total_pages} ({total_images} images)")

        # Get images for current page
        start_idx = self.current_caption_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, total_images)
        page_images = images[start_idx:end_idx]

        # Create image cards in a grid (4 columns)
        for i, img_path in enumerate(page_images):
            row_idx = i // 4
            col_idx = i % 4
            self.create_caption_image_card(img_path, row_idx, col_idx)

    def create_caption_image_card(self, img_path, row, col):
        """Create an image card with thumbnail and caption"""
        card_frame = ttk.Frame(self.caption_grid_frame, relief="solid", borderwidth=1)
        card_frame.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)

        # Edit + Remove pinned to the card's BOTTOM edge, packed first for priority. Cards in
        # a grid row share the tallest card's height (sticky=NSEW), so pinning puts every
        # row's buttons on the same level regardless of thumbnail shape.
        btn_row = tk.Frame(card_frame)
        btn_row.pack(side=tk.BOTTOM, pady=(2, 6))
        ttk.Button(btn_row, text="Edit",
                   command=lambda p=img_path: self.show_edit_caption_dialog(p)).pack(side=tk.LEFT, padx=(0, 4))
        rm_btn = ttk.Button(btn_row, text="Remove",
                            command=lambda p=img_path: self.remove_caption_image(p))
        rm_btn.pack(side=tk.LEFT)
        ToolTip(rm_btn, "Move this image + its caption to a 'removed' subfolder — nothing is\n"
                        "deleted, so it's easy to undo. Use after Image Prep to cull face\n"
                        "close-ups that came out soft or blurry.")

        # Create thumbnail (original resolution captured before thumbnail() shrinks it)
        img_res = None
        try:
            with self._open_training_frame(img_path) as img:
                img_res = getattr(img, "fizgig_source_size", img.size)
                img.thumbnail((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.caption_thumbnails[img_path] = photo  # Keep reference

                img_label = ttk.Label(card_frame, image=photo)
                img_label.pack(padx=5, pady=5)
        except Exception as e:
            ttk.Label(card_frame, text="Error loading image").pack(padx=5, pady=5)

        # Filename + original resolution — the res is what you're eyeballing for (tiny face
        # crops read as "(180×240)" here long before the blur is obvious in a 150px thumb).
        filename = os.path.basename(img_path)
        name_label = ttk.Label(card_frame, text=filename[:30] + "..." if len(filename) > 30 else filename)
        name_label.pack()
        if img_res:
            ttk.Label(card_frame, text=f"({img_res[0]}×{img_res[1]})",
                      foreground=COLORS["text_muted"]).pack()

        # Load and display caption if exists — wrapped to the cell's usable width
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                caption_preview = caption[:110] + "..." if len(caption) > 110 else caption
                caption_label = ttk.Label(card_frame, text=caption_preview, wraplength=270,
                                          justify=tk.LEFT, foreground=COLORS["text_secondary"])
                caption_label.pack(fill=tk.X, padx=8, pady=2)
            except Exception:
                pass
        else:
            ttk.Label(card_frame, text="[No caption]", foreground=COLORS["warning"]).pack(pady=2)

    def remove_caption_image(self, img_path):
        """Move an image + its caption .txt to <folder>/removed/ — the never-delete pattern
        ('originals', 'excluded_by_look') applied to manual culling. Subfolders aren't globbed
        by the dataset builder, so removed files simply stop being training data."""
        folder = os.path.dirname(img_path)
        dest_dir = os.path.join(folder, "removed")
        try:
            os.makedirs(dest_dir, exist_ok=True)
            for p in (img_path, os.path.splitext(img_path)[0] + ".txt"):
                if not os.path.exists(p):
                    continue
                dest = os.path.join(dest_dir, os.path.basename(p))
                stem, ext = os.path.splitext(dest)
                n = 1
                while os.path.exists(dest):
                    dest = f"{stem}_{n}{ext}"
                    n += 1
                os.rename(p, dest)
        except OSError as e:
            messagebox.showerror("Remove failed", f"Could not move the file:\n{e}")
            return
        self.refresh_caption_images()

    def caption_prev_page(self):
        """Go to previous page of images"""
        if self.current_caption_page > 0:
            self.current_caption_page -= 1
            self.refresh_caption_images()

    def caption_next_page(self):
        """Go to next page of images"""
        images = self.get_caption_image_files()
        total_pages = max(1, (len(images) + self.images_per_page - 1) // self.images_per_page)
        if self.current_caption_page < total_pages - 1:
            self.current_caption_page += 1
            self.refresh_caption_images()

    def show_edit_caption_dialog(self, img_path):
        """Live caption editor: no Save button, no confirmation popups. Edits save themselves
        when you navigate or close; ◀ ▶ (or Ctrl+←/→ — plain arrows stay cursor movement)
        walk the whole folder without leaving the window. Shaped by user feedback: the old
        flow was nine clicks per image and people were editing captions in other apps."""
        folder = os.path.dirname(img_path)
        from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
        _exts = {e.lower() for e in IMAGE_EXTENSIONS}
        try:
            files = sorted(f for f in os.listdir(folder)
                           if os.path.splitext(f)[1].lower() in _exts)
        except Exception:
            files = [os.path.basename(img_path)]
        state = {"path": img_path, "loaded": "", "dirty_grid": False, "speech_busy": False}

        dialog = tk.Toplevel(self.master)
        dialog.configure(bg=BG_COLOR)
        # No fixed geometry: content height varies (a portrait thumbnail is up to 300 px tall).
        # The button row packs side=BOTTOM *first* so it can never be clipped off the edge.
        dialog.minsize(600, 380)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        status = tk.Label(dialog, text="Edits save automatically when you move to another "
                                       "image or close.", font=(FONT_FAMILY, 9),
                          fg=COLORS["text_explain"], bg=BG_COLOR)
        status.pack(side=tk.BOTTOM, pady=(0, 2))

        img_label = ttk.Label(dialog)
        img_label.pack(pady=(10, 2))
        size_label = ttk.Label(dialog, foreground=COLORS["text_muted"])
        size_label.pack()
        ttk.Label(dialog, text="Caption:").pack(anchor=tk.W, padx=10)
        caption_text = tk.Text(dialog, height=5, width=60, bg=COLORS["bg_surface"],
                               fg=COLORS["text_primary"], font=(FONT_FAMILY, 10), wrap="word",
                               insertbackground=COLORS["text_primary"])
        caption_text.pack(padx=10, pady=5, fill=tk.X)

        def _cap_path():
            return os.path.splitext(state["path"])[0] + ".txt"

        def _load(path):
            state["path"] = path
            dialog.title(f"Edit Caption — {os.path.basename(path)}"
                         + (f"   ({files.index(os.path.basename(path)) + 1} / {len(files)})"
                            if os.path.basename(path) in files else ""))
            try:
                with self._open_training_frame(path) as img:
                    _w, _h = getattr(img, "fizgig_source_size", img.size)
                    img.thumbnail((300, 300), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    img_label.configure(image=photo)
                    img_label.image = photo
                    size_label.configure(text=f"{_w}×{_h} px")
            except Exception:
                img_label.configure(image="")
                img_label.image = None
                size_label.configure(text="(image preview unavailable)")
            caption = ""
            if os.path.exists(_cap_path()):
                try:
                    with open(_cap_path(), 'r', encoding='utf-8-sig', errors="replace") as f:
                        caption = f.read().strip()
                except Exception:
                    pass
            caption_text.delete("1.0", tk.END)
            caption_text.insert("1.0", caption)
            state["loaded"] = caption
            _speech_refresh()               # video with sound → the Append Speech button shows

        def _save():
            """Write the caption if it changed. Silent and inline — never a popup."""
            text = caption_text.get("1.0", tk.END).strip()
            if text == state["loaded"]:
                return
            if not text:
                status.config(fg="#E74C3C",
                              text="Caption box is empty — not saved (the previous caption "
                                   "is kept). Type something or move on.")
                return
            try:
                with open(_cap_path(), 'w', encoding='utf-8') as f:
                    f.write(text)
                state["loaded"] = text
                state["dirty_grid"] = True
                status.config(fg="#2ECC71", text=f"Saved ✓  {os.path.basename(_cap_path())}")
            except Exception as e:
                status.config(fg="#E74C3C", text=f"Save failed: {e}")

        def _nav(step):
            _save()
            base = os.path.basename(state["path"])
            if base not in files or len(files) < 2:
                return
            _load(os.path.join(folder, files[(files.index(base) + step) % len(files)]))

        def _close():
            _save()
            if state["dirty_grid"]:
                self.refresh_caption_images()   # once, on close — not per save (scroll reset)
            dialog.destroy()

        def regenerate():
            _save()
            dialog.destroy()
            self.caption_single_image(state["path"])

        # --- Append Speech: Whisper the clip's audio into the caption -------------------------
        # Only for a training VIDEO that isn't muted — where "muted" is Gizmo's _mute filename
        # convention, the same one the trainer reads. Reuses Gizmo's machinery wholesale: its
        # ffmpeg finder, its Whisper model (local-first — the Preferences downloader pre-fetches
        # it; otherwise a one-time ~300 MB download, exactly like Gizmo), its language
        # preference, and its hallucination-loop detector. Un-captioned speech is a lie the
        # model must explain away — this is the one-click fix for clips that never went
        # through Gizmo.
        def _speech_refresh():
            # Muted is a FILENAME convention, not a stream probe: Gizmo exports silent clips
            # as <stem>_NN_mute.mp4 (MUTE_SUFFIX) and that name is the whole contract.
            path = state["path"]
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            if (os.path.splitext(path)[1].lower() in self.TRAINING_VIDEO_EXTENSIONS
                    and not stem.endswith("_mute")):
                speech_btn.pack(side=tk.LEFT, padx=5, before=close_btn)
            else:
                speech_btn.pack_forget()

        def _append_speech():
            if state["speech_busy"]:
                return
            state["speech_busy"] = True
            path = state["path"]
            import gizmo as _gz
            speech_btn.configure(state=tk.DISABLED, text="⏳ Transcribing…")
            _warm = _gz.local_whisper_dir() or hasattr(self, "_caption_whisper_pipe")
            status.config(fg=COLORS["text_explain"],
                          text="Transcribing…" if _warm else
                               "Transcribing — first use downloads Whisper (~300 MB), "
                               "please wait…")

            def worker():
                text, err = None, None
                try:
                    import tempfile
                    import wave as _wave
                    import numpy as _np
                    _ff = _gz.find_ffmpeg()
                    if not _ff:
                        raise RuntimeError("ffmpeg not found")
                    wav = os.path.join(tempfile.gettempdir(), "fizgig_caption_whisper.wav")
                    p = _gz._run([_ff, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                                  "-vn", "-ac", "1", "-ar", "16000", "-t", "120",
                                  "-c:a", "pcm_s16le", wav])
                    if p.returncode != 0:
                        raise RuntimeError("could not extract the clip's audio")
                    with _wave.open(wav, "rb") as w:
                        frames = w.readframes(w.getnframes())
                        span = w.getnframes() / float(w.getframerate() or 16000)
                    samples = _np.frombuffer(frames, dtype=_np.int16).astype(_np.float32) / 32768.0
                    if not hasattr(self, "_caption_whisper_pipe"):
                        from transformers import pipeline
                        self._caption_whisper_pipe = pipeline(
                            "automatic-speech-recognition",
                            model=_gz.local_whisper_dir() or _gz._WHISPER_REPO, device=-1)
                    # Gizmo's saved language preference; English when the user never chose one —
                    # auto-detect's wrong guess on a short clip is the classic loop trigger, so
                    # the unset default here is deliberately NOT auto.
                    lang = None
                    try:
                        with open(_gz.SETTINGS_FILE, encoding="utf-8") as f:
                            lang = json.load(f).get("whisper_language")
                    except Exception:
                        pass
                    lang = lang or "English"
                    lang = None if lang == "Auto detect" else lang.lower()

                    def hear(language):
                        kw = ({"generate_kwargs": {"language": language, "task": "transcribe"}}
                              if language else {})
                        return (self._caption_whisper_pipe(
                            {"array": samples, "sampling_rate": 16000},
                            chunk_length_s=30, **kw).get("text") or "").strip()

                    text = hear(lang)
                    if lang is None and _gz.Gizmo._whisper_degenerate(text, span):
                        text = hear("english")
                    if _gz.Gizmo._whisper_degenerate(text, span):
                        text = None
                        err = ("Whisper looped on this clip — it hallucinated repeating text. "
                               "Pick the language in Gizmo's ⚙ settings, or type the words.")
                except Exception as exc:
                    err = f"Whisper could not run: {exc}"
                self.master.after(0, lambda: _speech_done(path, text, err))

            def _speech_done(for_path, text, err):
                state["speech_busy"] = False
                if not dialog.winfo_exists():
                    return
                speech_btn.configure(state=tk.NORMAL, text="🎤 Append Transcription")
                if state["path"] != for_path:
                    status.config(fg=COLORS["text_explain"],
                                  text="Transcript arrived after you moved on — discarded.")
                    return
                if err:
                    status.config(fg="#E74C3C", text=err)
                    return
                if not text:
                    status.config(fg=COLORS["text_explain"],
                                  text="Whisper heard no words in this clip.")
                    return
                # Gizmo's caption grammar: <description> saying "…" — the form the trainer's
                # doctrine expects. The words land in the box for editing; saves on move/close.
                cap = caption_text.get("1.0", tk.END).strip().rstrip(".")
                caption_text.delete("1.0", tk.END)
                caption_text.insert("1.0", (cap + " " if cap else "") + f'saying "{text}"')
                status.config(fg="#2ECC71",
                              text="Transcript appended — edit freely; saves when you move on.")

            threading.Thread(target=worker, daemon=True).start()

        ttk.Button(btn_frame, text="◀ Prev", command=lambda: _nav(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Next ▶", command=lambda: _nav(+1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Regenerate (AI)", command=regenerate).pack(side=tk.LEFT, padx=(20, 5))
        speech_btn = ttk.Button(btn_frame, text="🎤 Append Transcription", command=_append_speech)
        close_btn = ttk.Button(btn_frame, text="Close", command=_close)
        close_btn.pack(side=tk.LEFT, padx=5)
        # Ctrl+arrows so plain arrows keep moving the text cursor while typing.
        dialog.bind("<Control-Left>", lambda e: (_nav(-1), "break")[1])
        dialog.bind("<Control-Right>", lambda e: (_nav(+1), "break")[1])
        dialog.protocol("WM_DELETE_WINDOW", _close)
        _load(img_path)

    def unload_florence_model(self, silent=False):
        """Stop the caption worker subprocess and release its GPU memory."""
        if getattr(self, "_captioning_running", False):
            self.update_caption_log("Captioning is still running — press Stop first.\n")
            if not silent:
                messagebox.showinfo(
                    "Captioning in progress",
                    "A captioning job is still running.\n\nPress Stop and wait for it to finish.")
            return
        proc = getattr(self, "caption_process", None)
        if proc is None or proc.poll() is not None:
            if not silent:
                messagebox.showinfo("Info", "No caption model is loaded.")
            return
        self.update_caption_log("Unloading caption model...\n")

        def _bg_unload():
            self._stop_caption_worker(silent=True, wait=True)
            self.master.after(0, lambda: self._finish_caption_unload(silent))

        threading.Thread(target=_bg_unload, daemon=True).start()

    def _finish_caption_unload(self, silent: bool) -> None:
        vram = self._read_vram()
        msg = "Caption model unloaded."
        if vram:
            used, tot = vram
            msg += f"\n\nDevice VRAM {used / 1e9:.1f} / {tot / 1e9:.1f} GB"
        self.update_caption_log(msg + "\n")
        if not silent:
            messagebox.showinfo("Caption model unloaded", msg)

    def _reset_caption_worker_state(self) -> None:
        self.caption_process = None
        self._caption_worker_stdin = None
        self._caption_worker_key = None
        self._caption_worker_warm = False
        self._caption_worker_ready.clear()

    def _caption_job_dir(self) -> str:
        cache = ""
        try:
            cache = self.prefs_vars["cache_dir"].get().strip()
        except Exception:
            pass
        if not cache:
            cache = os.path.join(_FIZGIG_DIR, "cache")
        d = os.path.join(cache, "_caption_job")
        os.makedirs(d, exist_ok=True)
        return d

    def _caption_script_path(self) -> str:
        return os.path.join(_FIZGIG_DIR, "src", "fizgig", "scripts", "batch_caption.py")

    def _caption_worker_config_key(self) -> tuple:
        if self._is_qwen_captioner():
            return ("qwen", self._qwen_captioner_path() or "", bool(self._is_minimax_arch()))
        return ("florence", self.caption_model_var.get(), bool(self._is_minimax_arch()))

    def _write_caption_worker_config(self) -> str:
        job_dir = self._caption_job_dir()
        config_path = os.path.join(job_dir, "worker_config.json")
        config = {
            "backend": "qwen" if self._is_qwen_captioner() else "florence",
            "include_video": self._is_minimax_arch(),
        }
        if config["backend"] == "qwen":
            config["text_encoder"] = self._qwen_captioner_path()
        else:
            model_name = self.caption_model_var.get()
            config["florence_model"] = model_name
            rev = FLORENCE_REVISIONS.get(model_name)
            code_rev = FLORENCE_CODE_REVISIONS.get(model_name)
            if rev:
                config["florence_revision"] = rev
            if code_rev:
                config["florence_code_revision"] = code_rev
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return config_path

    def _write_caption_job(self, images: list[str]) -> tuple[str, str]:
        job_dir = self._caption_job_dir()
        list_file = os.path.join(job_dir, "images.txt")
        instr_file = os.path.join(job_dir, "instruction.txt")
        job_path = os.path.join(job_dir, "job.json")
        stop_file = os.path.join(job_dir, "stop")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in images:
                f.write(p + "\n")
        try:
            os.remove(stop_file)
        except OSError:
            pass
        try:
            max_tokens = int(self.caption_max_tokens_var.get())
        except (ValueError, tk.TclError):
            max_tokens = 120
        trigger = self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else ""
        job = {
            "list_file": list_file,
            "max_new_tokens": max_tokens,
            "trigger": trigger,
            "stop_file": stop_file,
        }
        if self._is_qwen_captioner():
            te = self._qwen_captioner_path()
            if not te:
                raise FileNotFoundError("Qwen3-VL text encoder path is not set or not found")
            with open(instr_file, "w", encoding="utf-8") as f:
                f.write(self._resolve_caption_instruction())
            job["instruction_file"] = instr_file
        else:
            job["florence_task"] = self.caption_task_var.get()
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f)
        return job_path, stop_file

    def _caption_worker_env(self) -> dict:
        env = self._cuda_env_for_subprocess(os.environ.copy())
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _start_caption_worker_reader(self, proc) -> None:
        def reader():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._handle_caption_subprocess_line(line.rstrip("\r\n"))
            finally:
                rc = proc.wait()
                self.master.after(0, lambda r=rc, p=proc: self._on_caption_worker_exit(r, p))

        threading.Thread(target=reader, daemon=True).start()

    def _caption_worker_alive(self) -> bool:
        proc = getattr(self, "caption_process", None)
        try:
            return proc is not None and proc.poll() is None
        except Exception:
            return False

    def _stop_caption_worker(self, silent: bool = False, wait: bool = True,
                             graceful: bool = True) -> None:
        import subprocess

        proc = getattr(self, "caption_process", None)
        if proc is None or proc.poll() is not None:
            self._reset_caption_worker_state()
            return

        stdin = getattr(self, "_caption_worker_stdin", None)
        self._caption_worker_stdin = None
        self._caption_worker_warm = False
        self._caption_worker_key = None
        self._caption_worker_ready.clear()
        self.caption_process = None

        if wait:
            if graceful:
                try:
                    if stdin is not None:
                        stdin.write("QUIT\n")
                        stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            else:
                try:
                    if stdin is not None:
                        stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
        else:
            try:
                if stdin is not None:
                    stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass

            def _reap(p):
                try:
                    p.wait(timeout=30)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

            threading.Thread(target=_reap, args=(proc,), daemon=True).start()

    def _stop_caption_worker_async(self, callback, *, graceful: bool = True) -> None:
        """Stop the caption worker on a background thread, then run callback on the UI thread."""
        if not self._caption_worker_alive():
            self._reset_caption_worker_state()
            self.master.after(0, callback)
            return

        def _bg():
            try:
                self._stop_caption_worker(silent=True, wait=True, graceful=graceful)
            except Exception:
                pass
            self.master.after(0, callback)

        threading.Thread(target=_bg, daemon=True).start()

    def _ensure_caption_worker(self) -> bool:
        import subprocess

        key = self._caption_worker_config_key()
        proc = getattr(self, "caption_process", None)
        if (
            proc is not None
            and proc.poll() is None
            and self._caption_worker_warm
            and self._caption_worker_key == key
        ):
            return True

        self._stop_caption_worker(silent=True)

        try:
            config_path = self._write_caption_worker_config()
        except FileNotFoundError:
            raise

        cmd = [
            self._venv_python(),
            self._caption_script_path(),
            "--serve",
            "--config", config_path,
        ]
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid

        self._caption_worker_ready.clear()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            env=self._caption_worker_env(),
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        self.caption_process = proc
        self._caption_worker_stdin = proc.stdin
        self._caption_worker_key = key
        self._start_caption_worker_reader(proc)

        # The exit handler SETS the event too (with warm still False), so a worker that
        # dies during model load — bad path, missing download — fails this promptly
        # instead of blocking the full 600 s and popping a stale error dialog later.
        if not self._caption_worker_ready.wait(timeout=600) or not self._caption_worker_warm:
            self._stop_caption_worker(silent=True)
            return False
        return True

    def _handle_caption_subprocess_line(self, line: str) -> None:
        line = (line or "").strip()
        if not line:
            return
        if line == "READY":
            self._caption_worker_warm = True
            self._caption_worker_ready.set()
            return
        if line == "DONE":
            self.master.after(0, self._on_caption_job_done)
            return
        if line == "STOPPED":
            self.master.after(0, lambda: self._on_caption_job_done(stopped=True))
            return
        if line.startswith("PROGRESS:"):
            try:
                _tag, cur_s, tot_s = line.split(None, 2)
                cur, tot = int(cur_s), int(tot_s)
            except (ValueError, IndexError):
                return
            self.master.after(0, lambda p=(cur / tot) * 100.0 if tot else 0.0, c=cur, t=tot:
                              self.update_caption_progress(p, c, t))
        elif line.startswith("OK:"):
            self.master.after(0, lambda n=line[3:].strip(): self.update_caption_log(f"✓ {n}\n"))
        elif line.startswith("FAIL:"):
            self.master.after(0, lambda d=line[5:].strip(): self.update_caption_log(f"✗ {d}\n"))
        elif line.startswith("INFO:"):
            self.master.after(0, lambda m=line[5:].strip(): self.update_caption_log(f"{m}\n"))

    def _on_caption_job_done(self, stopped: bool = False) -> None:
        self._captioning_running = False
        self.caption_stop_btn.configure(state=tk.DISABLED)
        self._set_caption_buttons_running(False)
        stop_file = getattr(self, "_caption_stop_file", "")
        if stop_file:
            try:
                os.remove(stop_file)
            except OSError:
                pass
        self._caption_stop_file = ""
        if stopped:
            self.update_caption_log("\nCaptioning stopped.\n")
        else:
            self.update_caption_log("\nCaptioning complete!\n")
        self.refresh_caption_images()

    def _on_caption_worker_exit(self, returncode: int, proc=None) -> None:
        # A worker we DELIBERATELY replaced (model switch: _ensure_caption_worker stops the
        # old one, which nulls caption_process before the exit lands) must not be mistaken
        # for the current worker dying — that cleared _captioning_running out from under the
        # brand-new job, whose send then silently no-opped: Qwen loaded and sat idle
        # (field, first NVIDIA test). Only the CURRENT worker's exit means anything.
        if proc is not None and proc is not getattr(self, "caption_process", None):
            return
        was_running = getattr(self, "_captioning_running", False)
        self.caption_process = None
        self._caption_worker_stdin = None
        self._caption_worker_key = None
        self._caption_worker_warm = False
        # SET, not clear: release any thread blocked in _ensure_caption_worker's ready-wait
        # so a load failure reports immediately (warm stays False, which is the signal).
        # The next _ensure_caption_worker clears it via _stop_caption_worker before spawning.
        self._caption_worker_ready.set()
        if was_running:
            self._on_caption_job_done(stopped=True)
            self.update_caption_log(
                f"Caption worker exited unexpectedly (code {returncode}).\n")
        elif returncode not in (0, None):
            self.update_caption_log(f"Caption worker exited (code {returncode}).\n")

    def _run_ai_caption_subprocess(self, images: list[str]) -> None:
        if not images:
            return
        proc = getattr(self, "caption_process", None)
        warm = (
            proc is not None
            and proc.poll() is None
            and self._caption_worker_warm
            and self._caption_worker_key == self._caption_worker_config_key()
        )
        try:
            job_path, stop_file = self._write_caption_job(images)
        except FileNotFoundError as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.captioning_stop_flag = False
        self._captioning_running = True
        self.caption_stop_btn.configure(state=tk.NORMAL)
        self._set_caption_buttons_running(True)
        self._caption_stop_file = stop_file
        if warm:
            self.update_caption_log(f"Captioning {len(images)} image(s)...\n")
        else:
            self.update_caption_log(
                f"Starting AI captioning of {len(images)} image(s) (separate process)...\n")

        def _ensure_and_run():
            try:
                if not self._ensure_caption_worker():
                    self.master.after(0, lambda: self._on_caption_worker_start_failed())
                    return
                self.master.after(0, lambda: self._send_caption_job(job_path))
            except Exception as exc:
                self.master.after(0, lambda e=exc: self._on_caption_worker_start_failed(str(e)))

        threading.Thread(target=_ensure_and_run, daemon=True).start()

    def _on_caption_worker_start_failed(self, detail: str = "") -> None:
        self._captioning_running = False
        self.caption_stop_btn.configure(state=tk.DISABLED)
        self._set_caption_buttons_running(False)
        self._caption_stop_file = ""
        msg = "Could not start the caption worker or load the model."
        if detail:
            msg += f"\n\n{detail}"
        msg += "\nCheck the caption log for details."
        self.update_caption_log(msg + "\n")
        messagebox.showerror("Caption worker failed", msg)
        self._stop_caption_worker(silent=True)

    def _send_caption_job(self, job_path: str) -> None:
        if not getattr(self, "_captioning_running", False):
            return
        try:
            assert self._caption_worker_stdin is not None
            self._caption_worker_stdin.write(f"RUN {job_path}\n")
            self._caption_worker_stdin.flush()
        except Exception as exc:
            self._captioning_running = False
            self._set_caption_buttons_running(False)
            messagebox.showerror("Caption worker error", str(exc))
            self._stop_caption_worker(silent=True)

    # === Bilingual translation (Qwen3-8B chat) ===

    def _has_cjk(self, text: str) -> bool:
        """Heuristic: does the text contain CJK Unified Ideographs?"""
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _load_translator(self):
        """Lazy-load Helsinki-NLP/opus-mt-en-zh for English→Chinese translation.

        First-time use downloads ~300MB from HuggingFace to the standard HF cache.
        We use this dedicated MT model (not Klein's Qwen3) because Klein's distributed
        text_encoder safetensors strips the LM head — it can extract hidden states for
        Klein's training but cannot do generation reliably.
        """
        if getattr(self, "_translator_model", None) is not None:
            return True
        try:
            self.update_caption_log("Loading translator (Helsinki-NLP/opus-mt-en-zh)...\n")
            import torch as _torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            from fizgig.utils.hf_cache import from_pretrained_cache_first
            model_id = "Helsinki-NLP/opus-mt-en-zh"
            self._translator_tokenizer = from_pretrained_cache_first(AutoTokenizer, model_id)
            self._translator_model = from_pretrained_cache_first(
                AutoModelForSeq2SeqLM,
                model_id,
                torch_dtype=_torch.float16 if device == "cuda" else _torch.float32,
            ).to(device)
            self._translator_model.eval()
            self._translator_device = device
            self.update_caption_log(f"Translator loaded on {device}.\n")
            return True
        except Exception as e:
            self.update_caption_log(f"Failed to load translator: {e}\n")
            messagebox.showerror(
                "Error",
                f"Failed to load Helsinki-NLP/opus-mt-en-zh:\n{e}\n\n"
                "First-time use needs internet to download ~300MB from HuggingFace."
            )
            self._translator_model = None
            self._translator_tokenizer = None
            return False

    def _unload_translator(self):
        """Free VRAM after batch translation."""
        if getattr(self, "_translator_model", None) is not None:
            import torch as _torch
            del self._translator_model
            del self._translator_tokenizer
            self._translator_model = None
            self._translator_tokenizer = None
            _torch.cuda.empty_cache()
            self.update_caption_log("Translator unloaded.\n")

    def _translate_to_chinese(self, english: str) -> str:
        """Single-string EN→ZH translation via Helsinki MT. Returns Chinese only, or '' on failure."""
        if not english.strip():
            return ""
        import torch as _torch
        inputs = self._translator_tokenizer(
            english.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._translator_device)
        with _torch.no_grad():
            outputs = self._translator_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )
        translation = self._translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = translation.strip().strip('"').strip("'").strip()
        translation = " ".join(translation.split())
        return translation

    def _translate_captions_in_folder(self):
        """Threaded entry point for the 'Translate Captions in Folder' button."""
        folder = self.image_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Pick a valid caption folder first.")
            return
        import threading
        threading.Thread(target=self._translate_captions_worker, args=(folder,), daemon=True).start()

    def _translate_captions_worker(self, folder: str):
        """Background worker: load Qwen3, walk folder, translate each .txt, write back."""
        self._translating = True
        import glob
        # glob.escape: a dataset folder containing [brackets] is a glob character class,
        # so an unescaped pattern silently matches nothing.
        files = sorted(glob.glob(os.path.join(glob.escape(folder), "*.txt")))
        if not files:
            self.master.after(0, lambda: self.update_caption_log(f"No .txt files in {folder}\n"))
            return
        self.master.after(0, lambda: self.update_caption_log(
            f"\n=== Bilingual translation: {len(files)} files in {folder} ===\n"
        ))
        if not self._load_translator():
            return
        skip_existing = self.skip_bilingual_var.get()
        translated = 0
        skipped = 0
        failed = 0
        try:
            for i, fp in enumerate(files):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        original = f.read().strip()
                    if not original:
                        skipped += 1
                        continue
                    if skip_existing and self._has_cjk(original):
                        skipped += 1
                        self.master.after(0, lambda fp=fp: self.update_caption_log(
                            f"  [skip already-bilingual] {os.path.basename(fp)}\n"
                        ))
                        continue
                    # Trigger-word preservation: split on first ", "
                    if ", " in original:
                        trigger, eng_rest = original.split(", ", 1)
                    else:
                        trigger, eng_rest = "", original
                    chinese = self._translate_to_chinese(eng_rest if eng_rest else original)
                    if not chinese:
                        failed += 1
                        self.master.after(0, lambda fp=fp: self.update_caption_log(
                            f"  [fail empty translation] {os.path.basename(fp)} (kept original)\n"
                        ))
                        continue
                    if trigger:
                        new_text = f"{trigger}, {eng_rest} - {chinese}"
                    else:
                        new_text = f"{original} - {chinese}"
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    translated += 1
                    self.master.after(0, lambda fp=fp, c=chinese: self.update_caption_log(
                        f"  [ok] {os.path.basename(fp)}  →  ...{c[:40]}\n"
                    ))
                except Exception as e:
                    failed += 1
                    self.master.after(0, lambda fp=fp, e=e: self.update_caption_log(
                        f"  [fail {type(e).__name__}] {os.path.basename(fp)}: {e}\n"
                    ))
                self.master.after(0, lambda i=i: self.update_caption_progress(
                    (i + 1) / len(files) * 100, i + 1, len(files)
                ))
        finally:
            self._unload_translator()
            self._translating = False
        self.master.after(0, lambda: self.update_caption_log(
            f"=== Done: {translated} translated, {skipped} skipped, {failed} failed ===\n"
        ))

    def _caption_model_blocked_by_training(self) -> bool:
        """True if a training run is live and a caption subprocess likely won't co-fit."""
        proc = getattr(self, "current_process", None)
        try:
            running = proc is not None and proc.poll() is None
        except Exception:
            running = False
        if not running:
            return False
        need_gb = 8.0 if self._is_qwen_captioner() else 2.5
        free_gb = None
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                free_gb = _torch.cuda.mem_get_info()[0] / 1e9
        except Exception:
            free_gb = None
        margin = need_gb + 3.0
        if free_gb is not None and free_gb >= margin:
            self.update_caption_log(
                f"Training is running, but {free_gb:.0f} GB of VRAM is free — captioning "
                f"alongside it.\n")
            return False
        model = "Qwen3-VL" if self._is_qwen_captioner() else "the caption model"
        _detail = (f"about {free_gb:.0f} GB free right now" if free_gb is not None
                   else "free VRAM can't be measured from here")
        messagebox.showwarning(
            "Training is running",
            f"Captioning with {model} needs about {need_gb:.0f} GB of VRAM on top of the "
            f"training run, and there's {_detail}.\n\n"
            "Wait for training to finish.")
        return True

    def caption_all_florence(self):
        """Caption all images using the selected AI model (Florence-2 or Qwen3-VL)."""
        folder = self.image_folder_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid image folder")
            return

        images = self.get_caption_image_files()
        if not images:
            messagebox.showinfo("Info", "No images found in folder")
            return
        if self._caption_model_blocked_by_training():
            return

        overwrite = self.overwrite_captions_var.get()
        if not overwrite:
            images = [img for img in images if not os.path.exists(os.path.splitext(img)[0] + ".txt")]
            if not images:
                messagebox.showinfo("Info", "All images already have captions. Enable 'Overwrite' to regenerate.")
                return

        if getattr(self, "_captioning_running", False):
            messagebox.showinfo("Already running",
                                "A captioning job is already running. Press Stop to end it first.")
            return

        self._run_ai_caption_subprocess(images)

    def caption_single_image(self, img_path):
        """Caption a single image (Regenerate AI on the thumbnail grid)."""
        if getattr(self, "_captioning_running", False):
            messagebox.showinfo("Captioning in progress",
                                "A captioning job is running — wait for it to finish (or press Stop).")
            return
        if self._caption_model_blocked_by_training():
            return
        self._run_ai_caption_subprocess([img_path])

    def _set_caption_buttons_running(self, running: bool):
        """Grey the job-starting buttons while a captioning job is in flight."""
        state = tk.DISABLED if running else tk.NORMAL
        for attr in ("caption_all_btn", "caption_static_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                try:
                    btn.configure(state=state)
                except tk.TclError:
                    pass

    def stop_captioning(self):
        """Stop AI captioning — subprocess jobs honour a stop file between images."""
        proc = getattr(self, "caption_process", None)
        if proc is not None and proc.poll() is None:
            stop_file = getattr(self, "_caption_stop_file", "")
            if stop_file:
                try:
                    with open(stop_file, "w", encoding="utf-8"):
                        pass
                except OSError:
                    pass
            self.caption_stop_btn.configure(state=tk.DISABLED)
            self.update_caption_log("Stopping — finishing the current image first...\n")
            return
        self.captioning_stop_flag = True
        self.caption_stop_btn.configure(state=tk.DISABLED)
        self.update_caption_log("Stopping — finishing the current image first...\n")

    def update_caption_progress(self, progress, current, total):
        """Update caption progress bar and label"""
        self.caption_progress_var.set(progress)
        self.caption_progress_label.config(text=f"{current}/{total} images")

    def _smart_text_insert(self, widget, text):
        """Insert text and only auto-scroll if user was already at the bottom.
        Preserves manual scroll position so streaming output doesn't yank the view away."""
        try:
            at_bottom = widget.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        if at_bottom:
            widget.see(tk.END)
        widget.configure(state="disabled")

    def update_caption_log(self, text):
        """Update the caption log (preserves user scroll position).

        Thread-safe: several captioning/translation workers call this directly from their
        threads (fifteen call sites — some marshalled via after(), some not). Touching Tk
        widgets + pumping update_idletasks from a worker thread is the classic Tkinter
        hang, so marshal here, once, for every caller."""
        if threading.current_thread() is not threading.main_thread():
            try:
                self.master.after(0, lambda t=text: self.update_caption_log(t))
            except Exception:
                pass   # window closed mid-caption — drop the line rather than crash the worker
            return
        self._append_global_log(text)
        self._smart_text_insert(self.caption_log, text)
        self.master.update_idletasks()

    def find_replace_in_captions(self, preview_only=False):
        """Find and replace text in all caption files (case insensitive)"""
        import re

        folder = self.image_folder_var.get()
        find_text = self.find_text_var.get()
        replace_text = self.replace_text_var.get()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return []

        if not find_text:
            messagebox.showerror("Error", "Please enter text to find")
            return []

        results = []
        txt_files = glob.glob(os.path.join(glob.escape(folder), "*.txt"))

        # Compile case-insensitive pattern
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if pattern.search(content):
                    # Replacement via a function so re NEVER parses it as a template: a
                    # Windows path typed into the box used to write real newlines/tabs
                    # into captions (\n, \t), and \1 raised "invalid group reference".
                    new_content = pattern.sub(lambda _m: replace_text, content)
                    results.append({
                        'file': txt_file,
                        'old': content,
                        'new': new_content
                    })

                    if not preview_only:
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
            except Exception as e:
                self.update_caption_log(f"Error processing {txt_file}: {e}\n")

        if not preview_only:
            self.update_caption_log(f"Replaced in {len(results)} files\n")
            self.refresh_caption_images()

        return results

    def preview_find_replace(self):
        """Preview find/replace changes"""
        results = self.find_replace_in_captions(preview_only=True)

        if not results:
            messagebox.showinfo("Preview", "No matches found")
            return

        # Show preview dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Find & Replace Preview")
        dialog.geometry("700x500")
        dialog.configure(bg=BG_COLOR)

        ttk.Label(dialog, text=f"Found {len(results)} files with matches:", font=("Arial", 11, "bold")).pack(pady=10)

        # Scrollable text area
        preview_text = scrolledtext.ScrolledText(dialog, height=20, width=80, bg=ENTRY_BG, fg=FG_COLOR, wrap="word")
        preview_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        for result in results:
            filename = os.path.basename(result['file'])
            preview_text.insert(tk.END, f"\n=== {filename} ===\n")
            preview_text.insert(tk.END, f"BEFORE: {result['old'][:200]}...\n" if len(result['old']) > 200 else f"BEFORE: {result['old']}\n")
            preview_text.insert(tk.END, f"AFTER:  {result['new'][:200]}...\n" if len(result['new']) > 200 else f"AFTER:  {result['new']}\n")

        preview_text.configure(state="disabled")

        # Apply button
        def apply_changes():
            self.find_replace_in_captions(preview_only=False)
            dialog.destroy()
            messagebox.showinfo("Done", f"Replaced text in {len(results)} files")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Apply Changes", command=apply_changes).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def browse_caption_folder(self):
        """Browse for caption folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_var.set(folder)

    def generate_captions(self):
        """Generate caption files for all images in the selected folder"""
        folder = self.image_folder_var.get()
        # Read the WIDGET-BOUND trigger var. caption_text_var is an orphan StringVar no
        # widget binds (default: the literal placeholder "trigger_word") — reading it here
        # wrote the word "trigger_word" into every caption file regardless of what the
        # user typed in the visible Trigger Word box.
        caption_text = self.caption_trigger_var.get().strip()
        overwrite = self.overwrite_captions_var.get()

        if not folder:
            messagebox.showerror("Error", "Please select a folder.")
            return

        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder does not exist.")
            return

        if not caption_text:
            messagebox.showerror("Error", "Please enter a trigger word in the Trigger Word box first.")
            return

        # Supported image extensions — plus video clips under MiniMax H3, where they are
        # training items and need a .txt exactly like a still does.
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        if self._is_minimax_arch():
            image_extensions = image_extensions | self.TRAINING_VIDEO_EXTENSIONS

        # Clear log
        self.caption_log.configure(state="normal")
        self.caption_log.delete(1.0, tk.END)

        created = 0
        skipped = 0
        errors = 0

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue

            # Create caption file path
            caption_path = os.path.splitext(filepath)[0] + ".txt"

            try:
                if os.path.exists(caption_path) and not overwrite:
                    self.caption_log.insert(tk.END, f"⊘ Skipped (exists): {filename}\n")
                    skipped += 1
                else:
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption_text)
                    self.caption_log.insert(tk.END, f"✓ Created: {os.path.basename(caption_path)}\n")
                    created += 1
            except Exception as e:
                self.caption_log.insert(tk.END, f"✗ Error ({filename}): {str(e)}\n")
                errors += 1

        # Summary
        self.caption_log.insert(tk.END, f"\n--- Summary ---\n")
        self.caption_log.insert(tk.END, f"Created: {created}\n")
        self.caption_log.insert(tk.END, f"Skipped: {skipped}\n")
        self.caption_log.insert(tk.END, f"Errors: {errors}\n")
        self.caption_log.insert(tk.END, f"Total images found: {created + skipped + errors}\n")

        self.caption_log.configure(state="disabled")
        self.caption_log.see(tk.END)

