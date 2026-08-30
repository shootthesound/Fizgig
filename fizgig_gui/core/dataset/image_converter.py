import math
import os
import subprocess
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import json
from PIL import Image, ImageTk

from face_utils import FaceDetector, draw_face_boxes, FaceEmbedder, crop_to_face
from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR
from fizgig_gui.core.config.constants import FACE_DETECTION_AVAILABLE

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class ImageConverterMixin:
    def create_image_converter(self):
        """Create the Image Prep tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.image_converter_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Image Prep",
            "Resize, convert to PNG, and optionally face-crop your training images. "
            "Optional — skip straight to Captions if your images are already prepared.",
        )

        # Card 1: Training folder — display only. Everything happens IN this folder; the old
        # optional Output Folder is gone (it silently diverged from the training source set on
        # the Start tab, which is never what a training workflow wants).
        folders_card = self._start_section_card(
            outer, "Training Folder",
            "Everything below happens inside the training folder from the Start tab — "
            "prepared images land there, ready for the Captions tab and training.",
        )
        folders_card.grid_columnconfigure(1, weight=1)
        ttk.Label(folders_card, text="Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        tk.Label(folders_card, textvariable=self.image_folder_var,
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                 anchor="w").grid(row=0, column=1, sticky=tk.W, pady=4)
        tk.Label(folders_card, text="(set on the Start tab)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=1, sticky=tk.W, pady=(0, 4)
        )

        # Working from video or audio — sits ABOVE the prep steps because clips and voice
        # segments are cut first, and both then land in the same training folder as everything
        # else. Shown to everyone rather than only under MiniMax: the person who needs it is
        # the one who has not chosen a model yet, looking at an hour of footage or a voice
        # memo, wondering where to start.
        gizmo_card = self._start_section_card(
            outer, "Working from video or audio?",
            "Gizmo cuts training clips AND voice segments. Video: scrub to a moment, pick a "
            "length, save — frame rate, frame count, sizing and sound all come out on spec. "
            "Audio: open a recording (or a video, for just its sound), scrub the waveform, "
            "caption the voice — with optional Whisper transcription — and export ready "
            "training segments. Video and voice training are MiniMax H3 only; still images "
            "need none of this.",
        )
        _gz_row = tk.Frame(gizmo_card, bg=COLORS["bg_surface"])
        _gz_row.pack(anchor=tk.W)
        tk.Button(_gz_row, text="🎬🎙  Open Gizmo", command=self._launch_gizmo,
                  bg=COLORS["accent"], fg=COLORS["text_primary"],
                  activebackground=COLORS["accent_hover"],
                  activeforeground=COLORS["text_primary"], font=(FONT_FAMILY, 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=16, pady=8, cursor="hand2").pack(side=tk.LEFT)
        tk.Label(_gz_row, text="opens in its own window — Fizgig keeps running",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=12)

        # Card 2: What to do — one radio per outcome, plain-language hint under each. The radio
        # VALUES stay the historical mode strings so persistence and the convert pipeline are
        # untouched; only the visible labels changed.
        mode_card = self._start_section_card(outer, "1 · What to do", None)

        def _mode_radio(text, value, hint, hint_fg=None):
            rb = ttk.Radiobutton(mode_card, text=text, variable=self.prep_mode_var,
                                 value=value, command=self._on_prep_mode_changed)
            rb.pack(anchor=tk.W, pady=(6, 0))
            lbl = tk.Label(mode_card, text=hint, font=(FONT_FAMILY, 10),
                           fg=hint_fg or COLORS["text_explain"], bg=COLORS["bg_surface"],
                           wraplength=680, justify=tk.LEFT)
            lbl.pack(anchor=tk.W, padx=(24, 0))
            return rb

        _mode_radio(
            "Resize + face close-ups — recommended for people",
            "Auto Prep (Face Crops)",
            "Every photo is resized and saved as PNG, PLUS a zoomed-in copy of the face saved "
            "beside it — more detail shots for better likeness.\n"
            "\U0001F4A1 Works best on high-res originals: if your photos are already shrunk to "
            "training size, the face close-ups come out soft. Start from the biggest versions "
            "you have.")
        _mode_radio(
            "Resize only",
            "Resize Only",
            "Just resize + convert to PNG. Use for styles, objects, or already-cropped sets.")
        _mode_radio(
            "Face close-ups only",
            "Face Crop Only",
            "Keep only the cropped face from each photo — the full shot is not kept.")

        # Options row: max size always live; face options grey out in Resize Only (kept visible
        # so the layout doesn't jump and users learn they exist).
        opts_row = tk.Frame(mode_card, bg=COLORS["bg_surface"])
        opts_row.pack(anchor=tk.W, pady=(12, 0))
        ttk.Label(opts_row, text="Target megapixels:").pack(side=tk.LEFT, padx=(0, 4))
        _max_combo = ttk.Combobox(opts_row, textvariable=self.prep_megapixels_var,
                                  values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4",
                                          "3.0", "4.2"],
                                  state="readonly", width=6)
        _max_combo.pack(side=tk.LEFT)
        _max_combo.bind("<<ComboboxSelected>>", lambda e: self._update_prep_note())
        tk.Label(opts_row, text="MP  (larger images shrink to fit; smaller are left alone)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(
            side=tk.LEFT, padx=(4, 16))
        self._face_target_label = ttk.Label(opts_row, text="Face:")
        self._face_target_label.pack(side=tk.LEFT, padx=(0, 4))
        self._face_target_combo = ttk.Combobox(
            opts_row, textvariable=self.face_selection_var,
            values=["Largest Face", "Largest Male Face", "Largest Female Face"],
            state="readonly" if FACE_DETECTION_AVAILABLE else "disabled", width=18,
        )
        self._face_target_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._face_padding_label = ttk.Label(opts_row, text="Padding:")
        self._face_padding_label.pack(side=tk.LEFT, padx=(0, 4))
        self._face_padding_entry = ttk.Entry(opts_row, textvariable=self.face_padding_var, width=5)
        self._face_padding_entry.pack(side=tk.LEFT)
        self._face_pct_label = tk.Label(opts_row, text="% around the face",
                                        font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                                        bg=COLORS["bg_surface"])
        self._face_pct_label.pack(side=tk.LEFT, padx=(4, 0))
        if not FACE_DETECTION_AVAILABLE:
            self._face_unavail_label = ttk.Label(
                opts_row, text="(Run install_fizgig.py to enable)",
                foreground=COLORS["warning"],
            )
            self._face_unavail_label.pack(side=tk.LEFT, padx=(8, 0))
        else:
            self._face_unavail_label = None

        # Why this is megapixels and not a "max size" any more. Training picks its resolution by
        # AREA, so a longest-edge cap quietly shrank every non-square image below the training
        # target — a 3:4 photo kept only 75% of the pixels it could have trained at, 16:9 just 56%.
        tk.Label(mode_card,
                 text="Sizing is by target area (megapixels), not longest side — this matches how "
                      "training buckets your images, so prepping no longer costs you resolution. "
                      "Aspect ratio is preserved; nothing is cropped.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))
        # Only shown when prep is set BELOW the Training tab's target — the one harmful direction
        # (prepping higher is free: training simply downscales at cache time).
        self._prep_mp_warn_var = tk.StringVar(value="")
        self._prep_mp_warn_label = tk.Label(
            mode_card, textvariable=self._prep_mp_warn_var,
            font=(FONT_FAMILY, 9), fg=COLORS["warning"], bg=COLORS["bg_surface"],
            wraplength=760, justify=tk.LEFT)
        # packed/unpacked by _update_prep_note

        # Card 3: Your originals — the one real destination question, as an explicit choice
        # (replaces the old inverted "Replace originals" checkbox). Keep-safe is the default.
        orig_card = self._start_section_card(outer, "2 · Your originals", None)
        ttk.Radiobutton(
            orig_card, text="Keep them safe — moved to an 'originals' subfolder",
            variable=self.delete_originals_var, value=False,
            command=self._update_prep_note).pack(anchor=tk.W, pady=(4, 0))
        ttk.Radiobutton(
            orig_card, text="Replace them  ⚠ originals are gone after this",
            variable=self.delete_originals_var, value=True,
            command=self._update_prep_note).pack(anchor=tk.W, pady=(4, 2))

        # Card 4: What will happen — the single honest summary, computed from ALL the settings
        # (the old one-line note ignored half of them). Accent border so it reads as the answer.
        summary_card = self._start_section_card(outer, "\U0001F4CB What will happen", None,
                                                accent_border=True)
        self._prep_note_var = tk.StringVar()
        self._prep_note_label = tk.Label(
            summary_card, textvariable=self._prep_note_var,
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
            wraplength=700, justify=tk.LEFT,
        )
        self._prep_note_label.pack(anchor=tk.W)

        # Card 5: Run it — one unmistakable primary action (label carries the live image count,
        # set by _update_prep_note), with the face-detection test framed as the optional,
        # nothing-is-written side step it actually is.
        action_card = self._start_section_card(outer, "3 · Run it", None)

        self.prepare_images_btn = tk.Button(
            action_card, text="✨ Prepare Images Now", command=self.convert_images,
            font=(FONT_FAMILY, 12, "bold"),
            fg="#FFFFFF", bg=COLORS["accent"],
            activeforeground="#FFFFFF", activebackground=COLORS["accent_hover"],
            relief="flat", bd=0, padx=24, pady=8, cursor="hand2",
        )
        self.prepare_images_btn.pack(anchor=tk.W, pady=(4, 10))

        test_row = tk.Frame(action_card, bg=COLORS["bg_surface"])
        test_row.pack(anchor=tk.W)
        tk.Label(test_row, text="Want to check first?",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))
        self.preview_faces_btn = ttk.Button(
            test_row, text="Test face detection on one photo…", command=self.preview_faces,
            state="normal" if FACE_DETECTION_AVAILABLE else "disabled",
        )
        self.preview_faces_btn.pack(side=tk.LEFT)
        tk.Label(test_row, text="optional and safe — shows the crop, writes nothing",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_muted"],
                 bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(8, 0))

        # Apply initial state (face-control greying + summary + button count)
        self._on_prep_mode_changed()

        # Card 5: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)

        self.convert_log = scrolledtext.ScrolledText(
            log_card, height=12, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.convert_log.pack(fill=tk.BOTH, expand=True)

        # Card 6: Look Consistency Filter — deliberately the LAST card: it scores the images as
        # they'll actually be trained, so it only makes sense after resize/crop/captioning is done.
        filter_card = self._start_section_card(
            outer, "Final Step: Look Consistency Filter (faces)",
            "Run this LAST, after every other prep stage — it scores the finished training folder. "
            "Pick THREE baseline images that nail the look you want; every image's face is scored "
            "against all three and averaged (ArcFace embedding similarity) — one baseline photo "
            "would bake its own angle/expression/lighting bias into every score, three cancel it "
            "out. Great for weeding out synthetic images that "
            "drifted off-look — the subtle near-misses a loss curve can never see. Click images to "
            "mark them, or let Auto-Suggest flag the statistical outliers, then move the marked "
            "ones out of the dataset in one go (they go to an 'excluded_by_look' subfolder — "
            "nothing is deleted). Real-but-unusual low scorers (tight angles, profiles) that you "
            "KEEP can ease into training gently — the scan saves its scores with the dataset, and "
            "the Training tab's 'Warm up look outliers' toggle (Krea 2) ramps their LR up over "
            "the first few epochs instead of letting them fight the forming identity.",
        )
        self._face_filter_btn = ttk.Button(
            filter_card, text="🔍 Open Look Filter…", command=self._open_face_filter_window,
            state="normal" if FACE_DETECTION_AVAILABLE else "disabled",
        )
        self._face_filter_btn.pack(anchor=tk.W)
        if not FACE_DETECTION_AVAILABLE:
            ttk.Label(filter_card, text="(Run install_fizgig.py to enable face tools)",
                      foreground=COLORS["warning"]).pack(anchor=tk.W, pady=(4, 0))

        self._add_youtube_help_button(outer, "image_prep")

    @property
    def face_detector(self):
        """Lazy-loaded face detector instance"""
        if self._face_detector is None and FACE_DETECTION_AVAILABLE:
            self._face_detector = FaceDetector()
        return self._face_detector

    def _on_prep_mode_changed(self, *args):
        """Grey out face-related controls in Resize Only (kept visible — layout doesn't jump,
        and users learn the options exist)."""
        mode = self.prep_mode_var.get()
        face_on = mode != "Resize Only"
        muted, secondary = COLORS["text_muted"], COLORS["text_secondary"]
        self._face_target_combo.configure(
            state=("readonly" if (face_on and FACE_DETECTION_AVAILABLE) else "disabled"))
        self._face_padding_entry.configure(state=("normal" if face_on else "disabled"))
        for lbl in (self._face_target_label, self._face_padding_label):
            try:
                lbl.configure(foreground=(secondary if face_on else muted))
            except tk.TclError:
                pass
        self._face_pct_label.configure(fg=(secondary if face_on else muted))
        self.preview_faces_btn.configure(
            state=("normal" if (face_on and FACE_DETECTION_AVAILABLE) else "disabled"))

        self._update_prep_note()

    def _prep_source_stats(self, max_sample=40):
        """(image_count, median_area_px, median_size) for the training folder's top-level images.

        AREA rather than longest edge, because that's what both prep and training size by.
        Read from image HEADERS only (PIL .size — no pixel decode), sampled at most
        `max_sample` files, so it's cheap enough to run on every settings change. Returns
        (0, None, None) when the folder is unset/empty."""
        folder = self.image_folder_var.get().strip()
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        try:
            files = [os.path.join(folder, f) for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in exts]
        except OSError:
            return 0, None, None
        sizes = []
        for p in files[:max_sample]:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                pass
        if not sizes:
            return len(files), None, None
        sizes.sort(key=lambda wh: wh[0] * wh[1])
        median_size = sizes[len(sizes) // 2]
        return len(files), median_size[0] * median_size[1], median_size

    def _update_prep_note(self, *args):
        """The 'What will happen' summary — computed from ALL the settings (mode, originals
        choice, target megapixels, live folder contents). The one-line note this replaces ignored the
        output folder entirely and taught users the wrong answer to 'does this touch my
        folder?'."""
        if not hasattr(self, '_prep_note_var'):
            return
        # A voice folder has nothing for this tab to do — resize, crop and face detection are
        # image operations. Say so instead of promising to process "your 0 images".
        if self._training_folder_audio_only():
            self._prep_note_var.set(
                "🎙 Audio-only training set — this tab prepares images, and voice recordings "
                "need none of it. Segments are cut, captioned and sized in Gizmo's audio tab; "
                "your files here are already ready to train.")
            return
        mode = self.prep_mode_var.get()
        replace = self.delete_originals_var.get()
        try:
            prep_mp = float(self.prep_megapixels_var.get())
        except (ValueError, tk.TclError):
            prep_mp = 1.0
        target_area = self._prep_target_area(prep_mp)
        n, median_area, median_size = self._prep_source_stats()

        # A worked example in the user's own aspect ratio, so "1.0 MP" is a concrete size.
        example = ""
        if median_size:
            _w, _h = self._prep_output_size(median_size, target_area)
            if (_w, _h) != tuple(median_size):
                example = f" (your typical {median_size[0]}×{median_size[1]} → {_w}×{_h})"

        count = f"your {n} images" if n else "your images"
        sized = f"resized to about {prep_mp:g} MP{example} and saved as PNG"
        if mode == "Auto Prep (Face Crops)":
            what = (f"{count} → {sized}, plus one face "
                    f"close-up each{f' (≈{n * 2} files)' if n else ''}")
        elif mode == "Face Crop Only":
            what = (f"{count} → replaced by just the cropped face from each photo, "
                    f"saved as PNG")
        else:
            what = f"{count} → {sized}"

        where = "Everything lands in your training folder."
        if replace:
            originals = "Your original files are replaced ⚠ there is no undo."
        else:
            originals = ("Your originals are moved to the 'originals' subfolder — "
                         "nothing is deleted.")

        lines = [f"{what}. {where} {originals}"]
        # Soft-crop warning: face modes cropping from images that are already training-size
        # produce small, blurry faces. Header-read median AREA across a sample of the folder.
        if mode != "Resize Only" and median_area is not None and median_area <= target_area:
            lines.append(f"⚠ Your images are already at or below {prep_mp:g} MP — face "
                         f"close-ups cut from them will be soft. If you have higher-res "
                         f"versions, prep from those instead.")
        if mode != "Resize Only":
            lines.append("Next: eyeball the face close-ups on the Captions tab and Remove any "
                         "blurry ones before captioning.")
        self._prep_note_var.set("\n".join(lines))

        # Prep BELOW the training target is the one harmful direction: training never upscales,
        # so those pixels are gone for good. Prepping higher is free — training just downscales.
        if hasattr(self, "_prep_mp_warn_label"):
            try:
                train_mp = float(self.dataset_megapixels_var.get())
            except (ValueError, tk.TclError):
                train_mp = prep_mp
            if prep_mp < train_mp:
                self._prep_mp_warn_var.set(
                    f"⚠ Training is set to {train_mp:g} MP but prep is set to {prep_mp:g} MP — "
                    f"your images would be shrunk below what training asks for, and training "
                    f"cannot get that detail back. Match them, or prep higher.")
                if not self._prep_mp_warn_label.winfo_manager():
                    self._prep_mp_warn_label.pack(anchor=tk.W, pady=(4, 0))
            else:
                self._prep_mp_warn_var.set("")
                if self._prep_mp_warn_label.winfo_manager():
                    self._prep_mp_warn_label.pack_forget()

        # The Run button carries the live count — "Prepare 34 Images Now" answers "run on what?"
        if hasattr(self, "prepare_images_btn"):
            self.prepare_images_btn.configure(
                text=(f"✨ Prepare {n} Image{'s' if n != 1 else ''} Now" if n
                      else "✨ Prepare Images Now"))

    def _get_face_selection_mode(self):
        """Parse face selection mode from Face Target dropdown."""
        mode_text = self.face_selection_var.get()
        if "Male" in mode_text:
            return "largest_male"
        elif "Female" in mode_text:
            return "largest_female"
        return "largest_face"

    def preview_faces(self):
        """Preview face detection on a single image"""
        if not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available. Run install_fizgig.py first.")
            return

        filepath = filedialog.askopenfilename(
            title="Select image to preview faces",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif")]
        )
        if not filepath:
            return

        try:
            # Detect faces
            faces = self.face_detector.detect_all(filepath)

            if not faces:
                messagebox.showinfo("No Faces", f"No faces detected in:\n{os.path.basename(filepath)}")
                return

            # Find the largest face (or by gender based on current mode)
            crop_mode = self._get_face_selection_mode()
            if crop_mode == "largest_male":
                selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            elif crop_mode == "largest_female":
                selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            else:
                selected = self.face_detector.get_largest(faces)

            # Get highlight index
            highlight_idx = faces.index(selected) if selected in faces else None

            # Load image and draw boxes
            with Image.open(filepath) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                preview_img = draw_face_boxes(img, faces, highlight_index=highlight_idx)

                # Create preview window
                self._show_face_preview_window(preview_img, faces, filepath, highlight_idx)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect faces:\n{str(e)}")

    def _show_face_preview_window(self, preview_img, faces, filepath, highlight_idx):
        """Show a popup window with the face detection preview"""
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"Face Preview - {os.path.basename(filepath)}")
        preview_window.configure(bg=BG_COLOR)

        # Resize for display if too large
        display_img = preview_img.copy()
        max_display = 800
        if display_img.width > max_display or display_img.height > max_display:
            ratio = min(max_display / display_img.width, max_display / display_img.height)
            new_size = (int(display_img.width * ratio), int(display_img.height * ratio))
            display_img = display_img.resize(new_size, Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)

        # Image label
        img_label = ttk.Label(preview_window, image=photo)
        img_label.image = photo  # Keep reference
        img_label.pack(padx=10, pady=10)

        # Info frame
        info_frame = ttk.Frame(preview_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        # Face count
        ttk.Label(info_frame, text=f"Faces detected: {len(faces)}", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        # Face details
        for i, face in enumerate(faces):
            marker = " [SELECTED]" if i == highlight_idx else ""
            gender = face.gender.capitalize() if face.gender != 'unknown' else '?'
            ttk.Label(
                info_frame,
                text=f"  Face {i+1}: {gender}, {face.area:,} px{marker}"
            ).pack(anchor=tk.W)

        # Legend
        legend_frame = ttk.Frame(preview_window)
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(legend_frame, text="Green = Selected for cropping", foreground="green").pack(side=tk.LEFT, padx=10)
        ttk.Label(legend_frame, text="Yellow = Other faces", foreground="yellow").pack(side=tk.LEFT, padx=10)

        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

    def convert_images(self):
        """Prepare images based on selected prep mode."""
        self._originals_dir_cache = {}  # Reset per run
        source_folder = self.image_folder_var.get()
        output_folder = self.convert_output_var.get() or source_folder
        # Target AREA in pixels, from the megapixel selector (see _resize_image for why area).
        try:
            target_area = self._prep_target_area(float(self.prep_megapixels_var.get()))
        except (ValueError, tk.TclError):
            target_area = self._prep_target_area(1.0)
        replace_originals = self.delete_originals_var.get()
        prep_mode = self.prep_mode_var.get()
        face_mode = self._get_face_selection_mode()

        try:
            face_padding = float(self.face_padding_var.get())
        except ValueError:
            face_padding = 20.0

        if not source_folder:
            messagebox.showerror("Error", "Please select a source folder.")
            return
        if not os.path.isdir(source_folder):
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        # Check face detection for modes that need it
        if prep_mode != "Resize Only" and not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available.\nRun install_fizgig.py to enable.")
            return

        os.makedirs(output_folder, exist_ok=True)

        if getattr(self, "_prep_running", False):
            messagebox.showinfo("Already running", "An image prep job is already running.")
            return

        # Clear log
        self.convert_log.configure(state="normal")
        self.convert_log.delete(1.0, tk.END)

        # Worker thread, NOT inline: face detection is ONNX inference per image plus full-size
        # PIL decode/encode, and running the batch on the Tk main thread froze the whole window
        # ("Not Responding") for minutes on a big folder. Every Tk read happened above; the
        # workers only touch the UI through _log, which already marshals via after() — it was
        # written for this thread and waiting for it. The button is disabled for the duration
        # so the job can't be double-started.
        self._prep_running = True
        try:
            self.prepare_images_btn.config(state="disabled", text="Preparing…")
        except Exception:
            pass

        def _prep_worker():
            try:
                if prep_mode == "Auto Prep (Face Crops)":
                    self._auto_prep_images(source_folder, output_folder, target_area, face_mode, face_padding, replace_originals)
                elif prep_mode == "Resize Only":
                    self._resize_only_images(source_folder, output_folder, target_area, replace_originals)
                elif prep_mode == "Face Crop Only":
                    self._face_crop_only_images(source_folder, output_folder, target_area, face_mode, face_padding, replace_originals)
            except Exception as e:
                self._log(f"\nERROR: prep failed — {type(e).__name__}: {e}\n")
            finally:
                self.master.after(0, self._prep_finished)

        threading.Thread(target=_prep_worker, daemon=True).start()

    def _prep_finished(self):
        """Main-thread epilogue for a prep run: finalize the log, re-arm the button."""
        self._prep_running = False
        try:
            self.prepare_images_btn.config(state="normal", text="✨ Prepare Images Now")
        except Exception:
            pass
        self.convert_log.configure(state="disabled")
        self.convert_log.see(tk.END)

    def _resize_only_images(self, source_folder, output_folder, target_area, replace_originals):
        """Resize Only mode: convert/resize images, no face detection."""
        self._log("Mode: Resize Only\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                img, resized = self._resize_image(img, target_area)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")

    def _face_crop_only_images(self, source_folder, output_folder, target_area, face_mode, face_padding, replace_originals):
        """Face Crop Only mode: face crop replaces the output."""
        self._log(f"Mode: Face Crop Only ({face_mode}, padding {face_padding}%)\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors, face_crops, no_face = 0, 0, 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                cropped = False
                crop_info = ""

                try:
                    faces = self.face_detector.detect_from_pil(img)
                    if faces:
                        selected, note = self._select_face(faces, face_mode)
                        if note:
                            self._log(note)
                        if selected:
                            img = crop_to_face(img, selected, face_padding)
                            cropped = True
                            face_crops += 1
                            crop_info = f" [face: {selected.gender}]"
                    else:
                        self._log(f"  No face in {filename}, skipping crop\n")
                        no_face += 1
                except Exception as fe:
                    self._log(f"  Face error ({filename}): {fe}\n")

                img, resized = self._resize_image(img, target_area)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized and not cropped:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if (resized or cropped) else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]{crop_info}\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops: {face_crops} | No face: {no_face}\n")

    def _auto_prep_images(self, source_folder, output_folder, target_area, face_mode, face_padding, replace_originals):
        """Auto Prep mode: resize originals + generate face crops from the HIGH-RES original
        (before it gets overwritten/moved), then handle originals."""
        self._log(f"Mode: Auto Prep (Face Crops)\n")
        self._log(f"Face target: {face_mode}, padding: {face_padding}%\n")
        self._log(f"Output: {output_folder}\n\n")

        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0
        face_crops_created, no_face = 0, 0
        crop_index = self._get_next_facecrop_index(output_folder)

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            base_name = os.path.splitext(filename)[0]

            # Skip existing FaceCrop derivatives
            if base_name.startswith("FaceCrop_"):
                self._log(f"Skipped (derivative): {filename}\n")
                skipped += 1
                continue

            try:
                # Load original at full resolution
                original_img = self._load_image(filepath)
                original_size = original_img.size

                # --- Face crop from the HIGH-RES original (before resize) ---
                try:
                    faces = self.face_detector.detect_from_pil(original_img)
                    if faces:
                        selected, note = self._select_face(faces, face_mode)
                        if note:
                            self._log(note)
                        if selected:
                            cropped = crop_to_face(original_img, selected, face_padding)
                            cropped, _ = self._resize_image(cropped, target_area)
                            crop_name = f"FaceCrop_{crop_index:03d}.png"
                            crop_path = os.path.join(output_folder, crop_name)
                            cropped.save(crop_path, "PNG")
                            cw, ch = cropped.size
                            self._log(f"Face crop: {crop_name} ({cw}x{ch}) from {filename} ({original_size[0]}x{original_size[1]}) [{selected.gender}]\n")
                            face_crops_created += 1
                            crop_index += 1
                            cropped.close()
                        else:
                            no_face += 1
                    else:
                        self._log(f"No face: {filename}\n")
                        no_face += 1
                except Exception as e:
                    self._log(f"Face crop error ({filename}): {e}\n")

                # --- Resize and save the main image ---
                resized_img, resized = self._resize_image(original_img, target_area)
                w, h = resized_img.size
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"OK (no changes): {filename}\n")
                    skipped += 1
                    resized_img.close()
                    continue

                output_path = self._safe_output_path(filepath, output_path)
                # Keep-safe + in-place PNG means this save OVERWRITES the original and
                # _handle_original below has nothing left to move — the issue #43 failure, which
                # was fixed in the other two modes but not here, i.e. in the DEFAULT one.
                self._stash_original_if_inplace(filepath, output_path, output_folder, replace_originals)
                self._atomic_png_save(resized_img, output_path)
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                resized_img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\n")
        self._log(f"Originals converted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops created: {face_crops_created} | No face: {no_face}\n")
        self._log(f"Total files in output: {len(self._get_image_files(output_folder))}\n")